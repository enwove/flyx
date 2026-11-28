// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/model_host.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <optional>
#include <thread>

#include "apps/common/defaults.h"
#include "apps/common/numbers.h"
#include "apps/flyx-server/drafter_speeds.h"
#include "core/interfaces/options.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace flyx::server {

namespace {

namespace fs = std::filesystem;

using os::to_path;

ojson want_json(const ModelHost::Want& w) {
  return {{"model", w.model},
          {"drafters", w.drafters},
          {"ctx", w.eo.ctx_given ? (w.eo.ctx_max ? ojson("max") : ojson(w.eo.max_ctx)) : ojson(nullptr)},
          {"drafter", w.eo.drafter},
          {"speculative", w.eo.speculative},
          {"kv", w.eo.kv == FLYX_KV_FP8 ? ojson("fp8") : w.eo.kv == FLYX_KV_16 ? ojson("16") : ojson(nullptr)},
          {"kv_ram", w.eo.kv_ram < 0 ? ojson("auto") : ojson(double(w.eo.kv_ram) / 1e9)},
          {"kv_gpu", w.eo.kv_gpu < 0 ? ojson("auto") : ojson(w.eo.kv_gpu)},
          {"vram_reserve", w.eo.vram_reserve < 0 ? ojson("auto") : ojson(double(w.eo.vram_reserve) / 1e9)},
          {"ram_reserve", w.eo.ram_reserve < 0 ? ojson("auto") : ojson(double(w.eo.ram_reserve) / 1e9)},
          {"width", w.eo.width < 0 ? ojson(nullptr) : ojson(w.eo.width)},
          {"draft_trees", w.eo.draft_trees_given ? ojson(w.eo.draft_trees) : ojson(nullptr)},  // null: the drafter's default
          {"lookup", w.eo.lookup},
          {"draft_head", w.draft_head.empty() ? ojson(nullptr) : ojson(w.draft_head)},
          {"draft_weights", w.draft_weights.empty() ? ojson(nullptr) : ojson(w.draft_weights)},
          {"mmproj", w.eo.mmproj_file.empty() ? ojson(nullptr) : ojson(w.eo.mmproj_file)},
          {"lora", w.eo.lora_file.empty() ? ojson(nullptr) : ojson(w.eo.lora_file)},
          {"vision", w.eo.vision},
          {"vision_device", w.eo.vision_device.empty() ? ojson("disk") : ojson(w.eo.vision_device)},
          {"vision_quant", w.eo.vision_quant.empty() ? ojson("q8") : ojson(w.eo.vision_quant)},
          {"vision_max_tokens", w.eo.vision_max_tokens > 0 ? ojson(w.eo.vision_max_tokens) : ojson(nullptr)},
          {"name", w.name}};
}

std::mutex& settings_mutex() {
  static std::mutex mu;
  return mu;
}

// serve.json held by this process's threads (settings_mutex) and by every other flyx server sharing the file
// (serve.json.lock, as drafter_speeds.cpp keeps its file), so that one's read-modify-replace never writes back what
// another has just changed: a LAN setting turned off coming back on, a saved model undone.  Reads take it too, and
// open the file only when they have it: on Windows a replace fails while another process holds the file open.  A
// writer waits 5 s for it at most, a reader (telemetry's, a second) 1 s: it has the settings last read to fall back on.
class SettingsLock {
 public:
  explicit SettingsLock(int wait_ms) : guard_(settings_mutex()) {
    const std::string path = ModelHost::settings_path();
    if (path.empty()) {  // no file to share
      held_ = true;
      return;
    }
    std::error_code ec;
    fs::create_directories(to_path(path).parent_path(), ec);
    file_.emplace(path + ".lock");
    for (int waited = 0;; waited += 25) {
      const os::FileLock::Result r = file_->try_lock(&why_);
      if (r == os::FileLock::Result::kLocked) {
        held_ = true;
        return;
      }
      if (r == os::FileLock::Result::kError) return;
      if (waited >= wait_ms) {
        why_ = path + ".lock: held by another flyx server for " + std::to_string(wait_ms / 1000) + " s";
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
  }
  bool held() const { return held_; }
  const std::string& why() const { return why_; }

 private:
  std::lock_guard<std::mutex> guard_;
  std::optional<os::FileLock> file_;
  bool held_ = false;
  std::string why_;
};
constexpr int kSettingsWriteWaitMs = 5000, kSettingsReadWaitMs = 1000;

// The settings as this process last read or wrote them (under settings_mutex): what a reader is given while
// serve.json cannot be read.
ojson& last_known_settings() {
  static ojson root = ojson::object();
  return root;
}

// Why the settings could not be read, on standard error once a reason (telemetry reads them every second).
void say_settings_problem(const std::string& why) {
  static std::string said;
  if (why == said) return;
  std::fprintf(stderr, "flyx-server: %s\n", why.c_str());
  said = why;
}

// serve.json's object, under SettingsLock: an empty one while there is no file.  false, with *err, for a file that
// is there but cannot be read, is not JSON, or holds no object: a save then leaves its bytes as they are (the last
// model, the network settings in it are not thrown away for an empty object).
bool read_settings_locked(ojson* root, std::string* err) {
  *root = ojson::object();
  const std::string path = ModelHost::settings_path();
  if (path.empty()) return true;
  const fs::path p = to_path(path);
  std::error_code ec;
  const fs::file_status st = fs::status(p, ec);
  if (st.type() == fs::file_type::not_found) return true;
  const std::string keep = " (left as it is: mend or remove it to save settings)";
  if (ec) return *err = path + ": " + ec.message() + keep, false;
  std::string text;
  if (!read_whole_file(path, &text)) return *err = path + ": cannot be read" + keep, false;
  ojson j = ojson::parse(text, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) return *err = path + ": not JSON" + keep, false;
  if (!j.is_object()) return *err = path + ": holds no JSON object" + keep, false;
  *root = std::move(j);
  return true;
}

// The settings for a reader holding `lock`: the file's, else the ones this process last read or wrote, *why saying
// why not (the lock not had: the file is then not opened; the file unreadable).
ojson read_settings_with(const SettingsLock& lock, std::string* why) {
  std::string err;
  ojson root;
  if (!lock.held()) err = "could not read settings from " + ModelHost::settings_path() + ": " + lock.why();
  else if (read_settings_locked(&root, &err)) return last_known_settings() = std::move(root);
  say_settings_problem(err);
  if (why) *why = err;
  return last_known_settings();
}

ojson read_settings(std::string* why = nullptr) {
  SettingsLock lock(kSettingsReadWaitMs);
  return read_settings_with(lock, why);
}

bool write_settings_locked(const ojson& root, std::string* err) {
  const fs::path p = to_path(ModelHost::settings_path());
  if (p.empty()) return true;
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  fs::path tmp = p;
  tmp += "." + std::to_string(os::process_id()) + ".tmp";
  std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
  f << root.dump(2) << "\n";
  f.close();
  if (f) fs::rename(tmp, p, ec);
  if (!f || ec) {
    const std::string cause = ec ? ": " + ec.message() : "";
    fs::remove(tmp, ec);
    if (err) *err = "could not save settings to " + ModelHost::settings_path() + cause;
    return false;
  }
  return true;
}

// serve.json's object as it is now, `change` applied and written back under SettingsLock; `committed`, with what
// was written, before the lock is let go.  Nothing is written where the file cannot be read as an object.
bool update_settings(const std::function<void(ojson&)>& change, std::string* err,
                     const std::function<void(const ojson&)>& committed = {}) {
  SettingsLock lock(kSettingsWriteWaitMs);
  if (!lock.held()) {
    if (err) *err = "could not save settings to " + ModelHost::settings_path() + ": " + lock.why();
    return false;
  }
  ojson root;
  std::string why;
  if (!read_settings_locked(&root, &why)) {
    if (err) *err = "could not save settings: " + why;
    return false;
  }
  change(root);
  if (!write_settings_locked(root, err)) return false;
  last_known_settings() = root;
  if (committed) committed(root);
  return true;
}

// A served scheduler's end: set once it is destroyed (its worker joined, its engine and the backend's model closed),
// on whichever thread let go of it last.  A weak pointer expires as that owner starts the destructor, before any of
// the memory comes back.
struct Teardown {
  std::mutex mu;
  std::condition_variable cv;
  bool done = false;
};

struct SignalTeardown {
  std::shared_ptr<Teardown> teardown;
  void operator()(Scheduler* s) const {
    delete s;
    {
      std::lock_guard<std::mutex> lock(teardown->mu);
      teardown->done = true;
    }
    teardown->cv.notify_all();
  }
};

// A model's requests end, and its memory comes back once the last of them lets go of it: returns when it has.
void retire(std::shared_ptr<Scheduler> old) {
  if (!old) return;
  old->close();
  const SignalTeardown* signal = std::get_deleter<SignalTeardown>(old);
  const std::shared_ptr<Teardown> teardown = signal ? signal->teardown : nullptr;
  std::weak_ptr<Scheduler> gone = old;
  old.reset();
  if (!teardown) {  // not one ModelHost::open made: only its last owner is known to have let go
    while (!gone.expired()) os::sleep_ms(50);
    return;
  }
  std::unique_lock<std::mutex> lock(teardown->mu);
  for (int i = 0; !teardown->cv.wait_for(lock, std::chrono::milliseconds(50), [&] { return teardown->done; }); ++i)
    if (i == 100) std::fprintf(stderr, "flyx-server: waiting for the last model's requests to end\n");
}

ojson drafters_json(const std::vector<apps::DrafterSource>& ds, const char* where) {
  ojson out = ojson::array();
  for (const apps::DrafterSource& d : ds)
    out.push_back({{"kind", d.kind}, {where, d.from}, {"size", d.size ? ojson(*d.size) : ojson(nullptr)}});
  return out;
}

bool drafter_active(std::string_view active, std::string_view kind) {
  if (active.empty() || active == "none") return false;
  size_t i = 0;
  while (i < active.size()) {
    size_t j = active.find('+', i);
    if (j == std::string_view::npos) j = active.size();
    if (active.substr(i, j - i) == kind) return true;
    i = j + 1;
  }
  return false;
}

ojson degraded_json(const std::vector<LoadAdjustment>& list) {
  ojson out = ojson::array();
  for (const LoadAdjustment& a : list)
    out.push_back({{"kind", a.kind}, {"from", a.from}, {"to", a.to}, {"detail", a.detail}});
  return out;
}

// The model a drafter's speeds are kept under (drafter_speeds.h): a Hub name as named, else its file's absolute path.
std::string speed_model(const std::string& named, const std::string& path) {
  std::error_code ec;
  hub::Name n;
  return !fs::exists(to_path(named), ec) && hub::parse_name(named, &n) ? named
                                                                         : os::from_path(fs::absolute(to_path(path), ec));
}

void save_settings(const ModelHost::Want& w) {
  // A local model may be relaunched from another directory. Keep Hub names as requested
  // so their revision and quantization still go through the normal fetch path.
  auto remember_path = [](const std::string& spec) {
    const fs::path path = to_path(spec);
    std::error_code ec;
    if (path.is_absolute() || !fs::exists(path, ec)) return spec;
    const auto full = fs::absolute(path, ec);
    return ec ? spec : os::from_path(full);
  };
  ojson last = want_json(w);
  last["model"] = remember_path(w.model);
  for (size_t i = 0; i < w.drafters.size(); ++i) last["drafters"][i] = remember_path(w.drafters[i]);
  if (!w.eo.mmproj_file.empty()) last["mmproj"] = remember_path(w.eo.mmproj_file);
  if (!w.eo.lora_file.empty()) last["lora"] = remember_path(w.eo.lora_file);
  std::string err;
  if (!update_settings([&](ojson& root) { root["last"] = std::move(last); }, &err))
    std::fprintf(stderr, "flyx-server: could not save the last model: %s\n", err.c_str());
}

}  // namespace

bool read_chunks(const ChunkSource& src, std::string* text) {
  text->clear();
  char chunk[1 << 14];
  for (;;) {
    const size_t got = src.read(chunk, sizeof chunk);
    text->append(chunk, got);
    if (got < sizeof chunk) break;  // the end, or an error: which, the stream's own state says
  }
  return !src.error() && src.at_end();
}

bool read_whole_file(const std::string& path, std::string* text) {
#if defined(_WIN32)
  std::FILE* f = _wfopen(to_path(path).c_str(), L"rb");
#else
  std::FILE* f = std::fopen(path.c_str(), "rb");
#endif
  if (!f) return false;
  const bool whole = read_chunks({[f](char* b, size_t n) { return std::fread(b, 1, n, f); },
                                  [f] { return std::ferror(f) != 0; }, [f] { return std::feof(f) != 0; }},
                                 text);
  std::fclose(f);
  return whole;
}

bool want_from_json(const ojson& j, const EngineOptions& base, ModelHost::Want* w, std::string* err) {
  if (!j.is_object() || !j.contains("model") || !j["model"].is_string() || j["model"].get<std::string>().empty())
    return *err = "\"model\": a path or a Hugging Face name (org/name:QUANT) is needed", false;
  ModelHost::Want out;
  out.model = j["model"].get<std::string>();
  out.eo = base;
  out.eo.drafter_files.clear();
  out.eo.mmproj_file.clear();
  out.eo.lora_file.clear();
  if (j.contains("drafters") && !j["drafters"].is_null()) {
    if (!j["drafters"].is_array()) return *err = "\"drafters\": a list of paths or Hub names", false;
    for (const auto& d : j["drafters"]) {
      if (!d.is_string()) return *err = "\"drafters\": a list of paths or Hub names", false;
      if (!d.get<std::string>().empty()) out.drafters.push_back(d.get<std::string>());
    }
  }
  if (j.contains("ctx")) {
    if (j["ctx"].is_null()) {
      out.eo.ctx_max = true;
      out.eo.ctx_given = false;
    } else if (j["ctx"] == "max") {  // the model's own, or the longest the GPU and host memory hold
      out.eo.ctx_max = out.eo.ctx_given = true;
    } else if (!j["ctx"].is_number_integer() || j["ctx"].get<int64_t>() < 0 ||
               j["ctx"].get<int64_t>() > apps::kMaxCtx) {
      return *err = "\"ctx\": tokens, a whole number, or \"max\"", false;
    } else if (j["ctx"].get<int>() > 0) {
      out.eo.ctx_max = false;
      out.eo.max_ctx = j["ctx"].get<int>();
      out.eo.ctx_given = true;
    } else {
      out.eo.ctx_max = true;
      out.eo.ctx_given = false;
    }
  }
  if (j.contains("drafter") && !j["drafter"].is_null()) {
    interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
    if (!j["drafter"].is_string() || !interfaces::parse_drafter_mode(j["drafter"].get<std::string>(), &mode))
      return *err = "\"drafter\": " + interfaces::drafter_mode_words(), false;
    out.eo.drafter = j["drafter"].get<std::string>();
  }
  if (j.contains("speculative") && !j["speculative"].is_null()) {
    if (!j["speculative"].is_boolean()) return *err = "\"speculative\": true or false", false;
    out.eo.speculative = j["speculative"].get<bool>();
  }
  if (j.contains("kv") && !j["kv"].is_null()) {
    if (!j["kv"].is_string()) return *err = "\"kv\": fp8, 16 or auto", false;
    const std::string kv = j["kv"].get<std::string>();
    if (!kv.empty()) {  // "auto": the engine's default (Engine::load), whatever the server's --kv
      if (kv != "fp8" && kv != "16" && kv != "auto") return *err = "\"kv\": fp8, 16 or auto", false;
      out.eo.kv = kv == "fp8" ? FLYX_KV_FP8 : kv == "16" ? FLYX_KV_16 : FLYX_KV_DEFAULT;
    }
  }
  auto parse_gb_field = [&](const char* key, const char* desc, int64_t* dst) {
    if (!j.contains(key) || j[key].is_null()) return true;
    if (j[key].is_string() && j[key].get<std::string>() == "auto") {
      *dst = -1;
      return true;
    }
    if (!j[key].is_number() || !apps::gb_to_bytes(j[key].get<double>(), dst))
      return *err = std::string("\"") + key + "\": " + desc + ", or \"auto\"", false;
    return true;
  };
  // Host memory for the K/V past the GPU's part: GB, or "auto"; absent: the server's own setting.
  if (!parse_gb_field("kv_ram", "GB of host memory", &out.eo.kv_ram)) return false;
  // The K/V positions on the GPU at most: a whole number, or "auto" (as many as fit, whatever the server's --kv-gpu);
  // absent or null: the server's own setting.
  if (j.contains("kv_gpu") && !j["kv_gpu"].is_null()) {
    const bool automatic = j["kv_gpu"].is_string() && j["kv_gpu"].get<std::string>() == "auto";
    if (!automatic && (!j["kv_gpu"].is_number_integer() || j["kv_gpu"].get<int64_t>() < 0 ||
                       j["kv_gpu"].get<int64_t>() > INT32_MAX))
      return *err = "\"kv_gpu\": positions, a whole number, or \"auto\"", false;
    out.eo.kv_gpu = automatic ? -1 : j["kv_gpu"].get<int>();
  }
  if (!parse_gb_field("vram_reserve", "GB to keep free", &out.eo.vram_reserve)) return false;
  if (!parse_gb_field("ram_reserve", "GB to keep free", &out.eo.ram_reserve)) return false;
  if (j.contains("name") && !j["name"].is_null()) {
    if (!j["name"].is_string()) return *err = "\"name\": model id string", false;
    out.name = j["name"].get<std::string>();
  }
  // A LoRA adapter (--lora): a PEFT directory on the server's machine or a Hub name; absent or null: none.
  if (j.contains("lora") && !j["lora"].is_null()) {
    if (!j["lora"].is_string()) return *err = "\"lora\": a LoRA adapter's directory or Hugging Face name, or null", false;
    out.eo.lora_file = j["lora"].get<std::string>();
  }
  bool mmproj_given = false;
  if (j.contains("mmproj")) {
    if (j["mmproj"].is_null()) {
      out.eo.mmproj_file.clear();
    } else {
      if (!j["mmproj"].is_string()) return *err = "\"mmproj\": path to vision projector GGUF", false;
      out.eo.mmproj_file = j["mmproj"].get<std::string>();
      mmproj_given = !out.eo.mmproj_file.empty();
    }
  }
  const bool vision_given = j.contains("vision") && !j["vision"].is_null();
  if (vision_given) {
    if (!j["vision"].is_boolean()) return *err = "\"vision\": true or false", false;
    out.eo.vision = j["vision"].get<bool>();
  } else if (mmproj_given) {
    out.eo.vision = true;
  }
  if (!out.eo.vision) out.eo.mmproj_file.clear();
  if (j.contains("vision_device") && !j["vision_device"].is_null()) {
    if (!j["vision_device"].is_string()) return *err = "\"vision_device\": disk, host, gpu or auto", false;
    const std::string vd = j["vision_device"].get<std::string>();
    if (!interfaces::parse_vision_device(vd, nullptr, &out.eo.vision_device, err)) {
      *err = "\"vision_device\": disk, host, gpu or auto";
      return false;
    }
  }
  if (j.contains("vision_quant") && !j["vision_quant"].is_null()) {
    if (!j["vision_quant"].is_string()) return *err = "\"vision_quant\": q8, exact or nvfp4", false;
    const std::string vq = j["vision_quant"].get<std::string>();
    if (!interfaces::parse_vision_quant(vq, nullptr, &out.eo.vision_quant, err)) {
      *err = "\"vision_quant\": q8, exact or nvfp4";
      return false;
    }
  }
  if (j.contains("vision_max_tokens")) {
    if (j["vision_max_tokens"].is_null()) {
      out.eo.vision_max_tokens = 0;
    } else if (!j["vision_max_tokens"].is_number_integer()) {
      return *err = "\"vision_max_tokens\": whole number >= 16, or null for auto", false;
    } else {
      const int64_t n = j["vision_max_tokens"].get<int64_t>();
      if (n != 0 && (n < 16 || n > apps::kMaxCtx))
        return *err = "\"vision_max_tokens\": 16 to " + std::to_string(apps::kMaxCtx) + ", or null for auto", false;
      out.eo.vision_max_tokens = int(n);
    }
  }
  // Drafting, as --width, --draft-trees and --no-lookup set it; absent or null: the server's own setting.
  if (j.contains("width") && !j["width"].is_null()) {
    if (!j["width"].is_number_integer() || j["width"].get<int64_t>() < 0 ||
        j["width"].get<int64_t>() > apps::kMaxDraftWidth)
      return *err = "\"width\": drafts verified a round, 0 to 64, or null to choose each round", false;
    out.eo.width = j["width"].get<int>();
  }
  if (j.contains("draft_trees") && !j["draft_trees"].is_null()) {
    if (!j["draft_trees"].is_boolean()) return *err = "\"draft_trees\": true or false", false;
    out.eo.draft_trees = j["draft_trees"].get<bool>();
    out.eo.draft_trees_given = true;
  }
  if (j.contains("lookup") && !j["lookup"].is_null()) {
    if (!j["lookup"].is_boolean()) return *err = "\"lookup\": true or false", false;
    out.eo.lookup = j["lookup"].get<bool>();
    out.eo.lookup_given = true;
  }
  // The drafters' CUDA backend options (--backend-opt draft_head=..., draft_weights=...), overriding any value from
  // the server's command line. Explicit null/default resets to the default so the dashboard can clear CLI overrides.
  // Skip appending CUDA-only options when running on cpu_ref, which rejects unknown backend options.
  auto drafting_option = [&](const char* key, const char* a, const char* b, std::string* to) {
    if (!j.contains(key)) return true;
    if (j[key].is_null()) {
      out.eo.backend_options = interfaces::backend_opt_strip(out.eo.backend_options, key);
      to->clear();
      return true;
    }
    const std::string v = j[key].is_string() ? j[key].get<std::string>() : "";
    if (v != a && v != b) return *err = std::string("\"") + key + "\": " + a + " or " + b, false;
    *to = v == a ? "" : v;
    if (out.eo.backend != "cpu_ref")
      out.eo.backend_options = interfaces::backend_opt_set(out.eo.backend_options, key, v);
    else
      out.eo.backend_options = interfaces::backend_opt_strip(out.eo.backend_options, key);
    return true;
  };
  if (!drafting_option("draft_head", "pick", "full", &out.draft_head)) return false;
  if (j.contains("draft_weights")) {
    if (j["draft_weights"].is_null()) {
      out.draft_weights.clear();
      out.eo.backend_options = interfaces::backend_opt_strip(out.eo.backend_options, "draft_weights");
    } else {
      if (!j["draft_weights"].is_string()) return *err = "\"draft_weights\": auto, nvfp4, q8 or exact", false;
      std::string canon;
      if (!interfaces::parse_draft_weights(j["draft_weights"].get<std::string>(), nullptr, &canon, nullptr))
        return *err = "\"draft_weights\": auto, nvfp4, q8 or exact", false;
      out.draft_weights = canon == "auto" ? "" : canon;
      if (canon != "auto")
        out.eo.backend_options = interfaces::backend_opt_set(out.eo.backend_options, "draft_weights", canon);
      else
        out.eo.backend_options = interfaces::backend_opt_strip(out.eo.backend_options, "draft_weights");
    }
  }
  *w = std::move(out);
  return true;
}

std::string ModelHost::settings_path() {
  const std::string base = os::config_dir();
  return base.empty() ? "" : base + "/flyx/serve.json";
}

bool ModelHost::remembered(Want* w, const EngineOptions& base) {
  const ojson s = read_settings();
  std::string err;
  return s.is_object() && s.contains("last") && want_from_json(s["last"], base, w, &err);
}

std::string ModelHost::remembered_hub_endpoint() {
  const ojson s = read_settings();
  if (s.is_object() && s.contains("hub_endpoint") && s["hub_endpoint"].is_string())
    return s["hub_endpoint"].get<std::string>();
  return "";
}

bool ModelHost::save_hub_endpoint(const std::string& endpoint, std::string* err) {
  return update_settings(
      [&](ojson& root) {
        if (endpoint.empty()) root.erase("hub_endpoint");
        else root["hub_endpoint"] = endpoint;
      },
      err);
}

bool ModelHost::lan_access() { return read_network_settings().lan_access; }

bool ModelHost::remote_manage() { return read_network_settings().remote_manage; }

namespace {
ModelHost::NetworkSettingsSnapshot snapshot_of(const ojson& root,
                                               const std::function<bool()>& read_live_remote_manage) {
  ModelHost::NetworkSettingsSnapshot snap;
  if (root.is_object()) {
    snap.lan_access = root.contains("lan_access") && root["lan_access"] == true;
    snap.remote_manage = root.contains("remote_manage") && root["remote_manage"] == true;
  }
  snap.live_remote_manage = read_live_remote_manage ? read_live_remote_manage() : snap.remote_manage;
  return snap;
}
}  // namespace

ModelHost::NetworkSettingsSnapshot ModelHost::read_network_settings(
    const std::function<bool()>& read_live_remote_manage) {
  SettingsLock lock(kSettingsReadWaitMs);
  std::string why;
  const ojson root = read_settings_with(lock, &why);
  NetworkSettingsSnapshot snap = snapshot_of(root, read_live_remote_manage);
  snap.error = why;
  return snap;
}

bool ModelHost::save_network_settings(std::optional<bool> lan_access, std::optional<bool> remote_manage,
                                      const std::function<void(bool)>& on_remote_manage_committed,
                                      NetworkSettingsSnapshot* out_snapshot, std::string* err,
                                      const std::function<bool()>& read_live_remote_manage) {
  if (settings_path().empty()) {
    *err = "no configuration directory available to save network settings";
    return false;
  }
  return update_settings(
      [&](ojson& root) {
        if (lan_access.has_value()) root["lan_access"] = *lan_access;
        if (remote_manage.has_value()) root["remote_manage"] = *remote_manage;
      },
      err,
      [&](const ojson& root) {
        if (remote_manage.has_value() && on_remote_manage_committed) on_remote_manage_committed(*remote_manage);
        if (out_snapshot) *out_snapshot = snapshot_of(root, read_live_remote_manage);
      });
}

ModelHost::~ModelHost() {
  stop_ = true;
  if (worker_.joinable()) worker_.join();
}

ModelHost::Served ModelHost::served() const {
  std::lock_guard<std::mutex> lock(mu_);
  return served_;
}

void ModelHost::cancel() { stop_ = true; }

bool ModelHost::try_begin_locked(std::string* err) {
  if (busy_) return *err = "a model is being loaded (" + target_ + ")", false;
  busy_ = true;
  return true;
}

void ModelHost::reset_load_state_locked(const std::string& state, const std::string& target) {
  state_ = state;
  target_ = target;
  error_.clear();
  notes_.clear();
  progress_ = {};
  since_ms_ = os::now_ms();
}

bool ModelHost::unload(std::string* err) {
  std::shared_ptr<Scheduler> old;
  std::string name;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    old = std::move(served_.sched);
    name = served_.name;
    served_ = {}, fetched_ = {};
    reset_load_state_locked("none", "");
  }
  BusyGuard guard(this);
  if (old) std::fprintf(stderr, "flyx-server: unloading %s\n", name.c_str());
  retire(std::move(old));
  return true;
}

bool ModelHost::with_files_in_use(const std::function<void(const std::vector<std::string>&)>& fn, std::string* err) {
  std::vector<std::string> in_use;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    if (served_.sched) {
      in_use.push_back(fetched_.model);
      in_use.insert(in_use.end(), fetched_.drafters.begin(), fetched_.drafters.end());
      if (!fetched_.mmproj.empty()) in_use.push_back(fetched_.mmproj);
      if (!fetched_.lora.empty()) in_use.push_back(fetched_.lora);
    }
  }
  BusyGuard guard(this);
  fn(in_use);
  return true;
}

std::shared_ptr<Scheduler> ModelHost::open(const std::string& model, const apps::Fetched& f,
                                           const EngineOptions& options, double* load_s, std::string* err) const {
  EngineOptions eo = options;
  eo.drafter_files = f.drafters;
  eo.mmproj_file = f.mmproj;
  eo.lora_file = f.lora;
  std::fprintf(stderr, "flyx-server: loading %s\n", f.model.c_str());
  const double t0 = os::now_ms();
  // Each request that runs to its end and drafts counts for the speed of the Hub drafters the model drafts with
  // (drafting: once loaded, the kinds the engine drafts with), by model: a Hub name as named, a path made absolute.
  SchedulerOptions so = so_;
  std::vector<std::pair<std::string, std::string>> hub_drafters;  // repository, kind
  std::error_code ec;
  for (const apps::DrafterSource& d : f.drafter_sources) {
    hub::Name n;
    if (d.kind != "mtp" && !fs::exists(to_path(d.from), ec) && hub::parse_name(d.from, &n))
      hub_drafters.push_back({n.repo, d.kind});
  }
  const std::string model_key = speed_model(model, f.model);
  auto drafting = std::make_shared<std::vector<std::string>>();  // filled once the engine says what it drafts with
  auto drafting_mu = std::make_shared<std::mutex>();
  if (eo.speculative && !hub_drafters.empty())
    so.on_finished = [model_key, drafting, drafting_mu](const RequestRecord& r) {
      if (r.status == "cancelled" || r.status == "error" || r.drafted <= 0) return;
      std::lock_guard<std::mutex> lock(*drafting_mu);
      add_drafter_speed(model_key, *drafting, r.output_tokens, r.decode_ms);
    };
  std::shared_ptr<Scheduler> sched;
  try {
    // A deleter that says when it is gone, for retire() to wait on before the next model loads.
    sched = std::shared_ptr<Scheduler>(new Scheduler(f.model, eo, so), SignalTeardown{std::make_shared<Teardown>()});
  } catch (const std::exception& e) {
    // A model file no check foresaw: a load that failed with its reason, not the end of the server.
    *err = e.what();
    std::fprintf(stderr, "flyx-server: %s: %s\n", f.model.c_str(), err->c_str());
    return nullptr;
  }
  if (!sched->ok(err)) {
    std::fprintf(stderr, "flyx-server: %s: %s\n", f.model.c_str(), err->c_str());
    return nullptr;
  }
  if (so.on_finished && sched->engine()) {  // the Hub drafters among those the engine drafts with ("mtp+dspark")
    const std::string with = std::string("+") + sched->engine()->drafter() + "+";
    std::lock_guard<std::mutex> lock(*drafting_mu);
    for (const auto& [repo, kind] : hub_drafters)
      if (with.find("+" + kind + "+") != std::string::npos) drafting->push_back(repo);
  }
  *load_s = (os::now_ms() - t0) / 1000;
  return sched;
}

bool ModelHost::start(const Want& w, std::string* err) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!try_begin_locked(err)) return false;
  stop_ = false;
  reset_load_state_locked("fetching", w.model);
  if (worker_.joinable()) worker_.join();  // the last one is done: it let go of busy_ as it ended
  worker_ = std::thread([this, w] {
    BusyGuard guard(this);
    std::string e;
    try {
      work(w, &e);
    } catch (const std::exception& ex) {  // e.g. no memory for what a Hub listing declared
      std::fprintf(stderr, "flyx-server: %s: %s\n", w.model.c_str(), ex.what());
      std::lock_guard<std::mutex> lock(mu_);
      error_ = w.model + ": " + ex.what(), state_ = served_.sched ? "ready" : "failed";
    }
  });
  return true;
}

bool ModelHost::run(const Want& w, std::string* err) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (!try_begin_locked(err)) return false;
    stop_ = false;
    reset_load_state_locked("fetching", w.model);
  }
  BusyGuard guard(this);
  return work(w, err);
}

bool ModelHost::work(const Want& w, std::string* err) {
  apps::Fetched f;
  auto logged = apps::make_progress_logger("flyx-server", 5000, [this](const hub::Progress& p) {
    std::lock_guard<std::mutex> lock(mu_);
    progress_ = p;
  });
  if (!apps::fetch_model(w.model, w.drafters, apps::wants_mtp(w.eo), logged, &stop_, &f, err, w.eo.mmproj_file,
                         w.eo.vision, apps::wants_dspark(w.eo, w.drafters), w.eo.lora_file, w.eo.allow_path)) {
    std::fprintf(stderr, "flyx-server: %s: %s\n", w.model.c_str(), err->c_str());
    std::lock_guard<std::mutex> lock(mu_);
    error_ = *err, state_ = served_.sched ? "ready" : "failed";
    return false;
  }
  for (const std::string& n : f.notes) std::fprintf(stderr, "flyx-server: %s\n", n.c_str());
  // Unload the model served now so its GPU memory is free before the new one loads.
  std::shared_ptr<Scheduler> old;
  {
    std::lock_guard<std::mutex> lock(mu_);
    state_ = "loading", notes_ = f.notes;
    old = std::move(served_.sched);
    served_ = {}, fetched_ = {};
  }
  const bool switching = old != nullptr;
  retire(std::move(old));
  std::string why;
  double load_s = 0;
  auto sched = open(w.model, f, w.eo, &load_s, &why);
  if (!sched) {
    std::fprintf(stderr, "flyx-server: %s did not load: none served\n", w.model.c_str());
    std::lock_guard<std::mutex> lock(mu_);
    *err = error_ = switching ? (w.model + ": " + why) : why;
    state_ = "failed";
    return false;
  }
  const std::string name = !w.name.empty() ? w.name : f.name;
  log_loaded(*sched, name, load_s);
  {
    std::lock_guard<std::mutex> lock(mu_);
    served_ = {sched, name}, fetched_ = std::move(f);
    state_ = "ready", last_ = w;
    last_.eo.allow_path = {};
  }
  save_settings(w);
  return true;
}

ojson ModelHost::status() const {
  // A slow settings drive must not hold up requests for the model already in memory.
  const ojson s = read_settings();
  std::shared_ptr<Scheduler> active_sched;
  {
    std::lock_guard<std::mutex> lock(mu_);
    active_sched = served_.sched;
  }
  ModelInfo active_model;
  bool has_active = false;
  if (active_sched) {
    TelemetrySnapshot snap;
    flyx_device_stats dev{};
    bool dev_ok = false;
    active_sched->telemetry(&snap, &dev, &dev_ok);
    if (snap.loaded) active_model = std::move(snap.model), has_active = true;
  }
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<apps::DrafterSource> active_drafters;
  if (served_.sched) {
    for (const apps::DrafterSource& d : fetched_.drafter_sources)
      if (!has_active || drafter_active(active_model.drafter, d.kind)) active_drafters.push_back(d);
  }
  ojson j = {{"state", state_},
             {"model", served_.sched ? ojson(served_.name) : ojson(nullptr)},
             {"target", target_.empty() ? ojson(nullptr) : ojson(target_)},
             {"error", error_.empty() ? ojson(nullptr) : ojson(error_)},
             {"notes", notes_},
             {"fallback", ojson(nullptr)},
             {"degraded", served_.sched && has_active ? degraded_json(active_model.load_adjustments) : ojson::array()},
             {"drafters", drafters_json(active_drafters, "from")},
             {"suggested", served_.sched
                               ? known_drafters_json(speed_model(last_.model, fetched_.model), fetched_.suggested)
                               : ojson::array()},
             {"vision_projector",
              served_.sched && served_.sched->has_vision() && !fetched_.mmproj_source.empty()
                  ? ojson({{"from", fetched_.mmproj_source},
                           {"size", fetched_.mmproj_size ? ojson(*fetched_.mmproj_size) : ojson(nullptr)}})
                  : ojson(nullptr)}};
  if (busy_) j["elapsed_s"] = (os::now_ms() - since_ms_) / 1000;
  if (state_ == "fetching")
    j["progress"] = {{"file", progress_.file}, {"done", progress_.done}, {"total", progress_.total},
                     {"bytes_per_s", progress_.bytes_per_s}, {"checking", progress_.checking}};
  if (served_.sched) j["loaded"] = want_json(last_);
  // The file served, absolute: what a library row of it is, whatever name or path the load was given ("loaded" keeps
  // that, for a reload).
  if (served_.sched && !fetched_.model.empty()) {
    std::error_code ec;
    const fs::path file = fs::absolute(to_path(fetched_.model), ec).lexically_normal();
    j["loaded_file"] = ec ? ojson(nullptr) : ojson(os::from_path(file));
  }
  j["remembered"] = s.is_object() && s.contains("last") ? s["last"] : ojson(nullptr);
  return j;
}

void log_loaded(const Scheduler& sched, const std::string& name, double load_s) {
  TelemetrySnapshot snap;
  flyx_device_stats dev{};
  bool dev_ok = false;
  sched.telemetry(&snap, &dev, &dev_ok);
  const ModelInfo& m = snap.model;
  const std::string dw = m.draft_weights.empty() ? "" : (m.draft_weights + ", ");
  std::fprintf(stderr, "flyx-server: %s on %s (%s), context %d, drafter %s (%sup to %d per round), loaded in %.1f s\n",
               name.c_str(), m.device.c_str(), m.backend.c_str(), m.max_ctx, m.drafter.c_str(), dw.c_str(), m.width,
               load_s);
  std::fprintf(stderr, "flyx-server: memory: %s; up to %d session%s\n", m.memory.c_str(),
               m.max_sessions, m.max_sessions == 1 ? "" : "s");
  if (m.vision) {
    std::fprintf(stderr, "flyx-server: vision: %s (%s)%s\n", m.vision_device.c_str(), m.vision_quant.c_str(),
                 m.vision_max_tokens > 0 ? (" max " + std::to_string(m.vision_max_tokens) + " tok/img").c_str() : "");
  }
  for (const LoadAdjustment& a : m.load_adjustments)
    std::fprintf(stderr, "flyx-server: VRAM adjustment: %s\n", a.detail.c_str());
  std::fprintf(stderr, "flyx-server: chat template: %s\n",
               !m.chat_ok ? "none (chat endpoints answer 400)"
               : m.chat_native
                   ? (m.chat_format + " (its own renderer)").c_str()
                   : (m.chat_format + " (Jinja engine)").c_str());
}

}  // namespace flyx::server
