// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/engine.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <thread>

#include "core/exec/generation.h"
#include "core/interfaces/options.h"
#include "core/os/env.h"
#include "core/os/log.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "core/speculation/calibrate.h"

namespace flyx {

namespace {

int32_t to_abi_vision_device(interfaces::VisionDeviceMode m) {
  switch (m) {
    case interfaces::VisionDeviceMode::kDisk: return FLYX_VISION_DEV_DISK;
    case interfaces::VisionDeviceMode::kHost: return FLYX_VISION_DEV_HOST;
    case interfaces::VisionDeviceMode::kGpu: return FLYX_VISION_DEV_GPU;
    case interfaces::VisionDeviceMode::kAuto: return FLYX_VISION_DEV_AUTO;
  }
  return FLYX_VISION_DEV_DISK;
}

int32_t to_abi_vision_quant(interfaces::VisionQuantMode m) {
  switch (m) {
    case interfaces::VisionQuantMode::kQ8: return FLYX_VISION_QUANT_Q8;
    case interfaces::VisionQuantMode::kExact: return FLYX_VISION_QUANT_EXACT;
    case interfaces::VisionQuantMode::kNvfp4: return FLYX_VISION_QUANT_NVFP4;
  }
  return FLYX_VISION_QUANT_Q8;
}

int32_t to_abi_draft_weights(interfaces::DraftWeightsMode m) {
  switch (m) {
    case interfaces::DraftWeightsMode::kAuto: return FLYX_DRAFT_WEIGHTS_AUTO;
    case interfaces::DraftWeightsMode::kNvfp4: return FLYX_DRAFT_WEIGHTS_NVFP4;
    case interfaces::DraftWeightsMode::kQ8: return FLYX_DRAFT_WEIGHTS_Q8;
    case interfaces::DraftWeightsMode::kExact: return FLYX_DRAFT_WEIGHTS_EXACT;
  }
  return FLYX_DRAFT_WEIGHTS_AUTO;
}

// Require meaningful VRAM savings (>= 128 MiB and >= 10% of drafter weights) before elastic loading steps
// draft_weights down to NVFP4 — avoids spurious downgrades on already-4-bit GGUF drafters like Q4_K_M.
bool meaningful_draft_savings(const flyx_load_estimate& est) {
  constexpr uint64_t kMinSavingsBytes = 128ull << 20;
  return est.draft_compact_savings >= kMinSavingsBytes &&
         est.draft_compact_savings * 10 >= est.draft_weights_bytes;
}

bool set_kv_method(const EngineOptions& opt, plan::KvTiers* tiers, std::string* err) {
  interfaces::KvMethodSpec spec;
  if (!interfaces::parse_kv_method(opt.kv_method, &spec, err)) return false;
  if (spec.kind == interfaces::KvMethodKind::kExact) return true;
  const bool strata = spec.kind == interfaces::KvMethodKind::kStrata;
  if (opt.draft_trees && strata)
    return *err = "--draft-trees: strata attention reads chains of drafts only", false;
  // Trees of drafts read every position even in auto mode.
  if (!opt.draft_trees) tiers->method = strata ? FLYX_KV_STRATA : FLYX_KV_AUTO;
  tiers->method_arg = spec.bits;
  return true;
}

// A model file read from disk on a thread of its own while the load runs (ModelFile::read_ahead): the backend's
// upload then finds its bytes in memory instead of reading each page as it first touches it (the 27B's 2-bit file
// from an uncached NVMe disk: 2.6 GB/s by page faults, 6 GB/s read ahead). Only for files that fit
// in the memory available, so that reading ahead does not push out what the upload has yet to read.
class ReadAhead {
 public:
  explicit ReadAhead(const ModelFile& file) : available_(os::system_memory().available) {
    if (available_ && file.bytes() < available_) thread_ = std::thread([this, &file] { file.read_ahead(stop_); });
  }
  ~ReadAhead() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
  }
  // For the load's report.
  std::string what() const {
    return thread_.joinable() ? "read ahead" : "not read ahead: " + gb(available_) + " of memory available";
  }

 private:
  const uint64_t available_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// The context a round's cost is measured at a second time (its cost per token of context from the two): 1024
// tokens, a multiple of the prefill alignment (a checkpoint there brings the scratch sequence back), where the
// context takes it; 0: none (a context too short to tell).
int long_calibration(int max_ctx, int align) {
  const int at = std::min(1024, max_ctx - 640) / align * align;
  return at >= 256 ? at : 0;
}

std::string model_drafters_str(const interfaces::Model& m) {
  std::string out;
  for (int d = 0; d < m.drafters(); ++d) out += (out.empty() ? "" : "+") + std::string(m.drafter_name(d));
  return out;
}

bool model_has_drafter(const interfaces::Model& m, std::string_view name) {
  for (int d = 0; d < m.drafters(); ++d)
    if (name == m.drafter_name(d)) return true;
  return false;
}

}  // namespace

Engine::~Engine() {
  if (api_) {
    plans_.free(api_);
    if (bm_) api_->close(bm_);
  }
}

void apply_environment(EngineOptions* o) {
  if (o->backend.empty()) o->backend = os::getenv_str(env::kBackend);
  if (o->backend_dir.empty()) o->backend_dir = os::getenv_str(env::kBackendDir);
  if (o->drafter.empty()) o->drafter = os::getenv_str(env::kDrafter);
  if (o->activations.empty()) o->activations = os::getenv_str(env::kActivations);
  const std::string env_opts = os::getenv_str(env::kBackendOptions);
  if (!env_opts.empty()) o->backend_options = o->backend_options.empty() ? env_opts : env_opts + "," + o->backend_options;
  if (!o->chat_native_given)
    if (const auto v = os::env_bool(env::kChatNative)) o->chat_native = *v;
  if (!o->lookup_given)
    if (const auto v = os::env_bool(env::kLookup)) o->lookup = *v;
  if (!o->draft_trees_given)
    if (const auto v = os::env_bool(env::kDraftTrees)) {
      o->draft_trees = *v;
      o->draft_trees_given = true;
    }
  if (!o->adaptive_given)
    if (const auto v = os::env_bool(env::kAdaptive)) o->adaptive = *v;
}

namespace {

std::mutex& builtin_backends_mu() {
  static std::mutex mu;
  return mu;
}

std::vector<std::pair<std::string, flyx_backend_entry_fn>>& builtin_backends() {
  static std::vector<std::pair<std::string, flyx_backend_entry_fn>> list;
  return list;
}

flyx_backend_entry_fn find_builtin_backend(const std::string& name) {
  std::lock_guard<std::mutex> lock(builtin_backends_mu());
  for (const auto& [n, fn] : builtin_backends())
    if (n == name) return fn;
  return nullptr;
}

}  // namespace

bool open_backend(const EngineOptions& opt, std::unique_ptr<os::DynLib>* out_lib, const flyx_backend_api** out_api,
                  flyx_device_info* out_info, std::string* err) {
  std::string dir = opt.backend_dir.empty() ? os::executable_dir() : opt.backend_dir;
  std::vector<std::string> names;
  std::string want = opt.backend;
  if (!want.empty()) names = {want};
  else names = {"cuda", "cpu_ref"};
  std::string tried;
  for (const auto& n : names) {
    if (opt.backend_dir.empty()) {
      if (auto entry = find_builtin_backend(n)) {
        const flyx_backend_api* api = entry();
        if (!api || api->abi_version != FLYX_BACKEND_ABI_VERSION || api->struct_size < sizeof(flyx_backend_api)) {
          tried += "\n  built-in " + n + ": incompatible backend ABI";
          continue;
        }
        flyx_device_info info{};
        if (api->probe(&info) != 0) {
          tried += "\n  built-in " + n + ": " + api->last_error();
          continue;
        }
        if (n != names.front())
          log::warn("flyx: the %s backend is not usable here, so %s runs:%s", names.front().c_str(), n.c_str(),
                    tried.c_str());
        *out_lib = nullptr;
        *out_api = api;
        *out_info = info;
        return true;
      }
    }
    // build tree (bin/), release package (bin/ -> ../lib/flyx, lib/ -> flyx/), wheel (_lib/flyx)
    const std::string file = "libflyx-" + n + os::kDynLibSuffix;
    std::vector<std::string> paths = {dir + "/" + file, dir + "/../lib/flyx/" + file, dir + "/flyx/" + file,
                                      dir + "/backends/" + file};
    std::string failures;
    for (const auto& path : paths) {
      std::string e;
      auto lib = std::make_unique<os::DynLib>();
      if (!lib->open(path, &e)) {
        // A missing dependency can produce the same loader error as a missing backend library.
        std::error_code ec;
        if (std::filesystem::status(os::to_path(path), ec).type() != std::filesystem::file_type::not_found)
          failures += "\n  " + path + ": " + e;
        continue;
      }
      auto entry = reinterpret_cast<flyx_backend_entry_fn>(lib->symbol(FLYX_BACKEND_ENTRY));
      const flyx_backend_api* api = entry ? entry() : nullptr;
      if (!api || api->abi_version != FLYX_BACKEND_ABI_VERSION || api->struct_size < sizeof(flyx_backend_api)) {
        failures += "\n  " + path + ": incompatible backend ABI";
        continue;
      }
      flyx_device_info info{};
      if (api->probe(&info) != 0) {
        failures += "\n  " + path + ": " + api->last_error();
        continue;
      }
      // A backend passed over is said without -v: on Windows, Smart App Control once refused an unsigned
      // libflyx-cuda.dll and a real model ran on the float64 reference, a minute a 6000-token prefill, unexplained.
      if (n != names.front())
        log::warn("flyx: the %s backend is not usable here, so %s runs:%s", names.front().c_str(), n.c_str(),
                  tried.c_str());
      *out_lib = std::move(lib);
      *out_api = api;
      *out_info = info;
      return true;
    }
    tried += failures.empty() ? "\n  " + file + " not found in backend paths relative to " + dir : failures;
  }
  *err = "no usable backend" + tried;
  return false;
}

bool probe_device(const EngineOptions& o, std::string* backend, flyx_device_info* info, std::string* err) {
  static std::mutex mu;
  static std::unique_ptr<os::DynLib> kept;  // never unloaded: a CUDA runtime is not to be loaded twice
  std::lock_guard<std::mutex> lock(mu);
  EngineOptions opt = o;
  apply_environment(&opt);
  std::unique_ptr<os::DynLib> lib;
  const flyx_backend_api* api = nullptr;
  if (!open_backend(opt, &lib, &api, info, err)) return false;
  *backend = api->name;
  if (!kept) kept = std::move(lib);
  return true;
}

void register_builtin_backend(const char* name, flyx_backend_entry_fn entry) {
  if (!name || !*name || !entry) return;
  std::lock_guard<std::mutex> lock(builtin_backends_mu());
  auto& list = builtin_backends();
  for (auto& [n, fn] : list) {
    if (n == name) {
      fn = entry;
      return;
    }
  }
  list.emplace_back(name, entry);
}

bool Engine::load_backend(std::string* err) {
  flyx_device_info info{};
  if (!open_backend(opt_, &lib_, &api_, &info, err)) return false;  // lib_ stays loaded for the engine's lifetime
  if (opt_.verbose) log::info("flyx: backend %s on %s", api_->name, info.name);
  backend_name_ = api_->name;
  device_ = info;
  return true;
}

Grammar& Engine::grammar() {
  std::lock_guard<std::mutex> lock(grammar_mu_);
  if (!grammar_.ready()) {
    std::vector<int32_t> markers;  // the special tokens the chat format writes replies with
    for (const std::string& text : chat_.reply_tokens())
      if (const int32_t id = tok_.added_id(text); tok_.special(id)) markers.push_back(id);
    grammar_.init(tok_, model_->desc().vocab, eos_, markers);
  }
  return grammar_;
}

bool Engine::build_and_calibrate_plans(double* t_plans, double* t_rounds, std::string* err) {
  // Prefill passes run whole multiples of the backend's alignment (numerics.md: K).
  prefill_align_ = std::max(1, int(api_->prefill_align(bm_)));
  int chunk = opt_.prefill_chunk > 0 ? opt_.prefill_chunk : auto_prefill_chunk();
  if (prefill_cap_ > 0) chunk = std::min(chunk, prefill_cap_);
  prefill_chunk_ = std::max(prefill_align_, chunk / prefill_align_ * prefill_align_);

  // Lookup rounds are speculation too: none when the caller asks for plain decoding.
  if (!plan::build(api_, bm_, *model_, prefill_chunk_, opt_.width, opt_.lookup && opt_.speculative, opt_.draft_trees,
                   &plans_, err, opt_.adaptive))
    return false;
  // A drafter whose rounds the backend runs at no width is not in use: named by the caller, the load fails (as a
  // drafter's refused round does, plans.h); chosen by default, the rounds are the ones without it.
  const int refused = plans_.refused_drafter;
  if (refused >= 0 && opt_.drafter == model_->drafter_name(refused))
    return *err = "--drafter " + opt_.drafter + ": the backend does not run its rounds (" + plans_.refused + ")",
           false;
  // A measurement's rounds (--cycle) are indices into these plans: one past them is refused here, by the rounds there
  // are, not left to index past them on every reply.
  for (const size_t i : opt_.cycle)
    if (i >= plans_.widths.size()) {
      *err = "--cycle " + std::to_string(i) + ": the rounds are";
      for (size_t k = 0; k < plans_.widths.size(); ++k)
        *err += (k ? ", " : " ") + std::to_string(k) + " (" + round_name(k) + ")";
      return false;
    }
  drafter_.clear();
  for (int d = 0; d < model_->drafters(); ++d)
    if (d != refused) drafter_ += (drafter_.empty() ? "" : "+") + std::string(model_->drafter_name(d));
  if (drafter_.empty()) drafter_ = "none";
  round_ms_.assign(plans_.widths.size(), 1.0);
  round_ms_per_token_.assign(plans_.widths.size(), 0.0);
  round_tokens_.resize(plans_.widths.size());
  round_ms_at_ = round_ms_again_at_ = 0;
  round_trunks_.assign(plans_.widths.size(), 0);
  // A tree's trunk: the chain one pass of its drafter's block makes (ops/draft/ref.h); a drafter that drafts nothing
  // but trees has none.
  for (size_t i = 0; i < plans_.widths.size(); ++i)
    if (plans_.drafter_trees[i]) {
      const int chain = model_->block_drafts(plans_.drafters[i]);
      round_trunks_[i] = model_->drafts_trees(plans_.drafters[i])
                             ? -1
                             : std::min(plans_.widths[i], chain > 0 ? chain : plans_.widths[i]);
    }
  *t_plans = os::now_ms();
  // The rounds are timed where this prompt ends, on the drafts that follow it.  Prose, not a count: after "1, 2, 3, 4,
  // 5, 6," the drafts are digits and commas, and a sparse-MoE target routes a round of those few tokens to few experts,
  // so its wide rounds timed cheap (Inkling-2-Mini: 2 -> 3 drafts +0.7 ms timed, +1.5 ms on chat prompts).  A tokenizer
  // that makes more than 64 tokens of it keeps its end, and the longer context below ends with it too.
  std::vector<int32_t> prompt = tok_.encode("The quick brown fox jumps over the lazy dog. In the beginning was the word, and "
                                            "the word was with the engine, and the engine was fast. When the evening "
                                            "came, the travellers");
  if (prompt.size() > 64) prompt.erase(prompt.begin(), prompt.end() - 64);
  std::vector<std::string> labels, given_labels;
  for (size_t i = 0; i < plans_.widths.size(); ++i) labels.push_back(round_name(i));
  for (int k : plans_.given_widths) given_labels.push_back(std::to_string(k));
  // Every round runs at load, timed where the costs choose among them: a kernel's first launch loads its module, and
  // the first request of a process waited for that (the 27B's 2-bit file, plain decoding: 200 to 350 ms of its first
  // rounds).
  lookup_ms_.clear();
  if (plans_.widths.size() > 1 || !plans_.given.empty()) {
    // Where a drafter's state follows its own rounds only (DV's canvas: its warm-up among the rounds), the controller
    // enters and leaves it by the rounds' costs at the context they run at: every round timed
    // twice, the faster kept, and again after a longer context.  Other models' rounds are timed once at the prompt's
    // context, the costs their choice was measured with: timed so too, Qwen3.5-4B NVFP4's (plain decoding and MTP 1 to
    // 7) took 3.8 s of a 5.6 s load instead of 1.2 of 3.1, the costs at 1024 tokens within 0.4 ms of those at 64.
    bool restarting = false;
    for (size_t i = 0; i < plans_.widths.size(); ++i)
      restarting = restarting || (plans_.drafters[i] >= 0 && plans_.widths[i] == 0);
    std::vector<spec::TimedGroup> groups = {{&plans_.rounds, &labels, "round", &round_ms_, restarting ? 2 : 1}};
    if (!plans_.given.empty()) groups.push_back({&plans_.given, &given_labels, "lookup round", &lookup_ms_});
    spec::time_rounds(api_, bm_, plans_.prefill, prefill_chunk_, prompt, groups, opt_.verbose);
    round_ms_at_ = static_cast<int>(prompt.size());
    // And after a longer context: a round's attention reads every key, the more the more rows it has (DV's 32 verify
    // rows and 32 canvas rows a round of 31).  Its cost per token of context past the first.
    const int longer = restarting ? long_calibration(opt_.max_ctx, prefill_align_) : 0;
    if (longer > round_ms_at_ && !prompt.empty()) {
      std::vector<int32_t> context;  // the prompt repeated, its last copy whole: both timings draft after its end
      context.reserve(static_cast<size_t>(longer) + prompt.size());
      while (static_cast<int>(context.size()) < longer) context.insert(context.end(), prompt.begin(), prompt.end());
      context.erase(context.begin(), context.end() - longer);
      const std::string what = "round at " + std::to_string(longer) + " tokens";
      auto time_at = [&](const std::vector<int32_t>& tokens, const std::string& label) {
        std::vector<double> ms;
        spec::time_rounds(api_, bm_, plans_.prefill, prefill_chunk_, tokens,
                          {{&plans_.rounds, &labels, label.c_str(), &ms, 2}}, opt_.verbose);
        return ms;
      };
      std::vector<double> at = time_at(context, what);
      // Timings another program's work disturbed would steer the choice for the engine's life (a busy CPU delays a
      // round's launch or its wait now and then, by a round's time or more: plain decoding once timed 14.6 ms at 1024
      // tokens against 4.4 at 64): timed again, twice at most, the faster of each timing kept, until they are what a
      // device takes (spec::costs_plausible); else said, the output the same either way.
      size_t plain = plans_.widths.size();
      for (size_t i = 0; i < plans_.widths.size(); ++i)
        if (plans_.widths[i] == 0 && plans_.drafters[i] < 0) plain = i;
      bool plausible = plain == plans_.widths.size() || spec::costs_plausible(round_ms_, at, plain);
      for (int again = 0; again < 2 && !plausible; ++again) {
        const std::vector<double> near = time_at(prompt, "round (again)"), far = time_at(context, what + " (again)");
        for (size_t i = 0; i < at.size(); ++i) {
          round_ms_[i] = std::min(round_ms_[i], near[i]);
          at[i] = std::min(at[i], far[i]);
        }
        plausible = spec::costs_plausible(round_ms_, at, plain);
      }
      if (!plausible)
        log::warn("flyx: the rounds' costs timed at load disagree with each other (another program busy on this "
                  "machine?): rounds may be chosen where they do not pay; the output is the same");
      for (size_t i = 0; i < at.size(); ++i)
        round_ms_per_token_[i] = std::max(0.0, (at[i] - round_ms_[i]) / (longer - round_ms_at_));
      round_ms_again_at_ = longer;
    }
  } else {
    spec::warm_rounds(api_, bm_, plans_.prefill, plans_.rounds, prompt);
  }
  *t_rounds = os::now_ms();
  return true;
}

bool Engine::lower_and_time(std::string* err) {
  for (;;) {
    const double t0 = os::now_ms();
    double t_plans = 0, t_rounds = 0;
    if (build_and_calibrate_plans(&t_plans, &t_rounds, err)) {
      lower_ms_ += t_plans - t0;
      timed_ms_ += t_rounds - t_plans;
      return true;
    }
    lower_ms_ += os::now_ms() - t0;
    if (!plans_.oom) return false;
    if (!give_back()) {  // out of GPU memory with nothing left to give back: what would make room (as the budget's)
      flyx_memory_info mi{};
      const bool has_others =
          api_->memory && api_->memory(bm_, &mi) == 0 && mi.total > 0 && plan::closable(plan::held_of(mi));
      const bool has_drafter = model_->drafters() > 0;
      if (has_others)
        *err += std::string(". Close other apps") + (has_drafter ? ", disable speculative decoding" : "") +
                ", or use a smaller quantization.";
      else if (has_drafter)
        *err += ". Disable speculative decoding or use a smaller quantization.";
      else
        *err += ". Use a smaller quantization.";
      return false;
    }
  }
}

// Passes of 4096 rows where the weights leave the device 16 GB or more, 2048 from 8 GB, else 1024: the K/V and the
// passes' buffers (about 1.5 GB for the 27B at 4096 rows) still fit beside them, and a prompt's rows take fewer
// passes, each reading every weight once and each a prefix checkpoint for the server.  An RTX PRO 6000 (96 GB),
// QUASAR's NVFP4 27B through the server: a 3.6K-token prompt 350-390 -> 295 ms.  The same bits at any pass length
// (numerics.md: K).
int Engine::auto_prefill_chunk() const {
  flyx_memory_info info{};
  if (!api_->memory || api_->memory(bm_, &info) != 0 || info.total == 0) return 1024;
  const double free_gb = double(info.free) / 1e9;
  return free_gb >= 16 ? 4096 : free_gb >= 8 ? 2048 : 1024;
}

bool Engine::give_back() {
  const uint64_t freed = api_->trim ? api_->trim(bm_, 1) : 0;
  given_back_ += freed;
  if (freed) plans_.free(api_);
  return freed > 0;
}

bool Engine::plan_memory(plan::KvTiers* tiers, int session_limit, int* asked_ctx, std::string* err) {
  // Whether most of the matrices' values are BF16 (what the budget suggests when a session does not fit).
  double bf16 = 0, all = 0;
  for (int i = 0; i < model_->desc().n_weights; ++i) {
    const flyx_tensor_view* t = model_->desc().weights[i].t;
    if (!t || t->ndim != 2) continue;
    const double n = double(t->shape[0]) * double(t->shape[1]);
    all += n;
    if (t->dtype == FLYX_DT_BF16) bf16 += n;
  }
  if (plans_.refused_drafter >= 0)
    log::warn("flyx: drafter %s: the backend does not run its rounds (%s); decoding as without it (plain "
              "decoding%s)",
              model_->drafter_name(plans_.refused_drafter), plans_.refused.c_str(),
              plans_.given.empty() ? "" : " and lookup rounds");
  else if (!plans_.refused.empty())
    log::warn("flyx: drafter %s: rounds of %d drafts at most (the backend does not run the others: %s)",
              drafter_.c_str(), plans_.widest(), plans_.refused.c_str());
  if (opt_.ctx_max) opt_.max_ctx = model_->trained_ctx() > 0 ? model_->trained_ctx() : kCtxMaxUnnamed;
  *asked_ctx = opt_.max_ctx;
  // A file that drafts trees already locked AUTO to exact attention before preflight.
  const bool trees =
      std::find(plans_.drafter_trees.begin(), plans_.drafter_trees.end(), true) != plans_.drafter_trees.end();
  if (trees && tiers->method != FLYX_KV_READ_ALL)
    return *err = drafter_ + ": strata attention reads chains of drafts only", false;
  // The model file's pages the load read are cached, not used: on Windows they count as available only once the
  // process lets go of them, and the host memory K/V may take is measured next.
  os::release_working_set();
  tiers->gpu = opt_.kv_gpu;
  tiers->ram = opt_.kv_ram;
  tiers->copy_gpu = opt_.kv_copy_gpu;
  tiers->reserved = opt_.host_reserved;
  tiers->vram_reserve = opt_.vram_reserve;
  tiers->min_vram_reserve = os::gpu_memory_floor();
  tiers->ram_reserve = opt_.ram_reserve;
  tiers->gpu_paged = os::gpu_memory_paged();
  // --ctx max: the longest the GPU and host memory hold, or with no host memory for K/V the longest the GPU holds.
  const int min_ctx = opt_.ctx_max ? 1024 : opt_.ctx_given ? 0 : kMinFittedCtx;
  // Not even one session fits (in any layout account() takes, its K/V in host memory or its context shrunk
  // included): one step of what the backend keeps for speed, and the budget again.  Only when nothing is left to give
  // is it an error.
  while (!plan::account(api_, bm_, model_->drafters() > 0, 2 * bf16 > all, &opt_.max_ctx, min_ctx, *tiers, &ledger_,
                        &budget_, err, opt_.ctx_max)) {
    const std::string why = *err;
    if (!give_back()) {
      *err = why;
      if (given_back_) *err += " (" + gb(given_back_) + " the backend kept for speed already given back)";
      return false;
    }
    if (!lower_and_time(err)) {
      budget_.oom = plans_.oom;
      return false;
    }
  }
  if (given_back_)
    log::info("flyx: %s of GPU memory the backend kept only for speed given back so that a session fits (the same "
              "output, a little slower; -v says what)", gb(given_back_).c_str());
  if (session_limit > 1) plan::fit_second_session(api_, bm_, *tiers, ledger_, &budget_);
  return true;
}

void Engine::log_memory_plan(int asked_ctx) const {
  if (flyx_seq_size size{};
      budget_.layout.method != FLYX_KV_READ_ALL && api_->seq_size(bm_, &budget_.layout, &size) == 0) {
    if (budget_.layout.method == FLYX_KV_AUTO && size.strata_from < opt_.max_ctx) {
      const int bits = budget_.layout.method_arg ? budget_.layout.method_arg : 2;
      if (size.strata_from > 0)
        log::info("flyx: K/V read exactly up to %d tokens, then through a %d-bit copy, the positions each "
                  "token attends to most read exactly", size.strata_from, bits);
      else
        log::info("flyx: K/V read through a %d-bit copy, the positions each token attends to most read exactly", bits);
    }
    if (budget_.layout.copy_host > 0 && size.copies > 0)
      log::info("flyx: the copy of %d of the %d layers in host memory",
                std::min(budget_.layout.copy_host, size.copies), size.copies);
  }
  if (const int dev = budget_.layout.device_positions; dev == 0) {
    log::info("flyx: context %d tokens: K/V in host memory, read over the bus every step (%s of host memory "
              "a session with its checkpoints)",
              opt_.max_ctx, gb(budget_.session_host_bytes).c_str());
  } else if (dev < opt_.max_ctx) {
    log::info("flyx: context %d tokens: K/V of the first %d on the GPU, the rest in host memory, read over "
              "the bus once a conversation passes %d tokens (%s of host memory a session with its "
              "checkpoints)",
              opt_.max_ctx, dev, dev, gb(budget_.session_host_bytes).c_str());
  }
  if (opt_.ctx_max && model_->trained_ctx() <= 0)
    log::info("flyx: --ctx max: the model's file names no context of its own; %d tokens at most",
              kCtxMaxUnnamed);
  if (opt_.ctx_max && opt_.max_ctx == asked_ctx)
    log::info("flyx: --ctx max: context %d tokens, the model's own", opt_.max_ctx);
  else if (opt_.ctx_max)
    log::info("flyx: --ctx max: context %d tokens, the longest %s (the model's own: %d)", opt_.max_ctx,
              budget_.layout.device_positions < opt_.max_ctx ? "the GPU and host memory hold" : "the GPU holds",
              asked_ctx);
  else if (opt_.max_ctx != asked_ctx)
    log::info("flyx: context %d tokens, the longest whose session fits in GPU memory (%d does not; --ctx "
              "sets it)", opt_.max_ctx, asked_ctx);
  if (budget_.auto_exact && budget_.layout.device_positions < opt_.max_ctx)  // auto kept exact
    log::info("flyx: --kv-method auto reads every position exactly here: with its copy %s; decoding past "
              "%d tokens reads the K/V past them over the bus every step (a shorter --ctx keeps the copy)",
              budget_.copy_note.c_str(), budget_.layout.device_positions);
  if (!budget_.tier_note.empty() && opt_.kv_ram != 0)
    log::info("flyx: K/V not in host memory: %s", budget_.tier_note.c_str());
  if (const std::string why = plan::ctx_limit_note(budget_); !why.empty()) {
    if (ctx_short()) log::warn("flyx: the context is short (%d tokens): %s", opt_.max_ctx, why.c_str());
    else log::info("flyx: %s", why.c_str());
  }
  if (opt_.verbose) log::info("flyx: memory: %s", memory_report().c_str());
}

bool Engine::load(const std::string& path, const EngineOptions& opt, std::string* err, int session_limit) {
  opt_ = opt;
  load_adjustments_.clear();
  prefill_cap_ = 0;
  active_draft_weights_.clear();
  apply_environment(&opt_);
  interfaces::VisionDeviceMode vdev = interfaces::VisionDeviceMode::kDisk;
  interfaces::VisionQuantMode vquant = interfaces::VisionQuantMode::kQ8;
  if (!interfaces::parse_vision_device(opt_.vision_device, &vdev, &opt_.vision_device, err)) return false;
  if (!interfaces::parse_vision_quant(opt_.vision_quant, &vquant, &opt_.vision_quant, err)) return false;
  if (opt_.vision_max_tokens < 0)
    return *err = "invalid --vision-max-tokens (must be >= 0)", false;
  if (!opt_.vision) opt_.mmproj_file.clear();
  plan::KvTiers tiers;
  if (!set_kv_method(opt_, &tiers, err)) return false;
  // The load's steps, for its report (verbose): file, tokenizer and chat, backend, model, open, plans, rounds, memory.
  double t[7] = {os::now_ms()};
  if (!file_.open(path, err)) return false;
  for (const std::string& f : opt_.drafter_files)
    if (!file_.add(f, err)) return false;
  if (!opt_.lora_file.empty()) {
    const int before = file_.companions();
    if (!file_.add(opt_.lora_file, err)) return false;
    if (file_.header()["companions"][size_t(before)].value("kind", "") != "lora")
      return *err = opt_.lora_file + ": not a LoRA adapter (a directory of adapter_config.json and "
                                     "adapter_model.safetensors)",
             false;
  }
  if (opt_.vision && !opt_.mmproj_file.empty())
    if (!file_.add(opt_.mmproj_file, err)) return false;
  ReadAhead ahead(file_);  // until the load returns
  t[1] = os::now_ms();
  if (!tok_.load(file_, err)) return false;
  // Chat: a file without a usable template still runs raw prompts; a template
  // the caller gave must work.
  std::string chat_err;
  if (!chat_.load(file_, opt_.chat_template, &chat_err, opt_.chat_native)) {
    if (!opt_.chat_template.empty()) return *err = chat_err, false;
    if (opt_.verbose) log::warn("flyx: no chat: %s", chat_err.c_str());
  }
  t[2] = os::now_ms();
  // The device first: the inputs NVFP4 weights take depend on it.
  if (!load_backend(err)) return false;
  t[3] = os::now_ms();

  const int initial_max_ctx = opt_.max_ctx;
  const std::string initial_backend_opts = opt_.backend_options;
  const std::string req_dw_raw = interfaces::backend_opt_get(initial_backend_opts, "draft_weights");
  interfaces::DraftWeightsMode req_dw = interfaces::DraftWeightsMode::kAuto;
  const bool req_dw_valid =
      req_dw_raw.empty() || interfaces::parse_draft_weights(req_dw_raw, &req_dw, nullptr, nullptr);
  // Strip valid draft_weights= from the opaque options string and pass it via typed flyx_open_options::draft_weights
  // (ABI 48); leave invalid values intact so the backend reports its normal option error.
  std::string clean_backend_opts =
      req_dw_valid ? interfaces::backend_opt_strip(initial_backend_opts, "draft_weights") : initial_backend_opts;
  interfaces::DraftWeightsMode cur_dw = req_dw;

  // Legacy backend keys take precedence over the typed request once. An elastic retry can then pass disk
  // without a stale vision=gpu overriding it at open.
  for (const char* key : {"vision", "vision_quant"}) {
    const std::string value = interfaces::backend_opt_get(clean_backend_opts, key);
    if (value.empty()) continue;
    if (std::string_view(key) == "vision") {
      if (!interfaces::parse_vision_device(value, &vdev, &opt_.vision_device, err)) return false;
    } else if (!interfaces::parse_vision_quant(value, &vquant, &opt_.vision_quant, err)) {
      return false;
    }
    clean_backend_opts = interfaces::backend_opt_strip(clean_backend_opts, key);
  }

  interfaces::ModelOptions mo;
  mo.drafter = opt_.drafter;
  mo.speculative = opt_.speculative;
  mo.vision = opt_.vision;
  mo.activations = opt_.activations;
  mo.fp4_inputs = device_.fp4_inputs != 0;
  model_ = interfaces::open_model(file_, mo, err);
  if (!model_) return false;
  // Draft trees where the model's drafter takes them by default (DSpark), unless the caller said (--draft-trees,
  // FLYX_DRAFT_TREES) or asked for strata attention, which reads chains only.
  if (!opt_.draft_trees_given && opt_.speculative && model_->draft_trees_by_default() &&
      tiers.method != FLYX_KV_STRATA)
    opt_.draft_trees = true;
  if (opt_.vision && !opt_.mmproj_file.empty() && !model_->vision())
    return *err = "--mmproj " + opt_.mmproj_file + ": file is not a supported vision projector (mmproj)", false;
  if (opt_.verbose && device_.fp4_inputs == 0 && opt_.activations.empty()) {
    int nvfp4 = 0;
    for (int i = 0; i < model_->desc().n_weights; ++i) {
      const flyx_tensor_view* w = model_->desc().weights[i].t;
      nvfp4 += w && (w->dtype == FLYX_DT_NVFP4 || w->dtype == FLYX_DT_GGML_NVFP4);
    }
    if (nvfp4)
      log::info("flyx: %d NVFP4 weights take bf16 inputs on this GPU (no FP4 tensor cores)", nvfp4);
  }
  eos_ = file_.eos_ids();

  const bool orig_has_vision = model_->vision() != nullptr;
  const std::string orig_drafters_str = model_drafters_str(*model_);
  const bool gpu_paged = os::gpu_memory_paged();
  const uint64_t vram_floor = os::gpu_memory_floor();
  if (opt_.vram_reserve >= 0 && uint64_t(opt_.vram_reserve) < vram_floor) {
    log::info("flyx: --vram-reserve %s clamped to platform floor %s",
              gb(uint64_t(opt_.vram_reserve)).c_str(), gb(vram_floor).c_str());
  }

  auto upsert_adj = [&](const std::string& kind, const std::string& from, const std::string& to,
                        const std::string& detail) {
    for (LoadAdjustment& a : load_adjustments_) {
      if (a.kind == kind) {
        a.to = to;
        a.detail = detail;
        return;
      }
    }
    load_adjustments_.push_back({kind, from, to, detail});
  };
  auto erase_adj = [&](const std::string& kind) {
    load_adjustments_.erase(
        std::remove_if(load_adjustments_.begin(), load_adjustments_.end(),
                       [&](const LoadAdjustment& a) { return a.kind == kind; }),
        load_adjustments_.end());
  };

  auto query_vram = [&](uint64_t* free_out, uint64_t* total_out) {
    *free_out = device_.mem_free;
    *total_out = device_.mem_total;
    flyx_device_stats ds{};
    if (device_stats(&ds) && ds.mem_total > 0) {
      *total_out = ds.mem_total;
      *free_out = ds.mem_total > ds.mem_used ? ds.mem_total - ds.mem_used : 0;
    }
  };

  int auto_from = opt_.kv_gpu >= 0 && opt_.kv_ram != 0 ? opt_.kv_gpu : INT32_MAX;
  auto make_open_options = [&]() {
    flyx_open_options oo{};
    oo.max_ctx = opt_.max_ctx;
    oo.verbose = opt_.verbose;
    oo.lock_auto_from = tiers.method == FLYX_KV_AUTO;
    oo.auto_from = auto_from;
    oo.kv_fp8 = opt_.kv == FLYX_KV_FP8 ? 1 : 0;
    oo.options = clean_backend_opts.c_str();
    oo.vision_device = to_abi_vision_device(vdev);
    oo.vision_quant = to_abi_vision_quant(vquant);
    oo.draft_weights = to_abi_draft_weights(cur_dw);
    return oo;
  };

  // Resolve target arithmetic from the same text-only, drafter-free description as plain decoding.
  // This is a metadata open. No device allocations or timing runs are needed to set the contract.
  {
    std::unique_ptr<interfaces::Model> plain;
    if (model_->drafters() || model_->vision()) {
      interfaces::ModelOptions base = mo;
      base.drafter = "none";
      base.speculative = false;
      base.vision = false;
      plain = interfaces::open_model(file_, base, err);
      if (!plain) return false;
    }
    const auto& target = plain ? *plain : *model_;
    uint64_t weights = 0;
    for (int i = 0; i < target.desc().n_weights; ++i)
      if (const auto* w = target.desc().weights[i].t) weights += w->nbytes;
    roomy_ = device_.mem_free >= weights && device_.mem_free - weights >= 32000000000ull;
    if (opt_.kv == FLYX_KV_DEFAULT)
      opt_.kv = roomy() || !target.kv_fp8_by_default() ? FLYX_KV_16 : FLYX_KV_FP8;
    if (target.file_drafts_trees() && tiers.method == FLYX_KV_AUTO) tiers.method = FLYX_KV_READ_ALL;
    if (tiers.method == FLYX_KV_AUTO) {
      tiers.allow_auto_exact = false;
      // With automatic placement use the exact target K/V that fits beside the plain weights and runtime.
      // The physical split can still move. An explicit GPU cap or strata_from selects its own switch.
      if (opt_.kv_gpu < 0 && opt_.kv_ram != 0 && api_->estimate_load && device_.mem_total) {
        const auto oo = make_open_options();
        flyx_load_estimate base{};
        if (api_->estimate_load(&target.desc(), &oo, device_.mem_free, device_.mem_total, &base) == 0 &&
            base.target_kv_bytes_per_position) {
          uint64_t room = device_.mem_free;
          for (uint64_t held : {base.weights_bytes, base.min_runtime_bytes,
                                plan::effective_device_reserve(device_.mem_total, gpu_paged,
                                                               opt_.vram_reserve, vram_floor)})
            room -= std::min(room, held);
          auto_from = int(std::min<uint64_t>(INT32_MAX, room / base.target_kv_bytes_per_position));
        }
      }
    }
  }

  auto query_estimate = [&](uint64_t free_vram, uint64_t total_vram, flyx_load_estimate* out) -> bool {
    *out = {};
    if (!api_->estimate_load) return false;
    const flyx_open_options oo = make_open_options();
    return api_->estimate_load(&model_->desc(), &oo, free_vram, total_vram, out) == 0;
  };

  if (orig_has_vision && vdev == interfaces::VisionDeviceMode::kAuto) {
    uint64_t free_vram = 0, total_vram = 0;
    query_vram(&free_vram, &total_vram);
    flyx_load_estimate est{};
    if (query_estimate(free_vram, total_vram, &est)) {
      if (est.vision_device == FLYX_VISION_DEV_GPU) {
        vdev = interfaces::VisionDeviceMode::kGpu;
        opt_.vision_device = "gpu";
      } else if (est.vision_device == FLYX_VISION_DEV_DISK) {
        vdev = interfaces::VisionDeviceMode::kDisk;
        opt_.vision_device = "disk";
      }
    }
  }

  // A drafter the caller named (--drafter mtp, dflash2, dspark, dv) is never skipped: a load that cannot keep it
  // fails, as one without elastic loading does.  auto and all leave the drafters to the load.
  interfaces::DrafterMode asked_drafter = interfaces::DrafterMode::kAuto;
  const bool drafter_named = interfaces::parse_drafter_mode(opt_.drafter, &asked_drafter) &&
                             asked_drafter != interfaces::DrafterMode::kAuto &&
                             asked_drafter != interfaces::DrafterMode::kAll &&
                             asked_drafter != interfaces::DrafterMode::kNone;

  // Ordered VRAM degradation ladder (each step preserves identical target output tokens):
  //   1) vision_device: gpu -> disk (frees 100% vision VRAM while keeping multimodal)
  //   2) draft_weights -> nvfp4 (when backend estimates meaningful VRAM savings >= 128 MiB and >= 10%)
  //   3) drop heavy DFlash2 drafter while retaining DSpark beside it, or retaining or falling back to MTP
  //   4) drop remaining neural drafters, keeping zero-VRAM prompt lookup (or plain decoding if lookup is off)
  // Steps 3 and 4 only for drafters the load chose (drafter_named: none).
  auto step_down = [&](bool comfort_only, const flyx_load_estimate* est) -> bool {
    const bool has_mtp = model_has_drafter(*model_, "mtp");
    const bool has_dspark = model_has_drafter(*model_, "dspark");
    const bool has_dv = model_has_drafter(*model_, "dv");
    // The block drafter in the registry's block slot (DFlash2's block, DFlash's or DSpine's): the heaviest, its
    // context cache growing with the context.
    const char* block_title = nullptr;
    for (const auto& [name, title] : {std::pair{"dflash2", "DFlash2"}, {"dflash", "DFlash"}, {"dspine", "DSpine"}})
      if (!block_title && model_has_drafter(*model_, name)) block_title = title;
    const bool has_block = block_title != nullptr;

    if (!comfort_only && orig_has_vision && vdev == interfaces::VisionDeviceMode::kGpu) {
      vdev = interfaces::VisionDeviceMode::kDisk;
      opt_.vision_device = "disk";
      upsert_adj("vision_device", "gpu", "disk", "Vision encoder moved from GPU to disk (0 VRAM; multimodal kept)");
      return true;
    }
    if (req_dw_valid && cur_dw != interfaces::DraftWeightsMode::kNvfp4 && est && meaningful_draft_savings(*est) &&
        (!comfort_only || req_dw == interfaces::DraftWeightsMode::kAuto)) {
      cur_dw = interfaces::DraftWeightsMode::kNvfp4;
      const std::string detail = std::string(has_block ? block_title : has_dspark ? "DSpark" : "Drafter") +
                                 (has_block || has_dspark ? " drafter" : "") +
                                 " weights quantized to NVFP4 (4-bit) to save VRAM";
      upsert_adj("draft_weights", interfaces::draft_weights_name(req_dw), "nvfp4", detail);
      return true;
    }
    if (comfort_only || drafter_named) return false;
    // The drafters but one: the model opened with `keep` alone (its weights' precision as asked again).
    auto keep_only = [&](const char* keep, const std::string& detail) {
      interfaces::ModelOptions mo_keep = mo;
      mo_keep.drafter = keep;
      std::string probe_err;
      auto kept = interfaces::open_model(file_, mo_keep, &probe_err);
      if (!kept) return false;
      opt_.drafter = keep;
      model_ = std::move(kept);
      if (req_dw == interfaces::DraftWeightsMode::kAuto) {
        erase_adj("draft_weights");
        cur_dw = req_dw;
      } else if (cur_dw == interfaces::DraftWeightsMode::kNvfp4) {
        upsert_adj("draft_weights", interfaces::draft_weights_name(req_dw), "nvfp4",
                   "Drafter weights quantized to NVFP4 (4-bit) to save VRAM");
      }
      upsert_adj("drafter", orig_drafters_str, keep, detail);
      return true;
    };
    // DSpark beside the block drafter (--drafter all): the block drafter goes first, the larger of the two (its weights
    // and its context cache, which grows with the context where DSpark's stays bounded).
    const std::string skipped = std::string(has_block ? block_title : "") + " drafter skipped due to tight VRAM; ";
    if (has_block && has_dspark && keep_only("dspark", skipped + "running with DSpark")) return true;
    if (has_block && keep_only("mtp", skipped + (has_mtp ? "running with MTP" : "falling back to MTP"))) return true;
    if (model_->drafters() > 0) {
      interfaces::ModelOptions mo_none = mo;
      mo_none.drafter = "none";
      std::string none_err;
      if (auto plain_model = interfaces::open_model(file_, mo_none, &none_err)) {
        opt_.drafter = "none";
        model_ = std::move(plain_model);
        erase_adj("draft_weights");
        cur_dw = req_dw;
        const bool lookup_active = opt_.lookup && opt_.speculative;
        const std::string detail = "Speculative drafter (" + orig_drafters_str +
                                   ") skipped due to tight VRAM; " +
                                   (lookup_active ? "running with prompt lookup" : "running plain decoding");
        upsert_adj("drafter", orig_drafters_str, "none", detail);
        return true;
      }
    }
    return false;
  };

  // Past every other step, prefill passes of fewer rows (an automatic size only), down to kElasticMinPrefill: a
  // session's buffers are sized by them as the passes' own are, and a round's graphs are sized on such a session.
  // A prompt goes slower; its rows give a longer pass's bits but in a last pass of 16 rows or fewer, which computes
  // as a round (numerics.md: K).  So not beside a drafter the caller named: plain decoding, fitting with longer
  // passes, would give such a prompt other tokens, and a drafter never changes them (AGENTS.md rule 4); that load
  // fails with the reason, as one that cannot keep the drafter does.
  bool kept_for_drafter = false;
  auto smaller_prefill = [&] {
    if (opt_.prefill_chunk > 0) return false;
    // The size a pass then takes (a whole multiple of the alignment, as build_and_calibrate_plans makes it): a
    // retry only for fewer rows.
    const int cap = std::max(kElasticMinPrefill, prefill_chunk_ / 2);
    const int next = std::max(prefill_align_, cap / prefill_align_ * prefill_align_);
    if (next >= prefill_chunk_) return false;
    if (drafter_named) return kept_for_drafter = true, false;
    std::string from = std::to_string(prefill_chunk_);
    for (const LoadAdjustment& a : load_adjustments_)
      if (a.kind == "prefill_chunk") from = a.from;
    prefill_cap_ = next;
    const std::string to = std::to_string(next);
    upsert_adj("prefill_chunk", from, to,
               "Prefill passes of " + to + " rows instead of " + from + " (a slower prompt)");
    return true;
  };

  // Pre-flight backend-reported VRAM check (avoids wasting seconds uploading weights that cannot fit).
  if (opt_.elastic_load && api_->estimate_load) {
    uint64_t free_vram = 0, total_vram = 0;
    query_vram(&free_vram, &total_vram);
    if (total_vram > 0 && free_vram > 0) {
      const uint64_t reserve = plan::effective_device_reserve(total_vram, gpu_paged, opt_.vram_reserve, vram_floor);
      for (;;) {
        flyx_load_estimate est{};
        if (!query_estimate(free_vram, total_vram, &est)) break;
        const uint64_t hard_need = est.weights_bytes + reserve + est.min_runtime_bytes;
        const uint64_t comfort_need = hard_need + est.comfort_margin_bytes;
        if (hard_need > free_vram) {
          if (!step_down(/*comfort_only=*/false, &est)) break;
          continue;
        }
        if (req_dw == interfaces::DraftWeightsMode::kAuto && cur_dw == interfaces::DraftWeightsMode::kAuto &&
            meaningful_draft_savings(est) && comfort_need > free_vram) {
          if (step_down(/*comfort_only=*/true, &est)) continue;
        }
        break;
      }
    }
  }
  t[4] = os::now_ms();

  int asked_ctx = 0;
  for (;;) {
    opt_.max_ctx = initial_max_ctx;
    plan::KvTiers attempt_tiers = tiers;
    const flyx_open_options oo = make_open_options();

    std::string attempt_err;
    bool ok = true;
    bool oom = false;
    bool opened = false;  // the weights fit: what ran short came after them
    if (const int rc = api_->open(&model_->desc(), &oo, &bm_); rc != 0) {
      attempt_err = std::string("backend open: ") + api_->last_error();
      ok = false;
      oom = rc == FLYX_ERR_OOM;
    }
    t[5] = os::now_ms();
    opened = ok;
    if (ok) {
      lower_ms_ = timed_ms_ = 0;
      given_back_ = 0;
      ok = lower_and_time(&attempt_err);
      oom = !ok && plans_.oom;
    }
    if (ok) {
      ok = plan_memory(&attempt_tiers, session_limit, &asked_ctx, &attempt_err);
      oom = !ok && budget_.oom;
    }
    if (ok) break;

    if (!opt_.elastic_load || !oom) {
      *err = attempt_err;
      return false;
    }
    // Tear down partial GPU allocations before stepping down a tier and retrying.
    plans_.free(api_);
    if (bm_) {
      api_->close(bm_);
      bm_ = nullptr;
    }
    budget_ = plan::Budget{};
    ledger_.set_budget(0);

    uint64_t free_vram = 0, total_vram = 0;
    query_vram(&free_vram, &total_vram);
    flyx_load_estimate est{};
    const bool have_est = query_estimate(free_vram, total_vram, &est);
    if (!step_down(/*comfort_only=*/false, have_est ? &est : nullptr) && !(opened && smaller_prefill())) {
      *err = attempt_err;
      if (kept_for_drafter)
        *err += " Beside --drafter " + opt_.drafter + ", prefill passes stay " + std::to_string(prefill_chunk_) +
                " rows: shorter ones could make its tokens differ from plain decoding's.";
      return false;
    }
    log::warn("flyx: tight VRAM during load (%s); retrying with: %s", attempt_err.c_str(),
              load_adjustments_.back().detail.c_str());
  }

  // Record effective active drafter weight precision directly from the backend (ABI 48).
  if (drafter_ != "none" && api_->draft_weights) {
    if (const char* dw = api_->draft_weights(bm_); dw && *dw) active_draft_weights_ = dw;
  }

  if (opt_.verbose && has_vision()) log::info("flyx: vision encoder loaded");
  wanted_ctx_ = asked_ctx;
  log_memory_plan(asked_ctx);
  if (!load_adjustments_.empty()) {
    log::warn("flyx: adjusted load for available VRAM:");
    for (const LoadAdjustment& a : load_adjustments_) log::warn("flyx:   - %s", a.detail.c_str());
    const std::string dw_tag = active_draft_weights_.empty() ? "" : (" (" + active_draft_weights_ + ")");
    log::warn("flyx: active config: drafter=%s%s, vision=%s, context=%d", drafter_.c_str(), dw_tag.c_str(),
              has_vision() ? opt_.vision_device.c_str() : "off", opt_.max_ctx);
  }
  t[6] = os::now_ms();
  if (opt_.verbose)
    log::info("flyx: load %.2f s: file %.2f s (%s), tokenizer and chat template %.2f s, backend library "
              "%.2f s, model %.2f s, backend open %.2f s, plans %.2f s, rounds timed %.2f s, memory %.2f s",
              (t[6] - t[0]) / 1000, (t[1] - t[0]) / 1000, ahead.what().c_str(), (t[2] - t[1]) / 1000,
              (t[3] - t[2]) / 1000, (t[4] - t[3]) / 1000, (t[5] - t[4]) / 1000, lower_ms_ / 1000, timed_ms_ / 1000,
              std::max(0.0, t[6] - t[5] - lower_ms_ - timed_ms_) / 1000);
  return true;
}

std::string Engine::round_name(size_t i) const {
  const int d = plans_.drafters[i];
  const std::string width = std::to_string(plans_.widths[i]);
  if (d >= 0 && plans_.widths[i] == 0) return std::string(model_->drafter_name(d)) + " warm-up";
  return d < 0 ? width : std::string(model_->drafter_name(d)) + (plans_.drafter_trees[i] ? " tree " : " ") + width;
}

bool Engine::reserve_host_kv() {
  if (!budget_.session_host_bytes) return true;
  const uint64_t limit = budget_.host_budget / budget_.session_host_bytes;
  auto used = host_sessions_.load();
  while (used < limit) {
    if (host_sessions_.compare_exchange_weak(used, used + 1)) return true;
  }
  return false;
}

int Engine::sessions_that_fit() const {
  uint64_t n = ledger_.budget() && budget_.session_bytes ? ledger_.available() / budget_.session_bytes : 1 << 20;
  if (budget_.session_host_bytes) {
    const uint64_t limit = budget_.host_budget / budget_.session_host_bytes;
    const uint64_t used = host_sessions_.load();
    n = std::min(n, limit > used ? limit - used : 0);
  }
  return static_cast<int>(std::min<uint64_t>(n, std::numeric_limits<int>::max()));
}

std::string Engine::memory_report() const {
  std::string r = ledger_.report() + " | one session (" + std::to_string(opt_.max_ctx) + " tokens) " +
                  gb(budget_.session_bytes);
  if (budget_.layout.device_positions < opt_.max_ctx)
    r += " on the GPU (K/V of " + std::to_string(budget_.layout.device_positions) + " tokens) and " +
         gb(budget_.session_host_bytes) + " in host memory";
  return r;
}

int Engine::image_token_budget_for(int max_ctx, int vision_max_tokens, const std::string& detail) {
  const int avail = std::max(256, prompt_limit(max_ctx) - 256);
  std::string norm;
  if (!interfaces::normalize_image_detail(detail, &norm, nullptr)) norm = "auto";
  int budget = 0;
  if (norm == "low") {
    budget = std::min(256, avail);
  } else if (norm == "high") {
    budget = std::min(4096, avail);
  } else {
    // Keep per-image auto budget independent of conversation image count so adding an image on a later
    // turn never changes earlier turns' VisionCache keys or invalidates their KV/GDN prefix cache.
    budget = std::min(2048, std::max(256, avail / 2));
  }
  if (vision_max_tokens > 0) budget = std::min(budget, vision_max_tokens);
  return std::max(16, budget);
}

int Engine::image_token_budget(const std::string& detail, size_t /*n_images*/) const {
  return image_token_budget_for(opt_.max_ctx, opt_.vision_max_tokens, detail);
}

bool Engine::encode_image(const interfaces::ImageInput& img, int max_tokens,
                          std::shared_ptr<const vision::CachedVision>* out, std::string* err,
                          bool* client_error) const {
  auto fail_client = [&](const std::string& msg) {
    if (client_error) *client_error = true;
    *err = msg;
    return false;
  };
  auto fail_internal = [&](const std::string& msg) {
    if (client_error) *client_error = false;
    *err = msg;
    return false;
  };
  if (!has_vision()) return fail_client("this model has no vision encoder loaded (pass --mmproj <mmproj.gguf>)");
  std::string norm_detail;
  if (!interfaces::normalize_image_detail(img.detail, &norm_detail, err)) {
    if (client_error) *client_error = true;
    return false;
  }
  const vision::VisionCacheKey key{img.sha256, norm_detail, max_tokens};
  if (vision_cache_.get(key, out)) return true;

  const double t_enc0 = os::now_ms();
  interfaces::RgbImage rgb;
  if (!vision::decode_image_bytes(img.bytes.data(), img.bytes.size(), &rgb, err)) {
    if (client_error) *client_error = true;
    return false;
  }
  rgb.sha256 = img.sha256;

  interfaces::PreparedMedia pm;
  if (!model_->vision()->preprocess(rgb, max_tokens, &pm, err)) {
    if (client_error) *client_error = true;
    return false;
  }
  if (pm.n_patches <= 0 || pm.patch_dim <= 0 || pm.tokens.empty() || pm.n_embeds <= 0)
    return fail_client("vision preprocess produced empty output");
  const double t_pre = os::now_ms();

  flyx_vision_input vi{};
  vi.patches = pm.patches.data();
  vi.n_patches = pm.n_patches;
  vi.patch_dim = pm.patch_dim;
  vi.grid_thw = pm.grid_thw.empty() ? nullptr : pm.grid_thw.data();
  vi.n_grids = static_cast<int32_t>(pm.grid_thw.size() / 3);
  vi.cu_seqlens = pm.cu_seqlens.size() >= 2 ? pm.cu_seqlens.data() : nullptr;
  vi.n_segs = pm.cu_seqlens.size() >= 2 ? static_cast<int32_t>(pm.cu_seqlens.size() - 1) : 0;

  const int expected_rows = pm.n_embeds;
  const int out_h = model_->vision()->out_hidden() > 0 ? model_->vision()->out_hidden() : model_->desc().hidden;
  auto cv = std::make_shared<vision::CachedVision>();
  cv->embeds.resize(size_t(expected_rows) * size_t(out_h));
  int32_t got_rows = 0;
  if (api_->encode_vision(plans_.vision, &vi, cv->embeds.data(), expected_rows, &got_rows) != 0)
    return fail_internal(std::string("encode_vision: ") + api_->last_error());
  if (got_rows != expected_rows)
    return fail_internal("encode_vision: produced " + std::to_string(got_rows) + " rows, expected " +
                         std::to_string(expected_rows));
  if (opt_.verbose) {
    const double t_end = os::now_ms();
    log::info("flyx: vision encode %dx%d -> %d patches -> %d tokens: %.1f ms (preprocess %.1f ms, ViT %.1f ms)",
              rgb.width, rgb.height, pm.n_patches, got_rows, t_end - t_enc0, t_pre - t_enc0, t_end - t_pre);
  }
  cv->tokens = std::move(pm.tokens);
  cv->embed_offsets = std::move(pm.embed_offsets);
  cv->n_embeds = pm.n_embeds;
  cv->grid_thw = std::move(pm.grid_thw);
  cv->out_hidden = out_h;
  vision_cache_.put(key, cv);
  *out = std::move(cv);
  return true;
}

}  // namespace flyx
