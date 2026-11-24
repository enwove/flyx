// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/engine.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

#include "core/exec/generation.h"
#include "core/interfaces/options.h"
#include "core/os/env.h"
#include "core/os/log.h"
#include "core/os/os.h"
#include "core/speculation/calibrate.h"

namespace flyx {

namespace {

bool parse_vision_device(const std::string& s, int32_t* out, std::string* err) {
  if (s.empty() || s == "auto") return *out = FLYX_VISION_DEV_AUTO, true;
  if (s == "host" || s == "cpu") return *out = FLYX_VISION_DEV_HOST, true;
  if (s == "gpu") return *out = FLYX_VISION_DEV_GPU, true;
  return *err = "unknown --vision-device '" + s + "' (auto, gpu, or host)", false;
}

bool parse_vision_quant(const std::string& s, int32_t* out, std::string* err) {
  if (s.empty() || s == "exact" || s == "bf16") return *out = FLYX_VISION_QUANT_EXACT, true;
  if (s == "q8" || s == "q8_0") return *out = FLYX_VISION_QUANT_Q8, true;
  if (s == "nvfp4" || s == "fp4") return *out = FLYX_VISION_QUANT_NVFP4, true;
  return *err = "unknown --vision-quant '" + s + "' (exact, q8, or nvfp4)", false;
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
    if (const auto v = os::env_bool(env::kDraftTrees)) o->draft_trees = *v;
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
        if (n != names.front() && tried.find(names.front()) != std::string::npos)
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
    for (const auto& path : paths) {
      std::string e;
      auto lib = std::make_unique<os::DynLib>();
      if (!lib->open(path, &e)) {
        tried += "\n  " + path + ": " + e;
        continue;
      }
      auto entry = reinterpret_cast<flyx_backend_entry_fn>(lib->symbol(FLYX_BACKEND_ENTRY));
      const flyx_backend_api* api = entry ? entry() : nullptr;
      if (!api || api->abi_version != FLYX_BACKEND_ABI_VERSION || api->struct_size < sizeof(flyx_backend_api)) {
        tried += "\n  " + path + ": incompatible backend ABI";
        continue;
      }
      flyx_device_info info{};
      if (api->probe(&info) != 0) {
        tried += "\n  " + path + ": " + api->last_error();
        continue;
      }
      // A backend passed over is said without -v: on Windows, Smart App Control once refused an unsigned
      // libflyx-cuda.dll and a real model ran on the float64 reference, a minute a 6000-token prefill, unexplained.
      if (n != names.front() && tried.find(names.front()) != std::string::npos)
        log::warn("flyx: the %s backend is not usable here, so %s runs:%s", names.front().c_str(), n.c_str(),
                  tried.c_str());
      *out_lib = std::move(lib);
      *out_api = api;
      *out_info = info;
      return true;
    }
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
  if (!grammar_.ready()) grammar_.init(tok_, model_->desc().vocab, eos_);
  return grammar_;
}

bool Engine::build_and_calibrate_plans(double* t_plans, double* t_rounds, std::string* err) {
  // Prefill passes run whole multiples of the backend's alignment (numerics.md: K).
  prefill_align_ = std::max(1, int(api_->prefill_align(bm_)));
  const int chunk = opt_.prefill_chunk > 0 ? opt_.prefill_chunk : auto_prefill_chunk();
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
  std::vector<int32_t> prompt = tok_.encode("The quick brown fox jumps over the lazy dog. In the beginning was the word, and "
                                            "the word was with the engine, and the engine was fast. 1, 2, 3, 4, 5, 6,");
  if (prompt.size() > 64) prompt.resize(64);
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
    if (longer > round_ms_at_) {
      std::vector<int32_t> context;
      while (static_cast<int>(context.size()) < longer) context.insert(context.end(), prompt.begin(), prompt.end());
      context.resize(size_t(longer));
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
    if (err->find("out of GPU memory") == std::string::npos) return false;
    if (!give_back()) {  // out of GPU memory with nothing left to give back: what would make room (as the budget's)
      *err += std::string(". Close other apps") +
              (model_->drafters() > 0 ? ", disable speculative decoding" : "") +
              ", or use a smaller quantization.";
      return false;
    }
  }
}

// Passes of 4096 rows where the weights leave the device 16 GB or more, 2048 from 8 GB, else 1024: the K/V and the
// passes' buffers (about 1.5 GB for the 27B at 4096 rows) still fit beside them, and a prompt's rows take fewer
// passes, each reading every weight once and each a prefix checkpoint for the server.  An RTX PRO 6000 (96 GB),
// QUASAR's NVFP4 27B through the server: a 3.6K-token prompt 350-390 -> 295 ms.  The same bits at any pass length
// (numerics.md: K).
bool Engine::roomy() const {
  // The FP8 K/V default buys context on cards the weights nearly fill; where 32 GB stay free after them (a
  // 96 GB card under the 27B: 70), binary16 costs no context worth having and is finer and faster to fill (the 27B,
  // RTX PRO 6000: a 14K-token prefill 6 to 9% faster, decoding the same).
  uint64_t weights = 0;
  for (int i = 0; i < model_->desc().n_weights; ++i)
    if (const flyx_tensor_view* w = model_->desc().weights[i].t) weights += w->nbytes;
  return device_.mem_free > weights && device_.mem_free - weights >= uint64_t(32) * 1000000000;
}

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
  // Trees of drafts read every position, a drafter's that drafts nothing else (DV) as --draft-trees': set_kv_method
  // chose the method before the model and the plans said which drafters draft trees.  A file that holds DV reads every
  // position by default whether DV is asked for or not, so that naming it changes no token (AGENTS.md gate 4): auto
  // would read strata past its switch without DV.  Strata asked for stays where no round drafts a tree.
  const bool trees =
      std::find(plans_.drafter_trees.begin(), plans_.drafter_trees.end(), true) != plans_.drafter_trees.end();
  if (trees && tiers->method == FLYX_KV_STRATA)
    return *err = drafter_ + ": strata attention reads chains of drafts only", false;
  if (trees || (model_->file_drafts_trees() && tiers->method == FLYX_KV_AUTO)) tiers->method = FLYX_KV_READ_ALL;
  // The model file's pages the load read are cached, not used: on Windows they count as available only once the
  // process lets go of them, and the host memory K/V may take is measured next.
  os::release_working_set();
  tiers->gpu = opt_.kv_gpu;
  tiers->ram = opt_.kv_ram;
  tiers->copy_gpu = opt_.kv_copy_gpu;
  tiers->reserved = opt_.host_reserved;
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
    if (!lower_and_time(err)) return false;
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
  apply_environment(&opt_);
  plan::KvTiers tiers;
  if (!set_kv_method(opt_, &tiers, err)) return false;
  // The load's steps, for its report (verbose): file, tokenizer and chat, backend, model, open, plans, rounds, memory.
  double t[7] = {os::now_ms()};
  if (!file_.open(path, err)) return false;
  for (const std::string& f : opt_.drafter_files)
    if (!file_.add(f, err)) return false;
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
  // The model as blocks, with its drafters: the one asked for, else every one it has.
  interfaces::ModelOptions mo;
  mo.drafter = opt_.drafter;
  mo.speculative = opt_.speculative;
  mo.vision = opt_.vision;
  mo.activations = opt_.activations;
  mo.fp4_inputs = device_.fp4_inputs != 0;
  model_ = interfaces::open_model(file_, mo, err);
  if (!model_) return false;
  if (opt_.verbose && !mo.fp4_inputs && opt_.activations.empty()) {
    int nvfp4 = 0;
    for (int i = 0; i < model_->desc().n_weights; ++i) {
      const flyx_tensor_view* w = model_->desc().weights[i].t;
      nvfp4 += w && (w->dtype == FLYX_DT_NVFP4 || w->dtype == FLYX_DT_GGML_NVFP4);
    }
    if (nvfp4)
      log::info("flyx: %d NVFP4 weights take bf16 inputs on this GPU (no FP4 tensor cores)", nvfp4);
  }
  eos_ = file_.eos_ids();
  t[4] = os::now_ms();
  flyx_open_options oo{};
  oo.max_ctx = opt_.max_ctx;
  oo.verbose = opt_.verbose;
  if (opt_.kv == FLYX_KV_DEFAULT) opt_.kv = roomy() || !model_->kv_fp8_by_default() ? FLYX_KV_16 : FLYX_KV_FP8;
  oo.kv_fp8 = opt_.kv == FLYX_KV_FP8 ? 1 : 0;
  oo.options = opt_.backend_options.c_str();
  if (!parse_vision_device(opt_.vision_device, &oo.vision_device, err)) return false;
  if (!parse_vision_quant(opt_.vision_quant, &oo.vision_quant, err)) return false;
  if (api_->open(&model_->desc(), &oo, &bm_) != 0) {
    *err = std::string("backend open: ") + api_->last_error();
    return false;
  }
  t[5] = os::now_ms();
  // The plans, and every round run and timed: again when the backend gives back memory (trim).  Their time over every
  // pass, for the load's report.
  lower_ms_ = timed_ms_ = 0;
  given_back_ = 0;
  if (!lower_and_time(err)) return false;
  if (opt_.verbose && has_vision()) log::info("flyx: vision encoder loaded");
  int asked_ctx = 0;
  if (!plan_memory(&tiers, session_limit, &asked_ctx, err)) return false;
  wanted_ctx_ = asked_ctx;
  log_memory_plan(asked_ctx);
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

int Engine::image_token_budget(const std::string& detail, size_t n_images) const {
  const int avail = std::max(256, prompt_limit(opt_.max_ctx) - 256);
  int budget = 0;
  if (detail == "low") {
    budget = std::min(256, avail);
  } else if (detail == "high") {
    budget = std::min(4096, avail);
  } else {
    const int per_img = std::max(256, (avail / 2) / std::max(1, static_cast<int>(n_images)));
    budget = std::min({2048, per_img, avail});
  }
  if (opt_.vision_max_tokens > 0) budget = std::min(budget, opt_.vision_max_tokens);
  return std::max(16, budget);
}

bool Engine::encode_image(const interfaces::ImageInput& img, int max_tokens,
                          std::shared_ptr<const vision::CachedVision>* out, std::string* err) const {
  if (!has_vision()) {
    *err = "this model has no vision encoder loaded (pass --mmproj <mmproj.gguf>)";
    return false;
  }
  const vision::VisionCacheKey key{img.sha256, img.detail, max_tokens};
  if (vision_cache_.get(key, out)) return true;

  const double t_enc0 = os::now_ms();
  interfaces::RgbImage rgb;
  if (!vision::decode_image_bytes(img.bytes.data(), img.bytes.size(), &rgb, err)) return false;
  rgb.sha256 = img.sha256;

  interfaces::PreparedMedia pm;
  if (!model_->vision()->preprocess(rgb, max_tokens, &pm, err)) return false;
  if (pm.n_patches <= 0 || pm.patch_dim <= 0 || pm.tokens.empty() || pm.n_embeds <= 0) {
    *err = "vision preprocess produced empty output";
    return false;
  }
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
  if (api_->encode_vision(plans_.vision, &vi, cv->embeds.data(), expected_rows, &got_rows) != 0) {
    *err = std::string("encode_vision: ") + api_->last_error();
    return false;
  }
  if (got_rows != expected_rows) {
    *err = "encode_vision: produced " + std::to_string(got_rows) + " rows, expected " + std::to_string(expected_rows);
    return false;
  }
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
