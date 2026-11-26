// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Elastic VRAM loading (Engine::load, ABI 45): verifies the ordered degradation ladder on FLYX_ERR_OOM and
// backend-reported pre-flight estimates (estimate_load / draft_weights), meaningful NVFP4 savings threshold,
// strict non-degradation when elastic_load=false (bench mode), immediate failure on non-OOM errors (even if the
// error message mentions VRAM), and Gate 4 bit-identical greedy outputs across every degraded tier.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/session/session.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

const flyx_backend_api* g_backend = nullptr;
std::string g_err;
uint64_t g_total_vram = 0;
uint64_t g_free_vram = 0;
uint64_t g_target_bytes = 8ull << 30;
uint64_t g_dflash_bytes = 3800ull << 20;
uint64_t g_dflash_savings = 2700ull << 20;  // savings from compacting DFlash2 -> NVFP4
uint64_t g_mtp_bytes = 300ull << 20;
bool g_oom_non_nvfp4_dflash = false;
bool g_oom_any_dflash = false;
bool g_oom_any_drafter = false;
bool g_oom_always = false;
bool g_non_vram_error = false;
bool g_generic_rc_with_vram_text = false;
int g_open_calls = 0;
std::vector<int32_t> g_seen_dw;
std::string g_active_dw;
std::string g_opaque_options;
int g_vision_device = 0, g_vision_quant = 0;
uint64_t g_file_bytes = 0, g_kv_row_bytes = 0;
std::vector<std::pair<int, int>> g_target_options;
enum class Refusal { none, lower, size };
Refusal g_refusal = Refusal::none;


bool has_block(const flyx_model_desc* desc, int kind) {
  for (int i = 0; i < desc->n_blocks; ++i)
    if (desc->blocks[i].kind == kind) return true;
  return false;
}

int hook_probe(flyx_device_info* out) {
  const int rc = g_backend->probe(out);
  if (rc != 0) return rc;
  out->mem_total = g_total_vram;
  out->mem_free = g_free_vram;
  return 0;
}

int hook_device_stats(flyx_device_stats* out) {
  *out = {};
  if (g_total_vram == 0) return -1;
  out->mem_total = g_total_vram;
  out->mem_used = g_total_vram > g_free_vram ? g_total_vram - g_free_vram : 0;
  return 0;
}

int hook_estimate_load(const flyx_model_desc* desc, const flyx_open_options* opts, uint64_t /*free_vram*/,
                       uint64_t total_vram, flyx_load_estimate* out) {
  *out = {};
  const bool has_df = has_block(desc, FLYX_BLOCK_DFLASH2);
  const bool has_mtp = has_block(desc, FLYX_BLOCK_MTP);
  const int32_t dw = opts ? opts->draft_weights : FLYX_DRAFT_WEIGHTS_AUTO;
  uint64_t draft_b = 0, savings = 0;
  if (has_df) {
    const uint64_t df_nvfp4 = g_dflash_bytes > g_dflash_savings ? g_dflash_bytes - g_dflash_savings : 0;
    if (dw == FLYX_DRAFT_WEIGHTS_NVFP4) {
      draft_b += df_nvfp4;
    } else {
      draft_b += g_dflash_bytes;
      savings += g_dflash_savings;
    }
  }
  if (has_mtp) draft_b += g_mtp_bytes;
  out->weights_bytes = g_target_bytes + draft_b;
  out->target_kv_bytes_per_position = g_kv_row_bytes;
  out->draft_weights_bytes = draft_b;
  out->draft_compact_savings = savings;
  out->min_runtime_bytes = 512ull << 20;
  out->comfort_margin_bytes = total_vram > 0 ? (1536ull << 20) : 0;
  return 0;
}

const char* hook_draft_weights(flyx_bmodel*) { return g_active_dw.c_str(); }

int hook_open(const flyx_model_desc* desc, const flyx_open_options* opts, flyx_bmodel** out) {
  ++g_open_calls;
  g_file_bytes = 0;
  for (int i = 0; i < desc->n_weights; ++i)
    if (auto* t = desc->weights[i].t) g_file_bytes += t->nbytes;
  g_target_options.emplace_back(opts->kv_fp8, opts->lock_auto_from ? opts->auto_from : -1);
  const int32_t dw = opts ? opts->draft_weights : FLYX_DRAFT_WEIGHTS_AUTO;
  g_seen_dw.push_back(dw);
  g_opaque_options = opts->options;
  g_vision_device = opts->vision_device;
  g_vision_quant = opts->vision_quant;
  if (g_non_vram_error) {
    g_err = "options: unknown option \"bad_key\"";
    return -1;
  }
  if (g_generic_rc_with_vram_text) {
    g_err = "Not enough VRAM (simulated non-OOM status -1)";
    return -1;
  }
  if (g_oom_always) {
    g_err = "Not enough VRAM while loading weights (16.00 GB GPU). Use a smaller quantization.";
    return FLYX_ERR_OOM;
  }
  const bool has_df = has_block(desc, FLYX_BLOCK_DFLASH2);
  const bool has_mtp = has_block(desc, FLYX_BLOCK_MTP);
  if (g_oom_any_drafter && (has_df || has_mtp)) {
    g_err = "Not enough VRAM while loading weights: 4.00 / 16.00 GB used by other apps. Close other apps.";
    return FLYX_ERR_OOM;
  }
  if (g_oom_any_dflash && has_df) {
    g_err = "Not enough VRAM while loading weights: 3.00 / 16.00 GB used by other apps. Close other apps.";
    return FLYX_ERR_OOM;
  }
  if (g_oom_non_nvfp4_dflash && has_df && dw != FLYX_DRAFT_WEIGHTS_NVFP4) {
    g_err = "Not enough VRAM while loading weights: 2.00 / 16.00 GB used by other apps. Close other apps.";
    return FLYX_ERR_OOM;
  }
  if (has_df) {
    g_active_dw = dw == FLYX_DRAFT_WEIGHTS_NVFP4 ? "nvfp4"
                  : dw == FLYX_DRAFT_WEIGHTS_Q8  ? "q8"
                                                 : "exact";
  } else if (has_mtp) {
    g_active_dw = dw == FLYX_DRAFT_WEIGHTS_EXACT ? "exact"
                  : dw == FLYX_DRAFT_WEIGHTS_Q8  ? "q8"
                                                 : "nvfp4";
  } else {
    g_active_dw.clear();
  }
  // Forward to cpu_ref without CUDA-specific backend options (cpu_ref rejects unknown keys).
  flyx_open_options clean = *opts;
  clean.options = "";
  const int rc = g_backend->open(desc, &clean, out);
  if (rc != 0) g_err = g_backend->last_error();
  return rc;
}

int hook_lower(flyx_bmodel* m, const flyx_program* p, flyx_plan** out) {
  if (g_refusal == Refusal::lower && g_open_calls == 1) {
    g_err = "prefill allocation refused";
    return FLYX_ERR_OOM;
  }
  const int rc = g_backend->lower(m, p, out);
  if (rc) g_err = g_backend->last_error();
  return rc;
}

int hook_size(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq_size* out) {
  if (g_refusal == Refusal::size && g_open_calls == 1) {
    g_err = "round graph scratch allocation refused";
    return FLYX_ERR_OOM;
  }
  const int rc = g_backend->seq_size(m, layout, out);
  if (rc) g_err = g_backend->last_error();
  return rc;
}

const char* hook_last_error() { return g_err.c_str(); }

void reset_hooks() {
  g_err.clear();
  g_total_vram = 0;
  g_free_vram = 0;
  g_target_bytes = 8ull << 30;
  g_dflash_bytes = 3800ull << 20;
  g_dflash_savings = 2700ull << 20;
  g_mtp_bytes = 300ull << 20;
  g_oom_non_nvfp4_dflash = false;
  g_oom_any_dflash = false;
  g_oom_any_drafter = false;
  g_oom_always = false;
  g_non_vram_error = false;
  g_generic_rc_with_vram_text = false;
  g_open_calls = 0;
  g_target_options.clear();
  g_kv_row_bytes = 0;
  g_refusal = Refusal::none;
  g_seen_dw.clear();
  g_active_dw.clear();
}

std::vector<int32_t> tokens(Engine& e, int prompt_length = 0, bool sampled = false) {
  Session s(&e);
  std::string err;
  if (!s.init(&err)) return {};
  GenerateRequest rq;
  rq.prompt = e.tokenizer().encode("The quick brown fox");
  if (prompt_length > 0) rq.prompt.resize(size_t(prompt_length), rq.prompt.back());
  if (sampled) {
    rq.sampling.seed = 142;
    rq.sampling.temperature = 0.8f;
  }
  rq.max_new_tokens = 12;
  rq.ignore_eos = true;
  std::vector<int32_t> out;
  GenerateResult res;
  TextSink sink = [&](std::string_view, const int32_t* toks, int n) {
    out.insert(out.end(), toks, toks + n);
    return true;
  };
  if (generate(s, rq, sink, &res, &err) != 0) return {};
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: elastic_load_test TINY_MODEL.flyx\n");
    return 2;
  }
  const std::string model = argv[1];

  // Delegate execution to cpu_ref or CUDA while our hooks control VRAM, estimate_load, and FLYX_ERR_OOM.
  EngineOptions probe_eo;
  probe_eo.backend = argc > 2 ? argv[2] : "cpu_ref";
  std::unique_ptr<os::DynLib> lib;

  flyx_device_info info{};
  std::string err;
  if (!open_backend(probe_eo, &lib, &g_backend, &info, &err)) {
    std::fprintf(stderr, "cannot open delegate backend: %s\n", err.c_str());
    return 1;
  }

  flyx_backend_api delegated = *g_backend;
  delegated.name = "test_gpu";
  delegated.probe = hook_probe;
  delegated.device_stats = hook_device_stats;
  delegated.open = hook_open;
  delegated.lower = hook_lower;
  delegated.seq_size = hook_size;
  delegated.last_error = hook_last_error;
  delegated.estimate_load = hook_estimate_load;
  delegated.draft_weights = hook_draft_weights;
  flyx_test_delegate(&delegated);

  // Reference greedy tokens from plain decoding (Gate 4: degradation never changes output tokens).
  std::vector<int32_t> ref_tokens;
  {
    reset_hooks();
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.speculative = false;
    Engine plain;
    if (!plain.load(model, eo, &err)) { check(false, err); return 1; }
    ref_tokens = tokens(plain);
    check(ref_tokens.size() == 12, "plain reference generated 12 tokens");
  }

  // Exactly 32 decimal GB after the plain target. Adding any drafter crosses the old precision threshold.
  const uint64_t target_file_bytes = g_file_bytes;
  auto threshold_device = [&] {
    reset_hooks();
    g_total_vram = 40ull << 30;
    g_free_vram = 32000000000ull + target_file_bytes;
    // The target-only estimate leaves room for exactly 512 K/V positions. Generate on both sides of the switch.
    const uint64_t reserve = plan::device_reserve(g_total_vram, os::gpu_memory_paged());
    g_kv_row_bytes = (g_free_vram - g_target_bytes - (512ull << 20) - reserve) / 512;
  };
  EngineOptions contract;
  contract.backend = "test-host-kv";
  contract.max_ctx = 1024;
  contract.speculative = false;
  std::vector<int32_t> plain_greedy, plain_sampled;
  threshold_device();
  {
    Engine plain;
    if (!plain.load(model, contract, &err)) { check(false, err); return 1; }
    plain_greedy = tokens(plain, 510);
    plain_sampled = tokens(plain, 510, true);
    check(plain_greedy.size() == 12 && plain_sampled.size() == 12, "plain greedy and fixed-seed reference");
    check(g_target_options == std::vector<std::pair<int, int>>{{0, 512}}, "target-only binary16 and AUTO switch");
  }
  contract.speculative = true;
  contract.drafter = "all";
  for (Refusal refusal : {Refusal::lower, Refusal::size}) {
    threshold_device();
    g_refusal = refusal;
    if (refusal == Refusal::size) g_dflash_savings = 0;  // reopen with only MTP after the sizing failure
    Engine retried;
    if (!retried.load(model, contract, &err)) { check(false, err); return 1; }
    check(g_open_calls == 2 && g_file_bytes > target_file_bytes &&
              std::all_of(g_target_options.begin(), g_target_options.end(),
                          [](auto x) { return x == std::make_pair(0, 512); }),
          "lower/seq_size OOM reaches elastic retry with the plain target contract");
    if (probe_eo.backend == "cuda") {
      for (int positions : {0, INT32_MAX}) {
        flyx_seq_layout l = retried.kv_layout();
        l.device_positions = positions;
        flyx_seq_size size{};
        check(g_backend->seq_size(retried.bmodel(), &l, &size) == 0 && size.strata_from == 512,
              "physical K/V placement preserves the CUDA attention switch");
      }
    }
    check(tokens(retried, 510) == plain_greedy && tokens(retried, 510, true) == plain_sampled,
          "greedy and fixed-seed tokens across the AUTO switch match plain decoding");
  }
  threshold_device();
  g_dflash_bytes = g_free_vram;  // preflight drops DFlash2 before the first upload
  g_dflash_savings = 0;
  {
    Engine fitted;
    if (!fitted.load(model, contract, &err)) { check(false, err); return 1; }
    check(std::string(fitted.drafter()) == "mtp" &&
              g_target_options == std::vector<std::pair<int, int>>{{0, 512}},
          "preflight keeps the plain target contract on the first upload");
    check(tokens(fitted, 510) == plain_greedy && tokens(fitted, 510, true) == plain_sampled,
          "preflight preserves greedy and fixed-seed decoding across the switch");
  }
  if (probe_eo.backend == "cuda") return g_failures ? 1 : 0;

  // 1. Roomy VRAM: both MTP and DFlash2 load with zero adjustments and backend-reported draft_weights.
  {
    reset_hooks();
    g_total_vram = 32ull << 30;
    g_free_vram = 28ull << 30;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    eo.backend_options = "vision=host,vision=gpu,vision_quant=fp4";
    Engine e;
    check(e.load(model, eo, &err), "roomy VRAM loads mtp+dflash2: " + err);
    check(std::string(e.drafter()) == "mtp+dflash2" && e.draft_weights() == "exact" &&
              e.load_adjustments().empty() && g_open_calls == 1,
          "roomy VRAM keeps mtp+dflash2 (exact) with no adjustments and 1 open call");
    check(g_opaque_options.empty() && g_vision_device == FLYX_VISION_DEV_GPU &&
              g_vision_quant == FLYX_VISION_QUANT_NVFP4 &&
              e.options().vision_device == "gpu" && e.options().vision_quant == "nvfp4",
          "legacy vision keys resolve once into the typed options used for open and reporting");
    check(tokens(e) == ref_tokens, "roomy mtp+dflash2 matches plain greedy tokens");
  }

  // 2. FLYX_ERR_OOM on non-NVFP4 DFlash2: steps down typed oo.draft_weights -> NVFP4 and succeeds cleanly.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_non_nvfp4_dflash = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    Engine e;
    check(e.load(model, eo, &err), "FLYX_ERR_OOM on non-NVFP4 DFlash2 retries with draft_weights=nvfp4: " + err);
    check(std::string(e.drafter()) == "mtp+dflash2" && e.draft_weights() == "nvfp4",
          "active config reports mtp+dflash2 with draft_weights=nvfp4");
    check(e.load_adjustments().size() == 1 && e.load_adjustments()[0].kind == "draft_weights" &&
              e.load_adjustments()[0].to == "nvfp4",
          "single draft_weights -> nvfp4 adjustment reported");
    check(g_open_calls == 2 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_NVFP4,
          "second open call passed typed FLYX_DRAFT_WEIGHTS_NVFP4");
    check(tokens(e) == ref_tokens, "nvfp4-degraded mtp+dflash2 matches plain greedy tokens");
  }

  // 3. FLYX_ERR_OOM on any DFlash2: drops DFlash2, retains MTP, coalesces superseded draft_weights adjustment.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_any_dflash = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    Engine e;
    check(e.load(model, eo, &err), "OOM on DFlash2 drops DFlash2 and keeps MTP: " + err);
    check(std::string(e.drafter()) == "mtp" && e.draft_weights() == "nvfp4",
          "active drafter is mtp (nvfp4)");
    check(e.load_adjustments().size() == 1 && e.load_adjustments()[0].kind == "drafter" &&
              e.load_adjustments()[0].from == "mtp+dflash2" && e.load_adjustments()[0].to == "mtp",
          "superseded draft_weights adjustment cleaned up; single drafter mtp+dflash2 -> mtp adjustment reported");
    check(tokens(e) == ref_tokens, "MTP-fallback matches plain greedy tokens");
  }

  // 4. Explicit --drafter dflash2 on a model that also has embedded MTP: falls back to MTP cleanly.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_any_dflash = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "dflash2";
    Engine e;
    check(e.load(model, eo, &err), "explicit dflash2 OOM falls back to embedded MTP: " + err);
    check(std::string(e.drafter()) == "mtp" && e.load_adjustments().size() == 1 &&
              e.load_adjustments()[0].from == "dflash2" && e.load_adjustments()[0].to == "mtp",
          "dflash2 -> mtp fallback reported clearly");
    check(tokens(e) == ref_tokens, "dflash2->mtp fallback matches plain greedy tokens");
  }

  // 5. FLYX_ERR_OOM on all neural drafters: drops all drafters, keeps prompt lookup, single coalesced adjustment.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_any_drafter = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    eo.lookup = true;
    Engine e;
    check(e.load(model, eo, &err), "OOM on all neural drafters falls back to prompt lookup: " + err);
    check(std::string(e.drafter()) == "none" && !e.lookup_widths().empty() && e.draft_weights().empty(),
          "active drafter is none with prompt lookup intact and empty draft_weights");
    check(e.load_adjustments().size() == 1 && e.load_adjustments()[0].kind == "drafter" &&
              e.load_adjustments()[0].from == "mtp+dflash2" && e.load_adjustments()[0].to == "none" &&
              e.load_adjustments()[0].detail.find("prompt lookup") != std::string::npos,
          "single coalesced drafter -> none adjustment mentions prompt lookup");
    check(tokens(e) == ref_tokens, "lookup-only fallback matches plain greedy tokens");
  }

  // 6. Strict mode (elastic_load=false, used by flyx bench): fails immediately without silently degrading.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_any_dflash = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    eo.elastic_load = false;
    Engine e;
    err.clear();
    check(!e.load(model, eo, &err) && g_open_calls == 1 && err.find("Not enough VRAM") != std::string::npos,
          "elastic_load=false refuses OOM immediately without retrying or degrading");
  }

  // 7. Non-OOM error never triggers elastic retries (even if the message mentions VRAM — ABI 45 checks FLYX_ERR_OOM).
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_non_vram_error = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    Engine e;
    err.clear();
    check(!e.load(model, eo, &err) && g_open_calls == 1 && err.find("unknown option") != std::string::npos,
          "non-VRAM error fails immediately on first attempt without degrading");

    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_generic_rc_with_vram_text = true;
    Engine e2;
    err.clear();
    check(!e2.load(model, eo, &err) && g_open_calls == 1,
          "generic return code -1 never retries even when error text mentions VRAM");
  }

  // 8. Total OOM even at minimal plain config fails cleanly after exhausting ladder.
  {
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_oom_always = true;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    Engine e;
    err.clear();
    check(!e.load(model, eo, &err) && err.find("Not enough VRAM") != std::string::npos,
          "unrecoverable OOM fails cleanly with itemized VRAM error");
  }

  // 9. Pre-flight check via estimate_load: when free VRAM cannot hold weights + reserve + runtime, skips drafters
  // BEFORE the first open() call.
  {
    reset_hooks();
    g_total_vram = 16ull << 30;
    g_free_vram = 9ull << 30;  // target=8 GiB fits alone (~9.3 GiB hard need -> drops all drafters pre-flight)
    g_target_bytes = 7ull << 30;
    g_dflash_bytes = 3800ull << 20;
    g_dflash_savings = 2700ull << 20;
    g_mtp_bytes = 2ull << 30;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "all";
    Engine e;
    check(e.load(model, eo, &err), "pre-flight tight VRAM succeeds on first open call: " + err);
    check(g_open_calls == 1 && std::string(e.drafter()) == "none" && !e.load_adjustments().empty(),
          "pre-flight skipped drafters before uploading weights (1 open call, drafter=none)");
    check(tokens(e) == ref_tokens, "pre-flight degraded load matches plain greedy tokens");
  }

  // 10. Meaningful NVFP4 savings threshold: tiny savings (e.g. 40 MiB on a 1.1 GiB Q4_K_M GGUF drafter) must NOT
  // trigger a spurious auto -> nvfp4 comfort-margin downgrade, whereas large savings (2.7 GiB on BF16) must.
  {
    reset_hooks();
    g_total_vram = 16ull << 30;
    g_free_vram = 12ull << 30;  // hard_need (~10.9 GiB) fits, comfort_need (~12.4 GiB) exceeds 12 GiB
    g_target_bytes = 8500ull << 20;
    g_dflash_bytes = 1100ull << 20;
    g_dflash_savings = 40ull << 20;  // ~3.6% (< 128 MiB and < 10%)
    g_mtp_bytes = 0;
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 128;
    eo.drafter = "dflash2";
    Engine e;
    check(e.load(model, eo, &err), "small NVFP4 savings loads cleanly: " + err);
    check(g_open_calls == 1 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_AUTO && e.load_adjustments().empty(),
          "small (<128 MiB / <10%) NVFP4 savings does NOT trigger spurious auto -> nvfp4 downgrade");

    // Same comfort pressure, now with large meaningful savings (BF16 drafter -> NVFP4 saves 2.7 GiB / 71%):
    reset_hooks();
    g_total_vram = 16ull << 30;
    g_free_vram = 13500ull << 20;  // weights 8.5+3.8=12.3 + reserve ~0.8 + min 0.5 = 13.6 GiB (> 13.5) -> steps to NVFP4
    g_target_bytes = 8500ull << 20;
    g_dflash_bytes = 3800ull << 20;
    g_dflash_savings = 2700ull << 20;
    g_mtp_bytes = 0;
    Engine e2;
    check(e2.load(model, eo, &err), "large NVFP4 savings steps down pre-flight: " + err);
    check(g_open_calls == 1 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_NVFP4 && e2.draft_weights() == "nvfp4" &&
              e2.load_adjustments().size() == 1 && e2.load_adjustments()[0].kind == "draft_weights",
          "large (>=128 MiB & >=10%) NVFP4 savings triggers pre-flight auto -> nvfp4 in 1 open call");
  }

  std::printf("%d failures\n", g_failures);
  return g_failures ? 1 : 0;
}
