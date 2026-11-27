// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Elastic VRAM loading (Engine::load, ABI 48): verifies the ordered degradation ladder on FLYX_ERR_OOM and
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
#include "tests/common/check.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

const flyx_backend_api* g_backend = nullptr;
std::string g_err;
uint64_t g_total_vram = 0;
uint64_t g_free_vram = 0;
uint64_t g_reserve_floor = 0;
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
// Refusal::size past this many rows of a prefill pass lowered since the open (0: the first open's sessions).
int g_size_max_rows = 0, g_prefill_rows = 0;
int g_align = 0;  // the backend's prefill alignment, if given
bool g_custom_memory = false;
uint64_t g_mem_weights = 0;
uint64_t g_mem_workspace = 0;
uint64_t g_seq_per_ctx_bytes = 0;
uint64_t g_oom_raise_floor = 0;
int g_seq_new_calls = 0;


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
  out->reserve_floor = g_reserve_floor;
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
  g_prefill_rows = 0;
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
  if (p->phase == FLYX_PHASE_PREFILL) g_prefill_rows = std::max(g_prefill_rows, int(p->max_rows));
  if (g_refusal == Refusal::lower && g_open_calls == 1) {
    g_err = "prefill allocation refused";
    return FLYX_ERR_OOM;
  }
  const int rc = g_backend->lower(m, p, out);
  if (rc) g_err = g_backend->last_error();
  return rc;
}

int hook_size(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq_size* out) {
  if (g_refusal == Refusal::size && (g_size_max_rows > 0 ? g_prefill_rows > g_size_max_rows : g_open_calls == 1)) {
    g_err = "round graph scratch allocation refused";
    return FLYX_ERR_OOM;
  }
  const int rc = g_backend->seq_size(m, layout, out);
  if (rc) {
    g_err = g_backend->last_error();
  } else if (g_seq_per_ctx_bytes > 0) {
    const int dev_pos = std::min(layout->max_ctx, layout->device_positions);
    out->device = uint64_t(std::max(dev_pos, 0)) * g_seq_per_ctx_bytes + (64ull << 20);
    out->host = uint64_t(std::max(layout->max_ctx - std::max(dev_pos, 0), 0)) * g_seq_per_ctx_bytes;
  }
  return rc;
}

int hook_memory(flyx_bmodel* m, flyx_memory_info* out) {
  if (!g_custom_memory) return g_backend->memory(m, out);
  *out = {};
  out->total = g_total_vram;
  out->free = g_free_vram;
  out->weights = g_mem_weights;
  out->workspace = g_mem_workspace;
  out->reserve_floor = g_reserve_floor;
  return 0;
}

int hook_seq_new_in(flyx_bmodel* m, const flyx_seq_layout* layout, flyx_seq** out) {
  ++g_seq_new_calls;
  if (g_oom_raise_floor > 0) {
    g_reserve_floor = g_oom_raise_floor;
    g_oom_raise_floor = 0;
    g_err = "simulated late VMM fallback OOM";
    return FLYX_ERR_OOM;
  }
  const int rc = g_backend->seq_new_in(m, layout, out);
  if (rc) g_err = g_backend->last_error();
  return rc;
}

int32_t hook_align(flyx_bmodel* m) { return g_align > 0 ? g_align : g_backend->prefill_align(m); }

const char* hook_last_error() { return g_err.c_str(); }

void reset_hooks() {
  g_err.clear();
  g_size_max_rows = g_prefill_rows = g_align = 0;
  g_custom_memory = false;
  g_mem_weights = 0;
  g_mem_workspace = 0;
  g_seq_per_ctx_bytes = 0;
  g_oom_raise_floor = 0;
  g_seq_new_calls = 0;
  g_total_vram = 0;
  g_free_vram = 0;
  g_reserve_floor = 0;
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

void set_vram(uint64_t total, uint64_t free, uint64_t reserve_floor = 0) {
  g_total_vram = total;
  g_free_vram = free;
  g_reserve_floor = reserve_floor;
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
  flyx::test::print_passes = true;
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
  delegated.memory = hook_memory;
  delegated.seq_size = hook_size;
  delegated.seq_new_in = hook_seq_new_in;
  delegated.prefill_align = hook_align;
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
  if (probe_eo.backend == "cuda") {
    // A packed down projection selects Q8 for an otherwise floating target. A single packed member of a fixed
    // gate|up stack is converted to BF16 along with the other member. The drafter projection reveals AUTO's choice.
    const flyx_tensor_view dense{"float", FLYX_DT_BF16, 2, {3, 256, 0, 0}, nullptr, 3 * 512};
    const flyx_tensor_view packed{"down", FLYX_DT_GGML_Q4_K, 2, {3, 256, 0, 0}, nullptr, 3 * 144};
    const flyx_weight ws[] = {{&dense, nullptr, nullptr, FLYX_ACT_Q8, 0}, {&packed, nullptr, nullptr, FLYX_ACT_Q8, 0},
                              {&dense, nullptr, nullptr, FLYX_ACT_Q8, 0}};
    flyx_block blocks[2]{};
    blocks[0].kind = FLYX_BLOCK_MLP;
    blocks[0].mlp.gate = blocks[0].mlp.up = 0;
    blocks[0].mlp.down = 1;
    blocks[1].kind = FLYX_BLOCK_DFLASH2;
    auto& df = blocks[1].dflash2;
    df.fc = 2;
    df.hidden_norm.w = df.norm.w = df.sel_hid = df.sel_pred = df.sel_succ = -1;
    flyx_model_desc d{};
    d.n_weights = 3;
    d.weights = ws;
    d.n_blocks = 2;
    d.blocks = blocks;
    flyx_open_options opts{};
    flyx_load_estimate est{};
    auto draft_bytes = [&] {
      return g_backend->estimate_load(&d, &opts, 0, 0, &est) == 0 ? est.draft_weights_bytes : UINT64_MAX;
    };
    check(draft_bytes() == 3 * 272, "AUTO sees the packed down projection of a mixed target");
    std::swap(blocks[0].mlp.up, blocks[0].mlp.down);
    check(draft_bytes() == 3 * 512, "AUTO sees the BF16 resident format of the mixed gate/up stack");
    return flyx::test::finish("elastic_load_test");
  }

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

  // 3b. The same with z-lab's DFlash in the block slot (tiny_dflash.flyx, beside tiny.flyx): the block drafter goes
  // whatever its kind, and MTP stays.
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
    const std::string dflash_model = model.substr(0, model.find_last_of("/\\") + 1) + "tiny_dflash.flyx";
    check(e.load(dflash_model, eo, &err), "OOM on DFlash drops DFlash and keeps MTP: " + err);
    check(std::string(e.drafter()) == "mtp" && e.load_adjustments().size() == 1 &&
              e.load_adjustments()[0].kind == "drafter" && e.load_adjustments()[0].from == "mtp+dflash" &&
              e.load_adjustments()[0].to == "mtp" &&
              e.load_adjustments()[0].detail.find("DFlash drafter skipped") != std::string::npos,
          "a single drafter mtp+dflash -> mtp adjustment, naming DFlash");
  }

  // 4. Explicit --drafter dflash2 on a model that also has embedded MTP: a drafter the caller named is never skipped.
  // Its weights may still step down to NVFP4 (the drafter stays); when that does not fit either, the load fails as a
  // load without elastic loading does, and says why.
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
    err.clear();
    check(!e.load(model, eo, &err) && err.find("Not enough VRAM") != std::string::npos,
          "explicit dflash2 that does not fit fails instead of falling back to MTP: " + err);
    check(g_open_calls == 2 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_NVFP4,
          "explicit dflash2 tried its NVFP4 weights once, and no drafter was dropped");
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

  // 7. Non-OOM error never triggers elastic retries (even if the message mentions VRAM — ABI 48 checks FLYX_ERR_OOM).
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

  // 10. Nothing left to drop (no drafter) and a session that does not fit while prefill passes are longer than 256
  // rows (its buffers are sized by them, and a round's graphs on such a session): the load halves the passes until
  // it fits.  A prompt longer than two passes, its last pass over 16 rows, gives one pass's tokens (numerics.md: K).
  // A prefill size the caller gave is kept, and so is a pass no shorter for halving (an alignment of 512 rows).
  {
    const int prompt = 2 * kElasticMinPrefill + 88;
    reset_hooks();
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 1024;
    eo.drafter = "none";
    Engine plain;
    check(plain.load(model, eo, &err) && plain.prefill_chunk() >= prompt,
          "the reference: the prompt in one prefill pass (" + std::to_string(plain.prefill_chunk()) + " rows) " + err);
    const std::vector<int32_t> want = tokens(plain, prompt);
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_refusal = Refusal::size;
    g_size_max_rows = kElasticMinPrefill;
    Engine e;
    err.clear();
    const bool loaded = e.load(model, eo, &err);
    check(loaded, "a session that does not fit retries with shorter prefill passes: " + err);
    const auto& adj = e.load_adjustments();
    check(g_open_calls >= 2 && adj.size() == 1 && adj[0].kind == "prefill_chunk" &&
              adj[0].from == std::to_string(plain.prefill_chunk()) && e.prefill_chunk() == kElasticMinPrefill,
          "one adjustment: prefill passes halved to " + std::to_string(e.prefill_chunk()) + " rows, " +
              std::to_string(g_open_calls) + " opens");
    check(want.size() == 12 && tokens(e, prompt) == want,
          "a prompt of " + std::to_string(prompt) + " rows in passes of 256: one pass's greedy tokens");
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_refusal = Refusal::size;
    g_size_max_rows = kElasticMinPrefill;
    g_align = 512;
    Engine a;
    err.clear();
    const bool aligned = a.load(model, eo, &err);
    check(!aligned && g_open_calls == 2,
          "an alignment of 512 rows: passes of 512, then no retry that would not shorten them (" +
              std::to_string(g_open_calls) + " opens): " + err);
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_refusal = Refusal::size;
    g_size_max_rows = kElasticMinPrefill;
    eo.prefill_chunk = 1024;
    Engine f;
    err.clear();
    const bool given = f.load(model, eo, &err);
    check(!given && g_open_calls == 1, "a prefill size the caller gave is kept: " + err);
    // Beside a drafter the caller named, the passes keep their size (plain decoding, fitting with longer ones,
    // could give a prompt other tokens): the load fails and says so.
    reset_hooks();
    g_total_vram = 24ull << 30;
    g_free_vram = 20ull << 30;
    g_refusal = Refusal::size;
    g_size_max_rows = kElasticMinPrefill;
    eo.prefill_chunk = 0;
    eo.drafter = "mtp";
    Engine named;
    err.clear();
    const bool kept = named.load(model, eo, &err);
    bool halved = false;
    for (const auto& a : named.load_adjustments()) halved = halved || a.kind == "prefill_chunk";
    check(!kept && !halved && err.find("Beside --drafter mtp, prefill passes stay") != std::string::npos,
          "--drafter mtp: no shorter prefill passes, the load refused with the reason: " + err);
  }

  // 11. Explicit --vram-reserve and backend reserve_floor apply to both AUTO exact/Strata boundary (auto_from) and
  // elastic pre-flight checks.
  {
    reset_hooks();
    set_vram(16ull << 30, 12ull << 30);
    g_target_bytes = 8ull << 30;
    g_kv_row_bytes = 1ull << 20;  // 1 MiB per token
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 4096;
    eo.drafter = "none";
    eo.kv_method = "auto";
    eo.vram_reserve = int64_t(1ull << 30);  // 1 GiB reserve -> 2.5 GiB left -> 2560 tokens
    Engine e_lo;
    check(e_lo.load(model, eo, &err), "auto_from with 1 GiB vram_reserve loads: " + err);
    const int auto_from_1g = g_target_options.empty() ? -1 : g_target_options.back().second;

    reset_hooks();
    set_vram(16ull << 30, 12ull << 30);
    g_target_bytes = 8ull << 30;
    g_kv_row_bytes = 1ull << 20;
    eo.vram_reserve = int64_t(2ull << 30);  // 2 GiB reserve -> 1.5 GiB left -> 1536 tokens (1024 fewer)
    Engine e_hi;
    check(e_hi.load(model, eo, &err), "auto_from with 2 GiB vram_reserve loads: " + err);
    const int auto_from_2g = g_target_options.empty() ? -1 : g_target_options.back().second;
    check(auto_from_1g == 2560 && auto_from_2g == 1536,
          "vram_reserve shifts AUTO exact/Strata auto_from boundary (" + std::to_string(auto_from_1g) + " -> " +
              std::to_string(auto_from_2g) + ")");

    // Backend-reported reserve_floor clamps --vram-reserve 0 on auto_from.
    reset_hooks();
    set_vram(16ull << 30, 12ull << 30, 1ull << 30);
    g_target_bytes = 8ull << 30;
    g_kv_row_bytes = 1ull << 20;
    eo.vram_reserve = 0;
    Engine e_floor;
    check(e_floor.load(model, eo, &err), "auto_from with 0 vram_reserve and 1 GiB reserve_floor loads: " + err);
    const int auto_from_floor = g_target_options.empty() ? -1 : g_target_options.back().second;
    check(auto_from_floor == 2560,
          "backend reserve_floor clamps vram_reserve=0 on auto_from (got " + std::to_string(auto_from_floor) + ")");

    // Elastic pre-flight: target 8.5 GiB + dflash 3.8 GiB + min_runtime 0.5 GiB = 12.8 GiB.
    // With 15.0 GiB free and vram_reserve = 256 MiB: comfort_need = 14.55 GiB <= 15.0 GiB -> stays.
    reset_hooks();
    set_vram(16ull << 30, 15000ull << 20, 256ull << 20);
    g_target_bytes = 8500ull << 20;
    g_dflash_bytes = 3800ull << 20;
    g_dflash_savings = 2700ull << 20;
    g_mtp_bytes = 0;
    EngineOptions eo_pf;
    eo_pf.backend = "test-host-kv";
    eo_pf.max_ctx = 128;
    eo_pf.drafter = "dflash2";
    eo_pf.vram_reserve = 0;  // clamped to 256 MiB floor: hard=13068 MiB, comfort=14604 MiB <= 15000 MiB -> keeps auto
    Engine e_keep;
    check(e_keep.load(model, eo_pf, &err), "low reserve_floor keeps dflash2 in elastic pre-flight: " + err);
    check(g_open_calls == 1 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_AUTO && e_keep.load_adjustments().empty(),
          "low reserve_floor avoids unnecessary pre-flight step-down");

    reset_hooks();
    set_vram(16ull << 30, 15000ull << 20, 2048ull << 20);
    g_target_bytes = 8500ull << 20;
    g_dflash_bytes = 3800ull << 20;
    g_dflash_savings = 2700ull << 20;
    g_mtp_bytes = 0;
    eo_pf.vram_reserve = 0;  // clamped to 2048 MiB floor: hard=14860 MiB, comfort=16396 MiB > 15000 MiB -> steps to nvfp4
    Engine e_floor_step;
    check(e_floor_step.load(model, eo_pf, &err), "high reserve_floor steps down in elastic pre-flight: " + err);
    check(g_open_calls == 1 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_NVFP4 && !e_floor_step.load_adjustments().empty(),
          "backend reserve_floor clamps vram_reserve=0 in elastic pre-flight");

    reset_hooks();
    set_vram(16ull << 30, 15000ull << 20);
    g_target_bytes = 8500ull << 20;
    g_dflash_bytes = 3800ull << 20;
    g_dflash_savings = 2700ull << 20;
    g_mtp_bytes = 0;
    eo_pf.vram_reserve = int64_t(2048ull << 20);  // explicit 2 GiB reserve -> steps to nvfp4
    Engine e_step;
    check(e_step.load(model, eo_pf, &err), "high vram_reserve steps down in elastic pre-flight: " + err);
    check(g_open_calls == 1 && g_seen_dw.back() == FLYX_DRAFT_WEIGHTS_NVFP4 && !e_step.load_adjustments().empty(),
          "high vram_reserve triggers pre-flight auto -> nvfp4 step-down");
  }

  // 12. Late reserve_floor increase (e.g., backend VMM fallback on first session allocation): Session::init catches
  // FLYX_ERR_OOM, triggers Engine::replan_for_floor_increase to adopt the raised floor and shrink context, and retries
  // seq_new_in cleanly with identical greedy output.
  {
    reset_hooks();
    g_custom_memory = true;
    g_total_vram = 4ull << 30;
    g_mem_weights = 2ull << 30;
    g_mem_workspace = 0;
    g_free_vram = 1536ull << 20;      // have = 3.5 GiB; with 256 MiB floor -> 1280 MiB session room
    g_reserve_floor = 256ull << 20;
    g_seq_per_ctx_bytes = 128ull << 10;  // 8192 ctx -> 1088 MiB (fits 1280), 4096 ctx -> 576 MiB (fits 768)
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 8192;
    eo.drafter = "none";
    eo.kv_ram = 0;
    eo.vram_reserve = 0;
    Engine e;
    check(e.load(model, eo, &err), "initial load with low reserve_floor succeeds: " + err);
    check(e.options().max_ctx == 8192 && e.budget().device_reserve == (256ull << 20),
          "initial plan keeps 8192 ctx at 256 MiB floor");
    g_oom_raise_floor = 768ull << 20;
    const std::vector<int32_t> out = tokens(e);
    check(out == ref_tokens, "late reserve_floor replan succeeds and reproduces plain greedy tokens");
    check(g_seq_new_calls == 2 && e.budget().device_reserve == (768ull << 20) && e.options().max_ctx < 8192,
          "Session::init retried seq_new_in after replan_for_floor_increase raised reserve to 768 MiB and shrank ctx (" +
              std::to_string(e.options().max_ctx) + ")");
  }

  return flyx::test::finish("elastic_load_test");
}
