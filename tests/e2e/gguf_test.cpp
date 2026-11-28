// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A model from its GGUF file is the model.
//   gguf_e2e_test ASSETS BACKEND [BACKEND_OPTIONS]
// * The tiny Qwen3.5 as GGUF (float32 weights in llama.cpp's layout) gives the
//   golden outputs of the float64 reference, with and without the MTP
//   drafter; seeded sampling and edited prompts as from its .flyx file.
// * Attention dimensions the 27B family has (hidden not a multiple of heads).
// * Packed block weights (a backend other than cpu_ref): a tiny Qwen3 per GGML
//   block type and the tiny Qwen3.5 with its matrices quantized give the
//   tokens the float64 backend gives from the same file (8-bit inputs in
//   every pass, per type teacher forced, where a
//   near tie may part), through short prefill passes (the rounds' kernels)
//   and long ones; drafting changes nothing.
// * Drafters in files of their own: the MTP layer from a GGUF of its own, and
//   a DFlash2 drafter file.
// * NVFP4 weights whose inputs are NVFP4 in passes of more than 16 rows (the
//   W4A4 kernels, tiny_w4a4.flyx, a Qwen3.5 128 wide): the
//   float64 backend's tokens, teacher forced, with the same prefill passes
//   (the rounding depends on a pass's rows); asked for by name
//   (--activations nvfp4/bf16) in a file that says bf16 inputs, the same tokens.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "core/exec/generation.h"
#include "nlohmann/json.hpp"
#include "ops/dtype/ref.h"
#include "tests/common/check.h"
#include "tests/common/gpu.h"

using namespace flyx;
using json = nlohmann::json;

namespace {
std::string tested, tested_options;  // the backend under test and its options (the float64 backend takes none)
// The same tokens; else where the two part, said.
bool same_tokens(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
  if (got == want) return true;
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::printf("  (the tokens part at %zu of %zu)\n", i, want.size());
  return false;
}

// How a check loads its engine.
struct Load {
  bool speculative = false;
  int prefill_chunk = 16;
  int max_ctx = 256;
  std::string drafter = "mtp";  // when speculative
  std::vector<std::string> drafter_files{};
  std::string options{};  // backend options, after the tested backend's own
  std::string activations{};  // EngineOptions::activations ("": as the file has it)
  std::string lora_file{};  // EngineOptions::lora_file
  int kv = FLYX_KV_16;  // the golden tokens and drafts are binary16 K/V's
  bool lookup = false;  // prompt lookup's rounds (speculative, drafter "none")
};

// The engine; a load that fails throws, or with `failed` says why there.
std::unique_ptr<Engine> engine(const std::string& path, const std::string& backend, const Load& how = {},
                               std::string* failed = nullptr) {
  EngineOptions options;
  options.backend = backend;
  if (backend == tested) options.backend_options = tested_options;
  if (!how.options.empty()) options.backend_options += (options.backend_options.empty() ? "" : ",") + how.options;
  options.max_ctx = how.max_ctx;
  options.prefill_chunk = how.prefill_chunk;
  options.lookup = how.lookup;
  options.width = !how.speculative ? 0 : how.drafter == "mtp" ? 3 : 15;  // the reference's rounds
  options.speculative = how.speculative;
  options.drafter = how.speculative ? how.drafter : "none";
  options.drafter_files = how.drafter_files;
  options.activations = how.activations;
  options.lora_file = how.lora_file;
  options.kv = how.kv;
  auto result = std::make_unique<Engine>();
  std::string error;
  if (!result->load(path, options, &error)) {
    if (failed) return *failed = error, nullptr;
    throw std::runtime_error(error);
  }
  return result;
}

// Why a load is refused ("" when it is not).
std::string refusal(const std::string& path, const std::string& backend, const Load& how) {
  std::string why;
  if (engine(path, backend, how, &why)) return "";
  return why.empty() ? "refused, no reason given" : why;
}

// The inputs a file's NVFP4 matrices take here, for the float64 backend to round its the same way: as the file has
// it (""), or, on a GPU without FP4 tensor cores (before sm_120), where the engine gives them bf16 inputs by itself,
// bf16 ones ("exact", --activations exact). The file loads either way.
std::string fp4_inputs_here(const std::string& path, const std::string& backend, const std::string& what) {
  std::string why;
  const auto e = engine(path, backend, {.prefill_chunk = 64}, &why);
  check(e != nullptr, what + ": loads on this device" + (why.empty() ? "" : " (" + why + ")"));
  return e && e->device().fp4_inputs ? "" : "exact";
}

std::vector<int32_t> generate_tokens(Session& session, const std::vector<int32_t>& prompt, int n, bool sampling = false) {
  GenerateRequest request;
  request.prompt = prompt;
  request.max_new_tokens = n;
  request.ignore_eos = true;
  request.checkpoints = {int32_t(prompt.size())};
  if (sampling) request.sampling = {0.7f, 0.9f, 20, 123, 0.5f, 0.1f};
  std::string error;
  GenerateResult result;
  std::vector<int32_t> tokens;
  if (generate(session, request,
               [&](std::string_view, const int32_t* ids, int count) {
                 tokens.insert(tokens.end(), ids, ids + count);
                 return true;
               },
               &result, &error) != 0)
    throw std::runtime_error(error);
  return tokens;
}

// A fresh session of the engine on this prompt.
std::vector<int32_t> tokens_of(Engine& e, const std::vector<int32_t>& prompt, int n, bool sampling = false) {
  Session s(&e);
  std::string error;
  if (!s.init(&error)) throw std::runtime_error(error);
  return generate_tokens(s, prompt, n, sampling);
}

// The same, greedy, with every result the device gave (the prefill's, then a round's each).
std::vector<flyx_round_result> traced(Engine& e, const std::vector<int32_t>& prompt, int n,
                                      std::vector<int32_t>* tokens, bool sampling = false) {
  Session s(&e);
  std::string error;
  if (!s.init(&error)) throw std::runtime_error(error);
  std::vector<flyx_round_result> trace;
  s.trace = &trace;
  *tokens = generate_tokens(s, prompt, n, sampling);
  s.trace = nullptr;
  return trace;
}

// The state one prefill pass of the prompt leaves (snapshot bytes: every K/V position and the recurrent state), the
// rows `at` taking `embeds` in place of their tokens' embeddings (an image's rows).
std::vector<uint8_t> pass_state(Engine& e, const std::vector<int32_t>& prompt, const std::vector<int32_t>& at = {},
                                const std::vector<float>& embeds = {}) {
  Session s(&e);
  std::string error;
  if (!s.init(&error)) throw std::runtime_error(error);
  const auto* api = e.api();
  flyx_prefill_input in{};
  in.tokens = prompt.data();
  in.n = int32_t(prompt.size());
  in.embeds = embeds.empty() ? nullptr : embeds.data();
  in.embed_rows = at.empty() ? nullptr : at.data();
  in.n_embeds = int32_t(at.size());
  uint64_t bytes = 0;
  if (api->prefill_ex(e.prefill_plan(), s.seq(), &in, nullptr) != 0 || api->checkpoint(s.seq(), 0) != in.n ||
      api->snapshot_bytes(s.seq(), 0, 0, &bytes) != 0)
    throw std::runtime_error(std::string("a prefill pass's state: ") + api->last_error());
  std::vector<uint8_t> out(static_cast<size_t>(bytes));
  void* p = out.data();
  const flyx_host_pieces pieces{&p, 1, 0, std::max<uint64_t>(4096, bytes)};
  if (api->snapshot_save(s.seq(), 0, 0, &pieces, bytes) != in.n)
    throw std::runtime_error(std::string("a prefill pass's state: ") + api->last_error());
  return out;
}

// Teacher forced: at how many of the reference's tokens the engine, given the reference's tokens before it, picks
// the same one.
int forced_agreement(Engine& e, const std::vector<int32_t>& prompt, const std::vector<int32_t>& want) {
  int agree = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    std::vector<int32_t> p = prompt;
    p.insert(p.end(), want.begin(), want.begin() + std::ptrdiff_t(i));
    agree += tokens_of(e, p, 1)[0] == want[i];
  }
  return agree;
}

void golden(const std::string& dir, const std::string& backend) {
  auto canonical = engine(dir + "/tiny.flyx", backend);
  auto plain = engine(dir + "/tiny_gguf.gguf", backend);
  auto speculative = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const auto golden = json::parse(f);
  for (const auto& item : golden.at("cases")) {
    const auto prompt = item.at("prompt").get<std::vector<int32_t>>();
    const auto expected = item.at("tokens").get<std::vector<int32_t>>();
    Session a(canonical.get()), b(plain.get()), c(speculative.get());
    std::string error;
    if (!a.init(&error) || !b.init(&error) || !c.init(&error)) throw std::runtime_error(error);
    check(generate_tokens(b, prompt, 48) == expected, "GGUF plain output equals float64 golden");
    check(generate_tokens(c, prompt, 48) == expected, "GGUF MTP output equals float64 golden");
    check(generate_tokens(a, prompt, 32, true) == generate_tokens(b, prompt, 32, true),
          "sampling with one seed equals the .flyx model");
    check(generate_tokens(b, prompt, 32, true) == generate_tokens(c, prompt, 32, true),
          "MTP preserves seeded sampling and penalties");
    auto continued = prompt;
    continued.insert(continued.end(), expected.begin(), expected.begin() + 24);
    check(generate_tokens(b, continued, 24) == std::vector<int32_t>(expected.begin() + 24, expected.end()),
          "GGUF session continuation/rewind");
    auto edit = prompt;
    edit.back() = (edit.back() + 1) % 300;
    check(generate_tokens(a, edit, 24) == generate_tokens(b, edit, 24), "GGUF edited prompt equals the .flyx model");
  }
}

void explicit_head_dim(const std::string& dir, const std::string& backend) {
  auto native = engine(dir + "/tiny_explicit_hd.flyx", backend);
  auto imported = engine(dir + "/tiny_explicit_hd.gguf", backend);
  auto spec = engine(dir + "/tiny_explicit_hd.gguf", backend, {.speculative = true});
  const std::vector<int32_t> prompt = {20, 41, 89, 100};
  for (bool sampling : {false, true}) {
    const auto expected = tokens_of(*native, prompt, 32, sampling);
    check(tokens_of(*imported, prompt, 32, sampling) == expected,
          "explicit head dimension: GGUF equals native with hidden % heads != 0");
    check(tokens_of(*spec, prompt, 32, sampling) == expected, "explicit head dimension: MTP preserves greedy/sampled output");
  }
}

// Packed block weights against the float64 backend on the same file.
void packed(const std::string& dir, const std::string& backend) {
  std::vector<int32_t> prompt;
  for (int i = 0; i < 80; ++i) prompt.push_back(i + 20);
  for (const char* kind : {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
                           "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS",
                           "NVFP4"}) {
    const std::string path = dir + "/quant_" + kind + ".gguf";
    // Every pass takes its inputs in 8 bits, the reference's too. A group of 32 values is
    // scaled by its largest, so one bf16 step upstream (the float32 norm against the float64 one) can move a whole
    // group: where the reference's two best tokens nearly tie the backend may pick the other one, and from there
    // the replies part (Q5_K's, IQ3_XXS's and IQ4_XS's tiny models do, once each).  So the reply is compared
    // teacher forced: given the reference's tokens so far, the backend picks the reference's next one at all but at
    // most two of 32 places, through short prefill passes (the rounds' kernels) and long ones.
    const std::string act = std::string(kind) == "NVFP4" ? fp4_inputs_here(path, backend, kind) : "";
    const auto want = tokens_of(*engine(path, "cpu_ref", {.activations = act}), prompt, 32);
    for (int chunk : {16, 64}) {
      const int agree =
          forced_agreement(*engine(path, backend, {.prefill_chunk = chunk, .activations = act}), prompt, want);
      check(agree >= 30, std::string(kind) + ": teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                             " of 32 places, prefill in passes of up to " + std::to_string(chunk) + " rows");
    }
  }
  // The tiny Qwen3.5 quantized: gated delta nets, gated attention, a tied head, the MTP layer, mixed types in a stack.
  const std::string q8 = dir + "/tiny_q8.gguf";
  auto ref = engine(q8, "cpu_ref"), ref_long = engine(q8, "cpu_ref", {.prefill_chunk = 64});
  auto plain = engine(q8, backend), plain_long = engine(q8, backend, {.prefill_chunk = 64});
  auto spec = engine(q8, backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);  // named: a loop over json::parse(f).at(...) reads a destroyed temporary
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = tokens_of(*ref, p, 48);
    check(same_tokens(tokens_of(*plain, p, 48), want), "quantized Qwen3.5: the float64 backend's tokens");
    check(same_tokens(tokens_of(*spec, p, 48), want), "quantized Qwen3.5: the same with the MTP drafter");
    check(tokens_of(*plain, p, 32, true) == tokens_of(*spec, p, 32, true), "quantized Qwen3.5: MTP preserves seeded sampling");
    auto longer = p;
    for (int i = 0; i < 70; ++i) longer.push_back(want[size_t(i) % want.size()]);
    check(tokens_of(*plain, longer, 24) == tokens_of(*ref, longer, 24),
          "quantized Qwen3.5: a long prompt, prefill in passes of up to 16 rows");
    check(tokens_of(*plain_long, longer, 24) == tokens_of(*ref_long, longer, 24),
          "quantized Qwen3.5: a long prompt, prefill in passes of up to 64 rows (8-bit inputs)");
  }
  // The tiny Qwen3.5 with every matrix in GGML's NVFP4: repacked into the NVFP4 planes, gated
  // attention's q | gate rows reordered as they go, the MTP layer's too.
  const std::string nv = dir + "/tiny_nvfp4.gguf";
  const std::string act = fp4_inputs_here(nv, backend, "GGML NVFP4 Qwen3.5");
  auto nref = engine(nv, "cpu_ref", {.activations = act});
  auto nref_long = engine(nv, "cpu_ref", {.prefill_chunk = 64, .activations = act});
  auto nplain = engine(nv, backend, {.activations = act});
  auto nplain_long = engine(nv, backend, {.prefill_chunk = 64, .activations = act});
  auto nspec = engine(nv, backend, {.speculative = true, .activations = act});
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = tokens_of(*nref, p, 48);
    check(same_tokens(tokens_of(*nplain, p, 48), want), "GGML NVFP4 Qwen3.5: the float64 backend's tokens");
    check(same_tokens(tokens_of(*nspec, p, 48), want), "GGML NVFP4 Qwen3.5: the same with the MTP drafter");
    auto longer = p;
    for (int i = 0; i < 70; ++i) longer.push_back(want[size_t(i) % want.size()]);
    check(tokens_of(*nplain_long, longer, 24) == tokens_of(*nref_long, longer, 24),
          "GGML NVFP4 Qwen3.5: a long prompt, prefill in passes of up to 64 rows");
  }
}

// gpt-oss (tests/core/modelfile/tiny_gpt_oss.py): the tiny model as llama.cpp lays it out, with its experts in MXFP4
// and its attention in Q8_0, and the same in float32.  With exact inputs (every linear's input rounded as its format
// does, binary16 pairs for GGML blocks) both give transformers' tokens on every golden case (cases far from a tie in
// the next token and in every router's k-th expert): its biases, sinks, sliding windows (8 positions, a ring past
// them), YaRN rotary, routing and clamped expert activation, through prefill passes of up to 16 and 64 rows.  The file
// as the real model ships it, with the default 8-bit inputs (a coarser rounding than the golden cases' margins
// absorb): prompt lookup's rounds give plain decoding's tokens, greedy and seeded, and on CUDA the unfused kernels
// the fused ones' and, teacher forced, the float64 backend's next token at all but at most four of 24 places.
void gpt_oss(const std::string& dir, const std::string& backend) {
  for (const std::string stem : {"gpt_oss", "gpt_oss_f32"}) {
    std::ifstream f(dir + "/golden_" + stem + ".json");
    const json golden = json::parse(f);
    const std::string path = dir + "/tiny_" + stem + ".gguf";
    auto exact = engine(path, backend, {.activations = "exact"});
    auto exact_long = engine(path, backend, {.prefill_chunk = 64, .activations = "exact"});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      check(same_tokens(tokens_of(*exact, p, int(want.size())), want), "tiny " + stem + ": transformers' tokens");
      check(same_tokens(tokens_of(*exact_long, p, int(want.size())), want),
            "tiny " + stem + ": transformers' tokens, prefill in passes of up to 64 rows");
    }
  }
  const std::string path = dir + "/tiny_gpt_oss.gguf";
  std::ifstream f(dir + "/golden_gpt_oss.json");
  const json golden = json::parse(f);
  auto plain = engine(path, backend), lookup = engine(path, backend, {.speculative = true, .drafter = "none",
                                                                      .lookup = true});
  auto plain_bf16 = engine(path, backend, {.activations = "bf16"});
  auto lookup_bf16 = engine(path, backend, {.speculative = true, .drafter = "none", .activations = "bf16",
                                            .lookup = true});
  // MXFP8 inputs where the backend computes them (CUDA: sm_120's block-scaled tensor cores)
  std::string no_mxfp8;
  auto plain_mxfp8 = engine(path, backend, {.activations = "mxfp8"}, &no_mxfp8);
  auto lookup_mxfp8 = plain_mxfp8 ? engine(path, backend, {.speculative = true, .drafter = "none",
                                                       .activations = "mxfp8", .lookup = true})
                              : nullptr;
  if (!plain_mxfp8) std::printf("  skip MXFP8 inputs on %s: %s\n", backend.c_str(), no_mxfp8.c_str());
  // MXFP8 in prefill passes, 8 bits in rounds
  auto plain_mxfp8_q8 = plain_mxfp8 ? engine(path, backend, {.activations = "mxfp8/q8"}) : nullptr;
  auto lookup_mxfp8_q8 = plain_mxfp8 ? engine(path, backend, {.speculative = true, .drafter = "none",
                                                        .activations = "mxfp8/q8", .lookup = true})
                               : nullptr;
  std::unique_ptr<Engine> unfused, ref, ref_bf16, ref_mxfp8, ref_mxfp8_q8;
  if (backend != "cpu_ref") {
    unfused = engine(path, backend, {.options = "fusion=0"});
    ref = engine(path, "cpu_ref");
    ref_bf16 = engine(path, "cpu_ref", {.activations = "bf16"});
    ref_mxfp8 = engine(path, "cpu_ref", {.activations = "mxfp8"});
    ref_mxfp8_q8 = engine(path, "cpu_ref", {.activations = "mxfp8/q8"});
  }
  for (const auto& item : golden.at("cases")) {
    auto p = item.at("prompt").get<std::vector<int32_t>>();
    const size_t n = p.size();
    for (size_t i = 0; i < 2 * n; ++i) p.push_back(p[i % n]);  // the prompt three times over: lookup drafts
    const auto want = tokens_of(*plain, p, 24);
    check(tokens_of(*lookup, p, 24) == want, "tiny gpt_oss: prompt lookup's rounds give plain decoding's tokens");
    check(tokens_of(*lookup, p, 24, true) == tokens_of(*plain, p, 24, true),
          "tiny gpt_oss: prompt lookup preserves seeded sampling");
    check(tokens_of(*lookup_bf16, p, 24) == tokens_of(*plain_bf16, p, 24),
          "tiny gpt_oss, the experts' inputs in bf16: prompt lookup's rounds give plain decoding's tokens");
    if (plain_mxfp8)
      check(tokens_of(*lookup_mxfp8, p, 24) == tokens_of(*plain_mxfp8, p, 24),
            "tiny gpt_oss, the experts' inputs in MXFP8: prompt lookup's rounds give plain decoding's tokens");
    if (plain_mxfp8_q8)
      check(tokens_of(*lookup_mxfp8_q8, p, 24) == tokens_of(*plain_mxfp8_q8, p, 24),
            "tiny gpt_oss, the experts' inputs in MXFP8 in prefill, 8 bits in rounds: prompt lookup's rounds give "
            "plain decoding's tokens");
    if (unfused) {
      check(tokens_of(*unfused, p, 24) == want, "tiny gpt_oss: fusion=0 gives the fused kernels' tokens");
      const auto ref_tokens = tokens_of(*ref, p, 24);
      const int agree = forced_agreement(*plain, p, ref_tokens);
      // these prompts are not the golden cases' (no margin kept): where the reference's two best nearly tie
      check(agree >= 20, "tiny gpt_oss: teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                             " of 24 places (8-bit inputs)");
      const int agree_bf16 = forced_agreement(*plain_bf16, p, tokens_of(*ref_bf16, p, 24));
      check(agree_bf16 >= 20, "tiny gpt_oss, the experts' inputs in bf16: teacher forced, the float64 backend's "
                              "tokens at " + std::to_string(agree_bf16) + " of 24 places");
      if (plain_mxfp8) {
        const int agree_mxfp8 = forced_agreement(*plain_mxfp8, p, tokens_of(*ref_mxfp8, p, 24));
        check(agree_mxfp8 >= 20, "tiny gpt_oss, the experts' inputs in MXFP8: teacher forced, the float64 backend's "
                               "tokens at " + std::to_string(agree_mxfp8) + " of 24 places");
        const int agree_mxfp8_q8 = forced_agreement(*plain_mxfp8_q8, p, tokens_of(*ref_mxfp8_q8, p, 24));
        check(agree_mxfp8_q8 >= 20, "tiny gpt_oss, the experts' inputs in MXFP8 in prefill, 8 bits in rounds: teacher "
                                "forced, the float64 backend's tokens at " + std::to_string(agree_mxfp8_q8) +
                                " of 24 places");
      }
    }
  }
}

// Gemma 4 (tests/core/modelfile/tiny_gemma4.py): the tiny model as llama.cpp lays it out, its linears and embedding in
// Q8_0, and the same in float32.  With exact inputs both give transformers' tokens on every golden case: sliding
// windows of 8 (a ring past them) beside global layers of wider heads whose v is k's projection, proportional rotary
// on a quarter of those heads, per-head q, k and (weightless) v norms with q.k unscaled, norms after attention and the
// MLP, GELU (tanh) MLPs, each layer's scalar, the scaled embedding tied to the softcapped head, through prefill passes
// of up to 16 and 64 rows.  The file with its default 8-bit inputs: prompt lookup's rounds give plain decoding's
// tokens, greedy and seeded, and on CUDA the unfused kernels the fused ones' and, teacher forced, the float64
// backend's next token at all but at most four of 24 places.
// E2B's and E4B's layers (tiny_gemma4_e.gguf): an input of its own for every layer through a stage after each MLP,
// the last layers reading the K/V of the last of their kind before them.  Transformers' tokens, prefill in passes of
// up to 16 and 64 rows; prompt lookup's rounds and the unfused kernels give plain decoding's; its assistant, its head
// clustered as E4B's, drafts as transformers' does after each prompt (from its directory: its GGUF, as llama.cpp's
// converter writes one, has no clusters and scores every token) and its rounds give plain decoding's tokens.
void gemma4_e(const std::string& dir, const std::string& backend) {
  const std::string path = dir + "/tiny_gemma4_e.gguf", assistant = dir + "/tiny_gemma4_e_assistant";
  std::ifstream f(dir + "/golden_gemma4_e.json");
  const json golden = json::parse(f);
  auto exact = engine(path, backend, {.activations = "exact"});
  auto exact_long = engine(path, backend, {.prefill_chunk = 64, .activations = "exact"});
  auto flyx_exact = engine(dir + "/tiny_gemma4_e.flyx", backend, {.activations = "exact"});
  auto plain = engine(path, backend);
  auto lookup = engine(path, backend, {.speculative = true, .drafter = "none", .lookup = true});
  auto mtp = engine(path, backend, {.speculative = true, .drafter = "mtp", .drafter_files = {assistant}});
  auto mtp_exact = engine(path, backend, {.speculative = true, .drafter = "mtp", .drafter_files = {assistant},
                                          .options = "draft_weights=exact", .activations = "exact"});
  auto mtp_gguf = engine(path, backend, {.speculative = true, .drafter = "mtp",
                                         .drafter_files = {assistant + ".gguf"}});
  std::unique_ptr<Engine> unfused = backend != "cpu_ref" ? engine(path, backend, {.options = "fusion=0"}) : nullptr;
  size_t compared = 0;
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = item.at("tokens").get<std::vector<int32_t>>();
    check(same_tokens(tokens_of(*exact, p, int(want.size())), want), "tiny gemma4_e: transformers' tokens");
    check(same_tokens(tokens_of(*exact_long, p, int(want.size())), want),
          "tiny gemma4_e: transformers' tokens, prefill in passes of up to 64 rows");
    check(same_tokens(tokens_of(*flyx_exact, p, int(want.size())), want),
          "tiny gemma4_e .flyx (python/convert): transformers' tokens");
    auto q = p;
    const size_t n = q.size();
    for (size_t i = 0; i < 2 * n; ++i) q.push_back(q[i % n]);  // the prompt three times over: lookup drafts
    const auto plain_q = tokens_of(*plain, q, 24);
    check(tokens_of(*lookup, q, 24) == plain_q, "tiny gemma4_e: prompt lookup's rounds give plain decoding's tokens");
    if (unfused) check(tokens_of(*unfused, q, 24) == plain_q, "tiny gemma4_e: fusion=0 gives the fused kernels' tokens");
    const auto plain_p = tokens_of(*plain, p, 24);
    check(tokens_of(*mtp, p, 24) == plain_p, "tiny gemma4_e: its assistant's rounds give plain decoding's tokens");
    check(tokens_of(*mtp, p, 24, true) == tokens_of(*plain, p, 24, true),
          "tiny gemma4_e: its assistant's rounds preserve seeded sampling");
    check(tokens_of(*mtp_gguf, p, 24) == plain_p,
          "tiny gemma4_e: its assistant's GGUF (every token scored) rounds give plain decoding's tokens");
    const auto drafts = item.at("drafts").get<std::vector<int32_t>>();
    if (drafts.empty()) continue;
    std::vector<int32_t> got;
    const auto trace = traced(*mtp_exact, p, 24, &got);
    const flyx_round_result* first = trace.size() > 1 ? &trace[1] : nullptr;
    const size_t k = first ? std::min(size_t(first->n_drafts), drafts.size()) : 0;
    check(first && first->n_drafts > 0 && std::equal(drafts.begin(), drafts.begin() + std::ptrdiff_t(k), first->drafts),
          "tiny gemma4_e: its clustered assistant's first drafts are transformers'");
    compared += k;
  }
  check(compared >= 6, "tiny gemma4_e: " + std::to_string(compared) + " of its assistant's drafts compared");
}

void gemma4(const std::string& dir, const std::string& backend) {
  for (const std::string stem : {"gemma4", "gemma4_f32"}) {
    std::ifstream f(dir + "/golden_" + stem + ".json");
    const json golden = json::parse(f);
    const std::string path = dir + "/tiny_" + stem + ".gguf";
    auto exact = engine(path, backend, {.activations = "exact"});
    auto exact_long = engine(path, backend, {.prefill_chunk = 64, .activations = "exact"});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      check(same_tokens(tokens_of(*exact, p, int(want.size())), want), "tiny " + stem + ": transformers' tokens");
      check(same_tokens(tokens_of(*exact_long, p, int(want.size())), want),
            "tiny " + stem + ": transformers' tokens, prefill in passes of up to 64 rows");
    }
  }
  const std::string path = dir + "/tiny_gemma4.gguf";
  std::ifstream f(dir + "/golden_gemma4.json");
  const json golden = json::parse(f);
  auto plain = engine(path, backend), lookup = engine(path, backend, {.speculative = true, .drafter = "none",
                                                                      .lookup = true});
  std::unique_ptr<Engine> unfused, ref;
  if (backend != "cpu_ref") {
    unfused = engine(path, backend, {.options = "fusion=0"});
    ref = engine(path, "cpu_ref");
  }
  for (const auto& item : golden.at("cases")) {
    auto p = item.at("prompt").get<std::vector<int32_t>>();
    const size_t n = p.size();
    for (size_t i = 0; i < 2 * n; ++i) p.push_back(p[i % n]);  // the prompt three times over: lookup drafts
    const auto want = tokens_of(*plain, p, 24);
    check(tokens_of(*lookup, p, 24) == want, "tiny gemma4: prompt lookup's rounds give plain decoding's tokens");
    check(tokens_of(*lookup, p, 24, true) == tokens_of(*plain, p, 24, true),
          "tiny gemma4: prompt lookup preserves seeded sampling");
    if (unfused) {
      check(tokens_of(*unfused, p, 24) == want, "tiny gemma4: fusion=0 gives the fused kernels' tokens");
      const int agree = forced_agreement(*plain, p, tokens_of(*ref, p, 24));
      check(agree >= 20, "tiny gemma4: teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                             " of 24 places (8-bit inputs)");
    }
  }
  // With experts beside every MLP (26B-A4B's layers): transformers' tokens on float32 weights, and its tokens teacher
  // forced in passes of 64 rows (bf16 tensor cores) and on Q8_0 weights (half-precision inputs).  A row's experts
  // are the router's two best of eight: of the hundreds of choices a case makes some nearly tie, and the rounding of
  // an input moves an expert there, which can move a token.  Prompt lookup's rounds and the unfused kernels give
  // plain decoding's tokens; the float64 backend's, teacher forced, at 8-bit inputs.
  {
    const std::string moe = dir + "/tiny_gemma4_moe.gguf", moe32 = dir + "/tiny_gemma4_moe_f32.gguf";
    std::ifstream fm(dir + "/golden_gemma4_moe.json"), fm32(dir + "/golden_gemma4_moe_f32.json");
    const json gm = json::parse(fm), gm32 = json::parse(fm32);
    auto f32_exact = engine(moe32, backend, {.activations = "exact"});
    auto f32_long = engine(moe32, backend, {.prefill_chunk = 64, .activations = "exact"});
    auto q8_exact = engine(moe, backend, {.activations = "exact"});
    for (const auto& item : gm32.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      if (backend == "cpu_ref") {
        check(same_tokens(tokens_of(*f32_exact, p, int(want.size())), want), "tiny gemma4_moe_f32: transformers' tokens");
      } else {  // a pass's 1..16-row GEMMs read float32 inputs as two bf16 halves: a routing tie closer than that moves
        const int short_agree = forced_agreement(*f32_exact, p, want);
        check(short_agree + 1 >= int(want.size()), "tiny gemma4_moe_f32: transformers' tokens teacher forced, at " +
                                                       std::to_string(short_agree) + " of " +
                                                       std::to_string(want.size()));
      }
      const int agree = forced_agreement(*f32_long, p, want);
      check(agree + 1 >= int(want.size()), "tiny gemma4_moe_f32: transformers' tokens teacher forced in passes of up "
                                           "to 64 rows, at " + std::to_string(agree) + " of " +
                                           std::to_string(want.size()));
    }
    for (const auto& item : gm.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      const int agree = forced_agreement(*q8_exact, p, want);
      check(agree + 2 >= int(want.size()), "tiny gemma4_moe: transformers' tokens teacher forced, at " +
                                           std::to_string(agree) + " of " + std::to_string(want.size()));
    }
    auto m_plain = engine(moe, backend);
    auto m_lookup = engine(moe, backend, {.speculative = true, .drafter = "none", .lookup = true});
    std::unique_ptr<Engine> m_unfused, m_ref;
    if (backend != "cpu_ref") {
      m_unfused = engine(moe, backend, {.options = "fusion=0"});
      m_ref = engine(moe, "cpu_ref");
    }
    for (const auto& item : gm.at("cases")) {
      auto p = item.at("prompt").get<std::vector<int32_t>>();
      const size_t n = p.size();
      for (size_t i = 0; i < 2 * n; ++i) p.push_back(p[i % n]);
      const auto want = tokens_of(*m_plain, p, 24);
      check(tokens_of(*m_lookup, p, 24) == want, "tiny gemma4 MoE: prompt lookup's rounds give plain decoding's");
      if (m_unfused) {
        check(tokens_of(*m_unfused, p, 24) == want, "tiny gemma4 MoE: fusion=0 gives the fused kernels' tokens");
        const int agree = forced_agreement(*m_plain, p, tokens_of(*m_ref, p, 24));
        check(agree >= 18, "tiny gemma4 MoE: teacher forced, the float64 backend's tokens at " +
                               std::to_string(agree) + " of 24 places (8-bit inputs)");
      }
    }
  }
  // Its experts in GGML's NVFP4 as llama.cpp writes NVIDIA's 26B-A4B: gate and up apart, each over scales of its own
  // (CUDA stacks them under gate's and takes up's ratio to it into the expert's weight).  Transformers' tokens on the
  // values they hold: on the float64 backend, and teacher forced where the experts take bf16 inputs.
  {
    std::ifstream fn(dir + "/golden_gemma4_moe_nvfp4.json");
    const json gn = json::parse(fn);
    auto nv = engine(dir + "/tiny_gemma4_moe_nvfp4.gguf", backend, {.activations = "exact"});
    for (const auto& item : gn.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      if (backend == "cpu_ref") {
        check(same_tokens(tokens_of(*nv, p, int(want.size())), want), "tiny gemma4_moe_nvfp4: transformers' tokens");
        continue;
      }
      const int agree = forced_agreement(*nv, p, want);
      check(agree + 1 >= int(want.size()), "tiny gemma4_moe_nvfp4: transformers' tokens teacher forced, at " +
                                           std::to_string(agree) + " of " + std::to_string(want.size()));
    }
  }
  // Its assistant beside it (models/gemma4/assistant.h): the first round drafts what transformers' assistant drafts
  // on the same K/V (its drafts of a clear margin), and its rounds give plain decoding's tokens, greedy and seeded.
  const std::string assistant = dir + "/tiny_gemma4_assistant";
  auto mtp = engine(path, backend, {.speculative = true, .drafter = "mtp", .drafter_files = {assistant}});
  auto mtp_exact = engine(path, backend, {.speculative = true, .drafter = "mtp", .drafter_files = {assistant},
                                          .options = "draft_weights=exact", .activations = "exact"});
  // the same assistant as llama.cpp's GGUF of it (unsloth's MTP/mtp-*.gguf)
  auto mtp_gguf = engine(path, backend, {.speculative = true, .drafter = "mtp",
                                         .drafter_files = {assistant + ".gguf"}, .options = "draft_weights=exact",
                                         .activations = "exact"});
  size_t compared = 0;
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto drafts = item.at("drafts").get<std::vector<int32_t>>();
    const auto want = tokens_of(*plain, p, 24);
    std::vector<int32_t> got;
    check(tokens_of(*mtp, p, 24) == want, "tiny gemma4: its assistant's rounds give plain decoding's tokens");
    check(tokens_of(*mtp, p, 24, true) == tokens_of(*plain, p, 24, true),
          "tiny gemma4: its assistant's rounds preserve seeded sampling");
    const auto trace = traced(*mtp_exact, p, 24, &got);  // its inputs as transformers' float64 model reads them
    const flyx_round_result* first = trace.size() > 1 ? &trace[1] : nullptr;
    const size_t n = first ? std::min(size_t(first->n_drafts), drafts.size()) : 0;
    check(first && first->n_drafts > 0 && std::equal(drafts.begin(), drafts.begin() + std::ptrdiff_t(n), first->drafts),
          "tiny gemma4: its assistant's first drafts are transformers'");
    compared += n;
    std::vector<int32_t> got_gguf;
    const auto trace_gguf = traced(*mtp_gguf, p, 24, &got_gguf);
    check(got_gguf == got && trace_gguf.size() > 1 && trace_gguf[1].n_drafts == first->n_drafts &&
              std::equal(first->drafts, first->drafts + first->n_drafts, trace_gguf[1].drafts),
          "tiny gemma4: its assistant's GGUF drafts as its directory does");
  }
  check(compared >= 10, "tiny gemma4: " + std::to_string(compared) + " of its assistant's drafts compared");

  // The same weights from transformers' layout through python/convert (tiny_gemma4.flyx: float32, the blocks' values)
  // give transformers' tokens; its MLPs in NVFP4 as NVIDIA's checkpoints (tiny_gemma4_nvfp4.flyx) decode as their
  // own plain decoding under prompt lookup and the assistant, and the float64 backend's tokens teacher forced.
  auto flyx_exact = engine(dir + "/tiny_gemma4.flyx", backend, {.activations = "exact"});
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    const auto want = item.at("tokens").get<std::vector<int32_t>>();
    check(same_tokens(tokens_of(*flyx_exact, p, int(want.size())), want),
          "tiny gemma4 .flyx (python/convert): transformers' tokens");
  }
  // bf16 inputs: its K of 64 is under the W4A4 kernels' tile of 128 (tiny_w4a4.flyx tests NVFP4 inputs)
  const std::string fp4 = dir + "/tiny_gemma4_nvfp4.flyx";
  auto fp4_plain = engine(fp4, backend, {.activations = "exact"});
  auto fp4_lookup = engine(fp4, backend, {.speculative = true, .drafter = "none", .activations = "exact",
                                          .lookup = true});
  auto fp4_mtp = engine(fp4, backend, {.speculative = true, .drafter = "mtp", .drafter_files = {assistant},
                                       .activations = "exact"});
  std::unique_ptr<Engine> fp4_ref = backend != "cpu_ref" ? engine(fp4, "cpu_ref") : nullptr;
  for (const auto& item : golden.at("cases")) {
    auto p = item.at("prompt").get<std::vector<int32_t>>();
    const size_t n = p.size();
    for (size_t i = 0; i < 2 * n; ++i) p.push_back(p[i % n]);
    const auto want = tokens_of(*fp4_plain, p, 24);
    check(tokens_of(*fp4_lookup, p, 24) == want, "tiny gemma4 NVFP4: prompt lookup's rounds give plain decoding's");
    check(tokens_of(*fp4_mtp, p, 24) == want, "tiny gemma4 NVFP4: its assistant's rounds give plain decoding's");
    if (fp4_ref) {
      const int agree = forced_agreement(*fp4_plain, p, tokens_of(*fp4_ref, p, 24));
      check(agree >= 20, "tiny gemma4 NVFP4: teacher forced, the float64 backend's tokens at " +
                             std::to_string(agree) + " of 24 places");
    }
  }
  gemma4_e(dir, backend);
}

// The MTP layer in a file of its own (EngineOptions::drafter_files; unsloth ships the 27B's so): the quantized tiny
// Qwen3.5 from the two files gives what the one file gives, every round's drafts included; the model without the
// file has no MTP layer, and a file whose model has one takes no second.
void split_mtp(const std::string& dir, const std::string& backend) {
  const std::string whole = dir + "/tiny_q8.gguf", target = dir + "/tiny_q8_target.gguf", mtp = dir + "/tiny_q8_mtp.gguf";
  auto one = engine(whole, backend, {.speculative = true});
  auto two = engine(target, backend, {.speculative = true, .drafter_files = {mtp}});
  auto rounds_of = [](Engine& e, const std::vector<int32_t>& prompt, std::vector<int32_t>* tokens) {
    std::vector<std::vector<int32_t>> drafts;
    for (const auto& r : traced(e, prompt, 40, tokens)) drafts.emplace_back(r.drafts, r.drafts + r.n_drafts);
    return drafts;
  };
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);
  for (const auto& item : golden.at("cases")) {
    const auto p = item.at("prompt").get<std::vector<int32_t>>();
    std::vector<int32_t> ta, tb;
    const auto da = rounds_of(*one, p, &ta), db = rounds_of(*two, p, &tb);
    size_t drafted = 0;
    for (const auto& d : da) drafted += d.size();
    check(ta == tb && da == db && drafted > 0, "the MTP layer from a file of its own: the tokens and every round's drafts "
                                               "of the one file (" + std::to_string(drafted) + " drafts)");
  }
  std::string why = refusal(target, backend, {.speculative = true});
  check(why.find("no MTP layer") != std::string::npos, "the model without its MTP file has no MTP layer (" + why + ")");
  why = refusal(whole, backend, {.speculative = true, .drafter_files = {mtp}});
  check(why.find("has its MTP layer") != std::string::npos, "a model with its MTP layer takes no drafter file (" + why + ")");
}

// Prefill tokens [from, to) of t at the sequence's position, in passes of up to `rows` rows that choose no token.
bool prefill_span(Engine& e, Session& s, const std::vector<int32_t>& t, size_t from, size_t to, size_t rows = 64) {
  for (size_t off = from; off < to; off += rows) {
    const int32_t n = int32_t(std::min<size_t>(rows, to - off));
    if (e.api()->prefill(e.prefill_plan(), s.seq(), t.data() + off, n, nullptr) != 0) return false;
  }
  return true;
}

// From the sequence's position: `first` prefilled, choosing the token after it, then greedy rounds of the widest plan,
// one at a time, until n tokens are out: every token, every round's drafts, and the last result.
bool decode_from(Engine& e, Session& s, int32_t first, size_t n, std::vector<int32_t>* tokens,
                 std::vector<int32_t>* drafts, flyx_round_result* last = nullptr) {
  const auto* api = e.api();
  flyx_gen_params g{};
  g.top_p = 1.f;
  flyx_round_result rr{};
  if (api->prefill(e.prefill_plan(), s.seq(), &first, 1, &g) != 0 || api->read(s.seq(), &rr) != 0 || rr.n_tokens != 1)
    return false;
  tokens->push_back(rr.tokens[0]);
  while (tokens->size() < n) {
    if (api->launch(e.round_plan(e.widths().size() - 1), s.seq(), &g) != 0 || api->read(s.seq(), &rr) != 0 ||
        rr.n_tokens <= 0)
      return false;
    drafts->insert(drafts->end(), rr.drafts, rr.drafts + rr.n_drafts);
    tokens->insert(tokens->end(), rr.tokens, rr.tokens + rr.n_tokens);
  }
  if (last) *last = rr;
  return true;
}

// Greedy through a session, the prompt checkpointed at `checkpoints`: the tokens, every result the device gave, and
// how many of the prompt's tokens the session's state already held (and how it got them).
std::vector<int32_t> generate_traced(Session& s, const std::vector<int32_t>& prompt, int n,
                                     const std::vector<int32_t>& checkpoints, std::vector<flyx_round_result>* trace,
                                     int* reused = nullptr, std::string* how = nullptr) {
  GenerateRequest request;
  request.prompt = prompt;
  request.max_new_tokens = n;
  request.ignore_eos = true;
  request.checkpoints = checkpoints;
  std::string error;
  GenerateResult result;
  std::vector<int32_t> tokens;
  s.trace = trace;
  const int failed = generate(s, request,
                              [&](std::string_view, const int32_t* ids, int count) {
                                tokens.insert(tokens.end(), ids, ids + count);
                                return true;
                              },
                              &result, &error);
  s.trace = nullptr;
  if (failed) throw std::runtime_error(error);
  if (reused) *reused = result.usage.reused_tokens;
  if (how) *how = result.reuse;
  return tokens;
}

// The same results, round by round: tokens and drafts.
bool same_rounds(const std::vector<flyx_round_result>& a, const std::vector<flyx_round_result>& b) {
  if (a.size() != b.size() || a.size() < 2) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].n_tokens != b[i].n_tokens || !std::equal(a[i].tokens, a[i].tokens + a[i].n_tokens, b[i].tokens) ||
        a[i].n_drafts != b[i].n_drafts || !std::equal(a[i].drafts, a[i].drafts + a[i].n_drafts, b[i].drafts))
      return false;
  return true;
}

// Size both rings at a context where even the tiny FP8 MTP cache saves more than a 2 MiB VMM allocation granule.
constexpr int ring_sizing_ctx = 32768;

// A sliding-window DFlash2 drafter's context cache as a ring of slots (backend option dflash_ring, on by default)
// beside the same cache a slot a position (dflash_ring=0): at max_ctx 512, prefill passes of 64 rows and rounds of 16,
// the window of 8 in a ring of 128 slots (256 with wide rounds) that a 300-token prompt and its reply wrap around.
// * Fewer bytes, and none fewer where the ring would hold no fewer positions than the cache.
// * The flat cache's tokens and drafts across the wraps, and from a snapshot of the ring loaded elsewhere; plain
//   decoding's tokens.  The same with the drafter cut to one layer beside FP8 caches: its cache binary16, as its
//   kernels write and read it, not the FP8 of the MTP layer's (a one-layer cache too).
// * The ring holds the drafter's window of the last positions only.  A checkpoint whose window later positions
//   overwrote is neither restored nor saved; a rewind past prefill passes that overwrote the window there is refused
//   (the state reset, as outside the rollback window); a session then recomputes, with a fresh session's tokens and
//   drafts.  A checkpoint or round the ring still holds is restored, saved after more rows, or rewound to, with the
//   flat cache's tokens and drafts.
void dflash_ring(const std::string& dir, const std::string& backend, const std::vector<int32_t>& p) {
  const std::string model = dir + "/tiny_gguf.gguf";
  std::vector<int32_t> wrap_p, filler;
  while (wrap_p.size() < 300) wrap_p.insert(wrap_p.end(), p.begin(), p.end());
  wrap_p.resize(300);
  while (filler.size() < 320) filler.insert(filler.end(), p.rbegin(), p.rend());
  filler.resize(320);
  // Ring and flat side by side: what they take, the tokens and drafts across the wraps, a snapshot of the ring.
  auto against_flat = [&](const std::string& drafter, int kv, const std::string& which) {
    const Load ring_load = {.speculative = true,
                            .prefill_chunk = 64,
                            .max_ctx = 512,
                            .drafter = "dflash2",
                            .drafter_files = {drafter},
                            .kv = kv};
    Load flat_load = ring_load;
    flat_load.options = "dflash_ring=0";
    auto ring_eng = engine(model, backend, ring_load);
    auto flat_eng = engine(model, backend, flat_load);
    auto bytes = [&](Engine& e, int ctx) {
      uint64_t b = 0;
      return e.api()->seq_bytes(e.bmodel(), ctx, &b) == 0 ? b : uint64_t(0);
    };
    const uint64_t ring_bytes = bytes(*ring_eng, ring_sizing_ctx), flat_bytes = bytes(*flat_eng, ring_sizing_ctx);
    check(ring_bytes > 0 && ring_bytes < flat_bytes,
          which + ": the ring shrinks a session (" + std::to_string(ring_bytes) + " < " + std::to_string(flat_bytes) +
              " bytes)");
    // The cache's max_ctx + 32 positions: at 96 no more than the ring's 128 slots, at 64 fewer.
    for (int ctx : {64, 96})
      check(bytes(*ring_eng, ctx) > 0 && bytes(*ring_eng, ctx) == bytes(*flat_eng, ctx),
            "  at max_ctx " + std::to_string(ctx) + ", a cache no longer than the ring: no ring, the flat bytes");
    std::vector<int32_t> tr_toks, fl_toks;
    const auto tr = traced(*ring_eng, wrap_p, 48, &tr_toks);
    const auto fl = traced(*flat_eng, wrap_p, 48, &fl_toks);
    check(same_tokens(tr_toks, fl_toks) && same_rounds(tr, fl),
          which + ": across the ring's wraps, the flat cache's tokens and drafts");
    {
      Load plain_load = ring_load;
      plain_load.speculative = false;
      check(same_tokens(tr_toks, tokens_of(*engine(model, backend, plain_load), wrap_p, 48)),
            "  plain decoding's tokens");
    }
    const auto* api = ring_eng->api();
    Session src(ring_eng.get()), dst(ring_eng.get()), fresh(flat_eng.get());
    std::string err;
    bool ok = src.init(&err) && dst.init(&err) && fresh.init(&err) && prefill_span(*ring_eng, src, wrap_p, 0, 300);
    const int32_t at = ok ? api->checkpoint(src.seq(), 0) : -1;  // 300, or a chunk boundary the last pass crossed
    uint64_t sb = 0;
    ok = ok && at > 0 && at <= 300 && api->snapshot_bytes(src.seq(), 0, 0, &sb) == 0 && sb > 0;
    std::vector<uint8_t> buf(size_t(std::max<uint64_t>(1, sb)));
    void* q = buf.data();
    flyx_host_pieces pieces{&q, 1, 0, sb};
    const int32_t next = at > 0 && at < 300 ? wrap_p[size_t(at)] : p.front();
    ok = ok && api->snapshot_save(src.seq(), 0, 0, &pieces, sb) == at &&
         api->snapshot_load(dst.seq(), &pieces, sb, next) == at &&
         prefill_span(*flat_eng, fresh, wrap_p, 0, size_t(at));
    std::vector<int32_t> got, want, got_dr, want_dr;
    ok = ok && decode_from(*ring_eng, dst, next, 24, &got, &got_dr) &&
         decode_from(*flat_eng, fresh, next, 24, &want, &want_dr);
    check(ok && same_tokens(got, want) && !got_dr.empty() && got_dr == want_dr,
          which + ": a snapshot of the ring after its wraps, loaded elsewhere: the flat cache's tokens and drafts");
    return std::make_pair(std::move(ring_eng), std::move(flat_eng));
  };
  {
    auto [ring_eng, flat_eng] =
        against_flat(dir + "/tiny_draft_window.gguf", FLYX_KV_16, "sliding-window DFlash2 ring");
    const auto* api = ring_eng->api();
    // A checkpoint after the first pass, and one after the prompt; then 16 rows more.  The first one's window's slots
    // hold later positions by then; the second's still hold its window.
    {
      Session src(ring_eng.get()), dst(ring_eng.get()), fresh(flat_eng.get());
      std::string err;
      bool ok = src.init(&err) && dst.init(&err) && fresh.init(&err) && prefill_span(*ring_eng, src, wrap_p, 0, 64);
      ok = ok && api->checkpoint(src.seq(), 0) == 64 && prefill_span(*ring_eng, src, wrap_p, 64, 300);
      const int32_t at = ok ? api->checkpoint(src.seq(), 1) : -1;  // 300, or a chunk boundary the last pass crossed
      ok = ok && at > 0 && at <= 300 && prefill_span(*ring_eng, src, filler, 0, 16);
      uint64_t sb = 0;
      check(ok && api->restore(src.seq(), 0, wrap_p[64]) < 0 && api->snapshot_bytes(src.seq(), 0, 0, &sb) != 0,
            "  a checkpoint whose window later positions overwrote in the ring: neither restored nor saved");
      // The one the ring still holds, saved and restored after the rows past it: the flat cache's continuation.
      ok = ok && api->snapshot_bytes(src.seq(), 1, 0, &sb) == 0 && sb > 0;
      std::vector<uint8_t> buf(size_t(std::max<uint64_t>(1, sb)));
      void* q = buf.data();
      flyx_host_pieces pieces{&q, 1, 0, sb};
      const int32_t next = at > 0 && at < 300 ? wrap_p[size_t(at)] : p.front();
      ok = ok && api->snapshot_save(src.seq(), 1, 0, &pieces, sb) == at &&
           api->snapshot_load(dst.seq(), &pieces, sb, next) == at && api->restore(src.seq(), 1, next) == at &&
           prefill_span(*flat_eng, fresh, wrap_p, 0, size_t(at));
      std::vector<int32_t> want, want_dr, restored, restored_dr, loaded, loaded_dr;
      ok = ok && decode_from(*flat_eng, fresh, next, 24, &want, &want_dr) &&
           decode_from(*ring_eng, src, next, 24, &restored, &restored_dr) &&
           decode_from(*ring_eng, dst, next, 24, &loaded, &loaded_dr);
      check(ok && !want_dr.empty() && same_tokens(restored, want) && restored_dr == want_dr &&
                same_tokens(loaded, want) && loaded_dr == want_dr,
            "  one the ring still holds, restored, and saved after more rows: the flat cache's tokens and drafts");
    }
    // Rounds after a prompt, prefill passes past them (the next turn's prompt, its reply not asked for), then a rewind
    // into the last round: past one pass the ring still holds the window there, past five it does not.
    auto rounds_then_passes = [&](Engine& e, Session& s, size_t passes, std::vector<int32_t>* history, int32_t* pos) {
      std::vector<int32_t> tokens, drafts;
      flyx_round_result last{};
      if (!prefill_span(e, s, wrap_p, 0, 63) || !decode_from(e, s, wrap_p[63], 8, &tokens, &drafts, &last))
        return false;
      history->assign(wrap_p.begin(), wrap_p.begin() + 64);  // position 64 + i: tokens[i], the last the anchor
      history->insert(history->end(), tokens.begin(), tokens.end());
      *pos = last.position - 1;  // inside the last round: [position - n_tokens, position]
      return last.n_tokens >= 1 && prefill_span(e, s, filler, 0, passes * 64);
    };
    for (size_t passes : {size_t(1), size_t(5)}) {
      Session rs(ring_eng.get()), fs(flat_eng.get()), fresh(flat_eng.get());
      std::string err;
      std::vector<int32_t> hist_r, hist_f;
      int32_t pos_r = -1, pos_f = -1;
      bool ok = rs.init(&err) && fs.init(&err) && fresh.init(&err) &&
                rounds_then_passes(*ring_eng, rs, passes, &hist_r, &pos_r) &&
                rounds_then_passes(*flat_eng, fs, passes, &hist_f, &pos_f) && hist_r == hist_f && pos_r == pos_f &&
                pos_r >= 64;
      const int32_t anchor = ok ? hist_r[size_t(pos_r)] : 0;
      const int ring_rewind = ok ? api->rewind(rs.seq(), pos_r, anchor) : -1;
      const int flat_rewind = ok ? flat_eng->api()->rewind(fs.seq(), pos_r, anchor) : -1;
      std::vector<int32_t> got, got_dr, want, want_dr;
      if (passes == 1) {
        ok = ok && ring_rewind == 1 && flat_rewind == 1 && decode_from(*ring_eng, rs, anchor, 24, &got, &got_dr) &&
             decode_from(*flat_eng, fs, anchor, 24, &want, &want_dr);
        check(ok && same_tokens(got, want) && !want_dr.empty() && got_dr == want_dr,
              "  a rewind into the last round past one prefill pass: done, the flat cache's tokens and drafts");
      } else {
        // The ring's state reset: the sequence again from its start, as a fresh session of the flat cache.
        ok = ok && ring_rewind == 0 && flat_rewind == 1 && prefill_span(*ring_eng, rs, hist_r, 0, size_t(pos_r)) &&
             prefill_span(*flat_eng, fresh, hist_r, 0, size_t(pos_r)) &&
             decode_from(*ring_eng, rs, anchor, 24, &got, &got_dr) &&
             decode_from(*flat_eng, fresh, anchor, 24, &want, &want_dr);
        check(ok && same_tokens(got, want) && !want_dr.empty() && got_dr == want_dr,
              "  past five prefill passes that overwrote its window: refused (" + std::to_string(ring_rewind) +
                  ", flat " + std::to_string(flat_rewind) + "); recomputed, a fresh prefill's tokens and drafts");
      }
    }
    // Through sessions: a 300-token prompt checkpointed at 64, and its reply; then the prompt cut at 64 and three
    // other tokens, whose rounds draft from the window before 64.  The flat cache restores the checkpoint; the ring,
    // whose window there later positions overwrote, recomputes; both give a fresh session's tokens and drafts.
    {
      std::vector<int32_t> edited(wrap_p.begin(), wrap_p.begin() + 64);
      edited.insert(edited.end(), {wrap_p[64] == 5 ? 6 : 5, 17, 33});
      Session rs(ring_eng.get()), fs(flat_eng.get()), fresh(flat_eng.get());
      std::string err;
      if (!rs.init(&err) || !fs.init(&err) || !fresh.init(&err)) throw std::runtime_error(err);
      std::vector<flyx_round_result> first, tr, tf, tn;
      generate_traced(rs, wrap_p, 48, {64}, &first);
      generate_traced(fs, wrap_p, 48, {64}, &first);
      int reused_r = -1, reused_f = -1;
      std::string how_r, how_f;
      const auto got_r = generate_traced(rs, edited, 24, {}, &tr, &reused_r, &how_r);
      const auto got_f = generate_traced(fs, edited, 24, {}, &tf, &reused_f, &how_f);
      const auto want = generate_traced(fresh, edited, 24, {}, &tn);
      check(reused_r == 0 && reused_f == 64 && same_tokens(got_r, want) && same_tokens(got_f, want) &&
                same_rounds(tr, tn) && same_rounds(tf, tn),
            "  a prompt edited at a checkpoint whose window the ring lost: recomputed (" + how_r +
                "), the flat cache's restored (" + how_f + "); both a fresh session's tokens and drafts");
    }
  }
  // The engines above given back before these load (dflash_file: no more alive at a time than in its loop).
  against_flat(dir + "/tiny_draft_1layer.flyx", FLYX_KV_FP8, "a DFlash2 drafter of one layer beside FP8 caches");
}

// The MTP layer's cache past 8 of its windows as a ring of slots (backend option mtp_ring, on by default) beside the
// same cache a slot a position (mtp_ring=0), on the tiny Qwen3.5 whose GDN heads of 128 let a pass keep its state at a
// boundary inside it (prefill_align > 1): mtp_window=96 at max_ctx 800 and prefill passes of 128 rows, the ring its
// window, a pass and its first 16 positions (mtp_copy=0, binary16) or none (beside a 2-bit copy of every position,
// mtp_copy=2, FP8).  The ring goes to host memory with every checkpoint, as a sliding-window layer's does: the rows
// past a checkpoint overwrite the window its restore, its snapshot or a rewind reads (the ring once handed them over
// as they were).  Each gives the tokens and drafts of a fresh prefill there:
// * a checkpoint after a pass, restored before the rows past it went round the ring; one kept inside a pass
//   (prefill_snapshot), whose rows past it overwrote the window it reads; both restored, the newer then the older, and
//   saved, after rows that went round the ring;
// * a rewind into the last round past one short prefill pass: done; past passes that went round its window: refused
//   (the state reset, as outside the rollback window), then recomputed;
// * through sessions: a prompt edited at a checkpoint the reply went round restores it.
void mtp_ring(const std::string& dir, const std::string& backend, const std::vector<int32_t>& p) {
  const std::string model = dir + "/tiny_gdn128.flyx";
  std::vector<int32_t> prompt, filler;
  while (prompt.size() < 384) prompt.insert(prompt.end(), p.begin(), p.end());
  prompt.resize(384);
  while (filler.size() < 384) filler.insert(filler.end(), p.rbegin(), p.rend());
  filler.resize(384);
  struct Case {
    const char* options;
    int kv;
    const char* what;
  };
  const Case cases[] = {
      {"mtp_window=96,mtp_sink=16,mtp_copy=0", FLYX_KV_16, "MTP ring, its first 16 and last 96 positions"},
      {"mtp_window=96,mtp_copy=2", FLYX_KV_FP8, "MTP ring, its last 96 beside a 2-bit copy, FP8"}};
  for (const Case& c : cases) {
    const std::string which = c.what;
    const Load ring_load = {.speculative = true,
                            .prefill_chunk = 128,
                            .max_ctx = 800,
                            .drafter = "mtp",
                            .options = c.options,
                            .kv = c.kv};
    Load flat_load = ring_load;
    flat_load.options += ",mtp_ring=0";
    auto ring_eng = engine(model, backend, ring_load);
    auto flat_eng = engine(model, backend, flat_load);
    const auto* api = ring_eng->api();
    uint64_t ring_bytes = 0, flat_bytes = 0;
    const bool sized = api->seq_bytes(ring_eng->bmodel(), ring_sizing_ctx, &ring_bytes) == 0 &&
                       flat_eng->api()->seq_bytes(flat_eng->bmodel(), ring_sizing_ctx, &flat_bytes) == 0;
    check(sized && ring_bytes < flat_bytes,
          which + ": a ring of slots (" + std::to_string(ring_bytes) + " < " + std::to_string(flat_bytes) + " bytes)");
    // The flat cache's continuation of a fresh prefill of the prompt to `at`, in the same passes.
    auto fresh_from = [&](size_t at, std::vector<int32_t>* tokens, std::vector<int32_t>* drafts) {
      Session fresh(flat_eng.get());
      std::string err;
      return fresh.init(&err) && prefill_span(*flat_eng, fresh, prompt, 0, at, 128) &&
             decode_from(*flat_eng, fresh, prompt[at], 24, tokens, drafts);
    };
    auto same = [](const std::vector<int32_t>& got, const std::vector<int32_t>& got_dr,
                   const std::vector<int32_t>& want, const std::vector<int32_t>& want_dr) {
      return same_tokens(got, want) && !want_dr.empty() && got_dr == want_dr;
    };
    std::vector<int32_t> want128, want128_dr;
    const bool fresh_ok = fresh_from(128, &want128, &want128_dr);
    {
      Session src(ring_eng.get()), dst(ring_eng.get());
      std::string err;
      // A checkpoint after the first pass; 16 rows past it, none round the ring yet: restored.
      bool ok = fresh_ok && src.init(&err) && dst.init(&err) && prefill_span(*ring_eng, src, prompt, 0, 128, 128) &&
                api->checkpoint(src.seq(), 0) == 128 && prefill_span(*ring_eng, src, prompt, 128, 144, 128) &&
                api->restore(src.seq(), 0, prompt[128]) == 128;
      std::vector<int32_t> got, got_dr;
      ok = ok && decode_from(*ring_eng, src, prompt[128], 24, &got, &got_dr);
      check(ok && same(got, got_dr, want128, want128_dr),
            "  a checkpoint restored before the rows past it went round the ring: a fresh prefill's tokens and drafts");
      // From it again, a pass of 128 rows keeping the state at 192 inside it (prefill_snapshot), then 384 rows: the
      // ring gone round since either checkpoint.
      // Where the device's GDN prefill is chunked (gdn_chunked_device) a pass keeps a state inside it; elsewhere
      // (prefill_align 1) the checkpoint is the pass's end.
      const int align = ring_eng->prefill_align();
      const bool inside = align > 1 && 192 % align == 0 && api->prefill_snapshot;
      check(inside == flyx::test::gdn_chunked_device(api),
            "  passes that keep a state inside them where the device's GDN prefill is chunked (prefill_align " +
                std::to_string(align) + ")");
      ok = ok && api->restore(src.seq(), 0, prompt[128]) == 128 &&
           (!inside || api->prefill_snapshot(src.seq(), 192) == 0) &&
           prefill_span(*ring_eng, src, prompt, 128, 256, 128);
      const int32_t at = ok ? api->checkpoint(src.seq(), 1) : -1;
      ok = ok && at == (inside ? 192 : 256) && prefill_span(*ring_eng, src, filler, 0, 384, 128);
      std::vector<int32_t> want_at, want_at_dr;
      ok = ok && fresh_from(size_t(at), &want_at, &want_at_dr);
      // Saved now (from its copy, not from the ring as it is), before the sequence goes back before it.
      uint64_t sb = 0;
      ok = ok && api->snapshot_bytes(src.seq(), 1, 0, &sb) == 0 && sb > 0;
      std::vector<uint8_t> buf(size_t(std::max<uint64_t>(1, sb)));
      void* q = buf.data();
      flyx_host_pieces pieces{&q, 1, 0, sb};
      ok = ok && api->snapshot_save(src.seq(), 1, 0, &pieces, sb) == at;
      std::vector<int32_t> newer, newer_dr, older, older_dr;
      ok = ok && api->restore(src.seq(), 1, prompt[size_t(at)]) == at &&
           decode_from(*ring_eng, src, prompt[size_t(at)], 24, &newer, &newer_dr) &&
           api->restore(src.seq(), 0, prompt[128]) == 128 &&
           decode_from(*ring_eng, src, prompt[128], 24, &older, &older_dr);
      check(ok && same(newer, newer_dr, want_at, want_at_dr) && same(older, older_dr, want128, want128_dr),
            "  after the ring went round: one kept inside a pass (" + std::to_string(at) +
                ") restored, then the older one: each a fresh prefill's tokens and drafts");
      std::vector<int32_t> loaded, loaded_dr;
      ok = ok && api->snapshot_load(dst.seq(), &pieces, sb, prompt[size_t(at)]) == at &&
           decode_from(*ring_eng, dst, prompt[size_t(at)], 24, &loaded, &loaded_dr);
      check(ok && same(loaded, loaded_dr, want_at, want_at_dr),
            "  saved after the ring went round and loaded elsewhere: a fresh prefill's tokens and drafts");
    }
    // Rounds after a prompt, then prefill passes past them (the next turn's prompt, its reply not asked for), then a
    // rewind into the last round: past one pass of 64 rows the ring still holds the window there, past three not.
    auto rounds_then_passes = [&](Engine& e, Session& s, size_t rows, std::vector<int32_t>* history, int32_t* pos) {
      std::vector<int32_t> tokens, drafts;
      flyx_round_result last{};
      if (!prefill_span(e, s, prompt, 0, 255, 128) || !decode_from(e, s, prompt[255], 8, &tokens, &drafts, &last))
        return false;
      history->assign(prompt.begin(), prompt.begin() + 256);  // position 256 + i: tokens[i], the last the anchor
      history->insert(history->end(), tokens.begin(), tokens.end());
      *pos = last.position - 1;  // inside the last round: [position - n_tokens, position]
      return last.n_tokens >= 1 && prefill_span(e, s, filler, 0, rows, 64);
    };
    for (size_t rows : {size_t(64), size_t(192)}) {
      Session rs(ring_eng.get()), fs(flat_eng.get()), fresh(flat_eng.get());
      std::string err;
      std::vector<int32_t> hist_r, hist_f;
      int32_t pos_r = -1, pos_f = -1;
      bool ok = rs.init(&err) && fs.init(&err) && fresh.init(&err) &&
                rounds_then_passes(*ring_eng, rs, rows, &hist_r, &pos_r) &&
                rounds_then_passes(*flat_eng, fs, rows, &hist_f, &pos_f) && hist_r == hist_f && pos_r == pos_f &&
                pos_r >= 256;
      const int32_t anchor = ok ? hist_r[size_t(pos_r)] : 0;
      const int ring_rewind = ok ? api->rewind(rs.seq(), pos_r, anchor) : -1;
      const int flat_rewind = ok ? flat_eng->api()->rewind(fs.seq(), pos_r, anchor) : -1;
      std::vector<int32_t> got, got_dr, want, want_dr;
      if (rows == 64) {
        ok = ok && ring_rewind == 1 && flat_rewind == 1 && decode_from(*ring_eng, rs, anchor, 24, &got, &got_dr) &&
             decode_from(*flat_eng, fs, anchor, 24, &want, &want_dr);
        check(ok && same(got, got_dr, want, want_dr),
              "  a rewind into the last round past one pass of 64 rows: done, the flat cache's tokens and drafts");
      } else {
        ok = ok && ring_rewind == 0 && flat_rewind == 1 && prefill_span(*ring_eng, rs, hist_r, 0, size_t(pos_r)) &&
             prefill_span(*flat_eng, fresh, hist_r, 0, size_t(pos_r)) &&
             decode_from(*ring_eng, rs, anchor, 24, &got, &got_dr) &&
             decode_from(*flat_eng, fresh, anchor, 24, &want, &want_dr);
        check(ok && same(got, got_dr, want, want_dr),
              "  past 192 rows that went round its window: refused (" + std::to_string(ring_rewind) + ", flat " +
                  std::to_string(flat_rewind) + "); recomputed, a fresh prefill's tokens and drafts");
      }
    }
    // Through sessions: a 300-token prompt checkpointed at 256, and its reply; then the prompt cut at 256 and three
    // other tokens, whose rounds draft from the window before 256, which the reply's rows went round.  Both caches
    // restore the checkpoint, with a fresh session's tokens and drafts.
    {
      const std::vector<int32_t> first(prompt.begin(), prompt.begin() + 300);
      std::vector<int32_t> edited(prompt.begin(), prompt.begin() + 256);
      edited.insert(edited.end(), {prompt[256] == 5 ? 6 : 5, 17, 33});
      Session rs(ring_eng.get()), fs(flat_eng.get()), fresh(flat_eng.get());
      std::string err;
      if (!rs.init(&err) || !fs.init(&err) || !fresh.init(&err)) throw std::runtime_error(err);
      std::vector<flyx_round_result> ignored, tr, tf, tn;
      generate_traced(rs, first, 48, {256}, &ignored);
      generate_traced(fs, first, 48, {256}, &ignored);
      int reused_r = -1, reused_f = -1;
      std::string how_r, how_f;
      const auto got_r = generate_traced(rs, edited, 24, {}, &tr, &reused_r, &how_r);
      const auto got_f = generate_traced(fs, edited, 24, {}, &tf, &reused_f, &how_f);
      const auto want = generate_traced(fresh, edited, 24, {}, &tn);
      check(reused_r == 256 && reused_f == 256 && same_tokens(got_r, want) && same_tokens(got_f, want) &&
                same_rounds(tr, tn) && same_rounds(tf, tn),
            "  a prompt edited at a checkpoint the reply went round: restored (" + how_r +
                "), a fresh session's tokens and drafts");
    }
  }
}

// A DSpine drafter in a file of its own (tencent publishes them so), and the directory tencent publishes it as
// (hf_dspine), beside a GGUF model:
// * beside the tiny Qwen3.5 as GGUF (float32): the golden tokens (golden_dspine.json's), and its drafts the
//   reference's, round by round (on the float64 backend all of them);
// * beside the quantized GGUF (a packed embedding and head, the rows its walk gathers; its layers' copies Q8_0 beside
//   GGML block weights): it drafts, and the tokens are plain decoding's, greedy and seeded sampling's.
void dspine_file(const std::string& dir, const std::string& backend) {
  std::ifstream f(dir + "/golden_dspine.json");
  const json golden = json::parse(f);
  const std::string q8 = dir + "/tiny_q8_target.gguf";
  auto plain = engine(q8, backend);
  for (const char* file : {"tiny_draft_dspine.flyx", "hf_dspine"}) {
    std::printf("  the DSpine drafter from %s\n", file);
    // Its drafts from its exact weights (beside a float target CUDA's default is NVFP4 copies), as the reference's.
    auto spec = engine(dir + "/tiny_gguf.gguf", backend,
                       {.speculative = true, .drafter = "dspine", .drafter_files = {dir + "/" + file},
                        .options = "draft_weights=exact"});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      std::vector<int32_t> got;
      const auto trace = traced(*spec, p, 48, &got);
      int same = 0, total = 0;
      const json& rounds = item.at("rounds");
      for (size_t i = 1; i < trace.size() && i - 1 < rounds.size(); ++i) {
        const auto d = rounds[i - 1].at("drafts").get<std::vector<int32_t>>();
        for (int j = 0; j < trace[i].n_drafts && j < int(d.size()); ++j, ++total) same += trace[i].drafts[j] == d[size_t(j)];
      }
      check(same_tokens(got, want) && total > 0 && (backend == "cpu_ref" ? same == total : same * 10 >= total * 9),
            std::string(file) + " beside a GGUF model: the golden tokens; drafts equal to the reference's " +
                std::to_string(same) + " / " + std::to_string(total));
    }
    auto packed = engine(q8, backend, {.speculative = true, .drafter = "dspine", .drafter_files = {dir + "/" + file}});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      for (bool sampling : {false, true}) {
        std::vector<int32_t> got;
        const auto rounds = traced(*packed, p, 32, &got, sampling);
        const bool drafted = std::any_of(rounds.begin(), rounds.end(), [](const auto& r) { return r.n_drafts > 0; });
        check(drafted && same_tokens(got, tokens_of(*plain, p, 32, sampling)),
              std::string(file) + " beside the quantized GGUF: it drafts, and plain decoding's tokens (" +
                  (sampling ? "seeded sampling" : "greedy") + ")");
      }
    }
  }
}

// A DFlash2 drafter in a file of its own (z-lab publishes them so), beside a model of either format.
// * tiny.flyx's drafter alone (tiny_draft.flyx, converted from z-lab's layout) beside the tiny Qwen3.5 as GGUF
//   (float32, no drafter): the golden tokens, and on the float64 backend the reference's drafts, round by round;
//   with the GGUF's MTP layer beside it, either drafter as asked.
// * The quantized GGUF without its MTP layer (a packed embedding and head, which the drafter reads): plain
//   decoding's tokens, greedy and sampled, with the drafter and with its copy that attends a sliding window of 8;
//   on another backend, from exact weights and binary16 inputs, nearly all of the float64 backend's drafts on the
//   same files.  (With 8-bit inputs, a group of 32 scaled by its largest, one float32 step can move the target's
//   features enough to turn a near tie among the drafts, and the walk through the block carries a turned draft
//   on to every later one.)
// * A drafter file is no model; a drafter made for another model is refused; a model takes one block drafter.
void dflash_file(const std::string& dir, const std::string& backend) {
  const std::string drafter = dir + "/tiny_draft.flyx", window = dir + "/tiny_draft_window.flyx";
  const std::string drafter_gguf = dir + "/tiny_draft.gguf", window_gguf = dir + "/tiny_draft_window.gguf";
  // The GGUF drafter in Q8_0, and with the block types mixed as a Q4_K_M file mixes them: its stacked q|k|v and its
  // context k|v are then runs of more than one type.
  const std::string drafter_q8 = dir + "/tiny_draft_q8.gguf", drafter_mixed = dir + "/tiny_draft_mixed.gguf";
  std::ifstream f(dir + "/golden.json");
  const json golden = json::parse(f);
  // The drafter file (.flyx), the directory z-lab publishes it as, and the arch="dflash" GGUF.
  for (const std::string& source : {drafter, dir + "/hf_draft_zlab", drafter_gguf}) {
    std::printf("  the drafter from %s\n", source.substr(dir.size() + 1).c_str());
    auto dflash = engine(dir + "/tiny_gguf.gguf", backend,
                         {.speculative = true, .drafter = "dflash2", .drafter_files = {source}});
    auto mtp = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true, .drafter_files = {drafter}});
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      const auto want = item.at("tokens").get<std::vector<int32_t>>();
      std::vector<int32_t> got;
      const auto trace = traced(*dflash, p, 48, &got);
      int same = 0, total = 0;
      const json& rounds = item.at("rounds");
      for (size_t i = 1; i < trace.size() && i - 1 < rounds.size(); ++i) {
        const auto d = rounds[i - 1].at("drafts").get<std::vector<int32_t>>();
        for (int j = 0; j < trace[i].n_drafts && j < int(d.size()); ++j, ++total) same += trace[i].drafts[j] == d[size_t(j)];
      }
      check(same_tokens(got, want) && total > 0 && (backend != "cpu_ref" || same == total),
            "DFlash2 from a drafter file beside a GGUF model: the golden tokens; drafts equal to the reference's " +
                std::to_string(same) + " / " + std::to_string(total));
      check(same_tokens(tokens_of(*mtp, p, 48), want), "  the GGUF's MTP layer beside it, when asked for");
    }
  }
  const std::string q8 = dir + "/tiny_q8_target.gguf";
  auto plain = engine(q8, backend);
  for (const std::string& file : {drafter, window, dir + "/hf_draft_zlab", dir + "/hf_draft_window", drafter_gguf,
                                  window_gguf, drafter_q8, drafter_mixed}) {
    const std::string which = std::string(file.find("window") != std::string::npos ? "a DFlash2 drafter with a sliding window"
                                          : file.find("_q8") != std::string::npos  ? "a Q8_0 DFlash2 GGUF drafter"
                                          : file.find("_mixed") != std::string::npos
                                              ? "a DFlash2 GGUF drafter of mixed block types"
                                              : "a DFlash2 drafter") +
                              (file.find("/hf_") != std::string::npos ? " (its directory)"
                               : file.ends_with(".gguf")              ? " (GGUF)"
                                                                      : "");
    const Load drafted = {.speculative = true, .drafter = "dflash2", .drafter_files = {file}};
    Load in16 = drafted, exact = drafted;
    in16.activations = exact.activations = "exact";
    exact.options = "draft_weights=exact";
    auto spec = engine(q8, backend, drafted);
    auto ref = backend == "cpu_ref" ? nullptr : engine(q8, "cpu_ref", in16);
    auto spec_exact = backend == "cpu_ref" ? nullptr : engine(q8, backend, exact);
    for (const auto& item : golden.at("cases")) {
      const auto p = item.at("prompt").get<std::vector<int32_t>>();
      std::vector<int32_t> got;
      int accepted = 0;
      for (const auto& r : traced(*spec, p, 48, &got)) accepted += r.n_accepted;
      check(same_tokens(got, tokens_of(*plain, p, 48)),
            which + " beside the quantized GGUF: plain decoding's tokens (" + std::to_string(accepted) +
                " drafts accepted)");
      check(tokens_of(*plain, p, 32, true) == tokens_of(*spec, p, 32, true), "  and seeded sampling's");
      if (!ref) continue;
      std::vector<int32_t> ta, tb;
      const auto a = traced(*spec_exact, p, 48, &ta), b = traced(*ref, p, 48, &tb);
      int same = 0, total = 0;
      for (size_t i = 1; i < a.size() && i < b.size(); ++i)
        for (int j = 0; j < a[i].n_drafts && j < b[i].n_drafts; ++j, ++total) same += a[i].drafts[j] == b[i].drafts[j];
      check(ta == tb && total > 0 && same * 10 >= total * 9,
            "  from exact weights and binary16 inputs, the float64 backend's drafts on the same files: " +
                std::to_string(same) + " / " + std::to_string(total));
    }
  }
  // This test runs four times at once (ctest -j: fused, generic, PDL on and off) beside the golden tests, all on one
  // device: from here on an engine is given back before the next ones load, so that no more are alive at a time
  // than in the loop above.
  plain.reset();
  // The packed drafters beside a BF16 target (packed GGML weights inside DFlash2 when the target itself is
  // non-quantized), plus fusion=0 parity and draft_weights=nvfp4/q8 regressions on CUDA.
  {
    const auto p = golden.at("cases").at(0).at("prompt").get<std::vector<int32_t>>();
    const auto want = tokens_of(*engine(dir + "/tiny_gguf.gguf", backend), p, 32);
    for (const std::string& packed : {drafter_q8, drafter_mixed}) {
      const std::string which = packed.substr(dir.size() + 1);
      const Load drafted = {.speculative = true, .drafter = "dflash2", .drafter_files = {packed}};
      auto fused = engine(dir + "/tiny_gguf.gguf", backend, drafted);
      check(same_tokens(tokens_of(*fused, p, 32), want), which + " beside BF16 target: plain decoding's tokens");
      if (backend == "cpu_ref") continue;
      Load generic_load = drafted;
      generic_load.options = "fusion=0";
      auto generic = engine(dir + "/tiny_gguf.gguf", backend, generic_load);
      std::vector<int32_t> ta, tb;
      const auto a = traced(*fused, p, 32, &ta), b = traced(*generic, p, 32, &tb);
      bool same_drafts = ta == tb && a.size() == b.size();
      for (size_t i = 1; same_drafts && i < a.size(); ++i)
        same_drafts = a[i].n_drafts == b[i].n_drafts &&
                      std::equal(a[i].drafts, a[i].drafts + a[i].n_drafts, b[i].drafts);
      check(same_drafts, which + ": fused == generic (fusion=0) tokens and drafts");
      fused.reset();
      generic.reset();
      // Streaming GGML->NVFP4 conversion on Q8_0 and mixed Q4_0+Q8_0 stacks (where attn.v is shared across
      // layer w_qkv and context ctx): must never release shared WeightLoader::packs_ entries and must preserve output.
      Load fp4_load = drafted;
      fp4_load.prefill_chunk = 64;
      fp4_load.options = "draft_weights=nvfp4";
      auto fp4 = engine(dir + "/tiny_gguf.gguf", backend, fp4_load);
      Load fp4_gen = fp4_load;
      fp4_gen.options = "draft_weights=nvfp4,fusion=0";
      auto fp4_g = engine(dir + "/tiny_gguf.gguf", backend, fp4_gen);
      std::vector<int32_t> tfa, tfb;
      const auto fa = traced(*fp4, p, 32, &tfa), fb = traced(*fp4_g, p, 32, &tfb);
      bool same_fp4 = same_tokens(tfa, want) && tfa == tfb && fa.size() == fb.size();
      for (size_t i = 1; same_fp4 && i < fa.size(); ++i)
        same_fp4 = fa[i].n_drafts == fb[i].n_drafts &&
                   std::equal(fa[i].drafts, fa[i].drafts + fa[i].n_drafts, fb[i].drafts);
      check(same_fp4, which + " draft_weights=nvfp4: exact target tokens and fused == generic drafts");
    }
    // On-the-fly Q8_0 drafter weights beside a BF16 target (desc has no FLYX_ACT_Q8 weights) with >16-row prefill:
    // exercises RowNeeds::q8 propagation from bound drafter blocks so q8_parts/round_parts scratch is allocated.
    if (backend != "cpu_ref") {
      std::vector<int32_t> long_p = p;
      while (long_p.size() < 48) long_p.insert(long_p.end(), p.begin(), p.end());
      long_p.resize(48);
      const auto want_long = tokens_of(*engine(dir + "/tiny_gguf.gguf", backend, {.prefill_chunk = 64}), long_p, 24);
      auto bf16_q8_draft = engine(dir + "/tiny_gguf.gguf", backend,
                                  {.speculative = true,
                                   .prefill_chunk = 64,
                                   .drafter = "dflash2",
                                   .drafter_files = {drafter},
                                   .options = "draft_weights=q8"});
      check(same_tokens(tokens_of(*bf16_q8_draft, long_p, 24), want_long),
            "BF16 target + DFlash2 draft_weights=q8 with >16-row prefill: plain decoding's tokens");
    }
    // The sliding-window drafter's context cache as a ring of slots (dflash_ring), and the MTP layer's (mtp_ring).
    if (backend != "cpu_ref") dflash_ring(dir, backend, p);
    if (backend != "cpu_ref") mtp_ring(dir, backend, p);
  }
  // The dense targets with a DFlash2 GGUF drafter made for them: Qwen3 (GGUF and .flyx) and Llama (plain and with
  // its own adapters, a V-LoRA).
  {
    struct Dense {
      const char* model;
      const char* drafter;
    };
    const std::vector<int32_t> prompt = {12, 37, 81, 104};
    for (const Dense& t : {Dense{"tiny_qwen3_gguf.gguf", "tiny_qwen3_draft.gguf"},
                           Dense{"tiny_qwen3.flyx", "tiny_qwen3_draft.gguf"},
                           Dense{"tiny_llama.flyx", "tiny_llama_draft.gguf"},
                           Dense{"tiny_llama_lora.flyx", "tiny_llama_draft.gguf"}}) {
      const std::string model = dir + "/" + t.model;
      auto plain_dense = engine(model, backend);
      const auto want_g = tokens_of(*plain_dense, prompt, 32, false);
      const auto want_s = tokens_of(*plain_dense, prompt, 24, true);
      plain_dense.reset();
      auto spec_dense = engine(model, backend,
                               {.speculative = true, .drafter = "dflash2", .drafter_files = {dir + "/" + t.drafter}});
      std::vector<int32_t> got_g;
      int total_drafts = 0;
      for (const auto& r : traced(*spec_dense, prompt, 32, &got_g)) total_drafts += r.n_drafts;
      check(same_tokens(got_g, want_g) && total_drafts > 0,
            std::string(t.model) + " + DFlash2 GGUF: greedy matches plain (" + std::to_string(total_drafts) +
                " drafts)");
      check(tokens_of(*spec_dense, prompt, 24, true) == want_s,
            std::string(t.model) + " + DFlash2 GGUF: seeded sampling matches plain");
    }
  }
  {  // The drafters a model holds unless one is named (models/model.h: choose_drafters).
    auto own = engine(dir + "/tiny.flyx", backend, {.speculative = true, .drafter = ""});
    auto all = engine(dir + "/tiny.flyx", backend, {.speculative = true, .drafter = "all"});
    auto beside = engine(dir + "/tiny_gguf.gguf", backend, {.speculative = true, .drafter = "", .drafter_files = {drafter}});
    const std::string a = own->drafter(), b = all->drafter(), c = beside->drafter();
    check(a == "mtp" && b == "mtp+dflash2" && c == "mtp+dflash2",
          "the drafters held unless named: the model file's MTP layer of its two (" + a + "), all (" + b +
              "), and with a drafter file beside it, its drafter too (" + c + ")");
  }
  std::string why = refusal(drafter, backend, {});
  check(why.find("a drafter alone, not a model") != std::string::npos, "a drafter file alone is no model (" + why + ")");
  why = refusal(drafter_gguf, backend, {});
  check(why.find("a drafter alone, not a model") != std::string::npos,
        "a DFlash2 GGUF alone is no model (" + why + ")");
  why = refusal(dir + "/tiny_w4a4.flyx", backend, {.speculative = true, .drafter = "dflash2", .drafter_files = {drafter}});
  check(why.find("another model") != std::string::npos,
        "a drafter made for another model (twice as wide) is refused (" + why + ")");
  why = refusal(dir + "/tiny_gguf.gguf", backend,
                {.speculative = true, .drafter = "dflash2", .drafter_files = {drafter, window}});
  check(why.find("block drafter already") != std::string::npos, "a model takes one block drafter (" + why + ")");
}

// NVFP4 weights with NVFP4 inputs in passes of more than 16 rows against the float64 backend, which rounds the same
// inputs the same way in passes of the same rows: so the reference prefills in the same passes as the backend.  A
// pass of 64 rows rounds its inputs to 4 bits, and one step upstream (a float32 norm against a float64 one) can move
// a value across a rounding boundary: where the reference's two best tokens nearly tie the backend may pick the other,
// so the reply is compared teacher forced, as the packed types are.
void nvfp4_inputs(const std::string& dir, const std::string& backend) {
  const std::string path = dir + "/tiny_w4a4.flyx";
  if (!fp4_inputs_here(path, backend, "NVFP4 inputs").empty()) {  // what this checks needs FP4 tensor cores
    // Asked for by name where the device cannot compute them: refused at open, not bf16 inputs instead.
    const std::string why = refusal(dir + "/tiny_w4a4_a16.flyx", backend, {.activations = "nvfp4/bf16"});
    check(why.find("cannot compute") != std::string::npos,
          "NVFP4 inputs asked for without FP4 tensor cores: refused (" + why + ")");
    return;
  }
  std::vector<int32_t> prompt;
  for (int i = 0; i < 80; ++i) prompt.push_back((7 * i + 20) % 700);
  for (int chunk : {16, 64}) {
    const auto want = tokens_of(*engine(path, "cpu_ref", {.prefill_chunk = chunk}), prompt, 32);
    const int agree = forced_agreement(*engine(path, backend, {.prefill_chunk = chunk}), prompt, want);
    check(agree >= 30, "NVFP4 inputs: teacher forced, the float64 backend's tokens at " + std::to_string(agree) +
                           " of 32 places, prefill in passes of up to " + std::to_string(chunk) + " rows");
  }
  auto plain = engine(path, backend, {.prefill_chunk = 64});
  {  // the generic regions (fusion=0, gate 2) keep the fused path's bf16 rows: the same state, bit for bit, also
     // with rows of float32 embeddings injected (prefill_ex), which the fused path stores in bf16
    auto generic = engine(path, backend, {.prefill_chunk = 64, .options = "fusion=0"});
    const std::vector<int32_t> pass(prompt.begin(), prompt.begin() + 64), at = {3, 4, 5, 6};
    std::vector<float> embeds(at.size() * size_t(plain->model().desc().hidden));
    for (size_t i = 0; i < embeds.size(); ++i) embeds[i] = float(0.3 * std::sin(0.7 * double(i) + 0.1));
    check(pass_state(*generic, pass) == pass_state(*plain, pass),
          "NVFP4 inputs, a pass of 64 rows: fusion=0 leaves the fused path's state, bit for bit");
    check(pass_state(*generic, pass, at, embeds) == pass_state(*plain, pass, at, embeds),
          "  with 4 rows of float32 embeddings injected: the fused path's state");
    // Injected rows are stored as the residual is, in bf16: their bf16 values leave the same state, here and on the
    // float64 backend, which keeps the residual so in such a pass.
    std::vector<float> rounded(embeds.size());
    for (size_t i = 0; i < embeds.size(); ++i) rounded[i] = float(ops::round_bf16(embeds[i]));
    auto reference = engine(path, "cpu_ref", {.prefill_chunk = 64});
    for (Engine* e : {plain.get(), reference.get()})
      check(pass_state(*e, pass, at, embeds) == pass_state(*e, pass, at, rounded),
            "  " + e->backend_name() + ": the embeddings' bf16 values leave the same state (bf16 residual rows)");
  }
  auto spec = engine(path, backend, {.speculative = true, .prefill_chunk = 64});
  check(tokens_of(*plain, prompt, 32) == tokens_of(*spec, prompt, 32), "NVFP4 inputs: the MTP drafter changes nothing");
  // Asked for by name (--activations nvfp4/bf16) in a file of the same weights that says bf16 inputs: the tokens of
  // the file that says NVFP4, greedy and sampled; by default, that file's with --activations exact.
  const std::string a16 = dir + "/tiny_w4a4_a16.flyx";
  auto asked = engine(a16, backend, {.prefill_chunk = 64, .activations = "nvfp4/bf16"});
  auto as_said = engine(a16, backend, {.prefill_chunk = 64});
  auto exact = engine(path, backend, {.prefill_chunk = 64, .activations = "exact"});
  for (bool sampling : {false, true}) {
    const std::string how = sampling ? "sampled" : "greedy";
    check(tokens_of(*asked, prompt, 32, sampling) == tokens_of(*plain, prompt, 32, sampling),
          "NVFP4 inputs asked for in a file that says bf16: the tokens of the file that says NVFP4 (" + how + ")");
    check(tokens_of(*as_said, prompt, 32, sampling) == tokens_of(*exact, prompt, 32, sampling),
          "  not asked for: that file's with --activations exact (" + how + ")");
  }
  // Asked for where this backend cannot compute them, refused: 64-wide NVFP4 weights (the W4A4 kernels' tile of K is
  // 128). A model without NVFP4 weights gives what it gives without the word.
  const std::string why = refusal(dir + "/tiny_nvfp4.flyx", backend, {.activations = "nvfp4/bf16"});
  check(why.find("a multiple of 128") != std::string::npos,
        "NVFP4 inputs asked for with K = 64: refused (" + why + ")");
  const std::vector<int32_t> short_prompt(prompt.begin(), prompt.begin() + 40);
  auto no_nvfp4 = engine(dir + "/tiny.flyx", backend, {.prefill_chunk = 64, .activations = "nvfp4/bf16"});
  check(tokens_of(*no_nvfp4, short_prompt, 16) ==
            tokens_of(*engine(dir + "/tiny.flyx", backend, {.prefill_chunk = 64}), short_prompt, 16),
        "NVFP4 inputs asked for without NVFP4 weights: the tokens without the word");
  // PREFILL/DECODE: NVFP4 rounds are no mode of its (refused, naming the modes)
  const std::string bare = refusal(path, backend, {.activations = "nvfp4"});
  check(bare.find("nvfp4/bf16") != std::string::npos, "--activations nvfp4 (NVFP4 rounds too): refused (" + bare + ")");
}
void split_shards(const std::string& dir, const std::string& backend) {
  auto single = engine(dir + "/tiny_gguf.gguf", backend);
  auto split_plain = engine(dir + "/tiny_split-00001-of-00003.gguf", backend);
  auto split_spec = engine(dir + "/tiny_split-00001-of-00003.gguf", backend, {.speculative = true});
  std::ifstream f(dir + "/golden.json");
  const auto g = json::parse(f);
  const auto prompt = g.at("cases").at(0).at("prompt").get<std::vector<int32_t>>();
  const auto want = tokens_of(*single, prompt, 24);
  check(same_tokens(tokens_of(*split_plain, prompt, 24), want),
        "3-shard split GGUF gives the single GGUF's tokens (plain)");
  check(same_tokens(tokens_of(*split_spec, prompt, 24), want),
        "3-shard split GGUF gives the single GGUF's tokens (MTP)");
  auto qsingle = engine(dir + "/tiny_q8.gguf", backend);
  auto qsplit = engine(dir + "/tiny_q8_split-00001-of-00002.gguf", backend);
  check(same_tokens(tokens_of(*qsplit, prompt, 24), tokens_of(*qsingle, prompt, 24)),
        "2-shard quantized split GGUF gives the single quantized GGUF's tokens");
}

// Packed block weights beside a PEFT LoRA adapter (--lora: peft_quant/, on some of the linears) against the float64
// backend on the same file and adapter, as packed() holds them alone: the adapters' GEMMs read the float32 rows the
// block GEMMs round to 8 bits (backends/cuda/regions/layers.cpp attn_layer).  The adapter changes the tokens.
void packed_lora(const std::string& dir, const std::string& backend) {
  std::vector<int32_t> prompt;
  for (int i = 0; i < 80; ++i) prompt.push_back(i + 20);
  const Load adapted{.lora_file = dir + "/peft_quant"};
  for (const char* kind : {"Q8_0", "Q4_K", "Q6_K", "IQ4_XS"}) {
    const std::string path = dir + "/quant_" + kind + ".gguf";
    const auto want = tokens_of(*engine(path, "cpu_ref", adapted), prompt, 32);
    check(want != tokens_of(*engine(path, "cpu_ref"), prompt, 32),
          std::string(kind) + " with a LoRA adapter: the float64 backend's tokens are not the plain model's");
    for (int chunk : {16, 64}) {
      Load how = adapted;
      how.prefill_chunk = chunk;
      const int agree = forced_agreement(*engine(path, backend, how), prompt, want);
      check(agree >= 30, std::string(kind) + " with a LoRA adapter: teacher forced, the float64 backend's tokens at " +
                             std::to_string(agree) + " of 32 places, prefill in passes of up to " +
                             std::to_string(chunk) + " rows");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc != 3 && argc != 4) return 2;
  const std::string dir = argv[1], backend = argv[2];
  tested = backend;
  if (argc > 3) tested_options = argv[3];
  try {
    golden(dir, backend);
    split_shards(dir, backend);
    explicit_head_dim(dir, backend);
    split_mtp(dir, backend);
    dflash_file(dir, backend);
    dspine_file(dir, backend);
    gpt_oss(dir, backend);
    gemma4(dir, backend);
    if (backend != "cpu_ref") packed(dir, backend);
    if (backend != "cpu_ref") packed_lora(dir, backend);
    if (backend != "cpu_ref") nvfp4_inputs(dir, backend);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return flyx::test::finish("gguf_e2e_test");
}
