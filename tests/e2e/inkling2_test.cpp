// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// End-to-end verification of Inkling-2 (`inkling2_loop`) and its DSpark companion drafter (`dspark`)
// on self-contained synthetic GGUF models (no PyTorch/assets dependency):
//   * Looped core (prelude + 3-pass core + postlude), AttnRes, cross-pass loop gates, ShortConv (static + dynamic),
//     relative-position-bias attention (SWA ring + Global log-scaled), padded->unpadded vocab + softcap,
//     sparse MoE (SiTU, 3D NVFP4 with per-expert .scale, MXFP4, Q2_0, Q8_0, BF16), and BF16/F32 sinks and taps.
//   * Gate 2: fusion=0 == fusion=1 bit-identical.
//   * Gate 3: CUDA == float64 cpu_ref (greedy, seeded sampling with penalties, prefill chunking, rewind, edit).
//   * Gate 4: Plain == DSpark speculative decoding (greedy & seeded sampling with penalties), CUDA graphs on/off,
//     its drafts from both heads read whole and from their 1-bit copies (the two-stage draft head).
//   * SWA ring wraparound: checkpoint/restore and snapshot_save/snapshot_load past ring capacity.
//   * Multi-sequence concurrent batch (`launch_batch` fallback) and Inkling-2 chat template / format / grammar.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/exec/generation.h"
#include "core/grammar/grammar.h"
#include "core/render/chat.h"
#include "core/session/store.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

using namespace flyx;
using json = nlohmann::json;

namespace {

bool same_tokens(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
  if (got == want) return true;
  size_t i = 0;
  while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
  std::printf("    tokens diverge at %zu (got len %zu val %d vs want len %zu val %d)\n", i, got.size(),
              i < got.size() ? got[i] : -1, want.size(), i < want.size() ? want[i] : -1);
  return false;
}

// ---- Minimal GGUF v3 binary writer ----------------------------------------------------------

constexpr uint32_t kGgufTypeU32 = 4;
constexpr uint32_t kGgufTypeF32 = 6;
constexpr uint32_t kGgufTypeBool = 7;
constexpr uint32_t kGgufTypeStr = 8;
constexpr uint32_t kGgufTypeArr = 9;

constexpr uint32_t kGgmlF32 = 0;
constexpr uint32_t kGgmlQ8_0 = 8;
constexpr uint32_t kGgmlBF16 = 30;
constexpr uint32_t kGgmlMXFP4 = 39;
constexpr uint32_t kGgmlNVFP4 = 40;
constexpr uint32_t kGgmlQ2_0 = 42;

uint16_t f32_to_bf16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

uint16_t f32_to_f16(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  const uint16_t sign = uint16_t((u >> 16) & 0x8000u);
  const int exp = int((u >> 23) & 0xFFu) - 127 + 15;
  const uint32_t mant = u & 0x7FFFFFu;
  if (exp <= 0) return sign;
  if (exp >= 31) return uint16_t(sign | 0x7BFFu);
  return uint16_t(sign | (uint16_t(exp) << 10) | uint16_t((mant + 0x1000u) >> 13));
}

class GgufWriter {
 public:
  void add_str(const std::string& k, const std::string& v) {
    kv_.push_back({k, [v](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeStr);
                     put_str(b, v);
                   }});
  }
  void add_u32(const std::string& k, uint32_t v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeU32);
                     put_u32(b, v);
                   }});
  }
  void add_f32(const std::string& k, float v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeF32);
                     put_pod(b, v);
                   }});
  }
  void add_bool(const std::string& k, bool v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeBool);
                     b.push_back(v ? 1 : 0);
                   }});
  }
  void add_bool_arr(const std::string& k, const std::vector<bool>& v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeArr);
                     put_u32(b, kGgufTypeBool);
                     put_u64(b, v.size());
                     for (bool x : v) b.push_back(x ? 1 : 0);
                   }});
  }
  void add_u32_arr(const std::string& k, const std::vector<uint32_t>& v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeArr);
                     put_u32(b, kGgufTypeU32);
                     put_u64(b, v.size());
                     for (uint32_t x : v) put_u32(b, x);
                   }});
  }
  void add_str_arr(const std::string& k, const std::vector<std::string>& v) {
    kv_.push_back({k, [=](std::vector<uint8_t>& b) {
                     put_u32(b, kGgufTypeArr);
                     put_u32(b, kGgufTypeStr);
                     put_u64(b, v.size());
                     for (const auto& s : v) put_str(b, s);
                   }});
  }

  // Row-major shape (e.g. {N, K} or {E, N, K}); writer reverses dims to GGUF fastest-first order.
  void add_tensor(const std::string& name, const std::vector<uint64_t>& shape_rm, uint32_t ggml_type,
                  std::vector<uint8_t> bytes) {
    tensors_.push_back({name, shape_rm, ggml_type, std::move(bytes)});
  }

  void write(const std::filesystem::path& path) const {
    std::vector<uint8_t> buf;
    buf.reserve(1 << 20);
    buf.insert(buf.end(), {'G', 'G', 'U', 'F'});
    put_u32(buf, 3);
    put_u64(buf, tensors_.size());
    put_u64(buf, kv_.size());
    for (const auto& kv : kv_) {
      put_str(buf, kv.key);
      kv.emit(buf);
    }
    constexpr size_t kAlign = 32;
    size_t off = 0;
    for (const auto& t : tensors_) {
      put_str(buf, t.name);
      put_u32(buf, uint32_t(t.shape_rm.size()));
      for (size_t d = 0; d < t.shape_rm.size(); ++d) put_u64(buf, t.shape_rm[t.shape_rm.size() - 1 - d]);
      put_u32(buf, t.ggml_type);
      put_u64(buf, off);
      off += (t.bytes.size() + kAlign - 1) / kAlign * kAlign;
    }
    while (buf.size() % kAlign) buf.push_back(0);
    for (const auto& t : tensors_) {
      buf.insert(buf.end(), t.bytes.begin(), t.bytes.end());
      while (buf.size() % kAlign) buf.push_back(0);
    }
    std::filesystem::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + path.string());
    f.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size()));
  }

 private:
  struct Kv {
    std::string key;
    std::function<void(std::vector<uint8_t>&)> emit;
  };
  struct Tensor {
    std::string name;
    std::vector<uint64_t> shape_rm;
    uint32_t ggml_type;
    std::vector<uint8_t> bytes;
  };
  template <class T>
  static void put_pod(std::vector<uint8_t>& b, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    b.insert(b.end(), p, p + sizeof(T));
  }
  static void put_u32(std::vector<uint8_t>& b, uint32_t v) { put_pod(b, v); }
  static void put_u64(std::vector<uint8_t>& b, uint64_t v) { put_pod(b, v); }
  static void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u64(b, s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
  std::vector<Kv> kv_;
  std::vector<Tensor> tensors_;
};

// GPT-2 printable alphabet encoder for byte 0..255 (exact inverse of gguf_format.cpp's kGpt2ByteTable).
std::string gpt2_byte_token(int b) {
  int extra = 256;
  int cp = -1;
  for (int i = 0; i < 256; ++i) {
    const bool printable = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174;
    const int c = printable ? i : extra++;
    if (i == b) {
      cp = c;
      break;
    }
  }
  std::string s;
  if (cp < 0x80) {
    s.push_back(char(cp));
  } else {
    s.push_back(char(0xC0 | (cp >> 6)));
    s.push_back(char(0x80 | (cp & 0x3F)));
  }
  return s;
}

// The tiny models' draws, the same on every platform.  <random>'s engines are specified and its distributions are not
// (libstdc++ and MSVC make other values of one engine's bits), so Linux and Windows wrote two different models from
// the same seeds, and what a check counts against cpu_ref was another count on each (the mixed-quant model, teacher
// forced: 12 or 13 of 16 on Linux under a check that wants 15).  Integers from the engine, and exact arithmetic on
// them up to one rounding.
int draw_int(std::mt19937& rng, int lo, int hi) { return lo + int(rng() % uint32_t(hi - lo + 1)); }  // lo .. hi

// lo + (hi - lo) * u, u in [0, 1) of 24 bits: the product of two floats is exact in a double, so the sum rounds once
// whether or not a compiler fuses it.
float draw_uniform(std::mt19937& rng, float lo, float hi) {
  const double u = double(rng() >> 8) * 0x1p-24;
  return float(double(lo) + double(hi - lo) * u);
}

// A normal's mean and deviation and nearly its shape (Irwin-Hall: twelve uniforms of 16 bits, variance 1, within 6).
// The sum is exact in an integer and its product with sd exact in a double (20 and 24 bits).
float draw_normal(std::mt19937& rng, float mean, float sd) {
  uint32_t sum = 0;
  for (int i = 0; i < 12; ++i) sum += rng() >> 16;
  const double z = (double(sum) - 6.0 * 65536.0) * 0x1p-16;
  return float(double(mean) + double(sd) * z);
}

std::vector<uint8_t> gen_f32(std::mt19937& rng, size_t n, float mean, float sd) {
  std::vector<uint8_t> out(n * 4);
  for (size_t i = 0; i < n; ++i) {
    const float v = draw_normal(rng, mean, sd);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

std::vector<uint8_t> gen_bf16(std::mt19937& rng, size_t n, float sd) {
  std::vector<uint8_t> out(n * 2);
  for (size_t i = 0; i < n; ++i) {
    const uint16_t h = f32_to_bf16(draw_normal(rng, 0.f, sd));
    std::memcpy(out.data() + i * 2, &h, 2);
  }
  return out;
}

std::vector<uint8_t> gen_q8_0(std::mt19937& rng, size_t n, float sd) {
  const size_t nb = n / 32;
  std::vector<uint8_t> out(nb * 34);
  for (size_t b = 0; b < nb; ++b) {
    float vals[32], amax = 0.f;
    for (int i = 0; i < 32; ++i) {
      vals[i] = draw_normal(rng, 0.f, sd);
      amax = std::max(amax, std::fabs(vals[i]));
    }
    const float scale = amax > 0.f ? amax / 127.f : 1e-4f;
    const uint16_t dh = f32_to_f16(scale);
    uint8_t* dst = out.data() + b * 34;
    std::memcpy(dst, &dh, 2);
    for (int i = 0; i < 32; ++i) {
      const int q = std::clamp(int(std::lround(vals[i] / scale)), -127, 127);
      dst[2 + i] = uint8_t(int8_t(q));
    }
  }
  return out;
}

std::vector<uint8_t> gen_mxfp4(std::mt19937& rng, size_t n) {
  const size_t nb = n / 32;
  std::vector<uint8_t> out(nb * 17);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 17;
    dst[0] = uint8_t(draw_int(rng, 120, 124));
    for (int i = 0; i < 16; ++i) dst[1 + i] = uint8_t(draw_int(rng, 0, 255));
  }
  return out;
}

std::vector<uint8_t> gen_q2_0(std::mt19937& rng, size_t n) {
  const size_t nb = n / 64;
  std::vector<uint8_t> out(nb * 18);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 18;
    const uint16_t dh = f32_to_f16(draw_uniform(rng, 0.01f, 0.08f));
    std::memcpy(dst, &dh, 2);
    for (int i = 0; i < 16; ++i) dst[2 + i] = uint8_t(draw_int(rng, 0, 255));
  }
  return out;
}

std::vector<uint8_t> gen_nvfp4(std::mt19937& rng, size_t n) {
  const size_t nb = n / 64;
  std::vector<uint8_t> out(nb * 36);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 36;
    for (int g = 0; g < 4; ++g) dst[g] = uint8_t(draw_int(rng, 0x18, 0x38));  // positive E4M3 scales, none 0x7F/0xFF
    for (int i = 0; i < 32; ++i) dst[4 + i] = uint8_t(draw_int(rng, 0, 255));
  }
  return out;
}

struct TinySpec {
  int H = 64, nh = 4, nkv = 2, hd = 32, d_rel = 8, rel_extent = 16, window = 8, conv = 4;
  int log_floor = 8, attn_res_block = 2, n_phys = 6, core_start = 2, core_end = 4, n_passes = 3, gate_rank = 16;
  int n_experts = 4, top_k = 2, inter = 64, vocab_padded = 320, vocab_unpadded = 272;
  float eps = 1e-6f, qk_clip = 6.f, rel_clip = 4.f, log_alpha = 0.2f;
  float route_scale = 1.5f, situ_beta = 6.f, situ_linear_beta = 8.f, softcap = 30.f;
  bool packed_gate_nvfp4_down = false;
  std::vector<bool> is_swa = {true, false, true, false, true, false};
};

// Minimal Inkling-2 chat template exercising tojson(sort_keys=true), thinking, text, and tool invocation tags, and
// the thinking effort the real template writes (reasoning_effort, 0.9 unless given, "none" for no thinking).
const char* kInkling2Template = R"jinja(
{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}
<|message_system|><|content_text|>Thinking effort level: {{ 0 if eff == 'none' else eff }}<|end_message|>
{%- for m in messages -%}
{%- if m.role == 'user' -%}
<|message_user|><|content_text|>{{ m.content or '' }}<|end_message|>
{%- elif m.role == 'assistant' -%}
{%- if m.reasoning_content -%}
<|message_model|><|content_thinking|>{{ m.reasoning_content }}<|end_message|>
{%- endif -%}
{%- if m.content -%}
<|message_model|><|content_text|>{{ m.content }}<|end_message|>
{%- endif -%}
{%- if m.tool_calls -%}
{%- for tc in m.tool_calls -%}
<|message_model|>{{ tc.function.name }}<|content_invoke_tool_json|>{"name":"{{ tc.function.name }}","args":{{ tc.function.arguments | tojson(sort_keys=true) }}}<|end_message|>
{%- endfor -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|message_model|>
{%- endif -%}
)jinja";

enum class RoundMatrixCase { Generated, ExactBf16, OneLowBit };

void write_tiny_inkling2(const std::filesystem::path& path, const TinySpec& c, bool mixed_quant, uint32_t seed_val,
                         RoundMatrixCase round_case = RoundMatrixCase::Generated, uint32_t header_heads = 0) {
  std::mt19937 rng(seed_val);
  GgufWriter w;
  const std::string arch = "inkling2_loop";
  w.add_str("general.architecture", arch);
  w.add_u32("general.quantization_version", 2);
  w.add_u32(arch + ".context_length", 1024);
  w.add_u32(arch + ".block_count", uint32_t(c.n_phys));
  w.add_u32(arch + ".embedding_length", uint32_t(c.H));
  w.add_u32(arch + ".vocab_size", uint32_t(c.vocab_padded));
  w.add_u32(arch + ".vocab_size_unpadded", uint32_t(c.vocab_unpadded));
  w.add_u32(arch + ".attention.head_count", header_heads ? header_heads : uint32_t(c.nh));
  w.add_u32(arch + ".attention.head_count_kv", uint32_t(c.nkv));
  w.add_u32(arch + ".attention.key_length", uint32_t(c.hd));
  w.add_u32(arch + ".attention.value_length", uint32_t(c.hd));
  w.add_f32(arch + ".attention.layer_norm_rms_epsilon", c.eps);
  w.add_f32(arch + ".attention.qk_clip", c.qk_clip);
  w.add_u32(arch + ".attention.relative_length", uint32_t(c.d_rel));
  w.add_u32(arch + ".attention.relative_extent", uint32_t(c.rel_extent));
  w.add_f32(arch + ".attention.relative_clip", c.rel_clip);
  w.add_u32(arch + ".attention.log_scale_floor", uint32_t(c.log_floor));
  w.add_f32(arch + ".attention.log_scale_alpha", c.log_alpha);
  w.add_u32(arch + ".attention.sliding_window", uint32_t(c.window));
  w.add_bool_arr(arch + ".attention.sliding_window_pattern", c.is_swa);
  w.add_u32(arch + ".shortconv.l_cache", uint32_t(c.conv));
  w.add_u32(arch + ".attn_res.block_size", uint32_t(c.attn_res_block));
  w.add_u32(arch + ".expert_count", uint32_t(c.n_experts));
  w.add_u32(arch + ".expert_used_count", uint32_t(c.top_k));
  w.add_u32(arch + ".expert_feed_forward_length", uint32_t(c.inter));
  w.add_f32(arch + ".expert_weights_scale", c.route_scale);
  w.add_f32(arch + ".activation.situ_beta", c.situ_beta);
  w.add_f32(arch + ".activation.situ_linear_beta", c.situ_linear_beta);
  w.add_f32(arch + ".final_logit_softcapping", c.softcap);
  w.add_u32(arch + ".loop.start", uint32_t(c.core_start));
  w.add_u32(arch + ".loop.end", uint32_t(c.core_end));
  w.add_u32(arch + ".loop.count", uint32_t(c.n_passes));
  w.add_u32(arch + ".loop.gate_rank", uint32_t(c.gate_rank));
  w.add_bool(arch + ".loop.varying", false);

  // Tokenizer metadata (o200k / gpt-4o style byte-level BPE with 272 unpadded tokens).
  std::vector<std::string> toks;
  std::vector<uint32_t> ttypes;
  for (int b = 0; b < 256; ++b) {
    toks.push_back(gpt2_byte_token(b));
    ttypes.push_back(1);
  }
  const std::vector<std::string> extra_toks = {"he", "ll", "lo", "hel", "llo", "hello", "in", "kl"};
  const std::vector<std::string> merges = {"h e", "l l", "l o", "he l", "l lo", "hel lo", "i n", "k l"};
  for (const auto& s : extra_toks) {
    toks.push_back(s);
    ttypes.push_back(1);
  }
  const std::vector<std::string> ctrl = {
      "<|message_user|>",          "<|message_model|>",
      "<|content_thinking|>",      "<|content_text|>",
      "<|content_invoke_tool_json|>", "<|end_message|>",
      "<|content_model_end_sampling|>", "<|message_system|>"};
  for (const auto& s : ctrl) {
    toks.push_back(s);
    ttypes.push_back(3);
  }
  for (int t = int(toks.size()); t < c.vocab_unpadded; ++t) {  // a larger vocabulary's other tokens: never text
    toks.push_back("<|unused_" + std::to_string(t) + "|>");
    ttypes.push_back(3);
  }
  w.add_str("tokenizer.ggml.model", "gpt2");
  // Exercise both "inkling2" (unsloth GGUFs) and "gpt-4o" (o200k alias) pre-tokenizers.
  w.add_str("tokenizer.ggml.pre", mixed_quant ? "gpt-4o" : "inkling2");
  w.add_str_arr("tokenizer.ggml.tokens", toks);
  w.add_u32_arr("tokenizer.ggml.token_type", ttypes);
  w.add_str_arr("tokenizer.ggml.merges", merges);
  w.add_u32("tokenizer.ggml.eos_token_id", 269);  // <|end_message|>
  w.add_str("tokenizer.chat_template", kInkling2Template);

  const std::string core = "blk." + std::to_string(c.core_start) + ".";
  auto add_mat = [&](const std::string& name, int N, int K, uint32_t dt, float sd = 0.08f) {
    const size_t n = size_t(N) * K;
    if (round_case != RoundMatrixCase::Generated &&
        (name == core + "ffn_gate_inp.weight" || name == core + "loop_q_ctrl_forget.weight")) {
      auto data = gen_f32(rng, n, 0.f, sd);
      for (size_t i = 0; i < data.size(); i += 4) data[i] = data[i + 1] = 0;  // little-endian GGUF
      if (round_case == RoundMatrixCase::OneLowBit) data[data.size() - 4] = 1;
      w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlF32, std::move(data));
      return;
    }
    if (dt == kGgmlF32) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlF32, gen_f32(rng, n, 0.f, sd));
    else if (dt == kGgmlBF16) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlBF16, gen_bf16(rng, n, sd));
    else if (dt == kGgmlQ8_0) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlQ8_0, gen_q8_0(rng, n, sd));
    else if (dt == kGgmlMXFP4) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlMXFP4, gen_mxfp4(rng, n));
    else if (dt == kGgmlQ2_0) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlQ2_0, gen_q2_0(rng, n));
    else if (dt == kGgmlNVFP4) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlNVFP4, gen_nvfp4(rng, n));
  };
  auto add_vec = [&](const std::string& name, const std::vector<uint64_t>& shape, float mean, float sd) {
    size_t n = 1;
    for (uint64_t d : shape) n *= size_t(d);
    w.add_tensor(name, shape, kGgmlF32, gen_f32(rng, n, mean, sd));
  };

  add_mat("token_embd.weight", c.vocab_padded, c.H, mixed_quant ? kGgmlQ8_0 : kGgmlBF16, 0.4f);
  add_vec("token_embd_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_mat("output.weight", c.vocab_padded, c.H, mixed_quant ? kGgmlQ8_0 : kGgmlBF16, 0.15f);
  add_vec("output_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_vec("output_res_query.weight", {uint64_t(c.H)}, 0.f, 0.2f);

  const int q_dim = c.nh * c.hd, kv_dim = c.nkv * c.hd;
  for (int i = 0; i < c.n_phys; ++i) {
    const std::string b = "blk." + std::to_string(i) + ".";
    const bool swa = c.is_swa[size_t(i)];
    const int extent = swa ? c.window : c.rel_extent;
    // Exercise both the split BF16 companion GEMM and the exact float32 control projection beside Q8 inputs.
    const uint32_t companion_dt = mixed_quant && i % 3 != 1 ? kGgmlBF16 : kGgmlF32;
    if (swa) {
      add_vec(b + "attn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    } else {
      add_vec(b + "attn_q_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_vec(b + "attn_k_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_vec(b + "attn_v_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_mat(b + "attn_k_conv_proj.weight", c.conv * c.nkv, c.H, companion_dt, 0.05f);
      add_mat(b + "attn_v_conv_proj.weight", c.conv * c.nkv, c.H, companion_dt, 0.05f);
    }
    const uint32_t attn_dt = mixed_quant ? (i % 3 == 0 ? kGgmlQ8_0 : i % 3 == 1 ? kGgmlMXFP4 : kGgmlQ2_0) : kGgmlBF16;
    add_mat(b + "attn_q.weight", q_dim, c.H, attn_dt);
    add_mat(b + "attn_k.weight", kv_dim, c.H, attn_dt);
    add_mat(b + "attn_v.weight", kv_dim, c.H, attn_dt);
    add_mat(b + "attn_output.weight", c.H, q_dim, attn_dt);
    add_mat(b + "attn_gate.weight", q_dim, c.H, attn_dt);
    add_mat(b + "attn_rel.weight", c.d_rel * c.nh, c.H, attn_dt);
    add_vec(b + "attn_rel_bias.weight", {uint64_t(extent), uint64_t(c.d_rel)}, 0.f, 0.2f);
    add_vec(b + "attn_q_clip.weight", {uint64_t(c.nh), uint64_t(c.hd)}, 1.f, 0.1f);
    add_vec(b + "attn_rel_clip.weight", {uint64_t(c.nh), uint64_t(c.d_rel)}, 1.f, 0.1f);
    add_mat(b + "attn_sink_proj.weight", c.nh, c.H, companion_dt, 0.1f);
    add_vec(b + "attn_k_conv.weight", {uint64_t(kv_dim), uint64_t(c.conv)}, 0.f, 0.15f);
    add_vec(b + "attn_v_conv.weight", {uint64_t(kv_dim), uint64_t(c.conv)}, 0.f, 0.15f);
    add_vec(b + "attn_post_conv.weight", {uint64_t(c.H), uint64_t(c.conv)}, 0.f, 0.15f);
    if (i > 0) {
      if (swa) add_vec(b + "attn_res_query.weight", {uint64_t(c.H)}, 0.f, 0.2f);
      else add_vec(b + "attn_res_query.weight", {3, uint64_t(c.H)}, 0.f, 0.2f);
    }

    // MoE weights: in non-mixed model, test BF16 + 3D NVFP4 with per-expert .scale; in mixed model, test ALL 4!
    add_vec(b + "ffn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_mat(b + "ffn_gate_inp.weight", c.n_experts, c.H, kGgmlF32, 0.2f);
    add_vec(b + "exp_probs_b.bias", {uint64_t(c.n_experts)}, 0.f, 0.1f);
    uint32_t exp_dt = kGgmlBF16;
    if (!mixed_quant) {
      exp_dt = (i == 2 || i == 3) ? kGgmlNVFP4 : kGgmlBF16;
    } else {
      const uint32_t cycle[4] = {kGgmlNVFP4, kGgmlMXFP4, kGgmlQ2_0, kGgmlQ8_0};
      exp_dt = cycle[i % 4];
    }
    auto add_3d = [&](const std::string& name, int E, int N, int K, uint32_t dt) {
      const size_t n = size_t(E) * N * K;
      std::vector<uint8_t> data;
      if (dt == kGgmlBF16) data = gen_bf16(rng, n, 0.08f);
      else if (dt == kGgmlNVFP4) data = gen_nvfp4(rng, n);
      else if (dt == kGgmlMXFP4) data = gen_mxfp4(rng, n);
      else if (dt == kGgmlQ2_0) data = gen_q2_0(rng, n);
      else if (dt == kGgmlQ8_0) data = gen_q8_0(rng, n, 0.08f);
      w.add_tensor(name, {uint64_t(E), uint64_t(N), uint64_t(K)}, dt, std::move(data));
    };
    const uint32_t gate_type = c.packed_gate_nvfp4_down && exp_dt == kGgmlNVFP4 ? kGgmlQ8_0 : exp_dt;
    add_3d(b + "ffn_gate_up_exps.weight", c.n_experts, 2 * c.inter, c.H, gate_type);
    add_3d(b + "ffn_down_exps.weight", c.n_experts, c.H, c.inter, exp_dt);
    if (exp_dt == kGgmlNVFP4) {
      // Distinct positive per-expert F32 tensor scales!
      std::vector<uint8_t> sc_gu(size_t(c.n_experts) * 4), sc_dn(size_t(c.n_experts) * 4);
      for (int e = 0; e < c.n_experts; ++e) {
        // In double, where the products are exact: in float a compiler that fuses them rounds another bit.
        const float s1 = float(double(0.04f) + double(0.015f) * e);
        const float s2 = float(double(0.03f) + double(0.012f) * e);
        std::memcpy(sc_gu.data() + e * 4, &s1, 4);
        std::memcpy(sc_dn.data() + e * 4, &s2, 4);
      }
      w.add_tensor(b + "ffn_gate_up_exps.scale", {uint64_t(c.n_experts)}, kGgmlF32, std::move(sc_gu));
      w.add_tensor(b + "ffn_down_exps.scale", {uint64_t(c.n_experts)}, kGgmlF32, std::move(sc_dn));
    }
    add_vec(b + "ffn_post_conv.weight", {uint64_t(c.H), uint64_t(c.conv)}, 0.f, 0.15f);
    add_vec(b + "ffn_res_query.weight", {uint64_t(c.H)}, 0.f, 0.2f);

    // Core layers own cross-pass loop gate tensors.
    if (i >= c.core_start && i < c.core_end) {
      add_vec(b + "loop_msg_query.weight", {4, uint64_t(c.H)}, 0.f, 0.2f);
      for (const char* cn : {"loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl", "loop_ffn_ctrl"}) {
        add_mat(b + cn + "_down.weight", c.gate_rank, 2 * c.H, kGgmlBF16, 0.05f);
        add_mat(b + cn + "_gain.weight", c.H, c.gate_rank, kGgmlBF16, 0.05f);
        add_mat(b + cn + "_forget.weight", c.H, c.gate_rank, kGgmlBF16, 0.05f);
      }
    }
  }
  w.write(path);
}

void write_tiny_dspark(const std::filesystem::path& path, const TinySpec& c, bool packed_w1, uint32_t seed_val,
                       int rank = 32, uint32_t header_heads = 0) {
  std::mt19937 rng(seed_val);
  GgufWriter w;
  const std::string arch = "dflash";
  const int layers = 3, block = 8;
  const int n_visits = c.core_start + c.n_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);  // 10
  const std::vector<uint32_t> taps = {0, 3, 7, uint32_t(n_visits)};  // includes post output_norm tap!
  const std::vector<bool> swa_pat = {true, true, false};

  w.add_str("general.architecture", arch);
  w.add_u32("general.quantization_version", 2);
  w.add_bool(arch + ".has_markov_context", true);
  w.add_u32(arch + ".block_count", uint32_t(layers));
  w.add_u32(arch + ".embedding_length", uint32_t(c.H));
  w.add_u32(arch + ".feed_forward_length", uint32_t(c.inter));
  w.add_u32(arch + ".attention.head_count", header_heads ? header_heads : uint32_t(c.nh));
  w.add_u32(arch + ".attention.head_count_kv", uint32_t(c.nkv));
  w.add_u32(arch + ".attention.key_length", uint32_t(c.hd));
  w.add_u32(arch + ".block_size", uint32_t(block));
  w.add_f32(arch + ".rope.freq_base", 1000000.f);
  w.add_f32(arch + ".attention.layer_norm_rms_epsilon", c.eps);
  w.add_f32(arch + ".final_logit_softcapping", c.softcap);
  w.add_u32(arch + ".attention.sliding_window", uint32_t(c.window));
  w.add_bool_arr(arch + ".attention.sliding_window_pattern", swa_pat);
  w.add_u32_arr(arch + ".target_layers", taps);

  auto add_mat = [&](const std::string& name, int N, int K, uint32_t dt, float sd = 0.08f) {
    const size_t n = size_t(N) * K;
    if (dt == kGgmlBF16) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlBF16, gen_bf16(rng, n, sd));
    else if (dt == kGgmlQ8_0) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlQ8_0, gen_q8_0(rng, n, sd));
    else w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlF32, gen_f32(rng, n, 0.f, sd));
  };
  auto add_vec = [&](const std::string& name, const std::vector<uint64_t>& shape, float mean, float sd) {
    size_t n = 1;
    for (uint64_t d : shape) n *= size_t(d);
    w.add_tensor(name, shape, kGgmlF32, gen_f32(rng, n, mean, sd));
  };

  const int q_dim = c.nh * c.hd, kv_dim = c.nkv * c.hd;
  for (int l = 0; l < layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    add_vec(p + "attn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_vec(p + "ffn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_vec(p + "attn_q_norm.weight", {uint64_t(c.hd)}, 1.f, 0.1f);
    add_vec(p + "attn_k_norm.weight", {uint64_t(c.hd)}, 1.f, 0.1f);
    add_vec(p + "attn_k_ctx_norm.weight", {uint64_t(c.hd)}, 1.f, 0.1f);
    add_vec(p + "ctx_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_mat(p + "attn_q.weight", q_dim, c.H, kGgmlBF16);
    add_mat(p + "attn_k.weight", kv_dim, c.H, kGgmlBF16);
    add_mat(p + "attn_v.weight", kv_dim, c.H, kGgmlBF16);
    add_mat(p + "attn_output.weight", c.H, q_dim, kGgmlBF16);
    add_mat(p + "attn_k_ctx.weight", kv_dim, c.H, kGgmlBF16);
    add_mat(p + "attn_v_ctx.weight", kv_dim, c.H, kGgmlBF16);
    add_vec(p + "ctx_mix.weight", {uint64_t(taps.size())}, 0.25f, 0.1f);
    add_mat(p + "ffn_gate.weight", c.inter, c.H, kGgmlBF16);
    add_mat(p + "ffn_up.weight", c.inter, c.H, kGgmlBF16);
    add_mat(p + "ffn_down.weight", c.H, c.inter, kGgmlBF16);
  }
  add_vec("token_embd_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_vec("output_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_vec("markov_ctx_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_vec("mask_embd.weight", {uint64_t(block - 1), uint64_t(c.H)}, 0.f, 0.2f);
  add_vec("markov_ctx_mix.weight", {uint64_t(taps.size())}, 0.25f, 0.1f);
  add_mat("markov_w1.weight", c.vocab_unpadded, rank, packed_w1 ? kGgmlQ8_0 : kGgmlBF16, 0.1f);
  add_mat("markov_w2.weight", c.vocab_unpadded, rank, packed_w1 ? kGgmlQ8_0 : kGgmlBF16, 0.1f);
  add_vec("markov_w2.bias", {uint64_t(c.vocab_unpadded)}, 0.f, 0.1f);
  add_mat("joint_proj.weight", 3 * rank, 2 * c.H + 2 * rank, kGgmlBF16, 0.05f);
  add_vec("joint_proj.bias", {uint64_t(3 * rank)}, 0.f, 0.1f);
  w.write(path);
}

void write_tiny_dflash2_gguf(const std::filesystem::path& path, const TinySpec& c, bool q8_weights,
                             bool padded_codebook, uint32_t seed_val, int rank = 32, int d_nh = 0) {
  std::mt19937 rng(seed_val);
  GgufWriter w;
  const std::string arch = "dflash";
  const int nh = d_nh > 0 ? d_nh : c.nh;
  const int layers = 2, block = 8, conv_taps = 2, conv_group = 16, topk = 16;
  const int n_visits = c.core_start + c.n_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);  // 10
  const std::vector<uint32_t> taps = {2, 6, uint32_t(n_visits)};  // 1-based, ending at post-output_norm tap!
  const std::vector<bool> swa_pat(size_t(layers), true);
  const int sel_vocab = padded_codebook ? c.vocab_padded : c.vocab_unpadded;

  w.add_str("general.architecture", arch);
  w.add_u32("general.quantization_version", 2);
  w.add_u32(arch + ".block_count", uint32_t(layers));
  w.add_u32(arch + ".embedding_length", uint32_t(c.H));
  w.add_u32(arch + ".feed_forward_length", uint32_t(c.inter));
  w.add_u32(arch + ".attention.head_count", uint32_t(nh));
  w.add_u32(arch + ".attention.head_count_kv", uint32_t(c.nkv));
  w.add_u32(arch + ".attention.key_length", uint32_t(c.hd));
  w.add_u32(arch + ".attention.value_length", uint32_t(c.hd));
  w.add_bool(arch + ".attention.causal", false);
  w.add_f32(arch + ".rope.freq_base", 10000.f);
  w.add_f32(arch + ".attention.layer_norm_rms_epsilon", c.eps);
  w.add_u32(arch + ".block_size", uint32_t(block));
  w.add_u32(arch + ".conv_kernel_size", uint32_t(conv_taps));
  w.add_u32(arch + ".conv_group_size", uint32_t(conv_group));
  w.add_u32(arch + ".selector_rank", uint32_t(rank));
  w.add_u32(arch + ".selector_top_k", uint32_t(topk));
  w.add_f32(arch + ".final_logit_softcapping", c.softcap);
  w.add_u32(arch + ".attention.sliding_window", uint32_t(c.window));
  w.add_bool_arr(arch + ".attention.sliding_window_pattern", swa_pat);
  w.add_u32_arr(arch + ".target_layers", taps);
  w.add_u32("tokenizer.ggml.mask_token_id", 266);

  const uint32_t lin_dt = q8_weights ? kGgmlQ8_0 : kGgmlBF16;
  auto add_mat = [&](const std::string& name, int N, int K, uint32_t dt, float sd = 0.08f) {
    const size_t n = size_t(N) * K;
    if (dt == kGgmlBF16) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlBF16, gen_bf16(rng, n, sd));
    else if (dt == kGgmlQ8_0) w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlQ8_0, gen_q8_0(rng, n, sd));
    else w.add_tensor(name, {uint64_t(N), uint64_t(K)}, kGgmlF32, gen_f32(rng, n, 0.f, sd));
  };
  auto add_vec = [&](const std::string& name, const std::vector<uint64_t>& shape, float mean, float sd) {
    size_t n = 1;
    for (uint64_t d : shape) n *= size_t(d);
    w.add_tensor(name, shape, kGgmlF32, gen_f32(rng, n, mean, sd));
  };

  const int q_dim = nh * c.hd, kv_dim = c.nkv * c.hd;
  const int tg2 = 2 * conv_taps * (c.H / conv_group);
  for (int l = 0; l < layers; ++l) {
    const std::string p = "blk." + std::to_string(l) + ".";
    add_vec(p + "attn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_vec(p + "ffn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    add_vec(p + "attn_q_norm.weight", {uint64_t(c.hd)}, 1.f, 0.1f);
    add_vec(p + "attn_k_norm.weight", {uint64_t(c.hd)}, 1.f, 0.1f);
    add_mat(p + "attn_q.weight", q_dim, c.H, lin_dt);
    add_mat(p + "attn_k.weight", kv_dim, c.H, lin_dt);
    add_mat(p + "attn_v.weight", kv_dim, c.H, lin_dt);
    add_mat(p + "attn_output.weight", c.H, q_dim, lin_dt);
    add_mat(p + "ffn_gate.weight", c.inter, c.H, lin_dt);
    add_mat(p + "ffn_up.weight", c.inter, c.H, lin_dt);
    add_mat(p + "ffn_down.weight", c.H, c.inter, lin_dt);
    add_mat(p + "attn_conv_proj.weight", tg2, c.H, lin_dt, 0.05f);
    add_vec(p + "attn_conv_base", {2, uint64_t(conv_taps), uint64_t(c.H)}, 0.5f, 0.1f);
    add_mat(p + "ffn_conv_proj.weight", tg2, c.H, lin_dt, 0.05f);
    add_vec(p + "ffn_conv_base", {2, uint64_t(conv_taps), uint64_t(c.H)}, 0.5f, 0.1f);
  }
  add_mat("fc.weight", c.H, int(taps.size()) * c.H, lin_dt);
  add_vec("enc.output_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_vec("output_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
  add_mat("selector_predecessor.weight", sel_vocab, rank, kGgmlBF16, 0.1f);
  add_mat("selector_successor.weight", sel_vocab, rank, kGgmlBF16, 0.1f);
  add_mat("selector_hidden.weight", rank, c.H, lin_dt, 0.1f);
  w.write(path);
}

void write_tiny_dflash2_dir(const std::filesystem::path& dir, const TinySpec& c, uint32_t seed_val, int rank = 32) {
  std::filesystem::create_directories(dir);
  std::mt19937 rng(seed_val);
  const int layers = 2, block = 8, conv_taps = 2, conv_group = 16, topk = 16;
  const int n_visits = c.core_start + c.n_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);  // 10
  const json cfg = {
      {"architectures", {"DFlash2DraftModel"}},
      {"hidden_size", c.H},
      {"intermediate_size", c.inter},
      {"num_hidden_layers", layers},
      {"num_attention_heads", c.nh},
      {"num_key_value_heads", c.nkv},
      {"head_dim", c.hd},
      {"rms_norm_eps", c.eps},
      {"rope_parameters", {{"rope_theta", 10000.0}}},
      {"is_causal", false},
      {"layer_types", std::vector<std::string>(size_t(layers), "sliding_attention")},
      {"sliding_window", c.window},
      {"num_target_layers", n_visits},
      {"vocab_size", c.vocab_padded},
      {"dflash_config",
       {{"block_size", block},
        {"conv_group_size", conv_group},
        {"conv_kernel_size", conv_taps},
        {"final_logit_softcapping", c.softcap},
        {"mask_token_id", 266},
        {"selector_rank", rank},
        {"selector_softcap_mode", "after_correction"},
        {"selector_top_k", topk},
        {"target_layer_ids", {1, 5, n_visits}}}},
  };
  {
    std::ofstream out(dir / "config.json", std::ios::binary | std::ios::trunc);
    out << cfg.dump(2);
  }
  json hdr = json::object();
  std::vector<uint8_t> blob;
  auto add_t = [&](const std::string& name, const std::vector<int64_t>& shape, const std::string& dt,
                   const std::vector<uint8_t>& bytes) {
    const uint64_t start = blob.size();
    blob.insert(blob.end(), bytes.begin(), bytes.end());
    hdr[name] = {{"dtype", dt}, {"shape", shape}, {"data_offsets", {start, uint64_t(blob.size())}}};
  };
  auto add_bf16 = [&](const std::string& name, const std::vector<int64_t>& shape, float sd = 0.08f) {
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    add_t(name, shape, "BF16", gen_bf16(rng, n, sd));
  };
  auto add_f32 = [&](const std::string& name, const std::vector<int64_t>& shape, float mean, float sd) {
    size_t n = 1;
    for (int64_t d : shape) n *= size_t(d);
    add_t(name, shape, "F32", gen_f32(rng, n, mean, sd));
  };

  const int q_dim = c.nh * c.hd, kv_dim = c.nkv * c.hd;
  const int tg2 = 2 * conv_taps * (c.H / conv_group);
  for (int l = 0; l < layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    add_f32(p + "input_layernorm.weight", {c.H}, 1.f, 0.1f);
    add_f32(p + "post_attention_layernorm.weight", {c.H}, 1.f, 0.1f);
    add_f32(p + "self_attn.q_norm.weight", {c.hd}, 1.f, 0.1f);
    add_f32(p + "self_attn.k_norm.weight", {c.hd}, 1.f, 0.1f);
    add_bf16(p + "self_attn.q_proj.weight", {q_dim, c.H});
    add_bf16(p + "self_attn.k_proj.weight", {kv_dim, c.H});
    add_bf16(p + "self_attn.v_proj.weight", {kv_dim, c.H});
    add_bf16(p + "self_attn.o_proj.weight", {c.H, q_dim});
    add_bf16(p + "mlp.gate_proj.weight", {c.inter, c.H});
    add_bf16(p + "mlp.up_proj.weight", {c.inter, c.H});
    add_bf16(p + "mlp.down_proj.weight", {c.H, c.inter});
    add_bf16(p + "attention_conv.kernel_projection.weight", {tg2, c.H}, 0.05f);
    add_f32(p + "attention_conv.base_kernel", {2, conv_taps, c.H}, 0.5f, 0.1f);
    add_bf16(p + "mlp_conv.kernel_projection.weight", {tg2, c.H}, 0.05f);
    add_f32(p + "mlp_conv.base_kernel", {2, conv_taps, c.H}, 0.5f, 0.1f);
  }
  add_bf16("model.fc.weight", {c.H, 3 * c.H});
  add_f32("model.hidden_norm.weight", {c.H}, 1.f, 0.1f);
  add_f32("model.norm.weight", {c.H}, 1.f, 0.1f);
  add_bf16("model.candidate_selector.predecessor_codebook", {c.vocab_padded, rank}, 0.1f);
  add_bf16("model.candidate_selector.successor_codebook", {c.vocab_padded, rank}, 0.1f);
  add_bf16("model.candidate_selector.hidden_projection.weight", {rank, c.H}, 0.1f);

  const std::string hdr_s = hdr.dump();
  const uint64_t hlen = hdr_s.size();
  std::ofstream out(dir / "model.safetensors", std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hdr_s.data(), std::streamsize(hdr_s.size()));
  out.write(reinterpret_cast<const char*>(blob.data()), std::streamsize(blob.size()));
}

// LoadOpts::width: the rounds chosen every round as in real use (EngineOptions::width -1), adaptive rounds included.
constexpr int kChosen = -2;

struct LoadOpts {
  bool speculative = false;
  bool lookup = false;
  int max_ctx = 512;
  int prefill_chunk = 32;
  int width = -1;  // -1: pinned 7 when speculative; 0: no drafter's rounds (plain decoding, lookup rounds); kChosen
  int kv = FLYX_KV_16;
  int kv_gpu = -1;
  int kv_copy_gpu = -1;
  int64_t kv_ram = 0;
  std::string kv_method = "exact";
  std::string drafter = "none";
  std::vector<std::string> drafter_files{};
  std::string backend_opts{};
  std::string activations{};
  bool draft_trees = false;  // EngineOptions::draft_trees: the drafters' trees beside their chains
};

std::unique_ptr<Engine> load_engine(const std::string& path, const std::string& backend, const LoadOpts& o = {}) {
  EngineOptions eo;
  eo.backend = backend;
  eo.backend_options = o.backend_opts;
  eo.max_ctx = o.max_ctx;
  eo.prefill_chunk = o.prefill_chunk;
  eo.lookup = o.lookup;
  eo.lookup_given = o.lookup;
  eo.speculative = o.speculative;
  eo.drafter = o.speculative ? o.drafter : "none";
  eo.width = !o.speculative ? 0 : o.width == kChosen ? -1 : o.width >= 0 ? o.width : 7;
  eo.drafter_files = o.drafter_files;
  eo.activations = o.activations;
  eo.kv = o.kv;
  eo.kv_gpu = o.kv_gpu;
  eo.kv_copy_gpu = o.kv_copy_gpu;
  eo.kv_ram = o.kv_ram;
  eo.kv_method = o.kv_method;
  if (o.draft_trees) eo.draft_trees = eo.draft_trees_given = true;
  auto e = std::make_unique<Engine>();
  std::string err;
  if (!e->load(path, eo, &err)) throw std::runtime_error("load(" + backend + "): " + err);
  return e;
}

std::vector<int32_t> run_gen(Session& s, const std::vector<int32_t>& prompt, int n, bool sampled = false,
                             const std::vector<int32_t>& ckps = {}, int32_t keep_ckpt = -1,
                             GenerateResult* result = nullptr) {
  GenerateRequest req;
  req.prompt = prompt;
  req.max_new_tokens = n;
  req.ignore_eos = true;
  req.checkpoints = ckps.empty() ? std::vector<int32_t>{int32_t(prompt.size())} : ckps;
  req.keep_checkpoint = keep_ckpt;
  if (sampled) req.sampling = {0.75f, 0.9f, 20, 42, 0.4f, 0.15f};
  GenerateResult res;
  std::string err;
  std::vector<int32_t> out;
  if (generate(s, req,
               [&](std::string_view, const int32_t* ids, int cnt) {
                 out.insert(out.end(), ids, ids + cnt);
                 return true;
               },
               &res, &err) != 0)
    throw std::runtime_error("generate: " + err);
  if (result) *result = res;
  return out;
}

std::vector<int32_t> fresh_gen(Engine& e, const std::vector<int32_t>& prompt, int n, bool sampled = false) {
  Session s(&e);
  std::string err;
  if (!s.init(&err)) throw std::runtime_error("session init: " + err);
  return run_gen(s, prompt, n, sampled);
}

// Gate 2 on every row of a round: a fused round's producers, each writing what
// the next GEMM reads, against the generic layers.  All of a four-row round's
// logits, with its last draft rejected, then the next round's, and the round
// after a rewind into the accepted ones.
void test_fused_rounds(const std::filesystem::path& path, const std::string& model, const std::vector<int32_t>& prompt,
                       double kl_limit) {
  std::vector<int32_t> want;
  std::vector<float> resident_logits, generic_logits;
  for (const std::string fusion :
       {"fusion=0", "fusion=1", "fusion=0,gpu_experts=2", "fusion=1,gpu_experts=2", "fusion=0,gpu_experts=0",
        "fusion=1,gpu_experts=0", "fusion=0,gpu_experts=2,expert_prefill_min=2",
        "fusion=1,gpu_experts=2,expert_prefill_min=2", "fusion=0,gpu_experts=0,expert_prefill_min=2",
        "fusion=1,gpu_experts=0,expert_prefill_min=2"}) {
    const std::string opts = "head_cert=0,blocks=packed," + fusion;
    auto e = load_engine(path.string(), "cuda", {.speculative = true, .lookup = true, .backend_opts = opts});
    if (e->lookup_widths().empty() || e->lookup_widths().front() != 3 || e->widths().front() != 0)
      throw std::runtime_error("missing plain or three-draft test plan");
    const int vocab = e->model().desc().vocab;
    if (want.empty()) want = fresh_gen(*e, prompt, 5);
    if (want.size() != 5) throw std::runtime_error("missing reference continuation");

    Session s(e.get());
    std::string err;
    if (!s.init(&err)) throw std::runtime_error("session init: " + err);
    const auto* api = e->api();
    const int prefix = int(prompt.size());
    flyx_gen_params plain{};
    plain.top_p = 1.f;
    flyx_round_result r{};
    if (api->prefill(e->prefill_plan(), s.seq(), prompt.data(), prefix, &plain) || api->read(s.seq(), &r))
      throw std::runtime_error(api->last_error());
    bool matches = r.n_tokens == 1 && r.tokens[0] == want[0];
    std::vector<float> logits;
    auto round = [&](flyx_plan* p, const flyx_gen_params& g, int rows) {
      if (api->launch(p, s.seq(), &g) || api->read(s.seq(), &r)) throw std::runtime_error(api->last_error());
      for (int row = 0; row < rows; ++row) {
        const size_t at = logits.size();
        logits.resize(at + vocab);
        if (api->logits(s.seq(), row, logits.data() + at)) throw std::runtime_error(api->last_error());
      }
    };

    // Force rejection of only the last draft, then read the history folded from
    // the three accepted inputs.
    const int32_t drafts[] = {want[1], want[2], (want[3] + 1) % vocab};
    flyx_gen_params given = plain;
    given.drafts = drafts;
    given.n_drafts = 3;
    given.drafts_at = prefix + 1;
    round(e->lookup_plan(0), given, 4);
    matches = matches && r.n_accepted == 2 && r.n_tokens == 3 && std::equal(r.tokens, r.tokens + 3, want.begin() + 1);
    round(e->round_plan(0), plain, 1);
    matches = matches && r.n_tokens == 1 && r.tokens[0] == want[4];
    if (api->rewind(s.seq(), prefix + 1, want[1]) != 1) throw std::runtime_error("rewinding inside the round");
    round(e->round_plan(0), plain, 1);
    matches = matches && r.n_tokens == 1 && r.tokens[0] == want[2] &&
              std::memcmp(logits.data() + vocab, logits.data() + 5 * vocab, vocab * sizeof(float)) == 0;
    if (resident_logits.empty()) resident_logits = logits;
    if (fusion.compare(0, 8, "fusion=0") == 0) generic_logits = logits;
    check(matches && logits == generic_logits, model +
                                                   ": greedy tokens, rewind and all six rows of fused logits match "
                                                   "generic at this placement (" +
                                                   fusion + ")");

    // A one-ulp change in an expert sum can cross a quantization boundary in a
    // later layer. Compare every token probability across placements, including
    // the rejected row, and require each top-1.
    double worst_kl = 0;
    bool same_top1 = true;
    for (size_t row = 0; row < logits.size(); row += vocab) {
      const float* a = resident_logits.data() + row;
      const float* b = logits.data() + row;
      const auto ia = std::max_element(a, a + vocab) - a;
      const auto ib = std::max_element(b, b + vocab) - b;
      same_top1 &= ia == ib;
      double za = 0, zb = 0, delta = 0;
      for (int i = 0; i < vocab; ++i) {
        const double x = double(a[i]) - a[ia], y = double(b[i]) - b[ib];
        const double p = std::exp(x);
        za += p;
        zb += std::exp(y);
        delta += p * (x - y);
      }
      const double kl = delta / za + std::log(zb / za);
      worst_kl = std::isfinite(kl) ? std::max(worst_kl, kl) : INFINITY;
    }
    check(same_top1 && worst_kl <= kl_limit,
          model + ": every row matches resident top-1, maximum KL " + std::to_string(worst_kl) + " (" + fusion + ")");
  }
}

// Device bytes the backend's model holds (weights and the copies made of them).
uint64_t weight_bytes(const Engine& e) {
  flyx_memory_info mi{};
  if (!e.api()->memory || e.api()->memory(e.bmodel(), &mi) != 0) throw std::runtime_error("memory() failed");
  return mi.weights;
}

int teacher_forced_matches(Engine& e, const std::vector<int32_t>& prompt, const std::vector<int32_t>& want) {
  int agree = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    std::vector<int32_t> p = prompt;
    p.insert(p.end(), want.begin(), want.begin() + ptrdiff_t(i));
    if (fresh_gen(e, p, 1)[0] == want[i]) ++agree;
  }
  return agree;
}

void test_chat_and_format(const std::string& model_path) {
  std::printf("Inkling-2 chat format, Jinja tojson(sort_keys=true), and grammar\n");
  auto e = load_engine(model_path, "cpu_ref");
  const auto& tok = e->tokenizer();
  check(e->chat().ok(), "chat template loaded");
  check(std::string(e->chat().format()) == "inkling2", "detected chat format is inkling2");

  // Render with Jinja including tojson(sort_keys=true) on unsorted tool call arguments.
  std::vector<ChatMessage> msgs;
  {
    ChatMessage u1;
    u1.role = "user";
    u1.content = "hello";
    msgs.push_back(std::move(u1));
    ChatMessage a1;
    a1.role = "assistant";
    a1.content = "checking";
    a1.reasoning = "thinking step";
    ToolCall tc;
    tc.id = "call_1";
    tc.name = "lookup";
    tc.arguments = ojson::parse(R"({"z":1,"a":2,"m":3})");
    a1.tool_calls.push_back(std::move(tc));
    msgs.push_back(std::move(a1));
    ChatMessage u2;
    u2.role = "user";
    u2.content = "next";
    msgs.push_back(std::move(u2));
  }
  const std::vector<ojson> tools = {ojson::parse(R"({
    "type": "function",
    "function": {
      "name": "lookup",
      "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    }
  })")};
  ChatOptions copt;
  copt.add_generation_prompt = true;
  RenderedChat rc;
  std::string err;
  const bool ok_render = e->chat().render(msgs, tools, copt, &rc, &err, false);
  check(ok_render, "Jinja chat render succeeded" + (err.empty() ? "" : ": " + err));
  check(rc.text.find(R"({"a":2,"m":3,"z":1})") != std::string::npos ||
            rc.text.find(R"({"a": 2, "m": 3, "z": 1})") != std::string::npos,
        "tojson(sort_keys=true) sorted keys a, m, z in rendered prompt");

  // Thinking switched off reaches the template as the effort it reads (it has no enable_thinking), unless the request
  // names an effort itself.
  {
    auto effort = [&](bool thinking, const char* asked) {
      ChatOptions o;
      o.enable_thinking = thinking;
      if (asked) o.kwargs["reasoning_effort"] = asked;
      RenderedChat r;
      std::string why;
      if (!e->chat().render(msgs, tools, o, &r, &why, false)) return "error: " + why;
      const std::string key = "Thinking effort level: ";
      const size_t at = r.text.find(key);
      if (at == std::string::npos) return std::string("not written");
      return r.text.substr(at + key.size(), r.text.find("<|end_message|>", at) - at - key.size());
    };
    const std::string on = effort(true, nullptr), off = effort(false, nullptr), low = effort(false, "low"),
                      none = effort(true, "none");
    check(on == "0.9" && off == "0" && low == "low" && none == "0",
          "thinking off is the template's effort none; a request's own effort stays (" + on + ", " + off + ", " + low +
              ", " + none + ")");
  }

  // Parse Inkling-2 reply with thinking, visible text, and tool call.
  auto parser = e->chat().parser(rc, &tools);
  std::string reasoning, content;
  std::vector<ToolCall> calls;
  auto collect = [&](const std::vector<OutputEvent>& evs) {
    for (const auto& ev : evs) {
      if (ev.kind == OutputEvent::kReasoning) reasoning += ev.text;
      else if (ev.kind == OutputEvent::kContent) content += ev.text;
      else if (ev.kind == OutputEvent::kToolCall) calls.push_back(ev.call);
    }
  };
  collect(parser->feed("<|content_thinking|>plan<|end_message|><|message_model|><|content_text|>hello<|end_message|>"
                       "<|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"a\":2}}"
                       "<|end_message|>"));
  collect(parser->finish());
  check(reasoning == "plan", "parsed <|content_thinking|> section");
  check(content == "hello", "parsed <|content_text|> section");
  check(calls.size() == 1 && calls[0].name == "lookup" && calls[0].arguments.contains("a") &&
            calls[0].arguments["a"] == 2,
        "parsed <|content_invoke_tool_json|> tool call");

  // Grammar construction for tool_choice=required.
  ReplyRules rules;
  rules.tools = ReplyRules::kRequired;
  const std::string grm = e->chat().reply_grammar(tools, rules, rc);
  check(!grm.empty() && grm.find("<|content_invoke_tool_json|>") != std::string::npos,
        "Inkling-2 XGrammar structural tag generated");

  {
    std::vector<int32_t> markers;
    for (const std::string& text : e->chat().reply_tokens()) markers.push_back(tok.added_id(text));
    // A reply as the model writes it, every marker its control token: the grammar takes a marker's token where a
    // delimiter spells it and nowhere else.  A grammar that knows no marker takes none of them (they are special
    // tokens) and is left with their spelling in ordinary tokens.  The stop token as in the real model's file,
    // <|content_model_end_sampling|>: <|end_message|> closes a section and the reply may go on.
    const int32_t eos = tok.added_id("<|content_model_end_sampling|>");
    Grammar g;
    auto matcher = [](Grammar& grammar, const std::string& tag) {
      std::string why;
      auto m = grammar.matcher(tag, &why);
      if (!m) throw std::runtime_error(why);
      return m;
    };
    auto take = [](GrammarMatcher& m, const std::vector<int32_t>& ids) {
      for (int32_t id : ids)
        if (!m.accept(id)) return false;
      return true;
    };
    auto admits = [&](const std::string& tag, const std::string& reply) {
      auto m = matcher(g, tag);
      return take(*m, tok.encode(reply)) && m->can_end();
    };
    auto mask_allows = [&](GrammarMatcher& m, int32_t id) {
      std::vector<uint32_t> mask(g.words());
      m.allowed(mask.data());
      return (mask.at(size_t(id) / 32) & (1u << (size_t(id) % 32))) != 0;
    };
    const std::vector<TextSpan> no_literals;
    const std::string text_begin = "<|message_model|><|content_text|>";
    const std::string end_message = "<|end_message|>";
    const int32_t end_id = tok.added_id(end_message);
    const std::string thought = "<|message_model|><|content_thinking|>plan<|end_message|>";
    const std::string hello = text_begin + "hello" + end_message;
    const std::string call_begin = R"(<|message_model|>lookup<|content_invoke_tool_json|>{"name":"lookup","args":)";
    const std::string call = call_begin + R"({"a":2}})" + end_message;
    const std::vector<int32_t> written = tok.encode(thought + call);
    const int64_t as_tokens = std::count_if(written.begin(), written.end(), [&](int32_t id) {
      return std::find(markers.begin(), markers.end(), id) != markers.end();
    });
    check(markers.size() == 5 && std::none_of(markers.begin(), markers.end(), [](int32_t id) { return id < 0; }) &&
              as_tokens == 6,
          "the format names its reply markers, each a control token of the model");
    g.init(tok, tok.n_tokens(), {eos});
    {
      auto fallback = matcher(g, grm);
      check(!fallback->accept(tok.added_id("<|message_model|>")) &&
                take(*fallback, tok.encode(thought + call, &no_literals)) && fallback->can_end(),
            "a grammar that knows no markers requires their ordinary spellings");
    }
    g.init(tok, tok.n_tokens(), {eos}, markers);
    check(admits(grm, thought + call), "the same tag takes the control tokens after the vocabulary changes");

    const std::string auto_grm = e->chat().reply_grammar(tools, ReplyRules{}, rc);
    auto quoting = matcher(g, auto_grm);
    const int32_t text_id = tok.added_id("<|content_text|>");
    check(take(*quoting, tok.encode(text_begin + "a")) && !mask_allows(*quoting, text_id) &&
              !quoting->accept(text_id) && take(*quoting, tok.encode("<|content_text|>", &no_literals)) &&
              take(*quoting, tok.encode("b" + end_message)) && quoting->can_end(),
          "a section can quote an opening marker only in ordinary tokens");
    auto spelled_end = matcher(g, auto_grm);
    check(take(*spelled_end, tok.encode(text_begin + "a")) &&
              !take(*spelled_end, tok.encode(end_message, &no_literals)),
          "a section refuses an end marker spelled in ordinary tokens");
    auto closed = matcher(g, auto_grm);
    check(take(*closed, tok.encode(hello)) && closed->can_end() && !take(*closed, tok.encode("b")) &&
              closed->accept(tok.added_id("<|message_model|>")) &&
              !closed->accept(tok.added_id("<|content_thinking|>")),
          "the end token closes the text section, allowing a call header but no more text or thinking");
    auto middle = matcher(e->grammar(), grm);
    check(take(*middle, tok.encode("<|message_model|><|content_thinking|>plan")) && !middle->accept(end_id) &&
              !take(*middle, tok.encode(end_message, &no_literals)),
          "the tiny model's stop marker cannot close thinking before a required call, even spelled in ordinary tokens");
    auto stopped = matcher(e->grammar(), auto_grm);
    check(take(*stopped, tok.encode(text_begin + "hello")) && !stopped->can_end() &&
              stopped->accept(end_id) && stopped->can_end(),
          "the tiny model's stop marker completes a text reply at its section end");

    ReplyRules json_answer;
    json_answer.json = true;
    json_answer.schema = ojson::parse(R"({"type":"object","properties":{"s":{"type":"string"}},"required":["s"]})");
    auto string_case = [&](const std::string& tag, const std::string& prefix, const std::string& suffix) {
      auto m = matcher(g, tag);
      return take(*m, tok.encode(prefix)) && !mask_allows(*m, end_id) && !m->accept(end_id) &&
             take(*m, tok.encode(suffix)) && mask_allows(*m, end_id) && m->accept(end_id) && m->can_end();
    };
    check(string_case(e->chat().reply_grammar({}, json_answer, rc), text_begin + R"({"s":"a)", R"("})"),
          "JSON answer masks and refuses the end token inside a string and takes it after the object");
    auto string_tools = tools;
    auto& parameters = string_tools[0]["function"]["parameters"];
    parameters = json_answer.schema;
    parameters["properties"]["s"]["maxLength"] = 64;
    check(string_case(e->chat().reply_grammar(string_tools, rules, rc), call_begin + R"({"s":"a)", R"("}})"),
          "tool call masks and refuses the end token inside a bounded string argument and takes it after the call");

    std::vector<ojson> marked_tools;
    for (const std::string& marker : e->chat().reply_tokens()) {
      auto tool = tools.front();
      tool["function"]["name"] = "x" + marker + "y";
      marked_tools.push_back(std::move(tool));
    }
    // A name that holds a reply marker would put the marker in its call's header: no grammar offers that call.
    check(e->chat().reply_grammar(marked_tools, rules, rc).empty(),
          "tools whose names all hold a reply marker leave no call a grammar could offer");
    marked_tools.push_back(tools.front());
    check(e->chat().reply_grammar(marked_tools, rules, rc) == grm,
          "a tool whose name holds a reply marker is left out beside a callable one");

    // Thinking switched off: the reply has no thinking section, with tools and without, and is otherwise the
    // model's (as many text sections as it writes); left on, a reply with no tools is not held to anything.
    ReplyRules quiet;
    quiet.thinking = false;
    const std::string quiet_text = e->chat().reply_grammar({}, quiet, rc);
    const std::string quiet_tools = e->chat().reply_grammar(tools, quiet, rc);
    check(e->chat().reply_grammar({}, ReplyRules{}, rc).empty() && admits(quiet_text, hello) &&
              !admits(quiet_text, thought + hello) && admits(quiet_tools, hello) &&
              admits(quiet_tools, call) && !admits(quiet_tools, thought + call) &&
              admits(auto_grm, thought + hello) && admits(auto_grm, thought + call),
          "thinking off: text or calls, no thinking section; on: an optional one first");
    check(admits(quiet_text, hello + hello) && !admits(quiet_text, hello + thought) && !admits(quiet_text, hello + call),
          "thinking off, no tools: text sections and nothing else");
  }

  // Round-trip tokenizer encode/decode on control tokens and text.
  auto ids = tok.encode("<|message_user|><|content_text|>hello<|end_message|>");
  check(!ids.empty() && ids.front() == 264 && ids.back() == 269, "inkling2/o200k tokenizer encodes control tokens");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 3) {
    std::fprintf(stderr, "usage: inkling2_test WORK_DIR cpu_ref|cuda\n");
    return 2;
  }
  const std::filesystem::path dir(argv[1]);
  const std::string backend = argv[2];
  try {
    const TinySpec spec;
    const auto bf16_gguf = dir / "tiny_inkling2_bf16.gguf";
    const auto mixed_gguf = dir / "tiny_inkling2_mixed.gguf";
    const auto dspark_bf16 = dir / "dspark-tiny-bf16.gguf";
    const auto dspark_q8 = dir / "dspark-tiny-q8.gguf";
    const auto dflash2_bf16 = dir / "dflash2-tiny-bf16.gguf";
    const auto dflash2_q8 = dir / "dflash2-tiny-q8.gguf";
    const auto dflash2_dir = dir / "dflash2-tiny-hf";
    write_tiny_inkling2(bf16_gguf, spec, false, 1001);
    write_tiny_inkling2(mixed_gguf, spec, true, 2002);
    write_tiny_dspark(dspark_bf16, spec, false, 3003);
    write_tiny_dspark(dspark_q8, spec, true, 4004);
    write_tiny_dflash2_gguf(dflash2_bf16, spec, false, /*padded_codebook=*/true, 5005);
    // Use more query heads than the target (8 > 4) so wide-block DFlash2 rounds (B=16) guard attention partial sizing.
    write_tiny_dflash2_gguf(dflash2_q8, spec, true, /*padded_codebook=*/false, 6006, 32, spec.nh * 2);
    write_tiny_dflash2_dir(dflash2_dir, spec, 7007);

    for (bool drafter : {false, true}) {
      const auto bad = dir / (drafter ? "dspark_excess_heads.gguf" : "inkling_excess_heads.gguf");
      EngineOptions opts{};
      opts.backend = "cpu_ref";
      if (drafter) {
        write_tiny_dspark(bad, spec, false, 3003, 32, 1u << 27);
        opts.drafter = "dspark";
        opts.speculative = true;
        opts.drafter_files = {bad.string()};
      } else {
        write_tiny_inkling2(bad, spec, false, 1001, RoundMatrixCase::Generated, 1u << 27);
      }
      Engine engine;
      std::string why;
      check(!engine.load((drafter ? bf16_gguf : bad).string(), opts, &why) &&
                why.find(drafter ? "invalid DSpark drafter dimensions" : "invalid inkling2_loop GGUF configuration") !=
                    std::string::npos,
            "excessive attention head count rejected before forming weight dimensions");
    }
    test_chat_and_format(bf16_gguf.string());

    std::vector<int32_t> prompt;
    for (int i = 0; i < 20; ++i) prompt.push_back(10 + (i * 7) % 240);

    // ---- 1. Float64 cpu_ref baseline + cpu_ref DSpark + cpu_ref DFlash2 (GGUF & HF dir) + all ----
    std::printf("Inkling-2 bf16+nvfp4 target (%s vs cpu_ref)\n", backend.c_str());
    auto ref_plain = load_engine(bf16_gguf.string(), "cpu_ref");
    auto ref_dspark = load_engine(bf16_gguf.string(), "cpu_ref",
                                  {.speculative = true, .drafter = "dspark", .drafter_files = {dspark_bf16.string()}});
    auto ref_dflash2 = load_engine(bf16_gguf.string(), "cpu_ref",
                                   {.speculative = true, .drafter = "dflash2", .drafter_files = {dflash2_bf16.string()}});
    auto ref_dflash2_dir =
        load_engine(bf16_gguf.string(), "cpu_ref",
                    {.speculative = true, .drafter = "dflash2", .drafter_files = {dflash2_dir.string()}});
    auto ref_both = load_engine(bf16_gguf.string(), "cpu_ref",
                                {.speculative = true,
                                 .drafter = "all",
                                 .drafter_files = {dspark_bf16.string(), dflash2_bf16.string()}});
    check(std::string(ref_dspark->drafter()) == "dspark", "cpu_ref loaded DSpark companion drafter");
    check(std::string(ref_dflash2->drafter()) == "dflash2", "cpu_ref loaded DFlash2 GGUF companion drafter");
    check(std::string(ref_dflash2_dir->drafter()) == "dflash2", "cpu_ref loaded DFlash2 HF dir companion drafter");
    check(ref_both->model().drafters() == 2, "cpu_ref loaded DSpark + DFlash2 simultaneously (--drafter all)");
    const auto ref_greedy = fresh_gen(*ref_plain, prompt, 24, false);
    const auto ref_sampled = fresh_gen(*ref_plain, prompt, 24, true);
    check(same_tokens(fresh_gen(*ref_dspark, prompt, 24, false), ref_greedy),
          "cpu_ref DSpark greedy == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_dspark, prompt, 24, true), ref_sampled),
          "cpu_ref DSpark sampled+penalties == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_dflash2, prompt, 24, false), ref_greedy),
          "cpu_ref DFlash2 GGUF greedy == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_dflash2, prompt, 24, true), ref_sampled),
          "cpu_ref DFlash2 GGUF sampled+penalties == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_dflash2_dir, prompt, 24, false), ref_greedy),
          "cpu_ref DFlash2 HF dir greedy == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_both, prompt, 24, false), ref_greedy),
          "cpu_ref DSpark + DFlash2 (--drafter all) greedy == cpu_ref plain");
    // A MoE that routes each row to more experts than the CUDA kernels take (65 of 72; moe_route keeps 64): cpu_ref
    // runs it, and the CUDA load refuses it before allocating anything, saying why.
    {
      TinySpec routed = spec;
      routed.n_experts = 72;
      routed.top_k = 65;
      const auto path = dir / "tiny_inkling2_top65.gguf";
      write_tiny_inkling2(path, routed, false, 1001);
      check(fresh_gen(*load_engine(path.string(), "cpu_ref"), prompt, 4, false).size() == 4,
            "cpu_ref runs a MoE routing each row to 65 of 72 experts");
      if (backend == "cuda") {
        std::string why;
        try {
          (void)load_engine(path.string(), "cuda");
        } catch (const std::exception& e) {
          why = e.what();
        }
        check(why.find("route to up to 64") != std::string::npos,
              "the CUDA load refuses a MoE routing to 65 experts a row (" + why + ")");
      }
    }
    {
      const auto clip_gguf = dir / "mmproj-BF16.gguf";
      GgufWriter cw;
      cw.add_str("general.architecture", "clip");
      cw.add_tensor("v.dummy", {4}, kGgmlF32, std::vector<uint8_t>(16, 0));
      cw.write(clip_gguf);
      EngineOptions clip_opts{};
      clip_opts.backend = "cpu_ref";
      clip_opts.mmproj_file = clip_gguf.string();
      Engine clip_eng;
      std::string clip_err;
      const bool ok = clip_eng.load(bf16_gguf.string(), clip_opts, &clip_err);
      check(!ok && clip_err.find("vision projector") != std::string::npos &&
                clip_err.find("drafter file") == std::string::npos,
            "inkling2_loop refuses --mmproj clip companion with a vision projector message (" + clip_err + ")");
    }

    // A prompt asked again, its conversation ending where the prompt does (the real template opens a reply with
    // nothing of its own): the session kept the state its last prefill pass started from, and runs that pass again.
    // Eight rounds later no rollback reaches the prompt (FLYX_ROUND_SLOTS).  The boundary gives the first request no
    // pass of its own and moves none: its passes are those of a request that names no boundary (32 rows, then 24),
    // so its first token is computed as theirs is, and the one asked again computes it from the same 24 rows.
    {
      std::vector<int32_t> longer;
      for (int i = 0; i < 56; ++i) longer.push_back(10 + (i * 11) % 240);
      auto e = load_engine(bf16_gguf.string(), backend);
      struct Asked {
        std::vector<int32_t> tokens;
        std::vector<int> passes;  // the prompt tokens in the state as the request starts, then after each pass
        int reused = -1;
      };
      // boundary: the conversation ends where the prompt does; question: and that end is a question's.
      auto ask = [&](Session& s, const std::vector<int32_t>& prompt, bool boundary, bool question) {
        Asked a;
        GenerateRequest req;
        req.prompt = prompt;
        req.max_new_tokens = 8;
        req.ignore_eos = true;
        if (boundary) req.checkpoints = {int32_t(prompt.size())};
        if (question) req.keep_checkpoint = int32_t(prompt.size());
        req.on_prefill = [&](int done, int) { a.passes.push_back(done); };
        GenerateResult res;
        std::string gerr;
        if (generate(s, req,
                     [&](std::string_view, const int32_t* ids, int cnt) {
                       a.tokens.insert(a.tokens.end(), ids, ids + cnt);
                       return true;
                     },
                     &res, &gerr) != 0)
          throw std::runtime_error("generate: " + gerr);
        a.reused = res.usage.reused_tokens;
        return a;
      };
      Session s(e.get()), bare(e.get()), chat(e.get());
      std::string serr;
      if (!s.init(&serr) || !bare.init(&serr) || !chat.init(&serr)) throw std::runtime_error("session init: " + serr);
      const Asked first = ask(s, longer, true, true), again = ask(s, longer, true, true),
                  unmarked = ask(bare, longer, false, false);
      check(first.reused == 0 && first.passes == std::vector<int>{0, 32, 56} && unmarked.passes == first.passes &&
                same_tokens(first.tokens, unmarked.tokens),
            "a boundary at the prompt's end adds no prefill pass and moves none (" + backend + ")");
      check(again.reused == 32 && again.passes == std::vector<int>{32, 56} && same_tokens(again.tokens, first.tokens),
            "a prompt asked again runs its last pass again (" + std::to_string(again.reused) + " of " +
                std::to_string(longer.size()) + " reused), the same tokens (" + backend + ")");

      // A question that ends where its prompt does (one pass here: no pass start to keep): the checkpoint at its end
      // is the question's, kept through the turns that follow, six of them with a checkpoint each at its own end
      // where a sequence has four slots.  The next question's prompt leaves the conversation right after the first
      // (a template drops the thinking of the turns since), and starts from that checkpoint.
      const std::vector<int32_t> question(longer.begin(), longer.begin() + 24);
      ask(chat, question, true, true);
      std::vector<int32_t> turns = question;
      for (int t = 0; t < 6; ++t) {
        for (int i = 0; i < 9; ++i) turns.push_back(40 + int(turns.size() * 7 % 200));
        ask(chat, turns, true, false);
      }
      std::vector<int32_t> next = question;
      for (int i = 0; i < 12; ++i) next.push_back(3 + i * 5);
      const Asked asked_next = ask(chat, next, true, true);
      check(asked_next.reused == int(question.size()),
            "a question's checkpoint at its prompt's end is kept through six turns (" +
                std::to_string(asked_next.reused) + " of " + std::to_string(question.size()) + " reused, " + backend +
                ")");
    }

    if (backend == "cuda") {
      // The attention's producers on the mixed weights; the MoE's on experts whose gate|up GEMM reads 8-bit rows
      // and whose down GEMM is NVFP4.
      test_fused_rounds(bf16_gguf, "BF16 trunk", prompt, 1e-6);
      test_fused_rounds(mixed_gguf, "mixed weights", prompt, 2e-4);
      {
        TinySpec experts = spec;
        experts.packed_gate_nvfp4_down = true;
        const auto path = dir / "tiny_inkling2_mixed_experts.gguf";
        write_tiny_inkling2(path, experts, true, 2002);
        test_fused_rounds(path, "mixed experts", prompt, 2e-4);
      }
      {
        auto held = [&](RoundMatrixCase round_case, const char* filename) {
          const auto path = dir / filename;
          write_tiny_inkling2(path, spec, false, 1001, round_case);
          return weight_bytes(*load_engine(path.string(), "cuda"));
        };
        const uint64_t exact = held(RoundMatrixCase::ExactBf16, "tiny_inkling2_exact_round.gguf");
        const uint64_t inexact = held(RoundMatrixCase::OneLowBit, "tiny_inkling2_inexact_round.gguf");
        const size_t router = size_t(spec.n_experts) * spec.H, forget = size_t(spec.H) * spec.gate_rank;
        check(inexact == exact + (router + forget) * sizeof(float),
              "one low bit in each repeated core matrix retains exactly one F32 router and one F32 forget matrix");
      }
      // A softcap small enough to saturate: cap * tanh(z / cap) is the cap itself for every score well above the
      // cap, and the greedy choice the lowest token id of those, which a screen bounding z apart would have cut.  A
      // softcapped head holds no certified copy: greedy decoding with head_cert on (the default) reads the whole head,
      // in plain rounds and in lookup rounds of several rows, token for token what head_cert=0 gives.
      {
        TinySpec capped = spec;
        capped.softcap = 0.01f;
        const auto path = dir / "tiny_inkling2_saturated_cap.gguf";
        write_tiny_inkling2(path, capped, false, 1001);
        auto whole = load_engine(path.string(), "cuda", {.backend_opts = "head_cert=0"});
        auto cert = load_engine(path.string(), "cuda");
        auto cert_rows = load_engine(path.string(), "cuda", {.speculative = true, .lookup = true, .width = 0});
        const auto want = fresh_gen(*whole, prompt, 32, false);
        // The tie is there: the first token's capped scores have their highest at two or more ids, the lowest chosen.
        int ties = 0, lowest = -1;
        {
          Session s(whole.get());
          std::string err;
          if (!s.init(&err)) throw std::runtime_error(err);
          const auto* api = whole->api();
          flyx_gen_params gp{};
          gp.top_p = 1.f;
          flyx_round_result rr{};
          std::vector<float> z(size_t(whole->model().desc().vocab));
          if (api->prefill(whole->prefill_plan(), s.seq(), prompt.data(), int32_t(prompt.size()), &gp) ||
              api->read(s.seq(), &rr) || api->logits(s.seq(), 0, z.data()))
            throw std::runtime_error(api->last_error());
          const float top = *std::max_element(z.begin(), z.end());
          for (size_t v = 0; v < z.size(); ++v)
            if (z[v] == top && ties++ == 0) lowest = int(v);
        }
        check(ties >= 2 && !want.empty() && want[0] == lowest,
              "saturating softcap: " + std::to_string(ties) + " ids tie at the cap, the lowest (" +
                  std::to_string(lowest) + ") chosen");
        check(weight_bytes(*cert) == weight_bytes(*whole), "a softcapped head holds no certified copy");
        check(same_tokens(fresh_gen(*cert, prompt, 32, false), want),
              "saturating softcap: greedy with head_cert on == the whole head");
        check(same_tokens(fresh_gen(*cert_rows, prompt, 32, false), want),
              "saturating softcap: greedy with head_cert on, lookup rounds of several rows == the whole head");
      }
      // ---- 2. CUDA vs cpu_ref, Gate 2 (fusion=0 == fusion=1), Gate 4 (plain == DSpark), CUDA graphs ----
      auto cu_plain = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32});
      auto cu_unfused = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32, .backend_opts = "fusion=0"});
      auto cu_nographs = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32, .backend_opts = "graphs=0"});
      auto cu_small_chunk = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 8});
      auto cu_dspark = load_engine(bf16_gguf.string(), "cuda",
                                   {.speculative = true,
                                    .prefill_chunk = 32,
                                    .drafter = "dspark",
                                    .drafter_files = {dspark_bf16.string()}});
      auto cu_dspark_unfused = load_engine(bf16_gguf.string(), "cuda",
                                           {.speculative = true,
                                            .prefill_chunk = 32,
                                            .drafter = "dspark",
                                            .drafter_files = {dspark_bf16.string()},
                                            .backend_opts = "fusion=0"});
      check(std::string(cu_dspark->drafter()) == "dspark", "cuda loaded DSpark companion drafter");

      const auto cu_greedy = fresh_gen(*cu_plain, prompt, 24, false);
      for (const auto* cap :
           {"gpu_experts=2", "gpu_experts=0", "gpu_experts=2,fusion=0", "gpu_experts=0,expert_prefill_min=2"}) {
        auto hybrid = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32, .backend_opts = cap});
        check(same_tokens(fresh_gen(*hybrid, prompt, 24, false), cu_greedy),
              std::string("hybrid routed experts match resident greedy tokens: ") + cap);
      }
      {
        auto e = load_engine(bf16_gguf.string(), "cuda",
                             {.prefill_chunk = 32, .backend_opts = "gpu_experts=2,expert_prefill_min=2"});
        std::string error;
        {
          Session a(e.get()), b(e.get());
          if (!a.init(&error) || !b.init(&error)) throw std::runtime_error(error);
          GenerateRequest request;
          request.prompt = prompt;
          request.max_new_tokens = 24;
          request.ignore_eos = true;
          std::vector<int32_t> result[2];
          auto collect = [&](int index) {
            return [&, index](std::string_view, const int32_t* ids, int count) {
              result[index].insert(result[index].end(), ids, ids + count);
              return true;
            };
          };
          Generation first(&a, request, collect(0)), second(&b, request, collect(1));
          while (!first.done() || !second.done()) {
            if (!first.done()) first.advance(true);
            if (!second.done()) second.advance(true);
          }
          check(result[0] == cu_greedy && result[1] == cu_greedy, "interleaved hybrid sessions match resident tokens");
        }
        check(same_tokens(fresh_gen(*e, prompt, 24), cu_greedy), "fresh session after promotion");
      }
      const auto cu_sampled = fresh_gen(*cu_plain, prompt, 24, true);
      check(same_tokens(cu_greedy, ref_greedy), "cuda plain greedy == float64 cpu_ref");
      check(same_tokens(cu_sampled, ref_sampled), "cuda plain sampled+penalties == float64 cpu_ref");
      check(same_tokens(fresh_gen(*cu_unfused, prompt, 24, false), cu_greedy), "Gate 2: cuda fusion=0 == fusion=1");
      check(same_tokens(fresh_gen(*cu_nographs, prompt, 24, false), cu_greedy), "cuda graphs=0 == graphs=1");
      check(same_tokens(fresh_gen(*cu_small_chunk, prompt, 24, false), cu_greedy),
            "cuda prefill_chunk=8 (direct MoE) == prefill_chunk=32 (grouped MoE)");
      check(same_tokens(fresh_gen(*cu_dspark, prompt, 24, false), cu_greedy),
            "Gate 4: cuda DSpark greedy == cuda plain");
      check(same_tokens(fresh_gen(*cu_dspark, prompt, 24, true), cu_sampled),
            "Gate 4: cuda DSpark sampled+penalties == cuda plain");
      check(same_tokens(fresh_gen(*cu_dspark_unfused, prompt, 24, false), cu_greedy),
            "Gate 2+4: cuda DSpark fusion=0 == cuda plain");

      // ---- 2b. Inkling-2 + DFlash2 (GGUF BF16/Q8_0, HF safetensors dir) & DSpark + DFlash2 together ----
      std::printf("Inkling-2 DFlash2 (GGUF BF16/Q8_0, HF safetensors) and DSpark+DFlash2 simultaneously\n");
      {
        auto cu_df_bf16 = load_engine(bf16_gguf.string(), "cuda",
                                      {.speculative = true,
                                       .prefill_chunk = 32,
                                       .drafter = "dflash2",
                                       .drafter_files = {dflash2_bf16.string()}});
        auto cu_df_unfused = load_engine(bf16_gguf.string(), "cuda",
                                         {.speculative = true,
                                          .prefill_chunk = 32,
                                          .drafter = "dflash2",
                                          .drafter_files = {dflash2_bf16.string()},
                                          .backend_opts = "fusion=0"});
        auto cu_df_nographs = load_engine(bf16_gguf.string(), "cuda",
                                          {.speculative = true,
                                           .prefill_chunk = 32,
                                           .drafter = "dflash2",
                                           .drafter_files = {dflash2_bf16.string()},
                                           .backend_opts = "graphs=0"});
        auto cu_df_dir = load_engine(bf16_gguf.string(), "cuda",
                                     {.speculative = true,
                                      .prefill_chunk = 32,
                                      .drafter = "dflash2",
                                      .drafter_files = {dflash2_dir.string()}});
        auto cu_df_q8 = load_engine(mixed_gguf.string(), "cuda",
                                    {.speculative = true,
                                     .prefill_chunk = 32,
                                     .drafter = "dflash2",
                                     .drafter_files = {dflash2_q8.string()}});
        auto cu_mix_plain_df = load_engine(mixed_gguf.string(), "cuda", {.prefill_chunk = 32});
        auto cu_both = load_engine(bf16_gguf.string(), "cuda",
                                   {.speculative = true,
                                    .prefill_chunk = 32,
                                    .drafter = "all",
                                    .drafter_files = {dspark_bf16.string(), dflash2_bf16.string()}});
        check(std::string(cu_df_bf16->drafter()) == "dflash2", "cuda loaded DFlash2 GGUF on Inkling-2");
        check(std::string(cu_df_dir->drafter()) == "dflash2", "cuda loaded DFlash2 HF safetensors on Inkling-2");
        check(cu_both->model().drafters() == 2, "cuda loaded DSpark + DFlash2 simultaneously (--drafter all)");
        check(same_tokens(fresh_gen(*cu_df_bf16, prompt, 24, false), cu_greedy),
              "Gate 4: cuda DFlash2 GGUF greedy == cuda plain");
        check(same_tokens(fresh_gen(*cu_df_bf16, prompt, 24, true), cu_sampled),
              "Gate 4: cuda DFlash2 GGUF sampled+penalties == cuda plain");
        check(same_tokens(fresh_gen(*cu_df_unfused, prompt, 24, false), cu_greedy),
              "Gate 2+4: cuda DFlash2 GGUF fusion=0 == cuda plain");
        check(same_tokens(fresh_gen(*cu_df_nographs, prompt, 24, true), cu_sampled),
              "Gate 4: cuda DFlash2 GGUF graphs=0 sampled == cuda plain");
        check(same_tokens(fresh_gen(*cu_df_dir, prompt, 24, false), cu_greedy),
              "Gate 4: cuda DFlash2 HF safetensors greedy == cuda plain");
        check(same_tokens(fresh_gen(*cu_df_dir, prompt, 24, true), cu_sampled),
              "Gate 4: cuda DFlash2 HF safetensors sampled+penalties == cuda plain");
        const auto mix_g = fresh_gen(*cu_mix_plain_df, prompt, 24, false);
        const auto mix_s = fresh_gen(*cu_mix_plain_df, prompt, 24, true);
        check(same_tokens(fresh_gen(*cu_df_q8, prompt, 24, false), mix_g),
              "Gate 4: cuda mixed-quant Inkling-2 + Q8_0 DFlash2 GGUF greedy == plain");
        check(same_tokens(fresh_gen(*cu_df_q8, prompt, 24, true), mix_s),
              "Gate 4: cuda mixed-quant Inkling-2 + Q8_0 DFlash2 GGUF sampled+penalties == plain");
        check(same_tokens(fresh_gen(*cu_both, prompt, 24, false), cu_greedy),
              "Gate 4: cuda DSpark + DFlash2 (--drafter all) greedy == cuda plain");
        check(same_tokens(fresh_gen(*cu_both, prompt, 24, true), cu_sampled),
              "Gate 4: cuda DSpark + DFlash2 (--drafter all) sampled+penalties == cuda plain");

        // 16-row (width=15) row-count invariance + adaptive conditional graphs (rounds chosen) + oracle + lookup.
        for (const bool mixed : {false, true}) {
          const std::string tag = mixed ? "mixed-quant" : "bf16";
          const auto& tgt = mixed ? mixed_gguf : bf16_gguf;
          const auto& df = mixed ? dflash2_q8 : dflash2_bf16;
          auto e_plain = load_engine(tgt.string(), "cuda", {.prefill_chunk = 32, .backend_opts = "head_cert=0"});
          const auto want48 = fresh_gen(*e_plain, prompt, 48, false);
          const auto want48_s = fresh_gen(*e_plain, prompt, 48, true);

          // 16-row lookup round (width=15, 14 accepted + 1 rejected) logits vs 1-row plain and fusion=0 vs fusion=1.
          for (const std::string fusion : {"fusion=0", "fusion=1"}) {
            auto e16 = load_engine(tgt.string(), "cuda",
                                   {.speculative = true,
                                    .lookup = true,
                                    .prefill_chunk = 32,
                                    .width = 0,
                                    .backend_opts = "head_cert=0,blocks=packed," + fusion});
            size_t idx15 = SIZE_MAX;
            for (size_t i = 0; i < e16->lookup_widths().size(); ++i)
              if (e16->lookup_widths()[i] == 15) idx15 = i;
            check(idx15 != SIZE_MAX, tag + " (" + fusion + "): has lookup width=15 plan");
            if (idx15 != SIZE_MAX) {
              const int vocab = e16->model().desc().vocab;
              // Collect 1-row reference logits for steps 0..16.
              std::vector<float> ref_logits(17 * size_t(vocab));
              {
                Session sr(e16.get());
                std::string serr;
                if (!sr.init(&serr)) throw std::runtime_error(serr);
                flyx_gen_params gp{};
                gp.top_p = 1.f;
                flyx_round_result rr{};
                const auto* api = e16->api();
                if (api->prefill(e16->prefill_plan(), sr.seq(), prompt.data(), int32_t(prompt.size()), &gp) ||
                    api->read(sr.seq(), &rr))
                  throw std::runtime_error(api->last_error());
                for (int step = 0; step < 17; ++step) {
                  if (api->launch(e16->round_plan(0), sr.seq(), &gp) || api->read(sr.seq(), &rr))
                    throw std::runtime_error(api->last_error());
                  if (api->logits(sr.seq(), 0, ref_logits.data() + size_t(step) * vocab))
                    throw std::runtime_error(api->last_error());
                }
              }
              // Run 16-row lookup round with 14 accepted drafts + 1 wrong, then a 1-row follow-up round.
              Session s16(e16.get());
              std::string serr;
              if (!s16.init(&serr)) throw std::runtime_error(serr);
              flyx_gen_params gp{};
              gp.top_p = 1.f;
              flyx_round_result rr{};
              const auto* api = e16->api();
              if (api->prefill(e16->prefill_plan(), s16.seq(), prompt.data(), int32_t(prompt.size()), &gp) ||
                  api->read(s16.seq(), &rr))
                throw std::runtime_error(api->last_error());
              int32_t drafts15[15];
              for (int i = 0; i < 14; ++i) drafts15[i] = want48[size_t(1 + i)];
              drafts15[14] = (want48[15] + 1) % vocab;
              flyx_gen_params gv = gp;
              gv.drafts = drafts15;
              gv.n_drafts = 15;
              gv.drafts_at = int32_t(prompt.size()) + 1;
              if (api->launch(e16->lookup_plan(idx15), s16.seq(), &gv) || api->read(s16.seq(), &rr))
                throw std::runtime_error(api->last_error());
              bool ok16 = rr.n_accepted == 14 && rr.n_tokens == 15 &&
                          std::equal(rr.tokens, rr.tokens + 15, want48.begin() + 1);
              int bad_row = -1;
              float max_diff = 0.f;
              std::vector<float> got(static_cast<size_t>(vocab));
              for (int row = 0; row < 15; ++row) {
                if (api->logits(s16.seq(), row, got.data())) throw std::runtime_error(api->last_error());
                for (int v = 0; v < vocab; ++v) {
                  const float d = std::fabs(got[size_t(v)] - ref_logits[size_t(row) * vocab + size_t(v)]);
                  if (d > max_diff) {
                    max_diff = d;
                    if (bad_row < 0 && d > 0.f) bad_row = row;
                  }
                }
              }
              if (api->launch(e16->round_plan(0), s16.seq(), &gp) || api->read(s16.seq(), &rr))
                throw std::runtime_error(api->last_error());
              const bool next_ok = rr.n_tokens == 1 && rr.tokens[0] == want48[16];
              check(ok16 && next_ok && max_diff == 0.f,
                    tag + " (" + fusion + "): 16-row lookup round (14 accepted) bit-identical to 1-row plain" +
                        (max_diff > 0.f ? " [bad_row=" + std::to_string(bad_row) + " max_diff=" + std::to_string(max_diff) + "]" : ""));
            }
          }

          // Adaptive rounds (the rounds chosen every round, lookup rounds too): a block drafter's whole block and one
          // decision (DFlash2, stop_decide_block); DSpark's backbone once and its Markov head a checkpoint at a time,
          // each step's decision on the device (stop_decide_step: another step, or the verify body of the prefix that
          // pays).  Free, and with the oracle, whose accepted runs make the drafter's rounds pay so that they run and
          // reach the later checkpoints.  (Pinned at a width, 0 included, an engine has no adaptive plan.)
          for (const std::string which : {"dflash2", "dspark"}) {
            const auto& file = which == "dspark" ? (mixed ? dspark_q8 : dspark_bf16) : df;
            const std::string name = which == "dspark" ? "DSpark" : "DFlash2";
            auto e_adapt = load_engine(tgt.string(), "cuda",
                                       {.speculative = true,
                                        .lookup = true,
                                        .prefill_chunk = 32,
                                        .width = kChosen,
                                        .drafter = which,
                                        .drafter_files = {file.string()}});
            check(e_adapt->model().drafters() == 1 && e_adapt->adaptive_plan(0) != nullptr,
                  tag + " adaptive " + name + ": the drafter has an adaptive round plan");
            check(same_tokens(fresh_gen(*e_adapt, prompt, 48, false), want48),
                  tag + " Gate 4: adaptive " + name + " + lookup greedy == plain");
            check(same_tokens(fresh_gen(*e_adapt, prompt, 48, true), want48_s),
                  tag + " Gate 4: adaptive " + name + " + lookup sampled+penalties == plain");
            Session so(e_adapt.get());
            std::string serr;
            if (!so.init(&serr)) throw std::runtime_error(serr);
            std::vector<int32_t> oracle = prompt;
            oracle.insert(oracle.end(), want48.begin(), want48.end());
            if (e_adapt->api()->set_oracle(so.seq(), oracle.data(), int32_t(oracle.size())) != 0)
              throw std::runtime_error(e_adapt->api()->last_error());
            GenerateResult res;
            check(same_tokens(run_gen(so, prompt, 48, false, {}, -1, &res), want48),
                  tag + " Gate 4: adaptive " + name + " + oracle (forced accepted runs) greedy == plain");
            // Launched on the adaptive plan, not on a fixed round the engine falls back to (a round launched as the
            // reply ends is counted there and never read, so the count may pass the drafter's rounds read by one or
            // two); and drafts accepted beyond the lookup rounds'.
            const int drafted = res.drafter_rounds.empty() ? 0 : res.drafter_rounds[0];
            check(res.adaptive_rounds > 0 && drafted > 0 && res.usage.accepted_drafts > res.lookup_accepted,
                  tag + " adaptive " + name + " + oracle: rounds launched on the adaptive plan (" +
                      std::to_string(res.adaptive_rounds) + "; the drafter's rounds read " + std::to_string(drafted) +
                      ", " + std::to_string(res.usage.accepted_drafts - res.lookup_accepted) + " drafts accepted)");
          }
          // Fixed width=15 DFlash2 + oracle.
          auto e_w15 = load_engine(tgt.string(), "cuda",
                                   {.speculative = true,
                                    .lookup = true,
                                    .prefill_chunk = 32,
                                    .width = 15,
                                    .drafter = "dflash2",
                                    .drafter_files = {df.string()}});
          {
            Session so(e_w15.get());
            std::string serr;
            if (!so.init(&serr)) throw std::runtime_error(serr);
            std::vector<int32_t> oracle = prompt;
            oracle.insert(oracle.end(), want48.begin(), want48.end());
            if (e_w15->api()->set_oracle(so.seq(), oracle.data(), int32_t(oracle.size())) != 0)
              throw std::runtime_error(e_w15->api()->last_error());
            check(same_tokens(run_gen(so, prompt, 48, false), want48),
                  tag + " Gate 4: width=15 DFlash2 + oracle (15 accepted/round) greedy == plain");
          }
        }
      }

      // ---- 2c. Tree rounds: an Inkling-2 target verifies chains only ----
      // Relative-position attention scores a round's rows as consecutive positions and its ShortConvs (and the
      // MoE's) fold them in order, with no replay of a fork: of two siblings, the second would read the first.  The
      // backend refuses every tree round (given, and DFlash2's drafted), so that no tree reaches a launch; the engine
      // keeps plain rounds, its drafters' chains and lookup's.
      std::printf("Inkling-2 tree rounds refused (relative-position attention, ShortConv)\n");
      {
        auto e = load_engine(bf16_gguf.string(), "cuda",
                             {.speculative = true,
                              .lookup = true,
                              .prefill_chunk = 32,
                              .width = kChosen,
                              .drafter = "dflash2",
                              .drafter_files = {dflash2_bf16.string()},
                              .draft_trees = true});
        const auto& trees = e->round_trees();
        check(e->tree_widths().empty() && !e->lookup_widths().empty() && e->widths().size() > 1 &&
                  std::find(trees.begin(), trees.end(), true) == trees.end(),
              "with draft trees asked for, the engine lowers DFlash2's chains, lookup's and plain rounds, no tree");
        const auto* api = e->api();
        auto lowers = [&](std::vector<flyx_node> rnd, int k, std::string* why) {
          flyx_program rp{FLYX_PHASE_ROUND, k, k + 1, int32_t(rnd.size()), rnd.data()};
          flyx_plan* plan = nullptr;
          const int rc = api->lower(e->bmodel(), &rp, &plan);
          if (plan) api->plan_free(plan);
          *why = rc != 0 && api->last_error() ? api->last_error() : "";
          return rc == 0;
        };
        // The tree of three given drafts the lookup would send with parents {-1, -1, 0}: drafts 0 and 1 siblings.
        std::string why;
        auto given_tree = e->model().round_program(3, -1, true);
        for (auto& n : given_tree)
          if (n.kind == FLYX_NODE_GIVEN) n.arg1 = FLYX_GIVEN_TREE;
        check(!lowers(given_tree, 3, &why) && why.find("tree rounds on relative-position attention") != std::string::npos,
              "a round verifying a tree of given drafts (siblings) is refused at lower (" + why + ")");
        auto drafted_tree = e->model().round_program(9, 0, false);
        for (auto& n : drafted_tree)
          if (n.kind == FLYX_NODE_DRAFT) n.arg1 = FLYX_DRAFT_TREE;
        check(!lowers(drafted_tree, 9, &why) && why.find("tree rounds on relative-position attention") != std::string::npos,
              "a round verifying DFlash2's drafted tree is refused at lower (" + why + ")");
        check(lowers(e->model().round_program(3, -1, true), 3, &why) && lowers(e->model().round_program(7, 0, false), 7, &why),
              "the same rounds as chains (given drafts, DFlash2's) still lower");
        check(same_tokens(fresh_gen(*e, prompt, 24, false), cu_greedy),
              "Gate 4: DFlash2 + lookup with draft trees asked for (chains only) greedy == cuda plain");
      }

      // ---- 3. Rewind, edited prompt, and SWA ring wraparound checkpoint/restore & snapshot ----
      std::printf("Inkling-2 rewind, edit, and SWA ring wraparound checkpoint/snapshot\n");
      {
        Session s(cu_dspark.get());
        std::string err;
        if (!s.init(&err)) throw std::runtime_error(err);
        auto t0 = run_gen(s, prompt, 24, false);
        check(same_tokens(t0, cu_greedy), "session initial DSpark run");
        // Rewind inside last rounds and continue from prompt + first 12 generated tokens.
        auto cont = prompt;
        cont.insert(cont.end(), cu_greedy.begin(), cu_greedy.begin() + 12);
        auto t1 = run_gen(s, cont, 12, false);
        check(same_tokens(t1, std::vector<int32_t>(cu_greedy.begin() + 12, cu_greedy.end())),
              "session rewind + continuation (ShortConv fold + SWA ring)");
        // Prompt edit.
        auto edit = prompt;
        edit.back() = (edit.back() + 5) % 240;
        check(same_tokens(run_gen(s, edit, 16, false), fresh_gen(*cu_plain, edit, 16, false)),
              "session edited prompt == fresh session");
      }

      // SWA ring wraparound test: max_ctx = 512, prefill_chunk = 32 -> SWA ring = 128 slots (< 544)!
      // Save kept checkpoint at pos 260 (>= kSnapshotWorth = 256), advance past pos 420 (overwriting ALL 128 slots
      // of the live SWA ring past 260 + 128 = 388), save_to(store) while wrapped, then restore checkpoint AND load
      // snapshot into a fresh session -> both must match fresh prefill at pos 261!
      {
        std::vector<int32_t> long_p(420);
        for (int i = 0; i < 420; ++i) long_p[size_t(i)] = 5 + (i * 13) % 240;
        std::vector<int32_t> prefix261(long_p.begin(), long_p.begin() + 261);
        const auto want_from_261 = fresh_gen(*cu_plain, prefix261, 16, false);

        Session s(cu_plain.get());
        std::string err;
        if (!s.init(&err)) throw std::runtime_error(err);
        // Prefill to 260, keep checkpoint at 260 (prefix size 260 >= kSnapshotWorth), decode 4 tokens.
        auto first_out = run_gen(s, prefix261, 4, false, {260}, 260);
        check(same_tokens(first_out, std::vector<int32_t>(want_from_261.begin(), want_from_261.begin() + 4)),
              "SWA ring prefix261 initial run");

        // Advance to pos 420 (> 260 + 128 = 388) so every slot in the 128-slot SWA ring is overwritten.
        // Keep keep_checkpoint = -1 so the kept checkpoint at 260 stays protected!
        (void)run_gen(s, long_p, 4, false, {420}, -1);

        // Save checkpoints to SnapshotStore (includes kept pos 260 while live ring has wrapped to 424!).
        SnapshotStore store(64ull << 20, 64ull << 10);
        std::string why;
        const int saved = s.save_to(store, &why);
        check(saved >= 260, "SnapshotStore save_to after SWA ring wraparound (" + why + ")");

        // Restore checkpoint on the wrapped session and verify output from pos 261!
        auto after_restore = run_gen(s, prefix261, 16, false);
        check(same_tokens(after_restore, want_from_261), "checkpoint restore past SWA ring wraparound == fresh");

        // Load saved snapshot (pos 260) into a brand-new session and verify output from pos 261!
        Session s2(cu_plain.get());
        if (!s2.init(&err)) throw std::runtime_error(err);
        s2.set_store(&store);
        std::string how;
        const int kept = s2.prepare(prefix261, &how, &err);
        check(kept == 260 && how.find("snapshot") != std::string::npos,
              "fresh session prepared pos 260 from snapshot (" + how + ")");
        auto after_snap = run_gen(s2, prefix261, 16, false);
        check(same_tokens(after_snap, want_from_261), "snapshot_load after SWA ring wraparound == fresh");
      }

      // A rewind into a committed round after prefill passes appended past it: the passes move the position on and
      // not the round, so the round is still a rollback source of the recurrent state (ShortConv); the SWA ring (128
      // slots, window 8) holds the positions the row there reads only while the passes have not gone round it.
      // Within the ring, the rewind is taken and the rounds after it are those of the same rewind without the passes;
      // past it, the rewind is refused (0: the caller restores a checkpoint or prefills again), never a rollback that
      // reads slots overwritten since.
      {
        auto e = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32});
        const auto* api = e->api();
        const int vocab = e->model().desc().vocab;
        flyx_gen_params gp{};
        gp.top_p = 1.f;
        const int P = int(prompt.size());
        std::vector<int32_t> extra(240);
        for (int i = 0; i < 240; ++i) extra[size_t(i)] = 7 + (i * 19) % 230;
        // Prefill the prompt, three plain rounds (from positions P, P + 1 and P + 2), then `appended` tokens in passes
        // of at most 32 and of `appends` calls; then the rewind to P + 2 with another anchor, and 8 rounds.
        auto rewound = [&](int appended, int appends, int* rc) {
          Session s(e.get());
          std::string err;
          if (!s.init(&err)) throw std::runtime_error(err);
          flyx_round_result rr{};
          std::vector<int32_t> got;
          auto round = [&] {
            if (api->launch(e->round_plan(0), s.seq(), &gp) || api->read(s.seq(), &rr))
              throw std::runtime_error(api->last_error());
            got.push_back(rr.tokens[0]);
          };
          if (api->prefill(e->prefill_plan(), s.seq(), prompt.data(), P, &gp) || api->read(s.seq(), &rr))
            throw std::runtime_error(api->last_error());
          got.push_back(rr.tokens[0]);
          for (int i = 0; i < 3; ++i) round();
          for (int a = 0, done = 0; a < appends; ++a)
            for (const int end = appended * (a + 1) / appends; done < end;) {
              const int n = std::min(32, end - done);
              if (api->prefill(e->prefill_plan(), s.seq(), extra.data() + done, n, nullptr))
                throw std::runtime_error(api->last_error());
              done += n;
            }
          const int32_t anchor = (got[2] + 1) % vocab;
          *rc = api->rewind(s.seq(), P + 2, anchor);
          if (*rc != 1) return std::vector<int32_t>{};
          got.clear();
          for (int i = 0; i < 8; ++i) round();
          return got;
        };
        int rc_none = -1, rc_within = -1, rc_past = -1;
        const auto after_none = rewound(0, 1, &rc_none);
        const auto after_within = rewound(8, 1, &rc_within);
        (void)rewound(240, 2, &rc_past);
        check(rc_none == 1 && rc_within == 1 && after_none.size() == 8 && same_tokens(after_within, after_none),
              "a rewind into a round after 8 appended tokens (inside the SWA ring) is taken, its rounds those of the "
              "rewind without them");
        check(rc_past == 0,
              "a rewind into a round after prefill passes went round the SWA ring (240 tokens in two appends) is "
              "refused, not rolled back over overwritten slots (" + std::to_string(rc_past) + ")");

        // The same through sessions, from plain and DSpark rounds: a reply, a long continuation of the conversation
        // in two turns (prefill passes, no rounds of their own past the first token), then the conversation edited
        // back inside the reply: the state the session brings it to (a checkpoint and a prefill, the rewind refused)
        // continues as a fresh prefill of the edited prompt does.
        for (Engine* eng : {cu_plain.get(), cu_dspark.get()}) {
          const std::string tag = eng == cu_dspark.get() ? "DSpark" : "plain";
          Session s(eng);
          std::string err;
          if (!s.init(&err)) throw std::runtime_error(err);
          const auto reply = run_gen(s, prompt, 6, false);
          std::vector<int32_t> convo = prompt;
          convo.insert(convo.end(), reply.begin(), reply.end());
          for (int turn = 0; turn < 2; ++turn) {
            convo.insert(convo.end(), extra.begin() + 120 * turn, extra.begin() + 120 * (turn + 1));
            (void)run_gen(s, convo, 1, false, {int32_t(convo.size())});
          }
          std::vector<int32_t> edit = prompt;
          edit.insert(edit.end(), reply.begin(), reply.begin() + 3);
          edit.push_back((reply[3] + 1) % vocab);
          GenerateResult res;
          const auto got = run_gen(s, edit, 12, false, {}, -1, &res);
          check(same_tokens(got, fresh_gen(*cu_plain, edit, 12, false)),
                tag + ": the conversation edited back into a reply after 240 appended tokens == a fresh prefill (" +
                    std::to_string(res.usage.reused_tokens) + " of " + std::to_string(edit.size()) + " reused)");
        }
      }

      // Multi-sequence concurrent generation on one Engine (exercises launch_batch fallback + independent state).
      {
        std::vector<int32_t> prompt_b(16);
        for (int i = 0; i < 16; ++i) prompt_b[size_t(i)] = 30 + (i * 11) % 200;
        const auto want_a = cu_greedy;
        const auto want_b = fresh_gen(*cu_plain, prompt_b, 16, false);

        Session sa(cu_dspark.get()), sb(cu_dspark.get());
        std::string err;
        if (!sa.init(&err) || !sb.init(&err)) throw std::runtime_error("dual session init: " + err);
        std::vector<int32_t> out_a, out_b;
        GenerateRequest ra, rb;
        ra.prompt = prompt;
        ra.max_new_tokens = 24;
        ra.ignore_eos = true;
        rb.prompt = prompt_b;
        rb.max_new_tokens = 16;
        rb.ignore_eos = true;
        Generation ga(&sa, ra, [&](std::string_view, const int32_t* ids, int cnt) {
          out_a.insert(out_a.end(), ids, ids + cnt);
          return true;
        });
        Generation gb(&sb, rb, [&](std::string_view, const int32_t* ids, int cnt) {
          out_b.insert(out_b.end(), ids, ids + cnt);
          return true;
        });
        ga.set_batched(true);
        gb.set_batched(true);
        int batch_calls = 0;
        while (!ga.done() || !gb.done()) {
          if (!ga.done()) ga.advance(false);
          if (!gb.done()) gb.advance(false);
          // When either generation wants a round, dispatch via api()->launch_batch!
          std::vector<Generation*> ready;
          std::vector<Generation::NextRound> nr;
          for (Generation* g : {&ga, &gb}) {
            Generation::NextRound r;
            if (!g->done() && g->next_round(&r)) {
              ready.push_back(g);
              nr.push_back(r);
            }
          }
          if (!ready.empty()) {
            std::vector<flyx_plan*> plans(ready.size());
            std::vector<flyx_seq*> seqs(ready.size());
            std::vector<const flyx_gen_params*> gens(ready.size());
            for (size_t i = 0; i < ready.size(); ++i) {
              plans[i] = nr[i].plan;
              seqs[i] = ready[i]->session()->seq();
              gens[i] = &nr[i].g;
            }
            const int rc = cu_dspark->api()->launch_batch(plans.data(), seqs.data(), gens.data(), int32_t(ready.size()));
            if (rc != 0) throw std::runtime_error("launch_batch failed");
            for (size_t i = 0; i < ready.size(); ++i) ready[i]->launched(nr[i]);
            if (ready.size() == 2) ++batch_calls;
          } else {
            if (!ga.done()) ga.advance(true);
            if (!gb.done()) gb.advance(true);
          }
        }
        check(!ga.failed() && !gb.failed() && batch_calls > 0 && same_tokens(out_a, want_a) &&
                  same_tokens(out_b, want_b),
              "dual batched Inkling-2 + DSpark sessions via launch_batch (" + std::to_string(batch_calls) +
                  " paired batches) == fresh");
      }

      // ---- 4. Mixed quantization GGUF with BF16 and F32 sinks/taps, plus Q8 DSpark ----
      std::printf("Inkling-2 mixed-quant target (NVFP4 + MXFP4 + Q2_0 + Q8_0 + BF16/F32 sinks and taps)\n");
      auto ref_mix = load_engine(mixed_gguf.string(), "cpu_ref");
      auto cu_mix = load_engine(mixed_gguf.string(), "cuda", {.prefill_chunk = 32, .backend_opts = "head_cert=0"});
      auto cu_mix_unfused = load_engine(mixed_gguf.string(), "cuda",
                                       {.prefill_chunk = 32, .backend_opts = "fusion=0,head_cert=0"});
      auto cu_mix_dspark = load_engine(mixed_gguf.string(), "cuda",
                                       {.speculative = true,
                                        .prefill_chunk = 32,
                                        .drafter = "dspark",
                                        .drafter_files = {dspark_q8.string()}});
      const auto ref_mix_tok = fresh_gen(*ref_mix, prompt, 16, false);
      const int tf_agree = teacher_forced_matches(*cu_mix, prompt, ref_mix_tok);
      check(tf_agree >= 15, "mixed-quant cuda teacher-forced vs cpu_ref (" + std::to_string(tf_agree) + "/16)");
      std::vector<int32_t> cu_mix_greedy;
      {
        Session fused(cu_mix.get()), generic(cu_mix_unfused.get());
        std::string err;
        if (!fused.init(&err) || !generic.init(&err)) throw std::runtime_error("session init: " + err);
        cu_mix_greedy = run_gen(fused, prompt, 24);
        check(same_tokens(run_gen(generic, prompt, 24), cu_mix_greedy), "mixed-quant Gate 2: fusion=0 == fusion=1");
        std::vector<float> a(spec.vocab_unpadded), b(spec.vocab_unpadded);
        check(cu_mix->api()->logits(fused.seq(), 0, a.data()) == 0 &&
                  cu_mix_unfused->api()->logits(generic.seq(), 0, b.data()) == 0 &&
                  std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
              "mixed-quant Gate 2: identical final round logits with BF16 and F32 sinks and taps");
      }
      const auto cu_mix_sampled = fresh_gen(*cu_mix, prompt, 24, true);
      check(same_tokens(fresh_gen(*cu_mix_dspark, prompt, 24, false), cu_mix_greedy),
            "mixed-quant Gate 4: Q8 DSpark greedy == cuda plain");
      check(same_tokens(fresh_gen(*cu_mix_dspark, prompt, 24, true), cu_mix_sampled),
            "mixed-quant Gate 4: Q8 DSpark sampled+penalties == cuda plain");

      // Also verify exact activations mode (--activations exact: a16h on MXFP4/Q2_0/Q8_0) on CUDA vs cpu_ref + Gate 4.
      auto ref_exact = load_engine(mixed_gguf.string(), "cpu_ref", {.activations = "exact"});
      auto cu_exact = load_engine(mixed_gguf.string(), "cuda", {.prefill_chunk = 32, .activations = "exact"});
      auto cu_exact_dspark = load_engine(mixed_gguf.string(), "cuda",
                                         {.speculative = true,
                                          .prefill_chunk = 32,
                                          .drafter = "dspark",
                                          .drafter_files = {dspark_q8.string()},
                                          .activations = "exact"});
      const auto ref_exact_tok = fresh_gen(*ref_exact, prompt, 16, false);
      const int tf_exact = teacher_forced_matches(*cu_exact, prompt, ref_exact_tok);
      check(tf_exact >= 15,
            "mixed-quant --activations exact (a16h) teacher-forced vs cpu_ref (" + std::to_string(tf_exact) + "/16)");
      const auto cu_exact_sampled = fresh_gen(*cu_exact, prompt, 24, true);
      check(same_tokens(fresh_gen(*cu_exact_dspark, prompt, 24, true), cu_exact_sampled),
            "mixed-quant --activations exact Gate 4: DSpark sampled+penalties == cuda plain");

      // ---- 5. FP8 KV, 2/3-bit StrataFlux, auto switch, host-RAM windowed prefill, and DSpark KV rings ----
      std::printf("Inkling-2 FP8 KV, StrataFlux (2/3-bit & auto), windowed prefill, and DSpark KV rings\n");
      {
        const int64_t ram = 256ll << 20;
        const std::string narrow = "strata_denom=8,strata_least=1";
        // 5a. FP8 KV exact: plain == DSpark (greedy & sampled+penalties).
        auto fp8_plain = load_engine(bf16_gguf.string(), "cuda", {.prefill_chunk = 32, .kv = FLYX_KV_FP8});
        auto fp8_dspark = load_engine(bf16_gguf.string(), "cuda",
                                      {.speculative = true,
                                       .prefill_chunk = 32,
                                       .kv = FLYX_KV_FP8,
                                       .drafter = "dspark",
                                       .drafter_files = {dspark_bf16.string()}});
        const auto fp8_greedy = fresh_gen(*fp8_plain, prompt, 24, false);
        const auto fp8_sampled = fresh_gen(*fp8_plain, prompt, 24, true);
        check(same_tokens(fresh_gen(*fp8_dspark, prompt, 24, false), fp8_greedy),
              "FP8 KV Gate 4: DSpark greedy == plain");
        check(same_tokens(fresh_gen(*fp8_dspark, prompt, 24, true), fp8_sampled),
              "FP8 KV Gate 4: DSpark sampled+penalties == plain");

        // 5b. StrataFlux with every pair refined reproduces exact binary16 tokens identically.
        auto strata_all = load_engine(bf16_gguf.string(), "cuda",
                                      {.prefill_chunk = 32,
                                       .kv = FLYX_KV_16,
                                       .kv_method = "strata",
                                       .backend_opts = "strata_least=100000"});
        check(same_tokens(fresh_gen(*strata_all, prompt, 24, false), cu_greedy),
              "RelAttn strata:2 all pairs refined == exact binary16");

        // 5c. 2-bit & 3-bit StrataFlux (binary16 and FP8): plain == DSpark, host KV (+ hot cache), host copy streaming.
        for (const char* method : {"strata", "strata:3"}) {
          for (int kv_fmt : {FLYX_KV_16, FLYX_KV_FP8}) {
            const std::string tag = std::string(method) + (kv_fmt == FLYX_KV_FP8 ? " fp8" : " f16");
            auto st_plain = load_engine(bf16_gguf.string(), "cuda",
                                        {.prefill_chunk = 32, .kv = kv_fmt, .kv_method = method, .backend_opts = narrow});
            auto st_dspark = load_engine(bf16_gguf.string(), "cuda",
                                         {.speculative = true,
                                          .prefill_chunk = 32,
                                          .kv = kv_fmt,
                                          .kv_method = method,
                                          .drafter = "dspark",
                                          .drafter_files = {dspark_bf16.string()},
                                          .backend_opts = narrow});
            auto st_host_kv = load_engine(bf16_gguf.string(), "cuda",
                                          {.speculative = true,
                                           .prefill_chunk = 32,
                                           .kv = kv_fmt,
                                           .kv_gpu = 0,
                                           .kv_ram = ram,
                                           .kv_method = method,
                                           .drafter = "dspark",
                                           .drafter_files = {dspark_bf16.string()},
                                           .backend_opts = narrow});
            auto st_host_copy = load_engine(bf16_gguf.string(), "cuda",
                                            {.speculative = true,
                                             .prefill_chunk = 32,
                                             .kv = kv_fmt,
                                             .kv_gpu = 0,
                                             .kv_copy_gpu = 0,
                                             .kv_ram = ram,
                                             .kv_method = method,
                                             .drafter = "dspark",
                                             .drafter_files = {dspark_bf16.string()},
                                             .backend_opts = narrow});
            const auto t_plain = fresh_gen(*st_plain, prompt, 24, false);
            check(same_tokens(fresh_gen(*st_dspark, prompt, 24, false), t_plain),
                  tag + " Gate 4: DSpark == plain");
            check(same_tokens(fresh_gen(*st_host_kv, prompt, 24, false), t_plain),
                  tag + " host KV (+ hot cache) + DSpark == GPU plain");
            check(same_tokens(fresh_gen(*st_host_copy, prompt, 24, false), t_plain),
                  tag + " host KV + host copy streaming + DSpark == GPU plain");
            // Rounds whose rows pass 128 and 256 keys, where a row's first split is shorter than the row's before it
            // (attn_split_range), so that the round's last row is not the one whose split ends last: DSpark's rounds
            // still read every key of every row.  A prompt of 100 tokens, 180 decoded after it.
            std::vector<int32_t> longer;
            for (int i = 0; i < 100; ++i) longer.push_back(10 + (i * 11) % 240);
            const auto t_across = fresh_gen(*st_plain, longer, 180, false);
            check(same_tokens(fresh_gen(*st_dspark, longer, 180, false), t_across) &&
                      same_tokens(fresh_gen(*st_host_kv, longer, 180, false), t_across),
                  tag + " rounds across 128 and 256 keys: DSpark == plain, host KV too");
          }
        }

        // 5d. auto:2 across strata_from switch (default Qwen3.8/Inkling-2 path): exact before switch, DSpark == plain across.
        const int sw = int(prompt.size()) + 10;
        const std::string auto_opts = narrow + ",strata_from=" + std::to_string(sw);
        for (int kv_fmt : {FLYX_KV_16, FLYX_KV_FP8}) {
          const std::string tag = kv_fmt == FLYX_KV_FP8 ? "auto:2 fp8" : "auto:2 f16";
          auto au_plain = load_engine(bf16_gguf.string(), "cuda",
                                      {.prefill_chunk = 32, .kv = kv_fmt, .kv_method = "auto", .backend_opts = auto_opts});
          auto au_dspark = load_engine(bf16_gguf.string(), "cuda",
                                       {.speculative = true,
                                        .prefill_chunk = 32,
                                        .kv = kv_fmt,
                                        .kv_gpu = 0,
                                        .kv_ram = ram,
                                        .kv_method = "auto",
                                        .drafter = "dspark",
                                        .drafter_files = {dspark_bf16.string()},
                                        .backend_opts = auto_opts});
          const auto t_au = fresh_gen(*au_plain, prompt, 24, false);
          const auto& t_exact = kv_fmt == FLYX_KV_FP8 ? fp8_greedy : cu_greedy;
          check(same_tokens(std::vector<int32_t>(t_au.begin(), t_au.begin() + 10),
                            std::vector<int32_t>(t_exact.begin(), t_exact.begin() + 10)),
                tag + " before strata_from switch == exact");
          check(same_tokens(fresh_gen(*au_dspark, prompt, 24, false), t_au),
                tag + " across strata_from switch (host KV + DSpark) == plain");
        }

        // 5e. Windowed prefill staging (RelWindow) + DSpark Full-layer bounded ring (mtp_window=16, max_ctx=512 > 8*16).
        std::vector<int32_t> long_prompt(180);
        for (int i = 0; i < 180; ++i) long_prompt[size_t(i)] = 12 + (i * 17) % 230;
        auto gpu_strata = load_engine(bf16_gguf.string(), "cuda",
                                      {.prefill_chunk = 32,
                                       .kv = FLYX_KV_FP8,
                                       .kv_method = "strata",
                                       .backend_opts = narrow});
        auto win_dspark = load_engine(bf16_gguf.string(), "cuda",
                                      {.speculative = true,
                                       .max_ctx = 512,
                                       .prefill_chunk = 32,
                                       .kv = FLYX_KV_FP8,
                                       .kv_gpu = 0,
                                       .kv_ram = ram,
                                       .kv_method = "strata",
                                       .drafter = "dspark",
                                       .drafter_files = {dspark_bf16.string()},
                                       .backend_opts = narrow + ",stage_window=64,mtp_window=16"});
        const auto want_long = fresh_gen(*gpu_strata, long_prompt, 24, false);
        check(same_tokens(fresh_gen(*win_dspark, long_prompt, 24, false), want_long),
              "RelAttn windowed prefill (stage_window=64) + DSpark Full ring (mtp_window=16) == GPU strata");
      }

      // ---- 6. DSpark's two-stage draft head: 1-bit copies of the head and of the Markov head pick the candidates ----
      // The tiny models above are too small for the copies (a hidden width and a Markov rank of 128s, 1,025 tokens
      // or more), so their drafts read both heads whole; these take the two-stage path, from bf16 and from packed
      // weights, and draft_head=full reads the heads whole again.
      std::printf("Inkling-2 DSpark two-stage draft head (1-bit copies of the head and the Markov head)\n");
      {
        TinySpec wide;
        wide.H = 128;
        wide.vocab_padded = 1344;
        wide.vocab_unpadded = 1280;
        const int rank = 128;
        for (const bool packed : {false, true}) {
          const std::string tag = packed ? "Q8_0 heads" : "bf16 heads";
          const auto target = dir / (packed ? "wide_inkling2_mixed.gguf" : "wide_inkling2_bf16.gguf");
          const auto drafter = dir / (packed ? "dspark-wide-q8.gguf" : "dspark-wide-bf16.gguf");
          write_tiny_inkling2(target, wide, packed, packed ? 6006 : 5005);
          write_tiny_dspark(drafter, wide, packed, packed ? 8008 : 7007, rank);
          auto dspark = [&](const std::string& opts) {
            return load_engine(target.string(), "cuda",
                               {.speculative = true,
                                .prefill_chunk = 32,
                                .drafter = "dspark",
                                .drafter_files = {drafter.string()},
                                .backend_opts = opts});
          };
          auto plain = load_engine(target.string(), "cuda", {.prefill_chunk = 32});
          auto pick = dspark(""), pick_unfused = dspark("fusion=0"), pick_nographs = dspark("graphs=0");
          auto full = dspark("draft_head=full");
          const auto greedy = fresh_gen(*plain, prompt, 40, false);
          const auto sampled = fresh_gen(*plain, prompt, 40, true);
          check(same_tokens(fresh_gen(*pick, prompt, 40, false), greedy), tag + ": two-stage DSpark greedy == plain");
          check(same_tokens(fresh_gen(*pick, prompt, 40, true), sampled),
                tag + ": two-stage DSpark sampled+penalties == plain");
          check(same_tokens(fresh_gen(*pick_unfused, prompt, 40, false), greedy),
                tag + ": two-stage DSpark fusion=0 == plain");
          check(same_tokens(fresh_gen(*pick_nographs, prompt, 40, true), sampled),
                tag + ": two-stage DSpark graphs=0 sampled+penalties == plain");
          check(same_tokens(fresh_gen(*full, prompt, 40, false), greedy),
                tag + ": draft_head=full DSpark greedy == plain");
          check(same_tokens(fresh_gen(*full, prompt, 40, true), sampled),
                tag + ": draft_head=full DSpark sampled+penalties == plain");
          // What the default holds for it: the two 1-bit copies, where draft_head=full holds an NVFP4 copy of a bf16
          // head (larger than both) and no copy of a packed one.
          const int64_t held = int64_t(weight_bytes(*pick)) - int64_t(weight_bytes(*full));
          const int64_t rows = (wide.vocab_unpadded + 255) / 256 * 256;
          const int64_t copies = rows * (wide.H + rank) / 8 + rows * (wide.H + rank) / 64 * 2;  // signs, group scales
          const int64_t nvfp4_head = int64_t(wide.vocab_unpadded) * (wide.H / 2 + wide.H / 16 + 4);
          check(held == copies - (packed ? 0 : nvfp4_head),
                tag + ": the default holds the head's and the Markov head's 1-bit copies (" + std::to_string(held) +
                    " bytes against draft_head=full)");
        }
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL exception: %s\n", e.what());
    return 1;
  }
  return flyx::test::finish("inkling2_test (" + backend + ")");
}
