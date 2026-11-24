// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// End-to-end verification of Inkling-2 (`inkling2_loop`) and its DSpark companion drafter (`dspark`)
// on self-contained synthetic GGUF models (no PyTorch/assets dependency):
//   * Looped core (prelude + 3-pass core + postlude), AttnRes, cross-pass loop gates, ShortConv (static + dynamic),
//     relative-position-bias attention (SWA ring + Global log-scaled), padded->unpadded vocab + softcap,
//     sparse MoE (SiTU, 3D NVFP4 with per-expert .scale, MXFP4, Q2_0, Q8_0, BF16, mixed-dtype F32 sink_proj).
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

using namespace flyx;
using json = nlohmann::json;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& msg) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", msg.c_str());
  if (!ok) ++g_fail;
}

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

std::vector<uint8_t> gen_f32(std::mt19937& rng, size_t n, float mean, float sd) {
  std::normal_distribution<float> d(mean, sd);
  std::vector<uint8_t> out(n * 4);
  for (size_t i = 0; i < n; ++i) {
    const float v = d(rng);
    std::memcpy(out.data() + i * 4, &v, 4);
  }
  return out;
}

std::vector<uint8_t> gen_bf16(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.f, sd);
  std::vector<uint8_t> out(n * 2);
  for (size_t i = 0; i < n; ++i) {
    const uint16_t h = f32_to_bf16(d(rng));
    std::memcpy(out.data() + i * 2, &h, 2);
  }
  return out;
}

std::vector<uint8_t> gen_q8_0(std::mt19937& rng, size_t n, float sd) {
  const size_t nb = n / 32;
  std::normal_distribution<float> d(0.f, sd);
  std::vector<uint8_t> out(nb * 34);
  for (size_t b = 0; b < nb; ++b) {
    float vals[32], amax = 0.f;
    for (int i = 0; i < 32; ++i) {
      vals[i] = d(rng);
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
  std::uniform_int_distribution<int> ed(120, 124);
  std::uniform_int_distribution<int> qd(0, 255);
  std::vector<uint8_t> out(nb * 17);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 17;
    dst[0] = uint8_t(ed(rng));
    for (int i = 0; i < 16; ++i) dst[1 + i] = uint8_t(qd(rng));
  }
  return out;
}

std::vector<uint8_t> gen_q2_0(std::mt19937& rng, size_t n) {
  const size_t nb = n / 64;
  std::uniform_real_distribution<float> sd(0.01f, 0.08f);
  std::uniform_int_distribution<int> qd(0, 255);
  std::vector<uint8_t> out(nb * 18);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 18;
    const uint16_t dh = f32_to_f16(sd(rng));
    std::memcpy(dst, &dh, 2);
    for (int i = 0; i < 16; ++i) dst[2 + i] = uint8_t(qd(rng));
  }
  return out;
}

std::vector<uint8_t> gen_nvfp4(std::mt19937& rng, size_t n) {
  const size_t nb = n / 64;
  std::uniform_int_distribution<int> sd(0x18, 0x38);  // valid positive E4M3 scales (never 0x7F/0xFF)
  std::uniform_int_distribution<int> qd(0, 255);
  std::vector<uint8_t> out(nb * 36);
  for (size_t b = 0; b < nb; ++b) {
    uint8_t* dst = out.data() + b * 36;
    for (int g = 0; g < 4; ++g) dst[g] = uint8_t(sd(rng));
    for (int i = 0; i < 32; ++i) dst[4 + i] = uint8_t(qd(rng));
  }
  return out;
}

struct TinySpec {
  int H = 64, nh = 4, nkv = 2, hd = 32, d_rel = 8, rel_extent = 16, window = 8, conv = 4;
  int log_floor = 8, attn_res_block = 2, n_phys = 6, core_start = 2, core_end = 4, n_passes = 3, gate_rank = 16;
  int n_experts = 4, top_k = 2, inter = 64, vocab_padded = 320, vocab_unpadded = 272;
  float eps = 1e-6f, qk_clip = 6.f, rel_clip = 4.f, log_alpha = 0.2f;
  float route_scale = 1.5f, situ_beta = 6.f, situ_linear_beta = 8.f, softcap = 30.f;
  std::vector<bool> is_swa = {true, false, true, false, true, false};
};

// Minimal Inkling-2 chat template exercising tojson(sort_keys=true), thinking, text, and tool invocation tags.
const char* kInkling2Template = R"jinja(
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

void write_tiny_inkling2(const std::filesystem::path& path, const TinySpec& c, bool mixed_quant, uint32_t seed_val) {
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
  w.add_u32(arch + ".attention.head_count", uint32_t(c.nh));
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

  auto add_mat = [&](const std::string& name, int N, int K, uint32_t dt, float sd = 0.08f) {
    const size_t n = size_t(N) * K;
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
    if (swa) {
      add_vec(b + "attn_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
    } else {
      add_vec(b + "attn_q_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_vec(b + "attn_k_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_vec(b + "attn_v_inp_norm.weight", {uint64_t(c.H)}, 1.f, 0.1f);
      add_mat(b + "attn_k_conv_proj.weight", c.conv * c.nkv, c.H, kGgmlF32, 0.05f);
      add_mat(b + "attn_v_conv_proj.weight", c.conv * c.nkv, c.H, kGgmlF32, 0.05f);
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
    // Keep sink_proj F32 even in mixed_quant (matches real Inkling-2 GGUFs where sink_proj is F32!).
    add_mat(b + "attn_sink_proj.weight", c.nh, c.H, kGgmlF32, 0.1f);
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
    add_3d(b + "ffn_gate_up_exps.weight", c.n_experts, 2 * c.inter, c.H, exp_dt);
    add_3d(b + "ffn_down_exps.weight", c.n_experts, c.H, c.inter, exp_dt);
    if (exp_dt == kGgmlNVFP4) {
      // Distinct positive per-expert F32 tensor scales!
      std::vector<uint8_t> sc_gu(size_t(c.n_experts) * 4), sc_dn(size_t(c.n_experts) * 4);
      for (int e = 0; e < c.n_experts; ++e) {
        const float s1 = 0.04f + 0.015f * float(e);
        const float s2 = 0.03f + 0.012f * float(e);
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
                       int rank = 32) {
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
  w.add_u32(arch + ".attention.head_count", uint32_t(c.nh));
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

struct LoadOpts {
  bool speculative = false;
  int max_ctx = 512;
  int prefill_chunk = 32;
  int width = 0;
  int kv = FLYX_KV_16;
  int kv_gpu = -1;
  int kv_copy_gpu = -1;
  int64_t kv_ram = 0;
  std::string kv_method = "exact";
  std::string drafter = "none";
  std::vector<std::string> drafter_files{};
  std::string backend_opts{};
  std::string activations{};
};

std::unique_ptr<Engine> load_engine(const std::string& path, const std::string& backend, const LoadOpts& o = {}) {
  EngineOptions eo;
  eo.backend = backend;
  eo.backend_options = o.backend_opts;
  eo.max_ctx = o.max_ctx;
  eo.prefill_chunk = o.prefill_chunk;
  eo.lookup = false;
  eo.speculative = o.speculative;
  eo.drafter = o.speculative ? o.drafter : "none";
  eo.width = o.speculative ? (o.width > 0 ? o.width : 7) : 0;
  eo.drafter_files = o.drafter_files;
  eo.activations = o.activations;
  eo.kv = o.kv;
  eo.kv_gpu = o.kv_gpu;
  eo.kv_copy_gpu = o.kv_copy_gpu;
  eo.kv_ram = o.kv_ram;
  eo.kv_method = o.kv_method;
  auto e = std::make_unique<Engine>();
  std::string err;
  if (!e->load(path, eo, &err)) throw std::runtime_error("load(" + backend + "): " + err);
  return e;
}

std::vector<int32_t> run_gen(Session& s, const std::vector<int32_t>& prompt, int n, bool sampled = false,
                             const std::vector<int32_t>& ckps = {}, int32_t keep_ckpt = -1) {
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
  return out;
}

std::vector<int32_t> fresh_gen(Engine& e, const std::vector<int32_t>& prompt, int n, bool sampled = false) {
  Session s(&e);
  std::string err;
  if (!s.init(&err)) throw std::runtime_error("session init: " + err);
  return run_gen(s, prompt, n, sampled);
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

  // A reply as the model writes it, every marker its control token: the grammar takes it and lets it end there.  A
  // grammar that knows no marker takes none of them (they are special tokens), which would leave a model to spell
  // its sections out in ordinary tokens.  The stop token as in the real model's file, <|content_model_end_sampling|>.
  {
    std::vector<int32_t> markers;
    for (const std::string& text : e->chat().reply_tokens()) markers.push_back(tok.added_id(text));
    // How many of the reply's tokens the grammar takes, and whether it may end after the last.
    auto taken = [&](const std::string& tag, const std::string& reply, const std::vector<int32_t>& known,
                     bool* ends) {
      Grammar g;
      g.init(tok, tok.n_tokens(), {tok.added_id("<|content_model_end_sampling|>")}, known);
      std::string why;
      auto m = g.matcher(tag, &why);
      const std::vector<int32_t> ids = tok.encode(reply);
      size_t n = 0;
      while (m && n < ids.size() && m->accept(ids[n])) ++n;
      *ends = m && n == ids.size() && m->can_end();
      return n;
    };
    auto admits = [&](const std::string& tag, const std::string& reply, const std::vector<int32_t>& known) {
      bool ends = false;
      taken(tag, reply, known, &ends);
      return ends;
    };
    const std::string thought = "<|message_model|><|content_thinking|>plan<|end_message|>";
    const std::string call = "<|message_model|>lookup<|content_invoke_tool_json|>"
                             "{\"name\":\"lookup\",\"args\":{\"a\":2}}<|end_message|>";
    const std::vector<int32_t> written = tok.encode(thought + call);
    const int64_t as_tokens = std::count_if(written.begin(), written.end(), [&](int32_t id) {
      return std::find(markers.begin(), markers.end(), id) != markers.end();
    });
    check(markers.size() == 5 && std::none_of(markers.begin(), markers.end(), [](int32_t id) { return id < 0; }) &&
              as_tokens == 6,
          "the format names its reply markers, each a control token of the model");
    check(admits(grm, thought + call, markers), "the grammar takes a reply written with its control tokens");
    check(!admits(grm, thought + call, {}), "a grammar that knows no marker takes none of them");
    // A marker inside a section's text is not text: the section ends at its own <|end_message|>.
    const std::string auto_grm = e->chat().reply_grammar(tools, ReplyRules{}, rc);
    bool ends = false;
    const std::string nested = "<|message_model|><|content_text|>a<|content_text|>b<|end_message|>";
    const size_t n = taken(auto_grm, nested, markers, &ends);
    check(n == 3 && !ends, "a section's text holds no marker (" + std::to_string(n) + " tokens taken)");
  }

  // Round-trip tokenizer encode/decode on control tokens and text.
  auto ids = tok.encode("<|message_user|><|content_text|>hello<|end_message|>");
  check(!ids.empty() && ids.front() == 264 && ids.back() == 269, "inkling2/o200k tokenizer encodes control tokens");
}

}  // namespace

int main(int argc, char** argv) {
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
    write_tiny_inkling2(bf16_gguf, spec, false, 1001);
    write_tiny_inkling2(mixed_gguf, spec, true, 2002);
    write_tiny_dspark(dspark_bf16, spec, false, 3003);
    write_tiny_dspark(dspark_q8, spec, true, 4004);

    test_chat_and_format(bf16_gguf.string());

    std::vector<int32_t> prompt;
    for (int i = 0; i < 20; ++i) prompt.push_back(10 + (i * 7) % 240);

    // ---- 1. Float64 cpu_ref baseline + cpu_ref DSpark ---------------------------------------
    std::printf("Inkling-2 bf16+nvfp4 target (%s vs cpu_ref)\n", backend.c_str());
    auto ref_plain = load_engine(bf16_gguf.string(), "cpu_ref");
    auto ref_dspark = load_engine(bf16_gguf.string(), "cpu_ref",
                                  {.speculative = true, .drafter = "dspark", .drafter_files = {dspark_bf16.string()}});
    check(std::string(ref_dspark->drafter()) == "dspark", "cpu_ref loaded DSpark companion drafter");
    const auto ref_greedy = fresh_gen(*ref_plain, prompt, 24, false);
    const auto ref_sampled = fresh_gen(*ref_plain, prompt, 24, true);
    check(same_tokens(fresh_gen(*ref_dspark, prompt, 24, false), ref_greedy),
          "cpu_ref DSpark greedy == cpu_ref plain");
    check(same_tokens(fresh_gen(*ref_dspark, prompt, 24, true), ref_sampled),
          "cpu_ref DSpark sampled+penalties == cpu_ref plain");

    if (backend == "cuda") {
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

      // ---- 4. Mixed quantization GGUF (NVFP4 + MXFP4 + Q2_0 + Q8_0 + F32 sink_proj + Q8 DSpark) ----
      std::printf("Inkling-2 mixed-quant target (NVFP4 + MXFP4 + Q2_0 + Q8_0 + F32 sink_proj)\n");
      auto ref_mix = load_engine(mixed_gguf.string(), "cpu_ref");
      auto cu_mix = load_engine(mixed_gguf.string(), "cuda", {.prefill_chunk = 32});
      auto cu_mix_unfused = load_engine(mixed_gguf.string(), "cuda", {.prefill_chunk = 32, .backend_opts = "fusion=0"});
      auto cu_mix_dspark = load_engine(mixed_gguf.string(), "cuda",
                                       {.speculative = true,
                                        .prefill_chunk = 32,
                                        .drafter = "dspark",
                                        .drafter_files = {dspark_q8.string()}});
      const auto ref_mix_tok = fresh_gen(*ref_mix, prompt, 16, false);
      const int tf_agree = teacher_forced_matches(*cu_mix, prompt, ref_mix_tok);
      check(tf_agree >= 15, "mixed-quant cuda teacher-forced vs cpu_ref (" + std::to_string(tf_agree) + "/16)");
      const auto cu_mix_greedy = fresh_gen(*cu_mix, prompt, 24, false);
      const auto cu_mix_sampled = fresh_gen(*cu_mix, prompt, 24, true);
      check(same_tokens(fresh_gen(*cu_mix_unfused, prompt, 24, false), cu_mix_greedy),
            "mixed-quant Gate 2: fusion=0 == fusion=1");
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
  std::printf("inkling2_test (%s): %s (%d failures)\n", backend.c_str(), g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}
