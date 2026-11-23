// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A GGUF file as an architecture's model reads it: the
// reader (core/modelfile) hands over the file's metadata and its tensors
// under their GGUF names; the keys of an architecture and what its GGUF
// layout means are read here and in models/<arch>.  A key that is missing or
// of the wrong kind throws; open() turns that into "bad model file".
#pragma once

#include <climits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "core/modelfile/modelfile.h"

namespace flyx::models {

class GgufMeta {
 public:
  static bool is(const ModelFile& file) { return is_header(file.header()); }
  static bool is_header(const nlohmann::json& h) {
    return h.is_object() && h.contains("source") && h["source"].value("format", "") == "gguf" && h.contains("gguf") &&
           h.contains("arch");
  }
  // Keys are given without the architecture's prefix ("attention.head_count").
  explicit GgufMeta(const ModelFile& file) : meta_(file.header().at("gguf")), prefix_(file.arch() + ".") {}
  // A header of a GGUF file (ModelFile::header(), or one of its "companions").
  explicit GgufMeta(const nlohmann::json& header)
      : meta_(header.at("gguf")), prefix_(header.at("arch").get<std::string>() + ".") {}

  bool has(const std::string& key) const { return meta_.contains(prefix_ + key); }
  // An integer of at least `minimum`.
  int count(const std::string& key, int minimum = 1) const {
    const auto& v = at(key);
    if (!v.is_number_integer() || (v.is_number_unsigned() && v.get<uint64_t>() > uint64_t(INT_MAX)))
      throw std::runtime_error("GGUF " + prefix_ + key + ": not an integer in range");
    const int64_t n = v.get<int64_t>();
    if (n < minimum || n > INT_MAX) throw std::runtime_error("GGUF " + prefix_ + key + ": out of range");
    return int(n);
  }
  int count_or(const std::string& key, int fallback, int minimum = 1) const {
    return has(key) ? count(key, minimum) : fallback;
  }
  // A positive number.
  double real_or(const std::string& key, double fallback) const {
    if (!has(key)) return fallback;
    const auto& v = at(key);
    if (!v.is_number() || !(v.get<double>() > 0)) throw std::runtime_error("GGUF " + prefix_ + key + ": not a positive number");
    return v.get<double>();
  }
  std::string text_or(const std::string& key, const std::string& fallback) const {
    return has(key) && at(key).is_string() ? at(key).get<std::string>() : fallback;
  }
  const nlohmann::json& at(const std::string& key) const {
    if (!has(key)) throw std::runtime_error("GGUF without " + prefix_ + key);
    return meta_.at(prefix_ + key);
  }

  // What the dense decoders here do not implement: scaled rotary, sliding
  // windows, another activation, scaled embeddings or logits, experts.
  void expect_plain_decoder() const {
    if (text_or("rope.scaling.type", "none") != "none" || real_or("rope.scaling.factor", 1) != 1)
      throw std::runtime_error("GGUF rotary scaling is not supported");
    if (count_or("attention.sliding_window", 0, 0) != 0) throw std::runtime_error("GGUF sliding-window attention is not supported");
    if (text_or("hidden_activation", "silu") != "silu") throw std::runtime_error("GGUF activation is not SiLU");
    for (const char* key : {"embedding_scale", "residual_scale", "logit_scale"})
      if (real_or(key, 1) != 1) throw std::runtime_error(std::string("GGUF ") + key + " is not supported");
    if (count_or("expert_count", 0, 0) != 0) throw std::runtime_error("GGUF mixture-of-experts models are not supported");
  }

 private:
  const nlohmann::json& meta_;
  std::string prefix_;
};

// Common dense-decoder metadata shared by Qwen3 and Qwen3.5 GGUF files.
struct GgufDecoderBase {
  int hidden = 0, blocks = 0, vocab = 0, inter = 0, n_heads = 0, n_kv = 0, head_dim = 0, rot_dim = 0;
  double theta = 0, eps = 0;
  int trained_ctx = 0;
};

inline GgufDecoderBase read_gguf_decoder_base(const ModelFile& file, bool allow_partial_rotary) {
  const GgufMeta m(file);
  m.expect_plain_decoder();
  GgufDecoderBase f{};
  const auto* embed = file.tensor("token_embd.weight");
  if (!embed || embed->ndim != 2) throw std::runtime_error("GGUF without a token_embd.weight matrix");
  f.vocab = int(embed->shape[0]);
  f.hidden = m.count("embedding_length");
  f.blocks = m.count("block_count");
  f.inter = m.count("feed_forward_length");
  f.n_heads = m.count("attention.head_count");
  f.n_kv = m.count("attention.head_count_kv");
  if (!m.has("attention.key_length") && f.hidden % f.n_heads)
    throw std::runtime_error("GGUF hidden size is not a whole number of heads");
  f.head_dim = m.count_or("attention.key_length", f.hidden / f.n_heads);
  if (m.count_or("attention.value_length", f.head_dim) != f.head_dim)
    throw std::runtime_error("GGUF attention key and value lengths differ");
  f.rot_dim = m.count_or("rope.dimension_count", f.head_dim);
  if (!allow_partial_rotary && f.rot_dim != f.head_dim)
    throw std::runtime_error("GGUF rotary dimension is not the head's");
  f.theta = m.real_or("rope.freq_base", 10000.0);
  f.eps = m.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  f.trained_ctx = m.count_or("context_length", 0, 0);
  return f;
}

// Top-level and standard dense-layer GGUF tensor names (from python/convert's canonical names).
// `post_norm_name` varies by architecture ("ffn_norm.weight" in Qwen3 vs "post_attention_norm.weight" in Qwen3.5).
inline const char* gguf_top_name(std::string_view name) {
  if (name == "target.embed") return "token_embd.weight";
  if (name == "target.lm_head") return "output.weight";
  if (name == "target.norm") return "output_norm.weight";
  return nullptr;
}

inline const char* gguf_dense_layer_name(std::string_view suffix, const char* post_norm_name) {
  if (suffix == "in_norm") return "attn_norm.weight";
  if (suffix == "post_norm") return post_norm_name;
  if (suffix == "mlp.gate") return "ffn_gate.weight";
  if (suffix == "mlp.up") return "ffn_up.weight";
  if (suffix == "mlp.down") return "ffn_down.weight";
  if (suffix == "attn.q") return "attn_q.weight";
  if (suffix == "attn.k") return "attn_k.weight";
  if (suffix == "attn.v") return "attn_v.weight";
  if (suffix == "attn.o") return "attn_output.weight";
  if (suffix == "attn.q_norm") return "attn_q_norm.weight";
  if (suffix == "attn.k_norm") return "attn_k_norm.weight";
  return nullptr;
}

}  // namespace flyx::models
