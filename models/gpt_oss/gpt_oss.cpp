// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// gpt-oss from llama.cpp's GGUF layout (arch "gpt-oss"; convert_hf_to_gguf.py's GptOssModel).  What its file does
// not say, the architecture does (OpenAI's gpt_oss reference, Hugging Face's GptOssForCausalLM): the experts'
// activation's alpha 1.702 and limit 7, the routing (softmax over the k best logits), the sliding layers (the even
// ones), YaRN's ramp not rounded to whole dims ("truncate": false), and no q/k norms.
#include "models/gpt_oss/gpt_oss.h"

#include <climits>
#include <cmath>
#include <stdexcept>

#include "models/gguf.h"

namespace flyx::models::gpt_oss {
namespace {

using nlohmann::json;

constexpr float kActAlpha = 1.702f, kActLimit = 7.0f;  // gpt_oss/torch/model.py swiglu(alpha, limit)

struct Config {
  int hidden = 0, layers = 0, vocab = 0, n_heads = 0, n_kv = 0, head_dim = 0;
  int n_experts = 0, top_k = 0, inter = 0, window = 0;
  double theta = 0, eps = 0;
  flyx_rope_scaling rope{};
  int trained_ctx = 0;
};

Config read_config(const ModelFile& file) {
  if (!GgufMeta::is(file)) throw std::runtime_error("gpt-oss is read from its GGUF file");
  const GgufMeta m(file);
  Config c;
  const auto* embed = file.tensor("token_embd.weight");
  if (!embed || embed->ndim != 2) throw std::runtime_error("GGUF without a token_embd.weight matrix");
  c.vocab = int(embed->shape[0]);
  c.hidden = m.count("embedding_length");
  c.layers = m.count("block_count");
  c.n_heads = m.count("attention.head_count");
  c.n_kv = m.count("attention.head_count_kv");
  c.head_dim = m.count("attention.key_length");
  if (m.count("attention.value_length") != c.head_dim)
    throw std::runtime_error("attention key and value lengths differ");
  c.n_experts = m.count("expert_count");
  c.top_k = m.count("expert_used_count");
  c.inter = m.count("expert_feed_forward_length");
  c.window = m.count("attention.sliding_window");
  c.theta = m.real_or("rope.freq_base", 0);
  c.eps = m.real_or("attention.layer_norm_rms_epsilon", 0);
  c.trained_ctx = m.count_or("context_length", 0, 0);
  if (m.text_or("rope.scaling.type", "") != "yarn") throw std::runtime_error("gpt-oss rotary scaling is not YaRN");
  c.rope.kind = FLYX_ROPE_YARN;
  c.rope.factor = float(m.real_or("rope.scaling.factor", 0));
  c.rope.orig_ctx = m.count("rope.scaling.original_context_length");
  c.rope.beta_fast = float(m.real_or("rope.scaling.yarn_beta_fast", 0));
  c.rope.beta_slow = float(m.real_or("rope.scaling.yarn_beta_slow", 0));
  c.rope.truncate = 0;
  for (const char* key : {"rope.freq_base", "attention.layer_norm_rms_epsilon", "rope.scaling.factor",
                          "rope.scaling.yarn_beta_fast", "rope.scaling.yarn_beta_slow"})
    if (!m.has(key)) throw std::runtime_error(std::string("GGUF without gpt-oss.") + key);
  if (m.has("attention.sliding_window_pattern"))
    throw std::runtime_error("a gpt-oss GGUF naming its sliding layers (gpt-oss slides every even layer)");
  if (c.hidden > kMaxWidth || c.inter > kMaxWidth || c.vocab > kMaxVocab || c.layers > kMaxLayers ||
      c.n_heads > kMaxHeads || c.head_dim > kMaxHeadDim || c.n_heads % c.n_kv || c.head_dim % 2 ||
      c.top_k > c.n_experts || int64_t(c.n_experts) * std::max(c.inter, c.hidden) > INT_MAX ||
      !(c.rope.factor >= 1.f) || !(c.rope.beta_fast > c.rope.beta_slow) || !(c.rope.beta_slow > 0.f))
    throw std::runtime_error("invalid gpt-oss GGUF configuration");
  return c;
}

// A tensor's GGUF name from the canonical name Tables::attn_block asks for ("target.L<i>.attn.q.bias").
std::string gguf_name(const std::string& name) {
  if (const char* top = gguf_top_name(name)) return top;
  const size_t dot = name.find('.', 8);  // "target.L<i>."
  const std::string layer = "blk." + name.substr(8, dot - 8) + ".";
  const std::string_view suffix = std::string_view(name).substr(dot + 1);
  if (suffix == "post_norm") return layer + "post_attention_norm.weight";
  if (suffix == "attn.sinks") return layer + "attn_sinks.weight";
  for (const auto& [canon, gguf] : {std::pair{"attn.q.bias", "attn_q.bias"}, {"attn.k.bias", "attn_k.bias"},
                                    {"attn.v.bias", "attn_v.bias"}, {"attn.o.bias", "attn_output.bias"}})
    if (suffix == canon) return layer + gguf;
  if (const char* mapped = gguf_dense_layer_name(suffix, "post_attention_norm.weight")) return layer + mapped;
  throw std::out_of_range("unknown gpt-oss GGUF tensor: " + name);
}

class GptOss final : public Model {
 public:
  explicit GptOss(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  void add_layers(std::vector<flyx_node>* p) const;

  int embed_ = -1, head_ = -1;
  std::vector<int> layers_;  // per layer: attention block, MoE block
};

bool GptOss::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  const Config c = read_config(file);
  if (file.header().contains("companions") && !file.header().at("companions").empty())
    return *err = "a gpt-oss model takes no file beside it (no drafter or projector of its own)", false;
  trained_ctx_ = c.trained_ctx;
  kv_fp8_by_default_ = false;         // binary16 K/V until a real model's measures say FP8 costs nothing
  verify_scales_with_width_ = true;   // every row reads its own experts
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, DrafterSources{}, &drafters, err)) return false;
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;

  Tables& T = tables_;
  const int H = c.hidden, I = c.inter, E = c.n_experts;
  embed_ = T.embed_block(H, c.vocab, gguf_name);
  head_ = T.head_block(H, c.vocab, embed_, 0.0, c.eps, gguf_name);
  for (int i = 0; i < c.layers; ++i) {
    const std::string p = "target.L" + std::to_string(i) + ".";
    AttnBlockSpec a{H, c.n_heads, c.n_kv, c.head_dim, c.head_dim, FLYX_GATE_NONE, c.theta, 0.0, c.eps};
    a.qk_norm = false;
    a.bias = a.sinks = true;
    a.window = i % 2 == 0 ? c.window : 0;
    a.rope = c.rope;
    layers_.push_back(T.attn_block(p, a, gguf_name));

    const std::string blk = "blk." + std::to_string(i) + ".";
    flyx_block mb{};
    mb.kind = FLYX_BLOCK_MOE;
    auto& m = mb.moe;
    m.hidden = H;
    m.inter = I;
    m.n_experts = E;
    m.top_k = c.top_k;
    m.gating = 0;        // softmax over every logit, then the k best normalised: a softmax over the k best's
    m.norm_weights = 1;
    m.norm_eps = 0.f;
    m.route_scale = 1.f;
    m.act = FLYX_MOE_SWIGLU_OAI;
    m.act_alpha = kActAlpha;
    m.act_limit = kActLimit;
    m.norm = T.norm(blk + "post_attention_norm.weight", H, 0.0, c.eps);
    m.router = T.weight(blk + "ffn_gate_inp.weight", {E, H});
    m.router_logit_b = T.weight(blk + "ffn_gate_inp.bias", {E});
    m.router_bias = -1;
    m.gate_up = -1;
    m.gate = T.weight(blk + "ffn_gate_exps.weight", {E, I, H});
    m.up = T.weight(blk + "ffn_up_exps.weight", {E, I, H});
    m.down = T.weight(blk + "ffn_down_exps.weight", {E, H, I});
    m.gate_b = T.weight(blk + "ffn_gate_exps.bias", {E, I});
    m.up_b = T.weight(blk + "ffn_up_exps.bias", {E, I});
    m.down_b = T.weight(blk + "ffn_down_exps.bias", {E, H});
    m.post_conv = m.res_query = -1;
    m.loop_ctrl_down = m.loop_ctrl_gain = m.loop_ctrl_forget = -1;
    m.sconv = -1;
    layers_.push_back(T.block(mb));
  }
  finish(H, c.vocab);
  return true;
}

void GptOss::add_layers(std::vector<flyx_node>* p) const {
  for (int b : layers_) p->push_back({FLYX_NODE_RUN, b, 0, 0});
}

std::vector<flyx_node> GptOss::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  return p;
}

// Plain decoding (width 0) or a round of given drafts (prompt lookup).
std::vector<flyx_node> GptOss::round_program(int width, int /*d*/, bool given) const {
  std::vector<flyx_node> p;
  if (width > 0) {
    if (!given) throw std::logic_error("gpt-oss holds no drafter of its own");
    p.push_back({FLYX_NODE_GIVEN, -1, width, 0});
  }
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  p.push_back({FLYX_NODE_RUN, head_, 0, 0});
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<GptOss>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::gpt_oss
