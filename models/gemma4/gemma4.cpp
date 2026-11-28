// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gemma 4 from llama.cpp's GGUF layout (arch "gemma4"; convert_hf_to_gguf's Gemma4Model), or from a .flyx file
// (python/convert: its header's config, its tensors under the canonical names).  What a GGUF file says per
// layer: sliding or not (attention.sliding_window_pattern), its K/V heads (attention.head_count_kv), its MLP width
// (feed_forward_length); the sliding layers' and the global layers' head widths, rotary dims and thetas; whether a
// layer holds a v projection (none: v is k's, attention_k_eq_v).  The global layers' proportional rotary is in
// rope_freqs.weight as llama.cpp writes it (a factor 1 for a pair that turns, 1e30 for one that does not).  What
// the file does not say, the architecture does (Hugging Face's Gemma4ForCausalLM): plain RMSNorm weights, v normed
// without one, q.k unscaled, GELU's tanh form, and the embedding times sqrt(hidden) rounded to bf16 as the
// checkpoints' bf16 forward rounds it.  A layer with experts (26B-A4B: one with a router, ffn_gate_inp) runs its
// dense MLP beside them (models/model.h's MoE block: its shared MLP), each between norms of its own, the sum normed
// again before the residual add.  Per-layer embeddings and shared K/V layers (E2B, E4B) are refused here for now.
#include "models/gemma4/gemma4.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "models/drafters/registry.h"
#include "models/gemma4/assistant.h"
#include "models/gguf.h"

namespace flyx::models::gemma4 {
namespace {

struct Layer {
  bool sliding = false;
  int n_kv = 0, head_dim = 0, rot_dim = 0, inter = 0;
  double theta = 0;
  bool v_from_k = false;
  bool moe = false;  // experts beside its MLP
};

struct Config {
  int hidden = 0, vocab = 0, n_heads = 0, window = 0;
  double eps = 0;
  float softcap = 0.f;
  int trained_ctx = 0;
  bool gguf = false;
  int n_experts = 0, top_k = 0, moe_inter = 0;
  std::vector<Layer> layers;
};

void validate(const Config& c, int hd_full, int hd_swa) {
  if (c.hidden <= 0 || c.hidden > kMaxWidth || c.vocab <= 0 || c.vocab > kMaxVocab || c.layers.empty() ||
      int(c.layers.size()) > kMaxLayers || c.n_heads <= 0 || c.n_heads > kMaxHeads || hd_full <= 0 ||
      hd_full > kMaxHeadDim || hd_swa <= 0 || hd_swa > kMaxHeadDim || hd_full % 2 || hd_swa % 2 || !(c.eps > 0) ||
      c.window < 1)
    throw std::runtime_error("invalid Gemma 4 configuration");
  for (const Layer& l : c.layers)
    if (l.n_kv < 1 || c.n_heads % l.n_kv || l.inter < 1 || l.inter > kMaxWidth || l.rot_dim < 2 ||
        l.rot_dim > l.head_dim || l.rot_dim % 2 || !(l.theta > 0))
      throw std::runtime_error("invalid Gemma 4 layer configuration");
  const bool moe = std::any_of(c.layers.begin(), c.layers.end(), [](const Layer& l) { return l.moe; });
  if (moe && (c.n_experts < 1 || int64_t(c.n_experts) * std::max(c.moe_inter, c.hidden) > INT_MAX || c.top_k < 1 ||
              c.top_k > c.n_experts ||
              c.moe_inter < 1 || c.moe_inter > kMaxWidth))
    throw std::runtime_error("invalid Gemma 4 experts' configuration");
}

// A .flyx file's config (python/convert canonical_gemma4_config).
Config read_flyx_config(const ModelFile& file) {
  const nlohmann::json& j = file.header().at("config");
  Config c;
  c.hidden = j.at("hidden");
  c.vocab = j.at("vocab");
  c.n_heads = j.at("n_heads");
  c.window = j.at("window");
  c.eps = j.at("eps");
  c.softcap = j.value("softcap", 0.0f);
  c.trained_ctx = j.value("max_pos", 0);
  if (!j.value("tie_embed", true)) throw std::runtime_error("a Gemma 4 .flyx file with an untied head");
  const int hd = j.at("head_dim"), hd_full = j.at("head_dim_full"), n_kv = j.at("n_kv"), n_kv_full = j.at("n_kv_full");
  const int inter = j.at("inter"), rot_full = j.at("rot_dim_full");
  const double theta = j.at("rope_theta"), theta_swa = j.at("rope_theta_swa");
  const bool k_eq_v = j.at("k_eq_v");
  c.n_experts = j.value("experts", 0);
  c.top_k = j.value("top_k", 0);
  c.moe_inter = j.value("moe_inter", 0);
  const auto& types = j.at("layer_types");
  if (!types.is_array() || int(types.size()) != j.at("layers").get<int>())
    throw std::runtime_error("Gemma 4 layer_types: not one a layer");
  for (const auto& t : types) {
    Layer l;
    l.sliding = t == "sliding";
    if (!l.sliding && t != "full") throw std::runtime_error("Gemma 4 layer type: sliding or full");
    l.n_kv = l.sliding ? n_kv : n_kv_full;
    l.head_dim = l.sliding ? hd : hd_full;
    l.rot_dim = l.sliding ? hd : rot_full;
    l.theta = l.sliding ? theta_swa : theta;
    l.inter = inter;
    l.v_from_k = !l.sliding && k_eq_v;
    l.moe = c.n_experts > 0;  // Hugging Face's enable_moe_block: every layer's
    c.layers.push_back(l);
  }
  validate(c, hd_full, hd);
  return c;
}

// A float rounded to bf16 (to nearest, ties to even), as torch's .to(bfloat16) rounds it.
float round_bf16(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  std::memcpy(&v, &u, 4);
  return v;
}

// A per-layer key: a scalar for every layer, or an array of `layers` integers.
std::vector<int> per_layer(const GgufMeta& m, const std::string& key, int layers) {
  const auto& v = m.at(key);
  if (v.is_number_integer()) return std::vector<int>(size_t(layers), m.count(key));
  if (!v.is_array() || v.size() != size_t(layers))
    throw std::runtime_error("GGUF gemma4." + key + ": not one value or one per layer");
  std::vector<int> out;
  for (const auto& e : v) {
    if (!e.is_number_integer() || e.get<int64_t>() < 1 || e.get<int64_t>() > kMaxWidth)
      throw std::runtime_error("GGUF gemma4." + key + ": a value out of range");
    out.push_back(e.get<int>());
  }
  return out;
}

// The global layers' rotary dims from rope_freqs.weight: llama.cpp's factors over the head's pairs, 1 for each of the
// first that turn and 1e30 (a frequency of nothing) for the rest.
int proportional_rot_dim(const ModelFile& file, int head_dim) {
  const flyx_tensor_view* t = file.tensor("rope_freqs.weight");
  if (!t || t->dtype != FLYX_DT_F32 || t->ndim != 1 || t->shape[0] != head_dim / 2 || !t->data)
    throw std::runtime_error("GGUF without a rope_freqs.weight of the global head's pairs (float32)");
  const float* f = static_cast<const float*>(t->data);
  int turned = 0;
  while (turned < head_dim / 2 && f[turned] == 1.f) ++turned;
  for (int i = turned; i < head_dim / 2; ++i)
    if (!(f[i] >= 1e29f)) throw std::runtime_error("rope_freqs.weight: factors other than proportional rotary's");
  if (turned == 0) throw std::runtime_error("rope_freqs.weight: no pair turns");
  return 2 * turned;
}

Config read_config(const ModelFile& file) {
  if (!GgufMeta::is(file)) return read_flyx_config(file);
  const GgufMeta m(file);
  Config c;
  c.gguf = true;
  const auto* embed = file.tensor("token_embd.weight");
  if (!embed || embed->ndim != 2) throw std::runtime_error("GGUF without a token_embd.weight matrix");
  c.vocab = int(embed->shape[0]);
  c.hidden = m.count("embedding_length");
  const int layers = m.count("block_count");
  c.n_heads = m.count("attention.head_count");
  c.window = m.count("attention.sliding_window");
  c.eps = m.real_or("attention.layer_norm_rms_epsilon", 0);
  c.softcap = float(m.real_or("final_logit_softcapping", 0));
  c.trained_ctx = m.count_or("context_length", 0, 0);
  for (const char* key : {"attention.layer_norm_rms_epsilon", "rope.freq_base", "rope.freq_base_swa"})
    if (!m.has(key)) throw std::runtime_error(std::string("GGUF without gemma4.") + key);
  // What this build does not run yet: per-layer embeddings (E2B, E4B), layers reading another's K/V.
  if (m.count_or("embedding_length_per_layer_input", 0, 0) != 0)
    throw std::runtime_error("Gemma 4 with per-layer embeddings (E2B, E4B) is not supported yet");
  if (m.count_or("attention.shared_kv_layers", 0, 0) != 0)
    throw std::runtime_error("Gemma 4 with shared K/V layers is not supported yet");
  if (m.count_or("expert_count", 0, 0) != 0) {
    c.n_experts = m.count("expert_count");
    c.top_k = m.count("expert_used_count");
    c.moe_inter = m.count("expert_feed_forward_length");
  }
  const int hd_full = m.count("attention.key_length"), hd_swa = m.count("attention.key_length_swa");
  if (m.count("attention.value_length") != hd_full || m.count("attention.value_length_swa") != hd_swa)
    throw std::runtime_error("Gemma 4 attention key and value lengths differ");
  const int rot_swa = m.count_or("rope.dimension_count_swa", hd_swa);
  if (rot_swa != hd_swa) throw std::runtime_error("Gemma 4 sliding layers rotating part of the head");
  if (m.count_or("rope.dimension_count", hd_full) != hd_full)
    throw std::runtime_error("Gemma 4 global layers' rotary not over the head (llama.cpp's proportional layout)");
  const int rot_full = proportional_rot_dim(file, hd_full);
  const double theta = m.real_or("rope.freq_base", 0), theta_swa = m.real_or("rope.freq_base_swa", 0);
  const auto& pattern = m.at("attention.sliding_window_pattern");
  if (!pattern.is_array() || pattern.size() != size_t(layers))
    throw std::runtime_error("GGUF gemma4.attention.sliding_window_pattern: not one flag per layer");
  const std::vector<int> n_kv = per_layer(m, "attention.head_count_kv", layers);
  const std::vector<int> inter = per_layer(m, "feed_forward_length", layers);
  for (int i = 0; i < layers; ++i) {
    if (!pattern[size_t(i)].is_boolean())
      throw std::runtime_error("GGUF gemma4.attention.sliding_window_pattern: not one flag per layer");
    Layer l;
    l.sliding = pattern[size_t(i)].get<bool>();
    l.n_kv = n_kv[size_t(i)];
    l.head_dim = l.sliding ? hd_swa : hd_full;
    l.rot_dim = l.sliding ? rot_swa : rot_full;
    l.theta = l.sliding ? theta_swa : theta;
    l.inter = inter[size_t(i)];
    l.v_from_k = !file.tensor("blk." + std::to_string(i) + ".attn_v.weight");
    l.moe = file.tensor("blk." + std::to_string(i) + ".ffn_gate_inp.weight") != nullptr;
    if (l.moe && c.n_experts == 0) throw std::runtime_error("GGUF Gemma 4 router without gemma4.expert_count");
    if (l.sliding && l.v_from_k) throw std::runtime_error("Gemma 4 sliding layer without a v projection");
    if (c.n_heads % l.n_kv) throw std::runtime_error("Gemma 4 query heads not a multiple of a layer's K/V heads");
    c.layers.push_back(l);
  }
  validate(c, hd_full, hd_swa);
  return c;
}

// A tensor's GGUF name from the canonical name the Tables builders ask for ("target.L<i>.attn.post_norm").
std::string gguf_name(const std::string& name) {
  if (const char* top = gguf_top_name(name)) return top;
  const size_t dot = name.find('.', 8);  // "target.L<i>."
  const std::string layer = "blk." + name.substr(8, dot - 8) + ".";
  const std::string_view suffix = std::string_view(name).substr(dot + 1);
  if (suffix == "attn.post_norm") return layer + "post_attention_norm.weight";
  if (suffix == "mlp.post_norm") return layer + "post_ffw_norm.weight";
  if (suffix == "layer_scalar") return layer + "layer_output_scale.weight";
  // a layer with experts: the dense MLP's norm after it, the experts' and the router's tensors
  static const std::pair<std::string_view, const char*> kMoe[] = {
      {"mlp.post_norm_1", "post_ffw_norm_1.weight"},  {"moe.norm", "pre_ffw_norm_2.weight"},
      {"moe.post_norm", "post_ffw_norm_2.weight"},    {"moe.router", "ffn_gate_inp.weight"},
      {"moe.router_in", "ffn_gate_inp.scale"},        {"moe.expert_scale", "ffn_down_exps.scale"},
      {"moe.gate_up", "ffn_gate_up_exps.weight"},     {"moe.gate", "ffn_gate_exps.weight"},
      {"moe.up", "ffn_up_exps.weight"},               {"moe.down", "ffn_down_exps.weight"},
  };
  for (const auto& [c, f] : kMoe)
    if (suffix == c) return layer + f;
  // "post_norm" is the MLP's own norm before it (Hugging Face's pre_feedforward_layernorm)
  if (const char* mapped = gguf_dense_layer_name(suffix, "ffn_norm.weight")) return layer + mapped;
  throw std::out_of_range("unknown Gemma 4 GGUF tensor: " + name);
}

class Gemma4 final : public Model {
 public:
  explicit Gemma4(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  void add_layers(std::vector<flyx_node>* p) const;
  int moe_block(const std::string& p, const Config& c, int shared, const Tables::NameFn& n);

  int embed_ = -1, head_ = -1;
  std::vector<int> layers_;  // per layer: attention block, MLP block
  drafters::DrafterRegistry drafters_reg_;
};

bool Gemma4::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  const Config c = read_config(file);
  trained_ctx_ = c.trained_ctx;
  kv_fp8_by_default_ = false;  // binary16 K/V until a real model's measures say FP8 costs nothing
  verify_scales_with_width_ = c.n_experts > 0;  // with experts, every row reads its own
  drafters::TargetSpec target{};
  target.arch = file.arch();
  target.hidden = c.hidden;
  target.vocab = c.vocab;
  target.layers = int(c.layers.size());
  target.gguf = c.gguf;
  // Its assistant is its MTP drafter: a block of its own, built (below) once the layers whose K/V it reads are.
  AssistantTarget at;
  drafters::MtpHooks hooks{};
  hooks.claims = [&c](const nlohmann::json& h) { return is_assistant(h, c.hidden, c.vocab); };
  hooks.build = [this, &file, &at](int companion) { return build_assistant(tables_, file, companion, at); };
  std::vector<int> unclaimed;
  drafters_reg_.scan(file, target, &hooks, /*has_dv=*/false, &unclaimed);
  if (!unclaimed.empty())
    return *err = "companion file " + std::to_string(unclaimed.front() + 1) +
                  ": holds no drafter a Gemma 4 model takes (its assistant, or none)",
           false;
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, drafters_reg_.sources(), &drafters, err)) return false;
  for (Drafter d : drafters)
    if (d != Drafter::kMtp)
      return *err = std::string("a Gemma 4 model drafts with its assistant (mtp) or prompt lookup, not ") +
                    models::drafter_name(d),
             false;
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;

  Tables& T = tables_;
  const int H = c.hidden;
  // the file's name for a tensor: GGUF's, or the canonical one (.flyx)
  const Tables::NameFn n =
      c.gguf ? Tables::NameFn(gguf_name) : Tables::NameFn([](const std::string& s) { return s; });
  embed_ = T.embed_block(H, c.vocab, n);
  T.blocks[size_t(embed_)].embed.scale = round_bf16(std::sqrt(float(H)));
  head_ = T.head_block(H, c.vocab, embed_, 0.0, c.eps, n);
  T.blocks[size_t(head_)].head.softcap = c.softcap;
  for (size_t i = 0; i < c.layers.size(); ++i) {
    const Layer& l = c.layers[i];
    const std::string p = "target.L" + std::to_string(i) + ".";
    AttnBlockSpec a{H, c.n_heads, l.n_kv, l.head_dim, l.rot_dim, FLYX_GATE_NONE, l.theta, 0.0, c.eps};
    a.window = l.sliding ? c.window : 0;
    if (!l.sliding) a.rope.kind = FLYX_ROPE_PROPORTIONAL;
    a.scale = 1.f;
    a.v_from_k = l.v_from_k;
    a.v_norm = a.post_norm = true;
    layers_.push_back(T.attn_block(p, a, n));
    const int mlp = T.mlp_block(p, H, l.inter, 0.0, c.eps, n);
    flyx_mlp_block& f = T.blocks[size_t(mlp)].mlp;
    f.act = FLYX_MLP_GELU_TANH;
    f.has_post_norm = 1;
    if (!l.moe) {
      f.post_norm = T.norm(n(p + "mlp.post_norm"), H, 0.0, c.eps);
      f.has_out_scale = 1;
      f.out_scale = T.weight(n(p + "layer_scalar"), {1});
      layers_.push_back(mlp);
    } else {
      f.post_norm = T.norm(n(p + "mlp.post_norm_1"), H, 0.0, c.eps);
      layers_.push_back(moe_block(p, c, mlp, n));
    }
    const flyx_attn_block& ab = T.blocks[size_t(layers_[layers_.size() - 2])].attn;
    (l.sliding ? at.kv_sliding : at.kv_full) = ab.kv;  // its last layer of each kind's, by the end
    (l.sliding ? at.n_kv_sliding : at.n_kv_full) = l.n_kv;
    (l.sliding ? at.hd_sliding : at.hd_full) = l.head_dim;
  }
  at.hidden = H;
  at.vocab = c.vocab;
  at.embed = embed_;
  at.head = head_;
  at.window = c.window;
  target.embed = embed_;
  target.head = head_;
  for (Drafter kind : drafters) {
    DrafterReach reach;
    const int blk = drafters_reg_.attach(kind, file, T, target, &hooks, &reach);
    add_drafter(kind, blk, reach);
  }
  finish(H, c.vocab);
  return true;
}

// A layer's experts (Hugging Face's Gemma4TextRouter and Gemma4TextExperts), the dense MLP `shared` beside them:
// the router over the weightless RMSNorm of the layer's input times its scale and 1 / sqrt(hidden), a softmax over
// every expert, the top_k normalized and each times its expert's scale; the experts' GELU (tanh) GLU over their own
// normed input; their output normed and added to the MLP's normed output, the sum normed before the residual add,
// the layer times its scalar.
int Gemma4::moe_block(const std::string& p, const Config& c, int shared, const Tables::NameFn& n) {
  Tables& T = tables_;
  const int H = c.hidden, E = c.n_experts, I = c.moe_inter;
  flyx_block mb{};
  mb.kind = FLYX_BLOCK_MOE;
  auto& m = mb.moe;
  m.hidden = H;
  m.inter = I;
  m.n_experts = E;
  m.top_k = c.top_k;
  m.gating = 0;
  m.norm_weights = 1;
  m.norm_eps = 0.f;
  m.route_scale = 1.f;
  m.act = FLYX_MOE_GELU_TANH;
  m.norm = T.norm(n(p + "moe.norm"), H, 0.0, c.eps);
  m.router = T.weight(n(p + "moe.router"), {E, H});
  m.router_bias = m.router_logit_b = -1;
  if (T.has(n(p + "moe.gate_up"))) {
    m.gate_up = T.weight(n(p + "moe.gate_up"), {E, 2 * I, H});
    m.gate = m.up = -1;
  } else {
    m.gate_up = -1;
    m.gate = T.weight(n(p + "moe.gate"), {E, I, H});
    m.up = T.weight(n(p + "moe.up"), {E, I, H});
  }
  m.down = T.weight(n(p + "moe.down"), {E, H, I});
  m.gate_b = m.up_b = m.down_b = -1;
  m.post_conv = m.res_query = -1;
  m.loop_ctrl_down = m.loop_ctrl_gain = m.loop_ctrl_forget = -1;
  m.sconv = -1;
  m.has_router_norm = 1;
  m.router_norm = flyx_norm{-1, 0, 0.0, c.eps};
  m.router_in = T.weight(n(p + "moe.router_in"), {H});
  m.router_mul = float(1.0 / std::sqrt(double(H)));
  m.has_expert_scale = 1;
  m.expert_scale = T.weight(n(p + "moe.expert_scale"), {E});
  m.has_shared = 1;
  m.shared = shared;
  m.has_post_norm = 1;
  m.post_norm = T.norm(n(p + "moe.post_norm"), H, 0.0, c.eps);
  m.has_out_norm = 1;
  m.out_norm = T.norm(n(p + "mlp.post_norm"), H, 0.0, c.eps);
  m.out_scale = T.weight(n(p + "layer_scalar"), {1});
  return T.block(mb);
}

void Gemma4::add_layers(std::vector<flyx_node>* p) const {
  for (int b : layers_) p->push_back({FLYX_NODE_RUN, b, 0, 0});
}

std::vector<flyx_node> Gemma4::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  drafters_reg_.emit_contexts(&p);
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  return p;
}

// Plain decoding (width 0), a round of given drafts (prompt lookup) or of its assistant's.
std::vector<flyx_node> Gemma4::round_program(int width, int d, bool given) const {
  std::vector<flyx_node> p;
  if (width > 0)
    p.push_back(given ? flyx_node{FLYX_NODE_GIVEN, -1, width, 0}
                      : flyx_node{FLYX_NODE_DRAFT, drafters_.at(size_t(d)).block, width, 0});
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  p.push_back({FLYX_NODE_RUN, head_, 0, 0});
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  drafters_reg_.emit_contexts(&p);
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Gemma4>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::gemma4
