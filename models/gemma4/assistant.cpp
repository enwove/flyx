// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/gemma4/assistant.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "models/gguf.h"

namespace flyx::models::gemma4 {

using nlohmann::json;

namespace {

// llama.cpp's architecture name for the assistant's GGUF files (convert_hf_to_gguf's Gemma4AssistantModel).
constexpr const char* kGgufArch = "gemma4-assistant";

int config_int(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > (int64_t(1) << 30))
    throw std::runtime_error(std::string("Gemma 4 assistant ") + key + ": not a count");
  return int(v.get<int64_t>());
}

// What either file says of the assistant.
struct Layer {
  bool sliding = false;
  int head_dim = 0, n_kv = 0, rot_dim = 0;
  double theta = 0;
};
struct Config {
  int hidden = 0, vocab = 0, n_heads = 0, inter = 0, window = 0, backbone = 0;
  double eps = 0;
  std::vector<Layer> layers;
  int n_centroids = 0, top_k = 0;  // its clustered head (0: every token scored)
  Tables::NameFn name;             // the file's name for a canonical one
};

// The canonical names Tables asks for: "draft.L<i>.<a layer's>" or "draft.<top>".
struct Names {
  const char* top[6][2];
  const char* layer[11][2];
  const char* layer_prefix;  // before the layer's index
};
constexpr Names kHf = {
    {{"draft.pre_proj", "pre_projection.weight"},
     {"draft.post_proj", "post_projection.weight"},
     {"draft.norm", "model.norm.weight"},
     {"draft.head", "model.embed_tokens.weight"},
     {"draft.centroids", "masked_embedding.centroids.weight"},
     {"draft.token_order", "masked_embedding.token_ordering"}},
    {{"in_norm", "input_layernorm.weight"},
     {"attn.q", "self_attn.q_proj.weight"},
     {"attn.o", "self_attn.o_proj.weight"},
     {"attn.q_norm", "self_attn.q_norm.weight"},
     {"attn.post_norm", "post_attention_layernorm.weight"},
     {"post_norm", "pre_feedforward_layernorm.weight"},  // the MLP's own norm before it
     {"mlp.gate", "mlp.gate_proj.weight"},
     {"mlp.up", "mlp.up_proj.weight"},
     {"mlp.down", "mlp.down_proj.weight"},
     {"mlp.post_norm", "post_feedforward_layernorm.weight"},
     {"layer_scalar", "layer_scalar"}},
    "model.layers."};
constexpr Names kGguf = {
    {{"draft.pre_proj", "nextn.pre_projection.weight"},
     {"draft.post_proj", "nextn.post_projection.weight"},
     {"draft.norm", "output_norm.weight"},
     {"draft.head", "token_embd.weight"},
     {"draft.centroids", "masked_embd_centroids.weight"},
     {"draft.token_order", "masked_embd_ordering"}},
    {{"in_norm", "attn_norm.weight"},
     {"attn.q", "attn_q.weight"},
     {"attn.o", "attn_output.weight"},
     {"attn.q_norm", "attn_q_norm.weight"},
     {"attn.post_norm", "post_attention_norm.weight"},
     {"post_norm", "ffn_norm.weight"},
     {"mlp.gate", "ffn_gate.weight"},
     {"mlp.up", "ffn_up.weight"},
     {"mlp.down", "ffn_down.weight"},
     {"mlp.post_norm", "post_ffw_norm.weight"},
     {"layer_scalar", "layer_output_scale.weight"}},
    "blk."};

Tables::NameFn names(const Names& t, int comp) {
  return [&t, comp](const std::string& canonical) {
    for (const auto& [c, f] : t.top)
      if (canonical == c) return ModelFile::companion_name(comp, f);
    const size_t dot = canonical.find('.', 7);  // "draft.L<i>."
    if (canonical.rfind("draft.L", 0) == 0 && dot != std::string::npos) {
      const std::string suffix = canonical.substr(dot + 1);
      for (const auto& [c, f] : t.layer)
        if (suffix == c)
          return ModelFile::companion_name(comp, t.layer_prefix + canonical.substr(7, dot - 7) + "." + f);
    }
    throw std::out_of_range("unknown Gemma 4 assistant tensor: " + canonical);
  };
}

// transformers' config.json (Gemma4AssistantConfig, its text_config a Gemma4TextConfig).
Config dir_config(const json& sc, int comp) {
  const json& tc = sc.at("text_config");
  Config c;
  c.hidden = config_int(tc, "hidden_size");
  c.vocab = config_int(tc, "vocab_size");
  c.n_heads = config_int(tc, "num_attention_heads");
  c.inter = config_int(tc, "intermediate_size");
  c.window = config_int(tc, "sliding_window");
  c.backbone = config_int(sc, "backbone_hidden_size");
  c.eps = tc.at("rms_norm_eps").get<double>();
  const int layers = config_int(tc, "num_hidden_layers");
  if (tc.value("num_kv_shared_layers", 0) != layers)
    throw std::runtime_error("Gemma 4 assistant: a layer with K/V of its own (its layers read the target's)");
  if (tc.value("hidden_activation", "") != "gelu_pytorch_tanh" || tc.value("use_double_wide_mlp", false) ||
      tc.value("enable_moe_block", false) || tc.value("hidden_size_per_layer_input", 0) != 0 ||
      tc.value("attention_bias", false))
    throw std::runtime_error("Gemma 4 assistant: a layer of another kind than its dense ones (GELU, one MLP)");
  const json& rp = tc.at("rope_parameters");
  const json& rs = rp.at("sliding_attention");
  const json& rf = rp.at("full_attention");
  if (rs.value("rope_type", "default") != "default" || rf.value("rope_type", "") != "proportional")
    throw std::runtime_error("Gemma 4 assistant: a rotary its layers do not take");
  const double partial = rf.value("partial_rotary_factor", 1.0);
  const auto& types = tc.at("layer_types");
  if (!types.is_array() || int(types.size()) != layers)
    throw std::runtime_error("Gemma 4 assistant layer_types: not one a layer");
  // A layer's heads: a global one's global_head_dim and num_global_key_value_heads (as Google's configs give them),
  // or the layer's own entries in per_layer_config (as transformers writes them since)
  const json& plc = tc.value("per_layer_config", json::object());
  auto heads = [&](int i, bool sliding, const char* key, const char* global_key) {
    const json& l = plc.is_object() ? plc.value(std::to_string(i), json::object()) : json::object();
    if (l.is_object() && l.contains(key)) return config_int(l, key);
    if (!sliding && !tc.value(global_key, json()).is_null()) return config_int(tc, global_key);
    return config_int(tc, key);
  };
  for (int i = 0; i < layers; ++i) {
    Layer l;
    l.sliding = types[size_t(i)] == "sliding_attention";
    if (!l.sliding && types[size_t(i)] != "full_attention")
      throw std::runtime_error("Gemma 4 assistant layer type: sliding_attention or full_attention");
    l.head_dim = heads(i, l.sliding, "head_dim", "global_head_dim");
    l.n_kv = heads(i, l.sliding, "num_key_value_heads", "num_global_key_value_heads");
    l.rot_dim = l.sliding ? l.head_dim : 2 * int(partial * l.head_dim / 2);
    l.theta = (l.sliding ? rs : rf).at("rope_theta").get<double>();
    c.layers.push_back(l);
  }
  if (sc.value("use_ordered_embeddings", false)) {  // its clustered head: the tokens of its best clusters
    c.n_centroids = config_int(sc, "num_centroids");
    c.top_k = config_int(sc, "centroid_intermediate_top_k");
  }
  c.name = names(kHf, comp);
  return c;
}

// llama.cpp's GGUF file (arch "gemma4-assistant"): what it says per layer, the global layers' rotary in rope_freqs.
Config gguf_config(const ModelFile& file, const json& h, int comp) {
  const GgufMeta m(h);
  Config c;
  c.hidden = m.count("embedding_length");
  c.n_heads = m.count("attention.head_count");
  c.inter = m.count("feed_forward_length");
  c.window = m.count("attention.sliding_window");
  c.backbone = m.count("embedding_length_out");
  c.eps = m.real_or("attention.layer_norm_rms_epsilon", 0);
  const int layers = m.count("block_count");
  if (m.count_or("attention.shared_kv_layers", 0, 0) != layers)
    throw std::runtime_error("Gemma 4 assistant: a layer with K/V of its own (its layers read the target's)");
  if (m.count_or("embedding_length_per_layer_input", 0, 0) != 0 || m.count_or("expert_count", 0, 0) != 0)
    throw std::runtime_error("Gemma 4 assistant: a layer of another kind than its dense ones (GELU, one MLP)");
  const int hd_full = m.count("attention.key_length"), hd_swa = m.count("attention.key_length_swa");
  if (m.count_or("rope.dimension_count_swa", hd_swa) != hd_swa || m.count_or("rope.dimension_count", hd_full) != hd_full)
    throw std::runtime_error("Gemma 4 assistant: a rotary over part of a head (not llama.cpp's layout)");
  const auto& pattern = m.at("attention.sliding_window_pattern");
  const auto& n_kv = m.at("attention.head_count_kv");
  if (!pattern.is_array() || pattern.size() != size_t(layers) ||
      !(n_kv.is_number_integer() || (n_kv.is_array() && n_kv.size() == size_t(layers))))
    throw std::runtime_error("Gemma 4 assistant GGUF: not a sliding flag and K/V heads a layer");
  // the global layers' turning pairs: llama.cpp's factors over the head's pairs, 1 for each that turns
  int rot_full = hd_full;
  if (const flyx_tensor_view* f = file.tensor(ModelFile::companion_name(comp, "rope_freqs.weight"))) {
    if (f->dtype != FLYX_DT_F32 || f->ndim != 1 || f->shape[0] != hd_full / 2 || !f->data)
      throw std::runtime_error("Gemma 4 assistant GGUF: rope_freqs.weight not the global head's pairs (float32)");
    const float* v = static_cast<const float*>(f->data);
    int turned = 0;
    while (turned < hd_full / 2 && v[turned] == 1.f) ++turned;
    for (int i = turned; i < hd_full / 2; ++i)
      if (!(v[i] >= 1e29f)) throw std::runtime_error("rope_freqs.weight: factors other than proportional rotary's");
    rot_full = 2 * turned;
  }
  for (int i = 0; i < layers; ++i) {
    if (!pattern[size_t(i)].is_boolean()) throw std::runtime_error("Gemma 4 assistant GGUF: a sliding flag a layer");
    Layer l;
    l.sliding = pattern[size_t(i)].get<bool>();
    l.head_dim = l.sliding ? hd_swa : hd_full;
    const json& k = n_kv.is_array() ? n_kv[size_t(i)] : n_kv;
    if (!k.is_number_integer() || k.get<int64_t>() < 1 || k.get<int64_t>() > kMaxHeads)
      throw std::runtime_error("Gemma 4 assistant GGUF: a layer's K/V heads out of range");
    l.n_kv = k.get<int>();
    l.rot_dim = l.sliding ? hd_swa : rot_full;
    l.theta = l.sliding ? m.real_or("rope.freq_base_swa", 0) : m.real_or("rope.freq_base", 0);
    if (!(l.theta > 0)) throw std::runtime_error("Gemma 4 assistant GGUF without its rotary's bases");
    c.layers.push_back(l);
  }
  const flyx_tensor_view* e = file.tensor(ModelFile::companion_name(comp, "token_embd.weight"));
  if (!e || e->ndim != 2) throw std::runtime_error("Gemma 4 assistant GGUF without its token_embd.weight");
  c.vocab = int(e->shape[0]);
  c.name = names(kGguf, comp);
  return c;
}

bool is_gguf_assistant(const json& h) { return GgufMeta::is_header(h) && h.value("arch", "") == kGgufArch; }

}  // namespace

bool is_assistant(const json& h, int target_hidden, int target_vocab) {
  if (is_gguf_assistant(h)) {
    const int backbone = GgufMeta(h).count_or("embedding_length_out", 0, 0);
    if (backbone != target_hidden)
      throw std::runtime_error("a Gemma 4 assistant made for a model of another width (" + std::to_string(backbone) +
                               "; this one: " + std::to_string(target_hidden) + ")");
    return true;  // its vocabulary: its head's rows (build_assistant)
  }
  if (!h.is_object() || h.value("kind", "") != "drafter" || !h.contains("draft") || !h["draft"].is_object() ||
      h["draft"].value("kind", "") != "gemma4_mtp")
    return false;
  const json& t = h["draft"].value("target", json::object());
  if (!t.is_object() || t.value("hidden", -1) != target_hidden || t.value("vocab", -1) != target_vocab)
    throw std::runtime_error("a Gemma 4 assistant made for a model of another width or vocabulary (" +
                             t.dump() + "; this one: hidden " + std::to_string(target_hidden) + ", vocab " +
                             std::to_string(target_vocab) + ")");
  return true;
}

int build_assistant(Tables& T, const ModelFile& file, int comp, const AssistantTarget& t) {
  const json& h = file.header().at("companions").at(size_t(comp));
  const Config c =
      is_gguf_assistant(h) ? gguf_config(file, h, comp) : dir_config(file.companion(comp).directory_config, comp);
  const int H = c.hidden, V = c.vocab, layers = int(c.layers.size());
  if (c.backbone != t.hidden || V != t.vocab)
    throw std::runtime_error("Gemma 4 assistant: made for another model's width or vocabulary");
  if (layers < 1 || layers > kMaxLayers || H < 1 || H > kMaxWidth || c.n_heads < 1 || c.n_heads > kMaxHeads ||
      c.inter < 1 || c.inter > kMaxWidth || !(c.eps > 0))
    throw std::runtime_error("invalid Gemma 4 assistant configuration");
  // It reads its target's K/V no further back than they are kept.
  if (c.window < 2 || c.window > t.window)
    throw std::runtime_error("Gemma 4 assistant: a sliding window past its target's (" + std::to_string(c.window) +
                             " of " + std::to_string(t.window) + ")");

  const Tables::NameFn& n = c.name;
  const int first = int(T.blocks.size());
  for (int i = 0; i < layers; ++i) {
    const Layer& l = c.layers[size_t(i)];
    const std::string p = "draft.L" + std::to_string(i) + ".";
    // it reads its target's K/V as the target wrote them: the same heads
    if (l.head_dim != (l.sliding ? t.hd_sliding : t.hd_full) || l.n_kv != (l.sliding ? t.n_kv_sliding : t.n_kv_full))
      throw std::runtime_error("Gemma 4 assistant: its layer " + std::to_string(i) + "'s K/V heads are not its "
                               "target's (" + std::to_string(l.n_kv) + " of " + std::to_string(l.head_dim) + ")");
    AttnBlockSpec a{H, c.n_heads, l.n_kv, l.head_dim, l.rot_dim, FLYX_GATE_NONE, l.theta, 0.0, c.eps};
    a.draft = true;
    a.window = l.sliding ? c.window : 0;
    if (!l.sliding) a.rope.kind = FLYX_ROPE_PROPORTIONAL;
    a.scale = 1.f;
    a.post_norm = true;
    a.kv_from = l.sliding ? t.kv_sliding : t.kv_full;
    if (T.attn_block(p, a, n) != first + 2 * i) throw std::logic_error("Gemma 4 assistant: its blocks not in turn");
    const int mlp = T.mlp_block(p, H, c.inter, 0.0, c.eps, n);
    flyx_mlp_block& f = T.blocks[size_t(mlp)].mlp;
    f.act = FLYX_MLP_GELU_TANH;
    f.has_post_norm = 1;
    f.post_norm = T.norm(n(p + "mlp.post_norm"), H, 0.0, c.eps);
    f.has_out_scale = 1;
    f.out_scale = T.weight(n(p + "layer_scalar"), {1});
  }
  flyx_block hb{};
  hb.kind = FLYX_BLOCK_HEAD;
  hb.head.hidden = H;
  hb.head.vocab = V;
  hb.head.norm = T.norm(n("draft.norm"), H, 0.0, c.eps);
  hb.head.w = T.weight(n("draft.head"), {V, H});
  const int head = T.block(hb);

  flyx_block b{};
  b.kind = FLYX_BLOCK_GEMMA4_MTP;
  auto& g = b.gemma4_mtp;
  g.hidden = H;
  g.target_hidden = t.hidden;
  g.vocab = V;
  g.embed = t.embed;
  g.pre_proj = T.weight(n("draft.pre_proj"), {H, 2 * t.hidden});
  g.post_proj = T.weight(n("draft.post_proj"), {t.hidden, H});
  g.first_layer = first;
  g.layers = layers;
  g.head = head;
  g.target_head = t.head;
  g.centroids = g.token_order = -1;
  if (c.n_centroids > 0) {  // its clustered head: the tokens of its best clusters
    g.n_centroids = c.n_centroids;
    g.top_k = c.top_k;
    if (V % g.n_centroids || g.top_k < 1 || g.top_k > g.n_centroids)
      throw std::runtime_error("Gemma 4 assistant: clusters that do not split its vocabulary");
    g.centroids = T.weight(n("draft.centroids"), {g.n_centroids, H});
    g.token_order = T.weight(n("draft.token_order"), {V});
  }
  flyx_state ring{};
  ring.kind = FLYX_STATE_RING;
  ring.draft = 1;
  ring.width = t.hidden;
  g.ring = T.state(ring);
  return T.block(b);
}

}  // namespace flyx::models::gemma4
