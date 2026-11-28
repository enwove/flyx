// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/gemma4/assistant.h"

#include <stdexcept>
#include <string>

namespace flyx::models::gemma4 {

using nlohmann::json;

namespace {

int config_int(const json& j, const char* key) {
  const json& v = j.at(key);
  if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > (int64_t(1) << 30))
    throw std::runtime_error(std::string("Gemma 4 assistant ") + key + ": not a count");
  return int(v.get<int64_t>());
}

// The file's name for a canonical one: "draft.L<i>.<what Tables asks a layer for>" or "draft.<top>".
std::string hf_name(const std::string& canonical) {
  static const std::pair<const char*, const char*> kTop[] = {
      {"draft.pre_proj", "pre_projection.weight"},  {"draft.post_proj", "post_projection.weight"},
      {"draft.norm", "model.norm.weight"},          {"draft.head", "model.embed_tokens.weight"},
      {"draft.centroids", "masked_embedding.centroids.weight"},
      {"draft.token_order", "masked_embedding.token_ordering"},
  };
  for (const auto& [c, f] : kTop)
    if (canonical == c) return f;
  static const std::pair<const char*, const char*> kLayer[] = {
      {"in_norm", "input_layernorm.weight"},
      {"attn.q", "self_attn.q_proj.weight"},
      {"attn.o", "self_attn.o_proj.weight"},
      {"attn.q_norm", "self_attn.q_norm.weight"},
      {"attn.post_norm", "post_attention_layernorm.weight"},
      {"post_norm", "pre_feedforward_layernorm.weight"},  // the MLP's own norm before it
      {"mlp.gate", "mlp.gate_proj.weight"},
      {"mlp.up", "mlp.up_proj.weight"},
      {"mlp.down", "mlp.down_proj.weight"},
      {"mlp.post_norm", "post_feedforward_layernorm.weight"},
      {"layer_scalar", "layer_scalar"},
  };
  const size_t dot = canonical.find('.', 7);  // "draft.L<i>."
  if (canonical.rfind("draft.L", 0) == 0 && dot != std::string::npos) {
    const std::string layer = "model.layers." + canonical.substr(7, dot - 7) + ".";
    const std::string suffix = canonical.substr(dot + 1);
    for (const auto& [c, f] : kLayer)
      if (suffix == c) return layer + f;
  }
  throw std::out_of_range("unknown Gemma 4 assistant tensor: " + canonical);
}

}  // namespace

bool is_assistant(const json& h, int target_hidden, int target_vocab) {
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
  const json& sc = file.companion(comp).directory_config;
  const json& tc = sc.at("text_config");
  const int H = config_int(tc, "hidden_size"), V = config_int(tc, "vocab_size");
  const int layers = config_int(tc, "num_hidden_layers"), nh = config_int(tc, "num_attention_heads");
  const int inter = config_int(tc, "intermediate_size");
  // A layer's heads: a global one's global_head_dim and num_global_key_value_heads (as Google's configs give them),
  // or the layer's own entries in per_layer_config (as transformers writes them since)
  const json& plc = tc.value("per_layer_config", json::object());
  auto heads = [&](int i, bool sliding, const char* key, const char* global_key) {
    const json& l = plc.is_object() ? plc.value(std::to_string(i), json::object()) : json::object();
    if (l.is_object() && l.contains(key)) return config_int(l, key);
    if (!sliding && !tc.value(global_key, json()).is_null()) return config_int(tc, global_key);
    return config_int(tc, key);
  };
  const int window = config_int(tc, "sliding_window");
  const double eps = tc.at("rms_norm_eps").get<double>();
  if (config_int(sc, "backbone_hidden_size") != t.hidden || V != t.vocab)
    throw std::runtime_error("Gemma 4 assistant: made for another model's width or vocabulary");
  if (layers < 1 || layers > kMaxLayers || H < 1 || H > kMaxWidth || nh < 1 || nh > kMaxHeads || !(eps > 0))
    throw std::runtime_error("invalid Gemma 4 assistant configuration");
  if (tc.value("num_kv_shared_layers", 0) != layers)
    throw std::runtime_error("Gemma 4 assistant: a layer with K/V of its own (its layers read the target's)");
  if (tc.value("hidden_activation", "") != "gelu_pytorch_tanh" || tc.value("use_double_wide_mlp", false) ||
      tc.value("enable_moe_block", false) || tc.value("hidden_size_per_layer_input", 0) != 0 ||
      tc.value("attention_bias", false))
    throw std::runtime_error("Gemma 4 assistant: a layer of another kind than its dense ones (GELU, one MLP)");
  // It reads its target's K/V no further back than they are kept.
  if (window < 2 || window > t.window)
    throw std::runtime_error("Gemma 4 assistant: a sliding window past its target's (" + std::to_string(window) +
                             " of " + std::to_string(t.window) + ")");
  const json& rp = tc.at("rope_parameters");
  const json& rs = rp.at("sliding_attention");
  const json& rf = rp.at("full_attention");
  if (rs.value("rope_type", "default") != "default" || rf.value("rope_type", "") != "proportional")
    throw std::runtime_error("Gemma 4 assistant: a rotary its layers do not take");
  const double partial = rf.value("partial_rotary_factor", 1.0);
  const auto& types = tc.at("layer_types");
  if (!types.is_array() || int(types.size()) != layers)
    throw std::runtime_error("Gemma 4 assistant layer_types: not one a layer");

  const Tables::NameFn n = [comp](const std::string& c) { return ModelFile::companion_name(comp, hf_name(c)); };
  const int first = int(T.blocks.size());
  for (int i = 0; i < layers; ++i) {
    const bool sliding = types[size_t(i)] == "sliding_attention";
    if (!sliding && types[size_t(i)] != "full_attention")
      throw std::runtime_error("Gemma 4 assistant layer type: sliding_attention or full_attention");
    const std::string p = "draft.L" + std::to_string(i) + ".";
    const int hd = heads(i, sliding, "head_dim", "global_head_dim");
    const int n_kv = heads(i, sliding, "num_key_value_heads", "num_global_key_value_heads");
    // it reads its target's K/V as the target wrote them: the same heads
    if (hd != (sliding ? t.hd_sliding : t.hd_full) || n_kv != (sliding ? t.n_kv_sliding : t.n_kv_full))
      throw std::runtime_error("Gemma 4 assistant: its layer " + std::to_string(i) + "'s K/V heads are not its "
                               "target's (" + std::to_string(n_kv) + " of " + std::to_string(hd) + ")");
    const int rot = sliding ? hd : 2 * int(partial * hd / 2);
    AttnBlockSpec a{H,   nh, n_kv, hd, rot, FLYX_GATE_NONE,
                    sliding ? rs.at("rope_theta").get<double>() : rf.at("rope_theta").get<double>(), 0.0, eps};
    a.draft = true;
    a.window = sliding ? window : 0;
    if (!sliding) a.rope.kind = FLYX_ROPE_PROPORTIONAL;
    a.scale = 1.f;
    a.post_norm = true;
    a.kv_from = sliding ? t.kv_sliding : t.kv_full;
    if (T.attn_block(p, a, n) != first + 2 * i) throw std::logic_error("Gemma 4 assistant: its blocks not in turn");
    const int mlp = T.mlp_block(p, H, inter, 0.0, eps, n);
    flyx_mlp_block& f = T.blocks[size_t(mlp)].mlp;
    f.act = FLYX_MLP_GELU_TANH;
    f.has_post_norm = 1;
    f.post_norm = T.norm(n(p + "mlp.post_norm"), H, 0.0, eps);
    f.has_out_scale = 1;
    f.out_scale = T.weight(n(p + "layer_scalar"), {1});
  }
  flyx_block hb{};
  hb.kind = FLYX_BLOCK_HEAD;
  hb.head.hidden = H;
  hb.head.vocab = V;
  hb.head.norm = T.norm(n("draft.norm"), H, 0.0, eps);
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
  if (sc.value("use_ordered_embeddings", false)) {  // its clustered head: the tokens of its best clusters
    g.n_centroids = config_int(sc, "num_centroids");
    g.top_k = config_int(sc, "centroid_intermediate_top_k");
    if (g.n_centroids < 1 || V % g.n_centroids || g.top_k < 1 || g.top_k > g.n_centroids)
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
