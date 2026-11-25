// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/dspark.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {

using nlohmann::json;

bool is_dspark_companion(const json& header) {
  if (!GgufMeta::is_header(header) || header.value("arch", "") != "dflash") return false;
  const GgufMeta cm(header);
  return cm.has("has_markov_context") && cm.at("has_markov_context").is_boolean() &&
         cm.at("has_markov_context").get<bool>();
}

void check_dspark_made_for(const json& header, const TargetSpec& target) {
  const GgufMeta cm(header);
  auto mismatch = [&] {
    return "a drafter made for another model (this one: width " + std::to_string(target.hidden) + ", vocabulary " +
           std::to_string(target.vocab) + ", " + std::to_string(target.layers) + " layers)";
  };
  if (cm.count("embedding_length") != target.hidden) throw std::runtime_error(mismatch());
  if (cm.has("vocab_size")) {
    const int dv = cm.count("vocab_size");
    if (dv != target.vocab && dv != target.padded_vocab()) throw std::runtime_error(mismatch());
  }
}

DSparkSpec parse_dspark_gguf(const ModelFile& file, const json& header, const TargetSpec& target, int companion) {
  const GgufMeta cm(header);
  check_dspark_made_for(header, target);
  auto cn = [&](const std::string& name) { return ModelFile::companion_name(companion, name); };

  if (!cm.has("has_markov_context") || !cm.at("has_markov_context").is_boolean() ||
      !cm.at("has_markov_context").get<bool>())
    throw std::runtime_error("DSpark GGUF must have dflash.has_markov_context = true");

  DSparkSpec spec{};
  spec.companion = companion;
  flyx_dspark_block& d = spec.block;
  spec.layers = cm.count("block_count");
  d.hidden = cm.count("embedding_length");
  if (d.hidden != target.hidden)
    throw std::runtime_error("DSpark embedding_length does not match target hidden size");
  d.inter = cm.count("feed_forward_length");
  d.n_heads = cm.count("attention.head_count");
  d.n_kv = cm.count("attention.head_count_kv");
  if (!cm.has("attention.key_length") && d.hidden % d.n_heads)
    throw std::runtime_error("DSpark GGUF hidden size is not a whole number of heads");
  d.head_dim = cm.count_or("attention.key_length", d.hidden / d.n_heads);
  if (cm.count_or("attention.value_length", d.head_dim) != d.head_dim)
    throw std::runtime_error("DSpark GGUF attention key and value lengths differ");
  d.block = cm.count("block_size");
  d.theta = cm.real_or("rope.freq_base", 1000000.0);
  spec.eps = cm.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  const double softcap =
      cm.has("final_logit_softcapping") ? cm.at("final_logit_softcapping").get<double>() : double(target.softcap);
  if (!std::isfinite(softcap) || softcap < 0.0) throw std::runtime_error("DSpark final_logit_softcapping out of range");
  d.softcap = float(softcap);

  spec.swa_window = cm.count_or("attention.sliding_window", 0, 0);
  spec.swa_pattern.assign(size_t(spec.layers), false);
  if (spec.swa_window > 0 && cm.has("attention.sliding_window_pattern")) {
    const auto& pat = cm.at("attention.sliding_window_pattern");
    if (!pat.is_array() || int(pat.size()) != spec.layers)
      throw std::runtime_error("DSpark sliding_window_pattern does not match block_count");
    for (int i = 0; i < spec.layers; ++i) {
      if (!pat[size_t(i)].is_boolean())
        throw std::runtime_error("DSpark sliding_window_pattern entries must be booleans");
      spec.swa_pattern[size_t(i)] = pat[size_t(i)].get<bool>();
    }
  }

  const auto raw_taps = cm.at("target_layers").get<std::vector<int>>();
  if (raw_taps.empty()) throw std::runtime_error("DSpark target_layers is empty");
  for (int t : raw_taps)
    if (t < 0 || t > target.layers)
      throw std::runtime_error("DSpark target_layers index " + std::to_string(t) + " out of range 0.." +
                               std::to_string(target.layers));
  spec.taps = raw_taps;
  d.n_taps = int32_t(spec.taps.size());

  const flyx_tensor_view* w1_t = file.tensor(cn("markov_w1.weight"));
  if (!w1_t || w1_t->ndim != 2) throw std::runtime_error("DSpark missing markov_w1.weight");
  if (w1_t->shape[0] != target.vocab)
    throw std::runtime_error("DSpark markov_w1.weight has " + std::to_string(w1_t->shape[0]) +
                             " rows, expected unpadded vocab " + std::to_string(target.vocab) +
                             " (use mainline DSpark GGUF)");
  d.markov_rank = int32_t(w1_t->shape[1]);
  d.vocab_out = target.vocab;
  if (d.block < 2 || d.block > FLYX_MAX_ROWS || d.markov_rank <= 0 || d.markov_rank > kMaxWidth ||
      d.n_heads <= 0 || d.n_kv <= 0 || d.n_heads % d.n_kv != 0 || d.head_dim <= 0 || d.head_dim > kMaxHeadDim ||
      d.hidden <= 0 || d.hidden > kMaxWidth || d.inter <= 0 || d.inter > kMaxWidth || spec.layers <= 0 ||
      spec.layers > kMaxLayers || !(spec.eps > 0) || !std::isfinite(spec.eps) || !(d.theta > 0) ||
      !std::isfinite(d.theta))
    throw std::runtime_error("invalid DSpark drafter dimensions");
  return spec;
}

int build_dspark(Tables& T, const ModelFile& file, const DSparkSpec& spec, const TargetSpec& target) {
  auto cn = [&](const std::string& name) { return ModelFile::companion_name(spec.companion, name); };
  flyx_dspark_block d = spec.block;
  const int H = d.hidden, nh = d.n_heads, nkv = d.n_kv, hd = d.head_dim, I = d.inter;
  const int rank = d.markov_rank, n_taps = d.n_taps, layers = spec.layers;
  const double eps = spec.eps;

  d.embed = target.embed;
  d.head = target.head;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = layers;
  const int q_dim = nh * hd, kv_dim = nkv * hd;
  for (int l = 0; l < layers; ++l) {
    const std::string p = cn("blk." + std::to_string(l) + ".");
    flyx_block lb{};
    lb.kind = FLYX_BLOCK_DSPARK_LAYER;
    auto& y = lb.dspark_layer;
    flyx_state lkv{};
    lkv.kind = FLYX_STATE_KV;
    lkv.draft = 1;
    lkv.layers = 1;
    lkv.n_kv = nkv;
    lkv.head_dim = hd;
    lkv.window = spec.swa_pattern[size_t(l)] ? spec.swa_window : 0;
    y.window = lkv.window;
    y.kv = T.state(lkv);
    y.norm = T.norm(p + "attn_norm.weight", H, 0.0, eps);
    y.post_norm = T.norm(p + "ffn_norm.weight", H, 0.0, eps);
    y.q_norm = T.norm(p + "attn_q_norm.weight", hd, 0.0, eps);
    y.k_norm = T.norm(p + "attn_k_norm.weight", hd, 0.0, eps);
    y.k_ctx_norm = T.norm(p + "attn_k_ctx_norm.weight", hd, 0.0, eps);
    y.ctx_norm = T.norm(p + "ctx_norm.weight", H, 0.0, eps);
    y.q = T.weight(p + "attn_q.weight", {q_dim, H});
    y.k = T.weight(p + "attn_k.weight", {kv_dim, H});
    y.v = T.weight(p + "attn_v.weight", {kv_dim, H});
    y.o = T.weight(p + "attn_output.weight", {H, q_dim});
    y.k_ctx = T.weight(p + "attn_k_ctx.weight", {kv_dim, H});
    y.v_ctx = T.weight(p + "attn_v_ctx.weight", {kv_dim, H});
    y.ctx_mix = T.weight(p + "ctx_mix.weight", {n_taps});
    y.gate = T.weight(p + "ffn_gate.weight", {I, H});
    y.up = T.weight(p + "ffn_up.weight", {I, H});
    y.down = T.weight(p + "ffn_down.weight", {H, I});
    T.block(lb);
  }
  d.tok_norm = T.norm(cn("token_embd_norm.weight"), H, 0.0, eps);
  d.norm = T.norm(cn("output_norm.weight"), H, 0.0, eps);
  d.markov_ctx_norm = T.norm(cn("markov_ctx_norm.weight"), H, 0.0, eps);
  d.mask_embd = T.weight(cn("mask_embd.weight"), {d.block - 1, H});
  d.markov_ctx_mix = T.weight(cn("markov_ctx_mix.weight"), {n_taps});
  d.markov_w1 = T.weight(cn("markov_w1.weight"), {target.vocab, rank});
  d.markov_w2 = T.weight(cn("markov_w2.weight"), {target.vocab, rank});
  d.markov_w2_b = T.has(cn("markov_w2.bias")) ? T.weight(cn("markov_w2.bias"), {target.vocab}) : -1;
  d.joint_proj = T.weight(cn("joint_proj.weight"), {3 * rank, 2 * H + 2 * rank});
  d.joint_proj_b = T.weight(cn("joint_proj.bias"), {3 * rank});

  flyx_state ring{};
  ring.kind = FLYX_STATE_RING;
  ring.draft = 1;
  ring.width = H;
  d.ring = T.state(ring);

  flyx_block db{};
  db.kind = FLYX_BLOCK_DSPARK;
  db.dspark = d;
  return T.block(db);
}

}  // namespace flyx::models::drafters
