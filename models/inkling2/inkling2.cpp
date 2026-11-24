// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Inkling-2 looped MoE architecture (`inkling2_loop`) from its GGUF file, with optional DSpark companion GGUF:
//   * 24 physical layers unfolded across loop passes (middle core [core_start, core_end) run n_passes times),
//     sharing physical weight indices while each logical visit owns its own KV cache and ShortConv state.
//   * Relative-position-bias attention (no RoPE), per-query sink, Q/K/R L2 clipping, no-gamma V RMSNorm,
//     SWA-1024 + Global Full attention hybrid with position log-scaling and dynamic K/V shortconvs on global layers.
//   * Block Attention Residuals (depth-wise attention over block history) and low-rank cross-pass loop controllers.
//   * Sparse MoE (128 experts, top-8 sigmoid routing with bias, SiTU activation, 3D NVFP4 with per-expert scales).
//   * Padded weight vocab (201024) truncated to unpadded vocab (200058) at the head with final logit softcapping.
#include "models/inkling2/inkling2.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/gguf.h"

namespace flyx::models::inkling2 {
namespace {

using nlohmann::json;

int int_or_uniform_arr(const GgufMeta& m, const std::string& key, int expected_len, int min_v = 1) {
  const auto& v = m.at(key);
  if (v.is_number_integer()) return m.count(key, min_v);
  if (!v.is_array() || int(v.size()) != expected_len || v.empty())
    throw std::runtime_error("GGUF " + key + ": expected scalar or array of length " + std::to_string(expected_len));
  const int first = v[0].get<int>();
  if (first < min_v) throw std::runtime_error("GGUF " + key + ": value out of range");
  for (int i = 1; i < expected_len; ++i)
    if (v[size_t(i)].get<int>() != first)
      throw std::runtime_error("GGUF " + key + ": non-uniform per-layer dimensions are not supported");
  return first;
}

struct Config {
  int hidden = 0, n_phys = 0, vocab_padded = 0, vocab_unpadded = 0, inter = 0;
  int n_heads = 0, n_kv = 0, head_dim = 0;
  int d_rel = 0, rel_extent = 0, window = 0, conv = 0, log_scale_floor = 0;
  int attn_res_block = 0, n_experts = 0, top_k = 0;
  int core_start = 0, core_end = 0, n_passes = 0, gate_rank = 0;
  int trained_ctx = 0;
  double eps = 1e-6, qk_clip = 0, rel_clip = 0, log_scale_alpha = 0;
  double route_scale = 1.0, situ_beta = 0, situ_linear_beta = 0, softcap = 0;
  std::vector<bool> is_swa;
  bool dspark = false;
  int dspark_companion = -1;
};

Config read_gguf_config(const ModelFile& file) {
  if (!GgufMeta::is(file)) throw std::runtime_error("inkling2_loop currently requires a GGUF file");
  const GgufMeta m(file);
  Config c{};
  c.n_phys = m.count("block_count");
  c.hidden = m.count("embedding_length");
  const auto* embed = file.tensor("token_embd.weight");
  if (!embed || embed->ndim != 2) throw std::runtime_error("GGUF without token_embd.weight");
  c.vocab_padded = m.count_or("vocab_size", int(embed->shape[0]));
  c.vocab_unpadded = m.count_or("vocab_size_unpadded", c.vocab_padded);
  c.n_heads = int_or_uniform_arr(m, "attention.head_count", c.n_phys);
  c.n_kv = int_or_uniform_arr(m, "attention.head_count_kv", c.n_phys);
  c.head_dim = m.count_or("attention.key_length", c.hidden / c.n_heads);
  if (m.count_or("attention.value_length", c.head_dim) != c.head_dim)
    throw std::runtime_error("inkling2_loop: attention key and value head_dim must match");
  c.inter = int_or_uniform_arr(m, "expert_feed_forward_length", c.n_phys);
  c.n_experts = m.count("expert_count");
  c.top_k = m.count("expert_used_count");
  c.eps = m.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  c.d_rel = m.count("attention.relative_length");
  c.rel_extent = m.count("attention.relative_extent");
  c.window = m.count("attention.sliding_window");
  c.conv = m.count("shortconv.l_cache", 2);
  c.qk_clip = m.real_or("attention.qk_clip", 0.0);
  c.rel_clip = m.real_or("attention.relative_clip", 0.0);
  c.log_scale_floor = m.count("attention.log_scale_floor");
  c.log_scale_alpha = m.has("attention.log_scale_alpha") ? m.at("attention.log_scale_alpha").get<double>() : 0.0;
  c.attn_res_block = m.count("attn_res.block_size");
  c.route_scale = m.real_or("expert_weights_scale", 1.0);
  c.situ_beta = m.real_or("activation.situ_beta", 0.0);
  c.situ_linear_beta = m.real_or("activation.situ_linear_beta", 0.0);
  c.softcap = m.has("final_logit_softcapping") ? m.at("final_logit_softcapping").get<double>() : 0.0;
  c.trained_ctx = m.count_or("context_length", 0, 0);

  c.core_start = m.count("loop.start");
  c.core_end = m.count("loop.end");
  c.n_passes = m.count("loop.count");
  c.gate_rank = m.count("loop.gate_rank");
  if (m.has("loop.varying") && m.at("loop.varying").is_boolean() && m.at("loop.varying").get<bool>())
    throw std::runtime_error("inkling2_loop: variable-pass looping is not supported");

  const auto& pat = m.at("attention.sliding_window_pattern");
  if (!pat.is_array() || int(pat.size()) != c.n_phys)
    throw std::runtime_error("inkling2_loop: sliding_window_pattern does not match block_count");
  c.is_swa.reserve(size_t(c.n_phys));
  for (int i = 0; i < c.n_phys; ++i) c.is_swa.push_back(pat[size_t(i)].get<bool>());

  if (c.hidden <= 0 || c.hidden > kMaxWidth || c.inter <= 0 || c.inter > kMaxWidth ||
      c.vocab_padded <= 0 || c.vocab_padded > kMaxVocab || c.vocab_unpadded <= 0 ||
      c.vocab_unpadded > c.vocab_padded || c.n_phys <= 0 || c.n_phys > kMaxLayers ||
      c.n_heads <= 0 || c.n_kv <= 0 || c.n_heads % c.n_kv != 0 || c.head_dim <= 0 || c.head_dim > kMaxHeadDim ||
      c.top_k <= 0 || c.top_k > c.n_experts || c.n_passes < 2 || c.core_start <= 0 || c.core_start >= c.core_end ||
      c.core_end > c.n_phys || c.core_start % c.attn_res_block != 0 || c.core_end % c.attn_res_block != 0 ||
      c.conv < 2 || c.conv > 64 || !(c.eps > 0) || !(c.qk_clip > 0) || !(c.rel_clip > 0) || !(c.situ_beta > 0) ||
      !(c.situ_linear_beta > 0) || c.log_scale_alpha < 0.0 || !std::isfinite(c.log_scale_alpha))
    throw std::runtime_error("invalid inkling2_loop GGUF configuration");
  return c;
}

void take_companions(const ModelFile& file, Config* c) {
  for (int i = 0; i < file.companions(); ++i) {
    const json& h = file.header().at("companions").at(size_t(i));
    const std::string what = "drafter file " + std::to_string(i + 1) + ": ";
    if (!GgufMeta::is_header(h) || h.value("arch", "") != "dflash")
      throw std::runtime_error(what + "inkling2_loop takes only a DSpark (arch 'dflash') companion GGUF");
    if (c->dspark) throw std::runtime_error(what + "a DSpark drafter is already loaded");
    const GgufMeta cm(h);
    if (cm.count("embedding_length") != c->hidden)
      throw std::runtime_error(what + "DSpark embedding_length does not match target hidden size");
    if (!cm.has("has_markov_context") || !cm.at("has_markov_context").get<bool>())
      throw std::runtime_error(what + "DSpark GGUF must have dflash.has_markov_context = true");
    c->dspark = true;
    c->dspark_companion = i;
  }
}

class Inkling2 final : public Model {
 public:
  explicit Inkling2(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  struct LogicalLayer {
    int attn = -1;
    int moe = -1;
  };
  LogicalLayer build_logical_layer(int ip, int pass, const Config& c);
  int build_dspark(const ModelFile& file, const Config& c);
  void add_body(std::vector<flyx_node>* p) const;

  int embed_ = -1, head_ = -1, output_res_query_ = -1;
  int core_start_ = 0, core_end_ = 0, n_passes_ = 0, n_phys_ = 0, max_bank_ = 0;
  std::vector<LogicalLayer> visits_;  // 48 logical visits in execution order
  int dspark_block_ = -1;
  std::vector<int> dspark_taps_;      // 0..48 visit indices (48 = post output_norm)
};

Inkling2::LogicalLayer Inkling2::build_logical_layer(int ip, int pass, const Config& c) {
  Tables& T = tables_;
  const std::string blk = "blk." + std::to_string(ip) + ".";
  const bool swa = c.is_swa[size_t(ip)];
  const int H = c.hidden, nh = c.n_heads, nkv = c.n_kv, hd = c.head_dim;
  const int q_dim = nh * hd, kv_dim = nkv * hd;
  const int extent = swa ? c.window : c.rel_extent;
  const bool looped = pass > 0;

  // Each logical visit owns its own KV cache and ShortConv recurrent state.
  flyx_state kv_st{};
  kv_st.kind = FLYX_STATE_KV;
  kv_st.draft = 0;
  kv_st.layers = 1;
  kv_st.n_kv = nkv;
  kv_st.head_dim = hd;
  kv_st.window = swa ? c.window : 0;
  const int kv_id = T.state(kv_st);

  flyx_state sc_st{};
  sc_st.kind = FLYX_STATE_SHORTCONV;
  sc_st.draft = 0;
  sc_st.conv = c.conv;
  sc_st.channels = 2 * kv_dim + 2 * H;
  const int sc_id = T.state(sc_st);

  // Attention block
  flyx_block ab{};
  ab.kind = FLYX_BLOCK_REL_ATTN;
  auto& a = ab.rel_attn;
  a.hidden = H;
  a.n_heads = nh;
  a.n_kv = nkv;
  a.head_dim = hd;
  a.d_rel = c.d_rel;
  a.rel_extent = extent;
  a.window = swa ? c.window : 0;
  a.conv = c.conv;
  a.log_scale_floor = c.log_scale_floor;
  a.qk_clip = float(c.qk_clip);
  a.rel_clip = float(c.rel_clip);
  a.log_scale_alpha = float(c.log_scale_alpha);
  a.eps = float(c.eps);
  if (swa) {
    a.q_norm_in = a.k_norm_in = a.v_norm_in = T.norm(blk + "attn_norm.weight", H, 0.0, c.eps);
    a.k_conv_proj = -1;
    a.v_conv_proj = -1;
  } else {
    a.q_norm_in = T.norm(blk + "attn_q_inp_norm.weight", H, 0.0, c.eps);
    a.k_norm_in = T.norm(blk + "attn_k_inp_norm.weight", H, 0.0, c.eps);
    a.v_norm_in = T.norm(blk + "attn_v_inp_norm.weight", H, 0.0, c.eps);
    a.k_conv_proj = T.weight(blk + "attn_k_conv_proj.weight", {c.conv * nkv, H});
    a.v_conv_proj = T.weight(blk + "attn_v_conv_proj.weight", {c.conv * nkv, H});
  }
  a.q = T.weight(blk + "attn_q.weight", {q_dim, H});
  a.k = T.weight(blk + "attn_k.weight", {kv_dim, H});
  a.v = T.weight(blk + "attn_v.weight", {kv_dim, H});
  a.o = T.weight(blk + "attn_output.weight", {H, q_dim});
  a.gate = T.weight(blk + "attn_gate.weight", {q_dim, H});
  a.rel = T.weight(blk + "attn_rel.weight", {c.d_rel * nh, H});
  a.rel_bias = T.weight(blk + "attn_rel_bias.weight", {extent, c.d_rel});
  a.q_clip_w = T.weight(blk + "attn_q_clip.weight", {nh, hd});
  a.rel_clip_w = T.weight(blk + "attn_rel_clip.weight", {nh, c.d_rel});
  a.sink_proj = T.weight(blk + "attn_sink_proj.weight", {nh, H});
  a.k_conv = T.weight(blk + "attn_k_conv.weight", {kv_dim, c.conv});
  a.v_conv = T.weight(blk + "attn_v_conv.weight", {kv_dim, c.conv});
  a.post_conv = T.weight(blk + "attn_post_conv.weight", {H, c.conv});
  if (ip > 0) {
    a.res_query = swa ? T.weight(blk + "attn_res_query.weight", {H})
                      : T.weight(blk + "attn_res_query.weight", {3, H});
  } else {
    a.res_query = -1;
  }
  a.res_push = (ip % c.attn_res_block == 0) ? 1 : 0;
  if (looped) {
    a.loop_msg_query = T.weight(blk + "loop_msg_query.weight", {4, H});
    a.loop_rank = c.gate_rank;
    const char* ctrl_names[3] = {"loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl"};
    for (int j = 0; j < 3; ++j) {
      a.loop_ctrl_down[j] = T.weight(blk + ctrl_names[j] + "_down.weight", {c.gate_rank, 2 * H});
      a.loop_ctrl_gain[j] = T.weight(blk + ctrl_names[j] + "_gain.weight", {H, c.gate_rank});
      a.loop_ctrl_forget[j] = T.weight(blk + ctrl_names[j] + "_forget.weight", {H, c.gate_rank});
    }
  } else {
    a.loop_msg_query = -1;
    a.loop_rank = 0;
    for (int j = 0; j < 3; ++j) a.loop_ctrl_down[j] = a.loop_ctrl_gain[j] = a.loop_ctrl_forget[j] = -1;
  }
  a.kv = kv_id;
  a.sconv = sc_id;
  const int attn_id = T.block(ab);

  // MoE block
  flyx_block mb{};
  mb.kind = FLYX_BLOCK_MOE;
  auto& m = mb.moe;
  m.hidden = H;
  m.inter = c.inter;
  m.n_experts = c.n_experts;
  m.top_k = c.top_k;
  m.conv = c.conv;
  m.gating = 1;  // sigmoid
  m.norm_weights = 1;
  m.route_scale = float(c.route_scale);
  m.norm_eps = 1e-20f;
  m.situ_beta = float(c.situ_beta);
  m.situ_linear_beta = float(c.situ_linear_beta);
  m.norm = T.norm(blk + "ffn_norm.weight", H, 0.0, c.eps);
  m.router = T.weight(blk + "ffn_gate_inp.weight", {c.n_experts, H});
  m.router_bias = T.has(blk + "exp_probs_b.bias") ? T.weight(blk + "exp_probs_b.bias", {c.n_experts}) : -1;
  m.gate_up = T.weight(blk + "ffn_gate_up_exps.weight", {c.n_experts, 2 * c.inter, H}, blk + "ffn_gate_up_exps.scale");
  m.down = T.weight(blk + "ffn_down_exps.weight", {c.n_experts, H, c.inter}, blk + "ffn_down_exps.scale");
  m.post_conv = T.weight(blk + "ffn_post_conv.weight", {H, c.conv});
  m.res_query = T.weight(blk + "ffn_res_query.weight", {H});
  if (looped) {
    m.loop_rank = c.gate_rank;
    m.loop_ctrl_down = T.weight(blk + "loop_ffn_ctrl_down.weight", {c.gate_rank, 2 * H});
    m.loop_ctrl_gain = T.weight(blk + "loop_ffn_ctrl_gain.weight", {H, c.gate_rank});
    m.loop_ctrl_forget = T.weight(blk + "loop_ffn_ctrl_forget.weight", {H, c.gate_rank});
  } else {
    m.loop_rank = 0;
    m.loop_ctrl_down = m.loop_ctrl_gain = m.loop_ctrl_forget = -1;
  }
  m.sconv = sc_id;
  m.sconv_offset = 2 * kv_dim + H;
  const int moe_id = T.block(mb);
  return {attn_id, moe_id};
}

int Inkling2::build_dspark(const ModelFile& file, const Config& c) {
  Tables& T = tables_;
  const int comp = c.dspark_companion;
  const json& ch = file.header().at("companions").at(size_t(comp));
  const GgufMeta cm(ch);
  auto cn = [&](const std::string& name) { return ModelFile::companion_name(comp, name); };

  const int layers = cm.count("block_count");
  const int H = cm.count("embedding_length");
  const int I = cm.count("feed_forward_length");
  const int nh = cm.count("attention.head_count");
  const int nkv = cm.count("attention.head_count_kv");
  const int hd = cm.count_or("attention.key_length", H / nh);
  const int block = cm.count("block_size");
  const double theta = cm.real_or("rope.freq_base", 1000000.0);
  const double eps = cm.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  const double softcap = cm.has("final_logit_softcapping") ? cm.at("final_logit_softcapping").get<double>() : c.softcap;
  const int swa_win = cm.count_or("attention.sliding_window", 0, 0);
  std::vector<bool> swa_pat(size_t(layers), false);
  if (swa_win > 0 && cm.has("attention.sliding_window_pattern")) {
    const auto& pat = cm.at("attention.sliding_window_pattern");
    if (!pat.is_array() || int(pat.size()) != layers)
      throw std::runtime_error("DSpark sliding_window_pattern does not match block_count");
    for (int i = 0; i < layers; ++i) swa_pat[size_t(i)] = pat[size_t(i)].get<bool>();
  }
  const auto raw_taps = cm.at("target_layers").get<std::vector<int>>();
  const int n_visits = int(visits_.size());
  if (raw_taps.empty()) throw std::runtime_error("DSpark target_layers is empty");
  for (int t : raw_taps)
    if (t < 0 || t > n_visits)
      throw std::runtime_error("DSpark target_layers index " + std::to_string(t) + " out of range 0.." +
                               std::to_string(n_visits));
  dspark_taps_ = raw_taps;
  const int n_taps = int(dspark_taps_.size());

  const flyx_tensor_view* w1_t = file.tensor(cn("markov_w1.weight"));
  if (!w1_t || w1_t->ndim != 2) throw std::runtime_error("DSpark missing markov_w1.weight");
  if (w1_t->shape[0] != c.vocab_unpadded)
    throw std::runtime_error("DSpark markov_w1.weight has " + std::to_string(w1_t->shape[0]) +
                             " rows, expected unpadded vocab " + std::to_string(c.vocab_unpadded) +
                             " (use mainline DSpark GGUF)");
  const int rank = int(w1_t->shape[1]);
  if (block < 2 || block > FLYX_MAX_ROWS || rank <= 0 || rank > kMaxWidth || nh % nkv != 0)
    throw std::runtime_error("invalid DSpark drafter dimensions");

  flyx_dspark_block d{};
  d.hidden = H;
  d.n_heads = nh;
  d.n_kv = nkv;
  d.head_dim = hd;
  d.inter = I;
  d.block = block;
  d.n_taps = n_taps;
  d.markov_rank = rank;
  d.vocab_out = c.vocab_unpadded;
  d.theta = theta;
  d.softcap = float(softcap);
  d.embed = embed_;
  d.head = head_;
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
    lkv.window = swa_pat[size_t(l)] ? swa_win : 0;
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
  d.mask_embd = T.weight(cn("mask_embd.weight"), {block - 1, H});
  d.markov_ctx_mix = T.weight(cn("markov_ctx_mix.weight"), {n_taps});
  d.markov_w1 = T.weight(cn("markov_w1.weight"), {c.vocab_unpadded, rank});
  d.markov_w2 = T.weight(cn("markov_w2.weight"), {c.vocab_unpadded, rank});
  d.markov_w2_b = T.has(cn("markov_w2.bias")) ? T.weight(cn("markov_w2.bias"), {c.vocab_unpadded}) : -1;
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

bool Inkling2::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  Config c = read_gguf_config(file);
  take_companions(file, &c);
  trained_ctx_ = c.trained_ctx;
  // Default FP8 KV cache on non-roomy GPUs (matches Qwen3.8-27B default path; verified in docs/design/numerics.md).
  DrafterSources sources{};
  sources.dspark = c.dspark;
  sources.dspark_beside = c.dspark_companion >= 0;
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, sources, &drafters, err)) return false;
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;

  Tables& T = tables_;
  core_start_ = c.core_start;
  core_end_ = c.core_end;
  n_passes_ = c.n_passes;
  n_phys_ = c.n_phys;
  max_bank_ = (c.n_phys / c.attn_res_block) + 2;

  flyx_block eb{};
  eb.kind = FLYX_BLOCK_EMBED;
  eb.embed.hidden = c.hidden;
  eb.embed.vocab = c.vocab_padded;
  eb.embed.table = T.weight("token_embd.weight", {c.vocab_padded, c.hidden});
  eb.embed.has_norm = 1;
  eb.embed.norm = T.norm("token_embd_norm.weight", c.hidden, 0.0, c.eps);
  embed_ = T.block(eb);

  output_res_query_ = T.weight("output_res_query.weight", {c.hidden});

  flyx_block hb{};
  hb.kind = FLYX_BLOCK_HEAD;
  hb.head.hidden = c.hidden;
  hb.head.vocab = c.vocab_padded;
  hb.head.norm = T.norm("output_norm.weight", c.hidden, 0.0, c.eps);
  hb.head.w = T.has("output.weight") ? T.weight("output.weight", {c.vocab_padded, c.hidden}) : eb.embed.table;
  hb.head.vocab_out = c.vocab_unpadded;
  hb.head.softcap = float(c.softcap);
  head_ = T.block(hb);

  // Unfold physical layers into execution order: prelude [0..core_start), core [core_start..core_end) x n_passes,
  // epilogue [core_end..n_phys).
  for (int ip = 0; ip < c.core_start; ++ip) visits_.push_back(build_logical_layer(ip, 0, c));
  for (int pass = 0; pass < c.n_passes; ++pass)
    for (int ip = c.core_start; ip < c.core_end; ++ip) visits_.push_back(build_logical_layer(ip, pass, c));
  for (int ip = c.core_end; ip < c.n_phys; ++ip) visits_.push_back(build_logical_layer(ip, 0, c));

  for (Drafter kind : drafters) {
    if (kind == Drafter::kDSpark) {
      dspark_block_ = build_dspark(file, c);
      add_drafter(kind, dspark_block_, T.at(dspark_block_).dspark.block - 1);
    }
  }
  finish(c.hidden, c.vocab_unpadded);
  return true;
}

void Inkling2::add_body(std::vector<flyx_node>* p) const {
  auto tap_at = [&](int visit, int post_norm) {
    if (dspark_block_ < 0) return;
    for (size_t s = 0; s < dspark_taps_.size(); ++s)
      if (dspark_taps_[s] == visit) p->push_back({FLYX_NODE_TAP, dspark_block_, int32_t(s), post_norm});
  };
  p->push_back({FLYX_NODE_RES_RESET, -1, max_bank_, 0});
  int visit = 0;
  for (int i = 0; i < core_start_; ++i, ++visit) {
    tap_at(visit, 0);
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
  }
  p->push_back({FLYX_NODE_LOOP_PRELUDE, -1, 0, 0});
  const int core_len = core_end_ - core_start_;
  for (int pass = 0; pass < n_passes_; ++pass) {
    if (pass > 0) p->push_back({FLYX_NODE_LOOP_NEXT, -1, 0, 0});
    for (int j = 0; j < core_len; ++j, ++visit) {
      tap_at(visit, 0);
      p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
      p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
    }
  }
  for (int i = core_end_; i < n_phys_; ++i, ++visit) {
    tap_at(visit, 0);
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
  }
  p->push_back({FLYX_NODE_RES_MIX, -1, output_res_query_, 0});
  tap_at(visit, 1);  // tap final output_norm(cur) for DSpark (visit == 48)
}

std::vector<flyx_node> Inkling2::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_body(&p);
  if (dspark_block_ >= 0) p.push_back({FLYX_NODE_CONTEXT, dspark_block_, 0, 0});
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  return p;
}

std::vector<flyx_node> Inkling2::round_program(int width, int d, bool given) const {
  std::vector<flyx_node> p;
  if (width > 0) p.push_back(given ? flyx_node{FLYX_NODE_GIVEN, -1, width, 0}
                                   : flyx_node{FLYX_NODE_DRAFT, drafters_.at(size_t(d)).block, width, 0});
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_body(&p);
  p.push_back({FLYX_NODE_RUN, head_, 0, 0});
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  if (dspark_block_ >= 0) p.push_back({FLYX_NODE_CONTEXT, dspark_block_, 0, 0});
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Inkling2>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::inkling2
