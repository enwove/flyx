// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Inkling-2 looped MoE architecture (`inkling2_loop`) from its GGUF file, with optional DSpark / DFlash2 drafters:
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

#include "models/drafters/registry.h"
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
  void add_body(std::vector<flyx_node>* p) const;

  int embed_ = -1, head_ = -1, output_res_query_ = -1;
  int core_start_ = 0, core_end_ = 0, n_passes_ = 0, n_phys_ = 0, max_bank_ = 0;
  std::vector<LogicalLayer> visits_;  // 48 logical visits in execution order
  drafters::DrafterRegistry drafters_reg_;
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

bool Inkling2::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  Config c = read_gguf_config(file);
  const int n_visits = c.core_start + c.n_passes * (c.core_end - c.core_start) + (c.n_phys - c.core_end);
  drafters::TargetSpec target{};
  target.arch = "inkling2_loop";
  target.hidden = c.hidden;
  target.vocab = c.vocab_unpadded;
  target.vocab_padded = c.vocab_padded;
  target.layers = n_visits;
  target.softcap = float(c.softcap);
  target.gguf = true;
  target.has_post_norm_tap = true;
  target.dspark_taps = true;

  std::vector<int> unclaimed;
  drafters_reg_.scan(file, target, nullptr, /*has_dv=*/false, &unclaimed);
  if (!unclaimed.empty()) {
    const int i = unclaimed.front();
    const json& ch = file.header().at("companions").at(size_t(i));
    if (ch.is_object() && ch.value("arch", "") == "clip")
      return *err = "an inkling2_loop model does not support a vision projector (--mmproj)", false;
    return *err = "companion file " + std::to_string(i + 1) +
                  ": inkling2_loop takes only DSpark or DFlash2 companion drafters",
           false;
  }

  trained_ctx_ = c.trained_ctx;
  // Every row of a sparse MoE reads its own experts' weights, so verifying more drafts costs nearly in proportion:
  // a block drafter's rounds take the finer width ladder and stop on the device where its confidence falls
  // (Model::draft_widths, adaptive_checkpoints).
  verify_scales_with_width_ = true;
  // Default FP8 KV cache on non-roomy GPUs (matches Qwen3.8-27B default path; verified in docs/design/numerics.md).
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, drafters_reg_.sources(), &drafters, err)) return false;
  drafters_reg_.validate_selected(file, drafters, target);
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

  target.embed = embed_;
  target.head = head_;
  for (Drafter kind : drafters) {
    int max_drafts = 0;
    const int blk = drafters_reg_.attach(kind, file, T, target, nullptr, &max_drafts);
    add_drafter(kind, blk, max_drafts);
  }
  finish(c.hidden, c.vocab_unpadded);
  return true;
}

void Inkling2::add_body(std::vector<flyx_node>* p) const {
  p->push_back({FLYX_NODE_RES_RESET, -1, max_bank_, 0});
  int visit = 0;
  for (int i = 0; i < core_start_; ++i, ++visit) {
    drafters_reg_.emit_pre_visit_taps(p, visit);
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
    drafters_reg_.emit_taps(p, visit + 1);
  }
  p->push_back({FLYX_NODE_LOOP_PRELUDE, -1, 0, 0});
  const int core_len = core_end_ - core_start_;
  for (int pass = 0; pass < n_passes_; ++pass) {
    if (pass > 0) p->push_back({FLYX_NODE_LOOP_NEXT, -1, 0, 0});
    for (int j = 0; j < core_len; ++j, ++visit) {
      drafters_reg_.emit_pre_visit_taps(p, visit);
      p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
      p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
      drafters_reg_.emit_taps(p, visit + 1);
    }
  }
  for (int i = core_end_; i < n_phys_; ++i, ++visit) {
    drafters_reg_.emit_pre_visit_taps(p, visit);
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].attn, 0, 0});
    p->push_back({FLYX_NODE_RUN, visits_[size_t(visit)].moe, 0, 0});
    drafters_reg_.emit_taps(p, visit + 1);
  }
  p->push_back({FLYX_NODE_RES_MIX, -1, output_res_query_, 0});
  drafters_reg_.emit_post_norm_taps(p, visit);  // tap final output_norm(cur) for DSpark and DFlash2 (visit == 48)
}

std::vector<flyx_node> Inkling2::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_body(&p);
  drafters_reg_.emit_contexts(&p);
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
  drafters_reg_.emit_contexts(&p);
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Inkling2>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::inkling2
