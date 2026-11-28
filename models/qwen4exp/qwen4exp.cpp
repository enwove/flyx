// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/qwen4exp/qwen4exp.h"

#include <algorithm>
#include <climits>
#include <limits>

#include "models/gguf.h"

namespace flyx::models::qwen4exp {
namespace {

void require(bool valid, const std::string& what) {
  if (!valid) throw std::runtime_error("qwen4exp: " + what);
}

int product(int a, int b) {
  require(a > 0 && b > 0 && a <= INT_MAX / b, "dimension product outside int32 range");
  return a * b;
}

std::vector<uint64_t> integers(const GgufMeta& m, const char* name, size_t count) {
  const auto& array = m.at(name);
  require(array.is_array() && array.size() == count, std::string(name) + ": incorrect array length");
  std::vector<uint64_t> out;
  for (const auto& x : array) {
    require(x.is_number_integer() && (x.is_number_unsigned() || x.get<int64_t>() >= 0),
            std::string(name) + ": expected nonnegative integer");
    out.push_back(x.get<uint64_t>());
  }
  return out;
}

class Decoder final : public Model {
 public:
  explicit Decoder(const ModelFile& file) : Model(file), meta_(file) {}
  const char* load_note() const override {
    return dense_only_ ? "qwen4exp GGUF attention.compress_ratios are all zero. Using full dense attention (sparse QSA disabled)." : nullptr;
  }
  void build(const ModelFile& file, const ModelOptions& options);
  std::vector<flyx_node> prefill_program() const override {
    auto p = trunk_;
    p.push_back({FLYX_NODE_RUN, head_, 1, 0});
    return p;
  }
  std::vector<flyx_node> round_program(int width, int, bool given) const override {
    auto p = trunk_;
    if (width) {
      require(given, "a round with drafts requires supplied token IDs");
      p.insert(p.begin(), {FLYX_NODE_GIVEN, -1, width, 0});
    }
    p.push_back({FLYX_NODE_RUN, head_, 0, 0});
    p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
    p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
    return p;
  }

 private:
  void run(const flyx_block& b) { trunk_.push_back({FLYX_NODE_RUN, tables_.block(b), 0, 0}); }
  void hc(int phase, const std::string& prefix = "");
  void gdn(const std::string& prefix);
  void attention(const std::string& prefix, int ratio);
  void moe(const std::string& prefix);
  void ple(const std::string& prefix);
  GgufMeta meta_;
  int h_ = 0, streams_ = 0, rank_ = 0, wide_ = 0, vocab_ = 0, head_ = -1;
  bool dense_only_ = false;
  double eps_ = 0;
  std::vector<uint64_t> multipliers_, offsets_, moduli_;
  std::vector<flyx_node> trunk_;
};

void Decoder::hc(int phase, const std::string& prefix) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_HC;
  auto& c = b.hc;
  c.phase = phase;
  c.hidden = h_;
  c.streams = streams_;
  c.rank = rank_;
  if (phase == FLYX_HC_MIX || phase == FLYX_HC_COLLAPSE) {
    c.norm = tables_.norm(prefix + "norm.weight", wide_, 0, eps_);
    c.down = tables_.weight(prefix + "down.weight", {rank_, wide_});
    c.up = tables_.weight(prefix + "up.weight", {wide_, rank_});
    c.inject = phase == FLYX_HC_MIX ? tables_.weight(prefix + "inject.weight", {streams_, wide_}) : -1;
  }
  run(b);
}

void Decoder::gdn(const std::string& p) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_GDN;
  auto& g = b.gdn;
  g.body_only = g.out_gate_act = 1;
  g.norm.w = -1;
  g.hidden = h_;
  g.nk = meta_.count("ssm.group_count");
  g.nv = meta_.count("ssm.time_step_rank");
  g.dk = g.dv = meta_.count("ssm.state_size");
  g.conv = meta_.count("ssm.conv_kernel");
  const int v = product(g.nv, g.dv), q = product(2, product(g.nk, g.dk));
  require(g.nv % g.nk == 0 && q <= INT_MAX - v && v == meta_.count("ssm.inner_size"), "inconsistent GDN dimensions");
  const int channels = q + v;
  g.head_order = FLYX_GDN_HEADS_TILED;
  g.decay = FLYX_GDN_DECAY_NEG_EXP;
  g.qkv = tables_.weight(p + "attn_qkv.weight", {channels, h_});
  g.z = tables_.weight(p + "attn_gate.weight", {v, h_});
  g.b = tables_.weight(p + "ssm_beta.weight", {g.nv, h_});
  g.a = tables_.weight(p + "ssm_alpha.weight", {g.nv, h_});
  g.out = tables_.weight(p + "ssm_out.weight", {h_, v});
  g.conv_w = tables_.weight(p + "ssm_conv1d.weight", {channels, g.conv});
  g.A_log = tables_.weight(p + "ssm_a", {g.nv});
  g.dt_bias = tables_.weight(p + "ssm_dt.bias", {g.nv});
  g.out_norm = tables_.norm(p + "ssm_norm.weight", g.dv, 0, eps_);
  flyx_state st{};
  st.kind = FLYX_STATE_GDN;
  st.nk = g.nk;
  st.nv = g.nv;
  st.dk = g.dk;
  st.dv = g.dv;
  st.conv = g.conv;
  st.head_order = g.head_order;
  g.state = tables_.state(st);
  run(b);
}

void Decoder::attention(const std::string& p, int ratio) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_ATTN;
  auto& a = b.attn;
  a.body_only = 1;
  a.norm.w = -1;
  a.hidden = h_;
  a.n_heads = meta_.count("attention.head_count");
  a.n_kv = meta_.count("attention.head_count_kv");
  a.head_dim = meta_.count("attention.key_length");
  a.rot_dim = meta_.count("rope.dimension_count");
  require(!(a.rot_dim % 2) && a.rot_dim <= a.head_dim && a.n_heads % a.n_kv == 0 &&
              a.head_dim == meta_.count("attention.value_length"),
          "inconsistent attention dimensions");
  const auto sections = integers(meta_, "rope.dimension_sections", 4);
  for (int i = 0; i < 3; ++i) {
    require(sections[i] <= uint64_t(a.rot_dim / 2), "invalid RoPE section");
    a.mrope_section[i] = int(sections[i]);
  }
  require(sections[0] + sections[1] + sections[2] == uint64_t(a.rot_dim / 2) && sections[3] == 0,
          "invalid RoPE sections");
  a.mrope_interleaved = 1;
  a.theta = meta_.real_or("rope.freq_base", 10000000);
  a.gate = FLYX_GATE_PER_HEAD;
  const int q = product(a.n_heads, a.head_dim), kv = product(a.n_kv, a.head_dim);
  a.q = tables_.weight(p + "attn_q.weight", {product(2, q), h_});
  a.k = tables_.weight(p + "attn_k.weight", {kv, h_});
  a.v = tables_.weight(p + "attn_v.weight", {kv, h_});
  a.o = tables_.weight(p + "attn_output.weight", {h_, q});
  a.q_norm = tables_.norm(p + "attn_q_norm.weight", a.head_dim, 0, eps_);
  a.k_norm = tables_.norm(p + "attn_k_norm.weight", a.head_dim, 0, eps_);
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.layers = 1;
  st.n_kv = a.n_kv;
  st.head_dim = a.head_dim;
  a.kv = tables_.state(st);
  a.index_heads = meta_.count("attention.indexer.head_count");
  a.index_kv_heads = 1;
  a.index_dim = meta_.count("attention.indexer.key_length");
  a.token_budget = meta_.count("attention.indexer.top_k");
  require(a.index_dim >= a.rot_dim && (!ratio || (ratio >= 2 && a.token_budget % ratio == 0)),
          "invalid indexer dimensions");
  a.index_q = tables_.weight(p + "indexer.q_proj.weight", {product(a.index_heads, a.index_dim), h_});
  a.index_k = tables_.weight(p + "indexer.k_proj.weight", {a.index_dim, h_});
  a.index_q_norm = tables_.norm(p + "indexer.q_norm.weight", a.index_dim, 0, eps_);
  a.index_k_norm = tables_.norm(p + "indexer.k_norm.weight", a.index_dim, 0, eps_);
  if (ratio) {
    a.has_indexer = 1;
    a.group_size = ratio;
    st.kind = FLYX_STATE_INDEXER;
    st.n_kv = 1;
    st.head_dim = a.index_dim;
    a.index_state = tables_.state(st);
  }
  run(b);
}

void Decoder::moe(const std::string& p) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_MOE;
  auto& f = b.moe;
  f.hidden = h_;
  f.body_only = f.norm_weights = 1;
  f.norm.w = f.router_bias = f.res_query = f.post_conv = f.sconv = -1;
  f.route_scale = 1;
  f.n_experts = meta_.count("expert_count");
  f.top_k = meta_.count("expert_used_count");
  f.inter = meta_.count("expert_feed_forward_length");
  require(f.top_k <= f.n_experts, "invalid routed expert count");
  f.router = tables_.weight(p + "ffn_gate_inp.weight", {f.n_experts, h_});
  f.separate_gate_up = !tables_.has(p + "ffn_gate_up_exps.weight");
  require(f.separate_gate_up || (!tables_.has(p + "ffn_gate_exps.weight") &&
                                 !tables_.has(p + "ffn_up_exps.weight")),
          "ambiguous routed gate/up weights");
  if (f.separate_gate_up) {
    f.gate = tables_.weight(p + "ffn_gate_exps.weight", {f.n_experts, f.inter, h_});
    f.up = tables_.weight(p + "ffn_up_exps.weight", {f.n_experts, f.inter, h_});
    f.gate_up = -1;
  } else {
    f.gate_up = tables_.weight(p + "ffn_gate_up_exps.weight", {f.n_experts, product(2, f.inter), h_});
  }
  f.down = tables_.weight(p + "ffn_down_exps.weight", {f.n_experts, h_, f.inter});
  f.shared_inter = meta_.has("expert_shared_feed_forward_length")
                       ? meta_.count("expert_shared_feed_forward_length", 0)
                       : 0;
  if (f.shared_inter) {
    f.has_shared = 1;
    f.shared_gate =
        tables_.weight(p + "ffn_gate_shexp.weight", {f.shared_inter, h_});
    f.shared_up =
        tables_.weight(p + "ffn_up_shexp.weight", {f.shared_inter, h_});
    f.shared_down =
        tables_.weight(p + "ffn_down_shexp.weight", {h_, f.shared_inter});
    f.shared_router = tables_.weight(p + "ffn_gate_inp_shexp.weight", {h_});
  } else {
    for (const char *name :
         {"ffn_gate_shexp.weight", "ffn_up_shexp.weight",
          "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight"})
      require(!tables_.has(p + name),
              "shared weights require a shared intermediate dimension");
  }
  run(b);
}

void Decoder::ple(const std::string& prefix) {
  flyx_block b{};
  b.kind = FLYX_BLOCK_PLE;
  auto& p = b.ple;
  p.hidden = h_;
  p.streams = streams_;
  p.ngram = meta_.count("ple.ngram_size", 2);
  p.heads_per_order = meta_.count("ple.heads_per_ngram");
  const int heads = product(p.ngram - 1, p.heads_per_order);
  p.row_width = meta_.count("embedding_length_per_layer_input");
  const int embed = product(p.row_width, heads);
  p.eos = meta_.count("ple.eos_token_id", 0);
  require(p.eos < vocab_, "PLE EOS outside vocabulary");
  p.taps = meta_.count("ple.conv_kernel", 2);
  p.dilation = p.ngram;
  multipliers_ = integers(meta_, "ple.layer_multipliers", p.ngram);
  offsets_ = integers(meta_, "ple.head_offsets", heads);
  moduli_ = integers(meta_, "ple.head_vocab_sizes", heads);
  const auto* table = tables_.tensor(kPleTable);
  require(table && table->ndim == 2 && table->shape[0] > 0 && table->shape[0] <= INT_MAX, "invalid PLE table");
  for (uint64_t m : multipliers_)
    require((m & 1) && m <= uint64_t(INT64_MAX) / uint64_t(vocab_), "invalid PLE multiplier");
  uint64_t end = 0;
  for (int i = 0; i < heads; ++i) {
    require(offsets_[i] == end && moduli_[i] && moduli_[i] <= uint64_t(table->shape[0]) - end,
            "invalid PLE head range");
    end += moduli_[i];
  }
  p.multipliers = multipliers_.data();
  p.offsets = offsets_.data();
  p.moduli = moduli_.data();
  p.table = tables_.weight(kPleTable, {table->shape[0], p.row_width});
  p.key = tables_.weight(prefix + "ple_key.weight", {wide_, embed});
  p.value = tables_.weight(prefix + "ple_value.weight", {h_, embed});
  p.norm_key = tables_.norm(prefix + "ple_norm_key.weight", wide_, 0, eps_);
  p.norm_query = tables_.norm(prefix + "ple_norm_query.weight", wide_, 0, eps_);
  p.norm_conv = tables_.norm(prefix + "ple_norm_conv.weight", wide_, 0, eps_);
  p.conv_w = tables_.weight(prefix + "ple_conv1d.weight", {wide_, p.taps});
  flyx_state st{};
  st.kind = FLYX_STATE_PLE;
  st.width = p.ngram - 1;
  st.channels = wide_;
  const int history = product(p.taps - 1, p.dilation);
  require(history < INT_MAX, "invalid PLE convolution reach");
  st.conv = history + 1;
  p.state = tables_.state(st);
  run(b);
}

void Decoder::build(const ModelFile& file, const ModelOptions& options) {
  const auto& m = meta_;
  h_ = m.count("embedding_length");
  streams_ = m.count("hyper_connection.count", 2);
  rank_ = m.count("hyper_connection.low_rank");
  wide_ = product(h_, streams_);
  const auto* token = tables_.tensor("token_embd.weight");
  require(h_ <= kMaxWidth && token && token->ndim == 2 && token->shape[0] <= kMaxVocab, "invalid embedding dimensions");
  vocab_ = int(token->shape[0]);
  eps_ = m.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  trained_ctx_ = m.count("context_length");
  kv_fp8_by_default_ = false;
  std::string err;
  std::vector<Drafter> requested;
  if (!choose_drafters(options, {}, &requested, &err) ||
      !tables_.set_activations(options.activations, declared_activations(file), options.fp4_inputs, &err))
    throw std::runtime_error(err);
  const int n = m.count("block_count"), interval = m.count("full_attention_interval");
  require(n <= kMaxLayers, "too many decoder layers");
  const auto ratios = integers(m, "attention.compress_ratios", n);
  dense_only_ = std::all_of(ratios.begin(), ratios.end(), [](uint64_t r) { return r == 0; });
  int ple_layer = -1;
  if (m.has("ple.layers") && !m.at("ple.layers").empty()) {
    const auto layers = integers(m, "ple.layers", 1);
    require(layers[0] < uint64_t(n) && (layers[0] + 1) % interval != 0, "invalid PLE layer");
    ple_layer = int(layers[0]);
  }
  require((ple_layer >= 0) == tables_.has(kPleTable), "incomplete PLE table metadata");
  flyx_block emb{};
  emb.kind = FLYX_BLOCK_EMBED;
  emb.embed = {h_, vocab_, tables_.weight("token_embd.weight", {vocab_, h_})};
  run(emb);
  hc(FLYX_HC_EXPAND);
  for (int i = 0; i < n; ++i) {
    const auto p = "blk." + std::to_string(i) + ".";
    if (i == ple_layer) ple(p);
    hc(FLYX_HC_MIX, p + "hc_attn_");
    if ((i + 1) % interval == 0) {
      require(ratios[i] <= INT_MAX, "invalid QSA ratio");
      attention(p, int(ratios[i]));
    } else {
      require(ratios[i] == 0, "QSA ratio on recurrent layer");
      gdn(p);
    }
    hc(FLYX_HC_SCATTER);
    hc(FLYX_HC_MIX, p + "hc_ffn_");
    moe(p);
    hc(FLYX_HC_SCATTER);
  }
  hc(FLYX_HC_COLLAPSE, "output_hc_");
  flyx_block head{};
  head.kind = FLYX_BLOCK_HEAD;
  head.head.hidden = h_;
  head.head.vocab = vocab_;
  head.head.norm.w = -1;
  head.head.w = tables_.weight("output.weight", {vocab_, h_});
  head_ = tables_.block(head);
  finish(h_, vocab_);
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& options, std::string*) {
  auto model = std::make_unique<Decoder>(file);
  model->build(file, options);
  return model;
}

}  // namespace flyx::models::qwen4exp
