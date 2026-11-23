// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Qwen3.5 from its model file: the layer math as block flags (attention with
// an output gate, (1 + w) norms and partial rotary; GDN with a plain-weight
// output norm; the DFlash2 drafter with plain norms), and the two ways a file
// stores the same model:
//   .flyx  the config in the header, tensors under the names
//             python/convert writes ("target.L3.attn.q"), laid out for the
//             engine (the gate's rows after q's, value heads grouped by key
//             head, norm weights w, the decay as A_log)
//   GGUF      llama.cpp's metadata keys and tensor names ("blk.3.attn_q.weight"),
//             laid out as its converter writes them: norm weights as 1 + w,
//             each head's gate rows after its q rows, value heads tiled over
//             the key heads, the decay as -exp(A_log).  The description says
//             so in its flags (backend_abi.h); no tensor is rewritten.
#include "models/qwen35/qwen35.h"

#include <climits>
#include <stdexcept>

#include "models/gguf.h"

namespace flyx::models::qwen35 {
namespace {

using nlohmann::json;

struct Config {
  int hidden, layers, vocab, inter;
  int n_heads, n_kv, head_dim, rot_dim;
  int nk, nv, dk, dv, conv;
  double theta, eps;
  std::vector<bool> is_attn;
  int trained_ctx = 0;  // the context the model was made for (0: the file does not say)
  bool gguf = false;  // the file's names and layout (above)
  bool mtp = false, dflash2 = false;  // the drafters the model has
  int mtp_companion = -1;    // the companion (ModelFile::add) holding the MTP layer, -1: the model file
  int draft_companion = -1;  // the companion holding the DFlash2 drafter, -1: the model file
  json draft;                // the DFlash2 drafter's config
};

// A tensor's name in a GGUF file, from the name python/convert gives it; the
// MTP layer's in companion `companion` when >= 0.
std::string gguf_name(const std::string& name, int layers, int companion) {
  if (const char* top = gguf_top_name(name)) return top;
  static const std::map<std::string, std::string> mtp = {
      {"mtp.fc", "nextn.eh_proj.weight"}, {"mtp.norm_emb", "nextn.enorm.weight"},
      {"mtp.norm_hidden", "nextn.hnorm.weight"}, {"mtp.norm", "nextn.shared_head_norm.weight"}};
  static const std::map<std::string, std::string> gdn_layer = {
      {"gdn.qkv", "attn_qkv.weight"}, {"gdn.z", "attn_gate.weight"}, {"gdn.a", "ssm_alpha.weight"},
      {"gdn.b", "ssm_beta.weight"}, {"gdn.out", "ssm_out.weight"}, {"gdn.conv", "ssm_conv1d.weight"},
      {"gdn.A_log", "ssm_a"}, {"gdn.dt_bias", "ssm_dt.bias"}, {"gdn.norm", "ssm_norm.weight"}};
  // The MTP layer is the block after the target's.
  const std::string blk = "blk." + std::to_string(layers) + ".";
  const std::string mtp_block = companion >= 0 ? ModelFile::companion_name(companion, blk) : blk;
  if (auto m = mtp.find(name); m != mtp.end()) return mtp_block + m->second;
  const bool in_mtp = name.starts_with("mtp.L0.");
  const size_t dot = in_mtp ? 6 : name.find('.', 8);  // "target.L<i>."
  const std::string block = in_mtp ? mtp_block : "blk." + name.substr(8, dot - 8) + ".";
  const std::string suffix = name.substr(dot + 1);
  if (const char* mapped = gguf_dense_layer_name(suffix, "post_attention_norm.weight")) return block + mapped;
  return block + gdn_layer.at(suffix);
}

void validate_config(const Config& f) {
  if (f.hidden <= 0 || f.layers <= 0 || f.vocab <= 0 || f.inter <= 0 || f.n_heads <= 0 || f.n_kv <= 0 ||
      f.head_dim <= 0 || f.rot_dim <= 0 || f.rot_dim > f.head_dim || f.rot_dim % 2 != 0 || f.n_heads % f.n_kv != 0 ||
      f.nk <= 0 || f.nv <= 0 || f.dk <= 0 || f.dv <= 0 || f.nv % f.nk != 0 || f.conv <= 0 || !(f.eps > 0) ||
      !(f.theta > 0) || f.hidden > kMaxWidth || f.inter > kMaxWidth || f.vocab > kMaxVocab || f.layers > kMaxLayers ||
      f.n_heads > kMaxHeads || f.head_dim > kMaxHeadDim || f.nk > kMaxHeads || f.nv > kMaxHeads ||
      f.dk > kMaxHeadDim || f.dv > kMaxHeadDim || f.conv > kMaxHeadDim)
    throw std::runtime_error("invalid Qwen3.5 model config dimensions");
}

Config read_gguf_config(const ModelFile& file) {
  const GgufDecoderBase b = read_gguf_decoder_base(file, /*allow_partial_rotary=*/true);
  const GgufMeta m(file);
  Config f{};
  f.gguf = true;
  f.vocab = b.vocab;
  f.hidden = b.hidden;
  const int blocks = b.blocks, mtp = m.count_or("nextn_predict_layers", 0, 0);
  if (mtp > 1 || mtp >= blocks) throw std::runtime_error("GGUF with more than one MTP layer");
  f.mtp = mtp == 1;
  f.layers = blocks - mtp;
  f.inter = b.inter;
  f.n_heads = b.n_heads;
  f.n_kv = b.n_kv;
  f.head_dim = b.head_dim;
  f.rot_dim = b.rot_dim;
  f.theta = b.theta;
  f.eps = b.eps;
  f.trained_ctx = b.trained_ctx;
  f.nk = m.count("ssm.group_count");
  f.nv = m.count("ssm.time_step_rank");
  f.dk = m.count("ssm.state_size");
  const int inner = m.count("ssm.inner_size");
  if (f.nv % f.nk || inner % f.nv) throw std::runtime_error("GGUF gated delta net head dimensions do not divide");
  f.dv = inner / f.nv;
  f.conv = m.count("ssm.conv_kernel");
  if (m.has("attention.recurrent_layers")) {  // one flag per block, the MTP layer's too
    const auto& flags = m.at("attention.recurrent_layers");
    if (!flags.is_array() || int(flags.size()) != blocks)
      throw std::runtime_error("GGUF recurrent layer flags do not match block_count");
    for (int i = 0; i < f.layers; ++i) f.is_attn.push_back(!flags[size_t(i)].get<bool>());
  } else {
    const int interval = m.count("full_attention_interval");
    for (int i = 0; i < f.layers; ++i) f.is_attn.push_back((i + 1) % interval == 0);
  }
  validate_config(f);
  return f;
}

// The drafters in companions (ModelFile::add), after the model file's own:
//   a GGUF of this architecture and width whose one MTP layer is the block after the model's (unsloth ships the
//     27B's so), with a GGUF model: its other tensors (its own embedding and head) are not read, the model's are;
//   a drafter file (python/convert --draft alone) holding a DFlash2 drafter made for a target of this width and
//     vocabulary (and layer count, when it names one), with a model of either format.
void take_companions(const ModelFile& file, Config* f) {
  for (int i = 0; i < file.companions(); ++i) {
    const json& h = file.header().at("companions").at(size_t(i));
    const std::string what = "drafter file " + std::to_string(i + 1) + ": ";
    if (GgufMeta::is_header(h)) {
      if (!f->gguf) throw std::runtime_error(what + "an MTP layer from a GGUF file goes with a GGUF model");
      if (f->mtp) throw std::runtime_error(what + "the model has its MTP layer already");
      const GgufMeta cm(h);
      if (h.value("arch", "") != file.arch() || cm.count("embedding_length") != f->hidden ||
          cm.count_or("nextn_predict_layers", 0, 0) != 1 || cm.count("block_count") != f->layers + 1)
        throw std::runtime_error(what + "no MTP layer for this model (" + file.arch() + ", width " +
                                 std::to_string(f->hidden) + ", " + std::to_string(f->layers) + " layers)");
      f->mtp = true;
      f->mtp_companion = i;
    } else if (h.value("kind", "") == "drafter" && h.contains("draft") && h["draft"].is_object()) {
      if (f->dflash2) throw std::runtime_error(what + "the model has a DFlash2 drafter already");
      const json& d = h["draft"];
      const json t = d.value("target", json::object());
      if (d.value("kind", "") != "dflash2" || t.value("hidden", f->hidden) != f->hidden ||
          t.value("vocab", f->vocab) != f->vocab ||
          (t.contains("layers") && !t["layers"].is_null() && t["layers"].get<int>() != f->layers))
        throw std::runtime_error(what + "a drafter made for another model (this one: width " +
                                 std::to_string(f->hidden) + ", vocabulary " + std::to_string(f->vocab) + ", " +
                                 std::to_string(f->layers) + " layers)");
      f->dflash2 = true;
      f->draft_companion = i;
      f->draft = d;
    } else {
      throw std::runtime_error(what + "holds no drafter this model takes (a GGUF MTP layer or a DFlash2 drafter file)");
    }
  }
}

Config read_config(const json& h) {
  const json& c = h.at("config");
  Config f{};
  f.mtp = h.contains("mtp") && !h["mtp"].is_null();
  f.dflash2 = h.contains("draft") && !h["draft"].is_null();
  if (f.dflash2) f.draft = h["draft"];
  f.hidden = c.at("hidden");
  f.layers = c.at("layers");
  f.vocab = c.at("vocab");
  f.inter = c.at("inter");
  f.n_heads = c.at("n_heads");
  f.n_kv = c.at("n_kv");
  f.head_dim = c.at("head_dim");
  f.rot_dim = c.at("rot_dim");
  f.nk = c.at("lin_nk");
  f.nv = c.at("lin_nv");
  f.dk = c.at("lin_dk");
  f.dv = c.at("lin_dv");
  f.conv = c.at("conv_k");
  f.theta = c.at("rope_theta");
  f.eps = c.at("eps");
  f.trained_ctx = c.value("max_pos", 0);
  const auto lt = c.at("layer_types").get<std::vector<std::string>>();
  if (int(lt.size()) != f.layers) throw std::runtime_error("layer_types does not match layers");
  for (const auto& t : lt) {
    if (t != "attn" && t != "gdn") throw std::runtime_error("unknown layer_types entry: " + t);
    f.is_attn.push_back(t == "attn");
  }
  validate_config(f);
  return f;
}

class Qwen35 final : public Model {
 public:
  explicit Qwen35(const ModelFile& file) : Model(file) {}
  bool build(const ModelFile& file, const ModelOptions& o, std::string* err);
  std::vector<flyx_node> prefill_program() const override;
  std::vector<flyx_node> round_program(int width, int d, bool given) const override;

 private:
  int attention(const std::string& p, const Config& c, bool draft);
  int gdn(const std::string& p, const Config& c);
  int mlp(const std::string& p, const Config& c);
  int dflash2(const json& d, const Config& c);
  void add_layers(std::vector<flyx_node>* p) const;
  void add_contexts(std::vector<flyx_node>* p) const;  // every drafter takes the pass's rows

  // The file's name for a tensor, and the offset of its (offset + w) norms.
  std::string n(const std::string& name) const { return gguf_ ? gguf_name(name, layers_, mtp_companion_) : name; }
  Tables::NameFn name_fn() const { return [this](const std::string& name) { return n(name); }; }
  // A DFlash2 drafter's tensor ("draft.fc"): in the model file, or in its drafter file.
  std::string dn(const std::string& name) const {
    return draft_companion_ >= 0 ? ModelFile::companion_name(draft_companion_, name) : name;
  }
  double one() const { return gguf_ ? 0.0 : 1.0; }

  bool gguf_ = false;
  int mtp_companion_ = -1, draft_companion_ = -1;
  int layers_ = 0;
  int embed_ = -1, head_ = -1, dflash2_block_ = -1;
  std::vector<int> mixer_, mlp_;
  std::vector<int> taps_;  // DFlash2: features from the residual stream after taps_[s] layers
};

// "<p>attn.*" (p = "target.L3." or "mtp.L0."): q carries the output gate.
int Qwen35::attention(const std::string& p, const Config& c, bool draft) {
  const AttnBlockSpec spec{c.hidden, c.n_heads, c.n_kv, c.head_dim, c.rot_dim,
                           gguf_ ? FLYX_GATE_PER_HEAD : FLYX_GATE_AFTER_Q, c.theta, one(), c.eps, draft};
  return tables_.attn_block(p, spec, name_fn());
}

int Qwen35::gdn(const std::string& p, const Config& c) {
  Tables& T = tables_;
  const int H = c.hidden, C = 2 * c.nk * c.dk + c.nv * c.dv;
  flyx_state st{};
  st.kind = FLYX_STATE_GDN;
  st.nk = c.nk;
  st.nv = c.nv;
  st.dk = c.dk;
  st.dv = c.dv;
  st.conv = c.conv;
  st.head_order = gguf_ ? FLYX_GDN_HEADS_TILED : FLYX_GDN_HEADS_GROUPED;
  flyx_block b{};
  b.kind = FLYX_BLOCK_GDN;
  auto& g = b.gdn;
  g.hidden = H;
  g.nk = c.nk;
  g.nv = c.nv;
  g.dk = c.dk;
  g.dv = c.dv;
  g.conv = c.conv;
  g.head_order = st.head_order;
  g.decay = gguf_ ? FLYX_GDN_DECAY_NEG_EXP : FLYX_GDN_DECAY_LOG;
  g.norm = T.norm(n(p + "in_norm"), H, one(), c.eps);
  g.qkv = T.weight(n(p + "gdn.qkv"), {C, H});
  g.z = T.weight(n(p + "gdn.z"), {c.nv * c.dv, H});
  g.b = T.weight(n(p + "gdn.b"), {c.nv, H});
  g.a = T.weight(n(p + "gdn.a"), {c.nv, H});
  g.out = T.weight(n(p + "gdn.out"), {H, c.nv * c.dv});
  g.conv_w = T.weight(n(p + "gdn.conv"), {C, c.conv});
  g.A_log = T.weight(n(p + "gdn.A_log"), {c.nv});
  g.dt_bias = T.weight(n(p + "gdn.dt_bias"), {c.nv});
  g.out_norm = T.norm(n(p + "gdn.norm"), c.dv, 0.0, c.eps);
  g.state = T.state(st);
  return T.block(b);
}

int Qwen35::mlp(const std::string& p, const Config& c) {
  return tables_.mlp_block(p, c.hidden, c.inter, one(), c.eps, name_fn());
}

// Parse and validate DFlash2 drafter config before binding any tensors so mismatched hidden sizes
// (which share the target's embed/head) and 32-bit dimension-product overflows are always refused early.
struct DFlash2Spec {
  flyx_dflash2_block block{};
  int layers = 0;
  int tg2 = 0;
  double eps = 0.0;
  std::vector<int> taps;
};

int req_int(const json& j, const char* key) {
  const int64_t v = j.at(key).get<int64_t>();
  if (v < INT_MIN || v > INT_MAX) throw std::runtime_error("invalid DFlash2 drafter config");
  return int(v);
}

DFlash2Spec parse_dflash2_spec(const json& j, const Config& c) {
  DFlash2Spec spec{};
  flyx_dflash2_block& d = spec.block;
  d.hidden = req_int(j, "hidden");
  spec.layers = req_int(j, "layers");
  d.n_heads = req_int(j, "n_heads");
  d.n_kv = req_int(j, "n_kv");
  d.head_dim = req_int(j, "head_dim");
  d.inter = req_int(j, "inter");
  d.block = req_int(j, "block");
  d.mask_token = req_int(j, "mask_token");
  const auto raw_taps = j.at("taps").get<std::vector<int64_t>>();
  spec.taps.reserve(raw_taps.size());
  for (int64_t t : raw_taps) {
    if (t < 1 || t > c.layers) throw std::runtime_error("DFlash2 tap after layer " + std::to_string(t));
    spec.taps.push_back(int(t));
  }
  d.conv_taps = req_int(j, "conv_taps");
  d.conv_group = req_int(j, "conv_group");
  d.sel_rank = req_int(j, "sel_rank");
  d.sel_topk = req_int(j, "sel_topk");
  d.theta = j.at("rope_theta");
  const int64_t win = j.value("window", int64_t(0));
  if (win < 0 || win > INT_MAX) throw std::runtime_error("DFlash2 window out of range");
  d.window = int32_t(win);
  spec.eps = j.at("eps");
  if (d.block < 2 || d.block > FLYX_MAX_ROWS)
    throw std::runtime_error("DFlash2 block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  if (d.hidden != c.hidden) throw std::runtime_error("DFlash2 hidden size does not match target");
  if (d.hidden <= 0 || spec.layers <= 0 || d.n_heads <= 0 || d.n_kv <= 0 || d.head_dim <= 0 || d.head_dim % 2 != 0 ||
      d.inter <= 0 || d.n_heads % d.n_kv != 0 || d.conv_taps <= 0 || d.sel_rank <= 0 || d.sel_rank > kMaxWidth ||
      d.sel_topk <= 0 || d.sel_topk > c.vocab || d.mask_token < 0 || d.mask_token >= c.vocab || !(spec.eps > 0) ||
      !(d.theta > 0) || d.hidden > kMaxWidth || d.inter > kMaxWidth || spec.layers > kMaxLayers ||
      d.n_heads > kMaxHeads || d.head_dim > kMaxHeadDim || d.conv_taps > kMaxHeadDim || spec.taps.empty() ||
      spec.taps.size() > size_t(INT_MAX))
    throw std::runtime_error("invalid DFlash2 drafter config");
  d.n_taps = int32_t(spec.taps.size());
  if (d.conv_group < 1 || d.hidden % d.conv_group) throw std::runtime_error("DFlash2 conv group does not divide hidden");
  const int64_t tg2 = int64_t(2) * d.conv_taps * (d.hidden / d.conv_group);
  const int64_t tap_width = int64_t(d.n_taps) * c.hidden;
  const int64_t ctx_width = int64_t(2) * spec.layers * d.n_kv * d.head_dim;
  if (tg2 > INT_MAX || tap_width > INT_MAX || ctx_width > INT_MAX)
    throw std::runtime_error("invalid DFlash2 drafter config");
  spec.tg2 = int(tg2);
  return spec;
}

int Qwen35::dflash2(const json& j, const Config& c) {
  Tables& T = tables_;
  DFlash2Spec spec = parse_dflash2_spec(j, c);
  flyx_dflash2_block d = spec.block;
  const int layers = spec.layers;
  const double eps = spec.eps;
  taps_ = std::move(spec.taps);
  const int H = d.hidden, q = d.n_heads * d.head_dim, kv = d.n_kv * d.head_dim, I = d.inter;
  const int TG2 = spec.tg2;
  d.embed = embed_;
  d.head = head_;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = layers;
  for (int l = 0; l < layers; ++l) {
    const std::string p = dn("draft.L" + std::to_string(l) + ".");
    flyx_block b{};
    b.kind = FLYX_BLOCK_DFLASH2_LAYER;
    auto& y = b.dflash2_layer;
    y.norm = T.norm(p + "in_norm", H, 0.0, eps);
    y.post_norm = T.norm(p + "post_norm", H, 0.0, eps);
    y.q_norm = T.norm(p + "attn.q_norm", d.head_dim, 0.0, eps);
    y.k_norm = T.norm(p + "attn.k_norm", d.head_dim, 0.0, eps);
    y.q = T.weight(p + "attn.q", {q, H});
    y.k = T.weight(p + "attn.k", {kv, H});
    y.v = T.weight(p + "attn.v", {kv, H});
    y.o = T.weight(p + "attn.o", {H, q});
    y.gate = T.weight(p + "mlp.gate", {I, H});
    y.up = T.weight(p + "mlp.up", {I, H});
    y.down = T.weight(p + "mlp.down", {H, I});
    y.attn_conv_proj = T.weight(p + "attn_conv.proj", {TG2, H});
    y.attn_conv_base = T.weight(p + "attn_conv.base", {2, d.conv_taps, H});
    y.mlp_conv_proj = T.weight(p + "mlp_conv.proj", {TG2, H});
    y.mlp_conv_base = T.weight(p + "mlp_conv.base", {2, d.conv_taps, H});
    T.block(b);
  }
  d.fc = T.weight(dn("draft.fc"), {H, int64_t(d.n_taps) * c.hidden});
  d.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  d.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  d.sel_pred = T.weight(dn("draft.sel.pred"), {c.vocab, d.sel_rank});
  d.sel_succ = T.weight(dn("draft.sel.succ"), {c.vocab, d.sel_rank});
  d.sel_hid = T.weight(dn("draft.sel.hid"), {d.sel_rank, H});
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = layers;
  st.n_kv = d.n_kv;
  st.head_dim = d.head_dim;
  d.kv = T.state(st);
  flyx_block b{};
  b.kind = FLYX_BLOCK_DFLASH2;
  b.dflash2 = d;
  return T.block(b);
}

bool Qwen35::build(const ModelFile& file, const ModelOptions& o, std::string* err) {
  const json& h = file.header();
  Config c = GgufMeta::is(file) ? read_gguf_config(file) : read_config(h);
  take_companions(file, &c);
  trained_ctx_ = c.trained_ctx;
  gguf_ = c.gguf;
  mtp_companion_ = c.mtp_companion;
  draft_companion_ = c.draft_companion;
  layers_ = c.layers;
  std::vector<Drafter> drafters;
  if (!choose_drafters(o, {c.mtp, c.dflash2, c.mtp_companion >= 0, c.draft_companion >= 0}, &drafters, err))
    return false;
  for (Drafter kind : drafters)
    if (kind == Drafter::kDFlash2) (void)parse_dflash2_spec(c.draft, c);
  if (!tables_.set_activations(o.activations, declared_activations(file), o.fp4_inputs, err)) return false;
  Tables& T = tables_;
  const auto nf = name_fn();
  embed_ = T.embed_block(c.hidden, c.vocab, nf);
  head_ = T.head_block(c.hidden, c.vocab, embed_, one(), c.eps, nf);
  for (int i = 0; i < c.layers; ++i) {
    const std::string p = "target.L" + std::to_string(i) + ".";
    mixer_.push_back(c.is_attn[size_t(i)] ? attention(p, c, false) : gdn(p, c));
    mlp_.push_back(mlp(p, c));
  }
  for (Drafter kind : drafters) {
    if (kind == Drafter::kMtp) {
      flyx_block b{};
      b.kind = FLYX_BLOCK_MTP;
      auto& m = b.mtp;
      m.embed = embed_;
      m.head = head_;
      m.attn = attention("mtp.L0.", c, true);
      m.mlp = mlp("mtp.L0.", c);
      m.norm_emb = T.norm(n("mtp.norm_emb"), c.hidden, one(), c.eps);
      m.norm_hidden = T.norm(n("mtp.norm_hidden"), c.hidden, one(), c.eps);
      m.norm = T.norm(n("mtp.norm"), c.hidden, one(), c.eps);
      m.fc = T.weight(n("mtp.fc"), {c.hidden, 2 * c.hidden});
      flyx_state ring{};
      ring.kind = FLYX_STATE_RING;
      ring.draft = 1;
      ring.width = c.hidden;
      m.ring = T.state(ring);
      add_drafter(kind, T.block(b), FLYX_MAX_ROWS - 1);
    } else {
      dflash2_block_ = dflash2(c.draft, c);
      add_drafter(kind, dflash2_block_, c.draft.at("block").get<int>() - 1);
    }
  }
  finish(c.hidden, c.vocab);
  return true;
}

void Qwen35::add_layers(std::vector<flyx_node>* p) const {
  for (size_t i = 0; i < mixer_.size(); ++i) {
    p->push_back({FLYX_NODE_RUN, mixer_[i], 0, 0});
    p->push_back({FLYX_NODE_RUN, mlp_[i], 0, 0});
    if (dflash2_block_ >= 0)  // its features, whichever drafter drafts: its context stays current
      for (size_t s = 0; s < taps_.size(); ++s)
        if (taps_[s] == int(i) + 1) p->push_back({FLYX_NODE_TAP, dflash2_block_, int32_t(s), 0});
  }
}

void Qwen35::add_contexts(std::vector<flyx_node>* p) const {
  for (const DrafterBlock& d : drafters_) p->push_back({FLYX_NODE_CONTEXT, d.block, 0, 0});
}

std::vector<flyx_node> Qwen35::prefill_program() const {
  std::vector<flyx_node> p;
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  add_contexts(&p);
  p.push_back({FLYX_NODE_RUN, head_, 1, 0});
  return p;
}

std::vector<flyx_node> Qwen35::round_program(int width, int d, bool given) const {
  std::vector<flyx_node> p;
  if (width > 0) p.push_back(given ? flyx_node{FLYX_NODE_GIVEN, -1, width, 0}
                                   : flyx_node{FLYX_NODE_DRAFT, drafters_.at(size_t(d)).block, width, 0});
  p.push_back({FLYX_NODE_RUN, embed_, 0, 0});
  add_layers(&p);
  p.push_back({FLYX_NODE_RUN, head_, 0, 0});
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  add_contexts(&p);
  return p;
}

}  // namespace

std::unique_ptr<Model> open(const ModelFile& file, const ModelOptions& o, std::string* err) {
  auto m = std::make_unique<Qwen35>(file);
  return m->build(file, o, err) ? std::move(m) : nullptr;
}

}  // namespace flyx::models::qwen35
