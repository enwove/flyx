// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/dflash2.h"

#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "models/gguf.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

int req_int(const json& j, const char* key) {
  const auto& v = j.at(key);
  if (!v.is_number_integer()) throw std::runtime_error("invalid DFlash2 drafter config");
  const int64_t n = v.get<int64_t>();
  if (n < INT_MIN || n > INT_MAX) throw std::runtime_error("invalid DFlash2 drafter config");
  return int(n);
}

std::string mismatch_msg(const TargetSpec& target) {
  return "a drafter made for another model (this one: width " + std::to_string(target.hidden) + ", vocabulary " +
         std::to_string(target.vocab) + ", " + std::to_string(target.layers) + " layers)";
}

void validate_dflash2_spec(DFlash2Spec* spec, const TargetSpec& target) {
  flyx_dflash2_block& d = spec->block;
  if (d.block < 2 || d.block > FLYX_MAX_ROWS)
    throw std::runtime_error("DFlash2 block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  if (d.hidden != target.hidden) throw std::runtime_error("DFlash2 hidden size does not match target");
  const int max_vocab = target.padded_vocab();
  if (d.hidden <= 0 || spec->layers <= 0 || d.n_heads <= 0 || d.n_kv <= 0 || d.head_dim <= 0 || d.head_dim % 2 != 0 ||
      d.inter <= 0 || d.n_heads % d.n_kv != 0 || d.conv_taps <= 0 || d.sel_rank <= 0 || d.sel_rank > kMaxWidth ||
      d.sel_topk <= 0 || d.sel_topk > target.vocab || d.mask_token < 0 || d.mask_token >= max_vocab ||
      !(spec->eps > 0) || !std::isfinite(spec->eps) || !(d.theta > 0) || !std::isfinite(d.theta) ||
      d.hidden > kMaxWidth || d.inter > kMaxWidth || spec->layers > kMaxLayers || d.n_heads > kMaxHeads ||
      d.head_dim > kMaxHeadDim || d.conv_taps > kMaxHeadDim || spec->taps.empty() ||
      spec->taps.size() > size_t(INT_MAX) || d.window < 0)
    throw std::runtime_error("invalid DFlash2 drafter config");
  if (spec->sel_vocab != target.vocab && spec->sel_vocab != max_vocab)
    throw std::runtime_error(mismatch_msg(target));
  d.n_taps = int32_t(spec->taps.size());
  if (d.conv_group < 1 || d.hidden % d.conv_group)
    throw std::runtime_error("DFlash2 conv group does not divide hidden");
  const int64_t tg2 = int64_t(2) * d.conv_taps * (d.hidden / d.conv_group);
  const int64_t tap_width = int64_t(d.n_taps) * target.hidden;
  const int64_t ctx_width = int64_t(2) * spec->layers * d.n_kv * d.head_dim;
  if (tg2 > INT_MAX || tap_width > INT_MAX || ctx_width > INT_MAX)
    throw std::runtime_error("invalid DFlash2 drafter config");
  spec->tg2 = int(tg2);
}

int resolve_sel_vocab(const ModelFile* file, const std::string& pred_name, const TargetSpec& target) {
  if (file) {
    if (const flyx_tensor_view* t = file->tensor(pred_name)) {
      if (t->ndim == 2 && (t->shape[0] == target.vocab || t->shape[0] == target.padded_vocab()))
        return int(t->shape[0]);
      if (t->ndim == 2) throw std::runtime_error(mismatch_msg(target));
    }
  }
  return target.vocab;
}

}  // namespace

bool is_dflash_companion(const json& header) {
  if (!header.is_object()) return false;
  if (header.value("kind", "") == "drafter" && header.contains("draft") && header["draft"].is_object()) return true;
  return GgufMeta::is_header(header) && header.value("arch", "") == "dflash";
}

DFlash2Spec parse_dflash2_json(const json& j, const TargetSpec& target, int companion, const ModelFile* file) {
  if (j.value("kind", "") != "dflash2") throw std::runtime_error(mismatch_msg(target));
  if (j.contains("target") && j["target"].is_object()) {
    const json& t = j["target"];
    const int th = t.value("hidden", target.hidden);
    const int tv = t.value("vocab", target.vocab);
    const bool vocab_ok = tv == target.vocab || tv == target.padded_vocab();
    const bool layers_ok = !t.contains("layers") || t["layers"].is_null() || t["layers"].get<int>() == target.layers;
    if (th != target.hidden || !vocab_ok || !layers_ok) throw std::runtime_error(mismatch_msg(target));
  }

  DFlash2Spec spec{};
  spec.gguf = false;
  spec.companion = companion;
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
    if (t < 1 || t > target.layers) throw std::runtime_error("DFlash2 tap after layer " + std::to_string(t));
    spec.taps.push_back(int(t));
  }
  d.conv_taps = req_int(j, "conv_taps");
  d.conv_group = req_int(j, "conv_group");
  d.sel_rank = req_int(j, "sel_rank");
  d.sel_topk = req_int(j, "sel_topk");
  d.theta = j.at("rope_theta").get<double>();
  const int64_t win = j.value("window", int64_t(0));
  if (win < 0 || win > INT_MAX) throw std::runtime_error("DFlash2 window out of range");
  d.window = int32_t(win);
  spec.eps = j.at("eps").get<double>();
  const std::string pred_name =
      companion >= 0 ? ModelFile::companion_name(companion, "draft.sel.pred") : std::string("draft.sel.pred");
  spec.sel_vocab = resolve_sel_vocab(file, pred_name, target);
  validate_dflash2_spec(&spec, target);
  return spec;
}

DFlash2Spec parse_dflash2_gguf(const ModelFile& file, const json& header, const TargetSpec& target, int companion) {
  const GgufMeta cm(header);
  const json& g = header.at("gguf");
  if (cm.has("has_markov_context") && cm.at("has_markov_context").is_boolean() &&
      cm.at("has_markov_context").get<bool>())
    throw std::runtime_error("DSpark drafters (dflash.has_markov_context = true) are not supported by this build");
  if (cm.has("attention.causal") && cm.at("attention.causal").is_boolean() && cm.at("attention.causal").get<bool>())
    throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");

  DFlash2Spec spec{};
  spec.gguf = true;
  spec.companion = companion;
  flyx_dflash2_block& d = spec.block;
  d.hidden = cm.count("embedding_length");
  if (d.hidden != target.hidden) throw std::runtime_error(mismatch_msg(target));
  if (cm.has("vocab_size")) {
    const int dv = cm.count("vocab_size");
    if (dv != target.vocab && dv != target.padded_vocab()) throw std::runtime_error(mismatch_msg(target));
  }
  spec.layers = cm.count("block_count");
  d.inter = cm.count("feed_forward_length");
  d.n_heads = cm.count("attention.head_count");
  d.n_kv = cm.count("attention.head_count_kv");
  if (!cm.has("attention.key_length") && d.hidden % d.n_heads)
    throw std::runtime_error("DFlash2 GGUF hidden size is not a whole number of heads");
  d.head_dim = cm.count_or("attention.key_length", d.hidden / d.n_heads);
  if (cm.count_or("attention.value_length", d.head_dim) != d.head_dim)
    throw std::runtime_error("DFlash2 GGUF attention key and value lengths differ");
  d.theta = cm.real_or("rope.freq_base", 10000.0);
  spec.eps = cm.real_or("attention.layer_norm_rms_epsilon", 1e-6);
  d.block = cm.count("block_size");
  d.conv_taps = cm.count_or("conv_kernel_size", 2);
  d.conv_group = cm.count_or("conv_group_size", 16);
  d.sel_rank = cm.count_or("selector_rank", 256);
  d.sel_topk = cm.count_or("selector_top_k", 16);

  int win = cm.count_or("attention.sliding_window", 0, 0);
  if (cm.has("attention.sliding_window_pattern")) {
    const json& pat = cm.at("attention.sliding_window_pattern");
    if (!pat.is_array() || int(pat.size()) != spec.layers)
      throw std::runtime_error("DFlash2 GGUF attention.sliding_window_pattern does not match block_count");
    bool any_true = false, any_false = false;
    for (const auto& flag : pat) {
      if (!flag.is_boolean())
        throw std::runtime_error("DFlash2 GGUF attention.sliding_window_pattern must be booleans");
      if (flag.get<bool>()) any_true = true;
      else any_false = true;
    }
    if (any_true && any_false)
      throw std::runtime_error("DFlash2 GGUF attention.sliding_window_pattern must be uniform (all full or all sliding)");
    if (!any_true) win = 0;
    else if (win <= 0)
      throw std::runtime_error("DFlash2 GGUF sliding_window_pattern enabled without positive attention.sliding_window");
  }
  d.window = win;

  const json& raw_taps = cm.at("target_layers");
  if (!raw_taps.is_array() || raw_taps.empty())
    throw std::runtime_error("DFlash2 GGUF missing or empty dflash.target_layers");
  spec.taps.reserve(raw_taps.size());
  for (const auto& item : raw_taps) {
    if (!item.is_number_integer()) throw std::runtime_error("DFlash2 GGUF target_layers must be integers");
    const int64_t t = item.get<int64_t>();
    if (t < 1 || t > target.layers) throw std::runtime_error("DFlash2 tap after layer " + std::to_string(t));
    spec.taps.push_back(int(t));
  }

  int mask_token = -1;
  if (cm.has("mask_token_id")) {
    mask_token = cm.count("mask_token_id", 0);
  } else if (g.contains("tokenizer.ggml.mask_token_id") && g["tokenizer.ggml.mask_token_id"].is_number_integer()) {
    const int64_t mt = g["tokenizer.ggml.mask_token_id"].get<int64_t>();
    if (mt < 0 || mt > INT_MAX) throw std::runtime_error("DFlash2 GGUF mask_token_id out of range");
    mask_token = int(mt);
  } else {
    mask_token = find_added_token(file, "<|mask|>");
  }
  if (mask_token < 0)
    throw std::runtime_error("DFlash2 GGUF missing mask_token_id (expected dflash.mask_token_id or tokenizer.ggml.mask_token_id)");
  d.mask_token = mask_token;

  const std::string pred_name = ModelFile::companion_name(companion, "selector_predecessor.weight");
  spec.sel_vocab = resolve_sel_vocab(&file, pred_name, target);
  validate_dflash2_spec(&spec, target);
  return spec;
}

int build_dflash2(Tables& T, const DFlash2Spec& spec, const TargetSpec& target) {
  flyx_dflash2_block d = spec.block;
  const int layers = spec.layers;
  const double eps = spec.eps;
  const int H = d.hidden, q = d.n_heads * d.head_dim, kv = d.n_kv * d.head_dim, I = d.inter;
  const int TG2 = spec.tg2;
  auto dn = [&](const std::string& name) -> std::string {
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, name) : name;
  };

  d.embed = target.embed;
  d.head = target.head;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = layers;
  for (int l = 0; l < layers; ++l) {
    flyx_block b{};
    b.kind = FLYX_BLOCK_DFLASH2_LAYER;
    auto& y = b.dflash2_layer;
    if (spec.gguf) {
      const std::string p = dn("blk." + std::to_string(l) + ".");
      y.norm = T.norm(p + "attn_norm.weight", H, 0.0, eps);
      y.post_norm = T.norm(p + "ffn_norm.weight", H, 0.0, eps);
      y.q_norm = T.norm(p + "attn_q_norm.weight", d.head_dim, 0.0, eps);
      y.k_norm = T.norm(p + "attn_k_norm.weight", d.head_dim, 0.0, eps);
      y.q = T.weight(p + "attn_q.weight", {q, H});
      y.k = T.weight(p + "attn_k.weight", {kv, H});
      y.v = T.weight(p + "attn_v.weight", {kv, H});
      y.o = T.weight(p + "attn_output.weight", {H, q});
      y.gate = T.weight(p + "ffn_gate.weight", {I, H});
      y.up = T.weight(p + "ffn_up.weight", {I, H});
      y.down = T.weight(p + "ffn_down.weight", {H, I});
      y.attn_conv_proj = T.weight(p + "attn_conv_proj.weight", {TG2, H});
      y.attn_conv_base = T.weight(p + "attn_conv_base", {2, d.conv_taps, H});
      y.mlp_conv_proj = T.weight(p + "ffn_conv_proj.weight", {TG2, H});
      y.mlp_conv_base = T.weight(p + "ffn_conv_base", {2, d.conv_taps, H});
    } else {
      const std::string p = dn("draft.L" + std::to_string(l) + ".");
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
    }
    T.block(b);
  }

  if (spec.gguf) {
    d.fc = T.weight(dn("fc.weight"), {H, int64_t(d.n_taps) * target.hidden});
    d.hidden_norm = T.norm(dn("enc.output_norm.weight"), H, 0.0, eps);
    d.norm = T.norm(dn("output_norm.weight"), H, 0.0, eps);
    d.sel_pred = T.weight(dn("selector_predecessor.weight"), {spec.sel_vocab, d.sel_rank});
    d.sel_succ = T.weight(dn("selector_successor.weight"), {spec.sel_vocab, d.sel_rank});
    d.sel_hid = T.weight(dn("selector_hidden.weight"), {d.sel_rank, H});
  } else {
    d.fc = T.weight(dn("draft.fc"), {H, int64_t(d.n_taps) * target.hidden});
    d.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
    d.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
    d.sel_pred = T.weight(dn("draft.sel.pred"), {spec.sel_vocab, d.sel_rank});
    d.sel_succ = T.weight(dn("draft.sel.succ"), {spec.sel_vocab, d.sel_rank});
    d.sel_hid = T.weight(dn("draft.sel.hid"), {d.sel_rank, H});
  }

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

}  // namespace flyx::models::drafters
