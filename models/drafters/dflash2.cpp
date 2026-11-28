// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/drafters/dflash2.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <climits>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

#include "models/drafters/qwen3_layer.h"
#include "models/gguf.h"

#include "models/drafters/directory.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

// An arch "dflash" GGUF header: the drafter reads the target's embedding and head, so its width is the target's.
void check_gguf_target(const GgufMeta& cm, const TargetSpec& target) {
  if (cm.count("embedding_length") != target.hidden) throw std::runtime_error(mismatch_msg(target));
  if (cm.has("vocab_size")) {
    const int dv = cm.count("vocab_size");
    if (dv != target.vocab && dv != target.padded_vocab()) throw std::runtime_error(mismatch_msg(target));
  }
}

int parse_softcap_mode(const std::string& mode) {
  if (mode == "before_correction") return FLYX_DFLASH2_SOFTCAP_BEFORE;
  if (mode == "after_correction") return FLYX_DFLASH2_SOFTCAP_AFTER;
  throw std::runtime_error("unsupported DFlash2 selector_softcap_mode: " + mode);
}

// The rows of the selector's codebooks: a row a token, or a row a row of the target's padded embedding.  A file
// without the tensor is refused by name when the block is built.
int resolve_sel_vocab(const flyx_tensor_view* t, const TargetSpec& target) {
  if (!t || t->ndim != 2) return target.vocab;
  if (t->shape[0] != target.vocab && t->shape[0] != target.padded_vocab())
    throw std::runtime_error(mismatch_msg(target));
  return int(t->shape[0]);
}

// A drafter tensor's name in an arch "dflash" GGUF file, from the name python/convert gives it ("draft.fc",
// "draft.L<i>.attn.q"): its layers' projections and norms are named as a dense target's (models/gguf.h).
std::string gguf_draft_name(const std::string& name) {
  static const std::map<std::string_view, const char*> top = {
      {"draft.fc", "fc.weight"},
      {"draft.hidden_norm", "enc.output_norm.weight"},
      {"draft.norm", "output_norm.weight"},
      {"draft.sel.pred", "selector_predecessor.weight"},
      {"draft.sel.succ", "selector_successor.weight"},
      {"draft.sel.hid", "selector_hidden.weight"}};
  static const std::map<std::string_view, const char*> conv = {
      {"attn_conv.proj", "attn_conv_proj.weight"},
      {"attn_conv.base", "attn_conv_base"},
      {"mlp_conv.proj", "ffn_conv_proj.weight"},
      {"mlp_conv.base", "ffn_conv_base"}};
  if (const auto t = top.find(name); t != top.end()) return t->second;
  const size_t dot = name.find('.', 7);  // "draft.L<i>."
  const std::string_view suffix = std::string_view(name).substr(dot + 1);
  const char* dense = gguf_dense_layer_name(suffix, "ffn_norm.weight");
  return "blk." + name.substr(7, dot - 7) + "." + (dense ? dense : conv.at(suffix));
}

void validate_dflash2_spec(DFlash2Spec* spec, const TargetSpec& target) {
  flyx_dflash2_block& d = spec->block;
  // A round's rows: a block from the row after the anchor's (draft_from 1) has block - 1 drafts, one from the
  // anchor's row (draft_from 0) a draft every row.
  const int most = FLYX_MAX_ROWS - (1 - d.draft_from);
  const char* what = drafter_title(spec->kind);
  if (d.block < 2 || d.block > most)
    throw std::runtime_error(std::string(what) + " block size outside 2.." + std::to_string(most));
  if (d.hidden != target.hidden) throw std::runtime_error(std::string(what) + " hidden size does not match target");
  const int max_vocab = target.padded_vocab();
  if (d.hidden <= 0 || spec->layers <= 0 || d.n_heads <= 0 || d.n_kv <= 0 || d.head_dim <= 0 || d.head_dim % 2 != 0 ||
      d.inter <= 0 || d.n_heads % d.n_kv != 0 || d.conv_taps < 0 || d.sel_rank < 0 || d.sel_rank > kMaxWidth ||
      d.sel_topk <= 0 || d.sel_topk > target.vocab || d.mask_token < 0 || d.mask_token >= max_vocab ||
      !(spec->eps > 0) || !std::isfinite(spec->eps) || !(d.theta > 0) || !std::isfinite(d.theta) ||
      !std::isfinite(d.softcap) || d.softcap < 0.0f ||
      d.hidden > kMaxWidth || d.inter > kMaxWidth || spec->layers > kMaxLayers || d.n_heads > kMaxHeads ||
      d.head_dim > kMaxHeadDim || d.conv_taps > kMaxHeadDim || spec->taps.empty() ||
      spec->taps.size() > size_t(INT_MAX) || d.window < 0)
    throw std::runtime_error("invalid DFlash2 drafter config");
  if (d.sel_rank == 0 && d.conv_taps > 0)  // a block without a selector is DFlash's: plain layers
    throw std::runtime_error(std::string(what) + " without a selector has no convolutions");
  if (spec->sel_vocab != target.vocab && spec->sel_vocab != max_vocab)
    throw std::runtime_error(mismatch_msg(target));
  d.n_taps = int32_t(spec->taps.size());
  // No selector (sel_rank 0: the walk takes the head's candidates by their logits) has no hidden gate either.
  if (d.sel_rank == 0 && spec->sel_gate) throw std::runtime_error(std::string(what) + " gate without a selector");
  if (d.conv_group < 1 || d.hidden % d.conv_group)
    throw std::runtime_error("DFlash2 conv group does not divide hidden");
  const int64_t tg2 = int64_t(2) * d.conv_taps * (d.hidden / d.conv_group);
  const int64_t tap_width = int64_t(d.n_taps) * target.hidden;
  const int64_t ctx_width = int64_t(2) * spec->layers * d.n_kv * d.head_dim;
  if (tg2 > INT_MAX || tap_width > INT_MAX || ctx_width > INT_MAX)
    throw std::runtime_error("invalid DFlash2 drafter config");
  spec->tg2 = int(tg2);
}

// python/convert's canonical_dspark_config(config, target=None): DeepSpec's Qwen3DSparkModel.
json dspark_config(const json& sc) {
  const int64_t rank = sc.at("markov_rank").get<int64_t>();
  std::string head = sc.at("markov_head_type").get<std::string>();
  std::transform(head.begin(), head.end(), head.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (rank <= 0 || head != "vanilla") throw std::runtime_error("DSpark: a vanilla Markov head (markov_rank > 0) is supported");
  std::vector<int64_t> taps;
  for (const auto& i : sc.at("target_layer_ids")) {
    if (i.get<int64_t>() < 0) throw std::runtime_error("DSpark: a tap of the embedding (target layer -1) is not supported");
    taps.push_back(i.get<int64_t>() + 1);
  }
  if (sc.contains("layer_types") && sc["layer_types"].is_array())
    for (const auto& t : sc["layer_types"])
      if (t != "full_attention") throw std::runtime_error("DSpark layer types: full attention is supported");
  json d = layer_fields(sc);
  d["kind"] = "dspark";
  d["block"] = sc.at("block_size");
  d["mask_token"] = sc.at("mask_token_id");
  d["taps"] = taps;
  d["conv_taps"] = 0;
  d["conv_group"] = 1;
  d["sel_rank"] = rank;
  d["sel_topk"] = std::min<int64_t>(1024, sc.at("vocab_size").get<int64_t>());  // python/convert's DSPARK_CANDIDATES
  d["window"] = 0;
  d["draft_from"] = 0;
  d["sel_gate"] = false;
  d["target"] = {{"hidden", d["hidden"]}, {"vocab", sc.at("vocab_size")},
                 {"layers", sc.value("num_target_layers", json())}};
  return d;
}
// python/convert's canonical_draft_config(config, target=None).  DFlash (z-lab's DFlashDraftModel, what DFlash2 grew
// from): DFlash2's block without convolutions (conv_taps 0) or selector (sel_rank 0, sel_topk 1: each row's draft the
// head's best), block_size beside its dflash_config.
json draft_config(const json& sc, const std::string& kind) {
  if (kind == "dspark") return dspark_config(sc);
  json dc, tl;
  std::vector<int64_t> taps;
  int64_t vocab, target_hidden;
  json causal;  // null: causal where the layers slide (z-lab's code)
  const bool dflash = kind == "dflash";
  if (sc.contains("dflash_config")) {
    if (kind != "dflash2" && !dflash)
      throw std::runtime_error("drafter architectures " + sc.value("architectures", json::array()).dump() +
                               ": no block drafter flyx reads");
    dc = sc.at("dflash_config"), tl = sc;
    for (const char* k : {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"})
      if (dflash && dc.contains(k)) throw std::runtime_error(std::string("DFlash drafter: DFlash2's ") + k);
    // target_layer_ids name the layers whose outputs the drafter reads (hidden_states[id + 1] in z-lab's code), so
    // every id is below the target's layer count.  A list that reaches num_target_layers counts the hidden states
    // themselves instead (0 the embedding, the last the final norm's output): Inco's drafter for Inkling-2, whose
    // ids [5, 23, 41, 48] of 48 visits its reference hands, as they are, to the capture DSpark's ids go to.
    std::vector<int64_t> ids;
    for (const auto& i : dc.at("target_layer_ids")) ids.push_back(i.get<int64_t>());
    const json n_target = sc.value("num_target_layers", json());
    const bool hidden_states =
        n_target.is_number_integer() && std::find(ids.begin(), ids.end(), n_target.get<int64_t>()) != ids.end();
    for (int64_t id : ids) taps.push_back(hidden_states ? id : id + 1);
    vocab = tl.at("vocab_size").get<int64_t>(), target_hidden = tl.at("hidden_size").get<int64_t>();
    causal = sc.value("is_causal", json());
  } else {
    if (sc.value("speculators_model_type", "") != "dflash2")
      throw std::runtime_error("drafter type " + sc.value("speculators_model_type", json()).dump() +
                               " is not supported (need dflash2)");
    if (sc.value("sample_from_anchor", false)) throw std::runtime_error("sample_from_anchor=True is not supported");
    dc = sc, tl = sc.at("transformer_layer_config");
    for (const auto& i : sc.at("aux_hidden_state_layer_ids")) taps.push_back(i.get<int64_t>());
    const json dv = sc.value("draft_vocab_size", json());
    vocab = dv.is_number_integer() && dv.get<int64_t>() != 0 ? dv.get<int64_t>() : tl.at("vocab_size").get<int64_t>();
    const json th = sc.value("target_hidden_size", json());
    target_hidden = th.is_number() && th.get<int64_t>() != 0 ? th.get<int64_t>() : tl.at("hidden_size").get<int64_t>();
    causal = false;
  }
  json d = layer_fields(tl);
  d["kind"] = kind;
  const int64_t layers = d["layers"].get<int64_t>();
  std::vector<std::string> types;
  if (tl.contains("layer_types") && tl["layer_types"].is_array() && !tl["layer_types"].empty())
    types = tl["layer_types"].get<std::vector<std::string>>();
  else
    types.assign(size_t(layers), "full_attention");
  int64_t window = 0;
  auto all = [&](const char* t) { return std::all_of(types.begin(), types.end(), [&](const std::string& x) { return x == t; }); };
  if (all("full_attention")) {
    window = 0;
  } else if (all("sliding_attention") && tl.value("sliding_window", json()).is_number() && tl["sliding_window"].get<int64_t>()) {
    window = tl["sliding_window"].get<int64_t>();
  } else {
    throw std::runtime_error("draft layer types: all full or all sliding with one window");
  }
  const bool is_causal = causal.is_null() ? window > 0 : causal.get<bool>();
  if (is_causal) throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");
  d["block"] = dflash ? sc.at("block_size") : dc.at("block_size");
  d["mask_token"] = dc.at("mask_token_id");
  d["taps"] = taps;
  d["conv_taps"] = dflash ? json(0) : dc.value("conv_kernel_size", json(2));
  d["conv_group"] = dflash ? json(1) : dc.value("conv_group_size", json(16));
  d["sel_rank"] = dflash ? json(0) : dc.value("selector_rank", json(256));
  d["sel_topk"] = dflash ? json(1) : dc.value("selector_top_k", json(16));
  d["window"] = window;
  d["draft_from"] = 1;
  d["sel_gate"] = !dflash;
  if (dc.contains("final_logit_softcapping") && dc["final_logit_softcapping"].is_number()) {
    const double sc_val = dc["final_logit_softcapping"].get<double>();
    if (!std::isfinite(sc_val) || sc_val < 0.0) throw std::runtime_error("draft final_logit_softcapping out of range");
    if (sc_val > 0.0) d["softcap"] = sc_val;
  }
  if (dc.contains("selector_softcap_mode") && dc["selector_softcap_mode"].is_string()) {
    const std::string sm = dc["selector_softcap_mode"].get<std::string>();
    if (sm != "before_correction" && sm != "after_correction")
      throw std::runtime_error("unsupported draft selector_softcap_mode: " + sm);
    d["softcap_mode"] = sm;
  }
  d["target"] = {{"hidden", target_hidden}, {"vocab", vocab}, {"layers", sc.value("num_target_layers", json())}};
  return d;
}

}  // namespace

// python/convert's _D_SKIP and _D_RULES: the drafter's tensor names as the engine knows them, or "" to leave out.
// layers_only: a drafter without the convolutions and selector (DFlash's, DSpine's layers), whose names they are not.
bool canonical_layer_tensor(const std::string& name, std::string* out, bool layers_only) {
  static const std::regex skip(R"(^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding|)"
                               R"(confidence_head\.).*)");
  static const std::vector<std::pair<std::regex, std::string>> rules = {
      {std::regex(R"(fc\.weight)"), "fc"},
      {std::regex(R"(hidden_norm\.weight)"), "hidden_norm"},
      {std::regex(R"(norm\.weight)"), "norm"},
      {std::regex(R"(layers\.(\d+)\.input_layernorm\.weight)"), "L$1.in_norm"},
      {std::regex(R"(layers\.(\d+)\.post_attention_layernorm\.weight)"), "L$1.post_norm"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight)"), "L$1.attn.$2"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k)_norm\.weight)"), "L$1.attn.$2_norm"},
      {std::regex(R"(layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight)"), "L$1.mlp.$2"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.base_kernel)"), "L$1.$2_conv.base"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.kernel_projection\.weight)"), "L$1.$2_conv.proj"},
      {std::regex(R"(candidate_selector\.predecessor_codebook)"), "sel.pred"},
      {std::regex(R"(candidate_selector\.successor_codebook)"), "sel.succ"},
      {std::regex(R"(candidate_selector\.hidden_projection\.weight)"), "sel.hid"},
      {std::regex(R"(markov_head\.markov_w1\.weight)"), "sel.pred"},  // DSpark: W1[previous token]
      {std::regex(R"(markov_head\.markov_w2\.weight)"), "sel.succ"},  // W2: [vocab, rank]
  };
  const std::string inner = inner_name(name);
  if (std::regex_match(inner, skip)) return out->clear(), true;
  for (const auto& [re, fmt] : rules) {
    if (layers_only && (fmt.find("conv") != std::string::npos || fmt.rfind("sel.", 0) == 0)) continue;
    std::smatch m;
    if (!std::regex_match(inner, m, re)) continue;
    std::string s = m.format(fmt);
    const size_t at = s.find("attention_conv");
    if (at != std::string::npos) s.replace(at, 14, "attn_conv");
    *out = s;
    return true;
  }
  return false;
}
DirectoryDraft dflash_directory(const ModelFile::Contents& raw) {
  DirectoryDraft out;
  out.config = draft_config(raw.directory_config, raw.header.at("draft").at("kind"));
  const auto& draft = out.config;
  for (const auto& v : raw.tensors) {
    std::string name;
    if (!canonical_layer_tensor(v.name, &name, draft.at("kind") == "dflash"))
      throw std::runtime_error("unknown draft tensor " + std::string(v.name));
    if (!name.empty()) out.add(name, v);
  }
  for (const char* n : {"fc", "hidden_norm", "norm"}) out.require(n);
  if (draft.at("sel_rank").get<int64_t>() > 0)
    for (const char* n : {"sel.pred", "sel.succ"}) out.require(n);
  if (draft.at("sel_gate").get<bool>()) out.require("sel.hid");
  for (int64_t j = 0; j < draft.at("layers").get<int64_t>(); ++j) {
    const std::string prefix = "L" + std::to_string(j) + ".";
    require_layer(out, prefix);
    if (draft.at("conv_taps").get<int64_t>() > 0)
      for (const char* n : {"attn_conv.base", "attn_conv.proj", "mlp_conv.base", "mlp_conv.proj"}) out.require(prefix + n);
  }
  return out;
}

bool is_block_companion(const json& header) {
  if (!header.is_object()) return false;
  if (header.value("kind", "") == "drafter" && header.contains("draft") && header["draft"].is_object()) return true;
  if (!GgufMeta::is_header(header) || header.value("arch", "") != "dflash") return false;
  const GgufMeta cm(header);
  return !(cm.has("has_markov_context") && cm.at("has_markov_context").is_boolean() &&
           cm.at("has_markov_context").get<bool>());
}

void check_made_for(const json& header, const TargetSpec& target) {
  if (GgufMeta::is_header(header)) check_gguf_target(GgufMeta(header), target);
  else check_config_target(header.at("draft"), target);
}

DFlash2Spec parse_dflash2_json(const ModelFile& file, const json& j, const TargetSpec& target, int companion,
                               const flyx_tensor_view* selector) {
  check_config_target(j, target);
  DFlash2Spec spec{};
  spec.gguf = false;
  spec.companion = companion;
  flyx_dflash2_block& d = spec.block;
  Qwen3BlockConfig c = parse_qwen3_block(j, target, drafter_title(block_drafter_kind(j)));
  d.hidden = c.hidden, spec.layers = c.layers, d.n_heads = c.n_heads, d.n_kv = c.n_kv, d.head_dim = c.head_dim;
  d.inter = c.inter, d.block = c.block, d.mask_token = c.mask_token, d.theta = c.theta, spec.eps = c.eps;
  spec.taps = std::move(c.taps);
  d.conv_taps = config_int(j, "conv_taps", "DFlash2");
  d.conv_group = config_int(j, "conv_group", "DFlash2");
  d.sel_rank = config_int(j, "sel_rank", "DFlash2");
  d.sel_topk = config_int(j, "sel_topk", "DFlash2");
  d.window = config_int(j, "window", "DFlash2");  // below zero: validate_dflash2_spec refuses it
  const double sc = j.contains("softcap") && j["softcap"].is_number() ? j["softcap"].get<double>() : 0.0;
  if (!std::isfinite(sc) || sc < 0.0) throw std::runtime_error("DFlash2 softcap out of range");
  d.softcap = float(sc);
  d.softcap_mode = j.contains("softcap_mode") && j["softcap_mode"].is_string()
                       ? parse_softcap_mode(j["softcap_mode"].get<std::string>())
                       : FLYX_DFLASH2_SOFTCAP_BEFORE;
  const std::string pred_name =
      companion >= 0 ? ModelFile::companion_name(companion, "draft.sel.pred") : std::string("draft.sel.pred");
  spec.sel_vocab = resolve_sel_vocab(selector ? selector : file.tensor(pred_name), target);
  spec.kind = block_drafter_kind(j);
  // A DFlash2 file written before the two keys holds the block the converter still writes for DFlash2: drafts from
  // the row after the anchor's, a gated selector.  A DSpark file states both.
  const bool before_keys = spec.kind == Drafter::kDFlash2 && !j.contains("draft_from") && !j.contains("sel_gate");
  const int64_t from = before_keys ? 1 : config_int(j, "draft_from", "DFlash2");
  if (from != 0 && from != 1) throw std::runtime_error("block drafter draft_from " + std::to_string(from));
  d.draft_from = int32_t(from);
  if (!before_keys && (!j.contains("sel_gate") || !j["sel_gate"].is_boolean()))
    throw std::runtime_error("block drafter sel_gate missing");
  spec.sel_gate = before_keys || j["sel_gate"].get<bool>();
  validate_dflash2_spec(&spec, target);
  return spec;
}

DFlash2Spec parse_dflash2_gguf(const ModelFile& file, const json& header, const TargetSpec& target, int companion) {
  const GgufMeta cm(header);
  const json& g = header.at("gguf");
  check_gguf_target(cm, target);
  // What the layout can say that the block here does not run: refused, never drafted as the plain block.
  auto set = [&](const char* key) { return cm.has(key) && cm.at(key).is_boolean() && cm.at(key).get<bool>(); };
  if (set("has_markov_context"))
    throw std::runtime_error("DSpark drafters (dflash.has_markov_context = true) use the DSpark drafter path");
  if (set("attention.causal"))
    throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");
  if (set("sample_from_anchor")) throw std::runtime_error("sample_from_anchor=True is not supported");
  if (set("has_confidence_head"))
    throw std::runtime_error("a DFlash2 confidence head (dflash.has_confidence_head) is not supported");

  DFlash2Spec spec{};
  spec.gguf = true;
  spec.companion = companion;
  flyx_dflash2_block& d = spec.block;
  d.draft_from = 1;
  d.hidden = cm.count("embedding_length");
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
  if (cm.has("final_logit_softcapping")) {
    if (!cm.at("final_logit_softcapping").is_number())
      throw std::runtime_error("DFlash2 GGUF final_logit_softcapping must be a number");
    const double sc = cm.at("final_logit_softcapping").get<double>();
    if (!std::isfinite(sc) || sc < 0.0) throw std::runtime_error("DFlash2 GGUF final_logit_softcapping out of range");
    d.softcap = float(sc);
  }
  if (d.softcap > 0.0f) {
    if (cm.has("selector_softcap_mode") && cm.at("selector_softcap_mode").is_string())
      d.softcap_mode = parse_softcap_mode(cm.at("selector_softcap_mode").get<std::string>());
    else
      d.softcap_mode = FLYX_DFLASH2_SOFTCAP_BEFORE;  // llama.cpp GGUF default when softcap > 0
  }

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
      throw std::runtime_error(
          "DFlash2 GGUF attention.sliding_window_pattern must be uniform (all full or all sliding)");
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
    // t in [1, target.layers) taps the residual stream after layer t; t == target.layers taps post-output_norm.
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
    throw std::runtime_error(
        "DFlash2 GGUF missing mask_token_id (expected dflash.mask_token_id or tokenizer.ggml.mask_token_id)");
  d.mask_token = mask_token;

  const std::string pred_name = ModelFile::companion_name(companion, "selector_predecessor.weight");
  spec.sel_vocab = resolve_sel_vocab(file.tensor(pred_name), target);
  validate_dflash2_spec(&spec, target);
  return spec;
}

int build_dflash2(Tables& T, const DFlash2Spec& spec, const TargetSpec& target) {
  flyx_dflash2_block d = spec.block;
  const int layers = spec.layers;
  const double eps = spec.eps;
  const int H = d.hidden, TG2 = spec.tg2;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    const std::string in_file = spec.gguf ? gguf_draft_name(name) : name;
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, in_file) : in_file;
  };

  d.embed = target.embed;
  d.head = target.head;
  d.first_layer = int32_t(T.blocks.size());
  d.layers = layers;
  for (int l = 0; l < layers; ++l) {
    const std::string p = "draft.L" + std::to_string(l) + ".";
    flyx_block b{};
    b.kind = FLYX_BLOCK_DFLASH2_LAYER;
    auto& y = b.dflash2_layer;
    bind_qwen3_layer(T, y, dn, l, H, d.n_heads, d.n_kv, d.head_dim, d.inter, eps);
    const bool conv = d.conv_taps > 0;  // none: -1
    y.attn_conv_proj = conv ? T.weight(dn(p + "attn_conv.proj"), {TG2, H}) : -1;
    y.attn_conv_base = conv ? T.weight(dn(p + "attn_conv.base"), {2, d.conv_taps, H}) : -1;
    y.mlp_conv_proj = conv ? T.weight(dn(p + "mlp_conv.proj"), {TG2, H}) : -1;
    y.mlp_conv_base = conv ? T.weight(dn(p + "mlp_conv.base"), {2, d.conv_taps, H}) : -1;
    T.block(b);
  }
  d.fc = T.weight(dn("draft.fc"), {H, int64_t(d.n_taps) * target.hidden});
  d.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  d.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  const bool selector = d.sel_rank > 0;  // none: -1, the head's candidates by their logits
  d.sel_pred = selector ? T.weight(dn("draft.sel.pred"), {spec.sel_vocab, d.sel_rank}) : -1;
  d.sel_succ = selector ? T.weight(dn("draft.sel.succ"), {spec.sel_vocab, d.sel_rank}) : -1;
  d.sel_hid = spec.sel_gate ? T.weight(dn("draft.sel.hid"), {d.sel_rank, H}) : -1;  // -1: a unit gate
  d.kv = draft_kv_state(T, layers, d.n_kv, d.head_dim);
  flyx_block b{};
  b.kind = FLYX_BLOCK_DFLASH2;
  b.dflash2 = d;
  return T.block(b);
}

}  // namespace flyx::models::drafters
