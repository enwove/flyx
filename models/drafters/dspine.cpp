// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine from a "draft" config of kind "dspine" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/dspine.py): per layer L<j>.{in_norm, post_norm, attn.*, mlp.*, bridge.read,
// bridge.write}, once fc, hidden_norm, norm and bridge.{codebook, gate, pred, prev, gate_bias}.
#include "models/drafters/dspine.h"

#include <algorithm>
#include <cstring>
#include <regex>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/modelfile/modelfile.h"
#include "models/drafters/config.h"
#include "models/drafters/qwen3_layer.h"

#include "models/drafters/directory.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DSpine drafter: " + what);
}

// python/convert/flyx_convert/dspine.py's canonical_dspine_config(config, target=None): the variant both released
// drafters are (Tencent's DSpineDraftModel), anything else refused.
json dspine_config(const json& sc) {
  expect(sc.contains("dflash_config") && sc["dflash_config"].is_object(), "no dflash_config");
  const json& dc = sc["dflash_config"];
  const int64_t layers = sc.at("num_hidden_layers").get<int64_t>();
  expect(layers > 0 && layers <= 1024, "num_hidden_layers out of range");
  // Each key read where Tencent's two released configs give it, and nowhere else.
  expect(sc.value("model_type", json()) == "qwen3", "model_type (qwen3 layers)");
  expect(sc.value("hidden_act", json()) == "silu", "hidden_act (silu)");
  expect(sc.value("attention_bias", json()) == json(false), "attention biases");
  expect(dc.value("draft_block_causal", json()) == json(true), "draft_block_causal must be true");
  expect(dc.value("selected_prefix_terminal", json()) == json(false), "a terminal write");
  expect(dc.value("selected_prefix_decode_mode", "") == "serial", "decode mode (serial)");
  expect(dc.value("selected_prefix_feature_writes", json()) == json(true) &&
                     dc.value("selected_prefix_feature_write_mode", "") == "independent",
                 "independent writes after the layers");
  json every = json::array();
  for (int64_t i = 1; i <= layers; ++i) every.push_back(i);
  expect(dc.value("selected_prefix_feature_layers", json()) == every, "a write after every layer");
  expect(falsy(dc.value("selected_prefix_terminal_residual_rank", json())), "a terminal residual");
  expect(falsy(dc.value("selected_prefix_feature_next_token_rows", json())), "next-token rows");
  expect(dc.value("selected_prefix_feature_reliability", json()).is_null(), "feature reliability weights");
  expect(dc.value("attention_value_scale", json()).is_null(), "an attention value scale");
  for (const char* k : {"conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k", "projector_type",
                        "attention_sink_bias"})
    expect(falsy(dc.value(k, json())), std::string(k) + " (DFlash2's parts)");
  std::vector<int64_t> widths;
  // The refinement's weight and the bridge's kappa are training's (the refinement loss's weight, the code's scale in
  // training): Tencent's own readers check them (Tencent/DSpine sglang_dspine/models/dspine.py, dspine_mlx/config.py)
  // and no inference reads them; these readers take the same keys and values, and nothing else.
  const json refine = dc.value("selected_prefix_last_write_refine", json());
  if (!refine.is_null()) {
    const auto keys = {"kind", "version", "head_topk", "weight"};
    expect(refine.is_object() && refine.size() == keys.size() &&
                       std::all_of(keys.begin(), keys.end(), [&](const char* k) { return refine.contains(k); }) &&
                       refine["kind"] == "exact_sender" && refine["version"] == 1 &&
                       refine["head_topk"].is_number_integer() && refine["weight"].is_number() &&
                       refine["weight"].get<double>() >= 0,
                   "last-write refinement (exact_sender, version 1, head_topk, a weight >= 0)");
    widths.push_back(refine["head_topk"].get<int64_t>());
  }
  const json second = sc.value("code_bridge_second_pass", json());
  if (!second.is_null()) {
    expect(second.value("mode", "") == "serial_table" && second.value("first_exact", int64_t(-1)) == layers - 1,
                   "second pass (serial_table, the last layer's write)");
    widths.push_back(second.at("head_topk").get<int64_t>());
  }
  expect(!widths.empty() &&
                     std::all_of(widths.begin(), widths.end(), [&](int64_t w) { return w == widths[0]; }),
                 "refinement widths (one, given at least once)");
  expect(widths[0] >= 2 && widths[0] <= 16, "refinement width (2..16)");
  const json bridge = dc.value("selected_prefix_code_bridge", json());
  expect(bridge.is_object() && bridge.value("code_size", json()).is_number_integer() &&
                     bridge.value("rank", json()).is_number_integer() && bridge.value("eps", json()).is_number() &&
                     std::all_of(bridge.items().begin(), bridge.items().end(),
                                 [](const auto& e) {
                                   return e.key() == "code_size" || e.key() == "rank" || e.key() == "eps" ||
                                          (e.key() == "kappa" && (e.value().is_null() || e.value().is_number()));
                                 }),
                 "selected_prefix_code_bridge (code_size, rank, eps; kappa: training's)");
  const json rp = sc.value("rope_parameters", json());
  expect(rp.is_object() && rp.value("rope_type", json()) == "default" &&
                     rp.value("rope_theta", json()).is_number() && sc.value("rope_scaling", json()).is_null(),
                 "rope_parameters (default rotary, its theta)");
  const json lt = sc.value("layer_types", json());
  expect(lt.is_array() && lt.size() == size_t(layers), "layer_types (one a layer)");
  const std::vector<std::string> types = lt.get<std::vector<std::string>>();
  auto all = [&](const char* t) {
    return std::all_of(types.begin(), types.end(), [&](const std::string& x) { return x == t; });
  };
  int64_t window = 0;
  if (all("full_attention")) {
    expect(sc.value("use_sliding_window", json()) == json(false), "use_sliding_window without sliding layers");
  } else {
    const json w = sc.value("sliding_window", json());
    expect(all("sliding_attention") && sc.value("use_sliding_window", json()) == json(true) &&
                       sc.value("max_window_layers", json()) == json(layers) && w.is_number_integer() &&
                       w.get<int64_t>() > 0,
                   "layer types: all full, or all sliding with one window");
    window = w.get<int64_t>();
  }
  std::vector<int64_t> taps;
  for (const auto& i : dc.at("target_layer_ids")) taps.push_back(i.get<int64_t>() + 1);
  const int64_t vocab = sc.at("vocab_size").get<int64_t>();
  expect(sc.value("draft_vocab_size", json()) == json(vocab), "a draft vocabulary of its own");
  expect(sc.value("tie_word_embeddings", json()) == json(false), "tied embeddings (it reads the target's head)");
  json d = layer_fields(sc);
  d["kind"] = "dspine";
  d["block"] = sc.at("block_size");
  d["mask_token"] = dc.at("mask_token_id");
  d["taps"] = taps;
  d["window"] = window;
  d["code"] = bridge.at("code_size");
  d["rank"] = bridge.at("rank");
  d["bridge_eps"] = bridge.at("eps").get<double>();
  d["topk"] = widths[0];
  d["target"] = {{"hidden", d["hidden"]}, {"vocab", vocab}, {"layers", sc.value("num_target_layers", json())}};
  return d;
}

// A DSpine drafter's tensors that only say which variant it is: each holds 1 (flyx_convert/dspine.py's VERSIONS).
bool dspine_version(const std::string& inner) {
  return inner == "draft_block_causal_version" || inner == "last_write_refine_version" ||
         inner == "selected_prefix_bridge.independent_write_version" ||
         inner == "selected_prefix_bridge.terminal_disabled_version";
}
// Its terminal write's projection, which a drafter without one holds as zeros.
constexpr const char* kTerminal = "selected_prefix_bridge.out.weight";
// Its stacked tensors: 1 the last layer's read (read_weight), 2 the other layers' (early_read_weights), 3 every
// layer's write (feature_write_weights); 0 none.
int dspine_stacked(const std::string& inner) {
  if (inner == "selected_prefix_bridge.read_weight") return 1;
  if (inner == "selected_prefix_bridge.early_read_weights") return 2;
  if (inner == "selected_prefix_bridge.feature_write_weights") return 3;
  return 0;
}

// The same for a DSpine drafter (flyx_convert/dspine.py's _LAYER_RULES and _BRIDGE_NAMES): its layers' tensors as
// DFlash2's are named, the bridge's own; "" for a tensor that only says the variant (checked by the caller), or a
// stacked one the caller cuts (read_weight, early_read_weights, feature_write_weights).
bool canonical_dspine(const std::string& name, std::string* out) {
  static const std::map<std::string, std::string> bridge = {
      {"selected_prefix_bridge.codebook", "bridge.codebook"}, {"selected_prefix_bridge.gate.weight", "bridge.gate"},
      {"selected_prefix_bridge.gate.bias", "bridge.gate_bias"},
      {"selected_prefix_bridge.predecessor_gate.weight", "bridge.pred"},
      {"selected_prefix_bridge.prev.weight", "bridge.prev"}};
  if (auto it = bridge.find(inner_name(name)); it != bridge.end()) return *out = it->second, true;
  return canonical_layer_tensor(name, out, /*layers_only=*/true);
}

}  // namespace

DirectoryDraft dspine_directory(const ModelFile::Contents& raw) {
  DirectoryDraft out;
  out.config = dspine_config(raw.directory_config);
  for (const auto& v : raw.tensors) {
    const std::string inner = inner_name(v.name);
    const int stacked = dspine_stacked(inner);
    if (dspine_version(inner)) {
      // one integer of any width holding 1 (little-endian: its low bytes first)
      const uint64_t width =
          v.dtype == FLYX_DT_I64 ? 8 : v.dtype == FLYX_DT_I32 ? 4 : v.dtype == FLYX_DT_U8 ? 1 : 0;
      int64_t one = 0;
      if (!width || v.nbytes != width || (std::memcpy(&one, v.data, size_t(width)), one != 1))
        throw std::runtime_error("DSpine drafter: " + inner + " is not 1");
    } else if (inner == kTerminal) {
      // a sign bit alone (an element's top byte 0x80, the rest 0) is a zero too
      const auto* b = static_cast<const uint8_t*>(v.data);
      if (v.dtype != FLYX_DT_F64 && v.dtype != FLYX_DT_F32 && v.dtype != FLYX_DT_F16 && v.dtype != FLYX_DT_BF16)
        throw std::runtime_error("DSpine drafter: terminal projection must be floating point");
      const uint64_t es = v.dtype == FLYX_DT_F64 ? 8 : v.dtype == FLYX_DT_F32 ? 4 : 2;
      for (uint64_t i = 0; i < v.nbytes; ++i)
        if (b[i] && !(i % es == es - 1 && b[i] == 0x80))
          throw std::runtime_error("DSpine drafter: a terminal projection that is not zero");
    } else if (stacked) {
      // [n, ...] -> n views of one layer each; the last layer's read stands alone (read_weight)
      const int64_t layers = out.config.at("layers").get<int64_t>();
      const bool read = stacked != 3, last = stacked == 1;
      const int64_t n = last ? 1 : v.shape[0];
      if (!last && (v.ndim != 3 || n != (read ? layers - 1 : layers)))
        throw std::runtime_error("DSpine drafter: " + inner + " is not one per layer");
      if (n == 0) continue;  // a drafter of one layer: no layer before the last reads
      flyx_tensor_view part = v;
      if (!last) {
        part.ndim = 2;
        part.shape[0] = v.shape[1];
        part.shape[1] = v.shape[2];
        part.shape[2] = 0;
        part.nbytes = v.nbytes / uint64_t(n);
      }
      for (int64_t j = 0; j < n; ++j) {
        part.data = static_cast<const uint8_t*>(v.data) + uint64_t(j) * part.nbytes;
        out.add("L" + std::to_string(last ? layers - 1 : j) + (read ? ".bridge.read" : ".bridge.write"), part);
      }
    } else {
      std::string canon;
      if (!canonical_dspine(v.name, &canon)) throw std::runtime_error("unknown draft tensor " + std::string(v.name));
      if (!canon.empty()) out.add(canon, v);
    }
  }
  for (const char* n : {"fc", "hidden_norm", "norm", "bridge.codebook", "bridge.gate", "bridge.gate_bias",
                        "bridge.pred", "bridge.prev"}) out.require(n);
  for (int64_t j = 0; j < out.config.at("layers").get<int64_t>(); ++j) {
    const std::string prefix = "L" + std::to_string(j) + ".";
    require_layer(out, prefix);
    out.require(prefix + "bridge.read");
    out.require(prefix + "bridge.write");
  }
  return out;
}

DSpineSpec parse_dspine_json(const json& d, const TargetSpec& t, int companion) {
  check_config_target(d, t);
  DSpineSpec spec{};
  spec.companion = companion;
  flyx_dspine_block& b = spec.block;
  Qwen3BlockConfig c = parse_qwen3_block(d, t, "DSpine");
  b.hidden = c.hidden, spec.layers = c.layers, b.n_heads = c.n_heads, b.n_kv = c.n_kv, b.head_dim = c.head_dim;
  b.inter = c.inter, b.block = c.block, b.mask_token = c.mask_token, b.theta = c.theta, spec.eps = c.eps;
  spec.taps = std::move(c.taps);
  b.n_taps = int32_t(spec.taps.size());
  b.code = config_int(d, "code", "DSpine");
  b.rank = config_int(d, "rank", "DSpine");
  b.topk = config_int(d, "topk", "DSpine");
  b.window = config_int(d, "window", "DSpine");
  b.bridge_eps = d.at("bridge_eps").get<double>();
  expect(b.block >= 2 && b.block <= FLYX_MAX_ROWS, "block size outside 2.." + std::to_string(FLYX_MAX_ROWS));
  expect(b.topk >= 2 && b.topk <= FLYX_DSPINE_MAX_TOPK && b.topk <= t.vocab,
         "the walk's candidates outside 2.." + std::to_string(FLYX_DSPINE_MAX_TOPK));
  expect(b.window >= 0 && b.code > 0 && b.code <= kMaxWidth && b.rank > 0 && b.rank <= kMaxWidth &&
             int64_t(3) * b.rank <= INT_MAX && b.bridge_eps > 0 && std::isfinite(b.bridge_eps),
         "invalid dimensions");
  return spec;
}

int build_dspine(Tables& T, const DSpineSpec& spec, const TargetSpec& t) {
  flyx_dspine_block b = spec.block;
  const int H = b.hidden, C = b.code, R = b.rank;
  const double eps = spec.eps;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, name) : name;
  };
  b.embed = t.embed;
  b.head = t.head;
  b.first_layer = int32_t(T.blocks.size());
  b.layers = spec.layers;
  for (int l = 0; l < spec.layers; ++l) {
    const std::string p = "draft.L" + std::to_string(l) + ".";
    flyx_block blk{};
    blk.kind = FLYX_BLOCK_DSPINE_LAYER;
    auto& y = blk.dspine_layer;
    bind_qwen3_layer(T, y, dn, l, H, b.n_heads, b.n_kv, b.head_dim, b.inter, eps);
    y.read = T.weight(dn(p + "bridge.read"), {C, H});
    y.write = T.weight(dn(p + "bridge.write"), {H, R});
    T.block(blk);
  }
  b.fc = T.weight(dn("draft.fc"), {H, int64_t(b.n_taps) * t.hidden});
  b.hidden_norm = T.norm(dn("draft.hidden_norm"), H, 0.0, eps);
  b.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  b.codebook = T.weight(dn("draft.bridge.codebook"), {t.vocab, C});
  b.gate = T.weight(dn("draft.bridge.gate"), {R, C});
  b.pred = T.weight(dn("draft.bridge.pred"), {R, C});
  b.prev = T.weight(dn("draft.bridge.prev"), {R, C});
  b.gate_bias = T.weight(dn("draft.bridge.gate_bias"), {R});
  b.kv = draft_kv_state(T, spec.layers, b.n_kv, b.head_dim);
  flyx_block blk{};
  blk.kind = FLYX_BLOCK_DSPINE;
  blk.dspine = b;
  return T.block(blk);
}

}  // namespace flyx::models::drafters
