// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpine from a "draft" config of kind "dspine" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/dspine.py): per layer L<j>.{in_norm, post_norm, attn.*, mlp.*, bridge.read,
// bridge.write}, once fc, hidden_norm, norm and bridge.{codebook, gate, pred, prev, gate_bias}.
#include "models/drafters/dspine.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/modelfile/modelfile.h"
#include "models/drafters/dflash2.h"
#include "models/drafters/qwen3_layer.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DSpine drafter: " + what);
}

}  // namespace

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
