// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// EAGLE-4 from a "draft" config of kind "eagle4" and its draft.* tensors, the names python/convert writes
// (python/convert/flyx_convert/eagle4.py): fc, fc_norm.<i> (one a tap), L0.{in_norm, hidden_norm, post_norm, attn.q,
// attn.k, attn.v, attn.o, mlp.gate, mlp.up, mlp.down}, norm, head, cand_down, cand_up.
#include "models/drafters/eagle4.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "core/modelfile/modelfile.h"
#include "models/drafters/dflash2.h"

namespace flyx::models::drafters {
namespace {

using nlohmann::json;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("EAGLE-4 drafter: " + what);
}

}  // namespace

Eagle4Spec parse_eagle4_json(const json& d, const TargetSpec& t, int companion) {
  check_config_target(d, t);
  Eagle4Spec spec{};
  spec.companion = companion;
  flyx_eagle4_block& b = spec.block;
  const std::string title = "EAGLE-4";
  b.hidden = config_int(d, "hidden", title);
  const int layers = config_int(d, "layers", title);
  b.n_heads = config_int(d, "n_heads", title);
  b.n_kv = config_int(d, "n_kv", title);
  b.head_dim = config_int(d, "head_dim", title);
  b.inter = config_int(d, "inter", title);
  b.window = config_int(d, "window", title);
  b.cand_rank = config_int(d, "cand_rank", title);
  b.n_cands = config_int(d, "n_cands", title);
  b.theta = d.at("rope_theta").get<double>();
  spec.eps = d.at("eps").get<double>();
  for (int64_t tap : d.at("taps").get<std::vector<int64_t>>()) {
    // Its taps are residual rows (the outputs of target layers): none after the last layer, which flyx's taps take
    // after the final norm.
    expect(tap >= 1 && tap < t.layers,
           "tap after layer " + std::to_string(tap) + " of " + std::to_string(t.layers) + " (1.." +
               std::to_string(t.layers - 1) + ")");
    spec.taps.push_back(int(tap));
  }
  b.n_taps = int32_t(spec.taps.size());
  b.target_hidden = t.hidden;
  b.vocab = t.vocab;
  // Its head's rows as its file has them (python/convert writes "target" for every EAGLE-4): the target's
  // vocabulary, or the target's padded one.
  spec.rows = d.at("target").at("vocab").get<int>();
  expect(spec.rows == t.vocab || spec.rows == t.padded_vocab(), "a vocabulary of its own");
  expect(layers == 1, "one layer (it has " + std::to_string(layers) + ")");
  expect(b.n_taps >= 1 && b.n_taps <= FLYX_EAGLE4_MAX_TAPS,
         "1.." + std::to_string(FLYX_EAGLE4_MAX_TAPS) + " taps");
  expect(b.hidden == t.hidden, "hidden size " + std::to_string(b.hidden) + " is not the target's " +
                                   std::to_string(t.hidden));
  expect(b.hidden > 0 && b.hidden <= kMaxWidth && b.n_heads > 0 && b.n_heads <= kMaxHeads && b.n_kv > 0 &&
             b.n_heads % b.n_kv == 0 && b.head_dim > 0 && b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim &&
             b.inter > 0 && b.inter <= kMaxWidth && b.window >= 0 && spec.eps > 0 && std::isfinite(spec.eps) &&
             b.theta > 0 && std::isfinite(b.theta) &&
             int64_t(b.n_taps) * t.hidden <= INT_MAX && int64_t(b.n_heads + 2 * b.n_kv) * b.head_dim <= INT_MAX,
         "invalid dimensions");
  expect(b.cand_rank > 0 && b.cand_rank <= kMaxWidth && b.n_cands >= 1 && b.n_cands <= t.vocab,
         "candidate head outside rank 1.." + std::to_string(kMaxWidth) + ", 1..vocab candidates");
  return spec;
}

int build_eagle4(Tables& T, const Eagle4Spec& spec, const TargetSpec& t) {
  flyx_eagle4_block b = spec.block;
  const int H = b.hidden, q = b.n_heads * b.head_dim, kv = b.n_kv * b.head_dim, I = b.inter;
  const double eps = spec.eps;
  // The file's name for a drafter tensor ("draft.fc"): in the model file, or in its drafter file.
  auto dn = [&](const std::string& name) {
    return spec.companion >= 0 ? ModelFile::companion_name(spec.companion, name) : name;
  };
  b.embed = t.embed;
  b.fc = T.weight(dn("draft.fc"), {H, int64_t(b.n_taps) * t.hidden});
  for (int i = 0; i < FLYX_EAGLE4_MAX_TAPS; ++i)
    b.fc_norm[i] = i < b.n_taps ? T.norm(dn("draft.fc_norm." + std::to_string(i)), t.hidden, 0.0, eps)
                                : flyx_norm{-1, 0, 0.0, 0.0};
  b.in_norm = T.norm(dn("draft.L0.in_norm"), H, 0.0, eps);
  b.hidden_norm = T.norm(dn("draft.L0.hidden_norm"), H, 0.0, eps);
  b.post_norm = T.norm(dn("draft.L0.post_norm"), H, 0.0, eps);
  b.q = T.weight(dn("draft.L0.attn.q"), {q, 2 * H});
  b.k = T.weight(dn("draft.L0.attn.k"), {kv, 2 * H});
  b.v = T.weight(dn("draft.L0.attn.v"), {kv, 2 * H});
  b.o = T.weight(dn("draft.L0.attn.o"), {H, q});
  b.gate = T.weight(dn("draft.L0.mlp.gate"), {I, H});
  b.up = T.weight(dn("draft.L0.mlp.up"), {I, H});
  b.down = T.weight(dn("draft.L0.mlp.down"), {H, I});
  b.norm = T.norm(dn("draft.norm"), H, 0.0, eps);
  b.head = T.weight(dn("draft.head"), {spec.rows, H});
  b.cand_down = T.weight(dn("draft.cand_down"), {b.cand_rank, H});
  b.cand_up = T.weight(dn("draft.cand_up"), {spec.rows, b.cand_rank});
  flyx_state st{};
  st.kind = FLYX_STATE_KV;
  st.draft = 1;
  st.layers = 1;
  st.n_kv = b.n_kv;
  st.head_dim = b.head_dim;
  b.kv = T.state(st);  // every position's: its window is the attention's (flyx_eagle4_block.window)
  flyx_state ring{};
  ring.kind = FLYX_STATE_RING;
  ring.draft = 1;
  ring.width = H;
  b.ring = T.state(ring);
  b.float_weights = draft_weights_beside_float(Drafter::kEagle4);
  flyx_block blk{};
  blk.kind = FLYX_BLOCK_EAGLE4;
  blk.eagle4 = b;
  return T.block(blk);
}

}  // namespace flyx::models::drafters
