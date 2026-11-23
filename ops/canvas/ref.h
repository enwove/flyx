// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// DV's canvas (bifill-dv's drafter, docs/prior-art.md; python/reference/flyx_ref/dv.py): N slots after the anchor,
// run through the target's base weights with the canvas adapter.  What goes in (canvas_input), what a pass leaves
// (canvas_stats), and how the canvas moves on by the tokens a round commits (slide).  Slot w sits at the anchor's
// position + 1 + w; a round's trees (ops/draft/ref.h) read the slid canvas.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "ops/activation/ref.h"
#include "ops/dtype/ref.h"
#include "ops/linear/ref.h"
#include "ops/sampling/ref.h"

namespace flyx::ops {

// A slot's input (bifill-dv dv.py CanvasAdapter.embed), for N slots of hidden width D:
//   x_w = (tokens[w] == mask ? mask_emb : embed[tokens[w]]) + res_out (silu(g) * u),   [u | g] = res_in z_w
// res_in [2D, D] (u: its first D outputs), res_out [D, D], z [N, D] the slot's expected embedding from the round before
// (0 for a new slot, which adds nothing: silu(0) = 0).
inline void canvas_input(int slots, int hidden, const int32_t* tokens, int32_t mask, const double* z,
                         const flyx_tensor_view* embed, const flyx_tensor_view* mask_emb,
                         const flyx_tensor_view* res_in, const flyx_tensor_view* res_out, double* x) {
  const size_t D = size_t(hidden);
  std::vector<double> ug(size_t(slots) * 2 * D), a(size_t(slots) * D);
  linear(z, slots, hidden, res_in, 2 * hidden, ug.data());
  for (int w = 0; w < slots; ++w) {
    const double* u = ug.data() + size_t(w) * 2 * D;  // then g = u + D
    for (size_t i = 0; i < D; ++i) a[size_t(w) * D + i] = silu(u[D + i]) * u[i];
  }
  linear(a.data(), slots, hidden, res_out, hidden, x);
  for (int w = 0; w < slots; ++w) {
    const bool m = tokens[w] == mask;
    for (size_t i = 0; i < D; ++i)
      x[size_t(w) * D + i] += m ? load(mask_emb, i) : load(embed, size_t(tokens[w]) * D + i);
  }
}

// What a canvas pass leaves, a slot (a row of `vocab` logits) at a time: the log-sum-exp; the k highest (ties to the
// lower id) as ids[0..k) and their probabilities under the whole vocabulary p = exp(logit - lse), not renormalised;
// their mass sum p; and the expected embedding zn = sum_c p_c embed[ids_c] (hidden values), the slot's residual
// context in the next round.
inline void canvas_stats(const double* logits, int vocab, int k, const flyx_tensor_view* embed, int hidden,
                         int32_t* ids, double* p, double* lse, double* mass, double* zn) {
  double m = logits[0];
  for (int v = 1; v < vocab; ++v) m = std::max(m, logits[v]);
  double s = 0;
  for (int v = 0; v < vocab; ++v) s += std::exp(logits[v] - m);
  *lse = m + std::log(s);
  std::vector<int> idx(static_cast<size_t>(k));
  std::vector<double> val(static_cast<size_t>(k));
  topk(logits, vocab, k, idx.data(), val.data());
  *mass = 0;
  for (int c = 0; c < k; ++c) {
    ids[c] = idx[size_t(c)];
    p[c] = std::exp(val[size_t(c)] - *lse);
    *mass += p[c];
  }
  for (int i = 0; i < hidden; ++i) {
    double acc = 0;
    for (int c = 0; c < k; ++c) acc += p[c] * load(embed, size_t(ids[c]) * size_t(hidden) + size_t(i));
    zn[i] = acc;
  }
}

// The canvas after a round that committed n >= 1 tokens (its anchor and accepted drafts): slot w takes this round's
// slot j = w + n.  In: this round's N slots' ids/p [N, k] and zn [N, hidden] (canvas_stats) and hid [N, width] (what
// the selector reads of each slot's final-normed hidden state).  Out, for each slot w:
//   live[w] = j < N (a slot past the canvas holds no candidates);
//   cv[w] = live && p[j][0] >= fill ? ids[j][0] : mask   (its next input);
//   z[w] = live ? zn[j] : 0;
//   cand[w] = ids[jc], logp[w] = log max(p[jc], 1e-9), hid_out[w] = hid[jc], jc = min(j, N - 1) (a slot past the canvas
//   keeps the last slot's, which no tree reads).
inline void slide(int slots, int k, int hidden, int width, int n, double fill, int32_t mask, const int32_t* ids,
                  const double* p, const double* zn, const double* hid, int32_t* cv, double* z, int32_t* cand,
                  double* logp, uint8_t* live, double* hid_out) {
  const size_t K = size_t(k), H = size_t(hidden), Wd = size_t(width);
  for (int w = 0; w < slots; ++w) {
    const int j = w + n, jc = std::min(j, slots - 1);
    const bool ok = j < slots;
    live[w] = ok;
    cv[w] = ok && p[size_t(j) * K] >= fill ? ids[size_t(j) * K] : mask;
    for (size_t i = 0; i < H; ++i) z[size_t(w) * H + i] = ok ? zn[size_t(j) * H + i] : 0.0;
    for (size_t c = 0; c < K; ++c) {
      cand[size_t(w) * K + c] = ids[size_t(jc) * K + c];
      logp[size_t(w) * K + c] = std::log(std::max(p[size_t(jc) * K + c], 1e-9));
    }
    for (size_t i = 0; i < Wd; ++i) hid_out[size_t(w) * Wd + i] = hid[size_t(jc) * Wd + i];
  }
}

}  // namespace flyx::ops
