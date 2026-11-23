// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gated DeltaNet: the causal depthwise conv before the mixer and the
// gated delta rule, one row at a time.
#pragma once

#include <cmath>
#include <cstring>

#include "ops/activation/ref.h"

namespace flyx::ops {

// Causal depthwise conv over the window [prev (k-1) rows | this row], then SiLU.
// window: (k-1) rows of `dim`; row: this row's input; w: [dim, k].
inline void conv_row(const double* window, const double* row, const double* w, int dim, int k, double* out) {
  for (int c = 0; c < dim; ++c) {
    double acc = 0;
    for (int j = 0; j < k - 1; ++j) acc += window[size_t(j) * dim + c] * w[size_t(c) * k + j];
    acc += row[c] * w[size_t(c) * k + k - 1];
    out[c] = silu(acc);
  }
}

// Shift the window by one row.
inline void conv_push(double* window, const double* row, int dim, int k) {
  if (k <= 1) return;
  std::memmove(window, window + dim, sizeof(double) * size_t(k - 2) * dim);
  std::memcpy(window + size_t(k - 2) * dim, row, sizeof(double) * dim);
}

// gdn_row() for rows j0 .. j0 + J - 1 of the state, side by side: the CPU overlaps their sums (each still runs over
// i in order, so every row gets the bits it gets alone).
template <int J>
inline void gdn_rows(double* S, const double* q, const double* k, const double* v, double beta, double decay, int dk,
                     int j0, double* out) {
  double* s[J];
  double mem[J] = {}, delta[J], o[J] = {};
  for (int j = 0; j < J; ++j) s[j] = S + size_t(j0 + j) * dk;
  for (int i = 0; i < dk; ++i)
    for (int j = 0; j < J; ++j) {
      s[j][i] *= decay;
      mem[j] += s[j][i] * k[i];
    }
  for (int j = 0; j < J; ++j) delta[j] = (v[j0 + j] - mem[j]) * beta;
  for (int i = 0; i < dk; ++i)
    for (int j = 0; j < J; ++j) {
      s[j][i] += k[i] * delta[j];
      o[j] += s[j][i] * q[i];
    }
  if (out)
    for (int j = 0; j < J; ++j) out[j0 + j] = o[j];
}

// Gated delta rule, one row, one value head.  S is [dv][dk] (column-major in
// the usual math: S[j][i] = state[i][j]).  q already includes 1/sqrt(dk).
// Per row j of S: s *= e^g; mem = s . k; delta = (v[j] - mem) beta; s += k delta; out[j] = s . q.
inline void gdn_row(double* S, const double* q, const double* k, const double* v, double beta, double g, int dk,
                    int dv, double* out) {
  const double decay = std::exp(g);
  int j = 0;
  for (; j + 4 <= dv; j += 4) gdn_rows<4>(S, q, k, v, beta, decay, dk, j, out);
  for (; j < dv; ++j) gdn_rows<1>(S, q, k, v, beta, decay, dk, j, out);
}

}  // namespace flyx::ops
