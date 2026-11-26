// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Block Attention Residuals (depth-wise attention over block history) and low-rank cross-pass loop control (Inkling-2).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "ops/linear/ref.h"
#include "ops/norm/ref.h"

namespace flyx::ops {

// Depth-wise attention over M = n_blocks + 1 states (each [H]): the earlier block outputs and `cur`.
// Each state is RMS-normalised (no gamma, eps = 1e-6); each query q[j] ([H], for j in 0..n_queries-1)
// scores all M normalized states scaled by 1 / sqrt(H), takes softmax, and mixes the normalized states.
inline void attn_res_mix(const double* const* blocks, int n_blocks, const double* cur, const double* queries,
                         int n_queries, int H, double* out) {
  const int M = n_blocks + 1;
  std::vector<double> norm_s(size_t(M) * size_t(H));
  for (int m = 0; m < n_blocks; ++m) rmsnorm(blocks[m], nullptr, H, 1e-6, 0.0, &norm_s[size_t(m) * size_t(H)]);
  rmsnorm(cur, nullptr, H, 1e-6, 0.0, &norm_s[size_t(n_blocks) * size_t(H)]);
  const double inv_sqrt = 1.0 / std::sqrt(double(H));
  std::vector<double> scores(static_cast<size_t>(M));
  for (int q = 0; q < n_queries; ++q) {
    const double* qv = queries + size_t(q) * size_t(H);
    double mx = -INFINITY;
    for (int m = 0; m < M; ++m) {
      const double* sm = &norm_s[size_t(m) * size_t(H)];
      double dot = 0.0;
      for (int i = 0; i < H; ++i) dot += sm[i] * qv[i];
      scores[size_t(m)] = dot * inv_sqrt;
      mx = std::max(mx, scores[size_t(m)]);
    }
    double z = 0.0;
    for (int m = 0; m < M; ++m) z += (scores[size_t(m)] = std::exp(scores[size_t(m)] - mx));
    double* oq = out + size_t(q) * size_t(H);
    for (int i = 0; i < H; ++i) oq[i] = 0.0;
    for (int m = 0; m < M; ++m) {
      const double w = scores[size_t(m)] / z;
      const double* sm = &norm_s[size_t(m) * size_t(H)];
      for (int i = 0; i < H; ++i) oq[i] += w * sm[i];
    }
  }
}

// Cross-pass loop gate controller:
//   g = down([rmsnorm(cur, 1e-6) | rmsnorm(msg, 1e-6)])
//   out = (1 - tanh(forget(g))) * cur + tanh(gain(g)) * msg
inline void loop_control(const double* cur, const double* msg, int rows, int H, int rank,
                         const flyx_tensor_view* w_down, const flyx_tensor_view* w_gain,
                         const flyx_tensor_view* w_forget, double* out) {
  std::vector<double> cat(size_t(rows) * size_t(2 * H));
  for (int r = 0; r < rows; ++r) {
    rmsnorm(cur + size_t(r) * size_t(H), nullptr, H, 1e-6, 0.0, &cat[size_t(r) * size_t(2 * H)]);
    rmsnorm(msg + size_t(r) * size_t(H), nullptr, H, 1e-6, 0.0, &cat[size_t(r) * size_t(2 * H) + size_t(H)]);
  }
  std::vector<double> gate(size_t(rows) * size_t(rank));
  std::vector<double> gain(size_t(rows) * size_t(H));
  std::vector<double> forget(size_t(rows) * size_t(H));
  linear(cat.data(), rows, 2 * H, w_down, rank, gate.data());
  linear(gate.data(), rows, rank, w_gain, H, gain.data());
  linear(gate.data(), rows, rank, w_forget, H, forget.data());
  for (size_t i = 0; i < size_t(rows) * size_t(H); ++i) {
    const double gn = std::tanh(gain[i]);
    const double fg = std::tanh(forget[i]);
    out[i] = cur[i] - fg * cur[i] + gn * msg[i];
  }
}

}  // namespace flyx::ops
