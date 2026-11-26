// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Sparse Mixture-of-Experts routing and expert linear projections (3D stacked weights with optional per-expert scale).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "ops/activation/ref.h"
#include "ops/linear/ref.h"

namespace flyx::ops {

// Route one token's router logits [n_experts] to top_k experts.
// gating_sigmoid: 1 = sigmoid(logits), 0 = softmax(logits).
// Optional bias shifts selection score only; weights come from unbiased probabilities.
inline void moe_route(const double* logits, const double* bias, int n_experts, int top_k, bool gating_sigmoid,
                      bool norm_weights, double norm_eps, double route_scale, int32_t* idx_out, double* w_out) {
  std::vector<double> probs(static_cast<size_t>(n_experts));
  if (gating_sigmoid) {
    for (int e = 0; e < n_experts; ++e) probs[size_t(e)] = sigmoid(logits[e]);
  } else {
    double mx = -INFINITY;
    for (int e = 0; e < n_experts; ++e) mx = std::max(mx, logits[e]);
    double z = 0.0;
    for (int e = 0; e < n_experts; ++e) z += (probs[size_t(e)] = std::exp(logits[e] - mx));
    for (int e = 0; e < n_experts; ++e) probs[size_t(e)] /= z;
  }
  std::vector<double> sel(static_cast<size_t>(n_experts));
  for (int e = 0; e < n_experts; ++e) sel[size_t(e)] = probs[size_t(e)] + (bias ? bias[e] : 0.0);
  std::vector<int32_t> order(static_cast<size_t>(n_experts));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int32_t a, int32_t b) {
    const double sa = sel[size_t(a)], sb = sel[size_t(b)];
    return sa != sb ? sa > sb : a < b;
  });
  double sum = 0.0;
  for (int k = 0; k < top_k; ++k) {
    idx_out[k] = order[size_t(k)];
    w_out[k] = probs[size_t(idx_out[k])];
    sum += w_out[k];
  }
  const double denom = norm_weights ? (sum + norm_eps) : 1.0;
  for (int k = 0; k < top_k; ++k) w_out[k] = (w_out[k] / denom) * route_scale;
}

// Single-row linear against slice `expert` of a 3D weight [E, N, K] stored in float64,
// with the same input rounding contract as ops/linear/ref.h (q8 / nvfp4 / a16h / a16 / exact).
inline void linear_expert_f64(const double* x, int K, const double* W_all, int expert, int N, double* y) {
  const double* W = W_all + size_t(expert) * size_t(N) * size_t(K);
  for (int n = 0; n < N; ++n) {
    const double* wn = W + size_t(n) * size_t(K);
    double acc = 0.0;
    for (int k = 0; k < K; ++k) acc += x[k] * wn[k];
    y[n] = acc;
  }
}

}  // namespace flyx::ops
