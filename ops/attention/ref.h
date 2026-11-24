// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Causal attention of one query head (grouped-query: kv head = h / group), and
// a head of q or k as attention takes it.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "ops/norm/ref.h"
#include "ops/rope/ref.h"

namespace flyx::ops {

// One head of q or k (hd values) as attention takes it at position pos: its RMSNorm (w, offset, eps) where the
// block has q/k norms, none where w is null (Llama: the head as projected), then rotary on the first rot_dim dims.
inline void prep_head(const double* x, const double* w, int hd, double eps, double offset, int pos, double theta,
                      int rot_dim, double* y) {
  if (w) rmsnorm(x, w, hd, eps, offset, y);
  else if (y != x) std::copy(x, x + hd, y);
  rope(y, pos, theta, rot_dim);
}

// One query head against keys [0, n_keys): K and V are [n_keys, n_kv, hd]; kv head = h / group.
inline void attend(const double* q, const double* K, const double* V, int n_keys, int n_kv, int hd, int kv_head,
                   double scale, double* out) {
  std::vector<double> s(n_keys);
  double mx = -INFINITY;
  for (int j = 0; j < n_keys; ++j) {
    const double* k = K + (size_t(j) * n_kv + kv_head) * hd;
    double d = 0;
    for (int i = 0; i < hd; ++i) d += q[i] * k[i];
    s[j] = d * scale;
    mx = std::max(mx, s[j]);
  }
  double sum = 0;
  for (int j = 0; j < n_keys; ++j) sum += (s[j] = std::exp(s[j] - mx));
  for (int i = 0; i < hd; ++i) out[i] = 0;
  for (int j = 0; j < n_keys; ++j) {
    const double* v = V + (size_t(j) * n_kv + kv_head) * hd;
    double p = s[j] / sum;
    for (int i = 0; i < hd; ++i) out[i] += p * v[i];
  }
}

// Per-head L2 norm clipping with optional per-channel scale (Inkling-2 Q/K/R clipping):
// y = (x * scale) * clamp(threshold / (||x * scale||_2 + 1e-6), 0, 1).
inline void l2_clip(const double* x, const double* scale, int dim, double threshold, double* y) {
  double sq = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = scale ? x[i] * scale[i] : x[i];
    y[i] = v;
    sq += v * v;
  }
  const double norm = std::sqrt(sq) + 1e-6;
  const double factor = std::clamp(threshold / norm, 0.0, 1.0);
  for (int i = 0; i < dim; ++i) y[i] *= factor;
}

// Global-layer position log scaling (Inkling-2): 1 + alpha * log(max(1, (pos + 1) / floor)).
inline double log_pos_scale(int pos, int floor_n, double alpha) {
  if (floor_n <= 0 || alpha <= 0.0) return 1.0;
  return 1.0 + alpha * std::log(std::max(1.0, double(pos + 1) / double(floor_n)));
}

// Causal relative-position-bias attention with per-query sink and optional sliding window (Inkling-2):
// K, V are [n_keys, n_kv, hd] (indexed by absolute position 0..q_pos).  rel_bias is [rel_extent, d_rel].
// Distance d = q_pos - j in [0, rel_extent) adds dot(rel_bias[d], r); d >= rel_extent adds 0.
// Virtual sink has logit 0 after subtracting `sink` from every key logit.
inline void attend_rel(const double* q, const double* r, const double* K, const double* V, const double* rel_bias,
                       double sink, int q_pos, int window, int n_kv, int hd, int d_rel, int rel_extent, int kv_head,
                       double scale, double* out) {
  const int j0 = (window > 0 && q_pos - window + 1 > 0) ? (q_pos - window + 1) : 0;
  const int n = q_pos - j0 + 1;
  std::vector<double> btab(static_cast<size_t>(rel_extent) + 1, 0.0);
  for (int d = 0; d < rel_extent; ++d) {
    double acc = 0.0;
    const double* row = rel_bias + size_t(d) * size_t(d_rel);
    for (int m = 0; m < d_rel; ++m) acc += row[m] * r[m];
    btab[size_t(d)] = acc;
  }
  std::vector<double> s(static_cast<size_t>(n));
  double mx = 0.0;  // virtual sink sits at logit 0 after subtracting `sink` from every key logit
  for (int t = 0; t < n; ++t) {
    const int j = j0 + t;
    const double* k = K + (size_t(j) * size_t(n_kv) + size_t(kv_head)) * size_t(hd);
    double dot = 0.0;
    for (int i = 0; i < hd; ++i) dot += q[i] * k[i];
    const int dist = q_pos - j;
    const double b = (dist >= 0 && dist < rel_extent) ? btab[size_t(dist)] : 0.0;
    s[size_t(t)] = dot * scale + b - sink;
    mx = std::max(mx, s[size_t(t)]);
  }
  double sum = std::exp(0.0 - mx);
  for (int t = 0; t < n; ++t) sum += (s[size_t(t)] = std::exp(s[size_t(t)] - mx));
  for (int i = 0; i < hd; ++i) out[i] = 0.0;
  for (int t = 0; t < n; ++t) {
    const int j = j0 + t;
    const double* v = V + (size_t(j) * size_t(n_kv) + size_t(kv_head)) * size_t(hd);
    const double p = s[size_t(t)] / sum;
    for (int i = 0; i < hd; ++i) out[i] += p * v[i];
  }
}

}  // namespace flyx::ops
