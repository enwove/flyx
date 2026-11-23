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

}  // namespace flyx::ops
