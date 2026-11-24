// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// RMS norm (Qwen3.5's zero-centred weight or plain) and L2 norm.
#pragma once

#include <cmath>

namespace flyx::ops {

// offset 1: Qwen3.5 zero-centred weight (x * (1 + w)); offset 0: plain (x * w); w == nullptr: no gamma (x * inv).
inline void rmsnorm(const double* x, const double* w, int n, double eps, double offset, double* y) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  double inv = 1.0 / std::sqrt(ss / n + eps);
  if (!w) {
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv;
  } else {
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (offset + w[i]);
  }
}

inline void l2norm(double* x, int n, double eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  double inv = 1.0 / std::sqrt(ss + eps);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

}  // namespace flyx::ops
