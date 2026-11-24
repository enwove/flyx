// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// RMS norm (Qwen3.5's zero-centred weight or plain) and L2 norm.
#pragma once

#include <cmath>

namespace flyx::ops {

// offset 1: Qwen3.5 zero-centred weight (x * (1 + w)); offset 0: plain (x * w).
inline void rmsnorm(const double* x, const double* w, int n, double eps, double offset, double* y) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  double inv = 1.0 / std::sqrt(ss / n + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (offset + w[i]);
}

inline void l2norm(double* x, int n, double eps) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  double inv = 1.0 / std::sqrt(ss + eps);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Standard LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps) * w + b (w and b optional).
inline void layernorm(const double* x, const double* w, const double* b, int n, double eps, double* y) {
  double mean = 0;
  for (int i = 0; i < n; ++i) mean += x[i];
  mean /= n;
  double var = 0;
  for (int i = 0; i < n; ++i) {
    double d = x[i] - mean;
    var += d * d;
  }
  double inv = 1.0 / std::sqrt(var / n + eps);
  for (int i = 0; i < n; ++i) {
    double v = (x[i] - mean) * inv;
    if (w) v *= w[i];
    if (b) v += b[i];
    y[i] = v;
  }
}

}  // namespace flyx::ops
