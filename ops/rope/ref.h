// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Rotary position embedding.
#pragma once

#include <cmath>

namespace flyx::ops {

// Rotate-half RoPE on the first rot_dim dims of one head.
inline void rope(double* x, int pos, double theta, int rot_dim) {
  int half = rot_dim / 2;
  for (int i = 0; i < half; ++i) {
    double inv = 1.0 / std::pow(theta, (2.0 * i) / rot_dim);
    double a = pos * inv, c = std::cos(a), s = std::sin(a);
    double x1 = x[i], x2 = x[i + half];
    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

}  // namespace flyx::ops
