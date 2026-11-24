// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Pointwise activations.
#pragma once

#include <cmath>

namespace flyx::ops {

inline double silu(double v) { return v / (1.0 + std::exp(-v)); }
inline double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }
inline double softplus(double v) { return v > 20 ? v : std::log1p(std::exp(v)); }
inline double softcap(double v, double cap) { return cap > 0.0 ? cap * std::tanh(v / cap) : v; }
// SiTU gated activation (Inkling-2): beta * tanh(gate / beta) * sigmoid(gate) * (lb * tanh(up / lb)).
inline double situ(double gate, double up, double beta, double linear_beta) {
  const double act = beta * std::tanh(gate / beta) * sigmoid(gate);
  const double lin = linear_beta > 0.0 ? linear_beta * std::tanh(up / linear_beta) : up;
  return act * lin;
}

}  // namespace flyx::ops
