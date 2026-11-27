// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Pointwise activations.
#pragma once

#include <cmath>

#include "flyx/backend_abi.h"

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

// Exact Gaussian Error Linear Unit (PyTorch nn.GELU(approximate='none')).
inline double gelu(double v) { return 0.5 * v * (1.0 + std::erf(v * 0.70710678118654752440)); }

// Tanh-approximated GELU (PyTorch nn.GELU(approximate='tanh') / gelu_pytorch_tanh).
inline double gelu_tanh(double v) {
  constexpr double kSqrt2OverPi = 0.79788456080286535588;
  return 0.5 * v * (1.0 + std::tanh(kSqrt2OverPi * (v + 0.044715 * v * v * v)));
}

inline double quick_gelu(double v) { return v * sigmoid(1.702 * v); }

inline double vact(double v, int kind) {
  switch (kind) {
    case FLYX_VACT_GELU_TANH: return gelu_tanh(v);
    case FLYX_VACT_GELU: return gelu(v);
    case FLYX_VACT_SILU: return silu(v);
    case FLYX_VACT_QUICK_GELU: return quick_gelu(v);
    default: return v;
  }
}

}  // namespace flyx::ops
