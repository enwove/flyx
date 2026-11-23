// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Pointwise activations.
#pragma once

#include <cmath>

namespace flyx::ops {

inline double silu(double v) { return v / (1.0 + std::exp(-v)); }
inline double sigmoid(double v) { return 1.0 / (1.0 + std::exp(-v)); }
inline double softplus(double v) { return v > 20 ? v : std::log1p(std::exp(v)); }

}  // namespace flyx::ops
