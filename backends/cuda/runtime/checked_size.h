// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace flyx::cuda {

inline size_t size_add(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a) throw std::overflow_error("CUDA state byte count overflow");
  return a + b;
}

inline size_t size_mul(size_t a, size_t b) {
  if (a && b > std::numeric_limits<size_t>::max() / a) throw std::overflow_error("CUDA state byte count overflow");
  return a * b;
}

inline size_t checkpoint_align(size_t bytes) { return size_add(bytes, 255) & ~size_t(255); }

}  // namespace flyx::cuda
