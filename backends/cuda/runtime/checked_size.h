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

// Pageable spill buffers share the device arena's final VMM allocation.
inline size_t arena_padding(size_t device, size_t spill, size_t granularity) {
  const size_t remainder = (device + spill) % granularity;
  return remainder ? granularity - remainder : 0;
}

inline size_t checkpoint_align(size_t bytes) { return size_add(bytes, 255) & ~size_t(255); }

}  // namespace flyx::cuda
