// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>

namespace flyx::cuda {

inline uint64_t min_runtime(uint64_t weights) {
  return std::clamp<uint64_t>(weights / 20, 384ull << 20, 1ull << 30);
}

// Test by subtraction so a huge checkpoint or requested context cannot wrap the requirement.
inline bool ple_fits_gpu(uint64_t free, uint64_t total, uint64_t weights, uint64_t kv_per_pos, uint64_t context) {
  if (!total) return false;
  for (uint64_t held : {weights, std::max<uint64_t>(384ull << 20, total / 40), min_runtime(weights)}) {
    if (held > free) return false;
    free -= held;
  }
  return !kv_per_pos || context <= free / kv_per_pos;
}

}  // namespace flyx::cuda
