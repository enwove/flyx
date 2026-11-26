// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace flyx::cuda {

inline uint32_t f32_word(const void* values, size_t i) {
  uint32_t word;
  std::memcpy(&word, static_cast<const uint8_t*>(values) + i * sizeof(word), sizeof(word));
  return word;
}

inline bool all_exact_bf16(const void* values, size_t count, bool* has_nan = nullptr) {
  bool nan = false;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t word = f32_word(values, i);
    if (word & 0xffffu) return false;
    nan |= (word & 0x7fffffffu) > 0x7f800000u;
  }
  if (has_nan) *has_nan = nan;
  return true;
}

}  // namespace flyx::cuda
