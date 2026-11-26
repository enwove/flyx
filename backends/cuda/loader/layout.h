// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace flyx::cuda {

constexpr size_t kLoadSlotBytes = size_t(64) << 20;

// Every floating-point source and row order goes through a BF16 row in the pinned slot.
inline size_t quant_row_bytes(size_t columns, const char* name) {
  if (!columns || columns > kLoadSlotBytes / sizeof(uint16_t))
    throw std::runtime_error(std::string(name ? name : "weight") +
                             ": quantized upload needs a nonempty BF16 row of at most 64 MiB");
  return columns * sizeof(uint16_t);
}

struct VisionQuantLayout {
  size_t width, payload, scales, row_scales;  // padded columns, then bytes in each plane
  size_t bytes() const { return payload + scales + row_scales; }

  static VisionQuantLayout q8(size_t rows, size_t cols) {
    const size_t width = (cols + 31) / 32 * 32;
    return {width, rows * width, rows * (width / 32) * sizeof(uint16_t), 0};
  }
  static VisionQuantLayout nvfp4(size_t rows, size_t cols) {
    const size_t width = (cols + 63) / 64 * 64;
    return {width, rows * (width / 2), rows * (width / 16), rows * sizeof(float)};
  }
};

}  // namespace flyx::cuda
