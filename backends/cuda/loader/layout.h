// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "backends/cuda/kernels/block_layout.h"
#include "backends/cuda/runtime/checked_size.h"
#include "flyx/backend_abi.h"

namespace flyx::cuda {

constexpr size_t kLoadSlotBytes = size_t(64) << 20;

// Every floating-point source and row order goes through a BF16 row in the pinned slot.
inline size_t quant_row_bytes(size_t columns, const char* name) {
  if (!columns || columns > kLoadSlotBytes / sizeof(uint16_t))
    throw std::runtime_error(std::string(name ? name : "weight") +
                             ": quantized upload needs a nonempty BF16 row of at most 64 MiB");
  return columns * sizeof(uint16_t);
}

// Resident planes of a MoE matrix [experts, output rows, input columns]. Scales are counted in bytes.
struct ExpertLayout {
  enum Format { bf16_rows, ggml_blocks, nvfp4 };
  Format format;
  size_t rows, payload_row, scales_row, expert_scales;
  size_t payload_bytes() const { return size_mul(rows, payload_row); }
  size_t scale_bytes() const { return size_mul(rows, scales_row); }
  size_t bytes() const {
    return size_add(size_add(payload_bytes(), scale_bytes()), size_mul(expert_scales, sizeof(float)));
  }
};

inline ExpertLayout expert_layout(const flyx_weight& w, bool packed) {
  const auto& v = *w.t;
  if (v.ndim != 3 || v.shape[0] <= 0 || v.shape[1] <= 0 || v.shape[2] <= 0 || v.shape[2] > INT_MAX ||
      v.shape[0] > INT_MAX / v.shape[1])
    throw std::runtime_error("invalid MoE expert matrix dimensions");
  const size_t e = size_t(v.shape[0]), rows = e * size_t(v.shape[1]), k = size_t(v.shape[2]);
  if (v.dtype == FLYX_DT_GGML_NVFP4) {
    if (k % 64) throw std::runtime_error("GGML NVFP4 expert width must be a multiple of 64");
    return {ExpertLayout::nvfp4, rows, k / 2, k / 16, e};
  }
  const auto lo = packed ? blocks_layout(v.dtype) : BlockLayout{};
  if (lo.values && k % lo.values == 0) {
    const size_t blocks = k / lo.values;
    return {ExpertLayout::ggml_blocks, rows, blocks * lo.stride, blocks * lo.scales * sizeof(uint16_t), w.scale2 ? e : 0};
  }
  return {ExpertLayout::bf16_rows, rows, k * sizeof(uint16_t), 0, w.scale2 ? e : 0};
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
