// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
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

// Validate the bytes read by a dense projection or router before either estimating it or opening a staging slot.
inline void check_linear_payload(const flyx_weight& w) {
  if (!w.t || !w.t->data || w.t->ndim < 1 || w.t->ndim > 2) throw std::runtime_error("invalid linear payload");
  const auto& t = *w.t;
  size_t elements = 1;
  for (int i = 0; i < t.ndim; ++i) {
    if (t.shape[i] <= 0 || t.shape[i] > INT_MAX) throw std::runtime_error("invalid linear dimensions");
    elements = size_mul(elements, size_t(t.shape[i]));
  }
  const size_t k = size_t(t.shape[t.ndim - 1]);
  if (t.dtype == FLYX_DT_NVFP4) {
    const auto* scales = w.scale;
    const auto* gain = w.scale2;
    if (t.ndim != 2 || k % 32 || t.nbytes != elements / 2 || !scales || !scales->data ||
        scales->dtype != FLYX_DT_F8E4M3 || scales->ndim != 2 || scales->shape[0] != t.shape[0] ||
        scales->shape[1] != int64_t(k / 16) || scales->nbytes != elements / 16 || !gain || !gain->data ||
        gain->dtype != FLYX_DT_F32 || gain->nbytes != sizeof(float) || gain->ndim < 0 || gain->ndim > 1 ||
        (gain->ndim == 1 && gain->shape[0] != 1))
      throw std::runtime_error("invalid NVFP4 linear planes");
    return;
  }
  const int span = flyx_block_values(t.dtype);
  const int bytes = span                                                ? flyx_block_bytes(t.dtype)
                    : t.dtype == FLYX_DT_F32                            ? 4
                    : t.dtype == FLYX_DT_F64                            ? 8
                    : t.dtype == FLYX_DT_BF16 || t.dtype == FLYX_DT_F16 ? 2
                                                                        : 0;
  if (!bytes || (span && k % span) || size_mul(elements / (span ? span : 1), bytes) != t.nbytes)
    throw std::runtime_error("invalid linear row format or extent");
  if (w.scale2 && (!w.scale2->data || w.scale2->dtype != FLYX_DT_F32 || w.scale2->nbytes != sizeof(float) ||
                   w.scale2->ndim < 0 || w.scale2->ndim > 4 ||
                   !std::all_of(w.scale2->shape, w.scale2->shape + w.scale2->ndim, [](int64_t n) { return n == 1; })))
    throw std::runtime_error("invalid linear gain");
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
  if (!w.t || !w.t->data) throw std::runtime_error("missing expert tensor payload");
  const auto& v = *w.t;
  if (v.ndim != 3 || v.shape[0] <= 0 || v.shape[1] <= 0 || v.shape[2] <= 0 || v.shape[2] > INT_MAX ||
      v.shape[0] > INT_MAX / v.shape[1])
    throw std::runtime_error("invalid MoE expert matrix dimensions");
  const size_t e = size_t(v.shape[0]), rows = e * size_t(v.shape[1]), k = size_t(v.shape[2]);
  const int span = flyx_block_values(v.dtype);
  const int width = span                                                ? flyx_block_bytes(v.dtype)
                    : v.dtype == FLYX_DT_F32                            ? 4
                    : v.dtype == FLYX_DT_F64                            ? 8
                    : v.dtype == FLYX_DT_BF16 || v.dtype == FLYX_DT_F16 ? 2
                                                                        : 0;
  if (!width || (span && k % span) || size_mul(rows, size_mul(k / (span ? span : 1), width)) != v.nbytes)
    throw std::runtime_error("invalid expert row format or payload extent");
  if (w.scale) throw std::runtime_error("unsupported external expert block scale");
  if (const auto* scale = w.scale2) {
    if (!scale->data || scale->dtype != FLYX_DT_F32 || scale->ndim < 0 || scale->ndim > 4)
      throw std::runtime_error("invalid expert scale tensor");
    size_t values = 1;
    for (int i = 0; i < scale->ndim; ++i) {
      if (scale->shape[i] < 1) throw std::runtime_error("invalid expert scale dimension");
      values = size_mul(values, size_t(scale->shape[i]));
    }
    if ((values != 1 && values != e) || size_mul(values, sizeof(float)) != scale->nbytes)
      throw std::runtime_error("invalid expert scale extent");
  }
  if (v.dtype == FLYX_DT_GGML_NVFP4) {
    if (k / 64 * 36 > kLoadSlotBytes - 16) throw std::runtime_error("NVFP4 expert row exceeds load slot");
    if (k % 64) throw std::runtime_error("GGML NVFP4 expert width must be a multiple of 64");
    return {ExpertLayout::nvfp4, rows, k / 2, k / 16, e};
  }
  const auto lo = packed ? blocks_layout(v.dtype) : BlockLayout{};
  if (lo.values && k % lo.values == 0) {
    const size_t blocks = k / lo.values;
    if (size_mul(blocks, lo.stride + lo.scales * sizeof(uint16_t)) > kLoadSlotBytes - 16)
      throw std::runtime_error("packed expert row exceeds load slot");
    return {ExpertLayout::ggml_blocks, rows, blocks * lo.stride, blocks * lo.scales * sizeof(uint16_t), w.scale2 ? e : 0};
  }
  return {ExpertLayout::bf16_rows, rows, quant_row_bytes(k, v.name), 0, w.scale2 ? e : 0};
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
