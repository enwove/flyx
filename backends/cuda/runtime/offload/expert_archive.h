// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include "backends/cuda/loader/layout.h"

namespace flyx::cuda {
// Contiguous CUDA planes for a batch of experts. The largest single expert also fits when it exceeds the DMA target.
struct ExpertBankLayout {
  size_t payload, scale, gains, count;
  static constexpr size_t kTargetTransferBytes = size_t(16) << 20;
  static ExpertBankLayout from(const ExpertLayout& layout, size_t experts, size_t cold) {
    const size_t bytes = layout.bytes() / experts;
    return {layout.payload_bytes() / experts, layout.scale_bytes() / experts, layout.expert_scales ? sizeof(float) : 0,
            std::min(cold, std::max<size_t>(1, kTargetTransferBytes / bytes))};
  }
  size_t scale_offset(size_t n) const { return checkpoint_align(size_mul(payload, n)); }
  size_t gain_offset(size_t n) const { return checkpoint_align(size_add(scale_offset(n), size_mul(scale, n))); }
  size_t bytes(size_t n) const { return size_add(gain_offset(n), size_mul(gains, n)); }
  size_t archive_bytes(size_t n) const { return size_add(size_mul(n / count, bytes(count)), bytes(n % count)); }
};

void pack_expert_nvfp4(const uint8_t* in, size_t blocks, uint8_t* qo, uint8_t* so);
}  // namespace flyx::cuda
