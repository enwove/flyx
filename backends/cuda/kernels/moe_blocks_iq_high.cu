// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block MoE kernels: higher-bit IQ types (IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS).
#include "backends/cuda/kernels/moe_blocks_impl.cuh"

namespace flyx::cuda {

const MoeBlockOps* moe_block_ops_iq_high(int dtype) {
  return moe_block_ops_of<FIQ3_XXS, FIQ3_S, FIQ4_NL, FIQ4_XS>(dtype);
}

}  // namespace flyx::cuda
