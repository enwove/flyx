// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block MoE kernels: low-bit IQ types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S).
#include "backends/cuda/kernels/moe_blocks_impl.cuh"

namespace flyx::cuda {

const MoeBlockOps* moe_block_ops_iq_low(int dtype) {
  return moe_block_ops_of<FIQ1_S, FIQ1_M, FIQ2_XXS, FIQ2_XS, FIQ2_S>(dtype);
}

}  // namespace flyx::cuda
