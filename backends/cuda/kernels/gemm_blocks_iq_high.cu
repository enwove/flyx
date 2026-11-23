// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels on GGML block weights for IQ3 and IQ4 (codebooks of 3 bits, non-linear 4-bit values): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_impl.cuh"

namespace flyx::cuda {

const BlockOps* block_ops_iq_high(int dtype) { return block_ops_of<FIQ3_XXS, FIQ3_S, FIQ4_NL, FIQ4_XS>(dtype); }

}  // namespace flyx::cuda
