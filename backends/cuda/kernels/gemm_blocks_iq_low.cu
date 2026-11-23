// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels on GGML block weights for IQ1 and IQ2 (codebooks of 1 and 2 bits a weight): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_impl.cuh"

namespace flyx::cuda {

const BlockOps* block_ops_iq_low(int dtype) { return block_ops_of<FIQ1_S, FIQ1_M, FIQ2_XXS, FIQ2_XS, FIQ2_S>(dtype); }

}  // namespace flyx::cuda
