// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels on GGML block weights for the K types (blocks of 256 with sub-block scales): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_impl.cuh"

namespace flyx::cuda {

const BlockOps* block_ops_k(int dtype) { return block_ops_of<FQ2_K, FQ3_K, FQ4_K, FQ5_K, FQ6_K>(dtype); }

}  // namespace flyx::cuda
