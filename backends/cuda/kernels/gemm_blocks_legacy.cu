// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels on GGML block weights for the legacy types (blocks of 32 values): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_impl.cuh"

namespace flyx::cuda {

const BlockOps* block_ops_legacy(int dtype) {
  return block_ops_of<FQ4_0, FQ4_1, FQ5_0, FQ5_1, FQ8_0, FMXFP4, FQ2_0>(dtype);
}

}  // namespace flyx::cuda
