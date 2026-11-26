// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block MoE kernels: legacy 32-weight types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, Q2_0).
#include "backends/cuda/kernels/moe_blocks_impl.cuh"

namespace flyx::cuda {

const MoeBlockOps* moe_block_ops_legacy(int dtype) {
  return moe_block_ops_of<FQ4_0, FQ4_1, FQ5_0, FQ5_1, FQ8_0, FMXFP4, FQ2_0>(dtype);
}

}  // namespace flyx::cuda
