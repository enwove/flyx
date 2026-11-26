// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block MoE kernels: K-quant types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K).
#include "backends/cuda/kernels/moe_blocks_impl.cuh"

namespace flyx::cuda {

const MoeBlockOps* moe_block_ops_k(int dtype) {
  return moe_block_ops_of<FQ2_K, FQ3_K, FQ4_K, FQ5_K, FQ6_K>(dtype);
}

}  // namespace flyx::cuda
