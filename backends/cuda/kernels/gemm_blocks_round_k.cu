// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The round kernels on GGML block weights for the K types (blocks of 256 with sub-block scales): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_round_impl.cuh"

namespace flyx::cuda {

RoundFn round_ops_k(int dtype) { return round_ops_of<FQ2_K, FQ3_K, FQ4_K, FQ5_K, FQ6_K>(dtype); }
bool round_warps_k(int dtype) { return round_warps_of<FQ2_K, FQ3_K, FQ4_K, FQ5_K, FQ6_K>(dtype); }
// Stacks of this group's types first, in one launch (gemm_blocks_ops.h): every type then Q8_0, and the attention
// stacks of other types Qwen3.8-27B's UD-Q4_K_XL and UD-IQ2_S files have (q, then k and v; their headers).
RoundStackFn round_stack_ops_k(const int* dtype, int n) {
  return round_stacks_of<RoundStack<FQ2_K, FQ8_0>, RoundStack<FQ3_K, FQ8_0>, RoundStack<FQ4_K, FQ8_0>,
                        RoundStack<FQ5_K, FQ8_0>, RoundStack<FQ6_K, FQ8_0>, RoundStack<FQ4_K, FQ5_K>,
                        RoundStack<FQ4_K, FQ6_K, FQ8_0>, RoundStack<FQ5_K, FQ6_K, FQ8_0>,
                        RoundStack<FQ5_K, FQ4_K, FQ5_K>, RoundStack<FQ2_K, FIQ3_S, FQ4_K>,
                        RoundStack<FQ2_K, FIQ3_XXS, FQ4_K>>(dtype, n);
}

}  // namespace flyx::cuda
