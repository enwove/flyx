// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The round kernels on GGML block weights for IQ1 and IQ2 (codebooks of 1 and 2 bits a weight): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_round_impl.cuh"

namespace flyx::cuda {

RoundFn round_ops_iq_low(int dtype) { return round_ops_of<FIQ1_S, FIQ1_M, FIQ2_XXS, FIQ2_XS, FIQ2_S>(dtype); }
bool round_warps_iq_low(int dtype) { return round_warps_of<FIQ1_S, FIQ1_M, FIQ2_XXS, FIQ2_XS, FIQ2_S>(dtype); }
// Stacks of this group's types first, in one launch (gemm_blocks_ops.h): every type then Q8_0, and the attention
// stacks of other types Qwen3.8-27B's UD-Q4_K_XL and UD-IQ2_S files have (q, then k and v; their headers).
RoundStackFn round_stack_ops_iq_low(const int* dtype, int n) {
  return round_stacks_of<RoundStack<FIQ1_S, FQ8_0>, RoundStack<FIQ1_M, FQ8_0>, RoundStack<FIQ2_XXS, FQ8_0>,
                        RoundStack<FIQ2_XS, FQ8_0>, RoundStack<FIQ2_S, FQ8_0>, RoundStack<FIQ2_XS, FIQ4_XS>,
                        RoundStack<FIQ1_S, FIQ2_XS, FQ4_K>, RoundStack<FIQ2_XXS, FIQ3_S, FQ4_K>,
                        RoundStack<FIQ2_XXS, FIQ4_XS, FQ4_K>, RoundStack<FIQ2_XXS, FQ2_K, FQ4_K>>(dtype, n);
}

}  // namespace flyx::cuda
