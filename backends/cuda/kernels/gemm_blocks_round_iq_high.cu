// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The round kernels on GGML block weights for IQ3 and IQ4 (codebooks of 3 bits, non-linear 4-bit values): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_round_impl.cuh"

namespace flyx::cuda {

RoundFn round_ops_iq_high(int dtype) { return round_ops_of<FIQ3_XXS, FIQ3_S, FIQ4_NL, FIQ4_XS>(dtype); }
bool round_warps_iq_high(int dtype) { return round_warps_of<FIQ3_XXS, FIQ3_S, FIQ4_NL, FIQ4_XS>(dtype); }
// Stacks of this group's types first, in one launch (gemm_blocks_ops.h): every type then Q8_0, and the attention
// stacks of other types Qwen3.8-27B's UD-Q4_K_XL and UD-IQ2_S files have (q, then k and v; their headers).
RoundStackFn round_stack_ops_iq_high(const int* dtype, int n) {
  return round_stacks_of<RoundStack<FIQ3_XXS, FQ8_0>, RoundStack<FIQ3_S, FQ8_0>, RoundStack<FIQ4_NL, FQ8_0>,
                        RoundStack<FIQ4_XS, FQ8_0>, RoundStack<FIQ4_XS, FQ6_K>, RoundStack<FIQ3_XXS, FIQ4_XS, FQ4_K>,
                        RoundStack<FIQ4_XS, FQ6_K, FQ8_0>>(dtype, n);
}

}  // namespace flyx::cuda
