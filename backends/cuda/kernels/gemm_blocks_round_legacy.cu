// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The round kernels on GGML block weights for the legacy types (blocks of 32 values): one group's translation unit
// (gemm_blocks_ops.h), so the block types compile side by side.
#include "backends/cuda/kernels/gemm_blocks_round_impl.cuh"

namespace flyx::cuda {

RoundFn round_ops_legacy(int dtype) { return round_ops_of<FQ4_0, FQ4_1, FQ5_0, FQ5_1, FQ8_0>(dtype); }
bool round_warps_legacy(int dtype) { return round_warps_of<FQ4_0, FQ4_1, FQ5_0, FQ5_1, FQ8_0>(dtype); }
// Stacks of this group's types first, in one launch (gemm_blocks_ops.h): every type then Q8_0.
RoundStackFn round_stack_ops_legacy(const int* dtype, int n) {
  return round_stacks_of<RoundStack<FQ4_0, FQ8_0>, RoundStack<FQ4_1, FQ8_0>, RoundStack<FQ5_0, FQ8_0>,
                        RoundStack<FQ5_1, FQ8_0>>(dtype, n);
}

}  // namespace flyx::cuda
