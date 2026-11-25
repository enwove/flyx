// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Per-group dispatch table for GGML block sparse MoE kernels (moe_blocks_<group>.cu, from moe_blocks_impl.cuh):
// compiles the 21 GGML block types in four parallel translation units while moe.cu handles routing, staging,
// work-tile compaction, NVFP4 Tensor Core MoE, BF16 MoE, and public API dispatch.
#pragma once

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

struct MoeBlockOps {
  bool mins;  // F::kFlatRun != 0 (requires q8_s half-group sums during staging)
  void (*gu)(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
             float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out, const MoeScratch* sc,
             int max_tiles, bool use_mma, MoeDp4a dp4a, cudaStream_t st, bf16* act_bf16);
  bool (*down)(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows, int top_k,
               int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc, int max_tiles, bool use_mma,
               MoeDp4a dp4a, cudaStream_t st);
};

const MoeBlockOps* moe_block_ops_legacy(int dtype);   // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 MXFP4 Q2_0
const MoeBlockOps* moe_block_ops_k(int dtype);        // Q2_K Q3_K Q4_K Q5_K Q6_K
const MoeBlockOps* moe_block_ops_iq_low(int dtype);   // IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S
const MoeBlockOps* moe_block_ops_iq_high(int dtype);  // IQ3_XXS IQ3_S IQ4_NL IQ4_XS

}  // namespace flyx::cuda
