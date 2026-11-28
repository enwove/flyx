// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Per-group dispatch table for GGML block sparse MoE kernels (moe_blocks_<group>.cu, from moe_blocks_impl.cuh):
// compiles the 21 GGML block types in four parallel translation units while moe.cu handles routing, staging,
// work-tile compaction, NVFP4 Tensor Core MoE, BF16 MoE, and public API dispatch.
#pragma once

#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_act.cuh"
#include "flyx/gemm_tables.inc"

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

namespace flyx::cuda {


// Whether moe_dp4a configuration `cfg` (FLYX_MOE_DP4A) is compiled for block type `dtype`, op kind (`fused_gate_up`:
// fused gate|up vs unfused gate|up or down), row band (`rows == 1` vs `rows > 1`), and compute capability `cc`
// (0: union of all architectures).  Original tiles stay compiled everywhere; with FLYX_TUNE_CANDIDATES every
// configuration of kMoeDp4aTypes stays compiled for all ops and rows; else a release build compiles only fused gate|up
// configurations reached by tables on `cc` (host dispatch checks cc=0; each kernel body traps on unreachable arches).
constexpr bool moe_dp4a_cfg_compiled(int dtype, int cfg, bool fused_gate_up = true, int rows = 2, int cc = 0) {
  if (cfg == tables::kMoeDp4aOriginal) return true;
  if (cfg < 0 || cfg >= FLYX_MOE_DP4A_COUNT || !has_moe_dp4a_variants(dtype)) return false;
#if FLYX_TUNE_CANDIDATES
  (void)fused_gate_up, (void)rows, (void)cc;
  return true;
#else
  if (!fused_gate_up) return false;
  const int di = tables::ggml_type_index(dtype);
  return di >= 0 && (tables::kReachMoeDp4a[tables::arch_bucket(cc)][rows == 1 ? 0 : 1][di] & (1u << cfg)) != 0;
#endif
}

// Compile-time reachability predicate for a k_moe_blocks_q8_dp4a template instance on compute capability `cc`.
template <class F, int CFG, bool FUSE_ACT, bool DIRECT_R1>
#ifdef __CUDACC__
__host__ __device__
#endif
constexpr bool moe_dp4a_kernel_active(int cc) {
  if constexpr (CFG == tables::kMoeDp4aOriginal) {
    (void)cc;
    return true;
  } else {
#if FLYX_TUNE_CANDIDATES
    (void)cc;
    return CFG >= 0 && CFG < FLYX_MOE_DP4A_COUNT && has_moe_dp4a_variants(F::kDtype);
#else
    if constexpr (!FUSE_ACT || CFG < 0 || CFG >= FLYX_MOE_DP4A_COUNT || !has_moe_dp4a_variants(F::kDtype)) {
      (void)cc;
      return false;
    } else {
      constexpr int di = tables::ggml_type_index(F::kDtype);
      if constexpr (di < 0) return false;
      else return (tables::kReachMoeDp4a[tables::arch_bucket(cc)][DIRECT_R1 ? 0 : 1][di] & (1u << CFG)) != 0;
    }
#endif
  }
}

// mma_rows: the tensor-core kernels on the tasks sc holds, that many rows each (moe_mma_rows: 16 or 32), else 0: the
// dp4a kernel in configuration dp4a, one compiled for the type (moe_dp4a_cfg_compiled).
struct MoeBlockOps {
  bool mins;  // F::kFlatRun != 0 (requires q8_s half-group sums during staging)
  void (*gu)(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
             MoeAct mact, bool fuse_act, float* out, int ld_out, const MoeScratch* sc,
             int max_tiles, int mma_rows, int dp4a, cudaStream_t st, bf16* act_bf16);
  bool (*down)(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows, int top_k,
               int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc, int max_tiles,
               int mma_rows, int dp4a, cudaStream_t st, const float* bias);
};

const MoeBlockOps* moe_block_ops_legacy(int dtype);   // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 MXFP4 Q2_0
const MoeBlockOps* moe_block_ops_k(int dtype);        // Q2_K Q3_K Q4_K Q5_K Q6_K
const MoeBlockOps* moe_block_ops_iq_low(int dtype);   // IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S
const MoeBlockOps* moe_block_ops_iq_high(int dtype);  // IQ3_XXS IQ3_S IQ4_NL IQ4_XS

}  // namespace flyx::cuda
