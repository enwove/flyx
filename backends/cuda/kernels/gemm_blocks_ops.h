// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels of one GGML block type (GEMMs of 1..16 rows, prefill with binary16 or 8-bit inputs, the
// embedding), compiled in the translation unit of its group (gemm_blocks_<group>.cu, from gemm_blocks_impl.cuh):
// the groups compile side by side, and gemm_blocks.cu dispatches through these tables.
#pragma once

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

struct BlockOps {
  bool (*skinny)(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
  void (*rows)(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
  void (*q8)(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L,
             int spread, int runs);
  void (*embed)(const int* tokens, const BlockMat& W, int H, float* x, int rows, cudaStream_t st);
  void (*dots)(const BlockMat& W, int K, const int* idx, const int* count, float* out, size_t ld, int cap,
               const float* x, int ldx, int rows, cudaStream_t st);
};

// A group's types' kernels; nullptr for another type.
const BlockOps* block_ops_legacy(int dtype);   // Q4_0 Q4_1 Q5_0 Q5_1 Q8_0
const BlockOps* block_ops_k(int dtype);        // Q2_K Q3_K Q4_K Q5_K Q6_K
const BlockOps* block_ops_iq_low(int dtype);   // IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S
const BlockOps* block_ops_iq_high(int dtype);  // IQ3_XXS IQ3_S IQ4_NL IQ4_XS

// A group's rounds with inputs in 8 bits (gemm_blocks_round_<group>.cu, gemm_blocks_round_impl.cuh): kernel 0 dp4a
// in configuration cfg (variants: blocks_round), 1 the integer tensor cores; nullptr for another type.
// The most rows of the dp4a configurations compiled with a SwiGLU (Launch::glu): each, one part of the rows and not in
// windows, has another kernel beside its own, in gemm_blocks_round_glu.cu.  round_glu: a round's launch in
// configuration cfg that writes the SwiGLU too; false, nothing run, for the gate run's own launch where gate and up
// are runs of their own (the plain kernel's).
constexpr int kRoundGluRows = 4;
bool round_glu(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
using RoundFn = void (*)(int kernel, int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                         const Launch& L);
RoundFn round_ops_legacy(int dtype);
RoundFn round_ops_k(int dtype);
RoundFn round_ops_iq_low(int dtype);
RoundFn round_ops_iq_high(int dtype);
// Whether kernel 5 (k_round_blocks_warps) takes the group's type dtype.
bool round_warps_legacy(int dtype);
bool round_warps_k(int dtype);
bool round_warps_iq_low(int dtype);
bool round_warps_iq_high(int dtype);

// A stack's n runs (2 or 3, the first of the group's types dtype[0]) in one launch of the dp4a kernel in the first
// run's configuration cfg (false, nothing run: a configuration no rule gives that type, not compiled for stacks);
// nullptr for types not compiled together.  A stack's projections share their input but not always their type (a
// GGUF file picks one per tensor), and each run of one type was a launch of its own: a narrow run cost a launch's
// whole latency for a few columns, the 27B's GDN b|a (96 columns of Q8_0 after the in-projection's other 16384) 2.6
// us at 1 row, 5.4 at 4, 10.2 at 8, where its bytes take 0.3, the 2-bit file's attention k and v (1024 columns each,
// of other types than q) 3 to 14 us a layer (`cuda_bench stack`). Compiled:
// every type then Q8_0 (which GGUF files keep for small projections: GDN b and a, attention k and
// v), and the attention stacks of other types the 27B's files have; the runs after the first are narrow ones
// (gemm_blocks_round.cu: kStackNextColumns).  Two wide runs gain little or lose (the 27B's gate and up of different
// types: the second launch costs only its bytes; GDN qkv then z: z a column a turn is slower).
using RoundStackFn = bool (*)(int cfg, const BlockMat* W, int K, float* y, int ldy, int rows, bool accumulate,
                              const Launch& L);
RoundStackFn round_stack_ops_legacy(const int* dtype, int n);
RoundStackFn round_stack_ops_k(const int* dtype, int n);
RoundStackFn round_stack_ops_iq_low(const int* dtype, int n);
RoundStackFn round_stack_ops_iq_high(const int* dtype, int n);

}  // namespace flyx::cuda
