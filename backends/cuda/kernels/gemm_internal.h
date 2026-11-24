// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What the GEMM files share besides the public launchers (kernels.cuh).
#pragma once

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

// Whether the 1..16-row tensor-core path takes this input and weight (alignment, K % 32).
bool skinny_fits(const float* x, int ldx, const bf16* W, int K);

// The prefill GEMM on the rows a fused producer left in the launch's scratch (float32 in af32 when it has it, else
// bf16 in abuf): W (bf16) or Q/S/RS (NVFP4).
void gemm_tc_staged(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy,
                    int M, bool accumulate, const Launch& L);

// What the block-weight kernels decode with (gemm_blocks.cu), built once on
// the host (gemm_blocks_tables.cu) from ggml's own tables.  Values are
// binary16 bit patterns of integers: pairs are two in a word (the first in
// the low half), quads four in two words.
struct BlockTables {
  // Codebook rows as the selectors of their weights' magnitudes, a nibble a weight (the first weight lowest).
  const uint32_t* iq2xxs;    // [256]   8 selectors a row, 0..2
  const uint32_t* iq2xs;     // [512]
  const uint32_t* iq2s;      // [1024]
  const uint16_t* iq3xxs;    // [256]   4 selectors a row, 0..7
  const uint16_t* iq3s;      // [512]
  const uint32_t* iq1;       // [2048]  8 selectors a row: 0, 1, 2 for -1, 0, 1
  // Signs of 8 weights: as bit 2 of their selectors, or as the top bits of their high bytes (weights 0..3 | 4..7).
  const uint32_t* ksign_sel; // [128]   of a 7-bit sign code (the eighth sign is the parity)
  const uint32_t* sign_sel;  // [256]   of 8 sign bits
  const uint2* ksign_hi;     // [128]
  const uint2* sign_hi;      // [256]
  // A family's magnitudes times 2 s + 1 as byte tables for a selector (entry i in byte i).
  const uint4* iq2_scaled;     // [3][16] IQ2_XXS, _XS, _S; s: {high bytes, high bytes negated, low bytes, -}
  const uint4* iq3xxs_scaled;  // [16]    s: {high 0..3, high 4..7, low 0..3, low 4..7}
  const uint4* iq3s_scaled;    // [16]
  const uint2* iq1_scaled;     // [16]    (shift down) << 3 | s: {high, low} of (8 w +- 1)(2 s + 1), w = -1, 0, 1
  // Bit fields.
  const uint32_t* pair_u4;   // [256]  a byte's two nibbles (the low one first)
  const uint32_t* pair_s4;   // [256]  the same less 8
  const uint32_t* pair_iq4;  // [256]  the same through IQ4's 16 values
  const uint32_t* pair_mxfp4;  // [256] the same through MXFP4's 16 values (2 * e2m1)
  const uint2* quad_u2;      // [256]  a byte's four 2-bit codes
  const uint2* quad_s2;      // [256]  the same less 1 (Q2_0: -1, 0, 1, 2)
  const uint2* quad_hi6;     // [256]  16 c - 32 of them: the high bits of Q6_K's codes, with its bias
  const uint2* quad_mask;    // [16]   4 b - 4 of four bits: Q3_K's mask
  const uint16_t* int8;      // [256]  a signed byte
  // Scales of a group.
  const uint32_t* twice_u6;  // [64]   the pair (v, v)
  const uint32_t* twice_s6;  // [64]   (v - 32, v - 32)
  const uint32_t* parts_s8;  // [256]  a signed byte s = 16 hi + lo, lo 0..15: hi | lo << 16
  const uint32_t* parts_s6;  // [64]   v - 32 = 8 hi + lo, lo 0..7: hi | lo << 16
  // The kernel for inputs in 8 bits multiplies a weight's code as an int8 (byte i: selector i).
  uint32_t iq2_s8[3][2];         // IQ2_XXS, _XS, _S: the magnitudes, the magnitudes negated
  uint32_t iq3_s8[2][2];         // IQ3_XXS, IQ3_S: magnitudes 0..3, 4..7
  uint32_t iq1_s8[2];            // 8 w + 1, then 8 w - 1, for w = -1, 0, 1
  const uint16_t* pair_iq4_s8;   // [256]  a byte's two nibbles through IQ4's 16 values, as int8 (the low one first)
  const uint16_t* pair_mxfp4_s8; // [256]  the same through MXFP4's 16 values, as int8
  // IQ4's 16 values as int8 (byte i: value i), in registers: k_round_blocks_warps looks a nibble up with byte_perm.
  uint32_t iq4_s8[4];
};
const BlockTables& block_tables();

// A round (1..16 rows) on GGML block weights whose inputs are 8 bits (W.q8), on the rows blocks_stage left in the
// launch's scratch (gemm_blocks_round.cu).
void gemm_blocks_round_q8(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L);
// The same on a stack's n runs (2 or 3, their columns side by side in y) in one launch: whether it takes them (their
// types compiled together, gemm_blocks_ops.h, and the first run's rules run it on dp4a), and the launch.
bool gemm_blocks_round_q8_stack_takes(const BlockMat* W, int n, int rows, const Launch& L);
void gemm_blocks_round_q8_stack(const BlockMat* W, int n, int K, float* y, int ldy, int rows, bool accumulate,
                                const Launch& L);

}  // namespace flyx::cuda
