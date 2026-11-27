// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
// Cold slots use a valid temporary expert until the CPU result replaces them.
void moe_place_ids(int* ids, int pairs, const int* slots, uint32_t* counts, cudaStream_t st);
void moe_resident_ids(const int* ids, int* safe, int pairs, int hot, cudaStream_t st);

// For a bank, group just its routed pairs and leave down rows in sc->down_buf for one final ordered reduction.
int moe_bank_gate_up(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out, const MoeScratch* sc,
                const Launch& L, bool q8_ready, bf16* act_bf16, int first);
void moe_bank_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows, int top_k,
              int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc, const Launch& L,
              const bf16* act_bf16, int grouped, int first);
}  // namespace flyx::cuda
