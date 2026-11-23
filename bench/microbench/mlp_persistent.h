// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The persistent-MLP experiment (bench only; see mlp_persistent.cu).
#pragma once

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

struct MlpPersistentOptions {
  bool prefetch = false;  // weights of the next phase into L2 ahead of their use (measured slower)
  int mode = 3;           // phases to run (timing them alone): bit 0 gate|up + SwiGLU, bit 1 down
  int blocks_per_sm = 0;  // 0: as many as fit
};

// x [rows, H] += down(swiglu(gate_up(rmsnorm(x)))) on NVFP4 weights (gate|up
// [2I, H], down [H, I]), rows <= 8, in one cooperative launch on st.  act:
// rows * I bf16 of scratch; sync: 8 bytes, zero before the first launch and
// left at zero by every launch (one launch at a time per sync).  Returns the
// blocks per SM it ran with, 0 if it does not take these shapes, rows or
// tables (then nothing ran).
int mlp_persistent(float* x, int H, const float* norm_w, float offset, float eps, const uint8_t* Qgu,
                   const uint8_t* Sgu, const float* RSgu, const uint8_t* Qd, const uint8_t* Sd, const float* RSd, int I,
                   int rows, bf16* act, unsigned* sync, int tuning, const MlpPersistentOptions& o, cudaStream_t st);

}  // namespace flyx::cuda
