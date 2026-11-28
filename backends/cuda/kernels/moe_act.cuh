// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// An MoE block's activation of its experts' gate and up values (kernels.cuh MoeAct), for every expert kernel.
#pragma once

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

// mact's activation (kernels.cuh MoeAct) of expert e's gate value g and up value u at column col of I, its biases
// added first where it has them; SITU with no beta is SWIGLU.
__device__ __forceinline__ float moe_act_val(const MoeAct& mact, float g, float u, int e, int col, int I) {
  if (mact.bias) {
    const float* br = mact.bias + size_t(e) * 2 * I;
    g += br[col];
    u += br[I + col];
  }
  if (mact.kind == FLYX_MOE_SWIGLU_OAI) {
    const float gc = fminf(g, mact.b), uc = fminf(fmaxf(u, -mact.b), mact.b);
    return (uc + 1.f) * gc * sigm(mact.a * gc);
  }
  if (mact.kind == FLYX_MOE_GELU_TANH) return gelu_tanh(g) * u;
  if (mact.kind == FLYX_MOE_SITU && mact.a > 0.f) {
    const float act = mact.a * tanhf(g / mact.a) * sigm(g);
    return act * (mact.b > 0.f ? mact.b * tanhf(u / mact.b) : u);
  }
  return silu(g) * u;
}

}  // namespace flyx::cuda
