// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Builds without sm_120: no MXFP8 inputs for MXFP4 experts (moe_mxfp4_mxfp8.cu needs sm_120a).
#include <cstdlib>

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

bool moe_mxfp4_mxfp8_supported() { return false; }
void moe_mxfp4_mxfp8_mma(bool, bool, int, const uint8_t*, const uint8_t*, const ExpertMat&, int, int, int,
                         const MoeScratch&, int, int, MoeAct, float*, int, const Launch&) {
  std::abort();
}

}  // namespace flyx::cuda
