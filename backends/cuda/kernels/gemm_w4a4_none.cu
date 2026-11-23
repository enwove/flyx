// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Builds without sm_120: no W4A4 prefill GEMM (gemm_w4a4.cu needs sm_120a).
#include <cstdlib>

#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

bool gemm_w4a4_supported() { return false; }
void quantize_rows_nvfp4(const float*, int, int, int, uint8_t*, uint8_t*, float*, cudaStream_t) { std::abort(); }
void quantize_rows_nvfp4_bf16(const bf16*, int, int, int, const float*, uint8_t*, uint8_t*, float*, cudaStream_t) {
  std::abort();
}
void gemm_w4a4(const uint8_t*, const uint8_t*, const float*, const uint8_t*, const uint8_t*, const float*, int, int,
               float*, int, int, bool, cudaStream_t, int, bf16*, float*, bf16*) {
  std::abort();
}

}  // namespace flyx::cuda
