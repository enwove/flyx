// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__global__ void k_hc_rows(int phase, float* x, float* wide, const float* norm, const float* inject, const float* gain,
                          int streams, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i >= hidden) return;
  const size_t x_at = size_t(r) * hidden + i;
  float sum = 0;
  for (int c = 0; c < streams; ++c) {
    const size_t stream = size_t(r) * streams + c, j = stream * hidden + i;
    if (phase == FLYX_HC_EXPAND)
      wide[j] = x[x_at];
    else if (phase == FLYX_HC_SCATTER)
      wide[j] += (2.f / (1.f + expf(-inject[stream] / streams))) * x[x_at];
    else
      sum += norm[j] / (1.f + expf(-gain[j]));
  }
  if (phase == FLYX_HC_MIX || phase == FLYX_HC_COLLAPSE) x[x_at] = sum / streams;
}

__global__ void k_hc_down(float* x, int64_t n, int streams) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = silu(x[i] / streams);
}

__global__ void k_shared_add(float* x, const float* input, const float* router, const float* expert, int hidden) {
  const size_t row = size_t(blockIdx.x) * hidden;
  float dot = 0;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dot += input[row + i] * router[i];
  const float gate = 1.f / (1.f + expf(-block_sum(dot)));
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) x[row + i] += gate * expert[row + i];
}

}  // namespace

void hc_rows(int phase, float* x, float* wide, const float* norm, const float* inject, const float* gain, int rows,
             int streams, int hidden, cudaStream_t st) {
  k_hc_rows<<<dim3((hidden + 255) / 256, rows), 256, 0, st>>>(phase, x, wide, norm, inject, gain, streams, hidden);
}

void hc_down_silu(float* x, int64_t elements, int streams, cudaStream_t st) {
  k_hc_down<<<(elements + 255) / 256, 256, 0, st>>>(x, elements, streams);
}

void shared_expert_add(float* x, const float* input, const float* router, const float* expert, int rows, int hidden,
                       cudaStream_t st) {
  k_shared_add<<<rows, 256, 0, st>>>(x, input, router, expert, hidden);
}
}  // namespace flyx::cuda
