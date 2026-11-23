// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The certified output head. The e4m3 copy of the head scores
// every vocabulary row (gemm_fp8_skinny); a rigorous bound on how far each
// score can be from the full head's (gemm_skinny's) keeps only the rows that
// can hold a row's largest score; gemm_skinny_cols scores those exactly, with
// gemm_skinny's bits, and the rest become -inf.  The greedy choice is then
// exactly the full head's, for a third less to read.
//
// The bound: for row n of the head w_n and its copy q_n (RS[n] times the e4m3
// codes), and an activation row x,
//   |gemm_skinny(x, w_n) - gemm_fp8_skinny(x, q_n)|
//     <= |x.(w_n - q_n)| + the two GEMMs' rounding
//     <= ||x|| ||w_n - q_n|| + kHeadCertRounding ||x|| (||w_n|| + ||q_n||)
// (Cauchy-Schwarz; each GEMM's error is at most about K 2^-23 times
// sum |x_k w_k| <= ||x|| ||w||, 3e-4 relative for K = 2560 with the bf16 split
// of x, so 1e-3 leaves room).  coef[n] holds the factor of ||x||.
#include <cuda_fp8.h>

#include <climits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// Floats in an order atomicMax on int keeps: larger float, larger key.
__device__ __forceinline__ int order_key(float f) {
  const int i = __float_as_int(f);
  return i >= 0 ? i : i ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float key_value(int k) { return __int_as_float(k >= 0 ? k : k ^ 0x7FFFFFFF); }

// grid V, block 256.
__global__ void k_head_cert_coef(const bf16* __restrict__ W, const uint8_t* __restrict__ Q,
                                 const float* __restrict__ RS, int K, float* __restrict__ coef) {
  const int n = blockIdx.x;
  const float rs = RS[n];
  float e2 = 0.f, w2 = 0.f, q2 = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const float w = __bfloat162float(W[size_t(n) * K + k]);
    const float q = e4m3_to_float(Q[size_t(n) * K + k]) * rs;
    e2 += (w - q) * (w - q);
    w2 += w * w;
    q2 += q * q;
  }
  e2 = block_sum(e2);
  w2 = block_sum(w2);
  q2 = block_sum(q2);
  // 1e-4: the float sums above.
  if (threadIdx.x == 0) coef[n] = (sqrtf(e2) + kHeadCertRounding * (sqrtf(w2) + sqrtf(q2))) * 1.0001f;
}

// grid rows, block 256: hn[r] = ||x_r|| (rounded up), key[r] = the lowest key, *count = 0.
__global__ void k_cert_init(const float* __restrict__ x, int ldx, int K, float* __restrict__ hn, int* __restrict__ key,
                            int* __restrict__ count) {
  pdl_trigger();
  const int r = blockIdx.x;
  float s = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const float v = x[size_t(r) * ldx + k];
    s += v * v;
  }
  s = block_sum(s);
  if (threadIdx.x == 0) {
    hn[r] = sqrtf(s) * 1.0001f;
    key[r] = INT_MIN;
    if (r == 0) *count = 0;
  }
}

// grid ceil(V / 256), block 256: key[r] = max over n of the lower bound s[r][n] - hn[r] coef[n].
__global__ void k_cert_lower(const float* __restrict__ s, int lds, int V, int rows, const float* __restrict__ coef,
                             const float* __restrict__ hn, int* __restrict__ key) {
  pdl_trigger();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const float c = n < V ? coef[n] : 0.f;
  for (int r = 0; r < rows; ++r) {
    float lb = n < V ? s[size_t(r) * lds + n] - hn[r] * c : -INFINITY;
    lb = block_max(lb);
    if (threadIdx.x == 0) atomicMax(&key[r], order_key(lb));
  }
}

// grid ceil(V / 256), block 256: column n survives when some row's upper bound
// reaches that row's best lower bound; survivors are listed (in any order),
// the others' scores become -inf in every row.
__global__ void k_cert_select(float* __restrict__ s, int lds, int V, int rows, const float* __restrict__ coef,
                              const float* __restrict__ hn, const int* __restrict__ key, int* __restrict__ cols,
                              int* __restrict__ count) {
  pdl_trigger();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= V) return;
  const float c = coef[n];
  bool keep = false;
  for (int r = 0; r < rows; ++r) keep |= s[size_t(r) * lds + n] + hn[r] * c >= key_value(key[r]);
  if (keep) {
    cols[atomicAdd(count, 1)] = n;
  } else {
    for (int r = 0; r < rows; ++r) s[size_t(r) * lds + n] = -INFINITY;
  }
}

}  // namespace

void head_cert_coef(const bf16* W, const uint8_t* Q, const float* RS, int V, int K, float* coef, cudaStream_t st) {
  k_head_cert_coef<<<V, 256, 0, st>>>(W, Q, RS, K, coef);
}

void head_certify(const float* x, int ldx, const bf16* W, const float* coef, int V, int K, float* logits, int ldl,
                  int rows, float* hn, int* key, int* cols, int* count, const Launch& L) {
  k_cert_init<<<rows, 256, 0, L.st>>>(x, ldx, K, hn, key, count);
  k_cert_lower<<<ceil_div(V, 256), 256, 0, L.st>>>(logits, ldl, V, rows, coef, hn, key);
  k_cert_select<<<ceil_div(V, 256), 256, 0, L.st>>>(logits, ldl, V, rows, coef, hn, key, cols, count);
  gemm_skinny_cols(x, ldx, W, K, cols, count, logits, ldl, rows, L);
}

}  // namespace flyx::cuda
