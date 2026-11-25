// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The certified output head. gemm_fp8_skinny scores the vocabulary with an e4m3 copy of the head.
// A column survives if its score interval can contain any activation row's largest exact score.
// gemm_skinny_cols scores those columns with gemm_skinny's bits and the rest become -inf.
//
// For a weight row w_n and its scaled e4m3 copy q_n, let xs = bf16(x) on the sliced path and
// xs = x on the three-part path. In exact arithmetic x.w_n - xs.q_n = x.(w_n - q_n) + (x - xs).q_n.
// Cauchy-Schwarz on each term, with an allowance for the two GEMMs' arithmetic, gives
//   |gemm_skinny(x, w_n) - gemm_fp8_skinny(x, q_n)|
//     <= ||x|| (||w_n - q_n|| + kHeadCertRounding (||w_n|| + ||q_n||)) + ||x - xs|| ||q_n||.
// The 1e-3 allowance covers split and float32 GEMM arithmetic at head widths (K * 2^-23 is
// about 6.1e-4 at K = 5120, even before partitioning K). On the sliced path accumulation uses
// xs, up to 1 + 2^-8 times the magnitude of x for normal inputs. The last term covers the
// actual loss of the activation residual and vanishes on the three-part path.
#include <cuda_fp8.h>

#include <climits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// Squares accumulate upwards in each thread. This margin covers the float32 block reduction and final norm arithmetic.
constexpr float kNormPadding = 1.0001f;

// Floats in an order atomicMax on int keeps: larger float, larger key.
__device__ __forceinline__ int order_key(float f) {
  const int i = __float_as_int(f);
  return i >= 0 ? i : i ^ 0x7FFFFFFF;
}
__device__ __forceinline__ float key_value(int k) { return __int_as_float(k >= 0 ? k : k ^ 0x7FFFFFFF); }

// grid V, block 256.
__global__ void k_head_cert_coef(const bf16* __restrict__ W, const uint8_t* __restrict__ Q,
                                 const float* __restrict__ RS, int K, float* __restrict__ coef,
                                 float* __restrict__ qn) {
  const int n = blockIdx.x;
  const float rs = RS[n];
  float e2 = 0.f, w2 = 0.f, q2 = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const float w = __bfloat162float(W[size_t(n) * K + k]);
    const float q = e4m3_to_float(Q[size_t(n) * K + k]) * rs;
    const float e = w - q;
    e2 = __fmaf_ru(e, e, e2);
    w2 = __fmaf_ru(w, w, w2);
    q2 = __fmaf_ru(q, q, q2);
  }
  e2 = block_sum(e2);
  w2 = block_sum(w2);
  q2 = block_sum(q2);
  if (threadIdx.x == 0) {
    const float qnorm = sqrtf(q2);
    coef[n] = (sqrtf(e2) + kHeadCertRounding * (sqrtf(w2) + qnorm)) * kNormPadding;
    qn[n] = qnorm * kNormPadding;
  }
}

// grid rows, block 256. hn and dn bound ||x|| and ||x - xs|| for this row.
__global__ void k_cert_init(const float* __restrict__ x, int ldx, int K, float* __restrict__ hn,
                            float* __restrict__ dn, int* __restrict__ key, int* __restrict__ count) {
  pdl_trigger();
  const int r = blockIdx.x;
  const bool rounds_input = gemm_fp8_skinny_rounds_input(K);
  float sums[2] = {};
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const float v = x[size_t(r) * ldx + k];
    sums[0] = __fmaf_ru(v, v, sums[0]);
    if (rounds_input) {
      const float d = v - __bfloat162float(__float2bfloat16_rn(v));
      sums[1] = __fmaf_ru(d, d, sums[1]);
    }
  }
  block_sum_n(sums);
  if (threadIdx.x == 0) {
    hn[r] = sqrtf(sums[0]) * kNormPadding;
    dn[r] = sqrtf(sums[1]) * kNormPadding;
    key[r] = INT_MIN;
    if (r == 0) *count = 0;
  }
}

__device__ __forceinline__ float score_error_bound(float hn, float coef, float dn, float qn) {
  const float radius = __fmaf_ru(dn, qn, __fmul_ru(hn, coef));
  // Squared norms can overflow for finite inputs. A zero factor times such a norm gives no basis for pruning.
  return isnan(radius) ? INFINITY : radius;
}

// grid ceil(V / 256), block 256. key[r] is the largest lower bound in row r.
__global__ void k_cert_lower(const float* __restrict__ s, int lds, int V, int rows, const float* __restrict__ coef,
                             const float* __restrict__ qn, const float* __restrict__ hn, const float* __restrict__ dn,
                             int* __restrict__ key) {
  pdl_trigger();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const float c = n < V ? coef[n] : 0.f;
  const float q = n < V ? qn[n] : 0.f;
  for (int r = 0; r < rows; ++r) {
    float lb = n < V ? s[size_t(r) * lds + n] - score_error_bound(hn[r], c, dn[r], q) : -INFINITY;
    lb = block_max(lb);
    if (threadIdx.x == 0) atomicMax(&key[r], order_key(lb));
  }
}

// grid ceil(V / 256), block 256: column n survives when some row's upper bound
// reaches that row's best lower bound; survivors are listed (in any order),
// the others' scores become -inf in every row.
__global__ void k_cert_select(float* __restrict__ s, int lds, int V, int rows, const float* __restrict__ coef,
                              const float* __restrict__ qn, const float* __restrict__ hn, const float* __restrict__ dn,
                              const int* __restrict__ key, int* __restrict__ cols, int* __restrict__ count) {
  pdl_trigger();
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= V) return;
  const float c = coef[n], q = qn[n];
  bool keep = false;
  for (int r = 0; r < rows; ++r)
    keep |= s[size_t(r) * lds + n] + score_error_bound(hn[r], c, dn[r], q) >= key_value(key[r]);
  if (keep) {
    cols[atomicAdd(count, 1)] = n;
  } else {
    for (int r = 0; r < rows; ++r) s[size_t(r) * lds + n] = -INFINITY;
  }
}

}  // namespace

void head_cert_coef(const bf16* W, const uint8_t* Q, const float* RS, int V, int K, float* coef, cudaStream_t st) {
  k_head_cert_coef<<<V, 256, 0, st>>>(W, Q, RS, K, coef, coef + V);
}

void head_certify(const float* x, int ldx, const bf16* W, const float* coef, int V, int K, float* logits, int ldl,
                  int rows, float* norms, int* key, int* cols, int* count, const Launch& L) {
  float *hn = norms, *dn = norms + rows;
  const float* qn = coef + V;
  k_cert_init<<<rows, 256, 0, L.st>>>(x, ldx, K, hn, dn, key, count);
  k_cert_lower<<<ceil_div(V, 256), 256, 0, L.st>>>(logits, ldl, V, rows, coef, qn, hn, dn, key);
  k_cert_select<<<ceil_div(V, 256), 256, 0, L.st>>>(logits, ldl, V, rows, coef, qn, hn, dn, key, cols, count);
  gemm_skinny_cols(x, ldx, W, K, cols, count, logits, ldl, rows, L);
}

}  // namespace flyx::cuda
