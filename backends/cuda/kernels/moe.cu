// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Sparse MoE kernels: top-k routing, SiTU activation, and strictly row-independent expert GEMMs
// supporting NVFP4 (with per-expert F32 scale, bf16 activations), all 21 GGML block types (q8 and a16h),
// and BF16 weights.  Decode/verify (rows <= 16) runs zero-scratch per-(row, k) kernels; prefill (rows > 16)
// groups token-expert pairs by expert for weight reuse with the exact same per-row K-reduction order (0.00e+00).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_types.cuh"
#include "backends/cuda/kernels/gemm_internal.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kMoeWarps = 4;  // 128 threads per block
constexpr int kMoeKT = 1024;  // 32 lanes * 32 weights per lane: fixed K-tile across decode and grouped prefill

__device__ __forceinline__ float situ_val(float g, float u, float beta, float linear_beta) {
  if (beta > 0.f) {
    const float act = beta * tanhf(g / beta) * sigm(g);
    const float lin = linear_beta > 0.f ? (linear_beta * tanhf(u / linear_beta)) : u;
    return act * lin;
  }
  return silu(g) * u;
}

__device__ __forceinline__ float a16h_scale(float m) {
  if (!(m > 0.f) || !isfinite(m)) return 1.f;
  const uint32_t u = __float_as_uint(m);
  const int e = int((u >> 23) & 0xFFu) - 127;
  const int se = max(e, -111) - 15;
  return __uint_as_float(uint32_t(se + 127) << 23);
}

// ---- moe_route & situ_rows ----------------------------------------------------------------
constexpr int kRouteThreads = 128;

__global__ void __launch_bounds__(kRouteThreads) k_moe_route(const float* __restrict__ logits,
                                                             const float* __restrict__ bias, int E, int top_k,
                                                             int gating_sigmoid, int norm_weights, float norm_eps,
                                                             float route_scale, int* __restrict__ idx_out,
                                                             float* __restrict__ w_out) {
  pdl_trigger();
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* lr = logits + size_t(r) * E;
  extern __shared__ float sh[];  // probs[E], sel[E]
  float* probs = sh;
  float* sel = sh + E;
  __shared__ float s_val[kRouteThreads / 32];
  __shared__ int s_idx[kRouteThreads / 32];
  __shared__ float s_norm;

  if (gating_sigmoid) {
    for (int e = tid; e < E; e += kRouteThreads) {
      const float p = sigm(lr[e]);
      probs[e] = p;
      sel[e] = p + (bias ? bias[e] : 0.f);
    }
  } else {
    float mx = -INFINITY;
    for (int e = tid; e < E; e += kRouteThreads) mx = fmaxf(mx, lr[e]);
    mx = block_max(mx);
    float z = 0.f;
    for (int e = tid; e < E; e += kRouteThreads) {
      const float v = expf(lr[e] - mx);
      probs[e] = v;
      z += v;
    }
    z = block_sum(z);
    const float inv_z = 1.f / z;
    for (int e = tid; e < E; e += kRouteThreads) {
      const float p = probs[e] * inv_z;
      probs[e] = p;
      sel[e] = p + (bias ? bias[e] : 0.f);
    }
  }
  __syncthreads();

  int* out_i = idx_out + size_t(r) * top_k;
  float* out_w = w_out + size_t(r) * top_k;
  float sum_w = 0.f;
  for (int k = 0; k < top_k; ++k) {
    float bv = -INFINITY;
    int bi = -1;
    for (int e = tid; e < E; e += kRouteThreads) {
      const float v = sel[e];
      if (bi < 0 || score_better(v, e, bv, bi)) bv = v, bi = e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (oi >= 0 && (bi < 0 || score_better(ov, oi, bv, bi))) bv = ov, bi = oi;
    }
    if (lane == 0) {
      s_val[warp] = bv;
      s_idx[warp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      float best_v = -INFINITY;
      int best_i = 0;
      for (int w = 0; w < kRouteThreads / 32; ++w)
        if (s_idx[w] >= 0 && (w == 0 || score_better(s_val[w], s_idx[w], best_v, best_i)))
          best_v = s_val[w], best_i = s_idx[w];
      out_i[k] = best_i;
      const float pw = probs[best_i];
      out_w[k] = pw;
      sum_w += pw;
      sel[best_i] = -INFINITY;
    }
    __syncthreads();
  }
  if (tid == 0) s_norm = norm_weights ? (route_scale / (sum_w + norm_eps)) : route_scale;
  __syncthreads();
  for (int k = tid; k < top_k; k += kRouteThreads) out_w[k] *= s_norm;
}

__global__ void k_situ_rows(const float* __restrict__ gu, int ld_gu, int I, float beta, float linear_beta,
                            float* __restrict__ out, int ld_out) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i >= I) return;
  const float* row = gu + size_t(r) * ld_gu;
  out[size_t(r) * ld_out + i] = situ_val(row[i], row[I + i], beta, linear_beta);
}

// ---- Expert grouping for prefill (rows > 16) ----------------------------------------------
// Single-block sort of M = R * top_k pairs into per-expert ranges [offsets[e], offsets[e + 1]).
__global__ void k_moe_group_pairs(const int* __restrict__ eidx, int M, int E, int* __restrict__ counts,
                                  int* __restrict__ offsets, int* __restrict__ pairs) {
  const int tid = threadIdx.x;
  for (int e = tid; e < E; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  for (int i = tid; i < M; i += blockDim.x) atomicAdd(&counts[eidx[i]], 1);
  __syncthreads();
  if (tid == 0) {
    int sum = 0;
    for (int e = 0; e < E; ++e) {
      offsets[e] = sum;
      sum += counts[e];
      counts[e] = offsets[e];
    }
    offsets[E] = sum;
  }
  __syncthreads();
  for (int i = tid; i < M; i += blockDim.x) {
    const int e = eidx[i];
    const int pos = atomicAdd(&counts[e], 1);
    pairs[pos] = i;
  }
}

__global__ void k_moe_weighted_sum(const float* __restrict__ down_buf, const float* __restrict__ ew, int top_k, int H,
                                   float* __restrict__ out, int ld_out, int accumulate) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (n >= H) return;
  float total = 0.f;
  const float* w = ew + size_t(r) * top_k;
  const float* base = down_buf + size_t(r) * top_k * H + n;
  for (int k = 0; k < top_k; ++k) total = __fmaf_rn(w[k], base[size_t(k) * H], total);
  float* dst = out + size_t(r) * ld_out + n;
  *dst = accumulate ? (*dst + total) : total;
}

// ---- Shared-memory activation tile stagers (fixed per-tile layout for bit-identity) --------
template <int BT>
__device__ __forceinline__ void stage_f32_tile(const float* __restrict__ x, int ldx, const int* row_ids, int nb, int k0,
                                               int kt, float sm[BT][kMoeKT]) {
  for (int idx = threadIdx.x; idx < nb * kt; idx += blockDim.x) {
    const int b = idx / kt, j = idx - b * kt;
    sm[b][j] = x[size_t(row_ids[b]) * ldx + k0 + j];
  }
}

template <int BT>
__device__ __forceinline__ void stage_bf16_tile(const float* __restrict__ x, int ldx, const int* row_ids, int nb,
                                                int k0, int kt, bf16 sm[BT][kMoeKT]) {
  for (int idx = threadIdx.x; idx < nb * kt; idx += blockDim.x) {
    const int b = idx / kt, j = idx - b * kt;
    sm[b][j] = __float2bfloat16(x[size_t(row_ids[b]) * ldx + k0 + j]);
  }
}

// Stage Q8 groups of 32 for nb rows over [k0, k0 + kt). Each warp handles one 32-value group at a time.
template <int BT>
__device__ __forceinline__ void stage_q8_tile(const float* __restrict__ x, int ldx, const int* row_ids, int nb, int k0,
                                              int kt, int8_t sm_q[BT][kMoeKT], float sm_d[BT][kMoeKT / 32],
                                              float sm_s[BT][kMoeKT / 16]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int ng = kt >> 5;  // groups of 32 in this tile (<= 32)
  const int total_g = nb * ng;
  for (int gi = warp; gi < total_g; gi += kMoeWarps) {
    const int b = gi / ng, g = gi - b * ng;
    const float v = x[size_t(row_ids[b]) * ldx + k0 + g * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, s));
    const float dg = amax / 127.f;
    const int p = q8_of(v, dg);
    sm_q[b][g * 32 + lane] = int8_t(p);
    int s16 = p;
#pragma unroll
    for (int s = 8; s > 0; s >>= 1) s16 += __shfl_xor_sync(0xffffffffu, s16, s);
    if ((lane & 15) == 0) sm_s[b][g * 2 + (lane >> 4)] = dg * float(s16);
    if (lane == 0) sm_d[b][g] = dg;
  }
}

template <int BT>
__device__ __forceinline__ void stage_a16h_tile(const float* __restrict__ x, int ldx, const int* row_ids,
                                                const float* row_sc, int nb, int k0, int kt, float sm[BT][kMoeKT]) {
  for (int idx = threadIdx.x; idx < nb * kt; idx += blockDim.x) {
    const int b = idx / kt, j = idx - b * kt;
    const float bf = __bfloat162float(__float2bfloat16(x[size_t(row_ids[b]) * ldx + k0 + j]));
    const float sc = row_sc[b];
    sm[b][j] = __half2float(__float2half_rn(bf / sc)) * sc;
  }
}

// ---- BF16 Expert Kernels ------------------------------------------------------------------
template <int BT>
__device__ __forceinline__ void bf16_row_step(const bf16* __restrict__ w_row, int k0, int kt, int lane, int nb,
                                              const float sm[BT][kMoeKT], float acc[BT]) {
  for (int j = lane; j < kt; j += 32) {
    const float wv = __bfloat162float(w_row[k0 + j]);
#pragma unroll
    for (int b = 0; b < BT; ++b)
      if (b < nb) acc[b] = __fmaf_rn(sm[b][j], wv, acc[b]);
  }
}

template <bool FUSE_ACT>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_bf16_gu_decode(const float* __restrict__ x, int ldx, const bf16* __restrict__ W, const float* __restrict__ RS,
                         int N_exp, int I, int K, const int* __restrict__ eidx, int top_k, float situ_beta,
                         float situ_linear_beta, float* __restrict__ out, int ld_out) {
  pdl_trigger();
  __shared__ float sm[1][kMoeKT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I : 2 * I;
  const int k_slot = blockIdx.y, r = blockIdx.z;
  const int e = eidx[r * top_k + k_slot];
  const float rs = RS ? RS[e] : 1.f;
  const bf16* w0 = W + (size_t(e) * N_exp + min(col, N_exp - 1)) * K;
  const bf16* w1 = FUSE_ACT ? (W + (size_t(e) * N_exp + min(col + I, N_exp - 1)) * K) : nullptr;
  float ag[1] = {0.f}, au[1] = {0.f};
  const int rids[1] = {r};
  for (int k0 = 0; k0 < K; k0 += kMoeKT) {
    const int kt = min(kMoeKT, K - k0);
    stage_f32_tile<1>(x, ldx, rids, 1, k0, kt, sm);
    __syncthreads();
    if (col < N_work) {
      bf16_row_step<1>(w0, k0, kt, lane, 1, sm, ag);
      if constexpr (FUSE_ACT) bf16_row_step<1>(w1, k0, kt, lane, 1, sm, au);
    }
    __syncthreads();
  }
  if (col < N_work) {
    const float g = warp_sum(ag[0]) * rs;
    if constexpr (FUSE_ACT) {
      const float u = warp_sum(au[0]) * rs;
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
    } else {
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = g;
    }
  }
}

__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_bf16_down_decode(const float* __restrict__ act, int ld_act, const bf16* __restrict__ W,
                           const float* __restrict__ RS, int H, int K, const int* __restrict__ eidx,
                           const float* __restrict__ ew, int top_k, float* __restrict__ out, int ld_out,
                           int accumulate) {
  pdl_trigger();
  __shared__ float sm[1][kMoeKT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int r = blockIdx.y;
  float total = 0.f;
  for (int k_slot = 0; k_slot < top_k; ++k_slot) {
    const int e = eidx[r * top_k + k_slot];
    const float rs = RS ? RS[e] : 1.f;
    const bf16* w0 = W + (size_t(e) * H + min(col, H - 1)) * K;
    const int rids[1] = {r * top_k + k_slot};
    float acc[1] = {0.f};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      stage_f32_tile<1>(act, ld_act, rids, 1, k0, kt, sm);
      __syncthreads();
      if (col < H) bf16_row_step<1>(w0, k0, kt, lane, 1, sm, acc);
      __syncthreads();
    }
    if (col < H) {
      const float v = warp_sum(acc[0]) * rs;
      if (lane == 0) total = __fmaf_rn(ew[r * top_k + k_slot], v, total);
    }
  }
  if (col < H && lane == 0) {
    float* dst = out + size_t(r) * ld_out + col;
    *dst = accumulate ? (*dst + total) : total;
  }
}

template <bool FUSE_ACT, bool IS_DOWN>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_bf16_grouped(const float* __restrict__ x, int ldx, const bf16* __restrict__ W, const float* __restrict__ RS,
                       int N_exp, int I_or_H, int K, const int* __restrict__ offsets, const int* __restrict__ pairs,
                       int top_k, float situ_beta, float situ_linear_beta, float* __restrict__ out, int ld_out) {
  constexpr int BT = 4;
  __shared__ float sm[BT][kMoeKT];
  const int e = blockIdx.y;
  const int p_start = offsets[e], p_end = offsets[e + 1];
  if (p_start >= p_end) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const float rs = RS ? RS[e] : 1.f;
  const bf16* w0 = W + (size_t(e) * N_exp + min(col, N_exp - 1)) * K;
  const bf16* w1 = FUSE_ACT ? (W + (size_t(e) * N_exp + min(col + I_or_H, N_exp - 1)) * K) : nullptr;

  for (int p0 = p_start; p0 < p_end; p0 += BT) {
    const int nb = min(BT, p_end - p0);
    int rids[BT], pids[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) {
      pids[b] = b < nb ? pairs[p0 + b] : 0;
      rids[b] = IS_DOWN ? pids[b] : (pids[b] / top_k);
    }
    float ag[BT] = {}, au[BT] = {};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      stage_f32_tile<BT>(x, ldx, rids, nb, k0, kt, sm);
      __syncthreads();
      if (col < N_work) {
        bf16_row_step<BT>(w0, k0, kt, lane, nb, sm, ag);
        if constexpr (FUSE_ACT) bf16_row_step<BT>(w1, k0, kt, lane, nb, sm, au);
      }
      __syncthreads();
    }
    if (col < N_work) {
#pragma unroll
      for (int b = 0; b < BT; ++b) {
        if (b >= nb) break;
        const float g = warp_sum(ag[b]) * rs;
        if constexpr (FUSE_ACT) {
          const float u = warp_sum(au[b]) * rs;
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
        } else {
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = g;
        }
      }
    }
  }
}

// ---- NVFP4 Expert Kernels (bf16 activations + per-expert F32 scale rs[e]) ------------------
template <int BT>
__device__ __forceinline__ void nvfp4_row_step(const uint8_t* __restrict__ q_row, const uint8_t* __restrict__ s_row,
                                               int k0, int kt, int lane, int nb, const bf16 sm[BT][kMoeKT],
                                               float acc[BT]) {
  const int g = lane;
  if (g * 32 >= kt) return;
  const int k = k0 + g * 32;
  const uint4 qv = ldg_stream(q_row + (k >> 1));
  const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w};
  const float sc0 = e4m3_to_float(s_row[k >> 4]);
  const float sc1 = e4m3_to_float(s_row[(k >> 4) + 1]);

  float wf[32];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t b2[4];
    fp4x8_to_bf16(qw[q], b2);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 fv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&b2[p]));
      wf[q * 8 + p * 2 + 0] = fv.x;
      wf[q * 8 + p * 2 + 1] = fv.y;
    }
  }
#pragma unroll
  for (int b = 0; b < BT; ++b) {
    if (b >= nb) break;
    const __nv_bfloat162* xr = reinterpret_cast<const __nv_bfloat162*>(&sm[b][g * 32]);
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const float2 xv = __bfloat1622float2(xr[p]);
      s0 = __fmaf_rn(xv.x, wf[p * 2], s0);
      s0 = __fmaf_rn(xv.y, wf[p * 2 + 1], s0);
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const float2 xv = __bfloat1622float2(xr[8 + p]);
      s1 = __fmaf_rn(xv.x, wf[16 + p * 2], s1);
      s1 = __fmaf_rn(xv.y, wf[16 + p * 2 + 1], s1);
    }
    acc[b] = __fmaf_rn(s0, sc0, acc[b]);
    acc[b] = __fmaf_rn(s1, sc1, acc[b]);
  }
}

template <bool FUSE_ACT>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_nvfp4_gu_decode(const float* __restrict__ x, int ldx, const uint8_t* __restrict__ Q,
                          const uint8_t* __restrict__ S, const float* __restrict__ RS, int N_exp, int I, int K,
                          const int* __restrict__ eidx, int top_k, float situ_beta, float situ_linear_beta,
                          float* __restrict__ out, int ld_out) {
  pdl_trigger();
  __shared__ bf16 sm[1][kMoeKT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I : 2 * I;
  const int k_slot = blockIdx.y, r = blockIdx.z;
  const int e = eidx[r * top_k + k_slot];
  const float rs = RS ? RS[e] : 1.f;
  const size_t r0_idx = size_t(e) * N_exp + min(col, N_exp - 1);
  const size_t r1_idx = FUSE_ACT ? (size_t(e) * N_exp + min(col + I, N_exp - 1)) : 0;
  const uint8_t* q0 = Q + r0_idx * (K >> 1);
  const uint8_t* s0 = S + r0_idx * (K >> 4);
  const uint8_t* q1 = FUSE_ACT ? (Q + r1_idx * (K >> 1)) : nullptr;
  const uint8_t* s1 = FUSE_ACT ? (S + r1_idx * (K >> 4)) : nullptr;

  float ag[1] = {0.f}, au[1] = {0.f};
  const int rids[1] = {r};
  for (int k0 = 0; k0 < K; k0 += kMoeKT) {
    const int kt = min(kMoeKT, K - k0);
    stage_bf16_tile<1>(x, ldx, rids, 1, k0, kt, sm);
    __syncthreads();
    if (col < N_work) {
      nvfp4_row_step<1>(q0, s0, k0, kt, lane, 1, sm, ag);
      if constexpr (FUSE_ACT) nvfp4_row_step<1>(q1, s1, k0, kt, lane, 1, sm, au);
    }
    __syncthreads();
  }
  if (col < N_work) {
    const float g = warp_sum(ag[0]) * rs;
    if constexpr (FUSE_ACT) {
      const float u = warp_sum(au[0]) * rs;
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
    } else {
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = g;
    }
  }
}

__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_nvfp4_down_decode(const float* __restrict__ act, int ld_act, const uint8_t* __restrict__ Q,
                            const uint8_t* __restrict__ S, const float* __restrict__ RS, int H, int K,
                            const int* __restrict__ eidx, const float* __restrict__ ew, int top_k,
                            float* __restrict__ out, int ld_out, int accumulate) {
  pdl_trigger();
  __shared__ bf16 sm[1][kMoeKT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int r = blockIdx.y;
  float total = 0.f;
  for (int k_slot = 0; k_slot < top_k; ++k_slot) {
    const int e = eidx[r * top_k + k_slot];
    const float rs = RS ? RS[e] : 1.f;
    const size_t r0_idx = size_t(e) * H + min(col, H - 1);
    const uint8_t* q0 = Q + r0_idx * (K >> 1);
    const uint8_t* s0 = S + r0_idx * (K >> 4);
    const int rids[1] = {r * top_k + k_slot};
    float acc[1] = {0.f};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      stage_bf16_tile<1>(act, ld_act, rids, 1, k0, kt, sm);
      __syncthreads();
      if (col < H) nvfp4_row_step<1>(q0, s0, k0, kt, lane, 1, sm, acc);
      __syncthreads();
    }
    if (col < H) {
      const float v = warp_sum(acc[0]) * rs;
      if (lane == 0) total = __fmaf_rn(ew[r * top_k + k_slot], v, total);
    }
  }
  if (col < H && lane == 0) {
    float* dst = out + size_t(r) * ld_out + col;
    *dst = accumulate ? (*dst + total) : total;
  }
}

template <bool FUSE_ACT, bool IS_DOWN>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_nvfp4_grouped(const float* __restrict__ x, int ldx, const uint8_t* __restrict__ Q,
                        const uint8_t* __restrict__ S, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                        const int* __restrict__ offsets, const int* __restrict__ pairs, int top_k, float situ_beta,
                        float situ_linear_beta, float* __restrict__ out, int ld_out) {
  constexpr int BT = 8;
  __shared__ bf16 sm[BT][kMoeKT];
  const int e = blockIdx.y;
  const int p_start = offsets[e], p_end = offsets[e + 1];
  if (p_start >= p_end) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const float rs = RS ? RS[e] : 1.f;
  const size_t r0_idx = size_t(e) * N_exp + min(col, N_exp - 1);
  const size_t r1_idx = FUSE_ACT ? (size_t(e) * N_exp + min(col + I_or_H, N_exp - 1)) : 0;
  const uint8_t* q0 = Q + r0_idx * (K >> 1);
  const uint8_t* s0 = S + r0_idx * (K >> 4);
  const uint8_t* q1 = FUSE_ACT ? (Q + r1_idx * (K >> 1)) : nullptr;
  const uint8_t* s1 = FUSE_ACT ? (S + r1_idx * (K >> 4)) : nullptr;

  for (int p0 = p_start; p0 < p_end; p0 += BT) {
    const int nb = min(BT, p_end - p0);
    int rids[BT], pids[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) {
      pids[b] = b < nb ? pairs[p0 + b] : 0;
      rids[b] = IS_DOWN ? pids[b] : (pids[b] / top_k);
    }
    float ag[BT] = {}, au[BT] = {};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      stage_bf16_tile<BT>(x, ldx, rids, nb, k0, kt, sm);
      __syncthreads();
      if (col < N_work) {
        nvfp4_row_step<BT>(q0, s0, k0, kt, lane, nb, sm, ag);
        if constexpr (FUSE_ACT) nvfp4_row_step<BT>(q1, s1, k0, kt, lane, nb, sm, au);
      }
      __syncthreads();
    }
    if (col < N_work) {
#pragma unroll
      for (int b = 0; b < BT; ++b) {
        if (b >= nb) break;
        const float g = warp_sum(ag[b]) * rs;
        if constexpr (FUSE_ACT) {
          const float u = warp_sum(au[b]) * rs;
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
        } else {
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = g;
        }
      }
    }
  }
}

// ---- Packed GGML Block Expert Kernels (ALL 21 block types, Q8 dp4a and a16h exact) ---------
template <class F, int BT>
__device__ __forceinline__ void block_q8_row_step(const Tables& T, const typename F::Local& local,
                                                  const uint8_t* __restrict__ w_row, const uint16_t* __restrict__ d_row,
                                                  int k0, int kt, int lane, int nb, const int8_t sm_q[BT][kMoeKT],
                                                  const float sm_d[BT][kMoeKT / 32], const float sm_s[BT][kMoeKT / 16],
                                                  float acc[BT]) {
  static constexpr int GROUPS = F::kValues / 32, SUB = F::kSub;
  static constexpr bool MINS = F::kFlatRun != 0;
  if (lane * 32 >= kt) return;
  const int u = (k0 >> 5) + lane;
  const int G = lane % GROUPS;
  const size_t blk = size_t(u / GROUPS);
  const typename F::Unit raw = block_unit<F>(w_row, d_row, blk, G);
  const float d0 = scale_bits<F>(d_row[blk * F::kScales]);
  const float d1 = F::kScales > 1 ? scale_bits<F>(d_row[blk * F::kScales + 1]) : 0.f;
  uint32_t w[8];
  float f[2], m[2];
  F::unit_codes(T, local, raw, G, w);
  F::unit_factors(raw, G, d0, d1, f, m);
#pragma unroll
  for (int b = 0; b < BT; ++b) {
    if (b >= nb) break;
    const uint4 x0 = *reinterpret_cast<const uint4*>(&sm_q[b][lane * 32]);
    const uint4 x1 = *reinterpret_cast<const uint4*>(&sm_q[b][lane * 32 + 16]);
    const float dx = sm_d[b][lane];
    int s0 = 0;
    s0 = __dp4a(int(w[0]), int(x0.x), s0);
    s0 = __dp4a(int(w[1]), int(x0.y), s0);
    s0 = __dp4a(int(w[2]), int(x0.z), s0);
    s0 = __dp4a(int(w[3]), int(x0.w), s0);
    int s1 = (SUB == 32) ? s0 : 0;
    s1 = __dp4a(int(w[4]), int(x1.x), s1);
    s1 = __dp4a(int(w[5]), int(x1.y), s1);
    s1 = __dp4a(int(w[6]), int(x1.z), s1);
    s1 = __dp4a(int(w[7]), int(x1.w), s1);
    if constexpr (SUB == 32) s0 = s1;
    acc[b] = fmaf(float(s0), dx * f[0], acc[b]);
    if constexpr (MINS) {
      const float ds0 = sm_s[b][lane * 2], ds1 = sm_s[b][lane * 2 + 1];
      acc[b] = fmaf(SUB == 16 ? ds0 : (ds0 + ds1), m[0], acc[b]);
    }
    if constexpr (SUB == 16) {
      acc[b] = fmaf(float(s1), dx * f[1], acc[b]);
      if constexpr (MINS) acc[b] = fmaf(sm_s[b][lane * 2 + 1], m[1], acc[b]);
    }
  }
}

template <class F, int BT>
__device__ __forceinline__ void block_a16h_row_step(const Tables& T, const typename F::Local& local,
                                                    const uint8_t* __restrict__ w_row,
                                                    const uint16_t* __restrict__ d_row, int k0, int kt, int lane,
                                                    int nb, const float sm[BT][kMoeKT], float acc[BT]) {
  static constexpr int GROUPS = F::kValues / 32, SUB = F::kSub;
  static constexpr bool MINS = F::kFlatRun != 0;
  if (lane * 32 >= kt) return;
  const int u = (k0 >> 5) + lane;
  const int G = lane % GROUPS;
  const size_t blk = size_t(u / GROUPS);
  const typename F::Unit raw = block_unit<F>(w_row, d_row, blk, G);
  const float d0 = scale_bits<F>(d_row[blk * F::kScales]);
  const float d1 = F::kScales > 1 ? scale_bits<F>(d_row[blk * F::kScales + 1]) : 0.f;
  uint32_t w[8];
  float f[2], m[2];
  F::unit_codes(T, local, raw, G, w);
  F::unit_factors(raw, G, d0, d1, f, m);
  float wf[32];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int sub = (SUB == 16 && i >= 4) ? 1 : 0;
    const float fs = f[sub], ms = MINS ? m[sub] : 0.f;
    const uint32_t word = w[i];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int8_t c = int8_t((word >> (8 * j)) & 0xFFu);
      wf[i * 4 + j] = MINS ? fmaf(float(c), fs, ms) : (float(c) * fs);
    }
  }
#pragma unroll
  for (int b = 0; b < BT; ++b) {
    if (b >= nb) break;
    const float* xr = &sm[b][lane * 32];
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) s = __fmaf_rn(xr[j], wf[j], s);
    acc[b] += s;
  }
}

template <class F, bool FUSE_ACT, bool Q8>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_blocks_gu_decode(Tables T, const float* __restrict__ x, int ldx, const uint8_t* __restrict__ W,
                           const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I, int K,
                           const int* __restrict__ eidx, int top_k, float situ_beta, float situ_linear_beta,
                           float* __restrict__ out, int ld_out) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __shared__ int8_t sm_q[1][Q8 ? kMoeKT : 1];
  __shared__ float sm_d[1][Q8 ? kMoeKT / 32 : 1];
  __shared__ float sm_s[1][Q8 ? kMoeKT / 16 : 1];
  __shared__ float sm_f[1][Q8 ? 1 : kMoeKT];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I : 2 * I;
  const int k_slot = blockIdx.y, r = blockIdx.z;
  const int e = eidx[r * top_k + k_slot];
  const float rs = RS ? RS[e] : 1.f;
  const size_t bpr = size_t(K / F::kValues);
  const size_t r0_idx = size_t(e) * N_exp + min(col, N_exp - 1);
  const size_t r1_idx = FUSE_ACT ? (size_t(e) * N_exp + min(col + I, N_exp - 1)) : 0;
  const uint8_t* w0 = W + r0_idx * bpr * F::kStride;
  const uint16_t* d0 = D + r0_idx * bpr * F::kScales;
  const uint8_t* w1 = FUSE_ACT ? (W + r1_idx * bpr * F::kStride) : nullptr;
  const uint16_t* d1 = FUSE_ACT ? (D + r1_idx * bpr * F::kScales) : nullptr;

  float row_sc[1] = {1.f};
  if constexpr (!Q8) {
    float rm = 0.f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
      rm = fmaxf(rm, fabsf(__bfloat162float(__float2bfloat16(x[size_t(r) * ldx + k]))));
    row_sc[0] = a16h_scale(block_max(rm));
  }

  float ag[1] = {0.f}, au[1] = {0.f};
  const int rids[1] = {r};
  for (int k0 = 0; k0 < K; k0 += kMoeKT) {
    const int kt = min(kMoeKT, K - k0);
    if constexpr (Q8) stage_q8_tile<1>(x, ldx, rids, 1, k0, kt, sm_q, sm_d, sm_s);
    else stage_a16h_tile<1>(x, ldx, rids, row_sc, 1, k0, kt, sm_f);
    __syncthreads();
    if (col < N_work) {
      if constexpr (Q8) {
        block_q8_row_step<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_q, sm_d, sm_s, ag);
        if constexpr (FUSE_ACT) block_q8_row_step<F, 1>(T, local, w1, d1, k0, kt, lane, 1, sm_q, sm_d, sm_s, au);
      } else {
        block_a16h_row_step<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_f, ag);
        if constexpr (FUSE_ACT) block_a16h_row_step<F, 1>(T, local, w1, d1, k0, kt, lane, 1, sm_f, au);
      }
    }
    __syncthreads();
  }
  if (col < N_work) {
    const float g = warp_sum(ag[0]) * rs;
    if constexpr (FUSE_ACT) {
      const float u = warp_sum(au[0]) * rs;
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
    } else {
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = g;
    }
  }
}

template <class F, bool Q8>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_blocks_down_decode(Tables T, const float* __restrict__ act, int ld_act, const uint8_t* __restrict__ W,
                             const uint16_t* __restrict__ D, const float* __restrict__ RS, int H, int K,
                             const int* __restrict__ eidx, const float* __restrict__ ew, int top_k,
                             float* __restrict__ out, int ld_out, int accumulate) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __shared__ int8_t sm_q[1][Q8 ? kMoeKT : 1];
  __shared__ float sm_d[1][Q8 ? kMoeKT / 32 : 1];
  __shared__ float sm_s[1][Q8 ? kMoeKT / 16 : 1];
  __shared__ float sm_f[1][Q8 ? 1 : kMoeKT];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int r = blockIdx.y;
  const size_t bpr = size_t(K / F::kValues);
  float total = 0.f;
  for (int k_slot = 0; k_slot < top_k; ++k_slot) {
    const int e = eidx[r * top_k + k_slot];
    const float rs = RS ? RS[e] : 1.f;
    const size_t r0_idx = size_t(e) * H + min(col, H - 1);
    const uint8_t* w0 = W + r0_idx * bpr * F::kStride;
    const uint16_t* d0 = D + r0_idx * bpr * F::kScales;
    const int rids[1] = {r * top_k + k_slot};
    float row_sc[1] = {1.f};
    if constexpr (!Q8) {
      float rm = 0.f;
      for (int k = threadIdx.x; k < K; k += blockDim.x)
        rm = fmaxf(rm, fabsf(__bfloat162float(__float2bfloat16(act[size_t(rids[0]) * ld_act + k]))));
      row_sc[0] = a16h_scale(block_max(rm));
    }
    float acc[1] = {0.f};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      if constexpr (Q8) stage_q8_tile<1>(act, ld_act, rids, 1, k0, kt, sm_q, sm_d, sm_s);
      else stage_a16h_tile<1>(act, ld_act, rids, row_sc, 1, k0, kt, sm_f);
      __syncthreads();
      if (col < H) {
        if constexpr (Q8) block_q8_row_step<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_q, sm_d, sm_s, acc);
        else block_a16h_row_step<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_f, acc);
      }
      __syncthreads();
    }
    if (col < H) {
      const float v = warp_sum(acc[0]) * rs;
      if (lane == 0) total = __fmaf_rn(ew[r * top_k + k_slot], v, total);
    }
  }
  if (col < H && lane == 0) {
    float* dst = out + size_t(r) * ld_out + col;
    *dst = accumulate ? (*dst + total) : total;
  }
}

template <class F, bool FUSE_ACT, bool IS_DOWN, bool Q8>
__global__ void __launch_bounds__(32 * kMoeWarps)
    k_moe_blocks_grouped(Tables T, const float* __restrict__ x, int ldx, const uint8_t* __restrict__ W,
                         const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                         const int* __restrict__ offsets, const int* __restrict__ pairs, int top_k, float situ_beta,
                         float situ_linear_beta, float* __restrict__ out, int ld_out) {
  constexpr int BT = Q8 ? 8 : 4;
  __shared__ typename F::Local local;
  F::tables(T, local);
  __shared__ int8_t sm_q[BT][Q8 ? kMoeKT : 1];
  __shared__ float sm_d[BT][Q8 ? kMoeKT / 32 : 1];
  __shared__ float sm_s[BT][Q8 ? kMoeKT / 16 : 1];
  __shared__ float sm_f[BT][Q8 ? 1 : kMoeKT];
  __syncthreads();

  const int e = blockIdx.y;
  const int p_start = offsets[e], p_end = offsets[e + 1];
  if (p_start >= p_end) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const float rs = RS ? RS[e] : 1.f;
  const size_t bpr = size_t(K / F::kValues);
  const size_t r0_idx = size_t(e) * N_exp + min(col, N_exp - 1);
  const size_t r1_idx = FUSE_ACT ? (size_t(e) * N_exp + min(col + I_or_H, N_exp - 1)) : 0;
  const uint8_t* w0 = W + r0_idx * bpr * F::kStride;
  const uint16_t* d0 = D + r0_idx * bpr * F::kScales;
  const uint8_t* w1 = FUSE_ACT ? (W + r1_idx * bpr * F::kStride) : nullptr;
  const uint16_t* d1 = FUSE_ACT ? (D + r1_idx * bpr * F::kScales) : nullptr;

  for (int p0 = p_start; p0 < p_end; p0 += BT) {
    const int nb = min(BT, p_end - p0);
    int rids[BT], pids[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) {
      pids[b] = b < nb ? pairs[p0 + b] : 0;
      rids[b] = IS_DOWN ? pids[b] : (pids[b] / top_k);
    }
    float row_sc[BT] = {};
    if constexpr (!Q8) {
      for (int b = 0; b < nb; ++b) {
        float rm = 0.f;
        for (int k = threadIdx.x; k < K; k += blockDim.x)
          rm = fmaxf(rm, fabsf(__bfloat162float(__float2bfloat16(x[size_t(rids[b]) * ldx + k]))));
        row_sc[b] = a16h_scale(block_max(rm));
      }
    }
    float ag[BT] = {}, au[BT] = {};
    for (int k0 = 0; k0 < K; k0 += kMoeKT) {
      const int kt = min(kMoeKT, K - k0);
      if constexpr (Q8) stage_q8_tile<BT>(x, ldx, rids, nb, k0, kt, sm_q, sm_d, sm_s);
      else stage_a16h_tile<BT>(x, ldx, rids, row_sc, nb, k0, kt, sm_f);
      __syncthreads();
      if (col < N_work) {
        if constexpr (Q8) {
          block_q8_row_step<F, BT>(T, local, w0, d0, k0, kt, lane, nb, sm_q, sm_d, sm_s, ag);
          if constexpr (FUSE_ACT) block_q8_row_step<F, BT>(T, local, w1, d1, k0, kt, lane, nb, sm_q, sm_d, sm_s, au);
        } else {
          block_a16h_row_step<F, BT>(T, local, w0, d0, k0, kt, lane, nb, sm_f, ag);
          if constexpr (FUSE_ACT) block_a16h_row_step<F, BT>(T, local, w1, d1, k0, kt, lane, nb, sm_f, au);
        }
      }
      __syncthreads();
    }
    if (col < N_work) {
#pragma unroll
      for (int b = 0; b < BT; ++b) {
        if (b >= nb) break;
        const float g = warp_sum(ag[b]) * rs;
        if constexpr (FUSE_ACT) {
          const float u = warp_sum(au[b]) * rs;
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
        } else {
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = g;
        }
      }
    }
  }
}

template <class F>
void launch_blocks_gu(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                      float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out,
                      const MoeScratch* sc, cudaStream_t st) {
  const Tables T = block_tables();
  const int N_work = fuse_act ? I : 2 * I;
  const bool grouped = rows > 16 && sc && sc->counts && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
  if (!grouped) {
    const dim3 grid(ceil_div(N_work, kMoeWarps), top_k, rows);
    if (W.blk.q8) {
      if (fuse_act)
        k_moe_blocks_gu_decode<F, true, true><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
      else
        k_moe_blocks_gu_decode<F, false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    } else {
      if (fuse_act)
        k_moe_blocks_gu_decode<F, true, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
      else
        k_moe_blocks_gu_decode<F, false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    }
    return;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                         sc->pairs);
    const dim3 grid(ceil_div(N_work, kMoeWarps), W.n_experts);
    const float* xc = x + size_t(r0) * ldx;
    float* oc = out + size_t(r0) * top_k * ld_out;
    if (W.blk.q8) {
      if (fuse_act)
        k_moe_blocks_grouped<F, true, false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, xc, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta,
            situ_linear_beta, oc, ld_out);
      else
        k_moe_blocks_grouped<F, false, false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, xc, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta,
            situ_linear_beta, oc, ld_out);
    } else {
      if (fuse_act)
        k_moe_blocks_grouped<F, true, false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, xc, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta,
            situ_linear_beta, oc, ld_out);
      else
        k_moe_blocks_grouped<F, false, false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            T, xc, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta,
            situ_linear_beta, oc, ld_out);
    }
  }
}

template <class F>
void launch_blocks_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows,
                        int top_k, int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc,
                        cudaStream_t st) {
  const Tables T = block_tables();
  const bool grouped =
      rows > 16 && sc && sc->counts && sc->down_buf && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
  if (!grouped) {
    const dim3 grid(ceil_div(H, kMoeWarps), rows);
    if (W.blk.q8)
      k_moe_blocks_down_decode<F, true><<<grid, 32 * kMoeWarps, 0, st>>>(
          T, act, ld_act, W.blk.data, W.blk.scales, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
    else
      k_moe_blocks_down_decode<F, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          T, act, ld_act, W.blk.data, W.blk.scales, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
    return;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                         sc->pairs);
    const dim3 grid(ceil_div(H, kMoeWarps), W.n_experts);
    const float* ac = act + size_t(r0) * top_k * ld_act;
    if (W.blk.q8)
      k_moe_blocks_grouped<F, false, true, true><<<grid, 32 * kMoeWarps, 0, st>>>(
          T, ac, ld_act, W.blk.data, W.blk.scales, W.rs, H, H, K, sc->offsets, sc->pairs, top_k, 0.f, 0.f, sc->down_buf,
          H);
    else
      k_moe_blocks_grouped<F, false, true, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          T, ac, ld_act, W.blk.data, W.blk.scales, W.rs, H, H, K, sc->offsets, sc->pairs, top_k, 0.f, 0.f, sc->down_buf,
          H);
    k_moe_weighted_sum<<<dim3(ceil_div(H, 256), rc), 256, 0, st>>>(sc->down_buf, ew + size_t(r0) * top_k, top_k, H,
                                                                   out + size_t(r0) * ld_out, ld_out, int(accumulate));
  }
}

}  // namespace

void moe_route(const float* logits, const float* bias, int rows, int n_experts, int top_k, bool gating_sigmoid,
               bool norm_weights, float norm_eps, float route_scale, int* idx_out, float* w_out, cudaStream_t st) {
  if (rows <= 0 || n_experts <= 0 || top_k <= 0 || top_k > n_experts)
    throw std::runtime_error("moe_route: invalid dimensions");
  const size_t smem = size_t(2) * size_t(n_experts) * sizeof(float);
  k_moe_route<<<rows, kRouteThreads, smem, st>>>(logits, bias, n_experts, top_k, int(gating_sigmoid), int(norm_weights),
                                                 norm_eps, route_scale, idx_out, w_out);
}

void situ_rows(const float* gu, int ld_gu, int I, float beta, float linear_beta, float* out, int ld_out, int rows,
               cudaStream_t st) {
  if (rows <= 0 || I <= 0) return;
  k_situ_rows<<<dim3(ceil_div(I, 256), rows), 256, 0, st>>>(gu, ld_gu, I, beta, linear_beta, out, ld_out);
}

void moe_gate_up(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                 float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out, const MoeScratch* sc,
                 const Launch& L) {
  cudaStream_t st = L.st;
  const int N_work = fuse_act ? I : 2 * I;
  const bool grouped = rows > 16 && sc && sc->counts && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
  if (W.packed()) {
    blocks_init();
    switch (W.blk.dtype) {
#define X(F)                                                                                                      \
  case F::kDtype:                                                                                                 \
    launch_blocks_gu<F>(x, ldx, W, eidx, rows, top_k, I, K, situ_beta, situ_linear_beta, fuse_act, out, ld_out,   \
                        sc, st);                                                                                  \
    return;
      FLYX_BLOCK_TYPES(X)
#undef X
      default: throw std::runtime_error("moe_gate_up: unsupported GGML block dtype " + std::to_string(W.blk.dtype));
    }
  }
  if (W.q) {
    if (!grouped) {
      const dim3 grid(ceil_div(N_work, kMoeWarps), top_k, rows);
      if (fuse_act)
        k_moe_nvfp4_gu_decode<true><<<grid, 32 * kMoeWarps, 0, st>>>(
            x, ldx, W.q, W.s, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
      else
        k_moe_nvfp4_gu_decode<false><<<grid, 32 * kMoeWarps, 0, st>>>(
            x, ldx, W.q, W.s, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
      return;
    }
    const int chunk = std::max(1, sc->chunk_rows);
    for (int r0 = 0; r0 < rows; r0 += chunk) {
      const int rc = std::min(chunk, rows - r0);
      k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                           sc->pairs);
      const dim3 grid(ceil_div(N_work, kMoeWarps), W.n_experts);
      const float* xc = x + size_t(r0) * ldx;
      float* oc = out + size_t(r0) * top_k * ld_out;
      if (fuse_act)
        k_moe_nvfp4_grouped<true, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            xc, ldx, W.q, W.s, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta, situ_linear_beta, oc, ld_out);
      else
        k_moe_nvfp4_grouped<false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
            xc, ldx, W.q, W.s, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta, situ_linear_beta, oc, ld_out);
    }
    return;
  }
  if (!grouped) {
    const dim3 grid(ceil_div(N_work, kMoeWarps), top_k, rows);
    if (fuse_act)
      k_moe_bf16_gu_decode<true><<<grid, 32 * kMoeWarps, 0, st>>>(
          x, ldx, W.w, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    else
      k_moe_bf16_gu_decode<false><<<grid, 32 * kMoeWarps, 0, st>>>(
          x, ldx, W.w, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    return;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                         sc->pairs);
    const dim3 grid(ceil_div(N_work, kMoeWarps), W.n_experts);
    const float* xc = x + size_t(r0) * ldx;
    float* oc = out + size_t(r0) * top_k * ld_out;
    if (fuse_act)
      k_moe_bf16_grouped<true, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          xc, ldx, W.w, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta, situ_linear_beta, oc, ld_out);
    else
      k_moe_bf16_grouped<false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          xc, ldx, W.w, W.rs, W.N, I, K, sc->offsets, sc->pairs, top_k, situ_beta, situ_linear_beta, oc, ld_out);
  }
}

void moe_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows, int top_k,
              int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc, const Launch& L) {
  cudaStream_t st = L.st;
  const bool grouped =
      rows > 16 && sc && sc->counts && sc->down_buf && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
  if (W.packed()) {
    blocks_init();
    switch (W.blk.dtype) {
#define X(F)                                                                                                      \
  case F::kDtype:                                                                                                 \
    launch_blocks_down<F>(act, ld_act, W, eidx, ew, rows, top_k, H, K, out, ld_out, accumulate, sc, st);          \
    return;
      FLYX_BLOCK_TYPES(X)
#undef X
      default: throw std::runtime_error("moe_down: unsupported GGML block dtype " + std::to_string(W.blk.dtype));
    }
  }
  if (W.q) {
    if (!grouped) {
      k_moe_nvfp4_down_decode<<<dim3(ceil_div(H, kMoeWarps), rows), 32 * kMoeWarps, 0, st>>>(
          act, ld_act, W.q, W.s, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
      return;
    }
    const int chunk = std::max(1, sc->chunk_rows);
    for (int r0 = 0; r0 < rows; r0 += chunk) {
      const int rc = std::min(chunk, rows - r0);
      k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                           sc->pairs);
      const dim3 grid(ceil_div(H, kMoeWarps), W.n_experts);
      k_moe_nvfp4_grouped<false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
          act + size_t(r0) * top_k * ld_act, ld_act, W.q, W.s, W.rs, H, H, K, sc->offsets, sc->pairs, top_k, 0.f, 0.f,
          sc->down_buf, H);
      k_moe_weighted_sum<<<dim3(ceil_div(H, 256), rc), 256, 0, st>>>(sc->down_buf, ew + size_t(r0) * top_k, top_k, H,
                                                                     out + size_t(r0) * ld_out, ld_out, int(accumulate));
    }
    return;
  }
  if (!grouped) {
    k_moe_bf16_down_decode<<<dim3(ceil_div(H, kMoeWarps), rows), 32 * kMoeWarps, 0, st>>>(
        act, ld_act, W.w, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
    return;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    k_moe_group_pairs<<<1, 256, 0, st>>>(eidx + size_t(r0) * top_k, rc * top_k, W.n_experts, sc->counts, sc->offsets,
                                         sc->pairs);
    const dim3 grid(ceil_div(H, kMoeWarps), W.n_experts);
    k_moe_bf16_grouped<false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
        act + size_t(r0) * top_k * ld_act, ld_act, W.w, W.rs, H, H, K, sc->offsets, sc->pairs, top_k, 0.f, 0.f,
        sc->down_buf, H);
    k_moe_weighted_sum<<<dim3(ceil_div(H, 256), rc), 256, 0, st>>>(sc->down_buf, ew + size_t(r0) * top_k, top_k, H,
                                                                   out + size_t(r0) * ld_out, ld_out, int(accumulate));
  }
}

}  // namespace flyx::cuda
