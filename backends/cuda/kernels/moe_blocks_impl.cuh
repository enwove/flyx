// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Templated GGML block sparse MoE kernels, included by four per-group translation units (moe_blocks_<group>.cu).
//
// Strict Gate 4 bit-identity (0.00e+00 across R=1, R=2..16, R=256, fused vs unfused, sc vs nullptr):
//   Every q8 path splits a column's K/32 groups into 32 fixed streams (stream s = groups s, s+32, s+64, ...),
//   accumulating each group in ascending K order with exact integer products and identical float32 fmaf order,
//   then combines the 32 streams via the fixed balanced binary tree (xor 1, 2, 4, 8, 16):
//     * k_moe_blocks_q8_dp4a: 8 warps/block, C=1,2,4 columns/warp, U=1,2 prefetched units, pre-staged q8.
//     * k_moe_blocks_q8_mma:  INT8 tensor cores (mma.m16n8k32), 16 rows x 8 cols per 4-warp tile, 2-buffer
//       cp.async span pipeline, compacted per-expert 16-row work tiles in sufficiently dense prefill.
//     * k_moe_blocks_q8_nostage: fallback when sc==nullptr (unit tests), same 32-stream dp4a + xor tree.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_types.cuh"
#include "backends/cuda/kernels/gemm_internal.h"
#include "backends/cuda/kernels/moe_blocks_ops.h"

namespace flyx::cuda {
namespace {

constexpr int kMoeBlkKT = 1024;
constexpr int kMoeDp4aWarps = 8;
constexpr int kMoeMmaTileWarps = 4;
constexpr int kMoeMmaCT = 2;
constexpr int kMoeMmaSpanRow = 32 * 32 + 32;  // 1056 bytes: 8-row bank-conflict-free uint2 loads
constexpr int kMoeMmaRowBufs = 2;

__device__ __forceinline__ float moe_situ(float g, float u, float beta, float linear_beta) {
  if (beta > 0.f) {
    const float act = beta * tanhf(g / beta) * sigm(g);
    const float lin = linear_beta > 0.f ? (linear_beta * tanhf(u / linear_beta)) : u;
    return act * lin;
  }
  return silu(g) * u;
}

__device__ __forceinline__ float moe_a16h_scale(float m) {
  if (!(m > 0.f) || !isfinite(m)) return 1.f;
  const uint32_t u = __float_as_uint(m);
  const int e = int((u >> 23) & 0xFFu) - 127;
  const int se = max(e, -111) - 15;
  return __uint_as_float(uint32_t(se + 127) << 23);
}

// Fixed balanced binary tree over 32 streams: (((0+1)+(2+3))+((4+5)+(6+7))) + ... (matches k_moe_blocks_q8_mma).
__device__ __forceinline__ float warp_tree_sum(float v) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ---- Fast pre-staged Q8 dp4a MoE kernel (R=1 direct or R>=2 compacted tasks, BT <= 4) ----------
template <class F, bool FUSE_ACT, int C, int U, int BT>
__device__ __forceinline__ void moe_dp4a_tile(
    const Tables& T, const typename F::Local& local, const int8_t* __restrict__ q8_a, const float* __restrict__ q8_d,
    const float* __restrict__ q8_s, const uint8_t* __restrict__ W, const uint16_t* __restrict__ D,
    const float* __restrict__ RS, int N_exp, int I_or_H, int K, int e, int nb, const int* s_pids, const int* s_rids,
    float situ_beta, float situ_linear_beta, float* __restrict__ out, int ld_out, bf16* __restrict__ act_bf16) {
  static constexpr int GROUPS = F::kValues / 32, SUB = F::kSub;
  static constexpr bool MINS = F::kFlatRun != 0;
  static constexpr int NC = FUSE_ACT ? 2 * C : C;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int col0 = (int(blockIdx.x) * kMoeDp4aWarps + warp) * C;
  if (col0 >= N_work) return;

  const float rs = RS ? RS[e] : 1.f;
  const int units = K >> 5;
  const size_t blocks = size_t(K / F::kValues);
  const int G = lane % GROUPS;

  const uint8_t* wn[NC];
  const uint16_t* dn[NC];
#pragma unroll
  for (int c = 0; c < C; ++c) {
    const int n0 = min(col0 + c, N_exp - 1);
    const size_t r0 = size_t(e) * N_exp + n0;
    wn[c] = W + r0 * blocks * F::kStride;
    dn[c] = D + r0 * blocks * F::kScales;
    if constexpr (FUSE_ACT) {
      const int n1 = min(col0 + c + I_or_H, N_exp - 1);
      const size_t r1 = size_t(e) * N_exp + n1;
      wn[C + c] = W + r1 * blocks * F::kStride;
      dn[C + c] = D + r1 * blocks * F::kScales;
    }
  }

  int rids[BT];
#pragma unroll
  for (int b = 0; b < BT; ++b) rids[b] = b < nb ? s_rids[b] : 0;

  typename F::Unit raw[U][NC];
  uint32_t d0[U][NC], d1[U][NC];
  auto load = [&](int slot, int u) {
    const size_t blk = size_t(u / GROUPS);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      raw[slot][c] = block_unit<F>(wn[c], dn[c], blk, G);
      d0[slot][c] = dn[c][blk * F::kScales];
      d1[slot][c] = F::kScales > 1 ? dn[c][blk * F::kScales + 1] : 0u;
    }
  };
#pragma unroll
  for (int i = 0; i < U; ++i)
    if (lane + 32 * i < units) load(i, lane + 32 * i);

  struct Activation {
    uint4 lo, hi;
    float dx;
    float2 ds;
  };
  auto activation = [&](int rid, int u) {
    const size_t group = size_t(rid) * units + u;
    const int8_t* ar = q8_a + size_t(rid) * K + size_t(u) * 32;
    Activation x{*reinterpret_cast<const uint4*>(ar), *reinterpret_cast<const uint4*>(ar + 16), q8_d[group], {}};
    if constexpr (MINS) x.ds = *reinterpret_cast<const float2*>(q8_s + group * 2);
    return x;
  };

  float acc[BT][NC];
#pragma unroll
  for (int b = 0; b < BT; ++b)
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[b][c] = 0.f;

  for (int u0 = lane; u0 < units; u0 += 32 * U) {
#pragma unroll
    for (int i = 0; i < U; ++i) {
      const int u = u0 + 32 * i;
      if (u >= units) break;
      // A one-row tile loads the two vectors and scale once for all NC dots. For longer tiles,
      // hold one column's decoded weights across rows, bounding the live activation registers.
      Activation single{};
      if constexpr (BT == 1) single = activation(rids[0], u);
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        uint32_t w[8];
        float f[2], m[2];
        F::unit_codes(T, local, raw[i][c], G, w);
        F::unit_factors(raw[i][c], G, scale_bits<F>(d0[i][c]),
                        F::kScales > 1 ? scale_bits<F>(d1[i][c]) : 0.f, f, m);
#pragma unroll
        for (int b = 0; b < BT; ++b) {
          if (b >= nb) break;
          const Activation x = BT == 1 ? single : activation(rids[b], u);
          int s0 = 0;
          s0 = __dp4a(int(w[0]), int(x.lo.x), s0);
          s0 = __dp4a(int(w[1]), int(x.lo.y), s0);
          s0 = __dp4a(int(w[2]), int(x.lo.z), s0);
          s0 = __dp4a(int(w[3]), int(x.lo.w), s0);
          int s1 = (SUB == 32) ? s0 : 0;
          s1 = __dp4a(int(w[4]), int(x.hi.x), s1);
          s1 = __dp4a(int(w[5]), int(x.hi.y), s1);
          s1 = __dp4a(int(w[6]), int(x.hi.z), s1);
          s1 = __dp4a(int(w[7]), int(x.hi.w), s1);
          if constexpr (SUB == 32) s0 = s1;
          float& ac = acc[b][c];
          ac = fmaf(float(s0), x.dx * f[0], ac);
          if constexpr (MINS) ac = fmaf(SUB == 16 ? x.ds.x : (x.ds.x + x.ds.y), m[0], ac);
          if constexpr (SUB == 16) {
            ac = fmaf(float(s1), x.dx * f[1], ac);
            if constexpr (MINS) ac = fmaf(x.ds.y, m[1], ac);
          }
        }
      }
      if (u + 32 * U < units) load(i, u + 32 * U);
    }
  }

#pragma unroll
  for (int b = 0; b < BT; ++b) {
    if (b >= nb) break;
    const size_t out_row = size_t(s_pids[b]) * ld_out;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const float g = warp_tree_sum(acc[b][c]) * rs;
      if constexpr (FUSE_ACT) {
        const float u = warp_tree_sum(acc[b][C + c]) * rs;
        if (lane == 0 && col0 + c < N_work) {
          const float v = moe_situ(g, u, situ_beta, situ_linear_beta);
          if (act_bf16) act_bf16[out_row + col0 + c] = __float2bfloat16_rn(v);
          else out[out_row + col0 + c] = v;
        }
      } else {
        if (lane == 0 && col0 + c < N_work) out[out_row + col0 + c] = g;
      }
    }
  }
}

template <class F, bool FUSE_ACT, bool IS_DOWN, bool DIRECT_R1, int C, int U, bool EXPLICIT_CONFIG>
__global__ void __launch_bounds__(32 * kMoeDp4aWarps)
    k_moe_blocks_q8_dp4a(Tables T, const int8_t* __restrict__ q8_a, const float* __restrict__ q8_d,
                         const float* __restrict__ q8_s, const uint8_t* __restrict__ W,
                         const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                         const int* __restrict__ eidx, const MoeTask* __restrict__ tasks,
                         const int* __restrict__ n_tasks, const int* __restrict__ pairs, int top_k, float situ_beta,
                         float situ_linear_beta, float* __restrict__ out, int ld_out, bf16* __restrict__ act_bf16) {
  pdl_trigger();
  constexpr int BT = DIRECT_R1 ? 1 : 4;
  int e, nb;
  __shared__ typename F::Local local;
  __shared__ int s_pids[BT];
  __shared__ int s_rids[BT];

  if constexpr (DIRECT_R1) {
    e = eidx[blockIdx.y];
    nb = 1;
    if (threadIdx.x == 0) {
      s_pids[0] = int(blockIdx.y);
      s_rids[0] = IS_DOWN ? int(blockIdx.y) : 0;
    }
  } else {
    const int tid_y = int(blockIdx.y);
    if (tid_y >= *n_tasks) return;
    const MoeTask tk = tasks[tid_y];
    e = tk.expert;
    nb = tk.nb;
    if (int(threadIdx.x) < nb) {
      const int pid = pairs[tk.p0 + int(threadIdx.x)];
      s_pids[threadIdx.x] = pid;
      s_rids[threadIdx.x] = IS_DOWN ? pid : (pid / top_k);
    }
  }
  if constexpr (EXPLICIT_CONFIG) F::unit_tables(T, local);
  else F::tables(T, local);
  __syncthreads();

  if constexpr (EXPLICIT_CONFIG && !DIRECT_R1) {
    if (nb == 1) {
      moe_dp4a_tile<F, FUSE_ACT, C, U, 1>(T, local, q8_a, q8_d, q8_s, W, D, RS, N_exp, I_or_H, K, e, 1,
                                          s_pids, s_rids, situ_beta, situ_linear_beta, out, ld_out, act_bf16);
      return;
    }
  }
  moe_dp4a_tile<F, FUSE_ACT, C, U, BT>(T, local, q8_a, q8_d, q8_s, W, D, RS, N_exp, I_or_H, K, e, nb,
                                       s_pids, s_rids, situ_beta, situ_linear_beta, out, ld_out, act_bf16);
}

// ---- INT8 Tensor Core mma.m16n8k32 grouped MoE kernel (BM = 16 rows/task, CT = 2 -> 16 cols/block) ----
template <class F, int CT>
struct MoeMmaSmem {
  static constexpr bool MINS = F::kFlatRun != 0;
  alignas(16) int8_t rows[kMoeMmaRowBufs][16][kMoeMmaSpanRow];
  float d[kMoeMmaRowBufs][16][33];
  alignas(8) float sums[MINS ? kMoeMmaRowBufs : 1][MINS ? 16 : 1][66];
  float tree[CT][kMoeMmaTileWarps][32][4];
  int pids[16];
  int rids[16];
  typename F::Local local;
};

template <class F, bool FUSE_ACT, bool IS_DOWN, int CT>
__global__ void __launch_bounds__(32 * kMoeMmaTileWarps * CT)
    k_moe_blocks_q8_mma(Tables T, const int8_t* __restrict__ q8_a, const float* __restrict__ q8_d,
                        const float* __restrict__ q8_s, const uint8_t* __restrict__ W,
                        const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                        const MoeTask* __restrict__ tasks, const int* __restrict__ n_tasks,
                        const int* __restrict__ pairs, int top_k, float situ_beta, float situ_linear_beta,
                        float* __restrict__ out, int ld_out) {
  pdl_trigger();
  using S = MoeMmaSmem<F, CT>;
  static constexpr int GROUPS = F::kValues / 32, BPW = 8 / GROUPS, SUB = F::kSub;
  static constexpr int THREADS = 32 * kMoeMmaTileWarps * CT;
  static constexpr bool MINS = S::MINS;

  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S*>(smem_raw);
  const int tid = threadIdx.x;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
  }
  F::tables(T, sm.local);
  __syncthreads();

  const int lane = tid & 31, warp = tid >> 5, wi = warp % kMoeMmaTileWarps, ct = warp / kMoeMmaTileWarps;
  const int g = lane >> 2, t4 = lane & 3;
  const int units = K >> 5, spans = (units + 31) >> 5, blocks = K / F::kValues;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int n0 = (int(blockIdx.x) * CT + ct) * 8;
  const float rs = RS ? RS[e] : 1.f;

  const int col0 = min(n0 + g, N_exp - 1);
  const size_t r0_idx = size_t(e) * N_exp + col0;
  const uint8_t* wn0 = W + r0_idx * blocks * F::kStride;
  const uint16_t* dn0 = D + r0_idx * blocks * F::kScales;

  auto stage = [&](int j, int buf) {
    const int n_s = min(32, units - 32 * j);
    const int chunks = 2 * n_s;
    for (int i = tid; i < 16 * chunks; i += THREADS) {
      const int r = i / chunks, c = i - r * chunks;
      const bool ok = r < nb;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * 1024 + 16 * c) : 0;
      cp_async16(&sm.rows[buf][r][16 * c], q8_a + src, ok);
    }
    for (int i = tid; i < 16 * n_s; i += THREADS) {
      const int r = i / n_s, c = i - r * n_s;
      const bool ok = r < nb;
      const size_t src = ok ? (size_t(sm.rids[r]) * units + 32 * j + c) : 0;
      cp_async4(&sm.d[buf][r][c], q8_d + src, ok);
    }
    if constexpr (MINS) {
      for (int i = tid; i < 16 * chunks; i += THREADS) {
        const int r = i / chunks, c = i - r * chunks;
        const bool ok = r < nb;
        const size_t src = ok ? ((size_t(sm.rids[r]) * units + 32 * j) * 2 + c) : 0;
        cp_async4(&sm.sums[buf][r][c], q8_s + src, ok);
      }
    }
    cp_async_commit();
  };

  // Single 32-float accumulator reused sequentially for gate then up when FUSE_ACT=true:
  // cuts register usage from ~172 to ~100, doubling occupancy on sm_120 while keeping exact 32-stream tree bits.
  float acc[8][4];
  auto zero_acc = [&]() {
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int e4 = 0; e4 < 4; ++e4) acc[s][e4] = 0.f;
  };
  zero_acc();

  auto compute_half = [&](int j, int buf, const uint8_t* wn, const uint16_t* dn) {
    const int base_blk = (32 * j + 8 * wi) / GROUPS;
    if (base_blk >= blocks) return;
    typename F::Raw cur = block_load<F>(wn, dn, size_t(base_blk), t4);
    float s0_cur = scale_of<F>(dn, size_t(base_blk) * F::kScales);
    float s1_cur = F::kScales > 1 ? scale_of<F>(dn, size_t(base_blk) * F::kScales + 1) : 0.f;
#pragma unroll
    for (int bi = 0; bi < BPW; ++bi) {
      const int b_idx = base_blk + bi;
      if (b_idx >= blocks) break;
      const typename F::Lane ln = F::share(cur, t4, lane);
      const float sc0 = s0_cur, sc1 = s1_cur;
      if constexpr (BPW > 1) {
        if (bi + 1 < BPW && b_idx + 1 < blocks) {
          cur = block_load<F>(wn, dn, size_t(b_idx + 1), t4);
          s0_cur = scale_of<F>(dn, size_t(b_idx + 1) * F::kScales);
          s1_cur = F::kScales > 1 ? scale_of<F>(dn, size_t(b_idx + 1) * F::kScales + 1) : 0.f;
        }
      }
#pragma unroll
      for (int G = 0; G < GROUPS; ++G) {
        const int s = bi * GROUPS + G, ul = 8 * wi + s;
        uint32_t w[2];
        float f, m;
        F::codes(T, sm.local, ln, G, w);
        F::factors(ln, G, sc0, sc1, &f, &m);
        float fc[2][2], mc[2][2];
#pragma unroll
        for (int h = 0; h < 32 / SUB; ++h) {
          const int o = SUB == 16 ? 2 * h : 0;
          fc[0][h] = __shfl_sync(0xffffffffu, f, 8 * t4 + o);
          fc[1][h] = __shfl_sync(0xffffffffu, f, 8 * t4 + 4 + o);
          if constexpr (MINS) {
            mc[0][h] = __shfl_sync(0xffffffffu, m, 8 * t4 + o);
            mc[1][h] = __shfl_sync(0xffffffffu, m, 8 * t4 + 4 + o);
          }
        }
        const uint2 x0 = *reinterpret_cast<const uint2*>(&sm.rows[buf][g][32 * ul + 8 * t4]);
        const uint2 x8 = *reinterpret_cast<const uint2*>(&sm.rows[buf][g + 8][32 * ul + 8 * t4]);
        const float dr[2] = {sm.d[buf][g][ul], sm.d[buf][g + 8][ul]};
        float2 sr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
        if constexpr (MINS) {
          sr[0] = *reinterpret_cast<const float2*>(&sm.sums[buf][g][2 * ul]);
          sr[1] = *reinterpret_cast<const float2*>(&sm.sums[buf][g + 8][2 * ul]);
        }
        const uint32_t a[4] = {x0.x, x8.x, x0.y, x8.y};
        int c0[4], c1[4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
        constexpr bool from_zero = SUB == 32;
#else
        constexpr bool from_zero = false;
#endif
        if constexpr (from_zero) {
          mma_s8_k32_zero(c0, a, w);
        } else {
          c0[0] = c0[1] = c0[2] = c0[3] = kMagicBits;
          if constexpr (SUB == 32) {
            mma_s8_k32(c0, a, w);
          } else {
            c1[0] = c1[1] = c1[2] = c1[3] = kMagicBits;
            const uint32_t lo[2] = {t4 < 2 ? w[0] : 0u, t4 < 2 ? w[1] : 0u};
            const uint32_t hi[2] = {t4 < 2 ? 0u : w[0], t4 < 2 ? 0u : w[1]};
            mma_s8_k32(c0, a, lo);
            mma_s8_k32(c1, a, hi);
          }
        }
#pragma unroll
        for (int e4 = 0; e4 < 4; ++e4) {
          const int h = e4 >> 1, c = e4 & 1;
          float& v = acc[s][e4];
          const float s0f = from_zero ? __int2float_rn(c0[e4]) : (__int_as_float(c0[e4]) - kMagic);
          v = fmaf(s0f, dr[h] * fc[c][0], v);
          if constexpr (MINS) v = fmaf(SUB == 16 ? sr[h].x : (sr[h].x + sr[h].y), mc[c][0], v);
          if constexpr (SUB == 16) {
            v = fmaf(__int_as_float(c1[e4]) - kMagic, dr[h] * fc[c][1], v);
            if constexpr (MINS) v = fmaf(sr[h].y, mc[c][1], v);
          }
        }
      }
    }
  };

  stage(0, 0);
  for (int j = 0; j < spans; ++j) {
    cp_async_wait<0>();
    __syncthreads();
    if (j + 1 < spans) stage(j + 1, (j + 1) & 1);
    if (n0 < N_work) compute_half(j, j & 1, wn0, dn0);
  }

  // Exact same 32-stream balanced binary tree as k_moe_blocks_q8_dp4a.
#pragma unroll
  for (int e4 = 0; e4 < 4; ++e4)
    sm.tree[ct][wi][lane][e4] =
        ((acc[0][e4] + acc[1][e4]) + (acc[2][e4] + acc[3][e4])) +
        ((acc[4][e4] + acc[5][e4]) + (acc[6][e4] + acc[7][e4]));
  __syncthreads();
  float vg[4];
  if (wi == 0) {
#pragma unroll
    for (int e4 = 0; e4 < 4; ++e4)
      vg[e4] = ((sm.tree[ct][0][lane][e4] + sm.tree[ct][1][lane][e4]) +
                (sm.tree[ct][2][lane][e4] + sm.tree[ct][3][lane][e4])) *
               rs;
  }
  if constexpr (FUSE_ACT) {
    __syncthreads();
    zero_acc();
    const size_t r1_idx = size_t(e) * N_exp + min(n0 + g + I_or_H, N_exp - 1);
    const uint8_t* wn1 = W + r1_idx * blocks * F::kStride;
    const uint16_t* dn1 = D + r1_idx * blocks * F::kScales;
    if (spans == 1) {
      if (n0 < N_work) compute_half(0, 0, wn1, dn1);
    } else {
      stage(0, 0);
      for (int j = 0; j < spans; ++j) {
        cp_async_wait<0>();
        __syncthreads();
        if (j + 1 < spans) stage(j + 1, (j + 1) & 1);
        if (n0 < N_work) compute_half(j, j & 1, wn1, dn1);
      }
    }
#pragma unroll
    for (int e4 = 0; e4 < 4; ++e4)
      sm.tree[ct][wi][lane][e4] =
          ((acc[0][e4] + acc[1][e4]) + (acc[2][e4] + acc[3][e4])) +
          ((acc[4][e4] + acc[5][e4]) + (acc[6][e4] + acc[7][e4]));
    __syncthreads();
    if (wi == 0) {
#pragma unroll
      for (int e4 = 0; e4 < 4; ++e4) {
        const int r = g + 8 * (e4 >> 1), col = n0 + 2 * t4 + (e4 & 1);
        if (r < nb && col < N_work) {
          const float vu = ((sm.tree[ct][0][lane][e4] + sm.tree[ct][1][lane][e4]) +
                            (sm.tree[ct][2][lane][e4] + sm.tree[ct][3][lane][e4])) *
                           rs;
          out[size_t(sm.pids[r]) * ld_out + col] = moe_situ(vg[e4], vu, situ_beta, situ_linear_beta);
        }
      }
    }
  } else if (wi == 0) {
#pragma unroll
    for (int e4 = 0; e4 < 4; ++e4) {
      const int r = g + 8 * (e4 >> 1), col = n0 + 2 * t4 + (e4 & 1);
      if (r < nb && col < N_work) out[size_t(sm.pids[r]) * ld_out + col] = vg[e4];
    }
  }
}

// ---- No-scratch fallback (sc == nullptr) & exact a16h kernels (exact same stream/tree bits) ----
template <int BT>
__device__ __forceinline__ void stage_q8_smem(const float* __restrict__ x, int ldx, const int* row_ids, int nb, int k0,
                                              int kt, int8_t sm_q[BT][kMoeBlkKT], float sm_d[BT][kMoeBlkKT / 32],
                                              float sm_s[BT][kMoeBlkKT / 16]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  const int ng = kt >> 5, total_g = nb * ng;
  for (int gi = warp; gi < total_g; gi += nw) {
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
__device__ __forceinline__ void stage_a16h_smem(const float* __restrict__ x, int ldx, const int* row_ids,
                                                const float* row_sc, int nb, int k0, int kt,
                                                float sm[BT][kMoeBlkKT]) {
  for (int idx = threadIdx.x; idx < nb * kt; idx += blockDim.x) {
    const int b = idx / kt, j = idx - b * kt;
    const float bf = __bfloat162float(__float2bfloat16(x[size_t(row_ids[b]) * ldx + k0 + j]));
    const float sc = row_sc[b];
    sm[b][j] = __half2float(__float2half_rn(bf / sc)) * sc;
  }
}

template <class F, int BT>
__device__ __forceinline__ void step_q8_smem(const Tables& T, const typename F::Local& local,
                                             const uint8_t* __restrict__ w_row, const uint16_t* __restrict__ d_row,
                                             int k0, int kt, int lane, int nb, const int8_t sm_q[BT][kMoeBlkKT],
                                             const float sm_d[BT][kMoeBlkKT / 32],
                                             const float sm_s[BT][kMoeBlkKT / 16], float acc[BT]) {
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
__device__ __forceinline__ void step_a16h_smem(const Tables& T, const typename F::Local& local,
                                               const uint8_t* __restrict__ w_row, const uint16_t* __restrict__ d_row,
                                               int k0, int kt, int lane, int nb, const float sm[BT][kMoeBlkKT],
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
__global__ void __launch_bounds__(128)
    k_moe_blocks_gu_nostage(Tables T, const float* __restrict__ x, int ldx, const uint8_t* __restrict__ W,
                            const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I, int K,
                            const int* __restrict__ eidx, int top_k, float situ_beta, float situ_linear_beta,
                            float* __restrict__ out, int ld_out) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __shared__ int8_t sm_q[1][Q8 ? kMoeBlkKT : 1];
  __shared__ float sm_d[1][Q8 ? kMoeBlkKT / 32 : 1];
  __shared__ float sm_s[1][Q8 ? kMoeBlkKT / 16 : 1];
  __shared__ float sm_f[1][Q8 ? 1 : kMoeBlkKT];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = int(blockIdx.x) * 4 + warp;
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
    row_sc[0] = moe_a16h_scale(block_max(rm));
  }
  float ag[1] = {0.f}, au[1] = {0.f};
  const int rids[1] = {r};
  for (int k0 = 0; k0 < K; k0 += kMoeBlkKT) {
    const int kt = min(kMoeBlkKT, K - k0);
    if constexpr (Q8) stage_q8_smem<1>(x, ldx, rids, 1, k0, kt, sm_q, sm_d, sm_s);
    else stage_a16h_smem<1>(x, ldx, rids, row_sc, 1, k0, kt, sm_f);
    __syncthreads();
    if (col < N_work) {
      if constexpr (Q8) {
        step_q8_smem<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_q, sm_d, sm_s, ag);
        if constexpr (FUSE_ACT) step_q8_smem<F, 1>(T, local, w1, d1, k0, kt, lane, 1, sm_q, sm_d, sm_s, au);
      } else {
        step_a16h_smem<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_f, ag);
        if constexpr (FUSE_ACT) step_a16h_smem<F, 1>(T, local, w1, d1, k0, kt, lane, 1, sm_f, au);
      }
    }
    __syncthreads();
  }
  if (col < N_work) {
    const float g = warp_tree_sum(ag[0]) * rs;
    if constexpr (FUSE_ACT) {
      const float u = warp_tree_sum(au[0]) * rs;
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = moe_situ(g, u, situ_beta, situ_linear_beta);
    } else if (lane == 0) {
      out[size_t(r * top_k + k_slot) * ld_out + col] = g;
    }
  }
}

template <class F, bool Q8>
__global__ void __launch_bounds__(128)
    k_moe_blocks_down_nostage(Tables T, const float* __restrict__ act, int ld_act, const uint8_t* __restrict__ W,
                              const uint16_t* __restrict__ D, const float* __restrict__ RS, int H, int K,
                              const int* __restrict__ eidx, const float* __restrict__ ew, int top_k,
                              float* __restrict__ out, int ld_out, int accumulate) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __shared__ int8_t sm_q[1][Q8 ? kMoeBlkKT : 1];
  __shared__ float sm_d[1][Q8 ? kMoeBlkKT / 32 : 1];
  __shared__ float sm_s[1][Q8 ? kMoeBlkKT / 16 : 1];
  __shared__ float sm_f[1][Q8 ? 1 : kMoeBlkKT];
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = int(blockIdx.x) * 4 + warp;
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
      row_sc[0] = moe_a16h_scale(block_max(rm));
    }
    float acc[1] = {0.f};
    for (int k0 = 0; k0 < K; k0 += kMoeBlkKT) {
      const int kt = min(kMoeBlkKT, K - k0);
      if constexpr (Q8) stage_q8_smem<1>(act, ld_act, rids, 1, k0, kt, sm_q, sm_d, sm_s);
      else stage_a16h_smem<1>(act, ld_act, rids, row_sc, 1, k0, kt, sm_f);
      __syncthreads();
      if (col < H) {
        if constexpr (Q8) step_q8_smem<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_q, sm_d, sm_s, acc);
        else step_a16h_smem<F, 1>(T, local, w0, d0, k0, kt, lane, 1, sm_f, acc);
      }
      __syncthreads();
    }
    if (col < H) {
      const float v = warp_tree_sum(acc[0]) * rs;
      if (lane == 0) total = __fmaf_rn(ew[r * top_k + k_slot], v, total);
    }
  }
  if (col < H && lane == 0) {
    float* dst = out + size_t(r) * ld_out + col;
    *dst = accumulate ? (*dst + total) : total;
  }
}

template <class F, bool FUSE_ACT, bool IS_DOWN>
__global__ void __launch_bounds__(128)
    k_moe_blocks_a16h_grouped(Tables T, const float* __restrict__ x, int ldx, const uint8_t* __restrict__ W,
                              const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H,
                              int K, const MoeTask* __restrict__ tasks, const int* __restrict__ n_tasks,
                              const int* __restrict__ pairs, int top_k, float situ_beta, float situ_linear_beta,
                              float* __restrict__ out, int ld_out) {
  constexpr int BT = 4;
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb_tot = tk.nb;

  __shared__ typename F::Local local;
  __shared__ float sm_f[BT][kMoeBlkKT];
  F::tables(T, local);
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = int(blockIdx.x) * 4 + warp;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const float rs = RS ? RS[e] : 1.f;
  const size_t bpr = size_t(K / F::kValues);
  const size_t r0_idx = size_t(e) * N_exp + min(col, N_exp - 1);
  const size_t r1_idx = FUSE_ACT ? (size_t(e) * N_exp + min(col + I_or_H, N_exp - 1)) : 0;
  const uint8_t* w0 = W + r0_idx * bpr * F::kStride;
  const uint16_t* d0 = D + r0_idx * bpr * F::kScales;
  const uint8_t* w1 = FUSE_ACT ? (W + r1_idx * bpr * F::kStride) : nullptr;
  const uint16_t* d1 = FUSE_ACT ? (D + r1_idx * bpr * F::kScales) : nullptr;

  for (int off = 0; off < nb_tot; off += BT) {
    const int nb = min(BT, nb_tot - off);
    int rids[BT], pids[BT];
#pragma unroll
    for (int b = 0; b < BT; ++b) {
      pids[b] = b < nb ? pairs[tk.p0 + off + b] : 0;
      rids[b] = IS_DOWN ? pids[b] : (pids[b] / top_k);
    }
    float row_sc[BT] = {};
    for (int b = 0; b < nb; ++b) {
      float rm = 0.f;
      for (int k = threadIdx.x; k < K; k += blockDim.x)
        rm = fmaxf(rm, fabsf(__bfloat162float(__float2bfloat16(x[size_t(rids[b]) * ldx + k]))));
      row_sc[b] = moe_a16h_scale(block_max(rm));
    }
    float ag[BT] = {}, au[BT] = {};
    for (int k0 = 0; k0 < K; k0 += kMoeBlkKT) {
      const int kt = min(kMoeBlkKT, K - k0);
      stage_a16h_smem<BT>(x, ldx, rids, row_sc, nb, k0, kt, sm_f);
      __syncthreads();
      if (col < N_work) {
        step_a16h_smem<F, BT>(T, local, w0, d0, k0, kt, lane, nb, sm_f, ag);
        if constexpr (FUSE_ACT) step_a16h_smem<F, BT>(T, local, w1, d1, k0, kt, lane, nb, sm_f, au);
      }
      __syncthreads();
    }
    if (col < N_work) {
#pragma unroll
      for (int b = 0; b < BT; ++b) {
        if (b >= nb) break;
        const float g = warp_tree_sum(ag[b]) * rs;
        if constexpr (FUSE_ACT) {
          const float u = warp_tree_sum(au[b]) * rs;
          if (lane == 0) out[size_t(pids[b]) * ld_out + col] = moe_situ(g, u, situ_beta, situ_linear_beta);
        } else if (lane == 0) {
          out[size_t(pids[b]) * ld_out + col] = g;
        }
      }
    }
  }
}

template <class F, bool FUSE_ACT, bool IS_DOWN>
void launch_mma_q8(const ExpertMat& W, int I_or_H, int K, int top_k, float situ_beta, float situ_linear_beta,
                   float* out, int ld_out, const MoeScratch* sc, int max_tiles, cudaStream_t st) {
  constexpr size_t smem = sizeof(MoeMmaSmem<F, kMoeMmaCT>);
  static const bool ready = set_dynamic_smem(k_moe_blocks_q8_mma<F, FUSE_ACT, IS_DOWN, kMoeMmaCT>, smem);
  (void)ready;
  const int N_work = FUSE_ACT ? I_or_H : W.N;
  const dim3 grid(ceil_div(N_work, 8 * kMoeMmaCT), max_tiles);
  k_moe_blocks_q8_mma<F, FUSE_ACT, IS_DOWN, kMoeMmaCT><<<grid, 32 * kMoeMmaTileWarps * kMoeMmaCT, smem, st>>>(
      block_tables(), sc->q8_a, sc->q8_d, sc->q8_s, W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc->tasks,
      sc->n_tasks, sc->pairs, top_k, situ_beta, situ_linear_beta, out, ld_out);
}

template <int C, int U>
struct MoeDp4aConfig {
  static constexpr int columns = C;
  static constexpr int units = U;  // zero selects the original prefetch depth and grouped loop
};

// The configuration a fused gate|up GEMM of 2 or more rows takes by default, by block type and rows (`cuda_bench moe`
// on Inkling-2-Mini's experts, RTX 5070 Ti, against the original tiles): the 2-bit IQ types two gate/up pairs a warp
// (9 to 25% faster at 2 to 16 rows); IQ3_XXS and IQ4_NL one pair up to 4 rows and two above (IQ3_XXS: 6% at 2 and
// 4 rows, 8 to 12% at 6 to 16).  Q4_K keeps the original tiles, as do one-row launches and the projections without a
// fused activation.  Every configuration gives the same bits.
template <class F>
constexpr MoeDp4a grouped_gate_up_dp4a(int rows) {
  switch (F::kDtype) {
    case FLYX_DT_GGML_IQ2_XXS:
    case FLYX_DT_GGML_IQ2_XS:
    case FLYX_DT_GGML_IQ2_S: return kMoeDp4aC2U1;
    case FLYX_DT_GGML_IQ3_XXS:
    case FLYX_DT_GGML_IQ4_NL: return rows <= 4 ? kMoeDp4aC1U1 : kMoeDp4aC2U1;
    default: return kMoeDp4aDefault;
  }
}

template <class F, bool FUSE_ACT, bool IS_DOWN>
void launch_dp4a_q8(MoeDp4a mode, const Tables& T, const ExpertMat& W, int I_or_H, int K, const int* eidx, int rows,
                    int top_k, float situ_beta, float situ_linear_beta, float* out, int ld_out, const MoeScratch* sc,
                    int max_tiles, cudaStream_t st, bf16* act_bf16) {
  const int N_exp = IS_DOWN ? I_or_H : W.N;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  auto run = [&](auto config, auto direct) {
    constexpr bool DIRECT_R1 = decltype(direct)::value;
    constexpr int C = decltype(config)::columns, requested_u = decltype(config)::units;
    constexpr int default_u = DIRECT_R1 && sizeof(typename F::Unit) <= 32 ? 2 : 1;
    constexpr int U = requested_u ? requested_u : default_u;
    const dim3 grid(ceil_div(N_work, kMoeDp4aWarps * C), DIRECT_R1 ? top_k : max_tiles);
    k_moe_blocks_q8_dp4a<F, FUSE_ACT, IS_DOWN, DIRECT_R1, C, U, (requested_u != 0)>
        <<<grid, 32 * kMoeDp4aWarps, 0, st>>>(T, sc->q8_a, sc->q8_d, sc->q8_s, W.blk.data, W.blk.scales, W.rs,
                                              N_exp, I_or_H, K, eidx, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                                              situ_beta, situ_linear_beta, out, ld_out, act_bf16);
  };
  auto launch = [&](auto config) {
    if (rows == 1) run(config, std::true_type{});
    else run(config, std::false_type{});
  };
  if constexpr (has_moe_dp4a_variants(F::kDtype)) {
    if (FUSE_ACT && mode == kMoeDp4aDefault && rows > 1) mode = grouped_gate_up_dp4a<F>(rows);
    switch (mode) {
      case kMoeDp4aC1U1: return launch(MoeDp4aConfig<1, 1>{});
      case kMoeDp4aC1U2: return launch(MoeDp4aConfig<1, 2>{});
      case kMoeDp4aC2U1: return launch(MoeDp4aConfig<2, 1>{});
      case kMoeDp4aC2U2: return launch(MoeDp4aConfig<2, 2>{});
      case kMoeDp4aC4U1: return launch(MoeDp4aConfig<4, 1>{});
      case kMoeDp4aC4U2: return launch(MoeDp4aConfig<4, 2>{});
      default: break;
    }
  }
  launch(MoeDp4aConfig<FUSE_ACT ? 1 : 2, 0>{});
}

template <class F>
void moe_blocks_gu(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                   float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out,
                   const MoeScratch* sc, int max_tiles, bool use_mma, MoeDp4a dp4a, cudaStream_t st, bf16* act_bf16) {
  const Tables T = block_tables();
  const int N_work = fuse_act ? I : 2 * I;
  const bool has_q8 = W.blk.q8 && sc && sc->q8_a && sc->stage_elems >= size_t(rows) * size_t(K);
  // Only the dp4a kernels that apply the activation write it in bf16: a round's, which moe_round_q8 promises.
  if (act_bf16 && !(fuse_act && has_q8 && !use_mma && (rows == 1 || (sc->tasks && max_tiles > 0))))
    throw std::runtime_error("moe_gate_up: bf16 activations asked of a kernel that writes float32");
  if (has_q8 && (rows == 1 || (sc->tasks && max_tiles > 0))) {
    if (rows > 1 && use_mma) {
      if (fuse_act)
        launch_mma_q8<F, true, false>(W, I, K, top_k, situ_beta, situ_linear_beta, out, ld_out, sc, max_tiles, st);
      else
        launch_mma_q8<F, false, false>(W, I, K, top_k, situ_beta, situ_linear_beta, out, ld_out, sc, max_tiles, st);
    } else if (fuse_act) {
      launch_dp4a_q8<F, true, false>(dp4a, T, W, I, K, eidx, rows, top_k, situ_beta, situ_linear_beta, out, ld_out,
                                      sc, max_tiles, st, act_bf16);
    } else {
      launch_dp4a_q8<F, false, false>(dp4a, T, W, I, K, eidx, rows, top_k, situ_beta, situ_linear_beta, out, ld_out,
                                       sc, max_tiles, st, nullptr);
    }
    return;
  }
  if (!W.blk.q8 && rows > 1 && sc && sc->tasks && max_tiles > 0) {
    const dim3 grid(ceil_div(N_work, 4), max_tiles);
    if (fuse_act)
      k_moe_blocks_a16h_grouped<F, true, false><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, situ_beta,
          situ_linear_beta, out, ld_out);
    else
      k_moe_blocks_a16h_grouped<F, false, false><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, situ_beta,
          situ_linear_beta, out, ld_out);
    return;
  }
  const dim3 grid(ceil_div(N_work, 4), top_k, rows);
  if (W.blk.q8) {
    if (fuse_act)
      k_moe_blocks_gu_nostage<F, true, true><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    else
      k_moe_blocks_gu_nostage<F, false, true><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
  } else {
    if (fuse_act)
      k_moe_blocks_gu_nostage<F, true, false><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    else
      k_moe_blocks_gu_nostage<F, false, false><<<grid, 128, 0, st>>>(
          T, x, ldx, W.blk.data, W.blk.scales, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
  }
}

template <class F>
bool moe_blocks_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows,
                     int top_k, int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc,
                     int max_tiles, bool use_mma, MoeDp4a dp4a, cudaStream_t st) {
  const Tables T = block_tables();
  const bool has_q8 =
      W.blk.q8 && sc && sc->q8_a && sc->down_buf && sc->stage_elems >= size_t(rows) * size_t(top_k) * size_t(K);
  if (has_q8 && (rows == 1 || (sc->tasks && max_tiles > 0))) {
    if (rows > 1 && use_mma) {
      launch_mma_q8<F, false, true>(W, H, K, top_k, 0.f, 0.f, sc->down_buf, H, sc, max_tiles, st);
    } else {
      launch_dp4a_q8<F, false, true>(dp4a, T, W, H, K, eidx, rows, top_k, 0.f, 0.f, sc->down_buf, H, sc, max_tiles,
                                      st, nullptr);
    }
    return true;
  }
  if (!W.blk.q8 && rows > 1 && sc && sc->tasks && sc->down_buf && max_tiles > 0) {
    k_moe_blocks_a16h_grouped<F, false, true><<<dim3(ceil_div(H, 4), max_tiles), 128, 0, st>>>(
        T, act, ld_act, W.blk.data, W.blk.scales, W.rs, H, H, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, 0.f, 0.f,
        sc->down_buf, H);
    return true;
  }
  // Fallback (sc == nullptr or !down_buf): writes directly to out with weighted sum.
  const dim3 grid(ceil_div(H, 4), rows);
  if (W.blk.q8)
    k_moe_blocks_down_nostage<F, true><<<grid, 128, 0, st>>>(
        T, act, ld_act, W.blk.data, W.blk.scales, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
  else
    k_moe_blocks_down_nostage<F, false><<<grid, 128, 0, st>>>(
        T, act, ld_act, W.blk.data, W.blk.scales, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
  return false;
}

template <class F>
const MoeBlockOps* moe_block_ops_one() {
  static const MoeBlockOps ops{F::kFlatRun != 0, &moe_blocks_gu<F>, &moe_blocks_down<F>};
  return &ops;
}

template <class... Fs>
const MoeBlockOps* moe_block_ops_of(int dtype) {
  const MoeBlockOps* found = nullptr;
  ((found = !found && dtype == Fs::kDtype ? moe_block_ops_one<Fs>() : found), ...);
  return found;
}

}  // namespace
}  // namespace flyx::cuda
