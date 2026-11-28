// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Sparse MoE kernels: top-k routing, SiTU activation, compacted work-tile grouping, one-pass activation staging,
// Tensor Core NVFP4 MoE (mma_bf16), BF16 MoE, and dispatch to modular GGML block MoE kernels (moe_blocks_*.cu) with
// the dp4a kernel's configuration the tuning tables give (variants/candidates.toml: moe_dp4a, moe_gate_up).
//
// Strict Gate 4 row-count invariance (0.00e+00 across R=1, R=2..16, R=256, fused vs unfused, sc vs nullptr):
//   * NVFP4: one fixed MMA span per K (bf16 inputs); option moe_vec=1: a fixed per-column F32 dot across all rows.
//   * GGML blocks (q8): fixed 32-stream ascending K accumulation + balanced xor tree (1,2,4,8,16) shared by
//     both k_moe_blocks_q8_dp4a and INT8 Tensor Core k_moe_blocks_q8_mma.
//   * BF16: fixed kMoeKT=1024 per-lane FMA + balanced xor tree across decode and compacted grouped tasks.
#include <algorithm>
#include <cuda/atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_internal.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_blocks_ops.h"

namespace flyx::cuda {
namespace {

constexpr int kMoeWarps = 4;
constexpr int kMoeKT = 1024;
constexpr int kMaxGridY = 65535;
constexpr int kRouteThreads = 128, kRouteTop = kMoeRouteMaxTopK;  // threads a row; the most experts a row takes

__device__ __forceinline__ float situ_val(float g, float u, float beta, float linear_beta) {
  if (beta > 0.f) {
    const float act = beta * tanhf(g / beta) * sigm(g);
    const float lin = linear_beta > 0.f ? (linear_beta * tanhf(u / linear_beta)) : u;
    return act * lin;
  }
  return silu(g) * u;
}

__device__ __forceinline__ float warp_tree_sum(float v) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

inline int max_work_tiles(int M, int E, int bm) {
  const int k_max = std::min(E, M);
  return k_max + (M - k_max) / bm;
}

const MoeBlockOps* moe_block_ops(int dtype) {
  for (const MoeBlockOps* (*grp)(int) :
       {moe_block_ops_legacy, moe_block_ops_k, moe_block_ops_iq_low, moe_block_ops_iq_high})
    if (const MoeBlockOps* ops = grp(dtype)) return ops;
  return nullptr;
}

// ---- moe_route & situ_rows ----------------------------------------------------------------
__global__ void __launch_bounds__(kRouteThreads) k_moe_route(const float* logits,
                                                             const float* __restrict__ bias, int E, int top_k,
                                                             int gating_sigmoid, int norm_weights, float norm_eps,
                                                             float route_scale, int* __restrict__ idx_out,
                                                             float* __restrict__ w_out) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* lr = logits + size_t(r) * E;
  extern __shared__ float sh[];  // probs[E], sel[E]
  float* probs = sh;
  float* sel = sh + E;
  __shared__ int s_top_i[kRouteTop];
  __shared__ float s_top_w[kRouteTop];
  __shared__ float s_norm;

  if (gating_sigmoid) {
    for (int e = tid; e < E; e += kRouteThreads) {
      const float p = sigm(lr[e]);
      probs[e] = p;
      const float score = p + (bias ? bias[e] : 0.f);
      sel[e] = fmaxf(score, -INFINITY);  // CUDA fmax maps a NaN score to the lower endpoint.
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
      const float score = p + (bias ? bias[e] : 0.f);
      sel[e] = fmaxf(score, -INFINITY);  // CUDA fmax maps a NaN score to the lower endpoint.
    }
  }
  __syncthreads();

  // The top_k best in order: a thread counts, for each of its experts, those better than it (score_better is a total
  // order, ties to the lower index), and an expert with fewer than top_k before it is the choice of that rank: what
  // choosing one best at a time gave, each behind two barriers, with two barriers for all of them (a round runs this
  // once a layer visit, and a launch lasts as long as its barriers).
  for (int e = tid; e < E; e += kRouteThreads) {
    const float v = sel[e];
    int before = 0;
    for (int j = 0; j < E; ++j) before += score_better(sel[j], j, v, e);
    if (before < top_k) {
      s_top_i[before] = e;
      s_top_w[before] = probs[e];
    }
  }
  __syncthreads();
  if (tid == 0) {
    float sum_w = 0.f;
    for (int k = 0; k < top_k; ++k) sum_w += s_top_w[k];
    s_norm = norm_weights ? (route_scale / (sum_w + norm_eps)) : route_scale;
  }
  __syncthreads();
  int* out_i = idx_out + size_t(r) * top_k;
  float* out_w = w_out + size_t(r) * top_k;
  for (int k = tid; k < top_k; k += kRouteThreads) {
    out_i[k] = s_top_i[k];
    out_w[k] = s_top_w[k] * s_norm;
  }
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

// ---- Compacted work-tile grouping & one-pass activation staging ---------------------------
// Sorts M = rc * top_k pairs into per-expert ranges in pairs[] and emits load-balanced BM-row
// tiles into tasks[0 .. *n_tasks) so hot experts split across SMs and empty experts consume
// zero blocks (fixed upper-bound grid, 100% CUDA Graph safe).
// One block of kGroupThreads: the counts in shared memory, one block-wide scan of every thread's run of experts (their
// pairs and their tiles), then every thread writes its own experts' tiles; no thread walks all the experts
// (a walk through global memory, one expert after another, took 20 us of a round's every MoE GEMM).  Tiles go in
// ascending expert order; an expert's pairs in the order the atomics give them, which no result depends on: each
// pair's row is its own.
constexpr int kGroupThreads = 256;

template <bool Filter>
__global__ void __launch_bounds__(kGroupThreads)
    k_moe_group_tasks(const int* __restrict__ eidx, int M, int E, int bm, int first, int* __restrict__ pairs,
                      MoeTask* __restrict__ tasks, int* __restrict__ n_tasks) {
  pdl_trigger();
  extern __shared__ int s_group[];  // [E] pairs an expert has, then [E] where its next pair goes
  int* cnt = s_group;
  int* cur = s_group + E;
  __shared__ int s_warp_pairs[kGroupThreads / 32], s_warp_tiles[kGroupThreads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += kGroupThreads) cnt[e] = 0;
  __syncthreads();
  for (int i = tid; i < M; i += kGroupThreads) {
    int e = eidx[i] - (Filter ? first : 0);
    if (!Filter || unsigned(e) < unsigned(E)) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  const int per = (E + kGroupThreads - 1) / kGroupThreads;
  const int e0 = min(E, tid * per), e1 = min(E, e0 + per);
  int my_pairs = 0, my_tiles = 0;
  for (int e = e0; e < e1; ++e) {
    my_pairs += cnt[e];
    my_tiles += (cnt[e] + bm - 1) / bm;
  }
  int in_pairs = my_pairs, in_tiles = my_tiles;  // inclusive scans within the warp
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int a = __shfl_up_sync(0xffffffffu, in_pairs, o), b = __shfl_up_sync(0xffffffffu, in_tiles, o);
    if (lane >= o) {
      in_pairs += a;
      in_tiles += b;
    }
  }
  if (lane == 31) {
    s_warp_pairs[warp] = in_pairs;
    s_warp_tiles[warp] = in_tiles;
  }
  __syncthreads();
  int p = in_pairs - my_pairs, t = in_tiles - my_tiles;
  for (int w = 0; w < warp; ++w) {
    p += s_warp_pairs[w];
    t += s_warp_tiles[w];
  }
  for (int e = e0; e < e1; ++e) {
    const int c = cnt[e];
    cur[e] = p;
    for (int off = 0; off < c; off += bm) tasks[t++] = MoeTask{e, p + off, min(bm, c - off)};
    p += c;
  }
  if (tid == kGroupThreads - 1) *n_tasks = t;  // its run is the last: its end is the total
  __syncthreads();
  for (int i = tid; i < M; i += kGroupThreads) {
    int e = eidx[i] - (Filter ? first : 0);
    if (!Filter || unsigned(e) < unsigned(E)) pairs[atomicAdd(&cur[e], 1)] = i;
  }
}

void group_tasks(const int* eidx, int M, int E, int bm, const MoeScratch* sc, cudaStream_t st, int first) {
  const size_t bytes = size_t(2) * E * sizeof(int);
  if (first < 0)
    k_moe_group_tasks<false><<<1, kGroupThreads, bytes, st>>>(eidx, M, E, bm, 0, sc->pairs, sc->tasks, sc->n_tasks);
  else
    k_moe_group_tasks<true><<<1, kGroupThreads, bytes, st>>>(eidx, M, E, bm, first, sc->pairs, sc->tasks, sc->n_tasks);
}

// Whether the tiles moe_gate_up left (`grouped` rows a tile, every one of `rows` rows in one chunk) serve a kernel
// that takes `bm` rows a tile over the same rows: they are its tiles where both hold all the rows an expert can have,
// as a row picks an expert once.
bool tiles_serve(int grouped, int bm, int rows) {
  return grouped > 0 && (grouped == bm || (grouped >= rows && bm >= rows));
}

template <bool MINS>
__global__ void __launch_bounds__(256)
    k_moe_stage_q8(const float* __restrict__ x, int ldx, int K, int8_t* __restrict__ q8_a, float* __restrict__ q8_d,
                   float* __restrict__ q8_s) {
  pdl_trigger();
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int units = K >> 5;
  const int sub = lane & 7, g_in_warp = lane >> 3;
  const size_t row_at = size_t(r) * ldx;
  const bool aligned4 = (ldx & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
  for (int base = warp * 4; base < units; base += (int(blockDim.x) >> 5) * 4) {
    const int grp = base + g_in_warp;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (grp < units) {
      const float* src = x + row_at + size_t(grp) * 32 + 4 * sub;
      if (aligned4) v = *reinterpret_cast<const float4*>(src);
      else v = make_float4(src[0], src[1], src[2], src[3]);
    }
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (grp < units) {
      const float dg = m / 127.f;
      const int p0 = q8_of(v.x, dg), p1 = q8_of(v.y, dg), p2 = q8_of(v.z, dg), p3 = q8_of(v.w, dg);
      const uint32_t packed = uint32_t(uint8_t(int8_t(p0))) | (uint32_t(uint8_t(int8_t(p1))) << 8) |
                              (uint32_t(uint8_t(int8_t(p2))) << 16) | (uint32_t(uint8_t(int8_t(p3))) << 24);
      reinterpret_cast<uint32_t*>(q8_a + size_t(r) * K + size_t(grp) * 32)[sub] = packed;
      if (sub == 0) q8_d[size_t(r) * units + grp] = dg;
      if constexpr (MINS) {
        int s16 = p0 + p1 + p2 + p3;
        s16 += __shfl_xor_sync(0xffffffffu, s16, 1);
        s16 += __shfl_xor_sync(0xffffffffu, s16, 2);
        if ((sub & 3) == 0) q8_s[(size_t(r) * units + grp) * 2 + (sub >> 2)] = dg * float(s16);
      }
    }
  }
}

__global__ void __launch_bounds__(256)
    k_moe_stage_bf16(const float* x, int ldx, int K, bf16* __restrict__ out) {
  pdl_trigger();
  pdl_wait();  // x: the kernel before
  const int r = blockIdx.x;
  const float* src = x + size_t(r) * ldx;
  bf16* dst = out + size_t(r) * K;
  const bool aligned4 = (ldx & 3) == 0 && (K & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
  if (aligned4) {
    const int k4 = K >> 2;
    for (int i = threadIdx.x; i < k4; i += blockDim.x) {
      const float4 v = reinterpret_cast<const float4*>(src)[i];
      const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
      const __nv_bfloat162 b = __floats2bfloat162_rn(v.z, v.w);
      reinterpret_cast<uint2*>(dst)[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&a),
                                                    *reinterpret_cast<const uint32_t*>(&b));
    }
  } else {
    for (int i = threadIdx.x; i < K; i += blockDim.x) dst[i] = __float2bfloat16_rn(src[i]);
  }
}

__global__ void k_moe_weighted_sum(const float* down_buf, const float* ew, int top_k, int H, float* out, int ld_out,
                                   int accumulate) {
  pdl_trigger();
  pdl_wait();
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

// ---- Tensor Core NVFP4 MoE (mma_bf16 m16n8k16, exact e2m1 * e4m3 in bf16) -----------------
// Unified bit-identical math across R=1 (DIRECT_R1), R=2..16, and R=256 (compacted 16-row tasks).
template <int SPAN, bool FUSE_ACT, bool IS_DOWN, bool DIRECT_R1, int NW, int NT>
__global__ void __launch_bounds__(32 * NW)
    k_moe_nvfp4_mma(const float* x, int ldx, const bf16* bf16_a, const uint8_t* __restrict__ Q,
                    const uint8_t* __restrict__ S, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                    const int* eidx, const MoeTask* tasks, const int* n_tasks, const int* pairs, int top_k,
                    float situ_beta, float situ_linear_beta, float* __restrict__ out, int ld_out) {
  pdl_trigger();
  pdl_wait();  // x or bf16_a, the experts and the tasks: the kernels before
  static constexpr int VPL = SPAN / 4, WORDS = VPL / 8, BW = VPL / 2;
  static constexpr int THREADS = 32 * NW;
  static constexpr int SM_ROWS = DIRECT_R1 ? 1 : 16;

  int e, nb;
  __shared__ alignas(16) bf16 sm_x[2][SM_ROWS][SPAN];
  __shared__ int s_pids[SM_ROWS];
  __shared__ int s_rids[SM_ROWS];

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
  __syncthreads();

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int n_base = (int(blockIdx.x) * NW + warp) * 8 * NT;
  const int spans = K / SPAN;
  const float rs = RS ? RS[e] : 1.f;

  const uint8_t *qp0[NT], *sp0[NT];
  const uint8_t *qp1[FUSE_ACT ? NT : 1], *sp1[FUSE_ACT ? NT : 1];
  bool wv0[NT];
  float acc0[NT][4], acc1[FUSE_ACT ? NT : 1][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int col = n_base + i * 8 + g;
    wv0[i] = col < N_work;
    const size_t r0 = size_t(e) * N_exp + min(col, N_exp - 1);
    qp0[i] = Q + r0 * (K >> 1) + (VPL / 2) * t4;
    sp0[i] = S + r0 * (K >> 4) + (VPL / 16) * t4;
    acc0[i][0] = acc0[i][1] = acc0[i][2] = acc0[i][3] = 0.f;
    if constexpr (FUSE_ACT) {
      const size_t r1 = size_t(e) * N_exp + min(col + I_or_H, N_exp - 1);
      qp1[i] = Q + r1 * (K >> 1) + (VPL / 2) * t4;
      sp1[i] = S + r1 * (K >> 4) + (VPL / 16) * t4;
      acc1[i][0] = acc1[i][1] = acc1[i][2] = acc1[i][3] = 0.f;
    }
  }

  auto stage = [&](int buf, int k) {
    if (bf16_a) {
      constexpr int chunks = SPAN / 8;  // 16 bytes = 8 bf16 values
      for (int idx = threadIdx.x; idx < nb * chunks; idx += THREADS) {
        const int b = DIRECT_R1 ? 0 : (idx / chunks);
        const int c = DIRECT_R1 ? idx : (idx - b * chunks);
        cp_async16(&sm_x[buf][b][8 * c], bf16_a + size_t(s_rids[b]) * K + k + 8 * c);
      }
      cp_async_commit();
    } else {
      for (int idx = threadIdx.x; idx < nb * SPAN; idx += THREADS) {
        const int b = DIRECT_R1 ? 0 : (idx / SPAN);
        const int j = DIRECT_R1 ? idx : (idx - b * SPAN);
        sm_x[buf][b][j] = __float2bfloat16_rn(x[size_t(s_rids[b]) * ldx + k + j]);
      }
    }
  };

  // Decode one fp4x8 word (4 bf16x2 regs) at a time and feed two mma_bf16 steps immediately:
  // keeps exact K-accumulation order (0.00e+00 vs R=1/R=256/serial) while slashing registers from ~89 to ~56.
  // A span's weights and scales, loaded one span ahead of their products (the next span's loads are in flight while
  // this span's mma run: twice the bytes in flight a warp, the same products in the same order).
  struct Wspan {
    uint32_t wq[NT][WORDS];
    uint32_t sc[NT];
  };
  auto load_w = [&](const uint8_t* const* qp, const uint8_t* const* sp, int k, Wspan& w) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      if constexpr (SPAN == 128) {
        const uint4 v = wv0[i] ? ldg_stream(qp[i] + (k >> 1)) : make_uint4(0, 0, 0, 0);
        w.wq[i][0] = v.x, w.wq[i][1] = v.y, w.wq[i][2] = v.z, w.wq[i][3] = v.w;
        w.sc[i] = wv0[i] ? *reinterpret_cast<const uint16_t*>(sp[i] + (k >> 4)) : 0u;
      } else {
        const uint2 v = wv0[i] ? ldg_stream2(qp[i] + (k >> 1)) : make_uint2(0, 0);
        w.wq[i][0] = v.x, w.wq[i][1] = v.y;
        w.sc[i] = wv0[i] ? sp[i][k >> 4] : 0u;
      }
    }
  };
  auto mma_half = [&](const Wspan& w, const uint32_t* av, const uint32_t* bv, float (*acc)[4]) {
    const auto& wq = w.wq;
    const auto& sc = w.sc;
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
      for (int b = 0; b < VPL / 16; ++b) {
        const __nv_bfloat162 s2 = e4m3_bf16x2((sc[i] >> (8 * b)) & 0xFFu);
#pragma unroll
        for (int q_in_b = 0; q_in_b < 2; ++q_in_b) {
          const int q = 2 * b + q_in_b;
          uint32_t bw[4];
          fp4x8_to_bf16(wq[i][q], bw);
#pragma unroll
          for (int e4 = 0; e4 < 4; ++e4) bw[e4] = bf16x2_mul(bw[e4], s2);
          const uint32_t a0[4] = {av[4 * q], DIRECT_R1 ? 0u : bv[4 * q], av[4 * q + 1],
                                  DIRECT_R1 ? 0u : bv[4 * q + 1]};
          const uint32_t bf0[2] = {bw[0], bw[1]};
          mma_bf16(acc[i], a0, bf0);
          const uint32_t a1[4] = {av[4 * q + 2], DIRECT_R1 ? 0u : bv[4 * q + 2], av[4 * q + 3],
                                  DIRECT_R1 ? 0u : bv[4 * q + 3]};
          const uint32_t bf1[2] = {bw[2], bw[3]};
          mma_bf16(acc[i], a1, bf1);
        }
      }
    }
  };

  const bool ra = g < nb, rb = !DIRECT_R1 && ((g + 8) < nb);
  const bool works = n_base < N_work;
  // One row (DIRECT_R1) loads a span ahead; more rows keep a span's loads beside its products (ahead measured 1-2%
  // slower on 3-7 row rounds there: the registers, where the rows already keep a warp's loads busy).
  constexpr bool kAhead = DIRECT_R1;
  // One row loads two spans ahead (a0/a1 hold span s + 1 while span s + 2 loads): one row's 4-5 warps an SM keep too
  // few bytes in flight with one (RTX 5090 Inkling-2-Mini decode +1.1% over one ahead).
  constexpr bool kAhead2 = DIRECT_R1;
  Wspan w0, w1;  // w1 unused (and dead) unless FUSE_ACT
  Wspan a0, a1;  // span s + 1 (kAhead2)
  if (kAhead && works) {
    load_w(qp0, sp0, 0, w0);
    if constexpr (FUSE_ACT) load_w(qp1, sp1, 0, w1);
    if constexpr (kAhead2) {
      if (spans > 1) {
        load_w(qp0, sp0, SPAN, a0);
        if constexpr (FUSE_ACT) load_w(qp1, sp1, SPAN, a1);
      }
    }
  }
  stage(0, 0);
  for (int s_idx = 0; s_idx < spans; ++s_idx) {
    if (bf16_a) cp_async_wait<0>();
    __syncthreads();
    if (s_idx + 1 < spans) stage((s_idx + 1) & 1, (s_idx + 1) * SPAN);
    if (n_base < N_work) {
      const int buf = s_idx & 1;
      uint32_t av[BW];
      uint32_t bv[DIRECT_R1 ? 1 : BW];
      const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int q = 0; q < VPL / 8; ++q) {
        const uint4 va = ra ? *reinterpret_cast<const uint4*>(&sm_x[buf][DIRECT_R1 ? 0 : g][VPL * t4 + 8 * q]) : z;
        av[4 * q] = va.x, av[4 * q + 1] = va.y, av[4 * q + 2] = va.z, av[4 * q + 3] = va.w;
        if constexpr (!DIRECT_R1) {
          const uint4 vb = rb ? *reinterpret_cast<const uint4*>(&sm_x[buf][g + 8][VPL * t4 + 8 * q]) : z;
          bv[4 * q] = vb.x, bv[4 * q + 1] = vb.y, bv[4 * q + 2] = vb.z, bv[4 * q + 3] = vb.w;
        }
      }
      Wspan n0, n1;
      const int ahead = kAhead2 ? 2 : 1;
      const bool more = kAhead && s_idx + ahead < spans;
      if (!kAhead) {
        load_w(qp0, sp0, s_idx * SPAN, w0);
        if constexpr (FUSE_ACT) load_w(qp1, sp1, s_idx * SPAN, w1);
      }
      if (more) {
        const int kn = (s_idx + ahead) * SPAN;
        load_w(qp0, sp0, kn, n0);
        if constexpr (FUSE_ACT) load_w(qp1, sp1, kn, n1);
      }
      mma_half(w0, av, bv, acc0);
      if constexpr (FUSE_ACT) mma_half(w1, av, bv, acc1);
      if constexpr (kAhead2) {
        w0 = a0;
        if constexpr (FUSE_ACT) w1 = a1;
        if (more) {
          a0 = n0;
          if constexpr (FUSE_ACT) a1 = n1;
        }
      } else if (more) {
        w0 = n0;
        if constexpr (FUSE_ACT) w1 = n1;
      }
    }
  }

  if (n_base < N_work) {
#pragma unroll
    for (int e_row = 0; e_row < (DIRECT_R1 ? 2 : 4); e_row += 2) {
      const int b = g + (e_row >> 1) * 8;
      if (b >= nb) continue;
      const size_t out_row = size_t(s_pids[DIRECT_R1 ? 0 : b]) * ld_out;
#pragma unroll
      for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int col = n_base + i * 8 + 2 * t4 + j;
          if (col < N_work) {
            const float gv = acc0[i][e_row + j] * rs;
            if constexpr (FUSE_ACT) {
              const float uv = acc1[i][e_row + j] * rs;
              out[out_row + col] = situ_val(gv, uv, situ_beta, situ_linear_beta);
            } else {
              out[out_row + col] = gv;
            }
          }
        }
      }
    }
  }
}

// ---- NVFP4 expert products a column at a time (all pass sizes) ----------------
// A warp takes kVecRows weight rows of one expert (FUSE_ACT: two columns' gate and up rows); lane l takes the
// 16-value groups l, l + 32, ... of K, each group's sum of fp4 * x in value order times its e4m3 scale into the
// lane's float32 sum, then the warp's xor tree and the expert's scale.  Every routed (row, expert) pair's sums are
// these, in this order, whichever other rows the round holds: a column's bits do not depend on the round's width
// (plain decode == speculative verify).  float32 activations (no bf16 staging); weights read once a block for up
// to BMAX pairs of the expert.
constexpr int kVecWarps = 4, kVecRows = 4;

template <bool FUSE_ACT, bool IS_DOWN, int BMAX>
__device__ __forceinline__ void nvfp4_vec_products(const float* x, int ldx, const uint8_t* Q, const uint8_t* S,
                                                    const float* RS, int N_exp, int I_or_H, int K, int top_k, int e,
                                                    const int (&pid)[BMAX], int nb, int c0,
                                                    float (&acc)[BMAX][kVecRows]) {
  const int lane = threadIdx.x & 31;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const uint8_t* qr[kVecRows];
  const uint8_t* sr[kVecRows];
#pragma unroll
  for (int i = 0; i < kVecRows; ++i) {
    const int col = min(c0 + (FUSE_ACT ? i / 2 : i), N_work - 1);
    const int row = FUSE_ACT && (i & 1) ? col + I_or_H : col;
    const size_t r0 = size_t(e) * N_exp + row;
    qr[i] = Q + r0 * size_t(K >> 1);
    sr[i] = S + r0 * size_t(K >> 4);
  }
  const float* xr[BMAX];
#pragma unroll
  for (int b = 0; b < BMAX; ++b) xr[b] = x + size_t(IS_DOWN ? pid[b] : pid[b] / top_k) * ldx;

#pragma unroll
  for (int b = 0; b < BMAX; ++b)
#pragma unroll
    for (int i = 0; i < kVecRows; ++i) acc[b][i] = 0.f;

  const int groups = K >> 4;
#pragma unroll 2
  for (int s = lane; s < groups; s += 32) {
    uint2 wq[kVecRows];
    float sc[kVecRows];
#pragma unroll
    for (int i = 0; i < kVecRows; ++i) {
      wq[i] = ldg_stream2(qr[i] + 8 * size_t(s));
      sc[i] = __low2float(e4m3_bf16x2(sr[i][s]));
    }
#pragma unroll
    for (int b = 0; b < BMAX; ++b) {
      if (b >= nb) break;
      float xv[16];
      const float* src = xr[b] + 16 * size_t(s);
      if ((reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const float4 v = reinterpret_cast<const float4*>(src)[q];
          xv[4 * q] = v.x, xv[4 * q + 1] = v.y, xv[4 * q + 2] = v.z, xv[4 * q + 3] = v.w;
        }
      } else {
#pragma unroll
        for (int q = 0; q < 16; ++q) xv[q] = src[q];
      }
#pragma unroll
      for (int i = 0; i < kVecRows; ++i) {
        uint32_t h[8];
        fp4x8_to_bf16(wq[i].x, h);
        fp4x8_to_bf16(wq[i].y, h + 4);
        float g = 0.f;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&h[p]));
          g = __fmaf_rn(f.x, xv[2 * p], g);
          g = __fmaf_rn(f.y, xv[2 * p + 1], g);
        }
        acc[b][i] = __fmaf_rn(g, sc[i], acc[b][i]);
      }
    }
  }

  const float rs = RS ? RS[e] : 1.f;
#pragma unroll
  for (int b = 0; b < BMAX; ++b) {
    if (b >= nb) break;
#pragma unroll
    for (int i = 0; i < kVecRows; ++i) acc[b][i] = __fmul_rn(warp_tree_sum(acc[b][i]), rs);
  }
}

template <bool FUSE_ACT, bool IS_DOWN, int BMAX>
__global__ void __launch_bounds__(32 * kVecWarps)
    k_moe_nvfp4_vec(const float* x, int ldx, const uint8_t* __restrict__ Q, const uint8_t* __restrict__ S,
                    const float* __restrict__ RS, int N_exp, int I_or_H, int K, const int* eidx, const MoeTask* tasks,
                    const int* n_tasks, const int* pairs, int top_k, float situ_beta, float situ_linear_beta,
                    float* __restrict__ out, int ld_out) {
  pdl_trigger();
  pdl_wait();  // x, the experts and the tasks: the kernels before
  int e, nb;
  int pid[BMAX];
  if constexpr (BMAX == 1) {
    pid[0] = int(blockIdx.y);
    e = eidx[pid[0]];
    nb = 1;
  } else {
    if (int(blockIdx.y) >= *n_tasks) return;
    const MoeTask tk = tasks[blockIdx.y];
    e = tk.expert;
    nb = tk.nb;
#pragma unroll
    for (int b = 0; b < BMAX; ++b) pid[b] = pairs[tk.p0 + (b < nb ? b : 0)];
  }
  constexpr int COLS = FUSE_ACT ? kVecRows / 2 : kVecRows;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = (int(blockIdx.x) * kVecWarps + warp) * COLS;
  if (c0 >= N_work) return;

  float acc[BMAX][kVecRows];
  nvfp4_vec_products<FUSE_ACT, IS_DOWN>(x, ldx, Q, S, RS, N_exp, I_or_H, K, top_k, e, pid, nb, c0, acc);
  if (lane != 0) return;
#pragma unroll
  for (int b = 0; b < BMAX; ++b) {
    if (b >= nb) break;
    float* orow = out + size_t(pid[b]) * ld_out;
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
      const int col = c0 + c;
      if (col >= N_work) break;
      if constexpr (FUSE_ACT)
        orow[col] = situ_val(acc[b][2 * c], acc[b][2 * c + 1], situ_beta, situ_linear_beta);
      else
        orow[col] = acc[b][c];
    }
  }
}

// With no per-expert output scratch, keep one row's weighted sum in registers.
__global__ void k_moe_nvfp4_vec_sum(const float* act, int ld, const uint8_t* Q, const uint8_t* S, const float* RS,
                                    const int* ids, const float* weights, int top_k, int H, int K,
                                    float* out, int ld_out, bool accumulate) {
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.y, lane = threadIdx.x & 31;
  const int c0 = (int(blockIdx.x) * kVecWarps + (threadIdx.x >> 5)) * kVecRows;
  if (c0 >= H) return;
  float sum[kVecRows]{};
  for (int k = 0; k < top_k; ++k) {
    const int pid[1] = {row * top_k + k};
    float dot[1][kVecRows];
    nvfp4_vec_products<false, true>(act, ld, Q, S, RS, H, H, K, top_k, ids[pid[0]], pid, 1, c0, dot);
    if (lane == 0) {
#pragma unroll
      for (int j = 0; j < kVecRows; ++j) sum[j] = __fmaf_rn(weights[pid[0]], dot[0][j], sum[j]);
    }
  }
  if (lane == 0) {
    float* dst = out + size_t(row) * ld_out + c0;
#pragma unroll
    for (int j = 0; j < kVecRows; ++j)
      if (c0 + j < H) dst[j] = accumulate ? dst[j] + sum[j] : sum[j];
  }
}

// One arithmetic contract for both rounds and prefill: the tensor-core path (bf16 inputs, MMA at every width) by
// default; option moe_vec=1 (KernelOptions::moe_vec) selects the F32 column-at-a-time vector kernels instead.
// Measured on an RTX 5070 Ti (Inkling-2-Mini UD-IQ3_XXS, --ctx 4096, exact K/V): the vector kernels gave plain
// decoding 1% more, but DSpark's rounds of 5 and 7 rows took 9-10% longer (decode -8% to -9% against the tensor-core
// path) and prefill, split into 16-row chunks, lost 18-25%; the tensor-core path also keeps the greedy outputs of the
// build before these kernels bit for bit.

// No-scratch fallback for NVFP4 moe_down (sc == nullptr): identical mma_bf16 span order + ew FMA order.
template <int SPAN, int NW, int NT>
__global__ void __launch_bounds__(32 * NW)
    k_moe_nvfp4_down_serial(const float* __restrict__ act, int ld_act, const uint8_t* __restrict__ Q,
                            const uint8_t* __restrict__ S, const float* __restrict__ RS, int H, int K,
                            const int* __restrict__ eidx, const float* __restrict__ ew, int top_k,
                            float* __restrict__ out, int ld_out, int accumulate) {
  pdl_trigger();
  static constexpr int VPL = SPAN / 4, STEPS = VPL / 4, WORDS = VPL / 8, BW = VPL / 2;
  __shared__ alignas(16) bf16 sm_x[SPAN];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t4 = lane & 3;
  const int n_base = (int(blockIdx.x) * NW + warp) * 8 * NT;
  const int r = int(blockIdx.y), spans = K / SPAN;

  float total[NT][2];
#pragma unroll
  for (int i = 0; i < NT; ++i) total[i][0] = total[i][1] = 0.f;

  for (int k_slot = 0; k_slot < top_k; ++k_slot) {
    const int e = eidx[r * top_k + k_slot];
    const float rs = RS ? RS[e] : 1.f;
    const float w_e = ew[r * top_k + k_slot];
    const uint8_t *qp[NT], *sp[NT];
    bool wv[NT];
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int col = n_base + i * 8 + g;
      wv[i] = col < H;
      const size_t r0 = size_t(e) * H + min(col, H - 1);
      qp[i] = Q + r0 * (K >> 1) + (VPL / 2) * t4;
      sp[i] = S + r0 * (K >> 4) + (VPL / 16) * t4;
      acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    const float* arow = act + size_t(r * top_k + k_slot) * ld_act;
    for (int s_idx = 0; s_idx < spans; ++s_idx) {
      const int k = s_idx * SPAN;
      for (int j = threadIdx.x; j < SPAN; j += blockDim.x) sm_x[j] = __float2bfloat16_rn(arow[k + j]);
      __syncthreads();
      if (n_base < H) {
        uint32_t av[BW];
        const uint4 z = make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int q = 0; q < VPL / 8; ++q) {
          const uint4 va = (g == 0) ? *reinterpret_cast<const uint4*>(&sm_x[VPL * t4 + 8 * q]) : z;
          av[4 * q] = va.x, av[4 * q + 1] = va.y, av[4 * q + 2] = va.z, av[4 * q + 3] = va.w;
        }
        uint32_t wq[NT][WORDS], sc[NT], bw[NT][BW];
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          if constexpr (SPAN == 128) {
            const uint4 v = wv[i] ? ldg_stream(qp[i] + (k >> 1)) : z;
            wq[i][0] = v.x, wq[i][1] = v.y, wq[i][2] = v.z, wq[i][3] = v.w;
            sc[i] = wv[i] ? *reinterpret_cast<const uint16_t*>(sp[i] + (k >> 4)) : 0u;
          } else {
            const uint2 v = wv[i] ? ldg_stream2(qp[i] + (k >> 1)) : make_uint2(0, 0);
            wq[i][0] = v.x, wq[i][1] = v.y;
            sc[i] = wv[i] ? sp[i][k >> 4] : 0u;
          }
#pragma unroll
          for (int q = 0; q < WORDS; ++q) fp4x8_to_bf16(wq[i][q], bw[i] + 4 * q);
#pragma unroll
          for (int b = 0; b < VPL / 16; ++b) {
            const __nv_bfloat162 s2 = e4m3_bf16x2((sc[i] >> (8 * b)) & 0xFFu);
#pragma unroll
            for (int e8 = 0; e8 < 8; ++e8) bw[i][8 * b + e8] = bf16x2_mul(bw[i][8 * b + e8], s2);
          }
        }
#pragma unroll
        for (int j = 0; j < STEPS; ++j) {
          const uint32_t a[4] = {av[2 * j], 0u, av[2 * j + 1], 0u};
#pragma unroll
          for (int i = 0; i < NT; ++i) {
            const uint32_t bf[2] = {bw[i][2 * j], bw[i][2 * j + 1]};
            mma_bf16(acc[i], a, bf);
          }
        }
      }
      __syncthreads();
    }
    if (g == 0 && n_base < H) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        total[i][0] = __fmaf_rn(w_e, acc[i][0] * rs, total[i][0]);
        total[i][1] = __fmaf_rn(w_e, acc[i][1] * rs, total[i][1]);
      }
    }
  }
  if (g == 0 && n_base < H) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = n_base + i * 8 + 2 * t4 + j;
        if (col < H) {
          float* dst = out + size_t(r) * ld_out + col;
          *dst = accumulate ? (*dst + total[i][j]) : total[i][j];
        }
      }
    }
  }
}

template <int SPAN>
int launch_nvfp4_gu(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                    float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out,
                    const MoeScratch* sc, const Launch& L, int first) {
  cudaStream_t st = L.st;
  const int N_work = fuse_act ? I : 2 * I;
  const bool has_sc = sc && sc->tasks && sc->n_tasks && sc->pairs && W.n_experts <= sc->max_experts &&
                      top_k <= sc->max_top_k;
  const bool vec = opts(L).moe_vec;
  const int capacity = has_sc ? std::max(1, sc->chunk_rows) : rows;
  const int chunk = vec ? std::min(capacity, FLYX_MAX_ROWS) : capacity;
  int grouped = 0;
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    const float* xc = x + size_t(r0) * ldx;
    const int* ec = eidx + size_t(r0) * top_k;
    float* oc = out + size_t(r0) * top_k * ld_out;
    if (vec) {
      const int gx = ceil_div(N_work, kVecWarps * (fuse_act ? kVecRows / 2 : kVecRows));
      if (rc == 1 || !has_sc) {
        if (fuse_act)
          launch_dependent(k_moe_nvfp4_vec<true, false, 1>, dim3(gx, rc * top_k), dim3(32 * kVecWarps), L,
                           xc, ldx, W.q, W.s, W.rs, W.N, I, K, ec, nullptr, nullptr, nullptr, top_k, situ_beta,
                           situ_linear_beta, oc, ld_out);
        else
          launch_dependent(k_moe_nvfp4_vec<false, false, 1>, dim3(gx, rc * top_k), dim3(32 * kVecWarps), L,
                           xc, ldx, W.q, W.s, W.rs, W.N, I, K, ec, nullptr, nullptr, nullptr, top_k, situ_beta,
                           situ_linear_beta, oc, ld_out);
      } else {
        const int M = rc * top_k;
        const int mt = max_work_tiles(M, W.n_experts, 4);
        group_tasks(ec, M, W.n_experts, 4, sc, st, first);
        grouped = rc == rows ? 4 : 0;
        if (fuse_act)
          launch_dependent(k_moe_nvfp4_vec<true, false, 4>, dim3(gx, mt), dim3(32 * kVecWarps), L,
                           xc, ldx, W.q, W.s, W.rs, W.N, I, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                           situ_beta, situ_linear_beta, oc, ld_out);
        else
          launch_dependent(k_moe_nvfp4_vec<false, false, 4>, dim3(gx, mt), dim3(32 * kVecWarps), L,
                           xc, ldx, W.q, W.s, W.rs, W.N, I, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                           situ_beta, situ_linear_beta, oc, ld_out);
      }
      continue;
    }
    const bf16* bf_a = nullptr;
    if (rc > 1 && sc && sc->bf16_a && sc->stage_elems >= size_t(rc) * size_t(K)) {
      launch_dependent(k_moe_stage_bf16, dim3(rc), dim3(256), L, xc, ldx, K, sc->bf16_a);
      bf_a = sc->bf16_a;
    }
    if (rc == 1) {
      if (fuse_act)
        launch_dependent(k_moe_nvfp4_mma<SPAN, true, false, true, 4, 1>, dim3(ceil_div(N_work, 32), top_k),
                         dim3(128), L, xc, ldx, bf_a, W.q, W.s, W.rs, W.N, I, K, ec, nullptr, nullptr, nullptr, top_k,
                         situ_beta, situ_linear_beta, oc, ld_out);
      else
        launch_dependent(k_moe_nvfp4_mma<SPAN, false, false, true, 4, 2>, dim3(ceil_div(N_work, 64), top_k),
                         dim3(128), L, xc, ldx, bf_a, W.q, W.s, W.rs, W.N, I, K, ec, nullptr, nullptr, nullptr, top_k,
                         situ_beta, situ_linear_beta, oc, ld_out);
    } else if (has_sc) {
      const int M = rc * top_k;
      const int mt = max_work_tiles(M, W.n_experts, 16);
      group_tasks(ec, M, W.n_experts, 16, sc, st, first);
      grouped = rc == rows ? 16 : 0;
      if (fuse_act)
        launch_dependent(k_moe_nvfp4_mma<SPAN, true, false, false, 4, 1>, dim3(ceil_div(N_work, 32), mt), dim3(128), L,
                         xc, ldx, bf_a, W.q, W.s, W.rs, W.N, I, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                         situ_beta, situ_linear_beta, oc, ld_out);
      else
        launch_dependent(k_moe_nvfp4_mma<SPAN, false, false, false, 4, 2>, dim3(ceil_div(N_work, 64), mt), dim3(128),
                         L, xc, ldx, bf_a, W.q, W.s, W.rs, W.N, I, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                         situ_beta, situ_linear_beta, oc, ld_out);
    } else {
      // No-scratch multi-row fallback: 1 row at a time via DIRECT_R1 (exact same bits).
      for (int rr = 0; rr < rc; ++rr) {
        if (fuse_act)
          launch_dependent(k_moe_nvfp4_mma<SPAN, true, false, true, 4, 1>, dim3(ceil_div(N_work, 32), top_k),
                           dim3(128), L, xc + size_t(rr) * ldx, ldx, nullptr, W.q, W.s, W.rs, W.N, I, K,
                           ec + size_t(rr) * top_k, nullptr, nullptr, nullptr, top_k, situ_beta, situ_linear_beta,
                           oc + size_t(rr) * top_k * ld_out, ld_out);
        else
          launch_dependent(k_moe_nvfp4_mma<SPAN, false, false, true, 4, 2>, dim3(ceil_div(N_work, 64), top_k),
                           dim3(128), L, xc + size_t(rr) * ldx, ldx, nullptr, W.q, W.s, W.rs, W.N, I, K,
                           ec + size_t(rr) * top_k, nullptr, nullptr, nullptr, top_k, situ_beta, situ_linear_beta,
                           oc + size_t(rr) * top_k * ld_out, ld_out);
      }
    }
  }
  return grouped;
}

template <int SPAN>
void launch_nvfp4_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows,
                       int top_k, int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc,
                       const Launch& L, const bf16* act_bf16, int grouped, int first, bool defer_reduce) {
  cudaStream_t st = L.st;
  const bool has_down = sc && sc->down_buf && sc->tasks && sc->n_tasks && sc->pairs &&
                        W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
  const bool vec = !act_bf16 && opts(L).moe_vec;
  if (!vec && !has_down) {
    k_moe_nvfp4_down_serial<SPAN, 4, 2><<<dim3(ceil_div(H, 64), rows), 128, 0, st>>>(
        act, ld_act, W.q, W.s, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
    return;
  }
  const int capacity = has_down ? std::max(1, sc->chunk_rows) : rows;
  const int chunk = vec ? std::min(capacity, FLYX_MAX_ROWS) : capacity;
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    const float* ac = act_bf16 ? nullptr : act + size_t(r0) * top_k * ld_act;
    const int* ec = eidx + size_t(r0) * top_k;
    const int M = rc * top_k;
    if (vec) {
      const int gx = ceil_div(H, kVecWarps * kVecRows);
      if (!has_down) {
        launch_dependent(k_moe_nvfp4_vec_sum, dim3(gx, rc), dim3(32 * kVecWarps), L, ac, ld_act, W.q, W.s, W.rs,
                         ec, ew + size_t(r0) * top_k, top_k, H, K, out + size_t(r0) * ld_out, ld_out, accumulate);
        continue;
      }
      if (rc == 1) {
        launch_dependent(k_moe_nvfp4_vec<false, true, 1>, dim3(gx, top_k), dim3(32 * kVecWarps), L,
                         ac, ld_act, W.q, W.s, W.rs, H, H, K, ec, nullptr, nullptr, nullptr, top_k, 0.f, 0.f,
                         sc->down_buf, H);
      } else {
        const int mt = max_work_tiles(M, W.n_experts, 4);
        if (rc != rows || !tiles_serve(grouped, 4, rc)) group_tasks(ec, M, W.n_experts, 4, sc, st, first);
        launch_dependent(k_moe_nvfp4_vec<false, true, 4>, dim3(gx, mt), dim3(32 * kVecWarps), L,
                         ac, ld_act, W.q, W.s, W.rs, H, H, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k,
                         0.f, 0.f, sc->down_buf, H);
      }
      if (!defer_reduce)
        launch_dependent(k_moe_weighted_sum, dim3(ceil_div(H, 256), rc), dim3(256), L, sc->down_buf,
                         ew + size_t(r0) * top_k, top_k, H, out + size_t(r0) * ld_out, ld_out, int(accumulate));
      continue;
    }
    const bf16* bf_a = act_bf16;
    if (!bf_a && rc > 1 && sc->bf16_a && sc->stage_elems >= size_t(M) * size_t(K)) {
      launch_dependent(k_moe_stage_bf16, dim3(M), dim3(256), L, ac, ld_act, K, sc->bf16_a);
      bf_a = sc->bf16_a;
    }
    if (rc == 1) {
      launch_dependent(k_moe_nvfp4_mma<SPAN, false, true, true, 4, 1>, dim3(ceil_div(H, 32), top_k), dim3(128), L,
                       ac, ld_act, bf_a, W.q, W.s, W.rs, H, H, K, ec, nullptr, nullptr, nullptr, top_k, 0.f, 0.f,
                       sc->down_buf, H);
    } else {
      const int mt = max_work_tiles(M, W.n_experts, 16);
      if (rc != rows || !tiles_serve(grouped, 16, rc)) group_tasks(ec, M, W.n_experts, 16, sc, st, first);
      launch_dependent(k_moe_nvfp4_mma<SPAN, false, true, false, 4, 2>, dim3(ceil_div(H, 64), mt), dim3(128), L,
                       ac, ld_act, bf_a, W.q, W.s, W.rs, H, H, K, ec, sc->tasks, sc->n_tasks, sc->pairs, top_k, 0.f,
                       0.f, sc->down_buf, H);
    }
    if (!defer_reduce)
      launch_dependent(k_moe_weighted_sum, dim3(ceil_div(H, 256), rc), dim3(256), L, sc->down_buf,
                       ew + size_t(r0) * top_k, top_k, H, out + size_t(r0) * ld_out, ld_out, int(accumulate));
  }
}

// ---- BF16 Expert Kernels (float32 activations, compacted work tiles + xor tree) -----------
template <int BT>
__device__ __forceinline__ void stage_f32_tile(const float* __restrict__ x, int ldx, const int* row_ids, int nb, int k0,
                                               int kt, float sm[BT][kMoeKT]) {
  for (int idx = threadIdx.x; idx < nb * kt; idx += blockDim.x) {
    const int b = idx / kt, j = idx - b * kt;
    sm[b][j] = x[size_t(row_ids[b]) * ldx + k0 + j];
  }
}

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
    const float g = warp_tree_sum(ag[0]) * rs;
    if constexpr (FUSE_ACT) {
      const float u = warp_tree_sum(au[0]) * rs;
      if (lane == 0) out[size_t(r * top_k + k_slot) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
    } else if (lane == 0) {
      out[size_t(r * top_k + k_slot) * ld_out + col] = g;
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
      const float v = warp_tree_sum(acc[0]) * rs;
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
                       int N_exp, int I_or_H, int K, const MoeTask* __restrict__ tasks, const int* __restrict__ n_tasks,
                       const int* __restrict__ pairs, int top_k, float situ_beta, float situ_linear_beta,
                       float* __restrict__ out, int ld_out) {
  constexpr int BT = 8;
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;

  __shared__ float sm[BT][kMoeKT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int col = blockIdx.x * kMoeWarps + warp;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const float rs = RS ? RS[e] : 1.f;
  const bf16* w0 = W + (size_t(e) * N_exp + min(col, N_exp - 1)) * K;
  const bf16* w1 = FUSE_ACT ? (W + (size_t(e) * N_exp + min(col + I_or_H, N_exp - 1)) * K) : nullptr;

  int rids[BT], pids[BT];
#pragma unroll
  for (int b = 0; b < BT; ++b) {
    pids[b] = b < nb ? pairs[tk.p0 + b] : 0;
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
      const float g = warp_tree_sum(ag[b]) * rs;
      if constexpr (FUSE_ACT) {
        const float u = warp_tree_sum(au[b]) * rs;
        if (lane == 0) out[size_t(pids[b]) * ld_out + col] = situ_val(g, u, situ_beta, situ_linear_beta);
      } else if (lane == 0) {
        out[size_t(pids[b]) * ld_out + col] = g;
      }
    }
  }
}

// The rows of a tensor-core task for a pass of rc rows and M pairs.  Every pass of more than FLYX_MAX_ROWS rows takes
// the prompt kernel's 64-row tasks whatever its density, so a row's sums do not depend on the pass (numerics class K);
// with option moe_mmq=0 a dense one (2 or more rows an expert on average) takes the 16-row kernel with the rounds'
// bits, as before.  Rounds and the passes of 16 rows or fewer stay on the low-register dp4a kernels (4-row tasks).
int moe_mma_rows(const KernelOptions& o, int rc, int M, int n_experts) {
  if (rc <= FLYX_MAX_ROWS) return 0;
  if (o.moe_mmq) return kMoeMmqRows;
  return M >= 2 * n_experts ? 16 : 0;
}

bool moe_round_scratch(const ExpertMat& W, int rows, int top_k, const MoeScratch* sc) {
  return rows > 0 && rows <= FLYX_MAX_ROWS && sc && rows <= sc->chunk_rows && W.n_experts <= sc->max_experts &&
         top_k > 0 && top_k <= sc->max_top_k && sc->pairs && sc->tasks && sc->n_tasks;
}

// The moe_dp4a configuration of a packed expert dp4a GEMM of `rows` rows of block type `dtype`: the options' where
// they force one compiled for this op and row band, else a fused gate|up GEMM's by the rules, else the original tiles.
int moe_dp4a_cfg(const KernelOptions& o, int dtype, int rows, bool fused_gate_up) {
  if (o.moe_dp4a >= 0 && moe_dp4a_compiled(dtype, o.moe_dp4a, fused_gate_up, rows)) return o.moe_dp4a;
  return fused_gate_up ? moe_gate_up_dp4a(o.tuning, dtype, rows) : tables::kMoeDp4aOriginal;
}

// ---- expert reads ahead (moe_prefetch) ------------------------------------------------------
// While a round's attention runs, a stream of its own guesses the experts the MoE after it will take (its router on
// the current residual row, normed by the MoE norm's gain only, in place of the mix the MoE will see) and asks L2 for
// their NVFP4 weights.  Only L2 is touched: the MoE's own loads and sums are unchanged, so a wrong guess costs
// bandwidth, never bits.
constexpr int kGuessWarps = 8;

template <class W>
__global__ void __launch_bounds__(32 * kGuessWarps)
    k_moe_guess(const float* x, const float* __restrict__ norm_w, float norm_offset, const W* __restrict__ router,
                const float* __restrict__ bias, int sigmoid, int E, int H, float* scores) {
  const int lane = threadIdx.x & 31, e = blockIdx.x * kGuessWarps + (threadIdx.x >> 5);
  if (e >= E) return;
  const W* w = router + size_t(e) * H;
  float acc = 0.f;
  for (int k = lane; k < H; k += 32) {
    const float g = norm_w ? (norm_offset + __ldg(norm_w + k)) : 1.f;
    acc = __fmaf_rn(float(w[k]), x[k] * g, acc);
  }
  acc = warp_tree_sum(acc);
  if (lane == 0) scores[e] = (sigmoid ? sigm(acc) : acc) + (bias ? bias[e] : 0.f);
}

__device__ __forceinline__ void l2_prefetch_bulk(const void* p, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#endif
}

// L2 warm-up of small weights a later latency-bound kernel reads (l2_warm): thread t of the grid asks for the
// t-th 16 KB piece of the ranges laid end to end.
__global__ void k_l2_warm(L2Ranges r) {
  constexpr size_t kPiece = 16384;
  size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int i = 0; i < r.count; ++i) {
    const size_t pieces = (r.n[i] + kPiece - 1) / kPiece;
    if (t < pieces) {
      const size_t off = t * kPiece;
      const uint32_t n = uint32_t(min(kPiece, r.n[i] - off)) & ~15u;
      const uint8_t* a = static_cast<const uint8_t*>(r.p[i]) + off;
      if (n && (reinterpret_cast<uintptr_t>(a) & 15u) == 0) l2_prefetch_bulk(a, n);
      return;
    }
    t -= pieces;
  }
}

// An expert-major plane: one expert's bytes contiguous, per_expert of them.
struct PrefetchPlane {
  const uint8_t* base;
  size_t per_expert;
};

// One block of 256 threads (E <= 256): the top_k scores (a block argmax top_k times, lower index on ties), then
// those experts' four planes asked of L2 in 32 KB pieces over the block.
__global__ void __launch_bounds__(256) k_moe_prefetch(const float* scores, int E, int top_k, PrefetchPlane p0,
                                                      PrefetchPlane p1, PrefetchPlane p2, PrefetchPlane p3) {
  __shared__ float s_v[8];
  __shared__ int s_i[8], s_pick[kRouteTop];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float v = tid < E ? fmaxf(scores[tid], -FLT_MAX) : -INFINITY;  // keep picked/padding below every candidate
  for (int j = 0; j < top_k; ++j) {
    float bv = v;
    int bi = tid;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) bv = ov, bi = oi;
    }
    if (lane == 0) s_v[warp] = bv, s_i[warp] = bi;
    __syncthreads();
    if (tid == 0) {
      float b = s_v[0];
      int i = s_i[0];
      for (int w = 1; w < 8; ++w)
        if (s_v[w] > b || (s_v[w] == b && s_i[w] < i)) b = s_v[w], i = s_i[w];
      s_pick[j] = i;
    }
    __syncthreads();
    if (tid == s_pick[j]) v = -INFINITY;
  }
  constexpr size_t kPiece = 32768;
  const PrefetchPlane planes[4] = {p0, p1, p2, p3};
  for (int j = 0; j < top_k; ++j) {
    const int e = s_pick[j];
    if (e < 0 || e >= E) continue;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const PrefetchPlane& pl = planes[q];
      if (!pl.base) continue;
      const uint8_t* a = pl.base + size_t(e) * pl.per_expert;
      const size_t pieces = (pl.per_expert + kPiece - 1) / kPiece;
      for (size_t c = tid; c < pieces; c += blockDim.x) {
        const size_t off = c * kPiece;
        const uint32_t n = uint32_t(min(kPiece, pl.per_expert - off)) & ~15u;
        if (n && ((reinterpret_cast<uintptr_t>(a + off) & 15u) == 0)) l2_prefetch_bulk(a + off, n);
      }
    }
  }
}

// Many blocks: block (j, q) asks L2 for plane q of the expert whose score ranks j-th (higher
// first, lower index on ties), a 16 KB piece a thread, so the requests go out at once.
__global__ void __launch_bounds__(256) k_moe_prefetch_wide(const float* scores, int E, PrefetchPlane p0,
                                                           PrefetchPlane p1, PrefetchPlane p2, PrefetchPlane p3) {
  __shared__ float s_sc[256];
  __shared__ int s_e;
  const int tid = threadIdx.x, j = blockIdx.x;
  if (tid == 0) s_e = -1;
  if (tid < E) s_sc[tid] = fmaxf(scores[tid], -FLT_MAX);  // NaNs tie at the bottom, ordered by expert index
  __syncthreads();
  if (tid < E) {
    const float v = s_sc[tid];
    int rank = 0;
    for (int i = 0; i < E; ++i) rank += s_sc[i] > v || (s_sc[i] == v && i < tid);
    if (rank == j) s_e = tid;
  }
  __syncthreads();
  const int e = s_e;
  const PrefetchPlane pl = blockIdx.y == 0 ? p0 : blockIdx.y == 1 ? p1 : blockIdx.y == 2 ? p2 : p3;
  if (e < 0 || !pl.base) return;
  constexpr size_t kPiece = 16384;
  const uint8_t* a = pl.base + size_t(e) * pl.per_expert;
  const size_t pieces = (pl.per_expert + kPiece - 1) / kPiece;
  for (size_t c = tid; c < pieces; c += blockDim.x) {
    const size_t off = c * kPiece;
    const uint32_t n = uint32_t(min(kPiece, pl.per_expert - off)) & ~15u;
    if (n && ((reinterpret_cast<uintptr_t>(a + off) & 15u) == 0)) l2_prefetch_bulk(a + off, n);
  }
}

}  // namespace

const MoeDp4aConfig kMoeDp4aConfigs[] = {
#define X(i, C, U) {C, U},
    FLYX_MOE_DP4A(X)
#undef X
};
const int kMoeDp4aConfigCount = FLYX_MOE_DP4A_COUNT;

const char* moe_dp4a_name(int cfg) {
  static constexpr const char* kNames[] = {
#define X(i, C, U) C == 0 ? "original" : #C "x" #U,
      FLYX_MOE_DP4A(X)
#undef X
  };
  return cfg >= 0 && cfg < FLYX_MOE_DP4A_COUNT ? kNames[cfg] : "default";
}

bool moe_dp4a_compiled(int dtype, int cfg, bool fused_gate_up, int rows) {
#if FLYX_TUNE_CANDIDATES
  return moe_dp4a_cfg_compiled(dtype, cfg, fused_gate_up, rows, 0);
#else
  if (cfg == tables::kMoeDp4aOriginal) return true;
  const int arch = compiled_arch();
  return arch > 0 && moe_dp4a_cfg_compiled(dtype, cfg, fused_gate_up, rows, arch);
#endif
}

int moe_gate_up_dp4a(int tuning, int dtype, int rows) {
  const int s = tables::rules_sku(tuning, tables::kFamilyMoeGateUp);
  for (const tables::MoeGateUpRule& r : tables::kMoeGateUp)
    if (r.sku == s && (r.dtype == dtype || r.dtype < 0) && rows >= r.lo && rows <= r.hi) return r.cfg;
  // gen_gemm_tables.py: every list of these rules ends with one for the rest
  throw std::runtime_error("moe_gate_up: no rule for " + std::to_string(rows) + " rows (variants)");
}

void moe_guess_scores(const float* x, const float* norm_w, float norm_offset, const bf16* router_b16,
                      const float* router_f32, const float* bias, bool gating_sigmoid, int E, int H, float* scores,
                      cudaStream_t st) {
  if (!sm90_or_newer() || E <= 0 || E > 256 || H <= 0 || !x || !scores || (!router_b16 && !router_f32)) return;
  const int grid = (E + kGuessWarps - 1) / kGuessWarps;
  if (router_b16)
    k_moe_guess<bf16><<<grid, 32 * kGuessWarps, 0, st>>>(x, norm_w, norm_offset, router_b16, bias, int(gating_sigmoid),
                                                          E, H, scores);
  else
    k_moe_guess<float><<<grid, 32 * kGuessWarps, 0, st>>>(x, norm_w, norm_offset, router_f32, bias,
                                                           int(gating_sigmoid), E, H, scores);
}

void moe_prefetch(const float* scores, int E, int top_k, int H, int I, const ExpertMat& gate_up, const ExpertMat& down,
                  int planes, int guess_k, cudaStream_t st) {
  if (!sm90_or_newer() || !planes || !scores || E < 1 || E > 256 || top_k < 1 || top_k > std::min(E, kRouteTop) ||
      gate_up.packed() || down.packed()) return;
  const size_t gu_n = size_t(gate_up.N), dn_n = size_t(down.N);
  if (planes < 0) {
    k_moe_prefetch<<<1, 256, 0, st>>>(scores, E, top_k, PrefetchPlane{gate_up.q, gu_n * size_t(H) / 2},
                                      PrefetchPlane{gate_up.s, gu_n * size_t(H) / 16},
                                      PrefetchPlane{down.q, dn_n * size_t(I) / 2},
                                      PrefetchPlane{down.s, dn_n * size_t(I) / 16});
    return;
  }
  const PrefetchPlane none{nullptr, 0};
  // The best-scored guesses only (option guess_k, default 4): the rest are wrong more often than the bandwidth pays.
  const int k = guess_k > 0 ? std::min(guess_k, top_k) : top_k;
  k_moe_prefetch_wide<<<dim3(k, 4), 256, 0, st>>>(
      scores, E, (planes & 1) ? PrefetchPlane{gate_up.q, gu_n * size_t(H) / 2} : none,
      (planes & 2) ? PrefetchPlane{gate_up.s, gu_n * size_t(H) / 16} : none,
      (planes & 4) ? PrefetchPlane{down.q, dn_n * size_t(I) / 2} : none,
      (planes & 8) ? PrefetchPlane{down.s, dn_n * size_t(I) / 16} : none);
}

void moe_stage_q8(const float* x, int ldx, int K, int rows, const MoeScratch& sc, bool mins, cudaStream_t st) {
  if (mins) k_moe_stage_q8<true><<<rows, 256, 0, st>>>(x, ldx, K, sc.q8_a, sc.q8_d, sc.q8_s);
  else k_moe_stage_q8<false><<<rows, 256, 0, st>>>(x, ldx, K, sc.q8_a, sc.q8_d, sc.q8_s);
}

void l2_warm(const L2Ranges& r, cudaStream_t st) {
  if (!sm90_or_newer()) return;
  size_t pieces = 0;
  for (int i = 0; i < r.count; ++i) pieces += (r.n[i] + 16383) / 16384;
  if (!pieces) return;
  k_l2_warm<<<unsigned((pieces + 127) / 128), 128, 0, st>>>(r);
}

void moe_route(const float* logits, const float* bias, int rows, int n_experts, int top_k, bool gating_sigmoid,
               bool norm_weights, float norm_eps, float route_scale, int* idx_out, float* w_out, const Launch& L) {
  if (rows <= 0 || n_experts <= 0 || n_experts > kMoeRouteMaxExperts || top_k <= 0 || top_k > n_experts ||
      top_k > kRouteTop)
    throw std::runtime_error("moe_route: invalid dimensions");
  const size_t smem = size_t(2) * size_t(n_experts) * sizeof(float);
  launch_dependent_smem(k_moe_route, dim3(rows), dim3(kRouteThreads), smem, L, logits, bias, n_experts, top_k,
                        int(gating_sigmoid), int(norm_weights), norm_eps, route_scale, idx_out, w_out);
}

void situ_rows(const float* gu, int ld_gu, int I, float beta, float linear_beta, float* out, int ld_out, int rows,
               cudaStream_t st) {
  if (rows <= 0 || I <= 0) return;
  for (int at = 0; at < rows;) {
    const int count = std::min(rows - at, kMaxGridY);
    k_situ_rows<<<dim3(ceil_div(I, 256), count), 256, 0, st>>>(gu + size_t(at) * ld_gu, ld_gu, I, beta, linear_beta,
                                                               out + size_t(at) * ld_out, ld_out);
    at += count;
  }
}

// counts[e] += n, saturating at UINT_MAX (the same result as n saturating increments).
__device__ __forceinline__ void saturating_add(uint32_t* count, uint32_t n) {
  unsigned old = atomicAdd(count, 0u);
  while (old != UINT_MAX) {
    const unsigned next = old > UINT_MAX - n ? UINT_MAX : old + n;
    const unsigned prior = atomicCAS(count, old, next);
    if (prior == old) break;
    old = prior;
  }
}

// A prompt pass's block counts its pairs' experts in shared memory and adds each expert's count once: its thousands
// of pairs on a few hot experts no longer retry compare-and-swap on the same global words (0.22 ms a layer visit of
// 1,024 rows on Inkling-2, 5070 Ti).  A round's few pairs (more than kPlaceSharedPerExpert an expert on average: not)
// keep the direct increments, without the block's clearing and barriers.  The counts are the same either way.
constexpr int kPlaceThreads = 256, kPlacePairs = 2048, kPlaceSharedExperts = 4096, kPlaceSharedPerExpert = 4;
__global__ void __launch_bounds__(kPlaceThreads)
    k_moe_place_ids(int* ids, int pairs, int experts, const int* slots, uint32_t* counts, int* invalid,
                    ResidentRoutes resident, bool local) {
  extern __shared__ uint32_t s_counts[];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (local) {
    for (int e = threadIdx.x; e < experts; e += blockDim.x) s_counts[e] = 0;
    __syncthreads();
  }
  // All producers of a ticketed packet belong to one CTA.
  for (size_t j = size_t(i); j < size_t(resident.width); j += size_t(blockDim.x) * gridDim.x)
    resident.host_input[j] = resident.input[j];
  for (size_t slot = i; slot < size_t(pairs); slot += size_t(blockDim.x)*gridDim.x) {
    const int id = ids[slot];
    if (unsigned(id) >= unsigned(experts)) {
      // A streamed pass still writes this pair via slot zero. Decode checks the sentinel during its handoff.
      if (invalid) atomicExch(invalid, 1);
      ids[slot] = invalid ? 0 : -1;
      if (resident.ids) resident.ids[slot] = 0;
      if (resident.host_ids) resident.host_ids[slot] = -1;
    } else {
      if (local) atomicAdd(s_counts + id, 1u);
      else if (counts) saturating_add(counts + id, 1u);
      const int physical = slots ? slots[id] : id;
      ids[slot] = physical;
      if (resident.host_ids) resident.host_ids[slot] = physical;
      if (resident.ids) resident.ids[slot] = unsigned(physical) < unsigned(resident.experts) ? physical : 0;
    }
  }
  if (local) {
    __syncthreads();
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
      if (const uint32_t n = s_counts[e]) saturating_add(counts + e, n);
  }
  // Each producer participates before the one system-scope release. One CTA owns a ticketed packet.
  if (resident.ticket) {
    __syncthreads();
    if (threadIdx.x == 0)
      ::cuda::atomic_ref<uint32_t, ::cuda::thread_scope_system>(*resident.ticket)
          .store(resident.generation, ::cuda::memory_order_release);
  }
}

void moe_place_ids(int* ids, int pairs, int experts, const int* slots, uint32_t* counts, int* invalid, cudaStream_t st,
                   ResidentRoutes resident) {
  const int blocks = resident.ticket ? 1 : std::max(1, ceil_div(pairs, kPlacePairs));
  const bool local = counts && experts <= kPlaceSharedExperts && pairs > kPlaceSharedPerExpert * experts;
  const size_t smem = local ? size_t(experts) * sizeof(uint32_t) : 0;
  k_moe_place_ids<<<blocks, kPlaceThreads, smem, st>>>(ids, pairs, experts, slots, counts, invalid, resident, local);
}

__global__ void k_moe_resident_ids(const int* ids, int* safe, int pairs, int hot) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pairs) safe[i] = unsigned(ids[i]) < unsigned(hot) ? ids[i] : 0;
}

void moe_resident_ids(const int* ids, int* safe, int pairs, int hot, cudaStream_t st) {
  k_moe_resident_ids<<<ceil_div(pairs, 128), 128, 0, st>>>(ids, safe, pairs, hot);
}

// Cold output is published by the host before launch. Each slot retains its original FMA position.
__global__ void k_moe_sum_mapped(const float* slots, const float* cold, const int* ids, int resident,
                                 const float* weights, int top_k, int hidden, float* out, bool accumulate) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= hidden) return;
  const size_t row = blockIdx.y;
  float total = 0.f;
  for (int k = 0; k < top_k; ++k) {
    const size_t pair = row * top_k + k;
    const float* source = ids[pair] >= resident ? cold : slots;
    total = __fmaf_rn(weights[pair], source[pair * hidden + n], total);
  }
  float& result = out[row * hidden + n];
  result = accumulate ? result + total : total;
}

// This block's share of expert `id`'s planes, to copy `slot` of the planes' targets.
__device__ __forceinline__ void copy_expert_planes(const GatherPlanes& planes, int id, int slot) {
  int block = blockIdx.x, which = 0;
  while (block >= planes.plane[which].blocks) block -= planes.plane[which++].blocks;
  const auto plane = planes.plane[which];
  const auto* src = plane.source + size_t(id) * plane.stride;
  auto* dst = plane.target + size_t(slot) * plane.stride;
  size_t i = size_t(block) * blockDim.x + threadIdx.x;
  const size_t step = size_t(plane.blocks) * blockDim.x;
  if (!(plane.stride % 16)) {
    for (; i < plane.stride / 16; i += step) reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
  } else {
    for (; i < plane.stride; i += step) dst[i] = src[i];
  }
}

__global__ void k_gather_expert_planes(GatherPlanes planes, const int* ids, int* compact) {
  const int pair = blockIdx.y, id = ids[pair];
  int slot = 0;
  while (ids[slot] != id) ++slot;
  if (!blockIdx.x && !threadIdx.x) compact[pair] = slot;
  if (slot != pair) return;
  copy_expert_planes(planes, id, pair);
}

void gather_expert_planes(GatherPlanes planes, const int* ids, int* compact, int pairs, cudaStream_t st) {
  k_gather_expert_planes<<<dim3(planes.blocks, pairs), 128, 0, st>>>(planes, ids, compact);
}

// The pairs' ids in shared memory once; each pair finds its expert's first pair there, not through global memory.
__global__ void __launch_bounds__(256) k_gather_plan(const int* ids, int pairs, const uint8_t* resident, int split,
                                                     int copies, int* out_ids, int* copy_slot) {
  __shared__ int s_ids[kGatherPlanPairs], s_rank[kGatherPlanPairs];
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) s_ids[p] = ids[p];
  __syncthreads();
  auto first_of = [&](int p) {
    int first = 0;
    while (s_ids[first] != s_ids[p]) ++first;
    return first;
  };
  // An id outside the bank (a router's invalid one) stays as it is, as a resident expert's.
  for (int p = threadIdx.x; p < pairs; p += blockDim.x)
    s_rank[p] = first_of(p) == p && unsigned(s_ids[p]) < unsigned(split) && !resident[s_ids[p]];
  __syncthreads();
  if (threadIdx.x == 0)
    for (int p = 0, rank = 0; p < pairs; ++p) {
      const int takes = s_rank[p];
      s_rank[p] = takes ? rank : -1;
      rank += takes;
    }
  __syncthreads();
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
    const int first = first_of(p), rank = s_rank[first];
    const bool copied = rank >= 0 && rank < copies;
    out_ids[p] = copied ? split + rank : s_ids[p];
    copy_slot[p] = copied && first == p ? rank : -1;
  }
}

void gather_plan(const int* ids, int pairs, const uint8_t* resident, int split, int copies, int* out_ids,
                 int* copy_slot, cudaStream_t st) {
  if (pairs < 1 || pairs > kGatherPlanPairs || !resident) throw std::logic_error("gather_plan: invalid pairs");
  k_gather_plan<<<1, 256, 0, st>>>(ids, pairs, resident, split, copies, out_ids, copy_slot);
}

__global__ void k_gather_copy(GatherPlanes planes, const int* ids, const int* copy_slot) {
  const int pair = blockIdx.y, slot = copy_slot[pair];
  if (slot >= 0) copy_expert_planes(planes, ids[pair], slot);
}

void gather_copy(GatherPlanes planes, const int* ids, const int* copy_slot, int pairs, cudaStream_t st) {
  if (planes.blocks) k_gather_copy<<<dim3(planes.blocks, pairs), 128, 0, st>>>(planes, ids, copy_slot);
}

void moe_sum_mapped(const float* slots, const float* cold, const int* ids, int resident, const float* weights,
                    int rows, int top_k, int hidden, float* out, bool accumulate, cudaStream_t st) {
  for (int at = 0; at < rows;) {
    const int count = std::min(rows-at, kMaxGridY);
    const size_t first = size_t(at)*top_k;
    k_moe_sum_mapped<<<dim3(ceil_div(hidden, 256), count), 256, 0, st>>>(
        slots + first*hidden, cold + first*hidden, ids + first, resident, weights + first, top_k, hidden,
        out + size_t(at)*hidden, accumulate);
    at += count;
  }
}

void moe_sum_slots(const float* slots, const float* weights, int rows, int top_k, int hidden, float* out,
                   bool accumulate, cudaStream_t st) {
  for (int at = 0; at < rows;) {
    const int count = std::min(rows - at, kMaxGridY);
    const size_t first = size_t(at) * top_k;
    k_moe_weighted_sum<<<dim3(ceil_div(hidden, 256), count), 256, 0, st>>>(
        slots + first * hidden, weights + first, top_k, hidden, out + size_t(at) * hidden, hidden, int(accumulate));
    at += count;
  }
}

bool moe_q8_mins(const ExpertMat& W) {
  const MoeBlockOps* ops = W.packed() ? moe_block_ops(W.blk.dtype) : nullptr;
  return ops && ops->mins;
}

bool moe_round_q8(const ExpertMat& W, int rows, int top_k, int K, const MoeScratch* sc) {
  if (!W.packed() || !W.blk.q8 || K <= 0 || K % 32 != 0 || !moe_round_scratch(W, rows, top_k, sc)) return false;
  // A format with minima: whole warps of four groups only (K % 128), where k_moe_stage_q8's half-group sums, which
  // rmsnorm_moe_q8 has to reproduce, have every lane of the warp behind them.
  const bool sums_ok = !moe_q8_mins(W) || (sc->q8_s && K % 128 == 0);
  return sums_ok && sc->q8_a && sc->q8_d && sc->stage_elems >= size_t(rows) * K;
}

bool moe_round_bf16_down(const ExpertMat& W, int rows, int top_k, int K, const MoeScratch* sc, const Launch& L) {
  return !opts(L).moe_vec && !W.packed() && W.q && K > 0 && K % 64 == 0 && moe_round_scratch(W, rows, top_k, sc) &&
         sc->down_buf && sc->bf16_a && sc->stage_elems >= size_t(rows) * top_k * K;
}

// Bank calls operate on one grouped tile. Direct kernels index the complete expert table.
void check_bank(BankRange bank, const ExpertMat& weights, int rows, int k, const MoeScratch* sc, const Launch& L) {
  if (bank.first < 0 && bank.count == 0) return;
  if (bank.first < 0 || bank.count != weights.n_experts || rows < 2 || !sc || !sc->tasks || !sc->pairs ||
      !sc->n_tasks || rows > sc->chunk_rows || k > sc->max_top_k || bank.count > sc->max_experts ||
      (weights.q && opts(L).moe_vec && rows > FLYX_MAX_ROWS))
    throw std::runtime_error("MoE bank requires grouped projection scratch for the entire tile");
}

int moe_bank_gate_up(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                     float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out,
                     const MoeScratch* sc, const Launch& L, bool q8_ready, bf16* act_bf16, BankRange bank) {
  check_bank(bank, W, rows, top_k, sc, L);
  const int first = bank.first;
  cudaStream_t st = L.st;
  const int N_work = fuse_act ? I : 2 * I;
  if ((q8_ready || act_bf16) && !moe_round_q8(W, rows, top_k, K, sc))
    throw std::runtime_error("moe_gate_up: pre-staged input and BF16 output require a staged Q8 round");
  if (act_bf16 && (!fuse_act || ld_out != I))
    throw std::runtime_error("moe_gate_up: BF16 output requires contiguous fused activations");
  if (W.packed()) {
    blocks_init();
    const MoeBlockOps* ops = moe_block_ops(W.blk.dtype);
    if (!ops) throw std::runtime_error("moe_gate_up: unsupported GGML block dtype " + std::to_string(W.blk.dtype));
    const bool has_sc = sc && sc->tasks && sc->pairs && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
    const int chunk = has_sc ? std::max(1, sc->chunk_rows) : rows;
    int grouped = 0;
    for (int r0 = 0; r0 < rows; r0 += chunk) {
      const int rc = std::min(chunk, rows - r0);
      const float* xc = x + size_t(r0) * ldx;
      const int* ec = eidx + size_t(r0) * top_k;
      float* oc = act_bf16 ? nullptr : out + size_t(r0) * top_k * ld_out;
      const bool can_q8 = W.blk.q8 && sc && sc->q8_a && sc->stage_elems >= size_t(rc) * size_t(K);
      if (can_q8 && !q8_ready) moe_stage_q8(xc, ldx, K, rc, *sc, ops->mins, st);
      const int M = rc * top_k;
      const int mma = can_q8 && has_sc ? moe_mma_rows(opts(L), rc, M, W.n_experts) : 0;
      int mt = 0;
      if (rc > 1 && has_sc) {
        const int bm = mma ? mma : 4;
        mt = max_work_tiles(M, W.n_experts, bm);
        group_tasks(ec, M, W.n_experts, bm, sc, st, first);
        grouped = rc == rows ? bm : 0;
      }
      ops->gu(xc, ldx, W, ec, rc, top_k, I, K, situ_beta, situ_linear_beta, fuse_act, oc, ld_out, sc, mt, mma,
              moe_dp4a_cfg(opts(L), W.blk.dtype, rc, fuse_act), st, act_bf16);
    }
    return grouped;
  }
  if (W.q) {
    if (K % 128 == 0)
      return launch_nvfp4_gu<128>(x, ldx, W, eidx, rows, top_k, I, K, situ_beta, situ_linear_beta, fuse_act, out,
                                  ld_out, sc, L, first);
    if (K % 64 == 0)
      return launch_nvfp4_gu<64>(x, ldx, W, eidx, rows, top_k, I, K, situ_beta, situ_linear_beta, fuse_act, out,
                                 ld_out, sc, L, first);
    throw std::runtime_error("moe_gate_up: NVFP4 requires K multiple of 64 (got " + std::to_string(K) + ")");
  }
  const bool tiled = rows > 1 && sc && sc->tasks && sc->pairs && W.n_experts <= sc->max_experts &&
                     top_k <= sc->max_top_k;
  if (!tiled) {
    const dim3 grid(ceil_div(N_work, kMoeWarps), top_k, rows);
    if (fuse_act)
      k_moe_bf16_gu_decode<true><<<grid, 32 * kMoeWarps, 0, st>>>(
          x, ldx, W.w, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    else
      k_moe_bf16_gu_decode<false><<<grid, 32 * kMoeWarps, 0, st>>>(
          x, ldx, W.w, W.rs, W.N, I, K, eidx, top_k, situ_beta, situ_linear_beta, out, ld_out);
    return 0;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    const int M = rc * top_k;
    const int mt = max_work_tiles(M, W.n_experts, 8);
    group_tasks(eidx + size_t(r0) * top_k, M, W.n_experts, 8, sc, st, first);
    const dim3 grid(ceil_div(N_work, kMoeWarps), mt);
    const float* xc = x + size_t(r0) * ldx;
    float* oc = out + size_t(r0) * top_k * ld_out;
    if (fuse_act)
      k_moe_bf16_grouped<true, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          xc, ldx, W.w, W.rs, W.N, I, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, situ_beta, situ_linear_beta, oc,
          ld_out);
    else
      k_moe_bf16_grouped<false, false><<<grid, 32 * kMoeWarps, 0, st>>>(
          xc, ldx, W.w, W.rs, W.N, I, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, situ_beta, situ_linear_beta, oc,
          ld_out);
  }
  return rows <= chunk ? 8 : 0;
}

void moe_bank_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows,
                   int top_k, int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc,
                   const Launch& L, const bf16* act_bf16, int grouped, BankDispatch dispatch) {
  check_bank(dispatch.range, W, rows, top_k, sc, L);
  const int first = dispatch.range.first;
  if ((dispatch.defer_reduce || first >= 0) && (!sc || !sc->down_buf))
    throw std::runtime_error("MoE bank needs reduction slots");
  cudaStream_t st = L.st;
  if (act_bf16 && !moe_round_bf16_down(W, rows, top_k, K, sc, L))
    throw std::runtime_error("moe_down: pre-staged BF16 input requires an NVFP4 round");
  if (W.packed()) {
    blocks_init();
    const MoeBlockOps* ops = moe_block_ops(W.blk.dtype);
    if (!ops) throw std::runtime_error("moe_down: unsupported GGML block dtype " + std::to_string(W.blk.dtype));
    const bool has_sc =
        sc && sc->tasks && sc->pairs && sc->down_buf && W.n_experts <= sc->max_experts && top_k <= sc->max_top_k;
    const int chunk = has_sc ? std::max(1, sc->chunk_rows) : rows;
    for (int r0 = 0; r0 < rows; r0 += chunk) {
      const int rc = std::min(chunk, rows - r0);
      const float* ac = act + size_t(r0) * top_k * ld_act;
      const int* ec = eidx + size_t(r0) * top_k;
      const float* ewc = ew + size_t(r0) * top_k;
      float* oc = out + size_t(r0) * ld_out;
      const int M = rc * top_k;
      const bool can_q8 = W.blk.q8 && sc && sc->q8_a && sc->down_buf && sc->stage_elems >= size_t(M) * size_t(K);
      if (can_q8) moe_stage_q8(ac, ld_act, K, M, *sc, ops->mins, st);
      const int mma = can_q8 && has_sc ? moe_mma_rows(opts(L), rc, M, W.n_experts) : 0;
      int mt = 0;
      if (rc > 1 && has_sc) {
        const int bm = mma ? mma : 4;
        mt = max_work_tiles(M, W.n_experts, bm);
        if (rc != rows || !tiles_serve(grouped, bm, rc)) group_tasks(ec, M, W.n_experts, bm, sc, st, first);
      }
      const bool wrote_buf = ops->down(ac, ld_act, W, ec, ewc, rc, top_k, H, K, oc, ld_out, accumulate, sc, mt,
                                       mma, moe_dp4a_cfg(opts(L), W.blk.dtype, rc, false), st);
      if (wrote_buf && !dispatch.defer_reduce)
        k_moe_weighted_sum<<<dim3(ceil_div(H, 256), rc), 256, 0, st>>>(sc->down_buf, ewc, top_k, H, oc, ld_out,
                                                                       int(accumulate));
    }
    return;
  }
  if (W.q) {
    if (K % 128 == 0)
      launch_nvfp4_down<128>(act, ld_act, W, eidx, ew, rows, top_k, H, K, out, ld_out, accumulate, sc, L, act_bf16,
                             grouped, first, dispatch.defer_reduce);
    else if (K % 64 == 0)
      launch_nvfp4_down<64>(act, ld_act, W, eidx, ew, rows, top_k, H, K, out, ld_out, accumulate, sc, L, act_bf16,
                            grouped, first, dispatch.defer_reduce);
    else
      throw std::runtime_error("moe_down: NVFP4 requires K multiple of 64 (got " + std::to_string(K) + ")");
    return;
  }
  const bool tiled = rows > 1 && sc && sc->tasks && sc->pairs && sc->down_buf && W.n_experts <= sc->max_experts &&
                     top_k <= sc->max_top_k;
  if (!tiled) {
    k_moe_bf16_down_decode<<<dim3(ceil_div(H, kMoeWarps), rows), 32 * kMoeWarps, 0, st>>>(
        act, ld_act, W.w, W.rs, H, K, eidx, ew, top_k, out, ld_out, int(accumulate));
    return;
  }
  const int chunk = std::max(1, sc->chunk_rows);
  for (int r0 = 0; r0 < rows; r0 += chunk) {
    const int rc = std::min(chunk, rows - r0);
    const int M = rc * top_k;
    const int mt = max_work_tiles(M, W.n_experts, 8);
    if (rc != rows || !tiles_serve(grouped, 8, rc)) group_tasks(eidx + size_t(r0) * top_k, M, W.n_experts, 8, sc, st, first);
    const dim3 grid(ceil_div(H, kMoeWarps), mt);
    k_moe_bf16_grouped<false, true><<<grid, 32 * kMoeWarps, 0, st>>>(
        act + size_t(r0) * top_k * ld_act, ld_act, W.w, W.rs, H, H, K, sc->tasks, sc->n_tasks, sc->pairs, top_k, 0.f,
        0.f, sc->down_buf, H);
    if (!dispatch.defer_reduce)
      k_moe_weighted_sum<<<dim3(ceil_div(H, 256), rc), 256, 0, st>>>(
          sc->down_buf, ew + size_t(r0) * top_k, top_k, H, out + size_t(r0) * ld_out, ld_out, int(accumulate));
  }
}

int moe_gate_up(const float* x, int ldx, const ExpertMat& W, const int* eidx, int rows, int top_k, int I, int K,
                float situ_beta, float situ_linear_beta, bool fuse_act, float* out, int ld_out, const MoeScratch* sc,
                const Launch& L, bool q8_ready, bf16* act_bf16) {
  return moe_bank_gate_up(x, ldx, W, eidx, rows, top_k, I, K, situ_beta, situ_linear_beta, fuse_act, out, ld_out, sc, L,
                          q8_ready, act_bf16, {});
}

void moe_down(const float* act, int ld_act, const ExpertMat& W, const int* eidx, const float* ew, int rows, int top_k,
              int H, int K, float* out, int ld_out, bool accumulate, const MoeScratch* sc, const Launch& L,
              const bf16* act_bf16, int grouped) {
  moe_bank_down(act, ld_act, W, eidx, ew, rows, top_k, H, K, out, ld_out, accumulate, sc, L, act_bf16, grouped, {});
}

}  // namespace flyx::cuda
