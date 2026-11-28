// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// MXFP4 experts with 8-bit inputs (FLYX_ACT_Q8, llama.cpp's) in prefill passes (more than FLYX_MAX_ROWS rows): the
// rows as k_moe_stage_q8 stages them (an int8 a value, a float32 scale a block of 32), the experts' GGML values (twice
// the e2m1 codes, int8: mxfp4_s8) and their blocks' scales; mma.sync m16n8k32 s8 gives a unit's exact integer sum, and
// a float32 sum an output takes it times the two scales, unit after unit in K's order (fmaf(sum, ws * xs, acc)).  So
// every tile shape gives the same bits (a pass's rows do not depend on its length: numerics.md K) and the shape follows
// the rows an expert has (moe_mxfp4_q8_pass_rows).  Rounds take moe_blocks_q8's kernels (their own bits; these within
// float32 sums of them, as the dense prefill GEMMs and the 1..16-row ones).
//
// Swap-AB, as moe_mxfp4_mxfp8_pass: A the weight rows (fused gate|up: an m16 tile 8 columns' gate rows then their up
// rows, the activation's pair in one thread), B the task's rows, C[weight row][row].  A block: 16 MT WMW weight rows x
// 8 NT WT rows of one expert; warps WMW x WT, a warp MT m16 x NT n8 tiles; spans of SU units, the rows, their scales,
// the weights and theirs staged a span ahead (cp.async), NS stages.
//
// A unit's 16 bytes: byte b's low nibble value b, its high nibble value b + 16.  The s8 MMA's A registers of rows g and
// g + 8 are values 4 t4 .. 4 t4 + 3 and 16 + 4 t4 .. 16 + 4 t4 + 3: the low and the high nibbles of bytes 4 t4 .. 4 t4
// + 3, one 4-byte load a row; its B registers of row g the same values of the row's 32 bytes.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_types.cuh"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_act.cuh"

namespace flyx::cuda {
namespace {

template <int WROWS, int TOK, int NS, int SU, bool W8>
struct Q8PassSmem {
  struct Stage {
    alignas(16) int8_t x[TOK][SU * 32 + 16];     // rows: 4 banks apart
    alignas(16) float xs[TOK][SU];               // their blocks' scales
    alignas(16) uint8_t w[WROWS][SU * 16 + 16];  // codes
    alignas(16) uint16_t ws[WROWS][SU + 2];      // a weight row's span of scales (bf16 bits)
  };
  Stage st[NS];
  alignas(16) int8_t w8[W8 ? WROWS : 1][SU * 32 + 16];  // W8: the span's weights as int8, decoded once for every warp
  int pids[TOK];
  int rids[TOK];
};

template <bool FUSE_ACT, bool IS_DOWN, int MT, int NT, int WMW, int WT, int NS, int SU, bool W8>
__global__ void __launch_bounds__(32 * WMW * WT)
    k_moe_mxfp4_q8_pass(const int8_t* xa, const float* xsc, const uint8_t* __restrict__ W,
                        const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                        const MoeTask* tasks, const int* n_tasks, const int* pairs, int top_k, MoeAct mact,
                        float* __restrict__ out, int ld_out) {
  constexpr int WROWS = 16 * MT * WMW, TOK = 8 * NT * WT, NTH = 32 * WMW * WT;
  constexpr int COLS = FUSE_ACT ? WROWS / 2 : WROWS;  // output columns a block
  using S_t = Q8PassSmem<WROWS, TOK, NS, SU, W8>;
  pdl_trigger();
  pdl_wait();  // the staged rows and the tasks: the kernels before
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S_t*>(smem_raw);
  const int tid = threadIdx.x;
  const int units = K >> 5, spans = (units + SU - 1) / SU;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, ww = warp / WT, wt = warp % WT;
  const int g = lane >> 2, t4 = lane & 3;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = int(blockIdx.x) * COLS;
  const float rs = RS ? RS[e] : 1.f;
  const int rows_st = min(TOK, (nb + 7) & ~7);
  // smem weight row s -> the expert's row
  auto wrow = [&](int s) {
    if constexpr (FUSE_ACT) {
      const int col = min(c0 + 8 * (s >> 4) + (s & 7), N_work - 1);
      return (s & 8) ? I_or_H + col : col;
    } else {
      return min(c0 + s, N_exp - 1);
    }
  };
  auto stage = [&](int j) {
    if (j >= spans) {
      cp_async_commit();
      return;
    }
    auto& S = sm.st[j % NS];
    const int n_s = min(SU, units - SU * j);
    for (int i = tid; i < rows_st * (SU * 2); i += NTH) {  // 16 bytes a thread
      const int r = i / (SU * 2), c = i % (SU * 2);
      const bool ok = r < nb && c < 2 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * (SU * 32) + 16 * c) : 0;
      cp_async16(&S.x[r][16 * c], xa + src, ok);
    }
    for (int i = tid; i < rows_st * SU; i += NTH) {  // past the task's rows or the span: zero
      const int r = i / SU, u = i % SU;
      const bool ok = r < nb && u < n_s;
      cp_async4(&S.xs[r][u], xsc + (ok ? size_t(sm.rids[r]) * units + size_t(SU) * j + u : 0), ok);
    }
    for (int i = tid; i < WROWS * SU; i += NTH) {  // a unit's 16 bytes a thread
      const int s = i / SU, u = i % SU;
      const size_t r = size_t(e) * N_exp + wrow(s);
      const bool ok = u < n_s;
      cp_async16(&S.w[s][16 * u], W + (r * size_t(units) + size_t(SU) * j + (ok ? u : 0)) * 16, ok);
    }
    if ((units & 1) == 0) {  // their scales two at a time (4 bytes, aligned where a row's units are even)
      for (int i = tid; i < WROWS * (SU / 2); i += NTH) {
        const int s = i / (SU / 2), u = 2 * (i % (SU / 2));
        const size_t r = size_t(e) * N_exp + wrow(s);
        const bool ok = u < n_s;
        cp_async4(&S.ws[s][u], D + r * size_t(units) + size_t(SU) * j + (ok ? u : 0), ok);
      }
    } else {
      for (int i = tid; i < WROWS * n_s; i += NTH) {
        const int s = i / n_s, u = i - s * n_s;
        S.ws[s][u] = D[(size_t(e) * N_exp + wrow(s)) * size_t(units) + size_t(SU) * j + u];
      }
    }
    cp_async_commit();
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
  auto mul = [&](int j) {
    const auto& S = sm.st[j % NS];
    const int n_s = min(SU, units - SU * j);
    if constexpr (W8) {  // the span's codes as GGML's int8 values: a word of codes its low nibbles' and its high ones'
      for (int i = tid; i < WROWS * SU * 4; i += NTH) {
        const int s = i / (SU * 4), u = (i / 4) % SU, q = i % 4;
        const uint32_t w = *reinterpret_cast<const uint32_t*>(&S.w[s][16 * u + 4 * q]);
        *reinterpret_cast<uint32_t*>(&sm.w8[s][32 * u + 4 * q]) = mxfp4_s8(w);
        *reinterpret_cast<uint32_t*>(&sm.w8[s][32 * u + 16 + 4 * q]) = mxfp4_s8(w >> 4);
      }
      __syncthreads();
    }
#pragma unroll
    for (int ul = 0; ul < SU; ++ul) {
      if (ul >= n_s) break;
      uint32_t a[MT][4], b[NT][2];
      float wsc[MT][2], xs[NT][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {  // weight rows 16 t + g and + 8
        const int ra = 16 * (MT * ww + mt) + g, rb = ra + 8;
        if constexpr (W8) {
          a[mt][0] = *reinterpret_cast<const uint32_t*>(&sm.w8[ra][32 * ul + 4 * t4]);
          a[mt][1] = *reinterpret_cast<const uint32_t*>(&sm.w8[rb][32 * ul + 4 * t4]);
          a[mt][2] = *reinterpret_cast<const uint32_t*>(&sm.w8[ra][32 * ul + 16 + 4 * t4]);
          a[mt][3] = *reinterpret_cast<const uint32_t*>(&sm.w8[rb][32 * ul + 16 + 4 * t4]);
        } else {
          const uint32_t qa = *reinterpret_cast<const uint32_t*>(&S.w[ra][16 * ul + 4 * t4]);
          const uint32_t qb = *reinterpret_cast<const uint32_t*>(&S.w[rb][16 * ul + 4 * t4]);
          a[mt][0] = mxfp4_s8(qa), a[mt][1] = mxfp4_s8(qb);
          a[mt][2] = mxfp4_s8(qa >> 4), a[mt][3] = mxfp4_s8(qb >> 4);
        }
        wsc[mt][0] = __uint_as_float(uint32_t(S.ws[ra][ul]) << 16);
        wsc[mt][1] = __uint_as_float(uint32_t(S.ws[rb][ul]) << 16);
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {  // row 8 t + g: its values 4 t4 .. + 3 and 16 + 4 t4 .. + 3
        const int tr = 8 * (NT * wt + nt);
        b[nt][0] = *reinterpret_cast<const uint32_t*>(&S.x[tr + g][32 * ul + 4 * t4]);
        b[nt][1] = *reinterpret_cast<const uint32_t*>(&S.x[tr + g][32 * ul + 16 + 4 * t4]);
        xs[nt][0] = S.xs[tr + 2 * t4][ul];  // C's rows 2 t4 and 2 t4 + 1 of the n8 tile
        xs[nt][1] = S.xs[tr + 2 * t4 + 1][ul];
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if (8 * (NT * wt + nt) >= nb) break;
        int c[MT][4];  // the tile's MMAs first, then their sums' scaling: no FMA waits on the MMA just issued
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma_s8_k32_zero(c[mt], a[mt], b[nt]);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int k = 0; k < 4; ++k)
            acc[mt][nt][k] = __fmaf_rn(float(c[mt][k]), wsc[mt][k >> 1] * xs[nt][k & 1], acc[mt][nt][k]);
      }
    }
  };
#pragma unroll
  for (int p = 0; p < NS - 1; ++p) stage(p);
  for (int j = 0; j < spans; ++j) {
    cp_async_wait<NS - 2>();
    __syncthreads();
    stage(j + NS - 1);
    mul(j);
  }
  // C[weight row][row]: the lane's weight rows 16 t + g (+ 8), rows 2 t4 (+ 1) of each n8 tile
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int tr = 8 * (NT * wt + nt) + 2 * t4 + (k & 1);
        if (tr >= nb) continue;
        if constexpr (FUSE_ACT) {
          if (k >= 2) continue;  // the up row's sum: k + 2
          const int col = c0 + 8 * (MT * ww + mt) + g;
          if (col >= N_work) continue;
          out[size_t(sm.pids[tr]) * ld_out + col] =
              moe_act_val(mact, acc[mt][nt][k] * rs, acc[mt][nt][k + 2] * rs, e, col, I_or_H);
        } else {
          const int col = c0 + 16 * (MT * ww + mt) + g + 8 * (k >> 1);
          if (col >= N_work) continue;
          out[size_t(sm.pids[tr]) * ld_out + col] = acc[mt][nt][k] * rs;
        }
      }
}

template <bool FUSE_ACT, bool IS_DOWN, int MT, int NT, int WMW, int WT, int NS, int SU, bool W8>
void launch_q8_pass_cfg(const int8_t* xa, const float* xs, const ExpertMat& W, int N_work, int I_or_H, int K,
                        const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out,
                        const Launch& L) {
  constexpr int WROWS = 16 * MT * WMW, TOK = 8 * NT * WT, COLS = FUSE_ACT ? WROWS / 2 : WROWS;
  using S_t = Q8PassSmem<WROWS, TOK, NS, SU, W8>;
  auto* kernel = k_moe_mxfp4_q8_pass<FUSE_ACT, IS_DOWN, MT, NT, WMW, WT, NS, SU, W8>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(S_t));
  if (!ready) throw std::runtime_error("moe_mxfp4_q8_pass: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, COLS), max_tiles), dim3(32 * WMW * WT), sizeof(S_t), L, xa, xs,
                        W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks, sc.n_tasks, sc.pairs, top_k, mact,
                        out, ld_out);
}

// The shapes by the rows an expert has: 16 rows a task, 128 weight rows (8 warps, spans of 16); 64, 128 (8 warps
// 4 x 2, spans of 8), each warp decoding its weight rows' codes; 128, 128 (4 warps 2 x 2, spans of 2), the codes
// decoded into shared memory once for both warps of a column (gpt-oss-20b's experts at 512 rows an expert, gate|up
// and down: 3,260 us against 3,875 with 256 weight rows decoded by each of four warps; moe_blocks_q8: 7,079).
template <bool FUSE_ACT, bool IS_DOWN, int TOK>
void launch_q8_pass(const int8_t* xa, const float* xs, const ExpertMat& W, int N_work, int I_or_H, int K,
                    const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out,
                    const Launch& L) {
  constexpr int MT = TOK == 16 ? 1 : TOK == 64 ? 2 : 4, NT = TOK == 16 ? 2 : TOK == 64 ? 4 : 8;
  constexpr int WMW = TOK == 16 ? 8 : TOK == 64 ? 4 : 2, WT = TOK == 16 ? 1 : 2;
  constexpr int NS = 2, SU = TOK == 16 ? 16 : TOK == 64 ? 8 : 2;
  static_assert(8 * NT * WT == TOK);
  launch_q8_pass_cfg<FUSE_ACT, IS_DOWN, MT, NT, WMW, WT, NS, SU, TOK == 128>(xa, xs, W, N_work, I_or_H, K, sc,
                                                                             max_tiles, top_k, mact, out, ld_out, L);
}

}  // namespace

int moe_mxfp4_q8_pass_rows(int pairs, int n_experts) {
  return pairs <= 24 * n_experts ? 16 : pairs <= 96 * n_experts ? 64 : 128;
}

void moe_mxfp4_q8_pass(bool fuse_act, bool is_down, int tok, const int8_t* xa, const float* xs, const ExpertMat& W,
                       int N_work, int I_or_H, int K, const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact,
                       float* out, int ld_out, const Launch& L) {
  if (tok != 16 && tok != 64 && tok != 128)
    throw std::runtime_error("moe_mxfp4_q8_pass: tasks of 16, 64 or 128 rows");
  auto go = [&](auto tc) {
    constexpr int T = decltype(tc)::value;
    if (is_down)
      launch_q8_pass<false, true, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act)
      launch_q8_pass<true, false, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch_q8_pass<false, false, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (tok == 128) go(std::integral_constant<int, 128>{});
  else if (tok == 64) go(std::integral_constant<int, 64>{});
  else go(std::integral_constant<int, 16>{});
}

}  // namespace flyx::cuda
