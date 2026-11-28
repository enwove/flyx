// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// MXFP4 experts with bf16 inputs (FLYX_ACT_BF16, gpt-oss's experts as vLLM runs them) on the bf16 tensor cores:
// mma.sync m16n8k16, A the inputs rounded once to bf16 (k_moe_stage_mxfp4_bf16), B the experts' e2m1 codes times their
// blocks' scales (exact in bf16), float32 sums.
//
// Tasks of 16 or 32 rows of one expert (moe_mxfp4_bm: 32 where the experts average more than 14 rows), column sets of
// 8 columns of 4 warps; spans of 16 units of K (a unit a GGML block of 32 values), the rows, the weights and their
// scales staged in shared memory a span ahead (cp.async), warp wi of a set its units 4 wi .. 4 wi + 3 of each span:
// one float32 sum a lane, MMA after MMA in K's order, then the four warps' sums in a fixed tree.  Every row gets the
// same MMAs in the same order whatever other rows its task holds and whichever task size: the same bits at every row
// count (rounds and passes take these kernels alike).
//
// A unit's 16 bytes: byte b's low nibble value b, its high nibble value b + 16.  Lane t4 takes bytes 4 t4 .. 4 t4 + 3
// of its column's block, eight codes whose bf16 pairs (mxfp4x8_to_bf16) are values (4 t4 + i, 4 t4 + 16 + i),
// i = 0..3: B's pairs of K for two MMAs as they come.  The staged inputs hold each block's values in
// the same pairs (position 2 c and 2 c + 1: values c and c + 16), so a lane's 16 bytes of a row are A's pairs for those
// MMAs: a dot product of the same pairs.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_act.cuh"

namespace flyx::cuda {
namespace {

constexpr int kSetWarps = 4;

// Rows of float32 values as bf16 (nearest), each block of 32 values interleaved: positions 2 c, 2 c + 1 values c,
// c + 16.  A block a row and 256 threads; a thread a pair.
__global__ void __launch_bounds__(256) k_moe_stage_mxfp4_bf16(const float* x, int ldx, int K,
                                                          bf16* __restrict__ out) {
  pdl_trigger();
  pdl_wait();  // x: the kernel before
  const int r = int(blockIdx.x);
  const float* src = x + size_t(r) * ldx;
  uint32_t* dst = reinterpret_cast<uint32_t*>(out + size_t(r) * K);
  for (int i = threadIdx.x; i < (K >> 1); i += blockDim.x) {
    const int b = i >> 4, c = i & 15;
    const __nv_bfloat162 v = __floats2bfloat162_rn(src[32 * b + c], src[32 * b + c + 16]);
    dst[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}

// Eight e2m1 codes of a word (nibble i: value i) as bf16 pairs of twice their values (GGML's MXFP4 values: 2 e2m1),
// pair p values 2 p (low half), 2 p + 1: fp4x8_to_bf16 with tables of 0, 1, 2, 3, 4, 6, 8, 12.
__device__ __forceinline__ void mxfp4x8_to_bf16(uint32_t w, uint32_t out[4]) {
  constexpr uint32_t kHi0 = 0x40403F00u, kHi1 = 0x41414040u, kLo0 = 0x40008000u, kLo1 = 0x4000C080u;
  const uint32_t m = w & 0x77777777u;  // 3-bit selectors (bit 3 of a prmt selector would sign-extend)
  const uint32_t hb0 = __byte_perm(kHi0, kHi1, m & 0xFFFFu), lb0 = __byte_perm(kLo0, kLo1, m & 0xFFFFu);
  const uint32_t hb1 = __byte_perm(kHi0, kHi1, m >> 16), lb1 = __byte_perm(kLo0, kLo1, m >> 16);
  const uint32_t E = (w << 4) & 0x80808080u;  // byte b bit 7: sign of value 2b
  const uint32_t O = w & 0x80808080u;         // byte b bit 7: sign of value 2b + 1
  out[0] = __byte_perm(lb0, hb0, 0x5140) | (__byte_perm(E, O, 0x4000) & 0x80008000u);
  out[1] = __byte_perm(lb0, hb0, 0x7362) | (__byte_perm(E, O, 0x5010) & 0x80008000u);
  out[2] = __byte_perm(lb1, hb1, 0x5140) | (__byte_perm(E, O, 0x6020) & 0x80008000u);
  out[3] = __byte_perm(lb1, hb1, 0x7362) | (__byte_perm(E, O, 0x7030) & 0x80008000u);
}

// Spans of 16 units, the rows, the weights and their scales staged in shared memory a span ahead (cp.async): warp wi
// of a set takes units 4 wi .. 4 wi + 3 of each span.  BM rows a task (16, or 32: two tiles of 16 decoding each unit
// of weights once), SETS column sets of 8 columns a block (the bits do not depend on either: a column's MMAs and tree
// are the same).
constexpr int kSpan = 16;
constexpr int kRow = kSpan * 32 + 32;   // bf16 a row: rows 16 banks apart
constexpr int kWcol = kSpan * 16 + 16;  // bytes a column: 68 words, 4 banks apart (a warp's 8 columns x 4 lanes)
constexpr int kScol = kSpan + 4;        // scales a column (u16): 10 words
template <int BM, int SETS>
struct Smem {
  struct Stage {
    alignas(16) bf16 rows[BM][kRow];
    alignas(16) uint8_t w[8 * SETS][kWcol];
    alignas(16) uint16_t s[8 * SETS][kScol];
  };
  union {
    Stage st[2];
    float tree[SETS][kSetWarps][32][BM / 4];  // after a pass's spans
  };
  int pids[BM];
  int rids[BM];
};

template <bool FUSE_ACT, bool IS_DOWN, int BM, int SETS>
__global__ void __launch_bounds__(32 * kSetWarps * SETS)
    k_moe_mxfp4_bf16_mma(const bf16* xa, const uint8_t* __restrict__ W, const uint16_t* __restrict__ D,
                     const float* __restrict__ RS, int N_exp, int I_or_H, int K, const MoeTask* tasks,
                     const int* n_tasks, const int* pairs, int top_k, MoeAct mact,
                     float* __restrict__ out, int ld_out) {
  constexpr int T = BM / 16, NT = 32 * kSetWarps * SETS, NC = 8 * SETS;
  pdl_trigger();
  pdl_wait();  // the staged rows and the tasks: the kernels before
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<Smem<BM, SETS>*>(smem_raw);
  const int tid = threadIdx.x;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, wi = warp % kSetWarps, ct = warp / kSetWarps;
  const int g = lane >> 2, t4 = lane & 3;
  const int units = K >> 5, spans = (units + kSpan - 1) / kSpan;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = int(blockIdx.x) * NC;
  const float rs = RS ? RS[e] : 1.f;
  const bool two = T == 2 && nb > 16;    // the second tile holds rows
  const int rows_st = two ? 32 : 16;     // the rows staged

  // span j of the rows and of the block's columns starting at row0 of the expert (a short last span: its units alone,
  // the rest never read)
  auto stage = [&](int row0, int j) {
    auto& S = sm.st[j & 1];
    const int n_s = min(kSpan, units - kSpan * j);
    for (int i = tid; i < rows_st * 4 * kSpan; i += NT) {  // 16 bytes a thread
      const int r = i / (4 * kSpan), c = i % (4 * kSpan);
      const bool ok = r < nb && c < 4 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * (kSpan * 32) + 8 * c) : 0;
      cp_async16(&S.rows[r][8 * c], xa + src, ok);
    }
    for (int i = tid; i < NC * kSpan; i += NT) {  // a unit's 16 bytes a thread
      const int c = i / kSpan, u = i % kSpan;
      const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
      const bool ok = u < n_s;
      cp_async16(&S.w[c][16 * u], W + (r * size_t(units) + size_t(kSpan) * j + (ok ? u : 0)) * 16, ok);
    }
    if ((units & 1) == 0) {  // their scales two at a time (4 bytes, aligned where a row's units are even)
      for (int i = tid; i < NC * (kSpan / 2); i += NT) {
        const int c = i / (kSpan / 2), u = 2 * (i % (kSpan / 2));
        const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
        const bool ok = u < n_s;
        cp_async4(&S.s[c][u], D + r * size_t(units) + size_t(kSpan) * j + (ok ? u : 0), ok);
      }
    } else {
      for (int i = tid; i < NC * n_s; i += NT) {
        const int c = i / n_s, u = i - c * n_s;
        const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
        S.s[c][u] = D[r * size_t(units) + size_t(kSpan) * j + u];
      }
    }
    cp_async_commit();
  };
  float acc[T][4];
  auto mul = [&](int j) {
    const auto& S = sm.st[j & 1];
    const int cl = ct * 8 + g, n_w = min(4, units - kSpan * j - 4 * wi);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int ul = 4 * wi + s;
      if (s >= n_w) break;
      const uint32_t q = *reinterpret_cast<const uint32_t*>(&S.w[cl][16 * ul + 4 * t4]);
      // the stored scale 0.5 2^(e - 127) on both halves: times GGML's values, 2 e2m1
      const uint32_t sb = __byte_perm(uint32_t(S.s[cl][ul]), 0u, 0x1010);
      const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sb);
      uint32_t bw[4];
      mxfp4x8_to_bf16(q, bw);
#pragma unroll
      for (int i = 0; i < 4; ++i) bw[i] = bf16x2_mul(bw[i], s2);
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t == 1 && !two) break;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&S.rows[16 * t + g][32 * ul + 8 * t4]);
        const uint4 x8 = *reinterpret_cast<const uint4*>(&S.rows[16 * t + g + 8][32 * ul + 8 * t4]);
        const uint32_t a0[4] = {x0.x, x8.x, x0.y, x8.y}, b0[2] = {bw[0], bw[1]};
        mma_bf16(acc[t], a0, b0);
        const uint32_t a1[4] = {x0.z, x8.z, x0.w, x8.w}, b1[2] = {bw[2], bw[3]};
        mma_bf16(acc[t], a1, b1);
      }
    }
  };
  auto pass = [&](int row0) {
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    stage(row0, 0);
    for (int j = 0; j < spans; ++j) {
      cp_async_wait<0>();
      __syncthreads();
      if (j + 1 < spans) stage(row0, j + 1);
      mul(j);
    }
    __syncthreads();  // the stages free: the tree overlays them
  };
  auto to_tree = [&] {
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int e4 = 0; e4 < 4; ++e4) sm.tree[ct][wi][lane][4 * t + e4] = acc[t][e4];
  };
  auto from_tree = [&](int k) {
    return ((sm.tree[ct][0][lane][k] + sm.tree[ct][1][lane][k]) + (sm.tree[ct][2][lane][k] + sm.tree[ct][3][lane][k])) *
           rs;
  };
  const int n0 = c0 + ct * 8;
  pass(0);
  to_tree();
  __syncthreads();
  float vg[4 * T];
  if (wi == 0)
#pragma unroll
    for (int k = 0; k < 4 * T; ++k) vg[k] = from_tree(k);
  if constexpr (FUSE_ACT) {
    __syncthreads();
    pass(I_or_H);
    to_tree();
    __syncthreads();
  }
  if (wi == 0) {
#pragma unroll
    for (int k = 0; k < 4 * T; ++k) {
      const int r = 16 * (k >> 2) + g + 8 * ((k & 3) >> 1), col = n0 + 2 * t4 + (k & 1);
      if (r >= nb || col >= N_work) continue;
      float v = vg[k];
      if constexpr (FUSE_ACT) v = moe_act_val(mact, v, from_tree(k), e, col, I_or_H);
      out[size_t(sm.pids[r]) * ld_out + col] = v;
    }
  }
}

// 16 rows a task: 8 column sets (1024 threads); 32: 6 (768 threads, the stages' larger rows in shared memory).
// gpt-oss-20b's experts on an RTX PRO 6000, us an expert GEMM (gate|up and down averaged), 16-row tasks / 32-row:
// 91 rows 172 / 181, 131 rows 199 / 188, 512 rows 526 / 435.
template <bool FUSE_ACT, bool IS_DOWN, int BM>
void launch(const bf16* xa, const ExpertMat& W, int N_work, int I_or_H, int K, const MoeScratch& sc, int max_tiles,
            int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  constexpr int SETS = BM == 32 ? 6 : 8;
  auto* kernel = k_moe_mxfp4_bf16_mma<FUSE_ACT, IS_DOWN, BM, SETS>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(Smem<BM, SETS>));
  if (!ready) throw std::runtime_error("moe_mxfp4_bf16_mma: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, 8 * SETS), max_tiles), dim3(32 * kSetWarps * SETS),
                        sizeof(Smem<BM, SETS>), L, xa, W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks,
                        sc.n_tasks, sc.pairs, top_k, mact, out, ld_out);
}

// ---- Prefill passes (more than FLYX_MAX_ROWS rows): swap-AB tiles ----------------------------------------------
// A the experts' weight rows (their codes times the stored scale, exact in bf16), B the task's rows (bf16, staged in
// the blocks' pairs (c, c + 16)), C[weight row][row]: one float32 sum an output, MMA after MMA in K's order, so every
// tile shape gives the same bits (numerics.md K) and the shape follows the rows an expert has
// (moe_mxfp4_bf16_pass_rows).  Rounds take the K-split kernels above.  A block: 16 MT WMW weight rows (fused gate|up:
// an m16 tile 8 columns' gate rows then their up rows) x 8 NT WT rows of one expert; warps WMW x WT, a warp MT m16 x NT
// n8 tiles; spans of SU units staged a span ahead (cp.async), NS stages.
template <int WROWS, int TOK, int NS, int SU>
struct PassSmem {
  struct Stage {
    alignas(16) bf16 x[TOK][SU * 32 + 32];       // rows: 16 banks apart (a quarter warp's 16-byte reads)
    alignas(16) uint8_t w[WROWS][SU * 16 + 16];  // codes: 4 banks apart
    alignas(16) uint16_t ws[WROWS][SU + 2];      // a weight row's span of scales (bf16 bits)
  };
  Stage st[NS];
  int pids[TOK];
  int rids[TOK];
};

template <bool FUSE_ACT, bool IS_DOWN, int MT, int NT, int WMW, int WT, int NS, int SU>
__global__ void __launch_bounds__(32 * WMW * WT)
    k_moe_mxfp4_bf16_pass(const bf16* xa, const uint8_t* __restrict__ W, const uint16_t* __restrict__ D,
                          const float* __restrict__ RS, int N_exp, int I_or_H, int K, const MoeTask* tasks,
                          const int* n_tasks, const int* pairs, int top_k, MoeAct mact, float* __restrict__ out,
                          int ld_out) {
  constexpr int WROWS = 16 * MT * WMW, TOK = 8 * NT * WT, NTH = 32 * WMW * WT;
  constexpr int COLS = FUSE_ACT ? WROWS / 2 : WROWS;
  using S_t = PassSmem<WROWS, TOK, NS, SU>;
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
    for (int i = tid; i < rows_st * (SU * 4); i += NTH) {  // 16 bytes (8 bf16) a thread
      const int r = i / (SU * 4), c = i % (SU * 4);
      const bool ok = r < nb && c < 4 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * (SU * 32) + 8 * c) : 0;
      cp_async16(&S.x[r][8 * c], xa + src, ok);
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
#pragma unroll
    for (int ul = 0; ul < SU; ++ul) {
      if (ul >= n_s) break;
      // weight rows 16 t + g and + 8: bytes 4 t4 .. 4 t4 + 3 of the block, pairs (4 t4 + i, 4 t4 + 16 + i) times the
      // scale; rows 8 t + g: their 16 bytes at 8 t4, the same pairs
      uint32_t a[MT][2][4], b[NT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int ra = 16 * (MT * ww + mt) + g, rb = ra + 8;
        const uint32_t qa = *reinterpret_cast<const uint32_t*>(&S.w[ra][16 * ul + 4 * t4]);
        const uint32_t qb = *reinterpret_cast<const uint32_t*>(&S.w[rb][16 * ul + 4 * t4]);
        const uint32_t sa = __byte_perm(uint32_t(S.ws[ra][ul]), 0u, 0x1010);
        const uint32_t sb = __byte_perm(uint32_t(S.ws[rb][ul]), 0u, 0x1010);
        uint32_t pa[4], pb[4];
        mxfp4x8_to_bf16(qa, pa);
        mxfp4x8_to_bf16(qb, pb);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          pa[i] = bf16x2_mul(pa[i], *reinterpret_cast<const __nv_bfloat162*>(&sa));
          pb[i] = bf16x2_mul(pb[i], *reinterpret_cast<const __nv_bfloat162*>(&sb));
        }
        // the unit's two MMAs: pairs 0, 1 then 2, 3 (A: rows g, g + 8 at k-slots 2 t4 and 2 t4 + 8)
        a[mt][0][0] = pa[0], a[mt][0][1] = pb[0], a[mt][0][2] = pa[1], a[mt][0][3] = pb[1];
        a[mt][1][0] = pa[2], a[mt][1][1] = pb[2], a[mt][1][2] = pa[3], a[mt][1][3] = pb[3];
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int tr = 8 * (NT * wt + nt) + g;
        const uint4 x = *reinterpret_cast<const uint4*>(&S.x[tr][32 * ul + 8 * t4]);
        b[nt][0] = x.x, b[nt][1] = x.y, b[nt][2] = x.z, b[nt][3] = x.w;
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if (8 * (NT * wt + nt) >= nb) break;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint32_t b0[2] = {b[nt][0], b[nt][1]}, b1[2] = {b[nt][2], b[nt][3]};
          mma_bf16(acc[mt][nt], a[mt][0], b0);
          mma_bf16(acc[mt][nt], a[mt][1], b1);
        }
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

// The three shapes, by the rows an expert has: 16 rows a task, 128 weight rows (8 warps, spans of 8); 64, 128 (8 warps
// 4 x 2, spans of 4); 128, 128 (16 warps 4 x 4, spans of 2, 3 stages).
template <bool FUSE_ACT, bool IS_DOWN, int MT, int NT, int WMW, int WT, int NS, int SU>
void launch_pass_cfg(const bf16* xa, const ExpertMat& W, int N_work, int I_or_H, int K, const MoeScratch& sc,
                     int max_tiles, int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  constexpr int WROWS = 16 * MT * WMW, TOK = 8 * NT * WT, COLS = FUSE_ACT ? WROWS / 2 : WROWS;
  using S_t = PassSmem<WROWS, TOK, NS, SU>;
  auto* kernel = k_moe_mxfp4_bf16_pass<FUSE_ACT, IS_DOWN, MT, NT, WMW, WT, NS, SU>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(S_t));
  if (!ready) throw std::runtime_error("moe_mxfp4_bf16_pass: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, COLS), max_tiles), dim3(32 * WMW * WT), sizeof(S_t), L, xa,
                        W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks, sc.n_tasks, sc.pairs, top_k, mact,
                        out, ld_out);
}

template <bool FUSE_ACT, bool IS_DOWN, int TOK>
void launch_pass(const bf16* xa, const ExpertMat& W, int N_work, int I_or_H, int K, const MoeScratch& sc,
                 int max_tiles, int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  // 128 rows: a warp's 16 weight rows dequantized once for 8 n8 tiles (gpt-oss-20b's experts at 512 rows an expert,
  // gate|up and down: 3,629 us against 4,119 with 32 weight rows x 4 n8 tiles a warp)
  constexpr int MT = TOK == 64 ? 2 : 1, NT = TOK == 128 ? 8 : 4;
  constexpr int WMW = TOK == 64 ? 4 : 8, WT = TOK == 32 ? 1 : 2;
  constexpr int NS = TOK == 128 ? 3 : 2, SU = TOK == 32 ? 8 : TOK == 64 ? 4 : 2;
  static_assert(8 * NT * WT == TOK);
  launch_pass_cfg<FUSE_ACT, IS_DOWN, MT, NT, WMW, WT, NS, SU>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out,
                                                             ld_out, L);
}

}  // namespace

void moe_mxfp4_bf16_stage(const float* x, int ldx, int K, int rows, bf16* out, const Launch& L) {
  if (K % 32) throw std::runtime_error("moe_mxfp4_bf16_stage: K a multiple of 32 (whole MXFP4 blocks)");
  launch_dependent(k_moe_stage_mxfp4_bf16, dim3(rows), dim3(256), L, x, ldx, K, out);
}

int moe_mxfp4_bm(int pairs, int n_experts) { return pairs > 14 * n_experts ? 32 : 16; }

// The pass kernels' rows a task by the rows an expert averages (RTX PRO 6000, gpt-oss-20b's experts, gate|up and down
// in us: bf16 at 18 rows an expert 377 with 32-row tasks against 537 with 64; at 64 rows 774 with 64 against 899 with
// 128; at 128 rows 1,135 with 128 against 1,284 with 64.  MXFP8 at 18 rows 346 with 16 against 372 with 64; at 32
// rows 399 with 64 against 442 with 16; at 128 rows 764 with 128 against 916 with 64).  Either kernel gives every
// shape the same bits.
int moe_mxfp4_bf16_pass_rows(int pairs, int n_experts) {
  return pairs <= 40 * n_experts ? 32 : pairs <= 96 * n_experts ? 64 : 128;
}
int moe_mxfp4_mxfp8_pass_rows(int pairs, int n_experts) {
  return pairs <= 24 * n_experts ? 16 : pairs <= 96 * n_experts ? 64 : 128;
}

void moe_mxfp4_bf16_mma(bool fuse_act, bool is_down, int bm, const bf16* xa, const ExpertMat& W, int N_work, int I_or_H,
                    int K, const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out,
                    const Launch& L) {
  if (bm != 16 && bm != 32) throw std::runtime_error("moe_mxfp4_bf16_mma: tasks of 16 or 32 rows");
  auto go = [&](auto bmc) {
    constexpr int B = decltype(bmc)::value;
    if (is_down) launch<false, true, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act) launch<true, false, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch<false, false, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (bm == 32) go(std::integral_constant<int, 32>{});
  else go(std::integral_constant<int, 16>{});
}

void moe_mxfp4_bf16_pass(bool fuse_act, bool is_down, int tok, const bf16* xa, const ExpertMat& W, int N_work,
                         int I_or_H, int K, const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out,
                         int ld_out, const Launch& L) {
  if (tok != 32 && tok != 64 && tok != 128)
    throw std::runtime_error("moe_mxfp4_bf16_pass: tasks of 32, 64 or 128 rows");
  auto go = [&](auto tc) {
    constexpr int T = decltype(tc)::value;
    if (is_down) launch_pass<false, true, T>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act)
      launch_pass<true, false, T>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch_pass<false, false, T>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (tok == 128) go(std::integral_constant<int, 128>{});
  else if (tok == 64) go(std::integral_constant<int, 64>{});
  else go(std::integral_constant<int, 32>{});
}

}  // namespace flyx::cuda
