// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The kernels of rounds (1..16 rows: decoding, verifying drafts) on GGML block weights whose
// inputs are 8 bits (FLYX_ACT_Q8): the rows as
// k_stage_q8_round leaves them, integers p with a scale d per 32 values.
//
// A warp computes a column over the whole of K: lane l takes the column's
// groups of 32 weights l, l + 32, l + 64, ... (group u is group u % G of
// block u / G, G groups a block), so the warp reads the column's bytes in
// order, and a group's 32 codes are one lane's (the types' unit_codes).
// Per group and row, the 32 products of two 8-bit integers are added up
// exactly (dp4a, from the bits of 1.5 x 2^23 as the prefill kernel's tensor
// cores do), then in float32, sub-group by sub-group as the prefill kernel
// does:  acc = fma(sum, d f, acc)  and, for a type with a minimum,
// acc = fma(d sum p, m, acc).  A lane adds its groups in the order of K: a
// column's sum is 32 "streams", lane l's being the groups l, l + 32, ...;
// the streams are added by a fixed tree (xor 1, 2, 4, 8, 16).  None of it
// depends on how many rows the round has, how many warps a block has or how
// many blocks there are: a row's result is the same in every round.
//
// Three kernels compute that, and give the same bits:
// * k_round_blocks_q8 (a few rows): dp4a, a warp a few columns at once.  A
//   column's codes are decoded once for all rows, but every row has its own
//   products, so it costs more the more rows the round has.  The blocks of
//   threads stay (as many as fit the GPU at once) and take columns in turn,
//   so the type's tables go to shared memory once per block; the rows are
//   read from global memory (the first-level cache holds them: every warp
//   reads the same), once for all the warp's columns; a lane has its next
//   groups' bytes on their way while it multiplies.  How many columns and
//   groups ahead: variants/candidates.toml (blocks_round), by the type, the
//   round's rows and the layer's columns.
// * k_round_blocks_mma (more rows): the integer tensor cores, one
//   mma.m16n8k32 for 16 rows x 8 columns of a group; see there.
// * k_round_blocks_cols: the integer tensor cores the other way round, one
//   mma.m16n8k32 for 16 columns x 8 rows of a group; see there.
//
// Compiled per group of block types (gemm_blocks_round_<group>.cu, gemm_blocks_ops.h), side by side;
// gemm_blocks_round.cu picks the kernel and configuration and dispatches.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <type_traits>

#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_ops.h"
#include "backends/cuda/kernels/gemm_blocks_types.cuh"
#include "flyx/gemm_tables.inc"

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

namespace flyx::cuda {

namespace {

constexpr int kRoundWarps = 8;  // warps a block of threads

// A configuration's columns a warp takes at once (a row's 8-bit values are loaded once for all of them) and groups a
// lane loads ahead per column; 0 takes them by the type's unit: 4 columns (2 for units over 32 bytes), and as many
// groups ahead as registers hold 4 units of 16 bytes or 2 of more.  Every choice gives the same bits: a column's sums
// depend neither on its neighbours nor on when its bytes arrive.
template <class F, int C>
constexpr int kRoundColumns = C ? C : sizeof(typename F::Unit) > 32 ? 2 : 4;
template <class F, int C, int U>
constexpr int kUnitsAhead = U ? U : std::max(1, (sizeof(typename F::Unit) <= 16 ? 4 : 2) / kRoundColumns<F, C>);

// A warp's C columns of block type F in a round of the dp4a kernel: each slot's first groups on their way (start),
// then every row's sums over K and the column's tree, written out (finish).  U groups a lane loads ahead per column.
template <class F, int R, int C, int U>
struct RoundColumns {
  static constexpr int GROUPS = F::kValues / 32, SUB = F::kSub;
  static constexpr bool MINS = F::kFlatRun != 0;  // the types with a minimum
  const uint8_t* wn[C];
  const uint16_t* dn[C];
  typename F::Unit raw[U][C];
  // The scales as loaded (binary16), converted where they are used: converting them here would wait for the load
  // and leave nothing on its way (cuda_bench round, RTX 5090: 2 to 6% at 2..8 rows).
  uint32_t d0[U][C], d1[U][C];

  __device__ __forceinline__ void load(int i, int u) {
    const int G = (threadIdx.x & 31) % GROUPS;  // the group of every block the lane takes (32 is a multiple of GROUPS)
    const size_t b = size_t(u / GROUPS);
#pragma unroll
    for (int c = 0; c < C; ++c) {
      raw[i][c] = block_unit<F>(wn[c], dn[c], b, G);
      d0[i][c] = dn[c][b * F::kScales];
      d1[i][c] = F::kScales > 1 ? dn[c][b * F::kScales + 1] : 0u;
    }
  }
  // Columns n0 .. n0 + C - 1 of W [N][K] (units = K / 32), and each slot's first group of them on its way.
  __device__ __forceinline__ void start(const uint8_t* W, const uint16_t* D, int N, int units, int n0) {
    const int lane = threadIdx.x & 31, blocks = units / GROUPS;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const int n = min(n0 + c, N - 1);  // a column past N reads the last one's bytes and writes nothing
      wn[c] = W + size_t(n) * blocks * F::kStride;
      dn[c] = D + size_t(n) * blocks * F::kScales;
    }
#pragma unroll
    for (int i = 0; i < U; ++i)
      if (lane + 32 * i < units) load(i, lane + 32 * i);
  }
  float acc[R][C];
  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int c = 0; c < C; ++c) acc[r][c] = 0.f;
  }
  // The columns' sums with the round's rows (a: their 8-bit values [rows][K], da: the scales [rows][K / 32], sa: the
  // sums [rows][K / 16]), into y [rows][ldy] at columns n0 .. (those below N).
  // sm: the columns' sums also to sm[r 64 + sj + c] (a SwiGLU's group, round_glu).
  __device__ __forceinline__ void finish(const Tables& T, const typename F::Local& local, const int8_t* a,
                                         const float* da, const float* sa, int K, int units, int rows, int n0, int N,
                                         float* y, int ldy, int accumulate, float* sm = nullptr, int sj = 0) {
    zero();
    sums(T, local, a, size_t(K), da, sa, units, 0, units, rows, 0, units, units);
    write(rows, n0, N, y, ldy, accumulate, sm, sj);
  }
  // Groups ub .. ue - 1 into the sums; the rows' values of group u at a + r a_stride + round_q8_at(u) - 32 ubase, its
  // scale at da[r d_stride + u - ubase] (sums: sa, two a group), so a window of K from group ubase on reads as all of
  // K; the slots' next groups loaded while below pf.
  __device__ __forceinline__ void sums(const Tables& T, const typename F::Local& local, const int8_t* a,
                                       size_t a_stride, const float* da, const float* sa, int d_stride, int ubase,
                                       int units, int rows, int ub, int ue, int pf) {
    const int lane = threadIdx.x & 31, G = lane % GROUPS;
    for (int u0 = ub + lane; u0 < ue; u0 += 32 * U) {
#pragma unroll
      for (int i = 0; i < U; ++i) {
        const int u = u0 + 32 * i;
        if (u >= ue) break;
        uint32_t w[C][8];
        float f[C][2], m[C][2];
#pragma unroll
        for (int c = 0; c < C; ++c) {
          F::unit_codes(T, local, raw[i][c], G, w[c]);
          F::unit_factors(raw[i][c], G, scale_bits<F>(d0[i][c]), F::kScales > 1 ? scale_bits<F>(d1[i][c]) : 0.f, f[c], m[c]);
        }
        if (u + 32 * U < pf) load(i, u + 32 * U);  // the slot's next group, on its way meanwhile
        const int at0 = round_q8_at(u, units, 0) - 32 * ubase, at1 = round_q8_at(u, units, 1) - 32 * ubase;
#pragma unroll
        for (int r = 0; r < R; ++r) {
          if (r >= rows) break;
          const int8_t* ar = a + size_t(r) * a_stride;
          const uint4 x0 = *reinterpret_cast<const uint4*>(ar + at0), x1 = *reinterpret_cast<const uint4*>(ar + at1);
          const float dx = da[size_t(r) * d_stride + u - ubase];
          float2 ds = make_float2(0.f, 0.f);
          if constexpr (MINS) {
            ds = *reinterpret_cast<const float2*>(sa + (size_t(r) * d_stride + u - ubase) * 2);
          }
#pragma unroll
          for (int c = 0; c < C; ++c) {
            // the first 16 values; the last 16 go on in the same sum, or, sub-groups of 16, in a sum of their own
            int s0 = kMagicBits;
            s0 = __dp4a(int(w[c][0]), int(x0.x), s0);
            s0 = __dp4a(int(w[c][1]), int(x0.y), s0);
            s0 = __dp4a(int(w[c][2]), int(x0.z), s0);
            s0 = __dp4a(int(w[c][3]), int(x0.w), s0);
            int s1 = SUB == 32 ? s0 : kMagicBits;
            s1 = __dp4a(int(w[c][4]), int(x1.x), s1);
            s1 = __dp4a(int(w[c][5]), int(x1.y), s1);
            s1 = __dp4a(int(w[c][6]), int(x1.z), s1);
            s1 = __dp4a(int(w[c][7]), int(x1.w), s1);
            if constexpr (SUB == 32) {
              s0 = s1;
            }
            float& ac = acc[r][c];
            ac = fmaf(__int_as_float(s0) - kMagic, dx * f[c][0], ac);
            if constexpr (MINS) {
              ac = fmaf(SUB == 16 ? ds.x : ds.x + ds.y, m[c][0], ac);
            }
            if constexpr (SUB == 16) {
              ac = fmaf(__int_as_float(s1) - kMagic, dx * f[c][1], ac);
              if constexpr (MINS) ac = fmaf(ds.y, m[c][1], ac);
            }
          }
        }
      }
    }
  }
  // Every row's sums by the tree, into y.
  __device__ __forceinline__ void write(int rows, int n0, int N, float* y, int ldy, int accumulate,
                                        float* sm = nullptr, int sj = 0) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r >= rows) break;
#pragma unroll
      for (int c = 0; c < C; ++c) {
        float v = acc[r][c];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);  // every lane: the same sum
        if (lane == (r * C + c) % 32 && n0 + c < N) {
          float* dst = y + size_t(r) * ldy + n0 + c;
          *dst = accumulate ? *dst + v : v;
          if (sm) sm[r * 64 + sj + c] = v;
        }
      }
    }
  }
};

// A round's SwiGLU in its gate|up GEMM (Launch::glu): the launch's y is column col0 of the gate|up rows gu [rows][ld]
// (gate 0..I-1, up I..2I-1), all of them (N = 2 I) or the up run's alone (col0 = I; the gate's launch before has
// written the gate's).  act(gate) * up goes to o in 8 bits as swiglu_split leaves it: put_q8 gives put_q8_pairs's
// bits (the same largest magnitude, integers and sums).
struct RoundGluArgs {
  const float* gu = nullptr;  // null: no SwiGLU
  int ld = 0, I = 0, col0 = 0;
  Out o{nullptr, nullptr, nullptr, nullptr};
  int act = 0;  // the gate's activation (flyx_mlp_act)
};

// The launch's column of a SwiGLU's group g whose j-th column (0..63: 32 of gate, then 32 of up; the up run alone:
// 0..31) a warp takes; N where j is past the group's (a block of 64 columns a pass on the up run alone).
__device__ __forceinline__ int glu_column(const RoundGluArgs& g, int N, int grp, int j) {
  if (N == 2 * g.I) return j < 32 ? 32 * grp + j : g.I + 32 * grp + j - 32;
  return j < 32 ? 32 * grp + j : N;
}

// The runs after the first in the same launch (a stack's next projections on the same rows, regions/linear.cpp;
// gemm_blocks_staged_runs): each run's N columns are written after the run's before it.  None: N = 0.
struct RoundNext {
  const uint8_t* W = nullptr;
  const uint16_t* D = nullptr;
  int N = 0;
};
constexpr int kRoundMore = 2;  // the most runs after the first
struct RoundMore {
  RoundNext run[kRoundMore];
};
template <class F>
constexpr bool kTypeMins = F::kFlatRun != 0;  // a type with a minimum: the rows' sums are staged too
template <>
constexpr bool kTypeMins<void> = false;
template <class F>
struct LocalOf {  // a type's tables in shared memory
  using type = typename F::Local;
};
template <>
struct LocalOf<void> {
  struct type {};
};

// A run after the first: its columns are turns t0 .. t0 + N - 1 of the grid's (turn t is warp t % warps's), one
// column each, as RoundColumns computes any column.
template <class F, int R>
__device__ __forceinline__ void round_more(const Tables& T, const typename F::Local& local, const int8_t* a,
                                           const float* da, const float* sa, int K, int units, int rows, int warp,
                                           int warps, int t0, const RoundNext& run, float* y, int ldy,
                                           int accumulate) {
  if (rows <= 0) return;
  int t = warp;
  if (t < t0) t += (t0 - t + warps - 1) / warps * warps;
  RoundColumns<F, R, 1, kUnitsAhead<F, 1, 0>> cols;
  for (; t < t0 + run.N; t += warps) {
    cols.start(run.W, run.D, run.N, units, t - t0);
    cols.finish(T, local, a, da, sa, K, units, rows, t - t0, run.N, y, ldy, accumulate);
  }
}

// Exact reachability check for a compiled dp4a kernel instance on compute capability `cc` (0: union of all archs).
template <class F, int R, int C, int U, int NW, int STAGED, int P, bool HAS_MORE, bool GLU>
__host__ __device__ constexpr bool round_kernel_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const int di = tables::ggml_type_index(F::kDtype);
  if (di < 0) return false;
  const int b = tables::arch_bucket(cc);
  const unsigned mask = GLU        ? tables::kReachRoundGlu[b][di]
                        : HAS_MORE ? tables::kReachRoundStack[b][di]
                                   : tables::kReachRoundDp4a[b][di];
  bool hit = false;
#define X(i, CR, CC, CU, CNW, CS, CP)                                                                         \
  if (mask & (1u << i)) {                                                                                     \
    constexpr int ec = kRoundColumns<F, CC>, eu = kUnitsAhead<F, CC, CU>;                                     \
    if (CR == R && ec == C && eu == U && CNW == NW && CS == STAGED && CP == P) hit = true;                    \
    if (CS == 1 && CR == R && ec == C && eu == U && kRoundWarps == NW && STAGED == 0 && CP == P) hit = true;  \
  }
  FLYX_BLOCKS_ROUND(X)
#undef X
  return hit;
#endif
}

template <class F, int Bit>
__host__ __device__ constexpr bool round_tc_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const int di = tables::ggml_type_index(F::kDtype);
  if (di < 0) return false;
  return (tables::kReachRoundTc[tables::arch_bucket(cc)][di] & (1u << Bit)) != 0;
#endif
}

// The rows through shared memory a window of kRoundWindow groups of K at a time (STAGED == 2), in two buffers: the
// block's warps take a window (each its columns, the same chains as ever: the same bits) while the next is copied in.
// The warps of a block walk the windows together, a pass of the block's columns at a time.
constexpr int kRoundWindow = 64;  // groups of 32 (two spans of round_q8_at); the sums' slots ahead stay within one
template <class F, int R>
__host__ __device__ constexpr size_t round_window_bytes(bool mins) {
  return size_t(R) * kRoundWindow * 32 + size_t(R) * kRoundWindow * 4 + (mins ? size_t(R) * kRoundWindow * 8 : 0);
}
template <class F, int R, int C, int U, int NW, int P, class Cols>
__device__ __forceinline__ void round_windows(const Tables& T, const typename F::Local& local, const int8_t* A,
                                              const float* dA, const float* sA, const uint8_t* W, const uint16_t* D,
                                              int N, int K, float* y, int ldy, int rows, int accumulate, Cols& cols,
                                              int warp, int warps, int row0, int my_rows, int first, bool early,
                                              unsigned char* smem) {
  static_assert(kRoundWindow % (32 * U) == 0, "a window holds whole turns of a lane's slots");
  constexpr bool MINS = F::kFlatRun != 0;
  constexpr int RP = R / P;
  const int units = K / 32, windows = (units + kRoundWindow - 1) / kRoundWindow, nr = min(rows, R);
  const size_t buf = round_window_bytes<F, R>(MINS);
  auto stage = [&](int w, int b) {  // window w's rows into buffer b
    unsigned char* base = smem + b * buf;
    int8_t* xa = reinterpret_cast<int8_t*>(base);
    float* xd = reinterpret_cast<float*>(base + size_t(R) * kRoundWindow * 32);
    float* xs = xd + size_t(R) * kRoundWindow;
    const int u0 = w * kRoundWindow, wu = min(kRoundWindow, units - u0);
    for (int i = threadIdx.x; i < nr * wu * 2; i += blockDim.x) {  // 16 bytes a piece
      const int r = i / (wu * 2), j = i % (wu * 2);
      cp_async16(xa + size_t(r) * kRoundWindow * 32 + 16 * j, A + size_t(r) * K + size_t(u0) * 32 + 16 * j);
    }
    for (int i = threadIdx.x; i < nr * wu; i += blockDim.x) {
      const int r = i / wu, j = i % wu;
      cp_async4(xd + size_t(r) * kRoundWindow + j, dA + size_t(r) * units + u0 + j);
    }
    if constexpr (MINS) {
      for (int i = threadIdx.x; i < nr * wu * 2; i += blockDim.x) {
        const int r = i / (wu * 2), j = i % (wu * 2);
        cp_async4(xs + size_t(r) * kRoundWindow * 2 + j, sA + size_t(r) * units * 2 + size_t(u0) * 2 + j);
      }
    }
    cp_async_commit();
  };
  const int block_first = (blockIdx.x * (NW / P)) * C;  // the block's first warp's first column
  for (int pass = 0;; ++pass) {
    const int n0 = first + pass * warps * C;
    if (block_first + pass * warps * C >= N) break;  // the same for every warp of the block
    const bool active = n0 < N && my_rows > 0;
    if (active) {
      if (!early || pass > 0) cols.start(W, D, N, units, n0);
      cols.zero();
    }
    stage(0, 0);
    for (int w = 0; w < windows; ++w) {
      if (w + 1 < windows) {
        stage(w + 1, (w + 1) & 1);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();  // window w is in for every warp
      if (active) {
        const unsigned char* base = smem + (w & 1) * buf;
        const int8_t* xa = reinterpret_cast<const int8_t*>(base) + size_t(row0) * kRoundWindow * 32;
        const float* xd = reinterpret_cast<const float*>(base + size_t(R) * kRoundWindow * 32);
        const float* xs = xd + size_t(R) * kRoundWindow;
        const int u0 = w * kRoundWindow;
        cols.sums(T, local, xa, size_t(kRoundWindow) * 32, xd + size_t(row0) * kRoundWindow,
                  xs + size_t(row0) * kRoundWindow * 2, kRoundWindow, u0, units, my_rows, u0,
                  min(units, u0 + kRoundWindow), units);
      }
      __syncthreads();  // every warp is done with buffer w & 1 before window w + 2 goes there
    }
    if (active) cols.write(my_rows, n0, N, y + size_t(row0) * ldy, ldy, accumulate);
  }
  (void)RP;
}

// The columns of a round's gate|up GEMM that also writes its SwiGLU (k_round_blocks_q8, GLU): block b takes the groups of
// 32 outputs b, b + gridDim.x, ..., a group's 64 columns (32 of gate, then 32 of up; the up run alone: its 32) NW C a
// pass, each warp C of them as ever (the same sums, into y), and their sums also in shared memory; once the block's
// warps are done with a group (a barrier), one warp (the group's turn) writes silu(gate) * up in 8 bits, the gate's
// from shared memory or, the up run alone, from y.  No warp waits on another block: a group's columns are one block's.
template <class F, int R, int C, int U, int NW>
__device__ __forceinline__ void round_glu(const Tables& T, const typename F::Local& local, const int8_t* a,
                                          const float* da, const float* sa, int K, int units, int rows,
                                          const uint8_t* W, const uint16_t* D, int N, float* y, int ldy,
                                          const RoundGluArgs& glu, RoundColumns<F, R, C, U>& cols, bool early) {
  static_assert(64 % (NW * C) == 0, "a block's pass within a group (the up run alone: 32 columns, some warps idle)");
  __shared__ float group_sums[2][R][64];  // a group's in turn: the warp that writes one reads it while the next fills
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, groups = glu.I / 32;
  const bool both = N == 2 * glu.I;
  const int passes = ((both ? 64 : 32) + NW * C - 1) / (NW * C);
  if (int(blockIdx.x) >= groups) pdl_trigger();
  for (int g = blockIdx.x, k = 0; g < groups; g += gridDim.x, ++k) {
    float* s = &group_sums[k & 1][0][0];
    for (int q = 0; q < passes; ++q) {
      const int j = q * NW * C + w * C, n0 = glu_column(glu, N, g, j);
      // The kernel after (the down GEMM) may launch once every block is at its last pass: launched from the start,
      // its blocks share the SMs this launch leaves free (a group a block: fewer blocks than turns) and their early
      // weights took this launch's bandwidth (2 rows, IQ3_XXS gate|up: 48 -> 57 us).
      if (g + int(gridDim.x) >= groups && q + 1 == passes) pdl_trigger();
      if (n0 >= N) continue;  // none of the group's for this warp
      if (!early || k || q) cols.start(W, D, N, units, n0);
      cols.finish(T, local, a, da, sa, K, units, rows, n0, N, y, ldy, 0, s, both ? j : 32 + j);
    }
    __syncthreads();  // the group's sums are in
    if (w == k % NW) {
      for (int r = 0; r < rows; ++r) {
        const float gate = both ? s[r * 64 + lane] : __ldcg(glu.gu + size_t(r) * glu.ld + 32 * g + lane);
        put_q8(glu.o, r, g, glu_act(glu.act, gate) * s[r * 64 + 32 + lane]);
      }
    }
  }
}

// STAGED: the block first copies the round's rows to shared memory and reads them there (the dynamic shared memory
// round_rows_bytes gives).  Read from global memory, the rows of a wide layer come back from L2 for every column a
// warp takes, each row's after the row before has been used (the registers are the same): at 8 rows two thirds of
// the row loop is waiting for them.
// Fn, Fn2: the block types of the runs after the first (RoundMore), or void.  Their columns are turns of their own,
// one column each, after the first run's ceil(N / C) (a column of K takes a lane about as many loads as C columns of
// the first run: the same groups ahead in all), so the first run's columns go as they would alone and the others'
// even out the last turn.  Which warp takes a column changes no bits.
// P: neighbouring warps that take the same columns, each R / P of the rows (part p rows p R / P ..): more warps on
// a layer of few columns, each with a shorter row loop between its loads; a row's sums are the same 32 chains.
// GLU: the launch also writes a round's SwiGLU (glu; round_glu takes the columns a block a group at a time).
template <class F, int R, int C, int U, int NW, int STAGED, int P = 1, class Fn = void, class Fn2 = void,
          bool GLU = false>
__global__ void __launch_bounds__(32 * NW) k_round_blocks_q8(Tables T, const int8_t* A, const float* dA, const float* sA,
                                                             const uint8_t* __restrict__ W,
                                                             const uint16_t* __restrict__ D, int N, int K,
                                                             float* __restrict__ y, int ldy, int rows, int accumulate,
                                                             RoundMore more, RoundGluArgs glu) {
  static_assert(!GLU || (P == 1 && STAGED != 2 && std::is_void_v<Fn>), "a SwiGLU: one part of the rows, one run");
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!round_kernel_active<F, R, C, U, NW, STAGED, P, !std::is_void_v<Fn>, GLU>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  if constexpr (!GLU) pdl_trigger();  // round_glu: at its block's last pass
  constexpr bool MINS = kTypeMins<F> || kTypeMins<Fn> || kTypeMins<Fn2>;
  __shared__ typename F::Local local;
  __shared__ typename LocalOf<Fn>::type local_next;
  __shared__ typename LocalOf<Fn2>::type local_next2;
  extern __shared__ __align__(16) unsigned char staged_rows[];  // [R][K] values, [R][K / 32] scales, [..][2] sums
  F::tables(T, local);
  if constexpr (!std::is_void_v<Fn>) Fn::tables(T, local_next);
  if constexpr (!std::is_void_v<Fn2>) Fn2::tables(T, local_next2);
  __syncthreads();
  const int units = K / 32;
  int8_t* const xa = reinterpret_cast<int8_t*>(staged_rows);
  float* const xd = reinterpret_cast<float*>(staged_rows + size_t(R) * K);
  float* const xs = xd + size_t(R) * units;
  constexpr int RP = R / P;  // the rows a warp holds
  const int wid = blockIdx.x * NW + int(threadIdx.x >> 5);
  const int warp = wid / P, warps = gridDim.x * (NW / P), row0 = (wid % P) * RP;
  const int my_rows = min(rows - row0, RP);  // this warp's rows (none: the round has fewer than its part's first)
  int first = my_rows > 0 ? warp * C : N;
  if constexpr (GLU) first = int(blockIdx.x) < glu.I / 32 ? glu_column(glu, N, blockIdx.x, int(threadIdx.x >> 5) * C) : N;
  RoundColumns<F, RP, C, U> cols;
  // The weights are not the kernel before's: in rounds of up to 4 rows the first ones go out before waiting for it,
  // and arrive while it finishes (programmatic dependent launch; rounds of 8 rows lost by it); the next columns' at
  // the top of their pass, as their registers free.
  constexpr bool EARLY = R <= 4;
  if (EARLY && first < N) cols.start(W, D, N, units, first);
  pdl_wait();  // the rows come from the kernel before
  if constexpr (STAGED == 2) {  // the rows a window of K at a time (this function returns from in there)
    static_assert(std::is_void_v<Fn>, "windows: one run");
    round_windows<F, R, C, U, NW, P>(T, local, A, dA, sA, W, D, N, K, y, ldy, rows, accumulate, cols, warp, warps,
                                     row0, my_rows, first, EARLY, staged_rows);
    return;
  }
  if constexpr (STAGED == 1) {
    const int nr = min(rows, R);
    for (int i = threadIdx.x; i < nr * K / 16; i += blockDim.x) cp_async16(xa + 16 * size_t(i), A + 16 * size_t(i));
    for (int i = threadIdx.x; i < nr * units; i += blockDim.x) cp_async4(xd + i, dA + i);
    if constexpr (MINS) {
      for (int i = threadIdx.x; i < nr * units * 2; i += blockDim.x) cp_async4(xs + i, sA + i);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
  }
  const int8_t* const a = (STAGED == 1 ? xa : A) + size_t(row0) * K;
  const float* const da = (STAGED == 1 ? xd : dA) + size_t(row0) * units;
  const float* const sa = (STAGED == 1 ? xs : sA) + size_t(row0) * units * 2;
  if constexpr (GLU) {
    round_glu<F, RP, C, U, NW>(T, local, a, da, sa, K, units, my_rows, W, D, N, y, ldy, glu, cols, EARLY);
  } else {  // (not after a return: nvcc calls the loop below unreachable in every GLU instance)
    float* const yr = y + size_t(row0) * ldy;
    for (int n0 = first; n0 < N; n0 += warps * C) {
      if (!EARLY || n0 != first) cols.start(W, D, N, units, n0);
      cols.finish(T, local, a, da, sa, K, units, my_rows, n0, N, yr, ldy, accumulate);
    }
    const int turns = (N + C - 1) / C;  // the first run's
    if constexpr (!std::is_void_v<Fn>) {
      round_more<Fn, RP>(T, local_next, a, da, sa, K, units, my_rows, warp, warps, turns, more.run[0], yr + N, ldy,
                         accumulate);
    }
    if constexpr (!std::is_void_v<Fn2>) {
      round_more<Fn2, RP>(T, local_next2, a, da, sa, K, units, my_rows, warp, warps, turns + more.run[0].N,
                          more.run[1], yr + N + more.run[0].N, ldy, accumulate);
    }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- rounds of more rows: the integer tensor cores ------------------------------------------------
// The same sums, the 16 rows of a group at once.  A tile of 8 columns takes 4 warps: warp w keeps the streams
// 8 w .. 8 w + 7 of its columns, which in every span of 32 groups are one block of 256 weights (or 8 blocks of
// 32).  Its lanes decode them as the prefill kernel's do (lane t of column g: weights 8 t .. 8 t + 7 of a group,
// the types' load / share / codes / factors), and mma.m16n8k32 multiplies them by the group's 16 rows into exact
// int32 sums from the bits of 1.5 x 2^23, one per sub-group (for a sub-group of 16 the other half's codes are
// zero).  Then the dp4a kernel's float32 steps, element by element, into the stream's own sums; at the end its
// tree: xor 1, 2 and 4 within the warp's 8 streams, xor 8 and 16 among the tile's 4 warps.  So a row's result is
// k_round_blocks_q8's, bit for bit (cuda_ops checks it).
//
// The rows go through shared memory a span at a time (every row's 32 groups; cp.async into a ring of three, two
// spans ahead, so one barrier a span both says a span is in and frees the buffer of the span before it), a
// group's 32 values in 32 bytes, a row padded so that the eight rows one load reads are in different banks.  Every
// block of threads re-reads the rows for its columns, so a block takes many (CT tiles); the blocks stay and take
// the tiles in turn.
constexpr int kTileWarps = 4;           // warps a tile of 8 columns: 32 streams, 8 a warp
constexpr int kSpanRow = 32 * 32 + 32;  // bytes of a row's span in shared memory: 32 groups, and 32 of padding
constexpr int kRowBuffers = 3;

template <class F, int R, int CT>
struct MmaSmem {
  static constexpr bool MINS = F::kFlatRun != 0;  // the types with a minimum
  alignas(16) int8_t rows[kRowBuffers][R][kSpanRow];
  float d[kRowBuffers][R][33];
  alignas(8) float sums[MINS ? kRowBuffers : 1][MINS ? R : 1][66];
  float tree[CT][kTileWarps][32][R / 4];  // a warp's 8 streams added up, for the tile's first warp
  typename F::Local local;
};

template <class F, int R, int CT>
__global__ void __launch_bounds__(32 * kTileWarps * CT)
k_round_blocks_mma(Tables T, const int8_t* A, const float* dA, const float* sA, const uint8_t* __restrict__ W,
                   const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y, int ldy, int rows, int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!round_tc_active<F, R == 8 ? 0 : 1>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  using S = MmaSmem<F, R, CT>;
  // BPW: blocks a warp takes a span; E: the sums a lane holds a stream (rows g, g + 8 with R 16; columns 2 t, 2 t + 1)
  constexpr int GROUPS = F::kValues / 32, BPW = 8 / GROUPS, SUB = F::kSub, E = R / 4, THREADS = 32 * kTileWarps * CT;
  constexpr bool MINS = S::MINS;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S*>(smem_raw);
  F::tables(T, sm.local);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wi = warp % kTileWarps, ct = warp / kTileWarps;
  const int g = lane >> 2, t = lane & 3;
  const int units = K / 32, spans = (units + 31) / 32, blocks = K / F::kValues, tiles = (N + 8 * CT - 1) / (8 * CT);
  auto stage = [&](int j, int b) {  // span j of the rows into buffer b, on its way (absent rows: zeros)
    const int n_s = min(32, units - 32 * j);
    for (int i = tid; i < R * 2 * n_s; i += THREADS) {  // a span is its groups' first halves, then their second
      const int r = i / (2 * n_s), c = i - r * 2 * n_s, at = c < n_s ? 32 * c : 32 * (c - n_s) + 16;
      cp_async16(&sm.rows[b][r][at], A + size_t(r < rows ? r : 0) * K + size_t(j) * 1024 + 16 * c, r < rows);
    }
    for (int i = tid; i < R * n_s; i += THREADS) {
      const int r = i / n_s, c = i - r * n_s;
      cp_async4(&sm.d[b][r][c], dA + size_t(r < rows ? r : 0) * units + 32 * j + c, r < rows);
    }
    if constexpr (MINS) {
      for (int i = tid; i < R * 2 * n_s; i += THREADS) {
        const int r = i / (2 * n_s), c = i - r * 2 * n_s;
        cp_async4(&sm.sums[b][r][c], sA + (size_t(r < rows ? r : 0) * units + 32 * j) * 2 + c, r < rows);
      }
    }
    // (Braces above: without them nvcc 13 dropped this commit where MINS is false, and the copies raced the reads.)
    cp_async_commit();
  };
  pdl_wait();  // the rows come from the kernel before
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int n0 = tile * 8 * CT + ct * 8, n = min(n0 + g, N - 1);  // a column past N: the last one's bytes, not written
    const uint8_t* wn = W + size_t(n) * blocks * F::kStride;
    const uint16_t* dn = D + size_t(n) * blocks * F::kScales;
    // The warp's weights a span: SLOTS spans of them in registers, a span's loaded SLOTS - 1 spans ahead into the
    // slot the span before it left (no copies between slots: a load is waited for where it is used).
    constexpr int SLOTS = BPW == 1 ? 3 : 2;
    typename F::Raw raw[SLOTS][BPW];
    float rs[SLOTS][BPW][2];
    auto load = [&](int j, typename F::Raw* r, float (*s)[2]) {
#pragma unroll
      for (int bi = 0; bi < BPW; ++bi) {
        const int b0 = (32 * j + 8 * wi) / GROUPS + bi, b = b0 < blocks ? b0 : blocks - 1;
        r[bi] = block_load<F>(wn, dn, size_t(b), t);
        s[bi][0] = scale_of<F>(dn, size_t(b) * F::kScales);
        s[bi][1] = F::kScales > 1 ? scale_of<F>(dn, size_t(b) * F::kScales + 1) : 0.f;
      }
    };
    float acc[8][E];
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int e = 0; e < E; ++e) acc[s][e] = 0.f;
    // Span j: the warp's 8 streams, its weights in r, s; the rows in buffer buf.
    auto compute = [&](int j, int buf, const typename F::Raw* r, const float (*sc)[2]) {
#pragma unroll
      for (int bi = 0; bi < BPW; ++bi) {
        if ((32 * j + 8 * wi) / GROUPS + bi >= blocks) continue;  // past K: the same for the whole warp
        const typename F::Lane ln = F::share(r[bi], t, lane);
#pragma unroll
        for (int G = 0; G < GROUPS; ++G) {
          const int s = bi * GROUPS + G, ul = 8 * wi + s;  // the warp's stream; the group within the span
          uint32_t w[2];
          float f, m;
          F::codes(T, sm.local, ln, G, w);
          F::factors(ln, G, sc[bi][0], sc[bi][1], &f, &m);
          // The factors of columns 2 t and 2 t + 1 (of each sub-group: lanes 0, 1 and 2, 3 of a column hold one each).
          float fc[2][2], mc[2][2];
#pragma unroll
          for (int h = 0; h < 32 / SUB; ++h) {
            const int o = SUB == 16 ? 2 * h : 0;
            fc[0][h] = __shfl_sync(0xffffffffu, f, 8 * t + o);
            fc[1][h] = __shfl_sync(0xffffffffu, f, 8 * t + 4 + o);
            if constexpr (MINS) {
              mc[0][h] = __shfl_sync(0xffffffffu, m, 8 * t + o);
              mc[1][h] = __shfl_sync(0xffffffffu, m, 8 * t + 4 + o);
            }
          }
          // rows g and g + 8 (R 8: zeros), the lane's 8 values of the group: k 4 t .. and 16 + 4 t .. of the MMA
          const uint2 x0 = *reinterpret_cast<const uint2*>(&sm.rows[buf][g][32 * ul + 8 * t]);
          uint2 x8 = make_uint2(0u, 0u);
          float dr[2] = {sm.d[buf][g][ul], 0.f};
          float2 sr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
          if constexpr (MINS) {
            sr[0] = *reinterpret_cast<const float2*>(&sm.sums[buf][g][2 * ul]);
          }
          if constexpr (R == 16) {
            x8 = *reinterpret_cast<const uint2*>(&sm.rows[buf][g + 8][32 * ul + 8 * t]);
            dr[1] = sm.d[buf][g + 8][ul];
            if constexpr (MINS) sr[1] = *reinterpret_cast<const float2*>(&sm.sums[buf][g + 8][2 * ul]);
          }
          const uint32_t a[4] = {x0.x, x8.x, x0.y, x8.y};
          int c0[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits}, c1[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          if constexpr (SUB == 32) {
            mma_s8_k32(c0, a, w);
          } else {  // the first 16 weights are lanes 0, 1's, the last 16 lanes 2, 3's
            const uint32_t lo[2] = {t < 2 ? w[0] : 0u, t < 2 ? w[1] : 0u}, hi[2] = {t < 2 ? 0u : w[0], t < 2 ? 0u : w[1]};
            mma_s8_k32(c0, a, lo);
            mma_s8_k32(c1, a, hi);
          }
#pragma unroll
          for (int e = 0; e < E; ++e) {
            const int h = e >> 1, c = e & 1;  // row g (+ 8 h), column 2 t + c
            float& v = acc[s][e];
            v = fmaf(__int_as_float(c0[e]) - kMagic, dr[h] * fc[c][0], v);
            if constexpr (MINS) {
              v = fmaf(SUB == 16 ? sr[h].x : sr[h].x + sr[h].y, mc[c][0], v);
            }
            if constexpr (SUB == 16) {
              v = fmaf(__int_as_float(c1[e]) - kMagic, dr[h] * fc[c][1], v);
              if constexpr (MINS) v = fmaf(sr[h].y, mc[c][1], v);
            }
          }
        }
      }
    };
#pragma unroll
    for (int k = 0; k + 1 < SLOTS; ++k) {
      if (k < spans) load(k, raw[k], rs[k]);
    }
    stage(0, 0);
    if (spans > 1) stage(1, 1);
#pragma unroll 1
    for (int j0 = 0; j0 < spans; j0 += SLOTS) {
#pragma unroll
      for (int k = 0; k < SLOTS; ++k) {
        const int j = j0 + k;
        if (j >= spans) break;
        if (j + 1 < spans) cp_async_wait<1>();  // span j's rows; the next span's may still be on their way
        else cp_async_wait<0>();
        // Span j's rows are in for every warp, and every warp is done with span j - 1 (the first time: the tables
        // are in too): its buffer takes span j + 2.
        __syncthreads();
        if (j + 2 < spans) stage(j + 2, (j + 2) % kRowBuffers);
        if (j + SLOTS - 1 < spans) load(j + SLOTS - 1, raw[(k + SLOTS - 1) % SLOTS], rs[(k + SLOTS - 1) % SLOTS]);
        compute(j, j % kRowBuffers, raw[k], rs[k]);
      }
    }
    // The tree: xor 1, 2, 4 within the warp's streams; xor 8, 16 among the tile's warps.
#pragma unroll
    for (int e = 0; e < E; ++e)
      sm.tree[ct][wi][lane][e] =
          ((acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e])) + ((acc[4][e] + acc[5][e]) + (acc[6][e] + acc[7][e]));
    __syncthreads();
    if (wi == 0) {
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const float v = (sm.tree[ct][0][lane][e] + sm.tree[ct][1][lane][e]) +
                        (sm.tree[ct][2][lane][e] + sm.tree[ct][3][lane][e]);
        const int r = g + 8 * (e >> 1), col = n0 + 2 * t + (e & 1);
        if (r < rows && col < N) {
          float* dst = y + size_t(r) * ldy + col;
          *dst = accumulate ? *dst + v : v;
        }
      }
    }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- rounds of a few rows on the tensor cores, the columns in the MMA's first operand ----------------
// The same sums once more, with the weights as the MMA's first operand: mma.m16n8k32 takes 16 columns x 32 values
// of a group and the group's values of 8 rows, so a round of up to 8 rows fills it (two MMAs for 16), where
// k_round_blocks_mma, its rows the first operand, leaves half of it empty with 8 and decodes a column for every 8
// columns of output.  Lane t of quad g decodes the values 8 t .. 8 t + 7 of the group for columns g and g + 8 (the
// types' load / share / codes / factors, once a column; the rows' values in the same order of k, so the integer
// sums are exact as before), and its sums are those two columns' for rows 2 t and 2 t + 1: the factors are the
// lane's own (sub-groups of 16: the other half's from lane t ^ 2).  The streams, the rows in shared memory and the
// tree are k_round_blocks_mma's: a row's result is k_round_blocks_q8's, bit for bit (cuda_ops checks it).
template <class F, int R, int CT>
struct ColsSmem {  // (as MmaSmem; the tree holds R / 2 sums a lane)
  static constexpr bool MINS = F::kFlatRun != 0;  // the types with a minimum
  alignas(16) int8_t rows[kRowBuffers][R][kSpanRow];
  float d[kRowBuffers][R][33];
  alignas(8) float sums[MINS ? kRowBuffers : 1][MINS ? R : 1][66];
  float tree[CT][kTileWarps][32][R / 2];  // a warp's 8 streams added up, for the tile's first warp
  typename F::Local local;
};

// SLOTS: spans of a warp's weights in registers (loads in flight: SLOTS - 1 spans ahead).
template <class F, int R, int CT, int SLOTS>
__global__ void __launch_bounds__(32 * kTileWarps * CT)
k_round_blocks_cols(Tables T, const int8_t* A, const float* dA, const float* sA, const uint8_t* __restrict__ W,
                    const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y, int ldy, int rows,
                    int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!round_tc_active<F, 2>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  using S = ColsSmem<F, R, CT>;
  // BPW: blocks a warp takes a span; NT: MMAs of 8 rows; E: the sums a lane holds a stream (columns g, g + 8 x rows
  // 2 t, 2 t + 1, of each 8 rows)
  constexpr int GROUPS = F::kValues / 32, BPW = 8 / GROUPS, SUB = F::kSub, NT = R / 8, E = 4 * NT;
  constexpr int THREADS = 32 * kTileWarps * CT;
  constexpr bool MINS = S::MINS;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S*>(smem_raw);
  F::tables(T, sm.local);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wi = warp % kTileWarps, ct = warp / kTileWarps;
  const int g = lane >> 2, t = lane & 3;
  const int units = K / 32, spans = (units + 31) / 32, blocks = K / F::kValues, tiles = (N + 16 * CT - 1) / (16 * CT);
  auto stage = [&](int j, int b) {  // span j of the rows into buffer b, on its way (absent rows: zeros)
    const int n_s = min(32, units - 32 * j);
    for (int i = tid; i < R * 2 * n_s; i += THREADS) {  // a span is its groups' first halves, then their second
      const int r = i / (2 * n_s), c = i - r * 2 * n_s, at = c < n_s ? 32 * c : 32 * (c - n_s) + 16;
      cp_async16(&sm.rows[b][r][at], A + size_t(r < rows ? r : 0) * K + size_t(j) * 1024 + 16 * c, r < rows);
    }
    for (int i = tid; i < R * n_s; i += THREADS) {
      const int r = i / n_s, c = i - r * n_s;
      cp_async4(&sm.d[b][r][c], dA + size_t(r < rows ? r : 0) * units + 32 * j + c, r < rows);
    }
    if constexpr (MINS) {
      for (int i = tid; i < R * 2 * n_s; i += THREADS) {
        const int r = i / (2 * n_s), c = i - r * 2 * n_s;
        cp_async4(&sm.sums[b][r][c], sA + (size_t(r < rows ? r : 0) * units + 32 * j) * 2 + c, r < rows);
      }
    }
    cp_async_commit();
  };
  pdl_wait();  // the rows come from the kernel before
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int n0 = tile * 16 * CT + ct * 16;
    // columns g and g + 8 (a column past N: the last one's bytes, not written)
    const uint8_t* wn[2];
    const uint16_t* dn[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = min(n0 + g + 8 * h, N - 1);
      wn[h] = W + size_t(n) * blocks * F::kStride;
      dn[h] = D + size_t(n) * blocks * F::kScales;
    }
    // The warp's weights a span (both columns): SLOTS spans of them in registers, as k_round_blocks_mma keeps them.
    typename F::Raw raw[SLOTS][BPW][2];
    float rs[SLOTS][BPW][2][2];
    auto load = [&](int j, typename F::Raw (*r)[2], float (*s)[2][2]) {
#pragma unroll
      for (int bi = 0; bi < BPW; ++bi) {
        const int b0 = (32 * j + 8 * wi) / GROUPS + bi, b = b0 < blocks ? b0 : blocks - 1;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          r[bi][h] = block_load<F>(wn[h], dn[h], size_t(b), t);
          s[bi][h][0] = scale_of<F>(dn[h], size_t(b) * F::kScales);
          s[bi][h][1] = F::kScales > 1 ? scale_of<F>(dn[h], size_t(b) * F::kScales + 1) : 0.f;
        }
      }
    };
    float acc[8][E];
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
      for (int e = 0; e < E; ++e) acc[s][e] = 0.f;
    // Span j: the warp's 8 streams, its weights in r, s; the rows in buffer buf.
    auto compute = [&](int j, int buf, const typename F::Raw (*r)[2], const float (*sc)[2][2]) {
#pragma unroll
      for (int bi = 0; bi < BPW; ++bi) {
        if ((32 * j + 8 * wi) / GROUPS + bi >= blocks) continue;  // past K: the same for the whole warp
        const typename F::Lane ln[2] = {F::share(r[bi][0], t, lane), F::share(r[bi][1], t, lane)};
#pragma unroll
        for (int G = 0; G < GROUPS; ++G) {
          const int s = bi * GROUPS + G, ul = 8 * wi + s;  // the warp's stream; the group within the span
          uint32_t w[2][2];
          float fc[2][2], mc[2][2];  // [column g, g + 8][sub-group]
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            float f, m;
            F::codes(T, sm.local, ln[h], G, w[h]);
            F::factors(ln[h], G, sc[bi][h][0], sc[bi][h][1], &f, &m);
            if constexpr (SUB == 32) {
              fc[h][0] = f;
              mc[h][0] = m;
            } else {  // lanes t < 2 hold the first sub-group's, t >= 2 the second's
              const float fo = __shfl_xor_sync(0xffffffffu, f, 2);
              fc[h][0] = t < 2 ? f : fo;
              fc[h][1] = t < 2 ? fo : f;
              if constexpr (MINS) {
                const float mo = __shfl_xor_sync(0xffffffffu, m, 2);
                mc[h][0] = t < 2 ? m : mo;
                mc[h][1] = t < 2 ? mo : m;
              }
            }
          }
          // columns g and g + 8: k 4 t .. and 16 + 4 t .. of the MMA are the lane's 8 values of the group
          const uint32_t a[4] = {w[0][0], w[1][0], w[0][1], w[1][1]};
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            // the second operand: row 8 nt + g's same 8 values
            const uint2 x = *reinterpret_cast<const uint2*>(&sm.rows[buf][8 * nt + g][32 * ul + 8 * t]);
            // the scales (and sums) of the lane's rows 8 nt + 2 t, + 1
            float dr[2];
            float2 sr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
            for (int i = 0; i < 2; ++i) {
              dr[i] = sm.d[buf][8 * nt + 2 * t + i][ul];
              if constexpr (MINS) sr[i] = *reinterpret_cast<const float2*>(&sm.sums[buf][8 * nt + 2 * t + i][2 * ul]);
            }
            int c0[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits}, c1[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
            if constexpr (SUB == 32) {
              const uint32_t b[2] = {x.x, x.y};
              mma_s8_k32(c0, a, b);
            } else {  // the first 16 values are lanes 0, 1's, the last 16 lanes 2, 3's
              const uint32_t lo[2] = {t < 2 ? x.x : 0u, t < 2 ? x.y : 0u}, hi[2] = {t < 2 ? 0u : x.x, t < 2 ? 0u : x.y};
              mma_s8_k32(c0, a, lo);
              mma_s8_k32(c1, a, hi);
            }
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              const int h = e >> 1, i = e & 1;  // column g (+ 8 h), row 8 nt + 2 t + i
              float& v = acc[s][4 * nt + e];
              v = fmaf(__int_as_float(c0[e]) - kMagic, dr[i] * fc[h][0], v);
              if constexpr (MINS) {
                v = fmaf(SUB == 16 ? sr[i].x : sr[i].x + sr[i].y, mc[h][0], v);
              }
              if constexpr (SUB == 16) {
                v = fmaf(__int_as_float(c1[e]) - kMagic, dr[i] * fc[h][1], v);
                if constexpr (MINS) v = fmaf(sr[i].y, mc[h][1], v);
              }
            }
          }
        }
      }
    };
#pragma unroll
    for (int k = 0; k + 1 < SLOTS; ++k) {
      if (k < spans) load(k, raw[k], rs[k]);
    }
    stage(0, 0);
    if (spans > 1) stage(1, 1);
#pragma unroll 1
    for (int j0 = 0; j0 < spans; j0 += SLOTS) {
#pragma unroll
      for (int k = 0; k < SLOTS; ++k) {
        const int j = j0 + k;
        if (j >= spans) break;
        if (j + 1 < spans) cp_async_wait<1>();  // span j's rows; the next span's may still be on their way
        else cp_async_wait<0>();
        // Span j's rows are in for every warp, and every warp is done with span j - 1 (the first time: the tables
        // are in too): its buffer takes span j + 2.
        __syncthreads();
        if (j + 2 < spans) stage(j + 2, (j + 2) % kRowBuffers);
        if (j + SLOTS - 1 < spans) load(j + SLOTS - 1, raw[(k + SLOTS - 1) % SLOTS], rs[(k + SLOTS - 1) % SLOTS]);
        compute(j, j % kRowBuffers, raw[k], rs[k]);
      }
    }
    // The tree: xor 1, 2, 4 within the warp's streams; xor 8, 16 among the tile's warps.
#pragma unroll
    for (int e = 0; e < E; ++e)
      sm.tree[ct][wi][lane][e] =
          ((acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e])) + ((acc[4][e] + acc[5][e]) + (acc[6][e] + acc[7][e]));
    __syncthreads();
    if (wi == 0) {
#pragma unroll
      for (int e = 0; e < E; ++e) {
        const float v = (sm.tree[ct][0][lane][e] + sm.tree[ct][1][lane][e]) +
                        (sm.tree[ct][2][lane][e] + sm.tree[ct][3][lane][e]);
        const int r = 8 * (e >> 2) + 2 * t + (e & 1), col = n0 + g + 8 * ((e >> 1) & 1);
        if (r < rows && col < N) {
          float* dst = y + size_t(r) * ldy + col;
          *dst = accumulate ? *dst + v : v;
        }
      }
    }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- rounds on the tensor cores with the rows held: the streams split among blocks of threads ----------------
// k_round_blocks_mma's blocks copy the whole of the round's rows again for every 32 columns they take (a tile's 4
// warps take its 32 streams, 8 each, and a block holds 4 tiles' sums): with 16 rows those copies are about as many
// bytes as the weights (RTX 5090, Q4_K 17408 x 5120: 42 us with 1 row, 72 with 16; dp4a 32 with 1 row).  Here the
// streams go to kSplitSets blocks of threads, 8 each (set q: groups 8 q .. 8 q + 7 of every span), and a block holds
// its set's share of the rows in shared memory for the whole launch (a quarter of them: up to 16 rows of K 17408 fit)
// while its warps take tiles of 8 columns in turn, each warp all of a tile's 8 streams of the set on its own: after
// the rows are in, no barrier.  A warp computes its streams as k_round_blocks_mma's warp of the same streams does (the
// same MMAs and sums in the same order), adds them up (xor 1, 2, 4) and leaves that in the scratch (round_parts); the
// last of a tile's kSplitSets warps to arrive adds the sets (xor 8, 16: (P0 + P1) + (P2 + P3)), writes the tile and
// sets its counter (round_count) back to zero.  The same bits as k_round_blocks_mma and k_round_blocks_q8.
constexpr int kSplitSets = 4;

// The tables k_round_blocks_split decodes with: with BIG a type's MmaLocal where it has one, else its Local.
template <class F, bool BIG, class = void>
struct SplitTables {
  using type = typename F::Local;
  static __device__ __forceinline__ void fill(const Tables& T, type& L) { F::tables(T, L); }
};
template <class F>
struct SplitTables<F, true, std::void_t<typename F::MmaLocal>> {
  using type = typename F::MmaLocal;
  static __device__ __forceinline__ void fill(const Tables& T, type& L) { F::mma_tables(T, L); }
};
template <class F, class = void>
constexpr bool kHasMmaLocal = false;
template <class F>
constexpr bool kHasMmaLocal<F, std::void_t<typename F::MmaLocal>> = true;

// The shared memory of k_round_blocks_split for R rows of K: the set's rows (8 groups of 32 a span, each row padded
// by 32 bytes so that the eight rows a load reads are in different banks), their scales and (types with a minimum)
// their sums: a group's two, or (sub-groups of 32) the two added as the kernels add them where they are used.
template <class F, int R, bool BIG>
struct SplitLayout {
  static constexpr bool MINS = F::kFlatRun != 0;
  static constexpr int SUMS = F::kSub == 16 ? 2 : 1;  // sums a group
  int spans, row_stride, d_stride, s_stride;          // bytes; floats; floats
  size_t rows_at, d_at, s_at, local_at, bytes;
  __host__ __device__ explicit SplitLayout(int K) {
    const int units = K / 32;
    spans = (units + 31) / 32;
    row_stride = spans * 256 + 32;
    d_stride = spans * 8 + 2;  // two groups' scales a load (8-byte aligned), the eight rows a load reads in other banks
    s_stride = spans * 8 * SUMS + 2;
    rows_at = 0;
    d_at = rows_at + size_t(R) * row_stride;
    s_at = d_at + size_t(R) * d_stride * 4;
    local_at = (s_at + (MINS ? size_t(R) * s_stride * 4 : 0) + 15) / 16 * 16;
    bytes = local_at + sizeof(typename SplitTables<F, BIG>::type);
  }
};

// COLS (rounds of up to 8 rows): the columns in the MMA's first operand, 16 a tile (lane g, t: columns g and g + 8,
// the same 8 weights of a group as with the rows there), the rows its second (the round's 8, row g a lane's): no
// MMA multiplies rows of zeros, and a lane's sums are its own columns', whose factors it decoded itself (no
// shuffles).  Every sum is a stream's of the same groups in the same order, so the bits are the same.
template <class F, int R, int NW, bool BIG, bool COLS = false>
__global__ void __launch_bounds__(32 * NW, 1)
k_round_blocks_split(Tables T, const int8_t* A, const float* dA, const float* sA, const uint8_t* __restrict__ W,
                     const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y, int ldy, int rows,
                     int accumulate, float* __restrict__ parts, int* __restrict__ count) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!round_tc_active<F, COLS ? 5 : (R == 8 ? 3 : 4)>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  static_assert(!COLS || R == 8, "the columns' kernel: rounds of up to 8 rows");
  // BPW: blocks a warp takes a span; E: the sums a lane holds a stream (rows g, g + 8 with R 16; columns 2 t, 2 t + 1;
  // COLS: columns g, g + 8, rows 2 t, 2 t + 1); TW: a tile's columns
  constexpr int GROUPS = F::kValues / 32, BPW = 8 / GROUPS, SUB = F::kSub, E = COLS ? 4 : R / 4, TW = COLS ? 16 : 8;
  // Spans of the warp's weights in registers, loads one span ahead.  Three (two ahead) where a span is one block of 256
  // values lost on the RTX 5090: alone some layers' launches were faster (IQ4_XS, IQ3_S), others slower (Q5_K, Q6_K),
  // but rounds were faster with two for every type (tree_bench, 16 rows: UD-IQ3_XXS 15.84 -> 15.23 ms, UD-Q4_K_XL
  // 18.63 -> 17.81).
  constexpr int SLOTS = 2;
  using Lay = SplitLayout<F, R, BIG>;
  constexpr bool MINS = Lay::MINS;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const Lay lay(K);
  int8_t* srows = reinterpret_cast<int8_t*>(smem_raw + lay.rows_at);
  float* sd = reinterpret_cast<float*>(smem_raw + lay.d_at);
  float* ss = reinterpret_cast<float*>(smem_raw + lay.s_at);
  auto& local = *reinterpret_cast<typename SplitTables<F, BIG>::type*>(smem_raw + lay.local_at);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int q = blockIdx.x % kSplitSets, m = blockIdx.x / kSplitSets, M = gridDim.x / kSplitSets;
  const int units = K / 32, spans = lay.spans, blocks = K / F::kValues, tiles = (N + TW - 1) / TW;
  // The warp's tiles: first, first + stride, ...; its items: (tile, span) in that order.
  const int first = m * NW + warp, stride = M * NW;
  const int my_tiles = first < tiles ? (tiles - first + stride - 1) / stride : 0, items = my_tiles * spans;
  constexpr int NC = COLS ? 2 : 1;  // the columns a lane decodes: g (and g + 8)
  typename F::Raw raw[SLOTS][NC][BPW];
  float rs[SLOTS][NC][BPW][2];
  // Item (ti, j): the warp's tile ti, span j.
  auto load = [&](int ti, int j, typename F::Raw (*r)[BPW], float (*s)[BPW][2]) {
#pragma unroll
    for (int h = 0; h < NC; ++h) {
      const int n = min((first + ti * stride) * TW + 8 * h + g, N - 1);
      const uint8_t* wn = W + size_t(n) * blocks * F::kStride;
      const uint16_t* dn = D + size_t(n) * blocks * F::kScales;
#pragma unroll
      for (int bi = 0; bi < BPW; ++bi) {
        const int b0 = (32 * j + 8 * q) / GROUPS + bi, b = b0 < blocks ? b0 : blocks - 1;
        r[h][bi] = block_load<F>(wn, dn, size_t(b), t);
        s[h][bi][0] = scale_of<F>(dn, size_t(b) * F::kScales);
        s[h][bi][1] = F::kScales > 1 ? scale_of<F>(dn, size_t(b) * F::kScales + 1) : 0.f;
      }
    }
  };
  // The weights do not depend on the kernel before: the first ones on their way before the wait.
#pragma unroll
  for (int k = 0; k + 1 < SLOTS; ++k)
    if (k < items) load(k / spans, k % spans, raw[k], rs[k]);
  SplitTables<F, BIG>::fill(T, local);
  pdl_wait();  // the rows come from the kernel before
  // The set's rows, once: span j's groups 8 q .. 8 q + 7 (their first halves, then their second, as k_stage_q8_round
  // lays a span out: round_q8_at); absent rows and groups past K: zeros (their sums are never written).
  for (int i = tid; i < R * spans * 16; i += 32 * NW) {
    const int r = i / (spans * 16), rest = i - r * spans * 16, j = rest >> 4, c = rest & 15, s = c & 7;
    const int n_s = min(32, units - 32 * j), u = 8 * q + s;
    const int at = 32 * j * 32 + (c < 8 ? 16 * u : 16 * n_s + 16 * u);
    const bool in = r < rows && u < n_s;
    cp_async16(srows + size_t(r) * lay.row_stride + j * 256 + 32 * s + (c < 8 ? 0 : 16),
               A + size_t(in ? r : 0) * K + (in ? at : 0), in);
  }
  for (int i = tid; i < R * spans * 8; i += 32 * NW) {
    const int r = i / (spans * 8), rest = i - r * spans * 8, j = rest >> 3, s = rest & 7;
    const int u = 32 * j + 8 * q + s;
    const bool in = r < rows && u < units;
    cp_async4(sd + r * lay.d_stride + rest, dA + (in ? size_t(r) * units + u : 0), in);
    if constexpr (MINS) {
      float2 v = make_float2(0.f, 0.f);
      if (in) v = *reinterpret_cast<const float2*>(sA + (size_t(r) * units + u) * 2);
      if constexpr (Lay::SUMS == 2) {
        *reinterpret_cast<float2*>(ss + r * lay.s_stride + 2 * rest) = v;
      } else {
        ss[r * lay.s_stride + rest] = v.x + v.y;
      }
    }
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float acc[8][E];
  // Span j of the warp's tile: its 8 streams, its weights in r, sc (COLS: columns g and g + 8).
  auto compute = [&](int j, const typename F::Raw (*r)[BPW], const float (*sc)[BPW][2]) {
#pragma unroll
    for (int bi = 0; bi < BPW; ++bi) {
      if ((32 * j + 8 * q) / GROUPS + bi >= blocks) continue;  // past K: the same for the whole warp
      typename F::Lane ln[NC];
#pragma unroll
      for (int h = 0; h < NC; ++h) ln[h] = F::share(r[h][bi], t, lane);
      float2 dpair[2];
#pragma unroll
      for (int G = 0; G < GROUPS; ++G) {
        const int s = bi * GROUPS + G, col = j * 8 + s;  // the warp's stream; the group among the set's
        uint32_t w[NC][2];
        float f[NC], mn[NC];
#pragma unroll
        for (int h = 0; h < NC; ++h) {
          F::codes(T, local, ln[h], G, w[h]);
          F::factors(ln[h], G, sc[h][bi][0], sc[h][bi][1], &f[h], &mn[h]);
        }
        if constexpr (COLS) {
          // A column's factors are the lane's own (SUB 16: lanes t < 2 decode a group's first 16 weights, t >= 2 its
          // last; the other half's factor from lane t ^ 2, which decodes the same column).
          float fc[2][2], mc[2][2];
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            if constexpr (SUB == 16) {
              const float of = __shfl_xor_sync(0xffffffffu, f[c], 2);
              fc[c][0] = t < 2 ? f[c] : of;
              fc[c][1] = t < 2 ? of : f[c];
              if constexpr (MINS) {
                const float om = __shfl_xor_sync(0xffffffffu, mn[c], 2);
                mc[c][0] = t < 2 ? mn[c] : om;
                mc[c][1] = t < 2 ? om : mn[c];
              }
            } else {
              fc[c][0] = f[c];
              if constexpr (MINS) mc[c][0] = mn[c];
            }
          }
          // Row g's values (the MMA's second operand); rows 2 t and 2 t + 1's scales and sums (its sums' rows).
          const uint2 x0 =
              *reinterpret_cast<const uint2*>(srows + size_t(g) * lay.row_stride + j * 256 + 32 * s + 8 * t);
          if ((G & 1) == 0 && GROUPS > 1) {
            dpair[0] = *reinterpret_cast<const float2*>(sd + (2 * t) * lay.d_stride + col);
            dpair[1] = *reinterpret_cast<const float2*>(sd + (2 * t + 1) * lay.d_stride + col);
          }
          float dr[2];
          float2 sr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
#pragma unroll
          for (int k = 0; k < 2; ++k) {
            dr[k] = GROUPS > 1 ? ((G & 1) ? dpair[k].y : dpair[k].x) : sd[(2 * t + k) * lay.d_stride + col];
            if constexpr (MINS) {
              if constexpr (Lay::SUMS == 2)
                sr[k] = *reinterpret_cast<const float2*>(ss + (2 * t + k) * lay.s_stride + 2 * col);
              else
                sr[k].x = ss[(2 * t + k) * lay.s_stride + col];
            }
          }
          const uint32_t b[2] = {x0.x, x0.y};
          int c0[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          int c1[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          if constexpr (SUB == 32) {
            const uint32_t a[4] = {w[0][0], w[1][0], w[0][1], w[1][1]};
            mma_s8_k32(c0, a, b);
          } else {  // the first 16 weights are lanes 0, 1's, the last 16 lanes 2, 3's
            const bool lo = t < 2;
            const uint32_t a_lo[4] = {lo ? w[0][0] : 0u, lo ? w[1][0] : 0u, lo ? w[0][1] : 0u, lo ? w[1][1] : 0u};
            const uint32_t a_hi[4] = {lo ? 0u : w[0][0], lo ? 0u : w[1][0], lo ? 0u : w[0][1], lo ? 0u : w[1][1]};
            mma_s8_k32(c0, a_lo, b);
            mma_s8_k32(c1, a_hi, b);
          }
#pragma unroll
          for (int e = 0; e < E; ++e) {
            const int c = e >> 1, k = e & 1;  // column g (+ 8 c), row 2 t + k
            float& v = acc[s][e];
            v = fmaf(__int_as_float(c0[e]) - kMagic, dr[k] * fc[c][0], v);
            if constexpr (MINS) {
              v = fmaf(sr[k].x, mc[c][0], v);  // SUB 32: the two sums as the other kernels add them
            }
            if constexpr (SUB == 16) {
              v = fmaf(__int_as_float(c1[e]) - kMagic, dr[k] * fc[c][1], v);
              if constexpr (MINS) v = fmaf(sr[k].y, mc[c][1], v);
            }
          }
        } else {
          float fc[2][2], mc[2][2];
#pragma unroll
          for (int h = 0; h < 32 / SUB; ++h) {
            const int o = SUB == 16 ? 2 * h : 0;
            fc[0][h] = __shfl_sync(0xffffffffu, f[0], 8 * t + o);
            fc[1][h] = __shfl_sync(0xffffffffu, f[0], 8 * t + 4 + o);
            if constexpr (MINS) {
              mc[0][h] = __shfl_sync(0xffffffffu, mn[0], 8 * t + o);
              mc[1][h] = __shfl_sync(0xffffffffu, mn[0], 8 * t + 4 + o);
            }
          }
          const int8_t* row0 = srows + size_t(g) * lay.row_stride + j * 256 + 32 * s + 8 * t;
          const uint2 x0 = *reinterpret_cast<const uint2*>(row0);
          uint2 x8 = make_uint2(0u, 0u);
          // The rows' scales of groups s and s + 1 in one load, at the even one.
          if ((G & 1) == 0 && GROUPS > 1) {
            dpair[0] = *reinterpret_cast<const float2*>(sd + g * lay.d_stride + col);
            if constexpr (R == 16) dpair[1] = *reinterpret_cast<const float2*>(sd + (g + 8) * lay.d_stride + col);
          }
          float dr[2] = {GROUPS > 1 ? ((G & 1) ? dpair[0].y : dpair[0].x) : sd[g * lay.d_stride + col], 0.f};
          float2 sr[2] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
          if constexpr (MINS) {
            if constexpr (Lay::SUMS == 2) sr[0] = *reinterpret_cast<const float2*>(ss + g * lay.s_stride + 2 * col);
            else sr[0].x = ss[g * lay.s_stride + col];
          }
          if constexpr (R == 16) {
            x8 = *reinterpret_cast<const uint2*>(row0 + 8 * lay.row_stride);
            dr[1] = GROUPS > 1 ? ((G & 1) ? dpair[1].y : dpair[1].x) : sd[(g + 8) * lay.d_stride + col];
            if constexpr (MINS) {
              if constexpr (Lay::SUMS == 2)
                sr[1] = *reinterpret_cast<const float2*>(ss + (g + 8) * lay.s_stride + 2 * col);
              else
                sr[1].x = ss[(g + 8) * lay.s_stride + col];
            }
          }
          const uint32_t a[4] = {x0.x, x8.x, x0.y, x8.y};
          int c0[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          int c1[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          if constexpr (SUB == 32) {
            mma_s8_k32(c0, a, w[0]);
          } else {  // the first 16 weights are lanes 0, 1's, the last 16 lanes 2, 3's
            const uint32_t lo[2] = {t < 2 ? w[0][0] : 0u, t < 2 ? w[0][1] : 0u};
            const uint32_t hi[2] = {t < 2 ? 0u : w[0][0], t < 2 ? 0u : w[0][1]};
            mma_s8_k32(c0, a, lo);
            mma_s8_k32(c1, a, hi);
          }
#pragma unroll
          for (int e = 0; e < E; ++e) {
            const int h = e >> 1, c = e & 1;  // row g (+ 8 h), column 2 t + c
            float& v = acc[s][e];
            v = fmaf(__int_as_float(c0[e]) - kMagic, dr[h] * fc[c][0], v);
            if constexpr (MINS) {
              v = fmaf(sr[h].x, mc[c][0], v);  // SUB 32: the two sums as the other kernels add them
            }
            if constexpr (SUB == 16) {
              v = fmaf(__int_as_float(c1[e]) - kMagic, dr[h] * fc[c][1], v);
              if constexpr (MINS) v = fmaf(sr[h].y, mc[c][1], v);
            }
          }
        }
      }
    }
  };
  // The tile's end: the warp's streams added up (xor 1, 2, 4) into the scratch; the last set's warp adds the sets.
  auto finish = [&](int tile) {
    float p[E];
#pragma unroll
    for (int e = 0; e < E; ++e)
      p[e] = ((acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e])) + ((acc[4][e] + acc[5][e]) + (acc[6][e] + acc[7][e]));
    float* slot = parts + (size_t(tile) * kSplitSets * 32 + lane) * E;
#pragma unroll
    for (int e = 0; e < E; ++e) slot[size_t(q) * 32 * E + e] = p[e];
    __threadfence();
    __syncwarp();
    int arrived = 0;
    if (lane == 0) arrived = atomicAdd(count + tile, 1);
    arrived = __shfl_sync(0xffffffffu, arrived, 0);
    if (arrived != kSplitSets - 1) return;
    __threadfence();
    float set[kSplitSets][E];
#pragma unroll
    for (int o = 0; o < kSplitSets; ++o)
#pragma unroll
      for (int e = 0; e < E; ++e) set[o][e] = o == q ? p[e] : __ldcg(slot + size_t(o) * 32 * E + e);
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const float v = (set[0][e] + set[1][e]) + (set[2][e] + set[3][e]);
      const int r = COLS ? 2 * t + (e & 1) : g + 8 * (e >> 1);
      const int col = COLS ? tile * TW + g + 8 * (e >> 1) : tile * TW + 2 * t + (e & 1);
      if (r < rows && col < N) {
        float* dst = y + size_t(r) * ldy + col;
        *dst = accumulate ? *dst + v : v;
      }
    }
    if (lane == 0) count[tile] = 0;
  };
  static_assert(kSplitSets == 4, "the sets' tree above is for four");
  // The item's tile and span, and the loaded item's, as running counters: an item's two divisions by spans took
  // 2 to 5% of the rounds of 16 rows of the IQ types.
  int ti = 0, j = 0, tl = (SLOTS - 1) / spans, jl = (SLOTS - 1) % spans;
#pragma unroll 1
  for (int i0 = 0; i0 < items; i0 += SLOTS) {
#pragma unroll
    for (int k = 0; k < SLOTS; ++k) {
      const int i = i0 + k;
      if (i >= items) break;
      if (j == 0) {
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
          for (int e = 0; e < E; ++e) acc[s][e] = 0.f;
      }
      if (i + SLOTS - 1 < items) load(tl, jl, raw[(k + SLOTS - 1) % SLOTS], rs[(k + SLOTS - 1) % SLOTS]);
      if (++jl == spans) jl = 0, ++tl;
      compute(j, raw[k], rs[k]);
      if (j == spans - 1) finish(first + ti * stride);
      if (++j == spans) j = 0, ++ti;
    }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- rounds on the tensor cores, the streams split among a block's warps ------------------------------------
// The same sums once more, with no shared memory until a tile's end.  A block of threads takes a tile of 16 columns
// at a time (the blocks stay and take the tiles in turn) and its 32 / Q warps Q streams each: warp p the streams
// Q p .. Q p + Q - 1, which are groups G0 .. G0 + Q - 1 of block q = Q p / 8 of every span (G0 = Q p % 8; blocks
// of 256).  The columns are the MMA's first operand as in
// k_round_blocks_cols (lane g, t: columns g and g + 8), the rows its second (8 a pack, two packs for 16 rows, row g a
// lane's), read where k_stage_q8_round left them (two 4-byte loads a row and group: the first-level cache holds the
// rows for the tile's warps).  A lane keeps its Q streams' sums of its outputs (columns g, g + 8 by rows 2 t, 2 t + 1
// of each pack): 16 floats with Q 4 and 16 rows, so many warps fit.  It adds them by the tree's first levels (xor 1,
// 2 with Q 4), the block's warps then by its last levels, in the warps' order, through shared memory.  A lane's
// bytes of a span are together in the type's payload (IQ4_XS lane-major: its codes 4 t .. 4 t + 3 of Q groups in one
// load), decoded in registers.  The same bits as k_round_blocks_q8: every sum is a stream's of the same groups in the
// same order, the same tree.  RTX 5070 Ti, IQ4_XS 12288 x 5120: 8 rows 45 us (the rules before: 59-69), 16 rows 54-65
// (85-95)

// A word's eight 4-bit codes through IQ4's 16 values (tb: four words, value i in byte i): lo = the values of the low
// nibbles of its bytes 0..3, hi = of the high nibbles.  byte_perm takes a nibble a byte; its bit 3 (which byte_perm
// reads as "replicate the sign") is overruled by a second byte_perm that takes such a byte from the table's upper
// half.  The idea is llama.cpp's (get_int_from_table_16; docs/prior-art.md).
__device__ __forceinline__ void iq4_codes(uint32_t x, const uint32_t* tb, uint32_t* lo, uint32_t* hi) {
  const uint32_t pick = 0x32103210u | ((x & 0x88888888u) >> 1);  // a result byte from the lower or the upper half
  uint32_t v[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t sel = x >> (16 * i);
    const uint32_t l = __byte_perm(tb[0], tb[1], sel), h = __byte_perm(tb[2], tb[3], sel);
    v[i] = __byte_perm(l, h, pick >> (16 * i));
  }
  // v[i]: the values of the nibbles of bytes 2 i, 2 i + 1 in nibble order (lo, hi, lo, hi)
  *lo = __byte_perm(v[0], v[1], 0x6420);
  *hi = __byte_perm(v[0], v[1], 0x7531);
}

// A type's decoding for k_round_blocks_warps (kTakes false: the kernel does not take it; kernel 5 runs the split
// kernel instead).  Raw<Q>: what a lane loads of a block for groups G0 .. G0 + Q - 1 of its column (load); codes:
// group G0 + i's two words of the column, w[0] for the MMA's k slots 4 t .. 4 t + 3, w[1] for 16 + 4 t .. 16 + 4 t + 3;
// factors: the group's f and m (two of each for sub-groups of 16), as the type's factors give them.  kRows8t: which
// values of a group a lane's slots hold, so which of a row's the kernel loads for them: values 8 t .. 8 t + 7 (the
// other tensor-core kernels' order, one 8-byte load from k_stage_q8_round's layout), or 4 t .. 4 t + 3 and
// 16 + 4 t .. 16 + 4 t + 3 (IQ4_XS: a code byte holds values j and j + 16; two 4-byte loads).
template <class F>
struct WarpsType {
  static constexpr bool kTakes = false;
};
template <>
struct WarpsType<FIQ4_XS> {
  static constexpr bool kTakes = true, kRows8t = false;
  template <int Q>
  struct Raw {
    uint32_t q[Q];
    uint2 head;
  };
  template <int Q>
  static __device__ __forceinline__ Raw<Q> load(const uint8_t* W, const uint16_t* D, size_t at, int G0, int t) {
    Raw<Q> r;
    const uint8_t* p = W + at * FIQ4_XS::kStride + 32 * t + 4 * G0;  // lane-major: one load
    if constexpr (Q == 8) {
      put(r.q, ld128(p));
      put(r.q + 4, ld128(p + 16));
    } else if constexpr (Q == 4) {
      put(r.q, ld128(p));
    } else {
      static_assert(Q == 2, "IQ4_XS: 2, 4 or 8 streams a warp");
      put(r.q, ld64(p));
    }
    r.head = *reinterpret_cast<const uint2*>(D + at * FIQ4_XS::kScales);
    return r;
  }
  template <int Q>
  static __device__ __forceinline__ void codes(const Tables& T, const Raw<Q>& r, int i, int, uint32_t* w) {
    iq4_codes(r.q[i], T.iq4_s8, &w[0], &w[1]);
  }
  template <int Q>
  static __device__ __forceinline__ void factors(const Raw<Q>& r, int, int G, float* f, float* m) {  // FIQ4_XS's
    const uint32_t s6 = ((byte_of(r.head.y, G >> 1) >> (4 * (G & 1))) & 15u) | ((r.head.x >> (16 + 2 * G)) & 3u) << 4;
    f[0] = half_of(r.head.x & 0xFFFFu) * float(int(s6) - 32), m[0] = 0.f;
  }
};
// Q4_K and Q5_K (FIVE: the fifth code bits): the nibbles of the group pair c (low: group 2 c) at QS + 32 c, a lane's
// 8 bytes of each pair it needs; Q5_K's fifth bits (bit G of byte j: value j of group G) at 16, a lane's 8 bytes; the
// scales and minimums packed in the block's first 16 bytes; d and dmin in the scales plane.
template <class F, bool FIVE, int QS>
struct WarpsTypeK4 {
  static constexpr bool kTakes = true, kRows8t = true;
  template <int Q>
  struct Raw {
    uint32_t q[Q], high[2], groups[Q / 2];
    float d0, d1;
  };
  template <int Q>
  static __device__ __forceinline__ Raw<Q> load(const uint8_t* W, const uint16_t* D, size_t at, int G0, int t) {
    static_assert(Q % 4 == 0, "the K types: whole halves of a block a warp");
    Raw<Q> r;
    const uint8_t* p = W + at * F::kStride;
    uint32_t head[4];
    put(head, ld128(p));
#pragma unroll
    for (int k = 0; k < Q / 2; ++k) {
      const int c = G0 / 2 + k;
      put(r.q + 2 * k, ld64(p + QS + 32 * c + 8 * t));
      r.groups[k] = k_groups(head, c);
    }
    if constexpr (FIVE) put(r.high, ld64(p + 16 + 8 * t));
    else r.high[0] = r.high[1] = 0u;
    r.d0 = scale_at(D, at * F::kScales), r.d1 = scale_at(D, at * F::kScales + 1);
    return r;
  }
  template <int Q>
  static __device__ __forceinline__ void codes(const Tables&, const Raw<Q>& r, int i, int G, uint32_t* w) {  // F's
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      w[k] = (r.q[2 * (i >> 1) + k] >> (4 * (G & 1))) & 0x0F0F0F0Fu;
      if constexpr (FIVE) w[k] |= ((r.high[k] >> G) & 0x01010101u) << 4;
    }
  }
  template <int Q>
  static __device__ __forceinline__ void factors(const Raw<Q>& r, int i, int G, float* f, float* m) {  // F's
    const uint32_t g = r.groups[i >> 1];
    f[0] = r.d0 * float(byte_of(g, G & 1)), m[0] = -r.d1 * float(byte_of(g, 2 + (G & 1)));
  }
};
template <>
struct WarpsType<FQ4_K> : WarpsTypeK4<FQ4_K, false, 16> {};
template <>
struct WarpsType<FQ5_K> : WarpsTypeK4<FQ5_K, true, 48> {};
// Q6_K: the groups of a half block h (4 of them: Q 4) take its low nibbles' rows 2 h and 2 h + 1 (groups c and c + 2
// in one byte, groups c & 1 in the two rows; pair-major: a lane's 8 bytes of each in one load), its high bits at
// 128 + 32 h (two a group), the signed scales of their 8 sub-groups of 16 at 192 + 8 h; d in the scales plane.
template <>
struct WarpsType<FQ6_K> {
  static constexpr bool kTakes = true, kRows8t = true;
  template <int Q>
  struct Raw {
    uint32_t low[4], high[2], scales[2];
    float d0;
  };
  template <int Q>
  static __device__ __forceinline__ Raw<Q> load(const uint8_t* W, const uint16_t* D, size_t at, int G0, int t) {
    static_assert(Q == 4, "Q6_K: a half block a warp");
    Raw<Q> r;
    const uint8_t* p = W + at * FQ6_K::kStride;
    const int h = G0 >> 2;
    put(r.low, ld128(p + 64 * h + 16 * t));
    put(r.high, ld64(p + 128 + 32 * h + 8 * t));
    put(r.scales, ld64(p + 192 + 8 * h));
    r.d0 = scale_at(D, at * FQ6_K::kScales);
    return r;
  }
  template <int Q>
  static __device__ __forceinline__ void codes(const Tables&, const Raw<Q>& r, int, int G, uint32_t* w) {  // FQ6_K's
    const int c = G & 3;
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const uint32_t x = (r.low[2 * (c & 1) + k] >> (4 * (c >> 1))) & 0x0F0F0F0Fu;
      w[k] = __vsub4(x | ((r.high[k] >> (2 * c)) & 0x03030303u) << 4, 0x20202020u);
    }
  }
  template <int Q>
  static __device__ __forceinline__ void factors(const Raw<Q>& r, int i, int, float* f, float* m) {  // FQ6_K's
#pragma unroll
    for (int k = 0; k < 2; ++k) {
      const int s = 2 * i + k;  // of the half block's
      f[k] = r.d0 * float(int(int8_t(byte_of(r.scales[s >> 2], s & 3)))), m[k] = 0.f;
    }
  }
};

// The first levels of the streams' tree over a lane's Q neighbouring streams.
template <int Q>
__device__ __forceinline__ float streams_tree(const float* s) {
  if constexpr (Q == 1) return s[0];
  else if constexpr (Q == 2) return s[0] + s[1];
  else if constexpr (Q == 4) return (s[0] + s[1]) + (s[2] + s[3]);
  else return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));
}

template <class F, int R, int Q>
__global__ void __launch_bounds__(32 * (32 / Q))
k_round_blocks_warps(Tables T, const int8_t* A, const float* dA, const float* sA, const uint8_t* __restrict__ W,
                     const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y, int ldy, int rows,
                     int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!round_tc_active<F, R == 8 ? 6 : 7>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  using WT = WarpsType<F>;
  static_assert(WT::kTakes && F::kValues == 256 && Q % 4 == 0, "k_round_blocks_warps: blocks of 256, 4 or 8 streams");
  constexpr int PARTS = 32 / Q, PACKS = R / 8, E = 4 * PACKS, SUB = F::kSub;
  constexpr bool MINS = F::kFlatRun != 0;  // the types with a minimum
  // The other warps' trees of their streams for the first warp, a lane's outputs (lane-fastest: no bank shares).
  // As little as that: the shared memory a block takes is first-level cache the rows lose (RTX 5070 Ti, 16 rows of
  // 12288 x 5120: 19 KB a block, 108 us; 10 KB, 57 us).
  __shared__ float sums[PARTS - 1][E][32];
  const int lane = threadIdx.x & 31, part = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int q = Q * part / 8, G0 = Q * part % 8;
  const int units = K / 32, blocks = K / F::kValues, spans = (units + 31) / 32, tiles = (N + 15) / 16;
  // A column's first block (a column past N reads the last one's and writes nothing).
  auto col_at = [&](int tile, int h) { return size_t(min(tile * 16 + g + 8 * h, N - 1)) * blocks; };
  typename WT::template Raw<Q> w[2];
  // The first tile's first weights do not depend on the kernel before: on their way before the wait.
  if (blockIdx.x < tiles && q < blocks) {
#pragma unroll
    for (int h = 0; h < 2; ++h) w[h] = WT::template load<Q>(W, D, col_at(blockIdx.x, h) + q, G0, t);
  }
  pdl_wait();  // the rows come from the kernel before
  // The blocks stay and take the tiles in turn (round_grid).
#pragma unroll 1
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    float acc[Q][PACKS][4];
#pragma unroll
    for (int i = 0; i < Q; ++i)
#pragma unroll
      for (int P = 0; P < PACKS; ++P)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[i][P][e] = 0.f;
    const size_t at0 = col_at(tile, 0), at1 = col_at(tile, 1);
#pragma unroll 1
    for (int j = 0, b = q; j < spans && b < blocks; ++j, b += 4) {
      if (j > 0 || tile != int(blockIdx.x)) {
        w[0] = WT::template load<Q>(W, D, at0 + b, G0, t);
        w[1] = WT::template load<Q>(W, D, at1 + b, G0, t);
      }
      // Rows 2 t, 2 t + 1 of each pack: the scales of the span's Q groups and (a minimum) their sums as the other
      // kernels add them, in vector loads (a group's own loads: 5 a pack and group with a minimum, most of the time).
      float dq[PACKS][2][Q];
      float2 sq[PACKS][2][MINS ? Q : 1];
#pragma unroll
      for (int P = 0; P < PACKS; ++P)
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int r = 8 * P + 2 * t + k;
          const bool in = r < rows;
          const size_t at = size_t(in ? r : 0) * units + 8 * b + G0;
#pragma unroll
          for (int v = 0; v < Q / 4; ++v) {
            const float4 x = in ? reinterpret_cast<const float4*>(dA + at)[v] : make_float4(0.f, 0.f, 0.f, 0.f);
            dq[P][k][4 * v] = x.x, dq[P][k][4 * v + 1] = x.y, dq[P][k][4 * v + 2] = x.z, dq[P][k][4 * v + 3] = x.w;
          }
          if constexpr (MINS) {
#pragma unroll
            for (int v = 0; v < Q / 2; ++v) {
              const float4 x =
                  in ? reinterpret_cast<const float4*>(sA + 2 * at)[v] : make_float4(0.f, 0.f, 0.f, 0.f);
              sq[P][k][2 * v] = SUB == 16 ? make_float2(x.x, x.y) : make_float2(x.x + x.y, 0.f);
              sq[P][k][2 * v + 1] = SUB == 16 ? make_float2(x.z, x.w) : make_float2(x.z + x.w, 0.f);
            }
          } else {
            sq[P][k][0] = make_float2(0.f, 0.f);
          }
        }
#pragma unroll
      for (int i = 0; i < Q; ++i) {
        const int G = G0 + i, u = 8 * b + G;
        uint32_t w0[2], w1[2];  // columns g and g + 8
        WT::template codes<Q>(T, w[0], i, G, w0);
        WT::template codes<Q>(T, w[1], i, G, w1);
        const uint32_t a[4] = {w0[0], w1[0], w0[1], w1[1]};
        float f0[2], m0[2], f1[2], m1[2];
        WT::template factors<Q>(w[0], i, G, f0, m0);
        WT::template factors<Q>(w[1], i, G, f1, m1);
        // Sub-groups of 16: the MMA's slots that hold a group's first 16 values, and its last.
        uint32_t a_lo[4] = {a[0], a[1], a[2], a[3]}, a_hi[4] = {0u, 0u, 0u, 0u};
        if constexpr (SUB == 16) {
          const bool first = !WT::kRows8t || t < 2;  // values 8 t ..: lanes t < 2 the first 16; else slots 4 t ..
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const bool in_first = WT::kRows8t ? first : k < 2;
            a_lo[k] = in_first ? a[k] : 0u;
            a_hi[k] = in_first ? 0u : a[k];
          }
        }
        // A row's values for the lane's slots, where k_stage_q8_round left them.
        const int ra0 = WT::kRows8t ? round_q8_at(u, units, t >> 1) + 8 * (t & 1) : round_q8_at(u, units, 0) + 4 * t;
        const int ra1 = round_q8_at(u, units, 1) + 4 * t;
#pragma unroll
        for (int P = 0; P < PACKS; ++P) {
          const int rg = 8 * P + g;
          uint32_t b2[2] = {0u, 0u};
          if (rg < rows) {
            const int8_t* row = A + size_t(rg) * K;
            if constexpr (WT::kRows8t) {
              const uint2 v = *reinterpret_cast<const uint2*>(row + ra0);
              b2[0] = v.x, b2[1] = v.y;
            } else {
              b2[0] = *reinterpret_cast<const uint32_t*>(row + ra0);
              b2[1] = *reinterpret_cast<const uint32_t*>(row + ra1);
            }
          }
          const float dr[2] = {dq[P][0][i], dq[P][1][i]};
          const float2 sr[2] = {sq[P][0][MINS ? i : 0], sq[P][1][MINS ? i : 0]};
          int c0[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          int c1[4] = {kMagicBits, kMagicBits, kMagicBits, kMagicBits};
          mma_s8_k32(c0, a_lo, b2);
          if constexpr (SUB == 16) mma_s8_k32(c1, a_hi, b2);
#pragma unroll
          for (int e = 0; e < 4; ++e) {  // column g (+ 8 (e >> 1)), row 2 t + (e & 1): the split kernel's steps
            const int k = e & 1;
            const bool g8 = e >> 1;
            float& v = acc[i][P][e];
            v = fmaf(__int_as_float(c0[e]) - kMagic, dr[k] * (g8 ? f1[0] : f0[0]), v);
            if constexpr (MINS) v = fmaf(sr[k].x, g8 ? m1[0] : m0[0], v);
            if constexpr (SUB == 16) {
              v = fmaf(__int_as_float(c1[e]) - kMagic, dr[k] * (g8 ? f1[1] : f0[1]), v);
              if constexpr (MINS) v = fmaf(sr[k].y, g8 ? m1[1] : m0[1], v);
            }
          }
        }
      }
    }
    float mine[E];  // the warp's streams by the tree's first levels
#pragma unroll
    for (int P = 0; P < PACKS; ++P)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float s4[Q];
#pragma unroll
        for (int i = 0; i < Q; ++i) s4[i] = acc[i][P][e];
        mine[4 * P + e] = streams_tree<Q>(s4);
      }
    if (part > 0)
#pragma unroll
      for (int e = 0; e < E; ++e) sums[part - 1][e][lane] = mine[e];
    __syncthreads();
    if (part == 0) {  // the parts by the tree's last levels, in the parts' order
#pragma unroll
      for (int e = 0; e < E; ++e) {
        float v[PARTS];
        v[0] = mine[e];
#pragma unroll
        for (int p = 1; p < PARTS; ++p) v[p] = sums[p - 1][e][lane];
#pragma unroll
        for (int width = PARTS; width > 1; width >>= 1)
#pragma unroll
          for (int p = 0; p < width / 2; ++p) v[p] = v[2 * p] + v[2 * p + 1];
        const int r = 8 * (e >> 2) + 2 * t + (e & 1), col = tile * 16 + g + 8 * ((e >> 1) & 1);
        if (r < rows && col < N) {
          float* dst = y + size_t(r) * ldy + col;
          *dst = accumulate ? *dst + v[0] : v[0];
        }
      }
    }
    __syncthreads();  // the sums read before the next tile's are written
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// The dynamic shared memory a staged block of the dp4a kernel takes for a round of at most R rows of K values (and
// runs after the first of types Fn, Fn2).
template <class F, int R, class Fn = void, class Fn2 = void>
size_t round_rows_bytes(int K) {
  constexpr bool mins = kTypeMins<F> || kTypeMins<Fn> || kTypeMins<Fn2>;
  return size_t(R) * K + size_t(R) * (K / 32) * 4 + (mins ? size_t(R) * (K / 32) * 8 : 0);
}

// The SwiGLU a launch of the dp4a kernel on W into y (rows of ldy) writes as well (Launch::glu): none (gu null) without
// one, and in the gate's launch of a layer whose gate and up are runs of their own (the up's launch writes it).
inline RoundGluArgs round_glu_args(const Launch& L, const BlockMat& W, const float* y, int ldy, int rows,
                                   bool accumulate) {
  RoundGluArgs g;
  if (!L.glu) return g;
  const Scratch& sc = scratch(L);
  const RoundGlu& u = *L.glu;
  const ptrdiff_t col0 = y - u.gu;
  const bool gate = col0 == 0 && W.rows == u.I, both = col0 == 0 && W.rows == 2 * u.I,
             up = col0 == u.I && W.rows == u.I;
  if (ldy != u.ld || !(gate || both || up) || accumulate || u.I % 64 || rows < 1 || rows > kRoundGluRows ||
      size_t(rows) * u.I > sc.q8_round_cap) {
    std::fprintf(stderr, "flyx-cuda: a round's SwiGLU in its GEMM takes 1..%d rows of the whole gate|up layer or of "
                         "its gate or up run, I %% 64 == 0, no accumulation (I %d, rows %d)\n", kRoundGluRows, u.I,
                 rows);
    std::abort();
  }
  if (gate) return g;
  g.gu = u.gu;
  g.ld = u.ld;
  g.I = u.I;
  g.col0 = int(col0);
  g.act = u.act;
  g.o.q8 = sc.q8_glu;
  g.o.q8_d = sc.q8_glu_scale;
  g.o.q8_sums = sc.q8_glu_sums;
  g.o.q8_k = u.I;
  return g;
}

// A launch of the dp4a kernel that writes a SwiGLU too: as launch_round's, a block a group of 32 outputs at a time.
template <class F, int R, int C, int U, int NW, int S>
void launch_round_glu(const BlockMat& W, int K, float* y, int ldy, int rows, const Launch& L, const RoundGluArgs& glu) {
  const Scratch& sc = scratch(L);
  const int groups = glu.I / 32;
  if constexpr (S != 0) {
    static const int limit = optin_dynamic_smem(k_round_blocks_q8<F, R, C, U, NW, 1, 1, void, void, true>);
    const size_t smem = round_rows_bytes<F, R>(K);
    if (smem <= size_t(limit)) {
      static ResidentCache cache;
      const int resident = cache.get(k_round_blocks_q8<F, R, C, U, NW, 1, 1, void, void, true>, 32 * NW, smem);
      launch_dependent_smem(k_round_blocks_q8<F, R, C, U, NW, 1, 1, void, void, true>,
                            dim3(balanced_grid(groups, 1, resident)), dim3(32 * NW), smem, L, block_tables(), sc.q8,
                            sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows, 0, RoundMore{}, glu);
      return;
    }
  }
  constexpr int GW = S ? kRoundWarps : NW;
  static const int resident = resident_blocks(k_round_blocks_q8<F, R, C, U, GW, 0, 1, void, void, true>, 32 * GW, 0);
  launch_dependent(k_round_blocks_q8<F, R, C, U, GW, 0, 1, void, void, true>, dim3(balanced_grid(groups, 1, resident)),
                   dim3(32 * GW), L, block_tables(), sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y,
                   ldy, rows, 0, RoundMore{}, glu);
}

// A configuration of the dp4a kernel (variants: blocks_round): R rows held, columns and groups ahead as it names them,
// NW warps a block, and S: the rows staged in shared memory where they fit (where they do not, K too long: the same
// configuration on 8 warps reading them from global memory, which gives the same bits).  Fn, Fn2: the types of the
// runs after the first in the same launch (more), or void.  The grid's turns: ceil(N / C) of the first run, a column
// each of the others'.
template <class F, int R, int CC, int CU, int NW, int S, int P = 1, class Fn = void, class Fn2 = void>
void launch_round(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L,
                  RoundMore more = {}) {
  // static: the launch-once lambdas below name them in template arguments (MSVC treats a captured constant as a
  // variable there)
  static constexpr int C = kRoundColumns<F, CC>, U = kUnitsAhead<F, CC, CU>;
  const Scratch& sc = scratch(L);
  const int turns = ceil_div(W.rows, C) + more.run[0].N + more.run[1].N;
  if (more.run[0].N && L.glu) {  // the SwiGLU's launches are round_glu's (gemm_blocks_round_glu.cu); a stack takes none
    std::fprintf(stderr, "flyx-cuda: a round's SwiGLU in a stack\n");
    std::abort();
  }
  if constexpr (S == 2) {
    constexpr size_t smem = 2 * round_window_bytes<F, R>(F::kFlatRun != 0);
    static const int resident = [&] {
      set_dynamic_smem(k_round_blocks_q8<F, R, C, U, NW, 2, P, Fn, Fn2>, smem);
      return resident_blocks(k_round_blocks_q8<F, R, C, U, NW, 2, P, Fn, Fn2>, 32 * NW, smem);
    }();
    const int grid = balanced_grid(turns, NW / P, resident);
    launch_dependent_smem(k_round_blocks_q8<F, R, C, U, NW, 2, P, Fn, Fn2>, dim3(grid), dim3(32 * NW), smem, L,
                          block_tables(), sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows,
                          int(accumulate), more, RoundGluArgs{});
    return;
  } else if constexpr (S != 0) {
    static const int limit = optin_dynamic_smem(k_round_blocks_q8<F, R, C, U, NW, 1, P, Fn, Fn2>);
    const size_t smem = round_rows_bytes<F, R, Fn, Fn2>(K);
    if (smem <= size_t(limit)) {
      static ResidentCache cache;
      const int resident = cache.get(k_round_blocks_q8<F, R, C, U, NW, 1, P, Fn, Fn2>, 32 * NW, smem);
      const int grid = balanced_grid(turns, NW / P, resident);
      launch_dependent_smem(k_round_blocks_q8<F, R, C, U, NW, 1, P, Fn, Fn2>, dim3(grid), dim3(32 * NW), smem, L,
                            block_tables(), sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows,
                            int(accumulate), more, RoundGluArgs{});
      return;
    }
  }
  constexpr int GW = S ? kRoundWarps : NW;  // the rows from global memory (P divides 8 as it divides NW)
  static const int resident = resident_blocks(k_round_blocks_q8<F, R, C, U, GW, 0, P, Fn, Fn2>, 32 * GW, 0);
  const int grid = balanced_grid(turns, GW / P, resident);
  launch_dependent(k_round_blocks_q8<F, R, C, U, GW, 0, P, Fn, Fn2>, dim3(grid), dim3(32 * GW), L, block_tables(),
                   sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows, int(accumulate), more,
                   RoundGluArgs{});
}

template <class F>
void launch_round_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  constexpr int di = tables::ggml_type_index(F::kDtype);
  switch (cfg) {
#define X(i, R, C, U, NW, S, P)                                                                                  \
  case i:                                                                                                        \
    if constexpr (FLYX_TUNE_CANDIDATES || (di >= 0 && (tables::kReachRoundDp4a[tables::kArchBucketAny][di] &     \
                                                       (1u << i)))) {                                            \
      return launch_round<F, R, C, U, NW, S, P>(W, K, y, ldy, rows, accumulate, L);                              \
    }                                                                                                            \
    break;
    FLYX_BLOCKS_ROUND(X)
#undef X
    default:
      break;
  }
  std::fprintf(stderr, "flyx-cuda: no round configuration %d\n", cfg);
  std::abort();
}

// Tiles of 8 columns a block of the tensor-core kernel takes at once: every block reads the rows again for its
// columns, so a block takes as many as its registers allow (4: 16 warps of 128 registers fill an SM; at 16 rows on
// the 27B's layers 4 ran 1.3 to 1.5 times 2).
constexpr int kMmaTiles = 4;

template <class F, int R>
void launch_round_mma(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  using S = MmaSmem<F, R, kMmaTiles>;
  constexpr int threads = 32 * kTileWarps * kMmaTiles;
  static const int resident = [&] {
    if (sizeof(S) > device_max_smem_optin()) {
      std::fprintf(stderr, "flyx-cuda: a round on block weights on the tensor cores needs %zu bytes of shared memory "
                           "a block\n", sizeof(S));
      std::abort();
    }
    set_dynamic_smem(k_round_blocks_mma<F, R, kMmaTiles>, sizeof(S));
    return resident_blocks(k_round_blocks_mma<F, R, kMmaTiles>, threads, sizeof(S));
  }();
  const Scratch& sc = scratch(L);
  const int grid = balanced_grid(W.rows, 8 * kMmaTiles, resident);
  launch_dependent_smem(k_round_blocks_mma<F, R, kMmaTiles>, dim3(grid), dim3(threads), sizeof(S), L, block_tables(),
                        sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows, int(accumulate));
}

// The kernel with the columns in the MMA's first operand: CT tiles of 16 columns a block, SLOTS spans of weights in
// registers.
template <class F, int R, int CT, int SLOTS>
void launch_round_cols(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  using S = ColsSmem<F, R, CT>;
  constexpr int threads = 32 * kTileWarps * CT;
  static const int resident = [&] {
    if (sizeof(S) > device_max_smem_optin()) {
      std::fprintf(stderr, "flyx-cuda: a round on block weights on the tensor cores needs %zu bytes of shared memory "
                           "a block\n", sizeof(S));
      std::abort();
    }
    set_dynamic_smem(k_round_blocks_cols<F, R, CT, SLOTS>, sizeof(S));
    return resident_blocks(k_round_blocks_cols<F, R, CT, SLOTS>, threads, sizeof(S));
  }();
  const Scratch& sc = scratch(L);
  const int grid = balanced_grid(W.rows, 16 * CT, resident);
  launch_dependent_smem(k_round_blocks_cols<F, R, CT, SLOTS>, dim3(grid), dim3(threads), sizeof(S), L, block_tables(),
                        sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows, int(accumulate));
}

// The streams split among a block's warps, Q a warp (k_round_blocks_warps): the blocks stay and take tiles of 16
// columns in turn.
template <class F, int R, int Q>
void launch_round_warps(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  constexpr int threads = 32 * (32 / Q);
  static const int resident = resident_blocks(k_round_blocks_warps<F, R, Q>, threads);
  const Scratch& sc = scratch(L);
  launch_dependent(k_round_blocks_warps<F, R, Q>, dim3(balanced_grid(W.rows, 16, resident)), dim3(threads), L,
                   block_tables(), sc.q8, sc.q8_scale, sc.q8_sums, W.data, W.scales, W.rows, K, y, ldy, rows,
                   int(accumulate));
}

// The streams split among kSplitSets blocks of threads of NW warps (k_round_blocks_split): false, nothing run, where
// the scratch has no room for partial sums or a block's rows do not fit in its shared memory.  A layer wider than
// the scratch's partial sums take (an output head) goes a slice of columns a launch.
template <class F, int R, int NW, bool BIG, bool COLS = false>
bool launch_round_split_as(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  constexpr int E = COLS ? 4 : R / 4, TW = COLS ? 16 : 8;  // a lane's sums a stream; a tile's columns
  const Scratch& sc = scratch(L);
  const int slice = int(std::min(sc.round_parts_cap / (kSplitSets * 32 * E), sc.round_count_cap)) * TW;
  if (!sc.round_parts || slice < TW) return false;
  static const int limit = optin_dynamic_smem(k_round_blocks_split<F, R, NW, BIG, COLS>);
  const SplitLayout<F, R, BIG> lay(K);
  if (lay.bytes > size_t(limit)) return false;
  static ResidentCache cache;
  const int resident = cache.get(k_round_blocks_split<F, R, NW, BIG, COLS>, 32 * NW, lay.bytes);
  const size_t blocks = size_t(K / F::kValues);
  for (int n0 = 0; n0 < W.rows; n0 += slice) {
    const int N = std::min(slice, W.rows - n0);
    const int per_set = balanced_grid(ceil_div(N, TW), NW, std::max(1, resident / kSplitSets));
    launch_dependent_smem(k_round_blocks_split<F, R, NW, BIG, COLS>, dim3(per_set * kSplitSets), dim3(32 * NW), lay.bytes, L,
                          block_tables(), sc.q8, sc.q8_scale, sc.q8_sums, W.data + size_t(n0) * blocks * F::kStride,
                          W.scales + size_t(n0) * blocks * F::kScales, N, K, y + n0, ldy, rows, int(accumulate),
                          sc.round_parts, sc.round_count);
  }
  return true;
}
// The split kernel with a type's larger tables (MmaLocal) where the rows leave room for them, else with its Local.
template <class F, int R, int NW, bool COLS = false>
bool launch_round_split(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  if (launch_round_split_as<F, R, NW, true, COLS>(W, K, y, ldy, rows, accumulate, L)) return true;
  if constexpr (kHasMmaLocal<F>) return launch_round_split_as<F, R, NW, false, COLS>(W, K, y, ldy, rows, accumulate, L);
  return false;
}
constexpr int kSplitWarps = 16;

// Blocks of the kernel with the columns in the MMA's first operand: 4 tiles of 16 columns, 2 spans of weights in
// registers.  (2 tiles and 2 or 3 spans, 255 registers a thread: slower on most of the 27B's layers; with 16 rows
// it spills, so rounds of more than 8 rows take the kernel with the rows there.)
constexpr int kColsTiles = 4, kColsSpans = 2;

// A round of type F: kernel 0 dp4a in configuration cfg, 1 the tensor cores with the rows in the MMA's first operand
// (the fewest of 8, 16 rows that take the round's), 2 with the columns there (8 rows; more: kernel 1), 3 and 4 the
// streams split among blocks of threads (the rows, the columns in the MMA's first operand), 5 the streams split among
// a block's warps (the types WarpsType takes; else kernel 3).
template <class F>
void launch_round_kernel(int kernel, int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                         const Launch& L) {
  if (kernel == 5) {
    if constexpr (WarpsType<F>::kTakes) {
      if (rows <= 8) {
        if constexpr (round_tc_active<F, 6>(0)) {
          launch_round_warps<F, 8, 4>(W, K, y, ldy, rows, accumulate, L);
          return;
        }
      } else {
        if constexpr (round_tc_active<F, 7>(0)) {
          launch_round_warps<F, 16, 4>(W, K, y, ldy, rows, accumulate, L);
          return;
        }
      }
    }
    kernel = 3;
  }
  if (kernel == 4) {  // the streams split among blocks, the columns in the MMA's first operand; else kernel 3
    if constexpr (round_tc_active<F, 5>(0)) {
      if (rows <= 8 && launch_round_split<F, 8, kSplitWarps, true>(W, K, y, ldy, rows, accumulate, L)) return;
    }
    kernel = 3;
  }
  if (kernel == 3) {  // the streams split among blocks; where that does not fit, kernel 1
    bool done = false;
    if (rows <= 8) {
      if constexpr (round_tc_active<F, 3>(0))
        done = launch_round_split<F, 8, kSplitWarps>(W, K, y, ldy, rows, accumulate, L);
    } else {
      if constexpr (round_tc_active<F, 4>(0))
        done = launch_round_split<F, 16, kSplitWarps>(W, K, y, ldy, rows, accumulate, L);
    }
    if (done) return;
    kernel = 1;
  }
  if (kernel == 2 && rows <= 8) {
    if constexpr (round_tc_active<F, 2>(0)) {
      launch_round_cols<F, 8, kColsTiles, kColsSpans>(W, K, y, ldy, rows, accumulate, L);
      return;
    }
    std::abort();
  } else if (kernel == 1 || kernel == 2) {
    if (rows <= 8) {
      if constexpr (round_tc_active<F, 0>(0)) {
        launch_round_mma<F, 8>(W, K, y, ldy, rows, accumulate, L);
        return;
      }
    } else {
      if constexpr (round_tc_active<F, 1>(0)) {
        launch_round_mma<F, 16>(W, K, y, ldy, rows, accumulate, L);
        return;
      }
    }
    std::abort();
  } else {
    launch_round_cfg<F>(cfg, W, K, y, ldy, rows, accumulate, L);
  }
}

// The round launcher of whichever of Fs has this dtype; nullptr if none does.
template <class... Fs>
RoundFn round_ops_of(int dtype) {
  RoundFn found = nullptr;
  ((found = !found && dtype == Fs::kDtype ? &launch_round_kernel<Fs> : found), ...);
  return found;
}

// Whether kernel 5 takes whichever of Fs has this dtype.
template <class... Fs>
bool round_warps_of(int dtype) {
  return ((dtype == Fs::kDtype && WarpsType<Fs>::kTakes) || ...);
}

// Whether a rule of any table (or the defaults) dispatches a stack (2048 < N <= 65536) on type `dtype` to dp4a
// configuration cfg (exact first-match reachability from gen_gemm_tables.py, excluding windowed S == 2).
constexpr bool round_cfg_ruled(int dtype, int cfg) {
  const int di = tables::ggml_type_index(dtype);
  return di >= 0 && (tables::kReachRoundStack[tables::kArchBucketAny][di] & (1u << cfg)) != 0;
}

// A stack's runs W[0 .. n - 1] of types F, Fn (and Fn2) in one launch of the dp4a kernel, in the first run's
// configuration cfg; false, nothing run, for a configuration no rule gives F (not compiled for several runs).
template <class F, class Fn, class Fn2>
bool launch_round_stack(int cfg, const BlockMat* W, int K, float* y, int ldy, int rows, bool accumulate,
                        const Launch& L) {
#if !FLYX_TUNE_CANDIDATES
  const int arch = compiled_arch();
  constexpr int di = tables::ggml_type_index(F::kDtype);
  if (arch <= 0 || di < 0 || !(tables::kReachRoundStack[tables::arch_bucket(arch)][di] & (1u << cfg))) return false;
#endif
  RoundMore more;
  more.run[0] = {W[1].data, W[1].scales, W[1].rows};
  if constexpr (!std::is_void_v<Fn2>) more.run[1] = {W[2].data, W[2].scales, W[2].rows};
  switch (cfg) {
#define X(i, R, C, U, NW, S, P)                                                                \
  case i:                                                                                      \
    if constexpr (S != 2 && round_cfg_ruled(F::kDtype, i)) {                                   \
      launch_round<F, R, C, U, NW, S, P, Fn, Fn2>(W[0], K, y, ldy, rows, accumulate, L, more); \
      return true;                                                                             \
    }                                                                                          \
    return false;
    FLYX_BLOCKS_ROUND(X)
#undef X
    default:
      return false;
  }
}
// The types of a stack compiled to run in one launch: F, then Fn, then Fn2 (void: two runs).
template <class F, class Fn, class Fn2 = void>
struct RoundStack {
  using First = F;
  using Next = Fn;
  using Next2 = Fn2;
};
// The launcher of whichever of Ss (RoundStack) has these n types (2 or 3); nullptr if none does.
template <class... Ss>
RoundStackFn round_stacks_of(const int* dtype, int n) {
  RoundStackFn found = nullptr;
  auto one = [&](auto stack) {
    using S = decltype(stack);
    constexpr int runs = std::is_void_v<typename S::Next2> ? 2 : 3;
    if (found || n != runs || dtype[0] != S::First::kDtype || dtype[1] != S::Next::kDtype) return;
    if constexpr (runs == 3) {
      if (dtype[2] != S::Next2::kDtype) return;
    }
    found = &launch_round_stack<typename S::First, typename S::Next, typename S::Next2>;
  };
  (one(Ss{}), ...);
  return found;
}

}  // namespace

}  // namespace flyx::cuda
