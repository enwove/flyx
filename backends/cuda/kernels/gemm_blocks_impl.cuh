// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The GEMM and embedding kernels on GGML block weights, templated on the block type (gemm_blocks_types.cuh), and
// their launchers: included by one translation unit per group of types (gemm_blocks_<group>.cu), which
// instantiates them for its types only, so the groups compile side by side; gemm_blocks.cu holds the staging of
// the inputs and the public API and reaches a type's kernels through BlockOps (gemm_blocks_ops.h).  Why the
// kernels are as they are: the comments of gemm_blocks.cu and below.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_ops.h"
#include "backends/cuda/kernels/gemm_blocks_types.cuh"
#include "flyx/gemm_tables.inc"

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

namespace flyx::cuda {
namespace {

template <int NT, int NW, int KW>
__host__ __device__ constexpr bool blocks_skinny_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const unsigned mask = tables::kReachBlocksSkinny[tables::arch_bucket(cc)];
  bool hit = false;
#define X(i, nt, nw, kw) \
  if ((mask & (1u << i)) && nt == NT && nw == NW && kw == KW) hit = true;
  FLYX_BLOCKS_SKINNY(X)
#undef X
  return hit;
#endif
}

template <class F, int BN, int QK, int BM>
__host__ __device__ constexpr bool blocks_q8_active(int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)cc;
  return true;
#else
  const int di = tables::ggml_type_index(F::kDtype);
  if (di < 0) return false;
  const unsigned mask = tables::kReachBlocksQ8[tables::arch_bucket(cc)][di];
  bool hit = false;
#define X(i, bn, qk, bm) \
  if ((mask & (1u << i)) && bn == BN && qk == QK && bm == BM) hit = true;
  FLYX_BLOCKS_Q8(X)
#undef X
  return hit;
#endif
}

// Group G of a block as lane t reads it, for a thread on its own (G is not a constant there: the groups are
// tried in turn, so that each call sees a constant one).
template <class F>
__device__ __forceinline__ void words_at(const typename F::Local& local, const typename F::Lane& ln, int G, uint32_t* w0,
                                         uint32_t* w1) {
#pragma unroll
  for (int c = 0; c < F::kValues / 32; ++c)
    if (c == G) F::words(local, ln, c, w0, w1);
}

// ---- 1..16 rows (decode, verification, drafting) ---------------------------------------------
// A warp computes 16 rows x 8 NT columns: lane (g, t) holds column g's
// weights 8 t .. of every 32 and rows g and g + 8 of their activations.  It
// walks K a window of kWindow weights at a time.  The window's activations
// (for 16 rows 8 KB, far more than its weights) go through shared memory,
// one window ahead, and the window's weights are loaded one window ahead,
// so that neither is waited for.
//
// A block of threads is KW groups of NW warps.  The warps of a group are on
// neighbouring columns and share their windows; the groups each take a run
// of K (of the kChunks runs below) on the same columns.  A wide layer wants
// many columns per block of threads (a window is staged once for all of
// them), a narrow one many runs of K at once (there are few columns to keep
// the GPU busy with), a layer of a few columns nothing else.
//
// The sums: the tensor cores add up a block from zero, plane by plane; the
// lane's sums (columns 2 t and 2 t + 1, whose coefficients live in lanes 8 t
// and 8 t + 4) take the block's times the coefficients.  K is cut into kChunks runs of whole windows, each
// added up on its own, and the runs are added as a balanced tree.  So a
// row's result depends on nothing but its own activations, whatever the row
// count and the configuration (NT, NW, KW).
constexpr int kWindow = 256;  // values of K (a block, or eight blocks of 32)
constexpr int kChunks = 8;    // runs of K; KW divides it

// The shared memory of a block of threads of k_gemm_blocks: the type's tables, the groups' sums (KW > 1) and
// per group two windows of slots + 1 rows (the last: zeros, for the lanes of absent rows).
template <class F, int NT, int NW, int KW>
struct BlocksSmem {
  static constexpr size_t kTables = (sizeof(typename F::Local) + 15) / 16 * 16;
  static constexpr size_t kSums = KW > 1 ? size_t(KW) * NW * NT * 32 * 4 * sizeof(float) : 0;
  static constexpr size_t bytes(int slots) {
    return kTables + kSums + size_t(KW) * 2 * (slots + 1) * kWindow * sizeof(uint32_t);
  }
};

template <class F, int NT, int NW, int KW>
__global__ void __launch_bounds__(32 * NW * KW) k_gemm_blocks(Tables T, const uint8_t* pairs, const float* row_scale,
                                                              const uint8_t* __restrict__ W,
                                                              const uint16_t* __restrict__ D, int N, int K,
                                                              float* __restrict__ y, int ldy, int rows,
                                                              int accumulate) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!blocks_skinny_active<NT, NW, KW>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  static_assert(kChunks % KW == 0 && (KW & (KW - 1)) == 0, "the groups each take the same number of runs of K");
  using Smem = BlocksSmem<F, NT, NW, KW>;
  constexpr int GT = 32 * NW, GROUPS = F::kValues / 32, P = F::kPlanes, PER = kWindow / F::kValues;
  constexpr int CPG = kChunks / KW, LEVELS = CPG == 8 ? 3 : CPG == 4 ? 2 : CPG == 2 ? 1 : 0;
  extern __shared__ __align__(16) unsigned char smem[];
  auto& local = *reinterpret_cast<typename F::Local*>(smem);
  float* sums = reinterpret_cast<float*>(smem + Smem::kTables);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int kw = warp / NW, nw = warp % NW, tid = int(threadIdx.x) - kw * GT;  // the thread within its group
  const int slots = rows < 8 ? rows : 8, slot = g < slots ? g : slots;       // absent rows: the zeros
  // The group's two windows, [2][slots + 1][kWindow]: a word per value of K (rows g and g + 8).
  uint32_t* windows = reinterpret_cast<uint32_t*>(smem + Smem::kTables + Smem::kSums) + size_t(kw) * 2 * (slots + 1) * kWindow;
  F::tables(T, local);
#pragma unroll
  for (int j = 0; j < (2 * kWindow + GT - 1) / GT; ++j) {
    const int i = tid + j * GT;
    if (i < 2 * kWindow) windows[((i / kWindow) * (slots + 1) + slots) * kWindow + i % kWindow] = 0u;
  }
  const int n_base = (blockIdx.x * NW + nw) * 8 * NT, blocks = K / F::kValues, nwin = (K + kWindow - 1) / kWindow;
  // The group's runs of K are kw CPG .. : windows w_first .. w_last - 1.  Every group makes as many turns as the
  // longest (the barriers are the block's).
  const int w_first = kw * CPG * nwin / kChunks, w_last = (kw + 1) * CPG * nwin / kChunks;
  const int turns = CPG * nwin / kChunks + 1;
  const uint8_t* wp[NT];
  const uint16_t* dp[NT];
  bool wv[NT];
  float acc[NT][4], level[LEVELS > 0 ? LEVELS : 1][NT][4];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    const size_t first = size_t(wv[i] ? n : 0) * blocks;
    wp[i] = W + first * F::kStride;
    dp[i] = D + first * F::kScales;
    acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  }
  // A window of the rows into shared memory: 16 bytes are 4 values of K, two rows.
  auto stage = [&](int which, int w) {
#pragma unroll
    for (int j = 0; j < (8 * (kWindow / 4) + GT - 1) / GT; ++j) {
      const int i = tid + j * GT, s = i / (kWindow / 4), k = 4 * (i % (kWindow / 4));
      if (s < slots) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (w * kWindow + k < K) v = ld128(pairs + (size_t(s) * K + size_t(w) * kWindow + k) * 4);
        *reinterpret_cast<uint4*>(&windows[(which * (slots + 1) + s) * kWindow + k]) = v;
      }
    }
  };
  // A run of K has ended: its sums (acc) join the tree of the group's runs; `done` runs ended before it.
  auto close = [&](int done) {
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = acc[i][e];
        bool carry = true;
#pragma unroll
        for (int l = 0; l < LEVELS; ++l)
          if (carry) {
            if ((done >> l) & 1) v = level[l][i][e] + v;
            else level[l][i][e] = v, carry = false;
          }
        acc[i][e] = carry ? v : 0.f;  // after the group's last run: all of it
      }
  };
  pdl_wait();  // the activations come from the kernel before
  if (w_first < w_last) stage(0, w_first);
  // The weights of the window ahead: of every block of it (eight blocks of 32 are eight small loads, and a
  // warp that waited for each in turn would have 16 bytes a column on their way at a time).
  typename F::Raw raw[NT][PER];
  auto load_window = [&](int w) {
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int bw = 0; bw < PER; ++bw) raw[i][bw] = block_load<F>(wp[i], dp[i], size_t(min(w * PER + bw, blocks - 1)), t);
  };
  load_window(w_first);
  int chunk = kw * CPG, done = 0;
#pragma unroll 1
  for (int turn = 0; turn < turns; ++turn) {
    const int w = w_first + turn;
    __syncthreads();  // this window is there (the first time: the tables and the zeros too); the one before is read
    if (w >= w_last) continue;
    while ((chunk + 1) * nwin / kChunks <= w) close(done), ++chunk, ++done;  // the runs that ended before this window
    if (w + 1 < w_last) stage((turn + 1) & 1, w + 1);
    const uint32_t* a_lane = windows + ((turn & 1) * (slots + 1) + slot) * kWindow + 8 * t;
    typename F::Lane ln[NT][PER];
    float cf[NT][PER][P];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int bw = 0; bw < PER; ++bw) {
        ln[i][bw] = F::share(raw[i][bw], t, lane);
        const size_t at = size_t(min(w * PER + bw, blocks - 1)) * F::kScales;
        const float d0 = scale_at(dp[i], at), d1 = F::kScales > 1 ? scale_at(dp[i], at + 1) : 0.f;
#pragma unroll
        for (int p = 0; p < P; ++p) cf[i][bw][p] = wv[i] ? F::coef(p, d0, d1) : 0.f;  // a column past N: nothing
      }
    load_window(w + 1 < w_last ? w + 1 : w);  // the next window's bytes, on their way
#pragma unroll
    for (int bw = 0; bw < PER; ++bw) {
      if (w * PER + bw >= blocks) break;
      float part[NT][P][4];
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int p = 0; p < P; ++p) part[i][p][0] = part[i][p][1] = part[i][p][2] = part[i][p][3] = 0.f;
#pragma unroll
      for (int G = 0; G < GROUPS; ++G) {
        const uint32_t* ap = a_lane + bw * F::kValues + 32 * G;
        const uint4 x0 = *reinterpret_cast<const uint4*>(ap), x1 = *reinterpret_cast<const uint4*>(ap + 4);
        const uint32_t a0[4] = {x0.x, x0.y, x0.z, x0.w}, a1[4] = {x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          uint32_t w0[4], w1[4];
          F::words(local, ln[i][bw], G, w0, w1);
          mma_f16(part[i][0], a0, w0);
          mma_f16(part[i][0], a1, w0 + 2);
          if (P > 1) {
            mma_f16(part[i][P - 1], a0, w1);
            mma_f16(part[i][P - 1], a1, w1 + 2);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int p = 0; p < P; ++p) {
          const float c0 = __shfl_sync(0xffffffffu, cf[i][bw][p], 8 * t);
          const float c1 = __shfl_sync(0xffffffffu, cf[i][bw][p], 8 * t + 4);
          acc[i][0] += c0 * part[i][p][0];
          acc[i][1] += c1 * part[i][p][1];
          acc[i][2] += c0 * part[i][p][2];
          acc[i][3] += c1 * part[i][p][3];
        }
    }
  }
  while (done < CPG) close(done), ++done;  // the last run, and runs without a window
  if (KW > 1) {  // the groups' sums, as the tree goes on
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) sums[((size_t(kw) * NW * NT + nw * NT + i) * 32 + lane) * 4 + e] = acc[i][e];
    __syncthreads();
    if (kw != 0) return;
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v[KW];
#pragma unroll
        for (int q = 0; q < KW; ++q) v[q] = sums[((size_t(q) * NW * NT + nw * NT + i) * 32 + lane) * 4 + e];
#pragma unroll
        for (int width = KW; width > 1; width /= 2)
#pragma unroll
          for (int q = 0; q < width / 2; ++q) v[q] = v[2 * q] + v[2 * q + 1];
        acc[i][e] = v[0];
      }
  }
#pragma unroll
  for (int e = 0; e < 4; e += 2) {
    const int r = g + (e >> 1) * 8;
    if (r >= rows) continue;
    const float s = row_scale[r];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int col = n_base + i * 8 + 2 * t + j;
        if (col < N) {
          float* dst = y + size_t(r) * ldy + col;
          const float v = acc[i][e + j] * s;
          *dst = accumulate ? *dst + v : v;
        }
      }
  }
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- more rows (prefill) -----------------------------------------------------------------------
// Block tile 128 rows x 64 columns, 8 warps of 32 x 32.  K goes by in windows
// of 256 weights.  A window's weights are decoded once for the 128 rows, a
// thread a lane's part of a column's blocks as in the kernel above (what the
// lane loads is read once, the rest of the block comes from its neighbours),
// into shared memory; then the rows (binary16 over their scales,
// k_stage_rows) come through in tiles of 32 values of K, one MMA step per 16
// weights.  A block's sums start from zero and are added times the
// coefficients when the block ends, as above.  Every output adds its blocks
// in order with the same operations whatever the tile it is in, so a row's
// result does not depend on how many rows the pass has.
//
// Nothing waits for memory: a window's bytes and a tile's rows are loaded
// into registers one window and one tile ahead, while the MMAs before them
// run.  A block of threads keeps to half an SM's shared memory and registers
// where its tables allow it, so that two run on one, each decoding while the
// other multiplies.
//
// A second plane that is one value per run of 16 or 32 weights (kFlatRun: a
// minimum) costs no MMA step per 16 weights: its part of an output is the sum
// over the runs of (the row's sum over the run) x (the run's value), one MMA
// step per 8 runs on the sums k_stage_rows left.  A sum is held as two
// binary16 values, high and low part of it over 64 (22 bits of it: what a
// partial sum loses anyway is its bits past 24).
//
// The pieces of 16 bytes of a row in shared memory are stored with their index
// bits swapped by the row's, so that the eight rows one ldmatrix reads are in
// eight bank groups.
constexpr int kRowsK = 32, kRowsM = 128, kRowsN = 64;
// Where piece c of row r is (in binary16 values) in rows of `pieces` pieces: 2, 4, or a multiple of 8.
__device__ __forceinline__ int piece_at(int r, int c, int pieces) {
  return 8 * (c ^ (pieces >= 8 ? r & 7 : pieces == 4 ? (r >> 1) & 3 : (r >> 2) & 1));
}
// Two blocks of threads on an SM where a block's shared memory allows it (the registers are then held to half).
constexpr size_t kSmemOfTwo = 49 * 1024;
constexpr float kSumsOver = 64.f;  // a run's sum over this is a binary16 value: 32 values under 2^16

template <class F>
struct RowsSmem {
  static constexpr int P = F::kPlanes, RUN = F::kFlatRun, PF = RUN ? 1 : P, ENDS = kWindow / F::kValues;
  static constexpr int RUNS = RUN ? kWindow / RUN : 8;  // of a window; an MMA step takes 8
  static constexpr int PC = RUN && ENDS > 1 ? 1 : P;    // planes with a coefficient per block
  alignas(16) uint16_t a[kRowsM][kRowsK];
  alignas(16) uint16_t b[PF][kRowsN][kWindow];
  // With a flat plane, per 8 runs of the window: the rows' sums (8 high parts, 8 low parts) and the columns'
  // values (twice).
  alignas(16) uint16_t sums[RUN ? kRowsM : 1][2 * RUNS];
  alignas(16) uint16_t flat[RUN ? kRowsN : 1][2 * RUNS];
  float coefs[PC][kRowsN][ENDS];
  typename F::Local local;
};

template <class F>
__global__ void __launch_bounds__(4 * kRowsN, sizeof(RowsSmem<F>) <= kSmemOfTwo ? 2 : 1)
k_gemm_blocks_rows(Tables T, const uint16_t* A, int lda, const float* row_scale, const float* __restrict__ sums,
                   const uint8_t* __restrict__ W, const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y,
                   int ldy, int M, int accumulate) {
  pdl_trigger();
  using S = RowsSmem<F>;
  constexpr int P = S::P, PF = S::PF, PC = S::PC, RUN = S::RUN, RUNS = S::RUNS;
  constexpr int THREADS = 4 * kRowsN, PIECES = kRowsM * (kRowsK / 8) / THREADS, TILES = kWindow / kRowsK;
  constexpr int UNIT = F::kValues, PER = kWindow / UNIT, GROUPS = UNIT / 32;
  constexpr int MINE = (PER + 3) / 4;  // blocks of a window whose scales a lane loads: t, t + 4
  constexpr int SUMS = RUN ? kRowsM * 8 / THREADS : 1;  // pairs of a window's 16 sums of a row a thread stages
  static_assert(PER == 1 || RUN == 0 || RUN == UNIT, "a flat plane of a small block is the block's");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S*>(smem_raw);
  F::tables(T, sm.local);
  __syncthreads();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * kRowsM, n0 = blockIdx.x * kRowsN, blocks = K / UNIT;
  float acc[2][4][4], part[PF][2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        acc[mi][ni][e] = 0.f;
#pragma unroll
        for (int p = 0; p < PF; ++p) part[p][mi][ni][e] = 0.f;
      }
  // Sums of the tensor cores times the columns' coefficients (c0, c1: columns 2 t and 2 t + 1 of the eight).
  auto take = [&](int p, int ni, float c0, float c1) {
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      acc[mi][ni][0] += c0 * part[p][mi][ni][0];
      acc[mi][ni][1] += c1 * part[p][mi][ni][1];
      acc[mi][ni][2] += c0 * part[p][mi][ni][2];
      acc[mi][ni][3] += c1 * part[p][mi][ni][3];
      part[p][mi][ni][0] = part[p][mi][ni][1] = part[p][mi][ni][2] = part[p][mi][ni][3] = 0.f;
    }
  };
  // A block of the weights has ended (`end`: which of the window's).
  auto close = [&](int end) {
#pragma unroll
    for (int p = 0; p < PF; ++p)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int c = wn * 32 + ni * 8 + 2 * t;
        take(p, ni, sm.coefs[p][c][end], sm.coefs[p][c + 1][end]);
      }
  };

  // The weights: this thread is lane t of column tid >> 2 (the lanes of a column are neighbours in a warp).
  const int col = tid >> 2, n = n0 + col;
  typename F::Raw raw[PER];
  float scales[MINE][2];
  auto block_at = [&](int k) { return n < N && k < K ? size_t(n) * blocks + k / UNIT : size_t(0); };
  auto load_window = [&](int k0) {  // loads only
#pragma unroll
    for (int bw = 0; bw < PER; ++bw) raw[bw] = block_load<F>(W, D, block_at(k0 + bw * UNIT), t);
#pragma unroll
    for (int i = 0; i < MINE; ++i) {
      const int bw = t + 4 * i;
      if (bw >= PER) continue;
      const size_t at = block_at(k0 + bw * UNIT) * F::kScales;
      scales[i][0] = scale_at(D, at);
      scales[i][1] = F::kScales > 1 ? scale_at(D, at + 1) : 0.f;
    }
  };
  // A run's value of the flat plane, for the MMA step of its eight runs.
  auto put_flat = [&](int run, uint32_t half) {
    const int at = run & 7, step = run >> 3;
    sm.flat[col][piece_at(col, 2 * step, RUNS / 4) + at] = uint16_t(half);
    sm.flat[col][piece_at(col, 2 * step + 1, RUNS / 4) + at] = uint16_t(half);
  };
  auto decode_window = [&](int k0) {  // what load_window(k0) loaded, into shared memory
    float2 two[SUMS];  // the rows' sums over the window's runs, on their way meanwhile
    if (RUN) {
#pragma unroll
      for (int i = 0; i < SUMS; ++i) {
        const int idx = tid + i * THREADS, r = idx >> 3, c = idx & 7, gm = m0 + r;
        two[i] = make_float2(0.f, 0.f);
        if (gm < M && k0 + 32 * c < K)
          two[i] = *reinterpret_cast<const float2*>(sums + size_t(gm) * (K / 16) + k0 / 16 + 2 * c);
      }
    }
#pragma unroll
    for (int bw = 0; bw < PER; ++bw) {
      const bool in = n < N && k0 + bw * UNIT < K;
      const typename F::Lane ln = F::share(raw[bw], t, lane);
#pragma unroll
      for (int G = 0; G < GROUPS; ++G) {
        uint32_t w[2][4];
        F::words(sm.local, ln, G, w[0], w[P - 1]);
        const int c = (bw * UNIT + 32 * G) / 8 + t;  // the piece of the window: its weights from 8 c on
#pragma unroll
        for (int p = 0; p < PF; ++p)
          *reinterpret_cast<uint4*>(&sm.b[p][col][piece_at(col, c, kWindow / 8)]) =
              in ? make_uint4(w[p][0], w[p][1], w[p][2], w[p][3]) : make_uint4(0, 0, 0, 0);
        if (RUN && PER == 1 && (8 * t) % (RUN ? RUN : 1) == 0) put_flat((32 * G + 8 * t) / (RUN ? RUN : 1), in ? w[1][0] : 0u);
      }
    }
#pragma unroll
    for (int i = 0; i < MINE; ++i) {  // the blocks' coefficients, for when they end
      const int bw = t + 4 * i;
      if (bw >= PER) continue;
      const bool in = n < N && k0 + bw * UNIT < K;
#pragma unroll
      for (int p = 0; p < PC; ++p)
        sm.coefs[p][col][bw] = in ? F::coef(p, scales[i][0], scales[i][1]) * (RUN && p ? kSumsOver : 1.f) : 0.f;
      if (RUN && PER > 1) {  // ones under a coefficient that is a binary16 value: the plane's value is that
        const __half h = __float2half(F::coef(1, scales[i][0], scales[i][1]));
        put_flat(bw, in ? uint32_t(*reinterpret_cast<const uint16_t*>(&h)) : 0u);
      }
    }
    if (RUN) {
#pragma unroll
      for (int i = 0; i < SUMS; ++i) {
        const int idx = tid + i * THREADS, r = idx >> 3, c = idx & 7;
        // 16 sums of a row a window: 8 runs of 32 (this pair is one), or 16 runs of 16.
#pragma unroll
        for (int j = 0; j < (RUN == 16 ? 2 : 1); ++j) {
          const float v = (RUN == 16 ? (j ? two[i].y : two[i].x) : two[i].x + two[i].y) * (1.f / kSumsOver);
          const __half high = __float2half_rn(v), low = __float2half_rn(v - __half2float(high));
          const int run = RUN == 16 ? 2 * c + j : c, at = run & 7, step = run >> 3;
          sm.sums[r][piece_at(r, 2 * step, RUNS / 4) + at] = *reinterpret_cast<const uint16_t*>(&high);
          sm.sums[r][piece_at(r, 2 * step + 1, RUNS / 4) + at] = *reinterpret_cast<const uint16_t*>(&low);
        }
      }
    }
  };
  // The rows: 128 x 4 pieces of 16 bytes a tile of K, PIECES of them this thread's.
  uint4 rows[PIECES];
  auto load_rows = [&](int kq) {
#pragma unroll
    for (int i = 0; i < PIECES; ++i) {
      const int idx = tid + i * THREADS, r = idx / (kRowsK / 8), c = idx % (kRowsK / 8), gm = m0 + r;
      rows[i] = make_uint4(0, 0, 0, 0);
      if (gm < M && kq + 8 * c < K) rows[i] = *reinterpret_cast<const uint4*>(A + size_t(gm) * lda + kq + 8 * c);
    }
  };
  auto store_rows = [&] {
#pragma unroll
    for (int i = 0; i < PIECES; ++i) {
      const int idx = tid + i * THREADS, r = idx / (kRowsK / 8), c = idx % (kRowsK / 8);
      *reinterpret_cast<uint4*>(&sm.a[r][piece_at(r, c, kRowsK / 8)]) = rows[i];
    }
  };

  load_window(0);
  pdl_wait();  // the activations come from the kernel before
  load_rows(0);
  for (int k0 = 0; k0 < K; k0 += kWindow) {
    decode_window(k0);
    if (k0 + kWindow < K) load_window(k0 + kWindow);
#pragma unroll 1
    for (int q = 0; q < TILES; ++q) {
      const int kq = k0 + q * kRowsK;
      if (kq < K) {
        store_rows();
        if (kq + kRowsK < K) load_rows(kq + kRowsK);
      }
      __syncthreads();
      if (kq < K) {
        uint32_t af[2][4], bf[PF][4][2];
#pragma unroll
        for (int s = 0; s < kRowsK / 16; ++s) {
          const int step = (kRowsK / 16) * q + s;  // of the window's 16
#pragma unroll
          for (int mi = 0; mi < 2; ++mi) {
            const int r = wm * 32 + mi * 16 + (lane & 15);
            ldmatrix_x4(af[mi], &sm.a[r][piece_at(r, 2 * s + (lane >> 4), kRowsK / 8)]);
          }
#pragma unroll
          for (int p = 0; p < PF; ++p)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
              const int c = wn * 32 + ni * 8 + (lane & 7);
              ldmatrix_x2(bf[p][ni], &sm.b[p][c][piece_at(c, 2 * step + ((lane >> 3) & 1), kWindow / 8)]);
            }
#pragma unroll
          for (int p = 0; p < PF; ++p)
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
              for (int ni = 0; ni < 4; ++ni) mma_f16(part[p][mi][ni], af[mi], bf[p][ni]);
          if (PER > 1 && (step & 1)) close(step >> 1);
        }
        const bool last = q == TILES - 1 || kq + kRowsK >= K;  // of the window
        if (PER == 1 && last) close(0);
        if (RUN && last) {  // the flat plane of the window's blocks
#pragma unroll
          for (int step = 0; step < RUNS / 8; ++step) {
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
              const int r = wm * 32 + mi * 16 + (lane & 15);
              ldmatrix_x4(af[mi], &sm.sums[r][piece_at(r, 2 * step + (lane >> 4), RUNS / 4)]);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
              const int c = wn * 32 + ni * 8 + (lane & 7);
              ldmatrix_x2(bf[0][ni], &sm.flat[c][piece_at(c, 2 * step + ((lane >> 3) & 1), RUNS / 4)]);
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
              for (int ni = 0; ni < 4; ++ni) mma_f16(part[0][mi][ni], af[mi], bf[0][ni]);
          }
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int c = wn * 32 + ni * 8 + 2 * t;
            if (PER == 1) take(0, ni, sm.coefs[PC - 1][c][0], sm.coefs[PC - 1][c + 1][0]);
            else take(0, ni, kSumsOver, kSumsOver);
          }
        }
      }
      __syncthreads();
    }
  }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int e = 0; e < 4; e += 2) {
      const int r = m0 + wm * 32 + mi * 16 + g + (e >> 1) * 8;
      if (r >= M) continue;
      const float s = row_scale[r];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int c = n0 + wn * 32 + ni * 8 + 2 * t + j;
          if (c < N) {
            float* dst = y + size_t(r) * ldy + c;
            const float v = acc[mi][ni][e + j] * s;
            *dst = accumulate ? *dst + v : v;
          }
        }
    }
}

// ---- more rows, inputs in 8 bits (FLYX_ACT_Q8) ------------------------------------------------
// The rows as k_stage_q8 leaves them: 8-bit integers p and a scale d per 32
// values (the value is d p).  A weight is f c + m: c a small integer (codes),
// f and m the factors of its sub-group of kSub weights (factors).  A
// sub-group's part of an output is d (f sum c p + m sum p): the tensor cores'
// integer mode adds up sum c p exactly (products of two 8-bit integers, 32 or
// 16 of them), the rest is float32, one sub-group after the other in the
// order of K whatever the tile.  So a row's result does not depend on how
// many rows the pass has.
//
// Every block of threads reads all the rows, so a block takes BN columns
// (warps of 32 x 32): 128 (16 warps, one block an SM) reads them half as
// often as 64 (8 warps, two an SM), which keeps more blocks for a narrow
// layer.  And every tile of rows costs two barriers, at which the block's
// warps all wait for the slowest: a tile of QK values of K, 64 to a whole
// window (256), is 32 steps of m16n8k32 between them for the longest (or of
// two m16n8k16, for sub-groups of 16); tiles 64 long ran 2 to 14% slower than
// the best. A block of
// 64 columns holds a tile of 128 and still fits two to an SM.
// variants/candidates.toml chooses by the layer's columns, and every choice
// gives the same bits.  A block adds up the sums of 16 it needs itself (dp4a)
// rather than reading them.  Windows of weights and loads ahead as in the
// kernel above.
// Byte offsets: piece c (16 bytes) of row r of a tile of rows QK values long (rows of 64 bytes share a line of
// banks two by two); byte k of a column's window of weights.
template <int QK>
__device__ __forceinline__ int q8_row_at(int r, int c) { return 16 * (c ^ (QK == 64 ? (r >> 1) & 3 : r & 7)); }
__device__ __forceinline__ int q8_col_at(int col, int k) { return 16 * ((k >> 4) ^ (col & 7)) + (k & 15); }

template <class F, int BN, int QK, int BM>
struct Q8Smem {
  static constexpr int SUBS = kWindow / F::kSub;
  static constexpr bool MINS = F::kFlatRun != 0;  // the types with a minimum
  alignas(16) uint8_t a[BM][QK];
  alignas(16) uint8_t b[BN][kWindow];
  float da[BM][QK / 32];
  float ds[MINS ? BM : 1][QK / 16];
  alignas(8) float wf[SUBS][BN];
  alignas(8) float wm[MINS ? SUBS : 1][BN];
  typename F::Local local;
};

// The runs of K that have a block each (gridDim.z == runs) fold onto y as one block taking them in turn would: y = [y
// +] run 0, then + run 1, ...; the same additions, the same bits.  Two ways, by the tile's counter (zero between
// GEMMs).  With parts (room for every run's sums): each run leaves its sums there, and the tile's last to finish folds
// them all.  Without: run z adds its sums onto y once runs 0 .. z - 1 have; it waits only for runs of its own tile,
// whose blocks come before it in the grid (blockIdx.z), so they have started.  The wait chains the runs' folds, a few
// microseconds that a short pass notices (64 rows: 2 to 6% slower than parts), and needs no scratch, so a long pass
// can spread too.  each_out(fn) calls fn(row, column, sum).
template <class EachOut>
__device__ __forceinline__ void fold_runs(EachOut&& each_out, float* y, int ldy, int M, int N, int accumulate, int runs,
                                          float* parts, int* count) {
  int* tile_count = count + blockIdx.y * gridDim.x + blockIdx.x;
  const int z = blockIdx.z;
  if (parts) {
    float* mine = parts + size_t(z) * M * N;
    each_out([&](int r, int c, float& v) { mine[size_t(r) * N + c] = v; });
    __threadfence();
    __syncthreads();
    __shared__ int last;
    if (threadIdx.x == 0) last = atomicAdd(tile_count, 1) == runs - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    each_out([&](int r, int c, float& v) {
      float* dst = y + size_t(r) * ldy + c;
      float sum = 0.f;
      for (int i = 0; i < runs; ++i) {
        const float p = i == z ? v : __ldcg(parts + (size_t(i) * M + r) * N + c);
        sum = i == 0 ? (accumulate ? *dst + p : p) : sum + p;
      }
      *dst = sum;
    });
    if (threadIdx.x == 0) *tile_count = 0;  // for the next GEMM
    return;
  }
  if (threadIdx.x == 0) {
    int in;
    while (true) {
      asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(in) : "l"(tile_count) : "memory");
      if (in == z) break;
      __nanosleep(100);
    }
  }
  __syncthreads();
  each_out([&](int r, int c, float& v) {
    float* dst = y + size_t(r) * ldy + c;
    __stcg(dst, z == 0 && !accumulate ? v : __ldcg(dst) + v);
  });
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int next = z + 1 == runs ? 0 : z + 1;  // the last run leaves zero for the next GEMM
    asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(tile_count), "r"(next) : "memory");
  }
}

// K is added up in `runs` runs of whole windows, fixed per shape (gemm_blocks_q8_runs): each run from zero, in
// the order of K, and the runs fold onto y in order (y = [y +] run 0, then + run 1, ...).  With gridDim.z == 1 a
// block takes every run of its tile, folding each into y as it ends; with gridDim.z == runs a block takes run
// blockIdx.z and fold_runs adds it in its place.  The same float32 additions in the same order either way, so a layer
// whose tiles leave much of the GPU idle can spread its runs over it without changing a bit.  A block's tile of rows is BM tall (128, or 64 for short passes): its four rows of
// warps take BM / 4 rows each, as BM / 64 steps of 16; the same sums whatever the tile.
// Blocks of 64 columns two an SM, so at most 128 registers a thread: the tiles of 128 rows then keep some values on
// the stack (16 to 176 bytes a thread), and lifting the cap (one block an SM, up to 252 registers, nothing on the
// stack) was 2 to 27% slower for every type measured but Q5_1's most crowded configuration.
template <class F, int BN, int QK, int BM>
__global__ void __launch_bounds__(4 * BN, BN == 64 ? 2 : 1)
k_gemm_blocks_q8(Tables T, const uint8_t* A, const float* d_a, const uint8_t* __restrict__ W,
                 const uint16_t* __restrict__ D, int N, int K, float* __restrict__ y, int ldy, int M, int accumulate,
                 int runs, float* __restrict__ parts, int* __restrict__ count) {
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  if constexpr (!blocks_q8_active<F, BN, QK, BM>(__CUDA_ARCH__ / 10)) {
    __trap();
  } else {
#endif
  pdl_trigger();
  using S = Q8Smem<F, BN, QK, BM>;
  constexpr int UNIT = F::kValues, PER = kWindow / UNIT, GROUPS = UNIT / 32, SUB = F::kSub, HALVES = 32 / SUB;
  constexpr bool MINS = S::MINS;
  constexpr int TILES = kWindow / QK, THREADS = 4 * BN, PIECES = BM * (QK / 16) / THREADS, WN = BN / 32;
  constexpr int PR = QK / 16;  // pieces of 16 values a row of the tile
  constexpr int MI = BM / 64, RW = BM / 4;  // a warp's steps of 16 rows, and its rows
  static_assert((BM == 64 || BM == 128) && PIECES >= 1 && PIECES * THREADS == BM * PR, "a tile of rows the threads share");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S*>(smem_raw);
  F::tables(T, sm.local);
  __syncthreads();
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp % WN, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN, blocks = K / UNIT;  // the tiles of rows side by side
  float acc[MI][4][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[mi][ni][e] = 0.f;

  // The weights: this thread is lane t of column tid >> 2.
  const int col = tid >> 2, n = n0 + col;
  typename F::Raw raw[PER];
  // A block's binary16 scales as stored, both in one word where there are two (they sit side by side), made floats
  // as the window is decoded: half the loads and registers a window holds for the types with a minimum.
  uint32_t ds[PER];
  auto block_at = [&](int k) { return n < N && k < K ? size_t(n) * blocks + k / UNIT : size_t(0); };
  auto load_window = [&](int k0) {  // loads only
#pragma unroll
    for (int bw = 0; bw < PER; ++bw) {
      const size_t at = block_at(k0 + bw * UNIT);
      raw[bw] = block_load<F>(W, D, at, t);
      ds[bw] = F::kScales > 1 ? *reinterpret_cast<const uint32_t*>(D + at * F::kScales) : uint32_t(D[at]);
    }
  };
  auto decode_window = [&](int k0) {  // what load_window(k0) loaded: codes and factors into shared memory
#pragma unroll
    for (int bw = 0; bw < PER; ++bw) {
      const bool in = n < N && k0 + bw * UNIT < K;
      const typename F::Lane ln = F::share(raw[bw], t, lane);
      const float d0 = half_of(uint16_t(ds[bw] & 0xFFFFu)), d1 = F::kScales > 1 ? half_of(uint16_t(ds[bw] >> 16)) : 0.f;
#pragma unroll
      for (int G = 0; G < GROUPS; ++G) {
        uint32_t w[2];
        F::codes(T, sm.local, ln, G, w);
        const int k = bw * UNIT + 32 * G;  // of the window
        *reinterpret_cast<uint2*>(&sm.b[col][q8_col_at(col, k + 8 * t)]) = in ? make_uint2(w[0], w[1]) : make_uint2(0u, 0u);
        float f, m;
        F::factors(ln, G, d0, d1, &f, &m);
        if (SUB == 16 ? !(t & 1) : !t) {  // a lane of each sub-group
          const int sub = k / SUB + (SUB == 16 ? t >> 1 : 0);
          sm.wf[sub][col] = in ? f : 0.f;
          if constexpr (MINS) sm.wm[sub][col] = in ? m : 0.f;
        }
      }
    }
  };
  // The rows: 128 x PR pieces of 16 bytes (16 values) a tile, PIECES of them this thread's, with their scales.
  uint4 piece[PIECES];
  float pd[PIECES];
  auto load_rows = [&](int kq) {
#pragma unroll
    for (int i = 0; i < PIECES; ++i) {
      const int idx = tid + i * THREADS, pr = idx / PR, pc = idx % PR, pm = m0 + pr;
      const bool in = pm < M && kq + 16 * pc < K;
      piece[i] = in ? *reinterpret_cast<const uint4*>(A + size_t(pm) * K + kq + 16 * pc) : make_uint4(0u, 0u, 0u, 0u);
      pd[i] = in ? d_a[size_t(pm) * (K / 32) + kq / 32 + (pc >> 1)] : 0.f;
    }
  };
  auto store_rows = [&] {
#pragma unroll
    for (int i = 0; i < PIECES; ++i) {
      const int idx = tid + i * THREADS, pr = idx / PR, pc = idx % PR;
      *reinterpret_cast<uint4*>(&sm.a[pr][q8_row_at<QK>(pr, pc)]) = piece[i];
      if (!(pc & 1)) sm.da[pr][pc >> 1] = pd[i];
      if constexpr (MINS) {  // d times the sum of the piece's 16 values
        const uint4 v = piece[i];
        const int sum = __dp4a(int(v.x), 0x01010101, __dp4a(int(v.y), 0x01010101,
                               __dp4a(int(v.z), 0x01010101, __dp4a(int(v.w), 0x01010101, 0))));
        sm.ds[pr][pc] = pd[i] * float(sum);
      }
    }
  };

  // This thread's outputs: fn(row, column, its sum) for each within M x N.
  auto each_out = [&](auto&& fn) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int e = 0; e < 4; e += 2) {
        const int r = m0 + wm * RW + mi * 16 + g + (e >> 1) * 8;
        if (r >= M) continue;
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const int c = n0 + wn * 32 + ni * 8 + 2 * t + j;
            if (c < N) fn(r, c, acc[mi][ni][e + j]);
          }
      }
  };
  // Where run i starts (window-aligned; the last ends at K).
  const int windows = (K + kWindow - 1) / kWindow;
  auto run_start = [&](int i) { return min(i * windows / runs * kWindow, K); };
  const bool spread = gridDim.z > 1;
  const int kb = spread ? run_start(blockIdx.z) : 0, ke = spread ? run_start(blockIdx.z + 1) : K;
  int run = 0, run_end = run_start(1);

  load_window(kb);
  pdl_wait();  // the rows come from the kernel before
  load_rows(kb);
  for (int k0 = kb; k0 < ke; k0 += kWindow) {
    decode_window(k0);
    if (k0 + kWindow < ke) load_window(k0 + kWindow);
#pragma unroll 1
    for (int q = 0; q < TILES; ++q) {
      const int kq = k0 + q * QK;
      if (kq < ke) {
        store_rows();
        if (kq + QK < ke) load_rows(kq + QK);
      }
      __syncthreads();
      if (kq < ke) {
#pragma unroll
        for (int s = 0; s < QK / 32; ++s) {
          if (kq + 32 * s >= ke) break;
          uint32_t af[MI][4], bf[4][2];
#pragma unroll
          for (int mi = 0; mi < MI; ++mi) {
            const int r = wm * RW + mi * 16 + (lane & 15);
            ldmatrix_x4(af[mi], &sm.a[r][q8_row_at<QK>(r, 2 * s + (lane >> 4))]);
          }
          const int kw = q * QK + 32 * s;  // of the window
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int c = wn * 32 + ni * 8 + (lane & 7);
            ldmatrix_x2(bf[ni], &sm.b[c][q8_col_at(c, kw + 16 * ((lane >> 3) & 1))]);
          }
          float da[MI][2], sums[MI][2][2];  // the rows' scales; their sums of 16 times them
#pragma unroll
          for (int mi = 0; mi < MI; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int r = wm * RW + mi * 16 + g + 8 * h;
              da[mi][h] = sm.da[r][s];
              if constexpr (MINS) sums[mi][h][0] = sm.ds[r][2 * s], sums[mi][h][1] = sm.ds[r][2 * s + 1];
            }
#pragma unroll
          for (int u = 0; u < HALVES; ++u) {
            const int sub = kw / SUB + u;
            float2 f[4], m[4];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
              const int c = wn * 32 + ni * 8 + 2 * t;
              f[ni] = *reinterpret_cast<const float2*>(&sm.wf[sub][c]);
              if constexpr (MINS) m[ni] = *reinterpret_cast<const float2*>(&sm.wm[sub][c]);
              else m[ni] = make_float2(0.f, 0.f);
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
              int C[4][4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 860
              constexpr bool from_zero = SUB == 32;  // the sums from zero, converted (kMagic's note)
#else
              constexpr bool from_zero = false;
#endif
#pragma unroll
              for (int ni = 0; ni < 4; ++ni) {
                if constexpr (from_zero) {
                  mma_s8_k32_zero(C[ni], af[mi], bf[ni]);
                } else {
                  C[ni][0] = C[ni][1] = C[ni][2] = C[ni][3] = kMagicBits;
                  if (SUB == 32) mma_s8_k32(C[ni], af[mi], bf[ni]);
                  else mma_s8_k16(C[ni], af[mi][2 * u], af[mi][2 * u + 1], bf[ni][u]);
                }
              }
#pragma unroll
              for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                  const int h = e >> 1;
                  const float fj = e & 1 ? f[ni].y : f[ni].x, mj = e & 1 ? m[ni].y : m[ni].x;
                  const float sum = from_zero ? __int2float_rn(C[ni][e]) : __int_as_float(C[ni][e]) - kMagic;
                  acc[mi][ni][e] = fmaf(sum, da[mi][h] * fj, acc[mi][ni][e]);
                  if constexpr (MINS)
                    acc[mi][ni][e] = fmaf(SUB == 16 ? sums[mi][h][u] : sums[mi][h][0] + sums[mi][h][1], mj,
                                          acc[mi][ni][e]);
                }
            }
          }
        }
      }
      __syncthreads();
    }
    if (!spread && k0 + kWindow >= run_end) {  // a run ends with this window: onto y, and from zero again
      each_out([&](int r, int c, float& v) {
        float* dst = y + size_t(r) * ldy + c;
        *dst = run == 0 && !accumulate ? v : *dst + v;
        v = 0.f;
      });
      run_end = run_start(++run + 1);
    }
  }
  if (!spread) return;
  fold_runs(each_out, y, ldy, M, N, accumulate, runs, parts, count);
#if defined(__CUDA_ARCH__) && !FLYX_TUNE_CANDIDATES
  }
#endif
}

// ---- rows of a table (the embedding) -----------------------------------------------------------
// Weights k .. k + 7 (k % 8 == 0) of row n of W [..][H] as float32: the planes' integers times their coefficients in
// one fused operation (the second product is exact), so each is the float32 of the weight.  The embedding and the
// drafts' candidate scores (k_dots_blocks) both read a row's values through this.
template <class F>
__device__ __forceinline__ void row_values8(const typename F::Local& local, const uint8_t* __restrict__ W,
                                            const uint16_t* __restrict__ D, size_t n, int H, int k, float* v) {
  constexpr int P = F::kPlanes;
  const int blocks = H / F::kValues, kb = k % F::kValues;
  const size_t at = n * blocks + k / F::kValues;
  const typename F::Lane ln = block_lane<F>(W, D, at, (kb >> 3) & 3);
  uint32_t w[2][4];
  words_at<F>(local, ln, kb / 32, w[0], w[P - 1]);
  const float d0 = scale_at(D, at * F::kScales), d1 = F::kScales > 1 ? scale_at(D, at * F::kScales + 1) : 0.f;
  const float c0 = F::coef(0, d0, d1), c1 = P > 1 ? F::coef(P - 1, d0, d1) : 0.f;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&w[0][e]));
    float2 b = make_float2(0.f, 0.f);
    if (P > 1) b = __half22float2(*reinterpret_cast<const __half2*>(&w[P - 1][e]));
    v[2 * e] = __fmaf_rn(c0, a.x, c1 * b.x);
    v[2 * e + 1] = __fmaf_rn(c0, a.y, c1 * b.y);
  }
}

// x[r] = W[tokens[r]], the weights as float32.  A thread per 8 weights.
template <class F>
__global__ void k_embed_blocks(Tables T, const int* tokens, const uint8_t* __restrict__ W, const uint16_t* __restrict__ D,
                               int H, float* x, int rows) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __syncthreads();
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int per_row = H / 8;
  if (i >= size_t(rows) * per_row) return;
  pdl_wait();  // the tokens come from the kernel before
  const int r = int(i / per_row), k = int(i % per_row) * 8;
  float v[8];
  row_values8<F>(local, W, D, size_t(tokens[r]), H, k, v);
  float* out = x + size_t(r) * H + k;
#pragma unroll
  for (int e = 0; e < 8; ++e) out[e] = v[e];
}

// out[r ld + c] = W[idx[r ld + c]] . x[r] for c < count[r ld] (rows of W [..][K] as k_embed_blocks gives them; idx
// and count at the same row stride ld as out, in elements): a warp a dot, lanes over K (lane l the 4-value chunks
// l, l + 32, ..), fixed FMA chains and a fixed tree across the warp, so a dot is the one the row decoded to float32 and
// then read 4 values a lane in that order gives.  A lane's chunk is half of 8 values row_values8 decodes, the other
// half its neighbour's (lanes 2 i, 2 i + 1): the pair decodes two steps' 8 values once each, the even lane this step's,
// the odd the next's, and they swap the halves the other needs (decoding all 8 for 4 took the time: 30 us for one row
// of MTP's 512 candidates on the 27B's head).  grid (rows, cap / kDotsWarps); block 32 kDotsWarps: warp w of block y
// the dot kDotsWarps y + w.
constexpr int kDotsWarps = 4;
template <class F>
__global__ void __launch_bounds__(32 * kDotsWarps) k_dots_blocks(Tables T, const uint8_t* __restrict__ W,
                                                                 const uint16_t* __restrict__ D, int K, const int* idx,
                                                                 const int* count, float* out, size_t ld,
                                                                 const float* x, int ldx) {
  pdl_trigger();
  __shared__ typename F::Local local;
  F::tables(T, local);
  __syncthreads();
  pdl_wait();  // the indices and x come from the kernels before
  const int r = blockIdx.x, lane = threadIdx.x & 31, c = blockIdx.y * kDotsWarps + (threadIdx.x >> 5);
  if (c >= count[size_t(r) * ld]) return;
  const size_t n = size_t(idx[size_t(r) * ld + c]);
  const float4* xr = reinterpret_cast<const float4*>(x + size_t(r) * ldx);
  const bool odd = lane & 1;
  const int chunks = K / 4;  // K % 128 == 0: every lane has as many (chunks / 32)
  float s = 0.f;
  int q = lane;
  for (; q + 32 < chunks; q += 64) {
    // steps q (this one) and q + 32 (the next): their 8 values, the even lane this step's, the odd the next's
    float v[8];
    row_values8<F>(local, W, D, n, K, 8 * ((q + (odd ? 32 : 0)) >> 1), v);
    // the even lane gets the next step's first half (its next chunk), the odd this step's second (its chunk now)
    float o[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) o[e] = __shfl_xor_sync(0xffffffffu, odd ? v[e] : v[4 + e], 1);
    const float* now = odd ? o : v;       // this step: even lane values 0-3, odd lane 4-7
    const float* next = odd ? v + 4 : o;  // the next: even lane values 0-3, odd lane 4-7
    const float4 b0 = xr[q], b1 = xr[q + 32];
    s = __fmaf_rn(now[0], b0.x, s);
    s = __fmaf_rn(now[1], b0.y, s);
    s = __fmaf_rn(now[2], b0.z, s);
    s = __fmaf_rn(now[3], b0.w, s);
    s = __fmaf_rn(next[0], b1.x, s);
    s = __fmaf_rn(next[1], b1.y, s);
    s = __fmaf_rn(next[2], b1.z, s);
    s = __fmaf_rn(next[3], b1.w, s);
  }
  if (q < chunks) {  // an odd number of steps a lane: the last one alone
    float v[8];
    row_values8<F>(local, W, D, n, K, 8 * (q >> 1), v);
    const float* a = v + 4 * (q & 1);
    const float4 b = xr[q];
    s = __fmaf_rn(a[0], b.x, s);
    s = __fmaf_rn(a[1], b.y, s);
    s = __fmaf_rn(a[2], b.z, s);
    s = __fmaf_rn(a[3], b.w, s);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) out[size_t(r) * ld + c] = s;
}

// False: the configuration's windows for this many rows do not fit a block's shared memory on this device.
template <class F, int NT, int NW, int KW>
bool launch_skinny(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  using Smem = BlocksSmem<F, NT, NW, KW>;
  const size_t bytes = Smem::bytes(rows < 8 ? rows : 8);
  if (bytes > device_max_smem_optin()) return false;
  static const bool attr =
      set_dynamic_smem(k_gemm_blocks<F, NT, NW, KW>, std::min(Smem::bytes(8), device_max_smem_optin()));
  (void)attr;
  const Scratch& sc = scratch(L);
  launch_dependent_smem(k_gemm_blocks<F, NT, NW, KW>, dim3(ceil_div(W.rows, 8 * NT * NW)), dim3(32 * NW * KW), bytes, L,
                        block_tables(), reinterpret_cast<const uint8_t*>(sc.pairs), sc.row_scale, W.data, W.scales,
                        W.rows, K, y, ldy, rows, int(accumulate));
  return true;
}

template <class F>
bool launch_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  switch (cfg) {
#define X(i, nt, nw, kw)                                                                                      \
  case i:                                                                                                     \
    if constexpr (FLYX_TUNE_CANDIDATES || (tables::kReachBlocksSkinny[tables::kArchBucketAny] & (1u << i))) { \
      return launch_skinny<F, nt, nw, kw>(W, K, y, ldy, rows, accumulate, L);                                 \
    }                                                                                                         \
    break;
    FLYX_BLOCKS_SKINNY(X)
#undef X
    default: break;
  }
  std::abort();
}

// Whether a GEMM's runs of K go side by side (a block a run, fold_runs): when that takes clearly fewer waves of
// the GPU's blocks.  `slots` blocks run at once (SMs times the blocks an SM holds); in turn the tiles take
// ceil(tiles / slots) waves, side by side their runs ceil(tiles runs / slots) waves of 1 / runs of a tile each, and
// each run also folds its sums onto y, so side by side must save a tenth.  The same bits either way.  On 70 SMs, 160
// tiles went 5 to 13% faster side by side, 272 and 320 3 to 10% slower;
// on 170, the 27B's down projection (4 runs) at 520-640 rows (200 tiles: 2 waves in turn, 1.25 side by
// side) 0.91 -> 0.59 ms, at 512 (160 tiles: one wave either way) 1 to 12% slower side by side.
inline bool side_by_side(int64_t tiles, int runs, int slots) {
  if (runs < 2 || slots < 1) return false;
  const int64_t turn = (tiles + slots - 1) / slots, side = (tiles * runs + slots - 1) / slots;
  return double(side) / runs < 0.9 * double(turn);
}

// spread: -1 the runs side by side when that saves waves (side_by_side); 0 never; 1 always (gemm_blocks_q8_cfg).  The
// same bits every way.
template <class F, int BN, int QK, int BM>
void launch_q8(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L,
               int spread = -1, int want_runs = 0) {
  using S = Q8Smem<F, BN, QK, BM>;
  static const bool attr = [&] {  // a window of codes and a tile of rows: more shared memory than a kernel has unasked
    if (sizeof(S) > device_max_smem_optin()) {
      std::fprintf(stderr, "flyx-cuda: the GEMM on block weights with inputs in 8 bits needs %zu bytes of shared "
                           "memory a block\n", sizeof(S));
      std::abort();
    }
    return set_dynamic_smem(k_gemm_blocks_q8<F, BN, QK, BM>, sizeof(S));
  }();
  (void)attr;
  const Scratch& sc = scratch(L);
  const int runs =
      want_runs > 0 ? std::min(want_runs, (K + kWindow - 1) / kWindow) : gemm_blocks_q8_runs(opts(L).tuning, W.rows, K);
  const int64_t tiles = int64_t(ceil_div(rows, BM)) * ceil_div(W.rows, BN);
  const bool fits = runs > 1 && tiles <= kQ8Tiles;
  float* parts = size_t(runs) * rows * W.rows <= sc.q8_parts_cap ? sc.q8_parts : nullptr;  // else in order
  static const int slots = resident_blocks(k_gemm_blocks_q8<F, BN, QK, BM>, 4 * BN, sizeof(S));
  const bool side = fits && (spread == 1 || (spread < 0 && side_by_side(tiles, runs, slots)));
  // a column's weights read once by each of its tiles of rows (and runs of K, side by side)
  const dim3 grid(ceil_div(rows, BM), ceil_div(W.rows, BN), side ? runs : 1);
  k_gemm_blocks_q8<F, BN, QK, BM><<<grid, 4 * BN, sizeof(S), L.st>>>(
      block_tables(), reinterpret_cast<const uint8_t*>(sc.q8), sc.q8_scale, W.data, W.scales, W.rows, K, y, ldy, rows,
      int(accumulate), runs, parts, sc.q8_count);
}

template <class F>
void launch_q8_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L,
                   int spread = -1, int runs = 0) {
  constexpr int di = tables::ggml_type_index(F::kDtype);
  switch (cfg) {
#define X(i, bn, qk, bm)                                                                                     \
  case i:                                                                                                    \
    if constexpr (FLYX_TUNE_CANDIDATES || (di >= 0 && (tables::kReachBlocksQ8[tables::kArchBucketAny][di] &  \
                                                       (1u << i)))) {                                        \
      return launch_q8<F, bn, qk, bm>(W, K, y, ldy, rows, accumulate, L, spread, runs);                      \
    }                                                                                                        \
    break;
    FLYX_BLOCKS_Q8(X)
#undef X
    default: break;
  }
  std::abort();
}

template <class F>
void launch_rows(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  using S = RowsSmem<F>;
  static const bool attr = [&] {  // a window of decoded weights: more shared memory than a kernel has unasked
    if (sizeof(S) > device_max_smem_optin()) {
      std::fprintf(stderr, "flyx-cuda: the prefill GEMM on block weights needs %zu bytes of shared memory a block\n",
                   sizeof(S));
      std::abort();
    }
    return set_dynamic_smem(k_gemm_blocks_rows<F>, sizeof(S));
  }();
  (void)attr;
  const Scratch& sc = scratch(L);
  const dim3 grid(ceil_div(W.rows, kRowsN), ceil_div(rows, kRowsM));
  k_gemm_blocks_rows<F><<<grid, 4 * kRowsN, sizeof(S), L.st>>>(block_tables(), reinterpret_cast<const uint16_t*>(sc.abuf),
                                                                K, sc.row_scale, sc.run_sums, W.data, W.scales, W.rows,
                                                                K, y, ldy, rows, int(accumulate));
}

template <class F>
void launch_embed(const int* tokens, const BlockMat& W, int H, float* x, int rows, cudaStream_t st) {
  k_embed_blocks<F><<<ceil_div(rows * (H / 8), 256), 256, 0, st>>>(block_tables(), tokens, W.data, W.scales, H, x,
                                                                  rows);
}

template <class F>
void launch_dots(const BlockMat& W, int K, const int* idx, const int* count, float* out, size_t ld, int cap,
                 const float* x, int ldx, int rows, cudaStream_t st) {
  k_dots_blocks<F><<<dim3(rows, ceil_div(cap, kDotsWarps)), 32 * kDotsWarps, 0, st>>>(block_tables(), W.data, W.scales,
                                                                                      K, idx, count, out, ld, x, ldx);
}

// One type's kernels, as gemm_blocks.cu calls them.
template <class F>
const BlockOps* ops_for() {
  static const BlockOps ops{&launch_cfg<F>, &launch_rows<F>, &launch_q8_cfg<F>, &launch_embed<F>, &launch_dots<F>};
  return &ops;
}

// The kernels of whichever of Fs has this dtype; nullptr if none does.
template <class... Fs>
const BlockOps* block_ops_of(int dtype) {
  const BlockOps* found = nullptr;
  ((found = !found && dtype == Fs::kDtype ? ops_for<Fs>() : found), ...);
  return found;
}

}  // namespace
}  // namespace flyx::cuda
