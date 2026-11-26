// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Draft tokens without reading the whole head: a 1-bit copy of the head
// scores every token roughly, the best kDraftCands of a row are scored
// exactly with the bf16 head rows, and their top k are the drafts.
//
// The two-stage head is EAGLE-4's ("EAGLE Efficient LM Head": cheap candidate
// retrieval over the full vocabulary, exact scores for the candidates); its
// first stage there is a trained low-rank projection.  Here it is the head's
// signs with one scale per 64 weights (mean |w|), which needs no training: on
// Qwen3.5-4B hidden states the exact argmax is among the 256 best rough
// scores at 99.8% of positions, its top 16 at 98%.
// Drafts never reach the output (the target verifies them).
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "ops/sampling/sampling.h"

namespace flyx::cuda {
namespace {

constexpr int kSignGroup = 64;     // weights per scale
constexpr int kPickThreads = 1024;
constexpr int kPerThread = 4;      // each thread's best rough scores
constexpr int kPool = kPickThreads * kPerThread;
constexpr int kCandCap = kDraftCandCap;  // candidates scored exactly: the kDraftCands best and their ties
constexpr int kPickChunks = 16;    // 8-value loads a lane takes of a candidate row: K <= 4096

// grid N; block 128.  Bit j of word w of row n set when weight 32 w + j >=
// +0; the scale of each 64 weights their mean |w| (bf16).  Laid out by tiles
// of 256 rows and 128 weights (sign_word_at, sign_scale_at): a tile's 4
// words (and 2 scales) of a row sit together, the rows one after another, so
// consecutive rows read consecutive bytes.
__global__ void k_sign_encode(const bf16* __restrict__ W, int K, uint32_t* __restrict__ bits,
                              bf16* __restrict__ scales) {
  const int n = blockIdx.x;
  const uint16_t* row = reinterpret_cast<const uint16_t*>(W) + size_t(n) * K;
  for (int w = threadIdx.x; w < K / 32; w += blockDim.x) {
    uint32_t b = 0;
    for (int j = 0; j < 32; ++j) b |= uint32_t((row[32 * w + j] >> 15) == 0) << j;
    bits[sign_word_at(n, w, K)] = b;
  }
  for (int g = threadIdx.x; g < K / kSignGroup; g += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j < kSignGroup; ++j) s += fabsf(__bfloat162float(W[size_t(n) * K + kSignGroup * g + j]));
    scales[sign_scale_at(n, g, K)] = __float2bfloat16(s / kSignGroup);
  }
}

// The same copy from float32 rows (a packed head's, as embed_blocks decodes them): rows n0 + blockIdx.x of the
// head from W [n][K].  A weight's bit is set when it is >= +0, as for the bf16 rows.
__global__ void k_sign_encode_f32(const float* __restrict__ W, int K, int n0, uint32_t* __restrict__ bits,
                                  bf16* __restrict__ scales) {
  const int n = n0 + int(blockIdx.x);
  const float* row = W + size_t(blockIdx.x) * K;
  for (int w = threadIdx.x; w < K / 32; w += blockDim.x) {
    uint32_t b = 0;
    for (int j = 0; j < 32; ++j) b |= uint32_t(!signbit(row[32 * w + j])) << j;
    bits[sign_word_at(n, w, K)] = b;
  }
  for (int g = threadIdx.x; g < K / kSignGroup; g += blockDim.x) {
    float s = 0.f;
    for (int j = 0; j < kSignGroup; ++j) s += fabsf(row[kSignGroup * g + j]);
    scales[sign_scale_at(n, g, K)] = __float2bfloat16(s / kSignGroup);
  }
}

// The sign bits of a word's values p and p + 1 (p even) on the binary16 halves of a scale pair: two weights of the
// 1-bit head as an MMA operand register.
__device__ __forceinline__ uint32_t sign_pair(uint32_t scale2, uint32_t negative, int p) {
  return scale2 | ((negative >> p) & 1u) << 15 | ((negative >> (p + 1)) & 1u) << 31;
}

// Rough scores y [rows][N] of 9..16 bf16 rows against the 1-bit head, on
// tensor cores in k_gemm_skinny_fp4's layout (spans of 128: lane t takes
// values 32 t .. 32 t + 31, one word of signs and one scale; MMA step j its
// values 4 j + {0, 1}, 4 j + {2, 3}; four fixed K chunks).  A weight is its
// group's scale with the sign of its bit, exact in bf16.
template <int NT, int WARPS, int CH = 4>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_sign(const bf16* xh, int ldk, const uint32_t* __restrict__ bits,
                                                        const bf16* __restrict__ scales, int N, int K,
                                                        float* __restrict__ y, int ldy, int rows) {
  constexpr int SPAN = 128, CPW = CH / WARPS, VPL = 32, STEPS = 8, BW = 16;
  static_assert(CH % WARPS == 0, "warps must divide the chunks");
  __shared__ float red[CH][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * 8 * NT, spans = K / SPAN;
  const bool ra = g < rows, rb = g + 8 < rows;
  const size_t oa = size_t(g) * ldk + VPL * t, ob = size_t(g + 8) * ldk + VPL * t;
  const uint32_t* bp[NT];
  const uint16_t* sp[NT];
  bool wv[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int n = n_base + i * 8 + g;
    wv[i] = n < N;
    const int row = wv[i] ? n : 0;
    bp[i] = bits + sign_word_at(row, t, K);  // word 4 s + t: + 1024 s
    sp[i] = reinterpret_cast<const uint16_t*>(scales) + sign_scale_at(row, t >> 1, K);  // group 2 s + t / 2: + 512 s
  }
#pragma unroll 1
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp * CPW + c;
    const int s0 = chunk * spans / CH, s1 = (chunk + 1) * spans / CH;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll 2
    for (int s = s0; s < s1; ++s) {
      const int k = s * SPAN;
      uint32_t wb[NT], sc[NT];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        wb[i] = wv[i] ? ldg_stream1(bp[i] + 1024 * size_t(s)) : 0u;
        sc[i] = wv[i] ? uint32_t(__ldg(sp[i] + 512 * size_t(s))) : 0u;
      }
      uint32_t av[BW], bv[BW];
#pragma unroll
      for (int q = 0; q < VPL / 8; ++q) {
        const uint4 z = make_uint4(0, 0, 0, 0);
        const uint4 a = ra ? *reinterpret_cast<const uint4*>(xh + oa + k + 8 * q) : z;
        const uint4 b = rb ? *reinterpret_cast<const uint4*>(xh + ob + k + 8 * q) : z;
        av[4 * q] = a.x, av[4 * q + 1] = a.y, av[4 * q + 2] = a.z, av[4 * q + 3] = a.w;
        bv[4 * q] = b.x, bv[4 * q + 1] = b.y, bv[4 * q + 2] = b.z, bv[4 * q + 3] = b.w;
      }
#pragma unroll
      for (int j = 0; j < STEPS; ++j) {
        const uint32_t a[4] = {av[2 * j], bv[2 * j], av[2 * j + 1], bv[2 * j + 1]};
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          const uint32_t s2 = sc[i] | sc[i] << 16, nb = ~wb[i];
          const uint32_t bf[2] = {sign_pair(s2, nb, 4 * j), sign_pair(s2, nb, 4 * j + 2)};
          mma_bf16(acc[i], a, bf);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[i][e];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
    const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) v += red[c][i][l][e];
    const int r = (l >> 2) + (e >> 1) * 8, col = n_base + i * 8 + 2 * (l & 3) + (e & 1);
    if (r < rows && col < N) y[size_t(r) * ldy + col] = v;
  }
}

// The same for 2..8 rows with the operands the other way round: the head's columns are the MMA's 16 rows and the
// rows of x its 8 columns, so no part of a tile is rows of zeros (k_gemm_sign with 7 rows multiplied 9 rows of zeros
// too, and its tensor cores were what held it: 244 us for the 27B's 7 drafts a DFlash2 round).  A step's local k is
// the same values of the span as there, so each score is the sum of the same 16 products a step, the steps and the
// chunks in the same order: a row's bits are those k_gemm_sign gives it (cuda_op_test checks).  NT tiles of 16
// columns a block.
template <int NT, int WARPS, int CH = 4>
__global__ void __launch_bounds__(32 * WARPS) k_gemm_sign_cols(const bf16* xh, int ldk, const uint32_t* __restrict__ bits,
                                                             const bf16* __restrict__ scales, int N, int K,
                                                             float* __restrict__ y, int ldy, int rows) {
  constexpr int SPAN = 128, CPW = CH / WARPS, VPL = 32, STEPS = 8, BW = 16;
  static_assert(CH % WARPS == 0, "warps must divide the chunks");
  __shared__ float red[CH][NT][32][4];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int n_base = blockIdx.x * 16 * NT, spans = K / SPAN;
  const bool rx = g < rows;  // lane group g brings row g of x (the MMA's column g)
  const size_t ox = size_t(g) * ldk + VPL * t;
  const uint32_t* bp[NT][2];
  const uint16_t* sp[NT][2];
  bool wv[NT][2];
#pragma unroll
  for (int i = 0; i < NT; ++i)
#pragma unroll
    for (int h = 0; h < 2; ++h) {  // the tile's columns g (MMA row g) and g + 8 (row g + 8)
      const int n = n_base + 16 * i + 8 * h + g;
      wv[i][h] = n < N;
      const int row = wv[i][h] ? n : 0;
      bp[i][h] = bits + sign_word_at(row, t, K);  // word 4 s + t: + 1024 s
      sp[i][h] = reinterpret_cast<const uint16_t*>(scales) + sign_scale_at(row, t >> 1, K);  // + 512 s
    }
#pragma unroll 1
  for (int c = 0; c < CPW; ++c) {
    const int chunk = warp * CPW + c;
    const int s0 = chunk * spans / CH, s1 = (chunk + 1) * spans / CH;
    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll 2
    for (int s = s0; s < s1; ++s) {
      const int k = s * SPAN;
      uint32_t wb[NT][2], sc[NT][2];
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          wb[i][h] = wv[i][h] ? ldg_stream1(bp[i][h] + 1024 * size_t(s)) : 0u;
          sc[i][h] = wv[i][h] ? uint32_t(__ldg(sp[i][h] + 512 * size_t(s))) : 0u;
        }
      uint32_t xv[BW];
#pragma unroll
      for (int q = 0; q < VPL / 8; ++q) {
        const uint4 a = rx ? *reinterpret_cast<const uint4*>(xh + ox + k + 8 * q) : make_uint4(0, 0, 0, 0);
        xv[4 * q] = a.x, xv[4 * q + 1] = a.y, xv[4 * q + 2] = a.z, xv[4 * q + 3] = a.w;
      }
#pragma unroll
      for (int j = 0; j < STEPS; ++j) {
        const uint32_t b[2] = {xv[2 * j], xv[2 * j + 1]};  // values 4 j + {0, 1}, 4 j + {2, 3} of row g
#pragma unroll
        for (int i = 0; i < NT; ++i) {
          uint32_t a[4];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t s2 = sc[i][h] | sc[i][h] << 16, nb = ~wb[i][h];
            a[h] = sign_pair(s2, nb, 4 * j);          // column g + 8 h, local k 2 t, 2 t + 1
            a[2 + h] = sign_pair(s2, nb, 4 * j + 2);  // local k 2 t + 8, 2 t + 9
          }
          mma_bf16(acc[i], a, b);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) red[chunk][i][lane][e] = acc[i][e];
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < NT * 32 * 4; idx += blockDim.x) {
    const int i = idx / 128, l = (idx / 4) & 31, e = idx & 3;
    float v = 0.f;
#pragma unroll
    for (int c = 0; c < CH; ++c) v += red[c][i][l][e];
    const int r = 2 * (l & 3) + (e & 1), col = n_base + 16 * i + (l >> 2) + 8 * (e >> 1);
    if (r < rows && col < N) y[size_t(r) * ldy + col] = v;
  }
}

// One row (each MTP step) and bit-identical multi-row (2..4 concurrent MTP sequences): on CUDA cores, without the
// idle rows of an m16 tile.  The block builds, for every 4 consecutive values of x in a K tile, the signed sums over
// all 16 sign patterns (+/-x +/-y +/-z +/-w), so 4 weights cost one table read and one add with no group-sum buffer
// or 2*p - gsum fixup.  Tiling K in 1280-weight chunks for R = 1 keeps shared memory at 20 KB (up to 4 blocks/SM vs
// 1 block/SM for the 82 KB full-K table at K = 5120) while processing T = 2 tiles of 256 vocab rows per block to
// amortize table construction and overlap weight loads.  R = 2..4 uses the exact same signed-table formula and
// accumulation order so multi-row exact_1row matches R = 1 bit-for-bit.
constexpr int kSign1Rows = 256;
constexpr int kSignTiles1 = 2;
constexpr int kSignTilesR = 4;

template <int R, int T, int TILE_G2, bool ACCUM_BIAS = false>
__global__ void __launch_bounds__(kSign1Rows)
k_sign_rough_lut(const float* __restrict__ x, int ldx, const uint32_t* __restrict__ bits,
                 const bf16* __restrict__ scales, const float* __restrict__ bias, int N, int K, float* __restrict__ y,
                 int ldy) {
  constexpr int kTileQ = TILE_G2 * 32;
  __shared__ float tab[R][kTileQ][16];

  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * T * kSign1Rows + tid;
  const uint4* wb[T];
  const uint32_t* sc2[T];
  bool valid[T];
  float score[T][R];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const int n = n0 + t * kSign1Rows;
    valid[t] = n < N;
    if (valid[t]) {
      wb[t] = reinterpret_cast<const uint4*>(bits + sign_word_at(n, 0, K));
      sc2[t] = reinterpret_cast<const uint32_t*>(scales + sign_scale_at(n, 0, K));
    }
#pragma unroll
    for (int r = 0; r < R; ++r) score[t][r] = 0.f;
  }

  const int total_g2 = K / 128;
  for (int g2_base = 0; g2_base < total_g2; g2_base += TILE_G2) {
    const int ng2 = min(TILE_G2, total_g2 - g2_base);
    const int nq = ng2 * 32;
    const int q_base = g2_base * 32;

    const int total_entries = R * nq * 16;
    for (int idx = tid; idx < total_entries; idx += kSign1Rows) {
      const int m = idx & 15;
      const int rq = idx >> 4;
      const int r = (R == 1) ? 0 : rq / nq;
      const int q = (R == 1) ? rq : rq - r * nq;
      const float4 v = *reinterpret_cast<const float4*>(x + size_t(r) * ldx + 4 * (q_base + q));
      const float sx = (m & 1) ? 1.f : -1.f;
      const float sy = (m & 2) ? 1.f : -1.f;
      const float sz = (m & 4) ? 1.f : -1.f;
      const float sw = (m & 8) ? 1.f : -1.f;
      tab[r][q][m] = __fmaf_rn(sx, v.x, __fmaf_rn(sy, v.y, __fmaf_rn(sz, v.z, sw * v.w)));
    }
    __syncthreads();

#pragma unroll 2
    for (int dg = 0; dg < ng2; ++dg) {
      const int g2 = g2_base + dg;
      uint4 w[T];
      uint32_t s2[T];
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if constexpr (ACCUM_BIAS) {
          w[t] = valid[t] ? __ldg(wb[t] + 256 * size_t(g2)) : make_uint4(0, 0, 0, 0);
        } else {
          w[t] = valid[t] ? ldg_stream(wb[t] + 256 * size_t(g2)) : make_uint4(0, 0, 0, 0);
        }
        s2[t] = valid[t] ? __ldg(sc2[t] + 256 * size_t(g2)) : 0u;
      }
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (!valid[t]) continue;
        const uint32_t ws[4] = {w[t].x, w[t].y, w[t].z, w[t].w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int g = 2 * dg + h;
            float p = 0.f;
#pragma unroll
            for (int ww = 0; ww < 2; ++ww) {
              const uint32_t b = ws[2 * h + ww];
              const int q0 = (kSignGroup * g + 32 * ww) / 4;
#pragma unroll
              for (int j = 0; j < 8; ++j) p += tab[r][q0 + j][(b >> (4 * j)) & 15];
            }
            score[t][r] = __fmaf_rn(__uint_as_float(h ? s2[t] & 0xFFFF0000u : s2[t] << 16), p, score[t][r]);
          }
        }
      }
    }
    if (g2_base + TILE_G2 < total_g2) __syncthreads();
  }

#pragma unroll
  for (int t = 0; t < T; ++t) {
    if (!valid[t]) continue;
    const int n = n0 + t * kSign1Rows;
    if constexpr (ACCUM_BIAS) {
      const float bval = bias ? __ldg(bias + n) : 0.f;
#pragma unroll
      for (int r = 0; r < R; ++r) y[size_t(r) * ldy + n] = y[size_t(r) * ldy + n] + score[t][r] + bval;
    } else {
#pragma unroll
      for (int r = 0; r < R; ++r) y[size_t(r) * ldy + n] = score[t][r];
    }
  }
}

// Picking, in three kernels per call (rows in parallel).  Row r's rough
// scores, once read, give way to its candidates: [kCandCap] indices, then
// [kCandCap] exact logits, then the count, at the start of its rough row.
//
// Each of the kPickThreads logical scanners keeps its kPerThread best rough
// scores (indices rising: an equal score keeps the lower index).  For small
// row counts over large vocabularies, k_draft_select_part distributes the 1024
// logical scanners across kSelPartBlocks warps so the rough-score scan runs
// across many SMs in parallel before k_draft_select_finish runs radix select
// over the exact same kPool candidate set.
constexpr int kSelPartBlocks = 32;
constexpr int kSelPartBlockThreads = 32;
constexpr int kSelPartThreads = kSelPartBlocks * kSelPartBlockThreads;
static_assert(kSelPartThreads == kPickThreads,
              "partitioned select must match k_draft_select scanner count for bit-exact candidate sets");

__device__ __forceinline__ int draft_select_scan_thread(const float* __restrict__ row, int ld_rough, int V, int gtid,
                                                        uint32_t out_k[kPerThread], int out_i[kPerThread]) {
  float l0 = -INFINITY, l1 = -INFINITY, l2 = -INFINITY, l3 = -INFINITY;
  int i0 = INT_MAX, i1 = INT_MAX, i2 = INT_MAX, i3 = INT_MAX;
  auto insert = [&](float a, int v) {  // sorted l0 >= l1 >= l2 >= l3, in registers
    if (!(a > l3)) return;
    l3 = a, i3 = v;
    if (l3 > l2) { const float t = l2; l2 = l3, l3 = t; const int u = i2; i2 = i3, i3 = u; }
    if (l2 > l1) { const float t = l1; l1 = l2, l2 = t; const int u = i1; i1 = i2, i2 = u; }
    if (l1 > l0) { const float t = l0; l0 = l1, l1 = t; const int u = i0; i0 = i1, i1 = u; }
  };
  const float4* row4 = reinterpret_cast<const float4*>(row);
  const int V4 = (ld_rough % 4 == 0) ? V / 4 : 0;
  int v4 = gtid;
  for (; v4 + kPickThreads < V4; v4 += 2 * kPickThreads) {  // two loads in flight
    const float4 a = row4[v4], b = row4[v4 + kPickThreads];
    insert(a.x, 4 * v4), insert(a.y, 4 * v4 + 1), insert(a.z, 4 * v4 + 2), insert(a.w, 4 * v4 + 3);
    const int vb = 4 * (v4 + kPickThreads);
    insert(b.x, vb), insert(b.y, vb + 1), insert(b.z, vb + 2), insert(b.w, vb + 3);
  }
  for (; v4 < V4; v4 += kPickThreads) {
    const float4 a = row4[v4];
    insert(a.x, 4 * v4), insert(a.y, 4 * v4 + 1), insert(a.z, 4 * v4 + 2), insert(a.w, 4 * v4 + 3);
  }
  for (int v = 4 * V4 + gtid; v < V; v += kPickThreads) insert(row[v], v);
  const float lv[4] = {l0, l1, l2, l3};
  const int li[4] = {i0, i1, i2, i3};
  int valid = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    out_k[i] = li[i] == INT_MAX ? 0u : float_order_u32(lv[i]);
    out_i[i] = li[i];
    valid += li[i] != INT_MAX;
  }
  return valid;
}

__device__ __forceinline__ void draft_select_radix(uint32_t* keys, int* pool_i, int valid_local, float* __restrict__ row) {
  __shared__ int hist[256];
  __shared__ int s_digit, s_need, s_count;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid == 0) s_count = 0;
  __syncthreads();
  int wcnt = valid_local;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) wcnt += __shfl_xor_sync(0xffffffffu, wcnt, o);
  if (lane == 0 && wcnt > 0) atomicAdd(&s_count, wcnt);
  __syncthreads();
  // radix select: the key of the kDraftCands-th largest (0: every one, when there are no more)
  uint32_t prefix = 0, mask = 0;
  const bool few = s_count <= kDraftCands;
  if (tid == 0) s_need = kDraftCands;
  for (int shift = 24; shift >= 0 && !few; shift -= 8) {
    if (tid < 256) hist[tid] = 0;
    __syncthreads();
    for (int i = tid; i < kPool; i += kPickThreads)
      if ((keys[i] & mask) == prefix && pool_i[i] != INT_MAX) atomicAdd(&hist[(keys[i] >> shift) & 255], 1);
    __syncthreads();
    if (warp == 0) {  // the digit of the need-th largest among the keys matching the prefix
      // lane l holds bins 255 - 8 l .. 248 - 8 l; counts from the top, then the lane and bin it lands in
      int c[8], sum = 0;
#pragma unroll
      for (int j = 0; j < 8; ++j) sum += (c[j] = hist[255 - 8 * lane - j]);
      int incl = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += u;
      }
      const int need = s_need, excl = incl - sum;
      const unsigned hit = __ballot_sync(0xffffffffu, excl < need && incl >= need);
      if (hit && lane == __ffs(hit) - 1) {
        int left = need - excl, d = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j)
          if (left > 0 && d == j) {
            if (c[j] < left) left -= c[j], d = j + 1;
          }
        s_digit = 255 - 8 * lane - d;
        s_need = left;
      }
    }
    __syncthreads();
    prefix |= uint32_t(s_digit) << shift;
    mask |= 255u << shift;
  }
  __syncthreads();
  if (tid == 0) s_count = 0;
  __syncthreads();  // every rough score of the row has been read: the candidates may overwrite them
  int* cand_i = reinterpret_cast<int*>(row);
  for (int i = tid; i < kPool; i += kPickThreads)
    if (pool_i[i] != INT_MAX && keys[i] >= prefix) {
      const int at = atomicAdd(&s_count, 1);
      if (at < kCandCap) cand_i[at] = pool_i[i];
    }
  __syncthreads();
  const int count = min(s_count, kCandCap);
  if (tid == 0) cand_i[2 * kCandCap] = count;
  for (int i = count + tid; i < kCandCap; i += kPickThreads) cand_i[i] = 0;  // a row of the head: decodable
}

__global__ void __launch_bounds__(kPickThreads) k_draft_select(float* __restrict__ rough, int ld_rough, int V) {
  __shared__ uint32_t keys[kPool];
  __shared__ int pool_i[kPool];
  const int r = blockIdx.x, tid = threadIdx.x;
  float* row = rough + size_t(r) * ld_rough;
  uint32_t lk[kPerThread];
  int li[kPerThread];
  const int valid = draft_select_scan_thread(row, ld_rough, V, tid, lk, li);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    keys[i * kPickThreads + tid] = lk[i];
    pool_i[i * kPickThreads + tid] = li[i];
  }
  draft_select_radix(keys, pool_i, valid, row);
}

__global__ void __launch_bounds__(kSelPartBlockThreads)
k_draft_select_part(const float* __restrict__ rough, int ld_rough, int V, uint32_t* __restrict__ part_keys,
                    int* __restrict__ part_idx) {
  const int r = blockIdx.y;
  const int gtid = blockIdx.x * kSelPartBlockThreads + threadIdx.x;
  const float* row = rough + size_t(r) * ld_rough;
  uint32_t lk[kPerThread];
  int li[kPerThread];
  (void)draft_select_scan_thread(row, ld_rough, V, gtid, lk, li);
  uint32_t* pk = part_keys + size_t(r) * kPool;
  int* pi = part_idx + size_t(r) * kPool;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    pk[i * kPickThreads + gtid] = lk[i];
    pi[i * kPickThreads + gtid] = li[i];
  }
}

__global__ void __launch_bounds__(kPickThreads)
k_draft_select_finish(float* __restrict__ rough, int ld_rough, const uint32_t* __restrict__ part_keys,
                      const int* __restrict__ part_idx) {
  __shared__ uint32_t keys[kPool];
  __shared__ int pool_i[kPool];
  const int r = blockIdx.x, tid = threadIdx.x;
  const uint32_t* pk = part_keys + size_t(r) * kPool;
  const int* pi = part_idx + size_t(r) * kPool;
  int valid = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const int at = i * kPickThreads + tid;
    const uint32_t k = pk[at];
    const int idx = pi[at];
    keys[at] = k;
    pool_i[at] = idx;
    valid += idx != INT_MAX;
  }
  draft_select_radix(keys, pool_i, valid, rough + size_t(r) * ld_rough);
}

void launch_draft_select(float* rough, int ld, int V, int rows, const Launch& L) {
  constexpr size_t kRowBytes = size_t(kPool) * (sizeof(uint32_t) + sizeof(int));
  // Only partition when all rows fit in one Scratch::q8 buffer so CUDA round graphs add at most 1 node and
  // k_draft_select_part + k_draft_select_finish produces the exact same candidate pool as k_draft_select.
  if (V >= 16384 && L.scratch && L.scratch->q8 && L.scratch->q8_round_cap >= size_t(rows) * kRowBytes) {
    uint32_t* part_keys = reinterpret_cast<uint32_t*>(L.scratch->q8);
    int* part_idx = reinterpret_cast<int*>(part_keys + size_t(rows) * kPool);
    k_draft_select_part<<<dim3(kSelPartBlocks, rows), kSelPartBlockThreads, 0, L.st>>>(rough, ld, V, part_keys,
                                                                                       part_idx);
    k_draft_select_finish<<<rows, kPickThreads, 0, L.st>>>(rough, ld, part_keys, part_idx);
    return;
  }
  k_draft_select<<<rows, kPickThreads, 0, L.st>>>(rough, ld, V);
}

// grid (rows, kCandCap / 8); block 256: warp w scores candidate 8 blockIdx.y + w, the bf16 head row
// against x with lanes over K (lane l the 8-value chunks l, l + 32, ..; all its loads in flight), a
// fixed tree across the warp.
__global__ void __launch_bounds__(256) k_draft_score(float* __restrict__ rough, int ld_rough,
                                                     const bf16* __restrict__ W, int K, const float* __restrict__ x,
                                                     int ldx) {
  const int r = blockIdx.x, lane = threadIdx.x & 31, c = blockIdx.y * 8 + (threadIdx.x >> 5);
  float* row = rough + size_t(r) * ld_rough;
  const int* cand_i = reinterpret_cast<const int*>(row);
  if (c >= reinterpret_cast<const int*>(row)[2 * kCandCap]) return;
  const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(cand_i[c]) * K);
  const float* xr = x + size_t(r) * ldx;
  uint4 a[kPickChunks];
#pragma unroll
  for (int m = 0; m < kPickChunks; ++m) {
    const int q = lane + 32 * m;
    a[m] = q < K / 8 ? __ldg(wr + q) : make_uint4(0, 0, 0, 0);
  }
  float s = 0.f;
#pragma unroll
  for (int m = 0; m < kPickChunks; ++m) {
    const int q = lane + 32 * m;
    if (q >= K / 8) break;
    const float4 x0 = *reinterpret_cast<const float4*>(xr + 8 * q), x1 = *reinterpret_cast<const float4*>(xr + 8 * q + 4);
    const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    const uint32_t wa[4] = {a[m].x, a[m].y, a[m].z, a[m].w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wa[e]));
      s = __fmaf_rn(f.x, xv[2 * e], s);
      s = __fmaf_rn(f.y, xv[2 * e + 1], s);
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) row[kCandCap + c] = s;
}

// General bf16 candidate-scoring kernel for arbitrary K, optional accumulate into existing candidate scores,
// optional per-token bias, and optional logit softcap (shared by additive / softcapped draft heads).
__global__ void __launch_bounds__(256)
k_draft_score_gen(float* __restrict__ rough, int ld_rough, const bf16* __restrict__ W, const float* __restrict__ bias,
                  int K, const float* __restrict__ x, int ldx, bool accumulate, float softcap) {
  const int r = blockIdx.x, lane = threadIdx.x & 31, c = blockIdx.y * 8 + (threadIdx.x >> 5);
  float* row = rough + size_t(r) * ld_rough;
  const int* cand_i = reinterpret_cast<const int*>(row);
  if (c >= cand_i[2 * kCandCap]) return;
  const int vid = cand_i[c];
  const float* xr = x + size_t(r) * ldx;
  float s = 0.f;
  const bool vec8 =
      (K & 7) == 0 && (reinterpret_cast<uintptr_t>(W) & 15u) == 0 && (reinterpret_cast<uintptr_t>(xr) & 15u) == 0;
  if (vec8) {
    const uint4* wr = reinterpret_cast<const uint4*>(W + size_t(vid) * K);
    const int n8 = K >> 3;
    for (int q = lane; q < n8; q += 32) {
      const uint4 a = __ldg(wr + q);
      const float4 x0 = *reinterpret_cast<const float4*>(xr + 8 * q);
      const float4 x1 = *reinterpret_cast<const float4*>(xr + 8 * q + 4);
      const uint32_t wa[4] = {a.x, a.y, a.z, a.w};
      const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wa[e]));
        s = __fmaf_rn(f.x, xv[2 * e], s);
        s = __fmaf_rn(f.y, xv[2 * e + 1], s);
      }
    }
  } else {
    const bf16* wr = W + size_t(vid) * K;
    for (int k = lane; k < K; k += 32) s = __fmaf_rn(__bfloat162float(__ldg(wr + k)), xr[k], s);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) {
    float z = (accumulate ? row[kCandCap + c] : 0.f) + s + (bias ? bias[vid] : 0.f);
    if (softcap > 0.f) z = softcap * tanhf(z / softcap);
    row[kCandCap + c] = z;
  }
}

// grid rows; block kCandCap: combines an optional packed candidate-score vector + optional vocab bias + optional
// softcap into rough[r * ld_rough + kCandCap + c].
__global__ void __launch_bounds__(kCandCap)
k_draft_combine_cands(const float* __restrict__ add, int ld_add, const float* __restrict__ bias, float softcap,
                      float* __restrict__ rough, int ld_rough) {
  const int r = blockIdx.x, c = threadIdx.x;
  float* row = rough + size_t(r) * ld_rough;
  const int* cand_i = reinterpret_cast<const int*>(row);
  if (c >= cand_i[2 * kCandCap]) return;
  const int vid = cand_i[c];
  float z = row[kCandCap + c] + (add ? add[size_t(r) * ld_add + c] : 0.f) + (bias ? bias[vid] : 0.f);
  if (softcap > 0.f) z = softcap * tanhf(z / softcap);
  row[kCandCap + c] = z;
}

// One warp per row: the top k of the candidates by exact logit, high to low, the lowest index on ties.
// When G && P && G->temperature > 0 && k_out == 1, applies coupled Gumbel-Max sampling over the exact scored
// candidates with the same top_k/top_p/kSampleZMin filters and ops::sample_uniform(G->seed, pos, vid) noise as
// k_sample (ops/sampling/sampling.h), so probabilistic drafters couple directly to target verification while keeping
// the same CUDA graph for greedy and sampled requests.
// When prob_out or margin_out is non-null, records the top candidate's truncated softmax probability over all scored
// candidates and its logit margin above the runner-up (in Gumbel-perturbed logit space when sampling).
__global__ void k_draft_topk(float* __restrict__ rough, int ld_rough, int k_out, int* __restrict__ idx_out,
                             float* __restrict__ val_out, float* __restrict__ prob_out,
                             float* __restrict__ margin_out, const DevParams* P, int pos_add, const DevGen* G) {
  const int r = blockIdx.x, lane = threadIdx.x;
  float* row = rough + size_t(r) * ld_rough;
  int* cand_i = reinterpret_cast<int*>(row);
  const float* cand_v = row + kCandCap;
  const int count = cand_i[2 * kCandCap];
  if (G && P && G->temperature > 0.f && k_out == 1) {
    __shared__ float s_z[kCandCap];
    const float T = G->temperature, invT = 1.f / T;
    float m1 = -INFINITY;
    for (int c = lane; c < count; c += 32) m1 = fmaxf(m1, cand_v[c]);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, s));
    int nc = 0;
    for (int c = lane; c < count; c += 32) {
      const float d = cand_v[c] - m1;
      const float z = isinf(invT) ? __fdiv_rn(d, T) : d * invT;
      s_z[c] = z;
      nc += (z > ops::kSampleZMin) ? 1 : 0;
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) nc += __shfl_xor_sync(0xffffffffu, nc, s);
    uint32_t thr = 0;
    const int k = G->top_k;
    if (k > 0 && k < nc) {
      uint32_t lo = 0, hi = 0xFFFFFFFFu;
      while (lo < hi) {
        const uint32_t mid = uint32_t((uint64_t(lo) + hi + 1) >> 1);
        int cnt = 0;
        for (int c = lane; c < count; c += 32) {
          const float z = s_z[c];
          cnt += (z > ops::kSampleZMin && ops::float_key(__float_as_uint(z)) >= mid) ? 1 : 0;
        }
#pragma unroll
        for (int s = 16; s > 0; s >>= 1) cnt += __shfl_xor_sync(0xffffffffu, cnt, s);
        if (cnt >= k) lo = mid;
        else hi = mid - 1;
      }
      thr = lo;
    }
    if (G->top_p < 1.f) {
      float ms = 0.f;
      for (int c = lane; c < count; c += 32) {
        const float z = s_z[c];
        if (z > ops::kSampleZMin && ops::float_key(__float_as_uint(z)) >= thr) ms += __expf(z);
      }
      const float target = G->top_p * warp_sum(ms);
      uint32_t lo = thr, hi = 0xFFFFFFFFu;
      while (lo < hi) {
        const uint32_t mid = uint32_t((uint64_t(lo) + hi + 1) >> 1);
        float sm = 0.f;
        for (int c = lane; c < count; c += 32) {
          const float z = s_z[c];
          if (z > ops::kSampleZMin && ops::float_key(__float_as_uint(z)) >= mid) sm += __expf(z);
        }
        if (warp_sum(sm) >= target) lo = mid;
        else hi = mid - 1;
      }
      thr = lo;
    }
    const int64_t pos = int64_t(P->position) + pos_add + r;
    float best = -INFINITY;
    int bi = INT_MAX, bc = -1;
    for (int c = lane; c < count; c += 32) {
      const float z = s_z[c];
      if (!(z > ops::kSampleZMin) || ops::float_key(__float_as_uint(z)) < thr) continue;
      const int vid = cand_i[c];
      const float u = ops::sample_uniform(G->seed, pos, vid);
      const float gs = z - __logf(-__logf(u));
      if (score_better(gs, vid, best, bi)) best = gs, bi = vid, bc = c;
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, s);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, s), oc = __shfl_xor_sync(0xffffffffu, bc, s);
      if (score_better(ov, oi, best, bi)) best = ov, bi = oi, bc = oc;
    }
    if (prob_out || margin_out) {
      const float s1 = __shfl_sync(0xffffffffu, best, 0);
      const int c1 = __shfl_sync(0xffffffffu, bc, 0);
      float zsum = 0.f, s2 = -INFINITY;
      for (int c = lane; c < count; c += 32) {
        const float z = s_z[c];
        if (!(z > ops::kSampleZMin) || ops::float_key(__float_as_uint(z)) < thr) continue;
        const float u = ops::sample_uniform(G->seed, pos, cand_i[c]);
        const float gs = z - __logf(-__logf(u));
        zsum += expf(gs - s1);
        if (c != c1 && gs > s2) s2 = gs;
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        zsum += __shfl_xor_sync(0xffffffffu, zsum, s);
        s2 = fmaxf(s2, __shfl_xor_sync(0xffffffffu, s2, s));
      }
      if (lane == 0) {
        if (prob_out) prob_out[r] = (c1 >= 0 && zsum > 0.f) ? (1.f / zsum) : 0.f;
        if (margin_out) margin_out[r] = (c1 >= 0) ? (isfinite(s2) ? fmaxf(0.f, s1 - s2) : 12.f) : 0.f;
      }
    }
    if (lane == 0) {
      idx_out[r] = bi == INT_MAX ? 0 : bi;
      if (val_out) val_out[r] = bc >= 0 ? cand_v[bc] : -INFINITY;
    }
    return;
  }
  if (prob_out || margin_out) {
    float m1 = -INFINITY;
    int i1 = INT_MAX, c1 = -1;
    for (int c = lane; c < count; c += 32)
      if (score_better(cand_v[c], cand_i[c], m1, i1)) m1 = cand_v[c], i1 = cand_i[c], c1 = c;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, m1, s);
      const int oi = __shfl_xor_sync(0xffffffffu, i1, s), oc = __shfl_xor_sync(0xffffffffu, c1, s);
      if (score_better(ov, oi, m1, i1)) m1 = ov, i1 = oi, c1 = oc;
    }
    m1 = __shfl_sync(0xffffffffu, m1, 0);
    c1 = __shfl_sync(0xffffffffu, c1, 0);
    float zsum = 0.f, m2 = -INFINITY;
    for (int c = lane; c < count; c += 32) {
      const float v = cand_v[c];
      zsum += expf(v - m1);
      if (c != c1 && v > m2) m2 = v;
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      zsum += __shfl_xor_sync(0xffffffffu, zsum, s);
      m2 = fmaxf(m2, __shfl_xor_sync(0xffffffffu, m2, s));
    }
    if (lane == 0) {
      if (prob_out) prob_out[r] = (count > 0 && zsum > 0.f) ? (1.f / zsum) : 0.f;
      if (margin_out) margin_out[r] = (count > 1 && isfinite(m2)) ? (m1 - m2) : 0.f;
    }
  }
  for (int o = 0; o < k_out; ++o) {
    float bv = -INFINITY;
    int bi = INT_MAX, bc = -1;
    for (int c = lane; c < count; c += 32)
      if (cand_i[c] >= 0 && score_better(cand_v[c], cand_i[c], bv, bi)) bv = cand_v[c], bi = cand_i[c], bc = c;
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, s);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, s), oc = __shfl_xor_sync(0xffffffffu, bc, s);
      if (score_better(ov, oi, bv, bi)) bv = ov, bi = oi, bc = oc;
    }
    if (lane == 0) {
      idx_out[size_t(r) * k_out + o] = bi == INT_MAX ? 0 : bi;
      if (val_out) val_out[size_t(r) * k_out + o] = bv;
      if (bc >= 0) cand_i[bc] = -1;  // taken
    }
    __syncwarp();
  }
}

}  // namespace

size_t sign_head_bytes(int N, int K) {
  const size_t rows = size_t(N + 255) / 256 * 256;  // whole tiles
  return rows * K / 8 + rows * (K / kSignGroup) * sizeof(bf16);
}

void make_sign_head(const bf16* W, int N, int K, SignHead* h, cudaStream_t st) {
  k_sign_encode<<<N, 128, 0, st>>>(W, K, h->bits, h->scales);
}

void make_sign_head_rows(const float* W, int n0, int n, int K, SignHead* h, cudaStream_t st) {
  k_sign_encode_f32<<<n, 128, 0, st>>>(W, K, n0, h->bits, h->scales);
}

void gemm_sign_rough(const float* x, int ldx, const SignHead& h, int N, int K, float* y, int ldy, int rows,
                     const Launch& L, bool exact_1row) {
  const bool fits_1row = K % 128 == 0;
  if (rows == 1 && fits_1row) {
    const int grid = ceil_div(N, kSignTiles1 * kSign1Rows);
    k_sign_rough_lut<1, kSignTiles1, 10><<<grid, kSign1Rows, 0, L.st>>>(x, ldx, h.bits, h.scales, nullptr, N, K, y,
                                                                        ldy);
    return;
  }
  if (exact_1row && rows >= 2 && fits_1row) {
    const int grid = ceil_div(N, kSignTilesR * kSign1Rows);
    for (int r0 = 0; r0 < rows;) {
      const int chunk = std::min(rows - r0, 4);
      const float* xr = x + size_t(r0) * ldx;
      float* yr = y + size_t(r0) * ldy;
      if (chunk == 1) gemm_sign_rough(xr, ldx, h, N, K, yr, ldy, 1, L, true);
      else if (chunk == 2)
        k_sign_rough_lut<2, kSignTilesR, 4><<<grid, kSign1Rows, 0, L.st>>>(xr, ldx, h.bits, h.scales, nullptr, N, K, yr,
                                                                           ldy);
      else if (chunk == 3)
        k_sign_rough_lut<3, kSignTilesR, 4><<<grid, kSign1Rows, 0, L.st>>>(xr, ldx, h.bits, h.scales, nullptr, N, K, yr,
                                                                           ldy);
      else
        k_sign_rough_lut<4, kSignTilesR, 4><<<grid, kSign1Rows, 0, L.st>>>(xr, ldx, h.bits, h.scales, nullptr, N, K, yr,
                                                                           ldy);
      r0 += chunk;
    }
    return;
  }
  const Scratch& sc = scratch(L);
  check_fits(size_t(rows) * K, sc.split_cap, "a 1-bit draft head GEMM");
  rows_to_bf16(x, ldx, rows, K, sc.split, L.st);
  // 64 columns a block: each warp reuses its activations for 64 columns (8 tiles of 8; one tile: 15 rows took 2.6x
  // longer), or for 2..8 rows 4 tiles of 16
  const dim3 grid(ceil_div(N, 64));
  if (rows <= 8)
    k_gemm_sign_cols<4, 4><<<grid, 128, 0, L.st>>>(sc.split, K, h.bits, h.scales, N, K, y, ldy, rows);
  else
    k_gemm_sign<8, 4><<<grid, 128, 0, L.st>>>(sc.split, K, h.bits, h.scales, N, K, y, ldy, rows);
}

void gemm_sign_rough_add(const float* x, int ldx, const SignHead& h, const float* bias, int N, int K, float* y,
                         int ldy, const Launch& L) {
  if (N <= 0 || K <= 0) return;
  if (K % 128 != 0) throw std::invalid_argument("gemm_sign_rough_add: K must be a multiple of 128");
  const int grid = ceil_div(N, kSignTiles1 * kSign1Rows);
  if (K <= 256)
    k_sign_rough_lut<1, kSignTiles1, 2, true><<<grid, kSign1Rows, 0, L.st>>>(x, ldx, h.bits, h.scales, bias, N, K, y,
                                                                             ldy);
  else
    k_sign_rough_lut<1, kSignTiles1, 10, true><<<grid, kSign1Rows, 0, L.st>>>(x, ldx, h.bits, h.scales, bias, N, K, y,
                                                                              ldy);
}

void draft_select_cands(float* rough, int ld, int V, int rows, const Launch& L) {
  static_assert(2 * kCandCap + 1 <= kDraftPickFloats, "a row's candidates fit where its rough scores were");
  if (rows <= 0 || V <= 0) return;
  launch_draft_select(rough, ld, V, rows, L);
}

void draft_score_bf16(const float* x, int ldx, const bf16* W, const float* bias, int K, int rows, float* rough, int ld,
                      bool accumulate, float softcap, cudaStream_t st) {
  if (rows <= 0) return;
  const dim3 grid(rows, kCandCap / 8);
  const bool fast_aligned = (K & 7) == 0 && (ldx & 3) == 0 && (reinterpret_cast<uintptr_t>(W) & 15u) == 0 &&
                            (reinterpret_cast<uintptr_t>(x) & 15u) == 0;
  if (!accumulate && !bias && softcap <= 0.f && K <= 4096 && fast_aligned) {
    k_draft_score<<<grid, 256, 0, st>>>(rough, ld, W, K, x, ldx);
  } else {
    k_draft_score_gen<<<grid, 256, 0, st>>>(rough, ld, W, bias, K, x, ldx, accumulate, softcap);
  }
}

void draft_combine_cands(const float* add, int ld_add, const float* bias, float softcap, int rows, float* rough, int ld,
                         cudaStream_t st) {
  if (rows <= 0) return;
  k_draft_combine_cands<<<rows, kCandCap, 0, st>>>(add, ld_add, bias, softcap, rough, ld);
}

void draft_pick_topk(float* rough, int ld, int rows, int k, int* idx, float* val, cudaStream_t st, float* prob,
                     float* margin, const DevParams* P, int pos_add, const DevGen* G) {
  if (rows <= 0 || k <= 0) return;
  k_draft_topk<<<rows, 32, 0, st>>>(rough, ld, k, idx, val, prob, margin, P, pos_add, G);
}

void draft_pick(const float* x, int ldx, const SignHead& h, const bf16* W, int V, int K, int rows, int k, int* idx,
                float* val, float* rough, const Launch& L, bool exact_1row, float* prob, float* margin) {
  const int ld = std::max(V, kDraftPickFloats);
  gemm_sign_rough(x, ldx, h, V, K, rough, ld, rows, L, exact_1row);
  draft_select_cands(rough, ld, V, rows, L);
  draft_score_bf16(x, ldx, W, nullptr, K, rows, rough, ld, false, 0.f, L.st);
  draft_pick_topk(rough, ld, rows, k, idx, val, L.st, prob, margin);
}

void draft_pick_blocks(const float* x, int ldx, const SignHead& h, const BlockMat& W, int V, int K, int rows, int k,
                       int* idx, float* val, float* rough, const Launch& L, bool exact_1row, float* prob,
                       float* margin) {
  const int ld = std::max(V, kDraftPickFloats);
  gemm_sign_rough(x, ldx, h, V, K, rough, ld, rows, L, exact_1row);
  draft_select_cands(rough, ld, V, rows, L);
  // Every row's candidates from the packed rows at once (a row's candidates decoded to float32 first, a row at a
  // time, were 7 launch pairs and 141 us a DFlash2 round on the 27B; the same values, in the same order).
  int* cand = reinterpret_cast<int*>(rough);
  dots_blocks(W, K, cand, cand + 2 * kCandCap, rough + kCandCap, size_t(ld), kCandCap, x, ldx, rows, L.st);
  draft_pick_topk(rough, ld, rows, k, idx, val, L.st, prob, margin);
}

}  // namespace flyx::cuda
