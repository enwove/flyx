// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Decode and verification attention on tensor cores, for binary16 and FP8 caches (KernelOptions::rows_mma): the
// partials of attention_split's grouped kernel, for k_attention_combine.
//
// A block takes one KV head, one key split and up to kMaxWarps rows of a round, each row its own warps (the KV head's
// query heads in its columns: head j in warp j / 8 of the row, column j % 8), so a row's query vectors sit in the
// same places of the same instructions whatever rows share the block.  The keys go through shared memory in tiles of
// 16 (cp.async, two buffers), each read once for all the block's rows; there the float32 split kernel had each row read
// them again, and kept a 256-dim key in a lane's registers.
//
// Scores S^T = K Q^T on mma.m16n8k16, keys as m and query vectors as n: a warp's output O^T (dims x its 8 queries)
// is 64 floats a lane at head dim 256, not 128 as with queries as m, so its query fragments stay in registers.  The
// probabilities come out keys x queries; movmatrix turns them into the B operand of O^T += V^T P^T.
//
// Precision: q and p go into the tensor cores as binary16 parts (the nearest, then the nearest to what is left, ...):
// three (33 significant bits, more than float32's 24), or one (11, llama.cpp's own) on the shapes
// where llama.cpp rounds them to one binary16 itself in every row count and context (more than 4 query heads a KV head
// at head dim 256 or more: its fattn-mma-f16, never its float32 vector kernel); the caches are binary16, so every
// product is exact, and the sums are float32.  A query vector is first scaled by a power of two (exact) so its largest value is just below
// 2^14 and its low parts stay normal; p in (0, 1] by 2^14 likewise.  llama.cpp rounds q and p to binary16 in its MMA
// kernel and keeps them float32 in its vector one (numerics.md).
//
// FP8 caches: a tile's e4m3 bytes and its keys' scales go through shared memory, the bytes become
// binary16 there (exactly: every e4m3 value is one), a score is the products' sum times its key's scale, and a key's
// value scale goes into p (p times the scale, in parts, times 2^(14 - E), E the exponent of the largest value scale of
// the row's keys so far; the output is rescaled exactly when E grows), so every product stays exact as with binary16
// caches.
//
// Bits: a row's keys are split by its own position (attn_split_range), tiles start at multiples of 16 keys, and a tile
// with none of a row's keys leaves its query vectors' state exactly as it was (the maximum stays, the factor is 1, the
// products are zeros), so a row gets the same bits whatever rows share its block: verifying k rows gives each the bits
// decoding it alone gives.
#include <algorithm>
#include <climits>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kTileKeys = 16;        // keys a tile: the m of the scores' MMA
constexpr int kMaxWarps = 8;         // warps a block
constexpr int kFp8Warps = 4;         // warps a block at least, with FP8 caches
constexpr float kPScale = 16384.f;   // p into the tensor cores times 2^14, the output times 2^-14
constexpr int kQueryTop = 14;        // a query vector scaled so its largest |value| is below 2^14
constexpr int kQueryTop8 = 8;        // ... as e4m3: below 2^8 (e4m3's largest is 448)

// x as NP binary16 values: each the nearest to what the ones before left (the remainders are exact in float32).
template <int NP>
__device__ __forceinline__ void split_parts(float x, __half (&h)[NP]) {
#pragma unroll
  for (int i = 0; i < NP; ++i) {
    h[i] = __float2half_rn(x);
    x = __fsub_rn(x, __half2float(h[i]));
  }
}

// The shared memory a block takes: binary16 key and value tiles, two of each; with FP8 caches one of each (Q8: a
// value tile only, the scores read the keys' bytes), and the bytes and scales of two tiles staged beside them.
template <int HD, bool Q8>
__host__ __device__ constexpr int staged_stride() {  // Q8: a row's bytes padded by 16 (ldmatrix reads 8 rows at one offset)
  return Q8 ? HD + 16 : HD;
}
template <int HD, bool FP8, bool Q8>
constexpr size_t rows_mma_smem() {
  constexpr size_t tile = size_t(kTileKeys) * (HD + 8) * sizeof(kv16);
  constexpr size_t staged = size_t(2) * 2 * kTileKeys * (staged_stride<HD, Q8>() + sizeof(float));
  return FP8 ? (Q8 ? 1 : 2) * tile + staged : 4 * tile;
}

// grid (ceil(rows / rows a block), nkv, splits): the blocks of a round's rows for one split next to each other.
// Q8 (FP8 caches only): a query vector as e4m3 (times a power of two: its largest |value| below 2^8), the scores
// on mma.m16n8k32 with the keys' bytes as they are. q_e4m3 (not Q8): the same e4m3 query vector, held
// exactly in binary16 and summed on the binary16 tensor cores, for a GPU whose e4m3 sums are not float32's (sm_89).
template <int HD, int NP, bool FP8, bool Q8>  // NP: binary16 parts of q and p (Q8: of p)
__global__ void __launch_bounds__(32 * kMaxWarps) k_attention_rows_mma(
    const float* q, KvRef kv, int nh, int nkv, int rows, int rows_per_block, int causal, int extra, float scale,
    const DevParams* P, float* __restrict__ part, int pos_add, int from, const TreeRows* __restrict__ tree,
    int q_e4m3) {
  static_assert(FP8 || !Q8, "q in e4m3 needs an FP8 cache");
  pdl_trigger();
  constexpr int KS = HD / 16, LD = HD + 8;  // 16-dim steps of the scores, and 16-dim tiles of the output
  constexpr int K8 = HD / 32;               // Q8: 32-dim steps of the scores
  constexpr int NBK = FP8 ? (Q8 ? 0 : 1) : 2, NBV = FP8 ? 1 : 2;  // binary16 key and value tile buffers
  constexpr int RS = staged_stride<HD, Q8>();
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto Ks = reinterpret_cast<kv16 (*)[kTileKeys][LD]>(smem_raw);
  auto Vs = reinterpret_cast<kv16 (*)[kTileKeys][LD]>(smem_raw + size_t(NBK) * kTileKeys * LD * sizeof(kv16));
  // FP8: [buffer][K, V][key][dim] bytes, then [buffer][K, V][key] scales
  unsigned char* staged = smem_raw + size_t(NBK + NBV) * kTileKeys * LD * sizeof(kv16);
  auto Rb = reinterpret_cast<uint8_t (*)[2][kTileKeys][RS]>(staged);
  auto Sc = reinterpret_cast<float (*)[2][kTileKeys]>(staged + size_t(2) * 2 * kTileKeys * RS);
  const kv16* Kc = static_cast<const kv16*>(kv.k);
  const kv16* Vc = static_cast<const kv16*>(kv.v);
  const int G = nh / nkv, WR = (G + 7) / 8, kvh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
  const int r0 = blockIdx.x * rows_per_block, nr = min(rows_per_block, rows - r0);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int wrow = warp / WR, whead = (warp % WR) * 8;  // the warp's row, and the head of its column 0
  const int base = P->position + pos_add;  // a round's positions are set before it runs (no kernel of it writes them)
  // The keys of each of the block's rows, read once (a tree's depths come from its table: not a load at every use).
  int totals[kMaxWarps];
  {
    const bool on = tree_on(tree);
#pragma unroll
    for (int i = 0; i < kMaxWarps; ++i)
      totals[i] = causal ? base + (on && i < nr ? int(tree->depth[r0 + i]) : r0 + i) + 1 : base + extra;
  }
  auto total_of = [&](int i) { return totals[i]; };
  auto mine = [&](int i) { return i < nr && total_of(i) >= from; };  // rows of fewer keys: the float32 kernels'
  auto range = [&](int i, int* ks, int* ke) { attn_split_range(total_of(i), split, splits, ks, ke); };
  auto out_of = [&](int i, int h) { return ((size_t(r0 + i) * nh + kvh * G + h) * splits + split) * (HD + 2); };
  int lo = INT_MAX, hi = 0;  // the keys of this split any of the block's rows has
  bool any = false;
  for (int i = 0; i < nr; ++i) {
    int a, b;
    range(i, &a, &b);
    any = any || mine(i);
    if (mine(i) && a < b) lo = min(lo, a), hi = max(hi, b);
  }
  if (!any) return;  // the float32 kernels' rows only: gone while the kernel before still runs
  pdl_wait();        // q and the round's keys come from the kernels before
  if (lo >= hi) {  // empty partials
    for (int idx = threadIdx.x; idx < nr * G * (HD + 2); idx += blockDim.x) {
      const int c = idx / (HD + 2), e = idx - c * (HD + 2);
      if (mine(c / G)) part[out_of(c / G, c % G) + e] = e == 0 ? -INFINITY : 0.f;
    }
    return;
  }
  // The lane's two columns of the scores and the output: heads whead + 2 t + j of row wrow.
  int cks[2] = {0, 0}, cke[2] = {0, 0};
  bool cok[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    cok[j] = mine(wrow) && whead + 2 * t + j < G;
    if (cok[j]) range(wrow, &cks[j], &cke[j]);
  }
  // The B fragments of head whead + g of row wrow, scaled by 2^sh: dims 16 ks + 2 t (+1) and + 8 (+9) in binary16
  // parts; Q8: bq[0][kb] dims 32 kb + 4 t (+1, +2, +3) and + 16 (...) in e4m3.
  uint32_t bq[Q8 ? 1 : NP][Q8 ? K8 : KS][2];
  int sh;
  {
    const bool qok = mine(wrow) && whead + g < G;
    float qx[KS][4];
    float mx = 0.f;
    const float* qp = q + (size_t(r0 + (qok ? wrow : 0)) * nh + kvh * G + (qok ? whead + g : 0)) * HD;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const int x = 4 * ks + h, w = x % 8;  // Q8: value w of step x / 8
        const int d = Q8 ? 32 * (x / 8) + 4 * t + (w & 3) + 16 * (w >> 2) : 16 * ks + 2 * t + (h & 1) + 8 * (h >> 1);
        qx[ks][h] = qok ? qp[d] : 0.f;
        mx = fmaxf(mx, fabsf(qx[ks][h]));
      }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
    int ex = 0;
    frexpf(mx, &ex);  // mx < 2^ex (0: ex 0)
    sh = (Q8 ? kQueryTop8 : kQueryTop) - ex;
    if constexpr (Q8) {
#pragma unroll
      for (int kb = 0; kb < K8; ++kb)
#pragma unroll
        for (int f = 0; f < 2; ++f) {  // values 8 kb + 4 f .. + 3 of the lane: qx[2 kb + f][0..3]
          const float* v = qx[2 * kb + f];
          const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(ldexpf(v[0], sh), ldexpf(v[1], sh)), __NV_SATFINITE,
                                                       __NV_E4M3);
          const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(ldexpf(v[2], sh), ldexpf(v[3], sh)), __NV_SATFINITE,
                                                       __NV_E4M3);
          bq[0][kb][f] = (lo & 0xFFFFu) | (hi << 16);
        }
    } else {
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        __half p[4][NP];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          float x = ldexpf(qx[ks][h], sh);
          if (q_e4m3)  // Q8's rounding (its largest |value| below 2^kQueryTop8), then back up: one exact part
            x = ldexpf(e4m3_to_float(__nv_cvt_float_to_fp8(ldexpf(x, kQueryTop8 - kQueryTop), __NV_SATFINITE,
                                                            __NV_E4M3)),
                       kQueryTop - kQueryTop8);
          split_parts(x, p[h]);
        }
#pragma unroll
        for (int i = 0; i < NP; ++i) {
          bq[i][ks][0] = pack_halves(p[0][i], p[1][i]);
          bq[i][ks][1] = pack_halves(p[2][i], p[3][i]);
        }
      }
    }
  }
  float cs[2];  // column 2 t + j's score factor: the softmax scale times 2^-sh of its query vector (exact)
#pragma unroll
  for (int j = 0; j < 2; ++j) cs[j] = ldexpf(scale, -__shfl_sync(0xffffffffu, sh, 4 * (2 * t + j)));

  float o[KS][4];  // O^T times 2^14: tile mt holds dims 16 mt + g (+8), columns 2 t (+1)
#pragma unroll
  for (int mt = 0; mt < KS; ++mt) o[mt][0] = o[mt][1] = o[mt][2] = o[mt][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};  // l: this lane's keys' share
  int ve = INT_MIN;      // FP8: E of the output's scale 2^(14 - E) (INT_MIN: no value scale of the row's keys yet)
  int rks = 0, rke = 0;  // the warp's row's keys of this split
  if (mine(wrow)) range(wrow, &rks, &rke);
  // row >= 0: a tree's row r0 + row, its round keys from its ancestors' slots; -1: keys as they are (a chain's rows,
  // or the keys before the round, which every row of a tree shares).
  auto load = [&](int tile, int buf, int row) {
    // past hi: zeros (masked; zeros keep 0 x V a zero)
    auto at_of = [&](int key) { return key >= hi ? lo : row >= 0 ? tree_slot(tree, r0 + row, base, key) : key; };
    if constexpr (FP8) {
      for (int i = threadIdx.x; i < 2 * kTileKeys * (HD / 16); i += blockDim.x) {
        const int v = i / (kTileKeys * (HD / 16)), r = (i / (HD / 16)) % kTileKeys, c = i % (HD / 16);
        const int key = tile * kTileKeys + r;
        const uint8_t* src = static_cast<const uint8_t*>(v ? kv.v : kv.k) + (size_t(at_of(key)) * nkv + kvh) * HD;
        cp_async16(&Rb[buf][v][r][16 * c], src + 16 * c, key < hi);
      }
      if (threadIdx.x < 2 * kTileKeys) {
        const int v = threadIdx.x / kTileKeys, r = threadIdx.x % kTileKeys, key = tile * kTileKeys + r;
        const int at = at_of(key);
        cp_async4(&Sc[buf][v][r], (v ? kv.vscales(at) : kv.kscales(at)) + size_t(at) * nkv + kvh, key < hi);
      }
    } else {
      for (int i = threadIdx.x; i < 2 * kTileKeys * (HD / 8); i += blockDim.x) {
        const int v = i / (kTileKeys * (HD / 8)), r = (i / (HD / 8)) % kTileKeys, c = i % (HD / 8);
        const int key = tile * kTileKeys + r;
        const kv16* src = (v ? Vc : Kc) + (size_t(at_of(key)) * nkv + kvh) * HD + 8 * c;
        cp_async16(v ? &Vs[buf][r][8 * c] : &Ks[buf][r][8 * c], src, key < hi);
      }
    }
    cp_async_commit();
  };
  // FP8: a staged tile's bytes into the binary16 tiles (exact), by every thread of the block (Q8: the values only).
  auto convert = [&](int buf) {
    if constexpr (FP8) {
      constexpr int first = Q8 ? kTileKeys * (HD / 8) : 0;  // Q8: from the first value
      for (int i = first + threadIdx.x; i < 2 * kTileKeys * (HD / 8); i += blockDim.x) {
        const int v = i / (kTileKeys * (HD / 8)), r = (i / (HD / 8)) % kTileKeys, c = i % (HD / 8);
        const uint2 w = *reinterpret_cast<const uint2*>(&Rb[buf][v][r][8 * c]);
        uint32_t hw[4];
        e4m3x4_to_f16bits(w.x, hw);
        e4m3x4_to_f16bits(w.y, hw + 2);
        *reinterpret_cast<uint4*>(v ? &Vs[0][r][8 * c] : &Ks[0][r][8 * c]) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
      }
    }
  };
  // One tile of keys into the warp's row's state.
  auto step = [&](int tile, int buf, int sbuf) {
    // S^T: keys g (+8) x columns 2 t (+1), a part a chain (Q8: one, the keys' bytes as the A operand)
    constexpr int SP = Q8 ? 1 : NP;
    float sp[SP][4];
#pragma unroll
    for (int i = 0; i < SP; ++i) sp[i][0] = sp[i][1] = sp[i][2] = sp[i][3] = 0.f;
    if constexpr (Q8) {
#pragma unroll
      for (int kb = 0; kb < K8; ++kb) {
        uint32_t a[4];
        ldmatrix_x4(a, &Rb[sbuf][0][lane & 15][32 * kb + (lane >> 4) * 16]);
        mma_e4m3(sp[0], a, bq[0][kb]);
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, &Ks[buf][lane & 15][16 * ks + (lane >> 4) * 8]);
#pragma unroll
        for (int i = 0; i < NP; ++i) mma_f16(sp[i], a, bq[i][ks]);
      }
    }
    float v[4], tm[2];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int j = e & 1, key = tile * kTileKeys + g + 8 * (e >> 1);
      float s = sp[SP - 1][e];  // the parts' sums from the smallest: sp[0] + (sp[1] + sp[2])
#pragma unroll
      for (int i = SP - 2; i >= 0; --i) s = __fadd_rn(sp[i][e], s);
      if constexpr (FP8) s = __fmul_rn(s, Sc[sbuf][0][key - tile * kTileKeys]);  // the key's scale
      v[e] = cok[j] && key >= cks[j] && key < cke[j] ? __fmul_rn(s, cs[j]) : -INFINITY;
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      tm[j] = fmaxf(v[j], v[j + 2]);
#pragma unroll
      for (int x = 4; x < 32; x <<= 1) tm[j] = fmaxf(tm[j], __shfl_xor_sync(0xffffffffu, tm[j], x));
    }
    float corr[2], p[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float mn = fmaxf(m[j], tm[j]);
      corr[j] = mn == m[j] ? 1.f : __expf(m[j] - mn);  // exactly 1 when the tile changed nothing
      m[j] = mn;
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) p[e] = v[e] == -INFINITY ? 0.f : __expf(v[e] - m[e & 1]);
#pragma unroll
    for (int j = 0; j < 2; ++j) l[j] = __fadd_rn(__fmaf_rn(l[j], corr[j], p[j]), p[j + 2]);
    // FP8: the value scales of the row's keys in this tile; the output's scale follows the largest so far.
    float pscale = kPScale, ocorr = 1.f;
    if constexpr (FP8) {
      const int key = tile * kTileKeys + (lane & 15);
      float big = lane < 16 && key >= rks && key < rke ? Sc[sbuf][1][lane] : 0.f;
#pragma unroll
      for (int x = 16; x; x >>= 1) big = fmaxf(big, __shfl_xor_sync(0xffffffffu, big, x));  // the whole warp
      int eb = INT_MIN;
      if (big > 0.f) frexpf(big, &eb);  // every value scale of the row's keys in the tile below 2^eb
      if (eb > ve) {
        if (ve != INT_MIN) ocorr = ldexpf(1.f, ve - eb);  // exact: a power of two
        ve = eb;
      }
      if (ve != INT_MIN) pscale = ldexpf(1.f, kQueryTop - ve);
    }
#pragma unroll
    for (int mt = 0; mt < KS; ++mt)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[mt][e] = __fmul_rn(o[mt][e], FP8 ? __fmul_rn(corr[e & 1], ocorr) : corr[e & 1]);
    // P^T as the B operand: keys 2 t (+1) and + 8 (+9) of column g, in parts
    uint32_t bp[NP][2];
    {
      __half h[4][NP];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float pe = p[e];
        if constexpr (FP8) pe = __fmul_rn(pe, Sc[sbuf][1][g + 8 * (e >> 1)]);  // its key's value scale (before the transpose)
        split_parts(pe * pscale, h[e]);
      }
#pragma unroll
      for (int i = 0; i < NP; ++i) {
        bp[i][0] = movmatrix_trans(pack_halves(h[0][i], h[1][i]));
        bp[i][1] = movmatrix_trans(pack_halves(h[2][i], h[3][i]));
      }
    }
    // O^T += V^T P^T: A = V^T (dims x keys) from the value tile, transposed on the way
#pragma unroll
    for (int mt = 0; mt < KS; ++mt) {
      uint32_t a[4];
      ldmatrix_x4_t(a, &Vs[buf][(lane & 7) + ((lane >> 4) << 3)][16 * mt + ((lane >> 3) & 1) * 8]);
#pragma unroll
      for (int i = 0; i < NP; ++i) mma_f16(o[mt], a, bp[i]);
    }
  };
  const int t0 = lo / kTileKeys, t1 = (hi - 1) / kTileKeys;
  // A tree's tiles from the one holding the round's first key: each row's own keys there, so its rows take them one
  // at a time through buffer 0 (each row's warps step through the same tiles in the same order as decoding it alone);
  // the tiles before are every row's, read once for the block.
  const int ts = tree_on(tree) ? max(t0, min(t1 + 1, base / kTileKeys)) : t1 + 1;
  if (ts > t0) {
    load(t0, 0, -1);
    for (int tile = t0; tile < ts; ++tile) {
      const int buf = (tile - t0) & 1;
      if (tile + 1 < ts) {
        load(tile + 1, buf ^ 1, -1);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncthreads();
      if constexpr (FP8) {
        convert(buf);
        __syncthreads();
      }
      if (wrow < nr) step(tile, FP8 ? 0 : buf, buf);  // warps past the rows only load
      __syncthreads();  // buf is reloaded two tiles on (FP8: the binary16 tiles next tile)
    }
  }
  for (int tile = ts; tile <= t1; ++tile)
    for (int i = 0; i < nr; ++i) {
      load(tile, 0, i);
      cp_async_wait<0>();
      __syncthreads();
      if constexpr (FP8) {
        convert(0);
        __syncthreads();
      }
      if (wrow == i) step(tile, 0, 0);
      __syncthreads();
    }
  // the columns' p sums over the 8 lanes that hold their keys (a fixed tree), then the partials
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int x = 4; x < 32; x <<= 1) l[j] = __fadd_rn(l[j], __shfl_xor_sync(0xffffffffu, l[j], x));
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (!cok[j]) continue;
    float* pr = part + out_of(wrow, whead + 2 * t + j);
    if (g == 0) {
      pr[0] = m[j];
      pr[1] = l[j];
    }
    // the output's scale back: 2^-14, or 2^(E - 14) with FP8 caches (exact)
    const float back = FP8 && ve != INT_MIN ? ldexpf(1.f, ve - kQueryTop) : 1.f / kPScale;
#pragma unroll
    for (int mt = 0; mt < KS; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) pr[2 + 16 * mt + g + 8 * h] = __fmul_rn(o[mt][2 * h + j], back);
  }
}

template <int HD, bool FP8>
void launch_rows_mma(const float* q, KvRef kv, int nh, int nkv, int rows, bool causal, int extra, float scale,
                     const DevParams* P, float* partial, int splits, const Launch& L, int pos_add, int from,
                     const TreeRows* tree) {
  const int G = nh / nkv, WR = (G + 7) / 8, rows_per_block = kMaxWarps / WR;
  // FP8: at least kFp8Warps a block, the ones past its rows' loading and converting with the others (a row alone in a
  // warp of its own left the SMs idle: one row at 63K tokens on the 5070 Ti ran 4% slower than the float32 kernel).
  const int blocks = ceil_div(rows, rows_per_block);
  const int warps = std::max(FP8 ? kFp8Warps : 1, (rows < rows_per_block ? rows : rows_per_block) * WR);
  // One part where llama.cpp itself rounds q and p to one binary16 (two before), else
  // three; rows_attention_parts sets it.
  const int asked = opts(L).rows_mma_parts;
  const int parts = asked ? asked : G > 4 && HD >= 256 ? 1 : 3;
  // q as e4m3 over FP8 caches where asked: on FP8 tensor cores where their sums are float32's
  // (fp8_mma), else the same e4m3 query vector on the binary16 ones.
  const bool asked_q8 = FP8 && opts(L).rows_mma_q8, q8 = asked_q8 && opts(L).fp8_mma;
  auto pick = [&](auto q8_tag) {
    constexpr bool Q8 = decltype(q8_tag)::value;
    auto kernel = parts == 1   ? k_attention_rows_mma<HD, 1, FP8, Q8>
                  : parts == 2 ? k_attention_rows_mma<HD, 2, FP8, Q8>
                               : k_attention_rows_mma<HD, 3, FP8, Q8>;
    launch_dependent_smem(kernel, dim3(blocks, nkv, splits), dim3(32 * warps), rows_mma_smem<HD, FP8, Q8>(), L, q, kv,
                          nh, nkv, rows, rows_per_block, int(causal), extra, scale, P, partial, pos_add, from, tree,
                          int(asked_q8 && !Q8));
  };
  if constexpr (FP8) {
    if (q8) return pick(std::true_type{});
  }
  pick(std::false_type{});
}

template <int HD>
void launch_rows_mma(const float* q, KvRef kv, int nh, int nkv, int rows, bool causal, int extra, float scale,
                     const DevParams* P, float* partial, int splits, const Launch& L, int pos_add, int from,
                     const TreeRows* tree) {
  if (kv.fp8())
    launch_rows_mma<HD, true>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
  else
    launch_rows_mma<HD, false>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
}

}  // namespace

bool attention_rows_mma(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, bool causal, int extra,
                        float scale, const DevParams* P, float* partial, int splits, const Launch& L, int pos_add,
                        int from, const TreeRows* tree) {
  if (nkv <= 0 || nh % nkv != 0 || nh / nkv > 8 * kMaxWarps) return false;
  switch (hd) {
  case 32: launch_rows_mma<32>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
    break;
  case 64: launch_rows_mma<64>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
    break;
  case 128: launch_rows_mma<128>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
    break;
  case 256: launch_rows_mma<256>(q, kv, nh, nkv, rows, causal, extra, scale, P, partial, splits, L, pos_add, from, tree);
    break;
  default: return false;
  }
  return true;
}

}  // namespace flyx::cuda
