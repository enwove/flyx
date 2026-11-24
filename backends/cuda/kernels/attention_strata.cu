// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Strata attention for decode and verification rounds, after StrataFlux's selective refinement of a
// low-bit KV cache (Hyun et al., 2026).  Every position's K and V also have a low-bit copy on the device: rotated by
// the normalized Walsh-Hadamard transform (q.k and the softmax's weights do not change under it) and quantized
// uniformly per position and head (attn_prep writes it).  A round's attention reads the copy for every position
// (strata_scan), then picks each row's most attended pairs of positions by the copy's softmax, the largest over the
// query heads of a KV head (strata_select), and replaces their contributions with the K/V cache's own values, FP8 or
// binary16 wherever the cache is (strata_correct: StrataFlux's equation 8).  A row's result depends on its own query
// and position only, so it is the same bits alone or among a round's rows (rule 4).
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// x (n vectors of HD floats in shared memory) <- H x / sqrt(HD), H Sylvester's Hadamard matrix.  Every butterfly
// adds and subtracts the same operands whatever the block's size, so the scan and the correction rotate a query to
// the same bits (and attn_prep's rotation, one vector by HD threads, does the same arithmetic).
template <int HD>
__device__ void fwht(float* x, int n) {
  for (int len = 1; len < HD; len <<= 1) {
    __syncthreads();
    for (int i = threadIdx.x; i < n * HD / 2; i += blockDim.x) {
      const int v = i / (HD / 2), k = i % (HD / 2);
      const int lo = (k / len) * 2 * len + (k % len);
      float* p = x + v * HD;
      const float a = p[lo], b = p[lo + len];
      p[lo] = a + b;
      p[lo + len] = a - b;
    }
  }
  __syncthreads();
  const float s = rsqrtf(float(HD));
  for (int i = threadIdx.x; i < n * HD; i += blockDim.x) x[i] *= s;
  __syncthreads();
}

// The copy's codes (KvRef::kq) as StrataLayout lays them out. 1 + c / 2^BITS for the code at bit
// `at` of a word, exactly, without a conversion (int to float runs at an eighth of the rate of a multiply-add on
// sm_120): the code moved to the top of a float's mantissa.
template <int BITS>
__device__ __forceinline__ float code_mant(uint32_t w, int at) {
  constexpr int TOP = 23 - BITS;
  const uint32_t x = (at <= TOP ? (w << (TOP - at)) : (w >> (at - TOP))) & (((1u << BITS) - 1u) << TOP);
  return __uint_as_float(0x3f800000u | x);
}

template <int BITS>
__device__ __forceinline__ float code_at(uint32_t w, int at) {
  return float((w >> at) & ((1u << BITS) - 1u));
}

// The scan: grid (rows, nkv, splits), block 128 (4 warps; warp w the 32-key tiles t with t % 4 == w, as the grouped
// kernel).  Partials as k_attention_gqa writes them; and for each pair of positions (2i, 2i + 1) and query head, the
// log of the pair's exp-scores, binary16, [rows][nh][lse_stride].
// The 24-word hosted 256d three-bit layout stores the shared codes in bits 15 and 31 of the private words.
// Each residual scalar has one bit in three private words. The refinement and float32 scans read it by dimension.
template <bool Key>
__device__ __forceinline__ uint32_t compact3_code(const uint32_t* tile, int pos, int d) {
  constexpr StrataLayout L(256, 3);
  if (d < 240) {
    int index, shift;
    if constexpr (Key) L.k_code(pos, d, &index, &shift);
    else L.v_code(pos, d, &index, &shift);
    return (tile[index] >> shift) & 7u;
  }
  const int group = Key ? ((d >> 1) & 3) : (d & 7);
  const int first = Key ? 6 * group + 3 * ((d >> 3) - 30) : 3 * group;
  const int shift = 15 + 16 * (Key ? d & 1 : (d >> 3) & 1);
  uint32_t code = 0;
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const int at = Key ? L.k_index(pos, first + i) : L.v_index(pos, first + i);
    code |= ((tile[at] >> shift) & 1u) << i;
  }
  return code;
}

template <int HD, int G, int BITS, bool Compact = false>
__global__ void __launch_bounds__(128) k_strata_scan(const float* __restrict__ q, KvRef kv, int nh, int nkv,
                                                     float scale, const DevParams* P, float* __restrict__ part,
                                                     int slots, __half* __restrict__ lse, int lse_stride) {
  pdl_trigger();
  constexpr int DPL = HD / 32, VB = 4;
  constexpr StrataLayout SL(HD, BITS);
  constexpr int W = SL.kwords();
  constexpr uint32_t FULL = 0xffffffffu;
  __shared__ __align__(16) float Qs[G][HD];
  __shared__ float s_qsum[G];
  __shared__ float s_m[4][G], s_l[4][G];
  __shared__ float s_acc[4][G][HD];
  const int r = blockIdx.x, kvh = blockIdx.y, split = blockIdx.z, splits = gridDim.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int total = P->position + r + 1;
  int ks, ke;
  attn_split_range(total, split, splits, &ks, &ke);
  for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
    const int g = i / HD, d = i - g * HD;
    Qs[g][d] = q[(size_t(r) * nh + kvh * G + g) * HD + d];
  }
  fwht<HD>(&Qs[0][0], G);
  if (threadIdx.x < G) {
    float s = 0.f;
    for (int d = 0; d < HD; ++d) s += Qs[threadIdx.x][d];
    s_qsum[threadIdx.x] = s;
  }
  __syncthreads();
  float acc[G][DPL], m[G], lp[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m[g] = -INFINITY;
    lp[g] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) acc[g][e] = 0.f;
  }
  const size_t words = size_t(kv.words);
  if (ks < ke) {
    const int t0 = ks >> 5, t1 = (ke - 1) >> 5;
    for (int t = t0 + ((warp - t0) & 3); t <= t1; t += 4) {
      const int j = (t << 5) + lane;
      const int kn = min(32, ke - (t << 5));
      float sc[G];
      if (j < ke) {
        const uint32_t* kw = kv.kq + strata_words_at(j, kvh, nkv, int(words));
        const float2 sl = __half22float2(kv.kqs[size_t(j) * nkv + kvh]);
        uint32_t wv[W];
#pragma unroll
        for (int c = 0; c < W; ++c) wv[c] = !Compact || c < 24 ? __ldg(kw + SL.k_index(j & 15, c)) : 0;
        // dim d's word among the position's and its code's bit
        auto kcode = [&](int d) {
          if constexpr (Compact) if (d >= 240) return code_mant<3>(compact3_code<true>(kw, j & 15, d), 0);
          int w = 0, s = 0;
          SL.k_at((d >> 1) & 3, d >> 3, &w, &s);
          return code_mant<BITS>(wv[w], (d & 1) * 16 + s * BITS);
        };
        float dot[G];
#pragma unroll
        for (int g = 0; g < G; ++g) dot[g] = 0.f;
#pragma unroll
        for (int d = 0; d < HD; d += 4) {  // four codes against a float4 of each query (one shared load, not four)
          const float c0 = kcode(d), c1 = kcode(d + 1), c2 = kcode(d + 2), c3 = kcode(d + 3);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            const float4 a = *reinterpret_cast<const float4*>(&Qs[g][d]);
            float s = dot[g];
            s = __fmaf_rn(a.x, c0, s);
            s = __fmaf_rn(a.y, c1, s);
            s = __fmaf_rn(a.z, c2, s);
            dot[g] = __fmaf_rn(a.w, c3, s);
          }
        }
        // dot = q.(1 + c / 2^BITS): the key lo + (c + 1/2) step is a (1 + c / 2^BITS) + b, a = 2^BITS step.
        const float a = __fmul_rn(sl.y, float(1 << BITS)), b = __fsub_rn(__fmaf_rn(0.5f, sl.y, sl.x), a);
#pragma unroll
        for (int g = 0; g < G; ++g) sc[g] = __fmul_rn(__fmaf_rn(a, dot[g], __fmul_rn(b, s_qsum[g])), scale);
      } else {
#pragma unroll
        for (int g = 0; g < G; ++g) sc[g] = -INFINITY;
      }
      // The pair's log-sum-exp per query head, from its even lane.
      if (lse) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const float o = __shfl_xor_sync(FULL, sc[g], 1);
          if (!(lane & 1) && j < ke) {
            const float mx = fmaxf(sc[g], o);
            const float v = mx == -INFINITY ? -INFINITY : mx + __logf(__expf(sc[g] - mx) + __expf(o - mx));
            lse[(size_t(r) * nh + kvh * G + g) * lse_stride + (j >> 1)] = __float2half_rn(v);
          }
        }
      }
      float pj[G];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float mt = sc[g];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mt = fmaxf(mt, __shfl_xor_sync(FULL, mt, o));
        const float mn = fmaxf(m[g], mt);
        const float corr = mn == m[g] ? 1.f : __expf(m[g] - mn);
        pj[g] = j < ke ? __expf(sc[g] - mn) : 0.f;
        lp[g] = __fmaf_rn(lp[g], corr, pj[g]);
#pragma unroll
        for (int e = 0; e < DPL; ++e) acc[g][e] = __fmul_rn(acc[g][e], corr);
        m[g] = mn;
      }
      for (int k0 = 0; k0 < kn; k0 += VB) {
        uint32_t vc[VB][DPL];  // the words of the lane's dims, and their codes' bits
        int vat[DPL];
        float2 vs[VB];
#pragma unroll
        for (int b = 0; b < VB; ++b) {
          const int key = (t << 5) + min(k0 + b, kn - 1);
          const uint32_t* vw = kv.vq + strata_words_at(key, kvh, nkv, int(words));
#pragma unroll
          for (int e = 0; e < DPL; ++e) {
            int index = 0;
            SL.v_code(key & 15, lane * DPL + e, &index, &vat[e]);
            if constexpr (Compact) {
              vc[b][e] = compact3_code<false>(vw, key & 15, lane * DPL + e);
              vat[e] = 0;
            } else vc[b][e] = __ldg(vw + index);
          }
          vs[b] = __half22float2(kv.vqs[size_t(key) * nkv + kvh]);
        }
#pragma unroll
        for (int b = 0; b < VB; ++b) {
          if (k0 + b >= kn) break;  // uniform across the warp
          float vf[DPL];
          const float va_ = __fmul_rn(vs[b].y, float(1 << BITS)), vb_ = __fsub_rn(__fmaf_rn(0.5f, vs[b].y, vs[b].x), va_);
#pragma unroll
          for (int e = 0; e < DPL; ++e) vf[e] = __fmaf_rn(va_, code_mant<BITS>(vc[b][e], vat[e]), vb_);
#pragma unroll
          for (int g = 0; g < G; ++g) {
            const float p = __shfl_sync(FULL, pj[g], k0 + b);
#pragma unroll
            for (int e = 0; e < DPL; ++e) acc[g][e] = __fmaf_rn(p, vf[e], acc[g][e]);
          }
        }
      }
    }
  }
  // The warps' partial sums in order, as the grouped kernel.
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float l = lp[g];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
    if (lane == 0) {
      s_m[warp][g] = m[g];
      s_l[warp][g] = l;
    }
#pragma unroll
    for (int e = 0; e < DPL; ++e) s_acc[warp][g][lane * DPL + e] = acc[g][e];
  }
  __syncthreads();
  __shared__ float s_f[4][G], s_M[G], s_L[G];
  if (threadIdx.x < G) {
    const int g = threadIdx.x;
    float M = -INFINITY;
    for (int w = 0; w < 4; ++w) M = fmaxf(M, s_m[w][g]);
    float L = 0.f;
    for (int w = 0; w < 4; ++w) {
      float f = 0.f;
      if (M != -INFINITY) {
        f = __expf(s_m[w][g] - M);
        L += s_l[w][g] * f;
      }
      s_f[w][g] = f;
    }
    s_M[g] = M;
    s_L[g] = L;
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < G * HD; idx += blockDim.x) {
    const int g = idx / HD, d = idx - g * HD;
    const float M = s_M[g];
    float A = 0.f;
    if (M != -INFINITY)
      for (int w = 0; w < 4; ++w) A += s_acc[w][g][d] * s_f[w][g];
    float* pr = part + ((size_t(r) * nh + kvh * G + g) * slots + split) * (HD + 2);
    if (d == 0) {
      pr[0] = M;
      pr[1] = s_L[g];
    }
    pr[2 + d] = A;
  }
}

// The rows' query vectors in the copy's basis (fwht), once for the scan and the corrections: grid (rows, nh), block HD.
template <int HD>
__global__ void __launch_bounds__(HD) k_strata_rotate_q(const float* __restrict__ q, float* __restrict__ qr) {
  pdl_trigger();
  __shared__ float x[HD];
  const size_t at = (size_t(blockIdx.x) * gridDim.y + blockIdx.y) * HD;
  x[threadIdx.x] = q[at + threadIdx.x];
  fwht<HD>(x, 1);
  qr[at + threadIdx.x] = x[threadIdx.x];
}

// The scan on tensor cores: a warp takes one row, one KV head and one key split on its own, the KV
// head's query heads in the columns of mma.m16n8k16 (S^T = K Q^T, O^T += V^T P^T, as attention_rows_mma), and loads
// and decodes the copy's codes for its own fragments (StrataLayout: every load one 128-byte line), with no tile shared
// through shared memory and no barrier.  The codes go into the MMAs as they lie, the grids' steps and bases applied in
// float32 outside them: K as 1024 + c 2^(bits sp) (FoldSlots), its query scaled by 2^-(bits sp), the scores
// step (S - 1024 sum q) + base sum q; V as its exact codes, p times the key's step into the MMA, p times its base a
// column's sum.  A tile's scores are taken one tile ahead of its softmax, so that their MMAs run while the softmax
// before them waits on its own chain (FlashAttention-3's overlap within a warp).  A row's result depends on its own
// query and position only (rule 4).  Also each pair of positions' log-sum-exp per query head.
constexpr int kScanTile = 32;
constexpr int kScanQueryTop = 14;  // a query vector scaled so that its largest |value| is below 2^14
constexpr int kScanPTop = 14;      // p times a V step into the MMAs times 2^k, below 2^14 (k chosen a tile)

// 2^k for |k| < 127, exactly, by its bits.
__device__ __forceinline__ float pow2(int k) { return __int_as_float((127 + k) << 23); }

// (a & m) | c in one instruction.
__device__ __forceinline__ uint32_t and_or(uint32_t a, uint32_t m, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(m), "r"(c));
  return d;
}

// A word's code pair at slot s as two binary16 1024 + c 2^(BITS sp), exactly, by one mask: the slots of a half in
// groups of R, a group's codes within binary16's 10-bit fraction as they lie, the first group in place and the next
// shifted down once (`shifted`: the word >> R BITS); sp the slot within its group.
template <int BITS>
struct FoldSlots {
  static constexpr int R = (10 - BITS) / BITS + 1;
  __host__ __device__ static constexpr int sp(int s) { return s % R; }
  __device__ static uint32_t pair(uint32_t word, uint32_t shifted, int s) {
    return and_or(s / R ? shifted : word, (((1u << BITS) - 1u) << (BITS * sp(s))) * 0x10001u, 0x64006400u);
  }
};

// True when every row in [0, rows) gets the same starting key `cks` and is nonempty for `split`.
// At long contexts this holds for all 64 splits (including the unaligned final split), so all rows run packed in 1 pass.
__device__ __forceinline__ bool strata_split_same_cks(int pos, int rows, int split, int splits, int* cks_out,
                                                      int* e0_out = nullptr, int* el_out = nullptr) {
  int b0, e0;
  attn_split_range(pos + 1, split, splits, &b0, &e0);
  if (b0 >= e0) return false;
  int el = e0;
  for (int r = 1; r < rows; ++r) {
    int br, er;
    attn_split_range(pos + r + 1, split, splits, &br, &er);
    if (br != b0 || br >= er) return false;
    el = er;
  }
  *cks_out = b0;
  if (e0_out) *e0_out = e0;
  if (el_out) *el_out = el;
  return true;
}

template <int HD, int BITS, bool FullAligned, bool Compact = false>
__device__ __forceinline__ void strata_scan_1w_body(
    const float* qr, KvRef kv, int nh, int nkv, int G, int rows, float scale, float* __restrict__ part,
    __half* __restrict__ lse, int lse_stride, int split, int splits, int kvh, int global_warp, bool packed,
    int tail_row, int my_cks, int max_cke_warp, const bool (&cok)[2], const int (&lse_rh)[2], const int (&cke)[2]) {
  constexpr StrataLayout L(HD, BITS);
  constexpr int KS = HD / 16, SUB = kScanTile / 16, RB = FoldSlots<BITS>::R * BITS;
  constexpr int KWL = L.kown + (L.kleft ? 1 : 0), VWL = L.vown + (L.vleft ? 1 : 0);
  constexpr uint32_t FULL = 0xffffffffu;
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int total_cols = rows * G;

  uint32_t bq[KS][2];
  int sh = 0;
  float qo = 0.f, qs = 0.f;
  {
    bool qok;
    int qrow, qh;
    if (packed) {
      const int cg = global_warp * 8 + g;
      qok = cg < total_cols;
      qrow = qok ? cg / G : 0;
      qh = qok ? cg - qrow * G : 0;
    } else {
      qok = g < G;
      qrow = tail_row;
      qh = qok ? g : 0;
    }
    float qx[KS][4];
    float mx = 0.f;
    const float* qp = qr + (size_t(qrow) * nh + kvh * G + qh) * HD;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks)
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        qx[ks][h] = qok ? qp[16 * ks + 2 * t + (h & 1) + 8 * (h >> 1)] : 0.f;
        mx = fmaxf(mx, fabsf(qx[ks][h]));
      }
    mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
    int ex = 0;
    frexpf(mx, &ex);
    sh = kScanQueryTop - ex;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      int e[2];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        int w = 0, s = 0;
        L.k_at(t, 2 * ks + c, &w, &s);
        e[c] = sh - BITS * FoldSlots<BITS>::sp(s);
      }
      __half x[4];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        x[h] = __float2half_rn(ldexpf(qx[ks][h], e[h >> 1]));
        qo = __fadd_rn(qo, __half2float(x[h]));
        qs = __fadd_rn(qs, __half2float(__float2half_rn(ldexpf(qx[ks][h], sh))));
      }
      bq[ks][0] = pack_halves(x[0], x[1]);
      bq[ks][1] = pack_halves(x[2], x[3]);
    }
#pragma unroll
    for (int o = 1; o < 4; o <<= 1) {
      qo = __fadd_rn(qo, __shfl_xor_sync(FULL, qo, o));
      qs = __fadd_rn(qs, __shfl_xor_sync(FULL, qs, o));
    }
  }
  float cs[2], qoff[2], qsum[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int from = 4 * (2 * t + j);
    cs[j] = ldexpf(scale, -__shfl_sync(FULL, sh, from));
    qoff[j] = __fmul_rn(1024.f, __shfl_sync(FULL, qo, from));
    qsum[j] = __shfl_sync(FULL, qs, from);
  }
  float o[KS][4];
#pragma unroll
  for (int mt = 0; mt < KS; ++mt) o[mt][0] = o[mt][1] = o[mt][2] = o[mt][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, beta[2] = {0.f, 0.f};
  int vs[2] = {0, 0};

  const size_t stride16 = Compact ? size_t(nkv) * 384 : size_t(nkv) * 16 * kv.words;
  const uint32_t* kb = kv.kq + (Compact ? size_t(kvh) * 384 : size_t(kvh) * 16 * kv.words);
  const uint32_t* vb = kv.vq + (Compact ? size_t(kvh) * 384 : size_t(kvh) * 16 * kv.words);

  uint32_t kw[SUB][2][KWL], vw[SUB][2][2][VWL];
  __half2 kgrid[SUB][2], vgrid[SUB][2];
  auto load_k = [&](int tile) {
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb) {
      const uint32_t* p = kb + size_t(2 * tile + sb) * stride16;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int key = tile * kScanTile + 16 * sb + g + 8 * h;
#pragma unroll
        for (int i = 0; i < L.kown; ++i) kw[sb][h][i] = __ldg(p + (i * 2 + h) * 32 + lane);
        if constexpr (Compact) {
          uint32_t tail = 0;
#pragma unroll
          for (int i = 0; i < 6; ++i) tail |= ((kw[sb][h][i] >> 15) & 0x10001u) << i;
          kw[sb][h][KWL - 1] = tail << (6 * (t & 1));
        } else if (L.kleft) kw[sb][h][KWL - 1] = __ldg(p + L.kown * 64 + (h * 8 + g) * L.ksh + t / L.kgpw);
        kgrid[sb][h] = FullAligned || key < max_cke_warp ? __ldg(kv.kqs + size_t(key) * nkv + kvh)
                                                         : __floats2half2_rn(0.f, 0.f);
      }
    }
  };
  auto load_v = [&](int tile) {
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb) {
      const uint32_t* p = vb + size_t(2 * tile + sb) * stride16;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
#pragma unroll
          for (int i = 0; i < L.vown; ++i) vw[sb][h][e][i] = __ldg(p + ((i * 2 + h) * 2 + e) * 32 + lane);
          if constexpr (Compact) {
            uint32_t tail = 0;
#pragma unroll
            for (int i = 0; i < 3; ++i) tail |= ((vw[sb][h][e][i] >> 15) & 0x10001u) << i;
            vw[sb][h][e][VWL - 1] = tail << (3 * (g % 5));
          } else if (L.vleft) vw[sb][h][e][VWL - 1] = __ldg(p + L.vown * 128 + ((h * 2 + e) * 4 + t) * L.vsh + g / L.vgpw);
        }
        const int key = tile * kScanTile + 16 * sb + g + 8 * h;
        vgrid[sb][h] = FullAligned || key < max_cke_warp ? __ldg(kv.vqs + size_t(key) * nkv + kvh)
                                                         : __floats2half2_rn(0.f, 0.f);
      }
    }
  };
  auto qk_to_v = [&](int tile, float (&v_out)[SUB][4]) {
    float sp[SUB][2][4];
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int e = 0; e < 4; ++e) sp[sb][h][e] = 0.f;
    int cached_wi = -1;
    uint32_t kw_c[SUB][2], kw_s[SUB][2];
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int j0 = 2 * ks, j1 = 2 * ks + 1;
      const int wi0 = j0 < L.kown * L.cph ? j0 / L.cph : KWL - 1;
      const int wi1 = j1 < L.kown * L.cph ? j1 / L.cph : KWL - 1;
      if (wi0 != cached_wi) {
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            kw_c[sb][h] = kw[sb][h][wi0];
            kw_s[sb][h] = kw_c[sb][h] >> RB;
          }
        cached_wi = wi0;
      }
      uint32_t kw_n[SUB][2], kw_ns[SUB][2];
      if (wi1 != wi0) {
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            kw_n[sb][h] = kw[sb][h][wi1];
            kw_ns[sb][h] = kw_n[sb][h] >> RB;
          }
      }
      int w0 = 0, s0 = 0, w1 = 0, s1 = 0;
      L.k_at(t, j0, &w0, &s0);
      L.k_at(t, j1, &w1, &s1);
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb) {
        uint32_t a[4];
        a[0] = FoldSlots<BITS>::pair(kw_c[sb][0], kw_s[sb][0], s0);
        a[1] = FoldSlots<BITS>::pair(kw_c[sb][1], kw_s[sb][1], s0);
        if (wi1 == wi0) {
          a[2] = FoldSlots<BITS>::pair(kw_c[sb][0], kw_s[sb][0], s1);
          a[3] = FoldSlots<BITS>::pair(kw_c[sb][1], kw_s[sb][1], s1);
        } else {
          a[2] = FoldSlots<BITS>::pair(kw_n[sb][0], kw_ns[sb][0], s1);
          a[3] = FoldSlots<BITS>::pair(kw_n[sb][1], kw_ns[sb][1], s1);
        }
        mma_f16(sp[sb][ks & 1], a, bq[ks]);
      }
      if (wi1 != wi0) {
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            kw_c[sb][h] = kw_n[sb][h];
            kw_s[sb][h] = kw_ns[sb][h];
          }
        cached_wi = wi1;
      }
    }
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb) {
      float kstep[2], kbase[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float2 sl = __half22float2(kgrid[sb][h]);
        kstep[h] = sl.y;
        kbase[h] = __fmaf_rn(0.5f, sl.y, sl.x);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int j = e & 1, h = e >> 1, key = tile * kScanTile + 16 * sb + g + 8 * h;
        const float x = __fmaf_rn(kstep[h], __fsub_rn(__fadd_rn(sp[sb][0][e], sp[sb][1][e]), qoff[j]),
                                  __fmul_rn(kbase[h], qsum[j]));
        v_out[sb][e] = cok[j] && (FullAligned || key < cke[j]) ? __fmul_rn(x, cs[j]) : -INFINITY;
      }
    }
  };

  const __half2 k1024 = __float2half2_rn(1024.f);
  const int t0 = my_cks / kScanTile, t1 = (max_cke_warp - 1) / kScanTile;
  load_k(t0);
  load_v(t0);
  float v[SUB][4];
  qk_to_v(t0, v);
  load_k(min(t0 + 1, t1));

  for (int tile = t0; tile <= t1; ++tile) {
    float vn[SUB][4];
    qk_to_v(min(tile + 1, t1), vn);
    load_k(min(tile + 2, t1));
    float tm[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      tm[j] = fmaxf(v[0][j], v[0][j + 2]);
#pragma unroll
      for (int sb = 1; sb < SUB; ++sb) tm[j] = fmaxf(tm[j], fmaxf(v[sb][j], v[sb][j + 2]));
#pragma unroll
      for (int x = 4; x < 32; x <<= 1) tm[j] = fmaxf(tm[j], __shfl_xor_sync(FULL, tm[j], x));
    }
    float corr[2], p[SUB][4];
    bool moved = false;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float mn = fmaxf(m[j], tm[j]);
      corr[j] = mn == m[j] ? 1.f : __expf(m[j] - mn);
      moved |= mn != m[j];
      m[j] = mn;
    }
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
      for (int e = 0; e < 4; ++e) p[sb][e] = v[sb][e] == -INFINITY ? 0.f : __expf(v[sb][e] - m[e & 1]);

    if constexpr (FullAligned) {
      const bool upper = g & 1;
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float lo = p[sb][j], hi = p[sb][j + 2];
          const float local = upper ? hi : lo;
          const float peer = __shfl_xor_sync(FULL, upper ? lo : hi, 4);
          const int key = tile * kScanTile + 16 * sb + (g & ~1) + 8 * int(upper);
          if (cok[j]) {
            const float sum = upper ? __fadd_rn(peer, local) : __fadd_rn(local, peer);
            const float s = __fadd_rn(m[j], __logf(sum));
            lse[size_t(lse_rh[j]) * lse_stride + (key >> 1)] = __float2half_rn(s);
          }
        }
    } else if (lse) {
      const bool upper = g & 1;
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const float lo = p[sb][j], hi = p[sb][j + 2];
          const float local = upper ? hi : lo;
          const float peer = __shfl_xor_sync(FULL, upper ? lo : hi, 4);
          const int key = tile * kScanTile + 16 * sb + (g & ~1) + 8 * int(upper);
          if (cok[j] && key < cke[j]) {
            const float sum = upper ? __fadd_rn(peer, local) : __fadd_rn(local, peer);
            const float s = __fadd_rn(m[j], __logf(sum));
            lse[size_t(lse_rh[j]) * lse_stride + (key >> 1)] = __float2half_rn(s);
          }
        }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float x = __fmul_rn(l[j], corr[j]);
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb) x = __fadd_rn(__fadd_rn(x, p[sb][j]), p[sb][j + 2]);
      l[j] = x;
    }

    float vstep[SUB][2], vbase[SUB][2];
    if constexpr (FullAligned) {
      float most = 0.f;
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 sl = __half22float2(vgrid[sb][h]);
          vstep[sb][h] = sl.y;
          vbase[sb][h] = __fmaf_rn(0.5f, sl.y, sl.x);
          most = fmaxf(most, sl.y);
        }
#pragma unroll
      for (int x = 4; x < 32; x <<= 1) most = fmaxf(most, __shfl_xor_sync(FULL, most, x));
      const int ex = most > 0.f ? int((__float_as_uint(most) >> 23) & 0xff) - 126 : 0;
      const int vs_new = tile == t0 ? kScanPTop - ex : min(vs[0], kScanPTop - ex);
      if (__any_sync(FULL, moved) || vs_new != vs[0]) {
        float f[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) f[j] = __fmul_rn(corr[j], pow2(vs_new - vs[0]));
#pragma unroll
        for (int mt = 0; mt < KS; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[mt][e] = __fmul_rn(o[mt][e], f[e & 1]);
        vs[0] = vs[1] = vs_new;
      }
    } else {
      float most[2] = {0.f, 0.f};
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int key = tile * kScanTile + 16 * sb + g + 8 * h;
          const float2 sl = __half22float2(vgrid[sb][h]);
          vstep[sb][h] = sl.y;
          vbase[sb][h] = __fmaf_rn(0.5f, sl.y, sl.x);
#pragma unroll
          for (int j = 0; j < 2; ++j)
            if (key < cke[j]) most[j] = fmaxf(most[j], sl.y);
        }
      int vs_new[2];
      bool vs_changed = false;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) most[j] = fmaxf(most[j], __shfl_xor_sync(FULL, most[j], x));
        if (tile * kScanTile < cke[j]) {
          const int ex = most[j] > 0.f ? int((__float_as_uint(most[j]) >> 23) & 0xff) - 126 : 0;
          vs_new[j] = tile == t0 ? kScanPTop - ex : min(vs[j], kScanPTop - ex);
        } else {
          vs_new[j] = vs[j];
        }
        vs_changed |= vs_new[j] != vs[j];
      }
      if (__any_sync(FULL, moved || vs_changed)) {
        float f[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) f[j] = __fmul_rn(corr[j], pow2(vs_new[j] - vs[j]));
#pragma unroll
        for (int mt = 0; mt < KS; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[mt][e] = __fmul_rn(o[mt][e], f[e & 1]);
        vs[0] = vs_new[0];
        vs[1] = vs_new[1];
      }
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float x = __fmul_rn(beta[j], corr[j]);
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
        x = __fmaf_rn(p[sb][j + 2], vbase[sb][1], __fmaf_rn(p[sb][j], vbase[sb][0], x));
      beta[j] = x;
    }

    const float p2_0 = pow2(vs[0]), p2_1 = FullAligned ? p2_0 : pow2(vs[1]);
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb) {
      uint32_t bp[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const float f0 = __fmul_rn(vstep[sb][h], p2_0);
        const float f1 = FullAligned ? f0 : __fmul_rn(vstep[sb][h], p2_1);
        bp[h] = movmatrix_trans(
            pack_halves(__float2half_rn(__fmul_rn(p[sb][2 * h], f0)), __float2half_rn(__fmul_rn(p[sb][2 * h + 1], f1))));
      }
#pragma unroll
      for (int wi = 0; wi < VWL; ++wi) {
        uint32_t vlo[2], vhi[2], vlo_sh[2], vhi_sh[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          vlo[h] = __byte_perm(vw[sb][h][0][wi], vw[sb][h][1][wi], 0x5410);
          vhi[h] = __byte_perm(vw[sb][h][0][wi], vw[sb][h][1][wi], 0x7632);
          vlo_sh[h] = vlo[h] >> RB;
          vhi_sh[h] = vhi[h] >> RB;
        }
        const int mt0 = wi * L.cph;
        const int mt1 = (wi + 1) * L.cph < KS ? (wi + 1) * L.cph : KS;
#pragma unroll
        for (int mt = mt0; mt < mt1; ++mt) {
          int w = 0, s = 0;
          L.v_at(g, mt, &w, &s);
          uint32_t a[4];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t b0 = FoldSlots<BITS>::pair(vlo[h], vlo_sh[h], s);
            const uint32_t b1 = FoldSlots<BITS>::pair(vhi[h], vhi_sh[h], s);
            const __half2 c0 = __hsub2(*reinterpret_cast<const __half2*>(&b0), k1024);
            const __half2 c1 = __hsub2(*reinterpret_cast<const __half2*>(&b1), k1024);
            a[2 * h] = *reinterpret_cast<const uint32_t*>(&c0);
            a[2 * h + 1] = *reinterpret_cast<const uint32_t*>(&c1);
          }
          mma_f16(o[mt], a, bp);
        }
      }
    }
    load_v(min(tile + 1, t1));
#pragma unroll
    for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
      for (int e = 0; e < 4; ++e) v[sb][e] = vn[sb][e];
  }

#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int x = 4; x < 32; x <<= 1) {
      l[j] = __fadd_rn(l[j], __shfl_xor_sync(FULL, l[j], x));
      beta[j] = __fadd_rn(beta[j], __shfl_xor_sync(FULL, beta[j], x));
    }
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    if (!cok[j]) continue;
    float* pr = part + (size_t(lse_rh[j]) * splits + split) * (HD + 2);
    if (g == 0) {
      pr[0] = m[j];
      pr[1] = l[j];
    }
#pragma unroll
    for (int mt = 0; mt < KS; ++mt) {
      int w = 0, s = 0;
      L.v_at(g, mt, &w, &s);
      const float f = ldexpf(1.f, -vs[j] - BITS * FoldSlots<BITS>::sp(s));
#pragma unroll
      for (int h = 0; h < 2; ++h) pr[2 + 16 * mt + g + 8 * h] = __fmaf_rn(o[mt][2 * h + j], f, beta[j]);
    }
  }
}

template <int HD, int BITS, bool Compact = false>
__global__ void __launch_bounds__(32)
    k_strata_scan_mma(const float* qr, KvRef kv, int nh, int nkv, int rows, float scale, const DevParams* P,
                                float* __restrict__ part, __half* __restrict__ lse, int lse_stride,
                                int first_split = 0, int total_splits = 0) {
  pdl_trigger();
  const int split = (int(gridDim.z) - 1 - int(blockIdx.z)) + first_split;
  const int splits = total_splits ? total_splits : int(gridDim.z);
  const int G = nh / nkv;
  const int kvh = int(blockIdx.y);
  const int lane = int(threadIdx.x), t = lane & 3;
  const int global_warp = int(blockIdx.x);
  const int total_warps = int(gridDim.x);
  const int total_cols = rows * G;

  int b0 = 0, e0 = 0, el = 0;
  const bool packed = strata_split_same_cks(P->position, rows, split, splits, &b0, &e0, &el);
  if constexpr (HD == 256 && BITS == 2) {
    if (packed && lse && e0 == el && e0 % kScanTile == 0) {
      bool cok[2];
      int lse_rh[2], cke[2];
      const int col_base = global_warp * 8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int cc = col_base + 2 * t + j;
        cok[j] = cc < total_cols;
        const int cr = cok[j] ? cc / G : 0;
        const int ch = cok[j] ? cc - cr * G : 0;
        lse_rh[j] = cr * nh + kvh * G + ch;
        cke[j] = e0;
      }
      strata_scan_1w_body<HD, BITS, true, Compact>(
          qr, kv, nh, nkv, G, rows, scale, part, lse, lse_stride, split, splits, kvh, global_warp, true, 0, b0, el,
          cok, lse_rh, cke);
      return;
    }
  }

  const int passes = packed ? 1 : ((rows + total_warps - 1) / total_warps);
  for (int pass = 0; pass < passes; ++pass) {
    bool cok[2];
    int lse_rh[2], cke[2];
    int my_cks = b0, max_cke_warp = el, tail_row = 0;
    if (packed) {
      const int col_base = global_warp * 8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int cc = col_base + 2 * t + j;
        cok[j] = cc < total_cols;
        const int cr = cok[j] ? cc / G : 0;
        const int ch = cok[j] ? cc - cr * G : 0;
        lse_rh[j] = cr * nh + kvh * G + ch;
        if (cok[j]) {
          int rb, re;
          attn_split_range(P->position + cr + 1, split, splits, &rb, &re);
          cke[j] = re;
        } else {
          cke[j] = 0;
        }
      }
    } else {
      tail_row = global_warp + pass * total_warps;
      if (tail_row >= rows) break;
      int rb, re;
      attn_split_range(P->position + tail_row + 1, split, splits, &rb, &re);
      if (rb >= re) {
        for (int h = 0; h < G; ++h) {
          const size_t base = ((size_t(tail_row) * nh + kvh * G + h) * splits + split) * (HD + 2);
          for (int e = lane; e < HD + 2; e += 32) part[base + e] = e == 0 ? -INFINITY : 0.f;
        }
        continue;
      }
      my_cks = rb;
      max_cke_warp = re;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int c = 2 * t + j;
        cok[j] = c < G;
        lse_rh[j] = tail_row * nh + kvh * G + (cok[j] ? c : 0);
        cke[j] = cok[j] ? re : 0;
      }
    }
    strata_scan_1w_body<HD, BITS, false, Compact>(
        qr, kv, nh, nkv, G, rows, scale, part, lse, lse_stride, split, splits, kvh, global_warp, packed, tail_row,
        my_cks, max_cke_warp, cok, lse_rh, cke);
  }
}

// Packed multi-row scan CTA (3/4-bit): packs up to 8*W (row, query-head) columns across W warps in one
// CTA (e.g. 4 rows x 6 heads = 24 columns in 3 full warps), cooperatively staging each 32-key K/V tile once via
// cp.async into double-buffered s_k + triple-buffered s_v (one barrier per tile, 4 CTAs/SM = 12 warps/SM even for
// compact 3-bit). Per-column cke[2] and vs[2] handle all splits including the unaligned tail in one packed pass
// whenever all rows share the same starting key (rule 4, bit-identical).
template <int HD, int BITS, int W, bool Compact = false>
__global__ void __launch_bounds__(32 * W, W == 3 ? 4 : (W == 2 ? 6 : 3))
    k_strata_scan_mma_cta(const float* qr, KvRef kv, int nh, int nkv, int rows, float scale, const DevParams* P,
                             float* __restrict__ part, __half* __restrict__ lse, int lse_stride,
                             int first_split = 0, int total_splits = 0) {
  pdl_trigger();
  constexpr StrataLayout L(HD, BITS);
  constexpr int KS = HD / 16, SUB = kScanTile / 16, RB = FoldSlots<BITS>::R * BITS;
  constexpr int VWL = L.vown + (L.vleft ? 1 : 0);
  constexpr int KW_SUB = Compact ? 384 : 16 * L.kwords();
  constexpr int VW_SUB = Compact ? 384 : 16 * L.vwords();
  constexpr int K_V4 = KW_SUB / 4, V_V4 = VW_SUB / 4;
  constexpr int NBUF_K = 2, NBUF_V = 3;
  constexpr int THREADS = 32 * W;
  constexpr uint32_t FULL = 0xffffffffu;

  constexpr int BQ_SMEM = KS / 2;  // upper half of bq in SMEM (6 KB for W=3, HD=256): frees 16 regs -> 0 spills!
  __shared__ __align__(16) uint32_t s_k[NBUF_K][SUB][KW_SUB];
  __shared__ __align__(16) uint32_t s_v[NBUF_V][SUB][VW_SUB];
  __shared__ __align__(16) __half2 s_kg[NBUF_K][kScanTile];
  __shared__ __align__(16) __half2 s_vg[NBUF_V][kScanTile];
  __shared__ __align__(16) uint2 s_bq[W][BQ_SMEM][32];

  const int split = (int(gridDim.z) - 1 - int(blockIdx.z)) + first_split;
  const int splits = total_splits ? total_splits : int(gridDim.z);
  const int G = nh / nkv;
  const int kvh = int(blockIdx.y);
  const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int total_cols = rows * G;
  const int global_warp = int(blockIdx.x) * W + warp;
  const int total_warps = int(gridDim.x) * W;

  int b0 = 0;
  const bool packed = strata_split_same_cks(P->position, rows, split, splits, &b0);
  const int passes = packed ? 1 : ((rows + total_warps - 1) / total_warps);

  const size_t stride16 = Compact ? size_t(nkv) * 384 : size_t(nkv) * 16 * kv.words;
  const uint32_t* kb = kv.kq + (Compact ? size_t(kvh) * 384 : size_t(kvh) * 16 * kv.words);
  const uint32_t* vb = kv.vq + (Compact ? size_t(kvh) * 384 : size_t(kvh) * 16 * kv.words);
  const __half2 k1024 = __float2half2_rn(1024.f);

  for (int pass = 0; pass < passes; ++pass) {
    const int tail_row = global_warp + pass * total_warps;
    int my_cks = b0;
    int cke[2] = {0, 0};
    bool cok[2];
    int lse_rh[2];
    bool my_active = true;
    int t0_cta = b0 / kScanTile, t1_cta = -1, max_cke_cta = 0;

    if (packed) {
      const int col_base = global_warp * 8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int cc = col_base + 2 * t + j;
        cok[j] = cc < total_cols;
        const int cr = cok[j] ? cc / G : 0;
        const int ch = cok[j] ? cc - cr * G : 0;
        lse_rh[j] = cr * nh + kvh * G + ch;
        if (cok[j]) {
          int rb, re;
          attn_split_range(P->position + cr + 1, split, splits, &rb, &re);
          cke[j] = re;
        }
      }
      // CTA tile end = max over all rows in this split (last row has largest end).
      int rb_last, re_last;
      attn_split_range(P->position + rows, split, splits, &rb_last, &re_last);
      max_cke_cta = re_last;
      t1_cta = (max_cke_cta - 1) / kScanTile;
    } else {
      my_active = tail_row < rows;
      int re = 0;
      if (my_active) {
        attn_split_range(P->position + tail_row + 1, split, splits, &my_cks, &re);
        if (my_cks >= re) {
          for (int h = 0; h < G; ++h) {
            const size_t base = ((size_t(tail_row) * nh + kvh * G + h) * splits + split) * (HD + 2);
            for (int e = lane; e < HD + 2; e += 32) part[base + e] = e == 0 ? -INFINITY : 0.f;
          }
          my_active = false;
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int c = 2 * t + j;
        cok[j] = my_active && c < G;
        lse_rh[j] = (my_active ? tail_row : 0) * nh + kvh * G + (cok[j] ? c : 0);
        cke[j] = cok[j] ? re : 0;
      }
      t0_cta = INT_MAX;
      t1_cta = -1;
      max_cke_cta = 0;
      const int cta_row0 = int(blockIdx.x) * W + pass * total_warps;
      for (int w = 0; w < W; ++w) {
        const int rw = cta_row0 + w;
        if (rw >= rows) break;
        int rb, r_end;
        attn_split_range(P->position + rw + 1, split, splits, &rb, &r_end);
        if (rb < r_end) {
          t0_cta = min(t0_cta, rb / kScanTile);
          t1_cta = max(t1_cta, (r_end - 1) / kScanTile);
          max_cke_cta = max(max_cke_cta, r_end);
        }
      }
      if (t1_cta < t0_cta) continue;
    }

    const int my_t0 = my_active ? (my_cks / kScanTile) : (t1_cta + 1);

    uint32_t bq[KS - BQ_SMEM][2];
    int sh = 0;
    float qo = 0.f, qs = 0.f;
    {
      bool qok;
      int qrow, qh;
      if (packed) {
        const int cg = global_warp * 8 + g;
        qok = cg < total_cols;
        qrow = qok ? cg / G : 0;
        qh = qok ? cg - qrow * G : 0;
      } else {
        qok = my_active && g < G;
        qrow = my_active ? tail_row : 0;
        qh = qok ? g : 0;
      }
      float qx[KS][4];
      float mx = 0.f;
      const float* qp = qr + (size_t(qrow) * nh + kvh * G + qh) * HD;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks)
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          qx[ks][h] = qok ? qp[16 * ks + 2 * t + (h & 1) + 8 * (h >> 1)] : 0.f;
          mx = fmaxf(mx, fabsf(qx[ks][h]));
        }
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
      int ex = 0;
      frexpf(mx, &ex);
      sh = kScanQueryTop - ex;
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        int e[2];
#pragma unroll
        for (int c = 0; c < 2; ++c) {
          int w = 0, s = 0;
          L.k_at(t, 2 * ks + c, &w, &s);
          e[c] = sh - BITS * FoldSlots<BITS>::sp(s);
        }
        __half x[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          x[h] = __float2half_rn(ldexpf(qx[ks][h], e[h >> 1]));
          qo = __fadd_rn(qo, __half2float(x[h]));
          qs = __fadd_rn(qs, __half2float(__float2half_rn(ldexpf(qx[ks][h], sh))));
        }
        const uint32_t p0 = pack_halves(x[0], x[1]), p1 = pack_halves(x[2], x[3]);
        if (ks < KS - BQ_SMEM) {
          bq[ks][0] = p0;
          bq[ks][1] = p1;
        } else {
          s_bq[warp][ks - (KS - BQ_SMEM)][lane] = make_uint2(p0, p1);
        }
      }
#pragma unroll
      for (int o = 1; o < 4; o <<= 1) {
        qo = __fadd_rn(qo, __shfl_xor_sync(FULL, qo, o));
        qs = __fadd_rn(qs, __shfl_xor_sync(FULL, qs, o));
      }
    }
    float cs[2], qoff[2], qsum[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int from = 4 * (2 * t + j);
      cs[j] = ldexpf(scale, -__shfl_sync(FULL, sh, from));
      qoff[j] = __fmul_rn(1024.f, __shfl_sync(FULL, qo, from));
      qsum[j] = __shfl_sync(FULL, qs, from);
    }

    float o[KS][4];
#pragma unroll
    for (int mt = 0; mt < KS; ++mt) o[mt][0] = o[mt][1] = o[mt][2] = o[mt][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f}, beta[2] = {0.f, 0.f};
    int vs[2] = {0, 0};

    auto async_load_tile = [&](int tile, int kbuf, int vbuf) {
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb) {
        const uint32_t* kp = kb + size_t(2 * tile + sb) * stride16;
        const uint32_t* vp = vb + size_t(2 * tile + sb) * stride16;
#pragma unroll
        for (int i = tid; i < K_V4; i += THREADS) cp_async16(&s_k[kbuf][sb][i * 4], kp + i * 4);
#pragma unroll
        for (int i = tid; i < V_V4; i += THREADS) cp_async16(&s_v[vbuf][sb][i * 4], vp + i * 4);
      }
      if (tid < kScanTile) {
        const int key = tile * kScanTile + tid;
        cp_async4(&s_kg[kbuf][tid], kv.kqs + size_t(key) * nkv + kvh, key < max_cke_cta);
      }
      if (tid >= kScanTile && tid < 2 * kScanTile) {
        const int kidx = tid - kScanTile;
        const int key = tile * kScanTile + kidx;
        cp_async4(&s_vg[vbuf][kidx], kv.vqs + size_t(key) * nkv + kvh, key < max_cke_cta);
      }
      cp_async_commit();
    };

    auto qk_to_v = [&](int tile, int buf, float (&v_out)[SUB][4]) {
      float sp[SUB][2][4];
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int e = 0; e < 4; ++e) sp[sb][h][e] = 0.f;

      uint32_t ktail[SUB][2];
      if constexpr (Compact) {
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            uint32_t tail = 0;
#pragma unroll
            for (int i = 0; i < 6; ++i) tail |= ((s_k[buf][sb][(i * 2 + h) * 32 + lane] >> 15) & 0x10001u) << i;
            ktail[sb][h] = tail << (6 * (t & 1));
          }
      }
      auto fetch_kw = [&](int sb, int h, int wi) -> uint32_t {
        if (wi < L.kown) return s_k[buf][sb][(wi * 2 + h) * 32 + lane];
        if constexpr (Compact) return ktail[sb][h];
        return s_k[buf][sb][L.kown * 64 + (h * 8 + g) * L.ksh + t / L.kgpw];
      };
      int cached_wi = -1;
      uint32_t kw_c[SUB][2], kw_s[SUB][2];
#pragma unroll
      for (int ks = 0; ks < KS; ++ks) {
        const int j0 = 2 * ks, j1 = 2 * ks + 1;
        const int wi0 = j0 < L.kown * L.cph ? j0 / L.cph : L.kown;
        const int wi1 = j1 < L.kown * L.cph ? j1 / L.cph : L.kown;
        if (wi0 != cached_wi) {
#pragma unroll
          for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              kw_c[sb][h] = fetch_kw(sb, h, wi0);
              kw_s[sb][h] = kw_c[sb][h] >> RB;
            }
          cached_wi = wi0;
        }
        uint32_t kw_n[SUB][2], kw_ns[SUB][2];
        if (wi1 != wi0) {
#pragma unroll
          for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              kw_n[sb][h] = fetch_kw(sb, h, wi1);
              kw_ns[sb][h] = kw_n[sb][h] >> RB;
            }
        }
        int w0 = 0, s0 = 0, w1 = 0, s1 = 0;
        L.k_at(t, j0, &w0, &s0);
        L.k_at(t, j1, &w1, &s1);
        uint32_t bq_cur[2];
        if (ks < KS - BQ_SMEM) {
          bq_cur[0] = bq[ks][0];
          bq_cur[1] = bq[ks][1];
        } else {
          const uint2 bq2 = s_bq[warp][ks - (KS - BQ_SMEM)][lane];
          bq_cur[0] = bq2.x;
          bq_cur[1] = bq2.y;
        }
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb) {
          uint32_t a[4];
          a[0] = FoldSlots<BITS>::pair(kw_c[sb][0], kw_s[sb][0], s0);
          a[1] = FoldSlots<BITS>::pair(kw_c[sb][1], kw_s[sb][1], s0);
          if (wi1 == wi0) {
            a[2] = FoldSlots<BITS>::pair(kw_c[sb][0], kw_s[sb][0], s1);
            a[3] = FoldSlots<BITS>::pair(kw_c[sb][1], kw_s[sb][1], s1);
          } else {
            a[2] = FoldSlots<BITS>::pair(kw_n[sb][0], kw_ns[sb][0], s1);
            a[3] = FoldSlots<BITS>::pair(kw_n[sb][1], kw_ns[sb][1], s1);
          }
          mma_f16(sp[sb][ks & 1], a, bq_cur);
        }
        if (wi1 != wi0) {
#pragma unroll
          for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              kw_c[sb][h] = kw_n[sb][h];
              kw_s[sb][h] = kw_ns[sb][h];
            }
          cached_wi = wi1;
        }
      }
      const bool tile_ok = my_active && tile >= my_t0;
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb) {
        float kstep[2], kbase[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float2 sl = __half22float2(s_kg[buf][16 * sb + g + 8 * h]);
          kstep[h] = sl.y;
          kbase[h] = __fmaf_rn(0.5f, sl.y, sl.x);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int j = e & 1, h = e >> 1, key = tile * kScanTile + 16 * sb + g + 8 * h;
          const float x = __fmaf_rn(kstep[h], __fsub_rn(__fadd_rn(sp[sb][0][e], sp[sb][1][e]), qoff[j]),
                                    __fmul_rn(kbase[h], qsum[j]));
          v_out[sb][e] = tile_ok && cok[j] && key < cke[j] ? __fmul_rn(x, cs[j]) : -INFINITY;
        }
      }
    };

    async_load_tile(t0_cta, 0, 0);
    if (t0_cta + 1 <= t1_cta) async_load_tile(t0_cta + 1, 1, 1);
    else cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();

    float v[SUB][4];
    qk_to_v(t0_cta, 0, v);

    int kcur = 0, cur = 0;
    for (int tile = t0_cta; tile <= t1_cta; ++tile) {
      const int knxt = kcur ^ 1;
      const int next_buf = cur == 2 ? 0 : cur + 1;
      const int load_buf = cur == 0 ? 2 : cur - 1;
      cp_async_wait<0>();
      __syncthreads();
      if (tile + 2 <= t1_cta) async_load_tile(tile + 2, kcur, load_buf);
      else cp_async_commit();

      // Next tile's QK and this tile's softmax in ONE basic block for instruction-scheduler interleaving!
      float vn[SUB][4];
      const int next_tile = tile < t1_cta ? tile + 1 : tile;
      const int qk_buf = tile < t1_cta ? knxt : kcur;
      qk_to_v(next_tile, qk_buf, vn);

      const bool do_tile = my_active && tile >= my_t0;
      float tm[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        tm[j] = fmaxf(v[0][j], v[0][j + 2]);
#pragma unroll
        for (int sb = 1; sb < SUB; ++sb) tm[j] = fmaxf(tm[j], fmaxf(v[sb][j], v[sb][j + 2]));
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) tm[j] = fmaxf(tm[j], __shfl_xor_sync(FULL, tm[j], x));
      }
      float corr[2], p[SUB][4];
      bool moved = false;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float mn = fmaxf(m[j], tm[j]);
        corr[j] = mn == m[j] ? 1.f : __expf(m[j] - mn);
        moved |= mn != m[j];
        m[j] = mn;
      }
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int e = 0; e < 4; ++e) p[sb][e] = v[sb][e] == -INFINITY ? 0.f : __expf(v[sb][e] - m[e & 1]);

      if (lse) {
        const bool upper = g & 1;
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float lo = p[sb][j], hi = p[sb][j + 2];
            const float local = upper ? hi : lo;
            const float peer = __shfl_xor_sync(FULL, upper ? lo : hi, 4);
            const int key = tile * kScanTile + 16 * sb + (g & ~1) + 8 * int(upper);
            if (do_tile && cok[j] && key < cke[j]) {
              const float sum = upper ? __fadd_rn(peer, local) : __fadd_rn(local, peer);
              const float s = __fadd_rn(m[j], __logf(sum));
              lse[size_t(lse_rh[j]) * lse_stride + (key >> 1)] = __float2half_rn(s);
            }
          }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float x = __fmul_rn(l[j], corr[j]);
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb) x = __fadd_rn(__fadd_rn(x, p[sb][j]), p[sb][j + 2]);
        l[j] = x;
      }

      float vstep[SUB][2], vbase[SUB][2];
      float most[2] = {0.f, 0.f};
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int key = tile * kScanTile + 16 * sb + g + 8 * h;
          const float2 sl = __half22float2(s_vg[cur][16 * sb + g + 8 * h]);
          vstep[sb][h] = sl.y;
          vbase[sb][h] = __fmaf_rn(0.5f, sl.y, sl.x);
#pragma unroll
          for (int j = 0; j < 2; ++j)
            if (key < cke[j]) most[j] = fmaxf(most[j], sl.y);
        }
      int vs_new[2];
      bool vs_changed = false;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) most[j] = fmaxf(most[j], __shfl_xor_sync(FULL, most[j], x));
        if (do_tile && tile * kScanTile < cke[j]) {
          const int ex = most[j] > 0.f ? int((__float_as_uint(most[j]) >> 23) & 0xff) - 126 : 0;
          vs_new[j] = tile == my_t0 ? kScanPTop - ex : min(vs[j], kScanPTop - ex);
        } else {
          vs_new[j] = vs[j];
        }
        vs_changed |= vs_new[j] != vs[j];
      }
      if (__any_sync(FULL, moved || vs_changed)) {
        float f[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) f[j] = __fmul_rn(corr[j], pow2(vs_new[j] - vs[j]));
#pragma unroll
        for (int mt = 0; mt < KS; ++mt)
#pragma unroll
          for (int e = 0; e < 4; ++e) o[mt][e] = __fmul_rn(o[mt][e], f[e & 1]);
        vs[0] = vs_new[0];
        vs[1] = vs_new[1];
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        float x = __fmul_rn(beta[j], corr[j]);
#pragma unroll
        for (int sb = 0; sb < SUB; ++sb)
          x = __fmaf_rn(p[sb][j + 2], vbase[sb][1], __fmaf_rn(p[sb][j], vbase[sb][0], x));
        beta[j] = x;
      }

      const float p2_0 = pow2(vs[0]), p2_1 = pow2(vs[1]);
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb) {
        uint32_t bp[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float f0 = __fmul_rn(vstep[sb][h], p2_0);
          const float f1 = __fmul_rn(vstep[sb][h], p2_1);
          bp[h] = movmatrix_trans(
              pack_halves(__float2half_rn(__fmul_rn(p[sb][2 * h], f0)), __float2half_rn(__fmul_rn(p[sb][2 * h + 1], f1))));
        }
        uint32_t vtail[2][2];
        if constexpr (Compact) {
#pragma unroll
          for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 2; ++e) vtail[h][e] = 0;
        }
#pragma unroll
        for (int wi = 0; wi < VWL; ++wi) {
          uint32_t vlo[2], vhi[2], vlo_sh[2], vhi_sh[2];
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            uint32_t w0, w1;
            if (wi < L.vown) {
              w0 = s_v[cur][sb][((wi * 2 + h) * 2 + 0) * 32 + lane];
              w1 = s_v[cur][sb][((wi * 2 + h) * 2 + 1) * 32 + lane];
              if constexpr (Compact) {
                vtail[h][0] |= ((w0 >> 15) & 0x10001u) << wi;
                vtail[h][1] |= ((w1 >> 15) & 0x10001u) << wi;
              }
            } else if constexpr (Compact) {
              const int sh_g = 3 * (g % 5);
              w0 = vtail[h][0] << sh_g;
              w1 = vtail[h][1] << sh_g;
            } else {
              w0 = s_v[cur][sb][L.vown * 128 + ((h * 2 + 0) * 4 + t) * L.vsh + g / L.vgpw];
              w1 = s_v[cur][sb][L.vown * 128 + ((h * 2 + 1) * 4 + t) * L.vsh + g / L.vgpw];
            }
            vlo[h] = __byte_perm(w0, w1, 0x5410);
            vhi[h] = __byte_perm(w0, w1, 0x7632);
            vlo_sh[h] = vlo[h] >> RB;
            vhi_sh[h] = vhi[h] >> RB;
          }
          const int mt0 = wi * L.cph;
          const int mt1 = (wi + 1) * L.cph < KS ? (wi + 1) * L.cph : KS;
#pragma unroll
          for (int mt = mt0; mt < mt1; ++mt) {
            int w = 0, s = 0;
            L.v_at(g, mt, &w, &s);
            uint32_t a[4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t b0 = FoldSlots<BITS>::pair(vlo[h], vlo_sh[h], s);
              const uint32_t b1 = FoldSlots<BITS>::pair(vhi[h], vhi_sh[h], s);
              const __half2 c0 = __hsub2(*reinterpret_cast<const __half2*>(&b0), k1024);
              const __half2 c1 = __hsub2(*reinterpret_cast<const __half2*>(&b1), k1024);
              a[2 * h] = *reinterpret_cast<const uint32_t*>(&c0);
              a[2 * h + 1] = *reinterpret_cast<const uint32_t*>(&c1);
            }
            mma_f16(o[mt], a, bp);
          }
        }
      }
#pragma unroll
      for (int sb = 0; sb < SUB; ++sb)
#pragma unroll
        for (int e = 0; e < 4; ++e) v[sb][e] = vn[sb][e];
      kcur = knxt;
      cur = next_buf;
    }
    cp_async_wait<0>();
    __syncthreads();

    if (my_active) {
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int x = 4; x < 32; x <<= 1) {
          l[j] = __fadd_rn(l[j], __shfl_xor_sync(FULL, l[j], x));
          beta[j] = __fadd_rn(beta[j], __shfl_xor_sync(FULL, beta[j], x));
        }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        if (!cok[j]) continue;
        float* pr = part + (size_t(lse_rh[j]) * splits + split) * (HD + 2);
        if (g == 0) {
          pr[0] = m[j];
          pr[1] = l[j];
        }
#pragma unroll
        for (int mt = 0; mt < KS; ++mt) {
          int w = 0, s = 0;
          L.v_at(g, mt, &w, &s);
          const float f = ldexpf(1.f, -vs[j] - BITS * FoldSlots<BITS>::sp(s));
#pragma unroll
          for (int h = 0; h < 2; ++h) pr[2 + 16 * mt + g + 8 * h] = __fmaf_rn(o[mt][2 * h + j], f, beta[j]);
        }
      }
    }
  }
}

// The sum of v over the threads before this one (blockDim.x a multiple of 32, at most 1024; `tmp` 32 ints).
__device__ int block_exclusive_sum(int v, int* tmp) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) tmp[warp] = x;
  __syncthreads();
  int before = x - v;
  for (int w = 0; w < warp; ++w) before += tmp[w];
  __syncthreads();
  return before;
}

// The pairs a row refines: StrataSel::groups(total) of its (total + 1) / 2 pairs, the largest scores first (the
// largest over the KV head's query heads of the pair's softmax weight), ties to the lower position; their indices in
// ascending order to sel[row][kvh][..], their number to sel_n. In two steps, so that a long context's
// pairs spread over many blocks: strata_select_part, grid (rows, nkv, chunks), the keys of a chunk of the pairs and
// its own best groups(total), ties to the lower position, to cand; strata_select, grid (rows, nkv), the best of the
// chunks' candidates.  Every pair the whole would pick is among its chunk's best, ties included (a chunk keeps its
// pairs above the k-th key and its ties in position order, up to k), so the pick is the one-step selection's.
constexpr int kSelThreads = 1024, kSelPartThreads = 512, kSelBins = 2048;
constexpr int kSelPartPer = kStrataSelChunk / kSelPartThreads;
static_assert(kSelPartPer % 8 == 0, "a selection thread's run: whole loads of 8 log-sum-exps");

// The `want`-th largest of the block's keys, 11 + 11 + 10 bits at a time (each(f) calls f(key) for every key the
// thread holds): the key, and in *ties how many keys equal to it are among the `want` largest.  Every thread of the
// block calls it.
template <int THREADS, class Each>
__device__ uint32_t block_kth(int want, Each each, int* ties) {
  __shared__ int hist[kSelBins];
  __shared__ int s_digit, s_want;
  __shared__ int s_warp[THREADS / 32][2];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint32_t prefix = 0, mask = 0;
  const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
  for (int pass = 0; pass < 3; ++pass) {
    const int shift = shifts[pass];
    const uint32_t dmask = (1u << widths[pass]) - 1u;
    for (int b = tid; b < kSelBins; b += THREADS) hist[b] = 0;
    __syncthreads();
    each([&](uint32_t k) {
      if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & dmask], 1);
    });
    __syncthreads();
    {  // the digit where the count from the top reaches `want`: each thread PER bins, the threads' sums from the top
      constexpr int PER = kSelBins / THREADS;
      const int bins = int(dmask) + 1, b0 = tid * PER;
      int mine = 0;
      for (int b = 0; b < PER; ++b) mine += b0 + b < bins ? hist[b0 + b] : 0;
      int above = 0;  // the bins of the threads after this one: a suffix sum, warp by warp
      int x = mine;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_down_sync(0xffffffffu, x, o);
        if (lane + o < 32) x += y;
      }
      above = x - mine;  // within the warp
      if (lane == 0) s_warp[warp][0] = x;  // the warp's total
      __syncthreads();
      for (int w = warp + 1; w < THREADS / 32; ++w) above += s_warp[w][0];
      if (above < want && above + mine >= want) {
        int cum = above, d = b0 + PER - 1;
        for (; d > b0; --d) {
          const int h = d < bins ? hist[d] : 0;
          if (cum + h >= want) break;
          cum += h;
        }
        s_digit = d;
        s_want = want - cum;
      }
      __syncthreads();
    }
    prefix |= uint32_t(s_digit) << shift;
    mask |= dmask << shift;
    want = s_want;
    __syncthreads();
  }
  *ties = want;
  return prefix;
}

// The `want` largest of n keys key(i), i in [0, n), ties to the lower i: emit(slot, i) for each, slots 0 .. want - 1
// in ascending order of i.  The k-th largest key (block_kth), then the picks in order: each thread a run of i, its
// counts summed over the runs before it.  Every thread of the block calls it.
template <class Key, class Emit>
__device__ void block_top(int n, int want_all, Key key, Emit emit) {
  __shared__ int s_base[32];
  const int tid = threadIdx.x;
  int want = 0;
  const uint32_t prefix = block_kth<kSelThreads>(
      want_all, [&](auto f) {
        for (int i = tid; i < n; i += kSelThreads) f(key(i));
      },
      &want);
  const int per = (n + kSelThreads - 1) / kSelThreads, i0 = min(n, tid * per), i1 = min(n, i0 + per);
  int n_above = 0, n_tie = 0;
  for (int i = i0; i < i1; ++i) {
    const uint32_t k = key(i);
    n_above += k > prefix;
    n_tie += k == prefix;
  }
  const int tie_before = block_exclusive_sum(n_tie, s_base);
  const int ties = min(n_tie, max(0, want - tie_before));
  int at = block_exclusive_sum(n_above + ties, s_base);
  for (int i = i0, t = tie_before; i < i1; ++i) {
    const uint32_t k = key(i);
    if (k > prefix) emit(at++, i);
    else if (k == prefix && t++ < want) emit(at++, i);
  }
}

// block_top over keys the threads hold in registers: thread t keys[0 .. nk) for its run of indices, the runs in
// thread order; emit(slot, j, key) for the thread's j-th.  The same picks as block_top.
template <int THREADS, int PER, class Emit>
__device__ void block_top_regs(int want_all, const uint32_t (&keys)[PER], int nk, Emit emit) {
  __shared__ int s_base[32];
  int want = 0;
  const uint32_t prefix = block_kth<THREADS>(
      want_all, [&](auto f) {
#pragma unroll
        for (int j = 0; j < PER; ++j)
          if (j < nk) f(keys[j]);
      },
      &want);
  int n_above = 0, n_tie = 0;
#pragma unroll
  for (int j = 0; j < PER; ++j)
    if (j < nk) n_above += keys[j] > prefix, n_tie += keys[j] == prefix;
  const int tie_before = block_exclusive_sum(n_tie, s_base);
  const int ties = min(n_tie, max(0, want - tie_before));
  int at = block_exclusive_sum(n_above + ties, s_base), tc = tie_before;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    if (j >= nk) continue;
    if (keys[j] > prefix) emit(at++, j, keys[j]);
    else if (keys[j] == prefix && tc++ < want) emit(at++, j, keys[j]);
  }
}

// A chunk of the row's pairs: [c0, c1) of ng, chunks of equal size (a multiple of 16 pairs: whole loads of 8).
__device__ __forceinline__ void sel_chunk(int ng, int chunk, int chunks, int* c0, int* c1) {
  const int per = ((ng + chunks - 1) / chunks + 15) / 16 * 16;
  *c0 = min(ng, chunk * per);
  *c1 = min(ng, *c0 + per);
}

// 512 threads (2 blocks/SM on sm_120 -> 140 blocks/wave, fitting 200k x 4 rows in 1 wave) with warp-parallel head_mass.
__global__ void __launch_bounds__(kSelPartThreads, 2)
    k_strata_select_part(const float* part, int nh, int nkv, int G, int hd, int slots, int used_cap, const __half* lse,
                         int lse_stride, uint32_t* cand_key, int* cand, int* cand_n, int sel_stride,
                         const DevParams* P, StrataSel how) {
  pdl_trigger();
  __shared__ float s_c[16];
  const int r = blockIdx.x, kvh = blockIdx.y, chunk = blockIdx.z, chunks = gridDim.z, tid = threadIdx.x;
  const int total = P->position + r + 1, ng = (total + 1) / 2, k = how.groups(total);
  const size_t rc = (size_t(r) * nkv + kvh) * chunks + chunk;
  if (k == 0 || k >= ng) return;  // none, or every pair: strata_select decides alone
  int c0, c1;
  sel_chunk(ng, chunk, chunks, &c0, &c1);
  const int used = attn_split_used(total, used_cap);
  const int warp = tid >> 5, lane = tid & 31;
  if (warp < G) {  // 1 warp per query head (kSelPartThreads / 32 = 16 >= G)
    const float* pr = part + (size_t(r) * nh + kvh * G + warp) * slots * (hd + 2);
    float M = -INFINITY;
    for (int s = lane; s < used; s += 32) M = fmaxf(M, pr[s * (hd + 2)]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float L = 0.f;
    if (M != -INFINITY)
      for (int s = lane; s < used; s += 32) {
        const float pm = pr[s * (hd + 2)];
        if (pm != -INFINITY) L += pr[s * (hd + 2) + 1] * __expf(pm - M);
      }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    if (lane == 0) s_c[warp] = M + __logf(L);  // log of the head's softmax denominator
  }
  __syncthreads();
  // The thread's run of the chunk's pairs (a multiple of 8, in thread order), their keys in registers: the log-sum-exps
  // read once, 8 a load (the rows' strides and the chunks are multiples of 8 and 16; a load past ng stays in the row).
  const int n = c1 - c0, run = ((n + kSelPartThreads - 1) / kSelPartThreads + 7) / 8 * 8;
  if (n <= 0) {  // a chunk past the row's pairs (chunks are whole 16s)
    if (tid == 0) cand_n[rc] = 0;
    return;
  }
  const int i0 = min(n, tid * run), nk = min(n, i0 + run) - i0;
  const int recent_from = ng - how.recent_pairs(total);  // these first, whatever their scores
  float s[kSelPartPer];
#pragma unroll
  for (int j = 0; j < kSelPartPer; ++j) s[j] = -INFINITY;
#pragma unroll
  for (int g = 0; g < 8; ++g) {  // every load in flight at once (G is at most 8)
    if (g >= G) break;
    const __half* lg = lse + (size_t(r) * nh + kvh * G + g) * lse_stride + c0 + i0;
    const float c = s_c[g];
#pragma unroll
    for (int j8 = 0; j8 < kSelPartPer; j8 += 8) {
      if (j8 >= nk) break;
      const uint4 x = __ldg(reinterpret_cast<const uint4*>(lg + j8));
      const uint32_t w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const __half2 h2 = *reinterpret_cast<const __half2*>(&w[e >> 1]);
        const float v = (e & 1) ? __high2float(h2) : __low2float(h2);
        s[j8 + e] = fmaxf(s[j8 + e], v - c);
      }
    }
  }
  uint32_t key[kSelPartPer];
#pragma unroll
  for (int j = 0; j < kSelPartPer; ++j) key[j] = c0 + i0 + j >= recent_from ? 0xffffffffu : float_order_u32(s[j]);
  const int want = min(k, n);
  int* out = cand + rc * sel_stride;
  uint32_t* out_key = cand_key + rc * sel_stride;
  block_top_regs<kSelPartThreads>(want, key, nk, [&](int at, int j, uint32_t kk) {
    out[at] = c0 + i0 + j;
    out_key[at] = kk;
  });
  if (tid == 0) cand_n[rc] = want;
}

__global__ void __launch_bounds__(kSelThreads)
    k_strata_select(int nkv, const uint32_t* cand_key, const int* cand, const int* cand_n, int chunks, int* gathered,
                    uint32_t* gathered_key, int* sel, int sel_stride, int* sel_n, const DevParams* P, StrataSel how) {
  pdl_trigger();
  __shared__ int s_off[kStrataSelChunks + 1];
  const int r = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x;
  const int total = P->position + r + 1, ng = (total + 1) / 2, k = how.groups(total);
  int* out = sel + (size_t(r) * nkv + kvh) * sel_stride;
  if (k == 0) {  // none: the copy alone
    if (tid == 0) sel_n[size_t(r) * nkv + kvh] = 0;
    return;
  }
  if (k >= ng) {  // every pair: the result is the cache's own attention
    for (int i = tid; i < ng; i += kSelThreads) out[i] = i;
    if (tid == 0) sel_n[size_t(r) * nkv + kvh] = ng;
    return;
  }
  // the chunks' candidates, in chunk order (so in position order), into one list
  const size_t rc0 = (size_t(r) * nkv + kvh) * chunks;
  if (tid == 0) {
    s_off[0] = 0;
    for (int c = 0; c < chunks; ++c) s_off[c + 1] = s_off[c] + cand_n[rc0 + c];
  }
  __syncthreads();
  int* list = gathered + rc0 * sel_stride;
  uint32_t* lk = gathered_key + rc0 * sel_stride;
  for (int c = 0; c < chunks; ++c)
    for (int i = tid; i < s_off[c + 1] - s_off[c]; i += kSelThreads) {
      list[s_off[c] + i] = cand[(rc0 + c) * sel_stride + i];
      lk[s_off[c] + i] = cand_key[(rc0 + c) * sel_stride + i];
    }
  __syncthreads();
  block_top(s_off[chunks], k, [&](int i) { return lk[i]; }, [&](int at, int i) { out[at] = list[i]; });
  if (tid == 0) sel_n[size_t(r) * nkv + kvh] = k;
}

// Recent pairs share the highest priority. When they fill the quota, the radix selector emits this range in order.
__global__ void k_strata_select_recent(const DevParams* P, int pairs, int* sel, int stride, int* sel_n) {
  pdl_trigger();
  const int row = blockIdx.x, kvh = blockIdx.y, ng = StrataSel::pairs(P->position + row + 1), n = min(ng, pairs);
  const size_t at = size_t(row) * gridDim.y + kvh;
  for (int i = threadIdx.x; i < n; i += blockDim.x) sel[at * stride + i] = ng - n + i;
  if (threadIdx.x == 0) sel_n[at] = n;
}

// The pairs a row refines that are in host memory, kept on the device for the rounds after, in two
// steps after the selection.  A pair qualifies when its second position is in host memory and both were written
// before this round (positions a round writes, or a rewind rewrites, are never kept: set_params forgets them all).
// strata_claim: a qualifying pair its slot does not hold claims it (the smallest pair of the round claiming a slot
// wins, whatever the order); grid (rows, nkv), block 256.  strata_fill: each winner's bytes into its slot, then its
// tag; grid (rows, nkv, kStrataSlices) as the correction (the bus wants many reads in flight), block 256, a warp a
// pair.
__device__ __forceinline__ bool hot_pair(const KvRef& kv, int q, int written) {
  return 2 * q + 1 >= kv.host_from && 2 * q + 1 < written;
}
// The lower pair wins. If several rows select it, only the lowest row needs to fetch its bytes over the bus.
static_assert(size_t(kStrataSelChunk) * kStrataSelChunks * FLYX_MAX_ROWS < kHotFree);
__device__ __forceinline__ int hot_claim_key(int q) { return q * int(gridDim.x) + int(blockIdx.x); }

__global__ void __launch_bounds__(256) k_strata_claim(KvRef kv, int nkv, const int* __restrict__ sel, int sel_stride,
                                                      const int* __restrict__ sel_n, const DevParams* P) {
  pdl_trigger();
  const int r = blockIdx.x, kvh = blockIdx.y, n = sel_n[size_t(r) * nkv + kvh];
  const int* list = sel + (size_t(r) * nkv + kvh) * sel_stride;
  const int32_t* tag = kv.hot_tag + (size_t(kvh) << kv.hot_bits);
  int32_t* claim = kv.hot_claim + (size_t(kvh) << kv.hot_bits);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int q = list[i];
    if (!hot_pair(kv, q, P->position)) continue;
    const int sl = kv.hot_slot(q);
    if (tag[sl] != q) atomicMin(&claim[sl], hot_claim_key(q));
  }
}
template <int HD>
__global__ void __launch_bounds__(256) k_strata_fill(KvRef kv, int nkv, const int* __restrict__ sel, int sel_stride,
                                                     const int* __restrict__ sel_n, const DevParams* P) {
  pdl_trigger();
  const int r = blockIdx.x, kvh = blockIdx.y, n = sel_n[size_t(r) * nkv + kvh];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int per = (n + gridDim.z - 1) / gridDim.z, i0 = min(n, int(blockIdx.z) * per), i1 = min(n, i0 + per);
  const int* list = sel + (size_t(r) * nkv + kvh) * sel_stride;
  int32_t* tag = kv.hot_tag + (size_t(kvh) << kv.hot_bits);
  int32_t* claim = kv.hot_claim + (size_t(kvh) << kv.hot_bits);
  for (int i = i0 + warp; i < i1; i += 8) {
    const int q = list[i];
    if (!hot_pair(kv, q, P->position)) continue;
    const int sl = kv.hot_slot(q);
    if (tag[sl] == q || claim[sl] != hot_claim_key(q)) continue;
    uint8_t* dst = kv.hot + ((size_t(kvh) << kv.hot_bits) + sl) * 4 * HD;  // [K, V][2 positions][HD]
    for (int x = lane; x < HD / 2; x += 32) {  // 4 HD bytes, 8 a lane
      const int part = x / (HD / 8), c = x % (HD / 8);  // part: K of 2q, K of 2q + 1, V of 2q, V of 2q + 1
      const uint8_t* src = static_cast<const uint8_t*>(part < 2 ? kv.k : kv.v) +
                           (size_t(2 * q + (part & 1)) * nkv + kvh) * HD + 8 * c;
      *reinterpret_cast<uint2*>(dst + part * HD + 8 * c) = *reinterpret_cast<const uint2*>(src);
    }
    if (kv.hot_scale && lane < 4) {  // the pair's scales with its bytes
      const int pos = 2 * q + (lane & 1);
      kv.hot_scale[(((size_t(kvh) << kv.hot_bits) + sl) << 2) + lane] =
          (lane < 2 ? kv.kscales(pos) : kv.vscales(pos))[size_t(pos) * nkv + kvh];
    }
    __syncwarp();
    if (lane == 0) {
      tag[sl] = q;
      claim[sl] = kHotFree;
    }
  }
}

// Pageable host K/V gather pipeline for Strata attention (Windows WDDM overflow):
// Cold pairs [host_from, ring_lo) live in pageable RAM; positions < host_from and >= ring_lo live on GPU (slot_of).
// 1) k_strata_page_claim claims hot-cache slots across rows (deduplicating pairs shared across rows).
// 2) k_strata_page_list emits unique cold misses into mapped pinned PageGatherReq[] (and bounce indices for collisions).
// 3) strata_page_gather_host (captured as a CUDA graph host node via cudaLaunchHostFunc) copies cold pairs from
//    pageable RAM into mapped pinned bounce_kv/bounce_scale.
// 4) k_strata_page_install installs winning misses into VRAM kv.hot/kv.hot_scale and clears hot_claim.
struct PageBounce {
  const uint8_t* bounce_kv = nullptr;
  const float* bounce_scale = nullptr;
  const int32_t* miss_map = nullptr;
};

__device__ __forceinline__ int page_ring_lo(const KvRef& kv, int pos) {
  const int lo = pos > kv.ring_safe ? ((pos - kv.ring_safe) & ~1) : 0;
  return max(kv.host_from, lo);
}

__global__ void __launch_bounds__(256)
    k_strata_page_claim(KvRef kv, int nkv, const int* __restrict__ sel, int sel_stride, const int* __restrict__ sel_n,
                        const DevParams* P, int32_t* miss_count) {
  pdl_trigger();
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) *miss_count = 0;
  if (!kv.hot) return;
  const int r = blockIdx.x, kvh = blockIdx.y, n = sel_n[size_t(r) * nkv + kvh];
  const int ring_lo = page_ring_lo(kv, P->position);
  const int* list = sel + (size_t(r) * nkv + kvh) * sel_stride;
  const int32_t* tag = kv.hot_tag + (size_t(kvh) << kv.hot_bits);
  int32_t* claim = kv.hot_claim + (size_t(kvh) << kv.hot_bits);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int q = list[i];
    if (2 * q < kv.host_from || 2 * q + 1 >= ring_lo) continue;
    const int sl = kv.hot_slot(q);
    // Pin already-resident pairs (-1 < every non-negative claim key) so a colliding miss in the same
    // round cannot evict a hit whose miss_map entry is -1.
    atomicMin(&claim[sl], tag[sl] == q ? -1 : hot_claim_key(q));
  }
}

__global__ void __launch_bounds__(256)
    k_strata_page_list(KvRef kv, int nkv, const int* __restrict__ sel, int sel_stride, const int* __restrict__ sel_n,
                       const DevParams* P, PageGatherReq* req_dev, int32_t* miss_count, int32_t* miss_map) {
  pdl_trigger();
  const int r = blockIdx.x, kvh = blockIdx.y, rows = gridDim.x;
  const int n = sel_n[size_t(r) * nkv + kvh];
  const int ring_lo = page_ring_lo(kv, P->position);
  const int* list = sel + (size_t(r) * nkv + kvh) * sel_stride;
  int32_t* map_row = miss_map + (size_t(r) * nkv + kvh) * sel_stride;
  const int32_t* tag = kv.hot ? kv.hot_tag + (size_t(kvh) << kv.hot_bits) : nullptr;
  const int32_t* claim = kv.hot ? kv.hot_claim + (size_t(kvh) << kv.hot_bits) : nullptr;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int q = list[i];
    if (2 * q < kv.host_from || 2 * q + 1 >= ring_lo) {
      map_row[i] = -1;
      continue;
    }
    if (kv.hot) {
      const int sl = kv.hot_slot(q);
      if (tag[sl] == q) {
        map_row[i] = -1;
        continue;
      }
      const int win = claim[sl];
      if (win >= 0) {
        if (win == q * rows + r) {
          const int m = atomicAdd(miss_count, 1);
          req_dev[m] = PageGatherReq{kvh, q, sl, 0};
          map_row[i] = -1;
          continue;
        }
        if (win / rows == q) {
          map_row[i] = -1;
          continue;
        }
      }
    }
    const int m = atomicAdd(miss_count, 1);
    req_dev[m] = PageGatherReq{kvh, q, -1, 0};
    map_row[i] = m;
  }
}

void CUDART_CB strata_page_gather_host(void* ud) {
  const auto* c = static_cast<const PageGatherCtx*>(ud);
  const int n = *c->miss_count_host;
  if (n <= 0) return;
  const int lim = n < c->max_misses ? n : c->max_misses;
  const size_t head_bytes = size_t(c->hd) * size_t(c->elem);
  const size_t pos_stride = size_t(c->nkv) * head_bytes;
  for (int m = 0; m < lim; ++m) {
    const PageGatherReq req = c->req_host[m];
    const int p0 = 2 * req.q;
    const int p1 = p0 + 1;
    uint8_t* dst = c->bounce_kv_host + size_t(m) * 4 * head_bytes;
    const size_t h_off = size_t(req.kvh) * head_bytes;
    std::memcpy(dst, c->k_page + size_t(p0) * pos_stride + h_off, head_bytes);
    if (p1 < c->cap) std::memcpy(dst + head_bytes, c->k_page + size_t(p1) * pos_stride + h_off, head_bytes);
    else std::memset(dst + head_bytes, 0, head_bytes);
    std::memcpy(dst + 2 * head_bytes, c->v_page + size_t(p0) * pos_stride + h_off, head_bytes);
    if (p1 < c->cap) std::memcpy(dst + 3 * head_bytes, c->v_page + size_t(p1) * pos_stride + h_off, head_bytes);
    else std::memset(dst + 3 * head_bytes, 0, head_bytes);
    if (c->ks_page) {
      float* sd = c->bounce_scale_host + size_t(m) * 4;
      sd[0] = c->ks_page[size_t(p0) * c->nkv + req.kvh];
      sd[1] = p1 < c->cap ? c->ks_page[size_t(p1) * c->nkv + req.kvh] : 0.f;
      sd[2] = c->vs_page[size_t(p0) * c->nkv + req.kvh];
      sd[3] = p1 < c->cap ? c->vs_page[size_t(p1) * c->nkv + req.kvh] : 0.f;
    }
  }
}

template <int HD, int Elem>
__global__ void __launch_bounds__(256)
    k_strata_page_install(KvRef kv, int nkv, const PageGatherReq* __restrict__ req_dev,
                          const int32_t* __restrict__ miss_count, const uint8_t* __restrict__ bounce_kv,
                          const float* __restrict__ bounce_scale) {
  pdl_trigger();
  if (!kv.hot) return;
  const size_t hot_total = size_t(nkv) << kv.hot_bits;
  const size_t tid = size_t(blockIdx.x) * size_t(blockDim.x) + size_t(threadIdx.x);
  const size_t stride = size_t(gridDim.x) * size_t(blockDim.x);
  for (size_t i = tid; i < hot_total; i += stride) kv.hot_claim[i] = kHotFree;
  const int n = *miss_count;
  if (n <= 0) return;
  const int warp = int(tid >> 5);
  const int total_warps = int(stride >> 5);
  const int lane = int(threadIdx.x) & 31;
  constexpr size_t pair_bytes = 4 * size_t(HD) * size_t(Elem);
  constexpr size_t pair_u32 = pair_bytes / 4;
  for (int m = warp; m < n; m += total_warps) {
    const PageGatherReq req = req_dev[m];
    if (req.hot_slot < 0) continue;
    const size_t hidx = (size_t(req.kvh) << kv.hot_bits) + size_t(req.hot_slot);
    uint32_t* dst = reinterpret_cast<uint32_t*>(kv.hot + hidx * pair_bytes);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(bounce_kv + size_t(m) * pair_bytes);
    for (size_t x = lane; x < pair_u32; x += 32) dst[x] = src[x];
    if constexpr (Elem == 1) {
      if (kv.hot_scale && lane < 4) kv.hot_scale[(hidx << 2) + lane] = bounce_scale[size_t(m) * 4 + lane];
    }
    __syncwarp();
    if (lane == 0) kv.hot_tag[hidx] = req.q;
  }
}

// The correction (StrataFlux's equation 8), in two steps.  strata_delta: a slice of a row's selected positions, their
// copy's contributions (rotated basis) and the cache's own (unrotated), and the change to the softmax's sum, per query
// head; grid (rows, nkv, kStrataSlices), block 32 * kDeltaWarps (warps in order).  strata_finish: the copy's attention
// combined, the slices' changes in order, rotated back, normalized, gated; grid (rows, nh), block HD.
constexpr int kDeltaWarps = 2;

template <int HD, int G, int BITS, bool FP8, bool Compact = false>
__global__ void __launch_bounds__(32 * kDeltaWarps, 12 / kDeltaWarps)
    k_strata_delta(const float* __restrict__ q, const float* __restrict__ qr, KvRef kv, int nh, int nkv, float scale,
                   const DevParams* P, const DevParams* P_hot,
                   const float* __restrict__ part, int slots, int used_cap, const int* __restrict__ sel,
                   int sel_stride, const int* __restrict__ sel_n, float* __restrict__ delta, PageBounce pb = {}) {
  pdl_trigger();
  constexpr int DPL = HD / 32, NW = kDeltaWarps;
  constexpr StrataLayout SL(HD, BITS);
  constexpr int KW_POS = Compact ? 24 : SL.kwords();
  constexpr int VW_POS = Compact ? 24 : SL.vwords();
  constexpr bool COMB_KV = (KW_POS + VW_POS <= 32);
  constexpr uint32_t FULL = 0xffffffffu;
  // Transposed [G][DPL][33] eliminates 8-way bank conflicts; unioning Q/Qr with F/C halves SMEM (12.7 KB for HD=256, G=6).
  __shared__ union {
    struct {
      float q[G][DPL][33];
      float qr[G][DPL][33];
    } q;
    struct {
      float f[G][DPL][33];
      float c[G][DPL][33];
    } acc;
  } sm;
  __shared__ float s_M[G], s_dl[NW][G];
  const int r = blockIdx.x, kvh = blockIdx.y, slice = blockIdx.z, slices = gridDim.z;
  const int tid = int(threadIdx.x), warp = tid >> 5, lane = tid & 31;
  const int total = P->position + r + 1, used = attn_split_used(total, used_cap);

  // Warp-parallel M-only reduction across splits (0 __expf, bit-identical to head_mass's M).
  for (int g = warp; g < G; g += NW) {
    const float* pr = part + (size_t(r) * nh + kvh * G + g) * slots * (HD + 2);
    float M = -INFINITY;
    for (int s = lane; s < used; s += 32) M = fmaxf(M, pr[s * (HD + 2)]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(FULL, M, o));
    if (lane == 0) s_M[g] = M;
  }
  for (int i = tid; i < G * HD; i += 32 * NW) {
    const int g = i / HD, d = i - g * HD;
    const size_t qoff = (size_t(r) * nh + kvh * G + g) * HD + d;
    sm.q.q[g][d & (DPL - 1)][d / DPL] = __ldg(q + qoff);
    sm.q.qr[g][d & (DPL - 1)][d / DPL] = __ldg(qr + qoff);
  }
  __syncthreads();

  float accF[G][DPL], accC[G][DPL], dl[G];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    dl[g] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) accF[g][e] = accC[g][e] = 0.f;
  }
  const int n2 = 2 * sel_n[size_t(r) * nkv + kvh], per = (n2 + slices - 1) / slices;
  const int p0 = min(n2, slice * per), p1 = min(n2, p0 + per);
  const int* list = sel + (size_t(r) * nkv + kvh) * sel_stride;
  for (int p = p0 + warp; p < p1; p += NW) {
    const int j = 2 * list[p >> 1] + (p & 1);
    if (j >= total) continue;  // uniform across the warp
    const int cj = kv.slot_of(j);  // the exact values' slot (a ring's); the copy's is j's
    const size_t at = (size_t(cj) * nkv + kvh) * HD + lane * DPL;

    // Issue all exact-KV and low-bit-KV loads up front so DRAM / PCIe latencies overlap.
    float kscale = 1.f, vscale = 1.f;
    uint2 kx8 = {0, 0}, vx8 = {0, 0};
    uint4 kx16 = {0, 0, 0, 0}, vx16 = {0, 0, 0, 0};
    float kf_small[DPL], vf_small[DPL];
    constexpr int kElem = FP8 ? 1 : 2;
    const uint8_t* ksrc = static_cast<const uint8_t*>(kv.k) + at * kElem;
    const uint8_t* vsrc = static_cast<const uint8_t*>(kv.v) + at * kElem;
    const float *kscale_at = nullptr, *vscale_at = nullptr;
    if (pb.bounce_kv) {
      const int q_pair = j >> 1;
      const int ring_lo = page_ring_lo(kv, P_hot->position);
      if (2 * q_pair >= kv.host_from && 2 * q_pair + 1 < ring_lo) {
        const int sl = kv.hot ? kv.hot_slot(q_pair) : -1;
        if (kv.hot && kv.hot_tag[(size_t(kvh) << kv.hot_bits) + sl] == q_pair) {
          const uint8_t* held = kv.hot + ((size_t(kvh) << kv.hot_bits) + sl) * 4 * HD * kElem + lane * DPL * kElem;
          ksrc = held + (j & 1) * HD * kElem;
          vsrc = held + (2 + (j & 1)) * HD * kElem;
          if constexpr (FP8) {
            if (kv.hot_scale) {
              kscale_at = kv.hot_scale + ((((size_t(kvh) << kv.hot_bits) + sl) << 2) + (j & 1));
              vscale_at = kscale_at + 2;
            }
          }
        } else {
          const int m = pb.miss_map[(size_t(r) * nkv + kvh) * sel_stride + (p >> 1)];
          const uint8_t* held = pb.bounce_kv + size_t(m) * 4 * HD * kElem + lane * DPL * kElem;
          ksrc = held + (j & 1) * HD * kElem;
          vsrc = held + (2 + (j & 1)) * HD * kElem;
          if constexpr (FP8) {
            kscale_at = pb.bounce_scale + size_t(m) * 4 + (j & 1);
            vscale_at = kscale_at + 2;
          }
        }
      }
    } else if constexpr (FP8 && DPL == 8) {
      if (kv.hot) {
        const int q_pair = j >> 1, sl = kv.hot_slot(q_pair);
        if (hot_pair(kv, q_pair, P_hot->position) && kv.hot_tag[(size_t(kvh) << kv.hot_bits) + sl] == q_pair) {
          const uint8_t* held = kv.hot + ((size_t(kvh) << kv.hot_bits) + sl) * 4 * HD + lane * DPL;
          ksrc = held + (j & 1) * HD;
          vsrc = held + (2 + (j & 1)) * HD;
          if (kv.hot_scale) {
            kscale_at = kv.hot_scale + ((((size_t(kvh) << kv.hot_bits) + sl) << 2) + (j & 1));
            vscale_at = kscale_at + 2;
          }
        }
      }
    }
    if constexpr (FP8 && DPL == 8) {
      kx8 = *reinterpret_cast<const uint2*>(ksrc);
      vx8 = *reinterpret_cast<const uint2*>(vsrc);
      kscale = kscale_at ? *kscale_at : kv.kscales(cj)[size_t(cj) * nkv + kvh];
      vscale = vscale_at ? *vscale_at : kv.vscales(cj)[size_t(cj) * nkv + kvh];
    } else if constexpr (FP8) {
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf_small[e] = e4m3_to_float(ksrc[e]);
        vf_small[e] = e4m3_to_float(vsrc[e]);
      }
      kscale = kscale_at ? *kscale_at : kv.kscales(cj)[size_t(cj) * nkv + kvh];
      vscale = vscale_at ? *vscale_at : kv.vscales(cj)[size_t(cj) * nkv + kvh];
    } else if constexpr (DPL == 8) {
      kx16 = *reinterpret_cast<const uint4*>(ksrc);
      vx16 = *reinterpret_cast<const uint4*>(vsrc);
    } else {
      const kv16* k16 = reinterpret_cast<const kv16*>(ksrc);
      const kv16* v16 = reinterpret_cast<const kv16*>(vsrc);
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf_small[e] = __half2float(k16[e]);
        vf_small[e] = __half2float(v16[e]);
      }
    }

    // Cooperative warp load of the position's low-bit K/V words (1 __ldg/lane for 2-bit, <=2 for 3/4-bit).
    const size_t at16 = strata_words_at(j, kvh, nkv, kv.words);
    const int p16 = j & 15;
    uint32_t kword = 0u, vword = 0u;
    if constexpr (COMB_KV) {
      if (lane < KW_POS) kword = __ldg(kv.kq + at16 + SL.k_index(p16, lane));
      else if (lane < KW_POS + VW_POS) kword = __ldg(kv.vq + at16 + SL.v_index(p16, lane - KW_POS));
    } else {
      if (lane < KW_POS) kword = __ldg(kv.kq + at16 + SL.k_index(p16, lane));
      if (lane < VW_POS) vword = __ldg(kv.vq + at16 + SL.v_index(p16, lane));
    }
    const __half2 kqs_h = __ldg(kv.kqs + size_t(j) * nkv + kvh);
    const __half2 vqs_h = __ldg(kv.vqs + size_t(j) * nkv + kvh);

    // Phase 1: Exact K -> pf[G], then Exact V -> accF[G][DPL] (reusing one DPL temp array).
    float kv_tmp[DPL];
    if constexpr (FP8 && DPL == 8) {
      e4m3x4_to_float(kx8.x, kv_tmp);
      e4m3x4_to_float(kx8.y, kv_tmp + 4);
    } else if constexpr (!FP8 && DPL == 8) {
      const uint32_t kw4[4] = {kx16.x, kx16.y, kx16.z, kx16.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&kw4[e]));
        kv_tmp[2 * e] = a.x;
        kv_tmp[2 * e + 1] = a.y;
      }
    } else {
#pragma unroll
      for (int e = 0; e < DPL; ++e) kv_tmp[e] = kf_small[e];
    }
    float pf[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float df = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) df = __fmaf_rn(sm.q.q[g][e][lane], kv_tmp[e], df);
      df = warp_sum(df);
      pf[g] = __expf(__fmul_rn(__fmul_rn(df, kscale), scale) - s_M[g]);
    }
    if constexpr (FP8 && DPL == 8) {
      e4m3x4_to_float(vx8.x, kv_tmp);
      e4m3x4_to_float(vx8.y, kv_tmp + 4);
    } else if constexpr (!FP8 && DPL == 8) {
      const uint32_t vw4[4] = {vx16.x, vx16.y, vx16.z, vx16.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&vw4[e]));
        kv_tmp[2 * e] = b.x;
        kv_tmp[2 * e + 1] = b.y;
      }
    } else {
#pragma unroll
      for (int e = 0; e < DPL; ++e) kv_tmp[e] = vf_small[e];
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const float pv = __fmul_rn(pf[g], vscale);
#pragma unroll
      for (int e = 0; e < DPL; ++e) accF[g][e] = __fmaf_rn(pv, kv_tmp[e], accF[g][e]);
    }

    // Phase 2: Low-bit K -> pc[G] & dl[G], then Low-bit V -> accC[G][DPL] (reusing kv_tmp).
    {
      const float2 ksl = __half22float2(kqs_h);
      const float kbase = __fmaf_rn(0.5f, ksl.y, ksl.x);
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        const int d = lane * DPL + e;
        int kw_i = 0, ks = 0;
        SL.k_at((d >> 1) & 3, d >> 3, &kw_i, &ks);
        const uint32_t w = __shfl_sync(FULL, kword, Compact && d >= 240 ? 0 : kw_i);
        uint32_t code = (w >> ((d & 1) * 16 + ks * BITS)) & ((1u << BITS) - 1u);
        if constexpr (Compact) {
          const int grp = (d >> 1) & 3;
          const int first = d >= 240 ? (6 * grp + 3 * ((d >> 3) - 30)) : 0;
          const int sh = 15 + 16 * (d & 1);
          const uint32_t w0 = __shfl_sync(FULL, kword, first);
          const uint32_t w1 = __shfl_sync(FULL, kword, first + 1);
          const uint32_t w2 = __shfl_sync(FULL, kword, first + 2);
          if (d >= 240) code = ((w0 >> sh) & 1u) | (((w1 >> sh) & 1u) << 1) | (((w2 >> sh) & 1u) << 2);
        }
        kv_tmp[e] = __fmaf_rn(float(code), ksl.y, kbase);
      }
    }
    float pc[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      float dc = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) dc = __fmaf_rn(sm.q.qr[g][e][lane], kv_tmp[e], dc);
      dc = warp_sum(dc);
      pc[g] = __expf(__fmul_rn(dc, scale) - s_M[g]);
      dl[g] += pf[g] - pc[g];
    }
    {
      const float2 vsl = __half22float2(vqs_h);
      const float vbase = __fmaf_rn(0.5f, vsl.y, vsl.x);
      const uint32_t vsrc_w = COMB_KV ? kword : vword;
      constexpr int V_OFF = COMB_KV ? KW_POS : 0;
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        const int d = lane * DPL + e;
        int vw_i = 0, vs_s = 0;
        SL.v_at(d & 7, d >> 4, &vw_i, &vs_s);
        const uint32_t w = __shfl_sync(FULL, vsrc_w, V_OFF + (Compact && d >= 240 ? 0 : vw_i));
        uint32_t code = (w >> (((d >> 3) & 1) * 16 + vs_s * BITS)) & ((1u << BITS) - 1u);
        if constexpr (Compact) {
          const int first = 3 * (d & 7);
          const int sh = 15 + 16 * ((d >> 3) & 1);
          const uint32_t w0 = __shfl_sync(FULL, vsrc_w, first);
          const uint32_t w1 = __shfl_sync(FULL, vsrc_w, first + 1);
          const uint32_t w2 = __shfl_sync(FULL, vsrc_w, first + 2);
          if (d >= 240) code = ((w0 >> sh) & 1u) | (((w1 >> sh) & 1u) << 1) | (((w2 >> sh) & 1u) << 2);
        }
        kv_tmp[e] = __fmaf_rn(float(code), vsl.y, vbase);
      }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
      for (int e = 0; e < DPL; ++e) accC[g][e] = __fmaf_rn(pc[g], kv_tmp[e], accC[g][e]);
    }
  }
  __syncthreads();  // all warps done reading sm.q before sm.acc reuses the same shared memory
  for (int w = 0; w < NW; ++w) {  // the warps' sums in order
    if (warp == w) {
#pragma unroll
      for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int e = 0; e < DPL; ++e) {
          if (w == 0) {
            sm.acc.f[g][e][lane] = accF[g][e];
            sm.acc.c[g][e][lane] = accC[g][e];
          } else {
            sm.acc.f[g][e][lane] += accF[g][e];
            sm.acc.c[g][e][lane] += accC[g][e];
          }
        }
        if (lane == 0) s_dl[w][g] = dl[g];
      }
    }
    __syncthreads();
  }
  float* out = delta + ((size_t(r) * nkv + kvh) * slices + slice) * G * (2 * HD + 1);
  for (int i = tid; i < G * HD; i += 32 * NW) {
    const int g = i / HD, d = i - g * HD;
    out[g * (2 * HD + 1) + d] = sm.acc.f[g][d & (DPL - 1)][d / DPL];
    out[g * (2 * HD + 1) + HD + d] = sm.acc.c[g][d & (DPL - 1)][d / DPL];
  }
  if (tid < G) {
    float s = 0.f;
    for (int w = 0; w < NW; ++w) s += s_dl[w][tid];
    out[tid * (2 * HD + 1) + 2 * HD] = s;
  }
}

// grid (rows, nh), block HD: one query head's output, a thread a dimension (reads across the threads in a row).
template <int HD>
__global__ void __launch_bounds__(HD)
    k_strata_finish(int nh, int nkv, const DevParams* P, const float* __restrict__ part, int slots, int used_cap,
                    const float* __restrict__ delta, int slices, const float* gate, int ld_gate, Out out, int ld_out,
                    int from) {
  pdl_trigger();
  __shared__ float sA[HD];
  __shared__ float s_w[128];
  __shared__ float s_M, s_L;
  const int r = blockIdx.x, h = blockIdx.y, G = nh / nkv, kvh = h / G, g = h - kvh * G, d = threadIdx.x;
  if (P->position + r < from) return;  // exact attention's row
  const int total = P->position + r + 1, used = attn_split_used(total, used_cap);
  const float* dr = delta + (size_t(r) * nkv + kvh) * slices * G * (2 * HD + 1) + size_t(g) * (2 * HD + 1);
  const float* pr = part + (size_t(r) * nh + h) * slots * (HD + 2);
  if (d < 32) {
    float M = -INFINITY;
    for (int s = d; s < used; s += 32) M = fmaxf(M, pr[s * (HD + 2)]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    if (d == 0) s_M = M;
  }
  __syncthreads();
  const float M = s_M;
  if (M != -INFINITY) {
    for (int s = d; s < used && s < 128; s += HD) {
      const float pm = pr[s * (HD + 2)];
      s_w[s] = pm == -INFINITY ? 0.f : __expf(pm - M);
    }
  }
  __syncthreads();
  if (d == 0) {
    float L = 0.f;
    if (M != -INFINITY) {
      for (int s = 0; s < used; ++s) {
        const float w = s < 128 ? s_w[s] : (pr[s * (HD + 2)] == -INFINITY ? 0.f : __expf(pr[s * (HD + 2)] - M));
        if (w != 0.f) L += pr[s * (HD + 2) + 1] * w;
      }
    }
    for (int s = 0; s < slices; ++s) L += dr[size_t(s) * G * (2 * HD + 1) + 2 * HD];
    s_L = L;
  }
  float A = 0.f, f = 0.f;  // the copy's attention, as the combine; the slices' changes in order
  if (M != -INFINITY)
    for (int s = 0; s < used; ++s) {
      const float w = s < 128 ? s_w[s] : (pr[s * (HD + 2)] == -INFINITY ? 0.f : __expf(pr[s * (HD + 2)] - M));
      if (w != 0.f) A += pr[s * (HD + 2) + 2 + d] * w;
    }
  for (int s = 0; s < slices; ++s) {
    const float* ds = dr + size_t(s) * G * (2 * HD + 1);
    A -= ds[HD + d];
    f += ds[d];
  }
  sA[d] = A;
  fwht<HD>(sA, 1);  // back from the rotated basis (also synchronizes s_L before the divide)
  float o = s_L > 0.f ? (sA[d] + f) / s_L : 0.f;
  if (gate) o *= sigm(gate[size_t(r) * ld_gate + h * HD + d]);
  put(out, size_t(r) * ld_out + h * HD + d, size_t(r) * nh * HD + h * HD + d, o);
}

// k_strata_scan_mma_cta's shared-memory carveout, set once for each of its configurations.  A function of its own:
// MSVC (the host compiler under nvcc on Windows) takes a lambda's local constant for a member of the lambda that names
// it, never a constant there (C3493, C2664), so the static's initializer cannot sit inside launch_strata's lambda.
template <int HD, int BITS, int W, bool Compact>
void prefer_shared_scan_mma_cta() {
  static const bool set = [] {
    cudaFuncSetAttribute(k_strata_scan_mma_cta<HD, BITS, W, Compact>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return true;
  }();
  (void)set;
}

template <int HD, int G, int BITS, bool Compact = false>
void launch_strata(const float* q, KvRef kv, const StrataScratch& sc, int nh, int nkv, int rows, float scale,
                   const DevParams* P, const float* gate, int ld_gate, Out o, int ld_out, float* partial, int slots,
                   cudaStream_t st, bool cores, int from, cudaEvent_t second_ready, const DevParams* P_hot) {
  k_strata_rotate_q<HD><<<dim3(rows, nh), HD, 0, st>>>(q, sc.qr);
  auto wait_for_copy = [&] {
    const cudaError_t status = cudaStreamWaitEvent(st, second_ready, 0);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  };
  if (second_ready && (cores || slots < 2)) {
    wait_for_copy();
    second_ready = nullptr;
  }
  int splits;
  auto launch_scan_mma = [&](int offset, int count, bool subset) {
    const int total_warps = (rows * G + 7) / 8;
    const int fs = subset ? offset : 0;
    const int ts = subset ? splits : 0;
    if constexpr (BITS >= 3) {
      if (rows >= 2 && total_warps >= 2) {
        auto run_packed = [&](auto w_tag) {
          constexpr int W = decltype(w_tag)::value;
          prefer_shared_scan_mma_cta<HD, BITS, W, Compact>();
          const int bx = (total_warps + W - 1) / W;
          k_strata_scan_mma_cta<HD, BITS, W, Compact><<<dim3(bx, nkv, count), 32 * W, 0, st>>>(
              sc.qr, kv, nh, nkv, rows, scale, P, partial, sc.lse, sc.lse_stride, fs, ts);
        };
        if (total_warps == 2) run_packed(std::integral_constant<int, 2>{});
        else if (total_warps == 3 || total_warps % 3 == 0) run_packed(std::integral_constant<int, 3>{});
        else run_packed(std::integral_constant<int, 4>{});
        return;
      }
    }
    k_strata_scan_mma<HD, BITS, Compact><<<dim3(total_warps, nkv, count), 32, 0, st>>>(
        sc.qr, kv, nh, nkv, rows, scale, P, partial, sc.lse, sc.lse_stride, fs, ts);
  };
  if (cores) {  // the float32 scan (measurement): 64 splits measured slower than 32 (3.03 against 2.70 ms a step)
    splits = std::min(slots, kAttnCoresSplits);
    k_strata_scan<HD, G, BITS, Compact><<<dim3(rows, nkv, splits), 128, 0, st>>>(q, kv, nh, nkv, scale, P, partial, slots,
                                                                         sc.lse, sc.lse_stride);
  } else if (second_ready) {
    // Each block keeps the original split, key order and partial output slot. The first grid can read its
    // complete key ranges while DMA fills the remaining codes and grids.
    splits = slots;
    const int first = splits / 2;
    launch_scan_mma(0, first, true);
    wait_for_copy();
    launch_scan_mma(first, splits - first, true);
  } else {
    splits = slots;  // the partials' stride is the grid's splits
    launch_scan_mma(0, splits, false);
  }
  if (sc.lse) {
    k_strata_select_part<<<dim3(rows, nkv, sc.chunks), kSelPartThreads, 0, st>>>(
        partial, nh, nkv, G, HD, slots, splits, sc.lse, sc.lse_stride, sc.cand_key, sc.cand, sc.cand_n, sc.sel_stride, P,
        sc.how);
    k_strata_select<<<dim3(rows, nkv), kSelThreads, 0, st>>>(nkv, sc.cand_key, sc.cand, sc.cand_n, sc.chunks,
                                                              sc.gathered, sc.gathered_key, sc.sel, sc.sel_stride,
                                                              sc.sel_n, P, sc.how);
  } else {
    k_strata_select_recent<<<dim3(rows, nkv), 128, 0, st>>>(P, sc.how.most, sc.sel, sc.sel_stride, sc.sel_n);
  }
  PageBounce pb{};
  if (kv.page_ctx) {
    PageGatherCtx* pc = kv.page_ctx;
    k_strata_page_claim<<<dim3(rows, nkv), 256, 0, st>>>(kv, nkv, sc.sel, sc.sel_stride, sc.sel_n, P_hot,
                                                         pc->miss_count_dev);
    k_strata_page_list<<<dim3(rows, nkv), 256, 0, st>>>(kv, nkv, sc.sel, sc.sel_stride, sc.sel_n, P_hot, pc->req_dev,
                                                        pc->miss_count_dev, pc->miss_map);
    const cudaError_t host_err = cudaLaunchHostFunc(st, strata_page_gather_host, pc);
    if (host_err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(host_err));
    if (kv.hot) {
      if (kv.fp8())
        k_strata_page_install<HD, 1><<<kStrataSlices, 256, 0, st>>>(kv, nkv, pc->req_dev, pc->miss_count_dev,
                                                                    pc->bounce_kv_dev, pc->bounce_scale_dev);
      else
        k_strata_page_install<HD, 2><<<kStrataSlices, 256, 0, st>>>(kv, nkv, pc->req_dev, pc->miss_count_dev,
                                                                    pc->bounce_kv_dev, pc->bounce_scale_dev);
    }
    pb = {pc->bounce_kv_dev, pc->bounce_scale_dev, pc->miss_map};
  } else if (kv.hot && kv.fp8()) {
    k_strata_claim<<<dim3(rows, nkv), 256, 0, st>>>(kv, nkv, sc.sel, sc.sel_stride, sc.sel_n, P_hot);
    k_strata_fill<HD><<<dim3(rows, nkv, kStrataSlices), 256, 0, st>>>(kv, nkv, sc.sel, sc.sel_stride, sc.sel_n, P_hot);
  }
  const dim3 dg(rows, nkv, kStrataSlices);
  if (kv.fp8()) {
    static const bool delta_fp8_carveout = [] {
      cudaFuncSetAttribute(k_strata_delta<HD, G, BITS, true, Compact>,
                           cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
      return true;
    }();
    (void)delta_fp8_carveout;
    k_strata_delta<HD, G, BITS, true, Compact><<<dg, 32 * kDeltaWarps, 0, st>>>(
        q, sc.qr, kv, nh, nkv, scale, P, P_hot, partial, slots, splits, sc.sel, sc.sel_stride, sc.sel_n, sc.delta, pb);
  } else {
    static const bool delta_f16_carveout = [] {
      cudaFuncSetAttribute(k_strata_delta<HD, G, BITS, false, Compact>,
                           cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
      return true;
    }();
    (void)delta_f16_carveout;
    k_strata_delta<HD, G, BITS, false, Compact><<<dg, 32 * kDeltaWarps, 0, st>>>(
        q, sc.qr, kv, nh, nkv, scale, P, P_hot, partial, slots, splits, sc.sel, sc.sel_stride, sc.sel_n, sc.delta, pb);
  }
  k_strata_finish<HD><<<dim3(rows, nh), HD, 0, st>>>(nh, nkv, P, partial, slots, splits, sc.delta, kStrataSlices,
                                                      gate, ld_gate, o, ld_out, from);
}

template <int HD, int G>
bool by_bits(const float* q, KvRef kv, const StrataScratch& sc, int nh, int nkv, int rows, float scale,
             const DevParams* P, const float* gate, int ld_gate, Out o, int ld_out, float* partial, int slots,
             cudaStream_t st, bool cores, int from, cudaEvent_t second_ready, const DevParams* P_hot) {
  switch (kv.bits) {
    case 2: launch_strata<HD, G, 2>(q, kv, sc, nh, nkv, rows, scale, P, gate, ld_gate, o, ld_out, partial, slots, st,
                                    cores, from, second_ready, P_hot);
      return true;
    case 3:
      if constexpr (HD == 256) if (kv.words == 24) {
        launch_strata<HD, G, 3, true>(q, kv, sc, nh, nkv, rows, scale, P, gate, ld_gate, o, ld_out, partial, slots,
                                      st, cores, from, second_ready, P_hot);
        return true;
      }
      launch_strata<HD, G, 3>(q, kv, sc, nh, nkv, rows, scale, P, gate, ld_gate, o, ld_out, partial, slots, st,
                                    cores, from, second_ready, P_hot);
      return true;
    case 4: launch_strata<HD, G, 4>(q, kv, sc, nh, nkv, rows, scale, P, gate, ld_gate, o, ld_out, partial, slots, st,
                                    cores, from, second_ready, P_hot);
      return true;
    default: return false;
  }
}

}  // namespace

bool strata_supported(int nh, int nkv, int hd, int bits) {
  if (nkv <= 0 || nh % nkv != 0 || bits < 2 || bits > 4) return false;
  const int g = nh / nkv;
  if (hd == 256) return g == 1 || g == 2 || g == 4 || g == 6 || g == 8;
  if (hd == 128) return g == 1 || g == 2 || g == 4;
  if (hd == 64 || hd == 32) return g == 1 || g == 2 || g == 4;
  return false;
}

int strata_words(int hd, int bits) {  // codes of a position and head, in 32-bit words
  return StrataLayout(hd, bits).words();
}

void attention_strata(const float* q, KvRef kv, const StrataScratch& sc, int nh, int nkv, int hd, int rows,
                      float scale, const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out,
                      float* partial, int slots, const Launch& L, int from, cudaEvent_t second_ready,
                      const DevParams* P_hot) {
  const DevParams* hot_P = P_hot ? P_hot : P;
  const Out o = out ? Out{out, nullptr, nullptr} : split_out(L, rows, nh * hd);
  const int g = nh / nkv;
  bool ok = false;
#define FLYX_STRATA(HD_, G_) \
  if (hd == HD_ && g == G_) ok = by_bits<HD_, G_>(q, kv, sc, nh, nkv, rows, scale, P, gate, ld_gate, o, ld_out, partial, slots, L.st, \
                                    !opts(L).strata_mma, from, second_ready, hot_P);
  FLYX_STRATA(32, 1) FLYX_STRATA(32, 2) FLYX_STRATA(32, 4)
  FLYX_STRATA(64, 1) FLYX_STRATA(64, 2) FLYX_STRATA(64, 4)
  FLYX_STRATA(128, 1) FLYX_STRATA(128, 2) FLYX_STRATA(128, 4)
  FLYX_STRATA(256, 1) FLYX_STRATA(256, 2) FLYX_STRATA(256, 4) FLYX_STRATA(256, 6) FLYX_STRATA(256, 8)
#undef FLYX_STRATA
  if (!ok)  // the backend refuses other shapes at load (strata_supported)
    throw std::runtime_error("flyx-cuda: strata attention with head dim " + std::to_string(hd) + ", " +
                             std::to_string(g) + " query heads a KV head, " + std::to_string(kv.bits) + " bits");
}

}  // namespace flyx::cuda
