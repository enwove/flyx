// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GEMMs for many rows (prefill): bf16 / NVFP4 weights on tensor cores, the pipelined kernel and its configurations.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/gemm_internal.h"

namespace flyx::cuda {
namespace {

// ---- GEMM: many rows (prefill), bf16 tensor cores, fp32 accumulate ------------------------
// Block tile 64x64x32, 4 warps in 2x2, each warp 32x32 = 2 (m16) x 4 (n8) mma tiles.
__global__ void __launch_bounds__(128) k_gemm_mma(const float* __restrict__ x, int ldx, const bf16* __restrict__ W,
                                                  int N, int K, float* __restrict__ y, int ldy, int M,
                                                  int accumulate) {
  pdl_trigger();
  __shared__ __align__(16) bf16 As[64][40];
  __shared__ __align__(16) bf16 Bs[64][40];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  const int lr = tid >> 1, lc = (tid & 1) * 16;
  for (int k0 = 0; k0 < K; k0 += 32) {
    {  // A: 64 x 32 floats -> bf16
      const int gm = m0 + lr;
#pragma unroll
      for (int j = 0; j < 16; j += 4) {
        const int kk = k0 + lc + j;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gm < M && kk < K) v = *reinterpret_cast<const float4*>(x + size_t(gm) * ldx + kk);
        *reinterpret_cast<__nv_bfloat162*>(&As[lr][lc + j]) = __floats2bfloat162_rn(v.x, v.y);
        *reinterpret_cast<__nv_bfloat162*>(&As[lr][lc + j + 2]) = __floats2bfloat162_rn(v.z, v.w);
      }
    }
    {  // B: 64 x 32 bf16
      const int gn = n0 + lr;
#pragma unroll
      for (int j = 0; j < 16; j += 8) {
        const int kk = k0 + lc + j;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (gn < N && kk < K) v = *reinterpret_cast<const uint4*>(W + size_t(gn) * K + kk);
        *reinterpret_cast<uint4*>(&Bs[lr][lc + j]) = v;
      }
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 32; ks += 16) {
      uint32_t a[2][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16;
        a[mi][0] = *reinterpret_cast<const uint32_t*>(&As[r + g][ks + 2 * t]);
        a[mi][1] = *reinterpret_cast<const uint32_t*>(&As[r + g + 8][ks + 2 * t]);
        a[mi][2] = *reinterpret_cast<const uint32_t*>(&As[r + g][ks + 2 * t + 8]);
        a[mi][3] = *reinterpret_cast<const uint32_t*>(&As[r + g + 8][ks + 2 * t + 8]);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int c = wn * 32 + ni * 8;
        b[ni][0] = *reinterpret_cast<const uint32_t*>(&Bs[c + g][ks + 2 * t]);
        b[ni][1] = *reinterpret_cast<const uint32_t*>(&Bs[c + g][ks + 2 * t + 8]);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma_bf16(acc[mi][ni], a[mi], b[ni]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r = m0 + wm * 32 + mi * 16 + g;
      const int c = n0 + wn * 32 + ni * 8 + 2 * t;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int rr = r + (e >> 1) * 8, cc = c + (e & 1);
        if (rr < M && cc < N) {
          float* dst = y + size_t(rr) * ldy + cc;
          *dst = accumulate ? *dst + acc[mi][ni][e] : acc[mi][ni][e];
        }
      }
    }
}

// Prefill NVFP4 GEMM: as k_gemm_mma, the weight tile decoded to bf16 on its way to shared memory.
__global__ void __launch_bounds__(128) k_gemm_mma_fp4(const float* __restrict__ x, int ldx,
                                                      const uint8_t* __restrict__ Q, const uint8_t* __restrict__ S,
                                                      const float* __restrict__ RS, int N, int K,
                                                      float* __restrict__ y, int ldy, int M, int accumulate) {
  pdl_trigger();
  __shared__ __align__(16) bf16 As[64][40];
  __shared__ __align__(16) bf16 Bs[64][40];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp >> 1, wn = warp & 1, g = lane >> 2, t = lane & 3;
  const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
  float acc[2][4][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
  const int lr = tid >> 1, lc = (tid & 1) * 16;
  for (int k0 = 0; k0 < K; k0 += 32) {
    {
      const int gm = m0 + lr;
#pragma unroll
      for (int j = 0; j < 16; j += 4) {
        const int kk = k0 + lc + j;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (gm < M && kk < K) v = *reinterpret_cast<const float4*>(x + size_t(gm) * ldx + kk);
        *reinterpret_cast<__nv_bfloat162*>(&As[lr][lc + j]) = __floats2bfloat162_rn(v.x, v.y);
        *reinterpret_cast<__nv_bfloat162*>(&As[lr][lc + j + 2]) = __floats2bfloat162_rn(v.z, v.w);
      }
    }
    {  // 16 weights of row gn: 8 bytes and one scale
      const int gn = n0 + lr, kk = k0 + lc;
      uint32_t out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (gn < N && kk < K) {
        const uint2 q = *reinterpret_cast<const uint2*>(Q + size_t(gn) * (K >> 1) + (kk >> 1));
        fp4x8_to_bf16(q.x, out);
        fp4x8_to_bf16(q.y, out + 4);
        const __nv_bfloat162 s2 = e4m3_bf16x2(S[size_t(gn) * (K >> 4) + (kk >> 4)]);
#pragma unroll
        for (int e = 0; e < 8; ++e) out[e] = bf16x2_mul(out[e], s2);
      }
      *reinterpret_cast<uint4*>(&Bs[lr][lc]) = make_uint4(out[0], out[1], out[2], out[3]);
      *reinterpret_cast<uint4*>(&Bs[lr][lc + 8]) = make_uint4(out[4], out[5], out[6], out[7]);
    }
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 32; ks += 16) {
      uint32_t a[2][4], b[4][2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int r = wm * 32 + mi * 16;
        a[mi][0] = *reinterpret_cast<const uint32_t*>(&As[r + g][ks + 2 * t]);
        a[mi][1] = *reinterpret_cast<const uint32_t*>(&As[r + g + 8][ks + 2 * t]);
        a[mi][2] = *reinterpret_cast<const uint32_t*>(&As[r + g][ks + 2 * t + 8]);
        a[mi][3] = *reinterpret_cast<const uint32_t*>(&As[r + g + 8][ks + 2 * t + 8]);
      }
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        const int c = wn * 32 + ni * 8;
        b[ni][0] = *reinterpret_cast<const uint32_t*>(&Bs[c + g][ks + 2 * t]);
        b[ni][1] = *reinterpret_cast<const uint32_t*>(&Bs[c + g][ks + 2 * t + 8]);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) mma_bf16(acc[mi][ni], a[mi], b[ni]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      const int r = m0 + wm * 32 + mi * 16 + g;
      const int c = n0 + wn * 32 + ni * 8 + 2 * t;
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int rr = r + (e >> 1) * 8, cc = c + (e & 1);
        if (rr < M && cc < N) {
          const float v = acc[mi][ni][e] * RS[cc];
          float* dst = y + size_t(rr) * ldy + cc;
          *dst = accumulate ? *dst + v : v;
        }
      }
    }
}

// ---- GEMM: many rows (prefill), pipelined tensor cores ------------------------------------
// C[M, N] (=|+=) A[M, K] * W[N, K]^T * colscale.  A is bf16 (activations
// rounded once by k_split), W bf16 or NVFP4.  Block tile (64 WM) x 128 x 32,
// WM x WN warps of 64 x (128 / WN), an ST-stage cp.async pipeline, fragments
// by ldmatrix.  Every element adds its k-tiles in order with the same MMAs
// whatever the tile, so all configurations give the same bits.  The shared
// tiles' rows are swizzled rather than padded (tc_swz, below).
// NVFP4: the stage holds the packed bytes and scales; each thread decodes its
// own B fragments from them (the bytes t, t + 4, t + 8, t + 12 of its weight
// row hold its values at k = 2t.., 8 + 2t.., 16 + 2t.., 24 + 2t..), so no
// decoded tile goes through shared memory and no barrier waits for one.
constexpr int kTcBK = 32, kTcLd = 32;
// Tile rows are 64 bytes (no padding): the 16-byte chunk c of row r is stored
// at chunk c ^ ((r >> 1) & 3), so the 8 rows an ldmatrix phase reads hit 8
// different bank groups, and three stages fit two blocks per SM.
__device__ __forceinline__ int tc_swz(int r, int c) { return ((((c >> 3) ^ (r >> 1)) & 3) << 3) | (c & 7); }


template <bool FP4, int BM, int BN, int ST>
struct TcSmem {
  bf16 a[ST][BM][kTcLd];
  bf16 b[FP4 ? 1 : ST][FP4 ? 1 : BN][kTcLd];  // bf16 weights per stage (NVFP4: unused)
  uint8_t q[FP4 ? ST : 1][FP4 ? BN : 1][16];  // NVFP4: 32 values of a row = 16 bytes
  uint8_t s[FP4 ? ST : 1][FP4 ? BN : 1][2];   // NVFP4, K % 256 != 0: two e4m3 scales of a row
  uint8_t s8[FP4 ? 2 : 1][FP4 ? BN : 1][16];  // NVFP4, K % 256 == 0: the scales of eight k-tiles
};

// A block tile of BM = 64 WM rows and BN columns; WN warps along N, each
// warp a 64 x (BN / WN) tile.  Every output is the same sum in the same order
// (the k-tiles in turn, 16 k per mma) whatever BM, BN, WN and ST are, so all
// the configurations give the same bits.  With BN = 128, WN = 4 (warp tiles
// of 64 x 32) is used; WN = 2 (64 x 64) measured the same and needs 166
// registers instead of 122, which would cost the second block per SM.
// GLU: W is a gate|up weight and the N = 2 I columns follow glu_row (device.cuh);
// the epilogue writes glu [M][I] = bf16(act(g) u) instead of y (act: glu_act, flyx_mlp_act).
template <bool FP4, int WN, int WM, int ST, bool GLU, int BN>
__global__ void __launch_bounds__(32 * WM * WN) k_gemm_tc(const bf16* __restrict__ A, int lda, const bf16* __restrict__ W,
                                                    const uint8_t* __restrict__ Q, const uint8_t* __restrict__ S,
                                                    const float* __restrict__ RS, int N, int K, float* __restrict__ y,
                                                    int ldy, int M, int accumulate, bf16* __restrict__ glu,
                                                    int act) {
  pdl_trigger();
  constexpr int BM = 64 * WM, THREADS = 32 * WM * WN, NI = BN / WN / 8;
  constexpr int CPA = BM * 4 / THREADS, CPB = BN * 4 / THREADS;  // 16-byte copies per thread (A, bf16 W)
  // NVFP4: each thread loads at most one row's scale pair per k-tile (store_scales keeps one).
  static_assert(NI >= 1 && CPA >= 1 && CPB >= 1 && BN <= THREADS, "block tile and threads");
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<TcSmem<FP4, BM, BN, ST>*>(smem_raw);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp % WN;  // warp tile: rows wm*64.., cols wn*(BN/WN)..
  // Blocks walk down M first: the blocks that share a weight tile run together, so each weight
  // tile comes from DRAM once (the activations, a few MB, stay in L2).
  const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
  const int ktiles = (K + kTcBK - 1) / kTcBK;

  // Each thread copies CPA x 16 bytes of A (BM rows x 64 bytes) and, for bf16, CPB x 16 of W.
  // NVFP4 scales: two bytes per row and k-tile.  When K % 256 == 0 the 16
  // bytes of eight k-tiles are aligned: one cp.async with the stage of every
  // eighth k-tile brings them, into two slots (group g's slot is refilled for
  // group g + 2 at iteration 8 g + 17 - ST > 8 g + 7, after its last reader).
  // Otherwise a prefetch loads them into a register and the next iteration
  // stores them, before the barrier that precedes the stage's use: a
  // synchronous load per k-tile (with it, the GEMMs of a 5760-token prefill
  // take 3% longer, bench/trace; cuda_bench tc 7-13% on the wide shapes).
  const bool group_scales = FP4 && (K & 255) == 0;
  uint32_t scale_val = 0;
  int scale_stage = -1, scale_row = 0;
  auto store_scales = [&] {
    if (FP4 && scale_stage >= 0) {
      *reinterpret_cast<uint16_t*>(&sm.s[scale_stage][scale_row][0]) = uint16_t(scale_val);
      scale_stage = -1;
    }
  };
  auto load_stage = [&](int stage, int kt) {
    const int k0 = kt * kTcBK;
#pragma unroll
    for (int i = 0; i < CPA; ++i) {
      const int idx = tid + i * THREADS, r = idx >> 2, c = (idx & 3) * 8;
      const int gm = m0 + r;
      const bool ok = gm < M && k0 + c < K;
      cp_async16(&sm.a[stage][r][tc_swz(r, c)], ok ? A + size_t(gm) * lda + k0 + c : A, ok);
    }
    if constexpr (!FP4) {
#pragma unroll
      for (int i = 0; i < CPB; ++i) {
        const int idx = tid + i * THREADS, r = idx >> 2, c = (idx & 3) * 8;
        const int gn = n0 + r, wr = GLU ? glu_row(gn, N >> 1) : gn;
        const bool ok = gn < N && k0 + c < K;
        cp_async16(&sm.b[stage][r][tc_swz(r, c)], ok ? W + size_t(wr) * K + k0 + c : W, ok);
      }
    } else {
      for (int i = tid; i < 2 * BN; i += THREADS) {
        if (i < BN) {  // 16 bytes of codes per row
          const int gn = n0 + i, wr = GLU ? glu_row(gn, N >> 1) : gn;
          const bool ok = gn < N && k0 < K;
          cp_async16(&sm.q[stage][i][0], ok ? Q + size_t(wr) * (K >> 1) + (k0 >> 1) : Q, ok);
        } else {
          const int r = i - BN, gn = n0 + r, wr = GLU ? glu_row(gn, N >> 1) : gn;
          const bool ok = gn < N && k0 < K;
          if (group_scales) {  // with every eighth k-tile: the scales of it and the next seven
            if ((kt & 7) == 0)
              cp_async16(&sm.s8[(kt >> 3) & 1][r][0], ok ? S + size_t(wr) * (K >> 4) + (k0 >> 4) : S, ok);
          } else {  // two scale bytes (K % 32 == 0: an aligned pair), stored by store_scales()
            scale_val = ok ? *reinterpret_cast<const uint16_t*>(S + size_t(wr) * (K >> 4) + (k0 >> 4)) : 0u;
            scale_stage = stage;
            scale_row = r;
          }
        }
      }
    }
  };

  float acc[4][NI][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < ktiles) load_stage(s, s);
    store_scales();
    cp_async_commit();
  }
  for (int kt = 0; kt < ktiles; ++kt) {
    const int stage = kt % ST;
    store_scales();  // the tile prefetched last iteration (its stage is read from iteration kt + ST - 2 >= kt)
    cp_async_wait<ST - 2>();
    __syncthreads();
    {  // prefetch tile kt + stages - 1 into the stage freed last iteration
      const int nk = kt + ST - 1;
      if (nk < ktiles) load_stage(nk % ST, nk);
      cp_async_commit();
    }
    // NVFP4: this k-tile's B fragments, decoded in registers (k 0..15: [0], [1]; k 16..31: [2], [3]).
    uint32_t bq[FP4 ? NI : 1][4];
    if constexpr (FP4) {
      const int t = lane & 3;
      const uint32_t pick = uint32_t(t) | (uint32_t(t + 4) << 4);  // byte t of the first and second word
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const int r = wn * (BN / WN) + ni * 8 + (lane >> 2);
        const uint4 q = *reinterpret_cast<const uint4*>(&sm.q[stage][r][0]);
        const uint32_t w = __byte_perm(__byte_perm(q.x, q.y, pick), __byte_perm(q.z, q.w, pick), 0x5410);
        fp4x8_to_bf16(w, bq[ni]);
        const uint8_t* sr = group_scales ? &sm.s8[(kt >> 3) & 1][r][(kt & 7) * 2] : &sm.s[stage][r][0];
        const __nv_bfloat162 s0 = e4m3_bf16x2(sr[0]), s1 = e4m3_bf16x2(sr[1]);
        bq[ni][0] = bf16x2_mul(bq[ni][0], s0);
        bq[ni][1] = bf16x2_mul(bq[ni][1], s0);
        bq[ni][2] = bf16x2_mul(bq[ni][2], s1);
        bq[ni][3] = bf16x2_mul(bq[ni][3], s1);
      }
    }
#pragma unroll
    for (int ks = 0; ks < kTcBK; ks += 16) {
      uint32_t af[4][4], bfr[NI][2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r = wm * 64 + mi * 16 + (lane & 15), c = ks + (lane >> 4) * 8;
        ldmatrix_x4(af[mi], &sm.a[stage][r][tc_swz(r, c)]);
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        if constexpr (FP4) {
          bfr[ni][0] = bq[ni][ks / 8];
          bfr[ni][1] = bq[ni][ks / 8 + 1];
        } else {
          const int r = wn * (BN / WN) + ni * 8 + (lane & 7), c = ks + ((lane >> 3) & 1) * 8;
          ldmatrix_x2(bfr[ni], &sm.b[stage][r][tc_swz(r, c)]);
        }
      }
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_bf16(acc[mi][ni], af[mi], bfr[ni]);
    }
  }
  cp_async_wait<0>();
  const int g = lane >> 2, t = lane & 3;
  if constexpr (GLU) {  // tiles ni (gate) and ni + 1 (up) hold the columns j, j + 1 of the output
    static_assert(NI % 2 == 0, "gate and up tiles in pairs");
    const int I = N >> 1;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ni += 2)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = m0 + wm * 64 + mi * 16 + g + 8 * h;
          const int j = glu_col(n0 + wn * (BN / WN) + ni * 8 + 2 * t);
          if (r < M && j < I) {
            float g0 = acc[mi][ni][2 * h], g1 = acc[mi][ni][2 * h + 1];
            float u0 = acc[mi][ni + 1][2 * h], u1 = acc[mi][ni + 1][2 * h + 1];
            if (FP4) g0 *= RS[j], g1 *= RS[j + 1], u0 *= RS[I + j], u1 *= RS[I + j + 1];
            *reinterpret_cast<__nv_bfloat162*>(glu + size_t(r) * I + j) =
                __floats2bfloat162_rn(glu_act(act, g0) * u0, glu_act(act, g1) * u1);
          }
        }
  } else {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = m0 + wm * 64 + mi * 16 + g + (e >> 1) * 8;
          const int c = n0 + wn * (BN / WN) + ni * 8 + 2 * t + (e & 1);
          if (r < M && c < N) {
            float v = acc[mi][ni][e];
            if (FP4) v *= RS[c];
            float* dst = y + size_t(r) * ldy + c;
            *dst = accumulate ? *dst + v : v;
          }
        }
  }
}

}  // namespace

void gemm_mma(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
              cudaStream_t st) {
  dim3 grid(ceil_div(N, 64), ceil_div(rows, 64));
  k_gemm_mma<<<grid, 128, 0, st>>>(x, ldx, W, N, K, y, ldy, rows, accumulate);
}

void gemm_fp4_mma(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
                  int ldy, int rows, bool accumulate, cudaStream_t st) {
  dim3 grid(ceil_div(N, 64), ceil_div(rows, 64));
  k_gemm_mma_fp4<<<grid, 128, 0, st>>>(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate);
}

// Prefill GEMM configurations (block tile rows x columns, pipeline stages);
// all give the same bits.  0: 64 x 128, 2 stages, 24 KB of shared memory
// (NVFP4 less), four blocks per SM.  1: 128 x 128, 3 stages, two blocks.  On
// the RTX 5070 Ti (cuda_bench tc, Qwen3.5-4B shapes, 512-2048 rows) 0 is
// 2-24% faster than 1 with bf16 weights; with NVFP4 up to 15% faster on the
// narrow shapes and within 6% on the wide ones.  2: 64 x 64 and 3: 64 x 32
// (two warps along N), for a short pass whose 64 x 128 tiles leave SMs idle:
// a 2560-column weight is 20 such tiles per 64 rows.
template <bool FP4, int WN, int WM, int ST, bool GLU, int BN>
void launch_tc_cfg(const bf16* a, const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
                   float* y, int ldy, int M, bool accumulate, cudaStream_t st, bf16* glu, int act) {
  constexpr int BM = 64 * WM;
  using Smem = TcSmem<FP4, BM, BN, ST>;
  static const bool attr = set_dynamic_smem(k_gemm_tc<FP4, WN, WM, ST, GLU, BN>, sizeof(Smem), true);
  (void)attr;
  const dim3 grid(ceil_div(M, BM), ceil_div(N, BN));
  k_gemm_tc<FP4, WN, WM, ST, GLU, BN><<<grid, 32 * WM * WN, sizeof(Smem), st>>>(a, K, W, Q, S, RS, N, K, y, ldy, M,
                                                                                accumulate, glu, act);
}

// The configuration a prefill GEMM gets, by its 64 x 128 tiles per SM (b):
// 64 x 32 below b = 0.5, 64 x 64 below 6 (NVFP4) or 2.5 (bf16 weights), else
// 64 x 128.  From cuda_bench tc on the RTX 5070 Ti (Qwen3.5-4B shapes, 32-2048
// rows): within 5% of the
// best configuration on every shape and row count, and on 2560-column
// weights up to 1.7x the 64 x 128 tile at 32-64 rows.
template <bool FP4>
int tc_auto_config(int M, int N) {
  const int64_t sms = device_sm_count();
  const int64_t tiles = int64_t(ceil_div(M, 64)) * ceil_div(N, 128);
  if (2 * tiles < sms) return 3;
  if (FP4 ? tiles < 6 * sms : 2 * tiles < 5 * sms) return 2;
  return 0;
}

// glu non-null: the gate|up GEMM with SwiGLU in its epilogue (N = 2 I columns, glu [M][I]).
// cfg: one of kTcConfigs, -1 = tc_auto_config.
template <bool FP4, bool GLU>
void launch_tc_glu(int cfg, const bf16* a, const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N,
                   int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st, bf16* glu, int act = 0) {
  switch (cfg < 0 ? tc_auto_config<FP4>(M, N) : cfg) {
    case 1: launch_tc_cfg<FP4, 4, 2, 3, GLU, 128>(a, W, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, act); break;
    case 2: launch_tc_cfg<FP4, 4, 1, 2, GLU, 64>(a, W, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, act); break;
    case 3: launch_tc_cfg<FP4, 2, 1, 2, GLU, 32>(a, W, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, act); break;
    default: launch_tc_cfg<FP4, 4, 1, 2, GLU, 128>(a, W, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, act); break;
  }
}

template <bool FP4>
void launch_tc_by(int cfg, const bf16* a, const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N,
                  int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st, bf16* glu = nullptr,
                  int act = 0) {
  if (glu) launch_tc_glu<FP4, true>(cfg, a, W, Q, S, RS, N, K, y, ldy, M, false, st, glu, act);
  else launch_tc_glu<FP4, false>(cfg, a, W, Q, S, RS, N, K, y, ldy, M, accumulate, st, nullptr);
}

namespace {
// Whether NVFP4 weights of this K take NVFP4 inputs here (the W4A4 kernels): the model's weights do (checked at
// open: the device has the kernels, every such K % 128 == 0) and the scratch holds their codes.
bool w4a4(const Launch& L, int K) {
  const Scratch& sc = scratch(L);
  return opts(L).prefill_fp4 && K % 128 == 0 && sc.q4 && gemm_w4a4_supported();
}
// The W4A4 GEMM on float32 rows x [M][ldx], rounded once to NVFP4; glu: gemm_w4a4's (into the
// scratch's glu and glu_amax).
void w4a4_rows(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
               int ldy, int M, bool accumulate, const Launch& L, bool glu = false) {
  const Scratch& sc = scratch(L);
  quantize_rows_nvfp4(x, ldx, M, K, sc.q4, sc.q4_scale, sc.q4_inv, L.st);
  gemm_w4a4(sc.q4, sc.q4_scale, sc.q4_inv, Q, S, RS, N, K, y, ldy, M, accumulate, L.st, -1, glu ? sc.glu : nullptr,
            glu ? sc.glu_amax : nullptr, nullptr, glu ? L.glu_act : 0);
}
// float32 rows x [M][ldx] as bf16 in abuf, the bf16 tensor cores' input.
const bf16* to_abuf(const float* x, int ldx, int K, int M, const Launch& L) {
  const Scratch& sc = scratch(L);
  check_fits(size_t(M) * K, sc.abuf_cap, "a prefill GEMM");
  rows_to_bf16(x, ldx, M, K, sc.abuf, L.st);
  return sc.abuf;
}
// The bf16 tensor cores on bf16 rows a: NVFP4 weights (Q) or bf16 ones (W).
void tc_rows(int cfg, const bf16* a, const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K,
             float* y, int ldy, int M, bool accumulate, const Launch& L, bf16* glu = nullptr) {
  const int act = glu ? L.glu_act : 0;
  if (Q) launch_tc_by<true>(cfg, a, nullptr, Q, S, RS, N, K, y, ldy, M, accumulate, L.st, glu, act);
  else launch_tc_by<false>(cfg, a, W, nullptr, nullptr, nullptr, N, K, y, ldy, M, accumulate, L.st, glu, act);
}
}  // namespace

bool prefill_rows_bf16(const Launch& L, const uint8_t* Q, int K) {
  return L.scratch && L.scratch->af32 && Q && w4a4(L, K);
}

template <bool FP4>
void launch_tc(const float* x, int ldx, const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N,
               int K, float* y, int ldy, int M, bool accumulate, const Launch& L, int cfg) {
  if (FP4 && w4a4(L, K)) return w4a4_rows(x, ldx, Q, S, RS, N, K, y, ldy, M, accumulate, L);
  tc_rows(cfg, to_abuf(x, ldx, K, M, L), W, Q, S, RS, N, K, y, ldy, M, accumulate, L);
}

void gemm_tc(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int M, bool accumulate,
             const Launch& L, int cfg) {
  launch_tc<false>(x, ldx, W, nullptr, nullptr, nullptr, N, K, y, ldy, M, accumulate, L, cfg);
}

void gemm_fp4_tc(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
                 int ldy, int M, bool accumulate, const Launch& L, int cfg) {
  launch_tc<true>(x, ldx, nullptr, Q, S, RS, N, K, y, ldy, M, accumulate, L, cfg);
}

// The prefill GEMM on the rows a fused producer left in the scratch (gemm_presplit for more than 16 rows): float32
// in af32 where the scratch has it (NVFP4 inputs), else bf16 in abuf.
void gemm_tc_staged(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy,
                    int M, bool accumulate, const Launch& L) {
  const Scratch& sc = scratch(L);
  if (sc.af32 && Q && w4a4(L, K)) return w4a4_rows(sc.af32, K, Q, S, RS, N, K, y, ldy, M, accumulate, L);
  tc_rows(-1, sc.af32 ? to_abuf(sc.af32, K, K, M, L) : sc.abuf, W, Q, S, RS, N, K, y, ldy, M, accumulate, L);
}

bool gemm_staged_bf16(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, bf16* y16, int ldy, int M,
                      const Launch& L, bool accumulate) {
  const Scratch& sc = scratch(L);
  if (!prefill_rows_bf16(L, Q, K)) return false;
  quantize_rows_nvfp4(sc.af32, K, M, K, sc.q4, sc.q4_scale, sc.q4_inv, L.st);
  gemm_w4a4(sc.q4, sc.q4_scale, sc.q4_inv, Q, S, RS, N, K, nullptr, ldy, M, accumulate, L.st, -1, nullptr, nullptr,
            y16);
  return true;
}

bool gemm_on_glu_bf16(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, bf16* y16, int ldy, int M,
                      bool accumulate, const Launch& L) {
  const Scratch& sc = scratch(L);
  if (!sc.glu_w4a4 || !Q || !w4a4(L, K)) return false;
  quantize_rows_nvfp4_bf16(sc.glu, K, M, K, sc.glu_amax, sc.q4, sc.q4_scale, sc.q4_inv, L.st);
  gemm_w4a4(sc.q4, sc.q4_scale, sc.q4_inv, Q, S, RS, N, K, nullptr, ldy, M, accumulate, L.st, -1, nullptr, nullptr,
            y16);
  return true;
}

void gemm_glu_staged(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int I, int K, int M,
                     const Launch& L, int cfg) {
  const Scratch& sc = scratch(L);
  if (I % 8 || M <= 16) {
    std::fprintf(stderr, "flyx-cuda: gemm_glu_staged needs I %% 8 == 0 and more than 16 rows (I %d, rows %d)\n",
                 I, M);
    std::abort();
  }
  check_fits(size_t(M) * I, sc.glu_cap, "a SwiGLU GEMM");
  // NVFP4 inputs: the SwiGLU rows in bf16 with each row's largest |value|, which the down projection rounds from.
  sc.glu_w4a4 = prefill_rows_bf16(L, Q, K);
  if (sc.glu_w4a4) return w4a4_rows(sc.af32, K, Q, S, RS, 2 * I, K, nullptr, 0, M, false, L, true);
  tc_rows(cfg, sc.af32 ? to_abuf(sc.af32, K, K, M, L) : sc.abuf, W, Q, S, RS, 2 * I, K, nullptr, 0, M, false, L,
          sc.glu);
}

void gemm_on_glu(const bf16* W, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy,
                 int M, bool accumulate, const Launch& L) {
  const Scratch& sc = scratch(L);
  if (Q && w4a4(L, K)) {  // the down weights' inputs: the rows' maxima where the W4A4 gate|up GEMM left them
    quantize_rows_nvfp4_bf16(sc.glu, K, M, K, sc.glu_w4a4 ? sc.glu_amax : nullptr, sc.q4, sc.q4_scale, sc.q4_inv,
                             L.st);
    return gemm_w4a4(sc.q4, sc.q4_scale, sc.q4_inv, Q, S, RS, N, K, y, ldy, M, accumulate, L.st);
  }
  tc_rows(-1, sc.glu, W, Q, S, RS, N, K, y, ldy, M, accumulate, L);
}

}  // namespace flyx::cuda
