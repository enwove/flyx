// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gated delta net prefill in chunks: the chunkwise form of the delta rule
// (Yang et al., "Parallelizing Linear Transformers with the Delta Rule over
// Sequence Length", NeurIPS 2024; "Gated Delta Networks", ICLR 2025), on
// bf16 tensor cores with float32 sums and a float32 state.
//
// Per value head, a chunk of positions t (state S [dv][dk] at its start,
// gamma_t the cumulative log decay inside it, G_t = exp(gamma_t)):
//   L[t][s] = beta_t exp(gamma_t - gamma_s) (k_t . k_s)   s < t
//   T       = (I + L)^-1                                  (unit lower triangular)
//   U       = T (beta (V - G (K S^T)))                    the rows' deltas
//   O       = G (Q S^T) + A U,  A[t][s] = exp(gamma_t - gamma_s) (q_t . k_s), s <= t
//   S'      = G_last S + (f U)^T K,  f_t = exp(gamma_last - gamma_t)
// k_gdn_chunk_tri computes T and A for every (chunk, head) at once (one block
// per key head and chunk); k_gdn_chunk_scan walks the chunks of one head and
// block of value columns.
//
// Precision (as vLLM's and SGLang's chunked kernels, FLA's chunk_gated_delta_rule):
// every product's inputs rounded to bf16, the sums in float32; T is solved in
// float32 and the state stays float32.
//
// Chunks cover absolute positions [32 c, 32 c + 32): a pass that starts at a
// multiple of 32 computes every chunk as a longer pass over the same positions
// does.  Positions outside the pass enter as zeros, and nothing a row computes
// reads a later row (T and A are lower triangular, and so is every block step
// of the inverse), so a row's output does not depend on where the pass ends.
#include <algorithm>
#include <cstdio>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kChunk = kGdnChunkSize;  // positions per chunk
constexpr int kDK = 128;          // key dim (the only one the kernels take)
constexpr int kBV = 64;           // value columns per scan block
constexpr int kLdK = kDK + 8;     // bf16 row strides in shared memory, padded so fragment loads spread over banks
constexpr int kLdF = kChunk + 1;  // float32 [32][32] matrices

// Swizzled bf16 tile (zero row padding, bank-conflict-free 16-byte ldmatrix / put2): 8-bf16 chunks XORed by row.
template <int R, int C>
struct SwizTile {
  static_assert(C == 32 || C == 128, "swizzle mask matches 32- or 128-element bf16 rows");
  static constexpr int kRowMask = C == 128 ? 7 : 3;
  bf16 v[R][C];
  __device__ __forceinline__ static int sw(int r, int c) { return c ^ ((r & kRowMask) << 3); }
  __device__ __forceinline__ void put2(int r, int c, float a, float b) {
    *reinterpret_cast<uint32_t*>(&v[r][sw(r, c)]) = pack_bf16(a, b);
  }
  __device__ __forceinline__ void put(int r, int c, float a) { v[r][sw(r, c)] = __float2bfloat16_rn(a); }
};

__device__ __forceinline__ int sw_q(int r, int c) { return c ^ ((r & 7) << 2); }

// The tri kernel's outputs, per (chunk, value head): T and A (float32 [32][32]), gamma and beta (float32 [32]).
struct ChunkScratch {
  float *T, *A, *gam, *beta;
};

// A matrix in shared memory in bf16 (each value rounded to nearest), the products' operands.
template <int R, int LD>
struct Tile {
  bf16 v[R][LD];
  __device__ __forceinline__ void put2(int r, int c, float a, float b) {  // values (r, c) and (r, c + 1); c even
    *reinterpret_cast<uint32_t*>(&v[r][c]) = pack_bf16(a, b);
  }
  __device__ __forceinline__ void put(int r, int c, float a) { v[r][c] = __float2bfloat16_rn(a); }
};

// mma.m16n8k16 operands from shared memory: A row-major [m][k]; B n-major [n][k] (two n tiles at once).
__device__ __forceinline__ void frag_a(uint32_t a[4], const bf16* A, int lda, int row0, int k0, int lane) {
  ldmatrix_x4(a, A + (row0 + (lane & 15)) * lda + k0 + 8 * (lane >> 4));
}
__device__ __forceinline__ void frag_b2(uint32_t b[4], const bf16* Bn, int ldb, int n0, int k0, int lane) {
  ldmatrix_x4(b, Bn + (n0 + (lane & 7) + 8 * (lane >> 4)) * ldb + k0 + 8 * ((lane >> 3) & 1));
}

// acc (rows row0 .. row0 + 15, columns n0 .. n0 + 8 NT) += A B over k < K; NT even.
// Element e of acc[j]: row row0 + lane / 4 + 8 (e / 2), column n0 + 8 j + 2 (lane % 4) + e % 2.
template <int NT, int RA, int LDA, int RB, int LDB>
__device__ __forceinline__ void warp_mma(float (&acc)[NT][4], const Tile<RA, LDA>& A, int row0,
                                         const Tile<RB, LDB>& Bn, int n0, int K, int lane) {
  static_assert(NT % 2 == 0, "two n tiles per ldmatrix");
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4];
    frag_a(a, &A.v[0][0], LDA, row0, k0, lane);
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b[4];
      frag_b2(b, &Bn.v[0][0], LDB, n0 + 8 * j, k0, lane);
      mma_bf16(acc[j], a, b);
      mma_bf16(acc[j + 1], a, b + 2);
    }
  }
}

// Swizzled-tile MMA helpers (identical arithmetic order to warp_mma / warp_mma_kb / warp_mma_kq).
template <int RA, int CA>
__device__ __forceinline__ void frag_a_sw(uint32_t a[4], const SwizTile<RA, CA>& A, int row0, int k0, int lane) {
  const int r = row0 + (lane & 15), c = k0 + 8 * (lane >> 4);
  ldmatrix_x4(a, &A.v[r][SwizTile<RA, CA>::sw(r, c)]);
}
template <int RB, int CB>
__device__ __forceinline__ void frag_b2_sw(uint32_t b[4], const SwizTile<RB, CB>& Bn, int n0, int k0, int lane) {
  const int r = n0 + (lane & 7) + 8 * (lane >> 4), c = k0 + 8 * ((lane >> 3) & 1);
  ldmatrix_x4(b, &Bn.v[r][SwizTile<RB, CB>::sw(r, c)]);
}
template <int RB, int CB>
__device__ __forceinline__ void frag_b2_kmajor_sw(uint32_t b[4], const SwizTile<RB, CB>& Bk, int n0, int k0, int lane) {
  const int r = k0 + (lane & 7) + 8 * ((lane >> 3) & 1), c = n0 + 8 * (lane >> 4);
  ldmatrix_x4_t(b, &Bk.v[r][SwizTile<RB, CB>::sw(r, c)]);
}

template <int NT, int RA, int CA, int RB, int CB>
__device__ __forceinline__ void warp_mma(float (&acc)[NT][4], const SwizTile<RA, CA>& A, int row0,
                                         const SwizTile<RB, CB>& Bn, int n0, int K, int lane) {
  static_assert(NT % 2 == 0, "two n tiles per ldmatrix");
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4];
    frag_a_sw(a, A, row0, k0, lane);
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b[4];
      frag_b2_sw(b, Bn, n0 + 8 * j, k0, lane);
      mma_bf16(acc[j], a, b);
      mma_bf16(acc[j + 1], a, b + 2);
    }
  }
}

template <int NT, int RA, int CA, int RB, int CB>
__device__ __forceinline__ void warp_mma_kb(float (&acc)[NT][4], const SwizTile<RA, CA>& A, int row0,
                                            const SwizTile<RB, CB>& Bk, int n0, int K, int lane) {
  static_assert(NT % 2 == 0, "two n tiles per ldmatrix");
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4];
    frag_a_sw(a, A, row0, k0, lane);
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b[4];
      frag_b2_kmajor_sw(b, Bk, n0 + 8 * j, k0, lane);
      mma_bf16(acc[j], a, b);
      mma_bf16(acc[j + 1], a, b + 2);
    }
  }
}

// Two products over the 128 key dims with one B: p += K B (K swizzled in shared memory) and o += Q B, Q float32
// swizzled in shared memory [32][128] and rounded here as K's staging rounds it.
template <int NT, int RB>
__device__ __forceinline__ void warp_mma_kq(float (&p)[NT][4], float (&o)[NT][4], const SwizTile<kChunk, kDK>& K,
                                            const float (*Q)[kDK], int row0, const SwizTile<RB, kDK>& Bn, int n0,
                                            int lane) {
  static_assert(NT % 2 == 0, "two n tiles per ldmatrix");
  const int g = lane >> 2, t = lane & 3;
#pragma unroll 2
  for (int k0 = 0; k0 < kDK; k0 += 16) {
    uint32_t a[4], q[4];
    frag_a_sw(a, K, row0, k0, lane);
    const int r0 = row0 + g, r1 = r0 + 8;
    const float2 x0 = *reinterpret_cast<const float2*>(&Q[r0][sw_q(r0, k0 + 2 * t)]);
    const float2 x1 = *reinterpret_cast<const float2*>(&Q[r1][sw_q(r1, k0 + 2 * t)]);
    const float2 x2 = *reinterpret_cast<const float2*>(&Q[r0][sw_q(r0, k0 + 2 * t + 8)]);
    const float2 x3 = *reinterpret_cast<const float2*>(&Q[r1][sw_q(r1, k0 + 2 * t + 8)]);
    q[0] = pack_bf16(x0.x, x0.y), q[1] = pack_bf16(x1.x, x1.y), q[2] = pack_bf16(x2.x, x2.y);
    q[3] = pack_bf16(x3.x, x3.y);
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b[4];
      frag_b2_sw(b, Bn, n0 + 8 * j, k0, lane);
      mma_bf16(p[j], a, b);
      mma_bf16(p[j + 1], a, b + 2);
      mma_bf16(o[j], q, b);
      mma_bf16(o[j + 1], q, b + 2);
    }
  }
}

// Rows of chunk r0 .. r0 + 31 of key head kh (normalised q or k, float32) split into dst; positions outside
// rows [0, rows) are zero.
template <int NT>
__device__ __forceinline__ void stage_rows(Tile<kChunk, kLdK>& dst, const float* src, int nk, int kh, int r0,
                                           int rows, int tid) {
  for (int idx = tid; idx < kChunk * kDK / 4; idx += NT) {
    const int t = idx / (kDK / 4), i = (idx % (kDK / 4)) * 4, r = r0 + t;
    const float4 v = r >= 0 && r < rows ? *reinterpret_cast<const float4*>(src + (size_t(r) * nk + kh) * kDK + i)
                                        : make_float4(0.f, 0.f, 0.f, 0.f);
    dst.put2(t, i, v.x, v.y);
    dst.put2(t, i + 2, v.z, v.w);
  }
}

// gamma: the chunk's cumulative log decay, by one warp (lane l: position l; an inclusive scan over the lanes).
// Position t reads positions <= t only, so it does not depend on the pass end.
__device__ __forceinline__ void chunk_cumsum(float* g, int lane) {
  float p = g[lane];
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float q = __shfl_up_sync(0xffffffffu, p, o);
    if (lane >= o) p += q;
  }
  g[lane] = p;
}

static_assert(kChunk == 32, "a chunk is a warp's lanes (chunk_cumsum) and two 16-row blocks (the inverse)");

// grid (chunks, nk); block 128: K K^T and Q K^T once per key head, then L, A
// and T for each of its value heads in turn.  Warp w: the products' rows
// 16 (w % 2) .. + 15, columns 16 (w / 2) .. + 15.
__global__ void __launch_bounds__(128) k_gdn_chunk_tri(GdnBufs bufs, int rows, int off, ChunkScratch cs) {
  __shared__ __align__(16) union {
    struct {
      Tile<kChunk, kLdK> k, q;
    } in;
    struct {
      float l[kChunk][kLdF], t[kChunk][kLdF], x[16][17];  // L, T, and X = L10 T00
    } m;  // once K and Q are used
  } u;
  __shared__ float sg[kChunk], sb[kChunk];
  const GdnDims d = bufs.d;
  const int c = blockIdx.x, kh = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3, pr = 16 * (warp & 1), pc = 16 * (warp >> 1);
  const int rep = d.rep(), r0 = c * kChunk - off;
  const float* B = bufs.base;
  stage_rows<128>(u.in.k, B + bufs.o_kn(), d.nk, kh, r0, rows, tid);
  stage_rows<128>(u.in.q, B + bufs.o_qn(), d.nk, kh, r0, rows, tid);
  __syncthreads();
  float kk[2][4] = {}, qk[2][4] = {};
  warp_mma<2>(kk, u.in.k, pr, u.in.k, pc, kDK, lane);
  warp_mma<2>(qk, u.in.q, pr, u.in.k, pc, kDK, lane);
  for (int hv = 0; hv < rep; ++hv) {
    const int h = d.value_head(kh, hv);
    const size_t at = size_t(c) * d.nv + h;
    __syncthreads();  // K and Q are read (their space becomes L and T); the last head's T is out
    if (tid < kChunk) {
      const int r = r0 + tid;
      const bool in = r >= 0 && r < rows;
      sg[tid] = in ? B[bufs.o_g() + size_t(r) * d.nv + h] : 0.f;
      sb[tid] = in ? B[bufs.o_beta() + size_t(r) * d.nv + h] : 0.f;
    }
    __syncthreads();
    if (warp == 0) chunk_cumsum(sg, lane);
    __syncthreads();
    float* A = cs.A + at * kChunk * kChunk;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int t = pr + g + 8 * (e >> 1), s = pc + 8 * n + 2 * tq + (e & 1);
        const float dts = expf(sg[t] - sg[s]);  // used where s <= t: <= 1
        u.m.l[t][s] = s < t ? sb[t] * dts * kk[n][e] : 0.f;
        A[t * kChunk + s] = s <= t ? dts * qk[n][e] : 0.f;
      }
    __syncthreads();
    // T = (I + L)^-1.  The diagonal blocks (warp b block b), by forward
    // substitution, lane s building column s in registers:
    // T[t][s] = -(L[t][s] + sum_{s < m < t} L[t][m] T[m][s]), m ascending.
    if (warp < 2) {
      const int b0 = 16 * warp, s = lane & 15;
      float col[16];
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        float lrow[16];
#pragma unroll
        for (int m = 0; m < 16; ++m) lrow[m] = u.m.l[b0 + t][b0 + m];
        float acc = u.m.l[b0 + t][b0 + s];
#pragma unroll
        for (int m = 1; m < 16; ++m)
          if (m < t && m > s) acc = __fmaf_rn(lrow[m], col[m], acc);
        col[t] = s < t ? -acc : (s == t ? 1.f : 0.f);
        if (lane < 16) {
          u.m.t[b0 + t][b0 + s] = col[t];
          if (warp == 0) u.m.t[s][16 + t] = 0.f;  // the block above the diagonal: zero
        }
      }
    }
    __syncthreads();
    // The block below: X = L10 T00, then T10 = -T11 X (float32, k ascending), two entries a thread.
    for (int idx = tid; idx < 256; idx += 128) {
      const int i = idx >> 4, j = idx & 15;
      float x = 0.f;
      for (int k = j; k < 16; ++k) x = __fmaf_rn(u.m.l[16 + i][k], u.m.t[k][j], x);
      u.m.x[i][j] = x;
    }
    __syncthreads();
    for (int idx = tid; idx < 256; idx += 128) {
      const int i = idx >> 4, j = idx & 15;
      float y = 0.f;
      for (int k = 0; k <= i; ++k) y = __fmaf_rn(u.m.t[16 + i][16 + k], u.m.x[k][j], y);
      u.m.t[16 + i][j] = -y;
    }
    __syncthreads();
    float* T = cs.T + at * kChunk * kChunk;
    for (int idx = tid; idx < kChunk * kChunk; idx += 128) T[idx] = u.m.t[idx / kChunk][idx % kChunk];
    if (tid < kChunk) {
      cs.gam[at * kChunk + tid] = sg[tid];
      cs.beta[at * kChunk + tid] = sb[tid];
    }
  }
}

// Shared memory of the scan (49,408 B: swizzled tiles + s/uf union fit two blocks per 99/100 KB SM with zero extra barriers).
struct ScanSmem {
  SwizTile<kChunk, kDK> k;  // K of the chunk
  union {
    SwizTile<kBV, kDK> s;     // the state at the chunk's start (dead after warp_mma_kq, two barriers before uf)
    SwizTile<kBV, kChunk> uf; // (f U)^T, f_t = exp(gamma_last - gamma_t)
  };
  SwizTile<kChunk, kChunk> t;  // T
  SwizTile<kChunk, kChunk> a;  // A
  SwizTile<kBV, kChunk> yu;    // Y^T, then U^T
  alignas(16) float q[kChunk][kDK];  // Q of the chunk (float32, XOR-swizzled)
  float gam[kChunk], beta[kChunk];
};

constexpr int kScanWarps = 8;  // the products by chunk row and value column: warp w rows 16 (w % 2) .., columns
                               // 16 (w / 2) ..; the state: warp w rows 16 (w % 4) .., columns 64 (w / 4) ..

// grid (nv, dv / kBV); block 32 kScanWarps.  The state (float32) stays in registers.
__global__ void __launch_bounds__(32 * kScanWarps, 2) k_gdn_chunk_scan(float* S, size_t s_stride, int sel, GdnBufs bufs,
                                                                       const DevParams* P, float* out, int rows,
                                                                       int off, int chunks, ChunkScratch cs,
                                                                       int snap_sel, int snap_chunks) {
  constexpr int NT = 32 * kScanWarps;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  ScanSmem& sm = *reinterpret_cast<ScanSmem*>(smem_raw);
  const GdnDims d = bufs.d;
  const int h = blockIdx.x, j0 = blockIdx.y * kBV, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, tq = lane & 3, rw = 16 * (warp & 3), cw = warp >> 2;
  const int pr = 16 * (warp & 1), pc = 16 * (warp >> 1);
  const int kh = d.key_head(h), vd = d.nv * d.dv;
  const float* B = bufs.base;
  float* Sh = S + size_t(slot_of(P, sel)) * s_stride + size_t(h) * d.dv * kDK;
  float* Ssnap = snap_sel >= 0 ? S + size_t(slot_of(P, snap_sel)) * s_stride + size_t(h) * d.dv * kDK : nullptr;
  // state rows j = j0 + rw + g (+ 8), columns i = 64 cw + 8 n + 2 tq (+ 1): the C layout of (f U)^T K
  float st[kDK / 16][4];
  auto store_state = [&](float* dst) {
#pragma unroll
    for (int n = 0; n < kDK / 16; ++n) {
      const int j = j0 + rw + g, i = 64 * cw + 8 * n + 2 * tq;
      *reinterpret_cast<float2*>(dst + size_t(j) * kDK + i) = make_float2(st[n][0], st[n][1]);
      *reinterpret_cast<float2*>(dst + size_t(j + 8) * kDK + i) = make_float2(st[n][2], st[n][3]);
    }
  };
#pragma unroll
  for (int n = 0; n < kDK / 16; ++n) {
    const int j = j0 + rw + g, i = 64 * cw + 8 * n + 2 * tq;
    const float2 lo = *reinterpret_cast<const float2*>(Sh + size_t(j) * kDK + i);
    const float2 hi = *reinterpret_cast<const float2*>(Sh + size_t(j + 8) * kDK + i);
    st[n][0] = lo.x, st[n][1] = lo.y, st[n][2] = hi.x, st[n][3] = hi.y;
  }
  if (Ssnap && snap_chunks == 0) store_state(Ssnap);  // the pass starts at the boundary
  // A chunk's inputs (K, T, A, gamma, beta, V) do not depend on the chunk before: the next chunk's are read into
  // registers while this one computes, so their latency leaves the chain of chunks (only the state carries over).
  // The same values staged the same way: the same bits.
  constexpr int KQ = kChunk * kDK / 4 / NT, TQ = kChunk * kChunk / 2 / NT;  // a thread's float4s of K, float2s of T
  static_assert(KQ * NT == kChunk * kDK / 4 && TQ * NT == kChunk * kChunk / 2, "the chunk's inputs over the threads");
  struct Ahead {
    float4 k[KQ];
    float2 t[TQ], a[TQ];
    float gam, beta, vv[2][4];
  };
  auto fetch_kt = [&](int c, Ahead& f) {
    const int r0 = c * kChunk - off;
    const size_t at = size_t(c) * d.nv + h;
    const float* src = B + bufs.o_kn();
#pragma unroll
    for (int q = 0; q < KQ; ++q) {
      const int idx = tid + q * NT, t = idx / (kDK / 4), i = (idx % (kDK / 4)) * 4, r = r0 + t;
      f.k[q] = r >= 0 && r < rows ? *reinterpret_cast<const float4*>(src + (size_t(r) * d.nk + kh) * kDK + i)
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (int q = 0; q < TQ; ++q) {
      const int idx = tid + q * NT, t = idx / (kChunk / 2), s2 = 2 * (idx % (kChunk / 2));
      f.t[q] = *reinterpret_cast<const float2*>(cs.T + at * kChunk * kChunk + t * kChunk + s2);
      f.a[q] = *reinterpret_cast<const float2*>(cs.A + at * kChunk * kChunk + t * kChunk + s2);
    }
    if (tid < kChunk) f.gam = cs.gam[at * kChunk + tid], f.beta = cs.beta[at * kChunk + tid];
  };
  auto fetch_v = [&](int c, Ahead& f) {
    const int r0 = c * kChunk - off;
    const float* vbase = B + bufs.o_v() + h * d.dv + j0;
    // this thread's V elements (Y below): rows pr + g (+ 8), columns pc + 8 n + 2 tq (+ 1)
    const int r_lo = r0 + pr + g, r_hi = r_lo + 8;
    const bool in_lo = r_lo >= 0 && r_lo < rows, in_hi = r_hi >= 0 && r_hi < rows;
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      const int j = pc + 8 * n + 2 * tq;
      const float2 v0 = in_lo ? *reinterpret_cast<const float2*>(vbase + size_t(r_lo) * vd + j) : make_float2(0.f, 0.f);
      const float2 v1 = in_hi ? *reinterpret_cast<const float2*>(vbase + size_t(r_hi) * vd + j) : make_float2(0.f, 0.f);
      f.vv[n][0] = v0.x, f.vv[n][1] = v0.y, f.vv[n][2] = v1.x, f.vv[n][3] = v1.y;
    }
  };
  // Q (float32) by cp.async straight into shared memory: the next chunk's once every warp has read this one's
  auto fetch_q = [&](int c) {
    const int r0 = c * kChunk - off;
    const float* src = B + bufs.o_qn();
    for (int idx = tid; idx < kChunk * kDK / 4; idx += NT) {
      const int t = idx / (kDK / 4), i = (idx % (kDK / 4)) * 4, r = r0 + t;
      const bool in = r >= 0 && r < rows;
      cp_async16(&sm.q[t][sw_q(t, i)], in ? src + (size_t(r) * d.nk + kh) * kDK + i : src, in);  // outside: zeros
    }
    cp_async_commit();
  };
  Ahead cur;
  if (chunks > 0) fetch_kt(0, cur), fetch_v(0, cur), fetch_q(0);
  for (int c = 0; c < chunks; ++c) {
    const int r0 = c * kChunk - off;
    // the chunk's inputs, split: K, T, A; gamma and beta; the state
#pragma unroll
    for (int q = 0; q < KQ; ++q) {
      const int idx = tid + q * NT, t = idx / (kDK / 4), i = (idx % (kDK / 4)) * 4;
      sm.k.put2(t, i, cur.k[q].x, cur.k[q].y);
      sm.k.put2(t, i + 2, cur.k[q].z, cur.k[q].w);
    }
#pragma unroll
    for (int q = 0; q < TQ; ++q) {
      const int idx = tid + q * NT, t = idx / (kChunk / 2), s2 = 2 * (idx % (kChunk / 2));
      sm.t.put2(t, s2, cur.t[q].x, cur.t[q].y);
      sm.a.put2(t, s2, cur.a[q].x, cur.a[q].y);
    }
    if (tid < kChunk) sm.gam[tid] = cur.gam, sm.beta[tid] = cur.beta;
#pragma unroll
    for (int n = 0; n < kDK / 16; ++n) {
      const int j = rw + g, i = 64 * cw + 8 * n + 2 * tq;
      sm.s.put2(j, i, st[n][0], st[n][1]);
      sm.s.put2(j + 8, i, st[n][2], st[n][3]);
    }
    if (c + 1 < chunks) fetch_kt(c + 1, cur);
    cp_async_wait<0>();  // this chunk's Q
    __syncthreads();
    const float ga = expf(sm.gam[pr + g]), gb = expf(sm.gam[pr + g + 8]);
    // P = K S^T, then Y = beta (V - G P) -> Y^T; and G (Q S^T), the start of O
    float o[2][4] = {};
    {
      float p[2][4] = {};
      warp_mma_kq<2>(p, o, sm.k, sm.q, pr, sm.s, pc, lane);
      const float ba = sm.beta[pr + g], bb = sm.beta[pr + g + 8];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = pr + g + 8 * (e >> 1), j = pc + 8 * n + 2 * tq + (e & 1);
          sm.yu.put(j, t, (e < 2 ? ba : bb) * (cur.vv[n][e] - (e < 2 ? ga : gb) * p[n][e]));
        }
        o[n][0] *= ga, o[n][1] *= ga, o[n][2] *= gb, o[n][3] *= gb;
      }
      if (c + 1 < chunks) fetch_v(c + 1, cur);
    }
    __syncthreads();
    if (c + 1 < chunks) fetch_q(c + 1);  // every warp has read this chunk's Q
    // U = T Y -> U^T and (f U)^T
    {
      float uu[2][4] = {};
      warp_mma<2>(uu, sm.t, pr, sm.yu, pc, kChunk, lane);
      __syncthreads();  // every warp has read Y^T
      const float glast = sm.gam[kChunk - 1];
      const float fa = expf(glast - sm.gam[pr + g]), fb = expf(glast - sm.gam[pr + g + 8]);
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int t = pr + g + 8 * (e >> 1), j = pc + 8 * n + 2 * tq + (e & 1);
          sm.yu.put(j, t, uu[n][e]);
          sm.uf.put(j, t, (e < 2 ? fa : fb) * uu[n][e]);
        }
    }
    __syncthreads();
    // O = G (Q S^T) + A U
    warp_mma<2>(o, sm.a, pr, sm.yu, pc, kChunk, lane);
    {
      float* obase = out + h * d.dv + j0;
      const int r_lo = r0 + pr + g, r_hi = r_lo + 8;
      const bool in_lo = r_lo >= 0 && r_lo < rows, in_hi = r_hi >= 0 && r_hi < rows;
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        const int j = pc + 8 * n + 2 * tq;
        if (in_lo) *reinterpret_cast<float2*>(obase + size_t(r_lo) * vd + j) = make_float2(o[n][0], o[n][1]);
        if (in_hi) *reinterpret_cast<float2*>(obase + size_t(r_hi) * vd + j) = make_float2(o[n][2], o[n][3]);
      }
    }
    // S' = G_last S + (f U)^T K
    const float decay = expf(sm.gam[kChunk - 1]);
#pragma unroll
    for (int n = 0; n < kDK / 16; ++n)
#pragma unroll
      for (int e = 0; e < 4; ++e) st[n][e] *= decay;
    warp_mma_kb<kDK / 16>(st, sm.uf, rw, sm.k, 64 * cw, kChunk, lane);
    if (Ssnap && c + 1 == snap_chunks) store_state(Ssnap);  // the state at the boundary
    __syncthreads();  // the next chunk restages every buffer
  }
  store_state(Sh);
}

}  // namespace

int gdn_chunk_align() { return kChunk; }
bool gdn_chunk_device() {
  static const bool yes = [&] {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    const int cc = 10 * major + minor;
    // match llama.cpp's chunked bf16 schedule (GA10X, Ada, consumer Blackwell, DGX Spark);
    // datacenter cards (A100/H100/B200) stay on the float32 recurrence to match llama.cpp precision.
    return cc == 86 || cc == 89 || cc == 120 || cc == 121;
  }();
  return yes;
}
bool gdn_chunk_takes(const GdnDims& d) {
  return d.dk == kDK && d.dv % kBV == 0 && d.nk >= 1 && d.nv % d.nk == 0 && gdn_chunk_device();
}
int gdn_chunk_rows() { return 2 * kChunk; }

bool gdn_chunk(float* S, size_t s_stride, int sel, GdnBufs bufs, const DevParams* P, int pos0, float* out, int rows,
               cudaStream_t st, int snap_sel, int snap_row) {
  const GdnDims& d = bufs.d;
  if (rows < 1 || bufs.slot_stride != 0 || !gdn_chunk_takes(d) || reinterpret_cast<uintptr_t>(S) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(out) % 8 != 0 || s_stride % 4 != 0)
    return false;
  if (rows > bufs.rows_cap) return false;
  const int off = pos0 % kChunk, chunks = (off + rows + kChunk - 1) / kChunk;
  // The intermediates go into [0, bufs.o_v()) (sized for max_chunks of this prefill buffer).
  auto need = [&](int nc) { return size_t(nc) * d.nv * (2 * kChunk * kChunk + 2 * kChunk) * sizeof(float); };
  if (need((kChunk - 1 + bufs.rows_cap + kChunk - 1) / kChunk) > bufs.o_v() * sizeof(float))
    return false;
  const size_t ph = size_t(chunks) * d.nv;
  ChunkScratch cs;
  cs.T = bufs.base;
  cs.A = cs.T + ph * kChunk * kChunk;
  cs.gam = cs.A + ph * kChunk * kChunk;
  cs.beta = cs.gam + ph * kChunk;
  k_gdn_chunk_tri<<<dim3(chunks, d.nk), 128, 0, st>>>(bufs, rows, off, cs);
  static const bool smem_ok = set_dynamic_smem(k_gdn_chunk_scan, sizeof(ScanSmem), true);
  if (!smem_ok) return false;
  // snap_row: a chunk boundary (pos0 + snap_row a multiple of kChunk) in [0, rows); its state goes to slot snap_sel.
  const bool snap = snap_sel >= 0 && snap_row >= 0 && snap_row < rows && (pos0 + snap_row) % kChunk == 0;
  k_gdn_chunk_scan<<<dim3(d.nv, d.dv / kBV), 32 * kScanWarps, sizeof(ScanSmem), st>>>(
      S, s_stride, sel, bufs, P, out, rows, off, chunks, cs, snap ? snap_sel : -1, snap ? (off + snap_row) / kChunk : -1);
  return true;
}

}  // namespace flyx::cuda
