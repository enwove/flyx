// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Prefill GEMM with NVFP4 activations (W4A4: NVFP4 weights whose file says
// their inputs are NVFP4, in passes of more than 16 rows) on
// the block-scaled FP4 tensor cores of sm_120a:
// mma.sync m16n8k64 kind::mxf4nvf4, a ue4m3 scale per 16 values of A and of
// B, float32 accumulation.  Built for sm_120a only (its own library); other
// devices keep the bf16 path.
//
// Activations: each row gets a scale g = 6 * 448 / max|x| so its 16-value
// blocks' scales e4m3(max|block| / 6 * g) stay in range; the values are e2m1
// of x g / scale, nearest.  y = (sum over K of the block-scaled products) *
// RS[n] / g: the weights' per-tensor scale and the row's.
//
// Operand layouts on this instruction (measured): A and B as for bf16
// m16n8k16 with 4 bytes per 32-bit register (8 values), so a 64-value k step
// of a row is 32 bytes and ldmatrix loads it; the scales of A row g are a
// 32-bit word (4 blocks of 16) in lane 4 g, of row g + 8 in lane 4 g + 1; of B
// column g in lane 4 g.
#include <cstdio>
#include <cstring>
#include <type_traits>

#include <cuda.h>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kBM = 128;   // block tile rows (2 warps of 64)
constexpr int kWM = 2, kWN = 4, kThreads = 32 * kWM * kWN;

// BN: block tile columns (4 warps of BN / 4); ST: pipeline stages; KB: bytes of K a row and stage (64: 128 values,
// 128: 256; K a multiple of 2 KB).  Every configuration adds a column's 64-value k steps in the same order: the same
// bits.
template <int BN, int ST, int KB>
struct W4Smem {
  uint8_t a[ST][kBM][KB];
  uint8_t b[ST][BN][KB];
  uint8_t sa[ST][kBM][KB / 8];  // the block scales of a row's 2 KB values, one per 16
  uint8_t sb[ST][BN][KB / 8];
};

// The 16-byte chunk c of tile row r: in rows of 64 bytes at chunk c ^ ((r >> 1) & 3), of 128 at c ^ (r & 7); the
// 8 rows an ldmatrix phase reads hit 8 different bank groups.
template <int KB>
__device__ __forceinline__ int swz(int r, int c) {
  return (KB == 64 ? c ^ ((r >> 1) & 3) : c ^ (r & 7)) << 4;
}

__device__ __forceinline__ void mma_fp4(float* c, const uint32_t* a, const uint32_t* b, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(uint16_t(0)),
        "h"(uint16_t(0)), "r"(sfb), "h"(uint16_t(0)), "h"(uint16_t(0)));
}

// y[M, N] (=|+=) (A x B^T): A the quantized activations [M][K/2] with block
// scales As [M][K/16] and row factors Ar [M] (1 / g); B the NVFP4 weights.
// grid (M / 128, N / 128) walking M first (consecutive blocks share a weight tile).
// GLU: a gate|up weight, columns as glu_row (device.cuh), glu [M][N / 2] = bf16(silu(g) u) instead of y, and
// glu_amax[r] raised to the largest |value| stored in row r (zeroed by the caller).
template <int BN, int ST, int KB, bool GLU>
__global__ void __launch_bounds__(kThreads) k_gemm_w4a4(const uint8_t* __restrict__ Aq, const uint8_t* __restrict__ As,
                                                       const float* __restrict__ Ar, const uint8_t* __restrict__ Q,
                                                       const uint8_t* __restrict__ S, const float* __restrict__ RS,
                                                       int N, int K, float* __restrict__ y, int ldy, int M,
                                                       int accumulate, bf16* __restrict__ glu,
                                                       float* __restrict__ glu_amax, bf16* __restrict__ y16) {
  constexpr int NI = BN / 4 / 8;  // n tiles per warp
  extern __shared__ __align__(16) unsigned char smem_raw[];
  W4Smem<BN, ST, KB>& sm = *reinterpret_cast<W4Smem<BN, ST, KB>*>(smem_raw);
  constexpr int CH = KB / 16, KSTEPS = KB / 32;  // 16-byte chunks a tile row; 64-value k steps a stage
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / kWN, wn = warp % kWN;
  const int m0 = blockIdx.x * kBM, n0 = blockIdx.y * BN, ktiles = K / (2 * KB), RB = K / 2, KS = K / 16;
  auto load_stage = [&](int st, int kt) {
#pragma unroll
    for (int i = 0; i < kBM * CH / kThreads; ++i) {  // A: 128 rows x CH chunks
      const int idx = tid + i * kThreads, r = idx / CH, c = idx % CH, gm = m0 + r;
      cp_async16(&sm.a[st][r][swz<KB>(r, c)], gm < M ? Aq + size_t(gm) * RB + kt * KB + 16 * c : Aq, gm < M);
    }
#pragma unroll
    for (int i = 0; i < BN * CH / kThreads; ++i) {  // B: BN rows x CH chunks
      const int idx = tid + i * kThreads, r = idx / CH, c = idx % CH, gn = n0 + r, wr = GLU ? glu_row(gn, N >> 1) : gn;
      cp_async16(&sm.b[st][r][swz<KB>(r, c)], gn < N ? Q + size_t(wr) * RB + kt * KB + 16 * c : Q, gn < N);
    }
    for (int r = tid; r < kBM + BN; r += kThreads) {
      if (r < kBM) {
        const int gm = m0 + r;
        const uint8_t* src = gm < M ? As + size_t(gm) * KS + kt * (KB / 8) : As;
        if constexpr (KB == 64) cp_async8(&sm.sa[st][r][0], src, gm < M);
        else cp_async16(&sm.sa[st][r][0], src, gm < M);
      } else {
        const int rb = r - kBM, gn = n0 + rb, wr = GLU ? glu_row(gn, N >> 1) : gn;
        const uint8_t* src = gn < N ? S + size_t(wr) * KS + kt * (KB / 8) : S;
        if constexpr (KB == 64) cp_async8(&sm.sb[st][rb][0], src, gn < N);
        else cp_async16(&sm.sb[st][rb][0], src, gn < N);
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
    cp_async_commit();
  }
  const int g = lane >> 2;
  for (int kt = 0; kt < ktiles; ++kt) {
    const int st = kt % ST;
    cp_async_wait<ST - 2>();
    __syncthreads();
    {
      const int nk = kt + ST - 1;
      if (nk < ktiles) load_stage(nk % ST, nk);
      cp_async_commit();
    }
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {  // the stage's 64-value k steps, in order
      uint32_t af[4][4], bf[NI][2], sfa[4], sfb[NI];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi) {
        const int r = wm * 64 + mi * 16 + (lane & 15), c = 2 * ks + (lane >> 4);
        ldmatrix_x4(af[mi], &sm.a[st][r][swz<KB>(r, c)]);
        sfa[mi] = *reinterpret_cast<const uint32_t*>(&sm.sa[st][wm * 64 + mi * 16 + g + 8 * (lane & 1)][4 * ks]);
      }
#pragma unroll
      for (int ni = 0; ni < NI; ni += 2) {
        const int r = wn * (BN / 4) + ni * 8 + (lane & 7) + 8 * (lane >> 4), c = 2 * ks + ((lane >> 3) & 1);
        uint32_t b4[4];
        ldmatrix_x4(b4, &sm.b[st][r][swz<KB>(r, c)]);
        bf[ni][0] = b4[0], bf[ni][1] = b4[1], bf[ni + 1][0] = b4[2], bf[ni + 1][1] = b4[3];
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
        sfb[ni] = *reinterpret_cast<const uint32_t*>(&sm.sb[st][wn * (BN / 4) + ni * 8 + g][4 * ks]);
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_fp4(acc[mi][ni], af[mi], bf[ni], sfa[mi], sfb[ni]);
    }
  }
  cp_async_wait<0>();
  const int t = lane & 3;
  if constexpr (GLU) {  // tiles ni (gate) and ni + 1 (up) hold the columns j, j + 1 of the output
    static_assert(NI % 2 == 0, "gate and up tiles in pairs");
    const int I = N >> 1;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = m0 + wm * 64 + mi * 16 + g + 8 * h;
        float m = 0.f;
#pragma unroll
        for (int ni = 0; ni < NI; ni += 2) {
          const int j = glu_col(n0 + wn * (BN / 4) + ni * 8 + 2 * t);
          if (r < M && j < I) {
            const float g0 = acc[mi][ni][2 * h] * RS[j] * Ar[r], g1 = acc[mi][ni][2 * h + 1] * RS[j + 1] * Ar[r];
            const float u0 = acc[mi][ni + 1][2 * h] * RS[I + j] * Ar[r];
            const float u1 = acc[mi][ni + 1][2 * h + 1] * RS[I + j + 1] * Ar[r];
            const __nv_bfloat162 v = __floats2bfloat162_rn(silu(g0) * u0, silu(g1) * u1);
            *reinterpret_cast<__nv_bfloat162*>(glu + size_t(r) * I + j) = v;
            m = fmaxf(m, fmaxf(fabsf(__low2float(v)), fabsf(__high2float(v))));
          }
        }
        // the row's four lanes (lane % 4), then one atomic: a maximum of non-negative floats is their bits' maximum
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        if (t == 0 && r < M && m > 0.f) atomicMax(reinterpret_cast<int*>(glu_amax + r), __float_as_int(m));
      }
  } else {
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int r = m0 + wm * 64 + mi * 16 + g + (e >> 1) * 8;
          const int c = n0 + wn * (BN / 4) + ni * 8 + 2 * t + (e & 1);
          if (r < M && c < N) {
            const float v = acc[mi][ni][e] * RS[c] * Ar[r];
            if (y16) {  // rounded to bf16; accumulate: the sum in float32 first
              bf16* d16 = y16 + size_t(r) * ldy + c;
              *d16 = __float2bfloat16_rn(accumulate ? __fadd_rn(__bfloat162float(*d16), v) : v);  // no contraction
              continue;
            }
            float* dst = y + size_t(r) * ldy + c;
            *dst = accumulate ? *dst + v : v;
          }
        }
  }
}

// One row per block (128 threads): the row scale, then 16-value blocks; from the float32 values, one rounding
// (bf16 first before).
__global__ void __launch_bounds__(128) k_quant_rows_nvfp4(const float* __restrict__ x, int ldx, int K,
                                                          uint8_t* __restrict__ q, uint8_t* __restrict__ s,
                                                          float* __restrict__ inv) {
  const int r = blockIdx.x;
  const float* xr = x + size_t(r) * ldx;
  float amax = 0.f;
  for (int i = threadIdx.x; i < K; i += 128) amax = fmaxf(amax, fabsf(xr[i]));
  amax = block_max(amax);
  const float gs = amax > 0.f ? __fdiv_rn(kNvfp4RowRange, amax) : 1.f;
  if (threadIdx.x == 0) inv[r] = __fdiv_rn(1.f, gs);
  const bool vec = (reinterpret_cast<uintptr_t>(xr) & 15) == 0;  // a block's 16 values as four 16-byte loads
  for (int b = threadIdx.x; b < K / 16; b += 128) {
    float v[16], bmax = 0.f;
    if (vec) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float4 f = *reinterpret_cast<const float4*>(xr + 16 * b + 4 * c);
        v[4 * c] = f.x, v[4 * c + 1] = f.y, v[4 * c + 2] = f.z, v[4 * c + 3] = f.w;
      }
    } else {
#pragma unroll
      for (int e = 0; e < 16; ++e) v[e] = xr[16 * b + e];
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) bmax = fmaxf(bmax, fabsf(v[e]));
    const uint8_t sc = nvfp4_block_scale(bmax, gs);
    const float sv = e4m3_to_float(sc), f = sv > 0.f ? __fdiv_rn(gs, sv) : 0.f;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) lo |= uint32_t(e2m1_nearest(__fmul_rn(v[e], f))) << (4 * e);
#pragma unroll
    for (int e = 0; e < 8; ++e) hi |= uint32_t(e2m1_nearest(__fmul_rn(v[8 + e], f))) << (4 * e);
    *reinterpret_cast<uint2*>(q + size_t(r) * (K / 2) + 8 * b) = make_uint2(lo, hi);
    s[size_t(r) * (K / 16) + b] = sc;
  }
}

// The same rounding from bf16 rows (the SwiGLU rows of the gate|up GEMM) whose largest |value| per row is given
// (one read of the row) or not (a bf16 gate|up GEMM's: found first); its codes are k_quant_rows_nvfp4's on the rows
// as float32.
__global__ void __launch_bounds__(128) k_quant_rows_nvfp4_bf16(const bf16* __restrict__ x, int ldx, int K,
                                                               const float* __restrict__ amax_in,
                                                               uint8_t* __restrict__ q, uint8_t* __restrict__ s,
                                                               float* __restrict__ inv) {
  const int r = blockIdx.x;
  const bf16* xr = x + size_t(r) * ldx;
  float amax = 0.f;
  if (amax_in) {
    amax = amax_in[r];
  } else {
    for (int i = threadIdx.x; i < K; i += 128) amax = fmaxf(amax, fabsf(__bfloat162float(xr[i])));
    amax = block_max(amax);
  }
  const float gs = amax > 0.f ? __fdiv_rn(kNvfp4RowRange, amax) : 1.f;
  if (threadIdx.x == 0) inv[r] = __fdiv_rn(1.f, gs);
  const bool vec = (reinterpret_cast<uintptr_t>(xr) & 15) == 0;  // a block's 16 values as two 16-byte loads
  for (int b = threadIdx.x; b < K / 16; b += 128) {
    float v[16], bmax = 0.f;
    if (vec) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint4 u = *reinterpret_cast<const uint4*>(xr + 16 * b + 8 * c);
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
          v[8 * c + 2 * e] = __uint_as_float(w[e] << 16), v[8 * c + 2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
      }
    } else {
#pragma unroll
      for (int e = 0; e < 16; ++e) v[e] = __bfloat162float(xr[16 * b + e]);
    }
#pragma unroll
    for (int e = 0; e < 16; ++e) bmax = fmaxf(bmax, fabsf(v[e]));
    const uint8_t sc = nvfp4_block_scale(bmax, gs);
    const float sv = e4m3_to_float(sc), f = sv > 0.f ? __fdiv_rn(gs, sv) : 0.f;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int e = 0; e < 8; ++e) lo |= uint32_t(e2m1_nearest(__fmul_rn(v[e], f))) << (4 * e);
#pragma unroll
    for (int e = 0; e < 8; ++e) hi |= uint32_t(e2m1_nearest(__fmul_rn(v[8 + e], f))) << (4 * e);
    *reinterpret_cast<uint2*>(q + size_t(r) * (K / 2) + 8 * b) = make_uint2(lo, hi);
    s[size_t(r) * (K / 16) + b] = sc;
  }
}

}  // namespace

// ---- configuration 4: a warp-specialized TMA pipeline ----------------------------------------------------------
// Adapted from basalt's src/prefill/fp4_dense.cu (MIT, Copyright (c) 2026 Jesús Gulfo), which adapts Infernix's
// nvfp4_a4_tma_kernel (Apache-2.0, Ian Ranson and the Infernix contributors): TMA loads into a three-stage mbarrier
// full/empty ring, one producer warpgroup (setmaxnreg) beside eight consumer warps, 128 weight rows x 256 rows x 128
// values of K a stage, the tiles of rows walking fastest so the blocks that run together share a weight tile.  Here
// the operands are this file's: A codes [M][K/2] and block scales [M][K/16], W codes [N][K/2] and block scales
// [N][K/16], row-major (a 16-byte column of scales, 256 values of K, serves two stages), and the epilogue is the one
// above: the same 64-value k steps in the same order into the same accumulators, the same products after, so its bits
// are the other configurations'.  K % 256 == 0.
namespace tma4 {

constexpr int kBT = 256;                 // rows (tokens) a block
constexpr int kBR = 128;                 // weight rows a block (GLU: 64 gate rows and the same 64 up rows)
constexpr int kStages = 3;
constexpr int kConsumerWarps = 8;        // 4 quarters of the rows x 2 halves of the weight rows
constexpr int kProducerThreads = 128;    // one warpgroup, so setmaxnreg can move registers to the consumers
constexpr int kThreads = kConsumerWarps * 32 + kProducerThreads;
constexpr int kWarpTokens = kBT / 4, kWarpRows = kBR / 2;
constexpr int kMmaTokens = kWarpTokens / 16, kMmaRows = kWarpRows / 8;
constexpr int kK64 = 2;                  // 64-value MMA steps a stage (128 values of K)
constexpr int kScaleSlots = (kStages + 2) / 2;  // a 16-byte column of scales serves two stages

struct alignas(64) TmaDesc {
  unsigned long long w[16];
};
static_assert(sizeof(TmaDesc) == sizeof(CUtensorMap), "a descriptor argument is the descriptor's size");

struct Params {
  TmaDesc a_codes, a_scales, b_codes[2], b_scales[2];
  const float* Ar;
  const float* RS;
  float* y;
  bf16* glu;
  float* amax;  // GLU: each row's largest |stored value|, raised atomically (zeroed by the caller)
  bf16* y16;    // non-null: the product rounded to bf16 there [M][ldy] instead of y (no accumulation)
  int ldy, M, N, I, K, tok_tiles, accumulate;
};

struct alignas(128) Shared {
  alignas(128) uint8_t a_codes[kStages][kBT * 64];
  alignas(128) uint8_t b_codes[kStages][kBR * 64];
  alignas(16) uint8_t a_scales[kScaleSlots][kBT * 16];
  alignas(16) uint8_t b_scales[kScaleSlots][kBR * 16];
  alignas(8) uint64_t full[kStages];
  alignas(8) uint64_t empty[kStages];
  float ar[kBT];  // the tile's row scales Ar (zero past M) and weight row scales RS (GLU: gate, then up)
  float rs[kBR];
};
constexpr int kSmem = int(sizeof(Shared));
constexpr int kFetchStages = 12;  // stages before the K loop's end an accumulating call asks for its output's lines
static_assert(kSmem <= 99 * 1024, "one block's shared memory");

__device__ __forceinline__ uint32_t saddr(const void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(saddr(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred done;\nwait_loop_%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
      "@done bra wait_done_%=;\nbra wait_loop_%=;\nwait_done_%=:\n}\n" ::"r"(saddr(b)),
      "r"(phase), "r"(0x989680u)
      : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(saddr(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(saddr(b)), "r"(bytes) : "memory");
}
// a consumer's generic-proxy reads of a stage ordered before the producer's async-proxy refill of it
__device__ __forceinline__ void fence_reads() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void tma_load_2d(void* dst, const TmaDesc* map, int c0, int c1, uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          saddr(dst)),
      "l"(reinterpret_cast<const CUtensorMap*>(map)), "r"(c0), "r"(c1), "r"(saddr(bar))
      : "memory");
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
template <int N, class F>
__device__ __forceinline__ void with_live(int live, F&& f) {
  if (live >= N) f(std::integral_constant<int, N>{});
  else if constexpr (N > 1) with_live<N - 1>(live, f);
}

template <bool GLU>
__global__ void __launch_bounds__(kThreads, 1) k_gemm_w4a4_tma(const __grid_constant__ Params p) {
  constexpr int branches = GLU ? 2 : 1, loaded_rows = kBR / branches;
  extern __shared__ __align__(128) unsigned char shared_bytes[];
  Shared& sh = *reinterpret_cast<Shared*>(shared_bytes);
  const int tok_tile = int(blockIdx.x) % p.tok_tiles;
  const int row_begin = (int(blockIdx.x) / p.tok_tiles) * loaded_rows, token_begin = tok_tile * kBT;
  const int ktiles = p.K / 128;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) mbar_init(&sh.full[s], 1), mbar_init(&sh.empty[s], kConsumerWarps);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (threadIdx.x < kProducerThreads) {  // the producer warp; the warpgroup's other three give up their registers
    asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
    if (threadIdx.x >= 32) return;
    // the producer warp keeps the ring's pace; lane 0 issues the copies, every lane a share of the prefetches
    const int fetch_at = ktiles > kFetchStages ? ktiles - kFetchStages : 0;
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages;
      mbar_wait(&sh.empty[stage], 1u ^ ((kt / kStages) & 1u));
      if (!GLU && p.accumulate && kt == fetch_at) {  // the output tile read back by the epilogue, toward L2 ahead
#pragma unroll 4
        for (int i = 0; i < kBT * 4 / 32; ++i) {  // a row's 128 values: 4 lines of float32, 2 of bf16
          const int t = token_begin + int(threadIdx.x) * (kBT / 32) + (i >> 2), c = row_begin + (i & 3) * 32;
          if (t >= p.M || c >= p.N) continue;
          if (!p.y16) asm volatile("prefetch.global.L2 [%0];" ::"l"(p.y + size_t(t) * p.ldy + c) : "memory");
          else if ((i & 1) == 0)
            asm volatile("prefetch.global.L2 [%0];" ::"l"(p.y16 + size_t(t) * p.ldy + c) : "memory");
        }
      }
      if (threadIdx.x != 0) continue;
      const bool scales = (kt & 1) == 0;  // a column of scales: fetched on the even stage of its two
      const uint32_t tx = kBT * 64 + kBR * 64 + (scales ? kBT * 16 + kBR * 16 : 0);
      mbar_expect_tx(&sh.full[stage], tx);
      tma_load_2d(sh.a_codes[stage], &p.a_codes, kt * 64, token_begin, &sh.full[stage]);
#pragma unroll
      for (int b = 0; b < branches; ++b)
        tma_load_2d(sh.b_codes[stage] + b * loaded_rows * 64, &p.b_codes[b], kt * 64, row_begin, &sh.full[stage]);
      if (scales) {
        const int slot = (kt / 2) % kScaleSlots;
        tma_load_2d(sh.a_scales[slot], &p.a_scales, (kt / 2) * 16, token_begin, &sh.full[stage]);
#pragma unroll
        for (int b = 0; b < branches; ++b)
          tma_load_2d(sh.b_scales[slot] + b * loaded_rows * 16, &p.b_scales[b], (kt / 2) * 16, row_begin,
                      &sh.full[stage]);
      }
    }
    return;
  }

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");
  const int ct = int(threadIdx.x) - kProducerThreads, lane = ct & 31, warp = ct >> 5;
  {  // the epilogue's scales into shared memory while the first stages come in
    const int t = token_begin + ct;
    sh.ar[ct] = t < p.M ? p.Ar[t] : 0.f;
    if (ct < kBR) {
      const int b = ct / loaded_rows, row = row_begin + ct % loaded_rows;
      sh.rs[ct] = row < (GLU ? p.I : p.N) ? p.RS[b * p.I + row] : 0.f;
    }
    asm volatile("bar.sync 1, %0;" ::"n"(kConsumerWarps * 32) : "memory");
  }
  const int warp_m = warp >> 1, warp_n = warp & 1;  // quarter of the rows, half of the weight rows
  const int a_matrix = lane >> 3, a_row_offset = (lane & 7) + ((a_matrix & 1) << 3), a_column_byte = (a_matrix >> 1) * 16;
  const int b_row_offset = lane & 7, b_column_byte = ((lane >> 3) & 1) * 16;
  const int sfa_row = ((lane & 1) << 3) | (lane >> 2), sfb_row = lane >> 2;
  float acc[kMmaTokens][kMmaRows][4] = {};
  // `live` of the warp's 16-row fragments hold rows of the call: the others run no MMA and are never stored
  const auto consume = [&](const auto live) {
#pragma unroll 1
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages, slot = (kt / 2) % kScaleSlots;
      mbar_wait(&sh.full[stage], (kt / kStages) & 1u);
#pragma unroll
      for (int k64 = 0; k64 < kK64; ++k64) {
        uint32_t af[kMmaTokens][4], bf[kMmaRows][2], asc[kMmaTokens], bsc[kMmaRows];
        const int sw = ((kt & 1) * kK64 + k64) * 4;  // the step's 4 scale bytes in a row's column of 16
#pragma unroll
        for (int mm = 0; mm < kMmaTokens; ++mm) {
          if (mm >= decltype(live)::value) continue;
          const int plane_row = warp_m * kWarpTokens + mm * 16, row = plane_row + a_row_offset;
          const int lb = k64 * 32 + a_column_byte, pb = ((lb >> 4) ^ ((row >> 1) & 3)) * 16 + (lb & 15);
          ldsm_x4(af[mm], saddr(sh.a_codes[stage] + row * 64 + pb));
          asc[mm] = *reinterpret_cast<const uint32_t*>(sh.a_scales[slot] + (plane_row + sfa_row) * 16 + sw);
        }
#pragma unroll
        for (int mn = 0; mn < kMmaRows; ++mn) {
          constexpr int per_branch = kMmaRows / branches;
          const int branch = mn / per_branch, fragment = mn % per_branch;
          const int pair_row = warp_n * (kWarpRows / branches) + fragment * 8;
          const int row = branch * loaded_rows + pair_row + b_row_offset;
          const int lb = k64 * 32 + b_column_byte, pb = ((lb >> 4) ^ ((row >> 1) & 3)) * 16 + (lb & 15);
          ldsm_x2(bf[mn], saddr(sh.b_codes[stage] + row * 64 + pb));
          bsc[mn] = *reinterpret_cast<const uint32_t*>(sh.b_scales[slot] +
                                                       (branch * loaded_rows + pair_row + sfb_row) * 16 + sw);
        }
#pragma unroll
        for (int mm = 0; mm < kMmaTokens; ++mm) {
          if (mm >= decltype(live)::value) continue;
#pragma unroll
          for (int mn = 0; mn < kMmaRows; ++mn) mma_fp4(acc[mm][mn], af[mm], bf[mn], asc[mm], bsc[mn]);
        }
      }
      fence_reads();
      __syncwarp();
      if (lane == 0) mbar_arrive(&sh.empty[stage]);
    }
  };
  const int live = (p.M - token_begin - warp_m * kWarpTokens + 15) / 16;
  if (live <= 0) {  // no row of the call: the ring's protocol alone keeps the producer's refills going
#pragma unroll 1
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages;
      mbar_wait(&sh.full[stage], (kt / kStages) & 1u);
      fence_reads();
      __syncwarp();
      if (lane == 0) mbar_arrive(&sh.empty[stage]);
    }
    return;
  }
  with_live<kMmaTokens>(live, consume);
  // thread (lane) holds rows lane / 4 and lane / 4 + 8 of each 16-row fragment, weight rows 2 (lane & 3) and + 1 of
  // each 8-row fragment: the epilogue of the configurations above, product for product
  if constexpr (GLU) {
#pragma unroll
    for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = token_begin + warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
        float m = 0.f;
        if (r < p.M) {
          const float ar = sh.ar[r - token_begin];
#pragma unroll
          for (int f = 0; f < kMmaRows / 2; ++f) {
            const int jl = warp_n * (kWarpRows / 2) + f * 8 + 2 * (lane & 3), j = row_begin + jl;
            if (j >= p.I) continue;
            const float g0 = acc[mm][f][2 * h] * sh.rs[jl] * ar, g1 = acc[mm][f][2 * h + 1] * sh.rs[jl + 1] * ar;
            const float u0 = acc[mm][f + kMmaRows / 2][2 * h] * sh.rs[loaded_rows + jl] * ar;
            const float u1 = acc[mm][f + kMmaRows / 2][2 * h + 1] * sh.rs[loaded_rows + jl + 1] * ar;
            const __nv_bfloat162 v = __floats2bfloat162_rn(silu(g0) * u0, silu(g1) * u1);
            *reinterpret_cast<__nv_bfloat162*>(p.glu + size_t(r) * p.I + j) = v;
            m = fmaxf(m, fmaxf(fabsf(__low2float(v)), fabsf(__high2float(v))));
          }
        }
        // the row's four lanes (lane % 4: all live or none), then one atomic, as the configurations above
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        if ((lane & 3) == 0 && r < p.M && m > 0.f) atomicMax(reinterpret_cast<int*>(p.amax + r), __float_as_int(m));
      }
    return;
  }
#pragma unroll
  for (int mm = 0; mm < kMmaTokens; ++mm)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = token_begin + warp_m * kWarpTokens + mm * 16 + (lane >> 2) + 8 * h;
      if (r >= p.M) continue;
      if (p.y16) {  // bf16 rows; accumulate: each pair read first, the sum in float32, then rounded
        bf16* out = p.y16 + size_t(r) * p.ldy;
        const float ar = sh.ar[r - token_begin];
        const bool pairs16 = (p.ldy & 1) == 0 && (reinterpret_cast<uintptr_t>(p.y16) & 3) == 0;
        float2 res[kMmaRows];
        if (p.accumulate) {
#pragma unroll
          for (int f = 0; f < kMmaRows; ++f) {
            const int c = row_begin + warp_n * kWarpRows + f * 8 + 2 * (lane & 3);
            if (pairs16 && c + 1 < p.N) {
              res[f] = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(out + c));
            } else {
              res[f] = make_float2(c < p.N ? __bfloat162float(out[c]) : 0.f,
                                   c + 1 < p.N ? __bfloat162float(out[c + 1]) : 0.f);
            }
          }
        }
#pragma unroll
        for (int f = 0; f < kMmaRows; ++f) {
          const int c = row_begin + warp_n * kWarpRows + f * 8 + 2 * (lane & 3), cl = c - row_begin;
          if (c >= p.N) continue;
          float2 v = make_float2(acc[mm][f][2 * h] * sh.rs[cl] * ar, 0.f);
          if (c + 1 < p.N) v.y = acc[mm][f][2 * h + 1] * sh.rs[cl + 1] * ar;
          if (p.accumulate) v.x = __fadd_rn(res[f].x, v.x), v.y = __fadd_rn(res[f].y, v.y);  // the product rounded first
          if (pairs16 && c + 1 < p.N) {
            *reinterpret_cast<__nv_bfloat162*>(out + c) = __floats2bfloat162_rn(v.x, v.y);
          } else {
            out[c] = __float2bfloat16_rn(v.x);
            if (c + 1 < p.N) out[c + 1] = __float2bfloat16_rn(v.y);
          }
        }
        continue;
      }
      {
        float* out = p.y + size_t(r) * p.ldy;
        const bool pairs = (p.ldy & 1) == 0 && (reinterpret_cast<uintptr_t>(p.y) & 7) == 0;
        float2 res[kMmaRows];  // accumulate: every pair read before any is written (they are distinct)
        if (p.accumulate) {
#pragma unroll
          for (int f = 0; f < kMmaRows; ++f) {
            const int c = row_begin + warp_n * kWarpRows + f * 8 + 2 * (lane & 3);
            res[f] = pairs && c + 1 < p.N ? __ldcg(reinterpret_cast<const float2*>(out + c))
                                          : make_float2(c < p.N ? out[c] : 0.f, c + 1 < p.N ? out[c + 1] : 0.f);
          }
        }
#pragma unroll
        for (int f = 0; f < kMmaRows; ++f) {
          const int c = row_begin + warp_n * kWarpRows + f * 8 + 2 * (lane & 3);
          if (c >= p.N) continue;
          const int cl = c - row_begin;
          const float ar = sh.ar[r - token_begin];
          float2 v = make_float2(acc[mm][f][2 * h] * sh.rs[cl] * ar, 0.f);
          if (c + 1 < p.N) v.y = acc[mm][f][2 * h + 1] * sh.rs[cl + 1] * ar;
          if (p.accumulate) v.x = res[f].x + v.x, v.y = res[f].y + v.y;
          if (pairs && c + 1 < p.N) *reinterpret_cast<float2*>(out + c) = v;
          else {
            out[c] = v.x;
            if (c + 1 < p.N) out[c + 1] = v.y;
          }
        }
      }
    }
}

using EncodeFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                              const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                              CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
EncodeFn encode_fn() {
  static const EncodeFn f = [] {
    void* q = nullptr;
    cudaDriverEntryPointQueryResult r{};
    if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &q, 12000, cudaEnableDefault, &r) != cudaSuccess ||
        r != cudaDriverEntryPointSuccess) {
      cudaGetLastError();
      return EncodeFn(nullptr);
    }
    return reinterpret_cast<EncodeFn>(q);
  }();
  return f;
}
// A byte matrix [rows][cols] (cols bytes a row), boxes of [box_r][box_c]; rows past the matrix read as zeros.
bool map_2d(TmaDesc* d, const void* base, uint64_t cols, uint64_t rows, uint32_t box_c, uint32_t box_r,
            CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promo) {
  const EncodeFn f = encode_fn();
  if (!f) return false;
  CUtensorMap m;
  const cuuint64_t dim[2] = {cols, rows}, str[1] = {cols};
  const cuuint32_t box[2] = {box_c, box_r}, es[2] = {1, 1};
  if (f(&m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(base), dim, str, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, promo, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
    return false;
  std::memcpy(d, &m, sizeof(*d));
  return true;
}

bool prepare() {
  return encode_fn() != nullptr &&
         cudaFuncSetAttribute(k_gemm_w4a4_tma<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) ==
             cudaSuccess &&
         cudaFuncSetAttribute(k_gemm_w4a4_tma<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) ==
             cudaSuccess;
}

// false: the shapes or pointers do not fit it (the caller takes another configuration)
bool launch(const uint8_t* Aq, const uint8_t* As, const float* Ar, const uint8_t* Q, const uint8_t* S,
            const float* RS, int N, int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st, bf16* glu,
            float* glu_amax, bf16* y16) {
  const auto al16 = [](const void* v) { return (reinterpret_cast<uintptr_t>(v) & 15) == 0; };
  if (M < 1 || K % 256 || !al16(Aq) || !al16(As) || !al16(Q) || !al16(S) || (glu && (N % 2 || (N / 2) % 2)))
    return false;
  const int branches = glu ? 2 : 1, rows = glu ? N / 2 : N, loaded = kBR / branches;
  Params p{};
  bool ok = map_2d(&p.a_codes, Aq, K / 2, M, 64, kBT, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE) &&
            map_2d(&p.a_scales, As, K / 16, M, 16, kBT, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE);
  for (int b = 0; b < branches; ++b) {
    const size_t off = size_t(b) * rows;
    ok = ok &&
         map_2d(&p.b_codes[b], Q + off * (K / 2), K / 2, rows, 64, loaded, CU_TENSOR_MAP_SWIZZLE_64B,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B) &&
         map_2d(&p.b_scales[b], S + off * (K / 16), K / 16, rows, 16, loaded, CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_NONE);
  }
  if (!ok) return false;
  p.Ar = Ar, p.RS = RS, p.y = y, p.glu = glu, p.amax = glu_amax, p.y16 = y16, p.ldy = ldy, p.M = M, p.N = N, p.I = glu ? N / 2 : 0, p.K = K;
  p.tok_tiles = (M + kBT - 1) / kBT, p.accumulate = int(accumulate);
  const unsigned blocks = unsigned(((rows + loaded - 1) / loaded) * p.tok_tiles);
  if (glu) k_gemm_w4a4_tma<true><<<blocks, kThreads, kSmem, st>>>(p);
  else k_gemm_w4a4_tma<false><<<blocks, kThreads, kSmem, st>>>(p);
  return true;
}

}  // namespace tma4

template <int BN, int ST, int KB, bool GLU>
bool prepare_one() {
  cudaFuncAttributes a{};
  const bool image = cudaFuncGetAttributes(&a, k_gemm_w4a4<BN, ST, KB, GLU>) == cudaSuccess;
  cudaGetLastError();
  return image && cudaFuncSetAttribute(k_gemm_w4a4<BN, ST, KB, GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       int(sizeof(W4Smem<BN, ST, KB>))) == cudaSuccess;
}
template <int BN, int ST, int KB>
bool prepare_cfg() {
  return prepare_one<BN, ST, KB, false>() && prepare_one<BN, ST, KB, true>();
}

template <int BN, int ST, int KB>
void launch_cfg(const uint8_t* Aq, const uint8_t* As, const float* Ar, const uint8_t* Q, const uint8_t* S,
                const float* RS, int N, int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st, bf16* glu,
                float* glu_amax, bf16* y16) {
  const dim3 grid((M + kBM - 1) / kBM, (N + BN - 1) / BN);
  constexpr size_t smem = sizeof(W4Smem<BN, ST, KB>);
  if (glu)
    k_gemm_w4a4<BN, ST, KB, true><<<grid, kThreads, smem, st>>>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, 0, glu,
                                                               glu_amax, nullptr);
  else
    k_gemm_w4a4<BN, ST, KB, false><<<grid, kThreads, smem, st>>>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M,
                                                                int(accumulate), nullptr, nullptr, y16);
}

bool gemm_w4a4_supported() {
  static const bool ok = [&] {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    // Block-scaled FP4 MMA cubin is compiled for architecture-specific sm_120a (strictly sm_120).
    return major == 12 && minor == 0 && prepare_cfg<128, 4, 64>() && prepare_cfg<256, 3, 64>() &&
           prepare_cfg<128, 2, 64>() && prepare_cfg<128, 2, 128>() && tma4::prepare();
  }();
  return ok;
}

void quantize_rows_nvfp4(const float* x, int ldx, int M, int K, uint8_t* q, uint8_t* s, float* inv, cudaStream_t st) {
  k_quant_rows_nvfp4<<<M, 128, 0, st>>>(x, ldx, K, q, s, inv);
}

void quantize_rows_nvfp4_bf16(const bf16* x, int ldx, int M, int K, const float* amax, uint8_t* q, uint8_t* s,
                              float* inv, cudaStream_t st) {
  k_quant_rows_nvfp4_bf16<<<M, 128, 0, st>>>(x, ldx, K, amax, q, s, inv);
}

void gemm_w4a4(const uint8_t* Aq, const uint8_t* As, const float* Ar, const uint8_t* Q, const uint8_t* S,
               const float* RS, int N, int K, float* y, int ldy, int M, bool accumulate, cudaStream_t st, int cfg,
               bf16* glu, float* glu_amax, bf16* y16) {
  if (y16 && glu) std::abort();  // bf16 rows are a plain product's
  if (glu && cudaMemsetAsync(glu_amax, 0, size_t(M) * sizeof(float), st) != cudaSuccess) std::abort();
  // 3 is the default where K allows: stages of 256 values of K (half the waits a k step), one block of 72 KB an SM;
  // on the 27B's shapes 600-1010 TFLOPS at 512-4096 rows against 525-750 for 2 (the default before: two blocks an SM
  // of 128 values, best on Qwen3.5-4B's shapes on the RTX 5070 Ti), 476-763 for 0 and 380-819 for 1 (cuda_bench
  // w4a4, RTX 5090).
  // 4 (the default where K % 256 == 0 and the pass is long): the warp-specialized TMA pipeline, 256 rows a block.  At
  // 512 rows its blocks of 256 rows leave an RTX PRO 6000's 188 SMs short on the 27B's 5120-column layers (down, out:
  // 260-450 TFLOPS against 3's 720-900) and on the q|k|v and GDN in-projections (590-680 against 670-760); from 1024
  // rows it is ahead on every layer (1.0 to 1.3 times 3), and on gate|up from 512 (cuda_bench w4a4).
  if (cfg < 0) cfg = K % 256 != 0 ? 2 : M > 512 || glu ? 4 : 3;
  if (cfg == 4 && tma4::launch(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, glu_amax, y16)) return;
  if (cfg == 4 || (cfg == 3 && K % 256 != 0)) cfg = K % 256 == 0 ? 3 : 2;
  if (cfg == 1) launch_cfg<256, 3, 64>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, glu_amax, y16);
  else if (cfg == 2)
    launch_cfg<128, 2, 64>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, glu_amax, y16);
  else if (cfg == 3)
    launch_cfg<128, 2, 128>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, glu_amax, y16);
  else launch_cfg<128, 4, 64>(Aq, As, Ar, Q, S, RS, N, K, y, ldy, M, accumulate, st, glu, glu_amax, y16);
}

}  // namespace flyx::cuda
