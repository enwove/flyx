// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Persistent-MLP microbenchmark experiment (not used by the engine): a round's MLP
// (1..8 rows, NVFP4) as one persistent kernel.  RMS norm, gate|up, SwiGLU,
// down and the residual add, with the bits of the four kernels a fused MLP
// region runs (rmsnorm_split, gemm_fp4_presplit, swiglu_split,
// gemm_fp4_presplit); `cuda_bench mlp` checks the bits and times both.
//
// Every block stays resident (cooperative launch).  Block b takes gate|up
// pairs b, b + G, ... (SwiGLU of the pair in registers, written as the down
// GEMM's input), counts itself done, waits for all blocks, then takes down
// tiles b, b + G, ...  Optionally (prefetch) the first pair's weights and a
// block's first down tile's weights go to L2 before the norm and before the
// wait. Measured slower than the four kernels.
//
// Bits: every tile is computed as k_gemm_skinny_fp4<128, 4, 1, false, 4, U>
// computes it (the tables' K partition for both MLP shapes; the launcher
// checks), the norm adds in k_rmsnorm's order for its 512 threads, and the
// SwiGLU and residual add are the same float expressions.
#include "bench/microbench/mlp_persistent.h"

#include "backends/cuda/kernels/common/device.cuh"

namespace flyx::cuda {
namespace {

constexpr int kThreads = 128;      // one tile at a time: 4 warps, one K chunk each
constexpr int kSpan = 128, kChunks = 4;
constexpr int kMaxRows = 8;        // the tile's rows g < 8 (no second row group)
constexpr int kNormThreads = 512;  // k_rmsnorm's block size (rows.cu): the order of its sum
constexpr int kNormCache = 8;      // k_rmsnorm's values per thread before its tail loop (rows.cu)
constexpr int kMinBlocks = 9;      // per SM: registers capped at the skinny GEMM's 56

struct MlpSync {
  unsigned pairs_done, exited;
};

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Bytes (a multiple of 16) into L2 without waiting for them; nothing when p is not 16-byte aligned.
__device__ __forceinline__ void prefetch_l2(const void* p, unsigned bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if ((reinterpret_cast<uintptr_t>(p) & 15) == 0)
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#endif
}

// The codes and scales of the 8 weight rows n0.. of an NVFP4 matrix with K columns.
__device__ __forceinline__ void prefetch_rows(const uint8_t* Q, const uint8_t* S, int n0, int K) {
  prefetch_l2(Q + size_t(n0) * (K / 2), 8 * (K / 2));
  prefetch_l2(S + size_t(n0) * (K / 16), 8 * (K / 16));
}

// 1 / rms of a row exactly as k_rmsnorm computes it with 512 threads.  Our 128
// threads play its threads tid + 128 j: that one is lane tid & 31 of warp
// (tid >> 5) + 4 j, so each warp_sum below is the same butterfly over the same
// values, and warp 0 adds the 16 warp sums as block_sum does.
__device__ float rms_inv(const float* xr, int n, float eps, float* sh) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < kNormThreads / kThreads; ++j) {
    const int v = threadIdx.x + j * kThreads;
    float ss = 0.f;
#pragma unroll
    for (int u = 0; u < kNormCache; ++u) {
      const int i = v + u * kNormThreads;
      if (i < n) ss = __fmaf_rn(xr[i], xr[i], ss);
    }
    for (int i = v + kNormCache * kNormThreads; i < n; i += kNormThreads) ss = __fmaf_rn(xr[i], xr[i], ss);
    ss = warp_sum(ss);
    if (lane == 0) sh[warp + 4 * j] = ss;
  }
  __syncthreads();
  if (warp == 0) {
    float t = lane < kNormThreads / 32 ? sh[lane] : 0.f;
    t = warp_sum(t);
    if (lane == 0) sh[32] = t;
  }
  __syncthreads();
  return rsqrtf(sh[32] / n + eps);
}

// Columns n_base .. n_base + 7 of rows x [rows, K] (bf16, stride ldk) times
// the NVFP4 matrix (Q, S, RS) [N, K]: k_gemm_skinny_fp4<128, 4, 1, false, 4,
// UNROLL>'s arithmetic, warp w summing K chunk w and the chunks added in
// order.  Thread t gets the value of row *r, column *c, scaled by RS (valid
// when *r < rows).
template <int UNROLL>
__device__ __forceinline__ float fp4_tile(const bf16* xh, int ldk, const uint8_t* __restrict__ Q,
                                          const uint8_t* __restrict__ S, const float* __restrict__ RS, int K,
                                          int n_base, int rows, float (*red)[32][4], int* r_out, int* c_out) {
  constexpr int VPL = kSpan / 4, STEPS = VPL / 4, WORDS = VPL / 8, BW = VPL / 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int spans = K / kSpan;
  const bool ra = g < rows;
  const size_t oa = size_t(g) * ldk + VPL * t;
  const uint8_t* qp = Q + size_t(n_base + g) * (K >> 1) + (VPL / 2) * t;
  const uint8_t* sp = S + size_t(n_base + g) * (K >> 4) + (VPL / 16) * t;
  const int chunk = warp;
  const int s0 = chunk * spans / kChunks, s1 = (chunk + 1) * spans / kChunks;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll UNROLL
  for (int s = s0; s < s1; ++s) {
    const int k = s * kSpan;
    uint32_t wq[WORDS];
    const uint4 v = ldg_stream(qp + (k >> 1));
    wq[0] = v.x, wq[1] = v.y, wq[2] = v.z, wq[3] = v.w;
    const uint32_t sc = *reinterpret_cast<const uint16_t*>(sp + (k >> 4));
    uint32_t av[BW];
#pragma unroll
    for (int q = 0; q < VPL / 8; ++q) {
      const uint4 a = ra ? *reinterpret_cast<const uint4*>(xh + oa + k + 8 * q) : make_uint4(0, 0, 0, 0);
      av[4 * q] = a.x, av[4 * q + 1] = a.y, av[4 * q + 2] = a.z, av[4 * q + 3] = a.w;
    }
    uint32_t bw[BW];
#pragma unroll
    for (int q = 0; q < WORDS; ++q) fp4x8_to_bf16(wq[q], bw + 4 * q);
#pragma unroll
    for (int b = 0; b < VPL / 16; ++b) {
      const __nv_bfloat162 s2 = e4m3_bf16x2((sc >> (8 * b)) & 0xFF);
#pragma unroll
      for (int e = 0; e < 8; ++e) bw[8 * b + e] = bf16x2_mul(bw[8 * b + e], s2);
    }
#pragma unroll
    for (int j = 0; j < STEPS; ++j) {
      const uint32_t a[4] = {av[2 * j], 0u, av[2 * j + 1], 0u};
      const uint32_t bf[2] = {bw[2 * j], bw[2 * j + 1]};
      mma_bf16(acc, a, bf);
    }
  }
#pragma unroll
  for (int e = 0; e < 4; ++e) red[chunk][lane][e] = acc[e];
  __syncthreads();
  const int l = threadIdx.x >> 2, e = threadIdx.x & 3;
  float v = 0.f;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) v += red[c][l][e];
  const int r = (l >> 2) + (e >> 1) * 8, col = n_base + 2 * (l & 3) + (e & 1);
  __syncthreads();  // red is free for the next tile
  *r_out = r;
  *c_out = col;
  return r < rows ? v * RS[col] : 0.f;
}

template <int UNROLL_GU, int UNROLL_DOWN>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
    k_mlp_persistent(float* x, int H, const float* __restrict__ nw, float noff, float neps,
                     const uint8_t* __restrict__ Qgu, const uint8_t* __restrict__ Sgu, const float* __restrict__ RSgu,
                     const uint8_t* __restrict__ Qd, const uint8_t* __restrict__ Sd, const float* __restrict__ RSd,
                     int I, int rows, bf16* act, MlpSync* sync, int prefetch, int mode) {
  extern __shared__ __align__(16) unsigned char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);  // the normalized rows [rows, H]
  __shared__ float red[kChunks][32][4];
  __shared__ float sh[33];
  const unsigned pairs = (mode & 1) ? unsigned(I / 8) : 0u, tiles = (mode & 2) ? unsigned(H / 8) : 0u;
  const unsigned G = gridDim.x;

  if (prefetch && threadIdx.x == 0 && blockIdx.x < pairs) {
    prefetch_rows(Qgu, Sgu, 8 * int(blockIdx.x), H);
    prefetch_rows(Qgu, Sgu, I + 8 * int(blockIdx.x), H);
  }
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + size_t(r) * H;
    const float inv = rms_inv(xr, H, neps, sh);
    for (int i = threadIdx.x; i < H; i += kThreads)
      xs[size_t(r) * H + i] = __float2bfloat16_rn(xr[i] * inv * (noff + nw[i]));
  }
  __syncthreads();

  // Pair u: column block u of the gate and of the up half.  Thread t holds the
  // same (row, column) of both, so SwiGLU needs no exchange.  Blocks land on
  // SMs in turn, so every SM gets its share of pairs and of down tiles.
  for (unsigned u = blockIdx.x; u < pairs; u += G) {
    int r, c, r2, c2;
    const float gate = fp4_tile<UNROLL_GU>(xs, H, Qgu, Sgu, RSgu, H, 8 * int(u), rows, red, &r, &c);
    const float up = fp4_tile<UNROLL_GU>(xs, H, Qgu, Sgu, RSgu, H, I + 8 * int(u), rows, red, &r2, &c2);
    if (r < rows) act[size_t(r) * I + c] = __float2bfloat16_rn(silu(gate) * up);
  }

  // The down GEMM reads all of act: count this block, wait for the others.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(&sync->pairs_done, 1u);
    if (blockIdx.x < tiles) {
      if (prefetch) prefetch_rows(Qd, Sd, 8 * int(blockIdx.x), I);
      while (ld_acquire(&sync->pairs_done) < G) {
      }
      __threadfence();
    }
  }
  __syncthreads();
  // act comes from other blocks, yet plain (L1) loads see it: no block read it
  // in this kernel before the wait, so no SM holds an old copy.
  for (unsigned u = blockIdx.x; u < tiles; u += G) {
    int r, c;
    const float v = fp4_tile<UNROLL_DOWN>(act, I, Qd, Sd, RSd, I, 8 * int(u), rows, red, &r, &c);
    if (r < rows) {
      float* dst = x + size_t(r) * H + c;
      *dst = *dst + v;
    }
  }

  // The last block out leaves the counters at zero for the next launch.
  if (threadIdx.x == 0 && atomicAdd(&sync->exited, 1u) == G - 1) {
    sync->pairs_done = 0;
    __threadfence();
    sync->exited = 0;
  }
}

constexpr auto kKernel = k_mlp_persistent<4, 8>;

}  // namespace

int mlp_persistent(float* x, int H, const float* norm_w, float offset, float eps, const uint8_t* Qgu,
                   const uint8_t* Sgu, const float* RSgu, const uint8_t* Qd, const uint8_t* Sd, const float* RSd, int I,
                   int rows, bf16* act, unsigned* sync, int tuning, const MlpPersistentOptions& o, cudaStream_t st) {
  if (rows < 1 || rows > kMaxRows || H % kSpan || I % kSpan || H > kNormCache * kNormThreads) return 0;
  // The bits of the GEMMs this kernel stands for: the K partition the tables give both shapes.
  for (const auto& nk : {std::pair<int, int>{2 * I, H}, std::pair<int, int>{H, I}}) {
    const Fp4Config& c = kFp4Configs[gemm_config(tuning, 1, nk.first, nk.second, rows)];
    if (c.span != kSpan || c.chunks != kChunks) return 0;
  }
  const int smem = rows * H * int(sizeof(bf16));
  int dev = 0, sms = 0, per_sm = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  if (smem > 48 * 1024) cudaFuncSetAttribute(kKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kKernel, kThreads, smem);
  if (o.blocks_per_sm > 0 && o.blocks_per_sm < per_sm) per_sm = o.blocks_per_sm;
  if (per_sm <= 0) return 0;
  // Every block waits for every other once: all of them must be resident together.
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(unsigned(per_sm * sms));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = size_t(smem);
  cfg.stream = st;
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeCooperative;
  at.val.cooperative = 1;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kKernel, x, H, norm_w, offset, eps, Qgu, Sgu, RSgu, Qd, Sd, RSd, I,
                                           rows, act, reinterpret_cast<MlpSync*>(sync), o.prefetch ? 1 : 0, o.mode);
  return e == cudaSuccess ? per_sm : 0;
}

}  // namespace flyx::cuda
