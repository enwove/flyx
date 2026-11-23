// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block weights (GGUF files) kept packed on the device:
// the GEMMs read the blocks as the file stores them, 1.6 to 8.5 bits a
// weight, and decode them in registers; the block types and how a kernel
// reads them are in gemm_blocks_types.cuh.
//
// What decides the speed of these kernels is how often they touch memory: a
// load from global memory costs a few cycles for every cache line it
// touches, whatever the cache level that answers (measured: about three per
// line and warp).  So a block's bytes are loaded once per column (the four
// lanes of a column a quarter each, shared by shuffles), and the
// activations go through shared memory once per block of threads.  And
// nothing may wait for a load: what a warp is going to need is loaded a
// window ahead, and the kernels leave an SM room for other warps to run
// meanwhile (a block of threads that fills an SM's shared memory alone ran
// at the speed of its waits, whatever its instructions).
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_impl.cuh"
#include "flyx/gemm_tables.inc"

namespace flyx::cuda {

namespace {

// ---- the activations ------------------------------------------------------------------------------
// A row's scale: the power of two that puts its largest magnitude m (of the
// bf16 rounding) in [2^15, 2^16), and its inverse.  Both exact; zero rows and
// rows with an infinity or a NaN keep 1.
__device__ __forceinline__ float row_scale_of(float m, float* inverse) {
  const uint32_t e = (__float_as_uint(m) >> 23) & 255u;
  if (!(m > 0.f) || e == 255u) {
    *inverse = 1.f;
    return 1.f;
  }
  const uint32_t ec = e < 16u ? 16u : e;  // the scale itself stays a normal number
  *inverse = __uint_as_float((269u - ec) << 23);
  return __uint_as_float((ec - 15u) << 23);
}
// Two values of a row as bf16 (FLOATS: rounded from float32; else a producer's bf16 rows), then as float32.
template <bool FLOATS>
__device__ __forceinline__ float2 two_bf16(const void* x, size_t at) {
  if (FLOATS) {
    const float2 v = *reinterpret_cast<const float2*>(static_cast<const float*>(x) + at);
    return __bfloat1622float2(__floats2bfloat162_rn(v.x, v.y));
  }
  return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(static_cast<const bf16*>(x) + at));
}
__device__ __forceinline__ uint32_t two_halves(float2 v, float inverse) {
  const __half2 h = __floats2half2_rn(v.x * inverse, v.y * inverse);
  return *reinterpret_cast<const uint32_t*>(&h);
}
template <bool FLOATS>
__device__ __forceinline__ float row_max(const void* x, size_t row, int K) {
  float m = 0.f;
  for (int k = 2 * int(threadIdx.x); k < K; k += 2 * int(blockDim.x)) {
    const float2 v = two_bf16<FLOATS>(x, row + k);
    m = fmaxf(m, fmaxf(fabsf(v.x), fabsf(v.y)));
  }
  return block_max(m);
}

// 1..16 rows as the kernel below reads them: slot g (a block of threads
// each) holds rows g and g + 8 pair by pair: the 16 bytes at 4 k are the
// tensor cores' operand of rows g, g + 8 and weights k .. k + 3 (zeros for an
// absent row g + 8).  Slots of absent row pairs are not written.
template <bool FLOATS>
__global__ void __launch_bounds__(256) k_stage_pairs(const void* x, int ldx, int K, int rows, uint2* pairs,
                                                      float* row_scale) {
  pdl_trigger();
  const int g = blockIdx.x;
  uint2* out = pairs + size_t(g) * (K / 2);
  pdl_wait();  // the rows come from the kernel before
  const bool second = g + 8 < rows;
  float inv0, inv1 = 0.f;
  const float s0 = row_scale_of(row_max<FLOATS>(x, size_t(g) * ldx, K), &inv0);
  const float s1 = second ? row_scale_of(row_max<FLOATS>(x, size_t(g + 8) * ldx, K), &inv1) : 1.f;
  if (threadIdx.x == 0) {
    row_scale[g] = s0;
    if (second) row_scale[g + 8] = s1;
  }
  for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
    const uint32_t a = two_halves(two_bf16<FLOATS>(x, size_t(g) * ldx + 2 * i), inv0);
    const uint32_t b = second ? two_halves(two_bf16<FLOATS>(x, size_t(g + 8) * ldx + 2 * i), inv1) : 0u;
    out[i] = make_uint2(a, b);
  }
}

// More rows (prefill): each row as binary16 over its scale, [rows][K], and the sums of its runs of 16 values in
// order, [rows][K / 16] (what a plane of one value per run takes of a row); a block of threads a row.  A
// producer's bf16 rows are rewritten where they are (x == out).
template <bool FLOATS>
__global__ void __launch_bounds__(256) k_stage_rows(const void* x, int ldx, int K, uint32_t* out, float* row_scale,
                                                     float* sums) {
  pdl_trigger();
  const int r = blockIdx.x;
  pdl_wait();
  float inv;
  const float s = row_scale_of(row_max<FLOATS>(x, size_t(r) * ldx, K), &inv);
  if (threadIdx.x == 0) row_scale[r] = s;
  for (int j = threadIdx.x; j < K / 16; j += blockDim.x) {
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t two = two_halves(two_bf16<FLOATS>(x, size_t(r) * ldx + 16 * j + 2 * i), inv);
      out[size_t(r) * (K / 2) + 8 * j + i] = two;
      const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&two));
      sum += v.x;
      sum += v.y;
    }
    sums[size_t(r) * (K / 16) + j] = sum;
  }
}

// More rows with inputs in 8 bits (FLYX_ACT_Q8; ops/dtype/ref.h: round_q8): per 32 values of the float32 row
// d = (their largest magnitude) / 127 and p = the nearest integers of value / d, [rows][K]; d, [rows][K / 32].
// One rounding, as llama.cpp's; bf16 rows (a fused producer's, which no block with 8-bit inputs
// has) are taken as they are.  A warp a group of 32 at a time, a block of threads a row.
template <bool FLOATS>
__device__ __forceinline__ float one_value(const void* x, size_t at) {
  if (FLOATS) return static_cast<const float*>(x)[at];
  return __bfloat162float(static_cast<const bf16*>(x)[at]);
}
// The largest magnitude group `grp`'s scale covers, across the warp: its own 32 values (group 32), or every value of
// the window of `group` holding them (option act_group, a measurement of coarser scales).
template <bool FLOATS>
__device__ __forceinline__ float group_max(const void* x, size_t row_at, int grp, int units, int group, float own) {
  float m = fabsf(own);
  if (group > 32) {
    const int per = group / 32, first = grp / per * per, last = min(first + per, units);
    for (int g = first; g < last; ++g) m = fmaxf(m, fabsf(one_value<FLOATS>(x, row_at + 32 * g + (threadIdx.x & 31))));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  return m;
}
// Measurement (option act_fp4): value a as an NVFP4 activation would carry it (the row's
// scale gs = kNvfp4RowRange / max|row|, an e4m3 scale per 16 values, the value's nearest e2m1), before the 8-bit
// rounding: what 4-bit activations would cost on these weights.  A half warp holds a block of 16.
__device__ __forceinline__ float nvfp4_fake(float a, float gs) {
  float b = fabsf(a);
#pragma unroll
  for (int o = 8; o; o >>= 1) b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, o));
  const float sv = e4m3_to_float(nvfp4_block_scale(b, gs));
  if (!(sv > 0.f)) return 0.f;
  const float f = __fdiv_rn(gs, sv);
  const uint8_t c = e2m1_nearest(__fmul_rn(a, f));
  const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = mag[c & 7];
  return __fdiv_rn(c & 8 ? -v : v, f);
}
// Measurement (option act_fp4=2): value a as an MXFP8 activation would carry it (a power of two a group of 32 that
// puts its largest magnitude at e4m3's top binade, the value's nearest e4m3), what 8-bit float activations on the
// FP8 / FP6 / FP4 tensor cores (kind::f8f6f4: NVFP4 weights, FP8 inputs) would cost.  A warp holds the group.
__device__ __forceinline__ float mxfp8_fake(float a) {
  float m = fabsf(a);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (!(m > 0.f)) return 0.f;
  const float scale = exp2f(floorf(log2f(m)) - 8.f);  // e4m3's largest binade is 2^8 (448 = 1.75 * 2^8)
  const __nv_fp8_storage_t c = __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, __NV_E4M3);
  return e4m3_to_float(static_cast<uint8_t>(c)) * scale;
}
// The row's NVFP4 scale for nvfp4_fake, from every value of it (a whole block of threads).
template <bool FLOATS>
__device__ __forceinline__ float nvfp4_row_scale(const void* x, size_t row_at, int K) {
  float m = 0.f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) m = fmaxf(m, fabsf(one_value<FLOATS>(x, row_at + i)));
  m = block_max(m);
  return m > 0.f ? __fdiv_rn(kNvfp4RowRange, m) : 1.f;
}
template <bool FLOATS>
__device__ __forceinline__ float4 load_four(const void* x, size_t at, bool aligned4) {
  if constexpr (FLOATS) {
    const float* xf = static_cast<const float*>(x) + at;
    if (aligned4) return *reinterpret_cast<const float4*>(xf);
    const float2 a = *reinterpret_cast<const float2*>(xf);
    const float2 b = *reinterpret_cast<const float2*>(xf + 2);
    return make_float4(a.x, a.y, b.x, b.y);
  } else {
    const bf16* xb = static_cast<const bf16*>(x) + at;
    float2 a, b;
    if (aligned4) {
      const uint2 u = *reinterpret_cast<const uint2*>(xb);
      a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
      b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    } else {
      a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xb));
      b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(xb + 2));
    }
    return make_float4(a.x, a.y, b.x, b.y);
  }
}
__device__ __forceinline__ uint32_t pack_q8_four(int p0, int p1, int p2, int p3) {
  return uint32_t(uint8_t(int8_t(p0))) | (uint32_t(uint8_t(int8_t(p1))) << 8) |
         (uint32_t(uint8_t(int8_t(p2))) << 16) | (uint32_t(uint8_t(int8_t(p3))) << 24);
}

template <bool FLOATS>
__global__ void __launch_bounds__(256) k_stage_q8(const void* x, int ldx, int K, int8_t* q, float* d, int group, int fp4) {
  pdl_trigger();
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31, units = K / 32;
  pdl_wait();
  if (group <= 32 && fp4 == 0) {
    // 4 groups of 32 per warp (8 lanes/group, 4 values/lane): vectorized loads and 128-byte coalesced int8 stores.
    const int sub = lane & 7, g_in_warp = lane >> 3;
    const size_t row_at = size_t(r) * ldx;
    const bool aligned4 = (ldx & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & (FLOATS ? 15u : 7u)) == 0;
    for (int base = warp * 4; base < units; base += (int(blockDim.x) >> 5) * 4) {
      const int grp = base + g_in_warp;
      const float4 v = grp < units ? load_four<FLOATS>(x, row_at + 32 * grp + 4 * sub, aligned4)
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
      float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
      for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
      if (grp < units) {
        const float dg = m / 127.f;
        const bool ok = dg > 0.f && dg <= 3.402823466e38f;
        const int p0 = ok ? __float2int_rn(v.x / dg) : 0;
        const int p1 = ok ? __float2int_rn(v.y / dg) : 0;
        const int p2 = ok ? __float2int_rn(v.z / dg) : 0;
        const int p3 = ok ? __float2int_rn(v.w / dg) : 0;
        reinterpret_cast<uint32_t*>(q + size_t(r) * K + 32 * grp)[sub] = pack_q8_four(p0, p1, p2, p3);
        if (!sub) d[size_t(r) * units + grp] = dg;
      }
    }
    return;
  }
  const float gs = fp4 == 1 ? nvfp4_row_scale<FLOATS>(x, size_t(r) * ldx, K) : 1.f;
  for (int grp = warp; grp < units; grp += int(blockDim.x) >> 5) {
    float a = one_value<FLOATS>(x, size_t(r) * ldx + 32 * grp + lane);
    if (fp4 == 1) a = nvfp4_fake(a, gs);
    if (fp4 == 2) a = mxfp8_fake(a);
    const float m = group_max<FLOATS>(x, size_t(r) * ldx, grp, units, group, a);
    const float dg = m / 127.f;
    const int p = dg > 0.f && dg <= 3.402823466e38f ? __float2int_rn(a / dg) : 0;
    q[size_t(r) * K + 32 * grp + lane] = int8_t(p);
    if (!lane) d[size_t(r) * units + grp] = dg;
  }
}

// 1..16 rows with inputs in 8 bits, for the rounds' kernel (gemm_blocks_round.cu): each row rounded as above, its
// integers where round_q8_at puts them, d [rows][K / 32], and d times the sum of each 16 integers [rows][K / 16]
// (what a sub-group's minimum takes of a row; the prefill kernel computes the same products itself).  A warp a
// group of 32, kStageGroups groups a block of threads: a round's few rows still fill the GPU.
constexpr int kStageGroups = 4;
template <bool FLOATS>
__global__ void __launch_bounds__(32) k_stage_q8_round_fast(const void* x, int ldx, int K, int8_t* q, float* d,
                                                            float* sums) {
  pdl_trigger();
  const int r = blockIdx.y, lane = threadIdx.x, sub = lane & 7, units = K / 32;
  const int grp = blockIdx.x * kStageGroups + (lane >> 3);
  const bool aligned4 = (ldx & 3) == 0 && (reinterpret_cast<uintptr_t>(x) & (FLOATS ? 15u : 7u)) == 0;
  pdl_wait();
  const float4 v = grp < units ? load_four<FLOATS>(x, size_t(r) * ldx + 32 * grp + 4 * sub, aligned4)
                               : make_float4(0.f, 0.f, 0.f, 0.f);
  float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
#pragma unroll
  for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const float dg = m / 127.f;
  const bool ok = grp < units && dg > 0.f && dg <= 3.402823466e38f;
  const int p0 = ok ? __float2int_rn(v.x / dg) : 0;
  const int p1 = ok ? __float2int_rn(v.y / dg) : 0;
  const int p2 = ok ? __float2int_rn(v.z / dg) : 0;
  const int p3 = ok ? __float2int_rn(v.w / dg) : 0;
  int s = (p0 + p1) + (p2 + p3);  // the sum of the half-group's 16 (4 lanes)
#pragma unroll
  for (int o = 2; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (grp < units) {
    const int half = sub >> 2;
    reinterpret_cast<uint32_t*>(q + size_t(r) * K + round_q8_at(grp, units, half))[sub & 3] =
        pack_q8_four(p0, p1, p2, p3);
    if (!(sub & 3)) sums[(size_t(r) * units + grp) * 2 + half] = dg * float(s);
    if (!sub) d[size_t(r) * units + grp] = dg;
  }
}
template <bool FLOATS>
__global__ void __launch_bounds__(32 * kStageGroups) k_stage_q8_round(const void* x, int ldx, int K, int8_t* q, float* d,
                                                                      float* sums, int group, int fp4) {
  pdl_trigger();
  const int r = blockIdx.y, lane = threadIdx.x & 31, units = K / 32;
  const int grp = blockIdx.x * kStageGroups + int(threadIdx.x >> 5);
  pdl_wait();
  const float gs = fp4 == 1 ? nvfp4_row_scale<FLOATS>(x, size_t(r) * ldx, K) : 1.f;
  if (grp < units) {
    float a = one_value<FLOATS>(x, size_t(r) * ldx + 32 * grp + lane);
    if (fp4 == 1) a = nvfp4_fake(a, gs);
    if (fp4 == 2) a = mxfp8_fake(a);
    const float m = group_max<FLOATS>(x, size_t(r) * ldx, grp, units, group, a);
    const float dg = m / 127.f;
    const int p = dg > 0.f && dg <= 3.402823466e38f ? __float2int_rn(a / dg) : 0;
    q[size_t(r) * K + round_q8_at(grp, units, lane >> 4) + (lane & 15)] = int8_t(p);
    int s = p;  // the sum of the half warp's 16
#pragma unroll
    for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (!(lane & 15)) sums[(size_t(r) * units + grp) * 2 + (lane >> 4)] = dg * float(s);
    if (!lane) d[size_t(r) * units + grp] = dg;
  }
}


// ---- launchers ----------------------------------------------------------------------------------
[[noreturn]] void unsupported(const char* what, int dtype) {
  std::fprintf(stderr, "flyx-cuda: %s on dtype %d, which is not a GGML block type the kernels take\n", what, dtype);
  std::abort();
}

// A default rule chose a configuration this device has no room for: the table is wrong, nothing to recover from.
[[noreturn]] void unfit(int N, int rows) {
  std::fprintf(stderr, "flyx-cuda: the configuration chosen for %d rows of a %d-column layer of block weights does "
                       "not fit a block's shared memory here\n", rows, N);
  std::abort();
}

void check_input(const void* x, int ldx, int K) {
  if (K % 32 || ldx % 2 || reinterpret_cast<uintptr_t>(x) % 8) {
    std::fprintf(stderr, "flyx-cuda: a GEMM on block weights needs K %% 32 == 0 and aligned rows (K %d, ldx %d)\n", K,
                 ldx);
    std::abort();
  }
}

template <bool FLOATS>
void stage(const void* x, int ldx, int K, int rows, bool q8, const Launch& L) {
  const Scratch& sc = scratch(L);
  if (rows > 16 && q8) {
    check_fits(size_t(rows) * K, sc.abuf_cap, "a GEMM on block weights with inputs in 8 bits");
    launch_dependent(k_stage_q8<FLOATS>, dim3(rows), dim3(256), L, x, ldx, K, sc.q8, sc.q8_scale, opts(L).act_group,
                     opts(L).act_fp4);
  } else if (q8) {
    check_fits(size_t(rows) * K, sc.q8_round_cap, "a round on block weights with inputs in 8 bits");
    if (opts(L).act_group <= 32 && opts(L).act_fp4 == 0) {
      launch_dependent(k_stage_q8_round_fast<FLOATS>, dim3(ceil_div(K / 32, kStageGroups), rows), dim3(32), L, x, ldx,
                       K, sc.q8, sc.q8_scale, sc.q8_sums);
    } else {
      launch_dependent(k_stage_q8_round<FLOATS>, dim3(ceil_div(K / 32, kStageGroups), rows), dim3(32 * kStageGroups), L,
                       x, ldx, K, sc.q8, sc.q8_scale, sc.q8_sums, opts(L).act_group, opts(L).act_fp4);
    }
  } else if (rows <= 16) {
    const int slots = rows < 8 ? rows : 8;
    check_fits(size_t(slots) * 2 * K, sc.pairs_cap, "a 1..16-row GEMM on block weights");
    launch_dependent(k_stage_pairs<FLOATS>, dim3(slots), dim3(256), L, x, ldx, K, rows,
                     reinterpret_cast<uint2*>(sc.pairs), sc.row_scale);
  } else {
    check_fits(size_t(rows) * K, sc.abuf_cap, "a GEMM on block weights");
    launch_dependent(k_stage_rows<FLOATS>, dim3(rows), dim3(256), L, x, ldx, K, reinterpret_cast<uint32_t*>(sc.abuf),
                     sc.row_scale, sc.run_sums);
  }
}

// The configuration the rules (this GPU's table's, else the defaults) give a pass of `rows` rows with inputs in 8
// bits on a layer of N columns of block type `dtype`: the first rule that matches (a type's own rules come first and
// end with one for the rest).
int default_q8_config(int tuning, int dtype, int N, int rows) {
  const int s = tables::rules_sku(tuning, tables::kFamilyBlocksQ8);
  for (const tables::BlocksQ8Rule& r : tables::kBlocksQ8)
    if (r.sku == s && (r.dtype == dtype || r.dtype < 0) && rows >= r.lo && rows <= r.hi && N <= r.n_max) return r.cfg;
  return 0;
}

// The configuration the rules give `rows` rows of a layer of N columns (variants: this GPU's table, else the
// defaults in candidates.toml).
int default_config(int tuning, int N, int rows) {
  const int s = tables::rules_sku(tuning, tables::kFamilyBlocks);
  for (const tables::BlocksRule& r : tables::kBlocks)
    if (r.sku == s && N <= r.n_max && rows >= r.lo && rows <= r.hi) return r.cfg;
  return 0;
}

// A block type's kernels, from its group's translation unit (gemm_blocks_ops.h).
const BlockOps& ops_of(int dtype, const char* what) {
  for (const BlockOps* (*group)(int) : {block_ops_legacy, block_ops_k, block_ops_iq_low, block_ops_iq_high})
    if (const BlockOps* ops = group(dtype)) return *ops;
  unsupported(what, dtype);
}

}  // namespace

void blocks_init() {
  block_tables();
#define X(F) (void)ops_of(F::kDtype, "the kernels of a block type no group compiles");
  FLYX_BLOCK_TYPES(X)
#undef X
}

BlockLayout blocks_layout(int dtype) {
  switch (dtype) {
#define X(F) \
  case F::kDtype: return {F::kValues, F::kBytes, F::kStride, F::kScales};
    FLYX_BLOCK_TYPES(X)
#undef X
    default: return {0, 0, 0, 0};
  }
}

bool blocks_supported(int dtype, int K) {
  const BlockLayout l = blocks_layout(dtype);
  return l.values != 0 && K > 0 && K % l.values == 0;
}

void blocks_pack(int dtype, const uint8_t* file, size_t n, uint8_t* payload, uint16_t* scales) {
  int skip = 0, s0 = -1, s1 = -1;
  bool head = false;
  switch (dtype) {
#define X(F) \
  case F::kDtype: skip = F::kSkip, s0 = F::kS0, s1 = F::kS1, head = kHeadInScales<F>; break;
    FLYX_BLOCK_TYPES(X)
#undef X
    default: unsupported("blocks_pack", dtype);
  }
  const BlockLayout l = blocks_layout(dtype);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* b = file + i * size_t(l.bytes);
    uint8_t* p = payload + i * size_t(l.stride);
    uint16_t* s = scales + i * size_t(l.scales);
    if (dtype == FLYX_DT_GGML_IQ4_XS) {  // the codes lane-major (FIQ4_XS)
      for (int G = 0; G < 8; ++G)
        for (int j = 0; j < 16; ++j) p[FIQ4_XS::lane_major(G, j)] = b[skip + 16 * G + j];
    } else if (dtype == FLYX_DT_GGML_Q6_K) {  // the low bits' four rows pair-major (k_pair_major), the rest as is
      std::memcpy(p, b, size_t(l.stride));
      for (int c = 0; c < 4; ++c)
        for (int j = 0; j < 32; ++j) p[size_t(k_pair_major(c, j))] = b[offsetof(block_q6_K, ql) + size_t(32 * c + j)];
    } else {
      const size_t kept = size_t(std::min(l.stride, l.bytes - skip));
      std::memcpy(p, b + skip, kept);
      std::memset(p + kept, 0, size_t(l.stride) - kept);  // IQ3_S, Q3_K: 108 bytes in 112
    }
    if (head) {  // the block's head: its scales and their bits (kHeadInScales)
      std::memcpy(s, b + s0, 2 * size_t(l.scales));
      continue;
    }
    if (dtype == FLYX_DT_GGML_IQ1_M) {  // the block's scale, a nibble in the top of each of its four scale words
      uint16_t w[4];
      std::memcpy(w, b + offsetof(block_iq1_m, scales), 8);
      s[0] = uint16_t((w[0] >> 12) | ((w[1] >> 8) & 0x00f0) | ((w[2] >> 4) & 0x0f00) | (w[3] & 0xf000));
      continue;
    }
    std::memcpy(s, b + s0, 2);
    if (s1 >= 0) std::memcpy(s + 1, b + s1, 2);
  }
}

void blocks_stage(const float* x, int ldx, int K, int rows, const Launch& L, bool q8) {
  check_input(x, ldx, K);
  stage<true>(x, ldx, K, rows, q8, L);
}

void blocks_stage_presplit(int K, int rows, const Launch& L, bool q8) {
  const Scratch& sc = scratch(L);
  // A pass's rows in float32 where the scratch keeps them so (a model with NVFP4 inputs; split_out).
  if (rows > 16 && sc.af32) return stage<true>(sc.af32, K, K, rows, q8, L);
  stage<false>(rows <= 16 ? sc.split : sc.abuf, K, K, rows, q8, L);
}

void gemm_blocks_staged(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  const BlockOps& ops = ops_of(W.dtype, "a GEMM");
  const int tuning = opts(L).tuning;
  if (L.glu && (rows > 16 || !W.q8)) {  // gemm_blocks_round_glu_takes said so
    std::fprintf(stderr, "flyx-cuda: a SwiGLU in a GEMM that is not a round's in 8 bits\n");
    std::abort();
  }
  if (rows > 16 && W.q8)
    ops.q8(default_q8_config(tuning, W.dtype, W.rows, rows), W, K, y, ldy, rows, accumulate, L, -1, 0);
  else if (rows > 16) ops.rows(W, K, y, ldy, rows, accumulate, L);
  else if (W.q8) gemm_blocks_round_q8(W, K, y, ldy, rows, accumulate, L);
  else if (!ops.skinny(default_config(tuning, W.rows, rows), W, K, y, ldy, rows, accumulate, L)) unfit(W.rows, rows);
}

// The runs in launches: alone, or two or three neighbours in one where a round's launch takes them.  The fewest
// launches; of those, the one whose narrowest launch is widest (a narrow run is the costly one alone: GDN qkv, z,
// b|a as qkv alone, then z with b|a, rather than qkv with z, then b|a alone).
void gemm_blocks_staged_runs(const BlockMat* W, int n, int K, float* y, int ldy, int rows, bool accumulate,
                             const Launch& L) {
  struct Plan {
    int launches = 0, narrowest = 0, take = 1;  // the runs from i on; take: how many its first launch takes
  };
  std::vector<Plan> plan(size_t(n) + 1);
  plan[size_t(n)].narrowest = INT_MAX;
  for (int i = n - 1; i >= 0; --i) {
    int columns = 0;
    for (int take = 1; take <= 3 && i + take <= n; ++take) {
      columns += W[i + take - 1].rows;
      if (take > 1 && !(opts(L).stacks && gemm_blocks_round_q8_stack_takes(W + i, take, rows, L))) continue;
      const Plan& rest = plan[size_t(i + take)];
      const Plan p{rest.launches + 1, std::min(columns, rest.narrowest), take};
      Plan& best = plan[size_t(i)];
      if (best.launches == 0 || p.launches < best.launches ||
          (p.launches == best.launches && p.narrowest > best.narrowest))
        best = p;
    }
  }
  struct Go {
    int i, take, n0, columns;
  };
  std::vector<Go> go;
  for (int i = 0, n0 = 0; i < n; i += plan[size_t(i)].take) {
    Go g{i, plan[size_t(i)].take, n0, 0};
    for (int j = i; j < i + g.take; ++j) g.columns += W[j].rows;
    n0 += g.columns;
    go.push_back(g);
  }
  auto launch = [&](const Go& g, const Launch& on) {
    if (g.take == 1) gemm_blocks_staged(W[g.i], K, y + g.n0, ldy, rows, accumulate, on);
    else gemm_blocks_round_q8_stack(W + g.i, g.take, K, y + g.n0, ldy, rows, accumulate, on);
  };
  // A round's narrow launches beside its wide ones, on the side stream: alone after a wide launch (the split kernel
  // takes no stack) the GDN's b|a, 96 columns, added 9.3 us a layer to a 10-row round.
  int narrow = 0;
  for (const Go& g : go) narrow += g.columns <= kSideColumns;
  if (!(L.side && L.side->st && !L.glu && opts(L).side_stream && rows <= 16 && narrow > 0 &&
        narrow < int(go.size()))) {
    for (const Go& g : go) launch(g, L);
    return;
  }
  const Scratch beside = side_scratch(scratch(L));
  Launch on = L;
  on.st = L.side->st;
  on.scratch = &beside;
  on.side = nullptr;
  if (cudaEventRecord(L.side->fork, L.st) != cudaSuccess || cudaStreamWaitEvent(on.st, L.side->fork, 0) != cudaSuccess)
    throw std::runtime_error(std::string("flyx-cuda: a fork onto the side stream: ") +
                             cudaGetErrorString(cudaGetLastError()));
  for (const Go& g : go)
    if (g.columns <= kSideColumns) launch(g, on);
  if (cudaEventRecord(L.side->join, on.st) != cudaSuccess)
    throw std::runtime_error(std::string("flyx-cuda: a join from the side stream: ") +
                             cudaGetErrorString(cudaGetLastError()));
  for (const Go& g : go)
    if (g.columns > kSideColumns) launch(g, L);
  if (cudaStreamWaitEvent(L.st, L.side->join, 0) != cudaSuccess)
    throw std::runtime_error(std::string("flyx-cuda: a join from the side stream: ") +
                             cudaGetErrorString(cudaGetLastError()));
}

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

bool gemm_blocks_cfg_compiled(int cfg) {
#if FLYX_TUNE_CANDIDATES
  (void)cfg;
  return true;
#else
  const int arch = compiled_arch();
  return arch > 0 && cfg >= 0 && cfg < FLYX_BLOCKS_SKINNY_COUNT &&
         (tables::kReachBlocksSkinny[tables::arch_bucket(arch)] & (1u << cfg)) != 0;
#endif
}

bool gemm_blocks_q8_cfg_compiled(int dtype, int cfg) {
#if FLYX_TUNE_CANDIDATES
  (void)dtype, (void)cfg;
  return true;
#else
  const int arch = compiled_arch();
  const int di = tables::ggml_type_index(dtype);
  return arch > 0 && di >= 0 && cfg >= 0 && cfg < FLYX_BLOCKS_Q8_COUNT &&
         (tables::kReachBlocksQ8[tables::arch_bucket(arch)][di] & (1u << cfg)) != 0;
#endif
}

bool gemm_blocks_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  return ops_of(W.dtype, "a GEMM").skinny(cfg, W, K, y, ldy, rows, accumulate, L);
}

void gemm_blocks_q8_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                        const Launch& L, int spread, int runs) {
  ops_of(W.dtype, "a GEMM").q8(cfg, W, K, y, ldy, rows, accumulate, L, spread, runs);
}

// The rules (blocks_q8_runs: this GPU's table's, else the defaults) by the layer's columns and K, at most a run a
// window: it decides the bits of a row, whatever the pass.
int gemm_blocks_q8_runs(int tuning, int N, int K) {
  const int s = tables::rules_sku(tuning, tables::kFamilyBlocksQ8Runs);
  int runs = 1;
  for (const tables::BlocksQ8Runs& r : tables::kBlocksQ8Runs)
    if (r.sku == s && N <= r.n_max && K <= r.k_max) {
      runs = r.runs;
      break;
    }
  return std::max(1, std::min(runs, (K + kWindow - 1) / kWindow));
}

int gemm_blocks_q8_default(int tuning, int dtype, int N, int rows) { return default_q8_config(tuning, dtype, N, rows); }
int gemm_blocks_default(int tuning, int N, int rows) { return default_config(tuning, N, rows); }

const BlocksQ8Config kBlocksQ8Configs[] = {
#define X(i, bn, qk, bm) {bn, qk, bm},
    FLYX_BLOCKS_Q8(X)
#undef X
};
const int kBlocksQ8ConfigCount = FLYX_BLOCKS_Q8_COUNT;

const BlocksConfig kBlocksConfigs[] = {
#define X(i, nt, nw, kw) {nt, nw, kw},
    FLYX_BLOCKS_SKINNY(X)
#undef X
};
const int kBlocksConfigCount = FLYX_BLOCKS_SKINNY_COUNT;

void gemm_blocks(const float* x, int ldx, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                 const Launch& L) {
  blocks_stage(x, ldx, K, rows, L, W.q8);
  gemm_blocks_staged(W, K, y, ldy, rows, accumulate, L);
}

void embed_blocks(const int* tokens, const BlockMat& W, int H, float* x, int rows, cudaStream_t st) {
  ops_of(W.dtype, "an embedding").embed(tokens, W, H, x, rows, st);
}

void dots_blocks(const BlockMat& W, int K, const int* idx, const int* count, float* out, size_t ld, int cap,
                 const float* x, int ldx, int rows, cudaStream_t st) {
  ops_of(W.dtype, "dots with rows of a table").dots(W, K, idx, count, out, ld, cap, x, ldx, rows, st);
}

}  // namespace flyx::cuda
