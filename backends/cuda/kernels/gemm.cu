// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GEMM dispatch (path by rows and shape, configuration from the tuning tables), NVFP4 quantization.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "flyx/gemm_tables.inc"
#include "backends/cuda/kernels/gemm_internal.h"

namespace flyx::cuda {
namespace {

// ---- NVFP4 copy of a bf16 matrix ---------------------------------------------------------
__device__ __forceinline__ float e4m3_value(uint8_t b) {
  const int e = (b >> 3) & 0xF, m = b & 7;
  const float v = e == 0 ? m * 0.125f * 0.015625f : (1.f + m * 0.125f) * exp2f(float(e - 7));
  return (b & 0x80) ? -v : v;
}

// grid N; block 256.  Row n: RS[n] = max|W[n]| / (6 * 448) keeps the block
// scales in e4m3 range; each block of 16 gets scale e4m3(max / 6 / RS[n]) and
// codes e2m1(w / (scale * RS[n])), both rounded to nearest.
__global__ void k_quantize_nvfp4(const bf16* __restrict__ W, int K, uint8_t* __restrict__ Q, uint8_t* __restrict__ S,
                                 float* __restrict__ RS) {
  pdl_trigger();
  const int n = blockIdx.x;
  const bf16* w = W + size_t(n) * K;
  float amax = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(w[k])));
  amax = block_max(amax);
  const float rs = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
  if (threadIdx.x == 0) RS[n] = rs;
  for (int b = threadIdx.x; b < K / 16; b += blockDim.x) {
    float v[16], bmax = 0.f;
#pragma unroll
    for (int e = 0; e < 16; ++e) {
      v[e] = __bfloat162float(w[b * 16 + e]);
      bmax = fmaxf(bmax, fabsf(v[e]));
    }
    const uint8_t sc = __nv_cvt_float_to_fp8(bmax / 6.f / rs, __NV_SATFINITE, __NV_E4M3);
    const float scale = e4m3_value(sc) * rs, inv = scale > 0.f ? 1.f / scale : 0.f;
    S[size_t(n) * (K / 16) + b] = sc;
#pragma unroll
    for (int e = 0; e < 16; e += 2)
      Q[size_t(n) * (K / 2) + b * 8 + e / 2] = uint8_t(e2m1_nearest(v[e] * inv) | (e2m1_nearest(v[e + 1] * inv) << 4));
  }
}

// ---- Q8_0 copy of a bf16 matrix -----------------------------------------------------------
// A thread a block of 32 weights: d = binary16(max |w| / 127), q = round(w / d) (|q| <= 127), the codes as the
// kernels hold a Q8_0 block's payload (32 bytes) and d in the scales' plane; weight = d q.
__global__ void k_quantize_q8_0(const bf16* __restrict__ W, size_t blocks, uint8_t* __restrict__ payload,
                                uint16_t* __restrict__ scales) {
  const size_t b = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= blocks) return;
  const bf16* w = W + b * 32;
  float v[32], amax = 0.f;
#pragma unroll
  for (int e = 0; e < 32; ++e) {
    v[e] = __bfloat162float(w[e]);
    amax = fmaxf(amax, fabsf(v[e]));
  }
  const __half dh = __float2half_rn(amax / 127.f);
  const float d = __half2float(dh), inv = d > 0.f ? 1.f / d : 0.f;
  scales[b] = __half_as_ushort(dh);
  uint32_t words[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = max(-127, min(127, __float2int_rn(v[4 * i + j] * inv)));
      word |= uint32_t(uint8_t(int8_t(q))) << (8 * j);
    }
    words[i] = word;
  }
  uint4* out = reinterpret_cast<uint4*>(payload + b * 32);
  out[0] = make_uint4(words[0], words[1], words[2], words[3]);
  out[1] = make_uint4(words[4], words[5], words[6], words[7]);
}

}  // namespace

void quantize_q8_0(const bf16* W, int N, int K, uint8_t* payload, uint16_t* scales, cudaStream_t st) {
  const size_t blocks = size_t(N) * (K / 32);
  k_quantize_q8_0<<<unsigned((blocks + 255) / 256), 256, 0, st>>>(W, blocks, payload, scales);
}

namespace {

// The configuration for one GEMM: the table of this GPU (or the one it
// inherits) for the shape, else the rules for shapes no table lists (this
// GPU's, else the defaults).  Never depends on anything but the table, the
// format, the shape and the row count.
// k_gemm_skinny_mt's formats (17..64 rows, a bf16_mt configuration): a round's in three parts, a drafting pass's in
// one, its chunks K's (tables::draft_chunks) whatever the table.
constexpr int kFmtBf16Rows = 3, kFmtBf16Draft = 4;

int pick_config(int sku, int fmt, int N, int K, int rows) {
  for (int s = sku; s >= 0; s = tables::kSkus[s].parent)
    for (int i = 0; i < tables::kRuleCount; ++i) {
      const auto& r = tables::kRules[i];
      if (r.sku == s && r.fmt == fmt && r.n == N && r.k == K && rows >= r.lo && rows <= r.hi) return r.cfg;
    }
  if (fmt == 1) {
    const int s = tables::rules_sku(sku, tables::kFamilyNvfp4);
    for (const auto& r : tables::kNvfp4Rules)
      if (r.sku == s) return K % 128 == 0 ? r.k128 : r.k64;
  }
  if (fmt == kFmtBf16Rows) {
    const int s = tables::rules_sku(sku, tables::kFamilyBf16Rows);
    for (const auto& r : tables::kBf16RowsRules)
      if (r.sku == s && rows >= r.lo && rows <= r.hi) return r.cfg;
    std::abort();  // the generator checks that the rules cover 17..64 rows
  }
  if (fmt == kFmtBf16Draft) {
    const int s = tables::rules_sku(sku, tables::kFamilyBf16Draft);
    for (const auto& r : tables::kBf16DraftRules)
      if (r.sku == s && rows >= r.lo && rows <= r.hi)
        return tables::draft_chunks(K) == tables::kDraftChunksLong ? r.cfg_long : r.cfg;
    std::abort();
  }
  const int s = tables::rules_sku(sku, tables::kFamilyBf16);
  for (const auto& r : tables::kBf16Rows)
    if (r.sku == s && rows >= r.lo && rows <= r.hi) return r.cfg;
  return 0;
}

}  // namespace

bool device_supported() { return compiled_arch() > 0; }

void quantize_nvfp4(const bf16* W, int N, int K, uint8_t* Q, uint8_t* S, float* RS, cudaStream_t st) {
  k_quantize_nvfp4<<<N, 256, 0, st>>>(W, K, Q, S, RS);
}

namespace {
// The pipelined kernel needs K % 32, an even row stride and room in the bf16 scratch.
bool tc_fits(const float* x, int ldx, int K) {
  return K % 32 == 0 && ldx % 2 == 0 && (reinterpret_cast<uintptr_t>(x) % 8) == 0;
}
}  // namespace

int gemm_tuning_for(const char* device_name, int cc, bool* by_arch) {
  int sku = -1;
  for (int s = 0; s < tables::kSkuCount; ++s)
    for (int d = 0; d < tables::kSkus[s].n_devices; ++d)
      if (std::strcmp(tables::kSkus[s].devices[d], device_name) == 0) sku = s;
  if (by_arch) *by_arch = false;
  for (int i = 0; sku < 0 && i < tables::kArchTableCount; ++i)
    if (tables::kArchTables[i].cc == cc) {
      sku = tables::kArchTables[i].sku;
      if (by_arch) *by_arch = true;
    }
  return sku;
}

int gemm_tuning_named(const char* name) {
  for (int s = 0; s < tables::kSkuCount; ++s)
    if (std::strcmp(tables::kSkus[s].name, name) == 0) return s;
  return -1;
}

std::string gemm_tuning_names() {
  std::string all;
  for (int s = 0; s < tables::kSkuCount; ++s) all += (s ? ", " : "") + std::string(tables::kSkus[s].name);
  return all.empty() ? "none" : all;
}

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

const char* gemm_tuning_name(int tuning) { return tuning >= 0 ? tables::kSkus[tuning].name : nullptr; }

bool gemm_tuning_compiled_for(int tuning, int cc) {
#if FLYX_TUNE_CANDIDATES
  (void)tuning, (void)cc;
  return true;
#else
  if (tuning < 0 || tuning >= tables::kSkuCount) return true;
  const int sa = tables::kSkuArch[tuning];
  return sa == 0 || sa == cc || (cc == 80 && (sa == 86 || sa == 89));
#endif
}

bool gemm_tuning_pdl(int tuning) { return tuning >= 0 && tables::kSkus[tuning].pdl; }

int gemm_config(int tuning, int fmt, int N, int K, int rows) { return pick_config(tuning, fmt, N, K, rows); }

int gemm_draft_chunks(int K) { return tables::draft_chunks(K); }

void gemm_skinny(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                 const Launch& L) {
  gemm_skinny_cfg(pick_config(opts(L).tuning, 0, N, K, rows), x, ldx, W, N, K, y, ldy, rows, accumulate, L);
}

// The tile and unroll are the first weight's shape's; the chunks, and so every weight's sums, K's.
void gemm_draft(const bf16* x, int ldx, int K, const MtGroups& gs, int rows, const Launch& L) {
  gemm_draft_cfg(pick_config(opts(L).tuning, kFmtBf16Draft, gs.g[0].N, K, rows), x, ldx, K, gs, rows, L);
}

void gemm_w12_skinny(const float* x, int ldx, const W12& W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                     const Launch& L) {
  gemm_w12_skinny_cfg(pick_config(opts(L).tuning, 2, N, K, rows), x, ldx, W, N, K, y, ldy, rows, accumulate, L);
}

void gemm_presplit(const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  if (rows > 16)
    gemm_tc_staged(W, nullptr, nullptr, nullptr, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_skinny_cfg(pick_config(opts(L).tuning, 0, N, K, rows), nullptr, K, W, N, K, y, ldy, rows, accumulate, L);
}

namespace {
// grid (rows, ceil(n / 256)); block 256: u[r][c] *= scale, one correctly rounded product.
__global__ void k_scale_rows(float* u, int ldu, int n, float scale) {
  pdl_trigger();
  pdl_wait();
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  if (c < n) {
    float* p = u + size_t(blockIdx.x) * ldu + c;
    *p = __fmul_rn(*p, scale);
  }
}
}  // namespace

void lora_down(const float* x, int ldx, const bf16* a, int n, int K, float scale, float* u, int ldu, int rows,
               const Launch& L) {
  gemm(x, ldx, a, n, K, u, ldu, rows, false, L);
  if (scale != 1.f)
    launch_dependent(k_scale_rows, dim3(rows, ceil_div(n, 256)), dim3(256), L, u, ldu, n, scale);
}

namespace {
// grid ceil(outputs G / 128); block 128: G lanes an output column of one linear, lane l its b values 8 (l + G c), up
// to kUpRows rows at once (b read once for them); each row's G partial sums (k's order) added in a fixed tree.
constexpr int kUpRows = 8;
template <int G>
__global__ void __launch_bounds__(128) k_lora_up(const float* u, int ldu, int rank, float scale, LoraUp up, float* y,
                                                 int ldy, int rows) {
  pdl_trigger();
  const int t = int(blockIdx.x) * blockDim.x + int(threadIdx.x), l = t % G;
  int n = t / G, i = 0;
  while (i < up.linears && n >= up.n[i]) n -= up.n[i++];
  pdl_wait();
  const bool live = i < up.linears;  // every lane of a group alike (the shuffles want the whole warp)
  const int seg = live ? i : 0;
  const uint4* b = reinterpret_cast<const uint4*>(up.b[seg] + size_t(live ? n : 0) * rank);
  const int chunks = rank / (8 * G);
  for (int r0 = 0; r0 < rows; r0 += kUpRows) {
    const int nr = min(kUpRows, rows - r0);
    float acc[kUpRows];
#pragma unroll
    for (int r = 0; r < kUpRows; ++r) acc[r] = 0.f;
    if (live) {
      for (int c = 0; c < chunks; ++c) {
        const int k8 = l + G * c;
        const uint4 w = b[k8];
        const uint32_t p[4] = {w.x, w.y, w.z, w.w};
        float bv[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) bv[2 * e] = __uint_as_float(p[e] << 16), bv[2 * e + 1] = __uint_as_float(p[e] & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < kUpRows; ++r) {
          if (r >= nr) break;
          const float4* ur = reinterpret_cast<const float4*>(u + size_t(r0 + r) * ldu + up.u_col[seg] + 8 * k8);
          const float4 ua = ur[0], ub = ur[1];
          const float uv[8] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[r] = __fmaf_rn(__fmul_rn(uv[e], scale), bv[e], acc[r]);
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kUpRows; ++r) {
#pragma unroll
      for (int o = G / 2; o > 0; o >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
    }
    if (live && l == 0) {
#pragma unroll
      for (int r = 0; r < kUpRows; ++r) {
        if (r >= nr) break;
        float* yr = y + size_t(r0 + r) * ldy + up.y_col[seg] + n;
        *yr = *yr + acc[r];
      }
    }
  }
}
}  // namespace

namespace {
// grid (n, ceil(rows / 64)); block 256: output j of 64 rows, up to 16 rows a pass over a's row (8 values a thread a step,
// k = 8 (t + 256 s)), then each row's 256 sums added warp by warp in a fixed order.
constexpr int kLoraDownRows = 16, kLoraDownBlockRows = 64;
__global__ void __launch_bounds__(256) k_lora_down(const float* x, int ldx, const bf16* a, int K, float* u, int ldu,
                                                   int rows) {
  pdl_trigger();
  __shared__ float part[8][kLoraDownRows];
  const int j = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint4* ar = reinterpret_cast<const uint4*>(a + size_t(j) * K);
  const int r_end = min(rows, int(blockIdx.y + 1) * kLoraDownBlockRows);
  for (int r0 = int(blockIdx.y) * kLoraDownBlockRows; r0 < r_end; r0 += kLoraDownRows) {
    const int nr = min(kLoraDownRows, r_end - r0);
    float acc[kLoraDownRows];
#pragma unroll
    for (int r = 0; r < kLoraDownRows; ++r) acc[r] = 0.f;
    for (int k8 = threadIdx.x; k8 < K / 8; k8 += 256) {
      const uint4 w = ar[k8];
      const uint32_t p[4] = {w.x, w.y, w.z, w.w};
      float av[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) av[2 * e] = __uint_as_float(p[e] << 16), av[2 * e + 1] = __uint_as_float(p[e] & 0xffff0000u);
#pragma unroll
      for (int r = 0; r < kLoraDownRows; ++r) {
        if (r >= nr) break;
        const float* xr = x + size_t(r0 + r) * ldx + 8 * k8;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[r] = __fmaf_rn(xr[e], av[e], acc[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < kLoraDownRows; ++r) {
      if (r >= nr) break;
      const float v = warp_sum(acc[r]);
      if (!lane) part[warp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < nr) {
      float t = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) t += part[w][threadIdx.x];
      u[size_t(r0 + threadIdx.x) * ldu + j] = t;
    }
    __syncthreads();
  }
}
}  // namespace

void lora_down_round(const float* x, int ldx, const bf16* a, int n, int K, float* u, int ldu, int rows,
                     cudaStream_t st) {
  if (K % 8) throw std::logic_error("lora_down_round: K is no multiple of 8");
  k_lora_down<<<dim3(n, ceil_div(rows, kLoraDownBlockRows)), 256, 0, st>>>(x, ldx, a, K, u, ldu, rows);
}

namespace {
// Split over K (kLoraSliceK of it a slice, the slices K alone sets): grid (slices, ceil(rows / 16), ceil(n / 32));
// block 256 (8 warps): x's 16 rows of the slice in shared memory, a warp an output at a time (32 of them a block), a
// lane its 8 values of the slice (k = k0 + 8 lane + e, in order), each row's 32 lane sums added by warp_sum; the
// slice's sum to part[slice][row][j].  Then k_lora_down_sum adds a row's slices in order: a row's bits whatever the
// rows, the tiles or the outputs a block.
constexpr int kLoraSliceK = 256, kLoraTileRows = 16, kLoraBlockOut = 32;
// Split: a block a slice (grid.x), its sums to part.  Else a block every slice in order, each row's slice sums added
// up as k_lora_down_sum adds them (0 + the first, + each next: the same bits), to u.
template <bool Split>
__global__ void __launch_bounds__(256) k_lora_down_split(const float* x, int ldx, const bf16* a, int n, int K,
                                                         float* out, int ldu, int rows) {
  pdl_trigger();
  __shared__ __align__(16) float xs[kLoraTileRows][kLoraSliceK];
  const int r0 = int(blockIdx.y) * kLoraTileRows, j0 = int(blockIdx.z) * kLoraBlockOut;
  const int nr = min(kLoraTileRows, rows - r0), slices = (K + kLoraSliceK - 1) / kLoraSliceK;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float tot[kLoraBlockOut / 8][kLoraTileRows];
#pragma unroll
  for (int q = 0; q < kLoraBlockOut / 8; ++q)
#pragma unroll
    for (int r = 0; r < kLoraTileRows; ++r) tot[q][r] = 0.f;
  pdl_wait();
  for (int sl = Split ? int(blockIdx.x) : 0; sl < (Split ? int(blockIdx.x) + 1 : slices); ++sl) {
    const int k0 = sl * kLoraSliceK, kc = min(kLoraSliceK, K - k0);
    if (!Split) __syncthreads();  // the last slice's reads of xs
    for (int i = threadIdx.x; i < kLoraTileRows * kLoraSliceK; i += 256) {
      const int r = i / kLoraSliceK, k = i % kLoraSliceK;
      xs[r][k] = r < nr && k < kc ? x[size_t(r0 + r) * ldx + k0 + k] : 0.f;
    }
    __syncthreads();
    const bool live = 8 * lane < kc;
#pragma unroll
    for (int q = 0; q < kLoraBlockOut / 8; ++q) {
      const int j = j0 + warp + 8 * q;
      if (j >= n) break;
      float av[8] = {};
      if (live) {
        const uint4 w = *reinterpret_cast<const uint4*>(a + size_t(j) * K + k0 + 8 * lane);
        const uint32_t p[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int e = 0; e < 4; ++e)
          av[2 * e] = __uint_as_float(p[e] << 16), av[2 * e + 1] = __uint_as_float(p[e] & 0xffff0000u);
      }
#pragma unroll
      for (int r = 0; r < kLoraTileRows; ++r) {
        if (r >= nr) break;
        const float4 xa = *reinterpret_cast<const float4*>(&xs[r][8 * lane]);
        const float4 xb = *reinterpret_cast<const float4*>(&xs[r][8 * lane + 4]);
        const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
        float acc = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc = __fmaf_rn(xv[e], av[e], acc);
        acc = warp_sum(acc);
        if (Split) {
          if (!lane) out[(size_t(sl) * rows + r0 + r) * n + j] = acc;
        } else {
          tot[q][r] += acc;
        }
      }
    }
  }
  if (!Split && !lane) {
#pragma unroll
    for (int q = 0; q < kLoraBlockOut / 8; ++q) {
      const int j = j0 + warp + 8 * q;
      if (j >= n) break;
#pragma unroll
      for (int r = 0; r < kLoraTileRows; ++r)
        if (r < nr) out[size_t(r0 + r) * ldu + j] = tot[q][r];
    }
  }
}
// grid ceil(rows n / 256): u[r][j] = the slices' sums in order.
__global__ void k_lora_down_sum(const float* part, int slices, int n, float* u, int ldu, int rows) {
  pdl_trigger();
  const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  pdl_wait();
  if (i >= rows * n) return;
  const int r = i / n, j = i % n;
  float t = 0.f;
  for (int sl = 0; sl < slices; ++sl) t += part[(size_t(sl) * rows + r) * n + j];
  u[size_t(r) * ldu + j] = t;
}
}  // namespace

int lora_down_part_floats(int n, int K, int rows) { return ceil_div(K, kLoraSliceK) * n * rows; }

void lora_down_rows(const float* x, int ldx, const bf16* a, int n, int K, float* u, int ldu, int rows, float* part,
                    int part_rows, cudaStream_t st) {
  if (K % 8) throw std::logic_error("lora_down_rows: K is no multiple of 8");
  const int slices = ceil_div(K, kLoraSliceK), tiles = ceil_div(rows, kLoraTileRows), groups = ceil_div(n, kLoraBlockOut);
  // Split over K while the tiles are too few to fill the GPU (the same bits either way).
  if (part && rows <= part_rows && tiles * groups < 128) {
    k_lora_down_split<true><<<dim3(slices, tiles, groups), 256, 0, st>>>(x, ldx, a, n, K, part, 0, rows);
    k_lora_down_sum<<<ceil_div(rows * n, 256), 256, 0, st>>>(part, slices, n, u, ldu, rows);
  } else {
    k_lora_down_split<false><<<dim3(1, tiles, groups), 256, 0, st>>>(x, ldx, a, n, K, u, ldu, rows);
  }
}

namespace {
// A prefill pass's second product, one linear: grid (ceil(n / 64), ceil(rows / 64)); block 256 (16 x 16 threads, each 4
// rows by 4 outputs), u (scaled, as lora_up rounds it) and b through shared memory 32 of k at a time; each output's sum
// in k's order.
constexpr int kUpTile = 64, kUpChunk = 32;
__global__ void __launch_bounds__(256) k_lora_up_tile(const float* u, int ldu, int u_col, int rank, float scale,
                                                      const bf16* b, int n, float* y, int ldy, int y_col, int rows) {
  pdl_trigger();
  __shared__ float us[kUpChunk][kUpTile + 1];
  __shared__ float bs[kUpChunk][kUpTile + 1];
  const int r0 = int(blockIdx.y) * kUpTile, n0 = int(blockIdx.x) * kUpTile;
  const int tr = threadIdx.x / 16, tn = threadIdx.x % 16;
  float acc[4][4] = {};
  pdl_wait();
  for (int k0 = 0; k0 < rank; k0 += kUpChunk) {
    const int kc = min(kUpChunk, rank - k0);
    for (int i = threadIdx.x; i < kUpTile * kUpChunk; i += 256) {
      const int rr = i / kUpChunk, kk = i % kUpChunk;
      us[kk][rr] = r0 + rr < rows && kk < kc ? __fmul_rn(u[size_t(r0 + rr) * ldu + u_col + k0 + kk], scale) : 0.f;
      bs[kk][rr] = n0 + rr < n && kk < kc ? __bfloat162float(b[size_t(n0 + rr) * rank + k0 + kk]) : 0.f;
    }
    __syncthreads();
    for (int kk = 0; kk < kc; ++kk) {
      float a[4], w[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) a[i] = us[kk][tr * 4 + i], w[i] = bs[kk][tn + 16 * i];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(a[i], w[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int r = r0 + tr * 4 + i;
    if (r >= rows) break;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = n0 + tn + 16 * j;
      if (c < n) {
        float* yr = y + size_t(r) * ldy + y_col + c;
        *yr = *yr + acc[i][j];
      }
    }
  }
}
}  // namespace

void lora_up_rows(const float* u, int ldu, int rank, float scale, const LoraUp& up, float* y, int ldy, int rows,
                  const Launch& L) {
  if (rank % 8 || up.linears < 1 || up.linears > 5)
    throw std::logic_error("lora_up_rows: a rank that is no multiple of 8, or 1 to 5 linears");
  for (int i = 0; i < up.linears; ++i)
    launch_dependent(k_lora_up_tile, dim3(ceil_div(up.n[i], kUpTile), ceil_div(rows, kUpTile)), dim3(256), L, u, ldu,
                     up.u_col[i], rank, scale, up.b[i], up.n[i], y, ldy, up.y_col[i], rows);
}

void lora_up(const float* u, int ldu, int rank, float scale, const LoraUp& up, float* y, int ldy, int rows,
             const Launch& L) {
  if (rank % 8 || up.linears < 1 || up.linears > 5 || ldu % 4 || reinterpret_cast<uintptr_t>(u) % 16)
    throw std::logic_error("lora_up: a rank that is no multiple of 8, 1 to 5 linears, or u rows not on 16 bytes");
  int total = 0;
  for (int i = 0; i < up.linears; ++i) total += up.n[i];
  // G lanes an output, each 16 bytes of its b row a step, two steps at least; G the rank's alone, so a row's bits are
  // too.  Measured (flyx bench --prompts, -n 256, Qwen3.8-27B UD-Q4_K_M, RTX 5090 at 450 W, option lora_lanes; plain
  // decoding tok/s, MTP rounds/s): rank 16 one lane 85.7 / 57.5, two 84.4 / 55.9; rank 64 on every linear four 80.1 /
  // 50.5, one 79.0 / 48.0, two 80.2 / 49.7, eight 78.3 / 49.1.
  auto go = [&](auto g) {
    constexpr int G = decltype(g)::value;
    launch_dependent(k_lora_up<G>, dim3(ceil_div(total * G, 128)), dim3(128), L, u, ldu, rank, scale, up, y, ldy,
                     rows);
  };
  switch (lora_lanes(rank, opts(L).lora_lanes)) {
    case 8: go(std::integral_constant<int, 8>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    default: go(std::integral_constant<int, 1>{});
  }
}

void gemm_fp4_presplit(const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y, int ldy, int rows,
                       bool accumulate, const Launch& L) {
  if (rows > 16)
    gemm_tc_staged(nullptr, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_fp4_skinny_cfg(pick_config(opts(L).tuning, 1, N, K, rows), nullptr, K, Q, S, RS, N, K, y, ldy, rows,
                        accumulate, L);
}

void gemm_fp4(const float* x, int ldx, const uint8_t* Q, const uint8_t* S, const float* RS, int N, int K, float* y,
              int ldy, int rows, bool accumulate, const Launch& L) {
  if (rows <= 16 && (K % 64) == 0 && (ldx % 2) == 0 && (reinterpret_cast<uintptr_t>(x) % 8) == 0)
    gemm_fp4_skinny_cfg(pick_config(opts(L).tuning, 1, N, K, rows), x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate,
                        L);
  else if (rows > 16 && tc_fits(x, ldx, K))
    gemm_fp4_tc(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L);
  else
    gemm_fp4_mma(x, ldx, Q, S, RS, N, K, y, ldy, rows, accumulate, L.st);
}

void gemm(const float* x, int ldx, const bf16* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
          const Launch& L) {
  const bool row_path = opts(L).row_gemms;
  if (rows > 16 && (L.round_rows || L.draft)) {
    // Every tile of 16 rows the 1..16-row kernel's arithmetic, the weights read once (a round's: a linear's bf16
    // weights, regions/linear.cpp, and an adapter's; numerics.md R, C), or bf16 inputs (a drafting pass's:
    // deterministic).
    if (rows <= 64 && !row_path && skinny_fits(x, ldx, W, K)) {
      const int fmt = L.round_rows ? kFmtBf16Rows : kFmtBf16Draft;
      gemm_skinny_mt_cfg(pick_config(opts(L).tuning, fmt, N, K, rows), x, ldx, W, N, K, y, ldy, rows, accumulate, L);
      return;
    }
    if (L.round_rows) {  // a round's: the 1..16-row choice a tile of 16 rows at a time (numerics.md: R, C)
      for (int r0 = 0; r0 < rows; r0 += 16)
        gemm(x + size_t(r0) * ldx, ldx, W, N, K, y + size_t(r0) * ldy, ldy, std::min(16, rows - r0), accumulate, L);
      return;
    }
  }
  if (rows <= 16 && !row_path && skinny_fits(x, ldx, W, K))
    gemm_skinny(x, ldx, W, N, K, y, ldy, rows, accumulate, L);
  else if (rows <= 16 || row_path || (ldx % 4) != 0) gemm_rows(x, ldx, W, N, K, y, ldy, rows, accumulate, L.st);
  else if (tc_fits(x, ldx, K)) gemm_tc(x, ldx, W, N, K, y, ldy, rows, accumulate, L);
  else gemm_mma(x, ldx, W, N, K, y, ldy, rows, accumulate, L.st);
}

}  // namespace flyx::cuda
