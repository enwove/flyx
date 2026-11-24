// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Row-wise kernels: RMS norm, embedding, round rows, SwiGLU, residual add.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// ---- small elementwise / row kernels ------------------------------------------------------

// One row per block (512 threads).  Cache up to kNormCache elements and weights per thread (10 * 512 = 5120,
// covering 27B's hidden size in registers with zero second-pass DRAM reads of x or w).
constexpr int kNormCache = 10;

// Exact block_sum order for 512-thread (16-warp) blocks in a single barrier.
__device__ __forceinline__ float block512_sum(float v) {
  __shared__ float sh[16];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sh[warp] = v;
  __syncthreads();
  const float t = lane < 16 ? sh[lane] : 0.f;
  return warp_sum(t);
}

// A row's 1 / rms in block_sum's exact order for blocks of 512 threads, each thread its values tid, tid + 512, ...:
// the first kNormCache of them loaded in parallel with their weights (xv, wv), then the rest.  Every rmsnorm kernel
// takes the same order: same bits.
template <class TIn>
__device__ __forceinline__ float rms_inv(const TIn* xr, const float* w, int n, float eps, float (&xv)[kNormCache],
                                         float (&wv)[kNormCache]) {
#pragma unroll
  for (int u = 0; u < kNormCache; ++u) {
    const int i = threadIdx.x + u * 512;
    if (i < n) xv[u] = in_value(xr + i), wv[u] = w[i];
  }
  float ss = 0.f;
#pragma unroll
  for (int u = 0; u < kNormCache; ++u)
    if (threadIdx.x + u * 512 < n) ss = __fmaf_rn(xv[u], xv[u], ss);
  for (int i = threadIdx.x + kNormCache * 512; i < n; i += 512) {
    const float v = in_value(xr + i);
    ss = __fmaf_rn(v, v, ss);
  }
  return rsqrtf(block512_sum(ss) / n + eps);
}

// Same 1 / rms order for spread round kernels (one output tile blockIdx.y per block): issues all kNormCache loads of
// xr in parallel while loading only this block's single weight element wi (no wasted weights, zero post-barrier reads).
__device__ __forceinline__ float rms_inv_spread(const float* xr, int n, float eps, int tile_y, int want_i,
                                                float* x_out) {
  float xv[kNormCache];
#pragma unroll
  for (int u = 0; u < kNormCache; ++u) {
    const int i = threadIdx.x + u * 512;
    if (i < n) xv[u] = xr[i];
  }
  float ss = 0.f, xi = 0.f;
#pragma unroll
  for (int u = 0; u < kNormCache; ++u) {
    if (threadIdx.x + u * 512 < n) {
      ss = __fmaf_rn(xv[u], xv[u], ss);
      if (u == tile_y) xi = xv[u];
    }
  }
  for (int i = threadIdx.x + kNormCache * 512; i < n; i += 512) {
    const float v = xr[i];
    ss = __fmaf_rn(v, v, ss);
    if (i == want_i) xi = v;
  }
  *x_out = xi;
  return rsqrtf(block512_sum(ss) / n + eps);
}

__device__ __forceinline__ float rms_value(float x, float inv, float offset, float w) { return x * inv * (offset + w); }

template <class TIn = float>
__global__ void __launch_bounds__(512, 2) k_rmsnorm(const TIn* x, int ldx, const float* w, float offset, float eps,
                                                    Out o, int ldy, int n) {
  pdl_trigger();
  const TIn* xr = x + size_t(blockIdx.x) * ldx;
  float xv[kNormCache], wv[kNormCache];
  const float inv = rms_inv(xr, w, n, eps, xv, wv);
  if (o.q8) {  // 8-bit rows (n % 32 == 0) of a prefill pass: a warp holds 32 consecutive values, one group
#pragma unroll
    for (int u = 0; u < kNormCache; ++u) {
      const int i = threadIdx.x + u * 512;
      if (i < n) put_q8(o, blockIdx.x, i >> 5, rms_value(xv[u], inv, offset, wv[u]));
    }
    for (int i = threadIdx.x + kNormCache * 512; i < n; i += 512)
      put_q8(o, blockIdx.x, i >> 5, rms_value(in_value(xr + i), inv, offset, w[i]));
    return;
  }
#pragma unroll
  for (int u = 0; u < kNormCache; ++u) {
    const int i = threadIdx.x + u * 512;
    if (i < n) put(o, size_t(blockIdx.x) * ldy + i, size_t(blockIdx.x) * n + i, rms_value(xv[u], inv, offset, wv[u]));
  }
  for (int i = threadIdx.x + kNormCache * 512; i < n; i += 512)
    put(o, size_t(blockIdx.x) * ldy + i, size_t(blockIdx.x) * n + i, rms_value(in_value(xr + i), inv, offset, w[i]));
}

// The same rows in 8 bits (n % 32 == 0) for a round's few rows: grid (rows, ceil(n / 512)).  Every block finds the
// row's 1 / rms as k_rmsnorm does, then each of its 16 warps writes one group of 32 values, so a round's one row is
// spread over the GPU (one block a row ran the 27B's 160 groups one after another: 5.4 us against 2.3).  A prefill
// pass's rows fill the GPU anyway, and there every block would read its row again: k_rmsnorm writes them.
__global__ void __launch_bounds__(512) k_rmsnorm_q8(const float* x, int ldx, const float* w, float offset, float eps,
                                                    Out o, int n) {
  pdl_trigger();
  const float* xr = x + size_t(blockIdx.x) * ldx;
  const int grp = int(blockIdx.y) * 16 + int(threadIdx.x >> 5), i = 32 * grp + (threadIdx.x & 31);
  // Preload only this block's own weight element before the reduction so its latency overlaps with xr.
  const float wi = i < n ? w[i] : 0.f;
  float xi = 0.f;
  const float inv = rms_inv_spread(xr, n, eps, int(blockIdx.y), i, &xi);
  if (i < n) put_q8(o, blockIdx.x, grp, rms_value(xi, inv, offset, wi));
}

// The same spread for a round's few rows in any other form (float32, or bf16 parts for the skinny GEMMs): grid (rows,
// ceil(n / 512)), every block the row's 1 / rms as k_rmsnorm does, then a value a thread: the bits of k_rmsnorm,
// which ran a round's one row on one SM.
__global__ void __launch_bounds__(512) k_rmsnorm_spread(const float* x, int ldx, const float* w, float offset,
                                                        float eps, Out o, int ldy, int n) {
  pdl_trigger();
  const float* xr = x + size_t(blockIdx.x) * ldx;
  const int i = int(blockIdx.y) * 512 + int(threadIdx.x);
  const float wi = i < n ? w[i] : 0.f;
  float xi = 0.f;
  const float inv = rms_inv_spread(xr, n, eps, int(blockIdx.y), i, &xi);
  if (i < n) put(o, size_t(blockIdx.x) * ldy + i, size_t(blockIdx.x) * n + i, rms_value(xi, inv, offset, wi));
}

__global__ void k_embed(const int* tokens, const bf16* E, int H, float* x) {
  pdl_trigger();
  const size_t row = size_t(tokens[blockIdx.x]) * H;
  float* xr = x + size_t(blockIdx.x) * H;
  if (H % 8 == 0 && (reinterpret_cast<uintptr_t>(E) & 15) == 0) {  // 16-byte reads: the table may be across PCIe
    const uint4* er = reinterpret_cast<const uint4*>(E + row);
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x) {
      const uint4 u = er[i];
      const uint32_t w[4] = {u.x, u.y, u.z, u.w};
      float v[8];
#pragma unroll
      for (int e = 0; e < 4; ++e) v[2 * e] = __uint_as_float(w[e] << 16), v[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
      reinterpret_cast<float4*>(xr)[2 * i] = make_float4(v[0], v[1], v[2], v[3]);
      reinterpret_cast<float4*>(xr)[2 * i + 1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    return;
  }
  for (int i = threadIdx.x; i < H; i += blockDim.x) xr[i] = __bfloat162float(E[row + i]);
}

// The table's bf16 rows as they are (a prefill pass's bf16 residual), 16 bytes at a time where they align.
__global__ void k_embed16(const int* tokens, const bf16* E, int H, bf16* x) {
  pdl_trigger();
  const bf16* er = E + size_t(tokens[blockIdx.x]) * H;
  bf16* xr = x + size_t(blockIdx.x) * H;
  if (H % 8 == 0 && (reinterpret_cast<uintptr_t>(E) & 15) == 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
    for (int i = threadIdx.x; i < H / 8; i += blockDim.x)
      reinterpret_cast<uint4*>(xr)[i] = reinterpret_cast<const uint4*>(er)[i];
    return;
  }
  for (int i = threadIdx.x; i < H; i += blockDim.x) xr[i] = er[i];
}

__global__ void k_build_rows(const DevParams* P, const int* drafts, int fill, int* tokens, int rows) {
  pdl_trigger();
  for (int i = threadIdx.x; i < rows; i += blockDim.x)
    tokens[i] = i == 0 ? P->anchor : (fill >= 0 ? fill : drafts[i - 1]);
}

// grid (rows, pairs / 256): one pair of values per thread.
__global__ void k_swiglu(const float* gu, int ld, int I, Out o, int ld_out) {
  pdl_trigger();
  const int row = blockIdx.x, i = 2 * (blockIdx.y * blockDim.x + threadIdx.x);
  if (i >= I) return;
  const float* r = gu + size_t(row) * ld;
  if (i + 1 >= I) {
    put(o, size_t(row) * ld_out + i, size_t(row) * I + i, silu(r[i]) * r[I + i]);
    return;
  }
  float g0, g1, u0, u1;
  if (((reinterpret_cast<uintptr_t>(r + i) | reinterpret_cast<uintptr_t>(r + I + i)) & 7) == 0) {
    const float2 gv = *reinterpret_cast<const float2*>(r + i);
    const float2 uv = *reinterpret_cast<const float2*>(r + I + i);
    g0 = gv.x, g1 = gv.y, u0 = uv.x, u1 = uv.y;
  } else {
    g0 = r[i], g1 = r[i + 1], u0 = r[I + i], u1 = r[I + i + 1];
  }
  const float a = silu(g0) * u0;
  const float b = silu(g1) * u1;
  if (o.q8) {  // 8-bit rows (I % 64 == 0): a warp holds 64 consecutive values, two groups
    put_q8_pairs(o, row, i >> 6, a, b);
  } else if (o.mid) {  // the three parts of the skinny GEMMs' input, as put() makes them
    uint32_t h, m, l;
    split3(a, b, &h, &m, &l);
    *reinterpret_cast<uint32_t*>(o.hi + size_t(row) * I + i) = h;
    *reinterpret_cast<uint32_t*>(o.mid + size_t(row) * I + i) = m;
    *reinterpret_cast<uint32_t*>(o.lo + size_t(row) * I + i) = l;
  } else if (o.f32) {
    *reinterpret_cast<float2*>(o.f32 + size_t(row) * I + i) = make_float2(a, b);
  } else if (o.hi) {
    *reinterpret_cast<uint32_t*>(o.hi + size_t(row) * I + i) = pack_bf16(a, b);
  } else {
    o.y[size_t(row) * ld_out + i] = a;
    o.y[size_t(row) * ld_out + i + 1] = b;
  }
}

}  // namespace

void rmsnorm(const float* x, int ldx, const float* w, float offset, float eps, float* y, int ldy, int rows, int n,
             cudaStream_t st) {
  // The block size sets the order of the sum: rmsnorm_split uses the same, so both give the same bits.
  if (rows <= 16)
    k_rmsnorm_spread<<<dim3(rows, ceil_div(n, 512)), 512, 0, st>>>(x, ldx, w, offset, eps, Out{y, nullptr, nullptr},
                                                                   ldy, n);
  else k_rmsnorm<<<rows, 512, 0, st>>>(x, ldx, w, offset, eps, Out{y, nullptr, nullptr}, ldy, n);
}

void rmsnorm_split(const float* x, int ldx, const float* w, float offset, float eps, int rows, int n,
                   const Launch& L, bool q8) {
  if (q8 && rows <= 16)
    k_rmsnorm_q8<<<dim3(rows, ceil_div(n, 512)), 512, 0, L.st>>>(x, ldx, w, offset, eps, split_out(L, rows, n, true), n);
  else if (rows <= 16)
    k_rmsnorm_spread<<<dim3(rows, ceil_div(n, 512)), 512, 0, L.st>>>(x, ldx, w, offset, eps,
                                                                      split_out(L, rows, n, false), n, n);
  else k_rmsnorm<<<rows, 512, 0, L.st>>>(x, ldx, w, offset, eps, split_out(L, rows, n, q8), n, n);
}

void embed(const int* tokens, const bf16* E, int H, float* x, int rows, cudaStream_t st) {
  k_embed<<<rows, 256, 0, st>>>(tokens, E, H, x);
}

void embed(const int* tokens, const bf16* E, int H, bf16* x, int rows, cudaStream_t st) {
  k_embed16<<<rows, 256, 0, st>>>(tokens, E, H, x);
}

void rmsnorm(const bf16* x, int ldx, const float* w, float offset, float eps, float* y, int ldy, int rows, int n,
             cudaStream_t st) {
  k_rmsnorm<bf16><<<rows, 512, 0, st>>>(x, ldx, w, offset, eps, Out{y, nullptr, nullptr}, ldy, n);
}

void rmsnorm_split(const bf16* x, int ldx, const float* w, float offset, float eps, int rows, int n, const Launch& L,
                   bool q8) {
  if (rows <= 16) throw std::runtime_error("rmsnorm_split: bf16 rows are a prefill pass's (more than 16 rows)");
  k_rmsnorm<bf16><<<rows, 512, 0, L.st>>>(x, ldx, w, offset, eps, split_out(L, rows, n, q8), n, n);
}

void build_rows(const DevParams* P, const int* drafts, int fill, int* tokens, int rows, cudaStream_t st) {
  k_build_rows<<<1, 32, 0, st>>>(P, drafts, fill, tokens, rows);
}

void swiglu(const float* gu, int ld, int I, float* out, int ld_out, int rows, cudaStream_t st) {
  k_swiglu<<<dim3(rows, ceil_div(I, 512)), 256, 0, st>>>(gu, ld, I, Out{out, nullptr, nullptr}, ld_out);
}

void swiglu_split(const float* gu, int ld, int I, int rows, const Launch& L, bool q8) {
  k_swiglu<<<dim3(rows, ceil_div(I, 512)), 256, 0, L.st>>>(gu, ld, I, split_out(L, rows, I, q8), I);
}

namespace {

// K/V in tiers: a part of positions [from, min(cap, P->position + pos_add + span)) of a layer's K and
// V (kv_stage), from where they are (page-locked host memory past the device part) to their views (the staging).
// Every thread keeps four loads of each in flight: the bus's latency, not the GPU, sets the speed.
__global__ void k_kv_stage(const uint4* k, uint4* vk, const uint4* v, uint4* vv, size_t row16, int from, int cap,
                           const DevParams* P, int pos_add, int span, KvStagePart part, int per_row,
                           KvStageScales sc) {
  const int at = P->position + pos_add;
  const int first = part == kStagePass ? max(from, at) : from;
  const int end = min(cap, part == kStageBefore ? at : at + span);
  if (end <= first) return;
  if (sc.ks) {  // the positions' scales
    const size_t lo = size_t(first) * sc.nkv, n = size_t(end - first) * sc.nkv;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
      sc.vks[lo + i] = sc.ks[lo + i];
      sc.vvs[lo + i] = sc.vs[lo + i];
    }
  }
  const int r0 = first / per_row, r1 = (end + per_row - 1) / per_row;  // the rows holding them
  const size_t lo = size_t(r0) * row16, n = size_t(r1 - r0) * row16;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += 4 * stride) {
    uint4 a[4], b[4];
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (i + u * stride < n) a[u] = k[lo + i + u * stride], b[u] = v[lo + i + u * stride];
#pragma unroll
    for (int u = 0; u < 4; ++u)
      if (i + u * stride < n) vk[lo + i + u * stride] = a[u], vv[lo + i + u * stride] = b[u];
  }
}

}  // namespace

void kv_stage(const void* k, void* vk, const void* v, void* vv, size_t row, int from, int cap, const DevParams* P,
              int pos_add, int span, KvStagePart part, int blocks, cudaStream_t st, int per_row,
              const KvStageScales& scales) {
  k_kv_stage<<<blocks, 256, 0, st>>>(static_cast<const uint4*>(k), static_cast<uint4*>(vk),
                                     static_cast<const uint4*>(v), static_cast<uint4*>(vv), row / 16, from, cap, P,
                                     pos_add, span, part, per_row, scales);
}

namespace {

__global__ void k_kv_stage_ring(const uint4* k, uint4* vk, const uint4* v, uint4* vv, size_t row16, int sink, int ring,
                                int from, int cap, const DevParams* P, int pos_add, int span, KvStagePart part,
                                KvStageScales sc) {
  const int at = P->position + pos_add;
  const int first = part == kStagePass ? max(from, at) : from;
  const int end = min(cap, part == kStageBefore ? at : at + span);
  if (end <= first) return;
  const int tail = ring - sink;
  auto slot_of = [&](int pos) { return ring <= 0 || pos < sink ? pos : sink + (pos - sink) % tail; };
  if (sc.ks) {
    const int npos = end - first;
    const size_t total = size_t(npos) * sc.nkv;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
      const int pos = first + int(i / size_t(sc.nkv));
      const int h = int(i % size_t(sc.nkv));
      const int src = slot_of(pos);
      sc.vks[size_t(pos) * sc.nkv + h] = sc.ks[size_t(src) * sc.nkv + h];
      sc.vvs[size_t(pos) * sc.nkv + h] = sc.vs[size_t(src) * sc.nkv + h];
    }
  }
  const int npos = end - first;
  const size_t total = size_t(npos) * row16;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += size_t(gridDim.x) * blockDim.x) {
    const int pos = first + int(i / row16);
    const size_t col = i % row16;
    const int src = slot_of(pos);
    vk[size_t(pos) * row16 + col] = k[size_t(src) * row16 + col];
    vv[size_t(pos) * row16 + col] = v[size_t(src) * row16 + col];
  }
}

__global__ void k_kv_drain_pageable(const PageDrainLayer* layers, int n_layers, const DevParams* P,
                                    const flyx_round_result* ring_dev, int max_rows) {
  pdl_trigger();
  const int li = int(blockIdx.y);
  if (li >= n_layers) return;
  const unsigned long long seq = P->seq_no;
  const int rslot = int(seq % kRing);
  const flyx_round_result& e = ring_dev[rslot];
  const int n = e.n_tokens;
  const int r = int(blockIdx.x);
  if (n <= 0 || r >= n) return;
  const PageDrainLayer& L = layers[li];
  const int pos = e.position - n + r;
  if (pos < L.device_positions) return;
  const int cell = L.slot_of(pos);
  const size_t row16 = L.row_bytes / 16;
  const size_t dst_row = size_t(rslot) * size_t(max_rows) + size_t(r);
  const uint4* sk = reinterpret_cast<const uint4*>(L.k) + size_t(cell) * row16;
  const uint4* sv = reinterpret_cast<const uint4*>(L.v) + size_t(cell) * row16;
  uint4* dk = reinterpret_cast<uint4*>(L.drain_k) + dst_row * row16;
  uint4* dv = reinterpret_cast<uint4*>(L.drain_v) + dst_row * row16;
  for (size_t i = threadIdx.x; i < row16; i += blockDim.x) {
    dk[i] = sk[i];
    dv[i] = sv[i];
  }
  if (L.ks) {
    for (int h = int(threadIdx.x); h < L.nkv; h += int(blockDim.x)) {
      L.drain_ks[dst_row * size_t(L.nkv) + size_t(h)] = L.ks[size_t(cell) * size_t(L.nkv) + size_t(h)];
      L.drain_vs[dst_row * size_t(L.nkv) + size_t(h)] = L.vs[size_t(cell) * size_t(L.nkv) + size_t(h)];
    }
  }
}

}  // namespace

void kv_stage_ring(const void* k, void* vk, const void* v, void* vv, size_t row, int sink, int ring, int from, int cap,
                   const DevParams* P, int pos_add, int span, KvStagePart part, int blocks, cudaStream_t st,
                   const KvStageScales& scales) {
  k_kv_stage_ring<<<blocks, 256, 0, st>>>(static_cast<const uint4*>(k), static_cast<uint4*>(vk),
                                          static_cast<const uint4*>(v), static_cast<uint4*>(vv), row / 16, sink, ring,
                                          from, cap, P, pos_add, span, part, scales);
}

void kv_drain_pageable(const PageDrainLayer* layers, int n_layers, const DevParams* P, const flyx_round_result* ring_dev,
                       int max_rows, cudaStream_t st) {
  if (n_layers <= 0 || max_rows <= 0) return;
  k_kv_drain_pageable<<<dim3(max_rows, n_layers), 128, 0, st>>>(layers, n_layers, P, ring_dev, max_rows);
}

}  // namespace flyx::cuda
