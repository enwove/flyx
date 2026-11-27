// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Prefill attention on tensor cores (FlashAttention-2 style).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "flyx/gemm_tables.inc"

namespace flyx::cuda {
namespace {

// ---- prefill attention on tensor cores (FlashAttention-2 style) --------------------------
// One block: one query head, 64 query rows (4 warps x 16).  Keys and values
// stream through shared memory KT at a time (cp.async); S = Q K^T and
// O += P V on mma.m16n8k16 (binary16 in, as the caches: q and P rounded to
// it; fp32 accumulate); the softmax is online
// in registers; P goes from the S accumulators straight into the A fragments
// of the P V product.  Causal: row r (position base + r) sees keys up to it;
// key tiles past a warp's last row are skipped.  The output gate
// (sigmoid(gate)) is applied on the way out.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_s(uint32_t* r, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_t_s(uint32_t* r, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

template <int HD>
__host__ __device__ constexpr int flash_kt() { return HD >= 256 ? 32 : 64; }
template <int HD>
constexpr size_t flash_smem() { return size_t(64 + 2 * flash_kt<HD>()) * (HD + 8) * sizeof(kv16); }

// A window's softmax state (FlashWindow) of a warp's rows wrow0 + g and + 8 of head h, as the warp
// holds it: the maximum (the same in the four lanes of a row), each lane's sum, the FP8 value-scale exponent (p[5]),
// and the lane's output columns 8 n + 2 t (+1).  Exact both ways, so the windows give one pass's bits.
template <int NT, bool StoreVe = false>
__device__ __forceinline__ void flash_state(bool store, float* state, int nh, int h, int wrow0, int rows, int g, int t,
                                            float (&o)[NT][4], float (&m)[2], float (&l)[2], int (*ve)[2] = nullptr) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = wrow0 + g + 8 * half;
    if (row >= rows) continue;
    float* p = state + (size_t(row) * nh + h) * kFlashStateFloats(NT * 8);
    if (store) {
      if (t == 0) {
        p[0] = m[half];
        if constexpr (StoreVe) p[5] = __int_as_float((*ve)[half]);
      }
      p[1 + t] = l[half];
    } else {
      m[half] = p[0];
      l[half] = p[1 + t];
      if constexpr (StoreVe) (*ve)[half] = __float_as_int(p[5]);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float& x = p[8 + n * 8 + 2 * t + e];
        if (store) x = o[n][2 * half + e];
        else o[n][2 * half + e] = x;
      }
  }
}

template <int HD, bool FP8>
__global__ void __launch_bounds__(128) k_flash_prefill(const float* __restrict__ q, KvRef kv, int nh, int nkv, int rows,
                                                       float scale, const DevParams* P, int pos_add,
                                                       const float* __restrict__ gate, int ld_gate, Out out,
                                                       int ld_out, FlashWindow win) {
  pdl_trigger();
  constexpr int KT = flash_kt<HD>(), LD = HD + 8, NT = HD / 8, ST = KT / 8;
  extern __shared__ __align__(16) unsigned char sm_raw[];
  kv16(*Qs)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw);
  kv16(*Ks)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw + size_t(64) * LD * sizeof(kv16));
  kv16(*Vs)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw + size_t(64 + KT) * LD * sizeof(kv16));
  const int h = blockIdx.x, r0 = blockIdx.y * 64, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kvh = h / (nh / nkv);
  const int base = P->position + pos_add;
  for (int i = tid; i < 64 * HD / 2; i += 128) {
    const int r = i / (HD / 2), c = 2 * (i % (HD / 2));
    float2 v = make_float2(0.f, 0.f);
    if (r0 + r < rows) v = *reinterpret_cast<const float2*>(q + (size_t(r0 + r) * nh + h) * HD + c);
    *reinterpret_cast<__half2*>(&Qs[r][c]) = __floats2half2_rn(v.x, v.y);
  }
  const int wrow0 = r0 + warp * 16;           // this warp's first query row
  const int last_row = min(rows, r0 + 64) - 1;
  const int kend = base + last_row + 1;       // keys this block needs: [0, kend)
  const int kwin = min(kend, win.k_hi);       // of them, the window's: [win.k_lo, kwin)
  // a sliding window: the block's first row's first key on (later rows start later)
  const int kfirst = win.sliding ? max(win.k_lo, base + r0 + 1 - win.sliding) : win.k_lo;
  const float sl2 = scale * 1.4426950408889634f;
  float o[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  if (!win.first) flash_state<NT>(false, win.state, nh, h, wrow0, rows, g, t, o, m, l);
  for (int k0 = kfirst; k0 < kwin; k0 += KT) {
    __syncthreads();  // the previous tiles are no longer read (and Q is written, first time)
    if constexpr (FP8) {  // e4m3 * row scale -> binary16 tiles
      for (int i = tid; i < KT * HD / 8; i += 128) {
        const int r = i / (HD / 8), c = 8 * (i % (HD / 8)), pos = k0 + r;
        uint4 kb = make_uint4(0, 0, 0, 0), vb = kb;
        if (pos < kwin) {
          const size_t at = (size_t(pos) * nkv + kvh) * HD + c;
          const uint2 kq = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(kv.k) + at);
          const uint2 vq = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(kv.v) + at);
          const float ksc = kv.kscales(pos)[size_t(pos) * nkv + kvh], vsc = kv.vscales(pos)[size_t(pos) * nkv + kvh];
          float kf[8], vf[8];
          e4m3x4_to_float(kq.x, kf);
          e4m3x4_to_float(kq.y, kf + 4);
          e4m3x4_to_float(vq.x, vf);
          e4m3x4_to_float(vq.y, vf + 4);
          kb = make_uint4(pack_f16(kf[0] * ksc, kf[1] * ksc), pack_f16(kf[2] * ksc, kf[3] * ksc),
                          pack_f16(kf[4] * ksc, kf[5] * ksc), pack_f16(kf[6] * ksc, kf[7] * ksc));
          vb = make_uint4(pack_f16(vf[0] * vsc, vf[1] * vsc), pack_f16(vf[2] * vsc, vf[3] * vsc),
                          pack_f16(vf[4] * vsc, vf[5] * vsc), pack_f16(vf[6] * vsc, vf[7] * vsc));
        }
        *reinterpret_cast<uint4*>(&Ks[r][c]) = kb;
        *reinterpret_cast<uint4*>(&Vs[r][c]) = vb;
      }
    } else {
      const kv16* Kc = static_cast<const kv16*>(kv.k);
      const kv16* Vc = static_cast<const kv16*>(kv.v);
      for (int i = tid; i < KT * HD / 8; i += 128) {
        const int r = i / (HD / 8), c = 8 * (i % (HD / 8)), pos = k0 + r;
        const bool ok = pos < kwin;
        const int slot = ok ? (win.ring ? pos % win.ring : pos) : kfirst;
        const size_t at = (size_t(slot) * nkv + kvh) * HD + c;
        cp_async16(&Ks[r][c], Kc + at, ok);
        cp_async16(&Vs[r][c], Vc + at, ok);
      }
      cp_async_commit();
      cp_async_wait<0>();
    }
    __syncthreads();
    if (wrow0 >= rows || base + wrow0 + 15 < k0) continue;  // all of this warp's rows precede the tile
    if (win.sliding && k0 + KT <= base + wrow0 + 1 - win.sliding) continue;  // or are all past it
    float s[ST][4];
#pragma unroll
    for (int j = 0; j < ST; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int kk = 0; kk < HD; kk += 16) {
      uint32_t a[4];
      ldmatrix_x4(a, &Qs[warp * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
      for (int j = 0; j < ST; ++j) {
        uint32_t b[2];
        ldmatrix_x2(b, &Ks[j * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
        mma_f16(s[j], a, b);
      }
    }
    // scale, mask, online softmax (rows g and g+8 of the warp's 16)
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = wrow0 + g + 8 * half, qpos = base + row;
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < ST; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = k0 + j * 8 + 2 * t + e;
          float& v = s[j][2 * half + e];
          v = (row < rows && key <= qpos && (!win.sliding || key > qpos - win.sliding)) ? v * sl2 : -INFINITY;
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mnew = fmaxf(m[half], mx);
      const float alpha = mnew == -INFINITY ? 1.f : exp2f(m[half] - mnew);
      m[half] = mnew;
      l[half] *= alpha;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        o[n][2 * half] *= alpha;
        o[n][2 * half + 1] *= alpha;
      }
#pragma unroll
      for (int j = 0; j < ST; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& v = s[j][2 * half + e];
          v = mnew == -INFINITY ? 0.f : exp2f(v - mnew);
          l[half] += v;
        }
    }
#pragma unroll
    for (int kk = 0; kk < KT; kk += 16) {
      const int j = kk / 8;
      const uint32_t a[4] = {pack_f16(s[j][0], s[j][1]), pack_f16(s[j][2], s[j][3]), pack_f16(s[j + 1][0], s[j + 1][1]),
                             pack_f16(s[j + 1][2], s[j + 1][3])};
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        uint32_t b[2];
        ldmatrix_x2_trans(b, &Vs[kk + (lane & 15)][n * 8]);
        mma_f16(o[n], a, b);
      }
    }
  }
  if (!win.last) {  // the next window goes on from here
    flash_state<NT>(true, win.state, nh, h, wrow0, rows, g, t, o, m, l);
    return;
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float sum = l[half];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const int row = wrow0 + g + 8 * half;
    if (row >= rows) continue;
    // a sink joins the denominator, in the scores' log2 units (exp2 of its excess over the row's maximum)
    if (win.sinks) sum += exp2f(fminf(win.sinks[h] * 1.4426950408889634f - m[half], 126.f));
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int d = n * 8 + 2 * t + e;
        float v = o[n][2 * half + e] * inv;
        if (gate) v *= 1.f / (1.f + __expf(-gate[size_t(row) * ld_gate + h * HD + d]));
        put(out, size_t(row) * ld_out + h * HD + d, size_t(row) * nh * HD + h * HD + d, v);
      }
  }
}

// The same attention with the query heads of a KV head together (GQA): one
// block takes 16 query rows of HB of the G = nh / nkv heads that share a KV
// head (HB warps, warp w the head HB z + w of grid row z), so each K/V tile is
// read once for all of them (G / HB times for the KV head);
// the queries stay in registers (A fragments); K/V tiles of 16 keys are
// double-buffered (the next one loads while this one is used).  Two blocks
// per SM (255 registers x 128 threads, 34 KB of shared memory at head dim
// 256): one block's softmax overlaps the other's MMAs, which the 8 warps of
// one block, held together by a barrier per tile, could not do (llama.cpp's
// tensor-core kernel has the same shape: 16 rows x 4 heads, 4 warps, two
// blocks per SM).  When no row of a warp gets a larger maximum, o is not
// rescaled (alpha = 1 exactly: the same bits).  Key tiles sit on absolute key
// positions as in k_flash_prefill: a row's result does not depend on where
// its pass starts, nor on HB (a warp's head is computed alone).  Head dims 128
// and 256; G = 4 (HB 4), 6 and 8. FP8 K/V: a tile's e4m3 bytes are
// double-buffered (XOR-swizzled across 16-byte chunks so Vs + Rb fits in 24 KiB
// at HD=256: 4 blocks/SM for HB=2 on 99 KiB GPUs), while K/V row scales stay
// in warp registers.  Each warp unpacks K's e4m3 bytes directly into binary16 B
// fragments (exact) and multiplies each score by its key's scale.  The block
// unpacks V's e4m3 bytes directly into one binary16 tile (exact), and each warp
// folds the key's value scale times 2^(kGqaPTop - ve) into P (as attention_rows
// does), tracking the running maximum value-scale exponent ve in FlashWindow.
constexpr int kGqaRows = 16, kGqaKeys = 16;
constexpr int kGqaPTop = 14;  // FP8: P * vsc scaled by 2^(14 - ve) so every P stays inside normal binary16
constexpr int kGqaVeUnset = -32768;
// Pack both rows' FP8 value-scale exponents into one uint32_t (int16_t each) to keep register pressure minimal.
__device__ __forceinline__ int ve_get(uint32_t pack, int half) {
  return int(int16_t((pack >> (16 * half)) & 0xFFFFu));
}
__device__ __forceinline__ uint32_t ve_set(uint32_t pack, int half, int v) {
  const uint32_t mask = half ? 0x0000FFFFu : 0xFFFF0000u;
  return (pack & mask) | (uint32_t(uint16_t(int16_t(v))) << (16 * half));
}
__device__ __forceinline__ float pow2_f32(int e) { return __int_as_float((e + 127) << 23); }
// Exact equivalent of: int eb; frexpf(x, &eb); return min(max(eb, -110), 127); for positive finite x.
__device__ __forceinline__ int fp8_vexp(float x) {
  const int eb = int((__float_as_uint(x) >> 23) & 0xFFu) - 126;
  return min(max(eb, -110), 127);
}
// Exact equivalent of ldexpf(1.f, d) for d <= 0 (normal, subnormal, and underflow to 0).
__device__ __forceinline__ float pow2_neg_f32(int d) {
  return d >= -126 ? pow2_f32(d) : (d >= -149 ? __int_as_float(1 << (d + 149)) : 0.f);
}
// 16-byte chunk index swizzled by row & 7: bijection on 8 chunks (128 B = 32 banks), eliminating bank conflicts
// without row padding. At HD=256 FP8: Vs (8 KiB) + Rb (16 KiB) = 24,576 B (24 KiB exactly; 4 blocks/SM on 99 KiB).
__device__ __forceinline__ int gqa_swiz16(int row, int chunk16) { return chunk16 ^ (row & 7); }
template <int HD, bool FP8>
constexpr size_t gqa_smem() {
  constexpr size_t tile = size_t(kGqaKeys) * (FP8 ? HD : HD + 8) * sizeof(kv16);
  return FP8 ? tile + size_t(2) * 2 * kGqaKeys * HD : 4 * tile;
}
template <int HD, int HB, bool FP8, bool History = false>
__global__ void __launch_bounds__(32 * HB) k_flash_gqa(const float* __restrict__ q, KvRef kv, int nh, int nkv,
                                                       int rows, float scale, const DevParams* P, int pos_add,
                                                       const float* __restrict__ gate, int ld_gate, Out out,
                                                       int ld_out, FlashWindow win) {
  static_assert(HD % 128 == 0, "gqa_swiz16 spans 8 16-byte chunks");
  pdl_trigger();
  constexpr int KT = kGqaKeys, LD = FP8 ? HD : HD + 8, NT = HD / 8, ST = KT / 8, KQ = HD / 16, QR = kGqaRows;
  constexpr int NBK = FP8 ? 0 : 2, NBV = FP8 ? 1 : 2;  // binary16 key and value tile buffers
  constexpr int RS = HD;
  constexpr bool release_after_qk = FP8;
  extern __shared__ __align__(16) unsigned char sm_raw[];
  kv16(*Ks)[KT][LD] = reinterpret_cast<kv16(*)[KT][LD]>(sm_raw);                                  // [NBK][KT][LD]
  kv16(*Vs)[KT][LD] = reinterpret_cast<kv16(*)[KT][LD]>(sm_raw + size_t(NBK) * KT * LD * sizeof(kv16));
  // FP8: [buffer][K, V][key][dim] bytes; K/V row scales stay in warp registers (no shared memory).
  unsigned char* staged = sm_raw + size_t(NBK + NBV) * KT * LD * sizeof(kv16);
  auto Rb = reinterpret_cast<uint8_t (*)[2][KT][RS]>(staged);
  const kv16* Kc = static_cast<const kv16*>(kv.k);
  const kv16* Vc = static_cast<const kv16*>(kv.v);
  // the last rows see the most keys: they start first, the short blocks fill the tail
  const int kvh = blockIdx.x, r0 = (gridDim.y - 1 - blockIdx.y) * QR, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int h = kvh * (nh / nkv) + int(blockIdx.z) * HB + warp;  // this warp's query head
  const int wrow0 = r0;                    // and its first row
  const int base = P->position + pos_add;
  const int last_row = min(rows, r0 + QR) - 1;
  const int kend = base + last_row + 1;          // keys the block needs: [0, kend)
  const int kwin = History ? win.k_hi : min(kend, win.k_hi);  // this window's keys: [win.k_lo, kwin)
  const int it0 = win.k_lo / KT, ntiles = (kwin + KT - 1) / KT;  // its tiles: [it0, ntiles)
  const bool full_rows = History || (r0 + QR <= rows);
  auto load = [&](int buf, int k0) {  // past kwin: zeros
    if constexpr (FP8) {
      for (int i = tid; i < 2 * KT * (HD / 16); i += 32 * HB) {
        const int v = i / (KT * (HD / 16)), r = (i / (HD / 16)) % KT, ch = i % (HD / 16), pos = k0 + r;
        const bool ok = History || pos < kwin;
        const uint8_t* src =
            static_cast<const uint8_t*>(v ? kv.v : kv.k) + (size_t(ok ? pos : win.k_lo) * nkv + kvh) * HD;
        const int dst_c = 16 * (v ? ch : gqa_swiz16(r, ch));
        if constexpr (History) cp_async16(&Rb[buf][v][r][dst_c], src + 16 * ch);
        else cp_async16(&Rb[buf][v][r][dst_c], src + 16 * ch, ok);
      }
    } else {
      for (int i = tid; i < KT * HD / 8; i += 32 * HB) {
        const int r = i / (HD / 8), c = 8 * (i % (HD / 8)), pos = k0 + r;
        const bool ok = pos < kwin;
        const size_t at = (size_t(ok ? pos : win.k_lo) * nkv + kvh) * HD + c;
        cp_async16(&Ks[buf][r][c], Kc + at, ok);
        cp_async16(&Vs[buf][r][c], Vc + at, ok);
      }
    }
    cp_async_commit();
  };
  // FP8: unpack raw e4m3 values directly into the binary16 value tile (exact; value scales fold into P).
  auto convert = [&](int buf) {
    if constexpr (FP8) {
      for (int i = tid; i < KT * (HD / 8); i += 32 * HB) {
        const int r = i / (HD / 8), ch = i % (HD / 8), c = 8 * ch;
        const uint2 w = *reinterpret_cast<const uint2*>(&Rb[buf][1][r][c]);
        uint32_t hw[4];
        e4m3x4_to_f16bits(w.x, hw);
        e4m3x4_to_f16bits(w.y, hw + 2);
        *reinterpret_cast<uint4*>(&Vs[0][r][8 * gqa_swiz16(r, ch)]) = make_uint4(hw[0], hw[1], hw[2], hw[3]);
      }
    }
  };
  if (it0 < ntiles) load(0, it0 * KT);
  // the warp's queries as A fragments: rows g, g + 8; values kk + 2t (+1), kk + 2t + 8 (+1); FP8: kk + 4t (+1)
  // and kk + 4t + 2 (+3), the dims of the keys' bytes a lane holds
  uint32_t qa[KQ][4];
  {
    const int ra = wrow0 + g, rb = ra + 8;
    const float* qa_row = q + (size_t(ra) * nh + h) * HD;
    const float* qb_row = q + (size_t(rb) * nh + h) * HD;
#pragma unroll
    for (int kq = 0; kq < KQ; ++kq) {
      const int c = 16 * kq + (FP8 ? 4 : 2) * t, c2 = FP8 ? c + 2 : c + 8;
      const float2 z = make_float2(0.f, 0.f);
      const float2 a0 = ra < rows ? *reinterpret_cast<const float2*>(qa_row + c) : z;
      const float2 a1 = rb < rows ? *reinterpret_cast<const float2*>(qb_row + c) : z;
      const float2 a2 = ra < rows ? *reinterpret_cast<const float2*>(qa_row + c2) : z;
      const float2 a3 = rb < rows ? *reinterpret_cast<const float2*>(qb_row + c2) : z;
      qa[kq][0] = pack_f16(a0.x, a0.y);
      qa[kq][1] = pack_f16(a1.x, a1.y);
      qa[kq][2] = pack_f16(a2.x, a2.y);
      qa[kq][3] = pack_f16(a3.x, a3.y);
    }
  }
  const float sl2 = scale * 1.4426950408889634f;
  float o[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  uint32_t ve_pack = 0x80008000u;  // both halves kGqaVeUnset
  if (!win.first) {
    int ve_tmp[2] = {kGqaVeUnset, kGqaVeUnset};
    flash_state<NT, FP8>(false, win.state, nh, h, wrow0, rows, g, t, o, m, l, &ve_tmp);
    if constexpr (FP8) {
      ve_pack = ve_set(ve_set(0u, 0, ve_tmp[0]), 1, ve_tmp[1]);
    }
  }
  for (int it = it0; it < ntiles; ++it) {
    const int buf = (it - it0) & 1, k0 = it * KT;
    if (it + 1 < ntiles) {
      load(buf ^ 1, k0 + KT);  // the previous iteration released this buffer after its last shared read
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const int tb = FP8 ? 0 : buf;  // the binary16 tiles in use
    if constexpr (FP8) {
      convert(buf);
      if constexpr (!release_after_qk) __syncthreads();
    }
    if (History || (wrow0 < rows && base + wrow0 + 15 >= k0)) {
      // Load K scales (lanes 0..15) and V scales (lanes 16..31) ahead of the QK MMAs so latency overlaps.
      float sc_lane = 0.f;
      if constexpr (FP8) {
        const int is_v = lane >> 4, r = lane & 15, pos = k0 + r;
        if (History || pos < kwin) {
          const float* sp = is_v ? (History ? kv.vs : kv.vscales(pos)) : (History ? kv.ks : kv.kscales(pos));
          sc_lane = sp[size_t(pos) * nkv + kvh];
        }
      }
      float s[ST][4];
#pragma unroll
      for (int j = 0; j < ST; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
      if constexpr (FP8) {
        static_assert(ST == 2, "two key groups a tile");
        const uint32_t rb_row = static_cast<uint32_t>(__cvta_generic_to_shared(&Rb[buf][0][lane & 15][0]));
        const uint32_t sw = 16u * ((lane >> 4) ^ (lane & 7));
#pragma unroll
        for (int kq = 0; kq < KQ; kq += 2) {  // the bytes of 16 keys x 32 dims in one load: key groups 0 and 1 of
          uint32_t w[4];                       // dims 16 kq.., then of 16 (kq + 1)..
          const uint32_t off = 16u * kq;
          ldmatrix_x4_s(w, (rb_row + ((off & 127u) ^ sw)) + (off & ~127u));
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            uint32_t b[2];
            e4m3x4_to_f16bits(w[x], b);
            mma_f16(s[x & 1], qa[kq + (x >> 1)], b);
          }
        }
#pragma unroll
        for (int j = 0; j < ST; ++j) {  // each key's scale
          const float k0s = __shfl_sync(0xffffffffu, sc_lane, j * 8 + 2 * t);
          const float k1s = __shfl_sync(0xffffffffu, sc_lane, j * 8 + 2 * t + 1);
          s[j][0] *= k0s;
          s[j][1] *= k1s;
          s[j][2] *= k0s;
          s[j][3] *= k1s;
        }
      } else {
#pragma unroll
        for (int kq = 0; kq < KQ; ++kq)
#pragma unroll
          for (int j = 0; j < ST; j += 2) {  // key groups j and j + 1 in one load
            uint32_t b[4];
            ldmatrix_x4(b, &Ks[tb][(j + (lane >> 4)) * 8 + (lane & 7)][16 * kq + ((lane >> 3) & 1) * 8]);
            mma_f16(s[j], qa[kq], b);
            mma_f16(s[j + 1], qa[kq], b + 2);
          }
      }
      const bool hist_tile = History || (full_rows && k0 + KT <= min(base + r0, kwin));
      float tile_vbig = 0.f;
      if constexpr (FP8) {
        if (hist_tile) {
          // On a full history tile every row sees all 16 keys, so one warp reduction over lanes 16..31 gives |vsc|_max.
          float vb = (lane >= 16) ? fabsf(sc_lane) : 0.f;
          vb = fmaxf(vb, __shfl_xor_sync(0xffffffffu, vb, 1));
          vb = fmaxf(vb, __shfl_xor_sync(0xffffffffu, vb, 2));
          vb = fmaxf(vb, __shfl_xor_sync(0xffffffffu, vb, 4));
          vb = fmaxf(vb, __shfl_xor_sync(0xffffffffu, vb, 8));
          tile_vbig = __shfl_sync(0xffffffffu, vb, 16);
        }
      }
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = wrow0 + g + 8 * half, qpos = base + row;
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < ST; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int key = k0 + j * 8 + 2 * t + e;
            float& v = s[j][2 * half + e];
            v = (hist_tile || (row < rows && key <= qpos)) ? v * sl2 : -INFINITY;
            mx = fmaxf(mx, v);
          }
        // On long contexts the running maximum usually exceeds this tile in every lane.
        // The strict comparison keeps signed-zero ties, and non-finite maxima take the full update.
        const bool update_max = !__all_sync(0xffffffffu, mx < m[half] && isfinite(m[half]));
        float mnew = m[half], alpha = 1.f;
        if (update_max) {
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
          mnew = fmaxf(m[half], mx);
          alpha = mnew == -INFINITY ? 1.f : exp2f(m[half] - mnew);
        }
        m[half] = mnew;
        l[half] *= alpha;  // Keep the multiplication beside the following sum to preserve its rounding.
        float ocorr = 1.f;
        if constexpr (FP8) {
          int ve = ve_get(ve_pack, half);
          const float vlim = ve != kGqaVeUnset ? pow2_f32(ve) : 0.f;
          float vbig = tile_vbig;
          if (!hist_tile) {
#pragma unroll
            for (int j = 0; j < ST; ++j)
#pragma unroll
              for (int e = 0; e < 2; ++e) {
                const int key = k0 + j * 8 + 2 * t + e;
                const float vs = __shfl_sync(0xffffffffu, sc_lane, 16 + j * 8 + 2 * t + e);
                if (row < rows && key <= qpos) vbig = fmaxf(vbig, fabsf(vs));
              }
          }
          if (__any_sync(0xffffffffu, vbig > 0.f && vbig >= vlim)) {
            if (!hist_tile) {
              vbig = fmaxf(vbig, __shfl_xor_sync(0xffffffffu, vbig, 1));
              vbig = fmaxf(vbig, __shfl_xor_sync(0xffffffffu, vbig, 2));
            }
            const int eb = vbig > 0.f ? fp8_vexp(vbig) : kGqaVeUnset;
            if (eb > ve) {
              if (ve != kGqaVeUnset) ocorr = pow2_neg_f32(ve - eb);
              ve = eb;
              ve_pack = ve_set(ve_pack, half, ve);
            }
          }
        }
        const float oscale = FP8 ? alpha * ocorr : alpha;
        if (__any_sync(0xffffffffu, oscale != 1.f)) {
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            o[n][2 * half] *= oscale;
            o[n][2 * half + 1] *= oscale;
          }
        }
#pragma unroll
        for (int j = 0; j < ST; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float& v = s[j][2 * half + e];
            v = mnew == -INFINITY ? 0.f : exp2f(v - mnew);
            l[half] += v;
          }
      }
      if constexpr (FP8) {
        const int ve0 = ve_get(ve_pack, 0), ve1 = ve_get(ve_pack, 1);
        const float ps0 = ve0 != kGqaVeUnset ? pow2_f32(kGqaPTop - ve0) : 1.f;
        const float ps1 = ve1 != kGqaVeUnset ? pow2_f32(kGqaPTop - ve1) : 1.f;
#pragma unroll
        for (int j = 0; j < ST; ++j)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const float vs = __shfl_sync(0xffffffffu, sc_lane, 16 + j * 8 + 2 * t + e);
            // Skip zero probabilities so a masked future key with a large vs (excluded from ve) never yields 0 * +INF = NaN.
            s[j][e] = s[j][e] == 0.f ? 0.f : s[j][e] * (vs * ps0);
            s[j][2 + e] = s[j][2 + e] == 0.f ? 0.f : s[j][2 + e] * (vs * ps1);
          }
      }
      // All head warps share the same query rows. This barrier publishes converted V and releases
      // the raw K tile once every warp has finished QK. The next wait's barrier protects V reuse.
      if constexpr (release_after_qk) __syncthreads();
#pragma unroll
      for (int kk = 0; kk < KT; kk += 16) {
        const int j = kk / 8;
        const uint32_t a[4] = {pack_f16(s[j][0], s[j][1]), pack_f16(s[j][2], s[j][3]),
                               pack_f16(s[j + 1][0], s[j + 1][1]), pack_f16(s[j + 1][2], s[j + 1][3])};
        if constexpr (FP8) {
          const uint32_t vs_row = static_cast<uint32_t>(__cvta_generic_to_shared(&Vs[tb][kk + (lane & 15)][0]));
          const uint32_t sw = 16u * ((lane >> 4) ^ (lane & 7));
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const uint32_t vp = vs_row + ((32u * p) ^ sw);
#pragma unroll
            for (int b8 = 0; b8 < NT; b8 += 8) {  // value columns n and n + 1 in one load (independent columns)
              const int n = b8 + 2 * p;
              uint32_t b[4];
              ldmatrix_x4_t_s(b, vp + 16u * b8);
              mma_f16(o[n], a, b);
              mma_f16(o[n + 1], a, b + 2);
            }
          }
        } else {
#pragma unroll
          for (int n = 0; n < NT; n += 2) {  // value columns n and n + 1 in one load
            uint32_t b[4];
            ldmatrix_x4_t(b, &Vs[tb][kk + (lane & 15)][(n + (lane >> 4)) * 8]);
            mma_f16(o[n], a, b);
            mma_f16(o[n + 1], a, b + 2);
          }
        }
      }
    } else if constexpr (release_after_qk) {
      __syncthreads();
    }
    if constexpr (!release_after_qk) __syncthreads();  // buf is reloaded two iterations on
  }
  if (!win.last) {  // the next window goes on from here
    int ve_tmp[2] = {ve_get(ve_pack, 0), ve_get(ve_pack, 1)};
    flash_state<NT, FP8>(true, win.state, nh, h, wrow0, rows, g, t, o, m, l, &ve_tmp);
    return;
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float sum = l[half];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const int row = wrow0 + g + 8 * half;
    if (row >= rows) continue;
    if (win.sinks) sum += exp2f(fminf(win.sinks[h] * 1.4426950408889634f - m[half], 126.f));  // as k_flash_prefill
    const int ve = ve_get(ve_pack, half);
    const float vback = FP8 && ve != kGqaVeUnset ? pow2_f32(ve - kGqaPTop) : 1.f;
    const float inv = (sum > 0.f ? 1.f / sum : 0.f) * vback;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int d = n * 8 + 2 * t + e;
        float v = o[n][2 * half + e] * inv;
        if (gate) v *= 1.f / (1.f + __expf(-gate[size_t(row) * ld_gate + h * HD + d]));
        put(out, size_t(row) * ld_out + h * HD + d, size_t(row) * nh * HD + h * HD + d, v);
      }
  }
}

}  // namespace

// The grouped kernel with HB heads a block, G / HB blocks a KV head and row tile.
template <int HD, int HB, bool FP8, bool History = false>
void launch_flash_gqa(const float* q, KvRef kv, int nh, int nkv, int rows, float scale, const DevParams* P,
                      int pos_add, const float* gate, int ld_gate, Out out, int ld_out, cudaStream_t st,
                      const FlashWindow& win) {
  constexpr size_t smem = gqa_smem<HD, FP8>();
  static const bool attr = set_dynamic_smem(k_flash_gqa<HD, HB, FP8, History>, smem);
  (void)attr;
  k_flash_gqa<HD, HB, FP8, History><<<dim3(nkv, ceil_div(rows, kGqaRows), nh / nkv / HB), 32 * HB, smem, st>>>(
      q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, win);
}

template <int HD, bool FP8>
void launch_flash(const float* q, KvRef kv, int nh, int nkv, int rows, float scale, const DevParams* P, int pos_add,
                  const float* gate, int ld_gate, Out out, int ld_out, cudaStream_t st, const FlashWindow& win) {
  static const bool attr = set_dynamic_smem(k_flash_prefill<HD, FP8>, flash_smem<HD>());
  (void)attr;
  if constexpr (HD == 256 || HD == 128) {
    if (win.sliding) {  // the grouped kernels take no sliding window
      k_flash_prefill<HD, FP8><<<dim3(nh, ceil_div(rows, 64)), 128, flash_smem<HD>(), st>>>(
          q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, win);
      return;
    }
    // The query heads of a KV head together: Qwen3.5's 16 over 4 and Qwen3's 32 over 8 (G 4), Qwen3's 64 over 8
    // (G 8: two blocks of 4), the 27B's 24 over 4 (G 6: two blocks of 3, two blocks an SM; 21K tokens of its shape
    // took 49.3 ms a layer against 51.9 with one block of 6 and 54.8 with a block a head: cuda_bench flash 24 4).
    // FP8 caches too (k_flash_prefill read and converted a tile once for every query head).
    if (nh == 4 * nkv || nh == 8 * nkv) {
      launch_flash_gqa<HD, 4, FP8>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win);
      return;
    }
    if (nh == 6 * nkv) {
      if constexpr (HD == 256 && FP8) {
        // 24 KiB SMEM lets HB=2 run 4 blocks/SM (8 warps/SM vs HB=3's 6 warps/SM) once the grid fills the GPU;
        // smaller chunks (<512 rows) keep HB=3 for fewer redundant K/V tile reads per KV head.
        const int hb = rows >= 512 ? 2 : 3;
        if (win.history && rows % kGqaRows == 0 && win.k_lo % kGqaKeys == 0 && win.k_hi % kGqaKeys == 0 &&
            (win.k_lo >= kv.scale_split || win.k_hi <= kv.scale_split)) {
          kv.ks = kv.kscales(win.k_lo);
          kv.vs = kv.vscales(win.k_lo);
          if (hb == 2)
            launch_flash_gqa<256, 2, true, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out,
                                                  st, win);
          else
            launch_flash_gqa<256, 3, true, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out,
                                                  st, win);
          return;
        }
        if (hb == 2)
          launch_flash_gqa<256, 2, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win);
        else
          launch_flash_gqa<256, 3, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win);
        return;
      }
      launch_flash_gqa<HD, 3, FP8>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win);
      return;
    }
  }
  k_flash_prefill<HD, FP8><<<dim3(nh, ceil_div(rows, 64)), 128, flash_smem<HD>(), st>>>(
      q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, win);
}

bool attention_flash_gqa_cfg(int hb, const float* q, KvRef kv, int nh, int nkv, int hd, int rows, float scale,
                             const DevParams* P, const float* gate, int ld_gate, float* out_f, int ld_out,
                             const Launch& L, int pos_add, const FlashWindow& win) {
  if (!kv.fp8() || nh != 24 || nkv != 4 || hd != 256) return false;
  const Out out = out_f ? Out{out_f, nullptr, nullptr} : split_out(L, rows, nh * hd);
  // A complete historical window uses one set of scales and has no partial or causal tile.
  if ((hb == 2 || hb == 3 || hb == 6) && win.history && rows % kGqaRows == 0 && win.k_lo % kGqaKeys == 0 &&
      win.k_hi % kGqaKeys == 0 && (win.k_lo >= kv.scale_split || win.k_hi <= kv.scale_split)) {
    kv.ks = kv.kscales(win.k_lo);
    kv.vs = kv.vscales(win.k_lo);
    if (hb == 2)
      launch_flash_gqa<256, 2, true, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out,
                                            L.st, win);
    else if (hb == 3)
      launch_flash_gqa<256, 3, true, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out,
                                            L.st, win);
    else
      launch_flash_gqa<256, 6, true, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out,
                                            L.st, win);
    return true;
  }
  switch (hb) {
    case 2:
      launch_flash_gqa<256, 2, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, L.st, win);
      break;
    case 3:
      launch_flash_gqa<256, 3, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, L.st, win);
      break;
    case 6:
      launch_flash_gqa<256, 6, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, L.st, win);
      break;
    default: return false;
  }
  return true;
}

bool attention_flash(const float* q, KvRef kv, int nh, int nkv, int hd, int rows, float scale, const DevParams* P,
                     const float* gate, int ld_gate, float* out_f, int ld_out, const Launch& L, int pos_add,
                     const FlashWindow& win) {
  if (kv.fp8() && win.sliding) return false;  // a sliding window's caches are binary16
  if (kv.fp8()) {
    const int sku = opts(L).tuning, keys = win.k_hi - win.k_lo;
    if (win.state) {
      for (const auto& r : tables::kFlashGqa)
        if (r.sku == sku && r.nh == nh && r.nkv == nkv && r.hd == hd && r.rows == rows && r.keys == keys)
          return attention_flash_gqa_cfg(r.hb, q, kv, nh, nkv, hd, rows, scale, P, gate, ld_gate, out_f, ld_out, L,
                                         pos_add, win);
    }
    const int want_rows = rows >= 512 ? 1024 : 256;
    for (const auto& r : tables::kFlashGqa)
      if (r.sku == sku && r.nh == nh && r.nkv == nkv && r.hd == hd && r.rows == want_rows)
        return attention_flash_gqa_cfg(r.hb, q, kv, nh, nkv, hd, rows, scale, P, gate, ld_gate, out_f, ld_out, L,
                                       pos_add, win);
  }
  const cudaStream_t st = L.st;
  const Out out = out_f ? Out{out_f, nullptr, nullptr} : split_out(L, rows, nh * hd);  // null: the next GEMM's input
#define FLYX_FLASH(HD_)                                                                                       \
  case HD_:                                                                                                   \
    if (kv.fp8())                                                                                             \
      launch_flash<HD_, true>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win);  \
    else                                                                                                      \
      launch_flash<HD_, false>(q, kv, nh, nkv, rows, scale, P, pos_add, gate, ld_gate, out, ld_out, st, win); \
    return true;
  switch (hd) {
    FLYX_FLASH(32)
    FLYX_FLASH(64)
    FLYX_FLASH(128)
    FLYX_FLASH(256)
    default: return false;
  }
#undef FLYX_FLASH
}

bool attention_flash(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int rows, float scale,
                     const DevParams* P, const float* gate, int ld_gate, float* out, int ld_out, const Launch& L,
                     int pos_add) {
  return attention_flash(q, KvRef{const_cast<kv16*>(Kc), const_cast<kv16*>(Vc), nullptr, nullptr}, nh, nkv, hd, rows,
                         scale, P, gate, ld_gate, out, ld_out, L, pos_add);
}

}  // namespace flyx::cuda
