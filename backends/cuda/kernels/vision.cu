// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Vision encoder and multimodal prefill kernels: LayerNorm, bias add, LayerScale residual add,
// vision activations, float32 tiled GEMM (arbitrary K for patch conv stems), 2D bilinear position
// embeddings, 2D axial Vision RoPE, bidirectional block-diagonal vision attention (arbitrary hd <= 256),
// 2D spatial average pooling, and multimodal embedding scatter.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kLnThreads = 512;
constexpr int kLnCache = 10;  // 512 * 10 = 5120 floats cached in registers per block

__global__ void __launch_bounds__(kLnThreads) k_layernorm(const float* x, int ldx, const float* w, const float* b,
                                                          float eps, float* y, int ldy, int n) {
  pdl_trigger();
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* xr = x + size_t(r) * ldx;
  float* yr = y + size_t(r) * ldy;
  float xv[kLnCache];
  float s = 0.f;
#pragma unroll
  for (int u = 0; u < kLnCache; ++u) {
    const int i = tid + u * kLnThreads;
    if (i < n) {
      const float v = xr[i];
      xv[u] = v;
      s += v;
    }
  }
  for (int i = tid + kLnCache * kLnThreads; i < n; i += kLnThreads) s += xr[i];
  const float mean = block_sum(s) / float(n);

  float ss = 0.f;
#pragma unroll
  for (int u = 0; u < kLnCache; ++u) {
    const int i = tid + u * kLnThreads;
    if (i < n) {
      const float d = xv[u] - mean;
      ss = __fmaf_rn(d, d, ss);
    }
  }
  for (int i = tid + kLnCache * kLnThreads; i < n; i += kLnThreads) {
    const float d = xr[i] - mean;
    ss = __fmaf_rn(d, d, ss);
  }
  const float inv = rsqrtf(block_sum(ss) / float(n) + eps);

#pragma unroll
  for (int u = 0; u < kLnCache; ++u) {
    const int i = tid + u * kLnThreads;
    if (i < n) {
      const float norm = (xv[u] - mean) * inv;
      yr[i] = norm * (w ? w[i] : 1.f) + (b ? b[i] : 0.f);
    }
  }
  for (int i = tid + kLnCache * kLnThreads; i < n; i += kLnThreads) {
    const float norm = (xr[i] - mean) * inv;
    yr[i] = norm * (w ? w[i] : 1.f) + (b ? b[i] : 0.f);
  }
}

__global__ void k_add_bias(float* y, int ldy, const float* b, int cols) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cols) y[size_t(r) * ldy + c] += b[c];
}

__global__ void k_vision_residual_add(float* x, const float* y, const float* scale, int cols) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < cols) {
    const size_t idx = size_t(r) * cols + c;
    const float v = y[idx];
    x[idx] += scale ? v * scale[c] : v;
  }
}

__device__ __forceinline__ float eval_vact(float v, int act) {
  switch (act) {
    case FLYX_VACT_GELU_TANH: {
      constexpr float kSqrt2OverPi = 0.7978845608028654f;
      return 0.5f * v * (1.f + tanhf(kSqrt2OverPi * (v + 0.044715f * v * v * v)));
    }
    case FLYX_VACT_GELU: {
      constexpr float kInvSqrt2 = 0.7071067811865475f;
      return 0.5f * v * (1.f + erff(v * kInvSqrt2));
    }
    case FLYX_VACT_SILU:
      return v / (1.f + expf(-v));
    case FLYX_VACT_QUICK_GELU:
      return v / (1.f + expf(-1.702f * v));
    default:
      return v;
  }
}

__global__ void k_vact_apply(float* x, size_t n, int act) {
  pdl_trigger();
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) x[i] = eval_vact(x[i], act);
}

__global__ void k_vact_gated(const float* gate, const float* up, float* out, size_t n, int act) {
  pdl_trigger();
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = eval_vact(gate[i], act) * up[i];
}

// Tiled float32 GEMM: y[m, n] (=|+=) sum_k x[m, k] * W[n, k], arbitrary M, N, K.
// Block tile 64x64x16, 16x16 threads (each thread accumulates a 4x4 sub-tile) in fixed K order.
__global__ void __launch_bounds__(256) k_gemm_f32(const float* __restrict__ x, int ldx, const float* __restrict__ W,
                                                  int N, int K, float* __restrict__ y, int ldy, int M, int accumulate) {
  pdl_trigger();
  constexpr int BM = 64, BN = 64, BK = 16;
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BN][BK + 1];
  const int tid = threadIdx.y * 16 + threadIdx.x;
  const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;

  // Each of 256 threads loads 4 elements of As (64*16 = 1024) and 4 elements of Bs (64*16 = 1024).
  const int lr = tid >> 2;         // 0..63
  const int lc = (tid & 3) * 4;    // 0, 4, 8, 12
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int gm = m0 + lr;
    const int gn = n0 + lr;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int kk = k0 + lc + e;
      As[lr][lc + e] = (gm < M && kk < K) ? x[size_t(gm) * ldx + kk] : 0.f;
      Bs[lr][lc + e] = (gn < N && kk < K) ? W[size_t(gn) * K + kk] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float av[4], bv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) av[i] = As[threadIdx.y * 4 + i][kk];
#pragma unroll
      for (int j = 0; j < 4; ++j) bv[j] = Bs[threadIdx.x * 4 + j][kk];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int rr = m0 + threadIdx.y * 4 + i;
    if (rr >= M) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int cc = n0 + threadIdx.x * 4 + j;
      if (cc >= N) continue;
      float* dst = y + size_t(rr) * ldy + cc;
      *dst = accumulate ? *dst + acc[i][j] : acc[i][j];
    }
  }
}

__global__ void k_vision_pos_embed_add(float* x, const float* pos, int pos_rows, int H) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < H) x[size_t(r) * H + c] += pos[size_t(r % pos_rows) * H + c];
}

__global__ void k_vision_pos_embed_bilinear_add(float* x, const float* pos, int pos_h, int pos_w, int H,
                                                const int32_t* coords_h, const int32_t* coords_w,
                                                const int32_t* row_gh, const int32_t* row_gw) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const int h = coords_h[r], w = coords_w[r], gh = row_gh[r], gw = row_gw[r];
  const double sy = gh > 1 ? double(pos_h - 1) / double(gh - 1) : 0.0;
  const double sx = gw > 1 ? double(pos_w - 1) / double(gw - 1) : 0.0;
  const double yf = h * sy;
  const double xf = w * sx;
  const int y0 = min(pos_h - 1, max(0, int(floor(yf))));
  const int x0 = min(pos_w - 1, max(0, int(floor(xf))));
  const int y1 = min(pos_h - 1, y0 + 1);
  const int x1 = min(pos_w - 1, x0 + 1);
  const float dy = float(yf - y0);
  const float dx = float(xf - x0);
  const float w00 = (1.f - dy) * (1.f - dx);
  const float w01 = (1.f - dy) * dx;
  const float w10 = dy * (1.f - dx);
  const float w11 = dy * dx;
  const float p00 = pos[(size_t(y0) * pos_w + x0) * H + c];
  const float p01 = pos[(size_t(y0) * pos_w + x1) * H + c];
  const float p10 = pos[(size_t(y1) * pos_w + x0) * H + c];
  const float p11 = pos[(size_t(y1) * pos_w + x1) * H + c];
  x[size_t(r) * H + c] += w00 * p00 + w01 * p01 + w10 * p10 + w11 * p11;
}

// 2D axial Vision RoPE (ops/rope/ref.h rope_2d_axial): grid (rows), block 256.
// Caches cos/sin for all half = hd / 2 pairs in shared memory once per row, then rotates all nh Q and K heads.
__global__ void __launch_bounds__(256) k_vision_rope_2d(float* qkv, int ld, int q_off, int k_off, int nh, int hd,
                                                        double theta, const int32_t* coords_h,
                                                        const int32_t* coords_w) {
  pdl_trigger();
  extern __shared__ float cs[];  // [2 * half]: cos[0..half-1], sin[0..half-1]
  const int r = blockIdx.x;
  const int half = hd / 2;
  const int quarter = half / 2;
  const int ph = coords_h[r], pw = coords_w[r];
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const bool is_w = i >= quarter;
    const int j = is_w ? i - quarter : i;
    const int coord = is_w ? pw : ph;
    const double inv = pow(theta, -2.0 * j / double(half));
    float c, s;
    rope_cs_inv(coord, inv, &c, &s);
    cs[i] = c;
    cs[half + i] = s;
  }
  __syncthreads();
  float* row = qkv + size_t(r) * ld;
  const int total = nh * half;
  for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
    const int h = idx / half;
    const int i = idx - h * half;
    const float c = cs[i], s = cs[half + i];
    float* q = row + q_off + h * hd;
    const float q1 = q[i], q2 = q[i + half];
    q[i] = q1 * c - q2 * s;
    q[i + half] = q2 * c + q1 * s;
    float* k = row + k_off + h * hd;
    const float k1 = k[i], k2 = k[i + half];
    k[i] = k1 * c - k2 * s;
    k[i + half] = k2 * c + k1 * s;
  }
}

// Bidirectional block-diagonal vision attention (arbitrary hd <= 256, including hd = 72).
// Block = 256 threads (8 warps, 1 warp per query row). Each block cooperatively tiles BK keys and
// values into shared memory for 8x DRAM reuse and computes online softmax in IEEE float32.
template <int MAX_E, int BK>
__global__ void __launch_bounds__(256) k_vision_attention(const float* __restrict__ qkv, int ld, int q_off, int k_off,
                                                          int v_off, int nh, int hd, const int32_t* __restrict__ cu,
                                                          int n_segs, int rows, float scale, float* __restrict__ out,
                                                          int ld_out) {
  pdl_trigger();
  extern __shared__ float smem[];  // Ks[BK][hd_pad], Vs[BK][hd_pad]
  const int hd_pad = hd + 1;
  float* Ks = smem;
  float* Vs = smem + BK * hd_pad;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int r0 = blockIdx.x * 8;
  const int r = r0 + warp;
  const int h = blockIdx.y;

  // Find segment [s0, s1) for this warp's query row, and block-wide [tile_lo, tile_hi) across the 8 rows.
  int s0 = 0, s1 = 0;
  if (r < rows) {
    for (int seg = 0; seg < n_segs; ++seg) {
      if (r < cu[seg + 1]) {
        s0 = cu[seg];
        s1 = cu[seg + 1];
        break;
      }
    }
  }
  int tile_lo = rows, tile_hi = 0;
  const int r_end = min(rows, r0 + 8);
  for (int seg = 0; seg < n_segs; ++seg) {
    const int a = cu[seg], b = cu[seg + 1];
    if (b > r0 && a < r_end) {
      tile_lo = min(tile_lo, a);
      tile_hi = max(tile_hi, b);
    }
  }

  float qv[MAX_E], acc[MAX_E];
#pragma unroll
  for (int e = 0; e < MAX_E; ++e) {
    const int d = lane + 32 * e;
    qv[e] = (r < rows && d < hd) ? qkv[size_t(r) * ld + q_off + h * hd + d] : 0.f;
    acc[e] = 0.f;
  }
  float m = -INFINITY;
  float l = 0.f;

  for (int t = tile_lo; t < tile_hi; t += BK) {
    const int kn = min(BK, tile_hi - t);
    // Cooperative load of kn x hd keys and values into shared memory.
    for (int idx = threadIdx.x; idx < kn * hd; idx += 256) {
      const int j = idx / hd;
      const int d = idx - j * hd;
      const size_t base = size_t(t + j) * ld + h * hd + d;
      Ks[j * hd_pad + d] = qkv[base + k_off];
      Vs[j * hd_pad + d] = qkv[base + v_off];
    }
    __syncthreads();

    if (r < rows) {
      for (int j = 0; j < kn; ++j) {
        const int kpos = t + j;
        if (kpos < s0 || kpos >= s1) continue;
        float dot = 0.f;
#pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
          const int d = lane + 32 * e;
          if (d < hd) dot = __fmaf_rn(qv[e], Ks[j * hd_pad + d], dot);
        }
        const float sc = warp_sum(dot) * scale;
        const float mn = fmaxf(m, sc);
        const float corr = mn == m ? 1.f : expf(m - mn);
        const float p = expf(sc - mn);
        l = l * corr + p;
#pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
          const int d = lane + 32 * e;
          if (d < hd) acc[e] = __fmaf_rn(p, Vs[j * hd_pad + d], acc[e] * corr);
        }
        m = mn;
      }
    }
    __syncthreads();
  }

  if (r < rows) {
    const float inv_l = l > 0.f ? 1.f / l : 0.f;
    float* dst = out + size_t(r) * ld_out + h * hd;
#pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
      const int d = lane + 32 * e;
      if (d < hd) dst[d] = acc[e] * inv_l;
    }
  }
}

__global__ void k_vision_avg_pool_blocks(const float* x, int group, int hidden, float* out) {
  pdl_trigger();
  const int b = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden) return;
  float sum = 0.f;
  for (int g = 0; g < group; ++g) sum += x[(size_t(b) * group + g) * hidden + c];
  out[size_t(b) * hidden + c] = sum / float(group);
}

__global__ void k_embed_scatter(float* x, bf16* x16, int H, const float* embeds, const int32_t* embed_rows) {
  pdl_trigger();
  const int i = blockIdx.x;
  const int row = embed_rows[i];
  const float* src = embeds + size_t(i) * H;
  if (x) {
    float* dst = x + size_t(row) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = src[c];
  } else if (x16) {
    bf16* dst = x16 + size_t(row) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = __float2bfloat16_rn(src[c]);
  }
}

}  // namespace

void layernorm(const float* x, int ldx, const float* w, const float* b, float eps, float* y, int ldy, int rows, int n,
               cudaStream_t st) {
  if (rows <= 0) return;
  k_layernorm<<<rows, kLnThreads, 0, st>>>(x, ldx, w, b, eps, y, ldy, n);
}

void add_bias(float* y, int ldy, const float* b, int rows, int cols, cudaStream_t st) {
  if (rows <= 0 || cols <= 0) return;
  k_add_bias<<<dim3(ceil_div(cols, 256), rows), 256, 0, st>>>(y, ldy, b, cols);
}

void vision_residual_add(float* x, const float* y, const float* scale, int rows, int cols, cudaStream_t st) {
  if (rows <= 0 || cols <= 0) return;
  k_vision_residual_add<<<dim3(ceil_div(cols, 256), rows), 256, 0, st>>>(x, y, scale, cols);
}

void vact_apply(float* x, size_t n, int act, cudaStream_t st) {
  if (n == 0) return;
  k_vact_apply<<<unsigned((n + 255) / 256), 256, 0, st>>>(x, n, act);
}

void vact_gated(const float* gate, const float* up, float* out, size_t n, int act, cudaStream_t st) {
  if (n == 0) return;
  k_vact_gated<<<unsigned((n + 255) / 256), 256, 0, st>>>(gate, up, out, n, act);
}

void gemm_f32(const float* x, int ldx, const float* W, int N, int K, float* y, int ldy, int rows, bool accumulate,
              cudaStream_t st) {
  if (rows <= 0 || N <= 0) return;
  k_gemm_f32<<<dim3(ceil_div(N, 64), ceil_div(rows, 64)), dim3(16, 16), 0, st>>>(x, ldx, W, N, K, y, ldy, rows,
                                                                                 accumulate ? 1 : 0);
}

void vision_pos_embed_add(float* x, const float* pos, int pos_rows, int H, int rows, cudaStream_t st) {
  if (rows <= 0 || H <= 0 || pos_rows <= 0) return;
  k_vision_pos_embed_add<<<dim3(ceil_div(H, 256), rows), 256, 0, st>>>(x, pos, pos_rows, H);
}

void vision_pos_embed_bilinear_add(float* x, const float* pos, int pos_h, int pos_w, int H, const int32_t* coords_h,
                                   const int32_t* coords_w, const int32_t* row_gh, const int32_t* row_gw, int rows,
                                   cudaStream_t st) {
  if (rows <= 0 || H <= 0) return;
  k_vision_pos_embed_bilinear_add<<<dim3(ceil_div(H, 256), rows), 256, 0, st>>>(x, pos, pos_h, pos_w, H, coords_h,
                                                                                coords_w, row_gh, row_gw);
}

void vision_rope_2d(float* qkv, int ld, int q_off, int k_off, int nh, int hd, double theta, const int32_t* coords_h,
                    const int32_t* coords_w, int rows, cudaStream_t st) {
  if (rows <= 0 || nh <= 0 || hd <= 0) return;
  const size_t shared = size_t(hd) * sizeof(float);  // 2 * (hd / 2) floats
  k_vision_rope_2d<<<rows, 256, shared, st>>>(qkv, ld, q_off, k_off, nh, hd, theta, coords_h, coords_w);
}

void vision_attention(const float* qkv, int ld, int q_off, int k_off, int v_off, int nh, int hd,
                      const int32_t* cu_seqlens, int n_segs, int rows, float scale, float* out, int ld_out,
                      cudaStream_t st) {
  if (rows <= 0 || nh <= 0 || hd <= 0) return;
  if (hd > 256) throw std::runtime_error("vision_attention: head_dim > 256 not supported");
  const dim3 grid(ceil_div(rows, 8), nh);
  if (hd <= 128) {
    constexpr int BK = 32;
    const size_t shared = 2 * size_t(BK) * size_t(hd + 1) * sizeof(float);
    k_vision_attention<4, BK><<<grid, 256, shared, st>>>(qkv, ld, q_off, k_off, v_off, nh, hd, cu_seqlens, n_segs,
                                                         rows, scale, out, ld_out);
  } else {
    constexpr int BK = 16;
    const size_t shared = 2 * size_t(BK) * size_t(hd + 1) * sizeof(float);
    k_vision_attention<8, BK><<<grid, 256, shared, st>>>(qkv, ld, q_off, k_off, v_off, nh, hd, cu_seqlens, n_segs,
                                                         rows, scale, out, ld_out);
  }
}

void vision_avg_pool_blocks(const float* x, int n_blocks, int group, int hidden, float* out, cudaStream_t st) {
  if (n_blocks <= 0 || hidden <= 0) return;
  k_vision_avg_pool_blocks<<<dim3(ceil_div(hidden, 256), n_blocks), 256, 0, st>>>(x, group, hidden, out);
}

void embed_scatter(float* x, bf16* x16, int H, const float* embeds, const int32_t* embed_rows, int n_embeds,
                   cudaStream_t st) {
  if (n_embeds <= 0 || H <= 0) return;
  k_embed_scatter<<<n_embeds, 256, 0, st>>>(x, x16, H, embeds, embed_rows);
}

}  // namespace flyx::cuda
