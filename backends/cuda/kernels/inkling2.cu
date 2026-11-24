// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Inkling-2 target kernels: logit softcapping, Block Attention Residuals, cross-pass loop control,
// causal depthwise ShortConv (with optional per-head dynamic taps), and relative-position-bias attention.
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// ---- softcap_rows -------------------------------------------------------------------------
__global__ void k_softcap_rows(float* logits, int ld, int V, float cap) {
  pdl_trigger();
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (v >= V) return;
  float* p = logits + size_t(r) * ld + v;
  const float x = *p;
  if (isfinite(x)) *p = cap * tanhf(x / cap);
}

// ---- attn_res_mix -------------------------------------------------------------------------
// One block (256 threads) per row r. Computes up to 4 queries at a time over M = n_blocks + 1
// states (bank[0..n_blocks-1] and cur), each RMS-normalized (no gamma, eps = 1e-6).
constexpr int kMaxResStates = 64;

__global__ void __launch_bounds__(256) k_attn_res_mix(const float* __restrict__ bank, int n_blocks,
                                                      const float* __restrict__ cur, const float* __restrict__ queries,
                                                      int q0, int nq_chunk, int rows, int H, float* __restrict__ out) {
  pdl_trigger();
  const int r = blockIdx.x, tid = threadIdx.x;
  const int M = n_blocks + 1;
  __shared__ float s_inv[kMaxResStates];
  __shared__ float s_dot[4][kMaxResStates];
  __shared__ float s_alpha[4][kMaxResStates];
  __shared__ float s_part[kMaxResStates][5][8];  // a state's five sums, a warp each

  // Every state's sums as block_sum_n makes them (a warp's tree, then the eight warps'), with one barrier for all
  // the states where each had three: a warp leaves its sums, and thread m finishes state m.  The same bits, and a
  // round runs this two to three times a layer visit.
  const float inv_sqrt_h = rsqrtf(float(H));
  for (int m = 0; m < M; ++m) {
    const float* src = (m < n_blocks) ? (bank + (size_t(m) * rows + r) * H) : (cur + size_t(r) * H);
    float acc[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    for (int i = tid; i < H; i += 256) {
      const float v = src[i];
      acc[0] = __fmaf_rn(v, v, acc[0]);
      if (nq_chunk >= 1) acc[1] = __fmaf_rn(v, queries[size_t(q0 + 0) * H + i], acc[1]);
      if (nq_chunk >= 2) acc[2] = __fmaf_rn(v, queries[size_t(q0 + 1) * H + i], acc[2]);
      if (nq_chunk >= 3) acc[3] = __fmaf_rn(v, queries[size_t(q0 + 2) * H + i], acc[3]);
      if (nq_chunk >= 4) acc[4] = __fmaf_rn(v, queries[size_t(q0 + 3) * H + i], acc[4]);
    }
#pragma unroll
    for (int a = 0; a < 5; ++a) {
      const float w = warp_sum(acc[a]);
      if ((tid & 31) == 0) s_part[m][a][tid >> 5] = w;
    }
  }
  __syncthreads();
  if (tid < M) {
    float t[5];
#pragma unroll
    for (int a = 0; a < 5; ++a) {
      // block_sum_n's second tree over lanes 0..7 of a warp whose other lanes hold +0
      float p[8];
#pragma unroll
      for (int w = 0; w < 8; ++w) p[w] = s_part[tid][a][w] + 0.f;
      t[a] = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
    }
    const float inv = rsqrtf(t[0] / float(H) + 1e-6f);
    s_inv[tid] = inv;
    const float sc = inv * inv_sqrt_h;
    for (int q = 0; q < nq_chunk; ++q) s_dot[q][tid] = t[q + 1] * sc;
  }
  __syncthreads();
  if (tid == 0) {
    for (int q = 0; q < nq_chunk; ++q) {
      float mx = -INFINITY;
      for (int m = 0; m < M; ++m) mx = fmaxf(mx, s_dot[q][m]);
      float z = 0.f;
      for (int m = 0; m < M; ++m) {
        const float e = expf(s_dot[q][m] - mx);
        s_alpha[q][m] = e;
        z += e;
      }
      const float inv_z = 1.f / z;
      for (int m = 0; m < M; ++m) s_alpha[q][m] = (s_alpha[q][m] * inv_z) * s_inv[m];
    }
  }
  __syncthreads();
  for (int i = tid; i < H; i += 256) {
    float o[4] = {0.f, 0.f, 0.f, 0.f};
    for (int m = 0; m < M; ++m) {
      const float* src = (m < n_blocks) ? (bank + (size_t(m) * rows + r) * H) : (cur + size_t(r) * H);
      const float v = src[i];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        if (q < nq_chunk) o[q] = __fmaf_rn(s_alpha[q][m], v, o[q]);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
      if (q < nq_chunk) out[(size_t(q0 + q) * rows + r) * H + i] = o[q];
  }
}

// ---- loop_ctrl_prep & loop_ctrl_apply -----------------------------------------------------
__global__ void __launch_bounds__(512) k_loop_ctrl_prep(const float* __restrict__ cur, const float* __restrict__ msg,
                                                        int H, float* __restrict__ cat_out) {
  pdl_trigger();
  const int r = blockIdx.x, which = blockIdx.y, tid = threadIdx.x;
  const float* src = (which == 0 ? cur : msg) + size_t(r) * H;
  float* dst = cat_out + size_t(r) * (2 * H) + (which == 0 ? 0 : H);
  float ss = 0.f;
  for (int i = tid; i < H; i += 512) {
    const float v = src[i];
    ss = __fmaf_rn(v, v, ss);
  }
  const float inv = rsqrtf(block_sum(ss) / float(H) + 1e-6f);
  for (int i = tid; i < H; i += 512) dst[i] = src[i] * inv;
}

__global__ void k_loop_ctrl_apply(const float* __restrict__ cur, const float* __restrict__ msg,
                                  const float* __restrict__ gain, int ld_gain, const float* __restrict__ forget,
                                  int ld_forget, int H, float* __restrict__ out) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (i >= H) return;
  const size_t at = size_t(r) * H + i;
  const float c = cur[at];
  const float m = msg[at];
  const float gn = tanhf(gain[size_t(r) * ld_gain + i]);
  const float fg = tanhf(forget[size_t(r) * ld_forget + i]);
  out[at] = c - fg * c + gn * m;
}

// ---- loop_control & linear_f32: the controllers and the control projections as rounds run them -------------
// The weights are read cold from device memory each time (a token passes gigabytes of weights between two uses).
// A matrix with exact bf16 values needs only its high words. Many small blocks each read their share at once.
// A block takes NR rows (1, 2, 4 or 8: a template, since rows counted at run time in the inner loops made a one-row
// launch five times as long) and reads the weights once for those rows. Rows past the last repeat it and are not written.
struct LoopGates {
  LoopGate g[kLoopGates];
};
constexpr int kLoopBlock = 256, kLoopRows = 8;
inline int loop_rows(int rows) { return rows == 1 ? 1 : rows == 2 ? 2 : rows <= 4 ? 4 : kLoopRows; }

// A controller owns a grid plane, so the choice of width is uniform across a block and made outside the dot product.
template <typename Dot>
__device__ __forceinline__ void with_weight_row(RoundWeight w, size_t offset, Dot dot) {
  if (w.b16) dot(w.b16 + offset);
  else dot(w.f32 + offset);
}

// A block: one output of one controller's down projection for NR rows, a thread its inputs 256 apart in order and
// then the block's tree; cur's half and msg's half summed apart (raw[2 o], raw[2 o + 1]): each half's norm scale
// multiplies its sum, in k_loop_apply.  The blocks of output 0 also sum the rows' squares for those scales
// (raw[2 rank], raw[2 rank + 1]): once for all that read them, and before any output is written (out may be cur).
template <int NR>
__global__ void __launch_bounds__(kLoopBlock) k_loop_gate(LoopGates G, int H, int rank, int rows, float* raw) {
  pdl_trigger();
  const int o = blockIdx.x, r0 = blockIdx.y * NR, c = blockIdx.z, tid = threadIdx.x;
  const float *cur[NR], *msg[NR];
  float acc[2 * NR];
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    const int r = min(r0 + j, rows - 1);
    cur[j] = G.g[c].cur + size_t(r) * H;
    msg[j] = G.g[c].msg + size_t(r) * H;
    acc[2 * j] = acc[2 * j + 1] = 0.f;
  }
  with_weight_row(G.g[c].down, size_t(o) * size_t(2 * H), [&](const auto* w) {
    for (int i = tid; i < H; i += kLoopBlock) {
      const float wc = in_value(w + i), wm = in_value(w + H + i);
#pragma unroll
      for (int j = 0; j < NR; ++j) {
        acc[2 * j] = __fmaf_rn(wc, cur[j][i], acc[2 * j]);
        acc[2 * j + 1] = __fmaf_rn(wm, msg[j][i], acc[2 * j + 1]);
      }
    }
  });
  block_sum_n(acc);
  float* row0 = raw + (size_t(c) * rows + r0) * loop_row_floats(rank);
#pragma unroll
  for (int j = 0; j < 2 * NR; ++j)
    if (tid == j && r0 + (j >> 1) < rows) row0[size_t(j >> 1) * loop_row_floats(rank) + 2 * o + (j & 1)] = acc[j];
  if (o != 0) return;
#pragma unroll
  for (int j = 0; j < 2 * NR; ++j) acc[j] = 0.f;
  for (int i = tid; i < H; i += kLoopBlock) {
#pragma unroll
    for (int j = 0; j < NR; ++j) {
      const float a = cur[j][i], b = msg[j][i];
      acc[2 * j] = __fmaf_rn(a, a, acc[2 * j]);
      acc[2 * j + 1] = __fmaf_rn(b, b, acc[2 * j + 1]);
    }
  }
  block_sum_n(acc);
#pragma unroll
  for (int j = 0; j < 2 * NR; ++j)
    if (tid == j && r0 + (j >> 1) < rows) row0[size_t(j >> 1) * loop_row_floats(rank) + 2 * rank + (j & 1)] = acc[j];
}

// A warp: one output of one controller for NR rows, its lanes 32 of the rank apart and then their tree.  A lane
// makes the gates it reads from the raw sums and the norms' scales (every lane the same scales).
template <int NR>
__global__ void __launch_bounds__(kLoopBlock) k_loop_apply(LoopGates G, int H, int rank, int rows, const float* raw) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, h = blockIdx.x * (kLoopBlock / 32) + (threadIdx.x >> 5);
  const int r0 = blockIdx.y * NR, c = blockIdx.z;
  if (h >= H) return;
  const float* row[NR];
  float inv_cur[NR], inv_msg[NR], gn[NR], fg[NR];
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    row[j] = raw + (size_t(c) * rows + min(r0 + j, rows - 1)) * loop_row_floats(rank);
    inv_cur[j] = rsqrtf(row[j][2 * rank] / float(H) + 1e-6f);
    inv_msg[j] = rsqrtf(row[j][2 * rank + 1] / float(H) + 1e-6f);
    gn[j] = fg[j] = 0.f;
  }
  const size_t offset = size_t(h) * rank;
  with_weight_row(G.g[c].gain, offset, [&](const auto* gain) {
    with_weight_row(G.g[c].forget, offset, [&](const auto* forget) {
      for (int i = lane; i < rank; i += 32) {
        const float wg = in_value(gain + i), wf = in_value(forget + i);
#pragma unroll
        for (int j = 0; j < NR; ++j) {
          const float gi = __fmaf_rn(row[j][2 * i], inv_cur[j], __fmul_rn(row[j][2 * i + 1], inv_msg[j]));
          gn[j] = __fmaf_rn(wg, gi, gn[j]);
          fg[j] = __fmaf_rn(wf, gi, fg[j]);
        }
      }
    });
  });
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    const float g = warp_sum(gn[j]), f = warp_sum(fg[j]);
    if (lane == 0 && r0 + j < rows) {
      const size_t at = (size_t(r0) + j) * H + h;
      const float cv = G.g[c].cur[at], mv = G.g[c].msg[at];
      G.g[c].out[at] = __fmaf_rn(tanhf(g), mv, __fmaf_rn(-tanhf(f), cv, cv));
    }
  }
}

// A warp: one output for NR rows, its lanes 32 values of K apart in order, then their tree.
template <int NR, typename Weight>
__global__ void __launch_bounds__(kLoopBlock) k_linear_f32(const float* x, int ldx, const Weight* __restrict__ W, int N,
                                                           int K, float* y, int ldy, int rows, int accumulate) {
  pdl_trigger();
  const int lane = threadIdx.x & 31, n = blockIdx.x * (kLoopBlock / 32) + (threadIdx.x >> 5);
  const int r0 = blockIdx.y * NR;
  if (n >= N) return;
  const Weight* w = W + size_t(n) * K;
  const float* xr[NR];
  float acc[NR];
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    xr[j] = x + size_t(min(r0 + j, rows - 1)) * ldx;
    acc[j] = 0.f;
  }
  for (int k = lane; k < K; k += 32) {
    const float wk = in_value(w + k);
#pragma unroll
    for (int j = 0; j < NR; ++j) acc[j] = __fmaf_rn(wk, xr[j][k], acc[j]);
  }
#pragma unroll
  for (int j = 0; j < NR; ++j) {
    const float v = warp_sum(acc[j]);
    if (lane == 0 && r0 + j < rows) {
      float* out = y + (size_t(r0) + j) * ldy + n;
      *out = accumulate ? *out + v : v;
    }
  }
}

// ---- shortconv_pass & shortconv_fold ------------------------------------------------------
constexpr int kMaxShortConv = 64;

__global__ void k_shortconv_pass(const float* __restrict__ x, int ldx, const float* __restrict__ w,
                                 const float* __restrict__ dyn, float* __restrict__ win, size_t win_stride, int in_sel,
                                 int next_sel, int skip, float* __restrict__ replay, size_t replay_stride,
                                 int replay_sel, int ch_off, int ch_total, int C, int conv, int n_heads,
                                 const DevParams* P, int rows, int tile_rows, float* __restrict__ out, int ldo,
                                 int accumulate) {
  pdl_trigger();
  if ((skip == kSkipIfStopped && P->stopped) || (skip == kSkipIfNoop && P->noop)) return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int hist = conv - 1;
  const int r0 = blockIdx.y * tile_rows;
  const int r1 = min(rows, r0 + tile_rows);
  const float inv_scale = rsqrtf(float((dyn ? 2 : 1) * conv + 1));
  const int head_idx = dyn ? (c / (C / n_heads)) : 0;

  float wc[kMaxShortConv];
  const float* wrow = w + size_t(c) * conv;
  for (int k = 0; k < conv; ++k) wc[k] = wrow[k];

  float old_h[kMaxShortConv];
  if (blockIdx.y == 0) {
    const float* win_in = win + size_t(slot_of(P, in_sel)) * win_stride;
    for (int k = 0; k < hist; ++k) old_h[k] = win_in[size_t(k) * ch_total + ch_off + c];
  }

  float* rep_base = replay ? (replay + size_t(replay_slot_of(P, replay_sel)) * replay_stride) : nullptr;
  for (int r = r0; r < r1; ++r) {
    const float xr = x[size_t(r) * ldx + c];
    if (rep_base) rep_base[size_t(r) * ch_total + ch_off + c] = xr;
    float acc = 0.f;
    for (int k = 0; k < hist; ++k) {
      const int t = r - hist + k;
      const float hv = t < 0 ? old_h[hist + t] : x[size_t(t) * ldx + c];
      acc = __fmaf_rn(hv, wc[k], acc);
    }
    acc = __fmaf_rn(xr, wc[hist], acc);
    if (dyn) {
      const float* dh = dyn + (size_t(r) * n_heads + head_idx) * conv;
      acc = __fmaf_rn(xr, dh[0], acc);
      for (int lag = 1; lag < conv; ++lag) {
        const int t = r - lag;
        const float hv = t < 0 ? old_h[hist + t] : x[size_t(t) * ldx + c];
        acc = __fmaf_rn(hv, dh[lag], acc);
      }
    }
    const float y = (xr + acc) * inv_scale;
    float* dst = out + size_t(r) * ldo + c;
    *dst = accumulate ? (*dst + y) : y;
  }

  // Only blockIdx.y == 0 reads win_in (tile_rows >= hist), so writing win_out after computing its tile is race-free
  // even when next_sel == in_sel (prefill in-place update).
  if (next_sel >= 0 && blockIdx.y == 0) {
    float* win_out = win + size_t(slot_of(P, next_sel)) * win_stride;
    for (int k = 0; k < hist; ++k) {
      const int src_r = rows - hist + k;
      const float v = src_r < 0 ? old_h[hist + src_r] : x[size_t(src_r) * ldx + c];
      win_out[size_t(k) * ch_total + ch_off + c] = v;
    }
  }
}

__global__ void k_shortconv_fold(const ShortConvLayer* __restrict__ layers, size_t win_stride, size_t replay_stride,
                                 int in_sel, int out_sel, int replay_sel, int channels, int conv, const DevParams* P,
                                 int rows, const int* rows_dev, int skip_if_noop) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const ShortConvLayer L = layers[blockIdx.y];
  const int nr = rows_dev ? (*rows_dev + 1) : rows;
  const int hist = conv - 1;
  const float* win_in = L.win + size_t(slot_of(P, in_sel)) * win_stride;
  float* win_out = L.win + size_t(slot_of(P, out_sel)) * win_stride;
  const float* rep = L.replay + size_t(replay_slot_of(P, replay_sel)) * replay_stride;
  float old_h[kMaxShortConv];
  for (int k = 0; k < hist; ++k) old_h[k] = win_in[size_t(k) * channels + c];
  for (int k = 0; k < hist; ++k) {
    const int src_r = nr - hist + k;
    win_out[size_t(k) * channels + c] = src_r < 0 ? old_h[hist + src_r] : rep[size_t(src_r) * channels + c];
  }
}

// ---- rel_attn_prep ------------------------------------------------------------------------
// grid (nh + nkv, rows); block size = round_up_32(max(hd, d_rel)).
template <bool Compact = false>
__global__ void k_rel_attn_prep(const float* __restrict__ q_in, int ldq, const float* __restrict__ k_in, int ldk,
                                const float* __restrict__ v_in, int ldv, const float* __restrict__ r_in, int ldr,
                                const float* __restrict__ q_clip_w, const float* __restrict__ r_clip_w, float qk_clip,
                                float rel_clip, float v_eps, int log_floor, float log_alpha, int nh, int nkv, int hd,
                                int d_rel, const DevParams* P, int pos_sel, int is_swa, float* __restrict__ q_out,
                                float* __restrict__ r_out, KvRef kv, int pos_add, int strata_base) {
  pdl_trigger();
  extern __shared__ float sh[];
  const int hidx = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int base = (pos_sel == kPosStart ? P->start : P->position) + pos_add;
  const int pos = base + r;

  if (hidx < nh) {
    const int hh = hidx;
    const float tau =
        (!is_swa && log_floor > 0 && log_alpha > 0.f)
            ? float(1.0 + double(log_alpha) * log(fmax(1.0, double(pos + 1) / double(log_floor))))
            : 1.f;
    // Q head: L2 clip with q_clip_w, then * tau.
    const float qv = d < hd ? q_in[size_t(r) * ldq + hh * hd + d] * q_clip_w[size_t(hh) * hd + d] : 0.f;
    const float qsq = block_sum(qv * qv);
    const float qfac = fminf(1.f, fmaxf(0.f, qk_clip / (sqrtf(qsq) + 1e-6f))) * tau;
    if (d < hd) q_out[(size_t(r) * nh + hh) * hd + d] = qv * qfac;

    // R head: L2 clip with r_clip_w, then * tau.
    const float rv = d < d_rel ? r_in[size_t(r) * ldr + hh * d_rel + d] * r_clip_w[size_t(hh) * d_rel + d] : 0.f;
    const float rsq = block_sum(rv * rv);
    const float rfac = fminf(1.f, fmaxf(0.f, rel_clip / (sqrtf(rsq) + 1e-6f))) * tau;
    if (d < d_rel) r_out[(size_t(r) * nh + hh) * d_rel + d] = rv * rfac;
    return;
  }

  // KV head: K L2 clip (no scale), V no-gamma RMSNorm, write to kv.slot_of(pos).
  if (pos < 0) return;
  const int kh = hidx - nh;
  const float kv_val = d < hd ? k_in[size_t(r) * ldk + kh * hd + d] : 0.f;
  const float vv_val = d < hd ? v_in[size_t(r) * ldv + kh * hd + d] : 0.f;
  float sq[2] = {kv_val * kv_val, vv_val * vv_val};
  block_sum_n(sq);
  const float kfac = fminf(1.f, fmaxf(0.f, qk_clip / (sqrtf(sq[0]) + 1e-6f)));
  const float vinv = rsqrtf(sq[1] / float(hd) + v_eps);
  const float ko = kv_val * kfac;
  const float vo = vv_val * vinv;

  const int cell = kv.slot_of(pos);
  const size_t at = (size_t(cell) * nkv + kh) * hd + d;
  if (kv.fp8()) {
    float amax[2] = {d < hd ? fabsf(ko) : 0.f, d < hd ? fabsf(vo) : 0.f};
    block_max_n(amax);
    const float ksc = amax[0] / 448.f, vsc = amax[1] / 448.f;
    if (d < hd) {
      static_cast<uint8_t*>(kv.k)[at] = ksc > 0.f ? __nv_cvt_float_to_fp8(ko / ksc, __NV_SATFINITE, __NV_E4M3) : 0;
      static_cast<uint8_t*>(kv.v)[at] = vsc > 0.f ? __nv_cvt_float_to_fp8(vo / vsc, __NV_SATFINITE, __NV_E4M3) : 0;
    }
    if (d == 0) {
      kv.kscales(cell)[size_t(cell) * nkv + kh] = ksc;
      kv.vscales(cell)[size_t(cell) * nkv + kh] = vsc;
    }
  } else if (d < hd) {
    static_cast<kv16*>(kv.k)[at] = __float2half_rn(ko);
    static_cast<kv16*>(kv.v)[at] = __float2half_rn(vo);
  }

  // Low-bit Strata copy (FWHT + uniform quantization + StrataLayout packing, matching attn_prep bit-for-bit).
  if (kv.kq) {
    float* xk = sh;
    float* xv = xk + hd;
    float fk = d < hd ? ko : 0.f, fv = d < hd ? vo : 0.f;
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
      const float ok = __shfl_xor_sync(0xffffffffu, fk, len);
      const float ov = __shfl_xor_sync(0xffffffffu, fv, len);
      fk = (d & len) ? (ok - fk) : (fk + ok);
      fv = (d & len) ? (ov - fv) : (fv + ov);
    }
    if (hd > 32) {
      if (d < hd) {
        xk[d] = fk;
        xv[d] = fv;
      }
      __syncthreads();
      const int half_hd = hd >> 1;
      for (int len = 32; len < half_hd; len <<= 1) {
        if (d < half_hd) {
          const int lo = ((d & ~(len - 1)) << 1) | (d & (len - 1));
          const float ak = xk[lo], bk = xk[lo + len];
          xk[lo] = ak + bk;
          xk[lo + len] = ak - bk;
          const float av = xv[lo], bv = xv[lo + len];
          xv[lo] = av + bv;
          xv[lo + len] = av - bv;
        }
        __syncthreads();
      }
      if (d < hd) {
        const float sk = xk[d], pk = xk[d ^ half_hd];
        fk = (d & half_hd) ? (pk - sk) : (sk + pk);
        const float sv = xv[d], pv = xv[d ^ half_hd];
        fv = (d & half_hd) ? (pv - sv) : (sv + pv);
      }
    }
    const float inv_s = rsqrtf(float(hd));
    const float xrk = d < hd ? fk * inv_s : 0.f, xrv = d < hd ? fv * inv_s : 0.f;
    const int top = (1 << kv.bits) - 1;
    float lok, stepk, lov, stepv;
    if (kv.grid) {
      float ms[2] = {xrk, xrv};
      block_sum_n(ms);
      const float mk = ms[0] / float(hd), mv = ms[1] / float(hd);
      float vs[2] = {d < hd ? (xrk - mk) * (xrk - mk) : 0.f, d < hd ? (xrv - mv) * (xrv - mv) : 0.f};
      block_sum_n(vs);
      const float c = kv.bits == 2 ? 0.9957f : kv.bits == 3 ? 0.5860f : 0.3352f;
      stepk = c * sqrtf(vs[0] / float(hd));
      stepv = c * sqrtf(vs[1] / float(hd));
      lok = mk - float((top + 1) / 2) * stepk;
      lov = mv - float((top + 1) / 2) * stepv;
    } else {
      float mm[4] = {d < hd ? xrk : -INFINITY, d < hd ? -xrk : -INFINITY, d < hd ? xrv : -INFINITY,
                     d < hd ? -xrv : -INFINITY};
      block_max_n(mm);
      lok = -mm[1];
      stepk = (mm[0] - lok) / float(top + 1);
      lov = -mm[3];
      stepv = (mm[2] - lov) / float(top + 1);
    }
    const __half2 gridk = __floats2half2_rn(lok, stepk);
    const __half2 gridv = __floats2half2_rn(lov, stepv);
    lok = __low2float(gridk);
    stepk = __high2float(gridk);
    lov = __low2float(gridv);
    stepv = __high2float(gridv);
    uint8_t* codes_k = reinterpret_cast<uint8_t*>(xk);
    uint8_t* codes_v = codes_k + hd;
    if (d < hd) {
      codes_k[d] = uint8_t(stepk > 0.f ? min(top, max(0, int(floorf((xrk - lok) / stepk)))) : 0);
      codes_v[d] = uint8_t(stepv > 0.f ? min(top, max(0, int(floorf((xrv - lov) / stepv)))) : 0);
    }
    __syncthreads();
    if (d >= hd) return;
    const StrataLayout L(hd, kv.bits);
    const int sslot = pos - strata_base;
    const size_t woff = strata_words_at(sslot, kh, nkv, kv.words);
    const int p16 = sslot & 15;
    const int kw = Compact ? 24 : L.kwords();
    const int vw = Compact ? 24 : L.vwords();
    const int vbase = hd >= 64 ? ((kv.words + 31) & ~31) : kv.words;
    if (d < kw) {
      uint32_t w = 0;
      if (d < 4 * L.kown) {
        const int t = d / L.kown, j0 = (d % L.kown) * L.cph;
        for (int s = 0; s < L.cph; ++s) {
          const int first = 8 * (j0 + s) + 2 * t;
          w |= uint32_t(codes_k[first]) << (s * kv.bits);
          w |= uint32_t(codes_k[first + 1]) << (16 + s * kv.bits);
        }
      } else {
        const int t0 = (d - 4 * L.kown) * L.kgpw, t_end = min(4, t0 + L.kgpw);
        for (int t = t0; t < t_end; ++t) {
          for (int rr = 0; rr < L.kleft; ++rr) {
            const int s = (t - t0) * L.kleft + rr;
            const int first = 8 * (L.kown * L.cph + rr) + 2 * t;
            w |= uint32_t(codes_k[first]) << (s * kv.bits);
            w |= uint32_t(codes_k[first + 1]) << (16 + s * kv.bits);
          }
        }
      }
      if constexpr (Compact) {
        const int i = d % 6, bit = i % 3;
        const int first = 2 * (4 * (30 + i / 3) + d / 6);
        w |= uint32_t((codes_k[first] >> bit) & 1) << 15;
        w |= uint32_t((codes_k[first + 1] >> bit) & 1) << 31;
      }
      kv.kq[woff + L.k_index(p16, d)] = w;
    } else if (d < kv.words) {
      kv.kq[woff + d * 16 + p16] = 0u;
    } else if (d >= vbase && d < vbase + vw) {
      const int vd = d - vbase;
      uint32_t w = 0;
      if (vd < 8 * L.vown) {
        const int g = vd / L.vown, m0 = (vd % L.vown) * L.cph;
        for (int s = 0; s < L.cph; ++s) {
          const int first = 16 * (m0 + s) + g;
          w |= uint32_t(codes_v[first]) << (s * kv.bits);
          w |= uint32_t(codes_v[first + 8]) << (16 + s * kv.bits);
        }
      } else {
        const int g0 = (vd - 8 * L.vown) * L.vgpw, g_end = min(8, g0 + L.vgpw);
        for (int g = g0; g < g_end; ++g) {
          for (int rr = 0; rr < L.vleft; ++rr) {
            const int s = (g - g0) * L.vleft + rr;
            const int first = 16 * (L.vown * L.cph + rr) + g;
            w |= uint32_t(codes_v[first]) << (s * kv.bits);
            w |= uint32_t(codes_v[first + 8]) << (16 + s * kv.bits);
          }
        }
      }
      if constexpr (Compact) {
        const int i = vd % 3, bit = i;
        const int first = 240 + vd / 3;
        w |= uint32_t((codes_v[first] >> bit) & 1) << 15;
        w |= uint32_t((codes_v[first + 8] >> bit) & 1) << 31;
      }
      kv.vq[woff + L.v_index(p16, vd)] = w;
    } else if (d >= vbase + vw && d < vbase + kv.words) {
      kv.vq[woff + (d - vbase) * 16 + p16] = 0u;
    }
    if (d == 0) {
      kv.kqs[size_t(sslot) * nkv + kh] = gridk;
      kv.vqs[size_t(sslot) * nkv + kh] = gridv;
    }
  }
}

// ---- rel_bias_table -----------------------------------------------------------------------
// Precomputes out[(r * nh + h) * extent + dist] = dot(rel_bias[dist], r_q[r, h]).
__global__ void __launch_bounds__(128) k_rel_bias_table(const float* __restrict__ r_q,
                                                        const float* __restrict__ rel_bias, int nh, int d_rel,
                                                        int rel_extent, float* __restrict__ out) {
  pdl_trigger();
  const int dist = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y, r = blockIdx.z;
  extern __shared__ float sr[];
  const float* rq = r_q + (size_t(r) * nh + h) * d_rel;
  for (int m = threadIdx.x; m < d_rel; m += blockDim.x) sr[m] = rq[m];
  __syncthreads();
  if (dist >= rel_extent) return;
  const float* br = rel_bias + size_t(dist) * d_rel;
  float acc = 0.f;
  for (int m = 0; m < d_rel; ++m) acc = __fmaf_rn(__ldg(br + m), sr[m], acc);
  out[(size_t(r) * nh + h) * rel_extent + dist] = acc;
}

// ---- rel_attention: FlashAttention prefill (tensor cores) + GQA/split float32 decode ------
__device__ __forceinline__ int rel_splits_used(int span, int max_splits) {
  return min(max_splits, max(1, (span + 63) / 64));
}

__device__ __forceinline__ void rel_ldmatrix_x2_trans(uint32_t* r, const void* smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

template <int HD>
__host__ __device__ constexpr int rel_flash_kt() {
  return HD >= 128 ? 32 : 64;
}

template <int HD>
constexpr size_t rel_flash_smem(int d_rel) {
  const int rs_ld = d_rel > 0 ? ((d_rel + 3) / 4) * 4 : 0;
  return size_t(64 + 2 * rel_flash_kt<HD>()) * (HD + 8) * sizeof(kv16) + size_t(64) * rs_ld * sizeof(float);
}

template <int NT>
__device__ __forceinline__ void rel_flash_state(bool store, float* state, int nh, int h, int wrow0, int rows, int g,
                                                int t, float (&o)[NT][4], float (&m)[2], float (&l)[2]) {
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = wrow0 + g + 8 * half;
    if (row >= rows) continue;
    float* p = state + (size_t(row) * nh + h) * kFlashStateFloats(NT * 8);
    if (store) {
      if (t == 0) p[0] = m[half];
      p[1 + t] = l[half];
    } else {
      m[half] = p[0];
      l[half] = p[1 + t];
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

// Prefill relative-position-bias attention on tensor cores (mma.m16n8k16 binary16 in, fp32 accumulate).
// Supports linear and SWA ring caches (kv.slot_of), binary16 and FP8 caches, windowed prefill (RelWindow),
// exact fp32 relative-bias dot products, virtual 0 sink (m=0, l=(t==0?1:0)), and output sigmoid gate.
template <int HD, bool FP8>
__global__ void __launch_bounds__(128)
    k_rel_flash_prefill(const float* __restrict__ q, const float* __restrict__ r_q, KvRef kv,
                        const float* __restrict__ rel_bias, const float* __restrict__ sink, int ld_sink,
                        const float* __restrict__ gate, int ld_gate, int nh, int nkv, int d_rel, int rel_extent,
                        int window, int rows, float scale, const DevParams* P, int pos_add, int k_lo, int k_hi,
                        int first_win, int last_win, float* __restrict__ state, Out out, int ld_out) {
  pdl_trigger();
  constexpr int KT = rel_flash_kt<HD>(), LD = HD + 8, NT = HD / 8, ST = KT / 8;
  constexpr float kLog2e = 1.4426950408889634f;
  const int rs_ld = d_rel > 0 ? ((d_rel + 3) / 4) * 4 : 0;

  extern __shared__ __align__(16) unsigned char sm_raw[];
  kv16(*Qs)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw);
  kv16(*Ks)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw + size_t(64) * LD * sizeof(kv16));
  kv16(*Vs)[LD] = reinterpret_cast<kv16(*)[LD]>(sm_raw + size_t(64 + KT) * LD * sizeof(kv16));
  float* Rs = reinterpret_cast<float*>(sm_raw + size_t(64 + 2 * KT) * LD * sizeof(kv16));

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
  const bool use_rel = d_rel > 0 && rel_extent > 0 && rel_bias != nullptr;
  if (use_rel) {
    for (int i = tid; i < 64 * d_rel; i += 128) {
      const int r = i / d_rel, dm = i - r * d_rel;
      Rs[r * rs_ld + dm] = (r0 + r < rows) ? r_q[(size_t(r0 + r) * nh + h) * d_rel + dm] : 0.f;
    }
  }

  const int wrow0 = r0 + warp * 16;
  const int last_row = min(rows, r0 + 64) - 1;
  const int w_first = window > 0 ? max(0, base + r0 - window + 1) : 0;
  const int kstart = max(k_lo, w_first);
  const int kwin = min(base + last_row + 1, k_hi);

  float o[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  // Virtual 0 sink: logit 0 (0 in log2 units) with value 0 -> exp2(0 - 0) = 1 summed across the 4 lanes (t==0 holds 1).
  float m[2] = {0.f, 0.f};
  float l[2] = {t == 0 ? 1.f : 0.f, t == 0 ? 1.f : 0.f};
  if (state && !first_win) rel_flash_state<NT>(false, state, nh, h, wrow0, rows, g, t, o, m, l);

  float sv_half[2];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int row = wrow0 + g + 8 * half;
    sv_half[half] = row < rows ? sink[size_t(row) * ld_sink + h] : 0.f;
  }
  const bool rb_aligned16 = use_rel && d_rel == 16 && (reinterpret_cast<uintptr_t>(rel_bias) & 15u) == 0;

  for (int k0 = kstart; k0 < kwin; k0 += KT) {
    __syncthreads();
    if constexpr (FP8) {
      for (int i = tid; i < KT * HD / 8; i += 128) {
        const int r = i / (HD / 8), c = 8 * (i % (HD / 8)), pos = k0 + r;
        uint4 kb = make_uint4(0, 0, 0, 0), vb = kb;
        if (pos < kwin) {
          const int cell = kv.slot_of(pos);
          const size_t at = (size_t(cell) * nkv + kvh) * HD + c;
          const uint2 kq = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(kv.k) + at);
          const uint2 vq = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(kv.v) + at);
          const float ksc = kv.kscales(cell)[size_t(cell) * nkv + kvh];
          const float vsc = kv.vscales(cell)[size_t(cell) * nkv + kvh];
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
        const int cell = ok ? kv.slot_of(pos) : 0;
        const size_t at = (size_t(cell) * nkv + kvh) * HD + c;
        cp_async16(&Ks[r][c], Kc + at, ok);
        cp_async16(&Vs[r][c], Vc + at, ok);
      }
      cp_async_commit();
      cp_async_wait<0>();
    }
    __syncthreads();
    if (wrow0 >= rows || base + wrow0 + 15 < k0 || (window > 0 && k0 + KT - 1 < base + wrow0 - window + 1)) continue;

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

#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = wrow0 + g + 8 * half, qpos = base + row;
      const int row_first = window > 0 ? max(0, qpos - window + 1) : 0;
      const float sv = sv_half[half];
      float4 rq0, rq1, rq2, rq3;
      const float* rq_ptr = use_rel ? &Rs[(warp * 16 + g + 8 * half) * rs_ld] : nullptr;
      if (rb_aligned16 && row < rows) {
        rq0 = *reinterpret_cast<const float4*>(rq_ptr);
        rq1 = *reinterpret_cast<const float4*>(rq_ptr + 4);
        rq2 = *reinterpret_cast<const float4*>(rq_ptr + 8);
        rq3 = *reinterpret_cast<const float4*>(rq_ptr + 12);
      }
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < ST; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int key = k0 + j * 8 + 2 * t + e;
          float& v = s[j][2 * half + e];
          if (row < rows && key >= row_first && key <= qpos && key < kwin) {
            float rb = 0.f;
            if (use_rel) {
              const int dist = qpos - key;
              if (dist >= 0 && dist < rel_extent) {
                const float* br = rel_bias + size_t(dist) * d_rel;
                if (rb_aligned16) {
                  const float4 b0 = __ldg(reinterpret_cast<const float4*>(br));
                  const float4 b1 = __ldg(reinterpret_cast<const float4*>(br + 4));
                  const float4 b2 = __ldg(reinterpret_cast<const float4*>(br + 8));
                  const float4 b3 = __ldg(reinterpret_cast<const float4*>(br + 12));
                  rb = __fmaf_rn(rq0.x, b0.x, __fmaf_rn(rq0.y, b0.y, __fmaf_rn(rq0.z, b0.z, __fmul_rn(rq0.w, b0.w))));
                  rb = __fmaf_rn(rq1.x, b1.x, __fmaf_rn(rq1.y, b1.y, __fmaf_rn(rq1.z, b1.z, __fmaf_rn(rq1.w, b1.w, rb))));
                  rb = __fmaf_rn(rq2.x, b2.x, __fmaf_rn(rq2.y, b2.y, __fmaf_rn(rq2.z, b2.z, __fmaf_rn(rq2.w, b2.w, rb))));
                  rb = __fmaf_rn(rq3.x, b3.x, __fmaf_rn(rq3.y, b3.y, __fmaf_rn(rq3.z, b3.z, __fmaf_rn(rq3.w, b3.w, rb))));
                } else {
                  for (int dm = 0; dm < d_rel; ++dm) rb = __fmaf_rn(rq_ptr[dm], __ldg(br + dm), rb);
                }
              }
            }
            v = __fsub_rn(__fmaf_rn(v, scale, rb), sv) * kLog2e;
          } else {
            v = -INFINITY;
          }
          mx = fmaxf(mx, v);
        }
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      const float mnew = fmaxf(m[half], mx);
      if (mnew > m[half]) {
        const float alpha = exp2f(m[half] - mnew);
        m[half] = mnew;
        l[half] *= alpha;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          o[n][2 * half] *= alpha;
          o[n][2 * half + 1] *= alpha;
        }
      }
#pragma unroll
      for (int j = 0; j < ST; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float& v = s[j][2 * half + e];
          v = v == -INFINITY ? 0.f : exp2f(v - mnew);
          l[half] += v;
        }
    }

#pragma unroll
    for (int kk = 0; kk < KT; kk += 16) {
      const int j = kk / 8;
      const uint32_t a[4] = {pack_f16(s[j][0], s[j][1]), pack_f16(s[j][2], s[j][3]),
                             pack_f16(s[j + 1][0], s[j + 1][1]), pack_f16(s[j + 1][2], s[j + 1][3])};
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        uint32_t b[2];
        rel_ldmatrix_x2_trans(b, &Vs[kk + (lane & 15)][n * 8]);
        mma_f16(o[n], a, b);
      }
    }
  }

  if (state && !last_win) {
    rel_flash_state<NT>(true, state, nh, h, wrow0, rows, g, t, o, m, l);
    return;
  }
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    float sum = l[half];
    sum += __shfl_xor_sync(0xffffffffu, sum, 1);
    sum += __shfl_xor_sync(0xffffffffu, sum, 2);
    const int row = wrow0 + g + 8 * half;
    if (row >= rows) continue;
    const float inv = sum > 0.f ? 1.f / sum : 0.f;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int d = n * 8 + 2 * t + e;
        float v = o[n][2 * half + e] * inv;
        if (gate) v *= sigm(gate[size_t(row) * ld_gate + h * HD + d]);
        put(out, size_t(row) * ld_out + h * HD + d, size_t(row) * nh * HD + h * HD + d, v);
      }
  }
}

template <int DPL, bool FP8>
__device__ __forceinline__ void rel_load_kv(const KvRef& kv, int cell, int kvh, int nkv, int HD, int lane,
                                            float (&kf)[DPL], float (&vf)[DPL]) {
  const size_t at = (size_t(cell) * nkv + kvh) * HD + lane * DPL;
  if constexpr (FP8) {
    const float ksc = kv.kscales(cell)[size_t(cell) * nkv + kvh];
    const float vsc = kv.vscales(cell)[size_t(cell) * nkv + kvh];
    const uint8_t* kp = static_cast<const uint8_t*>(kv.k) + at;
    const uint8_t* vp = static_cast<const uint8_t*>(kv.v) + at;
    if constexpr (DPL == 4) {
      e4m3x4_to_float(*reinterpret_cast<const uint32_t*>(kp), kf);
      e4m3x4_to_float(*reinterpret_cast<const uint32_t*>(vp), vf);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        kf[e] *= ksc;
        vf[e] *= vsc;
      }
    } else if constexpr (DPL == 8) {
      const uint2 kq = *reinterpret_cast<const uint2*>(kp);
      const uint2 vq = *reinterpret_cast<const uint2*>(vp);
      e4m3x4_to_float(kq.x, kf);
      e4m3x4_to_float(kq.y, kf + 4);
      e4m3x4_to_float(vq.x, vf);
      e4m3x4_to_float(vq.y, vf + 4);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        kf[e] *= ksc;
        vf[e] *= vsc;
      }
    } else {
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf[e] = e4m3_to_float(kp[e]) * ksc;
        vf[e] = e4m3_to_float(vp[e]) * vsc;
      }
    }
  } else {
    const kv16* kp = static_cast<const kv16*>(kv.k) + at;
    const kv16* vp = static_cast<const kv16*>(kv.v) + at;
    if constexpr (DPL == 4) {
      const uint2 kr = *reinterpret_cast<const uint2*>(kp);
      const uint2 vr = *reinterpret_cast<const uint2*>(vp);
      const float2 k0 = __half22float2(*reinterpret_cast<const __half2*>(&kr.x));
      const float2 k1 = __half22float2(*reinterpret_cast<const __half2*>(&kr.y));
      const float2 v0 = __half22float2(*reinterpret_cast<const __half2*>(&vr.x));
      const float2 v1 = __half22float2(*reinterpret_cast<const __half2*>(&vr.y));
      kf[0] = k0.x; kf[1] = k0.y; kf[2] = k1.x; kf[3] = k1.y;
      vf[0] = v0.x; vf[1] = v0.y; vf[2] = v1.x; vf[3] = v1.y;
    } else if constexpr (DPL == 8) {
      const uint4 kr = *reinterpret_cast<const uint4*>(kp);
      const uint4 vr = *reinterpret_cast<const uint4*>(vp);
      const uint32_t kwords[4] = {kr.x, kr.y, kr.z, kr.w};
      const uint32_t vwords[4] = {vr.x, vr.y, vr.z, vr.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float2 kf2 = __half22float2(*reinterpret_cast<const __half2*>(&kwords[q]));
        const float2 vf2 = __half22float2(*reinterpret_cast<const __half2*>(&vwords[q]));
        kf[2 * q] = kf2.x; kf[2 * q + 1] = kf2.y;
        vf[2 * q] = vf2.x; vf[2 * q + 1] = vf2.y;
      }
    } else if constexpr (DPL == 2) {
      const float2 k0 = __half22float2(*reinterpret_cast<const __half2*>(kp));
      const float2 v0 = __half22float2(*reinterpret_cast<const __half2*>(vp));
      kf[0] = k0.x; kf[1] = k0.y;
      vf[0] = v0.x; vf[1] = v0.y;
    } else {
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf[e] = __half2float(kp[e]);
        vf[e] = __half2float(vp[e]);
      }
    }
  }
}

// GQA-grouped exact float32 decode/verify kernel: 1 row x G query heads per block (100% row-count independent).
// Reads K, V, and rel_bias once per KV head and shares them across all G query heads.
template <int HD, int G, bool FP8>
__global__ void __launch_bounds__(128)
    k_rel_attention_gqa(const float* __restrict__ q, const float* __restrict__ r_q, KvRef kv,
                        const float* __restrict__ rel_bias, const float* __restrict__ sink, int ld_sink, int nh,
                        int nkv, int d_rel, int rel_extent, int window, float scale, const DevParams* P,
                        float* __restrict__ part, int pos_add, int splits, int k_lo, int k_hi) {
  pdl_trigger();
  constexpr int DPL = HD / 32;
  const int r = blockIdx.x, kvh = blockIdx.y, sp = blockIdx.z;
  const int p = P->position + pos_add + r;
  const int w_first = window > 0 ? max(0, p - window + 1) : 0;
  const int first = max(w_first, k_lo);
  const int total = min(p + 1, k_hi);
  const int span = max(0, total - first);
  const int used = span > 0 ? rel_splits_used(span, splits) : 0;
  if (sp >= used) return;

  const int chunk = (span + used - 1) / used;
  const int ks = min(total, first + sp * chunk);
  const int ke = min(total, ks + chunk);

  __shared__ float s_m[4][G], s_l[4][G];
  __shared__ float s_acc[4][G][HD];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int h0 = kvh * G;

  float qv[G][DPL], acc[G][DPL], m[G], l[G], sink_v[G], rv_lane[G];
#pragma unroll
  for (int hi = 0; hi < G; ++hi) {
    const int h = h0 + hi;
    m[hi] = -INFINITY;
    l[hi] = 0.f;
    sink_v[hi] = sink[size_t(r) * ld_sink + h];
    rv_lane[hi] = lane < d_rel ? r_q[(size_t(r) * nh + h) * d_rel + lane] : 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      acc[hi][e] = 0.f;
      qv[hi][e] = q[(size_t(r) * nh + h) * HD + lane * DPL + e];
    }
  }

  for (int j = ks < ke ? ks + ((warp - ks) & 3) : ke; j < ke; j += 4) {
    const int cell = kv.slot_of(j);
    float kf[DPL], vf[DPL];
    rel_load_kv<DPL, FP8>(kv, cell, kvh, nkv, HD, lane, kf, vf);
    const int dist = p - j;
    const bool has_rb = dist >= 0 && dist < rel_extent;
    const float* br = has_rb ? (rel_bias + size_t(dist) * d_rel) : nullptr;
    const float br_lane = (has_rb && d_rel <= 32 && lane < d_rel) ? __ldg(br + lane) : 0.f;
#pragma unroll
    for (int hi = 0; hi < G; ++hi) {
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) d = __fmaf_rn(qv[hi][e], kf[e], d);
      float rb = 0.f;
      if (has_rb) {
        if (d_rel <= 32) {
          if (lane < d_rel) rb = __fmul_rn(br_lane, rv_lane[hi]);
        } else {
          const float* rr = r_q + (size_t(r) * nh + h0 + hi) * d_rel;
          for (int di = lane; di < d_rel; di += 32) rb = __fmaf_rn(__ldg(br + di), rr[di], rb);
        }
      }
      const float s = __fsub_rn(__fmaf_rn(warp_sum(d), scale, warp_sum(rb)), sink_v[hi]);
      const float nm = fmaxf(m[hi], s);
      const float ew = m[hi] == -INFINITY ? 0.f : __expf(__fsub_rn(m[hi], nm));
      const float w = __expf(__fsub_rn(s, nm));
      l[hi] = __fmaf_rn(l[hi], ew, w);
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[hi][e] = __fmaf_rn(w, vf[e], __fmul_rn(acc[hi][e], ew));
      m[hi] = nm;
    }
  }

#pragma unroll
  for (int hi = 0; hi < G; ++hi) {
    if (lane == 0) {
      s_m[warp][hi] = m[hi];
      s_l[warp][hi] = l[hi];
    }
#pragma unroll
    for (int e = 0; e < DPL; ++e) s_acc[warp][hi][lane * DPL + e] = acc[hi][e];
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int hi = 0; hi < G; ++hi) {
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < 4; ++w) M = fmaxf(M, s_m[w][hi]);
      float L = 0.f;
      float out_v[DPL];
#pragma unroll
      for (int e = 0; e < DPL; ++e) out_v[e] = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          if (s_m[w][hi] == -INFINITY) continue;
          const float f = __expf(__fsub_rn(s_m[w][hi], M));
          L = __fmaf_rn(s_l[w][hi], f, L);
#pragma unroll
          for (int e = 0; e < DPL; ++e) out_v[e] = __fmaf_rn(s_acc[w][hi][lane * DPL + e], f, out_v[e]);
        }
      }
      float* dst = part + ((size_t(r) * nh + h0 + hi) * splits + sp) * (HD + 2);
      if (lane == 0) {
        dst[0] = M;
        dst[1] = L;
      }
#pragma unroll
      for (int e = 0; e < DPL; ++e) dst[2 + lane * DPL + e] = out_v[e];
    }
  }
}

template <int HD, int RT>
__global__ void __launch_bounds__(128) k_rel_attention_split(const float* __restrict__ q, const float* __restrict__ r_q,
                                                             KvRef kv, const float* __restrict__ rel_bias,
                                                             const float* __restrict__ sink, int ld_sink, int nh,
                                                             int nkv, int d_rel, int rel_extent, int window, int rows,
                                                             float scale, const DevParams* P, float* __restrict__ part,
                                                             int pos_add, int splits, int k_lo, int k_hi) {
  pdl_trigger();
  constexpr int DPL = HD / 32;
  __shared__ float s_m[4][RT], s_l[4][RT];
  __shared__ float s_acc[4][RT][HD];
  const int h = blockIdx.x, sp = blockIdx.y, r0 = blockIdx.z * RT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kvh = h / (nh / nkv);
  const int base = P->position + pos_add;
  const int nrows = min(RT, rows - r0);

  int p[RT], ks[RT], ke[RT], lo = INT_MAX, hi = 0;
  float sink_v[RT];
  bool any_used = false;
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    p[i] = base + r0 + i;
    const int w_first = window > 0 ? max(0, p[i] - window + 1) : 0;
    const int first = max(w_first, k_lo);
    const int total = min(p[i] + 1, k_hi);
    const int span = max(0, total - first);
    const int used = span > 0 ? rel_splits_used(span, splits) : 0;
    if (i < nrows && sp < used) {
      any_used = true;
      const int chunk = (span + used - 1) / used;
      ks[i] = min(total, first + sp * chunk);
      ke[i] = min(total, ks[i] + chunk);
      if (ks[i] < ke[i]) {
        lo = min(lo, ks[i]);
        hi = max(hi, ke[i]);
      }
      sink_v[i] = sink[size_t(r0 + i) * ld_sink + h];
    } else {
      ks[i] = ke[i] = 0;
      sink_v[i] = 0.f;
    }
  }
  if (!any_used) return;

  float qv[RT][DPL], acc[RT][DPL], m[RT], l[RT];
  float rv_lane[RT];  // holds r_q[row, h, lane] when d_rel <= 32
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.f;
    rv_lane[i] = (i < nrows && lane < d_rel) ? r_q[(size_t(r0 + i) * nh + h) * d_rel + lane] : 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      acc[i][e] = 0.f;
      qv[i][e] = i < nrows ? q[(size_t(r0 + i) * nh + h) * HD + lane * DPL + e] : 0.f;
    }
  }

  const bool is_fp8 = kv.fp8();
  for (int j = lo < hi ? lo + ((warp - lo) & 3) : hi; j < hi; j += 4) {
    const int cell = kv.slot_of(j);
    float kf[DPL], vf[DPL];
    if (is_fp8) rel_load_kv<DPL, true>(kv, cell, kvh, nkv, HD, lane, kf, vf);
    else rel_load_kv<DPL, false>(kv, cell, kvh, nkv, HD, lane, kf, vf);

#pragma unroll
    for (int i = 0; i < RT; ++i) {
      if (i >= nrows) break;
      if (j < ks[i] || j >= ke[i]) continue;
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) d = __fmaf_rn(qv[i][e], kf[e], d);
      float rb = 0.f;
      const int dist = p[i] - j;
      if (dist >= 0 && dist < rel_extent) {
        const float* br = rel_bias + size_t(dist) * d_rel;
        if (d_rel <= 32) {
          if (lane < d_rel) rb = __fmul_rn(__ldg(br + lane), rv_lane[i]);
        } else {
          const float* rr = r_q + (size_t(r0 + i) * nh + h) * d_rel;
          for (int di = lane; di < d_rel; di += 32) rb = __fmaf_rn(__ldg(br + di), rr[di], rb);
        }
      }
      const float s = __fsub_rn(__fmaf_rn(warp_sum(d), scale, warp_sum(rb)), sink_v[i]);
      const float nm = fmaxf(m[i], s);
      const float ew = m[i] == -INFINITY ? 0.f : __expf(__fsub_rn(m[i], nm));
      const float w = __expf(__fsub_rn(s, nm));
      l[i] = __fmaf_rn(l[i], ew, w);
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[i][e] = __fmaf_rn(w, vf[e], __fmul_rn(acc[i][e], ew));
      m[i] = nm;
    }
  }

  // Fixed warp order 0..3 -> split partial.
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    if (i >= nrows) break;
    if (lane == 0) {
      s_m[warp][i] = m[i];
      s_l[warp][i] = l[i];
    }
#pragma unroll
    for (int e = 0; e < DPL; ++e) s_acc[warp][i][lane * DPL + e] = acc[i][e];
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      if (i >= nrows) break;
      const int w_first = window > 0 ? max(0, p[i] - window + 1) : 0;
      const int first = max(w_first, k_lo);
      const int total = min(p[i] + 1, k_hi);
      const int span = max(0, total - first);
      const int used = span > 0 ? rel_splits_used(span, splits) : 0;
      if (sp >= used) continue;
      float M = -INFINITY;
#pragma unroll
      for (int w = 0; w < 4; ++w) M = fmaxf(M, s_m[w][i]);
      float L = 0.f;
      float out_v[DPL];
#pragma unroll
      for (int e = 0; e < DPL; ++e) out_v[e] = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          if (s_m[w][i] == -INFINITY) continue;
          const float f = __expf(__fsub_rn(s_m[w][i], M));
          L = __fmaf_rn(s_l[w][i], f, L);
#pragma unroll
          for (int e = 0; e < DPL; ++e) out_v[e] = __fmaf_rn(s_acc[w][i][lane * DPL + e], f, out_v[e]);
        }
      }
      float* dst = part + ((size_t(r0 + i) * nh + h) * splits + sp) * (HD + 2);
      if (lane == 0) {
        dst[0] = M;
        dst[1] = L;
      }
#pragma unroll
      for (int e = 0; e < DPL; ++e) dst[2 + lane * DPL + e] = out_v[e];
    }
  }
}

__global__ void k_rel_attention_combine(const float* __restrict__ part, const float* __restrict__ gate, int ld_gate,
                                        int nh, int hd, int window, int splits, const DevParams* P, int pos_add,
                                        Out out, int ld_out) {
  pdl_trigger();
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int p = P->position + pos_add + r;
  const int first = window > 0 ? max(0, p - window + 1) : 0;
  const int used = rel_splits_used(max(0, p + 1 - first), splits);
  const float* pr = part + (size_t(r) * nh + h) * splits * (hd + 2);
  const float g = gate ? gate[size_t(r) * ld_gate + h * hd + d] : 0.f;

  // Virtual sink sits at logit 0 with zero value contribution.
  float M = 0.f;
  for (int s = 0; s < used; ++s) M = fmaxf(M, pr[s * (hd + 2)]);
  float L = __expf(__fsub_rn(0.f, M));
  float A = 0.f;
  for (int s = 0; s < used; ++s) {
    const float ms = pr[s * (hd + 2)];
    if (ms == -INFINITY) continue;
    const float f = __expf(__fsub_rn(ms, M));
    L = __fmaf_rn(pr[s * (hd + 2) + 1], f, L);
    A = __fmaf_rn(pr[s * (hd + 2) + 2 + d], f, A);
  }
  float o = L > 0.f ? A / L : 0.f;
  if (gate) o *= sigm(g);
  put(out, size_t(r) * ld_out + h * hd + d, size_t(r) * nh * hd + h * hd + d, o);
}

__global__ void k_rel_attention_combine_win(const float* __restrict__ part, const float* __restrict__ gate, int ld_gate,
                                            int nh, int hd, int window, int splits, const DevParams* P, int pos_add,
                                            int k_lo, int k_hi, int first_win, int last_win, float* __restrict__ state,
                                            Out out, int ld_out) {
  pdl_trigger();
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int p = P->position + pos_add + r;
  const int w_first = window > 0 ? max(0, p - window + 1) : 0;
  const int first = max(w_first, k_lo);
  const int total = min(p + 1, k_hi);
  const int span = max(0, total - first);
  const int used = span > 0 ? rel_splits_used(span, splits) : 0;
  const float* pr = part + (size_t(r) * nh + h) * splits * (hd + 2);
  float* st = state + (size_t(r) * nh + h) * kFlashStateFloats(hd);

  float M = first_win ? -INFINITY : st[0];
  for (int s = 0; s < used; ++s) M = fmaxf(M, pr[s * (hd + 2)]);
  if (last_win) M = fmaxf(M, 0.f);  // virtual sink at logit 0

  float L = last_win ? __expf(__fsub_rn(0.f, M)) : 0.f;
  float A = 0.f;
  if (!first_win && st[0] != -INFINITY) {
    const float fp = __expf(__fsub_rn(st[0], M));
    L = __fmaf_rn(st[1], fp, L);
    A = __fmaf_rn(st[2 + d], fp, A);
  }
  for (int s = 0; s < used; ++s) {
    const float ms = pr[s * (hd + 2)];
    if (ms == -INFINITY) continue;
    const float f = __expf(__fsub_rn(ms, M));
    L = __fmaf_rn(pr[s * (hd + 2) + 1], f, L);
    A = __fmaf_rn(pr[s * (hd + 2) + 2 + d], f, A);
  }

  if (!last_win) {
    if (d == 0) {
      st[0] = M;
      st[1] = L;
    }
    st[2 + d] = A;
    return;
  }
  float o = L > 0.f ? A / L : 0.f;
  if (gate) o *= sigm(gate[size_t(r) * ld_gate + h * hd + d]);
  put(out, size_t(r) * ld_out + h * hd + d, size_t(r) * nh * hd + h * hd + d, o);
}

template <int HD, bool FP8>
bool launch_rel_flash(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink, int ld_sink,
                      const float* gate, int ld_gate, int nh, int nkv, int d_rel, int rel_extent, int window, int rows,
                      float scale, const DevParams* P, float* out, int ld_out, const Launch& L, int pos_add,
                      const RelWindow& win) {
  if (d_rel > 64) return false;
  const size_t smem = rel_flash_smem<HD>(d_rel);
  static const bool ready = set_dynamic_smem(k_rel_flash_prefill<HD, FP8>, rel_flash_smem<HD>(64));
  (void)ready;
  const int k_lo = win.state ? win.k_lo : 0;
  const int k_hi = win.state ? win.k_hi : INT_MAX;
  const int first_win = win.state ? int(win.first) : 1;
  const int last_win = win.state ? int(win.last) : 1;
  const Out dst = (!win.state || win.last) ? (out ? Out{out, nullptr, nullptr} : split_out(L, rows, nh * HD, false))
                                           : Out{nullptr, nullptr, nullptr};
  k_rel_flash_prefill<HD, FP8><<<dim3(nh, ceil_div(rows, 64)), 128, smem, L.st>>>(
      q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window, rows, scale, P, pos_add,
      k_lo, k_hi, first_win, last_win, win.state, dst, ld_out);
  return true;
}

template <int HD>
bool launch_rel_gqa(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink, int ld_sink,
                    int nh, int nkv, int d_rel, int rel_extent, int window, int rows, float scale, const DevParams* P,
                    float* partial, int eff_splits, cudaStream_t st, int pos_add, int k_lo, int k_hi) {
  if (nkv <= 0 || nh % nkv != 0) return false;
  const int G = nh / nkv;
  const dim3 grid(rows, nkv, eff_splits);
#define FLYX_REL_GQA(G_)                                                                                               \
  if (G == (G_)) {                                                                                                     \
    if (kv.fp8())                                                                                                      \
      k_rel_attention_gqa<HD, G_, true><<<grid, 128, 0, st>>>(q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel,       \
                                                              rel_extent, window, scale, P, partial, pos_add,          \
                                                              eff_splits, k_lo, k_hi);                                 \
    else                                                                                                               \
      k_rel_attention_gqa<HD, G_, false><<<grid, 128, 0, st>>>(q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel,      \
                                                               rel_extent, window, scale, P, partial, pos_add,         \
                                                               eff_splits, k_lo, k_hi);                                \
    return true;                                                                                                       \
  }
  FLYX_REL_GQA(1) FLYX_REL_GQA(2) FLYX_REL_GQA(3) FLYX_REL_GQA(4)
  if constexpr (HD <= 128) {
    FLYX_REL_GQA(6) FLYX_REL_GQA(8)
  }
#undef FLYX_REL_GQA
  return false;
}

template <int HD>
void dispatch_rel_attention(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink,
                            int ld_sink, const float* gate, int ld_gate, int nh, int nkv, int d_rel, int rel_extent,
                            int window, int rows, float scale, const DevParams* P, float* out, int ld_out,
                            float* partial, int splits, const Launch& L, int pos_add, const RelWindow& win) {
  // Prefill (more causal rows than a round): tensor-core FlashAttention unless kAttnSplit is forced.
  if (rows > FLYX_MAX_ROWS && !L.round_rows && opts(L).attention != kAttnSplit) {
    const bool ok = kv.fp8()
                        ? launch_rel_flash<HD, true>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel,
                                                     rel_extent, window, rows, scale, P, out, ld_out, L, pos_add, win)
                        : launch_rel_flash<HD, false>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel,
                                                      rel_extent, window, rows, scale, P, out, ld_out, L, pos_add, win);
    if (ok) return;
  }
  const int eff_splits = std::min(std::max(1, splits), kAttnCoresSplits);
  const int k_lo = win.state ? win.k_lo : 0;
  const int k_hi = win.state ? win.k_hi : INT_MAX;
  const bool gqa_ok = opts(L).attention != kAttnPerHead &&
                      launch_rel_gqa<HD>(q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel, rel_extent, window, rows,
                                         scale, P, partial, eff_splits, L.st, pos_add, k_lo, k_hi);
  if (!gqa_ok) {
    if (rows == 1) {
      k_rel_attention_split<HD, 1><<<dim3(nh, eff_splits, 1), 128, 0, L.st>>>(
          q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel, rel_extent, window, rows, scale, P, partial, pos_add,
          eff_splits, k_lo, k_hi);
    } else {
      k_rel_attention_split<HD, 4><<<dim3(nh, eff_splits, ceil_div(rows, 4)), 128, 0, L.st>>>(
          q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel, rel_extent, window, rows, scale, P, partial, pos_add,
          eff_splits, k_lo, k_hi);
    }
  }
  const Out dst = (!win.state || win.last) ? (out ? Out{out, nullptr, nullptr} : split_out(L, rows, nh * HD, false))
                                           : Out{nullptr, nullptr, nullptr};
  if (win.state) {
    k_rel_attention_combine_win<<<dim3(nh, rows), HD, 0, L.st>>>(partial, gate, ld_gate, nh, HD, window, eff_splits, P,
                                                                 pos_add, k_lo, k_hi, int(win.first), int(win.last),
                                                                 win.state, dst, ld_out);
  } else {
    k_rel_attention_combine<<<dim3(nh, rows), HD, 0, L.st>>>(partial, gate, ld_gate, nh, HD, window, eff_splits, P,
                                                             pos_add, dst, ld_out);
  }
}

}  // namespace

void softcap_rows(float* logits, int ld, int V, int rows, float cap, cudaStream_t st) {
  if (!(cap > 0.f) || V <= 0 || rows <= 0) return;
  k_softcap_rows<<<dim3(ceil_div(V, 256), rows), 256, 0, st>>>(logits, ld, V, cap);
}

void attn_res_mix(const float* bank, int n_blocks, const float* cur, const float* queries, int n_queries, int rows,
                  int H, float* out, cudaStream_t st) {
  if (n_blocks < 0 || n_blocks >= kMaxResStates)
    throw std::runtime_error("attn_res_mix: bank depth exceeds kMaxResStates");
  for (int q0 = 0; q0 < n_queries; q0 += 4) {
    const int nq = std::min(4, n_queries - q0);
    k_attn_res_mix<<<rows, 256, 0, st>>>(bank, n_blocks, cur, queries, q0, nq, rows, H, out);
  }
}

void loop_ctrl_prep(const float* cur, const float* msg, int rows, int H, float* cat_out, cudaStream_t st) {
  k_loop_ctrl_prep<<<dim3(rows, 2), 512, 0, st>>>(cur, msg, H, cat_out);
}

void loop_ctrl_apply(const float* cur, const float* msg, const float* gain, int ld_gain, const float* forget,
                     int ld_forget, int rows, int H, float* out, cudaStream_t st) {
  k_loop_ctrl_apply<<<dim3(ceil_div(H, 256), rows), 256, 0, st>>>(cur, msg, gain, ld_gain, forget, ld_forget, H, out);
}

void loop_control(const LoopGate* g, int n, int rows, int H, int rank, float* scratch, cudaStream_t st) {
  if (n < 1 || n > kLoopGates) throw std::invalid_argument("loop_control: 1 to kLoopGates controllers");
  if (rows <= 0 || H <= 0 || rank <= 0) return;
  LoopGates G{};
  for (int c = 0; c < n; ++c) G.g[c] = g[c];
  const int nr = loop_rows(rows);
  const dim3 gate_grid(rank, ceil_div(rows, nr), n), apply_grid(ceil_div(H, kLoopBlock / 32), ceil_div(rows, nr), n);
#define FLYX_LOOP_LAUNCH(NR)                                                                case NR:                                                                                    k_loop_gate<NR><<<gate_grid, kLoopBlock, 0, st>>>(G, H, rank, rows, scratch);             k_loop_apply<NR><<<apply_grid, kLoopBlock, 0, st>>>(G, H, rank, rows, scratch);           break;
  switch (nr) {
    FLYX_LOOP_LAUNCH(1)
    FLYX_LOOP_LAUNCH(2)
    FLYX_LOOP_LAUNCH(4)
    FLYX_LOOP_LAUNCH(8)
  }
#undef FLYX_LOOP_LAUNCH
}

void linear_f32(const float* x, int ldx, RoundWeight W, int N, int K, float* y, int ldy, int rows, bool accumulate,
                cudaStream_t st) {
  if (rows <= 0 || N <= 0) return;
  const int nr = loop_rows(rows), acc = accumulate ? 1 : 0;
  const dim3 grid(ceil_div(N, kLoopBlock / 32), ceil_div(rows, nr));
  auto launch = [&](const auto* w) {
    switch (nr) {
      case 1: k_linear_f32<1><<<grid, kLoopBlock, 0, st>>>(x, ldx, w, N, K, y, ldy, rows, acc); break;
      case 2: k_linear_f32<2><<<grid, kLoopBlock, 0, st>>>(x, ldx, w, N, K, y, ldy, rows, acc); break;
      case 4: k_linear_f32<4><<<grid, kLoopBlock, 0, st>>>(x, ldx, w, N, K, y, ldy, rows, acc); break;
      default: k_linear_f32<8><<<grid, kLoopBlock, 0, st>>>(x, ldx, w, N, K, y, ldy, rows, acc); break;
    }
  };
  if (W.b16) launch(W.b16);
  else launch(W.f32);
}

void shortconv_pass(const float* x, int ldx, const float* w, const float* dyn, float* win, size_t win_stride,
                    int in_sel, int next_sel, int skip, float* replay, size_t replay_stride, int replay_sel,
                    int ch_off, int ch_total, int C, int conv, int n_heads, const DevParams* P, int rows, float* out,
                    int ldo, cudaStream_t st, bool accumulate) {
  if (conv < 2 || conv > kMaxShortConv) throw std::runtime_error("shortconv_pass: conv out of range");
  const int tile_rows = std::max(16, conv - 1);
  k_shortconv_pass<<<dim3(ceil_div(C, 256), ceil_div(rows, tile_rows)), 256, 0, st>>>(
      x, ldx, w, dyn, win, win_stride, in_sel, next_sel, skip, replay, replay_stride, replay_sel, ch_off, ch_total, C,
      conv, n_heads, P, rows, tile_rows, out, ldo, int(accumulate));
}

void shortconv_fold(const ShortConvLayer* layers, int n, size_t win_stride, size_t replay_stride, int in_sel,
                    int out_sel, int replay_sel, int channels, int conv, const DevParams* P, int rows,
                    const int* rows_dev, bool skip_if_noop, cudaStream_t st) {
  if (n <= 0 || channels <= 0) return;
  if (conv < 2 || conv > kMaxShortConv) throw std::runtime_error("shortconv_fold: conv out of range");
  k_shortconv_fold<<<dim3(ceil_div(channels, 256), n), 256, 0, st>>>(layers, win_stride, replay_stride, in_sel, out_sel,
                                                                     replay_sel, channels, conv, P, rows, rows_dev,
                                                                     int(skip_if_noop));
}

void rel_attn_prep(const float* q_in, int ldq, const float* k_in, int ldk, const float* v_in, int ldv,
                   const float* r_in, int ldr, const float* q_clip_w, const float* r_clip_w, float qk_clip,
                   float rel_clip, float v_eps, int log_floor, float log_alpha, int nh, int nkv, int hd, int d_rel,
                   int rows, const DevParams* P, int pos_sel, bool is_swa, float* q_out, float* r_out, KvRef kv,
                   cudaStream_t st, int pos_add, int strata_base) {
  const int threads = std::max(32, ((std::max(hd, d_rel) + 31) / 32) * 32);
  const size_t smem = kv.kq ? size_t(2) * size_t(hd) * sizeof(float) : 0;
  if (kv.kq && hd == 256 && kv.bits == 3 && kv.words == 24) {
    k_rel_attn_prep<true><<<dim3(nh + nkv, rows), threads, smem, st>>>(
        q_in, ldq, k_in, ldk, v_in, ldv, r_in, ldr, q_clip_w, r_clip_w, qk_clip, rel_clip, v_eps, log_floor, log_alpha,
        nh, nkv, hd, d_rel, P, pos_sel, int(is_swa), q_out, r_out, kv, pos_add, strata_base);
  } else {
    k_rel_attn_prep<false><<<dim3(nh + nkv, rows), threads, smem, st>>>(
        q_in, ldq, k_in, ldk, v_in, ldv, r_in, ldr, q_clip_w, r_clip_w, qk_clip, rel_clip, v_eps, log_floor, log_alpha,
        nh, nkv, hd, d_rel, P, pos_sel, int(is_swa), q_out, r_out, kv, pos_add, strata_base);
  }
}

void rel_bias_table(const float* r_q, const float* rel_bias, int nh, int d_rel, int rel_extent, int rows, float* out,
                    cudaStream_t st) {
  if (rows <= 0 || nh <= 0 || rel_extent <= 0) return;
  k_rel_bias_table<<<dim3(ceil_div(rel_extent, 128), nh, rows), 128, size_t(d_rel) * sizeof(float), st>>>(
      r_q, rel_bias, nh, d_rel, rel_extent, out);
}

void rel_attention(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink, int ld_sink,
                   const float* gate, int ld_gate, int nh, int nkv, int hd, int d_rel, int rel_extent, int window,
                   int rows, const DevParams* P, float* out, int ld_out, float* partial, int splits, const Launch& L,
                   int pos_add, const RelWindow& win) {
  const float scale = 1.f / std::sqrt(float(hd));
  switch (hd) {
    case 32:
      dispatch_rel_attention<32>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                 rows, scale, P, out, ld_out, partial, splits, L, pos_add, win);
      return;
    case 64:
      dispatch_rel_attention<64>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                 rows, scale, P, out, ld_out, partial, splits, L, pos_add, win);
      return;
    case 128:
      dispatch_rel_attention<128>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                  rows, scale, P, out, ld_out, partial, splits, L, pos_add, win);
      return;
    case 256:
      dispatch_rel_attention<256>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                  rows, scale, P, out, ld_out, partial, splits, L, pos_add, win);
      return;
    default: throw std::runtime_error("rel_attention: unsupported head dim " + std::to_string(hd));
  }
}

}  // namespace flyx::cuda
