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
    block_sum_n(acc);
    if (tid == 0) {
      const float inv = rsqrtf(acc[0] / float(H) + 1e-6f);
      s_inv[m] = inv;
      const float sc = inv * inv_sqrt_h;
      for (int q = 0; q < nq_chunk; ++q) s_dot[q][m] = acc[q + 1] * sc;
    }
    __syncthreads();
  }
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
__global__ void k_rel_attn_prep(const float* __restrict__ q_in, int ldq, const float* __restrict__ k_in, int ldk,
                                const float* __restrict__ v_in, int ldv, const float* __restrict__ r_in, int ldr,
                                const float* __restrict__ q_clip_w, const float* __restrict__ r_clip_w, float qk_clip,
                                float rel_clip, float v_eps, int log_floor, float log_alpha, int nh, int nkv, int hd,
                                int d_rel, const DevParams* P, int pos_sel, int is_swa, float* __restrict__ q_out,
                                float* __restrict__ r_out, KvRef kv, int pos_add) {
  pdl_trigger();
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
}

// ---- rel_attention (row-count independent split-K + virtual 0 sink combine) ----------------
__device__ __forceinline__ int rel_splits_used(int span, int max_splits) {
  return min(max_splits, max(1, (span + 63) / 64));
}

template <int HD, int RT>
__global__ void __launch_bounds__(128) k_rel_attention_split(const float* __restrict__ q, const float* __restrict__ r_q,
                                                             KvRef kv, const float* __restrict__ rel_bias,
                                                             const float* __restrict__ sink, int ld_sink, int nh,
                                                             int nkv, int d_rel, int rel_extent, int window, int rows,
                                                             float scale, const DevParams* P, float* __restrict__ part,
                                                             int pos_add, int splits) {
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
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    p[i] = base + r0 + i;
    const int first = window > 0 ? max(0, p[i] - window + 1) : 0;
    const int total = p[i] + 1;
    const int span = max(0, total - first);
    const int used = rel_splits_used(span, splits);
    if (i < nrows && sp < used) {
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
    const size_t at = (size_t(cell) * nkv + kvh) * HD + lane * DPL;
    float kf[DPL], vf[DPL];
    if (is_fp8) {
      const float ksc = kv.kscales(cell)[size_t(cell) * nkv + kvh];
      const float vsc = kv.vscales(cell)[size_t(cell) * nkv + kvh];
      const uint8_t* kp = static_cast<const uint8_t*>(kv.k) + at;
      const uint8_t* vp = static_cast<const uint8_t*>(kv.v) + at;
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf[e] = e4m3_to_float(kp[e]) * ksc;
        vf[e] = e4m3_to_float(vp[e]) * vsc;
      }
    } else {
      const kv16* kp = static_cast<const kv16*>(kv.k) + at;
      const kv16* vp = static_cast<const kv16*>(kv.v) + at;
#pragma unroll
      for (int e = 0; e < DPL; ++e) {
        kf[e] = __half2float(kp[e]);
        vf[e] = __half2float(vp[e]);
      }
    }

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
      const int total = p[i] + 1;
      const int first = window > 0 ? max(0, p[i] - window + 1) : 0;
      const int used = rel_splits_used(max(0, total - first), splits);
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

template <int HD>
void dispatch_rel_attention(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink,
                            int ld_sink, const float* gate, int ld_gate, int nh, int nkv, int d_rel, int rel_extent,
                            int window, int rows, float scale, const DevParams* P, float* out, int ld_out,
                            float* partial, int splits, const Launch& L, int pos_add) {
  const int eff_splits = std::max(1, splits);
  if (rows == 1) {
    k_rel_attention_split<HD, 1><<<dim3(nh, eff_splits, 1), 128, 0, L.st>>>(
        q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel, rel_extent, window, rows, scale, P, partial, pos_add,
        eff_splits);
  } else {
    k_rel_attention_split<HD, 4><<<dim3(nh, eff_splits, ceil_div(rows, 4)), 128, 0, L.st>>>(
        q, r, kv, rel_bias, sink, ld_sink, nh, nkv, d_rel, rel_extent, window, rows, scale, P, partial, pos_add,
        eff_splits);
  }
  const Out dst = out ? Out{out, nullptr, nullptr} : split_out(L, rows, nh * HD, false);
  k_rel_attention_combine<<<dim3(nh, rows), HD, 0, L.st>>>(partial, gate, ld_gate, nh, HD, window, eff_splits, P,
                                                           pos_add, dst, ld_out);
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
                   cudaStream_t st, int pos_add) {
  const int threads = std::max(32, ((std::max(hd, d_rel) + 31) / 32) * 32);
  k_rel_attn_prep<<<dim3(nh + nkv, rows), threads, 0, st>>>(q_in, ldq, k_in, ldk, v_in, ldv, r_in, ldr, q_clip_w,
                                                            r_clip_w, qk_clip, rel_clip, v_eps, log_floor, log_alpha,
                                                            nh, nkv, hd, d_rel, P, pos_sel, int(is_swa), q_out, r_out,
                                                            kv, pos_add);
}

void rel_attention(const float* q, const float* r, KvRef kv, const float* rel_bias, const float* sink, int ld_sink,
                   const float* gate, int ld_gate, int nh, int nkv, int hd, int d_rel, int rel_extent, int window,
                   int rows, const DevParams* P, float* out, int ld_out, float* partial, int splits, const Launch& L,
                   int pos_add) {
  const float scale = 1.f / std::sqrt(float(hd));
  switch (hd) {
    case 32:
      dispatch_rel_attention<32>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                 rows, scale, P, out, ld_out, partial, splits, L, pos_add);
      return;
    case 64:
      dispatch_rel_attention<64>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                 rows, scale, P, out, ld_out, partial, splits, L, pos_add);
      return;
    case 128:
      dispatch_rel_attention<128>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                  rows, scale, P, out, ld_out, partial, splits, L, pos_add);
      return;
    case 256:
      dispatch_rel_attention<256>(q, r, kv, rel_bias, sink, ld_sink, gate, ld_gate, nh, nkv, d_rel, rel_extent, window,
                                  rows, scale, P, out, ld_out, partial, splits, L, pos_add);
      return;
    default: throw std::runtime_error("rel_attention: unsupported head dim " + std::to_string(hd));
  }
}

}  // namespace flyx::cuda
