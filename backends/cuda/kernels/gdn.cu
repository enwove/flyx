// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gated delta net: causal conv, q/k normalisation and gates, the recurrence, gated norm; one kernel for a round.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include <type_traits>
#include <cooperative_groups.h>

namespace flyx::cuda {
namespace {

// ---- gated delta net ----------------------------------------------------------------------
// The pieces below are shared by the per-step kernels and k_gdn_round, so both
// compute every value with the same operations in the same order.  Products
// that feed a sum are rounded explicitly (__fmul_rn / __fmaf_rn): left to
// contraction, the compiler may fuse differently in two kernels.

// Causal conv (+SiLU) of one channel over up to NR rows from r0.
// x[i] is the input at row r0 - (ck - 1) + i: `mixed`, or before row 0 the
// window w ([ck - 1][C]).  All loads are issued before any use (a loop that
// loads a row after storing the last one waits a round trip per row).  CK:
// the kernel width at compile time (registers instead of local arrays), 0 = ck.
// NR: the most rows (registers: k_gdn_round's 4 rows hold 7 values, not 19).
constexpr int kConvRows = 16;
template <int CK, int NR = kConvRows>
struct GdnConvIn {
  static constexpr int kMaxCk = CK > 0 ? CK : 8;
  float x[NR + kMaxCk - 1], cw[kMaxCk];
  template <class T>
  __device__ __forceinline__ void load(const T* mixed, int ld, const float* conv_w, const float* w, int C, int ck,
                                       int c, int r0, int n) {
    load_before(conv_w, w, C, ck, c, r0);
    load_rows(mixed, ld, ck, c, r0, n);
  }
  // What no kernel of the round writes (the weights, the window before row 0), and the rows from `mixed`: the
  // first may load before waiting for the kernel before (k_gdn_round).
  __device__ __forceinline__ void load_before(const float* conv_w, const float* w, int C, int ck, int c, int r0) {
#pragma unroll
    for (int j = 0; j < kMaxCk; ++j)
      if (j < ck) cw[j] = conv_w[size_t(c) * ck + j];
#pragma unroll
    for (int i = 0; i < kMaxCk - 1; ++i)
      if (i < ck - 1 && r0 - (ck - 1) + i < 0) x[i] = w[size_t(r0 + i) * C + c];
  }
  template <class T>
  __device__ __forceinline__ void load_rows(const T* mixed, int ld, int ck, int c, int r0, int n) {
#pragma unroll
    for (int i = 0; i < NR + kMaxCk - 1; ++i)
      if (i < ck - 1 + n) {
        const int r = r0 - (ck - 1) + i;
        if (r >= 0) x[i] = in_value(mixed + size_t(r) * ld + c);
      }
  }
  // The input and the output at row r0 + r (r a constant after unrolling).
  __device__ __forceinline__ float in(int r, int ck) const { return x[r + ck - 1]; }
  __device__ __forceinline__ float out(int r, int ck) const {
    float acc = __fmul_rn(x[r + ck - 1], cw[ck - 1]);
#pragma unroll
    for (int j = 0; j < kMaxCk - 1; ++j)
      if (j < ck - 1) acc = __fmaf_rn(x[r + j], cw[j], acc);
    return silu(acc);
  }
  // The window after row r0 + n - 1 (its last ck - 1 inputs) into w_out ([ck - 1][C]).
  __device__ __forceinline__ void advance(float* w_out, int C, int ck, int c, int n) const {
#pragma unroll
    for (int j = 0; j < kMaxCk - 1; ++j)
      if (j < ck - 1) {
        float v = 0.f;
#pragma unroll
        for (int i = 0; i < NR + kMaxCk - 1; ++i)
          if (i == n + j) v = x[i];
        w_out[size_t(j) * C + c] = v;
      }
  }
};

// Streaming causal conv (+SiLU) of one channel: keeps only ck weights and the
// (ck - 1)-tap sliding window in registers (7 floats for ck = 4, independent of
// row count), avoiding local-memory frames in k_gdn_conv and k_gdn_round while
// preserving the exact __fmul_rn / __fmaf_rn order of GdnConvIn::out.
template <int CK>
struct GdnConvStream {
  static constexpr int kMaxCk = CK > 0 ? CK : 8;
  float cw[kMaxCk], w[kMaxCk - 1];
  __device__ __forceinline__ void load_before(const float* conv_w, const float* w_cur, int C, int ck, int c) {
    if constexpr (CK > 0) {
#pragma unroll
      for (int j = 0; j < CK; ++j) cw[j] = conv_w[size_t(c) * CK + j];
#pragma unroll
      for (int j = 0; j < CK - 1; ++j) w[j] = w_cur[size_t(j) * C + c];
    } else {
      for (int j = 0; j < ck; ++j) cw[j] = conv_w[size_t(c) * ck + j];
      for (int j = 0; j < ck - 1; ++j) w[j] = w_cur[size_t(j) * C + c];
    }
  }
  __device__ __forceinline__ float step(float x, int ck) {
    if constexpr (CK > 0) {
      float acc = __fmul_rn(x, cw[CK - 1]);
#pragma unroll
      for (int j = 0; j < CK - 1; ++j) acc = __fmaf_rn(w[j], cw[j], acc);
#pragma unroll
      for (int j = 0; j < CK - 2; ++j) w[j] = w[j + 1];
      if constexpr (CK > 1) w[CK - 2] = x;
      return silu(acc);
    } else {
      float acc = __fmul_rn(x, cw[ck - 1]);
      for (int j = 0; j < ck - 1; ++j) acc = __fmaf_rn(w[j], cw[j], acc);
      for (int j = 0; j < ck - 2; ++j) w[j] = w[j + 1];
      if (ck > 1) w[ck - 2] = x;
      return silu(acc);
    }
  }
  __device__ __forceinline__ void store_window(float* w_out, int C, int ck, int c) const {
    if constexpr (CK > 0) {
#pragma unroll
      for (int j = 0; j < CK - 1; ++j) w_out[size_t(j) * C + c] = w[j];
    } else {
      for (int j = 0; j < ck - 1; ++j) w_out[size_t(j) * C + c] = w[j];
    }
  }
};

// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias).
__device__ __forceinline__ void gdn_gates(float b, float a, float A_log, float dt_bias, float* beta, float* g) {
  *beta = sigm(b);
  *g = -__expf(A_log) * softplus(a + dt_bias);
}

// grid (ceil(C / 256), ceil(rows / kConvRows)): one thread per channel and
// block of rows.  The ck - 1 inputs before a block come from the window (first
// block) or from `mixed` (the others: kConvRows >= ck - 1, so only the first
// block reads the window).  The first block's thread also writes the advanced
// window to slot win_out_sel (>= 0; may be win_sel), unless skip says not to.
template <int CK, class TIn = float>
__global__ void k_gdn_conv(const TIn* __restrict__ mixed, int ld_mixed, const float* __restrict__ conv_w,
                           float* __restrict__ win, size_t win_stride, int win_sel, int win_out_sel, int skip,
                           GdnBufs bufs, int bufs_sel, const DevParams* P, int rows, int snap_sel, int snap_row) {
  pdl_trigger();
  if (bufs.slot_stride > 0 && P->stopped) return;  // noop in-flight round must not clobber a prior round's replay slot
  const int C = bufs.d.C(), ck = CK > 0 ? CK : bufs.d.ck;
  const int kd = bufs.d.nk * bufs.d.dk, vd = bufs.d.nv * bufs.d.dv;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int r0 = blockIdx.y * kConvRows, n = min(kConvRows, rows - r0);
  const float* w = win + size_t(slot_of(P, win_sel)) * win_stride;
  const bool advance = win_out_sel >= 0 && blockIdx.y == 0 && !(skip == kSkipIfStopped && P->stopped);
  float* __restrict__ B = slot_base(bufs, P, bufs_sel);
  const bool keep_rows = bufs.slot_stride > 0;  // rounds keep the pre-conv rows for a fold; prefill never folds
  const int snap_local = (snap_sel >= 0 && blockIdx.y == max(snap_row - 1, 0) / kConvRows) ? (snap_row - r0) : -1;
  float* ws = snap_local >= 0 ? win + size_t(slot_of(P, snap_sel)) * win_stride : nullptr;

  GdnConvStream<CK> st;
  if (blockIdx.y == 0) {
    st.load_before(conv_w, w, C, ck, c);
  } else if constexpr (CK > 0) {
#pragma unroll
    for (int j = 0; j < CK; ++j) st.cw[j] = conv_w[size_t(c) * CK + j];
#pragma unroll
    for (int j = 0; j < CK - 1; ++j) st.w[j] = in_value(mixed + size_t(r0 - (CK - 1) + j) * ld_mixed + c);
  } else {
    for (int j = 0; j < ck; ++j) st.cw[j] = conv_w[size_t(c) * ck + j];
    for (int j = 0; j < ck - 1; ++j) st.w[j] = in_value(mixed + size_t(r0 - (ck - 1) + j) * ld_mixed + c);
  }
  if (snap_local == 0) st.store_window(ws, C, ck, c);

  constexpr int kPipe = 4;
  float* out_dst = c < kd ? B + bufs.o_qn() + c
                          : (c < 2 * kd ? B + bufs.o_kn() + (c - kd) : B + bufs.o_v() + (c - 2 * kd));
  const size_t out_stride = c < 2 * kd ? size_t(kd) : size_t(vd);
  for (int rb = 0; rb < n; rb += kPipe) {
    float xv[kPipe];
#pragma unroll
    for (int k = 0; k < kPipe; ++k)
      if (rb + k < n) xv[k] = in_value(mixed + size_t(r0 + rb + k) * ld_mixed + c);
#pragma unroll
    for (int k = 0; k < kPipe; ++k) {
      const int r = rb + k;
      if (r < n) {
        const float y = st.step(xv[k], ck);
        out_dst[size_t(r0 + r) * out_stride] = y;
        if (keep_rows) B[size_t(r0 + r) * C + c] = xv[k];
        if (r + 1 == snap_local) st.store_window(ws, C, ck, c);
      }
    }
  }
  if (advance) {
    float* w_out = win + size_t(slot_of(P, win_out_sel)) * win_stride;
    if (rows <= kConvRows) {
      st.store_window(w_out, C, ck, c);
    } else {  // the last ck - 1 rows (rows > kConvRows >= ck - 1: all from mixed)
      for (int j = 0; j < ck - 1; ++j)
        w_out[size_t(j) * C + c] = in_value(mixed + size_t(rows - (ck - 1) + j) * ld_mixed + c);
    }
  }
}

// A tree round's rows (rows <= kConvRows; the window is folded at the commit): row r's inputs before it are its
// ancestors' at the depths above it, then the window's; the same products in the same order as GdnConvIn::out.  The
// loads go first as in k_gdn_conv; a thread's inputs then sit in shared memory, indexed by the path.  grid
// ceil(C / 256); block 256.
template <int CK>
__global__ void k_gdn_conv_tree(const float* mixed, int ld_mixed, const float* conv_w, const float* win,
                                size_t win_stride, int win_sel, GdnBufs bufs, int bufs_sel, const DevParams* P, int rows,
                                const TreeRows* tree) {
  pdl_trigger();
  if (bufs.slot_stride > 0 && P->stopped) return;
  constexpr int NX = kConvRows + GdnConvIn<CK>::kMaxCk - 1;
  __shared__ float xs[NX][256];
  __shared__ unsigned char t_depth[kConvRows], t_path[kConvRows][kConvRows];
  for (int i = threadIdx.x; i < kConvRows * kConvRows; i += blockDim.x)
    t_path[i / kConvRows][i % kConvRows] = tree->path[i / kConvRows][i % kConvRows];
  if (threadIdx.x < kConvRows) t_depth[threadIdx.x] = static_cast<unsigned char>(tree->depth[threadIdx.x]);
  __syncthreads();
  const int C = bufs.d.C(), ck = CK > 0 ? CK : bufs.d.ck;
  const int kd = bufs.d.nk * bufs.d.dk, vd = bufs.d.nv * bufs.d.dv;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const int n = min(kConvRows, rows);
  GdnConvIn<CK> in;
  in.load(mixed, ld_mixed, conv_w, win + size_t(slot_of(P, win_sel)) * win_stride, C, ck, c, 0, n);
#pragma unroll
  for (int i = 0; i < NX; ++i) xs[i][threadIdx.x] = in.x[i];  // the thread's own column: no barrier
  float* B = slot_base(bufs, P, bufs_sel);
  for (int r = 0; r < n; ++r) {
    const int depth = t_depth[r];
    float acc = __fmul_rn(xs[r + ck - 1][threadIdx.x], in.cw[ck - 1]);
#pragma unroll
    for (int j = 0; j < GdnConvIn<CK>::kMaxCk - 1; ++j)
      if (j < ck - 1) {
        const int e = depth - (ck - 1) + j;  // the input's depth; below 0: the window's
        acc = __fmaf_rn(xs[ck - 1 + (e >= 0 ? int(t_path[r][e]) : e)][threadIdx.x], in.cw[j], acc);
      }
    const float y = silu(acc);
    if (c < kd) B[bufs.o_qn() + size_t(r) * kd + c] = y;
    else if (c < 2 * kd) B[bufs.o_kn() + size_t(r) * kd + (c - kd)] = y;
    else B[bufs.o_v() + size_t(r) * vd + (c - 2 * kd)] = y;
    if (bufs.slot_stride > 0) B[size_t(r) * C + c] = xs[r + ck - 1][threadIdx.x];
  }
}

// layers non-null: layer blockIdx.y of them (gdn_fold_layers), its window and replay buffers in place of win's and
// bufs'.  Direct O(ck) copy of only the surviving tail (ck - 1 rows): avoids dead reads of evicted history and
// eliminates the local-memory shift buffer on the common CK == 4 path.
template <int CK = 0>
__global__ void k_gdn_conv_fold(float* win, size_t win_stride, int in_sel, int out_sel, GdnBufs bufs, int bufs_sel,
                                const DevParams* P, int rows, const int* rows_dev, int skip_if_noop,
                                const GdnLayer* layers) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  if (layers) {
    win = layers[blockIdx.y].win;
    bufs.base = layers[blockIdx.y].replay;
  }
  const int C = bufs.d.C(), ck = CK > 0 ? CK : bufs.d.ck;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  if (rows_dev) rows = *rows_dev + 1;
  if (rows < 0) rows = 0;
  const float* wi = win + size_t(slot_of(P, in_sel)) * win_stride;
  float* wo = win + size_t(slot_of(P, out_sel)) * win_stride;
  const float* mixed = slot_base(bufs, P, bufs_sel);
  if constexpr (CK == 4) {
    const int p0 = rows - 3, p1 = rows - 2, p2 = rows - 1;
    const float v0 = p0 >= 0 ? mixed[size_t(p0) * C + c] : wi[size_t(3 + p0) * C + c];
    const float v1 = p1 >= 0 ? mixed[size_t(p1) * C + c] : wi[size_t(3 + p1) * C + c];
    const float v2 = p2 >= 0 ? mixed[size_t(p2) * C + c] : wi[size_t(3 + p2) * C + c];
    wo[size_t(0) * C + c] = v0;
    wo[size_t(1) * C + c] = v1;
    wo[size_t(2) * C + c] = v2;
  } else {
    const int w = ck - 1;
    float hist[7];
    for (int j = 0; j < w; ++j) {
      const int pos = rows - w + j;
      hist[j] = pos >= 0 ? mixed[size_t(pos) * C + c] : wi[size_t(w + pos) * C + c];
    }
    for (int j = 0; j < w; ++j) wo[size_t(j) * C + c] = hist[j];
  }
}

// grid (nk + 1, ceil(rows / r_step)); block 128.  Blocks < nk normalise one q/k head; the last computes beta and g.
template <class TIn = float>
__global__ void k_gdn_prep(GdnBufs bufs, int bufs_sel, const TIn* b, const TIn* a, int ld_ba, const float* A_log,
                           const float* dt_bias, const DevParams* P, int rows, int r_step) {
  pdl_trigger();
  if (bufs.slot_stride > 0 && P->stopped) return;
  const GdnDims d = bufs.d;
  const int r0 = int(blockIdx.y) * r_step, r1 = min(rows, r0 + r_step);
  const int kh = blockIdx.x, i = threadIdx.x;
  float* B = slot_base(bufs, P, bufs_sel);
  if (kh < d.nk) {
    // Block is always 128 threads (4 warps): double-buffered sw buffers keep 1 barrier per row in block_sum's order.
    __shared__ float sq[2][4], sk[2][4];
    const int lane = i & 31, warp = i >> 5;
    const float inv_sq_dk = __frsqrt_rn(float(d.dk));
    float* q0 = B + bufs.o_qn() + (size_t(r0) * d.nk + kh) * d.dk;
    float* k0 = B + bufs.o_kn() + (size_t(r0) * d.nk + kh) * d.dk;
    const size_t stride = size_t(d.nk) * d.dk;
    float q = i < d.dk ? q0[i] : 0.f;
    float k = i < d.dk ? k0[i] : 0.f;
    for (int r = r0; r < r1; ++r) {
      const int b = (r - r0) & 1;
      float* qp = q0 + size_t(r - r0) * stride;
      float* kp = k0 + size_t(r - r0) * stride;
      const float wq = warp_sum(__fmul_rn(q, q));
      const float wk = warp_sum(__fmul_rn(k, k));
      float q_nxt = 0.f, k_nxt = 0.f;
      if (r + 1 < r1 && i < d.dk) q_nxt = qp[stride + i], k_nxt = kp[stride + i];
      if (lane == 0) sq[b][warp] = wq, sk[b][warp] = wk;
      __syncthreads();
      const float iq = rsqrtf(((sq[b][0] + sq[b][2]) + (sq[b][1] + sq[b][3])) + 1e-6f);
      const float ik = rsqrtf(((sk[b][0] + sk[b][2]) + (sk[b][1] + sk[b][3])) + 1e-6f);
      if (i < d.dk) {
        qp[i] = q * iq * inv_sq_dk;
        kp[i] = k * ik;
      }
      q = q_nxt;
      k = k_nxt;
    }
  } else {
    for (int h = i; h < d.nv; h += blockDim.x) {
      const float al = A_log[h], dtb = dt_bias[h];
      for (int r = r0; r < r1; ++r)
        gdn_gates(in_value(b + size_t(r) * ld_ba + h), in_value(a + size_t(r) * ld_ba + h), al, dtb,
                  &B[bufs.o_beta() + size_t(r) * d.nv + h], &B[bufs.o_g() + size_t(r) * d.nv + h]);
    }
  }
}

// grid (nv, ceil(dv / 8)); block 256: one warp per state column, 4 key dims per lane (dk <= 128).
__global__ void __launch_bounds__(256) k_gdn_recur(float* S, size_t s_stride, int in_sel, int out_sel, GdnBufs bufs,
                                                   int bufs_sel, const DevParams* P, float* out, int rows,
                                                   const int* rows_dev, int skip) {
  pdl_trigger();
  if ((skip == kSkipIfNoop && P->noop) || (skip == kSkipIfStopped && P->stopped)) return;
  const GdnDims d = bufs.d;
  const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.y * 8 + warp;
  if (j >= d.dv) return;
  if (rows_dev) rows = *rows_dev + 1;
  const int kh = d.key_head(h);
  const float* sin_ = S + size_t(slot_of(P, in_sel)) * s_stride + (size_t(h) * d.dv + j) * d.dk;
  const float* B = slot_base(bufs, P, bufs_sel);
  const float* vbuf = B + bufs.o_v();
  const int vd = d.nv * d.dv;
  float s[4];
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int i = lane + 32 * e;
    s[e] = i < d.dk ? sin_[i] : 0.f;
  }
  for (int r = 0; r < rows; ++r) {
    const float* kn = B + bufs.o_kn() + (size_t(r) * d.nk + kh) * d.dk;
    const float* qn = B + bufs.o_qn() + (size_t(r) * d.nk + kh) * d.dk;
    const float decay = __expf(B[bufs.o_g() + size_t(r) * d.nv + h]);
    const float beta = B[bufs.o_beta() + size_t(r) * d.nv + h];
    const float vj = vbuf[size_t(r) * vd + h * d.dv + j];
    float kk[4], mem = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = lane + 32 * e;
      kk[e] = i < d.dk ? kn[i] : 0.f;
      s[e] *= decay;
      mem += s[e] * kk[e];
    }
    const float delta = (vj - warp_sum(mem)) * beta;
    float o = 0.f;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = lane + 32 * e;
      s[e] += kk[e] * delta;
      if (out) o += s[e] * (i < d.dk ? qn[i] : 0.f);
    }
    if (out) {
      o = warp_sum(o);
      if (lane == 0) out[(size_t(r) * d.nv + h) * d.dv + j] = o;
    }
  }
  if (out_sel >= 0) {
    float* so = S + size_t(slot_of(P, out_sel)) * s_stride + (size_t(h) * d.dv + j) * d.dk;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int i = lane + 32 * e;
      if (i < d.dk) so[i] = s[e];
    }
  }
}

// The recurrence with four lanes per state row j: lane p of the group holds
// the key dims i = 16 m + 4 p + e (DK / 4 of them, float4 e).  Per row, both
// sums over i run on the old state in one pass, (S k) and (S q), each a
// per-lane sum ((e0 + e1) + (e2 + e3)) and two shuffles that leave the same
// value in all four lanes; then
//   mem = decay (S k),  delta = (v - mem) beta,
//   o = decay (S q) + delta (k . q),  S = decay S + delta k^T,
// which is S <- decay S, S += delta k^T, o = S q with o off the path to the
// next row.  Every rounding is spelled out (__fmaf_rn / __fmul_rn): left to
// contraction, the compiler may fuse a different product in the copy that
// writes outputs than in the one that does not, and a fold would then leave
// other bits than the verification that produced the outputs.  A row's
// arithmetic does not depend on how many rows run together: prefill,
// verification, fold and decode give the same bits for the same inputs.
constexpr int kGdnJB = 32, kGdnRT = 16;
// The most rows k_gdn_round takes, and the fewer its smaller instance is sized for: registers and shared memory sized
// for 8 rows cost a round of 1 row 7% (cuda_bench gdn 27b on the RTX 5090: 4.63 -> 4.97 us a layer), so rounds of
// up to 4 rows take the 4-row kernel and 5 to 8 the 8-row one (8 rows 13.3 us a layer against the four kernels'
// 15.4); from 9 rows the four kernels, the one kernel repeating its key head's q and k conv in every block.
constexpr int kGdnRoundRows = 8, kGdnRoundFew = 4;

// k . q of one row (normalised, in shared memory), summed by one warp; every lane gets it.
template <int DK>
__device__ __forceinline__ float gdn_kq(const float* k, const float* q, int lane) {
  float a = 0.f;
#pragma unroll
  for (int i = lane; i < DK; i += 32) a = __fmaf_rn(k[i], q[i], a);
  return warp_sum(a);
}

// One row for the lane group of state row j (lane p of the group): updates s,
// returns o_j when OUTPUT.  k, q: the row's normalised key and query (shared
// memory, 16-byte aligned); kq = gdn_kq(k, q).
template <int DK, bool OUTPUT>
__device__ __forceinline__ float gdn_lanes_step(float (&s)[DK / 16][4], const float* k, const float* q, float v,
                                                float beta, float g, float kq, int p) {
  constexpr int M = DK / 16;
  const float decay = __expf(g);
  float4 kk[M];
#pragma unroll
  for (int m = 0; m < M; ++m) kk[m] = *reinterpret_cast<const float4*>(k + 16 * m + 4 * p);
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
  for (int m = 0; m < M; ++m) {
    a0 = __fmaf_rn(s[m][0], kk[m].x, a0);
    a1 = __fmaf_rn(s[m][1], kk[m].y, a1);
    a2 = __fmaf_rn(s[m][2], kk[m].z, a2);
    a3 = __fmaf_rn(s[m][3], kk[m].w, a3);
    if (OUTPUT) {
      const float4 qq = *reinterpret_cast<const float4*>(q + 16 * m + 4 * p);
      b0 = __fmaf_rn(s[m][0], qq.x, b0);
      b1 = __fmaf_rn(s[m][1], qq.y, b1);
      b2 = __fmaf_rn(s[m][2], qq.z, b2);
      b3 = __fmaf_rn(s[m][3], qq.w, b3);
    }
  }
  float sk = (a0 + a1) + (a2 + a3), sq = (b0 + b1) + (b2 + b3);
  sk += __shfl_xor_sync(0xffffffffu, sk, 1);
  sk += __shfl_xor_sync(0xffffffffu, sk, 2);
  if (OUTPUT) {
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2);
  }
  const float delta = __fmul_rn(__fmaf_rn(-decay, sk, v), beta);
#pragma unroll
  for (int m = 0; m < M; ++m) {
    s[m][0] = __fmaf_rn(kk[m].x, delta, __fmul_rn(s[m][0], decay));
    s[m][1] = __fmaf_rn(kk[m].y, delta, __fmul_rn(s[m][1], decay));
    s[m][2] = __fmaf_rn(kk[m].z, delta, __fmul_rn(s[m][2], decay));
    s[m][3] = __fmaf_rn(kk[m].w, delta, __fmul_rn(s[m][3], decay));
  }
  return OUTPUT ? __fmaf_rn(delta, kq, __fmul_rn(decay, sq)) : 0.f;
}

// A state row (DK floats, 16-byte aligned) into the lane group's registers, and back.
template <int DK>
__device__ __forceinline__ void gdn_state_load(float (&s)[DK / 16][4], const float* row, int p) {
#pragma unroll
  for (int m = 0; m < DK / 16; ++m) {
    const float4 v = *reinterpret_cast<const float4*>(row + 16 * m + 4 * p);
    s[m][0] = v.x;
    s[m][1] = v.y;
    s[m][2] = v.z;
    s[m][3] = v.w;
  }
}
template <int DK>
__device__ __forceinline__ void gdn_state_store(const float (&s)[DK / 16][4], float* row, int p) {
#pragma unroll
  for (int m = 0; m < DK / 16; ++m)
    *reinterpret_cast<float4*>(row + 16 * m + 4 * p) = make_float4(s[m][0], s[m][1], s[m][2], s[m][3]);
}

// grid (nv, ceil(dv / kGdnJB)); block 128.  The per-row inputs (normalised k
// and q, v, beta, g) stream through shared memory kGdnRT rows at a time,
// double-buffered with cp.async; k . q is computed once per row while staging.
template <int DK, bool FOLD = false>
struct GdnStage {
  float k[kGdnRT][DK], q[kGdnRT][DK], v[kGdnRT][kGdnJB], beta[kGdnRT], g[kGdnRT], kq[kGdnRT];
};
template <int DK>
struct GdnStage<DK, true> {
  float k[kGdnRT][DK], v[kGdnRT][kGdnJB], beta[kGdnRT], g[kGdnRT];
};

// layers non-null: layer blockIdx.z of them (gdn_fold_layers), its state and replay buffers in place of S and bufs'.
// TREE (a round's tree, rows <= kGdnRT, no state written): row r goes on from its parent's state; where the parent is
// not the row before, the start state is loaded again and the parent's path replayed (the same steps), so each row
// gets the bits of its path decoded alone.
// FOLD (out == nullptr: commit fold): omits q/kq from shared memory (37 KB -> 20.7 KB) and compiles only the
// no-output step so 4 blocks/SM fit.
template <int DK, bool TREE = false, bool FOLD = false>
__global__ void __launch_bounds__(128) k_gdn_recur_lanes(float* S, size_t s_stride, int in_sel, int out_sel,
                                                         GdnBufs bufs, int bufs_sel, const DevParams* P, float* out,
                                                         int rows, const int* rows_dev, int skip,
                                                         const GdnLayer* layers, const TreeRows* tree = nullptr) {
  pdl_trigger();
  if ((skip == kSkipIfNoop && P->noop) || (skip == kSkipIfStopped && P->stopped)) return;
  if (layers) {
    S = layers[blockIdx.z].S;
    bufs.base = layers[blockIdx.z].replay;
  }
  __shared__ __align__(16) GdnStage<DK, FOLD> sm[2];
  // TREE: each row's depth and path, read from the tree table once (not two dependent loads between steps).
  __shared__ unsigned char t_depth[TREE ? kGdnRT : 1], t_path[TREE ? kGdnRT : 1][TREE ? kGdnRT : 1];
  const GdnDims d = bufs.d;
  const int h = blockIdx.x, tid = threadIdx.x, p = tid & 3, jl = tid >> 2, warp = tid >> 5, lane = tid & 31;
  if constexpr (TREE) {  // made visible by the first chunk's barrier
    for (int i = tid; i < kGdnRT * kGdnRT; i += blockDim.x)
      t_path[i / kGdnRT][i % kGdnRT] = tree->path[i / kGdnRT][i % kGdnRT];
    if (tid < kGdnRT) t_depth[tid] = static_cast<unsigned char>(tree->depth[tid]);
  }
  const int j0 = blockIdx.y * kGdnJB, j = j0 + jl;
  const bool live = j < d.dv;  // all four lanes of a group agree; dead groups still shuffle
  if (rows_dev) rows = *rows_dev + 1;
  const int kh = d.key_head(h), vd = d.nv * d.dv;
  const float* B = slot_base(bufs, P, bufs_sel);
  const bool b_aligned16 = (reinterpret_cast<uintptr_t>(B) & 15u) == 0;
  const bool k_aligned16 = b_aligned16 && ((bufs.o_kn() & 3u) == 0) && (FOLD || ((bufs.o_qn() & 3u) == 0));
  const int jn = min(kGdnJB, d.dv - j0);
  const bool v_aligned16 = b_aligned16 && ((bufs.o_v() & 3u) == 0) && ((d.dv & 3) == 0) && ((jn & 3) == 0);

  // Copies rows [r0, r0 + n) of the inputs into buffer `buf` (only the rows that exist).
  auto stage = [&](int buf, int r0, int n) {
    if (k_aligned16) {
      constexpr int DK4 = DK / 4;
      for (int idx = tid; idx < n * DK4; idx += 128) {
        const int r = idx / DK4, i = (idx % DK4) * 4;
        const size_t at = (size_t(r0 + r) * d.nk + kh) * DK + i;
        cp_async16(&sm[buf].k[r][i], B + bufs.o_kn() + at);
        if constexpr (!FOLD)
          if (out) cp_async16(&sm[buf].q[r][i], B + bufs.o_qn() + at);
      }
    } else {
      for (int idx = tid; idx < n * DK; idx += 128) {
        const int r = idx / DK, i = idx % DK;
        const size_t at = (size_t(r0 + r) * d.nk + kh) * DK + i;
        cp_async4(&sm[buf].k[r][i], B + bufs.o_kn() + at);
        if constexpr (!FOLD)
          if (out) cp_async4(&sm[buf].q[r][i], B + bufs.o_qn() + at);
      }
    }
    if (v_aligned16) {
      constexpr int JB4 = kGdnJB / 4;
      for (int idx = tid; idx < n * JB4; idx += 128) {
        const int r = idx / JB4, c = (idx % JB4) * 4;
        if (c < jn) cp_async16(&sm[buf].v[r][c], B + bufs.o_v() + size_t(r0 + r) * vd + h * d.dv + j0 + c);
      }
    } else {
      for (int idx = tid; idx < n * kGdnJB; idx += 128) {
        const int r = idx / kGdnJB, c = idx % kGdnJB;
        if (c < jn) cp_async4(&sm[buf].v[r][c], B + bufs.o_v() + size_t(r0 + r) * vd + h * d.dv + j0 + c);
      }
    }
    if (tid < n)
      cp_async4(&sm[buf].beta[tid], B + bufs.o_beta() + size_t(r0 + tid) * d.nv + h);
    else if (tid >= kGdnRT && tid < kGdnRT + n)
      cp_async4(&sm[buf].g[tid - kGdnRT], B + bufs.o_g() + size_t(r0 + tid - kGdnRT) * d.nv + h);
  };

  // Start chunk 0's async copy before loading S so input staging overlaps the DRAM state read.
  const int chunks = (rows + kGdnRT - 1) / kGdnRT;
  if (chunks > 0) stage(0, 0, min(kGdnRT, rows));
  cp_async_commit();
  float s[DK / 16][4];
  const float* s_in = S + size_t(slot_of(P, in_sel)) * s_stride + (size_t(h) * d.dv + (live ? j : 0)) * DK;
  gdn_state_load<DK>(s, s_in, p);
  for (int ch = 0; ch < chunks; ++ch) {
    const int buf = ch & 1, rn = min(kGdnRT, rows - ch * kGdnRT);
    if (ch + 1 < chunks)  // buf ^ 1 was released by the barrier ending ch - 1
      stage(buf ^ 1, (ch + 1) * kGdnRT, min(kGdnRT, rows - (ch + 1) * kGdnRT));
    cp_async_commit();
    cp_async_wait<1>();
    __syncthreads();
    GdnStage<DK, FOLD>& t = sm[buf];
    if constexpr (!FOLD) {
      if (out) {  // k . q per row: warp w takes rows w, w + 4, ...
        for (int rr = warp; rr < rn; rr += 4) {
          const float a = gdn_kq<DK>(t.k[rr], t.q[rr], lane);
          if (lane == 0) t.kq[rr] = a;
        }
        __syncthreads();
      }
    }
    for (int rr = 0; rr < rn; ++rr) {
      if constexpr (TREE) {  // rows < kGdnRT: rr is the row
        const int depth = t_depth[rr];
        if (rr > 0 && t_path[rr][depth - 1] != rr - 1) {
          gdn_state_load<DK>(s, s_in, p);
          for (int e = 0; e < depth; ++e) {
            const int a = t_path[rr][e];
            gdn_lanes_step<DK, false>(s, t.k[a], nullptr, t.v[a][jl], t.beta[a], t.g[a], 0.f, p);
          }
        }
      }
      if constexpr (FOLD) {
        gdn_lanes_step<DK, false>(s, t.k[rr], nullptr, t.v[rr][jl], t.beta[rr], t.g[rr], 0.f, p);
      } else if (out) {
        const float o = gdn_lanes_step<DK, true>(s, t.k[rr], t.q[rr], t.v[rr][jl], t.beta[rr], t.g[rr], t.kq[rr], p);
        if (live && p == 0) out[(size_t(ch * kGdnRT + rr) * d.nv + h) * d.dv + j] = o;
      } else {
        gdn_lanes_step<DK, false>(s, t.k[rr], nullptr, t.v[rr][jl], t.beta[rr], t.g[rr], 0.f, p);
      }
    }
    __syncthreads();  // buf is restaged at ch + 1
  }
  if (out_sel >= 0 && live)
    gdn_state_store<DK>(s, S + size_t(slot_of(P, out_sel)) * s_stride + (size_t(h) * d.dv + j) * DK, p);
}

// After a tree round's accept: the accepted path's rows taken[1 ..] of the round's replay slot (kSlotLast) to rows
// 1 .. n_taken, where the commit's fold and a rewind read them: the conv inputs and outputs, the normalised q and k,
// beta and g.  Rows go in order (taken[d] >= d, so a row is read before anything lands on it).  layers non-null: layer
// blockIdx.y of them.
__global__ void k_gdn_compact(GdnBufs bufs, const GdnLayer* layers, const DevParams* P, const TreeRows* tree) {
  pdl_trigger();
  if (!tree_on(tree) || P->noop) return;
  if (layers) bufs.base = layers[blockIdx.y].replay;
  const GdnDims d = bufs.d;
  const int C = d.C(), kd = d.nk * d.dk, vd = d.nv * d.dv, i = blockIdx.x * blockDim.x + threadIdx.x;
  float* B = slot_base(bufs, P, kSlotLast);
  for (int e = 1; e <= tree->n_taken; ++e) {
    const int from = tree->taken[e];
    if (from == e) continue;
    if (i < C) B[size_t(e) * C + i] = B[size_t(from) * C + i];
    if (i < vd) B[bufs.o_v() + size_t(e) * vd + i] = B[bufs.o_v() + size_t(from) * vd + i];
    if (i < kd) {
      B[bufs.o_qn() + size_t(e) * kd + i] = B[bufs.o_qn() + size_t(from) * kd + i];
      B[bufs.o_kn() + size_t(e) * kd + i] = B[bufs.o_kn() + size_t(from) * kd + i];
    }
    if (i < d.nv) {
      B[bufs.o_beta() + size_t(e) * d.nv + i] = B[bufs.o_beta() + size_t(from) * d.nv + i];
      B[bufs.o_g() + size_t(e) * d.nv + i] = B[bufs.o_g() + size_t(from) * d.nv + i];
    }
  }
}

// grid (nv, ceil(rows / r_step)); block = dv threads.
template <class TZ = float>
__global__ void k_gdn_gated_norm(const float* o, const TZ* z, int ld_z, const float* w, float eps, Out y, int nv,
                                 int dv, int rows, int r_step, bool sigmoid_output) {
  pdl_trigger();
  const int h = blockIdx.x, r0 = int(blockIdx.y) * r_step, r1 = min(rows, r0 + r_step);
  const int j = threadIdx.x;  // blockDim: dv rounded up to 32
  const float wj = j < dv ? w[j] : 0.f;
  const size_t o_stride = size_t(nv) * dv;
  if (blockDim.x == 128) {
    __shared__ float sw[2][4];
    const int lane = j & 31, warp = j >> 5;
    size_t at = size_t(r0) * o_stride + size_t(h) * dv + j;
    size_t zat = size_t(r0) * ld_z + size_t(h) * dv + j;
    float v = j < dv ? o[at] : 0.f;
    float zv = j < dv ? in_value(z + zat) : 0.f;
    for (int r = r0; r < r1; ++r) {
      const int b = (r - r0) & 1;
      const float ws = warp_sum(__fmul_rn(v, v));
      const float sz = j < dv ? (sigmoid_output ? 1.f / (1.f + expf(-zv)) : silu(zv)) : 0.f;
      // Prefetch next row's o and z before the barrier so their DRAM latency hides behind reduction + output store.
      float v_nxt = 0.f, z_nxt = 0.f;
      if (r + 1 < r1 && j < dv) {
        v_nxt = o[at + o_stride];
        z_nxt = in_value(z + zat + ld_z);
      }
      if (lane == 0) sw[b][warp] = ws;
      __syncthreads();
      const float inv = rsqrtf(((sw[b][0] + sw[b][2]) + (sw[b][1] + sw[b][3])) / dv + eps);
      if (y.q8) {  // 8-bit rows (dv % 32 == 0): a warp holds 32 consecutive values of the row, one group
        put_q8<false, true>(y, r, (h * dv + j) >> 5, v * inv * wj * sz);
      } else if (j < dv) {
        put<true>(y, at, at, v * inv * wj * sz);
      }
      v = v_nxt;
      zv = z_nxt;
      at += o_stride;
      zat += ld_z;
    }
    return;
  }
  for (int r = r0; r < r1; ++r) {
    const size_t at = size_t(r) * o_stride + size_t(h) * dv + j;
    const float v = j < dv ? o[at] : 0.f;
    const float zv = j < dv ? in_value(z + size_t(r) * ld_z + size_t(h) * dv + j) : 0.f;
    const float sz = j < dv ? (sigmoid_output ? 1.f / (1.f + expf(-zv)) : silu(zv)) : 0.f;
    const float inv = rsqrtf(block_sum(__fmul_rn(v, v)) / dv + eps);
    if (y.q8) {
      put_q8<false, true>(y, r, (h * dv + j) >> 5, v * inv * wj * sz);
    } else if (j < dv) {
      put<true>(y, at, at, v * inv * wj * sz);
    }
  }
}

// block_sum's order for a block of 4 warps: warp 0 adds the per-warp sums
// with the same shuffles, which for values in lanes 0..3 (zeros above) is this.
__device__ __forceinline__ float block4_sum(const float (&w)[4]) { return (w[0] + w[2]) + (w[1] + w[3]); }

// A round of up to RT rows (kGdnRoundFew or kGdnRoundRows) of one layer: k_gdn_conv, k_gdn_prep,
// k_gdn_recur_lanes and k_gdn_gated_norm in one kernel, every value from the
// same operations in the same order as there (the same bits; cuda_op_test
// checks).  grid (nv, ceil(dv / kGdnJB)) in clusters of gridDim.y blocks (one
// head); block 128.  Each block convolves and normalises its key head's q and
// k itself (all blocks of the key head do the same work; the first also
// writes them to the replay buffers).  The gated norm's sum over the head's dv
// outputs adds the cluster's per-block sums, read through distributed shared
// memory, as k_gdn_gated_norm's block_sum adds its warps' (block y of the
// cluster holds what warp y there holds).  next_sel >= 0 (plain decoding, one
// row): the advanced window and state go to that slot, unless P->stopped.
template <int DK, int CK, int RT>
__global__ void __launch_bounds__(128) k_gdn_round(const float* proj, int ld, const float* conv_w,
                                                   float* win, size_t win_stride, float* S, size_t s_stride,
                                                   int next_sel, const float* A_log, const float* dt_bias,
                                                   const float* gw, float eps, GdnBufs bufs, const DevParams* P,
                                                   int rows, Out y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
  __trap();  // clusters are sm_90's: below it gdn_round_supported() runs the layer's kernels one by one instead
#else
  pdl_trigger();
  __shared__ __align__(16) float sq[RT][DK], sk[RT][DK];
  __shared__ float sv[RT][kGdnJB], so[RT][kGdnJB], sbeta[RT], sg[RT], skq[RT];
  __shared__ float ssq[RT][4], ssk[RT][4], spart[RT];
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const GdnDims d = bufs.d;
  const int ck = CK > 0 ? CK : d.ck;
  const int h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int kh = d.key_head(h), C = d.C(), kd = d.nk * DK, vd = d.nv * d.dv;
  const int j0 = blockIdx.y * kGdnJB, jn = min(kGdnJB, d.dv - j0);
  const bool lead = d.first_of_key(h) && blockIdx.y == 0;  // writes the key head's q and k
  const bool save_replay = !P->stopped;  // noop in-flight round must not clobber a prior round's replay slot
  const bool advance = next_sel >= 0 && !P->stopped;
  float* B = slot_base(bufs, P, kSlotCur);
  const float* w_cur = win + size_t(slot_of(P, kSlotCur)) * win_stride;
  float* w_next = advance ? win + size_t(slot_of(P, next_sel)) * win_stride : nullptr;

  // The state rows first: their loads overlap the conv.
  const int p = tid & 3, jl = tid >> 2, j = j0 + jl;
  const bool live = jl < jn;
  float s[DK / 16][4];
  gdn_state_load<DK>(s, S + size_t(slot_of(P, kSlotCur)) * s_stride + (size_t(h) * d.dv + (live ? j : j0)) * DK, p);
  // Nor are the weights, the conv window or the gates' parameters (the round's kernels write none of them): on their
  // way before the wait too, so that only the projection rows are left to come after it.
  // conv (k_gdn_conv): thread tid takes q and k dim tid, and value channel j0 + tid
  const int cq = kh * DK + tid, cx = kd + kh * DK + tid, cv = 2 * kd + h * d.dv + j0 + tid;
  GdnConvStream<CK> iq, ik, iv;
  if (tid < DK) {
    iq.load_before(conv_w, w_cur, C, ck, cq);
    ik.load_before(conv_w, w_cur, C, ck, cx);
  }
  if (tid < jn) iv.load_before(conv_w, w_cur, C, ck, cv);
  // The gated norm's weights for this block's outputs, and the gates' parameters of the head.
  const float gate_w = lane < jn ? gw[j0 + lane] : 0.f;
  const float a_log = A_log[h], dt_b = dt_bias[h];
  pdl_wait();  // the projection rows come from the kernel before
  // The gated norm's z for row `warp` (each warp's only row up to 4 rows), loaded now rather than after the
  // recurrence.
  const float gate_z = warp < rows && lane < jn ? proj[size_t(warp) * ld + C + h * d.dv + j0 + lane] : 0.f;
  for (int r = 0; r < rows; ++r) {
    const float* pr = proj + size_t(r) * ld;
    const float xq = tid < DK ? pr[cq] : 0.f;
    const float xk = tid < DK ? pr[cx] : 0.f;
    const float xv = tid < jn ? pr[cv] : 0.f;
    float qv = 0.f, kv = 0.f;
    if (tid < DK) {
      qv = iq.step(xq, ck);
      kv = ik.step(xk, ck);
      sq[r][tid] = qv;
      sk[r][tid] = kv;
      if (lead && save_replay) {
        B[size_t(r) * C + cq] = xq;
        B[size_t(r) * C + cx] = xk;
      }
    }
    if (tid < jn) {
      const float vv = iv.step(xv, ck);
      sv[r][tid] = vv;
      if (save_replay) {
        B[bufs.o_v() + size_t(r) * vd + h * d.dv + j0 + tid] = vv;
        B[size_t(r) * C + cv] = xv;
      }
    }
    // q and k per-warp sums of squares computed directly from registers before storing (bit-identical to block_sum).
    const float a = warp_sum(__fmul_rn(qv, qv)), b = warp_sum(__fmul_rn(kv, kv));
    if (lane == 0) ssq[r][warp] = a, ssk[r][warp] = b;
  }
  if (w_next) {
    if (tid < DK && lead) {
      iq.store_window(w_next, C, ck, cq);
      ik.store_window(w_next, C, ck, cx);
    }
    if (tid < jn) iv.store_window(w_next, C, ck, cv);
  }
  if (tid < rows) {  // gates (k_gdn_prep)
    const float* pr = proj + size_t(tid) * ld + C + vd;
    gdn_gates(pr[h], pr[d.nv + h], a_log, dt_b, &sbeta[tid], &sg[tid]);
    if (blockIdx.y == 0 && save_replay) {
      B[bufs.o_beta() + size_t(tid) * d.nv + h] = sbeta[tid];
      B[bufs.o_g() + size_t(tid) * d.nv + h] = sg[tid];
    }
  }
  __syncthreads();
  if (tid < DK)
    for (int r = 0; r < rows; ++r) {
      const float iq = rsqrtf(block4_sum(ssq[r]) + 1e-6f), ik = rsqrtf(block4_sum(ssk[r]) + 1e-6f);
      const float qn = sq[r][tid] * iq * __frsqrt_rn(float(DK)), kn = sk[r][tid] * ik;
      sq[r][tid] = qn;
      sk[r][tid] = kn;
      if (lead && save_replay) {
        B[bufs.o_qn() + (size_t(r) * d.nk + kh) * DK + tid] = qn;
        B[bufs.o_kn() + (size_t(r) * d.nk + kh) * DK + tid] = kn;
      }
    }
  __syncthreads();
  for (int r = warp; r < rows; r += 4) {
    const float a = gdn_kq<DK>(sk[r], sq[r], lane);
    if (lane == 0) skq[r] = a;
  }
  __syncthreads();
  {  // the recurrence (k_gdn_recur_lanes)
    for (int r = 0; r < rows; ++r) {
      const float o = gdn_lanes_step<DK, true>(s, sk[r], sq[r], live ? sv[r][jl] : 0.f, sbeta[r], sg[r], skq[r], p);
      if (live && p == 0) so[r][jl] = o;
    }
    if (advance && live)
      gdn_state_store<DK>(s, S + size_t(slot_of(P, next_sel)) * s_stride + (size_t(h) * d.dv + j) * DK, p);
  }
  __syncthreads();
  // gated norm (k_gdn_gated_norm): this block's sum of squares per row, then the cluster's in block_sum's order
  for (int r = warp; r < rows; r += 4) {
    const float v = lane < jn ? so[r][lane] : 0.f;
    const float a = warp_sum(__fmul_rn(v, v));
    if (lane == 0) spart[r] = a;
  }
  cluster.sync();
  const int nb = int(cluster.num_blocks());
  for (int r = warp; r < rows; r += 4) {
    const float t = lane < nb ? *cluster.map_shared_rank(&spart[r], lane) : 0.f;
    const float inv = rsqrtf(warp_sum(t) / d.dv + eps);
    if (y.q8) {  // 8-bit rows (dv % 32 == 0, so jn == 32): the warp's 32 values, one group
      const float z = r == warp ? gate_z : proj[size_t(r) * ld + C + h * d.dv + j0 + lane];
      put_q8<false, true>(y, r, (h * d.dv + j0) >> 5, so[r][lane] * inv * gate_w * silu(z));
    } else if (lane < jn) {
      const int j = j0 + lane;
      const size_t at = size_t(r) * vd + size_t(h) * d.dv + j;
      const float z = r == warp ? gate_z : proj[size_t(r) * ld + C + h * d.dv + j];
      put<true>(y, at, at, so[r][lane] * inv * gate_w * silu(z));
    }
  }
  cluster.sync();  // the other blocks have read spart
#endif
}

}  // namespace

void gdn_conv(const float* mixed, int ld_mixed, const float* conv_w, float* win, size_t win_stride, int win_sel,
              int win_out_sel, int skip, GdnBufs bufs, int bufs_sel, const DevParams* P, int rows, cudaStream_t st,
              int snap_sel, int snap_row, const TreeRows* tree) {
  if (rows <= 0) return;
  if (snap_row < 0 || snap_row > rows) snap_sel = -1;
  if (tree) {
    if (rows > kConvRows || win_out_sel >= 0 || snap_sel >= 0)
      throw std::runtime_error("gdn_conv: a tree is a round's rows, its window folded at the commit");
    if (bufs.d.ck == 4)
      k_gdn_conv_tree<4><<<ceil_div(bufs.d.C(), 256), 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel,
                                                                     bufs, bufs_sel, P, rows, tree);
    else
      k_gdn_conv_tree<0><<<ceil_div(bufs.d.C(), 256), 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel,
                                                                     bufs, bufs_sel, P, rows, tree);
    return;
  }
  const dim3 grid(ceil_div(bufs.d.C(), 256), ceil_div(rows, kConvRows));
  if (bufs.d.ck == 4)
    k_gdn_conv<4><<<grid, 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel, win_out_sel, skip, bufs,
                                        bufs_sel, P, rows, snap_sel, snap_row);
  else
    k_gdn_conv<0><<<grid, 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel, win_out_sel, skip, bufs,
                                        bufs_sel, P, rows, snap_sel, snap_row);
}

namespace {
bool gdn_round_supported(const KernelOptions& opt, const GdnDims& d, int rows, const float* S, size_t s_stride) {
  const bool aligned = reinterpret_cast<uintptr_t>(S) % 16 == 0 && s_stride % 4 == 0;
  return !d.sigmoid_output && opt.gdn_round && sm90_or_newer() && aligned && rows >= 1 && rows <= kGdnRoundRows && d.nk > 0 && d.nv % d.nk == 0 &&
         (d.dk == 16 || d.dk == 32 || d.dk == 64 || d.dk == 128) && d.dv <= 8 * kGdnJB && d.ck >= 1 && d.ck <= 8;
}
}  // namespace

// grid (nv, ceil(dv / kGdnJB)), one head per cluster.  Launched as a
// programmatic dependent of the input projection: its state rows load while
// that GEMM finishes (+1% plain decode).
template <int DK, int CK, int RT, class... Args>
void launch_gdn_round(const GdnDims& d, cudaStream_t st, Args... args) {
  const int ny = ceil_div(d.dv, kGdnJB);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(d.nv, ny);
  cfg.blockDim = dim3(128);
  cfg.stream = st;
  cudaLaunchAttribute at[2]{};
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 1;
  at[0].val.clusterDim.y = ny;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  cudaLaunchKernelEx(&cfg, k_gdn_round<DK, CK, RT>, args...);
}

template <int DK, int RT, class... Args>
void launch_gdn_round_ck(const GdnDims& d, cudaStream_t st, Args... args) {
  if (d.ck == 4) launch_gdn_round<DK, 4, RT>(d, st, args...);
  else launch_gdn_round<DK, 0, RT>(d, st, args...);
}

bool gdn_round_fits(const GdnDims& d, int rows, const float* S, size_t s_stride, const Launch& L, bool q8) {
  return gdn_round_supported(opts(L), d, rows, S, s_stride) && (!q8 || d.dv % 32 == 0);
}

bool gdn_round(const float* proj, int ld, const float* conv_w, float* win, size_t win_stride, float* S,
               size_t s_stride, int next_sel, const float* A_log, const float* dt_bias, const float* gw, float eps,
               GdnBufs bufs, const DevParams* P, int rows, float* y, const Launch& L, bool q8,
               int total_rows, int row_offset, float* also) {
  if (!gdn_round_fits(bufs.d, rows, S, s_stride, L, q8)) return false;
  const cudaStream_t st = L.st;
  Out out = y ? Out{y, nullptr, nullptr}
              : split_out_slice(L, total_rows > 0 ? total_rows : rows, row_offset, bufs.d.nv * bufs.d.dv, q8);
  if (!y) out.also = also;
  auto launch = [&](auto dk) {
    constexpr int DK = decltype(dk)::value;
    if (rows <= kGdnRoundFew)
      launch_gdn_round_ck<DK, kGdnRoundFew>(bufs.d, st, proj, ld, conv_w, win, win_stride, S, s_stride, next_sel,
                                            A_log, dt_bias, gw, eps, bufs, P, rows, out);
    else
      launch_gdn_round_ck<DK, kGdnRoundRows>(bufs.d, st, proj, ld, conv_w, win, win_stride, S, s_stride, next_sel,
                                             A_log, dt_bias, gw, eps, bufs, P, rows, out);
    return true;
  };
  switch (bufs.d.dk) {
    case 16: return launch(std::integral_constant<int, 16>{});
    case 32: return launch(std::integral_constant<int, 32>{});
    case 64: return launch(std::integral_constant<int, 64>{});
    case 128: return launch(std::integral_constant<int, 128>{});
    default: return false;
  }
}

void gdn_prep(GdnBufs bufs, int bufs_sel, const float* b, const float* a, int ld_ba, const float* A_log,
              const float* dt_bias, const DevParams* P, int rows, cudaStream_t st) {
  const int r_step = rows >= 64 ? 4 : 1;
  k_gdn_prep<<<dim3(bufs.d.nk + 1, ceil_div(rows, r_step)), 128, 0, st>>>(bufs, bufs_sel, b, a, ld_ba, A_log, dt_bias,
                                                                          P, rows, r_step);
}

void gdn_prep(GdnBufs bufs, int bufs_sel, const bf16* b, const bf16* a, int ld_ba, const float* A_log,
              const float* dt_bias, const DevParams* P, int rows, cudaStream_t st) {
  const int r_step = rows >= 64 ? 4 : 1;
  k_gdn_prep<bf16><<<dim3(bufs.d.nk + 1, ceil_div(rows, r_step)), 128, 0, st>>>(bufs, bufs_sel, b, a, ld_ba, A_log,
                                                                                dt_bias, P, rows, r_step);
}

void gdn_conv(const bf16* mixed, int ld_mixed, const float* conv_w, float* win, size_t win_stride, int win_sel,
              int win_out_sel, GdnBufs bufs, int bufs_sel, const DevParams* P, int rows, cudaStream_t st, int snap_sel,
              int snap_row) {
  if (rows <= 0) return;
  if (snap_row < 0 || snap_row > rows) snap_sel = -1;
  const dim3 grid(ceil_div(bufs.d.C(), 256), ceil_div(rows, kConvRows));
  if (bufs.d.ck == 4)
    k_gdn_conv<4, bf16><<<grid, 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel, win_out_sel,
                                              kSkipNever, bufs, bufs_sel, P, rows, snap_sel, snap_row);
  else
    k_gdn_conv<0, bf16><<<grid, 256, 0, st>>>(mixed, ld_mixed, conv_w, win, win_stride, win_sel, win_out_sel,
                                              kSkipNever, bufs, bufs_sel, P, rows, snap_sel, snap_row);
}

namespace {
// Room for two 37 KB blocks of k_gdn_recur_lanes<DK> (or four 20.7 KB fold blocks) per SM, set once before first launch.
template <int DK>
void lanes_carveout() {
  static const bool done = [&] {
    cudaFuncSetAttribute(k_gdn_recur_lanes<DK>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         int(cudaSharedmemCarveoutMaxShared));
    cudaFuncSetAttribute(k_gdn_recur_lanes<DK, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         int(cudaSharedmemCarveoutMaxShared));
    cudaFuncSetAttribute(k_gdn_recur_lanes<DK, false, true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         int(cudaSharedmemCarveoutMaxShared));
    return true;
  }();
  (void)done;
}
}  // namespace

void gdn_recur(float* S, size_t s_stride, int in_sel, int out_sel, GdnBufs bufs, int bufs_sel, const DevParams* P,
               float* out, int rows, const int* rows_dev, int skip_if_noop, cudaStream_t st) {
  // Lane groups for key dims 16/32/64/128 (the state rows are read as float4);
  // the choice depends only on the shapes and the state buffer, so every
  // call of a session uses the same kernel.
  const GdnDims& d = bufs.d;
  const bool aligned = reinterpret_cast<uintptr_t>(S) % 16 == 0 && s_stride % 4 == 0;
  const dim3 lanes_grid(d.nv, ceil_div(d.dv, kGdnJB));
#define FLYX_GDN_LANES(DK)                                                                                            \
  case DK: {                                                                                                          \
    lanes_carveout<DK>();                                                                                             \
    if (out)                                                                                                          \
      k_gdn_recur_lanes<DK><<<lanes_grid, 128, 0, st>>>(S, s_stride, in_sel, out_sel, bufs, bufs_sel, P, out, rows,   \
                                                        rows_dev, skip_if_noop, nullptr);                             \
    else                                                                                                              \
      k_gdn_recur_lanes<DK, false, true><<<lanes_grid, 128, 0, st>>>(S, s_stride, in_sel, out_sel, bufs, bufs_sel, P, \
                                                                     nullptr, rows, rows_dev, skip_if_noop, nullptr); \
    return;                                                                                                           \
  }
  if (aligned) switch (d.dk) {
      FLYX_GDN_LANES(16)
      FLYX_GDN_LANES(32)
      FLYX_GDN_LANES(64)
      FLYX_GDN_LANES(128)
      default: break;
    }
#undef FLYX_GDN_LANES
  k_gdn_recur<<<dim3(d.nv, ceil_div(d.dv, 8)), 256, 0, st>>>(S, s_stride, in_sel, out_sel, bufs, bufs_sel, P, out,
                                                             rows, rows_dev, skip_if_noop);
}

bool gdn_recur_tree(const float* S, size_t s_stride, int in_sel, GdnBufs bufs, int bufs_sel, const DevParams* P,
                    float* out, int rows, const TreeRows* tree, cudaStream_t st) {
  // gdn_recur's lanes kernel (the same steps), its TREE instance; the shapes it takes.
  const GdnDims& d = bufs.d;
  const bool aligned = reinterpret_cast<uintptr_t>(S) % 16 == 0 && s_stride % 4 == 0;
  if (!aligned || rows > kGdnRT) return false;
  const dim3 grid(d.nv, ceil_div(d.dv, kGdnJB));
  float* Sm = const_cast<float*>(S);  // read only: out_sel < 0
#define FLYX_GDN_TREE(DK)                                                                                          \
  case DK:                                                                                                           \
    lanes_carveout<DK>();                                                                                            \
    k_gdn_recur_lanes<DK, true><<<grid, 128, 0, st>>>(Sm, s_stride, in_sel, -1, bufs, bufs_sel, P, out, rows, nullptr, \
                                                      kSkipNever, nullptr, tree);                                    \
    return true;
  switch (d.dk) {
    FLYX_GDN_TREE(16)
    FLYX_GDN_TREE(32)
    FLYX_GDN_TREE(64)
    FLYX_GDN_TREE(128)
    default: return false;
  }
#undef FLYX_GDN_TREE
}

void gdn_compact(const GdnLayer* layers, int n, GdnBufs bufs, const DevParams* P, const TreeRows* tree,
                 cudaStream_t st) {
  k_gdn_compact<<<dim3(ceil_div(bufs.d.C(), 256), layers ? n : 1), 256, 0, st>>>(bufs, layers, P, tree);
}

void gdn_conv_fold(float* win, size_t win_stride, int in_sel, int out_sel, GdnBufs bufs, int bufs_sel,
                   const DevParams* P, int rows, const int* rows_dev, bool skip_if_noop, cudaStream_t st) {
  const int blocks = ceil_div(bufs.d.C(), 256);
  if (bufs.d.ck == 4)
    k_gdn_conv_fold<4><<<blocks, 256, 0, st>>>(win, win_stride, in_sel, out_sel, bufs, bufs_sel, P, rows, rows_dev,
                                               skip_if_noop, nullptr);
  else
    k_gdn_conv_fold<0><<<blocks, 256, 0, st>>>(win, win_stride, in_sel, out_sel, bufs, bufs_sel, P, rows, rows_dev,
                                               skip_if_noop, nullptr);
}

bool gdn_fold_layers(const GdnLayer* layers, int n, size_t s_stride, size_t w_stride, GdnBufs bufs, int in_sel,
                     int out_sel, int bufs_sel, const DevParams* P, const int* rows_dev, cudaStream_t st) {
  const GdnDims& d = bufs.d;
  const dim3 lanes_grid(d.nv, ceil_div(d.dv, kGdnJB), n);
  switch (d.dk) {  // the kernels gdn_recur takes for these key dims (an aligned state: gdn_fold_layers' caller checks)
    case 16:
      lanes_carveout<16>();
      k_gdn_recur_lanes<16, false, true><<<lanes_grid, 128, 0, st>>>(nullptr, s_stride, in_sel, out_sel, bufs, bufs_sel,
                                                                     P, nullptr, 0, rows_dev, kSkipIfNoop, layers);
      break;
    case 32:
      lanes_carveout<32>();
      k_gdn_recur_lanes<32, false, true><<<lanes_grid, 128, 0, st>>>(nullptr, s_stride, in_sel, out_sel, bufs, bufs_sel,
                                                                     P, nullptr, 0, rows_dev, kSkipIfNoop, layers);
      break;
    case 64:
      lanes_carveout<64>();
      k_gdn_recur_lanes<64, false, true><<<lanes_grid, 128, 0, st>>>(nullptr, s_stride, in_sel, out_sel, bufs, bufs_sel,
                                                                     P, nullptr, 0, rows_dev, kSkipIfNoop, layers);
      break;
    case 128:
      lanes_carveout<128>();
      k_gdn_recur_lanes<128, false, true><<<lanes_grid, 128, 0, st>>>(
          nullptr, s_stride, in_sel, out_sel, bufs, bufs_sel, P, nullptr, 0, rows_dev, kSkipIfNoop, layers);
      break;
    default:
      return false;
  }
  const dim3 conv_grid(ceil_div(d.C(), 256), n);
  if (d.ck == 4)
    k_gdn_conv_fold<4><<<conv_grid, 256, 0, st>>>(nullptr, w_stride, in_sel, out_sel, bufs, bufs_sel, P, 0, rows_dev, 1,
                                                  layers);
  else
    k_gdn_conv_fold<0><<<conv_grid, 256, 0, st>>>(nullptr, w_stride, in_sel, out_sel, bufs, bufs_sel, P, 0, rows_dev, 1,
                                                  layers);
  return true;
}

void gdn_gated_norm(const float* o, const float* z, int ld_z, const float* w, float eps, float* y, int rows,
                    GdnDims d, cudaStream_t st) {
  const int r_step = rows >= 64 ? 4 : 1;
  k_gdn_gated_norm<<<dim3(d.nv, ceil_div(rows, r_step)), (d.dv + 31) / 32 * 32, 0, st>>>(
      o, z, ld_z, w, eps, Out{y, nullptr, nullptr}, d.nv, d.dv, rows, r_step, d.sigmoid_output);
}

void gdn_gated_norm_split(const float* o, const float* z, int ld_z, const float* w, float eps, int rows, GdnDims d,
                          const Launch& L, bool q8) {
  const int r_step = rows >= 64 ? 4 : 1;
  k_gdn_gated_norm<<<dim3(d.nv, ceil_div(rows, r_step)), (d.dv + 31) / 32 * 32, 0, L.st>>>(
      o, z, ld_z, w, eps, split_out(L, rows, d.nv * d.dv, q8), d.nv, d.dv, rows, r_step, d.sigmoid_output);
}

void gdn_gated_norm_split(const float* o, const bf16* z, int ld_z, const float* w, float eps, int rows, GdnDims d,
                          const Launch& L, bool q8) {
  const int r_step = rows >= 64 ? 4 : 1;
  k_gdn_gated_norm<bf16><<<dim3(d.nv, ceil_div(rows, r_step)), (d.dv + 31) / 32 * 32, 0, L.st>>>(
      o, z, ld_z, w, eps, split_out(L, rows, d.nv * d.dv, q8), d.nv, d.dv, rows, r_step, d.sigmoid_output);
}

}  // namespace flyx::cuda
