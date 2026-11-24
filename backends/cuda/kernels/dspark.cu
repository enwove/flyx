// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DSpark speculative drafter kernels (Inkling-2): context tap mixing + RMSNorm, block init,
// block-uniform SWA/Full attention, and Markov RNN head steps.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

constexpr int kMixCache = 10;  // 512 * 10 = 5120 floats cached in registers per row

__global__ void __launch_bounds__(512) k_dspark_ctx_mix_norm(const float* __restrict__ feats, int ld_feats,
                                                             const float* __restrict__ mix,
                                                             const float* __restrict__ norm_w, float eps, int n_taps,
                                                             int H, float* __restrict__ out) {
  pdl_trigger();
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* fr = feats + size_t(r) * ld_feats;
  float* dst = out + size_t(r) * H;

  float mv[kMixCache];
  float ss = 0.f;
#pragma unroll
  for (int u = 0; u < kMixCache; ++u) {
    const int i = tid + u * 512;
    if (i < H) {
      float acc = 0.f;
      for (int t = 0; t < n_taps; ++t) acc = __fmaf_rn(mix[t], fr[size_t(t) * H + i], acc);
      mv[u] = acc;
      ss = __fmaf_rn(acc, acc, ss);
    }
  }
  for (int i = tid + kMixCache * 512; i < H; i += 512) {
    float acc = 0.f;
    for (int t = 0; t < n_taps; ++t) acc = __fmaf_rn(mix[t], fr[size_t(t) * H + i], acc);
    ss = __fmaf_rn(acc, acc, ss);
  }
  const float inv = rsqrtf(block_sum(ss) / float(H) + eps);
#pragma unroll
  for (int u = 0; u < kMixCache; ++u) {
    const int i = tid + u * 512;
    if (i < H) dst[i] = mv[u] * inv * norm_w[i];
  }
  for (int i = tid + kMixCache * 512; i < H; i += 512) {
    float acc = 0.f;
    for (int t = 0; t < n_taps; ++t) acc = __fmaf_rn(mix[t], fr[size_t(t) * H + i], acc);
    dst[i] = acc * inv * norm_w[i];
  }
}

__global__ void __launch_bounds__(512) k_dspark_prep_block(const DevParams* P, const float* __restrict__ anchor_emb,
                                                           const float* __restrict__ tok_norm_w, float eps,
                                                           const float* __restrict__ mask_embd, int B, int H,
                                                           float* __restrict__ dx, int* __restrict__ prev_out,
                                                           float* __restrict__ state_out, int markov_rank) {
  pdl_trigger();
  const int r = blockIdx.x, tid = threadIdx.x;
  if (r == 0) {
    float ss = 0.f;
    for (int i = tid; i < H; i += 512) {
      const float v = anchor_emb[i];
      ss = __fmaf_rn(v, v, ss);
    }
    const float inv = rsqrtf(block_sum(ss) / float(H) + eps);
    for (int i = tid; i < H; i += 512) dx[i] = anchor_emb[i] * inv * tok_norm_w[i];
    if (tid == 0) *prev_out = P->anchor;
    for (int i = tid; i < markov_rank; i += 512) state_out[i] = 0.f;
  } else if (r < B) {
    const float* src = mask_embd + size_t(r - 1) * H;
    float* dst = dx + size_t(r) * H;
    for (int i = tid; i < H; i += 512) dst[i] = src[i];
  }
}

// Block-uniform non-causal attention: every query in 0..B-1 attends to keys [first, base + B)
// where base = P->position and first = (window > 0 ? max(0, base - window + 1) : 0).
__device__ __forceinline__ int dspark_splits_used(int span, int max_splits) {
  return min(max_splits, max(1, (span + 63) / 64));
}

template <int HD, int RT>
__global__ void __launch_bounds__(128) k_dspark_attn_split(const float* __restrict__ q, const kv16* __restrict__ Kc,
                                                           const kv16* __restrict__ Vc, int nh, int nkv, int B,
                                                           int window, float scale, const DevParams* P,
                                                           float* __restrict__ part, int splits) {
  pdl_trigger();
  constexpr int DPL = HD / 32;
  __shared__ float s_m[4][RT], s_l[4][RT];
  __shared__ float s_acc[4][RT][HD];
  const int h = blockIdx.x, sp = blockIdx.y, r0 = blockIdx.z * RT;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int kvh = h / (nh / nkv);
  const int base = P->position;
  const int first = window > 0 ? max(0, base - window + 1) : 0;
  const int total = base + B;
  const int span = max(0, total - first);
  const int used = dspark_splits_used(span, splits);
  if (sp >= used) return;

  const int nrows = min(RT, B - r0);
  const int chunk = (span + used - 1) / used;
  const int ks = min(total, first + sp * chunk);
  const int ke = min(total, ks + chunk);

  float qv[RT][DPL], acc[RT][DPL], m[RT], l[RT];
#pragma unroll
  for (int i = 0; i < RT; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.f;
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      acc[i][e] = 0.f;
      qv[i][e] = i < nrows ? q[(size_t(r0 + i) * nh + h) * HD + lane * DPL + e] : 0.f;
    }
  }

  for (int j = ks < ke ? ks + ((warp - ks) & 3) : ke; j < ke; j += 4) {
    const size_t at = (size_t(j) * nkv + kvh) * HD + lane * DPL;
    float kf[DPL], vf[DPL];
#pragma unroll
    for (int e = 0; e < DPL; ++e) {
      kf[e] = __half2float(Kc[at + e]);
      vf[e] = __half2float(Vc[at + e]);
    }
#pragma unroll
    for (int i = 0; i < RT; ++i) {
      if (i >= nrows) break;
      float d = 0.f;
#pragma unroll
      for (int e = 0; e < DPL; ++e) d = __fmaf_rn(qv[i][e], kf[e], d);
      const float s = warp_sum(d) * scale;
      const float nm = fmaxf(m[i], s);
      const float ew = m[i] == -INFINITY ? 0.f : __expf(m[i] - nm);
      const float w = __expf(s - nm);
      l[i] = l[i] * ew + w;
#pragma unroll
      for (int e = 0; e < DPL; ++e) acc[i][e] = acc[i][e] * ew + w * vf[e];
      m[i] = nm;
    }
  }

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
          const float f = __expf(s_m[w][i] - M);
          L += s_l[w][i] * f;
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

__global__ void k_dspark_attn_combine(const float* __restrict__ part, int nh, int hd, int B, int window, int splits,
                                      const DevParams* P, float* __restrict__ out, int ld_out) {
  pdl_trigger();
  const int h = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
  const int base = P->position;
  const int first = window > 0 ? max(0, base - window + 1) : 0;
  const int used = dspark_splits_used(max(0, base + B - first), splits);
  const float* pr = part + (size_t(r) * nh + h) * splits * (hd + 2);
  float M = -INFINITY;
  for (int s = 0; s < used; ++s) M = fmaxf(M, pr[s * (hd + 2)]);
  float L = 0.f, A = 0.f;
  if (M != -INFINITY) {
    for (int s = 0; s < used; ++s) {
      const float ms = pr[s * (hd + 2)];
      if (ms == -INFINITY) continue;
      const float f = __expf(ms - M);
      L += pr[s * (hd + 2) + 1] * f;
      A = __fmaf_rn(pr[s * (hd + 2) + 2 + d], f, A);
    }
  }
  out[size_t(r) * ld_out + h * hd + d] = L > 0.f ? (A / L) : 0.f;
}

template <int HD>
void dispatch_dspark_attn(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int B, int window,
                          float scale, const DevParams* P, float* out, int ld_out, float* partial, int splits,
                          const Launch& L) {
  const int eff = std::max(1, splits);
  k_dspark_attn_split<HD, 4><<<dim3(nh, eff, ceil_div(B, 4)), 128, 0, L.st>>>(q, Kc, Vc, nh, nkv, B, window, scale, P,
                                                                               partial, eff);
  k_dspark_attn_combine<<<dim3(nh, B), HD, 0, L.st>>>(partial, nh, HD, B, window, eff, P, out, ld_out);
}

__global__ void k_dspark_markov_pack(const DevParams* P, const float* __restrict__ state,
                                     const float* __restrict__ w1_row, const float* __restrict__ hidden_i,
                                     const float* __restrict__ hring, int Rk, int H, float* __restrict__ joint_out) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = 2 * Rk + 2 * H;
  if (i >= total) return;
  if (i < Rk) {
    joint_out[i] = state[i];
  } else if (i < 2 * Rk) {
    if (w1_row) joint_out[i] = w1_row[i - Rk];
  } else if (i < 2 * Rk + H) {
    joint_out[i] = hidden_i[i - 2 * Rk];
  } else {
    const int c = i - (2 * Rk + H);
    const int ctx = P->position;
    joint_out[i] = (ctx > 0 && hring) ? hring[size_t((ctx - 1) % kHring) * H + c] : 0.f;
  }
}

__global__ void k_dspark_markov_gate(const float* __restrict__ joint_out, const float* __restrict__ joint_bias, int Rk,
                                     float* __restrict__ state, float* __restrict__ markov_out) {
  pdl_trigger();
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= Rk) return;
  const float j0 = joint_out[j] + (joint_bias ? joint_bias[j] : 0.f);
  const float j1 = joint_out[Rk + j] + (joint_bias ? joint_bias[Rk + j] : 0.f);
  const float j2 = joint_out[2 * Rk + j] + (joint_bias ? joint_bias[2 * Rk + j] : 0.f);
  const float g = sigm(j0);
  const float c = tanhf(j1);
  state[j] = c + g * (state[j] - c);
  markov_out[j] = tanhf(j2);
}

__global__ void k_dspark_combine_logits(const float* __restrict__ base, const float* __restrict__ bias,
                                        const float* __restrict__ w2_b, int V, float softcap,
                                        float* __restrict__ out) {
  pdl_trigger();
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V) return;
  float z = base[v] + bias[v] + (w2_b ? w2_b[v] : 0.f);
  if (softcap > 0.f) z = softcap * tanhf(z / softcap);
  out[v] = z;
}

__global__ void k_dspark_pick_step(const DevParams* P, const int* __restrict__ oracle, int step, int* prev_tok,
                                   int* drafts) {
  pdl_trigger();
  if (threadIdx.x != 0) return;
  if (P->oracle_len > 0) {
    const int pos = P->position + 1 + step;
    const int tok = pos < P->oracle_len ? oracle[pos] : 0;
    drafts[step] = tok;
    *prev_tok = tok;
  } else {
    drafts[step] = *prev_tok;
  }
}

}  // namespace

void dspark_ctx_mix_norm(const float* feats, int ld_feats, const float* mix, const float* norm_w, float eps, int n_taps,
                         int rows, int H, float* out, cudaStream_t st) {
  if (rows <= 0 || H <= 0) return;
  k_dspark_ctx_mix_norm<<<rows, 512, 0, st>>>(feats, ld_feats, mix, norm_w, eps, n_taps, H, out);
}

void dspark_prep_block(const DevParams* P, const float* anchor_emb, const float* tok_norm_w, float eps,
                       const float* mask_embd, int B, int H, float* dx, int* prev_out, float* state_out,
                       int markov_rank, cudaStream_t st) {
  k_dspark_prep_block<<<B, 512, 0, st>>>(P, anchor_emb, tok_norm_w, eps, mask_embd, B, H, dx, prev_out, state_out,
                                         markov_rank);
}

void dspark_block_attention(const float* q, const kv16* Kc, const kv16* Vc, int nh, int nkv, int hd, int B, int window,
                            float scale, const DevParams* P, float* out, int ld_out, float* partial, int splits,
                            const Launch& L) {
  switch (hd) {
    case 32: dispatch_dspark_attn<32>(q, Kc, Vc, nh, nkv, B, window, scale, P, out, ld_out, partial, splits, L); return;
    case 64: dispatch_dspark_attn<64>(q, Kc, Vc, nh, nkv, B, window, scale, P, out, ld_out, partial, splits, L); return;
    case 128: dispatch_dspark_attn<128>(q, Kc, Vc, nh, nkv, B, window, scale, P, out, ld_out, partial, splits, L); return;
    case 256: dispatch_dspark_attn<256>(q, Kc, Vc, nh, nkv, B, window, scale, P, out, ld_out, partial, splits, L); return;
    default: throw std::runtime_error("dspark_block_attention: unsupported head dim " + std::to_string(hd));
  }
}

void dspark_markov_pack(const DevParams* P, const float* state, const float* w1_row, const float* hidden_i,
                        const float* hring, int Rk, int H, float* joint_out, cudaStream_t st) {
  const int total = 2 * Rk + 2 * H;
  k_dspark_markov_pack<<<ceil_div(total, 256), 256, 0, st>>>(P, state, w1_row, hidden_i, hring, Rk, H, joint_out);
}

void dspark_markov_gate(const float* joint_out, const float* joint_bias, int Rk, float* state, float* markov_out,
                        cudaStream_t st) {
  k_dspark_markov_gate<<<ceil_div(Rk, 256), 256, 0, st>>>(joint_out, joint_bias, Rk, state, markov_out);
}

void dspark_combine_logits(const float* base, const float* bias, const float* w2_b, int V, float softcap, float* out,
                           cudaStream_t st) {
  k_dspark_combine_logits<<<ceil_div(V, 256), 256, 0, st>>>(base, bias, w2_b, V, softcap, out);
}

void dspark_pick_step(const DevParams* P, const int* oracle, int step, int* prev_tok, int* drafts, cudaStream_t st) {
  k_dspark_pick_step<<<1, 1, 0, st>>>(P, oracle, step, prev_tok, drafts);
}

}  // namespace flyx::cuda
