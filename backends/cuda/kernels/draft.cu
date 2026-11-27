// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Drafting kernels: MTP inputs and hidden ring, the DFlash2 grouped convs, selector walk and tree, context K/V.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "ops/sampling/sampling.h"

namespace flyx::cuda {
namespace {

__global__ void k_grouped_conv(const float* x, int ldx, const float* delta, int ld_delta, const float* base, int H,
                               int taps, int group, float* y, int ldy, int accumulate) {
  pdl_trigger();
  const int r = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const int G = H / group;
  float acc = 0.f;
  for (int t = 0; t < taps && t <= r; ++t)
    acc += (base[size_t(t) * H + c] + delta[size_t(r) * ld_delta + t * G + c / group]) * x[size_t(r - t) * ldx + c];
  float* dst = y + size_t(r) * ldy + c;
  *dst = accumulate ? *dst + acc : acc;
}

__device__ __forceinline__ float dflash2_score(float u, float e, float softcap, int softcap_mode) {
  if (softcap > 0.f) {
    return softcap_mode == FLYX_DFLASH2_SOFTCAP_AFTER ? softcap * tanhf((u + e) / softcap)
                                                      : softcap * tanhf(u / softcap) + e;
  }
  return u + e;
}

// One block of 32 * min(k, 32) threads: warp w scores candidates w, w + 32, ...; thread 0 takes the best (the first
// on ties).  Sampling (G's temperature > 0): the scores coupled to the target's noise (ops::coupled_draft_score).
__global__ void k_selector_walk(const DevParams* P, const int* cand, const float* unary, int k, const float* hid,
                                int rank, const bf16* pred, const bf16* succ, int width, const int* oracle,
                                int* drafts, float softcap, int softcap_mode, const DevGen* G) {
  pdl_trigger();
  const float T = G ? G->temperature : 0.f;
  __shared__ float score[kWalkMax];
  __shared__ int prev;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  if (P->oracle_len > 0) {
    if (threadIdx.x < width) {
      const int pos = P->position + 1 + threadIdx.x;
      drafts[threadIdx.x] = pos < P->oracle_len ? oracle[pos] : 0;
    }
    return;
  }
  if (threadIdx.x == 0) prev = P->anchor;
  __syncthreads();
  for (int s = 0; s < width; ++s) {
    // rank 0 (no selector): the candidates' logits alone, no selector weights read
    const bf16* pr = rank ? pred + size_t(prev) * rank : nullptr;
    const float* hs = rank ? hid + size_t(s) * rank : nullptr;
    for (int c = warp; c < k; c += warps) {
      const int cid = cand[s * k + c];
      const bf16* sc = rank ? succ + size_t(cid) * rank : nullptr;
      float e = 0.f;
      for (int i = lane; i < rank; i += 32) e += __bfloat162float(pr[i]) * hs[i] * __bfloat162float(sc[i]);
      e = warp_sum(e);
      if (lane == 0) {
        const float sc = dflash2_score(unary[s * k + c], e, softcap, softcap_mode);
        score[c] = T > 0.f ? ops::coupled_draft_score(sc, T, G->seed, int64_t(P->position) + 1 + s, cid) : sc;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (score[c] > score[best]) best = c;
      prev = cand[s * k + best];
      drafts[s] = prev;
    }
    __syncthreads();
  }
}

// ops/draft/ref.h's best_first_tree on the selector's scores (the walk's chain, then branches best first), the tree
// written as the round reads it (TreeRows).  One block of 32 * min(k, 32) threads: warp w scores candidates w, w + 32,
// ... of the node being expanded (as k_selector_walk does), the block normalises them, warp 0 picks the next node.
// With the test's drafts (P->oracle_len), a chain of them.
__global__ void k_selector_tree(const DevParams* P, const int* cand, const float* unary, int k, const float* hid,
                                int rank, const bf16* pred, const bf16* succ, int width, int depths,
                                const int* oracle, int* drafts, TreeRows* tree, float softcap, int softcap_mode,
                                const DevGen* G) {
  pdl_trigger();
  constexpr int kRows = FLYX_MAX_ROWS;
  extern __shared__ float tree_smem[];
  float* val = tree_smem;              // [kRows * k]: node n's child c at n * k + c; -inf: none, or taken
  float* score = tree_smem + kRows * k;  // [k]: the node being expanded, its candidates' scores
  __shared__ int row_tok[kRows], row_parent[kRows], row_depth[kRows];
  __shared__ float row_val[kRows];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
  for (int i = threadIdx.x; i < kRows * k; i += blockDim.x) val[i] = -INFINITY;
  for (int i = threadIdx.x; i < kRows * kRows; i += blockDim.x) tree->path[i / kRows][i % kRows] = 0;
  if (threadIdx.x == 0) {
    row_tok[0] = P->anchor;
    row_parent[0] = -1;
    row_depth[0] = 0;
    row_val[0] = 0.f;
  }
  __syncthreads();
  if (P->oracle_len > 0) {
    if (threadIdx.x == 0)
      for (int i = 0; i < width; ++i) {
        const int pos = P->position + 1 + i;
        drafts[i] = pos < P->oracle_len ? oracle[pos] : 0;
        row_parent[i + 1] = i;
        row_depth[i + 1] = i + 1;
      }
  } else {
    const float T = G ? G->temperature : 0.f;
    for (int n = 0; n < width; ++n) {
      const int d = row_depth[n];
      if (d < depths) {  // node n's children: their log-probabilities over the node's k candidates
        const bf16* pr = rank ? pred + size_t(row_tok[n]) * rank : nullptr;
        const float* hs = rank ? hid + size_t(d) * rank : nullptr;
        for (int c = warp; c < k; c += warps) {
          const bf16* sc = rank ? succ + size_t(cand[d * k + c]) * rank : nullptr;
          float e = 0.f;
          for (int i = lane; i < rank; i += 32) e += __bfloat162float(pr[i]) * hs[i] * __bfloat162float(sc[i]);
          e = warp_sum(e);
          if (lane == 0) score[c] = dflash2_score(unary[d * k + c], e, softcap, softcap_mode);
        }
        __syncthreads();
        float m = -INFINITY;
        for (int c = threadIdx.x; c < k; c += blockDim.x) m = fmaxf(m, score[c]);
        m = block_max(m);
        float z = 0.f;
        for (int c = threadIdx.x; c < k; c += blockDim.x) z += expf(score[c] - m);
        z = block_sum(z);
        for (int c = threadIdx.x; c < k; c += blockDim.x) val[n * k + c] = row_val[n] + (score[c] - m - logf(z));
      }
      __syncthreads();
      if (T > 0.f && n < min(width, depths)) {  // the trunk, sampling: the walk's coupled choice (k_selector_walk)
        if (threadIdx.x == 0) {
          int best = 0;
          float bs = -INFINITY;
          for (int c = 0; c < k; ++c) {
            const float v =
                ops::coupled_draft_score(score[c], T, G->seed, int64_t(P->position) + 1 + d, cand[d * k + c]);
            if (v > bs) bs = v, best = c;
          }
          const int r = n + 1, bi = n * k + best;
          row_parent[r] = n;
          row_tok[r] = cand[d * k + best];
          row_val[r] = val[bi];
          val[bi] = -INFINITY;
          row_depth[r] = d + 1;
          drafts[n] = row_tok[r];
        }
        __syncthreads();
        continue;
      }
      if (warp == 0) {  // the highest-valued child: ties to the lower index (the earlier node, the lower candidate)
        // The trunk (the walk's chain): node n's children only, node n being the trunk's last.
        const int from = n < min(width, depths) ? n * k : 0;
        float bv = -INFINITY;
        int bi = -1;
        for (int i = from + lane; i < (n + 1) * k; i += 32)
          if (val[i] > bv) bv = val[i], bi = i;
        for (int o = 16; o > 0; o >>= 1) {
          const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
          const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
          if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) bv = ov, bi = oi;
        }
        if (lane == 0) {
          const int r = n + 1;
          if (bi < 0) {  // no child left: the anchor's first candidate again
            row_parent[r] = 0;
            row_tok[r] = cand[0];
            row_val[r] = -INFINITY;
          } else {
            const int u = bi / k;
            row_parent[r] = u;
            row_tok[r] = cand[row_depth[u] * k + bi % k];
            row_val[r] = bv;
            val[bi] = -INFINITY;
          }
          row_depth[r] = row_depth[row_parent[r]] + 1;
          drafts[n] = row_tok[r];
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {  // as set_tree writes a given tree
    tree->n = width + 1;
    tree->n_taken = 0;
    tree->depth[0] = 0;
    for (int r = 1; r <= width; ++r) {
      const int p = row_parent[r];
      tree->depth[r] = row_depth[r];
      for (int d = 0; d < row_depth[p] + 1; ++d) tree->path[r][d] = tree->path[p][d];
      tree->path[r][row_depth[r]] = static_cast<unsigned char>(r);
    }
  }
}

template <class TIn>
__global__ void k_tap_copy(const TIn* x, int H, float* feats, int ld, int slot) {
  pdl_trigger();
  for (int i = threadIdx.x; i < H; i += blockDim.x)
    feats[size_t(blockIdx.x) * ld + size_t(slot) * H + i] = in_value(x + size_t(blockIdx.x) * H + i);
}

// grid (layers * nkv, rows); block hd.
__global__ void k_draft_ctx_kv(const float* kv, int ld, const float* k_norms, float eps, double theta, int nkv, int hd,
                               const DevParams* P, int pos_sel, int skip_if_noop, kv16* dK, kv16* dV,
                               size_t layer_stride, const double* rope, int ring) {
  pdl_trigger();
  extern __shared__ float sh[];
  if (skip_if_noop && P->noop) return;
  const int layer = blockIdx.x / nkv, kh = blockIdx.x - layer * nkv, r = blockIdx.y, d = threadIdx.x;
  const int pos = (pos_sel == kPosStart ? P->start : P->position) + r;
  const float* row = kv + size_t(r) * ld + size_t(layer) * 2 * nkv * hd;
  const float v = row[kh * hd + d];
  const float inv = rsqrtf(block_sum(v * v) / hd + eps);
  const float nv = v * inv * k_norms[layer * hd + d];
  sh[d] = nv;
  __syncthreads();
  const int half = hd / 2, i = d < half ? d : d - half;
  float c, s;
  if (rope) rope_cs_inv(pos, rope[i], &c, &s);  // rope_table's factor, or its pow
  else rope_cs(pos, i, hd, theta, &c, &s);
  const float out = d < half ? nv * c - sh[d + half] * s : nv * c + sh[d - half] * s;
  const int cell = ring > 0 ? pos % ring : pos;
  const size_t at = size_t(layer) * layer_stride + (size_t(cell) * nkv + kh) * hd + d;
  dK[at] = kv16_of(out);
  dV[at] = kv16_of(row[nkv * hd + kh * hd + d]);
}

constexpr int kMtpThreads = 512;
constexpr int kMtpCache = 10;  // 512 * 10 = 5120: single-pass register cache covers hidden_dim up to 5120

__global__ void __launch_bounds__(kMtpThreads) k_mtp_gather(const DevParams* P, int pos_sel, int skip_if_noop,
                                                             const float* h0, const float* hring, const float* hn,
                                                             int ldh, const int* tokens, const bf16* E,
                                                             const float* emb, const float* we, const float* wh,
                                                             float offset, float eps, int H, float* cat) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int base = pos_sel == kPosStart ? P->start : P->position;
  const float* h = r > 0 ? hn + size_t(r - 1) * ldh
                 : h0    ? h0
                         : hring + size_t(((base - 1) % kHring + kHring) % kHring) * H;
  const bf16* e = E ? E + size_t(tokens[r]) * H : nullptr;
  const float* ef = emb + size_t(r) * H;  // read only without a table
  const bool cached = H <= kMtpThreads * kMtpCache;
  float ev[kMtpCache], hv[kMtpCache], wev[kMtpCache], whv[kMtpCache];
  float se = 0.f, sh = 0.f;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMtpCache; ++i) {
      const int c = tid + i * kMtpThreads;
      const bool in = c < H;
      const float ei = in ? (e ? __bfloat162float(e[c]) : ef[c]) : 0.f;
      const float hi = in ? h[c] : 0.f;
      ev[i] = ei;
      hv[i] = hi;
      wev[i] = in ? we[c] : 0.f;
      whv[i] = in ? wh[c] : 0.f;
      se += ei * ei;
      sh += hi * hi;
    }
  } else {
    for (int c = tid; c < H; c += kMtpThreads) {
      const float ei = e ? __bfloat162float(e[c]) : ef[c], hi = h[c];
      se += ei * ei;
      sh += hi * hi;
    }
  }
  __shared__ float s_e[kMtpThreads / 32], s_h[kMtpThreads / 32];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    se += __shfl_xor_sync(0xffffffffu, se, o);
    sh += __shfl_xor_sync(0xffffffffu, sh, o);
  }
  if (lane == 0) {
    s_e[warp] = se;
    s_h[warp] = sh;
  }
  __syncthreads();
  float be = s_e[lane & 15], bh = s_h[lane & 15];
#pragma unroll
  for (int o = 8; o > 0; o >>= 1) {
    be += __shfl_xor_sync(0xffffffffu, be, o);
    bh += __shfl_xor_sync(0xffffffffu, bh, o);
  }
  const float ie = rsqrtf(be / H + eps);
  const float ih = rsqrtf(bh / H + eps);
  float* out = cat + size_t(r) * 2 * H;
  if (cached) {
#pragma unroll
    for (int i = 0; i < kMtpCache; ++i) {
      const int c = tid + i * kMtpThreads;
      if (c < H) {
        out[c] = ev[i] * ie * (offset + wev[i]);
        out[H + c] = hv[i] * ih * (offset + whv[i]);
      }
    }
  } else {
    for (int c = tid; c < H; c += kMtpThreads) {
      out[c] = (e ? __bfloat162float(e[c]) : ef[c]) * ie * (offset + we[c]);
      out[H + c] = h[c] * ih * (offset + wh[c]);
    }
  }
}

__global__ void k_hring_put(const DevParams* P, int pos_sel, int skip_if_noop, const float* hn, int ldh, int rows,
                            float* hring, int H) {
  pdl_trigger();
  if (skip_if_noop && P->noop) return;
  const int r = blockIdx.x;
  if (r < rows - kHring) return;  // only the last kHring rows land in the ring
  const int base = pos_sel == kPosStart ? P->start : P->position;
  float* dst = hring + size_t((base + r) % kHring) * H;
  for (int c = threadIdx.x; c < H; c += blockDim.x) dst[c] = hn[size_t(r) * ldh + c];
}

// The hidden row 0 of mtp_gather reads (h0, else the ring's at base - 1) copied to out.
__global__ void k_pair_hidden(const DevParams* P, int pos_sel, const float* h0, const float* hring, int H, float* out) {
  pdl_trigger();
  const int base = pos_sel == kPosStart ? P->start : P->position;
  const float* h = h0 ? h0 : hring + size_t(((base - 1) % kHring + kHring) % kHring) * H;
  for (int c = threadIdx.x; c < H; c += blockDim.x) out[c] = h[c];
}

__global__ void k_mtp_oracle(const DevParams* P, const int* oracle, int step, int* mtok) {
  pdl_trigger();
  if (P->oracle_len <= 0) return;
  const int pos = P->position + 1 + step;
  mtok[step + 1] = pos < P->oracle_len ? oracle[pos] : 0;
}

__global__ void k_mtp_choose(const DevParams* P, const DevGen* G, const int* reply_count, const int* cand,
                             const float* val, int k, int vocab, int step, int* mtok) {
  pdl_trigger();
  // each candidate's count as the row verifying this draft counts it: the reply's and the round's drafts before it
  // (a candidate outside the vocabulary is none: draft_choice skips it)
  int count[ops::kMtpCands];
  const bool pen = G->presence != 0.f || G->frequency != 0.f;
  for (int c = 0; pen && c < k; ++c) {
    if (cand[c] < 0 || cand[c] >= vocab) continue;
    count[c] = reply_count[cand[c]];
    for (int j = 1; j <= step; ++j) count[c] += mtok[j] == cand[c];
  }
  mtok[step + 1] = ops::draft_choice(cand, val, pen ? count : nullptr, k, vocab, G->temperature, G->top_p, G->top_k,
                                     G->presence, G->frequency, G->seed, int64_t(P->position) + 1 + step);
}

__global__ void k_given_drafts(const DevParams* P, const int* oracle, int width, int* drafts) {
  pdl_trigger();
  const int i = threadIdx.x;
  if (i >= width) return;
  const int pos = P->position + 1 + i;
  drafts[i] = pos < P->oracle_len ? oracle[pos] : 0;
}

__global__ void k_copy_ints(const int* src, int* dst, int n) {
  pdl_trigger();
  if (int(threadIdx.x) < n) dst[threadIdx.x] = src[threadIdx.x];
}

}  // namespace

void mtp_gather(const DevParams* P, int pos_sel, bool skip_if_noop, const float* h0, const float* hring,
                const float* hn, int ldh, const int* tokens, const bf16* E, const float* emb, const float* we,
                const float* wh, float offset, float eps, int H, int rows, float* cat, cudaStream_t st) {
  k_mtp_gather<<<rows, kMtpThreads, 0, st>>>(P, pos_sel, skip_if_noop, h0, hring, hn, ldh, tokens, E, emb, we, wh,
                                             offset, eps, H, cat);
}

void hring_put(const DevParams* P, int pos_sel, bool skip_if_noop, const float* hn, int ldh, int rows, float* hring,
               int H, cudaStream_t st) {
  k_hring_put<<<rows, 256, 0, st>>>(P, pos_sel, skip_if_noop, hn, ldh, rows, hring, H);
}

void pair_hidden(const DevParams* P, int pos_sel, const float* h0, const float* hring, int H, float* out,
                 cudaStream_t st) {
  k_pair_hidden<<<1, 512, 0, st>>>(P, pos_sel, h0, hring, H, out);
}

void mtp_oracle(const DevParams* P, const int* oracle, int step, int* mtok, cudaStream_t st) {
  k_mtp_oracle<<<1, 1, 0, st>>>(P, oracle, step, mtok);
}

void mtp_choose(const DevParams* P, const DevGen* G, const int* reply_count, const int* cand, const float* val, int k,
                int vocab, int step, int* mtok, cudaStream_t st) {
  if (k < 1 || k > ops::kMtpCands || k > vocab) throw std::runtime_error("mtp_choose: candidates out of range");
  k_mtp_choose<<<1, 1, 0, st>>>(P, G, reply_count, cand, val, k, vocab, step, mtok);
}

__global__ void k_shift_params(const DevParams* P, DevParams* dst, int add) {
  pdl_trigger();
  *dst = *P;
  dst->position += add;
}

void shift_params(const DevParams* P, DevParams* dst, int add, cudaStream_t st) {
  k_shift_params<<<1, 1, 0, st>>>(P, dst, add);
}

void given_drafts(const DevParams* P, const int* oracle, int width, int* drafts, cudaStream_t st) {
  k_given_drafts<<<1, FLYX_MAX_WIDE_ROWS, 0, st>>>(P, oracle, width, drafts);  // a wide round's drafts too
}

void copy_ints(const int* src, int* dst, int n, cudaStream_t st) { k_copy_ints<<<1, 32, 0, st>>>(src, dst, n); }

void grouped_conv(const float* x, int ldx, const float* delta, int ld_delta, const float* base, int H, int taps,
                  int group, float* y, int ldy, int rows, bool accumulate, cudaStream_t st) {
  k_grouped_conv<<<dim3(ceil_div(H, 256), rows), 256, 0, st>>>(x, ldx, delta, ld_delta, base, H, taps, group, y, ldy,
                                                               accumulate);
}

void selector_walk(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, const int* oracle, int* drafts, cudaStream_t st,
                   float softcap, int softcap_mode, const DevGen* G) {
  if (k < 1 || k > kWalkMax || width < 1 || width > FLYX_MAX_WIDE_ROWS)
    throw std::runtime_error("selector_walk: candidates or width out of range");
  k_selector_walk<<<1, 32 * std::min(k, 32), 0, st>>>(P, cand, unary, k, hid, rank, pred, succ, width, oracle,
                                                      drafts, softcap, softcap_mode, G);
}

namespace {

// The dynamic shared memory a tree of k candidates a position takes (kWalkMax: 69 KB), within what the device gives
// the kernel beside its static shared memory (optin_dynamic_smem, opted in to once).
size_t tree_shared_bytes(int k) {
  static const int room = optin_dynamic_smem(k_selector_tree);
  const size_t bytes = (size_t(FLYX_MAX_ROWS) + 1) * size_t(k) * sizeof(float);
  if (bytes > size_t(room)) throw std::runtime_error("selector_tree: the device takes no tree this wide");
  return bytes;
}

}  // namespace

void selector_tree(const DevParams* P, const int* cand, const float* unary, int k, const float* hid, int rank,
                   const bf16* pred, const bf16* succ, int width, int depths, const int* oracle, int* drafts,
                   TreeRows* tree, cudaStream_t st, float softcap, int softcap_mode, const DevGen* G) {
  if (k < 1 || k > kWalkMax || width < 1 || width >= FLYX_MAX_ROWS)
    throw std::runtime_error("selector_tree: candidates or width out of range");
  k_selector_tree<<<1, 32 * std::min(k, 32), tree_shared_bytes(k), st>>>(P, cand, unary, k, hid, rank, pred, succ,
                                                                         width, depths, oracle, drafts, tree, softcap,
                                                                         softcap_mode, G);
}

void tap_copy(const float* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st) {
  k_tap_copy<<<rows, 256, 0, st>>>(x, H, feats, ld_feats, slot);
}

void tap_copy(const bf16* x, int H, float* feats, int ld_feats, int slot, int rows, cudaStream_t st) {
  k_tap_copy<<<rows, 256, 0, st>>>(x, H, feats, ld_feats, slot);
}

void draft_ctx_kv(const float* kv, int ld, const float* k_norms, float eps, double theta, int layers, int nkv, int hd,
                  int rows, const DevParams* P, int pos_sel, bool skip_if_noop, kv16* dK, kv16* dV,
                  size_t layer_stride, cudaStream_t st, int ring) {
  k_draft_ctx_kv<<<dim3(layers * nkv, rows), hd, hd * sizeof(float), st>>>(kv, ld, k_norms, eps, theta, nkv, hd, P,
                                                                            pos_sel, skip_if_noop, dK, dV,
                                                                            layer_stride, rope_table(theta, hd), ring);
}

}  // namespace flyx::cuda
