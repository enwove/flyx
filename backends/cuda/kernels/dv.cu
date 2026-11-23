// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV drafting (bifill-dv's drafter; ops/canvas/ref.h, ops/draft/ref.h): the selector's token tables, a canvas pass's
// input rows and what it leaves, the canvas sliding by the tokens a round committed, the round's selector table and
// its trees.
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

static_assert(kDvTop == kTopMax, "a slot's candidates are a top-k row");
// The constants of ops::beam_tree / ops::pool_tree / ops::slide rounded to float as their float instantiations round
// them (from the double literal: what torch does with a Python float in float32 arithmetic, too).
constexpr float kNoSlot = float(-1e9), kBeamPad = float(-2e9), kUndrafted = float(-1e8), kBeamStep = float(1e-4);
constexpr float kPoolDead = float(-1e4), kPoolStep = float(1e-6);
constexpr float kLogFloor = float(1e-9), kMassFloor = float(1e-6);
constexpr int kTreeThreads = 512;
constexpr int kTreeRowsCap = int(sizeof(TreeRows::depth) / sizeof(int));  // the rows a TreeRows holds

// ---- the selector's token tables ---------------------------------------------------------------------------------
constexpr int kTabTokens = 64, kTabRanks = 64, kTabK = 16;
// grid (ceil(vocab / 64), ceil(rank / 64), 2: TA then TB); block 256, a thread 4 tokens by 4 ranks.  Each value is one
// fma after another over hidden in order from 0, whatever the tile: the bits of the plain loop.
__global__ void __launch_bounds__(256) k_dv_selector_tables(const float* A, const float* B, int rank, const bf16* E,
                                                            int vocab, int hidden, float* TA, float* TB) {
  pdl_trigger();
  __shared__ float Es[kTabK][kTabTokens + 1], Ms[kTabK][kTabRanks + 1];
  const float* M = blockIdx.z ? B : A;
  float* out = blockIdx.z ? TB : TA;
  const int t0 = blockIdx.x * kTabTokens, j0 = blockIdx.y * kTabRanks;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  float acc[4][4] = {};
  for (int k0 = 0; k0 < hidden; k0 += kTabK) {
    for (int e = threadIdx.x; e < kTabTokens * kTabK; e += blockDim.x) {
      const int i = e / kTabK, kk = e % kTabK, t = t0 + i, k = k0 + kk;
      Es[kk][i] = t < vocab && k < hidden ? __bfloat162float(E[size_t(t) * hidden + k]) : 0.f;
    }
    for (int e = threadIdx.x; e < kTabRanks * kTabK; e += blockDim.x) {
      const int i = e / kTabK, kk = e % kTabK, j = j0 + i, k = k0 + kk;
      Ms[kk][i] = j < rank && k < hidden ? M[size_t(j) * hidden + k] : 0.f;
    }
    __syncthreads();
    const int kn = min(kTabK, hidden - k0);  // no step past hidden: the chain is over hidden alone
    for (int kk = 0; kk < kn; ++kk) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = Es[kk][ty + 16 * i];
        b[i] = Ms[kk][tx + 16 * i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) acc[i][jj] = __fmaf_rn(a[i], b[jj], acc[i][jj]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int t = t0 + ty + 16 * i, j = j0 + tx + 16 * jj;
      if (t < vocab && j < rank) out[size_t(t) * rank + j] = acc[i][jj];
    }
}

// ---- a canvas pass's input rows ------------------------------------------------------------------------------------
// A round whose canvas is not the one the last slide left (DvCanvas::next_start): its pass starts from MASK, z = 0.
__device__ __forceinline__ bool canvas_fresh(const DevParams* P, const int* next_start) {
  return *next_start != P->start;
}

// grid (ceil(hidden / 256), slots); block 256.  [u | g] -> silu(g) * u in u's place (0 on a fresh canvas).
__global__ void k_dv_glu(const DevParams* P, const int* next_start, float* ug, int hidden) {
  pdl_trigger();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= hidden) return;
  float* row = ug + size_t(blockIdx.y) * 2 * hidden;
  row[c] = canvas_fresh(P, next_start) ? 0.f : silu(row[hidden + c]) * row[c];
}

// grid (ceil(hidden / 256), slots); block 256.  x = (cv == mask ? mask_emb : E[cv]) + the residual (ug's second half).
__global__ void k_dv_embed(const DevParams* P, DvCanvas c, int mask, const bf16* E, const bf16* mask_emb,
                           const float* ug, int hidden, float* x) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
  if (i >= hidden) return;
  const bool fresh = canvas_fresh(P, c.next_start);
  const int tok = fresh ? mask : c.cv[r];
  const float e = __bfloat162float(tok == mask ? mask_emb[i] : E[size_t(tok) * hidden + i]);
  x[size_t(r) * hidden + i] = fresh ? e : e + ug[size_t(r) * 2 * hidden + hidden + i];
}

// ---- what a canvas pass leaves -------------------------------------------------------------------------------------
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// The block's best kDvTop of each thread's sorted kDvTop (lv, li), as topk_rows takes them: 16 rounds of the block's
// best, ties to the lower id.  A block of THREADS.  A thread's next candidate is picked with fixed indices (the lists
// stay in registers).
template <int THREADS>
__device__ void block_top(const float (&lv)[kDvTop], const int (&li)[kDvTop], float* top_v, int* top_id) {
  __shared__ float sv[THREADS];
  __shared__ int si[THREADS], st[THREADS];
  const int tid = threadIdx.x;
  int head = 0;
  for (int out = 0; out < kDvTop; ++out) {
    float hv = -INFINITY;
    int hi = INT32_MAX;
#pragma unroll
    for (int k = 0; k < kDvTop; ++k)
      if (k == head) hv = lv[k], hi = li[k];
    sv[tid] = hv;
    si[tid] = hi;
    st[tid] = tid;
    __syncthreads();
    for (int h = THREADS / 2; h > 0; h >>= 1) {
      if (tid < h && better(sv[tid + h], si[tid + h], sv[tid], si[tid])) {
        sv[tid] = sv[tid + h];
        si[tid] = si[tid + h];
        st[tid] = st[tid + h];
      }
      __syncthreads();
    }
    if (tid == 0) {
      top_id[out] = si[0];
      top_v[out] = sv[0];
    }
    if (tid == st[0]) ++head;
    __syncthreads();
  }
}

// grid: (kDvParts, slots); block 256.  Part q of row r's vocabulary, [q V / kDvParts, (q + 1) V / kDvParts): its best
// 16 (each thread its slice's sorted 16, then block_top), the whole row's best 16 among them.  A value better than a
// thread's 16th goes down its list by compare and swap, every index fixed, so the list stays in registers.
__global__ void __launch_bounds__(256) k_canvas_parts(const float* logits, int ld, int V, DvStats s) {
  pdl_trigger();
  __shared__ int top_id[kDvTop];
  __shared__ float top_v[kDvTop];
  const int q = blockIdx.x, r = blockIdx.y, tid = threadIdx.x;
  const int lo = int(int64_t(q) * V / kDvParts), hi = int(int64_t(q + 1) * V / kDvParts);
  const float* row = logits + size_t(r) * ld;
  float lv[kDvTop];
  int li[kDvTop];
#pragma unroll
  for (int i = 0; i < kDvTop; ++i) {
    lv[i] = -INFINITY;
    li[i] = INT32_MAX;
  }
  for (int i = lo + tid; i < hi; i += blockDim.x) {
    float v = row[i];
    int vi = i;
    if (!better(v, vi, lv[kDvTop - 1], li[kDvTop - 1])) continue;
#pragma unroll
    for (int p = 0; p < kDvTop; ++p) {
      if (better(v, vi, lv[p], li[p])) {
        const float tv = lv[p];
        const int ti = li[p];
        lv[p] = v;
        li[p] = vi;
        v = tv;
        vi = ti;
      }
    }
  }
  block_top<256>(lv, li, top_v, top_id);
  const size_t at = (size_t(r) * kDvParts + q) * kDvTop;
  if (tid < kDvTop) {
    s.part_i[at + tid] = top_id[tid];
    s.part_v[at + tid] = top_v[tid];
  }
}

// grid: slots; block 512.  The top 16 (the best of the parts' best 16, block_top: the same 16 as of the whole row);
// the log-sum-exp from the largest (each thread's terms in the order of the vocabulary, then block_sum's tree); each
// candidate's probability exp((l - max) - log sum), their mass in order, and the expected embedding, a dim a thread
// (the 16 terms in order).
__global__ void __launch_bounds__(512) k_canvas_stats(const float* logits, int ld, int V, const bf16* E, int H,
                                                      DvStats s) {
  pdl_trigger();
  static_assert(kDvParts * kDvTop <= 512, "a candidate a thread");
  __shared__ int top_id[kDvTop];
  __shared__ float top_v[kDvTop], top_p[kDvTop];
  const int r = blockIdx.x, tid = threadIdx.x;
  const float* row = logits + size_t(r) * ld;
  float lv[kDvTop];
  int li[kDvTop];
#pragma unroll
  for (int i = 0; i < kDvTop; ++i) {
    lv[i] = -INFINITY;
    li[i] = INT32_MAX;
  }
  if (tid < kDvParts * kDvTop) {
    lv[0] = s.part_v[size_t(r) * kDvParts * kDvTop + tid];
    li[0] = s.part_i[size_t(r) * kDvParts * kDvTop + tid];
  }
  block_top<512>(lv, li, top_v, top_id);
  const float m = top_v[0];
  float sum = 0.f;
  // Unrolled so a thread's loads go ahead of its adds (one block a slot: their latency was most of the kernel); the
  // terms are added in the same order.
#pragma unroll 8
  for (int i = tid; i < V; i += blockDim.x) sum += expf(row[i] - m);
  const float log_sum = logf(block_sum(sum));
  if (tid < kDvTop) top_p[tid] = expf((top_v[tid] - m) - log_sum);
  __syncthreads();
  if (tid < kDvTop) {
    s.ids[r * kDvTop + tid] = top_id[tid];
    s.p[r * kDvTop + tid] = top_p[tid];
  }
  if (tid == 0) {
    float mass = 0.f;
    for (int k = 0; k < kDvTop; ++k) mass += top_p[k];
    s.mass[r] = mass;
    s.lse[r] = m + log_sum;
  }
  for (int c = tid; c < H; c += blockDim.x) {
    float acc = 0.f;
    for (int k = 0; k < kDvTop; ++k) acc = __fmaf_rn(top_p[k], __bfloat162float(E[size_t(top_id[k]) * H + c]), acc);
    s.zn[size_t(r) * H + c] = acc;
  }
}

// grid ceil(rank / 8); block 256: warp w the selector's row j = 8 blockIdx.x + w of H for every slot (a lane's terms
// in the order of hidden, the lanes' sums by warp_sum's tree), divided by sqrt(rank).
__global__ void __launch_bounds__(256) k_canvas_hid(const float* h, int ldh, int H, const float* Hm, int rank,
                                                    float* hid) {
  pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, j = blockIdx.x * 8 + warp, r = blockIdx.y;
  if (j >= rank) return;  // a whole warp
  const float* w = Hm + size_t(j) * H;
  const float* x = h + size_t(r) * ldh;
  float acc = 0.f;
  for (int c = lane; c < H; c += 32) acc = __fmaf_rn(w[c], x[c], acc);
  acc = warp_sum(acc);
  if (lane == 0) hid[size_t(r) * rank + j] = acc / sqrtf(float(rank));
}

// ---- the slide -----------------------------------------------------------------------------------------------------
// grid: slots; block 256.  Slot w of the canvas takes the pass's slot j = w + n (a slot past the pass: no candidates,
// the last slot's clamped in as ops::slide keeps them).  A no-op round (the reply had stopped) committed nothing and
// moved neither P->start nor P->position: its pass followed no round, and the canvas stays as the last one left it.
__global__ void __launch_bounds__(256) k_dv_slide(const DevParams* P, DvStats s, int N, int H, int rank, float fill,
                                                  int mask, DvCanvas c) {
  pdl_trigger();
  if (P->noop) return;
  const int w = blockIdx.x, n = P->position - P->start, j = w + n, jc = min(j, N - 1);
  const bool ok = j < N;
  if (threadIdx.x == 0) {
    c.cv[w] = ok && s.p[size_t(j) * kDvTop] >= fill ? s.ids[size_t(j) * kDvTop] : mask;
    c.live[w] = ok;
    float mass = 0.f;  // cov's term from the clamped log-probabilities, as ops::slot_coverage takes it
    for (int k = 0; k < kDvTop; ++k) {
      const float lp = logf(fmaxf(s.p[size_t(jc) * kDvTop + k], kLogFloor));
      c.cand[w * kDvTop + k] = s.ids[size_t(jc) * kDvTop + k];
      c.logp[w * kDvTop + k] = lp;
      mass += expf(lp);
    }
    c.cov[w] = logf(fmaxf(mass, kMassFloor));
    if (w == 0) *c.next_start = P->position;  // the next round's: its tree and its pass read this canvas
  }
  for (int i = threadIdx.x; i < H; i += blockDim.x) c.z[size_t(w) * H + i] = ok ? s.zn[size_t(j) * H + i] : 0.f;
  for (int i = threadIdx.x; i < rank; i += blockDim.x) c.hid[size_t(w) * rank + i] = s.hid[size_t(jc) * rank + i];
}

// ---- the round's selector table ------------------------------------------------------------------------------------
// grid (depths, 16); block 512.  Block (d, i): slot d's 16 candidates after candidate i of slot d - 1 (the anchor for
// d = 0); warp b candidate b's sum over the rank of TA[prev] hid TB[cand_b] (a lane's terms in order, then warp_sum's
// tree), warp 0 their log-softmax.
__global__ void __launch_bounds__(512) k_dv_selector_table(const DevParams* P, DvCanvas c, const float* TA,
                                                           const float* TB, int rank, float alpha, float temp,
                                                           float* table) {
  pdl_trigger();
  if (*c.next_start != P->position) return;  // the round drafts DV's first tree: its candidates are not this canvas's
  __shared__ float score[kDvTop];
  const int d = blockIdx.x, i = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int prev = d == 0 ? P->anchor : c.cand[(d - 1) * kDvTop + i];
  {
    const float* a = TA + size_t(prev) * rank;
    const float* b = TB + size_t(c.cand[d * kDvTop + warp]) * rank;
    const float* h = c.hid + size_t(d) * rank;
    float acc = 0.f;
    for (int j = lane; j < rank; j += 32) acc = __fmaf_rn(__fmul_rn(a[j], h[j]), b[j], acc);
    acc = warp_sum(acc);
    if (lane == 0) score[warp] = __fdiv_rn(__fmaf_rn(alpha, c.logp[d * kDvTop + warp], acc), temp);
  }
  __syncthreads();
  if (warp == 0) {
    const float v = lane < kDvTop ? score[lane] : -INFINITY;
    float m = v;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float z = warp_sum(lane < kDvTop ? expf(v - m) : 0.f);
    if (lane < kDvTop) table[(size_t(d) * kDvTop + i) * kDvTop + lane] = (v - m) - logf(z);
  }
}

// ---- the trees -----------------------------------------------------------------------------------------------------
// What a round drafts: its tree; the test's drafts as a chain (P->oracle_len); DV's first tree, the canvas being
// another round's.
enum TreeMode : int { kTreeGrown = 0, kTreeOracle = 1, kTreeFirst = 2 };
__device__ __forceinline__ int tree_mode(const DevParams* P, const DvCanvas& c) {
  return P->oracle_len > 0 ? kTreeOracle : *c.next_start != P->position ? kTreeFirst : kTreeGrown;
}

// The drafts of a chain or of DV's first tree: draft t's token and parent.
__device__ void fixed_tree(int mode, const DevParams* P, const int* oracle, int width, int mask, int* tok, int* par) {
  for (int t = threadIdx.x; t < width; t += blockDim.x) {
    const int pos = P->position + 1 + t;
    tok[t] = mode == kTreeOracle ? (pos < P->oracle_len ? oracle[pos] : 0) : mask;
    par[t] = mode == kTreeOracle ? t - 1 : -1;
  }
}

// Draft t's token and parent (-1: the anchor; parents first) -> drafts, parents and the round's TreeRows as set_tree
// writes a given tree (row 0 the anchor, row t + 1 draft t, a row one deeper than its parent's: the rows the tree's
// parents describe, whatever slot a draft came from).  Every thread of the block calls it.
__device__ void write_tree(int width, const int* tok, const int* par, int* drafts, int* parents, TreeRows* tree) {
  for (int t = threadIdx.x; t < width; t += blockDim.x) {
    drafts[t] = tok[t];
    if (parents) parents[t] = par[t];
  }
  if (!tree) return;
  for (int i = threadIdx.x; i < kTreeRowsCap * kTreeRowsCap; i += blockDim.x)
    tree->path[i / kTreeRowsCap][i % kTreeRowsCap] = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    tree->n = width + 1;
    tree->n_taken = 0;
    tree->depth[0] = 0;
    for (int r = 1; r <= width; ++r) {
      const int p = par[r - 1] + 1, depth = tree->depth[p] + 1;
      tree->depth[r] = depth;
      for (int d = 0; d < depth; ++d) tree->path[r][d] = tree->path[p][d];
      tree->path[r][depth] = static_cast<unsigned char>(r);
    }
  }
}

// The better of two (value, index) candidates: the higher value, ties to the lower index; INT_MAX: none.
__device__ __forceinline__ void take_better(float& bv, int& bi, float ov, int oi) {
  if (oi != INT_MAX && (bi == INT_MAX || ov > bv || (ov == bv && oi < bi))) bv = ov, bi = oi;
}
__device__ __forceinline__ void warp_best(float& bv, int& bi) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    take_better(bv, bi, ov, oi);
  }
}

// One block of kTreeThreads: ops::beam_tree<float>.  Depth by depth: every child of the beam a thread (flat index
// j * 16 + c), its place among them by counting the better ones (ties to the lower index), the beam's `beam` best kept;
// then the `width` best of all depths' beam nodes, warp 0 merging the depths' lists (each high to low already, ties
// in node order) a node at a time; drafts in node order (depth-major), so a parent comes first.
__global__ void __launch_bounds__(kTreeThreads) k_dv_tree_beam(const DevParams* P, DvCanvas c, const float* table,
                                                               int width, int beam, int cov, int depths, int mask,
                                                               const int* oracle, int* drafts, int* parents,
                                                               TreeRows* tree) {
  pdl_trigger();
  constexpr int kNodes = kDvMaxDepths * kDvMaxBeam;
  __shared__ float s_val[kNodes];
  __shared__ int s_node_tok[kNodes], s_node_par[kNodes];  // a node's token; its parent's id (0 the anchor, 1 + node)
  __shared__ float s_flat[kDvMaxBeam * kDvTop];
  __shared__ int s_order[kDvMaxBeam];
  __shared__ float s_bval[kDvMaxBeam];
  __shared__ int s_bidx[kDvMaxBeam], s_bid[kDvMaxBeam];  // the beam: value, candidate index at its slot, id
  __shared__ int s_take[kDvMaxDepths], s_first[kDvMaxDepths + 1];
  __shared__ int s_tok[kDvMaxWidth], s_par[kDvMaxWidth];
  __shared__ int s_mode;
  const int tid = threadIdx.x;
  if (tid == 0) {
    s_mode = tree_mode(P, c);
    s_bval[0] = 0.f;
    s_bidx[0] = 0;
    s_bid[0] = 0;
  }
  __syncthreads();
  if (s_mode != kTreeGrown) {
    fixed_tree(s_mode, P, oracle, width, mask, s_tok, s_par);
    __syncthreads();
    write_tree(width, s_tok, s_par, drafts, parents, tree);
    return;
  }
  int m = 1;
  for (int d = 0; d < depths; ++d) {
    const int real = m * kDvTop, nf = max(real, beam);
    const bool live = c.live[d] != 0;
    const float bonus = cov ? c.cov[d] : 0.f;
    for (int e = tid; e < nf; e += blockDim.x) {
      float v = kBeamPad;
      if (e < real) {
        const int j = e / kDvTop;
        float edge = kNoSlot;
        if (live) {
          edge = table[(size_t(d) * kDvTop + s_bidx[j]) * kDvTop + e % kDvTop];
          if (cov) edge = __fadd_rn(edge, bonus);
        }
        v = __fsub_rn(__fadd_rn(s_bval[j], edge), kBeamStep);
      }
      s_flat[e] = v;
    }
    __syncthreads();
    for (int e = tid; e < nf; e += blockDim.x) {
      const float v = s_flat[e];
      int place = 0;
      for (int f = 0; f < nf && place < beam; ++f) place += s_flat[f] > v || (s_flat[f] == v && f < e);
      if (place < beam) s_order[place] = e;
    }
    __syncthreads();
    for (int r = tid; r < beam; r += blockDim.x) {  // a place past the children continues the last one
      const int e = s_order[r], i = min(e, real - 1), node = d * beam + r;
      s_val[node] = s_flat[e];
      s_node_tok[node] = c.cand[d * kDvTop + i % kDvTop];
      s_node_par[node] = s_bid[i / kDvTop];
    }
    __syncthreads();
    for (int r = tid; r < beam; r += blockDim.x) {
      const int e = s_order[r];
      s_bval[r] = s_flat[e];
      s_bidx[r] = min(e, real - 1) % kDvTop;
      s_bid[r] = 1 + d * beam + r;
    }
    __syncthreads();
    m = beam;
  }
  if (tid < 32) {  // lane l merges depths l and l + 32
    int head[2] = {0, 0};
    for (int t = 0; t < width; ++t) {
      float bv = 0.f;
      int bi = INT_MAX;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int d = tid + 32 * q;
        if (d < depths && head[q] < beam) take_better(bv, bi, s_val[d * beam + head[q]], d * beam + head[q]);
      }
      warp_best(bv, bi);
      const int wd = bi / beam;
      if (wd == tid) ++head[0];
      else if (wd == tid + 32) ++head[1];
    }
#pragma unroll
    for (int q = 0; q < 2; ++q)
      if (tid + 32 * q < depths) s_take[tid + 32 * q] = head[q];
  }
  __syncthreads();
  if (tid == 0) {
    s_first[0] = 0;
    for (int d = 0; d < depths; ++d) s_first[d + 1] = s_first[d] + s_take[d];
  }
  __syncthreads();
  // The taken nodes of a depth are its first s_take.  A parent not taken (a child worth more than it: an edge above
  // the depth's step, which a canvas's log-probabilities and cov terms, at most 0 but for rounding, never give) leaves
  // its child under the anchor, as ops::beam_tree does.
  for (int q = tid; q < depths * beam; q += blockDim.x) {
    const int d = q / beam, r = q % beam;
    if (r >= s_take[d]) continue;
    const int t = s_first[d] + r, up = s_node_par[q] - 1;  // the parent's node, -1 the anchor
    s_tok[t] = s_val[q] > kUndrafted ? s_node_tok[q] : mask;
    s_par[t] = up < 0 || up % beam >= s_take[up / beam] ? -1 : s_first[up / beam] + up % beam;
  }
  __syncthreads();
  write_tree(width, s_tok, s_par, drafts, parents, tree);
}

// The pool's layout (ops::pool_tree): tuple q of depth d (q = p - off[d - 1]): q = 0 the best path, else the deviation
// at slot s = (q - 1) / ranks to rank (q - 1) % ranks + 1; its parent tuple (-1: the anchor) and its rank at its slot.
__device__ __forceinline__ void pool_tuple(const int* off, int d, int p, int ranks, int* up, int* lastr) {
  const int q = p - off[d - 1], prev = d > 1 ? off[d - 2] : -1;
  *up = prev;
  *lastr = 0;
  if (q > 0) {
    const int s = (q - 1) / ranks, r = (q - 1) % ranks + 1;
    *up = d == 1 ? -1 : (s == d - 1 ? prev : prev + 1 + s * ranks + (r - 1));
    *lastr = s == d - 1 ? r : 0;
  }
}

size_t pool_shared_bytes(int pool, int depths) {
  return size_t(pool) * sizeof(float) + size_t(pool / 32 + 1) * sizeof(unsigned) + size_t(pool) +
         size_t(depths) * kDvTop * kDvTop;
}

// One block of kTreeThreads: ops::pool_tree<float>.  Each table row's candidates ordered high to low once; the tuples
// depth by depth (a thread a tuple); the `width` best by score (ties to the lower pool index) a block-wide argmax at
// a time; drafts in pool order.  Dynamic shared memory (pool_shared_bytes): every tuple's value, then score [pool]
// float, taken bits, its candidate index [pool] and the rows' orders [depths][16][16] bytes.
__global__ void __launch_bounds__(kTreeThreads) k_dv_tree_pool(const DevParams* P, DvCanvas c, const float* table,
                                                               int width, int ranks, int depths, int mask,
                                                               const int* oracle, int* drafts, int* parents,
                                                               TreeRows* tree) {
  pdl_trigger();
  extern __shared__ float pool_sh[];
  __shared__ int s_off[kDvMaxDepths + 1];
  __shared__ float s_wv[kTreeThreads / 32];
  __shared__ int s_wi[kTreeThreads / 32];
  __shared__ int s_sel[kDvMaxWidth];
  __shared__ int s_tok[kDvMaxWidth], s_par[kDvMaxWidth];
  __shared__ int s_mode;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    s_mode = tree_mode(P, c);
    s_off[0] = 0;
    for (int d = 1; d <= depths; ++d) s_off[d] = s_off[d - 1] + 1 + d * ranks;
  }
  __syncthreads();
  if (s_mode != kTreeGrown) {
    fixed_tree(s_mode, P, oracle, width, mask, s_tok, s_par);
    __syncthreads();
    write_tree(width, s_tok, s_par, drafts, parents, tree);
    return;
  }
  const int pool = s_off[depths];
  float* cs = pool_sh;
  unsigned* taken = reinterpret_cast<unsigned*>(cs + pool);
  uint8_t* idx = reinterpret_cast<uint8_t*>(taken + pool / 32 + 1);
  uint8_t* order = idx + pool;  // order[row * 16 + r]: row (d * 16 + i)'s r-th best candidate, ties to the lower one
  for (int i = tid; i < pool / 32 + 1; i += blockDim.x) taken[i] = 0u;
  for (int row = tid; row < depths * kDvTop; row += blockDim.x) {
    const float* v = table + size_t(row) * kDvTop;
    for (int k = 0; k < kDvTop; ++k) {
      int place = 0;
      for (int o = 0; o < kDvTop; ++o) place += v[o] > v[k] || (v[o] == v[k] && o < k);
      order[row * kDvTop + place] = static_cast<uint8_t>(k);
    }
  }
  __syncthreads();
  for (int d = 1; d <= depths; ++d) {
    const bool live = c.live[d - 1] != 0;
    for (int p = s_off[d - 1] + tid; p < s_off[d]; p += blockDim.x) {
      int up, lastr;
      pool_tuple(s_off, d, p, ranks, &up, &lastr);
      float v = 0.f;
      if (live) {
        const int row = (d - 1) * kDvTop + (d == 1 ? 0 : idx[up]);
        const int k = order[row * kDvTop + lastr];
        idx[p] = static_cast<uint8_t>(k);
        v = table[size_t(row) * kDvTop + k];
      } else {
        idx[p] = static_cast<uint8_t>(lastr);
      }
      cs[p] = __fadd_rn(__fadd_rn(d == 1 ? 0.f : cs[up], v), live ? 0.f : kPoolDead);
    }
    __syncthreads();
  }
  for (int d = 1; d <= depths; ++d)  // value -> score, in place: no tuple's value is read past here
    for (int p = s_off[d - 1] + tid; p < s_off[d]; p += blockDim.x)
      cs[p] = __fsub_rn(cs[p], __fmul_rn(kPoolStep, float(d)));
  __syncthreads();
  for (int t = 0; t < width; ++t) {
    float bv = 0.f;
    int bi = INT_MAX;
    for (int p = tid; p < pool; p += blockDim.x)
      if (!((taken[p >> 5] >> (p & 31)) & 1u)) take_better(bv, bi, cs[p], p);
    warp_best(bv, bi);
    if (lane == 0) {
      s_wv[warp] = bv;
      s_wi[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
      bv = 0.f;
      bi = INT_MAX;
      if (lane < int(blockDim.x >> 5)) bv = s_wv[lane], bi = s_wi[lane];
      warp_best(bv, bi);
      if (lane == 0) {
        s_sel[t] = bi;
        taken[bi >> 5] |= 1u << (bi & 31);
      }
    }
    __syncthreads();
  }
  if (tid == 0)  // pool order: depth-major, so a parent comes first
    for (int i = 1; i < width; ++i)
      for (int j = i; j > 0 && s_sel[j - 1] > s_sel[j]; --j) {
        const int x = s_sel[j];
        s_sel[j] = s_sel[j - 1];
        s_sel[j - 1] = x;
      }
  __syncthreads();
  for (int t = tid; t < width; t += blockDim.x) {
    const int p = s_sel[t];
    int d = 1;
    while (p >= s_off[d]) ++d;
    int up, lastr;
    pool_tuple(s_off, d, p, ranks, &up, &lastr);
    int parent = -1;
    for (int u = 0; u < t && d > 1; ++u)
      if (s_sel[u] == up) parent = u;
    s_tok[t] = c.live[d - 1] ? c.cand[(d - 1) * kDvTop + idx[p]] : mask;
    s_par[t] = parent;
  }
  __syncthreads();
  write_tree(width, s_tok, s_par, drafts, parents, tree);
}

// The dynamic shared memory a pool tree's block may take: what a block may opt in to, less the kernel's own arrays,
// opted in to once (host work at a graph's capture).  The block's whole share counts: a pool past 48 KB less those
// arrays without the opt-in, or past this, fails at its launch and writes nothing.
int pool_shared_limit() {
  static const int limit = [] {
    int dev = 0, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes a{};
    cudaFuncGetAttributes(&a, k_dv_tree_pool);
    const int own = int(a.sharedSizeBytes);
    if (cudaFuncSetAttribute(k_dv_tree_pool, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - own) == cudaSuccess)
      return optin - own;
    cudaGetLastError();  // no opt-in: a block's default 48 KB
    return 48 * 1024 - own;
  }();
  return limit;
}

}  // namespace

void dv_selector_tables(const float* A, const float* B, int rank, const bf16* E, int vocab, int hidden, float* TA,
                        float* TB, cudaStream_t st) {
  const dim3 grid(ceil_div(vocab, kTabTokens), ceil_div(rank, kTabRanks), 2);
  k_dv_selector_tables<<<grid, 256, 0, st>>>(A, B, rank, E, vocab, hidden, TA, TB);
}

void canvas_input(const DevParams* P, const DvCanvas& c, int slots, int mask, const bf16* E, const bf16* mask_emb,
                  const bf16* res_in, const bf16* res_out, int hidden, float* ug, float* x, const Launch& L) {
  const int ld = 2 * hidden;
  // A round's GEMMs (Launch::round_rows): the 1..16-row kernels' arithmetic on every 16 rows (their inputs as three bf16
  // parts), whose rows' bits do not depend on the others' (class R), so a slot's input is the same whatever the
  // canvas's size.
  Launch R = L;
  R.round_rows = true;
  R.draft = false;
  gemm(c.z, hidden, res_in, ld, hidden, ug, ld, slots, false, R);
  k_dv_glu<<<dim3(ceil_div(hidden, 256), slots), 256, 0, L.st>>>(P, c.next_start, ug, hidden);
  gemm(ug, ld, res_out, hidden, hidden, ug + hidden, ld, slots, false, R);
  k_dv_embed<<<dim3(ceil_div(hidden, 256), slots), 256, 0, L.st>>>(P, c, mask, E, mask_emb, ug, hidden, x);
}

void canvas_stats(const float* logits, int ld, int vocab, const bf16* E, const float* h, int ldh, int hidden,
                  const float* H, int rank, int slots, const DvStats& s, cudaStream_t st) {
  k_canvas_parts<<<dim3(kDvParts, slots), 256, 0, st>>>(logits, ld, vocab, s);
  k_canvas_stats<<<slots, 512, 0, st>>>(logits, ld, vocab, E, hidden, s);
  k_canvas_hid<<<dim3(ceil_div(rank, 8), slots), 256, 0, st>>>(h, ldh, hidden, H, rank, s.hid);
}

void dv_slide(const DevParams* P, const DvStats& s, int slots, int hidden, int rank, float fill, int mask,
              const DvCanvas& c, cudaStream_t st) {
  k_dv_slide<<<slots, 256, 0, st>>>(P, s, slots, hidden, rank, fill, mask, c);
}

namespace {
// Launched in order (no programmatic dependence): P->noop is the accept's, which runs before it.
__global__ void k_dv_live(cudaGraphConditionalHandle h, const DevParams* P) { cudaGraphSetConditional(h, !P->noop); }
}  // namespace

void dv_live(cudaGraphConditionalHandle h, const DevParams* P, cudaStream_t st) { k_dv_live<<<1, 1, 0, st>>>(h, P); }

void dv_selector_table(const DevParams* P, const DvCanvas& c, const float* TA, const float* TB, int rank, float alpha,
                       float temp, int depths, float* table, cudaStream_t st) {
  k_dv_selector_table<<<dim3(depths, kDvTop), 32 * kDvTop, 0, st>>>(P, c, TA, TB, rank, alpha, temp, table);
}

bool dv_tree_takes(int kind, int width, int fan, int depths, std::string* why) {
  auto no = [&](const std::string& w) { return *why = w, false; };
  if (width < 1 || width > kDvMaxWidth || depths < 1 || depths > kDvMaxDepths)
    return no("a tree of " + std::to_string(width) + " drafts over " + std::to_string(depths) + " slots (1 to " +
              std::to_string(kDvMaxWidth) + " drafts, 1 to " + std::to_string(kDvMaxDepths) + " slots)");
  if (kind == FLYX_DV_TREE_BEAM) {
    if (fan < 1 || fan > kDvMaxBeam) return no("a beam of " + std::to_string(fan) + " (1 to 32)");
    if (depths * fan < width) return no("a beam of " + std::to_string(fan) + " over " + std::to_string(depths) +
                                        " slots holds fewer than " + std::to_string(width) + " drafts");
    return true;
  }
  if (kind != FLYX_DV_TREE_POOL) return no("tree kind " + std::to_string(kind));
  if (fan < 1 || fan >= kDvTop) return no("a pool to rank " + std::to_string(fan) + " (1 to 15)");
  const int pool = depths * (2 + (depths + 1) * fan) / 2;  // sum over d of 1 + d ranks
  if (pool < width) return no("a pool of " + std::to_string(pool) + " tuples for " + std::to_string(width) + " drafts");
  const size_t shared = pool_shared_bytes(pool, depths);
  if (shared > size_t(pool_shared_limit()))
    return no("a pool of " + std::to_string(pool) + " tuples takes " + std::to_string(shared) + " bytes of shared "
              "memory, a block " + std::to_string(pool_shared_limit()));
  return true;
}

void dv_tree_beam(const DevParams* P, const DvCanvas& c, const float* table, int width, int beam, bool cov, int depths,
                  int mask, const int* oracle, int* drafts, int* parents, TreeRows* tree, cudaStream_t st) {
  std::string why;
  if (!dv_tree_takes(FLYX_DV_TREE_BEAM, width, beam, depths, &why) || (tree && width + 1 > kTreeRowsCap))
    throw std::runtime_error("dv_tree_beam: width, beam or depths out of range (" + why + ")");
  k_dv_tree_beam<<<1, kTreeThreads, 0, st>>>(P, c, table, width, beam, cov ? 1 : 0, depths, mask, oracle, drafts,
                                             parents, tree);
}

void dv_tree_pool(const DevParams* P, const DvCanvas& c, const float* table, int width, int ranks, int depths,
                  int mask, const int* oracle, int* drafts, int* parents, TreeRows* tree, cudaStream_t st) {
  std::string why;
  if (!dv_tree_takes(FLYX_DV_TREE_POOL, width, ranks, depths, &why) || (tree && width + 1 > kTreeRowsCap))
    throw std::runtime_error("dv_tree_pool: width, ranks or depths out of range (" + why + ")");
  const int pool = depths * (2 + (depths + 1) * ranks) / 2;  // sum over d of 1 + d ranks
  const size_t shared = pool_shared_bytes(pool, depths);
  k_dv_tree_pool<<<1, kTreeThreads, shared, st>>>(P, c, table, width, ranks, depths, mask, oracle, drafts, parents,
                                                  tree);
}

}  // namespace flyx::cuda
