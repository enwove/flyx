// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// EAGLE-4's draft trees (docs/design/tree-verification.md, "EAGLE-4's trees"): a beam of the best `beam` nodes a step,
// each node's children its head's best `beam` candidates, scored by the sum of their log-probabilities over the
// candidates scored exactly (the path's); sampling, by how likely each is the target's choice there, its noise at the
// position known (draft_coupled_scores), its row's best `beam` by that; every candidate a node, the round verifies the best `width` of them, a parent
// before its children, the trunk first: from the root each trunk node's child the chain would draft there (mtp_choose:
// the same candidates, counts and noise), so a tree's first drafts are the chain the controller learns acceptance by
// position from, sampling too.  Drafts only: the round's verification keeps the output plain decoding's.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "ops/sampling/sampling.h"

namespace flyx::cuda {
namespace {

// The most candidates a node's row has (eagle4_tree_cands: its beam, and its chain's kMtpCands).
constexpr int kEagleCandsMax = ops::kMtpCands > kEagleBeamMax ? ops::kMtpCands : kEagleBeamMax;
// A sampled tree's candidates scored by how likely each is the target's choice (draft_coupled_scores) at this tau:
// Qwen3.8-27B with its EAGLE-4 drafter, the six sets of docs/benchmarks.md section 7 at concurrency 1, 117.8 tok/s
// end to end at T = 0.7 and 110.1 at T = 1 against 116.0 and 104.9 at tau 3, 113.1 and 108.9 at tau 1, and 116.7
// and 109.0 with the drafter's own log-probabilities.
constexpr float kEagleCoupledTau = 2.f;

// A row's log-normaliser over its scored candidates from its best's value and probability (draft_pick_topk's prob).
__device__ __forceinline__ float row_lse(float best, float prob) {
  return prob > 0.f && isfinite(best) ? best - logf(prob) : best;
}

// The token the oracle (a test's given drafts) has at `pos`, or -1 with none.
__device__ __forceinline__ int oracle_at(const DevParams* P, const int* oracle, int pos) {
  if (P->oracle_len <= 0 || !oracle) return -1;
  return pos < P->oracle_len ? oracle[pos] : 0;
}

// The frontier front[0 .. fn - 1] (one depth, d; slots f0 ..) as the step's two trees: prep (attn_prep rotates row k
// at P->position - 1 + f0 + prep.depth[k] = P->position - 1 + d and writes it to slot f0 + k, its node's) and att
// (attention: row k at P->position - 1 + d, its round keys its ancestors' slots); their tokens, and the hidden each
// continues from: its parent's normed output (gnode by slot).
__device__ void frontier_rows(EagleTree* t, const float* gnode, float* hbuf, int H) {
  const int f0 = t->f0, fn = t->fn;
  if (threadIdx.x == 0) {
    t->prep.n = t->att.n = fn;
    t->prep.n_taken = t->att.n_taken = 0;
    for (int k = 0; k < fn; ++k) {
      const int node = t->front[k], d = t->depth[node];
      t->prep.depth[k] = d - f0;
      t->att.depth[k] = d;
      for (int e = d, u = node; e >= 0; --e, u = t->parent[u])
        t->att.path[k][e] = static_cast<unsigned char>(t->slot[u]);
      t->tokbuf[k] = t->tok[node];
    }
  }
  __syncthreads();
  for (int k = 0; k < fn; ++k) {
    const float* src = gnode + size_t(t->slot[t->parent[t->front[k]]]) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) hbuf[size_t(k) * H + c] = src[c];
  }
}

// A node: token `tok` below `parent`, of score `score`.
__device__ __forceinline__ int add_node(EagleTree* t, int n, int parent, int tok, float score) {
  t->tok[n] = tok;
  t->parent[n] = parent;
  t->depth[n] = t->depth[parent] + 1;
  t->score[n] = score;
  t->slot[n] = -1;
  t->on_trunk[n] = false;
  return n;
}

// Each of the row's kt best candidates counted as its verifying row counts it (the reply's count and the drafts above
// the node `parent`), into count; false (count unset) without penalties.
__device__ bool path_counts(const EagleTree* t, const DevGen* G, const int* reply_count, const int* cand, int kt,
                            int vocab, int parent, int* count) {
  if (G->presence == 0.f && G->frequency == 0.f) return false;
  for (int c = 0; c < kt; ++c) {
    if (cand[c] < 0 || cand[c] >= vocab) continue;
    count[c] = reply_count[cand[c]];
    for (int u = parent; u > 0; u = t->parent[u]) count[c] += t->tok[u] == cand[c];
  }
  return true;
}

// The draft the chain makes after the trunk node `parent` (k_mtp_choose's): from the row's kt best candidates, each
// counted as its verifying row counts it, at the target's noise of the draft's position.
__device__ int trunk_choice(const EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count,
                            const int* cand, const float* val, int kt, int vocab, int parent) {
  int count[ops::kMtpCands];
  const bool pen = path_counts(t, G, reply_count, cand, kt, vocab, parent, count);
  return ops::draft_choice(cand, val, pen ? count : nullptr, kt, vocab, G->temperature, G->top_p, G->top_k,
                           G->presence, G->frequency, G->seed, int64_t(P->position) + 1 + t->depth[parent]);
}

// A row's candidates' scores below `parent` and the order they are taken in (order[0 .. beam)): greedy (and sampling
// with none of its kt best kept) their log-probabilities over the scored candidates, in their order; sampling the
// coupled ones (draft_coupled_scores: how likely each is the target's choice, its noise at the position known), best
// first (ties: the earlier candidate).
__device__ void row_scores(const EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count,
                           const int* cand, const float* val, float prob, int kc, int kt, int beam, int vocab,
                           int parent, float* score, int* order) {
  const float lse = row_lse(val[0], prob);
  for (int j = 0; j < kc; ++j) score[j] = val[j] - lse;
  for (int j = 0; j < beam; ++j) order[j] = j;
  if (!(G->temperature > 0.f)) return;
  int count[ops::kMtpCands];
  float cs[ops::kMtpCands];
  const bool pen = path_counts(t, G, reply_count, cand, kt, vocab, parent, count);
  if (!ops::draft_coupled_scores(cand, val, pen ? count : nullptr, kt, vocab, G->temperature, G->top_p, G->top_k,
                                 G->presence, G->frequency, G->seed, int64_t(P->position) + 1 + t->depth[parent],
                                 kEagleCoupledTau, cs))
    return;
  for (int j = 0; j < kc; ++j) score[j] = j < kt ? cs[j] : -INFINITY;
  bool used[kEagleCandsMax] = {};
  for (int q = 0; q < beam; ++q) {
    int bj = -1;
    for (int j = 0; j < kt; ++j)
      if (!used[j] && (bj < 0 || score[j] > score[bj])) bj = j;
    if (bj >= 0) used[bj] = true;
    order[q] = bj >= 0 ? bj : q;  // fewer candidates than the beam: one of the rest (-inf, not taken)
  }
}

// Every frontier row's `beam` best candidates (cand, val: row k's kc at k * kc) as nodes, scored by path; the best
// `beam` of them by score (ties: the lower index, the earlier row and candidate) the next frontier, at slots from
// `slot0`.  The trunk node's row adds the chain's draft (trunk_choice, from its min(kMtpCands, vocab) best; added
// where not one of its `beam`), the next trunk node; with an oracle, its token at the depth below the node on its
// path is a node (added where not a candidate).  The oracle's node first of the frontier, then the trunk's.
__device__ void grow(EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count, const int* oracle,
                     const int* cand, const float* val, const float* prob, int rows, int kc, int beam, int vocab,
                     int slot0) {
  if (threadIdx.x != 0) return;
  const int n0 = t->n, kt = min(ops::kMtpCands, vocab);
  int n = n0;
  int forced = -1, trunk = -1;
  const int on = t->oracle_node;
  const int forced_tok = on >= 0 ? oracle_at(P, oracle, P->position + t->depth[on] + 1) : -1;
  for (int k = 0; k < rows; ++k) {
    const int parent = t->front[k];
    const int* rc = cand + size_t(k) * kc;
    const float* rv = val + size_t(k) * kc;
    float score[kEagleCandsMax];
    int order[kEagleBeamMax];
    row_scores(t, P, G, reply_count, rc, rv, prob[k], kc, kt, beam, vocab, parent, score, order);
    // a repeated row (fewer candidates than rows) adds nothing new
    const int trunk_tok =
        parent == t->trunk_node && trunk < 0 ? trunk_choice(t, P, G, reply_count, rc, rv, kt, vocab, parent) : -1;
    bool has_forced = false;
    for (int q = 0; q < beam && n < kEagleNodes; ++q) {
      const int j = order[q];
      if (rc[j] < 0 || (G->temperature > 0.f && score[j] == -INFINITY)) continue;  // sampling: a candidate not kept
      add_node(t, n, parent, rc[j], t->score[parent] + score[j]);
      if (parent == on && rc[j] == forced_tok) forced = n, has_forced = true;
      if (rc[j] == trunk_tok) trunk = n;
      ++n;
    }
    if (parent == on && forced_tok >= 0 && !has_forced && n < kEagleNodes) {  // not a candidate: its own node
      forced = add_node(t, n++, parent, forced_tok, t->score[parent]);
      if (forced_tok == trunk_tok) trunk = forced;
    }
    // outside the row's `beam`: its own node, scored as one (a token of the vocabulary: none from scores all NaN)
    if (trunk_tok >= 0 && trunk_tok < vocab && trunk < 0 && n < kEagleNodes) {
      float lp = -INFINITY;
      for (int j = 0; j < kc; ++j)
        if (rc[j] == trunk_tok) lp = score[j];
      trunk = add_node(t, n++, parent, trunk_tok, t->score[parent] + lp);
    }
  }
  if (trunk >= 0) t->on_trunk[trunk] = true;
  t->oracle_node = forced;
  t->trunk_node = trunk;
  int fn = 0;
  if (forced >= 0) t->front[fn++] = forced;
  if (trunk >= 0 && trunk != forced && fn < beam) t->front[fn++] = trunk;
  while (fn < beam) {
    int bi = -1;
    for (int u = n0; u < n; ++u) {
      bool in = false;
      for (int q = 0; q < fn; ++q) in = in || t->front[q] == u;
      if (!in && (bi < 0 || t->score[u] > t->score[bi])) bi = u;
    }
    if (bi < 0) bi = fn > 0 ? t->front[fn - 1] : n0 - 1;  // fewer candidates than rows: a row repeats one
    t->front[fn++] = bi;
  }
  for (int k = 0; k < fn; ++k) t->slot[t->front[k]] = slot0 + k;
  t->f0 = slot0;
  t->fn = fn;
  t->n = n;
}

__global__ void k_eagle_tree_start(EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count,
                                   const int* oracle, const int* anchor, const int* cand, const float* val,
                                   const float* prob, int kc, int beam, int vocab, const float* gnode, float* hbuf,
                                   int H) {
  pdl_trigger();
  if (threadIdx.x == 0) {
    t->n = 1;
    t->tok[0] = *anchor;
    t->parent[0] = 0;
    t->depth[0] = 0;
    t->score[0] = 0.f;
    t->slot[0] = 0;
    t->on_trunk[0] = true;
    t->front[0] = 0;
    t->f0 = 0;
    t->fn = 1;
    t->oracle_node = P->oracle_len > 0 && oracle ? 0 : -1;
    t->trunk_node = 0;
  }
  __syncthreads();
  grow(t, P, G, reply_count, oracle, cand, val, prob, 1, kc, beam, vocab, 1);
  __syncthreads();
  frontier_rows(t, gnode, hbuf, H);
}

__global__ void k_eagle_tree_grow(EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count,
                                  const int* oracle, const int* cand, const float* val, const float* prob, int kc,
                                  int beam, int vocab, const float* gnode, float* hbuf, int H) {
  pdl_trigger();
  __shared__ int slot0;
  if (threadIdx.x == 0) slot0 = t->f0 + t->fn;
  __syncthreads();
  grow(t, P, G, reply_count, oracle, cand, val, prob, t->fn, kc, beam, vocab, slot0);
  __syncthreads();
  frontier_rows(t, gnode, hbuf, H);
}

// The round's tree of `width` drafts: the oracle's path (its first `width`), else sampling the trunk's first `trunk`
// (the coupled chain, which a sampled round's accepted path follows); then the best nodes by score (ties: the lower
// index; a parent's score is never below its child's, so the nodes taken are a tree), in rows: the trunk (from the
// root each row's oracle child, else its trunk child, else its first taken child) and the others in node order, so a
// parent's row is before its child's.  Greedy the trunk is each node's best child, which the scores take first
// anyway: best first throughout (Qwen3.8-27B, 15 drafts on chat prompts: 4.01 tokens a round against 3.93 with
// the chain of 7 forced).
__global__ void k_eagle_tree_finish(const EagleTree* t, const DevGen* G, int width, int trunk, int* drafts,
                                    TreeRows* tree) {
  pdl_trigger();
  if (threadIdx.x != 0) return;
  const int n = t->n;
  bool taken[kEagleNodes] = {}, oracle[kEagleNodes] = {};
  // the oracle's path to `width` drafts deep at most (a tree of more steps holds deeper nodes than the round's rows)
  for (int v = t->oracle_node; v > 0; v = t->parent[v])
    if (t->depth[v] <= width) oracle[v] = taken[v] = true;
  int count = 0;
  for (int u = 1; u < n; ++u) count += taken[u];
  if (G->temperature <= 0.f) trunk = 0;
  if (count == 0)
    for (int u = 1; u < n && count < min(width, trunk); ++u)
      if (t->on_trunk[u]) taken[u] = true, ++count;
  while (count < width) {
    int best = -1;
    for (int u = 1; u < n; ++u)
      if (!taken[u] && (t->parent[u] == 0 || taken[t->parent[u]]) && (best < 0 || t->score[u] > t->score[best]))
        best = u;
    if (best < 0) break;
    taken[best] = true;
    ++count;
  }
  int row_of[kEagleNodes];
  for (int u = 0; u < n; ++u) row_of[u] = -1;
  row_of[0] = 0;
  int rows = 1;
  for (int u = 0;;) {  // the trunk: each row's oracle child, else its trunk child, else its first taken child
    int next = -1, rank = -1;
    for (int v = u + 1; v < n; ++v) {
      const int r = oracle[v] ? 2 : t->on_trunk[v] ? 1 : 0;
      if (taken[v] && t->parent[v] == u && r > rank) next = v, rank = r;
    }
    if (next < 0) break;
    row_of[next] = rows++;
    u = next;
  }
  for (int u = 1; u < n; ++u)
    if (taken[u] && row_of[u] < 0) row_of[u] = rows++;
  tree->n = rows;
  tree->n_taken = 0;
  tree->depth[0] = 0;
  tree->path[0][0] = 0;
  for (int u = 1; u < n; ++u) {
    const int r = row_of[u];
    if (r < 0) continue;
    drafts[r - 1] = t->tok[u];
    tree->depth[r] = t->depth[u];
    for (int e = t->depth[u], v = u; e >= 0; --e, v = t->parent[v])
      tree->path[r][e] = static_cast<unsigned char>(row_of[v]);
  }
}

}  // namespace

void eagle_tree_start(EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count, const int* oracle,
                      const int* anchor, const int* cand, const float* val, const float* prob, int kc, int beam,
                      int vocab, const float* gnode, float* hbuf, int H, cudaStream_t st) {
  if (beam < 1 || beam > kEagleBeamMax || kc < std::max(beam, std::min(ops::kMtpCands, vocab)) || kc > kEagleCandsMax)
    throw std::runtime_error("eagle_tree_start: candidates out of range");
  k_eagle_tree_start<<<1, 256, 0, st>>>(t, P, G, reply_count, oracle, anchor, cand, val, prob, kc, beam, vocab, gnode,
                                        hbuf, H);
}

void eagle_tree_grow(EagleTree* t, const DevParams* P, const DevGen* G, const int* reply_count, const int* oracle,
                     const int* cand, const float* val, const float* prob, int kc, int beam, int vocab,
                     const float* gnode, float* hbuf, int H, cudaStream_t st) {
  if (beam < 1 || beam > kEagleBeamMax || kc < std::max(beam, std::min(ops::kMtpCands, vocab)) || kc > kEagleCandsMax)
    throw std::runtime_error("eagle_tree_grow: candidates out of range");
  k_eagle_tree_grow<<<1, 256, 0, st>>>(t, P, G, reply_count, oracle, cand, val, prob, kc, beam, vocab, gnode, hbuf,
                                       H);
}

void eagle_tree_finish(const EagleTree* t, const DevGen* G, int width, int trunk, int* drafts, TreeRows* tree,
                       cudaStream_t st) {
  k_eagle_tree_finish<<<1, 32, 0, st>>>(t, G, width, trunk, drafts, tree);
}

}  // namespace flyx::cuda
