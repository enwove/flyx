// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// Drafting: a drafter's tree of drafts: its chain, then branches best first (FLYX_DRAFT_TREE;
// docs/design/tree-verification.md).
//
// A drafter that scores k candidates at each of `depths` positions after the round's anchor (candidate c at depth
// d + 1 is token cand[d * k + c]; score(p, d, c) is its score after the token p before it, a log-probability up to a
// constant) proposes `width` drafts as a tree rooted at the anchor:
//
//   * a child c of node u (depth(u) < depths) has log-probability
//       score(token(u), depth(u), c) - log sum_c' exp(score(token(u), depth(u), c'))
//     (the k candidates' softmax), and value = value(u) + that (the anchor's value: 0);
//   * the trunk: drafts 0 .. min(width, depths) - 1 are the chain the selector walk drafts, each the highest-scored
//     child of the draft before it (of the anchor for draft 0; ties: the lower candidate);
//   * every later draft is the highest-valued child not yet taken of a node already in the tree; ties: the child of
//     the node that entered the tree first, then the lower candidate;
//   * when no child is left (k = 1 and width > depths), the draft is the anchor's first candidate again (a duplicate,
//     which the round rejects with its sibling).
//
// parents[i] = the draft it continues (-1: the anchor), always < i.  The branches are EAGLE-2's dynamic tree
// (docs/prior-art.md) on the drafter's own scores; the trunk comes first because a drafter's softmax can rate its
// chain's later drafts far below how often they are accepted (DFlash2 on the 27B: a best-first tree of 15 accepted
// fewer drafts than the chain of 7), so the tree is the chain and
// what its budget adds.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace flyx::ops {

template <class Score>
void best_first_tree(int width, int depths, int k, int32_t anchor, const int32_t* cand, Score score, int32_t* drafts,
                     int32_t* parents) {
  const double none = -std::numeric_limits<double>::infinity();
  // Children of node n (0: the anchor, i + 1: draft i) at pool[n * k + c].
  std::vector<double> pool(size_t(width + 1) * size_t(k), none);
  std::vector<int32_t> token(size_t(width) + 1), depth(size_t(width) + 1, 0);
  std::vector<double> value(size_t(width) + 1, 0.0);
  token[0] = anchor;
  for (int n = 0; n < width; ++n) {
    if (depth[size_t(n)] < depths) {  // node n's children
      const int d = depth[size_t(n)];
      std::vector<double> s(static_cast<size_t>(k));
      double m = none;
      for (int c = 0; c < k; ++c) {
        s[size_t(c)] = score(token[size_t(n)], d, c);
        m = std::max(m, s[size_t(c)]);
      }
      double z = 0;
      for (int c = 0; c < k; ++c) z += std::exp(s[size_t(c)] - m);
      for (int c = 0; c < k; ++c) pool[size_t(n) * size_t(k) + size_t(c)] = value[size_t(n)] + s[size_t(c)] - m - std::log(z);
    }
    // The trunk: node n's children only (node n is the trunk's last); then every node's.
    const int trunk = std::min(width, depths), from = n < trunk ? n * k : 0;
    int best = -1;
    for (int i = from; i < (n + 1) * k; ++i)
      if (pool[size_t(i)] != none && (best < 0 || pool[size_t(i)] > pool[size_t(best)])) best = i;
    const int r = n + 1;
    if (best < 0) {
      parents[n] = -1;
      token[size_t(r)] = cand[0];
      value[size_t(r)] = none;
      depth[size_t(r)] = 1;
    } else {
      const int u = best / k;
      parents[n] = u - 1;
      token[size_t(r)] = cand[depth[size_t(u)] * k + best % k];
      value[size_t(r)] = pool[size_t(best)];
      depth[size_t(r)] = depth[size_t(u)] + 1;
      pool[size_t(best)] = none;
    }
    drafts[n] = token[size_t(r)];
  }
}

// ---- DV's trees (bifill-dv's drafter, docs/prior-art.md; python/reference/flyx_ref/dv.py) ---------------------------
//
// After a round the canvas holds, for depths d + 1 = 1 .. depths after the next anchor, slot d's k candidates
// cand[d * k + c] (its top k, highest first), their log-probabilities logp[d * k + c] = log max(p, 1e-9) under the
// canvas, and the selector's projection of its hidden state.  live[d] = 0 for a slot past the canvas (no candidates).

// The selector's table (bifill-dv selector.py PathSelector.table): table[(d * k + i) * k + c] is the log-probability
// of candidate c at slot d when the token before it is candidate i of slot d - 1 (for d = 0 the anchor, every i):
//   S = (alpha * logp[d * k + c] + sum_j pred[prev * rank + j] * hid[d * rank + j] * succ[cand[d * k + c] * rank + j]
//        / sqrt(rank)) / temp,                  table = S - log sum_c' exp S
// with pred[t] = A E[t] and succ[t] = B E[t] (rank values a token; E the target's embedding) and hid[d] = H h_d.
inline void selector_table(int slots, int k, int rank, int32_t anchor, const int32_t* cand, const double* logp,
                           const double* pred, const double* succ, const double* hid, double alpha, double temp,
                           double* table) {
  std::vector<double> s(static_cast<size_t>(k));
  for (int d = 0; d < slots; ++d)
    for (int i = 0; i < k; ++i) {
      const int32_t prev = d == 0 ? anchor : cand[(d - 1) * k + i];
      double m = -std::numeric_limits<double>::infinity();
      for (int c = 0; c < k; ++c) {
        const double* a = pred + size_t(prev) * size_t(rank);
        const double* b = succ + size_t(cand[d * k + c]) * size_t(rank);
        const double* h = hid + size_t(d) * size_t(rank);
        double dot = 0;
        for (int j = 0; j < rank; ++j) dot += a[j] * h[j] * b[j];
        s[size_t(c)] = (dot / std::sqrt(double(rank)) + alpha * logp[d * k + c]) / temp;
        m = std::max(m, s[size_t(c)]);
      }
      double z = 0;
      for (int c = 0; c < k; ++c) z += std::exp(s[size_t(c)] - m);
      double* out = table + (size_t(d) * size_t(k) + size_t(i)) * size_t(k);
      for (int c = 0; c < k; ++c) out[c] = s[size_t(c)] - m - std::log(z);
    }
}

// cov's term for a slot (DYN=...:cov): log max(sum_c exp(logp[c]), 1e-6), the log of its top k's mass.
inline double slot_coverage(const double* logp, int k) {
  double mass = 0;
  for (int c = 0; c < k; ++c) mass += std::exp(logp[c]);
  return std::log(std::max(mass, 1e-6));
}

// Indices 0 .. n - 1 by value, highest first, ties to the lower index.
template <class T>
std::vector<int> order_high_first(const std::vector<T>& v) {
  std::vector<int> o(v.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = int(i);
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return v[size_t(a)] > v[size_t(b)]; });
  return o;
}

// DV's dynamic tree (bifill-dv's DYN=width:beam[:cov], selector.py dyn_tree_fast; EAGLE-2's beam per depth):
//   * an edge into depth d + 1 is worth lp(i, d, c) (the table above: candidate c given its parent is candidate i of
//     depth d, i = 0 under the anchor), plus bonus[d] when bonus is given (cov: slot_coverage), or -1e9 when !live[d];
//   * depth d + 1 keeps the `beam` best children of depth d's beam: a child is worth (its parent's + edge) - 1e-4 (the
//     anchor: 0), so a parent is always worth more than its children; when fewer than `beam` children exist the rest
//     of the beam is -2e9 and continues the last child (its parent and candidate);
//   * the drafts are the `width` best of all depths * beam beam nodes, in beam order (depth-major, then beam rank;
//     parents first), the token of a node worth no more than -1e8 being `mask`.
// Ties go to the lower index (the flat child index, the beam node index).  depths * beam >= width.  An edge above the
// step (never a canvas's: its log-probabilities and cov terms are at most 0 but for rounding) can make a child worth
// more than its parent; a draft whose parent is not drafted goes under the anchor.
// T: the scalar the values are added in, double, or float as the CUDA kernel adds them (dv_tree_beam): the constants
// and every sum rounded to T in this order, so that a float table gives the kernel's trees bit for bit.
template <class T = double, class LogP>
void beam_tree(int width, int beam, int depths, int k, const int32_t* cand, const uint8_t* live, const T* bonus,
               LogP lp, int32_t mask, int32_t* drafts, int32_t* parents) {
  const T no_slot = T(-1e9), pad = T(-2e9), undrafted = T(-1e8), step = T(1e-4);
  std::vector<T> bval{T(0)};
  std::vector<int> bidx{0}, bid{0};  // a beam node's candidate at its slot; its id: 0 the anchor, 1 + d * beam + rank
  std::vector<T> val;
  std::vector<int32_t> tok;
  std::vector<int> par;
  for (int d = 0; d < depths; ++d) {
    const int m = int(bval.size());
    std::vector<T> flat(size_t(std::max(m * k, beam)), pad);
    for (int j = 0; j < m; ++j)
      for (int c = 0; c < k; ++c) {
        T edge = no_slot;
        if (live[d]) {
          edge = T(lp(bidx[size_t(j)], d, c));
          if (bonus) edge = edge + bonus[d];
        }
        flat[size_t(j) * size_t(k) + size_t(c)] = (bval[size_t(j)] + edge) - step;
      }
    const std::vector<int> o = order_high_first(flat);
    std::vector<T> nval(static_cast<size_t>(beam));
    std::vector<int> nidx(static_cast<size_t>(beam)), nid(static_cast<size_t>(beam));
    for (int r = 0; r < beam; ++r) {
      const int i = std::min(o[size_t(r)], m * k - 1);
      nval[size_t(r)] = flat[size_t(o[size_t(r)])];
      nidx[size_t(r)] = i % k;
      nid[size_t(r)] = 1 + d * beam + r;
      val.push_back(nval[size_t(r)]);
      tok.push_back(cand[d * k + i % k]);
      par.push_back(bid[size_t(i / k)]);
    }
    bval.swap(nval);
    bidx.swap(nidx);
    bid.swap(nid);
  }
  std::vector<int> pick = order_high_first(val);
  pick.resize(size_t(width));
  std::sort(pick.begin(), pick.end());
  std::vector<int> at(val.size() + 1, 0);  // a node's row: 0 the anchor, t + 1 draft t
  for (int t = 0; t < width; ++t) at[size_t(pick[size_t(t)]) + 1] = t + 1;
  for (int t = 0; t < width; ++t) {
    const size_t p = size_t(pick[size_t(t)]);
    drafts[t] = val[p] > undrafted ? tok[p] : mask;
    parents[t] = at[size_t(par[p])] - 1;
  }
}

// DV's selector pool tree (ST width/ranks/1: the earlier CUDA port's dv.cu, bifill-dv combo.py SelTreePool):
//   * the pool: for each depth d = 1 .. depths, the tuple that follows the selector's best path, then for each slot
//     s < d each rank r = 1 .. ranks there (the tuple leaves the best path once, at slot s, for its rank-r candidate);
//     its parent is the same tuple one depth up, or the best path's when s = d - 1;
//   * a tuple's candidate is the r-th best (high to low, ties to the lower index) of lp(i, d - 1, .) given its parent's
//     candidate i (0 under the anchor), r = ranks at its deviation's slot and 0 elsewhere, and its value is its
//     parent's (the anchor's: 0) + that log-probability; at a slot past the canvas (!live) the candidate is index r,
//     the log-probability 0, the value also -1e4 lower, and the token `mask`;
//   * the drafts are the `width` best tuples by value - 1e-6 * depth (ties: the lower pool index, so a parent before
//     its children), in pool order; a log-probability above 0 (never a table's) can make a tuple worth more than its
//     parent, and a draft whose parent is not drafted goes under the anchor.
// T as for beam_tree (dv_tree_pool: float).
template <class T = double, class LogP>
void pool_tree(int width, int ranks, int depths, int k, const int32_t* cand, const uint8_t* live, LogP lp,
               int32_t mask, int32_t* drafts, int32_t* parents) {
  const T dead = T(-1e4), step = T(1e-6);
  std::vector<int> depth, parent, lastr, first;  // first[d - 1]: depth d's best-path tuple
  for (int d = 1; d <= depths; ++d) {
    first.push_back(int(depth.size()));
    const int prev = d > 1 ? first[size_t(d - 2)] : -1;
    depth.push_back(d), parent.push_back(prev), lastr.push_back(0);
    for (int s = 0; s < d; ++s)
      for (int r = 1; r <= ranks; ++r) {
        depth.push_back(d);
        parent.push_back(d == 1 ? -1 : (s == d - 1 ? prev : prev + 1 + s * ranks + (r - 1)));
        lastr.push_back(s == d - 1 ? r : 0);
      }
  }
  const size_t P = depth.size();
  std::vector<int> idx(P);
  std::vector<T> cs(P), score(P), row(static_cast<size_t>(k));
  for (size_t p = 0; p < P; ++p) {
    const int d = depth[p], up = parent[p];
    T v = 0;
    if (live[d - 1]) {
      const int i = d == 1 ? 0 : idx[size_t(up)];
      for (int c = 0; c < k; ++c) row[size_t(c)] = T(lp(i, d - 1, c));
      idx[p] = order_high_first(row)[size_t(lastr[p])];
      v = row[size_t(idx[p])];
    } else {
      idx[p] = lastr[p];
    }
    cs[p] = ((d == 1 ? T(0) : cs[size_t(up)]) + v) + (live[d - 1] ? T(0) : dead);
    score[p] = cs[p] - step * T(d);
  }
  std::vector<int> sel = order_high_first(score);
  sel.resize(size_t(width));
  std::sort(sel.begin(), sel.end());
  std::vector<int> node(P, -1);
  for (int t = 0; t < width; ++t) node[size_t(sel[size_t(t)])] = t;
  for (int t = 0; t < width; ++t) {
    const size_t p = size_t(sel[size_t(t)]);
    const int d = depth[p];
    drafts[t] = live[d - 1] ? cand[(d - 1) * k + idx[p]] : mask;
    parents[t] = d == 1 ? -1 : node[size_t(parent[p])];
  }
}

// DSpark Markov RNN head step (Inkling-2): updates `state` [rank] in place and writes `markov` [rank]
// from joint_out = joint_proj([state | w1_prev | hidden_i | f_rnn]) + joint_proj_b ([3 * rank]).
inline void dspark_markov_gate(const double* joint_out, int rank, double* state, double* markov) {
  for (int j = 0; j < rank; ++j) {
    const double g = 1.0 / (1.0 + std::exp(-joint_out[j]));
    const double c = std::tanh(joint_out[rank + j]);
    state[j] = c + g * (state[j] - c);
    markov[j] = std::tanh(joint_out[2 * rank + j]);
  }
}

}  // namespace flyx::ops
