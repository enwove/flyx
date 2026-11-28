// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
//
// Token selection: what "sample the next token" means, shared by every
// backend (host and device code).
//
// Penalties (presence and frequency, as the OpenAI API defines them) come
// first.  With c_v the number of times token v was chosen earlier in this
// reply (from the reply's first token up to position pos - 1; the prompt
// does not count):
//   logit_v -= presence * [c_v > 0] + frequency * c_v
// and everything below starts from these logits.
//
// Greedy (temperature <= 0): the largest logit, lowest index on ties.
//
// Sampling (temperature T > 0), for the token that will sit at absolute
// position `pos`:
//   z_v = (logit_v - max) / T
//   candidates: z_v > kSampleZMin (the rest each carry less than e^-30 of the
//               top token's probability and are never chosen)
//   top_k > 0:  keep the candidates whose z is at least the k-th largest
//   top_p < 1:  of those, keep the ones whose z is at least the largest
//               threshold x with  mass(z >= x) >= top_p * mass(kept)
//               (the smallest top set holding top_p of the probability)
//   choose      argmax over kept of  z_v + gumbel(seed, pos, v)
//
// A mask (grammar-constrained decoding: the tokens allowed at this position)
// applies after the filters above: greedy takes the largest allowed logit;
// sampling takes the argmax over kept AND allowed, and when no kept token is
// allowed, the argmax of z_v + gumbel over all allowed tokens.  So whenever
// the unmasked choice is allowed, the masked choice is the same token: an
// engine may decode without the mask, check each token against the grammar,
// and redo with the mask only the first token the grammar rejects, and get
// exactly what masking every token would give (core/grammar).
//
// Gumbel-max with noise keyed on (seed, position, token) makes the choice a
// pure function of the logits, the position and (with penalties) the reply so
// far.  That gives speculative decoding an exact acceptance rule for free: a
// draft is accepted when it equals the token the target chooses at that
// position, exactly as in greedy decoding.  The accepted probability is
// p(draft), the same as standard rejection sampling with a deterministic
// drafter, the output distribution is the target's, and for one seed the
// output does not depend on whether speculation is on.
//
// With penalties the same holds because a round's rows count as the reply
// what they were given: the row that verifies draft i counts the reply up to
// the round's first token and the drafts before i.  Its choice is used only
// when those drafts were accepted, which is when they are the reply.  One
// expression (penalty() below), applied once per token and row to the total
// count, gives every path the same bits.
//
// A round then keeps the drafts the target chose itself, up to the first
// chosen token that stops the reply (accepted_drafts), and the reply ends
// after a round at a stop token, the request's limit, or the end of the
// context (reply_ends).
#pragma once

#include <cmath>
#include <cstdint>

#include "flyx/backend_abi.h"

#if defined(__CUDACC__)
#define FLYX_HD __host__ __device__ __forceinline__
#else
#define FLYX_HD inline
#endif

namespace flyx::ops {

constexpr float kSampleZMin = -30.f;

// What a token chosen `count` times earlier in the reply loses from its logit.
FLYX_HD float penalty(float presence, float frequency, int count) {
  return presence * (count > 0 ? 1.f : 0.f) + frequency * float(count);
}
FLYX_HD double penalty(double presence, double frequency, int count) {
  return presence * (count > 0 ? 1.0 : 0.0) + frequency * double(count);
}

FLYX_HD uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

// Uniform in (0, 1), 24 bits, keyed on (seed, position, token).
FLYX_HD float sample_uniform(uint64_t seed, int64_t pos, int32_t v) {
  const uint64_t h = mix64(mix64(seed + 0x9E3779B97F4A7C15ULL * (uint64_t(pos) + 1)) ^
                           (uint64_t(uint32_t(v)) * 0xD6E8FEB86659FD93ULL));
  const float u = (float(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
  // The top bin's midpoint rounds to 1 in float32, which would give infinite Gumbel noise.
  return fminf(u, 0x1.fffffep-1f);
}

// A drafter's score for token v at position pos when the reply samples (temperature T > 0): its own score s over T
// plus the Gumbel noise the target's choice at pos adds to v (above).  A drafter that takes the argmax of these drafts
// the token the target would sample from the drafter's distribution with the target's noise: where the two
// distributions agree the draft is the target's choice (accepted with probability near 1 - TV between them, against
// p(draft) for the drafter's argmax), and the output, the target's choice, is unchanged.  T <= 0: s.
FLYX_HD float coupled_draft_score(float s, float T, uint64_t seed, int64_t pos, int32_t v) {
  if (!(T > 0.f)) return s;
  const float u = sample_uniform(seed, pos, v);
  return s / T - logf(-logf(u));
}

// How many of its best logits a sequential drafter (MTP) drafts from (fewer with a smaller vocabulary).
constexpr int kMtpCands = 16;

// The candidates a sequential drafter drafts from (draft_choice, draft_coupled_scores): of its k <= kMtpCands best
// logits, those that are candidates (in [0, vocab), logit not NaN), less their penalties (count: null, none), in o
// (their indices into cand, larger logit first, then lower token); returns their number n.  T > 0: z[i] = o[i]'s logit
// less the best over T for the first *keep of them, the ones the target's rule keeps (z > kSampleZMin, top_k, top_p;
// 0: no logit finite to scale from).
FLYX_HD int draft_kept(const int32_t* cand, const float* logit, const int* count, int k, int vocab, float T,
                       float top_p, int top_k, float presence, float frequency, int* o, float* z, int* keep) {
  float l[kMtpCands];
  int n = 0;
  *keep = 0;
  for (int c = 0; c < k && c < kMtpCands; ++c) {
    if (cand[c] < 0 || cand[c] >= vocab || logit[c] != logit[c]) continue;
    l[c] = logit[c] - (count ? penalty(presence, frequency, count[c]) : 0.f);
    int at = n++;
    while (at > 0 && (l[c] > l[o[at - 1]] || (l[c] == l[o[at - 1]] && cand[c] < cand[o[at - 1]]))) {
      o[at] = o[at - 1];
      --at;
    }
    o[at] = c;
  }
  if (n == 0 || !(T > 0.f)) return n;
  const float top = l[o[0]], inv = 1.f / T;
  int kept = 0;
  for (; kept < n; ++kept) {
    const float d = l[o[kept]] - top;
    z[kept] = inv == INFINITY ? d / T : d * inv;  // as the target scales a logit (scaled_logit)
    if (!(z[kept] > kSampleZMin)) break;
  }
  if (kept > 0 && top_k > 0 && top_k < kept) {
    int m = top_k;
    while (m < kept && z[m] >= z[top_k - 1]) ++m;
    kept = m;
  }
  if (kept > 0 && top_p < 1.f) {
    float mass = 0.f;
    for (int c = 0; c < kept; ++c) mass += expf(z[c]);
    float acc = 0.f;
    for (int c = 0; c < kept; ++c) {
      acc += expf(z[c]);
      if (acc >= top_p * mass) {
        int m = c + 1;
        while (m < kept && z[m] == z[c]) ++m;
        kept = m;
        break;
      }
    }
  }
  *keep = kept;
  return n;
}

// A sequential drafter's draft from its k <= kMtpCands best logits (cand, logit: best first; count: each candidate's
// count in the reply as the row verifying the draft counts it, or null), by the target's rule (sample_rows) applied to
// the candidates alone: their penalties, then greedy the largest logit (the lower token on a tie); sampling, z = the
// logit less the best over T, those with z > kSampleZMin, top_k keeping every candidate tied with the k-th, top_p the
// fewest best whose mass (ties at the cut kept) reaches top_p of the kept candidates' (not the vocabulary's), and of
// those the largest z plus the target's noise at the draft's position pos, the lower token on a tie.  Where the
// drafter's logits are the target's within the candidates the draft is the target's choice; the output is the
// target's either way.  A candidate outside [0, vocab) or of NaN logit is not one; with none, token 0 (a draft only
// proposes: the round verifies it).
FLYX_HD int32_t draft_choice(const int32_t* cand, const float* logit, const int* count, int k, int vocab, float T,
                             float top_p, int top_k, float presence, float frequency, uint64_t seed, int64_t pos) {
  int o[kMtpCands], keep;
  float z[kMtpCands];
  const int n = draft_kept(cand, logit, count, k, vocab, T, top_p, top_k, presence, frequency, o, z, &keep);
  if (n == 0) return 0;
  // greedy, or no logit finite to scale from (the best infinite: every difference NaN): the best, as greedy takes it
  if (!(T > 0.f) || keep == 0) return cand[o[0]];
  float best = -INFINITY;
  int32_t bi = INT32_MAX;
  for (int c = 0; c < keep; ++c) {
    const int32_t v = cand[o[c]];
    const float g = z[c] - logf(-logf(sample_uniform(seed, pos, v)));
    if (g > best || (g == best && v < bi)) best = g, bi = v;
  }
  return bi;
}

// Sampling (T > 0), how likely each of the k <= kMtpCands candidates draft_choice takes from is the target's choice at
// pos, the target's noise there known: score[c] for cand[c] the log-softmax, over the kept candidates, of z plus the
// noise over tau (draft_choice's argmax the largest), -inf for the others.  The target's z differ from the drafter's:
// tau > 1 spreads the probability of the drafter's draft after the noise over the candidates nearest it, the
// target's likeliest alternatives whatever their order before it.  Returns false (scores unset) greedy or with no
// candidate kept.
FLYX_HD bool draft_coupled_scores(const int32_t* cand, const float* logit, const int* count, int k, int vocab,
                                  float T, float top_p, int top_k, float presence, float frequency, uint64_t seed,
                                  int64_t pos, float tau, float* score) {
  int o[kMtpCands], keep;
  float z[kMtpCands];
  const int n = draft_kept(cand, logit, count, k, vocab, T, top_p, top_k, presence, frequency, o, z, &keep);
  if (n == 0 || !(T > 0.f) || keep == 0) return false;
  for (int c = 0; c < k && c < kMtpCands; ++c) score[c] = -INFINITY;
  float best = -INFINITY;
  for (int c = 0; c < keep; ++c) {
    score[o[c]] = (z[c] - logf(-logf(sample_uniform(seed, pos, cand[o[c]])))) / tau;
    best = fmaxf(best, score[o[c]]);
  }
  float sum = 0.f;
  for (int c = 0; c < keep; ++c) sum += expf(score[o[c]] - best);
  const float lse = best + logf(sum);
  for (int c = 0; c < keep; ++c) score[o[c]] -= lse;
  return true;
}

// Reflect negative encodings around the key of zero so equal values, including both zeros, share a key.
FLYX_HD uint32_t float_key(uint32_t bits) { return (bits & 0x80000000u) ? -bits : (bits | 0x80000000u); }

// Token v allowed by a mask (null: every token).
FLYX_HD bool allowed(const uint32_t* mask, int32_t v) { return !mask || ((mask[v >> 5] >> (v & 31)) & 1u); }

// One of the request's stop tokens.
FLYX_HD bool stop_token(const int32_t* stop, int n_stop, int32_t t) {
  for (int i = 0; i < n_stop; ++i)
    if (stop[i] == t) return true;
  return false;
}

// How many of a round's `width` drafts it keeps: each that is the token the target chose at its row, up to the
// first chosen token that stops the reply (that token is emitted; the drafts after it are not looked at).
FLYX_HD int accepted_drafts(const int32_t* drafts, const int32_t* choice, int width, const int32_t* stop,
                            int n_stop) {
  int n = 0;
  while (n < width && drafts[n] == choice[n] && !stop_token(stop, n_stop, choice[n])) ++n;
  return n;
}

// Relaxed acceptance (flyx_gen_params.accept_min_p, accept_typical; lossy, off by default).  The target's
// distribution at a row: p_v = exp(z_v) / Z with z_v = (logit_v - max) / T (T the temperature, 1 when greedy; the
// logits after penalties), Z = sum exp(z), entropy H = log Z - sum(exp(z) z) / Z.  A draft token with z = zd
// qualifies under min-p when p / p_max = exp(zd) >= min_p (a fraction of the row's top probability: the user's DVC
// rule), under typical acceptance (Medusa, Cai et al. 2024) when p > min(eps, sqrt(eps) exp(-H)).  The target's own
// choice always qualifies.
FLYX_HD bool relaxed_qualifies(float zd, float Z, float H, float min_p, float typical) {
  if (min_p > 0.f && zd >= logf(min_p)) return true;
  if (typical > 0.f) {
    const float p = expf(zd) / Z;
    if (p > fminf(typical, sqrtf(typical) * expf(-H))) return true;
  }
  return false;
}

// A tree round (gen.parents): row r (0 the anchor, r >= 1 draft r - 1) has parent row parent[r] < r.  Acceptance
// walks from the anchor: at row `at`, the child to take is the first (in row order) whose draft is the target's choice
// at `at`, or with relaxed rules the most probable qualifying child (the choice first on a tie, then row order); a
// child whose token stops the reply ends the round with that token, the others continue from the child.  The round
// emits the taken children's tokens and then the target's choice at the last row taken.  With exact acceptance
// along a chain this is accepted_drafts above.  `qualifies(at, child)` and `prob_z(at, child)` give the relaxed rules'
// test and the child token's z (only called when relaxed).  The round emits `most` tokens at most (max_ctx - position:
// the context's end): the walk ends there, with the target's choice at the row it reached.  Writes the taken rows to
// taken[0 .. n] (taken[0] = 0) and the emitted tokens to tokens[0 .. n]; returns n, the drafts accepted.
template <class Parent, class Qual, class Z>
FLYX_HD int accept_tree(int rows, Parent parent, const int32_t* drafts, const int32_t* choice, bool relaxed,
                        Qual qualifies, Z prob_z, const int32_t* stop, int n_stop, int* taken, int32_t* tokens,
                        int most) {
  int at = 0, n = 0;
  taken[0] = 0;
  while (n + 1 < most) {
    int best = -1;
    float best_z = 0.f;
    for (int c = at + 1; c < rows; ++c) {
      if (parent(c) != at) continue;
      const int32_t t = drafts[c - 1];
      const bool own = t == choice[at];
      if (!relaxed) {
        if (own) {
          best = c;
          break;
        }
        continue;
      }
      if (!own && !qualifies(at, c)) continue;
      const float z = prob_z(at, c);
      const bool best_own = best >= 0 && drafts[best - 1] == choice[at];
      if (best < 0 || z > best_z || (z == best_z && own && !best_own)) {
        best = c;
        best_z = z;
      }
    }
    if (best < 0) break;
    const int32_t t = drafts[best - 1];
    if (stop_token(stop, n_stop, t)) {
      tokens[n] = t;
      return n;
    }
    tokens[n++] = t;
    taken[n] = best;
    at = best;
  }
  tokens[n] = choice[at];
  return n;
}

// Whether the reply ends after a round that left the sequence at `position` with `last` its last token: a stop
// token, the request's limit of positions (0: none), or no room for another round (flyx_room).
FLYX_HD bool reply_ends(int32_t last, int32_t position, int32_t limit, int32_t max_ctx, const int32_t* stop,
                        int n_stop) {
  return stop_token(stop, n_stop, last) || (limit > 0 && position >= limit) || !flyx_room(position, max_ctx);
}

}  // namespace flyx::ops
