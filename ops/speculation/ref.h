// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Float64 reference for intra-round confidence stopping decisions (sequential and parallel block drafters).
#pragma once

#include <algorithm>
#include <cmath>

#include "flyx/backend_abi.h"

namespace flyx::ops {

inline double stop_draft_prob_ref(double prob, double margin, int pos, const flyx_stop_policy& pol) {
  if (prob <= 0.0) return double(pol.p_pos[pos < FLYX_MAX_ROWS ? pos : FLYX_MAX_ROWS - 1]);
  const double p = std::clamp(prob, 1e-5, 1.0 - 1e-5);
  const double lp = std::clamp(std::log(p / (1.0 - p)), -6.0, 10.0);
  const double mg = std::clamp(margin, 0.0, 12.0);
  const double z = std::clamp(double(pol.w_bias) + double(pol.w_margin) * mg + double(pol.w_logit_p) * lp, -20.0, 20.0);
  return 1.0 / (1.0 + std::exp(-z));
}

// Step-wise drafter decision at checkpoint `bucket` covering drafts [0, cur_w).
// Updates *survival (cumulative survival through cur_w) and optional *verify_bucket (what to verify if stopping now:
// `bucket`, or with `earlier_prefix` the best bucket in [0, bucket]); returns 1 to continue drafting, 0 to stop and
// verify *verify_bucket.  The policy's costs are whole rounds', draft steps and verify rows: an earlier prefix is
// priced right only where the steps already taken cost little beside the rows not verified
// (flyx_program::earlier_checkpoints).
inline int stop_decide_step_ref(const double* prob, const double* margin, const flyx_stop_policy& pol, int bucket,
                                int cur_w, bool earlier_prefix, double* survival, int* verify_bucket = nullptr) {
  const int n = std::clamp(pol.n_buckets, 1, FLYX_MAX_STOP_BUCKETS);
  const int b_cur = std::clamp(bucket, 0, n - 1);
  const int hi = std::clamp(cur_w, 0, FLYX_MAX_ROWS);
  if (!pol.enabled) {
    *survival = 1.0;
    if (verify_bucket) *verify_bucket = b_cur;
    return (b_cur + 1 < n) ? 1 : 0;
  }
  double S[FLYX_MAX_ROWS];
  double prod = 1.0;
  for (int k = 0; k < hi; ++k) {
    prod *= stop_draft_prob_ref(prob[k], margin[k], k, pol);
    S[k] = prod;
  }
  *survival = prod;
  const double c0 = double(pol.costs[0]);
  int best = earlier_prefix ? 0 : b_cur;
  double best_val = -1e30;
  double etok_cur = 1.0;
  for (int b = best; b <= b_cur; ++b) {
    const int wb = std::clamp(pol.widths[b], 0, hi);
    double etok = 1.0;
    for (int k = 0; k < wb; ++k) etok += S[k];
    if (b == b_cur) etok_cur = etok;
    const double val = etok - double(pol.lambda) * (double(pol.costs[b]) - c0);
    if (val > best_val) {
      best_val = val;
      best = b;
    }
  }
  if (verify_bucket) *verify_bucket = best;
  if (b_cur + 1 >= n) return 0;
  for (int nxt = b_cur + 1; nxt < n; ++nxt) {
    const int w_nxt = std::clamp(pol.widths[nxt], hi, FLYX_MAX_ROWS);
    double exp_more = 0.0;
    double prod_p = prod;
    for (int j = hi; j < w_nxt; ++j) {
      prod_p *= double(pol.p_pos[j]);
      exp_more += prod_p;
    }
    const double val_nxt = (etok_cur + exp_more) - double(pol.lambda) * (double(pol.costs[nxt]) - c0);
    if (val_nxt > best_val) return 1;
  }
  return 0;
}

// Parallel block drafter decision: given all W_max drafts' confidences, picks the bucket index maximizing
// expected tokens - lambda * incremental verify cost.
inline int stop_decide_block_ref(const double* prob, const double* margin, const flyx_stop_policy& pol) {
  const int n = pol.n_buckets;
  if (n <= 1 || !pol.enabled) return n > 0 ? n - 1 : 0;
  const int w_max = pol.widths[n - 1];
  double S[FLYX_MAX_ROWS];
  double prod = 1.0;
  for (int k = 0; k < w_max; ++k) {
    prod *= stop_draft_prob_ref(prob[k], margin[k], k, pol);
    S[k] = prod;
  }
  int best = 0;
  double best_val = -1e30;
  const double c0 = double(pol.costs[0]);
  for (int b = 0; b < n; ++b) {
    const int wb = pol.widths[b];
    double etok = 1.0;
    for (int k = 0; k < wb; ++k) etok += S[k];
    const double val = etok - double(pol.lambda) * (double(pol.costs[b]) - c0);
    if (val > best_val) {
      best_val = val;
      best = b;
    }
  }
  return best;
}

}  // namespace flyx::ops
