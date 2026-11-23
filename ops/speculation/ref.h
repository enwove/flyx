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

// Sequential drafter decision at checkpoint `bucket` covering drafts [prev_w, cur_w).
// Updates *survival (initialized to 1.0 when bucket == 0) and returns 1 to continue drafting, 0 to verify now.
inline int stop_decide_step_ref(const double* prob, const double* margin, const flyx_stop_policy& pol, int bucket,
                                int prev_w, int cur_w, double* survival) {
  double S = bucket == 0 ? 1.0 : *survival;
  if (!pol.enabled) {
    *survival = S;
    return (bucket + 1 < pol.n_buckets) ? 1 : 0;
  }
  for (int k = prev_w; k < cur_w; ++k) S *= stop_draft_prob_ref(prob[k], margin[k], k, pol);
  *survival = S;
  if (bucket + 1 >= pol.n_buckets) return 0;
  const double c_cur = double(pol.costs[bucket]);
  for (int nxt = bucket + 1; nxt < pol.n_buckets; ++nxt) {
    const int w_nxt = pol.widths[nxt];
    double exp_more = 0.0;
    double prod_p = S;
    for (int j = cur_w; j < w_nxt; ++j) {
      prod_p *= double(pol.p_pos[j]);
      exp_more += prod_p;
    }
    const double dc = double(pol.costs[nxt]) - c_cur;
    if (exp_more > double(pol.lambda) * dc) return 1;
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
