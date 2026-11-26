// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Drafter- and model-family-agnostic online confidence calibrator for intra-round optimal stopping.
// Uses diagonal second-order (Newton) online logistic updates for invariance to each model family's
// logit scale, smooth per-position Bayesian Beta-Binomial acceptance rates, monotonic cost
// sanitization, and scale-invariant initial throughput lambda (tokens/ms).
#pragma once

#include <array>
#include <vector>

#include "flyx/backend_abi.h"

namespace flyx {

class ConfidenceStopper {
 public:
  // Configure sorted verify checkpoints `widths` and their measured solo costs `costs_ms` (ms),
  // plus `plain_ms` (plain decoding round cost in ms) for scale-invariant initial lambda.
  void configure(const std::vector<int>& widths, const std::vector<double>& costs_ms, double plain_ms);

  // Start a new reply: resets per-reply throughput and position acceptance counts while retaining
  // learned head calibration (with softened curvature so topic/format shifts adapt quickly).
  void begin();

  bool ready() const { return policy_.n_buckets > 1; }
  const flyx_stop_policy& policy() const { return policy_; }
  int max_width() const { return policy_.n_buckets > 0 ? policy_.widths[policy_.n_buckets - 1] : 0; }
  double bucket_cost(int width) const;

  // Update online calibrator from one completed round of `width` drafts (`accepted` accepted)
  // with optional per-draft confidence arrays (may be null or zero when unavailable).  `cost_ms`: what the round cost,
  // for a fixed round of the drafter (its width may be none of the checkpoints); 0 takes its bucket's cost.
  void observe(int width, int accepted, const float* draft_prob, const float* draft_margin, double cost_ms = 0);

 private:
  void init_rates();
  void refresh_pos_probs();

  flyx_stop_policy policy_{};
  double plain_ms_ = 1.0;
  // Diagonal Fisher curvature accumulators for scale-invariant online Newton updates.
  float h0_ = 4.f, h1_ = 16.f, h2_ = 16.f;
  // Per-position decayed Beta-Binomial sufficient statistics + pooled empirical base rate.
  std::array<float, FLYX_MAX_ROWS> pos_hits_{};
  std::array<float, FLYX_MAX_ROWS> pos_trials_{};
  float tot_hits_ = 0.f, tot_trials_ = 0.f;
};

}  // namespace flyx
