// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/confidence.h"

#include <algorithm>
#include <cmath>

namespace flyx {

namespace {
constexpr float kInitBias = -0.20f;
constexpr float kInitMargin = 0.25f;
constexpr float kInitLogitP = 0.20f;
constexpr float kInitH0 = 4.f;
constexpr float kInitH1 = 16.f;
constexpr float kInitH2 = 16.f;
constexpr float kCurvDecay = 0.985f;
constexpr float kWeightDecay = 0.005f;  // mild L2 pull toward 0 so uncalibrated heads degrade cleanly to base rate
constexpr float kPosPriorBase = 0.78f;
constexpr float kPosPriorStep = 0.99f;
constexpr float kPosPriorN = 2.0f;
constexpr float kPosCountDecay = 0.93f;
constexpr float kTotCountDecay = 0.96f;
constexpr float kLamDecay = 0.90f;

float sigmoidf(float z) {
  z = std::clamp(z, -20.f, 20.f);
  return 1.f / (1.f + std::exp(-z));
}
}  // namespace

void ConfidenceStopper::configure(const std::vector<int>& widths, const std::vector<double>& costs_ms,
                                  double plain_ms) {
  policy_ = {};
  plain_ms_ = plain_ms > 1e-6 ? plain_ms : 1.0;
  const int n = static_cast<int>(std::min<size_t>({widths.size(), costs_ms.size(), FLYX_MAX_STOP_BUCKETS}));
  if (n < 2) return;
  policy_.enabled = 1;
  policy_.n_buckets = n;
  for (int i = 0; i < n; ++i) {
    policy_.widths[i] = widths[size_t(i)];
    float c = static_cast<float>(costs_ms[size_t(i)] > 1e-6 ? costs_ms[size_t(i)] : plain_ms_);
    // Monotonic sanitization: clock jitter on fast small models must never make incremental cost <= 0.
    if (i > 0 && c <= policy_.costs[i - 1]) c = policy_.costs[i - 1] + 1e-3f;
    policy_.costs[i] = c;
  }
  policy_.w_bias = kInitBias;
  policy_.w_margin = kInitMargin;
  policy_.w_logit_p = kInitLogitP;
  h0_ = kInitH0;
  h1_ = kInitH1;
  h2_ = kInitH2;
  init_rates();
}

void ConfidenceStopper::refresh_pos_probs() {
  // Hierarchical Empirical Bayes: unobserved deeper positions inherit this model & prompt's live acceptance rate
  // (shrunk gently toward kPosPriorBase when cold), avoiding cold-start traps past shorter checkpoints.
  const float p_base = (kPosPriorN * kPosPriorBase + tot_hits_) / (kPosPriorN + tot_trials_);
  for (int k = 0; k < FLYX_MAX_ROWS; ++k) {
    const float prior_k = std::clamp(p_base * std::pow(kPosPriorStep, float(k)), 0.05f, 0.99f);
    policy_.p_pos[k] = (kPosPriorN * prior_k + pos_hits_[size_t(k)]) / (kPosPriorN + pos_trials_[size_t(k)]);
  }
}

void ConfidenceStopper::init_rates() {
  pos_hits_.fill(0.f);
  pos_trials_.fill(0.f);
  tot_hits_ = tot_trials_ = 0.f;
  refresh_pos_probs();
  // Scale-invariant initial lambda: best expected tokens/ms under the prior and this GPU's measured costs.
  double best = 1.0 / plain_ms_;
  for (int b = 0; b < policy_.n_buckets; ++b) {
    const int wb = std::min(policy_.widths[b], FLYX_MAX_ROWS - 1);
    double etok = 1.0, prod = 1.0;
    for (int k = 0; k < wb; ++k) {
      prod *= double(policy_.p_pos[k]);
      etok += prod;
    }
    const double c = double(policy_.costs[b]);
    if (c > 1e-6) best = std::max(best, etok / c);
  }
  policy_.lambda = static_cast<float>(best);
}

void ConfidenceStopper::begin() {
  if (!ready()) return;
  init_rates();
}

double ConfidenceStopper::bucket_cost(int width) const {
  if (policy_.n_buckets <= 0) return plain_ms_;
  for (int i = 0; i < policy_.n_buckets; ++i)
    if (policy_.widths[i] == width) return double(policy_.costs[i]);
  return double(policy_.costs[policy_.n_buckets - 1]);
}

void ConfidenceStopper::observe(int width, int accepted, const float* draft_prob, const float* draft_margin,
                                double cost_ms) {
  if (!ready() || width <= 0) return;
  const double c = cost_ms > 0 ? cost_ms : bucket_cost(width);
  const float rate = static_cast<float>((accepted + 1) / c);
  policy_.lambda = kLamDecay * policy_.lambda + (1.f - kLamDecay) * rate;

  const int obs = std::min({accepted + 1, width, FLYX_MAX_ROWS});
  for (int k = 0; k < obs; ++k) {
    const float y = k < accepted ? 1.f : 0.f;
    pos_hits_[size_t(k)] = kPosCountDecay * pos_hits_[size_t(k)] + y;
    pos_trials_[size_t(k)] = kPosCountDecay * pos_trials_[size_t(k)] + 1.f;
    tot_hits_ = kTotCountDecay * tot_hits_ + y;
    tot_trials_ = kTotCountDecay * tot_trials_ + 1.f;
    if (!draft_prob || !draft_margin || draft_prob[k] <= 0.f) continue;
    const float p = std::clamp(draft_prob[k], 1e-5f, 1.f - 1e-5f);
    const float lp = std::clamp(std::log(p / (1.f - p)), -6.f, 10.f);
    const float mg = std::clamp(draft_margin[k], 0.f, 12.f);
    const float pk = sigmoidf(policy_.w_bias + policy_.w_margin * mg + policy_.w_logit_p * lp);
    const float err = y - pk;
    const float var = std::max(pk * (1.f - pk), 0.05f);
    h0_ = kCurvDecay * h0_ + var;
    h1_ = kCurvDecay * h1_ + var * mg * mg;
    h2_ = kCurvDecay * h2_ + var * lp * lp;
    policy_.w_bias = std::clamp(policy_.w_bias + err / (h0_ + 2.f), -4.f, 4.f);
    policy_.w_margin =
        std::clamp(policy_.w_margin + (err * mg - kWeightDecay * policy_.w_margin) / (h1_ + 4.f), 0.f, 4.f);
    policy_.w_logit_p =
        std::clamp(policy_.w_logit_p + (err * lp - kWeightDecay * policy_.w_logit_p) / (h2_ + 4.f), 0.f, 4.f);
  }
  refresh_pos_probs();
}

}  // namespace flyx
