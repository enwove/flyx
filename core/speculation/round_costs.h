// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What a round of each kind costs at a context, learned from the rounds the sessions read.
#pragma once

#include <cstddef>
#include <mutex>
#include <vector>

#include "core/speculation/controller.h"

namespace flyx {

// The costs the speculation controller chooses by, one engine's, its sessions' rounds teaching them (any thread).
// The load's timings start them (spec::time_rounds at a short context; some drafters' rounds again at a longer one),
// but a round's cost moves with the context: the K/V its attention reads, past the GPU's part over the bus, past
// auto's switch through a 2-bit copy, and more of it the more rows a round verifies.  So round i costs, at a context
// of c tokens,
//   cost(i, c) = a_i + b_i * x,  x = max(0, c - at) / 1000,
// the line that fits the rounds of it the sessions read (observe(): a session reading alone, Generation::round says
// when its clock tells a round's cost), each observation weighing kKeep times less for every later one of the same
// round, held to the load's line by two priors: its cost at `at` counting as kPriorRounds rounds there, its cost per
// token (the load's, 0 where none was timed) as kSlopeRounds rounds' worth of slope; its slope no less than 0 and its
// cost at `at` no less than kFloor times the load's.  An observation is first clipped to within kClip times what the
// round is priced at, so a stall now and then (another program, a busy host) moves it little while a real change
// still moves it all the way in a few rounds.  A wider round of a drafter's chain is priced at no less than the next
// narrower one (reset's `narrower`; an adaptive round that stops early teaches the narrower one only).  A round no
// session has read at a long context is priced by its load's line: low, so the controller chooses it there, and its
// cost is then seen.  A round the load did not time (no cost, or none finite) is priced at the cheapest the load
// timed until it is read, and takes its first reading for the load's.
// The adaptive stoppers (ConfidenceStopper) keep the load's costs: at the learned costs of 35K tokens they stopped
// earlier and their rounds gave 2 to 3% fewer tokens a second (Qwen3.5-4B NVFP4's MTP, RTX 5060 Ti).
class RoundCosts final : public RoundCostModel {
 public:
  static constexpr double kKeep = 0.98, kPriorRounds = 8.0, kSlopeRounds = 2.0, kClip = 2.0, kFloor = 0.5;

  // The load's timings: base_ms[i] at a context of `at` tokens, per_token[i] ms more for each token past it (null:
  // none); narrower[i] the round of the same drafter's chain one width narrower (-1, or null: none).  What was learned
  // is forgotten.
  void reset(const std::vector<double>& base_ms, const std::vector<double>* per_token, int at,
             const std::vector<int>* narrower = nullptr);
  size_t size() const;
  double cost(size_t round, int context) const override;
  // Round `round`, launched at a context of `context` tokens, took `ms` (the clock between two reads).
  void observe(size_t round, int context, double ms);

 private:
  struct Line {
    double a0 = 0, b0 = 0;                          // the load's: ms at `at`, ms per 1000 tokens past it
    int narrower = -1;                              // reset()'s
    double s1 = 0, sx = 0, sxx = 0, sy = 0, sxy = 0;  // the observations' sums, each kept kKeep a later one
  };
  double x_of(int context) const;
  static double fit(const Line& l, double x);
  double priced(size_t round, double x) const;  // fit(), no lower than the narrower round's (mu_ held)

  mutable std::mutex mu_;
  std::vector<Line> lines_;
  double untimed_ = 1.0;  // an untimed round's price until it is read (reset())
  int at_ = 0;
};

}  // namespace flyx
