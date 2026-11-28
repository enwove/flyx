// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Joint round scheduling for sessions running concurrently.
//
// Each running session has a small set of candidate rounds (RoundChooser::candidates):
// its preferred solo choice plus narrower batchable widths (plain, narrower drafter
// widths, narrower lookup widths).  A shared batch reads the target's weights once for
// all of its rows (at most FLYX_MAX_ROWS), and sessions sharing a drafter also share its
// weight reads in lockstep/grouped batches.  BatchCostModel estimates a batch's cost from
// the round and lookup timings calibrated at load; plan_batches() compares unconstrained First-Fit Decreasing packing against 1-batch joint width allocation
// maximizing proportional fairness — the geometric mean of every session's expected
// tokens per millisecond, (prod E_i)^(1/m) / T_total — which equals E/T for solo and
// homogeneous batches while preventing high-acceptance sessions from starving slower
// peers in heterogeneous batches.
#pragma once

#include <cstddef>
#include <vector>

#include "core/speculation/chooser.h"

namespace flyx {

// Calibrated cost of a shared batch: target_ms(total_rows) + extra_sessions + shared_draft_ms(items).
class BatchCostModel {
 public:
  // Marginal fraction of a session's solo draft cost paid by additional sessions sharing the same drafter
  // in a batch: lockstep MTP and grouped DFlash2 read drafter weights and the 1-bit draft head once across
  // concurrent sessions, leaving per-session mixer/attention/table overhead beyond the widest member.
  static constexpr double kSharedDraftShare = 0.15;
  // Fraction of a 1-row pass each session past the first adds to a batch (its own GDN/attention mixers, accept and
  // commit): Qwen3.8-27B BF16 on an RTX PRO 6000, 8 sessions decoding plainly in one batch of 8 rows cost 45.6 ms
  // against 37.5 for a round of 8 rows of one session (1.16 ms each past the first, 3.2% of the 35.8 ms 1-row pass).
  static constexpr double kExtraSessionShare = 0.035;

  // Builds the cost model from the engine's calibrated round and lookup timings.
  void build(const std::vector<int>& widths, const std::vector<int>& drafters, const std::vector<double>& round_ms,
             const std::vector<int>& lookup_widths, const std::vector<double>& lookup_ms);
  bool ready() const { return !knots_.empty(); }

  // Target-only verification time (ms) for `rows` rows (1..FLYX_MAX_ROWS),
  // piecewise-linearly interpolated from the calibrated plain and lookup timings.
  double target_ms(int rows) const;
  // Total shared-batch time (ms) for `n_sessions` sharing `rows` rows plus their `draft_ms`:
  // includes per-session mixer/state overhead beyond the single session already in target_ms().
  double batch_ms(int rows, int n_sessions, double draft_ms) const;
  // Marginal drafting time (ms) of candidate `c` beyond target verification.
  double draft_ms(const RoundCandidate& c) const;
  // Total drafting time (ms) of a batch whose members may share drafters:
  // for each drafter d >= 0 present, max_d + kSharedDraftShare * (sum_d - max_d).
  double shared_draft_ms(const std::vector<const RoundCandidate*>& items) const;

 private:
  struct Knot {
    int rows = 1;
    double ms = 1.0;
  };
  std::vector<Knot> knots_;
};

struct ScheduledRound {
  size_t session = 0;  // index into the input `sessions` vector
  RoundChoice choice{};
};

struct ScheduledBatch {
  std::vector<ScheduledRound> items;
};

// Assigns each session in `sessions` one of its candidates and groups them into
// batches of at most `max_rows` total rows (FLYX_MAX_ROWS, or up to FLYX_MAX_WIDE_ROWS where the backend runs rounds
// that wide: Engine::batch_rows).  Non-batchable sessions (tree rounds, relaxed acceptance) are isolated into solo
// batches so they never prevent other sessions from sharing a batch.
std::vector<ScheduledBatch> plan_batches(const BatchCostModel& cost,
                                         const std::vector<std::vector<RoundCandidate>>& sessions,
                                         int max_rows = FLYX_MAX_ROWS);

}  // namespace flyx
