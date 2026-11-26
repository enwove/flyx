// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Which round a session launches next: the one place that decides it.
//
// Two sources of drafts: the model's drafters, the drafter and width
// SpecController picks from measured cost and the acceptance each has had,
// and the sequence itself (LookupDrafter):
// when the text just written occurred before, a round verifies what followed
// it there, with no drafter run.
//
// A lookup round's drafts continue the tokens before it, so the round ahead
// must have been read when it is launched: while the text just written also
// appears earlier, one round is in flight instead of two.  A lookup round
// that accepts fewer than kLookupWorth drafts cost more than the drafter's
// round would have (a chance match): lookup then rests for kLookupRest tokens.
// Nor is a round looked up (of any width) where the controller would run a drafter whose state
// follows its own rounds only (DV's canvas, SpecController): a lookup round would
// leave that state behind, and the drafter's next round would have to warm it up
// again.
//
// The generation loop launches what next() names and reports each result; it
// holds no speculation policy of its own.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/speculation/confidence.h"
#include "core/speculation/controller.h"
#include "core/speculation/lookup.h"

namespace flyx {

// A round to launch.
struct RoundChoice {
  bool lookup = false;    // its drafts are given (RoundChooser::drafts()); no drafter runs
  size_t index = 0;       // which plan: of the drafters' rounds, or of the lookup widths
  int width = 0;          // drafts the round verifies (max width when adaptive)
  int drafter = -1;       // the drafter that drafts them (-1: none, or looked up)
  bool adaptive = false;  // run the drafter's intra-round adaptive stopping plan
};

// One option a session can run on its next round, for joint scheduling across
// concurrent sessions (core/speculation/batch_schedule.h).
struct RoundCandidate {
  RoundChoice choice{};
  double expected_tokens = 1.0;  // expected tokens this round will produce
  double solo_ms = 1.0;          // measured solo cost in ms
  bool batchable = true;         // false for rounds that must run alone (trees, relaxed acceptance)
  bool preferred = false;        // the solo unconstrained choice
};

class RoundChooser {
 public:
  static constexpr int kLookupWorth = 4, kLookupRest = 32;

  // The rounds to choose from (plan::Plans): round i verifies widths[i] drafts
  // of drafter drafters[i] (-1: none, plain decoding) at its measured cost.
  // lookup_widths: the widths of rounds whose drafts are given (empty: no such
  // rounds).  trunks, cost_per_token, cost_at: as SpecController::reset (null:
  // no trees; a round's cost whatever the context).  lookup_cost_ms: measured
  // costs of lookup_widths (optional).  The vectors must outlive the chooser.
  // Starts the acceptance history over.
  void configure(const std::vector<int>* widths, const std::vector<int>* drafters, const std::vector<double>* cost_ms,
                 const std::vector<int>* lookup_widths, const std::vector<int>* trunks = nullptr,
                 const std::vector<double>* lookup_cost_ms = nullptr,
                 const std::vector<std::vector<int>>* adaptive_checkpoints = nullptr,
                 const std::vector<double>* cost_per_token = nullptr, int cost_at = 0,
                 SharedTokens* shared = nullptr);
  bool configured() const { return widths_ != nullptr; }

  // A reply begins (after its prompt's prefill).  lookup: it may take drafts
  // from the sequence; wide: its lookup rounds may verify more than
  // FLYX_MAX_ROWS - 1 drafts; adaptive: it may use intra-round confidence
  // stopping on drafters with an adaptive plan.  What the session has learned
  // (acceptance, the index of its tokens) is kept; no drafter's state follows a
  // round now.
  void begin(bool lookup, bool wide = true, bool adaptive = true);
  // Current online stopping policy for drafter `d` (null if unavailable).
  const flyx_stop_policy* stop_policy(int drafter) const;
  // Rounds that may be in flight together: FLYX_MAX_IN_FLIGHT (two), so
  // the device runs round k + 1 while the host handles round k; one while the
  // next round's drafts are to be looked up from where the round ahead ends.
  int in_flight_limit() const { return lookup_mode_ ? 1 : FLYX_MAX_IN_FLIGHT; }
  // The round to launch after `tokens` (the sequence so far), with
  // `in_flight` rounds launched and not yet read.
  RoundChoice next(const std::vector<int32_t>& tokens, int in_flight);
  // Candidate rounds for joint scheduling across concurrent sessions.  Does not
  // advance chooser counters; follow with commit() for the chosen candidate.
  // If lookup candidates are returned, drafts() holds the lookup proposal until
  // the next call.  A tree round (DV's among them) never shares a round with
  // other sessions' (batchable false), and a drafter's round that keeps a
  // state is never offered in place of the preferred.
  void candidates(const std::vector<int32_t>& tokens, int in_flight, std::vector<RoundCandidate>* out);
  // Commits one of the candidates returned by candidates().
  void commit(const RoundChoice& choice, int in_flight);
  // The drafts of the lookup round next() / candidates() just proposed (valid until the next call).
  const std::vector<int32_t>& drafts() const { return proposal_; }
  // The drafter's round whatever the text, at a context of `context` tokens:
  // the round that chooses a token again after the grammar rejected it.
  RoundChoice next_drafted(int context);
  // Tests: launch these rounds (indices into the plans, Engine::load checks
  // EngineOptions::cycle's) in turn instead of choosing among them; empty:
  // choose.
  void cycle(std::vector<size_t> rounds) {
    cycle_ = std::move(rounds);
    cycled_ = 0;
  }
  // The result of the oldest round in flight.  emitted: it produced tokens
  // (a round launched past the end of the reply does not).  tokens: the
  // sequence after it; in_flight: rounds still not read.  learn: the reply
  // goes on (false for rounds read after it ended).  trunk_accepted: a tree's
  // drafts its trunk accepted (SpecController::observe).
  void result(const RoundChoice& round, bool emitted, int accepted, const std::vector<int32_t>& tokens, int in_flight,
              bool learn, int trunk_accepted = 0, int actual_width = -1, const float* draft_prob = nullptr,
              const float* draft_margin = nullptr);

  // The sequence was cut back to n tokens (a rollback, a new prompt): no
  // drafter's state follows a round any more.
  void cut(size_t n) {
    lookup_.cut(n);
    spec_.restart();
  }
  // The sequence is gone (after an error).
  void forget() {
    lookup_.reset();
    spec_.restart();
  }
  // The session's rounds go out with other sessions' (the scheduler's rounds of several sessions) or alone again: a
  // drafter whose rounds run alone (DV) does not draft meanwhile (SpecController::batched).
  void batched(bool on) { spec_.batched(on); }

 private:
  // Whether a round may be looked up at this context: lookup is on and not
  // resting, and the controller would not run a drafter that keeps a state.
  bool looks_up(size_t context) const;

  SpecController spec_;
  LookupDrafter lookup_;
  std::vector<ConfidenceStopper> stoppers_;
  bool adaptive_on_ = false;
  const std::vector<int>* widths_ = nullptr;
  const std::vector<int>* drafters_ = nullptr;
  const std::vector<int>* lookup_widths_ = nullptr;
  const std::vector<double>* lookup_cost_ = nullptr;
  double lookup_accept_ = 0.75;  // EMA of accepted / width on lookup rounds
  bool lookup_on_ = false;       // this reply may use lookup rounds
  // The text just written appears earlier: the next round is to be a lookup
  // round.  Looked at again whenever nothing is in flight (where no round may
  // be looked up, as while a drafter that keeps a state drafts, it is cleared);
  // a rest clears it.
  bool lookup_mode_ = false;
  size_t rests_until_ = 0;       // sequence length before which lookup proposes nothing
  // Wide lookup rounds (lookup widths past FLYX_MAX_ROWS - 1, as YASAKANI's wide copy passes): taken after
  // wide_need_ lookup rounds of FLYX_MAX_ROWS - 1 drafts in a row were accepted whole (full_runs_); every
  // kWideJudge wide rounds (or at one not accepted whole) their tokens a millisecond, by the rounds' measured costs,
  // are held against a full round of FLYX_MAX_ROWS - 1: short of it, back to those rounds and wide_need_ doubles (up
  // to kWideNeedMost), else it is kWideNeed again.
  static constexpr int kWideNeed = 2, kWideNeedMost = 16, kWideJudge = 4;
  bool wide_ok_ = false;         // this reply may take wide rounds (no penalty, relaxed acceptance or grammar)
  int full_runs_ = 0, wide_need_ = kWideNeed, wide_rounds_ = 0;
  double wide_tokens_ = 0, wide_ms_ = 0;
  // Which width once widened: each lookup draft taken as accepted with one chance p (miss_ of the drafts the lookup
  // rounds put to the test, look_, the last ten rounds or so), a round of w drafts gives 1 + p (1 - p^w) / (1 - p)
  // tokens; the width that gives the most a millisecond (a copy that breaks every 30 tokens or so: 31, an unbroken
  // one: 63).  Before any round: one miss in 64 drafts.
  static constexpr double kLookDecay = 0.9, kLookPriorDrafts = 16, kLookPriorMisses = 0.25;
  double look_ = kLookPriorDrafts, miss_ = kLookPriorMisses;
  bool wide_ready() const { return wide_ok_ && full_runs_ >= wide_need_; }
  double lookup_cost(size_t k) const;
  double lookup_rate(size_t k) const;  // expected tokens a millisecond of lookup width k
  // A drafter round that kept almost all of a long chain (at least kRunWidth drafts, all but one accepted): the next
  // round of that drafter goes one width wider (a DFlash2 block of 8 drafting 15: basalt's wide block after a window
  // that kept most of a block).  Its drafts past the chain are accepted far more often after such a round than on
  // average, where the controller's per-width choice cannot see it (RTX PRO 6000, the 27B's 7 -> 15 drafts cost
  // 15.2 -> 19.1 ms).  An adaptive round counts by the drafts it verified (its checkpoint), so one that verified
  // kRunWidth or more and kept all but one starts a run too; a run the drafter has no wider round for holds its width,
  // and a width past the adaptive round's reach stays a fixed round while its runs last.
  // Only while it pays: every wider round of a drafter (past kRunWidth drafts) is held against the kRunWidth round it
  // replaced, as if that round had kept the same first drafts (its tokens min(accepted, kRunWidth) + 1, at its cost),
  // over the last ten or so such rounds of the reply; wider rounds that give fewer tokens a millisecond stop it (prose
  // and lists on the 27B, where a near-whole chain of 7 says little of the drafts past it), with a try every kRunProbe
  // runs.
  static constexpr int kRunWidth = 7, kRunProbe = 16;
  static constexpr double kRunDecay = 0.9;
  int run_drafter_ = -1, run_width_ = 0;  // the last round's drafter and width when it was such a run, else -1
  double wide_run_tok_ = 0, wide_run_ms_ = 0, alt_run_tok_ = 0, alt_run_ms_ = 0;
  int run_skips_ = 0;
  bool runs_pay() const { return wide_run_ms_ <= 0 || wide_run_tok_ * alt_run_ms_ >= alt_run_tok_ * wide_run_ms_; }
  std::vector<int32_t> proposal_;
  std::vector<size_t> cycle_;    // cycle()
  size_t cycled_ = 0;
  int context_ = 0;              // the context the last candidates() were for (commit()'s)
};

}  // namespace flyx
