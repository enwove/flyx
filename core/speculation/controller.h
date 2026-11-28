// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The speculation controller: which drafter drafts the next round, and how many drafts it verifies.
#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <memory>
#include <mutex>
#include <vector>

#include "flyx/backend_abi.h"

namespace flyx {

// What an engine's sessions' trees gave a round, by round (the counts kept kKeep a round): a tree without a trunk its
// tokens, a tree with one what its branches added past its trunk (a round is of one kind or the other); the means every
// reply's estimate of them starts from (SpecController::reply, a session's first reply too); and the engine's credit
// (ms) for probing a drafter whose state follows its own rounds only.  Any thread.
class SharedTokens {
 public:
  static constexpr double kKeep = 0.998;
  void resize(size_t rounds) {
    std::lock_guard<std::mutex> lock(mu_);
    sum_.assign(rounds, 0.0);
    n_.assign(rounds, 0.0);
    credit_ = 0;
  }
  void add(size_t round, double tokens) {
    std::lock_guard<std::mutex> lock(mu_);
    if (round >= n_.size()) return;
    sum_[round] = kKeep * sum_[round] + tokens;
    n_[round] = kKeep * n_[round] + 1;
  }
  // The mean, `guess` counting as `weight` rounds.
  double mean(size_t round, double guess, double weight) const {
    std::lock_guard<std::mutex> lock(mu_);
    const double s = round < n_.size() ? sum_[round] : 0.0, n = round < n_.size() ? n_[round] : 0.0;
    return (weight * guess + s) / (weight + n);
  }
  double credit() const {
    std::lock_guard<std::mutex> lock(mu_);
    return credit_;
  }
  // The credit plus `ms`, kept within [0, cap].
  void add_credit(double ms, double cap) {
    std::lock_guard<std::mutex> lock(mu_);
    credit_ = std::clamp(credit_ + ms, 0.0, std::max(0.0, cap));
  }

 private:
  mutable std::mutex mu_;
  std::vector<double> sum_, n_;
  double credit_ = 0;
};

// What a round costs at a context (RoundCosts: the load's timings, then the rounds the engine's sessions read).
class RoundCostModel {
 public:
  virtual ~RoundCostModel() = default;
  virtual double cost(size_t round, int context) const = 0;
};

// Chooses the next round among plain decoding and every drafter's widths
// by expected tokens per millisecond at the context the round runs at (each
// round's cost by a RoundCostModel where reset() is given one: measured at load
// and learned from the rounds read; else measured at load, and where a
// drafter's state follows its own rounds only, at two lengths of context: a
// round costs its cost at the first plus its cost per token of context past it):
//   * plain decoding or a drafted round, by each drafter's per-position
//     acceptance as a moving average (with one drafter, the whole choice);
//   * drafted, the drafter: the one whose best round its drafter estimate
//     rates highest, at its best width by the moving average.  Per position,
//     the drafter estimate's mean m = (kPrior * prior_i + hits_i) / n,
//     n = kPrior + rounds_i, its counts kept kKeepCounts a round it drafts and
//     forgotten kForget a round it does not; the estimate is optimistic by
//     kUcb standard errors, m + kUcb sqrt(m (1 - m) / n), so a drafter seen
//     little, or not for a while, drafts again before the others' estimates
//     settle it.  Every drafter drafts once first.
// Only the drafter that drafted a round learns from it, so after kProbe rounds
// of plain decoding in a row a drafter drafts the next one: plain decoding
// notices a text that has become predictable.  Replayed on recorded rounds of
// Qwen3.8-27B's two drafters, whichever is tried first, this keeps 0.95 of the
// better drafter's speed for the whole text (0.89 before: the drafter tried
// first kept the rounds; bench/replay/drafter_choice.py,
//).
//
// A drafter's tree (FLYX_DRAFT_TREE) is its chain of `trunk` drafts and
// branches.  Where its trunk stops (k of its drafts accepted, at the chain's
// acceptance by position, which its rounds teach as much as the chain's: the
// trunk is the chain) its branches add (B + sum_k) / (1 + n_k), the tokens its
// rounds of that k gave beyond their trunk's, so its tokens are the sum over k
// of P(the trunk accepts k) (1 + k + that): branches matter most where the trunk
// fails early, which one mean beside the chain's tokens leaves out (EAGLE-4's
// trees of 15 on chat prompts: where its chain's first drafts failed, the
// anchor's four children still gave a token most of the time, and one mean had
// chains of 3 chosen there over trees that paid more).  B, the tree's prior:
// (prior + sum) / (1 + n) over all its rounds, the prior what the engine's
// sessions' rounds of it added (SharedTokens, kBonusPrior counting as
// kGuessRounds rounds: a reply starts from what the replies before it learned;
// kBonusPrior is what DFlash2's first two branches add on Qwen3.8-27B, 0.3 to
// 0.46, EAGLE-4's add about a token).  The sums and counts are kept kBonusKeep
// a tree round (the gain is a few tenths of a token and noisy: some 30 rounds),
// and a tree its drafter has not drafted for kTreeProbe of its rounds drafts
// the next one (the one idle longest), so what its branches add stays known
// at a few percent of a round's cost where it loses.  So a tree wins its rounds
// where its rows cost little against what its branches add: the round costs
// are this GPU's, measured at load.
//
// A tree without a trunk (trunks[i] < 0: DV's, whose drafts are all
// branches) is rated by the tokens its rounds give: what the engine's sessions'
// rounds of it gave (SharedTokens: its mean, kept SharedTokens::kKeep a round,
// a guess counting as kGuessRounds rounds, the tokens a chain of its width gives
// at the prior acceptance: 4.0 for 15 drafts and for 31), times the session's
// level, how its drafter's rounds in this reply fared against those means:
// (kTreePrior + sum of tokens / mean) / (kTreePrior + n), the counts kept
// kKeepCounts a round it drafts and 1 - kLevelForget a round it does not (0: as
// they are; a level forgotten back to the engine's means would enter the
// drafter again where the reply has shown that it does not pay: probes look
// again), started over by a reply (reply()).  So the width is chosen by the
// engine's means alone (tokens per ms at the context), not by which width's own
// rounds were lately the
// luckier (a tree round gives 1 to 32 tokens), and a reply's text moves every
// width alike.  A tree its drafter has not drafted for kWidthTurn of its rounds
// drafts the next (the one idle longest), so every width's mean stays known.
//
// A drafter with a round of no drafts (widths[i] == 0, drafters[i] = d: its
// warm-up, Model::restarts) keeps a state that follows its own rounds only
// (DV's canvas): any other round, a prefill or a rewind (restart()) leaves it
// behind, and its next round must be the warm-up (plain decoding that brings
// the state along: one token, at its cost).  Its trees share no round of
// several sessions (a tree runs alone): while the engine batches the session's
// rounds with others' (batched()), the drafter does not draft.  Otherwise, at
// its best width (T its expected tokens there, c the round's cost) such a
// drafter
//   * stays while T / c is at least the best other round's tokens per ms (the
//     other drafters' and plain decoding's, as above), and after a probe's
//     warm-up for kProbeRounds rounds whatever its estimate (a probe that only
//     warmed up would learn nothing), up to kProbeMax while the probe's own
//     rounds pay (its few rounds move an estimate that counts the engine's
//     means as kTreePrior rounds too little to keep it where it pays);
//   * is entered (its warm-up, then its rounds) where an entry pays over
//     kPayBack rounds: (1 + kPayBack T) / (warm-up + kPayBack c) tokens per ms
//     against the best other round's;
//   * else is probed, after kProbe rounds away, when the engine's credit covers
//     a probe at its worst (the warm-up and a round accepting nothing, against
//     the two tokens the other rounds would give).  The credit is kProbeShare
//     of every round's time plus what the drafter's rounds gave over the other
//     rounds' rate for as long (less where they gave less), shared by the
//     engine's sessions (a reply after replies of another kind probes on what
//     they saved), 0 when the engine loads, between 0 and kCreditCap probes at
//     their worst at the context: where the drafter does not pay, its probes
//     cost about kProbeShare of the time.
// docs/design/dv.md has the numbers that set these (bench/replay/dv_choice
// replays the rule, its constants set otherwise, on recorded rounds).
class SpecController {
 public:
  static constexpr int kProbe = 32, kTreeProbe = 64, kWidthTurn = 64;
  static constexpr double kPrior = 2.0, kKeepCounts = 0.95, kForget = 0.03, kUcb = 2.0;
  static constexpr double kBonusPrior = 0.4, kBonusKeep = 0.97;
  static constexpr int kPayBack = 8, kProbeRounds = 2, kProbeMax = 8;
  static constexpr double kTreePrior = 8.0, kGuessRounds = 8.0, kLevelForget = 0.0, kProbeShare = 0.005;
  static constexpr double kCreditCap = 1.25;

  // Round i verifies widths[i] drafts of drafter drafters[i] (-1: none) and costs cost_ms[i] at a context of cost_at
  // tokens, cost_per_token[i] ms more for every token of context past it (null: none); trunks (null: no trees):
  // trunks[i] > 0 when round i is a tree whose first trunks[i] drafts are the drafter's chain, < 0 a tree without a
  // trunk.  shared: the engine's sessions' rounds of trees without a trunk and their probe credit (null: the
  // controller's own).  costs: what the rounds cost at a context in their place (null: cost_ms and
  // cost_per_token).  The vectors and the model must outlive the controller.  Starts the acceptance history over.
  void reset(const std::vector<int>* widths, const std::vector<int>* drafters, const std::vector<double>* cost_ms,
             const std::vector<int>* trunks = nullptr, const std::vector<double>* cost_per_token = nullptr,
             int cost_at = 0, SharedTokens* shared = nullptr, const RoundCostModel* costs = nullptr);
  // The round to launch next at a context of `context` tokens (index into widths), and the controller's account of
  // it (it counts as launched): preview(context), then commit().
  int choose(int context = 0);
  // What choose() would return now at a context of `context` tokens, without counting it (nor advancing the idle and
  // probe counters).  max_width: the most drafts a round the drafters that do not restart may verify; allow_trees:
  // false leaves their tree rounds out.  forced (optional): set true when a mandatory probe (kProbe, kTreeProbe)
  // forced the round; untried (optional): set true when the first untried drafter is being tried; other_ms
  // (optional): ms a token of the best round of the drafters that do not restart and plain decoding (what a
  // restarting drafter's rounds are held to).
  int preview(int context = 0, int max_width = kAnyWidth, bool allow_trees = true, bool* forced = nullptr,
              bool* untried = nullptr, double* other_ms = nullptr) const;
  int peek(int context = 0) const { return preview(context); }
  // Records that `round` was launched at a context of `context` tokens (the untried drafters, the idle and probe
  // counters, the forgotten counts, a restarting drafter's state and credit), whichever round the caller chose
  // (choose()'s, or one of several sessions' rounds scheduled together: core/speculation/batch_schedule.h).
  void commit(size_t round, int context = 0);
  // Round i's result: its drafts accepted, and for a tree those its trunk accepted (the common start of its accepted
  // tokens and its first trunks[i] drafts).
  void observe(size_t round, int accepted, int trunk_accepted = 0);
  // Every drafter's state that follows its own rounds only is behind (a prefill, a rewind, a restore, a lookup round:
  // RoundChooser::commit): its next round warms it up.
  void restart();
  // A reply begins (after its prompt's prefill): restart(), and the trees without a trunk start from the engine's
  // means again (the session's level forgotten).
  void reply();
  // Round i is a round of a drafter whose state follows its own rounds only (its warm-up included).
  bool keeps_state(size_t round) const;
  // The session's rounds go out in rounds of several sessions (the scheduler's), or alone again: a
  // drafter whose state follows its own rounds only drafts only alone (its rounds share no round with another's).
  void batched(bool on) { batched_ = on; }
  // Tokens a round of `width` drafts of `drafter` is expected to give (1 with no drafter): by the moving average,
  // or by the drafter estimate.
  double expected_tokens(int drafter, int width, bool drafter_estimate = false) const;
  // Tokens round i is expected to give (a tree: its trunk's and its branches'; a tree without a trunk: its own).
  double round_tokens(size_t round, bool drafter_estimate = false) const;
  // Round i's cost at a context of `context` tokens: the cost model's (reset's `costs`), else the load's; kLeastMs at
  // least (the rates divide by it).
  static constexpr double kLeastMs = 1e-3;
  double cost(size_t round, int context) const;
  // ... the load's in any case (what the adaptive stoppers' checkpoints are priced at).
  double load_cost(size_t round, int context) const;
  // Round i is a tree round, with a trunk or without one (such a round runs alone: launch_batch).
  bool is_tree(size_t round) const { return tree(round) || bare(round); }

 private:
  // A round's drafts by position: a tree round's rows at most (FLYX_MAX_WIDE_ROWS tokens, its drafts one fewer).
  static constexpr int kPositions = FLYX_MAX_WIDE_ROWS - 1;
  using PerPosition = std::array<double, kPositions>;
  static constexpr int kAnyDrafter = -2;
  static constexpr int kAnyWidth = INT_MAX;  // preview(): no limit on a round's drafts
  // The acceptance of drafter's draft i (its drafts before it accepted): the estimate's or the moving average.
  double accept_at(int drafter, int i, bool drafter_estimate) const;
  // The round with the most expected tokens per ms among drafter's (kAnyDrafter: every drafter's whose state does
  // not restart; with_plain: and plain decoding's), of max_width drafts at most, a tree only when allow_trees.
  int best_round(int drafter, bool with_plain, bool drafter_estimate, int context, int max_width = kAnyWidth,
                 bool allow_trees = true) const;
  // A restarting drafter's round at this context: its best tree (or the one whose turn it is), and the warm-up that
  // enters it when it is behind; -1 where it neither stays nor enters against `other` (tokens per ms).  *probe: the
  // warm-up is a probe's (the drafter's estimate does not say it pays).
  int restarting_round(int d, double other, int context, double* rate, bool* probe) const;
  // The worst a probe of drafter d's round i costs (ms): its warm-up and round i accepting nothing, against the two
  // tokens the other rounds give at ms_per_token.
  double worst_probe(int d, size_t i, double ms_per_token, int context) const;

  // A tree without a trunk's tokens a round: the engine's mean (SharedTokens, the guess counted) and the session's
  // level against it.
  double mean_tokens(size_t i) const { return shared_->mean(i, tok_guess_[i], kGuessRounds); }
  double level(int d) const { return (kTreePrior + level_sum_[size_t(d)]) / (kTreePrior + level_n_[size_t(d)]); }
  // A drafter's tree without a trunk with the most tokens per ms by the engine's means at this context (-1: none).
  int best_width(int d, int context) const;
  // The credit's bound at this context: kCreditCap probes at their worst (0 without a rate to hold them to).
  double credit_cap(int d, int context) const;

  bool tree(size_t i) const { return trunks_ && (*trunks_)[i] > 0; }
  bool bare(size_t i) const { return trunks_ && (*trunks_)[i] < 0; }
  bool restarts(int d) const { return d >= 0 && warm_up_[size_t(d)] >= 0; }

  const std::vector<int>* widths_ = nullptr;
  const std::vector<int>* drafters_ = nullptr;
  const std::vector<double>* cost_ = nullptr;
  const std::vector<double>* per_token_ = nullptr;
  int cost_at_ = 0;
  const RoundCostModel* costs_ = nullptr;
  const std::vector<int>* trunks_ = nullptr;
  std::vector<double> bonus_sum_, bonus_n_;  // per round (trees): tokens beyond the trunk, counted
  // per round (trees), by the trunk's accepted drafts k: the tokens beyond them, counted (kBonusKeep a round of that k)
  std::vector<std::array<double, kPositions + 1>> bonus_at_sum_, bonus_at_n_;
  std::vector<double> tok_guess_;              // per round (trees without a trunk): the guess of its tokens a round
  SharedTokens* shared_ = nullptr;             // the engine's sessions' rounds of them, and the probe credit
  std::unique_ptr<SharedTokens> own_;          // ... when the caller gives none
  std::vector<double> level_sum_, level_n_;    // per drafter: tokens a round / the engine's mean, counted
  std::vector<int> tree_idle_;               // per round (trees): its drafter's rounds since it last drafted
  std::vector<PerPosition> accept_;        // per drafter: P(draft i accepted | drafts before it were), moving average
  std::vector<PerPosition> hits_, rounds_;  // per drafter: the drafter estimate's counts
  std::vector<int> idle_;                  // per drafter: rounds chosen since it last drafted
  std::vector<int> untried_;               // drafters that have not drafted yet, in order
  // Per drafter whose state follows its own rounds only: its warm-up round (-1: none, the drafter does not restart),
  // its rounds since the state started over (0: behind); a probe's rounds chosen since its warm-up (-1: no probe), and
  // those read, and their tokens.
  std::vector<int> warm_up_, warm_, probe_chosen_, probe_seen_;
  std::vector<double> probe_tokens_;
  bool batched_ = false;       // batched()
  int context_ = 0;            // the context of the last round chosen
  double other_ms_ = 0;        // ms a token of the best round of the others then (what a restarting drafter's beat)
};

}  // namespace flyx
