// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/controller.h"

#include <algorithm>
#include <climits>
#include <cmath>

namespace flyx {

namespace {
// P(draft i accepted) before any round: optimistic, so a new session drafts first; never below 0 (positions past 40,
// a tree round's).
double prior(int i) { return std::max(0.0, 0.8 - 0.02 * i); }

// Tokens a chain of `width` drafts gives at the prior acceptance.
double prior_tokens(int width) {
  double e = 1, p = 1;
  for (int i = 0; i < width; ++i) e += p *= prior(i);
  return e;
}
}  // namespace

void SpecController::reset(const std::vector<int>* widths, const std::vector<int>* drafters,
                           const std::vector<double>* cost_ms, const std::vector<int>* trunks,
                           const std::vector<double>* cost_per_token, int cost_at, SharedTokens* shared,
                           const RoundCostModel* costs) {
  widths_ = widths;
  costs_ = costs;
  drafters_ = drafters;
  cost_ = cost_ms;
  per_token_ = cost_per_token;
  cost_at_ = cost_at;
  trunks_ = trunks;
  const size_t rounds = widths->size();
  bonus_sum_.assign(rounds, 0.0);
  bonus_n_.assign(rounds, 0.0);
  tok_guess_.assign(rounds, 1.0);
  tree_idle_.assign(rounds, 0);
  for (size_t i = 0; i < rounds; ++i) tok_guess_[i] = prior_tokens((*widths)[i]);
  if (!shared) {
    own_ = std::make_unique<SharedTokens>();
    own_->resize(rounds);
  }
  shared_ = shared ? shared : own_.get();
  const int n = drafters->empty() ? 0 : *std::max_element(drafters->begin(), drafters->end()) + 1;
  accept_.assign(size_t(n), {});
  for (auto& a : accept_)
    for (int i = 0; i < kPositions; ++i) a[size_t(i)] = prior(i);
  hits_.assign(size_t(n), {});
  rounds_.assign(size_t(n), {});
  idle_.assign(size_t(n), 0);
  level_sum_.assign(size_t(n), 0.0);
  level_n_.assign(size_t(n), 0.0);
  warm_up_.assign(size_t(n), -1);
  warm_.assign(size_t(n), 0);
  probe_chosen_.assign(size_t(n), -1);
  probe_seen_.assign(size_t(n), 0);
  probe_tokens_.assign(size_t(n), 0.0);
  for (size_t i = 0; i < rounds; ++i)
    if ((*widths)[i] == 0 && (*drafters)[i] >= 0) warm_up_[size_t((*drafters)[i])] = static_cast<int>(i);
  untried_.clear();
  for (int d = 0; d < n; ++d)
    if (!restarts(d)) untried_.push_back(d);  // a restarting drafter is entered where it pays, never just to try it
  context_ = 0;
  other_ms_ = 0;
}

void SpecController::restart() {
  std::fill(warm_.begin(), warm_.end(), 0);
  std::fill(probe_chosen_.begin(), probe_chosen_.end(), -1);
}

void SpecController::reply() {
  restart();
  std::fill(level_sum_.begin(), level_sum_.end(), 0.0);
  std::fill(level_n_.begin(), level_n_.end(), 0.0);
}

bool SpecController::keeps_state(size_t i) const { return restarts((*drafters_)[i]); }

double SpecController::cost(size_t i, int context) const {
  return std::max(kLeastMs, costs_ ? costs_->cost(i, context) : load_cost(i, context));
}

double SpecController::load_cost(size_t i, int context) const {
  const double more = per_token_ ? (*per_token_)[i] * std::max(0, context - cost_at_) : 0.0;
  return (*cost_)[i] + more;
}

double SpecController::expected_tokens(int drafter, int width, bool drafter_estimate) const {
  double e = 1, p = 1;
  for (int i = 0; drafter >= 0 && i < width && i < kPositions; ++i) {
    const size_t d = size_t(drafter), k = size_t(i);
    if (drafter_estimate) {
      const double n = kPrior + rounds_[d][k], m = (kPrior * prior(i) + hits_[d][k]) / n;
      p *= std::min(1.0, m + kUcb * std::sqrt(m * (1 - m) / n));
    } else {
      p *= accept_[d][k];
    }
    e += p;
  }
  return e;
}

double SpecController::round_tokens(size_t i, bool drafter_estimate) const {
  const int d = (*drafters_)[i];
  if (bare(i)) return level(d) * mean_tokens(i);
  if (!tree(i)) return expected_tokens(d, (*widths_)[i], drafter_estimate);
  const double bonus = (kBonusPrior + bonus_sum_[i]) / (1 + bonus_n_[i]);
  return expected_tokens(d, (*trunks_)[i], drafter_estimate) + bonus;
}

int SpecController::best_round(int drafter, bool with_plain, bool drafter_estimate, int context, int max_width,
                               bool allow_trees) const {
  int best = -1;
  double best_rate = 0;
  for (size_t i = 0; i < widths_->size(); ++i) {
    if ((*widths_)[i] > max_width || (!allow_trees && is_tree(i))) continue;
    const int d = (*drafters_)[i];
    if (d >= 0 && (*widths_)[i] == 0) continue;  // a warm-up: chosen only to enter its drafter
    if (drafter == kAnyDrafter ? restarts(d) : d != drafter && !(with_plain && d < 0)) continue;
    const double rate = round_tokens(i, drafter_estimate) / cost(i, context);
    if (best < 0 || rate > best_rate) {
      best = static_cast<int>(i);
      best_rate = rate;
    }
  }
  return best;
}

double SpecController::worst_probe(int d, size_t i, double ms_per_token, int context) const {
  return cost(size_t(warm_up_[size_t(d)]), context) + cost(i, context) - 2 * ms_per_token;
}

int SpecController::best_width(int d, int context) const {
  int best = -1;
  double best_rate = 0;
  for (size_t i = 0; i < widths_->size(); ++i) {
    if (!bare(i) || (*drafters_)[i] != d) continue;
    const double rate = mean_tokens(i) / cost(i, context);
    if (best < 0 || rate > best_rate) best = static_cast<int>(i), best_rate = rate;
  }
  return best;
}

double SpecController::credit_cap(int d, int context) const {
  const int i = best_width(d, context);
  return i >= 0 && other_ms_ > 0 ? kCreditCap * worst_probe(d, size_t(i), other_ms_, context) : 0.0;
}

int SpecController::restarting_round(int d, double other, int context, double* rate, bool* probe) const {
  *probe = false;
  if (batched_) return -1;  // its rounds run alone: in rounds of several sessions it does not draft
  const int i = best_width(d, context);
  if (i < 0) return -1;
  // At its best width by the engine's means, its tokens there by the session's level.
  const double t = round_tokens(size_t(i)), c = cost(size_t(i), context);
  if (warm_[size_t(d)] > 0) {  // its state follows: it stays while its rounds pay (or a probe holds it), widths in turn
    *rate = t / c;
    // A probe holds it kProbeRounds rounds, and up to kProbeMax while its own rounds pay.
    const int k = probe_chosen_[size_t(d)], seen = probe_seen_[size_t(d)];
    const bool held = k >= 0 && (k < kProbeRounds ||
                                 (k < kProbeMax && seen > 0 && probe_tokens_[size_t(d)] >= seen * other * c));
    if (*rate < other && !held) return -1;
    int turn = -1;
    for (size_t j = 0; j < widths_->size(); ++j)
      if (bare(j) && (*drafters_)[j] == d && tree_idle_[j] >= kWidthTurn &&
          (turn < 0 || tree_idle_[j] > tree_idle_[size_t(turn)]))
        turn = static_cast<int>(j);
    return turn >= 0 ? turn : i;
  }
  // Behind: its warm-up first, where kPayBack of its rounds pay for it, or as a probe the engine's credit covers.
  const double warm = cost(size_t(warm_up_[size_t(d)]), context);
  *rate = (1 + kPayBack * t) / (warm + kPayBack * c);
  if (*rate > other) return warm_up_[size_t(d)];
  *probe = idle_[size_t(d)] >= kProbe && other > 0 &&
           shared_->credit() >= worst_probe(d, size_t(i), 1 / other, context);
  return *probe ? warm_up_[size_t(d)] : -1;
}

int SpecController::preview(int context, int max_width, bool allow_trees, bool* forced, bool* untried,
                            double* other_ms) const {
  if (forced) *forced = false;
  if (untried) *untried = false;
  int best = -1;
  if (!untried_.empty()) {  // every drafter drafts once first: its best round
    best = best_round(untried_.front(), false, false, context, max_width, allow_trees);
    if (best >= 0 && untried) *untried = true;
  }
  if (best < 0) {
    // Plain decoding or a drafted round, by the moving averages (with one drafter, the whole choice).
    best = best_round(kAnyDrafter, true, false, context, max_width, allow_trees);
    if (best < 0) best = best_round(kAnyDrafter, true, false, context);  // a pinned width past max_width
    if (best >= 0 && (*drafters_)[size_t(best)] >= 0) {
      // Drafted: by the drafter whose best round the drafter estimate rates highest, at its best width.
      int pick = -1;
      double pick_rate = 0;
      for (int d = 0; d < static_cast<int>(idle_.size()); ++d) {
        if (restarts(d)) continue;
        const int i = best_round(d, false, true, context, max_width, allow_trees);
        if (i < 0) continue;
        const double rate = round_tokens(size_t(i), true) / cost(size_t(i), context);
        if (pick < 0 || rate > pick_rate) {
          pick = d;
          pick_rate = rate;
        }
      }
      if (pick >= 0) best = best_round(pick, false, false, context, max_width, allow_trees);
    } else if (best >= 0) {
      // Plain decoding, and this would be the kProbe-th round in a row without a drafter: it drafts it (the one
      // idle longest).  (A drafter idle while another drafts comes back through its forgotten counts.)
      int probe = -1;
      for (int d = 0; d < static_cast<int>(idle_.size()); ++d)
        if (!restarts(d) && idle_[size_t(d)] >= kProbe - 1 &&
            (probe < 0 || idle_[size_t(d)] > idle_[size_t(probe)]))
          probe = d;
      if (probe >= 0) {
        const int p = best_round(probe, false, false, context, max_width, allow_trees);
        if (p >= 0) {
          best = p;
          if (forced) *forced = true;
        }
      }
    }
  }
  // A tree its drafter has not drafted for kTreeProbe of its rounds drafts this one (the one idle longest).
  const int drafter = best >= 0 ? (*drafters_)[size_t(best)] : -1;
  if (allow_trees && drafter >= 0 && !tree(size_t(best))) {
    int probe = -1;
    for (size_t i = 0; i < widths_->size(); ++i)
      if (tree(i) && (*drafters_)[i] == drafter && (*widths_)[i] <= max_width && tree_idle_[i] >= kTreeProbe &&
          (probe < 0 || tree_idle_[i] > tree_idle_[size_t(probe)]))
        probe = static_cast<int>(i);
    if (probe >= 0) {
      best = probe;
      if (forced) *forced = true;
    }
  }
  // A drafter whose state follows its own rounds only, against that choice (none: against nothing).
  const double other = best >= 0 ? round_tokens(size_t(best), false) / cost(size_t(best), context) : 0.0;
  if (other_ms) *other_ms = other > 0 ? 1 / other : 0.0;
  double top = other;
  for (int d = 0; d < static_cast<int>(warm_up_.size()); ++d) {
    if (!restarts(d)) continue;
    double rate = 0;
    bool probe = false;
    const int i = restarting_round(d, other, context, &rate, &probe);
    if (i >= 0 && (best < 0 || !keeps_state(size_t(best)) || rate > top)) {
      best = i;
      top = rate;
    }
  }
  return best < 0 ? 0 : best;
}

int SpecController::choose(int context) {
  const int best = preview(context);
  commit(size_t(best), context);
  return best;
}

void SpecController::commit(size_t round, int context) {
  preview(context, kAnyWidth, true, nullptr, nullptr, &other_ms_);  // what a restarting drafter's rounds beat now
  const int best = static_cast<int>(round);
  const int drafter = (*drafters_)[round];
  if (drafter >= 0) {
    auto it = std::find(untried_.begin(), untried_.end(), drafter);
    if (it != untried_.end()) untried_.erase(it);
  }
  context_ = context;
  for (size_t i = 0; i < widths_->size(); ++i)
    if ((tree(i) || bare(i)) && (*drafters_)[i] == drafter) tree_idle_[i] = i == round ? 0 : tree_idle_[i] + 1;
  for (int d = 0; d < static_cast<int>(idle_.size()); ++d) {
    if (restarts(d)) {
      // The engine's credit: a share of every round's time, within kCreditCap probes at their worst.
      shared_->add_credit(kProbeShare * cost(size_t(best), context), credit_cap(d, context));
      if (best == warm_up_[size_t(d)]) {  // a probe's warm-up holds the drafter to its next rounds
        double rate = 0;
        bool probe = false;
        restarting_round(d, other_ms_ > 0 ? 1 / other_ms_ : 0.0, context, &rate, &probe);
        probe_chosen_[size_t(d)] = probe ? 0 : -1;
        probe_seen_[size_t(d)] = 0;
        probe_tokens_[size_t(d)] = 0;
      } else if (d == drafter) {
        if (probe_chosen_[size_t(d)] >= 0) ++probe_chosen_[size_t(d)];
      } else {
        probe_chosen_[size_t(d)] = -1;
      }
      warm_[size_t(d)] = d == drafter ? warm_[size_t(d)] + 1 : 0;  // another round leaves its state behind
      if (d != drafter) {  // its level while it is away
        level_sum_[size_t(d)] *= 1 - kLevelForget;
        level_n_[size_t(d)] *= 1 - kLevelForget;
      }
    }
    if (d == drafter) {
      idle_[size_t(d)] = 0;
      continue;
    }
    ++idle_[size_t(d)];
    for (int i = 0; i < kPositions; ++i) {  // an idle drafter's estimate goes back toward the prior
      hits_[size_t(d)][size_t(i)] *= 1 - kForget;
      rounds_[size_t(d)][size_t(i)] *= 1 - kForget;
    }
  }
}

void SpecController::observe(size_t round, int accepted, int trunk_accepted) {
  const int drafter = (*drafters_)[round];
  if (drafter < 0) return;
  const size_t d = size_t(drafter);
  // What a restarting drafter's round gave over the others' rate for as long: the engine's credit.
  if (restarts(drafter))
    shared_->add_credit(other_ms_ * (accepted + 1) - cost(round, context_), credit_cap(drafter, context_));
  if ((*widths_)[round] == 0) return;  // a warm-up drafts nothing
  if (bare(round)) {  // the session's level against the engine's mean, and the engine's mean
    if (probe_chosen_[d] >= 0) ++probe_seen_[d], probe_tokens_[d] += accepted + 1;
    level_sum_[d] = kKeepCounts * level_sum_[d] + (accepted + 1) / mean_tokens(round);
    level_n_[d] = kKeepCounts * level_n_[d] + 1;
    shared_->add(round, accepted + 1);
    return;
  }
  constexpr double kKeep = 0.9;  // moving average over roughly the last ten rounds
  int width = (*widths_)[round];
  if (tree(round)) {  // the trunk teaches the chain; the rest is the branches'
    const double more = accepted - trunk_accepted;
    bonus_sum_[round] = kBonusKeep * bonus_sum_[round] + more;
    bonus_n_[round] = kBonusKeep * bonus_n_[round] + 1;
    width = (*trunks_)[round];
    accepted = trunk_accepted;
  }
  for (int i = 0; i < width && i < kPositions; ++i) {
    if (i > accepted) break;  // drafts after the first rejection say nothing
    const double hit = i < accepted ? 1.0 : 0.0;
    const size_t k = size_t(i);
    accept_[d][k] = kKeep * accept_[d][k] + (1 - kKeep) * hit;
    hits_[d][k] = kKeepCounts * hits_[d][k] + hit;
    rounds_[d][k] = kKeepCounts * rounds_[d][k] + 1;
  }
}

}  // namespace flyx
