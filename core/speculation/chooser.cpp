// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/chooser.h"

#include <algorithm>
#include <cmath>

namespace flyx {

void RoundChooser::configure(const std::vector<int>* widths, const std::vector<int>* drafters,
                             const std::vector<double>* cost_ms, const std::vector<int>* lookup_widths,
                             const std::vector<int>* trunks, const std::vector<double>* lookup_cost_ms,
                             const std::vector<std::vector<int>>* adaptive_checkpoints,
                             const std::vector<double>* cost_per_token, int cost_at, SharedTokens* shared,
                             const RoundCostModel* costs, const RoundCostModel* lookup_costs) {
  widths_ = widths;
  drafters_ = drafters;
  lookup_widths_ = lookup_widths;
  lookup_cost_ = lookup_cost_ms;
  lookup_costs_ = lookup_costs;
  lookup_accept_ = 0.75;
  spec_.reset(widths, drafters, cost_ms, trunks, cost_per_token, cost_at, shared, costs);
  stoppers_.clear();
  if (adaptive_checkpoints && !adaptive_checkpoints->empty()) {
    double plain_ms = 1.0;
    for (size_t i = 0; i < widths->size(); ++i)
      if ((*drafters)[i] < 0) {
        plain_ms = (*cost_ms)[i];
        break;
      }
    stoppers_.resize(adaptive_checkpoints->size());
    for (size_t d = 0; d < adaptive_checkpoints->size(); ++d) {
      const auto& cps = (*adaptive_checkpoints)[d];
      if (cps.size() < 2) continue;
      std::vector<int> ws;
      std::vector<double> cs;
      for (int cp : cps) {
        for (size_t i = 0; i < widths->size(); ++i) {
          if ((*drafters)[i] == int(d) && (*widths)[i] == cp && !spec_.is_tree(i)) {
            ws.push_back(cp);
            cs.push_back((*cost_ms)[i]);
            break;
          }
        }
      }
      if (ws.size() == cps.size()) stoppers_[d].configure(ws, cs, plain_ms);
    }
  }
}

void RoundChooser::begin(bool lookup, bool wide, bool adaptive) {
  lookup_on_ = lookup && lookup_widths_ && !lookup_widths_->empty();
  lookup_mode_ = false;
  rests_until_ = 0;
  wide_ok_ = wide;
  adaptive_on_ = adaptive;
  for (auto& st : stoppers_) st.begin();
  run_drafter_ = -1;
  run_width_ = 0;
  run_skips_ = 0;
  wide_run_tok_ = wide_run_ms_ = alt_run_tok_ = alt_run_ms_ = 0;  // each reply judges its own text
  full_runs_ = wide_rounds_ = 0;
  wide_need_ = kWideNeed;
  wide_tokens_ = wide_ms_ = 0;
  look_ = kLookPriorDrafts;
  miss_ = kLookPriorMisses;
  spec_.reply();  // after the prompt's prefill no drafter's state follows a round
}

const flyx_stop_policy* RoundChooser::stop_policy(int drafter) const {
  if (drafter < 0 || size_t(drafter) >= stoppers_.size()) return nullptr;
  return stoppers_[size_t(drafter)].ready() ? &stoppers_[size_t(drafter)].policy() : nullptr;
}

double RoundChooser::lookup_rate(size_t k) const {
  const double p = std::clamp(1.0 - miss_ / look_, 0.0, 1.0 - 1e-9);
  const int w = (*lookup_widths_)[k];
  return (1.0 + p * (1.0 - std::pow(p, w)) / (1.0 - p)) / lookup_cost(k);
}

double RoundChooser::lookup_cost(size_t k) const {
  if (lookup_costs_ && lookup_cost_ && k < lookup_cost_->size())
    return std::max(SpecController::kLeastMs, lookup_costs_->cost(k, context_));
  if (lookup_cost_ && k < lookup_cost_->size()) return std::max(SpecController::kLeastMs, (*lookup_cost_)[k]);
  return 1.0 + 0.02 * (*lookup_widths_)[k];  // before any timing: a row costs little beside the weights
}

bool RoundChooser::looks_up(size_t context) const {
  return lookup_on_ && context >= rests_until_ && !spec_.keeps_state(size_t(spec_.peek(int(context))));
}

void RoundChooser::candidates(const std::vector<int32_t>& tokens, int in_flight, std::vector<RoundCandidate>* out) {
  out->clear();
  context_ = static_cast<int>(tokens.size());
  int max_lookup = -1;  // lookup rounds: up to the widest the proposal fills
  if (in_flight == 0 && looks_up(tokens.size())) {
    lookup_.sync(tokens);
    if (lookup_.propose(tokens, lookup_widths_->back(), &proposal_) > 0)
      for (size_t k = 0; k < lookup_widths_->size(); ++k)
        if ((*lookup_widths_)[k] <= static_cast<int>(proposal_.size())) max_lookup = static_cast<int>(k);
  }
  if (max_lookup >= 0) {
    // the widest a round of FLYX_MAX_ROWS rows takes, or a wider one once a copy has been accepted whole (solo)
    int narrow = -1;
    for (int k = 0; k <= max_lookup; ++k)
      if ((*lookup_widths_)[size_t(k)] <= FLYX_MAX_ROWS - 1) narrow = k;
    int pref = narrow < 0 ? max_lookup : narrow;
    if (wide_ready())  // the narrow round or a wide one, whichever is expected to give the most tokens a millisecond
      for (int k = std::max(narrow, 0); k <= max_lookup; ++k)
        if ((*lookup_widths_)[size_t(k)] > FLYX_MAX_ROWS - 1 && lookup_rate(size_t(k)) > lookup_rate(size_t(pref))) pref = k;
    for (int k = 0; k <= max_lookup; ++k) {
      const int w = (*lookup_widths_)[size_t(k)];
      const bool wide = w > FLYX_MAX_ROWS - 1;
      if (wide && k != pref) continue;
      out->push_back({{true, size_t(k), w, -1}, 1.0 + lookup_accept_ * w, lookup_cost(size_t(k)), !wide, k == pref});
    }
    return;
  }
  // A round that runs alone: a tree's (launch_batch), or one of more than FLYX_MAX_ROWS rows (a session's share of a
  // batch holds at most that).  (A drafter's warm-up is no tree: it may go out with other sessions' rounds, its canvas
  // pass a sequence at a time.)
  auto alone = [&](size_t i) {
    return ((*widths_)[i] > 0 && spec_.is_tree(i)) || (*widths_)[i] + 1 > FLYX_MAX_ROWS;
  };
  if (!cycle_.empty()) {
    const size_t i = cycle_[cycled_ % cycle_.size()];
    out->push_back({{false, i, (*widths_)[i], (*drafters_)[i]}, spec_.round_tokens(i), spec_.cost(i, context_),
                    !alone(i), true});
    return;
  }
  bool forced = false, untried = false;
  size_t pref = static_cast<size_t>(spec_.preview(context_, FLYX_MAX_ROWS - 1, true, &forced, &untried));
  bool widened = false;
  if (!forced && run_drafter_ >= 0 && (*drafters_)[pref] == run_drafter_ &&
      (runs_pay() || run_skips_ >= kRunProbe)) {  // a run: one width wider than its round (or hold max widened width)
    size_t wider = pref;
    for (size_t i = 0; i < widths_->size(); ++i)
      if ((*drafters_)[i] == run_drafter_ && !spec_.is_tree(i) && (*widths_)[i] > run_width_ &&
          (wider == pref || (*widths_)[i] < (*widths_)[wider]) && (*widths_)[i] > (*widths_)[pref])
        wider = i;
    if (wider == pref && run_width_ > (*widths_)[pref]) {
      for (size_t i = 0; i < widths_->size(); ++i)
        if ((*drafters_)[i] == run_drafter_ && !spec_.is_tree(i) && (*widths_)[i] == run_width_) {
          wider = i;
          break;
        }
    }
    widened = wider != pref;
    pref = wider;
  }
  const int pref_drafter = (*drafters_)[pref];
  // An intra-round stopping plan (adaptive) for a model drafter's chains; never DV's, whose rounds are trees and its
  // warm-up, and whose canvas follows its own rounds (keeps_state).  A run widened past what the drafter's adaptive
  // round reaches (e.g., a stretched DFlash2 width past its trained block) keeps its fixed width while the run holds.
  const bool in_wide_run =
      widened || (!forced && run_drafter_ >= 0 && run_drafter_ == pref_drafter &&
                  size_t(pref_drafter) < stoppers_.size() && run_width_ > stoppers_[size_t(pref_drafter)].max_width());
  const bool past_adaptive = in_wide_run && pref_drafter >= 0 && size_t(pref_drafter) < stoppers_.size() &&
                             (*widths_)[pref] > stoppers_[size_t(pref_drafter)].max_width();
  const bool use_adaptive = adaptive_on_ && pref_drafter >= 0 && !spec_.is_tree(pref) && !spec_.keeps_state(pref) &&
                            stop_policy(pref_drafter) != nullptr && !past_adaptive;
  if (use_adaptive) {
    // Scored as the widest plan it reaches where the controller prefers a wider one: it verifies no more drafts.
    const int reach = stoppers_[size_t(pref_drafter)].max_width();
    size_t scored = pref;
    if ((*widths_)[pref] > reach)
      for (size_t i = 0; i < widths_->size(); ++i)
        if ((*drafters_)[i] == pref_drafter && (*widths_)[i] == reach && !spec_.is_tree(i)) scored = i;
    out->push_back({{false, pref, reach, pref_drafter, true}, spec_.round_tokens(scored), spec_.cost(scored, context_),
                    false, true});
    // Still offer the fixed preferred width as batchable so concurrent scheduling can batch cleanly.
    out->push_back({{false, pref, (*widths_)[pref], pref_drafter, false}, spec_.round_tokens(pref),
                    spec_.cost(pref, context_), true, false});
  } else {
    out->push_back({{false, pref, (*widths_)[pref], pref_drafter, false}, spec_.round_tokens(pref),
                    spec_.cost(pref, context_), !alone(pref), true});
  }
  // Probes, plain-best rounds and a drafter's that keeps a state keep their exact solo choice.
  if (forced || pref_drafter < 0 || spec_.keeps_state(pref)) return;
  for (size_t i = 0; i < widths_->size(); ++i) {
    if (i == pref || alone(i) || spec_.keeps_state(i)) continue;  // DV's rounds never in place of another's
    const int d = (*drafters_)[i];
    // For an untried drafter, keep only its own widths so it still drafts once on round 1; otherwise also allow plain.
    if (d != pref_drafter && (untried || d >= 0)) continue;
    out->push_back({{false, i, (*widths_)[i], d, false}, spec_.round_tokens(i), spec_.cost(i, context_), true, false});
  }
}

void RoundChooser::commit(const RoundChoice& choice, int in_flight) {
  if (choice.lookup) {
    lookup_mode_ = true;
    spec_.restart();  // a round of none of the controller's plans: a drafter's state that follows its own is behind
    return;
  }
  if (in_flight == 0) lookup_mode_ = false;
  if (!cycle_.empty()) {
    ++cycled_;
    return;
  }
  spec_.commit(choice.index, context_);
}

RoundChoice RoundChooser::next(const std::vector<int32_t>& tokens, int in_flight) {
  std::vector<RoundCandidate> cands;
  candidates(tokens, in_flight, &cands);
  RoundChoice pick = cands.front().choice;
  for (const auto& c : cands)
    if (c.preferred) {
      pick = c.choice;
      break;
    }
  commit(pick, in_flight);
  return pick;
}

RoundChoice RoundChooser::next_drafted(int context) {
  const size_t i = cycle_.empty() ? static_cast<size_t>(spec_.choose(context)) : cycle_[cycled_++ % cycle_.size()];
  return {false, i, (*widths_)[i], (*drafters_)[i]};
}

void RoundChooser::result(const RoundChoice& round, bool emitted, int accepted, const std::vector<int32_t>& tokens,
                          int in_flight, bool learn, int trunk_accepted, int actual_width, const float* draft_prob,
                          const float* draft_margin) {
  if (emitted) {
    if (round.lookup) {
      if (round.width > 0) {  // the drafts put to the test: the accepted ones and the first rejected
        look_ = kLookDecay * look_ + std::min(accepted + 1, round.width);
        miss_ = kLookDecay * miss_ + (accepted < round.width ? 1.0 : 0.0);
      }
      if (round.width <= FLYX_MAX_ROWS - 1) {
        full_runs_ = round.width == FLYX_MAX_ROWS - 1 && accepted >= round.width ? full_runs_ + 1
                     : round.width == FLYX_MAX_ROWS - 1                         ? 0
                                                                                 : full_runs_;
      } else {  // a wide round: judged with the ones before it against a full round of FLYX_MAX_ROWS - 1
        wide_tokens_ += accepted + 1;
        wide_ms_ += lookup_cost(round.index);
        if (++wide_rounds_ >= kWideJudge || accepted < round.width) {
          size_t narrow = 0;
          for (size_t k = 0; k < lookup_widths_->size(); ++k)
            if ((*lookup_widths_)[k] == FLYX_MAX_ROWS - 1) narrow = k;
          if (wide_tokens_ / wide_ms_ < FLYX_MAX_ROWS / lookup_cost(narrow)) {
            wide_need_ = std::min(kWideNeedMost, 2 * wide_need_);
            full_runs_ = 0;
          } else {
            wide_need_ = kWideNeed;
          }
          wide_rounds_ = 0;
          wide_tokens_ = wide_ms_ = 0;
        }
      }
      if (round.width > 0) {
        const double frac = static_cast<double>(std::min(accepted, round.width)) / round.width;
        lookup_accept_ = 0.9 * lookup_accept_ + 0.1 * frac;
      }
      const int worth = std::min(kLookupWorth, round.width);
      if (accepted < worth) {  // a rest: the drafter's rounds, two in flight
        rests_until_ = tokens.size() + kLookupRest;
        lookup_mode_ = false;
      }
    } else if (learn) {
      size_t obs_index = round.index;
      const int eff_w = actual_width > 0 ? actual_width : round.width;
      if (round.adaptive && actual_width > 0) {
        for (size_t i = 0; i < widths_->size(); ++i)
          if ((*drafters_)[i] == round.drafter && (*widths_)[i] == actual_width && !spec_.is_tree(i)) {
            obs_index = i;
            break;
          }
      }
      spec_.observe(obs_index, accepted, trunk_accepted);  // the drafter's acceptance only
      // A fixed round's tokens are credited at its own plan's cost, at the context it ran at (tokens holds its own
      // already): its width may lie past the stopper's checkpoints.  The load's cost, as the checkpoints' are.
      if (!spec_.is_tree(round.index) && round.drafter >= 0 && size_t(round.drafter) < stoppers_.size() &&
          stoppers_[size_t(round.drafter)].ready()) {
        const int ran_at = std::max(0, static_cast<int>(tokens.size()) - (accepted + 1));
        stoppers_[size_t(round.drafter)].observe(eff_w, accepted, draft_prob, draft_margin,
                                                 round.adaptive ? 0.0 : spec_.load_cost(round.index, ran_at));
      }
    }
    // The runs are a model drafter's chains (DFlash2's block): no tree's, and none of DV, whose canvas follows its
    // own rounds.  An adaptive round whose chosen checkpoint reaches kRunWidth can also trigger run widening.
    const bool chains = !round.lookup && !spec_.is_tree(round.index) && round.drafter >= 0 &&
                        !spec_.keeps_state(round.index);
    const int eff_w = round.adaptive && actual_width > 0 ? actual_width : round.width;
    const int at = static_cast<int>(tokens.size());
    if (chains && !round.adaptive && round.width > kRunWidth) {  // a wider round against the narrow one
      double alt_cost = 0;
      for (size_t i = 0; i < widths_->size(); ++i)
        if ((*drafters_)[i] == round.drafter && !spec_.is_tree(i) && (*widths_)[i] <= kRunWidth)
          alt_cost = spec_.cost(i, at);  // the widest of the drafter's up to kRunWidth (widths ascend)
      if (alt_cost > 0) {
        wide_run_tok_ = kRunDecay * wide_run_tok_ + accepted + 1;
        wide_run_ms_ = kRunDecay * wide_run_ms_ + spec_.cost(round.index, at);
        alt_run_tok_ = kRunDecay * alt_run_tok_ + std::min(accepted, kRunWidth) + 1;
        alt_run_ms_ = kRunDecay * alt_run_ms_ + alt_cost;
      }
      run_skips_ = 0;
    }
    const bool run = chains && eff_w >= kRunWidth && accepted >= eff_w - 1;
    if (run && eff_w == kRunWidth && !runs_pay()) ++run_skips_;  // a run not widened (it did not pay)
    run_drafter_ = run ? round.drafter : -1;
    run_width_ = run ? eff_w : 0;
  }
  // Text that also appears earlier: launch nothing more until the round
  // ahead is read, then draft by lookup from where it ends.
  if (learn && in_flight > 0 && !lookup_mode_ && looks_up(tokens.size())) {
    lookup_.sync(tokens);
    lookup_mode_ = lookup_.propose(tokens, 1, &proposal_) > 0;
  }
}

}  // namespace flyx
