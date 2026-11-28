// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/batch_schedule.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "flyx/backend_abi.h"

namespace flyx {

namespace {

constexpr double kMinMs = 1e-6;

const RoundCandidate& preferred_of(const std::vector<RoundCandidate>& cands) {
  for (const auto& c : cands)
    if (c.preferred) return c;
  return cands.front();
}

}  // namespace

void BatchCostModel::build(const std::vector<int>& widths, const std::vector<int>& drafters,
                           const std::vector<double>& round_ms, const std::vector<int>& lookup_widths,
                           const std::vector<double>& lookup_ms, double session_ms) {
  knots_.clear();
  session_ms_ = std::max(0.0, session_ms);
  double plain_ms = 0.0;
  for (size_t i = 0; i < widths.size() && i < drafters.size() && i < round_ms.size(); ++i)
    if (drafters[i] < 0 && widths[i] == 0 && round_ms[i] > 0.0) {
      plain_ms = round_ms[i];
      knots_.push_back({1, plain_ms});
      break;
    }
  if (lookup_ms.size() == lookup_widths.size()) {
    for (size_t k = 0; k < lookup_widths.size(); ++k)
      if (lookup_widths[k] >= 0 && lookup_ms[k] > 0.0) knots_.push_back({lookup_widths[k] + 1, lookup_ms[k]});
  }
  std::sort(knots_.begin(), knots_.end(), [](const Knot& a, const Knot& b) {
    return a.rows != b.rows ? a.rows < b.rows : a.ms < b.ms;
  });
  // Deduplicate equal row counts (keeping the smallest measured cost).
  std::vector<Knot> uniq;
  for (const Knot& k : knots_)
    if (uniq.empty() || uniq.back().rows != k.rows) uniq.push_back(k);
  knots_ = std::move(uniq);

  // Fallback when lookup timings are unavailable (e.g. --no-lookup or pinned width).
  if (knots_.empty()) {
    double base = 1.0;
    for (double m : round_ms)
      if (m > 0.0) base = (base == 1.0) ? m : std::min(base, m);
    knots_.push_back({1, base});
    knots_.push_back({FLYX_MAX_ROWS, base * (1.0 + 0.05 * (FLYX_MAX_ROWS - 1))});
  } else if (knots_.size() == 1) {
    const double base = knots_[0].ms;
    double slope = 0.0;
    bool have_slope = false;
    for (size_t i = 0; i < widths.size() && i < round_ms.size(); ++i)
      if (widths[i] > 0 && round_ms[i] > base) {
        // Half the drafted per-width delta is a conservative target-row slope estimate
        // (the rest belongs to drafting).
        const double s = 0.5 * (round_ms[i] - base) / widths[i];
        slope = have_slope ? std::min(slope, s) : s;
        have_slope = true;
      }
    if (!have_slope) slope = 0.05 * base;
    slope = std::min(slope, 0.15 * base);
    knots_.push_back({FLYX_MAX_ROWS, base + slope * (FLYX_MAX_ROWS - knots_[0].rows)});
  }

  // Enforce non-decreasing target time with row count (guards against calibration noise on tiny models).
  for (size_t i = 1; i < knots_.size(); ++i)
    if (knots_[i].ms < knots_[i - 1].ms) knots_[i].ms = knots_[i - 1].ms;
}

double BatchCostModel::target_ms(int rows) const {
  if (knots_.empty()) return 1.0;
  if (rows <= knots_.front().rows || knots_.size() == 1) return std::max(kMinMs, knots_.front().ms);
  auto tiles = [](int r) { return (r + FLYX_MAX_ROWS - 1) / FLYX_MAX_ROWS; };
  for (size_t i = 1; i < knots_.size(); ++i) {
    if (rows <= knots_[i].rows) {
      const Knot& a = knots_[i - 1];
      const Knot& b = knots_[i];
      const bool by_tiles = rows > FLYX_MAX_ROWS && tiles(b.rows) > tiles(a.rows);
      const double t = by_tiles ? static_cast<double>(tiles(rows) - tiles(a.rows)) / (tiles(b.rows) - tiles(a.rows))
                                : static_cast<double>(rows - a.rows) / static_cast<double>(b.rows - a.rows);
      return std::max(kMinMs, a.ms + t * (b.ms - a.ms));
    }
  }
  const Knot& a = knots_[knots_.size() - 2];
  const Knot& b = knots_.back();
  const double slope = (b.ms - a.ms) / static_cast<double>(b.rows - a.rows);
  return std::max(kMinMs, b.ms + slope * static_cast<double>(rows - b.rows));
}

double BatchCostModel::batch_ms(int rows, int n_sessions, double draft_ms) const {
  const double extra_sessions = n_sessions > 1 ? session_ms_ * static_cast<double>(n_sessions - 1) : 0.0;
  return std::max(kMinMs, target_ms(rows) + extra_sessions + draft_ms);
}

double BatchCostModel::draft_ms(const RoundCandidate& c) const {
  if (c.choice.lookup || c.choice.drafter < 0) return 0.0;
  return std::max(0.0, c.solo_ms - target_ms(c.choice.width + 1));
}

double BatchCostModel::shared_draft_ms(const std::vector<const RoundCandidate*>& items) const {
  struct DrafterAgg {
    int drafter = -1;
    double max_ms = 0.0;
    double sum_ms = 0.0;
  };
  DrafterAgg agg[FLYX_MAX_ROWS];
  int n_agg = 0;
  double overflow = 0.0;
  for (const RoundCandidate* c : items) {
    if (!c) continue;
    const double g = draft_ms(*c);
    if (g <= 0.0) continue;
    const int d = c->choice.drafter;
    int slot = -1;
    for (int k = 0; k < n_agg; ++k)
      if (agg[k].drafter == d) {
        slot = k;
        break;
      }
    if (slot >= 0) {
      agg[slot].max_ms = std::max(agg[slot].max_ms, g);
      agg[slot].sum_ms += g;
    } else if (n_agg < FLYX_MAX_ROWS) {
      agg[n_agg++] = {d, g, g};
    } else {
      overflow += g;
    }
  }
  double total = overflow;
  for (int k = 0; k < n_agg; ++k) total += agg[k].max_ms + kSharedDraftShare * (agg[k].sum_ms - agg[k].max_ms);
  return total;
}

std::vector<ScheduledBatch> plan_batches(const BatchCostModel& cost,
                                         const std::vector<std::vector<RoundCandidate>>& sessions, int max_rows) {
  max_rows = std::clamp(max_rows, 1, FLYX_MAX_WIDE_ROWS);
  std::vector<ScheduledBatch> result;
  std::vector<size_t> batchable;
  batchable.reserve(sessions.size());

  for (size_t s = 0; s < sessions.size(); ++s) {
    if (sessions[s].empty()) continue;
    bool has_batchable = false;
    for (const auto& c : sessions[s])
      if (c.batchable) {
        has_batchable = true;
        break;
      }
    if (!has_batchable) {
      result.push_back({{{s, preferred_of(sessions[s]).choice}}});
    } else {
      batchable.push_back(s);
    }
  }

  if (batchable.empty()) return result;
  if (batchable.size() == 1) {
    const size_t s = batchable[0];
    result.push_back({{{s, preferred_of(sessions[s]).choice}}});
    return result;
  }

  // ---- Plan A: each session's preferred solo choice, FFD-packed into <= max_rows bins ----
  // Objective: proportional fairness, log U = (1/m) sum log(E_i) - log(T_total), i.e. geometric mean
  // of every session's expected tokens per millisecond.  Equals E/T for solo and identical sessions,
  // and trims low-marginal-return tail drafts without starving lower-acceptance sessions.
  const size_t m = batchable.size();
  const double inv_m = 1.0 / static_cast<double>(m);
  std::vector<ScheduledBatch> plan_a;
  double log_e_a = 0.0, ms_a = 0.0;
  struct FfdItem {
    size_t session;
    const RoundCandidate* cand;
    int rows;
  };
  std::vector<FfdItem> ffd;
  ffd.reserve(m);
  for (size_t s : batchable) {
    const RoundCandidate& p = preferred_of(sessions[s]);
    log_e_a += std::log(std::max(1.0, p.expected_tokens));
    if (!p.batchable) {
      plan_a.push_back({{{s, p.choice}}});
      ms_a += std::max(kMinMs, p.solo_ms);
    } else {
      ffd.push_back({s, &p, p.choice.width + 1});
    }
  }
  std::stable_sort(ffd.begin(), ffd.end(), [](const FfdItem& a, const FfdItem& b) { return a.rows > b.rows; });
  struct Bin {
    std::vector<ScheduledRound> items;
    std::vector<const RoundCandidate*> cands;
    int rows = 0;
  };
  std::vector<Bin> bins;
  for (const FfdItem& item : ffd) {
    size_t placed = bins.size();
    for (size_t b = 0; b < bins.size(); ++b) {
      if (bins[b].rows + item.rows <= max_rows) {
        placed = b;
        break;
      }
    }
    if (placed == bins.size()) bins.push_back({});
    bins[placed].items.push_back({item.session, item.cand->choice});
    bins[placed].cands.push_back(item.cand);
    bins[placed].rows += item.rows;
  }
  for (auto& bin : bins) {
    ms_a += cost.batch_ms(bin.rows, static_cast<int>(bin.items.size()), cost.shared_draft_ms(bin.cands));
    plan_a.push_back({std::move(bin.items)});
  }
  const double score_a = inv_m * log_e_a - std::log(std::max(kMinMs, ms_a));

  // ---- Plan B: 1-batch joint width allocation (exact cap-enumerated tangent-linearized DP) ----
  // Under shared_draft_ms, each drafter d pays (1 - alpha) * max_d + alpha * sum_d.  Enumerating the
  // small set of distinct max_d caps turns the remaining objective into an exact separable additive DP
  //   sum_i ((1/m) log(E_i) - alpha * draft_i / T_ref) - (target_ms(R) + (1 - alpha) * cap) / T_ref
  // over total rows R in 1..max_rows; every reachable R is scored on the exact non-linear log U.
  bool have_b = false;
  double score_b = -std::numeric_limits<double>::infinity();
  std::vector<size_t> best_pick_b(m, 0);

  if (m <= static_cast<size_t>(max_rows)) {
    // Collect active drafters and their distinct candidate draft_ms values (including 0.0).
    struct DrafterCaps {
      int drafter = -1;
      std::vector<double> caps;
    };
    std::vector<DrafterCaps> drafter_caps;
    for (size_t i = 0; i < m; ++i) {
      for (const RoundCandidate& c : sessions[batchable[i]]) {
        if (!c.batchable || c.choice.drafter < 0) continue;
        const double g = cost.draft_ms(c);
        if (g <= 0.0) continue;
        auto it = std::find_if(drafter_caps.begin(), drafter_caps.end(),
                               [&](const DrafterCaps& dc) { return dc.drafter == c.choice.drafter; });
        if (it == drafter_caps.end()) {
          drafter_caps.push_back({c.choice.drafter, {0.0, g}});
        } else {
          it->caps.push_back(g);
        }
      }
    }
    size_t total_combos = 1;
    for (auto& dc : drafter_caps) {
      std::sort(dc.caps.begin(), dc.caps.end());
      dc.caps.erase(std::unique(dc.caps.begin(), dc.caps.end(),
                                [](double a, double b) { return std::abs(a - b) <= 1e-9; }),
                    dc.caps.end());
      total_combos = total_combos <= 64 ? total_combos * dc.caps.size() : 65;
    }
    std::vector<std::vector<double>> cap_combos;
    if (drafter_caps.empty()) {
      cap_combos.push_back({});
    } else if (total_combos <= 64) {
      cap_combos.push_back({});
      for (const auto& dc : drafter_caps) {
        std::vector<std::vector<double>> next;
        next.reserve(cap_combos.size() * dc.caps.size());
        for (const auto& prefix : cap_combos)
          for (double v : dc.caps) {
            next.push_back(prefix);
            next.back().push_back(v);
          }
        cap_combos = std::move(next);
      }
    } else {
      cap_combos.emplace_back(drafter_caps.size(), std::numeric_limits<double>::infinity());
    }

    constexpr double kNegInf = -std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dp(m + 1, std::vector<double>(size_t(max_rows) + 1));
    std::vector<std::vector<int>> pick(m + 1, std::vector<int>(size_t(max_rows) + 1));
    std::vector<size_t> cur_pick(m, 0);
    std::vector<const RoundCandidate*> picked_ptrs(m, nullptr);

    for (const auto& caps : cap_combos) {
      auto allowed = [&](const RoundCandidate& c) {
        if (!c.batchable) return false;
        const int rc = c.choice.width + 1;
        if (rc <= 0 || rc > max_rows) return false;
        if (c.choice.drafter < 0) return true;
        const double g = cost.draft_ms(c);
        if (g <= 0.0) return true;
        for (size_t d = 0; d < drafter_caps.size(); ++d)
          if (drafter_caps[d].drafter == c.choice.drafter) return g <= caps[d] + 1e-9;
        return true;
      };

      int min_rows_sum = 0;
      double init_log_e = 0.0;
      std::vector<size_t> min_pick(m, 0);
      bool feasible = true;
      for (size_t i = 0; i < m; ++i) {
        const auto& cands = sessions[batchable[i]];
        size_t best_k = cands.size();
        int min_r = max_rows + 1;
        for (size_t k = 0; k < cands.size(); ++k) {
          if (!allowed(cands[k])) continue;
          const int r = cands[k].choice.width + 1;
          if (r < min_r || (r == min_r && cands[k].expected_tokens > cands[best_k].expected_tokens)) {
            min_r = r;
            best_k = k;
          }
        }
        if (best_k == cands.size()) {
          feasible = false;
          break;
        }
        min_rows_sum += min_r;
        min_pick[i] = best_k;
        picked_ptrs[i] = &cands[best_k];
        init_log_e += std::log(std::max(1.0, cands[best_k].expected_tokens));
      }
      if (!feasible || min_rows_sum > max_rows) continue;

      double t_ref = cost.batch_ms(min_rows_sum, static_cast<int>(m), cost.shared_draft_ms(picked_ptrs));
      const double init_u = inv_m * init_log_e - std::log(t_ref);
      if (!have_b || init_u > score_b) {
        have_b = true;
        score_b = init_u;
        best_pick_b = min_pick;
      }

      for (int iter = 0; iter < 6; ++iter) {
        const double inv_t = 1.0 / t_ref;
        for (size_t i = 0; i <= m; ++i) {
          std::fill(dp[i].begin(), dp[i].end(), kNegInf);
          std::fill(pick[i].begin(), pick[i].end(), -1);
        }
        dp[0][0] = 0.0;
        for (size_t i = 0; i < m; ++i) {
          const auto& cands = sessions[batchable[i]];
          for (size_t k = 0; k < cands.size(); ++k) {
            const RoundCandidate& c = cands[k];
            if (!allowed(c)) continue;
            const int rc = c.choice.width + 1;
            const double val =
                inv_m * std::log(std::max(1.0, c.expected_tokens)) -
                inv_t * (BatchCostModel::kSharedDraftShare * cost.draft_ms(c));
            for (int r = 0; r + rc <= max_rows; ++r) {
              if (dp[i][size_t(r)] == kNegInf) continue;
              const double cand_val = dp[i][size_t(r)] + val;
              if (cand_val > dp[i + 1][size_t(r + rc)]) {
                dp[i + 1][size_t(r + rc)] = cand_val;
                pick[i + 1][size_t(r + rc)] = static_cast<int>(k);
              }
            }
          }
        }

        double iter_best = kNegInf;
        double iter_t = t_ref;
        for (int R = min_rows_sum; R <= max_rows; ++R) {
          if (dp[m][size_t(R)] == kNegInf) continue;
          int r = R;
          double sum_log_e = 0.0;
          for (size_t i = m; i-- > 0;) {
            const size_t k = static_cast<size_t>(pick[i + 1][size_t(r)]);
            cur_pick[i] = k;
            const RoundCandidate& c = sessions[batchable[i]][k];
            picked_ptrs[i] = &c;
            sum_log_e += std::log(std::max(1.0, c.expected_tokens));
            r -= c.choice.width + 1;
          }
          const double t = cost.batch_ms(R, static_cast<int>(m), cost.shared_draft_ms(picked_ptrs));
          const double u = inv_m * sum_log_e - std::log(t);
          if (u > iter_best) {
            iter_best = u;
            iter_t = t;
          }
          if (u > score_b) {
            score_b = u;
            best_pick_b = cur_pick;
          }
        }
        if (std::abs(iter_t - t_ref) <= 1e-6 * t_ref) break;
        t_ref = iter_t;
      }
    }
  }

  // When Plan A already fits in 1 batch, keep each session's preferred choice: MTP drafting and
  // drafter context are batched in lockstep inside the CUDA backend, so trimming widths that already
  // fit in one weight read reduces tokens/round more than it saves in round time.  When Plan A would
  // spill into 2+ batches (multiple target weight reads), compare against 1-batch packing (Plan B).
  if (plan_a.size() > 1 && have_b && score_b > score_a + 1e-6) {
    ScheduledBatch one;
    one.items.reserve(m);
    for (size_t i = 0; i < m; ++i) {
      const size_t s = batchable[i];
      one.items.push_back({s, sessions[s][best_pick_b[i]].choice});
    }
    result.push_back(std::move(one));
  } else {
    for (auto& b : plan_a) result.push_back(std::move(b));
  }
  return result;
}

}  // namespace flyx
