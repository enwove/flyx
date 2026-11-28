// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/thinking_budget.h"

#include <utility>

namespace flyx {

namespace {

// How much of `marker` the tokens end with after t, when they ended with `at` of it before.  The markers are a few
// tokens that do not overlap themselves (<think>, </think>, Inkling-2's control tokens), so a mismatch starts over,
// at the marker's first token where t is that.
size_t advance(const std::vector<int32_t>& marker, size_t at, int32_t t) {
  if (at < marker.size() && marker[at] == t) return at + 1;
  return !marker.empty() && marker[0] == t ? 1 : 0;
}

}  // namespace

ThinkingBudget::ThinkingBudget(std::vector<int32_t> open, std::vector<int32_t> close, int budget, bool inside)
    : open_(std::move(open)), close_(std::move(close)), budget_(budget), inside_(inside) {}

int32_t ThinkingBudget::forced() const {
  if (!active() || !inside_ || spent_ < budget_) return -1;
  return close_[close_at_];
}

bool ThinkingBudget::allows(int32_t t) const {
  const int32_t f = forced();
  return f < 0 || t == f;
}

void ThinkingBudget::take(int32_t t) {
  if (!active()) return;
  if (!inside_) {
    open_at_ = advance(open_, open_at_, t);
    if (!open_.empty() && open_at_ == open_.size()) inside_ = true, open_at_ = 0, close_at_ = 0;
    return;
  }
  // The closing marker is not thinking, written by the model or by the budget: its tokens are held back while they
  // match it, and counted only where the match breaks (they were thinking after all).
  const bool forcing = forced() >= 0;
  forced_now_ = forced_now_ || forcing;
  const size_t at = advance(close_, close_at_, t);
  if (!forcing && at <= close_at_) {
    spent_ += int(close_at_);           // the tokens held back were not the marker
    if (at == 0) ++spent_;              // nor is t, unless it starts the marker again
  }
  close_at_ = at;
  if (close_at_ == close_.size()) {
    closed_early_ = closed_early_ || forced_now_;
    inside_ = false;
    close_at_ = open_at_ = 0;
    forced_now_ = false;
  }
}

}  // namespace flyx
