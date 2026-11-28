// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// ThinkingBudget (core/exec/thinking_budget.h): the tokens of a reply's thinking counted, and once the budget is spent
// only the closing marker's next token allowed.  No model: the markers are token ids picked here.
#include <string>
#include <vector>

#include "core/exec/thinking_budget.h"
#include "tests/common/check.h"

using namespace flyx;

namespace {

constexpr int32_t kOpen = 7, kClose1 = 8, kClose2 = 9, kText = 1;

// Feeds the tokens that are allowed; returns how many were.
int feed(ThinkingBudget& b, const std::vector<int32_t>& tokens) {
  int n = 0;
  for (int32_t t : tokens) {
    if (!b.allows(t)) break;
    b.take(t);
    ++n;
  }
  return n;
}

void spent_then_closed() {
  ThinkingBudget b({kOpen}, {kClose1, kClose2}, 3, /*inside=*/true);
  check(b.active() && b.forced() == -1, "inside, nothing spent: any token");
  check(feed(b, {kText, kText, kText}) == 3, "three tokens of thinking taken");
  check(b.forced() == kClose1 && !b.allows(kText) && b.allows(kClose1), "budget spent: the marker's first token only");
  b.take(kClose1);
  check(b.forced() == kClose2, "  then its second");
  b.take(kClose2);
  check(b.forced() == -1 && b.allows(kText) && b.closed_early(), "  then the answer, any token; closed by the budget");
  check(b.spent() == 3, "the marker is not counted as thinking (" + std::to_string(b.spent()) + ")");
}

void closed_by_the_model() {
  ThinkingBudget b({kOpen}, {kClose1, kClose2}, 5, true);
  check(feed(b, {kText, kClose1, kClose2, kText, kText, kText, kText, kText}) == 8,
        "the model closes its thinking before the budget: nothing forced after, the answer free");
  check(!b.closed_early(), "  not closed by the budget");
}

void a_marker_begun() {
  // The tokens of a marker the model has begun are not thinking until it turns out not to be one.
  ThinkingBudget b({kOpen}, {kClose1, kClose2}, 2, true);
  check(feed(b, {kText, kClose1}) == 2 && b.forced() == -1 && b.spent() == 1,
        "a token, then the marker's first: one token thought, nothing forced");
  const int fed = feed(b, {kText});  // first: MSVC builds a call's arguments right to left
  check(fed == 1 && b.spent() == 3 && b.forced() == kClose1,
        "  the match broken: both counted (" + std::to_string(b.spent()) + "), the budget spent, the marker forced");
  // Spent with the marker begun by the budget: its next token is the marker's next, never a second marker.
  b.take(kClose1);
  check(b.forced() == kClose2, "  the marker's second token next (" + std::to_string(b.forced()) + ")");
}

void counted_across_sections() {
  // Outside thinking any token goes; a section the model opens counts on from what earlier ones spent.
  ThinkingBudget b({kOpen}, {kClose1, kClose2}, 3, /*inside=*/false);
  check(feed(b, {kText, kText, kText, kText}) == 4 && b.forced() == -1, "outside thinking: nothing counted");
  check(feed(b, {kOpen, kText, kText, kClose1, kClose2}) == 5, "a section of 2 tokens (and its marker)");
  const int fed = feed(b, {kOpen, kText});  // first: MSVC builds a call's arguments right to left
  check(fed == 2 && b.forced() == kClose1,
        "a second section: the budget spent across both (" + std::to_string(b.spent()) + " of 3)");
}

void no_budget() {
  ThinkingBudget none;
  check(!none.active() && none.allows(kText) && none.forced() == -1, "no budget: inactive, any token");
  ThinkingBudget no_marker({kOpen}, {}, 3, true);
  check(!no_marker.active(), "no closing marker (a format without thinking): inactive");
  ThinkingBudget zero({kOpen}, {kClose1}, 0, true);
  check(zero.forced() == kClose1, "a budget of 0 inside thinking: the marker first");
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  spent_then_closed();
  closed_by_the_model();
  a_marker_begun();
  counted_across_sections();
  no_budget();
  return flyx::test::finish("thinking_budget_test");
}
