// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// RoundChooser (core/speculation/chooser.h): which round a session launches
// next.  The drafter's widths by acceptance and cost; with two drafters the
// better one, the other tried now and then; rounds of drafts looked up in the
// sequence when its last tokens occurred before; one round in flight while
// such a round is due; the rest after a poor one; a drafter whose state
// follows its own rounds only (DV) entered by its warm-up where
// it pays, left where it does not, probed again, at the costs of the context.
// No model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/speculation/batch_schedule.h"
#include "core/speculation/chooser.h"

using namespace flyx;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  ++failures;
  std::printf("FAIL %s\n", what.c_str());
}

std::vector<int32_t> range(int32_t from, int n) {
  std::vector<int32_t> v;
  for (int i = 0; i < n; ++i) v.push_back(from + i);
  return v;
}

void append(std::vector<int32_t>* to, const std::vector<int32_t>& more, size_t from = 0, size_t n = size_t(-1)) {
  for (size_t i = from; i < more.size() && i - from < n; ++i) to->push_back(more[i]);
}

const std::vector<int> kWidths = {0, 1, 3, 7};
const std::vector<int> kDrafters = {-1, 0, 0, 0};          // plain decoding, then the drafter's rounds
const std::vector<double> kCost = {5.0, 5.6, 6.3, 8.4};  // ms a round, as measured at load
const std::vector<int> kLookupWidths = {15};

}  // namespace

int main() {
  const std::vector<int32_t> passage = range(100, 20), filler = range(200, 10);

  // The lookup drafter's cursor: a reply copying its source keeps drafting from there; past an edit, kLocalN tokens
  // that agree again near the place it left put it back (the nearest such place, not a look-alike further on).
  {
    std::vector<int32_t> src;  // lines that differ in one token each: 1 2 3 4 k 5 6 7 8 9 (k = 50 + line)
    for (int line = 0; line < 6; ++line) {
      append(&src, {1, 2, 3, 4, 50 + line, 5, 6, 7, 8, 9});
    }
    std::vector<int32_t> seq = src, drafts;
    seq.push_back(0);                        // the reply starts
    append(&seq, src, 0, 10);                // copies line 0 (its 10 tokens)
    LookupDrafter d;
    d.sync(seq);
    check(d.propose(seq, 5, &drafts) > 0 && drafts == std::vector<int32_t>({1, 2, 3, 4, 51}),
          "a copy of line 0 drafts line 1 from the source");
    append(&seq, {1, 2, 3, 4});              // follows the source
    d.sync(seq);
    check(d.propose(seq, 3, &drafts) == LookupDrafter::kMaxN && drafts == std::vector<int32_t>({51, 5, 6}),
          "the reply kept to the source: the cursor drafts on from there");
    append(&seq, {77, 5, 6, 7, 8});          // an edit: 77 for 51, then the source again
    d.sync(seq);
    check(d.propose(seq, 3, &drafts) > 0 && drafts == std::vector<int32_t>({9, 1, 2}),
          "past an edit the cursor is back at line 1, not at a look-alike line");
    append(&seq, {9, 1, 2, 3, 4});
    d.sync(seq);
    check(d.propose(seq, 1, &drafts) > 0 && drafts == std::vector<int32_t>({52}), "and copies line 2 next");
  }

  // The drafter alone: it is tried first; drafts that are all rejected teach
  // plain decoding; plain decoding probes a draft every 32 rounds; drafts
  // that are accepted bring the widest round back.
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, nullptr);
    c.begin(true);
    const std::vector<int32_t> text = range(1, 10);
    RoundChoice r = c.next(text, 0);
    check(!r.lookup && r.width > 0 && kWidths[r.index] == r.width && r.drafter == 0, "a new session drafts first");
    check(c.in_flight_limit() == 2, "the drafter's rounds run two in flight");
    for (int i = 0; i < 40; ++i) {
      r = c.next(text, 0);
      c.result(r, true, 0, text, 0, true);
    }
    check(c.next(text, 0).width == 0, "drafts rejected every round: plain decoding");
    int probes = 0;
    for (int i = 0; i < 64; ++i) {
      r = c.next(text, 0);
      probes += r.width > 0;
      c.result(r, true, 0, text, 0, true);
    }
    check(probes == 2, "plain decoding tries a draft every 32 rounds (" + std::to_string(probes) + " in 64)");
    for (int i = 0; i < 200; ++i) {
      r = c.next(text, 0);
      c.result(r, true, r.width, text, 0, true);
    }
    check(c.next(text, 0).width == kWidths.back(), "drafts accepted every round: the widest round");
  }
  // Rounds read after the reply ended teach nothing.
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, nullptr);
    c.begin(true);
    const std::vector<int32_t> text = range(1, 10);
    const RoundChoice first = c.next(text, 0);
    for (int i = 0; i < 40; ++i) c.result(first, true, 0, text, 0, false);
    check(c.next(text, 0).width == first.width, "results after the reply ended do not change the choice");
  }
  // A pinned width.
  {
    const std::vector<int> one = {3}, drafter = {0};
    const std::vector<double> cost = {6.3};
    RoundChooser c;
    c.configure(&one, &drafter, &cost, &kLookupWidths);
    c.begin(false);
    const std::vector<int32_t> text = range(1, 10);
    bool same = true;
    for (int i = 0; i < 8; ++i) {
      const RoundChoice r = c.next(text, 0);
      same = same && !r.lookup && r.width == 3 && r.index == 0;
      c.result(r, true, 0, text, 0, true);
    }
    check(same, "a pinned width is the only choice");
  }
  // Two drafters, the second's rounds dearer: each drafts once first; then the one whose drafts are
  // accepted; when the other's become the better ones it takes over, the first, cheaper, drafting now and then (its
  // estimate optimistic while it is seen little); when neither's drafts are accepted, plain decoding, each drafter
  // drafting a round in 32.
  {
    const std::vector<int> widths = {0, 1, 3, 7, 3, 7}, drafters = {-1, 0, 0, 0, 1, 1};
    const std::vector<double> cost = {5.0, 5.6, 6.3, 8.4, 7.0, 9.0};
    RoundChooser c;
    c.configure(&widths, &drafters, &cost, nullptr);
    c.begin(true);
    const std::vector<int32_t> text = range(1, 10);
    int by[2] = {0, 0}, plain = 0;
    auto run = [&](int good, int rounds) {
      by[0] = by[1] = plain = 0;
      for (int i = 0; i < rounds; ++i) {
        const RoundChoice r = c.next(text, 0);
        check(widths[r.index] == r.width && drafters[r.index] == r.drafter, "a choice names its round");
        if (r.drafter >= 0) ++by[r.drafter];
        else ++plain;
        c.result(r, true, r.drafter == good ? r.width : 0, text, 0, true);
      }
    };
    run(0, 320);  // the first drafter's drafts accepted, the second's rejected
    check(by[0] == 319 && by[1] == 1,
          "each drafter drafts once first, then the better one (" + std::to_string(by[0]) + ", " +
              std::to_string(by[1]) + ")");
    run(1, 200);  // then the other way round
    const int switched = by[1];
    run(1, 100);
    check(switched >= 150 && by[1] >= 90,
          "the other drafter takes over when its drafts become the better ones (" + std::to_string(switched) +
              " of the next 200 rounds, " + std::to_string(by[1]) + " of the 100 after)");
    run(-1, 200);  // neither drafter's drafts accepted
    run(-1, 64);
    check(plain >= 59 && by[0] + by[1] >= 2 && by[0] + by[1] <= 5 && by[0] > 0 && by[1] > 0,
          "neither's drafts accepted: plain decoding, each drafter a round in 32 (" + std::to_string(plain) + ", " +
              std::to_string(by[0]) + ", " + std::to_string(by[1]) + " of 64)");
    // Pinned to 3 drafts: a round of each drafter's, no plain decoding; the better drafter drafts.
    const std::vector<int> three = {3, 3}, two = {0, 1};
    const std::vector<double> cost3 = {6.3, 7.0};
    RoundChooser p;
    p.configure(&three, &two, &cost3, nullptr);
    p.begin(false);
    int second = 0, none = 0;
    for (int i = 0; i < 64; ++i) {
      const RoundChoice r = p.next(text, 0);
      second += r.drafter == 1;
      none += r.width != 3;
      p.result(r, true, r.drafter == 1 ? 3 : 0, text, 0, true);
    }
    check(none == 0 && second >= 54, "pinned width, two drafters: every round 3 drafts, mostly the better drafter's (" +
                                          std::to_string(second) + " of 64)");
  }

  // A drafter's trees (FLYX_DRAFT_TREE): its chains of 3 and 7, its trees of 9 and 11 whose first 7 drafts are
  // its chain of 7.  Every round the drafter's drafts are accepted 3 deep.  Where a tree's branches add 3 more tokens,
  // the trees' trial rounds (one each in kTreeProbe) find it and the tree that gives the most tokens a millisecond
  // drafts; where they add nothing, the chain that does (the chain of 3), the trees tried once in kTreeProbe rounds.
  // A choice names the drafter and the width it verifies.
  {
    const std::vector<int> widths = {0, 3, 7, 9, 11}, drafters = {-1, 0, 0, 0, 0}, trunks = {0, 0, 0, 7, 7};
    const std::vector<double> cost = {5.0, 6.3, 8.4, 9.2, 10.0};
    RoundChooser c;
    c.configure(&widths, &drafters, &cost, nullptr, &trunks);
    c.begin(true);
    const std::vector<int32_t> text = range(1, 10);
    int by[5] = {}, named = 0;
    auto run = [&](int branches, int rounds) {
      for (int& b : by) b = 0;
      for (int i = 0; i < rounds; ++i) {
        const RoundChoice r = c.next(text, 0);
        named += widths[r.index] == r.width && drafters[r.index] == r.drafter;
        ++by[r.index];
        const bool tree = trunks[r.index] > 0;
        c.result(r, true, std::min(3, r.width) + (tree ? branches : 0), text, 0, true, tree ? 3 : 0);
      }
    };
    auto counts = [&] {
      return std::to_string(by[0]) + " " + std::to_string(by[1]) + " " + std::to_string(by[2]) + " " +
             std::to_string(by[3]) + " " + std::to_string(by[4]);
    };
    run(3, 300);  // 4 + 3 tokens in 9.2 ms: the tree of 9 (the chain of 3: 4 in 6.3)
    run(3, 200);
    check(named == 500, "a tree round names its drafter and width");
    check(by[3] >= 180, "branches that add tokens: the tree giving the most tokens a ms (rounds by kind: " + counts() +
                            ")");
    run(0, 200);  // the branches add nothing: the chain of 3
    run(0, 128);
    check(by[1] >= 122 && by[3] + by[4] <= 6,
          "branches that add nothing: the chain, each tree once in 64 rounds (rounds by kind: " + counts() + ")");
  }

  // Lookup: the last tokens occurred before, so a round verifies what
  // followed them there, and such a round runs alone.
  std::vector<int32_t> text = passage;
  append(&text, filler);
  append(&text, passage, 0, 8);  // the passage starts again
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, &kLookupWidths);
    c.begin(true);
    RoundChoice r = c.next(text, 0);
    std::vector<int32_t> want(passage.begin() + 8, passage.end());  // the rest of the passage, then what followed it
    append(&want, filler, 0, 3);
    check(r.lookup && r.width == 15 && r.index == 0 && c.drafts() == want, "text seen before: its continuation is the draft");
    check(c.in_flight_limit() == 1, "a lookup round runs alone");

    // All 15 accepted (and the token after them): the copy goes on.
    std::vector<int32_t> more = text;
    append(&more, want);
    more.push_back(filler[3]);
    c.result(r, true, 15, more, 0, true);
    r = c.next(more, 0);
    check(r.lookup && !c.drafts().empty() && c.drafts()[0] == filler[4], "a lookup round that was accepted is followed by another");

    // One draft accepted: a chance match.  Lookup rests for kLookupRest
    // tokens, even where the text matches again, then comes back.
    std::vector<int32_t> after = more;
    after.push_back(c.drafts()[0]);
    after.push_back(900);
    c.result(r, true, 1, after, 0, true);
    const size_t rests_until = after.size() + RoundChooser::kLookupRest;
    append(&after, passage, 0, 8);
    bool rested = true, two = true;
    for (int32_t t = 1000; after.size() < rests_until; ++t) {
      r = c.next(after, 0);
      rested = rested && !r.lookup;
      two = two && c.in_flight_limit() == 2;
      c.result(r, true, 0, after, 0, true);
      after.push_back(t);
    }
    check(rested, "a poor lookup round: no lookup for the next 32 tokens");
    check(two, "  the rest runs the drafter's rounds two in flight");
    append(&after, passage, 0, 8);
    check(c.next(after, 0).lookup, "after the rest, lookup again");
  }
  // The text starts to repeat while another round is in flight: nothing more
  // is launched until that round is read, then the lookup round starts where
  // it ended.
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, &kLookupWidths);
    c.begin(true);
    std::vector<int32_t> t = passage;
    append(&t, filler);
    const RoundChoice r1 = c.next(t, 0), r2 = c.next(t, 1);
    check(!r1.lookup && !r2.lookup && c.in_flight_limit() == 2, "new text: the drafter's rounds, two in flight");
    append(&t, passage, 0, 8);
    c.result(r1, true, r1.width, t, 1, true);
    check(c.in_flight_limit() == 1, "the text repeats with a round in flight: the next launch waits for it");
    append(&t, passage, 8, 2);
    c.result(r2, true, 1, t, 0, true);
    const RoundChoice r3 = c.next(t, 0);
    check(r3.lookup && c.drafts()[0] == passage[10], "then a lookup round from where that round ended");
  }
  // A reply that may not use lookup; the same sequence after a cut.
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, &kLookupWidths);
    c.begin(false);
    check(!c.next(text, 0).lookup && c.in_flight_limit() == 2, "lookup off for a reply: the drafter's rounds only");
    c.begin(true);
    check(c.next(text, 0).lookup, "lookup on again for the next reply");
    std::vector<int32_t> again(text.begin(), text.begin() + 30);  // cut back to before the repeat
    c.cut(again.size());
    check(!c.next(again, 0).lookup, "cut back to before the repeat: nothing to look up");
    append(&again, passage, 0, 8);
    const RoundChoice r = c.next(again, 0);
    check(r.lookup && c.drafts()[0] == passage[8], "the repeat written again after a cut is found again");
  }
  // No lookup rounds in the engine at all.
  {
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, nullptr);
    c.begin(true);
    check(!c.next(text, 0).lookup, "an engine without lookup rounds never looks up");
  }

  // Multi-width lookup ({3, 7, 15}): solo long match picks 15 and offers {3, 7, 15} as candidates;
  // shorter match (8 tokens) picks 7 instead of falling back; 3/3 accepted on lookup 3 does not rest,
  // while 2/3 accepted does.
  {
    const std::vector<int> multi_lookup = {3, 7, 15};
    const std::vector<double> lookup_ms = {16.0, 21.0, 36.0};
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, &multi_lookup, nullptr, &lookup_ms);
    c.begin(true);
    std::vector<RoundCandidate> cands;
    c.candidates(text, 0, &cands);
    check(cands.size() == 3 && cands[0].choice.width == 3 && cands[1].choice.width == 7 &&
              cands[2].choice.width == 15 && cands[2].preferred && !cands[0].preferred,
          "multi-width lookup offers {3, 7, 15} with 15 preferred on a long match");
    // Repeated candidates() calls without commit() do not mutate state.
    std::vector<RoundCandidate> again;
    c.candidates(text, 0, &again);
    check(again.size() == 3 && c.in_flight_limit() == 2, "candidates() is side-effect-free before commit()");
    // Commit width 3 and accept all 3 drafts: must NOT trigger a rest (worth = min(4, 3) = 3).
    c.commit(cands[0].choice, 0);
    check(c.in_flight_limit() == 1, "committing a lookup candidate enters lookup mode");
    std::vector<int32_t> next_text = text;
    append(&next_text, passage, 8, 4);
    c.result(cands[0].choice, true, 3, next_text, 0, true);
    c.candidates(next_text, 0, &cands);
    check(!cands.empty() && cands.back().choice.lookup, "3/3 accepted on lookup width 3 keeps lookup active");
    // Accept only 2/3 drafts on lookup width 3: triggers kLookupRest.
    c.commit(cands[0].choice, 0);
    c.result(cands[0].choice, true, 2, next_text, 0, true);
    c.candidates(next_text, 0, &cands);
    check(!cands.empty() && !cands.front().choice.lookup && c.in_flight_limit() == 2,
          "2/3 accepted on lookup width 3 triggers lookup rest");

    // Solo next() still picks the widest lookup width (15) on a match.
    RoundChooser cs;
    cs.configure(&kWidths, &kDrafters, &kCost, &multi_lookup, nullptr, &lookup_ms);
    cs.begin(true);
    const RoundChoice solo_lk = cs.next(text, 0);
    check(solo_lk.lookup && solo_lk.width == 15 && solo_lk.index == 2,
          "solo next() still picks widest lookup width (15) when {3, 7, 15} are configured");
  }

  // Wide lookup rounds ({15, 31, 63}): after two 15-draft rounds accepted whole, the widest (alone, never in a batch);
  // kept while it yields more tokens a millisecond than full 15-draft rounds, else back to 15 and twice the full
  // rounds before widening again.  A reply that may not widen (penalties, relaxed acceptance, grammar) never does.
  {
    const std::vector<int> wide_lookup = {15, 31, 63};
    const std::vector<double> wide_ms = {16.5, 20.0, 27.0};
    const std::vector<int32_t> source = range(1000, 400);
    std::vector<int32_t> seq = source;
    append(&seq, range(5000, 10));
    append(&seq, source, 0, 8);
    size_t copied = 8;  // the reply follows the source; `accepted` is what each round claims
    RoundChooser c;
    c.configure(&kWidths, &kDrafters, &kCost, &wide_lookup, nullptr, &wide_ms);
    c.begin(true);
    auto round = [&](int accepted, int* width, bool* batchable) {
      std::vector<RoundCandidate> cands;
      c.candidates(seq, 0, &cands);
      RoundChoice pick = cands.front().choice;
      *batchable = true;
      for (const auto& k : cands)
        if (k.preferred) {
          pick = k.choice;
          *batchable = k.batchable;
        }
      c.commit(pick, 0);
      *width = pick.lookup ? pick.width : -1;
      append(&seq, source, copied, size_t(std::min(accepted, pick.width) + 1));
      copied += size_t(std::min(accepted, pick.width) + 1);
      c.result(pick, true, std::min(accepted, pick.width), seq, 0, true);
    };
    int w = 0;
    bool batchable = true;
    round(15, &w, &batchable);
    check(w == 15 && batchable, "wide: the first copy round verifies 15 drafts");
    round(15, &w, &batchable);
    check(w == 15, "wide: one full round is not enough");
    round(40, &w, &batchable);
    check(w == 63 && !batchable, "wide: after two full rounds the widest, alone");
    round(10, &w, &batchable);
    check(w == 63, "wide: 41 tokens in 27 ms beat full 15-draft rounds: still wide");
    round(15, &w, &batchable);
    check(w == 15, "wide: 11 tokens in 27 ms fall short: back to 15 drafts");
    round(15, &w, &batchable);
    round(15, &w, &batchable);
    round(15, &w, &batchable);
    check(w == 15, "wide: after backing off, four full rounds before widening");
    round(63, &w, &batchable);
    check(w == 63, "wide: widened again after the fourth");

    RoundChooser n;
    n.configure(&kWidths, &kDrafters, &kCost, &wide_lookup, nullptr, &wide_ms);
    n.begin(true, false);
    bool narrow = true;
    for (int i = 0; i < 4; ++i) {
      const RoundChoice r = n.next(seq, 0);
      narrow = narrow && r.lookup && r.width == 15;
      std::vector<int32_t> more = seq;
      append(&more, source, copied, 16);
      n.result(r, true, 15, more, 0, true);
    }
    check(narrow, "wide: a reply that may not widen stays at 15 drafts");

    // A copy that breaks every dozen tokens or so (four 15-draft rounds cut at 12), then two full rounds: widened, the
    // width that gives the most tokens a millisecond for drafts that miss about one in 25 is 31, not 63.
    RoundChooser b;
    b.configure(&kWidths, &kDrafters, &kCost, &wide_lookup, nullptr, &wide_ms);
    b.begin(true);
    auto bround = [&](int accepted, int* width) {
      const RoundChoice r = b.next(seq, 0);
      *width = r.lookup ? r.width : -1;
      const int a = std::min(accepted, r.width);
      append(&seq, source, copied, size_t(a + 1));
      copied += size_t(a + 1);
      b.result(r, true, a, seq, 0, true);
    };
    for (int i = 0; i < 4; ++i) bround(12, &w);
    bround(15, &w);
    bround(15, &w);
    bround(31, &w);
    check(w == 31, "wide: drafts that miss one in 25 or so widen to 31, not 63 (" + std::to_string(w) + ")");
  }

  // A drafter round that kept almost all of a long chain (7 drafts, 6 or 7 accepted): the next one drafts one width
  // wider (15: a DFlash2 block of 8 drafting wide); after one that kept less, the controller's own choice again.
  {
    const std::vector<int> widths = {0, 3, 7, 15};
    const std::vector<int> drafters = {-1, 0, 0, 0};
    const std::vector<double> cost = {12.7, 14.4, 15.2, 19.1};
    RoundChooser c;
    c.configure(&widths, &drafters, &cost, nullptr);
    c.begin(false);
    const std::vector<int32_t> text = range(100, 20);
    RoundChoice r{false, 2, 7, 0};
    c.commit(r, 0);
    c.result(r, true, 6, text, 0, true);
    const RoundChoice wide = c.next(text, 0);
    check(!wide.lookup && wide.width == 15, "a chain of 7 that kept 6: the next round drafts 15 (" +
                                                std::to_string(wide.width) + ")");
    c.result(wide, true, 4, text, 0, true);
    const RoundChoice after = c.next(text, 0);
    check(after.width != 15 || after.drafter < 0, "a chain of 15 that kept 4: no run, the controller's choice (" +
                                                      std::to_string(after.width) + ")");
    c.result(after, true, 0, text, 0, true);
    RoundChoice three{false, 1, 3, 0};
    c.commit(three, 0);
    c.result(three, true, 3, text, 0, true);
    const RoundChoice short_run = c.next(text, 0);
    check(short_run.width != 15, "a chain of 3 kept whole is no long run (" + std::to_string(short_run.width) + ")");

    // Wider rounds that give fewer tokens a millisecond than the chain of 7 would have (15 drafts keeping 7 or 8 at
    // 19.1 ms against 8 tokens at 15.2): runs stop widening, and widen again once every kRunProbe runs.
    RoundChooser p;
    p.configure(&widths, &drafters, &cost, nullptr);
    p.begin(false);
    RoundChoice seven{false, 2, 7, 0}, fifteen{false, 3, 15, 0};
    for (int i = 0; i < 4; ++i) {
      p.commit(fifteen, 0);
      p.result(fifteen, true, 7, text, 0, true);
    }
    int widened = 0;
    for (int i = 0; i < 16; ++i) {
      p.commit(seven, 0);
      p.result(seven, true, 7, text, 0, true);
      std::vector<RoundCandidate> cands;
      p.candidates(text, 0, &cands);
      for (const auto& k : cands)
        if (k.preferred && k.choice.width == 15) ++widened;
    }
    check(widened == 1, "wider rounds that do not pay: runs stop widening, one try in " +
                            std::to_string(16) + " (" + std::to_string(widened) + ")");
  }

  // BatchCostModel + plan_batches():
  // 1) Heterogeneous 4-session batch packs into 1 shared batch, allocates wider drafts to higher-acceptance sessions,
  //    and beats naive preferred FFD packing.
  // 2) Exact Dinkelbach DP matches brute-force optimal 1-batch rate across diverse cost curves & cliffs.
  // 3) Non-batchable sessions (relaxed acceptance / tree probe) run alone without poisoning batchable peers.
  {
    const std::vector<int> widths = {0, 1, 2, 3, 4, 5, 7};
    const std::vector<int> drafters = {-1, 0, 0, 0, 0, 0, 0};
    const std::vector<double> round_ms = {14.0, 15.6, 17.7, 19.3, 22.8, 25.6, 31.3};
    const std::vector<int> lookup_w = {3, 7, 15};
    const std::vector<double> lookup_ms = {16.5, 24.5, 36.5};
    BatchCostModel cost;
    cost.build(widths, drafters, round_ms, lookup_w, lookup_ms);
    check(cost.target_ms(1) == 14.0 && cost.target_ms(4) == 16.5 && cost.target_ms(8) == 24.5 &&
              cost.target_ms(16) == 36.5,
          "BatchCostModel reproduces calibrated plain and lookup knots");

    // Helper: build a session's candidates with per-position acceptance p^k.
    auto make_session = [&](double p) {
      std::vector<RoundCandidate> out;
      double best_rate = -1;
      size_t best_idx = 0;
      for (size_t i = 0; i < widths.size(); ++i) {
        double e = 1.0, prod = 1.0;
        for (int k = 0; k < widths[i]; ++k) {
          prod *= p;
          e += prod;
        }
        const double rate = e / round_ms[i];
        if (rate > best_rate) {
          best_rate = rate;
          best_idx = i;
        }
        out.push_back({{false, i, widths[i], drafters[i]}, e, round_ms[i], true, false});
      }
      out[best_idx].preferred = true;
      return out;
    };

    // 4 high-acceptance sessions (0.92, 0.85, 0.78, 0.65): solo preferred widths overflow FLYX_MAX_ROWS
    // (8 + 8 + 6 + 4 = 26 -> 2 batches), while plan_batches packs all 4 into 1 batch with monotonic widths.
    std::vector<std::vector<RoundCandidate>> s4 = {make_session(0.92), make_session(0.85), make_session(0.78),
                                                   make_session(0.65)};
    const auto batches = plan_batches(cost, s4);
    check(batches.size() == 1 && batches[0].items.size() == 4,
          "4 heterogeneous sessions pack into 1 shared batch instead of splitting into 2");
    int w_by_sess[4] = {};
    int total_rows = 0;
    for (const auto& item : batches[0].items) {
      w_by_sess[item.session] = item.choice.width;
      total_rows += item.choice.width + 1;
    }
    check(total_rows <= FLYX_MAX_ROWS && w_by_sess[0] >= w_by_sess[1] && w_by_sess[1] >= w_by_sess[2] &&
              w_by_sess[2] >= w_by_sess[3],
          "1-batch allocation assigns wider drafts to higher-acceptance sessions (" + std::to_string(w_by_sess[0]) +
              ", " + std::to_string(w_by_sess[1]) + ", " + std::to_string(w_by_sess[2]) + ", " +
              std::to_string(w_by_sess[3]) + "; " + std::to_string(total_rows) + " rows)");

    // Brute-force oracle check on the same 4 sessions: verify the proportional-fairness cap-enumerated DP hits the
    // exact global 1-batch optimum under shared-drafter cost accounting without starving lower-acceptance sessions.
    double brute_best = -1e30;
    for (const auto& c0 : s4[0])
      for (const auto& c1 : s4[1])
        for (const auto& c2 : s4[2])
          for (const auto& c3 : s4[3]) {
            const int r = (c0.choice.width + 1) + (c1.choice.width + 1) + (c2.choice.width + 1) + (c3.choice.width + 1);
            if (r > FLYX_MAX_ROWS) continue;
            const double log_e = 0.25 * (std::log(c0.expected_tokens) + std::log(c1.expected_tokens) +
                                         std::log(c2.expected_tokens) + std::log(c3.expected_tokens));
            const double d = cost.shared_draft_ms({&c0, &c1, &c2, &c3});
            const double t = cost.batch_ms(r, 4, d);
            const double u = log_e - std::log(t);
            if (u > brute_best) brute_best = u;
          }
    double got_log_e = 0.0;
    std::vector<const RoundCandidate*> got_ptrs;
    for (const auto& item : batches[0].items) {
      for (const auto& c : s4[item.session])
        if (c.choice.index == item.choice.index && c.choice.lookup == item.choice.lookup) {
          got_log_e += 0.25 * std::log(c.expected_tokens);
          got_ptrs.push_back(&c);
          break;
        }
    }
    const double got_u = got_log_e - std::log(cost.batch_ms(total_rows, 4, cost.shared_draft_ms(got_ptrs)));
    check(got_u >= brute_best - 1e-6, "proportional-fairness 1-batch DP matches exact brute-force global optimum");
    check(w_by_sess[3] >= 2, "shared-drafter cost lets trailing session keep drafting instead of starving to plain");

    // Non-batchable session (e.g. relaxed acceptance) runs solo without blocking the other 3 from sharing 1 batch.
    std::vector<std::vector<RoundCandidate>> mixed = {make_session(0.85), make_session(0.75), make_session(0.60),
                                                      make_session(0.80)};
    for (auto& c : mixed[1]) c.batchable = false;
    const auto mb = plan_batches(cost, mixed);
    check(mb.size() == 2 &&
              ((mb[0].items.size() == 1 && mb[0].items[0].session == 1 && mb[1].items.size() == 3) ||
               (mb[1].items.size() == 1 && mb[1].items[0].session == 1 && mb[0].items.size() == 3)),
          "a non-batchable session runs solo without splitting the 3 batchable sessions");

    // Lookup session + 2 drafted sessions: lookup narrows from 15 to 7 or 3 so all 3 share 1 batch.
    std::vector<RoundCandidate> lk = {
        {{true, 0, 3, -1}, 1.0 + 0.75 * 3, 16.5, true, false},
        {{true, 1, 7, -1}, 1.0 + 0.75 * 7, 24.5, true, false},
        {{true, 2, 15, -1}, 1.0 + 0.75 * 15, 36.5, true, true},
    };
    std::vector<std::vector<RoundCandidate>> with_lk = {lk, make_session(0.75), make_session(0.65)};
    const auto lb = plan_batches(cost, with_lk);
    int lk_rows = 0;
    for (const auto& item : lb[0].items) lk_rows += item.choice.width + 1;
    check(lb.size() == 1 && lb[0].items.size() == 3 && lk_rows <= FLYX_MAX_ROWS,
          "multi-width lookup narrows to share one batch with drafted sessions (" + std::to_string(lk_rows) + " rows)");
  }

  // LookupDrafter flat hash table: latest match wins, periodic repeat, cut() invalidates stale entries, growth/rehash.
  {
    LookupDrafter ld;
    std::vector<int32_t> seq = range(1000, 12);
    append(&seq, range(2000, 4));  // first continuation after 1000..1007: 1008, 1009, 1010, 1011
    append(&seq, range(1000, 8));  // second occurrence of 1000..1007
    append(&seq, range(3000, 5));  // second continuation: 3000..3004
    append(&seq, range(1000, 8));  // tail matches both; latest (3000..3004) must win
    ld.sync(seq);
    std::vector<int32_t> draft;
    check(ld.propose(seq, 4, &draft) == 8 && draft == range(3000, 4), "lookup picks latest earlier continuation");

    // Cutting back before the second occurrence removes it so the first occurrence wins again.
    seq.resize(16);  // 1000..1011 + 2000..2003
    append(&seq, range(1000, 8));
    ld.cut(16);
    ld.sync(seq);
    check(ld.propose(seq, 4, &draft) == 8 && draft == range(1008, 4), "lookup cut invalidates stale entries");

    // Growth across multiple rehashes (5000 tokens -> 15k n-grams) + repeated reset generations.
    std::vector<int32_t> big = range(5000, 5000);
    append(&big, range(5100, 8));
    ld.reset();
    ld.sync(big);
    check(ld.propose(big, 6, &draft) == 8 && draft == range(5108, 6), "lookup stays exact after multi-step rehash");
    ld.reset();
    check(ld.propose(big, 6, &draft) == 0 && draft.empty(), "lookup reset clears all entries in O(1)");

    // Cutting back to n == indexed_ (last token rolled back and replaced) must drop the active copy cursor.
    std::vector<int32_t> tail_cut = range(7000, 12);
    append(&tail_cut, range(7000, 8));
    ld.sync(tail_cut);
    check(ld.propose(tail_cut, 4, &draft) == 8 && draft == range(7008, 4),
          "lookup sets active copy cursor before single-token rollback");
    tail_cut.pop_back();
    ld.cut(tail_cut.size());  // n == indexed_: must still clear src_/left_/at_
    tail_cut.push_back(999999);
    check(ld.propose(tail_cut, 4, &draft) == 0 && draft.empty(),
          "lookup cut(indexed_) invalidates active copy cursor on divergent continuation");
  }

    // A run widened past the drafter's adaptive round (checkpoints up to 7, the run's next width 15) keeps the fixed
    // width; without a run the adaptive round is the preferred one.
  {
    const std::vector<int> widths = {0, 3, 7, 15};
    const std::vector<int> drafters = {-1, 0, 0, 0};
    const std::vector<double> costs = {12.7, 14.4, 15.2, 19.1};
    const std::vector<int> lk_widths = {};
    const std::vector<std::vector<int>> cps = {{3, 7}};
    RoundChooser rc;
    rc.configure(&widths, &drafters, &costs, &lk_widths, nullptr, nullptr, &cps);
    rc.begin(false, true, true);
    const RoundChoice first = rc.next({}, 0);
    check(first.adaptive, "no run: the adaptive round");
    RoundChoice seven{false, 2, 7, 0, false};
    rc.commit(seven, 0);
    rc.result(seven, true, 6, {}, 0, true);
    const RoundChoice wide = rc.next({}, 0);
    check(!wide.adaptive && wide.width == 15, "a run widened to 15 past adaptive checkpoints up to 7: fixed 15 (" +
                                                  std::to_string(wide.width) + (wide.adaptive ? ", adaptive)" : ")"));
  }

    // ConfidenceStopper online calibration + RoundChooser adaptive candidate selection (solo adaptive, batched fixed).
  {
    const std::vector<int> widths = {0, 1, 3, 5, 7, 15};
    const std::vector<int> drafters = {-1, 0, 0, 0, 0, 0};
    const std::vector<double> costs = {15.6, 17.5, 20.7, 23.8, 26.8, 35.8};
    const std::vector<int> lk_widths = {};
    const std::vector<std::vector<int>> adaptive_cps = {{1, 3, 5, 7, 15}};
    RoundChooser rc;
    rc.configure(&widths, &drafters, &costs, &lk_widths, nullptr, nullptr, &adaptive_cps);
    rc.begin(false, true, true);
    check(rc.stop_policy(0) != nullptr && rc.stop_policy(0)->n_buckets == 5 && rc.stop_policy(0)->enabled == 1,
          "adaptive chooser exposes initialized stop_policy");
    const float lam_27b = rc.stop_policy(0)->lambda;
    // Scale-invariant initial lambda: a 10x faster model yields 10x higher initial lambda.
    ConfidenceStopper fast;
    fast.configure({1, 3, 5, 7, 15}, {1.75, 2.07, 2.38, 2.68, 3.58}, 1.56);
    check(std::fabs(fast.policy().lambda - 10.f * lam_27b) < 1e-4f, "ConfidenceStopper initial lambda is scale-invariant");

    // Monotonic cost sanitization: calibration jitter on fast models never yields dc <= 0.
    ConfidenceStopper jitter;
    jitter.configure({1, 3, 5}, {2.05, 2.00, 1.98}, 1.80);
    check(jitter.policy().costs[1] > jitter.policy().costs[0] && jitter.policy().costs[2] > jitter.policy().costs[1],
          "ConfidenceStopper enforces strictly increasing bucket costs under jitter");

    // Cross-model logit scale adaptation (4x smaller margins) and uncalibrated noise fallback.
    ConfidenceStopper small_m;
    small_m.configure({1, 3, 5, 7}, {17.5, 20.7, 23.8, 26.8}, 15.6);
    const float lo_p[FLYX_MAX_ROWS] = {0.55f, 0.55f, 0.55f};
    const float lo_m[FLYX_MAX_ROWS] = {0.2f, 0.2f, 0.2f};
    const float hi_p[FLYX_MAX_ROWS] = {0.92f, 0.92f, 0.92f};
    const float hi_m[FLYX_MAX_ROWS] = {1.5f, 1.5f, 1.5f};
    const float w1_init = small_m.policy().w_margin;
    for (int i = 0; i < 30; ++i) {
      small_m.observe(3, 3, hi_p, hi_m);
      small_m.observe(3, 0, lo_p, lo_m);
    }
    check(small_m.policy().w_margin > w1_init, "diagonal-Newton adapts w_margin up for flat-logit model families");

    ConfidenceStopper noisy;
    noisy.configure({1, 3, 5, 7}, {17.5, 20.7, 23.8, 26.8}, 15.6);
    for (int i = 0; i < 60; ++i) {
      noisy.observe(3, 0, hi_p, hi_m);  // high confidence rejected, low confidence accepted -> zero signal
      noisy.observe(3, 3, lo_p, lo_m);
    }
    check(noisy.policy().w_margin < w1_init && noisy.policy().w_margin >= 0.f,
          "uncalibrated drafter decays w_margin cleanly toward zero");

    std::vector<RoundCandidate> cands;
    rc.candidates({}, 0, &cands);
    check(!cands.empty() && cands.front().preferred && cands.front().choice.adaptive && !cands.front().batchable,
          "solo preferred candidate is adaptive and non-batchable");
    bool has_batchable = false;
    for (const auto& c : cands)
      if (!c.choice.adaptive && c.batchable && c.choice.drafter == 0) has_batchable = true;
    check(has_batchable, "adaptive chooser still emits fixed batchable candidates for multi-session scheduling");

    // Online calibration moves bias up when high-margin drafts are accepted and updates p_pos / lambda.
    const float bias0 = rc.stop_policy(0)->w_bias;
    const float hp[FLYX_MAX_ROWS] = {0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f};
    const float hm[FLYX_MAX_ROWS] = {10.f, 10.f, 10.f, 10.f, 10.f, 10.f, 10.f};
    const RoundChoice ch = rc.next({}, 0);
    check(ch.adaptive && ch.drafter == 0, "next() selects adaptive round when enabled");
    for (int i = 0; i < 10; ++i) rc.result(ch, true, 7, {}, 0, true, 0, 7, hp, hm);
    check(rc.stop_policy(0)->p_pos[0] > 0.90f && rc.stop_policy(0)->w_bias >= bias0,
          "online ConfidenceStopper updates p_pos and logistic bias from adaptive results");

    // Tree rounds must neither update linear ConfidenceStopper nor trigger linear-run widening.
    const std::vector<int> tw = {0, 1, 3, 5, 7, 7};
    const std::vector<int> td = {-1, 0, 0, 0, 0, 0};
    const std::vector<double> tc = {15.6, 17.5, 20.7, 23.8, 26.8, 27.2};
    const std::vector<int> trunks = {0, 0, 0, 0, 0, 4};
    const std::vector<std::vector<int>> t_cps = {{1, 3, 5, 7}};
    RoundChooser trc;
    trc.configure(&tw, &td, &tc, &lk_widths, &trunks, nullptr, &t_cps);
    trc.begin(false, true, true);
    const flyx_stop_policy before = *trc.stop_policy(0);
    const RoundChoice tree_round{false, 5, 7, 0, false};
    for (int i = 0; i < 20; ++i) trc.result(tree_round, true, 7, {}, 0, true, 4, 7, hp, hm);
    check(std::memcmp(&before, trc.stop_policy(0), sizeof(before)) == 0,
          "tree round results do not mutate linear ConfidenceStopper");
    trc.begin(false, true, false);  // non-adaptive: check tree result does not force linear-run widening
    trc.result(RoundChoice{false, 5, 7, 0, false}, true, 7, {}, 0, false, 4);
    const RoundChoice after_tree = trc.next({}, 0);
    check(after_tree.width != 7 || after_tree.index != 4,
          "tree round acceptance does not trigger linear-chain wide-run widening");
  }

  // A drafter whose state follows its own rounds only (DV's canvas): plain decoding, its trees of 15
  // and 31 (no trunk), its warm-up (a round of no drafts of it), the costs this GPU measured at 64 tokens of context
  // and per token past it.  A fake backend keeps the canvas: it follows after the warm-up or a DV round, not after any
  // other round or a cut; a DV round whose canvas does not follow drafts DV's first tree (nothing accepted).  Its
  // trees accept accept(width, tokens so far) drafts while the canvas follows.  The engine's sessions share a
  // SharedTokens where a test says so.
  {
    const std::vector<int> widths = {0, 15, 31, 0}, drafters = {-1, 0, 0, 0}, trunks = {0, -1, -1, 0};
    const std::vector<double> cost = {4.40, 12.85, 15.25, 11.73}, per_token = {0.000083, 0.00157, 0.00215, 0.00093};
    const size_t kPlain = 0, k15 = 1, k31 = 2, kWarm = 3;
    struct Dv {
      RoundChooser c;
      std::vector<int32_t> text;
      bool follows = false;  // the canvas follows the sequence's last round
      bool grows = true;     // the context grows by each round's tokens (false: it stays as long)
      std::function<int(int width, size_t at)> accept = [](int width, size_t) { return width == 15 ? 5 : 6; };
      int by[4] = {}, behind = 0, after_probe = 0;  // rounds by kind; DV trees on a canvas that did not follow
      double ms = 0;
      int32_t next_token = 1;
      void step(const std::vector<double>& cost, const std::vector<double>& per_token) {
        const RoundChoice r = c.next(text, 0);
        int accepted = 0;
        if (r.lookup) {
          follows = false;
        } else {
          const size_t i = r.index;
          ++by[i];
          ms += cost[i] + per_token[i] * std::max<int>(0, static_cast<int>(text.size()) - 64);
          if (r.width > 0) {
            if (!follows) ++behind;
            accepted = follows ? std::min(r.width, accept(r.width, text.size())) : 0;
          }
          follows = r.drafter >= 0;
        }
        for (int k = 0; k <= accepted; ++k) grows ? text.push_back(next_token++) : (void)next_token++;
        c.result(r, true, accepted, text, 0, true);
      }
      void run(int rounds, const std::vector<double>& cost, const std::vector<double>& per_token) {
        for (int& b : by) b = 0;
        ms = 0;
        for (int k = 0; k < rounds; ++k) step(cost, per_token);
      }
      int trees() const { return by[1] + by[2]; }
      std::string counts() const {
        return std::to_string(by[0]) + " plain, " + std::to_string(by[1]) + " of 15, " + std::to_string(by[2]) +
               " of 31, " + std::to_string(by[3]) + " warm-ups";
      }
    };
    // A reply after a prompt of `context` tokens that repeats nothing, in a new session (of the engine `shared`).
    auto make = [&](int context, bool grows = true, SharedTokens* shared = nullptr) {
      auto d = std::make_unique<Dv>();
      d->grows = grows;
      d->c.configure(&widths, &drafters, &cost, &kLookupWidths, &trunks, nullptr, nullptr, &per_token, 64, shared);
      d->c.begin(true);
      d->text = range(100000, context);
      return d;
    };
    auto probe_ms = [&](const Dv& d) {
      return d.by[k15] * (cost[k15] - cost[kPlain]) + d.by[k31] * (cost[k31] - cost[kPlain]) +
             d.by[kWarm] * (cost[kWarm] - cost[kPlain]);
    };
    // A reply at a short context (300 tokens, kept so): DV enters at once, its warm-up first; its trees pay, so it
    // drafts every round, at the width the engine's means give more tokens a millisecond.
    auto d = make(300, false);
    const RoundChoice first = d->c.next(d->text, 0);
    check(!first.lookup && first.index == kWarm && first.drafter == 0 && first.width == 0,
          "a restarting drafter that pays: its warm-up first (round " + std::to_string(first.index) + ")");
    d->c.result(first, true, 0, d->text, 0, true);
    d->follows = true;
    d->run(200, cost, per_token);
    check(d->trees() == 200 && d->behind == 0,
          "  its trees accepted 5 and 6 deep: DV every round (" + d->counts() + ")");
    check(d->by[k31] >= 1 && d->by[k31] <= 200 / SpecController::kWidthTurn + 1,
          "  rounds of 15 (6 tokens in 12.9 ms) over rounds of 31 (7 in 15.3), 31 drafting one round in " +
              std::to_string(SpecController::kWidthTurn) + " (" + d->counts() + ")");
    // Its trees stop paying (no draft accepted): plain decoding once its level says so; after it, a DV round only
    // after its warm-up (the canvas forgotten), and probes (each its warm-up and kProbeRounds trees) cost about
    // kProbeShare of the time.
    d->accept = [](int, size_t) { return 0; };
    d->run(40, cost, per_token);
    check(d->by[kPlain] >= 10, "trees that accept nothing: plain decoding within 30 rounds (" + d->counts() + ")");
    d->run(4000, cost, per_token);
    check(d->behind == 0 && d->by[kWarm] >= 2 && d->trees() >= SpecController::kProbeRounds * d->by[kWarm] &&
              probe_ms(*d) <= 2 * SpecController::kProbeShare * d->ms + 25,
          "  then plain decoding, probes at about " + std::to_string(SpecController::kProbeShare * 100) +
              "% of the time, each its warm-up and its trees (" + d->counts() + "; " +
              std::to_string(int(probe_ms(*d))) + " of " + std::to_string(int(d->ms)) + " ms), never a tree on a "
              "canvas that did not follow (" + std::to_string(d->behind) + ")");
    // They pay again: a probe finds it, and DV drafts again.
    d->accept = [](int width, size_t) { return width == 15 ? 5 : 6; };
    d->run(0, cost, per_token);  // the counts start over
    int found = -1;
    for (int k = 0; k < 1000 && found < 0; ++k) {
      d->step(cost, per_token);
      if (d->trees() > 0) found = k;
    }
    d->run(50, cost, per_token);
    check(found >= 0 && d->trees() >= 48 && d->behind == 0,
          "trees that pay again: found by a probe within 1000 rounds (" + std::to_string(found) +
              "), then DV every round (" + d->counts() + ")");
    // A cut (a rollback, a new prompt) forgets the canvas: the next DV round is a warm-up.
    d->c.cut(d->text.size() - 2);
    d->text.resize(d->text.size() - 2);
    d->follows = false;
    const RoundChoice after_cut = d->c.next(d->text, 0);
    check(!after_cut.lookup && after_cut.index == kWarm, "after a cut: DV's warm-up first (" +
                                                             std::to_string(after_cut.index) + ")");
    // A lookup round (one of several sessions' the scheduler took) leaves the canvas behind as any other round does:
    // the next DV round is a warm-up, not a tree whose drafts would count as the canvas's.
    {
      auto k = make(300, false);
      k->run(20, cost, per_token);
      const bool drafting = k->trees() >= 19 && k->behind == 0;
      k->c.commit({true, 0, kLookupWidths[0], -1}, 0);
      k->follows = false;
      const RoundChoice after_lookup = k->c.next(k->text, 0);
      check(drafting && !after_lookup.lookup && after_lookup.index == kWarm,
            "after a lookup round: DV's warm-up first (" + std::to_string(after_lookup.index) + "; before it " +
                k->counts() + ")");
    }
    // While DV drafts no round is looked up, even where the text repeats; once plain decoding drafts, lookup may.
    auto l = make(100);
    l->run(30, cost, per_token);
    std::vector<int32_t> rep(l->text.end() - 20, l->text.end() - 12);
    append(&l->text, rep);
    const RoundChoice dv_on = l->c.next(l->text, 0);
    check(!dv_on.lookup && dv_on.drafter == 0, "text that repeats while DV drafts: DV's round, not a lookup round");
    l->c.result(dv_on, true, 5, l->text, 0, true);
    // Wider trees that pay: rounds of 31 once its turns have taught the engine's mean that they give more tokens a
    // millisecond, 15 then drafting one round in kWidthTurn.
    auto w = make(100);
    w->accept = [](int width, size_t) { return width == 15 ? 4 : 14; };
    w->run(3 * SpecController::kWidthTurn, cost, per_token);
    w->run(100, cost, per_token);
    check(w->by[k31] >= 97, "trees of 31 accepting 14 against 15's 4: rounds of 31 (" + w->counts() + ")");
    // At a long context (3110 tokens) DV's rounds cost more (attention of 32 or 64 rows over every key): a reply there
    // does not enter DV on the guess of its tokens (4.0), nor probe it before the engine's credit (0 at load) covers
    // a probe.
    {
      auto x = make(3110);
      x->accept = [](int, size_t) { return 0; };
      x->run(200, cost, per_token);
      check(x->trees() + x->by[kWarm] == 0,
            "after a prompt of 3110 tokens: no DV round in the first 200 (" + x->counts() + ")");
      // The context growing past it (the credit's bound grows with it): probes come, each its warm-up and its
      // trees, at about kProbeShare of the time; where DV pays there (8 tokens a round), a probe finds it, its trees
      // going on while they pay, and DV stays.
      x->run(3300, cost, per_token);
      const int probes = x->by[kWarm];
      check(probes >= 2 && x->trees() >= SpecController::kProbeRounds * probes &&
                probe_ms(*x) <= 2 * SpecController::kProbeShare * x->ms + 40,
            "  the context growing to " + std::to_string(x->text.size()) + " tokens: probes go on (" + x->counts() +
                "; " + std::to_string(int(probe_ms(*x))) + " of " + std::to_string(int(x->ms)) + " ms)");
      x->accept = [](int, size_t) { return 7; };
      x->run(0, cost, per_token);
      int back = -1;
      for (int k = 0; k < 1500 && back < 0; ++k) {
        x->step(cost, per_token);
        if (x->trees() > SpecController::kProbeMax) back = k;  // past a probe's trees: DV stays
      }
      x->run(100, cost, per_token);
      check(back >= 0 && x->trees() >= 95, "  where DV pays there: found by a probe within 1500 rounds (" +
                                               std::to_string(back) + "), then DV drafts (" + x->counts() + ")");
    }
    // An engine whose replies of one kind DV did not pay on (24 of 300 rounds, its trees accepting nothing), then
    // replies it pays on (5 drafts a round): the engine's credit probes one of them, and once its rounds have raised
    // the engine's means, a reply enters DV at once.
    {
      SharedTokens engine;
      engine.resize(widths.size());
      int dv_before = 0;
      for (int q = 0; q < 24; ++q) {
        auto x = make(100, true, &engine);
        x->accept = [](int, size_t) { return 0; };
        x->run(300, cost, per_token);
        dv_before += x->trees();
      }
      int first_dv = -1, entered_at_once = -1;
      double last_share = 0;
      for (int q = 0; q < 6; ++q) {
        auto x = make(100, true, &engine);
        const RoundChoice r0 = x->c.next(x->text, 0);  // the reply's first round, as the engine would launch it
        x->c.result(r0, true, 0, x->text, 0, true);
        x->follows = r0.drafter >= 0;
        x->text.push_back(x->next_token++);
        x->run(300, cost, per_token);
        if (first_dv < 0 && x->trees() > 0) first_dv = q;
        if (entered_at_once < 0 && r0.index == kWarm) entered_at_once = q;
        last_share = double(x->trees()) / 300;
      }
      check(first_dv >= 0 && first_dv <= 3 && entered_at_once >= 0 && last_share >= 0.9,
            "after 24 replies DV did not pay on (" + std::to_string(dv_before) + " trees, probes), replies it pays "
                "on: DV drafts in reply " + std::to_string(first_dv) + ", one enters it at once (" +
                std::to_string(entered_at_once) + "), the last drafts it in " + std::to_string(int(last_share * 100)) +
                "% of its rounds");
    }
    // An engine whose sessions DV pays on (its means 5 tokens a round of 15): a reply whose first 14 trees give 2
    // tokens each (some of GSM8K's do) stays in DV through them (its level counts the engine's means as kTreePrior
    // rounds) and drafts the reply; and in a session whose reply left DV, the next reply starts from the engine's
    // means again (enters at once), its level not carried over.
    {
      SharedTokens engine;
      engine.resize(widths.size());
      for (int k = 0; k < 400; ++k) engine.add(k15, 5.0), engine.add(k31, 5.3);
      auto x = make(100, true, &engine);
      int dv = 0;
      x->accept = [&dv](int, size_t) { return ++dv <= 14 ? 1 : 4; };
      x->run(150, cost, per_token);
      check(x->by[kWarm] == 1 && x->trees() == 149, "trees that give 2 tokens for 14 rounds, then 5, the engine's "
                                                    "means 5: DV all through (" + x->counts() + ")");
      x->accept = [](int, size_t) { return 0; };
      x->run(60, cost, per_token);
      const bool left = x->by[kPlain] > 0;
      x->c.begin(true);  // the session's next reply
      x->follows = false;
      x->accept = [](int, size_t) { return 4; };
      const RoundChoice r0 = x->c.next(x->text, 0);
      check(left && r0.index == kWarm, "  a reply that left DV, then the session's next reply: its warm-up first (" +
                                           std::to_string(r0.index) + ")");
    }
    // Tree rounds whose tokens come in streaks (1 to 13 tokens, of 15 a round of 31 one more): rounds of 15, the
    // width the engine's means give more tokens a millisecond; 31 only on its turns, not when 15 has had a poor
    // streak.
    {
      auto x = make(100, false);
      uint32_t seed = 12345;
      int streak = 0, depth = 0;
      x->accept = [&](int width, size_t) {
        if (streak-- <= 0) {
          seed = seed * 1103515245u + 12345u;
          streak = 3 + int(seed >> 16) % 8;
          depth = int(seed >> 8) % 13;
        }
        return depth + (width == 31 ? 1 : 0);
      };
      x->run(2000, cost, per_token);
      check(x->by[k31] <= x->trees() / SpecController::kWidthTurn + 3 && x->trees() > 1500,
            "tree rounds in streaks, 31 a token more than 15: 31 on its turns alone (" + x->counts() + ")");
    }
    // After a prompt of 1970 tokens (a few-shot chat), the engine's means there 3.9 tokens a round of 15: DV pays
    // (3.9 in 15.8 ms against 4.6 ms a token) and is entered, its warm-up paid over kPayBack rounds.
    {
      SharedTokens engine;
      engine.resize(widths.size());
      for (int k = 0; k < 400; ++k) engine.add(k15, 3.9), engine.add(k31, 4.1);
      auto x = make(1970, true, &engine);
      int n = 0;
      x->accept = [&n](int, size_t) { return ++n % 10 < 9 ? 3 : 2; };
      x->run(300, cost, per_token);
      check(x->by[kWarm] >= 1 && x->trees() >= 240, "after 1970 tokens, DV's means 3.9 tokens: DV entered and "
                                                    "drafting (" + x->counts() + ")");
    }
    // Rounds of several sessions (the scheduler's): DV's rounds run alone, so while the session's rounds are batched
    // it leaves and does not draft; alone again, its warm-up and its trees.
    {
      auto x = make(300, false);
      x->run(20, cost, per_token);
      const int before = x->trees();
      x->c.batched(true);
      x->run(100, cost, per_token);
      const int batched = x->trees() + x->by[kWarm];
      x->c.batched(false);
      x->run(0, cost, per_token);
      x->step(cost, per_token);
      const bool warmed = x->by[kWarm] == 1;
      x->run(50, cost, per_token);
      check(before >= 18 && batched == 0 && warmed && x->trees() >= 48 && x->behind == 0,
            "batched with other sessions: no DV round (" + std::to_string(batched) + "); alone again, its warm-up, "
                "then its trees (" + x->counts() + ")");
    }
    // Two rounds in flight, as a generation runs them, on a text that repeats (lookup would draft it), DV's trees at
    // the margin (2 and 4 tokens in turn): when the round read says plain decoding next and the text repeats, the
    // chooser waits for the round in flight to look the next one up; when that round brings DV back, its rounds go
    // out two in flight again, never one at a time for a lookup round that is not launched.
    {
      RoundChooser c;
      c.configure(&widths, &drafters, &cost, &kLookupWidths, &trunks, nullptr, nullptr, &per_token, 64);
      c.begin(true);
      std::vector<int32_t> text;
      for (int i = 0; i < 200; ++i) text.push_back(1000 + i % 40);  // a prompt whose text repeats (period 40)
      int32_t next = 1000 + 200 % 40;
      std::vector<RoundChoice> flight;
      bool follows = false;
      int dv = 0, alone = 0, beside = 0;
      for (int step = 0; step < 600; ++step) {
        while (static_cast<int>(flight.size()) < c.in_flight_limit()) {
          const RoundChoice r = c.next(text, static_cast<int>(flight.size()));
          if (!r.lookup && r.drafter == 0 && r.width > 0 && flight.empty())
            (c.in_flight_limit() == 1 ? alone : beside) += 1;
          flight.push_back(r);
        }
        const RoundChoice r = flight.front();
        flight.erase(flight.begin());
        int accepted = 0;
        if (!r.lookup && r.drafter == 0 && r.width > 0) accepted = follows ? (++dv % 2 ? 1 : 3) : 0;
        follows = !r.lookup && r.drafter == 0;
        for (int k = 0; k <= accepted; ++k) text.push_back(next), next = 1000 + (next - 1000 + 1) % 40;
        c.result(r, true, accepted, text, static_cast<int>(flight.size()), true);
      }
      check(dv > 20 && alone == 0, "two in flight on a text that repeats, DV at the margin (" + std::to_string(dv) +
                                        " trees): DV rounds launched one at a time for a lookup round never "
                                        "launched: " + std::to_string(alone) + " of " +
                                        std::to_string(alone + beside));
    }
    // Pinned (DV's rounds alone, no warm-up round): every round DV's.
    {
      const std::vector<int> one = {31}, dv_only = {0}, no_trunk = {-1};
      const std::vector<double> cost31 = {15.25};
      RoundChooser c;
      c.configure(&one, &dv_only, &cost31, nullptr, &no_trunk);
      c.begin(false);
      const std::vector<int32_t> t = range(1, 50);
      bool all = true;
      for (int k = 0; k < 40; ++k) {
        const RoundChoice r = c.next(t, 0);
        all = all && r.index == 0;
        c.result(r, true, 0, t, 0, true);
      }
      check(all, "DV pinned: its rounds alone, whatever they accept");
    }
  }

  std::printf("round_chooser: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
