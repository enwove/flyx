// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// RoundCosts (core/speculation/round_costs.h): what a round costs at a context, the load's timings first, then the
// rounds read; and the controller choosing by it: a drafter's rounds that grow dear at a long context give way to
// plain decoding there, and win again at a short one.  No model.
#include <cmath>
#include <string>
#include <vector>

#include "core/speculation/controller.h"
#include "core/speculation/round_costs.h"
#include "tests/common/check.h"

using namespace flyx;

namespace {

bool near(double got, double want, double rel) { return std::fabs(got - want) <= rel * want; }
std::string ms(double v) { return std::to_string(v) + " ms"; }

void the_load_first() {
  RoundCosts c;
  c.reset({10.0, 12.0}, nullptr, 64);
  check(c.size() == 2 && c.cost(0, 64) == 10.0 && c.cost(0, 35000) == 10.0 && c.cost(1, 35000) == 12.0,
        "before any round is read: the load's cost at every context (" + ms(c.cost(1, 35000)) + ")");
  const std::vector<double> per_token = {0.0001, 0.0};
  c.reset({10.0, 12.0}, &per_token, 64);
  check(near(c.cost(0, 1064), 10.1, 1e-9) && c.cost(1, 1064) == 12.0,
        "a cost per token timed at load: the load's line (" + ms(c.cost(0, 1064)) + ")");
}

void learned_at_the_context() {
  RoundCosts c;
  c.reset({10.0, 12.0}, nullptr, 64);
  for (int k = 0; k < 40; ++k) c.observe(1, 35000, 30.0);
  check(near(c.cost(1, 35000), 30.0, 0.05), "rounds read at 35000 tokens: their cost there (" +
                                                 ms(c.cost(1, 35000)) + ")");
  check(near(c.cost(1, 100), 12.0, 0.10), "  at a short context still the load's (" + ms(c.cost(1, 100)) + ")");
  check(c.cost(0, 35000) == 10.0, "  another round's cost unmoved (" + ms(c.cost(0, 35000)) + ")");
  for (int k = 0; k < 150; ++k) c.observe(1, 35000, 14.0);
  check(near(c.cost(1, 35000), 14.0, 0.10), "  dearer rounds forgotten as cheaper ones are read (" +
                                                 ms(c.cost(1, 35000)) + ")");
}

void a_stall_moves_it_little() {
  RoundCosts c;
  c.reset({10.0}, nullptr, 64);
  c.observe(0, 64, 1000.0);
  check(c.cost(0, 64) < 12.0, "one round 100 times its cost (a stall): the cost moves a little (" +
                                  ms(c.cost(0, 64)) + ")");
}

// A round flat to the GPU's part of the K/V and dear past it: one line through both keeps its cost at a short context
// no lower than half the load's, and the short context's rounds bring it back.
void a_knee_keeps_the_short_cost() {
  RoundCosts c;
  c.reset({13.0}, nullptr, 64);
  for (int k = 0; k < 30; ++k) c.observe(0, 16000, 13.0);
  for (int k = 0; k < 30; ++k) c.observe(0, 35000, 45.0);
  check(c.cost(0, 64) >= 6.5, "rounds read either side of a knee: the cost at a short context no lower than half the "
                              "load's (" + ms(c.cost(0, 64)) + ")");
  for (int k = 0; k < 60; ++k) c.observe(0, 64, 13.0);
  check(near(c.cost(0, 64), 13.0, 0.25), "  and the short context's rounds bring it back (" + ms(c.cost(0, 64)) + ")");
}

// A drafter's chains of 1 and 3 drafts: an adaptive round that stops at one draft teaches the narrow chain only; the
// wide one costs no less than it, and the load's difference more.
void a_wider_chain_no_cheaper() {
  RoundCosts c;
  const std::vector<int> narrower = {-1, -1, 1};
  c.reset({10.0, 11.0, 12.0}, nullptr, 64, &narrower);
  for (int k = 0; k < 40; ++k) c.observe(1, 35000, 40.0);
  check(c.cost(2, 35000) >= c.cost(1, 35000) + 1.0 && near(c.cost(2, 64), 12.0, 0.01),
        "the chain of 3 drafts at no less than the chain of 1 (" + ms(c.cost(2, 35000)) + " against " +
            ms(c.cost(1, 35000)) + "), about the load's at a short context (" + ms(c.cost(2, 64)) + ")");
  check(c.cost(0, 35000) == 10.0, "  plain decoding no chain of the drafter's");
}

// A round the load did not time (0 ms, or not finite): priced at the cheapest the load timed until read (1 ms where
// it timed none; never nothing, which a rate would divide by), then at what it was read at.
void untimed_rounds() {
  RoundCosts timed;
  timed.reset({5.0, 0.0, 8.0}, nullptr, 64);
  check(timed.cost(1, 64) == 5.0, "an untimed round: the cheapest timed one's cost until read (" +
                                      ms(timed.cost(1, 64)) + ")");
  RoundCosts c;
  c.reset({0.0, std::nan("")}, nullptr, 64);
  check(c.cost(0, 64) == 1.0 && c.cost(1, 64) == 1.0, "none timed: 1 ms until read");
  for (int k = 0; k < 20; ++k) {
    c.observe(0, 64, 7.0);
    c.observe(1, 64, 9.0);
  }
  check(near(c.cost(0, 64), 7.0, 0.01) && near(c.cost(1, 64), 9.0, 0.01),
        "  then what they were read at (" + ms(c.cost(0, 64)) + ", " + ms(c.cost(1, 64)) + ")");
}

// Plain decoding at 12 ms at 35000 tokens, the drafter's chain of 1 draft at 40 (its adaptive rounds stopping there),
// its chain of 3 never read: plain decoding there, not the chain of 3 at its load's cost.
void the_controller_leaves_a_stopping_drafter() {
  const std::vector<int> widths = {0, 1, 3}, drafters = {-1, 0, 0}, narrower = {-1, -1, 1};
  const std::vector<double> load = {10.0, 11.0, 12.0};
  RoundCosts costs;
  costs.reset(load, nullptr, 64, &narrower);
  SpecController spec;
  spec.reset(&widths, &drafters, &load, nullptr, nullptr, 0, nullptr, &costs);
  spec.commit(2, 100);
  for (int k = 0; k < 64; ++k) spec.observe(2, k % 2 ? 3 : 0);  // half the first drafts accepted, then all
  for (int k = 0; k < 40; ++k) {
    costs.observe(1, 35000, 40.0);
    costs.observe(0, 35000, 12.0);
  }
  check(spec.preview(35000) == 0, "the chain of 1 at 40 ms at 35000 tokens, plain decoding at 12: plain decoding, "
                                  "not the chain of 3 (" + ms(spec.cost(2, 35000)) + ")");
  check(spec.preview(100) == 2, "  the chain of 3 at 100 tokens (" + ms(spec.cost(2, 100)) + ")");
}

// Plain decoding costs 10 ms and the drafter's round of 3 drafts 12 at load, the drafter accepting nearly every
// draft: its rounds win.  At 35000 tokens its rounds take 60 ms and plain decoding's 12: plain decoding wins there,
// the drafter still at 100 tokens.
void the_controller_follows() {
  const std::vector<int> widths = {0, 3}, drafters = {-1, 0};
  const std::vector<double> load = {10.0, 12.0};
  RoundCosts costs;
  costs.reset(load, nullptr, 64);
  SpecController spec;
  spec.reset(&widths, &drafters, &load, nullptr, nullptr, 0, nullptr, &costs);
  spec.commit(1, 100);  // tried: no longer the drafter every choice tries first
  for (int k = 0; k < 64; ++k) spec.observe(1, 3);
  check(spec.preview(35000) == 1 && spec.preview(100) == 1, "the load's costs: the drafter's rounds at any context");
  for (int k = 0; k < 40; ++k) {
    costs.observe(1, 35000, 60.0);
    costs.observe(0, 35000, 12.0);
  }
  check(spec.preview(35000) == 0, "its rounds read at 60 ms at 35000 tokens, plain decoding's at 12: plain decoding "
                                  "there (" + ms(spec.cost(1, 35000)) + " against " + ms(spec.cost(0, 35000)) + ")");
  check(spec.preview(100) == 1, "  the drafter's rounds at 100 tokens (" + ms(spec.cost(1, 100)) + ")");
  SpecController fixed;
  fixed.reset(&widths, &drafters, &load);
  fixed.commit(1, 100);
  for (int k = 0; k < 64; ++k) fixed.observe(1, 3);
  check(fixed.preview(35000) == 1, "without a cost model: the load's costs throughout, the drafter's rounds");
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  the_load_first();
  learned_at_the_context();
  a_stall_moves_it_little();
  the_controller_follows();
  a_knee_keeps_the_short_cost();
  a_wider_chain_no_cheaper();
  untimed_rounds();
  the_controller_leaves_a_stopping_drafter();
  return flyx::test::finish("round_costs_test");
}
