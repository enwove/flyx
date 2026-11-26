// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// spec::costs_plausible (core/speculation/calibrate.h): round costs timed at load at two contexts that a device can
// take, and those another program's work on the machine made up, which the load times again.  The costs are
// MiniCPM5-1B's DV rounds (plain decoding, DV's trees of 15 and 31, its warm-up) on an RTX 5060 Ti, at 64 and 1024
// tokens, on a quiet machine and while every core ran other work.  No model.
#include <cstdio>
#include <string>
#include <vector>

#include "core/speculation/calibrate.h"
#include "tests/common/check.h"

using flyx::spec::costs_plausible;

int main() {
  flyx::test::print_passes = true;
  using V = std::vector<double>;
  // Plain decoding first, as the plans have it.
  check(costs_plausible(V{4.35, 13.13, 15.42, 11.77}, V{4.41, 14.50, 17.38, 12.77}, 0), "a quiet machine's costs");
  check(costs_plausible(V{4.40, 13.10, 15.40, 11.80}, V{4.39, 14.50, 17.40, 12.80}, 0),
        "a round a little cheaper after the longer context, within the timings' spread");
  check(!costs_plausible(V{4.50, 13.06, 15.32, 11.91}, V{14.55, 14.41, 17.53, 12.78}, 0),
        "plain decoding's round dearer than DV's at 1024 tokens (one disturbed timing)");
  check(!costs_plausible(V{31.71, 45.45, 47.50, 58.03}, V{24.95, 22.64, 19.94, 22.20}, 0),
        "every round cheaper after the longer context (a build on every core)");
  check(!costs_plausible(V{4.41, 13.67, 16.49, 13.45}, V{5.65, 14.77, 18.17, 13.02}, 0),
        "the warm-up 3% cheaper after the longer context");
  check(!costs_plausible(V{4.35, 13.13}, V{4.41}, 0) && !costs_plausible(V{4.35}, V{4.41}, 1),
        "costs that do not pair up, or no plain decoding's round among them");
  return flyx::test::finish("calibrate_test");
}
