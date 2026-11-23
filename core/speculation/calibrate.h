// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The cost table the speculation controller chooses from: what one round of
// each kind costs on this device, measured at load.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/plan/plans.h"
#include "flyx/backend_abi.h"

namespace flyx::spec {

struct TimedGroup {
  const std::vector<flyx_plan*>* rounds;
  const std::vector<std::string>* labels;
  const char* what;
  std::vector<double>* out;
  // Each round timed this many times, the fastest kept: where a cost per token of context is the difference of two
  // timings, on a GPU other programs share, their work lands in one of them now and then.
  int repeats = 1;
};

// Milliseconds per round for each group of rounds (named by `labels`, e.g. "mtp 3") on one shared scratch sequence
// at the end of `prompt` (prefilled once in passes of `chunk` tokens, checkpointed, and restored between rounds):
// three rounds of each timed after an untimed one (graphs captured, modules loaded), the group's `repeats` times, with
// two rounds in flight as in real use.  Before the first timed round, runs a brief untimed burst so GPUs ramping from
// idle P-states reach steady-state clocks before width 0 is measured.  A round whose timing fails keeps 1.0.
// Verbose: printed as "<what> cost".
void time_rounds(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill, int chunk,
                 const std::vector<int32_t>& prompt, const std::vector<TimedGroup>& groups, int verbose);

// Whether costs of the same rounds timed at two contexts (`at_short`, `at_long`: ms a round, one each) are what a
// device takes, not what another program's work on the machine made them: plain decoding's round (`plain`: one row,
// nothing else) the cheapest at each context, and no round cheaper after the longer one (a round's attention reads
// every key), each within `tolerance` of the other round's cost for the timings' own spread.
bool costs_plausible(const std::vector<double>& at_short, const std::vector<double>& at_long, size_t plain,
                     double tolerance = 0.02);

// Each of `rounds` once on a scratch sequence that first prefills `prompt`, untimed: a kernel's first launch loads its
// module (CUDA loads them as first used), which would otherwise wait in a request's first rounds.
void warm_rounds(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill,
                 const std::vector<flyx_plan*>& rounds, const std::vector<int32_t>& prompt);

}  // namespace flyx::spec
