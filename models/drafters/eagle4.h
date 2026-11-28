// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// EAGLE-4 (backend_abi.h flyx_eagle4_block, python/reference/flyx_ref/eagle4.py) as a block of a target: one decoder
// layer run a step a draft (FLYX_BLOCK_EAGLE4), from a "draft" config of kind "eagle4"
// (python/convert/flyx_convert/eagle4.py) and its draft.* tensors, in the model file or a drafter file beside it.  It
// reads the target's embedding and the target's residual stream at its taps; its head is its own.  Every
// architecture that taps its layers takes it with no backend change.
#pragma once

#include <vector>

#include "flyx/backend_abi.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct Eagle4Spec {
  flyx_eagle4_block block{};
  double eps = 0.0;
  int rows = 0;           // its head's rows (the target's vocabulary, padded or not)
  std::vector<int> taps;  // the residual stream after that many target layers (1..target.layers - 1)
  int companion = -1;     // -1: in the target's file; >= 0: the companion file holding it
};

// A "draft" config of kind "eagle4", checked against the target: throws on one that does not fit, before any tensor
// is bound.
Eagle4Spec parse_eagle4_json(const nlohmann::json& d, const TargetSpec& target, int companion);

// Adds the drafter's state (its K/V and its ring of h) and its block to the tables; returns the block.
int build_eagle4(Tables& T, const Eagle4Spec& spec, const TargetSpec& target);

}  // namespace flyx::models::drafters
