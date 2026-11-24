// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// From programs to plans: the model definition's programs, lowered by the
// backend (which picks the kernels) into the plans one engine runs.
#pragma once

#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "flyx/backend_abi.h"

namespace flyx::plan {

struct Plans {
  flyx_plan* prefill = nullptr;   // prompt rows, prefill_chunk at a time
  flyx_plan* vision = nullptr;    // optional vision encoder (FLYX_PHASE_VISION)
  // The rounds to choose from: plain decoding, then each drafter's widths, ascending (a drafter that drafts nothing
  // but trees: those the backend lowers, then its warm-up, a round of 0 drafts of it, Model::restarts), then the
  // trees its drafter drafts (draft_tree_widths(), where the backend has them).
  std::vector<int> widths;           // drafts a round verifies (0 = plain decoding, or a drafter's warm-up)
  std::vector<int> drafters;         // the drafter that drafts them (-1: none)
  std::vector<bool> drafter_trees;   // the drafter drafts them as a tree (FLYX_DRAFT_TREE)
  std::vector<flyx_plan*> rounds; // one per round
  // Rounds whose drafts the engine gives (looked up in the text):
  // no drafter runs, so they can be wider than the drafter's.
  std::vector<int> given_widths;     // ascending
  std::vector<flyx_plan*> given;  // one per given width
  // Rounds that verify a tree of given drafts (gen.parents; docs/design/tree-verification.md), where the backend
  // takes them: tree_widths()'s.
  std::vector<int> tree_widths;      // ascending
  std::vector<flyx_plan*> trees;  // one per tree width
  // Optional intra-round adaptive confidence-stopping plan per drafter (indexed by drafter d; null if unsupported).
  std::vector<flyx_plan*> adaptive;
  std::vector<std::vector<int>> adaptive_checkpoints;
  // A tree drafter's widths (and warm-up) the backend refused, each with why ("dv 31: why; dv 15: why"); empty: none.
  std::string refused;
  // That drafter when the backend lowered none of its widths (the rounds are then the ones without it); else -1.
  int refused_drafter = -1;

  int widest() const;  // the most drafts a round verifies
  void free(const flyx_backend_api* api);
};

// The draft counts of rounds whose drafts are given: up to a full round. Solo
// picks the widest that the match fills (15 on long copies);
// narrower widths let a lookup round share a batch with other running sessions
// or use shorter matches instead of falling back to a drafter.
inline const std::vector<int>& given_widths() {
  static const std::vector<int> w = {3, 7, FLYX_MAX_ROWS - 1};
  return w;
}

// Wider rounds of given drafts (a long copy verified at once), where the backend lowers them (FLYX_MAX_WIDE_ROWS):
// the chooser widens to them only after full rounds of FLYX_MAX_ROWS - 1 (RoundChooser).
inline const std::vector<int>& wide_given_widths() {
  static const std::vector<int> w = {31, 47, FLYX_MAX_WIDE_ROWS - 1};
  return w;
}

// The drafts a tree round may verify (its nodes; rows past the tree a caller gives continue the last row's chain).
// None wider: the given trees of more than FLYX_MAX_ROWS rows (on a backend's wide rows: the cuda option wide) are
// lowered by those that run them (golden_test, tree_bench), not by every engine.
inline const std::vector<int>& tree_widths() {
  static const std::vector<int> w = {3, 7, FLYX_MAX_ROWS - 1};
  return w;
}

// The drafts of a round whose drafter drafts them as a tree (EngineOptions::draft_trees): the drafter's chain and 2,
// 4 or 8 branches (DFlash2: its chain of 7; ops/draft/ref.h).
inline const std::vector<int>& draft_tree_widths() {
  static const std::vector<int> w = {9, 11, FLYX_MAX_ROWS - 1};
  return w;
}

// The prefill plan and the round plans: plain decoding and, for every drafter,
// the draft counts the model names for it (Model::draft_widths), or with
// `fixed_width` >= 0 only that many drafts (at most a drafter's widest; 0:
// plain decoding alone); with `draft_trees` (and no fixed width), every drafter's trees of draft_tree_widths() the
// backend lowers (a drafter that drafts no trees has none); with `given`, also the rounds of given_widths(); and the
// tree rounds of tree_widths() where the backend has them (a backend that refuses them gets none).  A drafter that
// drafts nothing but trees (Model::drafts_trees: DV) has the widths the backend lowers, marked trees, and when its
// state follows its own rounds only (Model::restarts) its warm-up (round_program(0, d)); with `fixed_width`, the
// widest of them, at most that many drafts, the backend lowers, alone (no plain or lookup round beside it: DV's
// rounds every round, as its references draft).  Where the backend lowers none (or not its warm-up), the rounds are
// the ones without it (`refused_drafter`); `refused` says why of every width refused.  A refusal for want of GPU
// memory fails the build instead (the load gives memory back and builds again).
bool build(const flyx_backend_api* api, flyx_bmodel* bm, const interfaces::Model& model, int prefill_chunk,
           int fixed_width, bool given, bool draft_trees, Plans* out, std::string* err, bool adaptive = true);

}  // namespace flyx::plan
