// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV (bifill-dv's drafter, docs/design/dv.md) as blocks of a Llama target: a canvas of slots after the anchor run by
// the target's own attention and MLP blocks with the canvas's norms and adapters (FLYX_BLOCK_DV_LAYER), the drafter
// itself (FLYX_BLOCK_DV: embedding and head, the canvas's final norm, MASK embedding, residual context and selector)
// and its canvas (FLYX_STATE_CANVAS), from a .flyx file's "dv" section and dv.* tensors (python/convert --dv).
#pragma once

#include <vector>

#include "models/model.h"

namespace flyx::models::dv {

// The file has a DV drafter.
bool in_file(const ModelFile& file);

// The target the canvas runs: its sizes and its blocks (per layer an attention block and an MLP block, in order).
struct Target {
  int hidden = 0, vocab = 0, n_heads = 0, n_kv = 0, head_dim = 0, inter = 0;
  double eps = 0;
  int embed = -1, head = -1;
  std::vector<int> attn, mlp;
};

// Adds the canvas state, a DV_LAYER block per target layer and the DV block to the tables, from the file's "dv"
// section; returns the DV block.  Throws on a section or a tensor that does not fit the target.
int describe(Tables& T, const ModelFile& file, const Target& target);

// DV's round of `width` drafts around the verify pass `verify` (embedding, layers, head): its tree first, the
// target's check of it, then the canvas pass on what was accepted.  Of width 0, DV's warm-up: plain decoding, then
// the canvas pass (a canvas that followed no round starts over there, as after DV's first tree: the same canvas,
// for a round of one row instead of the tree's).
std::vector<flyx_node> round_program(int block, int width, const std::vector<flyx_node>& verify);

}  // namespace flyx::models::dv
