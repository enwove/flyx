// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What a model gives the engine: the description a backend
// runs (weights, per-sequence state, typed blocks; backend_abi.h), its
// drafters with the round widths each is worth, and the programs.  Core sees
// only this interface: drafters by number, their names for reports only.
// models/ implements it and opens a file as one (open_model, defined in
// models/model.cpp), so core includes nothing from models/.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/vision.h"
#include "flyx/backend_abi.h"

namespace flyx {
class ModelFile;
}

namespace flyx::interfaces {

struct ModelOptions {
  std::string drafter;      // kDrafterModes' words (options.h), or "" for the model's default set
  bool speculative = true;  // false: no drafter
  // "" = as each weight's file has it; "exact" = the formats' own rounding everywhere; "nvfp4" = NVFP4 weights take
  // NVFP4 inputs in passes of more rows whatever the file says, other weights as the file has it.
  std::string activations;
  // The device computes NVFP4 inputs (flyx_device_info::fp4_inputs).  False: NVFP4 weights take bf16 inputs whatever
  // the file says (finer), other weights as the file has it; with "nvfp4" NVFP4 ones, which the
  // backend refuses.
  bool fp4_inputs = true;
  bool vision = true;       // false: skip loading any vision encoder / mmproj companion (--no-vision)
};

class Model {
 public:
  virtual ~Model() = default;
  virtual const flyx_model_desc& desc() const = 0;
  // The drafters the model holds, numbered 0.. drafters - 1.
  virtual int drafters() const = 0;
  // Drafter d's name, and all of them ("mtp+dflash2"; "none" without one): for logs and reports.
  virtual const char* drafter_name(int d) const = 0;
  virtual const char* drafter_title(int d) const { return drafter_name(d); }
  // Single-drafter subsets to try when the active configuration runs out of memory.
  virtual std::vector<std::string> drafter_fallbacks() const { return {}; }
  virtual const char* drafter() const = 0;
  // The draft counts a round of drafter d may verify, ascending; the last is
  // the most it is worth in one round.  The engine lowers a round plan for
  // each (and one for plain decoding) and chooses among them every round by
  // their measured cost and the acceptance each drafter has had.
  virtual std::vector<int> draft_widths(int d) const = 0;
  // Sorted verify checkpoints (subset of draft_widths(d)) for an intra-round
  // adaptive stopping plan on drafter d; empty disables adaptive plans for d.
  virtual std::vector<int> adaptive_checkpoints(int /*d*/) const { return {}; }
  // Whether one more draft of drafter d costs little beside the verify row it adds (a block drafted at once and
  // refined a draft at a time): an adaptive round that stops may then verify fewer drafts than it has
  // (flyx_program::earlier_checkpoints).  Where a draft step is a pass of a layer, what is drafted is verified.
  virtual bool draft_steps_cheap(int /*d*/) const { return false; }
  // The drafts one pass of drafter d's block makes as it was trained (DFlash2: its block less one; MTP: its widest
  // chain): a tree round's trunk.  draft_widths may go past it (a DFlash2 block drafted wider).
  virtual int block_drafts(int d) const = 0;
  // The context the model was made for, as its file says (a GGUF's <arch>.context_length, a .flyx config's max_pos);
  // 0: the file does not say.  --ctx max goes no further.
  virtual int trained_ctx() const = 0;
  // Drafter d's rounds are always a tree of its drafts: its round programs say so themselves (FLYX_DRAFT_TREE), and
  // no chain of its drafts is a trunk the tree grows from (DV's).
  virtual bool drafts_trees(int) const { return false; }
  // Drafter d's trees beside its chains are of FLYX_MAX_ROWS - 1 drafts only (EAGLE-4's: its chain of 7 and the
  // branches beside it), none narrower.
  virtual bool widest_trees(int) const { return false; }
  // The target file requires exact reads even in plain decoding (embedded DV).
  virtual bool file_drafts_trees() const { return false; }
  // Draft trees (FLYX_DRAFT_TREE rounds beside the chains, the chooser taking them where they pay) unless the caller
  // says otherwise: a chosen drafter's whose trees verify more tokens a round than its chains on the targets it is
  // made for (DSpark's, EAGLE-4's).
  virtual bool draft_trees_by_default() const { return false; }
  // Drafter d keeps a state that follows its own rounds only (DV's canvas): a round of another kind, a prefill, a
  // rewind or a restore leaves it behind, and its next round drafts from a state that starts over (DV's first tree,
  // its star of MASKs, which commits one token).  round_program(0, d, false) is then its warm-up: plain decoding that
  // brings the state along (DV: the round's canvas pass), which the engine runs before the drafter's first round
  // after any other (core/speculation/controller.h).
  virtual bool restarts(int) const { return false; }
  // The target's K/V caches in FP8 when the caller does not say: only where a real model of the family showed no
  // visible drop against binary16 (AGENTS.md rule 5), binary16 otherwise (Llama).
  virtual bool kv_fp8_by_default() const = 0;

  // Prompt processing: every layer, every drafter's taps and context, the head on the last row.
  virtual std::vector<flyx_node> prefill_program() const = 0;
  // One round verifying `width` drafts (0 = plain decoding, one token per
  // round): drafter `d`'s, or, when `given`, the launch's (FLYX_NODE_GIVEN).
  // Every drafter takes the accepted rows as context, so the next round may
  // draft with any of them, but one whose state follows its own rounds only
  // (restarts): its round of width 0 is its warm-up.
  virtual std::vector<flyx_node> round_program(int width, int d, bool given) const = 0;

  // Optional multimodal vision encoder (null / empty for text-only models or when ModelOptions::vision == false).
  virtual const VisionAdapter* vision() const { return nullptr; }
  // Shared handle to the lightweight VisionAdapter metadata (survives idle engine unload in Scheduler::count_tokens).
  virtual std::shared_ptr<const VisionAdapter> vision_shared() const { return nullptr; }
  // Vision encoder execution graph (FLYX_PHASE_VISION): stem -> ViT layers -> projector/merger.
  virtual std::vector<flyx_node> vision_program() const { return {}; }
};

// The model in a file, by its architecture, with the drafters asked for.
// Null with *err set if the file cannot be read as a model this build knows.
std::unique_ptr<Model> open_model(const ModelFile& file, const ModelOptions& o, std::string* err);
// The architectures open_model knows, as model files name theirs, whose GGUF files it reads (a Llama: .flyx only):
// for saying before a file is fetched whether this build runs it.
std::vector<std::string> model_architectures();
// The architectures open_model opens with a vision projector (--mmproj): for deciding before fetching a
// repository's mmproj companion whether this build reads vision for the model.
std::vector<std::string> vision_architectures();

}  // namespace flyx::interfaces
