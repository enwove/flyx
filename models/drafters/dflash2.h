// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Architecture-agnostic DFlash2 speculative drafter parser and block builder; z-lab's DFlash (kind "dflash": the same
// block drafter without its convolutions or selector, each row's draft the head's best); and
// DeepSpec's DSpark (kind "dspark": the same block drafter without its convolutions, every row of the block a draft,
// the anchor's too, and a Markov head as the selector without a hidden gate).  DFlash and DSpark: from a .flyx or an
// authors' directory.
// Supports three sources uniformly:
//   * embedded in a .flyx target model header ("draft"),
//   * a standalone .flyx drafter file or Hugging Face safetensors directory ("kind": "drafter"),
//   * a GGUF drafter file with general.architecture = "dflash" (non-Markov DFlash2 layout).
// The blocks are built from the names python/convert gives the drafter's tensors ("draft.fc", "draft.L0.attn.q");
// a GGUF file's own names are mapped from those, as an architecture's are (models/gguf.h).
#pragma once

#include <string>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

struct DFlash2Spec {
  flyx_dflash2_block block{};
  int layers = 0;
  int tg2 = 0;
  int sel_vocab = 0;       // selector codebook row count (vocab or padded_vocab)
  double eps = 0.0;
  std::vector<int> taps;   // 1-based layer indices (1..target.layers; target.layers = post output_norm tap)
  bool gguf = false;       // the file's tensor names are llama.cpp's (arch "dflash")
  int companion = -1;      // -1: embedded in target file; >= 0: companion index
  Drafter kind = Drafter::kDFlash2;  // its config's kind (DFlash2, DSpark): the name messages give it
  bool sel_gate = true;              // the selector's hidden gate (draft.sel.hid), else a unit gate (sel_gate false)
  // The drafts one pass of its block makes as trained (DFlash2: block - 1; DSpark: block, the anchor's row too).
  int block_drafts() const { return block.block - block.draft_from; }
};

// The drafter of the target's taps a canonical config ("draft") says it is (its "kind": a block drafter or EAGLE-4),
// kNone when it is none.
Drafter block_drafter_kind(const nlohmann::json& draft);

// True if a companion header describes a block drafter of DFlash2's block or DSpine's (a drafter file's "draft"), or
// a non-Markov arch="dflash" GGUF.
bool is_block_companion(const nlohmann::json& header);

// Throws unless the drafter such a companion holds (or a DSpark one's: the same GGUF layout) was made for this
// target (its width and vocabulary, and its layer count where the drafter names one).  A drafter given beside
// another model is refused whether or not it is chosen; the rest of its config is read only when it drafts
// (parse_dflash2_*, parse_dspark_gguf, parse_dspine_json).
void check_made_for(const nlohmann::json& header, const TargetSpec& target);

// A drafter config's integer `key`: a whole number in int's range, else "invalid <drafter> drafter config".
int config_int(const nlohmann::json& j, const char* key, const std::string& drafter);

// A block drafter's Qwen3 layers and context as its canonical config gives them (DFlash2's block, DSpine's), read and
// checked against the target (`title` names the drafter in refusals): their dimensions, block and mask token, the
// target layers it taps after, rotary theta and norm epsilon.
struct Qwen3BlockConfig {
  int hidden = 0, layers = 0, n_heads = 0, n_kv = 0, head_dim = 0, inter = 0, block = 0, mask_token = 0;
  double theta = 0.0, eps = 0.0;
  std::vector<int> taps;  // 1-based (1..target.layers; target.layers: after the final norm)
};
Qwen3BlockConfig parse_qwen3_block(const nlohmann::json& j, const TargetSpec& target, const std::string& title);

// Throws unless a block drafter's canonical config ("draft") is of a kind this build reads and was made for this target
// where it says what it was made for (its width, vocabulary, padded or not, and layer count).
void check_config_target(const nlohmann::json& draft, const TargetSpec& target);

// Parse and validate a DFlash2 spec from canonical JSON ("draft" section in .flyx or drafter_dir); `file` holds its
// tensors (the selector's codebooks say whether they have the target's padding rows).
DFlash2Spec parse_dflash2_json(const ModelFile& file, const nlohmann::json& j, const TargetSpec& target,
                               int companion);

// Parse and validate a DFlash2 spec from a GGUF companion header with general.architecture == "dflash".
DFlash2Spec parse_dflash2_gguf(const ModelFile& file, const nlohmann::json& header, const TargetSpec& target,
                               int companion);

// Build FLYX_BLOCK_DFLASH2_LAYER blocks, draft KV state, and FLYX_BLOCK_DFLASH2 in `T`.
// Returns the FLYX_BLOCK_DFLASH2 block index.
int build_dflash2(Tables& T, const DFlash2Spec& spec, const TargetSpec& target);

}  // namespace flyx::models::drafters
