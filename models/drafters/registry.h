// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Unified speculative drafter discovery, attachment, and program-node emission shared across
// all target model architectures. Any architecture supplies a TargetSpec (and optional MtpHooks)
// to automatically support its block drafter (DFlash2: .flyx / HF safetensors / arch="dflash" GGUF; DSpark: DeepSpec's
// safetensors or .flyx as DFlash2's block, llama.cpp's DSpark GGUF as its own; DSpine: .flyx / HF safetensors) and MTP.
// A model holds one block drafter of DFlash2's block or DSpine's at most, and llama.cpp's DSpark GGUF beside it (one
// DSpark of either).
#pragma once

#include <optional>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"
#include "models/drafters/dflash2.h"
#include "models/drafters/dspark.h"
#include "models/drafters/dspine.h"
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

class DrafterRegistry {
 public:
  // Inspect embedded drafters in `file` and claim any drafter companions: a companion made for another model is
  // refused here, chosen or not; what a claimed drafter's own config says is read when it is chosen (below).
  // Non-drafter companion indices (such as clip/mmproj vision projectors) are appended to `unclaimed`.
  void scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
            std::vector<int>* unclaimed);

  const DrafterSources& sources() const { return sources_; }
  int mtp_companion() const { return mtp_companion_; }

  // Validate specs of chosen drafters before target tensor binding (refuses bad configs early).
  void validate_selected(const ModelFile& file, const std::vector<Drafter>& selected, const TargetSpec& target);

  // Build blocks for `kind` (kMtp, or the block drafter's) in `T`, returning the drafter block index and what it
  // drafts a round (DrafterReach).
  int attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target, const MtpHooks* mtp,
             DrafterReach* reach);

  // Emit FLYX_NODE_TAP nodes for drafters that tap the pre-layer input (0-based visit index, e.g., DSpark).
  void emit_pre_visit_taps(std::vector<flyx_node>* p, int visit_0based) const;

  // Emit FLYX_NODE_TAP nodes for drafters that tap after 1-based layer `layer_1based` (DFlash2).
  // When `target.has_post_norm_tap` is false, `layer_1based == target.layers` emits post-output_norm taps (FLYX_TAP_NORMED);
  // architectures with a final residual mix set `has_post_norm_tap = true` and call emit_post_norm_taps() after it.
  void emit_taps(std::vector<flyx_node>* p, int layer_1based) const;

  // Emit FLYX_NODE_TAP nodes (FLYX_TAP_NORMED: post output_norm) for taps equal to `n_layers` (DFlash2 and DSpark).
  void emit_post_norm_taps(std::vector<flyx_node>* p, int n_layers) const;

  // Emit FLYX_NODE_CONTEXT nodes for all attached drafters managed by this registry (MTP, the block drafter).
  void emit_contexts(std::vector<flyx_node>* p) const;

 private:
  // The block drafter's spec, read once: DSpine's own, or DFlash2's block (DFlash2, DSpark).
  const DFlash2Spec& ensure_dflash2_spec(const ModelFile& file, const TargetSpec& target);
  const DSparkSpec& ensure_dspark_spec(const ModelFile& file, const TargetSpec& target);
  const DSpineSpec& ensure_dspine_spec(const ModelFile& file, const TargetSpec& target);

  DrafterSources sources_{};
  int mtp_companion_ = -1;
  int block_companion_ = -1;   // the companion holding the block drafter (DFlash2's block, DSpine), -1: the model file
  int dspark_companion_ = -1;  // the companion holding llama.cpp's DSpark GGUF
  std::optional<DFlash2Spec> dflash2_spec_;
  std::optional<DSpineSpec> dspine_spec_;
  std::optional<DSparkSpec> dspark_spec_;

  int target_layers_ = 0;
  bool explicit_post_norm_ = false;
  struct BlockTaps {
    int block = -1;
    std::vector<int> taps;
    void emit(std::vector<flyx_node>* p, int position, int flags) const;
  };
  BlockTaps block_, dspark_;  // the block drafter's (DFlash2's block or DSpine's), llama.cpp's DSpark GGUF
  std::vector<int> context_blocks_;
};

}  // namespace flyx::models::drafters
