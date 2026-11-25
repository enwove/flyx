// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Unified speculative drafter discovery, attachment, and program-node emission shared across
// all target model architectures. Any architecture supplies a TargetSpec (and optional MtpHooks)
// to automatically support DFlash2 (.flyx / HF safetensors / arch="dflash" GGUF) and MTP.
#pragma once

#include <optional>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"
#include "models/drafters/dflash2.h"
#include "models/drafters/dspark.h"
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

  // Build blocks for `kind` (kMtp, kDFlash2, or kDSpark) in `T`, returning the drafter block index and its max drafts.
  int attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target, const MtpHooks* mtp,
             int* max_drafts_out);

  // Emit FLYX_NODE_TAP nodes for drafters that tap the pre-layer input (0-based visit index, e.g., DSpark).
  void emit_pre_visit_taps(std::vector<flyx_node>* p, int visit_0based) const;

  // Emit FLYX_NODE_TAP nodes for drafters that tap after 1-based layer `layer_1based` (DFlash2).
  // When `target.has_post_norm_tap` is false, `layer_1based == target.layers` emits post-output_norm taps (arg1 = 1);
  // architectures with a final residual mix set `has_post_norm_tap = true` and call emit_post_norm_taps() after it.
  void emit_taps(std::vector<flyx_node>* p, int layer_1based) const;

  // Emit FLYX_NODE_TAP nodes (arg1 = 1: post output_norm) for taps equal to `n_layers` (DFlash2 and DSpark).
  void emit_post_norm_taps(std::vector<flyx_node>* p, int n_layers) const;

  // Emit FLYX_NODE_CONTEXT nodes for all attached drafters managed by this registry (MTP, DFlash2, DSpark).
  void emit_contexts(std::vector<flyx_node>* p) const;

 private:
  const DFlash2Spec& ensure_dflash2_spec(const ModelFile& file, const TargetSpec& target);
  const DSparkSpec& ensure_dspark_spec(const ModelFile& file, const TargetSpec& target);

  DrafterSources sources_{};
  int mtp_companion_ = -1;
  int dflash2_companion_ = -1;  // the companion holding the DFlash2 drafter, -1: the model file ("draft")
  std::optional<DFlash2Spec> dflash2_spec_;
  int dspark_companion_ = -1;
  std::optional<DSparkSpec> dspark_spec_;

  int target_layers_ = 0;
  bool explicit_post_norm_ = false;
  int dflash2_block_ = -1;
  std::vector<int> dflash2_taps_;
  int dspark_block_ = -1;
  std::vector<int> dspark_taps_;
  std::vector<int> context_blocks_;
};

}  // namespace flyx::models::drafters
