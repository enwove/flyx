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
#include "models/drafters/target.h"
#include "models/model.h"

namespace flyx::models::drafters {

class DrafterRegistry {
 public:
  // Inspect embedded drafters in `file` and claim any drafter companions.
  // Non-drafter companion indices (such as clip/mmproj vision projectors) are appended to `unclaimed`.
  void scan(const ModelFile& file, const TargetSpec& target, const MtpHooks* mtp, bool has_dv,
            std::vector<int>* unclaimed);

  const DrafterSources& sources() const { return sources_; }
  int mtp_companion() const { return mtp_companion_; }

  // Validate specs of chosen drafters before target tensor binding (refuses bad configs early).
  void validate_selected(const ModelFile& file, const std::vector<Drafter>& selected, const TargetSpec& target);

  // Build blocks for `kind` (kMtp or kDFlash2) in `T`, returning the drafter block index and its max drafts.
  int attach(Drafter kind, const ModelFile& file, Tables& T, const TargetSpec& target, const MtpHooks* mtp,
             int* max_drafts_out);

  // Emit FLYX_NODE_TAP nodes for all active tap-consuming drafters after 1-based layer `layer_1based`.
  void emit_taps(std::vector<flyx_node>* p, int layer_1based) const;

  // Emit FLYX_NODE_CONTEXT nodes for all attached drafters managed by this registry (MTP, DFlash2).
  void emit_contexts(std::vector<flyx_node>* p) const;

  bool has_active_context() const { return !context_blocks_.empty(); }

 private:
  const DFlash2Spec& ensure_dflash2_spec(const ModelFile& file, const TargetSpec& target);

  DrafterSources sources_{};
  int mtp_companion_ = -1;
  bool dflash2_gguf_ = false;
  int dflash2_companion_ = -1;
  nlohmann::json dflash2_json_;  // canonical JSON when !dflash2_gguf_
  std::optional<DFlash2Spec> dflash2_spec_;

  int dflash2_block_ = -1;
  std::vector<int> dflash2_taps_;
  std::vector<int> context_blocks_;
};

}  // namespace flyx::models::drafters
