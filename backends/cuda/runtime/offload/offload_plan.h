// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/offload/layer_ring.h"

namespace flyx::cuda {
struct OffloadPlan {
  StreamEstimate stream;
  ColdShape cold;
  std::vector<char> experts;
  std::vector<int> hot; // resident slots of each physical expert projection
  std::vector<int> placement_blocks;
  std::array<size_t, 2> prefill_slots{};
  size_t largest_plane = 0;
  size_t resident_expert_bytes = 0, workspace_bytes = 0;
  size_t pinned_bytes = 0, pageable_bytes = 0;
  bool bounce_layers = false, direct = false;
  void fit_pinned(uint64_t budget);
};
OffloadPlan plan_offload(const flyx_model_desc& desc, const BackendOptions& options);
uint64_t query_pinned_budget(const BackendOptions& options);
}  // namespace flyx::cuda
