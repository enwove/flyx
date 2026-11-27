// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/offload/offload_plan.h"
#include "backends/cuda/runtime/offload/offload_stats.h"
#include "backends/cuda/runtime/row_pool.h"
#include "backends/cuda/runtime/offload/expert_order.h"

namespace flyx::cuda {
// Own the mutable planes as a projection pair. A MoeBlock's kernel-facing ExpertMat views stay read-only.
struct ExpertResidency {
  explicit ExpertResidency(const MoeBlock& f) : gate_up(f.cold_gu), down(f.cold_down), order(f.n_experts) {}
  std::shared_ptr<ColdMatrix> gate_up, down;
  ExpertOrder order;
  int* slots = nullptr;
  uint32_t* counts = nullptr;
};

struct ColdWorkspace {
  ColdWorkspace(size_t threads, OffloadStats& stats) : stats(stats), pool(threads, true) {}
  OffloadStats& stats;
  StagingRing<2> staging;
  size_t next_slot = 0;
  std::map<const ColdMatrix*, ExpertResidency> placements;
  std::map<int, std::vector<uint64_t>> global_scores;
  size_t sessions = 0;
  bool pending_counts = false, failed = false;
  RowPool pool;
  std::mutex mutex;
  int *ids = nullptr, *resident_ids = nullptr, *mapped_ids = nullptr;
  float* mapped_input = nullptr;
  float *input = nullptr, *cpu_slots = nullptr, *gpu_slots = nullptr, *ones = nullptr;
  std::vector<float> act;
  std::vector<ColdQ8> gu_q8, down_q8;
  std::vector<int> cold;
  CudaEvent ready, handoff, timer[5];
  const flyx_seq* owner_sequence = nullptr;
  cudaStream_t owner = nullptr;  // live sequence stream, protected by mutex; cleared before sequence destruction
  ~ColdWorkspace() {
    if (ready) cudaEventSynchronize(ready);
  }
};

}  // namespace flyx::cuda
