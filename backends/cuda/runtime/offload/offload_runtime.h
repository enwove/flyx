// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <cuda_runtime_api.h>

#include "backends/cuda/runtime/offload/offload_stats.h"

struct flyx_bmodel;
struct flyx_seq;
namespace flyx::cuda {
struct LayerRing;
struct ColdWorkspace;
class ExpertAddressSpace;

class OffloadRuntime {
 public:
  OffloadRuntime();
  ~OffloadRuntime();
  bool allows_graphs() const { return !layers_ && !cold_; }
  bool allows_batched_rounds() const { return allows_graphs(); }
  void on_session_open(flyx_bmodel* m, cudaStream_t stream = nullptr);
  void on_session_close(flyx_bmodel* m, const flyx_seq* sequence = nullptr);
  void maintain(flyx_bmodel* m);
  void finish_prompt(cudaStream_t stream);
  void report(FILE* file);
  void count_fully_resident(size_t routes);
  ExpertAddressSpace& mapped_experts();
  bool maps_experts() const { return bool(mapped_); }
  bool wraps_node(size_t block) const;
  auto lease_node(int block, bool round, cudaStream_t compute);
  LayerRing& init_layers(size_t blocks, bool bounce);
  LayerRing* layers() const { return layers_.get(); }
  ColdWorkspace* experts() const { return cold_.get(); }
  void init_experts(flyx_bmodel* m, const struct OffloadPlan& plan);

 private:
  static void maintain_locked(flyx_bmodel* m, ColdWorkspace& ws);
  OffloadStats stats_;
  std::unique_ptr<ExpertAddressSpace> mapped_;
  std::unique_ptr<LayerRing> layers_;
  std::unique_ptr<ColdWorkspace> cold_;
};

}  // namespace flyx::cuda
