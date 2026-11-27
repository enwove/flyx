// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/backend.h"
#include "backends/cuda/runtime/offload/expert_order.h"

#include "backends/cuda/runtime/offload/staging_ring.h"

#include <deque>
#include <map>

namespace flyx::cuda {
// Every round sees stable projection addresses. At quiescence, equally sized pages can exchange
// physical backing across weights without increasing the device budget or changing any value.
struct MappedExpertPair {
  struct Projection {
    ExpertMat mat;
    size_t payload = 0, scales = 0;
    size_t gather_bytes(int pairs) const;
    ExpertMat gather(uint8_t* buffer, const int* ids, int* compact, int pairs, cudaStream_t stream) const;
    ExpertMat stage(uint8_t* buffer, cudaStream_t stream) const;
  };
  Projection gate, down;
  const bool* gather = nullptr;
  int stage_min_rows = INT_MAX;
  class ExpertAddressSpace* owner = nullptr;
};

class ExpertAddressSpace {
public:
  void* allocate(flyx_bmodel* model, int weight, size_t bytes, size_t experts, size_t hot);
  void attach(flyx_bmodel* model, MoeBlock& block, int gate, int down);
  size_t gather_bytes(int pairs) const;
  void note_rows(int rows);
  int planned_rows() const { return max_rows_.load(std::memory_order_relaxed); }
  void init_prefill(flyx_bmodel* model, const std::array<size_t, 2>& slots);
  void count_resident(size_t routes);
  void prefill(const Pass& ps, const MoeBlock& block, const MoeScratch* sc, float* out, bool accumulate, bool fused);
  void open();
  void finish_prompt(cudaStream_t stream);
  void close(const flyx_seq* seq) noexcept;
  void report(FILE* out) const;

private:
  struct Plane {
    DevicePieces memory;
    int weight = 0;
    size_t stride = 0;
    uint8_t* data = nullptr;
    std::vector<uint8_t> hot;
  };
  struct Histogram {
    explicit Histogram(size_t n) : order(n), decode(n) {}
    ExpertOrder order;
    uint32_t* device = nullptr;
    int* slots = nullptr;
    std::vector<uint32_t> decode;
    size_t desired = 0;
    bool gather = false;
    int top_k = 0;
  };
  bool resident(std::pair<int, int> pair, size_t slot) const;
  void observe();
  void select_gather();
  void promote();
  void rank_experts();
  std::deque<MappedExpertPair> pairs_;
  StagingRing<2> prefill_ring_;
  std::deque<Plane> planes_;
  std::map<std::pair<int, int>, Histogram> histograms_;
  uint8_t* scratch_ = nullptr;
  size_t granule_ = 0, sessions_ = 0;
  uint64_t hits_ = 0, slots_ = 0, full_routes_ = 0;
  bool failed_ = false, pending_ = false, verbose_ = false;
  std::mutex mutex_;
  std::atomic<int> max_rows_{0};
};
} // namespace flyx::cuda
