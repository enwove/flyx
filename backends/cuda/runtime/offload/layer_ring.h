// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/runtime/offload/staging_ring.h"

namespace flyx::cuda {

// A target attention or GDN mixer starts a dense layer. Its MLP uses the same residency policy.
// Companions can refer to their internal blocks directly, so those blocks always stay resident.
std::vector<char> streamed_blocks(const flyx_model_desc& desc, const std::vector<char>& draft, int resident);

struct StreamEstimate {
  std::vector<char> blocks, host_weight;
  size_t host_bytes = 0, ring_bytes = 0;
};
StreamEstimate estimate_stream(const flyx_model_desc& d, const std::vector<char>& draft, const BackendOptions& options);

struct LayerRing {
  struct Image {
    uint8_t* host = nullptr;
    std::vector<uint8_t> pageable;
    size_t bytes = 0;
    int slot = 0;
    CudaEvent sample_start, sample_end;
    bool sample_pending = false, sample_round = false;
    std::vector<std::function<void(uint8_t*)>> fixups;  // released once both slot addresses are known
  };
  LayerRing(size_t blocks, bool bounce, OffloadStats& stats) : images(blocks), bounce_(bounce), stats_(stats) {}
  void finish_samples();
  bool wraps_node(size_t i) const { return images.at(i).host != nullptr; }
  StagingRing<2>::Lease lease_node(int block, bool round, cudaStream_t compute);
  void archive(flyx_bmodel* m, int block);
  void allocate(flyx_bmodel* m);

 private:
  std::vector<Image> images;
  bool bounce_;
  std::array<size_t, 2> capacities{};
  StagingRing<2> ring_;
  std::mutex stats_mutex_;
  size_t next = 0, total = 0;
  OffloadStats& stats_;
  bool collect(Image& image);
};

inline auto OffloadRuntime::lease_node(int block, bool round, cudaStream_t compute) {
  return layers_->lease_node(block, round, compute);
}

}  // namespace flyx::cuda
