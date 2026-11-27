// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/backend.h"

namespace flyx::cuda {

// A target attention or GDN mixer starts a dense layer. Its MLP uses the same residency policy.
// Companions can refer to their internal blocks directly, so those blocks always stay resident.
std::vector<char> streamed_blocks(const flyx_model_desc& desc, const std::vector<char>& draft, int resident);

struct StreamEstimate {
  std::vector<char> host_weight;
  size_t host_bytes = 0, ring_bytes = 0;
};
StreamEstimate estimate_stream(const flyx_model_desc& d, const std::vector<char>& draft, const BackendOptions& options);

struct LayerRing {
  struct Image {
    uint8_t* host = nullptr;
    size_t bytes = 0;
    int slot = 0;
    cudaEvent_t sample_start = nullptr, sample_end = nullptr;
    bool sample_pending = false, sample_round = false;
    std::vector<std::function<void(uint8_t*)>> fixups;  // released once both slot addresses are known
  };
  struct Slot {
    uint8_t* device = nullptr;
    size_t capacity = 0;
    cudaEvent_t ready = nullptr, consumed = nullptr;
  };
  explicit LayerRing(size_t blocks) : images(blocks) {}
  ~LayerRing();
  void archive(flyx_bmodel* m, int block);
  void allocate(flyx_bmodel* m);
  void begin(int block, bool round, cudaStream_t compute);
  void end(int block, cudaStream_t compute);
  std::mutex mutex;
  std::vector<Image> images;
  Slot slots[2];
  cudaStream_t copy = nullptr;
  size_t next = 0, total = 0;

 private:
  struct CopyStats {
    uint64_t bytes = 0, samples = 0;
    double milliseconds = 0;
  } stats[2];  // prefill, round
  bool collect(Image& image);
};

}  // namespace flyx::cuda
