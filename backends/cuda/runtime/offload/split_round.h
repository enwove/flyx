// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/backend.h"

#include <deque>

namespace flyx::cuda {

// Capture the ordinary lowering once, splitting only where submission needs a host action.
// Residual-bank counters and deferred convolutions therefore span graph boundaries unchanged.
class SplitRound {
 public:
  bool capture(Pass ps, int prefix);
  bool instantiate();
  void launch(const Pass& ps) const;
  const GraphSize& size() const { return size_; }
  void begin_layer(const Pass& ps, int block);
  void end_layer(const Pass& ps);
  void defer_experts(const Pass& ps, const MoeBlock& block, float* output, bool accumulate, bool fused);

 private:
  struct Step {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    int layer = -1;
    const MoeBlock* experts = nullptr;
    float* output = nullptr;
    bool accumulate = false, fused = false;
    Step() = default;
    Step(const Step&) = delete;
    ~Step();
  };
  void begin(const Pass& ps, int layer = -1);
  void end(const Pass& ps);
  std::deque<Step> steps_;
  GraphSize size_;
  bool capturing_ = false;
};

}  // namespace flyx::cuda
