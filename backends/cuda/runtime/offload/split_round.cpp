// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/split_round.h"

#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/offload/layer_ring.h"

namespace flyx::cuda {

SplitRound::Step::~Step() {
  if (exec) cudaGraphExecDestroy(exec);
  if (graph) cudaGraphDestroy(graph);
}

void SplitRound::begin(const Pass& ps, int layer) {
  steps_.emplace_back().layer = layer;
  CK(cudaStreamBeginCapture(ps.v.L.st, cudaStreamCaptureModeThreadLocal));
  capturing_ = true;
}

void SplitRound::end(const Pass& ps) {
  if (ps.res && ps.res->expert_work) {
    CK(cudaStreamWaitEvent(ps.v.L.st, ps.s->expert_side.join));
    ps.res->expert_work = false;
  }
  auto& step = steps_.back();
  capturing_ = false;
  CK(cudaStreamEndCapture(ps.v.L.st, &step.graph));
  size_t nodes = 0;
  CK(cudaGraphGetNodes(step.graph, nullptr, &nodes));
  if (!nodes) {
    steps_.pop_back();
    return;
  }
  const auto g = graph_size(step.graph, ps.s->m->graph_prices);
  size_.bytes += g.bytes;
  size_.kernels += g.kernels;
  size_.copies += g.copies;
  size_.sets += g.sets;
  size_.others += g.others;
  size_.forks += g.forks;
  size_.bodies += g.bodies;
  size_.bodies_unknown += g.bodies_unknown;
  size_.params_unknown += g.params_unknown;
  size_.params_most = std::max(size_.params_most, g.params_most);
}

void SplitRound::begin_layer(const Pass& ps, int block) {
  end(ps);
  begin(ps, block);
}
void SplitRound::end_layer(const Pass& ps) {
  end(ps);
  begin(ps);
}

void SplitRound::defer_experts(const Pass& ps, const MoeBlock& block, float* output, bool accumulate, bool fused) {
  end(ps);
  auto& step = steps_.emplace_back();
  step.experts = &block;
  step.output = output;
  step.accumulate = accumulate;
  step.fused = fused;
  begin(ps);
}

bool SplitRound::capture(Pass ps, int prefix) {
  // K/V staging can fork a transfer in one span and join it in another. Such sessions use eager submission.
  if (ps.s->dv.body || std::any_of(ps.s->kv.begin(), ps.s->kv.end(),
                                   [](const KvBuf& k) { return k.vk || k.kq_stage; }))
    return false;
  ps.split = this;
  try {
    begin(ps);
    forward(ps, prefix);
    end(ps);
    return size_.bytes && !size_.bodies_unknown;
  } catch (const std::exception& e) {
    if (capturing_) {
      cudaGraph_t discarded = nullptr;
      cudaStreamEndCapture(ps.v.L.st, &discarded);
      if (discarded) cudaGraphDestroy(discarded);
      capturing_ = false;
    }
    cudaGetLastError();
    if (ps.s->m->verbose) std::fprintf(stderr, "flyx-cuda: split graph capture: %s\n", e.what());
    return false;
  }
}

bool SplitRound::instantiate() {
  for (auto& step : steps_) {
    if (!step.graph) continue;
    if (cudaGraphInstantiate(&step.exec, step.graph, 0) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    CK(cudaGraphDestroy(step.graph));
    step.graph = nullptr;
  }
  return true;
}

void SplitRound::launch(const Pass& ps) const {
  const auto st = ps.v.L.st;
  for (size_t i = 0; i < steps_.size(); ++i) {
    const auto& step = steps_[i];
    if (step.experts) {
      const auto& sc = ps.v.b.moe_sc;
      hybrid_experts(ps, *step.experts, sc.pairs ? &sc : nullptr, step.output, step.accumulate, step.fused);
      continue;
    }
    const bool ahead = step.layer < 0 && i + 1 < steps_.size() && steps_[i + 1].layer >= 0;
    if (step.layer >= 0 || ahead) {
      const auto& streamed = ahead ? steps_[++i] : step;
      auto lease = ps.s->m->offload->lease_node(streamed.layer, true, st);
      if (ahead) CK(cudaGraphLaunch(step.exec, st));
      lease.wait();
      CK(cudaGraphLaunch(streamed.exec, st));
      lease.release();
    } else {
      CK(cudaGraphLaunch(step.exec, st));
    }
  }
}

}  // namespace flyx::cuda
