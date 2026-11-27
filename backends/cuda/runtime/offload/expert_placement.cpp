// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_placement.h"
#include "backends/cuda/runtime/offload/mapped_experts.h"

namespace flyx::cuda {
ExpertAddressSpace& OffloadRuntime::mapped_experts() {
  if (!mapped_) mapped_ = std::make_unique<ExpertAddressSpace>();
  return *mapped_;
}
OffloadRuntime::OffloadRuntime() = default;
OffloadRuntime::~OffloadRuntime() = default;
LayerRing& OffloadRuntime::init_layers(size_t blocks, bool bounce) {
  layers_ = std::make_unique<LayerRing>(blocks, bounce, stats_);
  return *layers_;
}
bool OffloadRuntime::wraps_node(size_t block) const { return layers_ && layers_->wraps_node(block); }
void OffloadRuntime::count_fully_resident(size_t routes) {
  if (mapped_) return mapped_->count_resident(routes);
  std::lock_guard lock(cold_->mutex);
  stats_.decode_slots += routes;
}
void OffloadRuntime::report(FILE* file) {
  if (mapped_) mapped_->report(file);
  if (layers_) layers_->finish_samples();
  stats_.report(file, cold_ ? cold_->pool.threads() : 0);
}

void OffloadRuntime::init_experts(flyx_bmodel* m, const OffloadPlan& plan) {
  const auto& shape = plan.cold;
  const size_t threads = m->opt.expert_threads ? std::min<size_t>(size_t(m->opt.expert_threads),
                                                                  std::max(1u, std::thread::hardware_concurrency()))
                                               : RowPool::default_threads();
  auto w = std::make_unique<ColdWorkspace>(threads, stats_);
  w->stats.verbose = m->verbose;
  w->stats.profile = m->opt.profile;
  if (w->stats.profile)
    for (auto& event : w->timer) event.create(cudaEventDefault);
  if (shape.bank_bytes) w->staging.allocate(m, {shape.bank_bytes, shape.bank_bytes}, true, &w->pool);
  w->resident_ids = m->alloc<int>(shape.pairs);
  if (shape.bank_bytes && m->opt.expert_adapt) {
    for (int i : plan.placement_blocks) {
      auto* f = &std::get<MoeBlock>(m->blocks.at(size_t(i)));
      auto [it, added] = w->placements.try_emplace(f->cold_gu.get(), *f);
      if (!added) continue;
      auto& placement = it->second;
      const size_t n = f->n_experts;
      placement.slots = m->alloc<int>(n);
      placement.counts = m->alloc<uint32_t>(n);
      CK(cudaMemset(placement.counts, 0, n * sizeof(uint32_t)));
      CK(cudaMemcpy(placement.slots, placement.order.slot.data(), n * sizeof(int), cudaMemcpyHostToDevice));
      m->host_bytes += n * (2 * sizeof(int) + sizeof(uint64_t) + sizeof(uint32_t));
      if (m->opt.expert_global && w->global_scores.try_emplace(int(n), n, 0).second)
        m->host_bytes += n * sizeof(uint64_t);
    }
  }
  int device = 0, can_map = 0;
  CK(cudaGetDevice(&device));
  CK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device));
  w->ids = m->host_pinned<int>(shape.pairs, can_map ? &w->mapped_ids : nullptr);
  w->input = m->host_pinned<float>(shape.inputs, can_map ? &w->mapped_input : nullptr);
  w->cpu_slots = m->host_pinned<float>(shape.outputs, can_map && m->opt.expert_mapped_output ? &w->mapped_output : nullptr);
  w->ticket = m->host_pinned<uint32_t>(1, can_map && m->opt.expert_ticket ? &w->mapped_ticket : nullptr);
  *w->ticket = 0;
  w->gpu_slots = m->alloc<float>(shape.outputs);
  w->ones = m->alloc<float>(shape.pairs);
  std::vector<float> ones(shape.pairs, 1.f);
  CK(cudaMemcpy(w->ones, ones.data(), shape.pairs * sizeof(float), cudaMemcpyHostToDevice));
  w->act.resize(shape.activations);
  w->gu_q8.resize((shape.inputs + 31) / 32);
  w->down_q8.resize((shape.activations + 31) / 32);
  m->host_bytes += shape.quant_bytes();
  w->cold.reserve(shape.pairs);
  m->host_bytes += shape.activations * sizeof(float) + shape.pairs * sizeof(int);
  w->handoff.create();
  w->ready.create();
  CK(cudaEventRecord(w->ready, nullptr));
  cold_ = std::move(w);
}

void OffloadRuntime::finish_prompt(cudaStream_t stream) {
  if (mapped_) mapped_->finish_prompt(stream);
}

void OffloadRuntime::on_session_open(flyx_bmodel* m, cudaStream_t stream) {
  if (mapped_) mapped_->open();
  if (!cold_) return;
  auto& ws = *cold_;
  std::lock_guard lock(ws.mutex);
  if (ws.failed) throw std::runtime_error("expert exchange failed on this model");
  maintain_locked(m, ws);
  CK(cudaStreamWaitEvent(stream, ws.ready));
  ++ws.sessions;
}

void OffloadRuntime::maintain(flyx_bmodel* m) {
  if (!cold_) return;
  std::lock_guard lock(cold_->mutex);
  maintain_locked(m, *cold_);
}

void OffloadRuntime::maintain_locked(flyx_bmodel* m, ColdWorkspace& ws) {
  if (ws.sessions || !ws.pending_counts) return;
  CK(cudaEventSynchronize(ws.ready));
  size_t swaps = 0;
  try {
    for (auto& [_, scores] : ws.global_scores) std::fill(scores.begin(), scores.end(), 0);
    for (auto& [_, p] : ws.placements) {
      auto& o = p.order;
      CK(cudaMemcpy(o.counts.data(), p.counts, o.counts.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CK(cudaMemset(p.counts, 0, o.counts.size() * sizeof(uint32_t)));
      o.observe();
      if (m->opt.expert_global) {
        auto& scores = ws.global_scores.at(int(o.slot.size()));
        for (size_t e = 0; e < scores.size(); ++e) scores[e] += o.scores[e];
      }
    }
    for (auto& [_, p] : ws.placements) {
      const int total = int(p.order.slot.size());
      const auto& scores = m->opt.expert_global ? ws.global_scores.at(total) : p.order.scores;
      const size_t changed = p.order.update(p.gate_up->first, scores.data(), [&](int hot, int cold) {
        p.gate_up->exchange(hot, cold, total, ws.staging.bounce(0));
        p.down->exchange(hot, cold, total, ws.staging.bounce(0));
      });
      if (changed)
        CK(cudaMemcpy(p.slots, p.order.slot.data(), p.order.slot.size() * sizeof(int), cudaMemcpyHostToDevice));
      swaps += changed;
    }
    CK(cudaEventRecord(ws.ready, nullptr));  // Publish the exchanged planes, slot maps and count resets.
    CK(cudaEventSynchronize(ws.ready));
  } catch (...) {
    // An interrupted pair exchange must never expose the old logical map to a new session.
    ws.failed = true;
    throw;
  }
  if (ws.stats.verbose) std::fprintf(stderr, "flyx-cuda: promoted %zu experts between sessions\n", swaps);
  ws.pending_counts = false;
}

void OffloadRuntime::on_session_close(flyx_bmodel*, const flyx_seq* sequence) {
  if (mapped_) mapped_->close(sequence);
  if (!cold_) return;
  auto& ws = *cold_;
  std::lock_guard lock(ws.mutex);
  if (sequence && ws.owner_sequence == sequence) {
    if (cudaEventRecord(ws.ready, ws.owner) != cudaSuccess) {
      cudaStreamSynchronize(ws.owner);
      ws.failed = true;
    }
    ws.owner = nullptr;
    ws.owner_sequence = nullptr;
  }
  if (--ws.sessions == 0 && ws.stats.verbose && ws.stats.decode_slots > ws.stats.last_decode_slots) {
    const auto slots = ws.stats.decode_slots - ws.stats.last_decode_slots;
    const auto cold = ws.stats.decode_cold - ws.stats.last_decode_cold;
    std::fprintf(stderr, "flyx-cuda: session decode expert cache hit rate %.2f%% (%llu/%llu routed slots)\n",
                 100. * (slots - cold) / slots, static_cast<unsigned long long>(slots - cold),
                 static_cast<unsigned long long>(slots));
    ws.stats.last_decode_slots = ws.stats.decode_slots;
    ws.stats.last_decode_cold = ws.stats.decode_cold;
  }
}

}  // namespace flyx::cuda
