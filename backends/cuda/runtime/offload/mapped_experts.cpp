// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/mapped_experts.h"

#include "backends/cuda/regions/layers.h"

#include <cmath>

namespace flyx::cuda {

size_t MappedExpertPair::Projection::gather_bytes(int pairs) const {
  return checkpoint_align(size_mul(pairs, payload / mat.n_experts)) +
         checkpoint_align(size_mul(pairs, scales / mat.n_experts)) + pairs * sizeof(float);
}

size_t ExpertAddressSpace::gather_bytes(int pairs) const {
  size_t max = 0;
  for (const auto& layer : pairs_)
    max = std::max({max, layer.gate.gather_bytes(pairs), layer.down.gather_bytes(pairs)});
  return max;
}

ExpertMat MappedExpertPair::Projection::gather(uint8_t* buffer, const int* ids, int* compact, int pairs,
                                               cudaStream_t st) const {
  GatherPlanes planes;
  auto copy = [&](auto& ptr, size_t stride, size_t offset) {
    if (!ptr || !stride) return;
    const int blocks = int(std::min<size_t>(128, (stride + 2047) / 2048));
    planes.plane[planes.count++] = {reinterpret_cast<const uint8_t*>(ptr), buffer + offset, stride, blocks};
    planes.blocks += blocks;
    ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(buffer + offset);
  };
  const size_t data = payload / mat.n_experts, scale = scales / mat.n_experts;
  const size_t scale_at = checkpoint_align(size_mul(data, pairs));
  const size_t gains_at = scale_at + checkpoint_align(size_mul(scale, pairs));
  auto staged = mat;
  copy(staged.w, data, 0);
  copy(staged.q, data, 0);
  copy(staged.blk.data, data, 0);
  copy(staged.s, scale, scale_at);
  copy(staged.blk.scales, scale, scale_at);
  copy(staged.rs, sizeof(float), gains_at);
  gather_expert_planes(planes, ids, compact, pairs, st);
  staged.n_experts = pairs;
  if (staged.blk.data) staged.blk.rows = pairs * staged.N;
  return staged;
}

ExpertMat MappedExpertPair::Projection::stage(uint8_t* buffer, cudaStream_t stream) const {
  auto staged = mat;
  auto copy = [&](auto& ptr, size_t bytes, size_t offset) {
    if (!ptr || !bytes) return;
    CK(cudaMemcpyAsync(buffer + offset, ptr, bytes, cudaMemcpyDeviceToDevice, stream));
    ptr = reinterpret_cast<std::remove_reference_t<decltype(ptr)>>(buffer + offset);
  };
  copy(staged.w, payload, 0);
  copy(staged.q, payload, 0);
  copy(staged.blk.data, payload, 0);
  copy(staged.s, scales, checkpoint_align(payload));
  copy(staged.blk.scales, scales, checkpoint_align(payload));
  return staged;
}

void ExpertAddressSpace::init_prefill(flyx_bmodel* model, const std::array<size_t, 2>& slots) {
  prefill_ring_.allocate(model, slots, false);
}

void ExpertAddressSpace::prefill(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate,
                                 bool fused) {
  const auto& b = ps.v.b;
  const auto& launch = ps.v.L;
  const int H = f.hidden, I = f.inter, K = f.top_k, R = ps.rows;
  int grouped = 0;
  for (int i = 0; i < 2; ++i) {
    const auto& projection = i ? f.mapped->down : f.mapped->gate;
    ExpertMat mat;
    StagingRing<2>::Lease lease(prefill_ring_, i, launch.st, [&](uint8_t* buffer, size_t size, cudaStream_t copy) {
      if (size_add(checkpoint_align(projection.payload), projection.scales) > size)
        throw std::logic_error("expert projection exceeds prefill slot");
      mat = projection.stage(buffer, copy);
    });
    lease.wait();
    if (!i) {
      grouped = moe_gate_up(b.h, H, mat, b.moe_idx, R, K, I, H, f.situ_beta, f.situ_linear_beta, fused,
                            fused ? b.moe_act : b.moe_gu, fused ? I : 2 * I, sc, launch);
      if (!fused) situ_rows(b.moe_gu, 2 * I, I, f.situ_beta, f.situ_linear_beta, b.moe_act, I, R * K, launch.st);
    } else {
      moe_down(b.moe_act, I, mat, b.moe_idx, b.moe_w, R, K, H, I, out, H, accumulate, sc, launch, nullptr, grouped);
    }
    lease.release();
  }
}

void* ExpertAddressSpace::allocate(flyx_bmodel* model, int weight, size_t bytes, size_t experts, size_t hot) {
  if (!bytes) return nullptr;
  granule_ = vmm_granularity();
  // A small gain plane takes its exact number of bytes using the model allocator.
  auto& plane = planes_.emplace_back();
  plane.weight = weight;
  plane.stride = bytes / experts;
  if (bytes < granule_ && hot) {
    plane.data = model->alloc<uint8_t>(bytes);
    return plane.data;
  }
  const size_t device = size_mul(hot, plane.stride) / granule_ * granule_;
  auto* address = expert_pages(bytes, device, &plane.memory);
  if (!address) throw std::runtime_error("mapped expert pages unavailable");
  plane.hot.resize(plane.memory.bytes / granule_);
  std::fill_n(plane.hot.begin(), device / granule_, 1);
  model->bytes = size_add(model->bytes, device);
  const size_t pinned = plane.memory.bytes - device;
  model->pinned_bytes = size_add(model->pinned_bytes, pinned);
  model->host_bytes = size_add(
      model->host_bytes, size_add(pinned, size_mul(plane.hot.size(), sizeof(uint8_t) + sizeof(unsigned long long))));
  plane.data = static_cast<uint8_t*>(address);
  return address;
}

void ExpertAddressSpace::attach(flyx_bmodel* model, MoeBlock& block, int gate, int down, int resident) {
  verbose_ = model->verbose != 0;
  const size_t n = block.n_experts;
  auto [it, added] = histograms_.try_emplace({gate, down}, n);
  auto& h = it->second;
  if (added) {
    h.device = model->alloc<uint32_t>(2 * n);
    h.slots = model->alloc<int>(n);
    CK(cudaMemset(h.device, 0, 2 * n * sizeof(uint32_t)));
    CK(cudaMemcpy(h.slots, h.order.slot.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    model->host_bytes += n * (2 * sizeof(uint32_t) + 2 * sizeof(int) + sizeof(uint64_t));
    if (model->opt.expert_adapt && !scratch_) scratch_ = model->alloc<uint8_t>(vmm_granularity());
  }
  {
    auto projection = [&](int weight, const ExpertMat& mat) {
      MappedExpertPair::Projection result{mat};
      for (const auto& plane : planes_) {
        if (plane.weight != weight) continue;
        const size_t bytes = plane.stride * block.n_experts;
        const void* addr = plane.data;
        if (addr == mat.w || addr == mat.q || addr == mat.blk.data) result.payload = bytes;
        if (addr == mat.s || addr == mat.blk.scales) result.scales = bytes;
      }
      return result;
    };
    auto& mapped = pairs_.emplace_back();
    mapped.gate = projection(gate, block.gate_up);
    mapped.down = projection(down, block.down);
    mapped.owner = this;
    mapped.gather = &h.gather;
    block.mapped = &mapped;
    h.top_k = block.top_k;
    h.gather = int64_t(block.n_experts - resident) * block.top_k >= block.n_experts;
    const int configured = model->opt.expert_prefill_min;
    if (model->opt.keep_expert_archive && h.gather)
      mapped.stage_min_rows = std::max(2, configured < 0 ? 1 + (block.n_experts - 1) / block.top_k : configured);
  }

  block.route_histogram = h.device;
  block.route_slots = h.slots;
}

// plan_lowered serializes writers with the model graph mutex. A sequence freezes its row capacity
// before sizing its arena and uses ordinary mapped loads for a wider plan lowered later.
void ExpertAddressSpace::note_rows(int rows) {
  if (rows > 1 && rows <= FLYX_MAX_ROWS && rows > planned_rows()) max_rows_.store(rows, std::memory_order_relaxed);
}

bool ExpertAddressSpace::resident(std::pair<int, int> pair, size_t slot) const {
  for (const auto& plane : planes_) {
    if (plane.hot.empty() || (plane.weight != pair.first && plane.weight != pair.second)) continue;
    const size_t first = slot * plane.stride / granule_, last = ((slot + 1) * plane.stride - 1) / granule_;
    for (size_t p = first; p <= last; ++p)
      if (!plane.hot[p]) return false;
  }
  return true;
}

void ExpertAddressSpace::select_gather() {
  for (auto& [pair, h] : histograms_) {
    if (observed_rows_) {
      h.gather = 2 * observed_remote_rows_ >= observed_rows_;
      continue;
    }
    uint64_t total = 0, cold = 0;
    for (size_t i = 0; i < h.order.scores.size(); ++i) {
      total += h.order.scores[i];
      if (!resident(pair, h.order.slot[i])) cold += h.order.scores[i];
    }
    // Gather repeated reads when at least one routed expert per row is expected to use the host.
    if (total) h.gather = double(cold) * h.top_k >= double(total);
  }
}

void ExpertAddressSpace::count_resident(size_t routes) {
  std::lock_guard lock(mutex_);
  full_routes_ += routes;
}
void ExpertAddressSpace::observe() {
  uint64_t visited = full_routes_, resident = full_routes_;
  full_routes_ = 0;
  double sampled_rows = 0, remote_rows = 0;
  for (auto& [pair, h] : histograms_) {
    const size_t n = h.order.scores.size();
    CK(cudaMemcpy(h.order.counts.data(), h.device, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.decode.data(), h.device + n, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CK(cudaMemset(h.device, 0, 2 * n * sizeof(uint32_t)));
    uint64_t count = 0, cold = 0;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t decode = h.decode[i];
      h.order.scores[i] = h.order.scores[i] / 2 + uint64_t(h.order.counts[i]) + decode;
      pending_ |= h.order.counts[i] != 0 || decode != 0;
      if (!decode) continue;
      count += decode;
      const bool hot = this->resident(pair, h.order.slot[i]);
      if (!hot) cold += decode;
    }
    if (count) {
      visited += count;
      resident += count - cold;
      // The histogram contains marginal slot frequencies. Estimate the chance that at least
      // one of the row's K slots is remote, using the placement that actually served the round.
      const double rows = double(count) / h.top_k;
      sampled_rows += rows;
      remote_rows += rows * (1 - std::pow(double(count - cold) / count, h.top_k));
    }
  }
  if (sampled_rows) {
    observed_rows_ = 0.5 * observed_rows_ + sampled_rows;
    observed_remote_rows_ = 0.5 * observed_remote_rows_ + remote_rows;
  }
  hits_ += resident;
  slots_ += visited;
  if (verbose_ && visited)
    std::fprintf(stderr, "flyx-cuda: session decode expert cache hit rate %.2f%% (%llu/%llu routed slots)\n",
                 100. * resident / visited, static_cast<unsigned long long>(resident),
                 static_cast<unsigned long long>(visited));
}

void ExpertAddressSpace::rank_experts() {
  struct Candidate {
    Histogram* layer;
    size_t bytes;
    double density;
  };
  std::vector<Candidate> candidates;
  std::map<int, int> uses;
  size_t budget = 0;
  for (const auto& plane : planes_) budget += std::count(plane.hot.begin(), plane.hot.end(), uint8_t(1)) * granule_;
  for (auto& [pair, h] : histograms_) {
    ++uses[pair.first];
    ++uses[pair.second];
    h.desired = 0;
    size_t bytes = 0;
    for (const auto& plane : planes_)
      if (!plane.hot.empty() && (plane.weight == pair.first || plane.weight == pair.second)) bytes += plane.stride;
    if (bytes)
      for (const auto score : h.order.scores) candidates.push_back({&h, bytes, double(score) / bytes});
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const auto& a, const auto& b) { return a.density > b.density; });
  for (const auto& c : candidates)
    if (c.bytes <= budget) {
      ++c.layer->desired;
      budget -= c.bytes;
    }

  // Group a layer's hot experts before moving pages. Neighbours then share device pages even
  // when an expert or its scale plane is smaller than the driver's mapping granularity.
  for (auto& [pair, h] : histograms_) {
    if (uses[pair.first] != 1 || uses[pair.second] != 1) continue;
    const auto changed = h.order.update(int(h.desired), h.order.scores.data(), [&](int a, int b) {
      for (const auto& plane : planes_) {
        if (plane.weight != pair.first && plane.weight != pair.second) continue;
        for (size_t i = 0; i < plane.stride; i += granule_) {
          const size_t n = std::min(granule_, plane.stride - i);
          auto* x = plane.data + size_t(a) * plane.stride + i;
          auto* y = plane.data + size_t(b) * plane.stride + i;
          CK(cudaMemcpyAsync(scratch_, x, n, cudaMemcpyDeviceToDevice));
          CK(cudaMemcpyAsync(x, y, n, cudaMemcpyDeviceToDevice));
          CK(cudaMemcpyAsync(y, scratch_, n, cudaMemcpyDeviceToDevice));
        }
      }
    });
    if (changed)
      CK(cudaMemcpy(h.slots, h.order.slot.data(), h.order.slot.size() * sizeof(int), cudaMemcpyHostToDevice));
  }
  CK(cudaStreamSynchronize(nullptr));
}

void ExpertAddressSpace::promote() {
  rank_experts();
  struct Page {
    double frequency;
    size_t order;
    Plane* plane;
    size_t index;
  };
  std::vector<Page> device, host;
  size_t order = 0;
  for (auto& plane : planes_) {
    std::vector<uint64_t> scores;
    for (const auto& [pair, h] : histograms_)
      if (plane.weight == pair.first || plane.weight == pair.second) {
        scores.resize(h.order.id.size());
        for (size_t slot = 0; slot < scores.size(); ++slot) scores[slot] += h.order.scores[h.order.id[slot]];
      }
    for (size_t p = 0; p < plane.hot.size(); ++p) {
      const size_t start = p * granule_, end = std::min(start + granule_, scores.size() * plane.stride);
      double demand = 0;
      for (size_t offset = start; offset < end;) {
        const size_t e = offset / plane.stride;
        const size_t next = std::min(end, (e + 1) * plane.stride);
        demand += double(scores[e]) * (next - offset);
        offset = next;
      }
      (plane.hot[p] ? device : host).push_back({demand, order++, &plane, p});
    }
  }
  auto less = [](const Page& a, const Page& b) {
    return a.frequency != b.frequency ? a.frequency < b.frequency : a.order < b.order;
  };
  std::sort(device.begin(), device.end(), less);
  std::sort(host.rbegin(), host.rend(), less);
  // Bound adaptation traffic; a tie retains the existing backing.
  const size_t limit = std::min({device.size(), host.size(), (order + 7) / 8});
  size_t i = 0;
  for (; i < limit && host[i].frequency > device[i].frequency; ++i) {
    auto &a = device[i], &b = host[i];
    copy_expert_pages(a.plane->memory, a.index, b.plane->memory, b.index, scratch_);
  }
  CK(cudaStreamSynchronize(nullptr));
  for (size_t j = 0; j < i; ++j) {
    auto &a = device[j], &b = host[j];
    exchange_expert_backing(a.plane->memory, a.index, b.plane->memory, b.index);
    a.plane->hot[a.index] = 0;
    b.plane->hot[b.index] = 1;
  }
  select_gather();
  if (verbose_) std::fprintf(stderr, "flyx-cuda: redistributed %zu expert pages at quiescence\n", i);
}

void ExpertAddressSpace::finish_prompt(cudaStream_t stream) {
  std::lock_guard lock(mutex_);
  if (!scratch_ || sessions_ != 1) return;
  CK(cudaStreamSynchronize(stream));
  try {
    observe();
    if (pending_) promote();
    pending_ = false;
  } catch (...) {
    failed_ = true;
    throw;
  }
}

void ExpertAddressSpace::open() {
  std::lock_guard lock(mutex_);
  if (failed_) throw std::runtime_error("mapped expert placement failed");
  if (!sessions_ && pending_ && scratch_) {
    try {
      promote();
    } catch (...) {
      failed_ = true;
      throw;
    }
    pending_ = false;
  }
  ++sessions_;
}

void ExpertAddressSpace::close(const flyx_seq* seq) noexcept {
  std::lock_guard lock(mutex_);
  --sessions_;
  try {
    if (seq) CK(cudaStreamSynchronize(seq->st));
    if (sessions_ == 0 && !failed_) observe();
  } catch (...) {
    failed_ = true;
  }
}

void ExpertAddressSpace::report(FILE* out) const {
  if (verbose_ && slots_)
    std::fprintf(out, "flyx-cuda: decode expert cache hit rate %.2f%% (%llu/%llu routed slots)\n",
                 100. * hits_ / slots_, static_cast<unsigned long long>(hits_),
                 static_cast<unsigned long long>(slots_));
}

} // namespace flyx::cuda
