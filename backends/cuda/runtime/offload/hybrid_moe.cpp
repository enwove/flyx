// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_placement.h"
#include "backends/cuda/runtime/offload/host_handoff.h"
#include "backends/cuda/runtime/offload/split_round.h"

namespace flyx::cuda {
namespace {
void streamed_prefill(ColdWorkspace& ws, const Pass& ps, const MoeBlock& f, const MoeScratch& scratch, float* out,
                      bool accumulate, bool fused) {
  const auto& b = ps.v.b;
  const auto& L = ps.v.L;
  const int H = f.hidden, K = f.top_k, I = f.inter;
  // Produce all gate/up rows before touching down_buf. Generic gate/up and down_buf share the same allocation.
  for (bool down : {false, true}) {
    const auto& resident = down ? f.down : f.gate_up;
    const ColdMatrix& cold = down ? *f.cold_down : *f.cold_gu;
    auto launch = [&](const ExpertMat& mat, int first) {
      const int capacity = opts(L).moe_vec && mat.q ? std::min(scratch.chunk_rows, FLYX_MAX_ROWS) : scratch.chunk_rows;
      for (int base = 0; base < ps.rows;) {
        int rows = std::min(ps.rows - base, capacity);
        // Filtered grouping needs at least two rows. Give the final tile two when the tail would be a singleton.
        if (ps.rows - base - rows == 1) --rows;
        auto sc = scratch;
        sc.down_buf += size_t(base) * K * H;
        const int* ids = b.moe_idx + size_t(base) * K;
        float* act = b.moe_act + size_t(base) * K * I;
        if (down) {
          moe_bank_down(act, I, mat, ids, b.moe_w + size_t(base) * K, rows, K, H, I, out + size_t(base) * H, H,
                        accumulate, &sc, L, nullptr, 0, {{first, mat.n_experts}, true});
        } else {
          float* gu = b.moe_gu + size_t(base) * K * 2 * I;
          moe_bank_gate_up(b.h + size_t(base) * H, H, mat, ids, rows, K, I, H, f.situ_beta, f.situ_linear_beta, fused,
                           fused ? act : gu, fused ? I : 2 * I, &sc, L, false, nullptr, {first, mat.n_experts});
        }
        base += rows;
      }
    };
    size_t offset = 0;
    const auto& bank = cold.bank;
    for (int first = cold.first; first < f.n_experts;) {
      const int n = std::min<int>(int(bank.count), f.n_experts - first);
      const size_t bytes = bank.bytes(n);
      StagingRing<2>::Lease lease(ws.staging, ws.next_slot++ % 2, cold.archive.data() + offset, bytes, L.st);
      if (first == cold.first && resident.n_experts) launch(resident, 0);
      ExpertMat mat{};
      mat.N = cold.rows;
      mat.n_experts = n;
      auto* scales = lease.device() + bank.scale_offset(n);
      if (cold.nvfp4) {
        mat.q = lease.device();
        mat.s = scales;
      } else if (cold.q8 || cold.a16h) {
        mat.blk = {lease.device(), reinterpret_cast<uint16_t*>(scales), cold.dtype, n * cold.rows, cold.q8};
      } else {
        mat.w = reinterpret_cast<bf16*>(lease.device());
      }
      if (bank.gains) mat.rs = reinterpret_cast<float*>(lease.device() + bank.gain_offset(n));
      lease.wait();
      launch(mat, first);
      lease.release();
      offset += bytes;
      first += n;
    }
    if (!down && !fused)
      situ_rows(b.moe_gu, 2 * I, I, f.situ_beta, f.situ_linear_beta, b.moe_act, I, ps.rows * K, L.st);
  }
  moe_sum_slots(scratch.down_buf, b.moe_w, ps.rows, K, H, out, accumulate, L.st);
}
}  // namespace

void hybrid_experts(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate, bool fused) {
  if (ps.split) {
    ps.split->defer_experts(ps, f, out, accumulate, fused);
    return;
  }
  auto& ws = *ps.s->m->offload->experts();
  std::lock_guard lock(ws.mutex);
  const Launch& L = ps.v.L;
  const auto& b = ps.v.b;
  int H = f.hidden, I = f.inter, K = f.top_k;
  if (ps.rows > INT_MAX / K) throw std::runtime_error("MoE pass exceeds routed pair index range");
  const bool vec = opts(L).moe_vec;
  // One sequence orders successive visits on its own stream. Fence the former owner when ownership changes.
  if (ws.owner_sequence != ps.s || ws.owner != L.st) {
    if (ws.owner_sequence) CK(cudaEventRecord(ws.ready, ws.owner));
    CK(cudaStreamWaitEvent(L.st, ws.ready));
    ws.owner = L.st;
    ws.owner_sequence = ps.s;
  }
  auto* t = ws.stats.profile ? &ws.stats.times[{&f, ps.round}] : nullptr;
  try {
    const int configured = ps.s->m->opt.expert_prefill_min;
    const int min_rows = configured < 0 ? 1 + (f.n_experts - 1) / K : configured;
    const bool stage =
        !ps.round && ws.staging.stream() && sc && sc->chunk_rows >= 3 && ps.rows >= std::max(2, min_rows);
    auto placement = ws.placements.find(f.cold_gu.get());
    const bool mapped_chunk = placement != ws.placements.end() && !stage && ps.rows <= ColdShape::chunk;
    const bool direct = mapped_chunk && ws.mapped_ids && ws.mapped_input && ps.rows * K <= 128;
    const bool ticketed = direct && ws.mapped_ticket;
    const auto generation = ticketed ? ++ws.next_ticket : 0;
    if (placement != ws.placements.end() || stage) {
      auto* map = placement == ws.placements.end() ? nullptr : &placement->second;
      moe_place_ids(b.moe_idx, ps.rows * K, f.n_experts, map ? map->slots : nullptr,
                    map && !ps.round ? map->counts : nullptr, stage ? ps.s->route_error : nullptr, L.st,
                    {mapped_chunk ? ws.resident_ids : nullptr, f.gate_up.n_experts, direct ? ws.mapped_ids : nullptr,
                     b.h, ws.mapped_input, direct ? ps.rows * H : 0, ticketed ? ws.mapped_ticket : nullptr, generation});
      if (map && !ps.round) ws.pending_counts = true;
    }
    if (stage) {
      streamed_prefill(ws, ps, f, *sc, out, accumulate, fused);
      return;
    }
    if (t) {
      ++t->visits;
      t->rows += ps.rows;
    }
    ++ws.stats.visits;
    if (ws.stats.verbose && ps.round) ++ws.stats.decode_visits;
    for (int base = 0; base < ps.rows; base += ColdShape::chunk) {
      int rows = std::min(ColdShape::chunk, ps.rows - base);
      auto* ids = b.moe_idx + size_t(base) * K;
      float* input = b.h + size_t(base) * H;
      auto start = ws.stats.verbose || t ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      HostHandoff handoff(L.st, ws.handoff, ids, ws.ids, size_t(rows) * K, input, ws.input, size_t(rows) * H, direct,
                          ticketed ? ws.ticket : nullptr, generation);
      if (t) CK(cudaEventRecord(ws.timer[0], L.st));
      if (f.gate_up.n_experts) {
        if (!mapped_chunk) moe_resident_ids(ids, ws.resident_ids, rows * K, f.gate_up.n_experts, L.st);
        moe_gate_up(input, H, f.gate_up, ws.resident_ids, rows, K, I, H, f.situ_beta, f.situ_linear_beta, fused,
                    fused ? b.moe_act : b.moe_gu, fused ? I : 2 * I, sc, L);
        if (!fused) situ_rows(b.moe_gu, 2 * I, I, f.situ_beta, f.situ_linear_beta, b.moe_act, I, rows * K, L.st);
        // One down projection per pair. The scratch holds chunk_rows * max_top_k pairs and all their input rows.
        MoeScratch pairs{};
        if (sc) {
          pairs = *sc;
          pairs.chunk_rows *= pairs.max_top_k;
          pairs.max_top_k = 1;
        }
        moe_down(b.moe_act, I, f.down, ws.resident_ids, ws.ones, rows * K, 1, H, I, ws.gpu_slots, H, false,
                 sc ? &pairs : nullptr, L);
      }
      if (t) CK(cudaEventRecord(ws.timer[1], L.st));
      handoff.wait();
      if (ws.stats.verbose || t) {
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ws.stats.route_seconds += dt;
        if (ps.round) ws.stats.decode_handoff += dt;
        if (t) t->handoff += dt * 1e6;
      }
      ws.cold.clear();
      for (int slot = 0; slot < rows * K; ++slot) {
        const int e = ws.ids[slot];
        if (e < 0 || e >= f.n_experts) throw std::runtime_error("invalid MoE router expert ID");
        if (e >= f.cold_gu->first) ws.cold.push_back(slot);
      }
      if (t) {
        t->pairs += rows * K;
        t->cold += ws.cold.size();
        t->bytes += ws.cold.size() * (f.cold_gu->stride * f.cold_gu->rows + f.cold_down->stride * f.cold_down->rows);
      }
      if (ws.stats.verbose) {
        if (ps.round) {
          ws.stats.decode_slots += rows * K;
          ws.stats.decode_cold += ws.cold.size();
        }
        ++ws.stats.chunks;
        ws.stats.slots += size_t(rows) * K;
        ws.stats.cpu_pairs += ws.cold.size();
      }
      if (!ws.cold.empty()) {
        if (ws.stats.verbose || t) start = std::chrono::steady_clock::now();
        int last_row = -1;
        for (int slot : ws.cold) {
          const int row = slot / K;
          if (row != last_row) {
            if (f.cold_gu->quant_dot)
              prepare_cold_q8(ws.input + size_t(row) * H, ws.gu_q8.data() + size_t(row) * (H / 32), H,
                              f.cold_gu->max_isa);
            else
              f.cold_gu->prepare(ws.input + size_t(row) * H, vec);
          }
          last_row = row;
        }
        // Group identical weight reads while preserving the router's slot order in the output.
        if (rows > 1) std::sort(ws.cold.begin(), ws.cold.end(), [&](int a, int b) { return ws.ids[a] < ws.ids[b]; });
        std::array<int, ColdShape::chunk * kMoeRouteMaxTopK + 1> starts;
        int groups = 0;
        for (int i = 0; i < int(ws.cold.size()); ++i)
          if (!i || ws.ids[ws.cold[i]] != ws.ids[ws.cold[i - 1]]) starts[groups++] = i;
        starts[groups] = int(ws.cold.size());
        const ColdMatrix::Groups group{ws.ids, ws.cold.data(), starts.data(), groups, K};
        f.cold_gu->project(ws.pool, group, true, f.situ_beta, f.situ_linear_beta, ws.input, ws.gu_q8.data(),
                           ws.act.data());
        for (int slot : ws.cold) {
          float* x = ws.act.data() + size_t(slot) * I;
          if (f.cold_down->quant_dot)
            prepare_cold_q8(x, ws.down_q8.data() + size_t(slot) * (I / 32), I, f.cold_down->max_isa);
          else
            f.cold_down->prepare(x, vec);
        }
        f.cold_down->project(ws.pool, group, false, 0.f, 0.f, ws.act.data(), ws.down_q8.data(), ws.cpu_slots);
        if (rows > 1) std::sort(ws.cold.begin(), ws.cold.end());
        if (ws.stats.verbose || t) {
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          ws.stats.cpu_seconds += dt;
          if (ps.round) {
            ws.stats.decode_cpu += dt;
            ws.stats.decode_bytes +=
                ws.cold.size() * (f.cold_gu->stride * f.cold_gu->rows + f.cold_down->stride * f.cold_down->rows);
          }
          if (t) t->cpu += dt * 1e6;
        }
        if (t) CK(cudaEventRecord(ws.timer[2], L.st));
        for (size_t i = 0; !ws.mapped_output && i < ws.cold.size();) {
          int slot = ws.cold[i++];
          size_t count = 1;
          while (i < ws.cold.size() && size_t(ws.cold[i]) == slot + count) {
            ++i;
            ++count;
          }
          const size_t offset = size_t(slot) * H;
          CK(cudaMemcpyAsync(ws.gpu_slots + offset, ws.cpu_slots + offset, count * H * sizeof(float),
                             cudaMemcpyHostToDevice, L.st));
        }
      }
      if (t) CK(cudaEventRecord(ws.timer[3], L.st));
      if (ws.mapped_output && !ws.cold.empty())
        moe_sum_mapped(ws.gpu_slots, ws.mapped_output, ids, f.cold_gu->first, b.moe_w + size_t(base) * K, rows, K, H,
                       out + size_t(base) * H, accumulate, L.st);
      else
        moe_sum_slots(ws.gpu_slots, b.moe_w + size_t(base) * K, rows, K, H, out + size_t(base) * H, accumulate, L.st);
      if (t) {
        CK(cudaEventRecord(ws.timer[4], L.st));
        CK(cudaEventSynchronize(ws.timer[4]));
        auto us = [&](int a, int b) {
          float ms = 0;
          CK(cudaEventElapsedTime(&ms, ws.timer[a], ws.timer[b]));
          return double(ms) * 1000;
        };
        t->hot_gpu += us(0, 1);
        if (!ws.cold.empty()) t->upload += us(2, 3);
        t->reduce += us(3, 4);
      }
    }
  } catch (...) {
    if (ws.staging.stream()) cudaStreamSynchronize(ws.staging.stream());
    cudaStreamSynchronize(L.st);  // any queued DMA must end before a second session can own the staging buffers
    throw;
  }
}

}  // namespace flyx::cuda
