// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/cold_experts.h"

#include "backends/cuda/backend.h"
#include "backends/cuda/kernels/moe_bank.h"
#include "backends/cuda/runtime/row_pool.h"
#include "ops/activation/ref.h"

namespace flyx::cuda {

ColdMatrix::ColdMatrix(const flyx_weight& w, int hot, const ExpertLayout& layout)
    : first(hot),
      rows(int(w.t->shape[1])),
      cols(int(w.t->shape[2])),
      dtype(w.t->dtype),
      q8(layout.format == ExpertLayout::ggml_blocks && w.activations == FLYX_ACT_Q8),
      a16h(layout.format == ExpertLayout::ggml_blocks && !q8),
      nvfp4(layout.format == ExpertLayout::nvfp4) {
  if (q8) quant_dot = cold_q8_kernel(dtype);
  const size_t count = size_t(w.t->shape[0] - hot), nrows = size_mul(count, size_t(rows));
  if (w.scale2) {
    if (!w.scale2->data) throw std::runtime_error("missing cold expert scales");
    size_t n = 1;
    for (int i = 0; i < w.scale2->ndim; ++i) n = size_mul(n, size_t(w.scale2->shape[i]));
    if (n != 1 && n != size_t(w.t->shape[0])) throw std::runtime_error("invalid cold expert scale count");
    scales.reserve(count);
    for (size_t i = 0; i < count; ++i) scales.push_back(float(ops::load(w.scale2, n == 1 ? 0 : hot + i)));
  }
  if (layout.format == ExpertLayout::bf16_rows && dtype != FLYX_DT_BF16) {
    dtype = FLYX_DT_BF16;
    stride = size_mul(size_t(cols), 2);
    data.resize(size_mul(nrows, stride));
    const size_t start = size_t(hot) * rows * cols;
    for (size_t i = 0; i < data.size() / 2; ++i) {
      uint16_t b = float_to_bf16(float(ops::load(w.t, start + i)));
      std::memcpy(data.data() + 2 * i, &b, 2);
    }
  } else {
    stride = dtype == FLYX_DT_BF16 ? size_mul(size_t(cols), 2)
                                   : size_mul(size_t(cols / flyx_block_values(dtype)), size_t(flyx_block_bytes(dtype)));
    if (size_mul(layout.rows, stride) != w.t->nbytes) throw std::runtime_error("invalid cold expert payload size");
    const auto* src = static_cast<const uint8_t*>(w.t->data) + size_t(hot) * rows * stride;
    data.assign(src, src + size_mul(nrows, stride));
    if (nvfp4) {
      for (size_t offset = 0; offset < data.size(); offset += 36)
        for (int g = 0; g < 4; ++g) {
          auto& scale = data[offset + g];
          if (scale == 255) throw std::runtime_error("invalid NVFP4 cold expert scale");
          scale = scale == 127 ? 0 : uint8_t(scale & 127);
        }
    }
  }
}

void pack_expert_nvfp4(const uint8_t* in, size_t blocks, uint8_t* qo, uint8_t* so) {
  for (size_t k = 0; k < blocks; ++k, in += 36) {
    const uint8_t* d = in;
    const uint8_t* qs = in + 4;
    for (int g = 0; g < 4; ++g) {
      if (d[g] == 0xFF)
        throw std::runtime_error("an NVFP4 scale of 0xFF (no e4m3 holds it)");
      so[4 * k + size_t(g)] = d[g] == 0x7F ? 0 : uint8_t(d[g] & 0x7F);
      auto code = [&](int l) { return l < 8 ? qs[8 * g + l] & 15 : qs[8 * g + l - 8] >> 4; };
      for (int i = 0; i < 8; ++i)
        qo[32 * k + 8 * size_t(g) + size_t(i)] = uint8_t(code(2 * i) | code(2 * i + 1) << 4);
    }
  }
}

void ColdMatrix::pack_banks(const ExpertLayout& layout, size_t all) {
  const size_t cold = all - first;
  bank = ExpertBankLayout::from(layout, all, cold);
  archive.resize(bank.archive_bytes(cold));
  size_t offset = 0;
  for (size_t e = 0; e < cold; e += bank.count) {
    const size_t n = std::min(bank.count, cold - e);
    uint8_t* dst = archive.data() + offset;
    const uint8_t* src = data.data() + e * size_t(rows) * stride;
    auto* scale = dst + bank.scale_offset(n);
    if (nvfp4)
      pack_expert_nvfp4(src, n * size_t(rows) * (cols / 64), dst, scale);
    else if (layout.format == ExpertLayout::ggml_blocks)
      blocks_pack(dtype, src, n * size_t(rows) * (cols / blocks_layout(dtype).values),
                  dst, reinterpret_cast<uint16_t*>(scale));
    else
      std::memcpy(dst, src, n * bank.payload);
    if (bank.gains) {
      auto* g = reinterpret_cast<float*>(dst + bank.gain_offset(n));
      if (scales.empty()) std::fill_n(g, n, 1.f);
      else std::copy_n(scales.data() + e, n, g);
    }
    offset += bank.bytes(n);
  }
}

void ColdMatrix::prepare(float* x, bool vector_nvfp4) const {
  if (q8) {
    for (int i = 0; i < cols; i += 32) {
      float max_abs = 0;
      for (int j = 0; j < 32; ++j) max_abs = std::max(max_abs, std::fabs(x[i + j]));
      const float d = max_abs / 127.f;
      const bool scaled = d > 0 && std::isfinite(d);
      for (int j = 0; j < 32; ++j) x[i + j] = (scaled ? std::nearbyint(x[i + j] / d) : 0.f) * d;
    }
  } else if (a16h) {
    ops::round_a16h(x, cols);
  } else if (nvfp4 && !vector_nvfp4) {
    for (int i = 0; i < cols; ++i) x[i] = float(ops::round_bf16(x[i]));
  }
}

float ColdMatrix::product(int expert, int row, const float* x, const ColdQ8* quant) const {
  size_t e = size_t(expert - first);
  const uint8_t* weights = data.data() + (e * rows + row) * stride;
  const float dot = quant_dot ? quant_dot(weights, quant, cols) : cold_dot(dtype, weights, x, cols);
  return dot * (scales.empty() ? 1.f : scales[e]);
}

struct ExpertTimes {
  uint64_t rows = 0, visits = 0, cold = 0, pairs = 0, bytes = 0;
  double handoff = 0, cpu = 0, hot_gpu = 0, upload = 0, reduce = 0;
};

struct ColdWorkspace {
  explicit ColdWorkspace(size_t threads) : pool(threads, true) {}
  struct Slot {
    uint8_t *host = nullptr, *device = nullptr;
    cudaEvent_t uploaded = nullptr, consumed = nullptr;
  };
  Slot staging[2];
  cudaStream_t copy = nullptr;
  size_t next_slot = 0;
  RowPool pool;
  std::mutex mutex;
  int *ids, *resident_ids;
  float *input, *cpu_slots, *gpu_slots, *ones;
  std::vector<float> act;
  std::vector<ColdQ8> gu_q8, down_q8;
  std::vector<int> cold;
  cudaEvent_t ready = nullptr;
  bool verbose = false;
  bool profile = false;
  cudaEvent_t timer[5]{};
  std::map<std::pair<const MoeBlock*, bool>, ExpertTimes> times;
  uint64_t decode_slots = 0, decode_cold = 0;
  uint64_t visits = 0, chunks = 0, slots = 0, cpu_pairs = 0;
  double route_seconds = 0, cpu_seconds = 0;
  ~ColdWorkspace() {
    if (copy) cudaStreamSynchronize(copy);
    for (auto& slot : staging) {
      if (slot.consumed) cudaEventSynchronize(slot.consumed);
      if (slot.uploaded) cudaEventDestroy(slot.uploaded);
      if (slot.consumed) cudaEventDestroy(slot.consumed);
    }
    if (copy) cudaStreamDestroy(copy);
    if (ready) {
      cudaEventSynchronize(ready);
      cudaEventDestroy(ready);
    }
    for (auto event : timer)
      if (event) cudaEventDestroy(event);
    for (const auto& [key, t] : times) {
      if (!t.visits) continue;
      std::fprintf(stderr, "flyx-cuda: expert profile block=%p phase=%s visits=%llu rows=%llu "
                   "handoff=%.2f cpu=%.2f hot_gpu=%.2f h2d=%.2f reduce=%.2f us/visit "
                   "cold_reads=%.2f GB/s resident=%.2f%%\n",
                   static_cast<const void*>(key.first), key.second ? "round" : "prefill",
                   static_cast<unsigned long long>(t.visits), static_cast<unsigned long long>(t.rows),
                   t.handoff / t.visits, t.cpu / t.visits, t.hot_gpu / t.visits, t.upload / t.visits,
                   t.reduce / t.visits, t.cpu ? t.bytes / (t.cpu * 1e3) : 0.,
                   100. * (t.pairs - t.cold) / t.pairs);
    }
    if (verbose && decode_slots)
      std::fprintf(stderr, "flyx-cuda: decode expert cache hit rate %.2f%% (%llu/%llu routed slots)\n",
                   100. * (decode_slots - decode_cold) / decode_slots,
                   static_cast<unsigned long long>(decode_slots - decode_cold),
                   static_cast<unsigned long long>(decode_slots));
    if (verbose && visits)
      std::fprintf(stderr,
                   "flyx-cuda: hybrid expert route/activation wait %.1f us per visit, CPU work %.1f us per visit "
                   "(%llu visits, %llu chunks, %llu of %llu routed pairs on CPU, %zu pool threads)\n",
                   route_seconds * 1e6 / visits, cpu_seconds * 1e6 / visits, static_cast<unsigned long long>(visits),
                   static_cast<unsigned long long>(chunks), static_cast<unsigned long long>(cpu_pairs),
                   static_cast<unsigned long long>(slots), pool.threads());
  }
};

void init_cold_experts(flyx_bmodel* m) {
  if (m->opt.gpu_experts == INT_MAX) return;
  ColdShape shape;
  for (const Block& b : m->blocks) {
    auto f = std::get_if<MoeBlock>(&b);
    if (!f || !f->cold_gu) continue;
    shape.include(f->hidden, f->inter, f->top_k);
    for (const auto* mat : {f->cold_gu.get(), f->cold_down.get()})
      shape.bank_bytes = std::max(shape.bank_bytes, mat->bank.bytes(mat->bank.count));
  }
  if (!shape.pairs) return;
  const size_t threads = m->opt.expert_threads ? std::min<size_t>(size_t(m->opt.expert_threads),
                                                                  std::max(1u, std::thread::hardware_concurrency()))
                                               : RowPool::default_threads();
  auto w = std::make_shared<ColdWorkspace>(threads);
  w->verbose = m->verbose;
  w->profile = m->opt.profile;
  if (w->profile)
    for (auto& event : w->timer) CK(cudaEventCreate(&event));
  if (shape.bank_bytes) {
    CK(cudaStreamCreateWithFlags(&w->copy, cudaStreamNonBlocking));
    for (auto& slot : w->staging) {
      slot.device = m->alloc<uint8_t>(shape.bank_bytes);
      slot.host = m->host_pinned<uint8_t>(shape.bank_bytes);
      CK(cudaEventCreateWithFlags(&slot.uploaded, cudaEventDisableTiming));
      CK(cudaEventCreateWithFlags(&slot.consumed, cudaEventDisableTiming));
    }
  }
  w->resident_ids = m->alloc<int>(shape.pairs);
  w->ids = m->host_pinned<int>(shape.pairs);
  w->input = m->host_pinned<float>(shape.inputs);
  w->cpu_slots = m->host_pinned<float>(shape.outputs);
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
  CK(cudaEventCreateWithFlags(&w->ready, cudaEventDisableTiming));
  m->cold = std::move(w);
  m->opt.graphs = false;
}

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
                        accumulate, &sc, L, nullptr, 0, first);
        } else {
          float* gu = b.moe_gu + size_t(base) * K * 2 * I;
          moe_bank_gate_up(b.h + size_t(base) * H, H, mat, ids, rows, K, I, H, f.situ_beta, f.situ_linear_beta,
                           fused, fused ? act : gu, fused ? I : 2 * I, &sc, L, false, nullptr, first);
        }
        base += rows;
      }
    };
    size_t offset = 0;
    const auto& bank = cold.bank;
    for (int first = cold.first; first < f.n_experts;) {
      const int n = std::min<int>(int(bank.count), f.n_experts - first);
      auto& slot = ws.staging[ws.next_slot++ % 2];
      const size_t bytes = bank.bytes(n);
      CK(cudaEventSynchronize(slot.uploaded));
      std::memcpy(slot.host, cold.archive.data() + offset, bytes);
      CK(cudaStreamWaitEvent(ws.copy, slot.consumed));
      CK(cudaMemcpyAsync(slot.device, slot.host, bytes, cudaMemcpyHostToDevice, ws.copy));
      CK(cudaEventRecord(slot.uploaded, ws.copy));
      if (first == cold.first && resident.n_experts) launch(resident, 0);
      ExpertMat mat{};
      mat.N = cold.rows;
      mat.n_experts = n;
      auto* scales = slot.device + bank.scale_offset(n);
      if (cold.nvfp4) {
        mat.q = slot.device;
        mat.s = scales;
      } else if (cold.q8 || cold.a16h) {
        mat.blk = {slot.device, reinterpret_cast<uint16_t*>(scales), cold.dtype, n * cold.rows, cold.q8};
      } else {
        mat.w = reinterpret_cast<bf16*>(slot.device);
      }
      if (bank.gains) mat.rs = reinterpret_cast<float*>(slot.device + bank.gain_offset(n));
      CK(cudaStreamWaitEvent(L.st, slot.uploaded));
      launch(mat, first);
      CK(cudaEventRecord(slot.consumed, L.st));
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
  auto& ws = *ps.s->m->cold;
  std::lock_guard lock(ws.mutex);
  const Launch& L = ps.v.L;
  const auto& b = ps.v.b;
  int H = f.hidden, I = f.inter, K = f.top_k;
  const bool vec = opts(L).moe_vec;
  // A different session can own the previous visit. Order reuse of both DMA buffers and device slots on its event.
  CK(cudaStreamWaitEvent(L.st, ws.ready));
  auto* t = ws.profile ? &ws.times[{&f, ps.round}] : nullptr;
  try {
    if (!ps.round && ws.copy && sc && ps.rows >= std::max(2, ps.s->m->opt.expert_prefill_min)) {
      streamed_prefill(ws, ps, f, *sc, out, accumulate, fused);
      CK(cudaEventRecord(ws.ready, L.st));
      return;
    }
    if (t) {
      ++t->visits;
      t->rows += ps.rows;
    }
    ++ws.visits;
    for (int base = 0; base < ps.rows; base += ColdShape::chunk) {
      int rows = std::min(ColdShape::chunk, ps.rows - base);
      auto* ids = b.moe_idx + size_t(base) * K;
      float* input = b.h + size_t(base) * H;
      auto start = ws.verbose || t ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      CK(cudaMemcpyAsync(ws.ids, ids, rows * K * sizeof(int), cudaMemcpyDeviceToHost, L.st));
      CK(cudaMemcpyAsync(ws.input, input, size_t(rows) * H * sizeof(float), cudaMemcpyDeviceToHost, L.st));
      CK(cudaEventRecord(ws.ready, L.st));
      // Submit the router copy before queuing the independent GPU projections (WDDM batches stream commands).
      const auto query = cudaStreamQuery(L.st);
      if (query != cudaSuccess && query != cudaErrorNotReady) CK(query);
      if (t) CK(cudaEventRecord(ws.timer[0], L.st));
      if (f.gate_up.n_experts) {
        moe_resident_ids(ids, ws.resident_ids, rows * K, f.gate_up.n_experts, L.st);
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
      const auto submitted = cudaStreamQuery(L.st);
      if (submitted != cudaSuccess && submitted != cudaErrorNotReady) CK(submitted);
      CK(cudaEventSynchronize(ws.ready));
      if (ws.verbose || t) {
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ws.route_seconds += dt;
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
      if (ws.verbose) {
        if (ps.round) {
          ws.decode_slots += rows * K;
          ws.decode_cold += ws.cold.size();
        }
        ++ws.chunks;
        ws.slots += size_t(rows) * K;
        ws.cpu_pairs += ws.cold.size();
      }
      if (!ws.cold.empty()) {
        if (ws.verbose || t) start = std::chrono::steady_clock::now();
        int last_row = -1;
        for (int slot : ws.cold) {
          const int row = slot / K;
          if (row != last_row) {
            if (f.cold_gu->quant_dot)
              prepare_cold_q8(ws.input + size_t(row) * H, ws.gu_q8.data() + size_t(row) * (H / 32), H);
            else
              f.cold_gu->prepare(ws.input + size_t(row) * H, vec);
          }
          last_row = row;
        }
        ws.pool.run(ws.cold.size() * I, [&](size_t a, size_t end) {
          for (size_t i = a; i < end; ++i) {
            int slot = ws.cold[i / I], j = int(i % I), e = ws.ids[slot];
            const float* x = ws.input + size_t(slot / K) * H;
            const ColdQ8* quant = ws.gu_q8.data() + size_t(slot / K) * (H / 32);
            float g = f.cold_gu->product(e, j, x, quant), u = f.cold_gu->product(e, I + j, x, quant);
            ws.act[size_t(slot) * I + j] =
                float(f.situ_beta > 0 ? ops::situ(g, u, f.situ_beta, f.situ_linear_beta) : ops::silu(g) * u);
          }
        });
        for (int slot : ws.cold) {
          float* x = ws.act.data() + size_t(slot) * I;
          if (f.cold_down->quant_dot) prepare_cold_q8(x, ws.down_q8.data() + size_t(slot) * (I / 32), I);
          else f.cold_down->prepare(x, vec);
        }
        ws.pool.run(ws.cold.size() * H, [&](size_t a, size_t end) {
          for (size_t i = a; i < end; ++i) {
            int slot = ws.cold[i / H], j = int(i % H);
            ws.cpu_slots[size_t(slot) * H + j] =
                f.cold_down->product(ws.ids[slot], j, ws.act.data() + size_t(slot) * I,
                                     ws.down_q8.data() + size_t(slot) * (I / 32));
          }
        });
        if (ws.verbose || t) {
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          ws.cpu_seconds += dt;
          if (t) t->cpu += dt * 1e6;
        }
        if (t) CK(cudaEventRecord(ws.timer[2], L.st));
        for (size_t i = 0; i < ws.cold.size();) {
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
    CK(cudaEventRecord(ws.ready, L.st));
  } catch (...) {
    if (ws.copy) cudaStreamSynchronize(ws.copy);
    cudaStreamSynchronize(L.st);  // any queued DMA must end before a second session can own the staging buffers
    throw;
  }
}
}  // namespace flyx::cuda
