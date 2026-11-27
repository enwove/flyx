// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/cold_experts.h"

#include "backends/cuda/backend.h"
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
  const size_t count = size_t(w.t->shape[0] - hot), nrows = size_mul(count, size_t(rows));
  if (w.scale2) {
    size_t n = 1;
    for (int i = 0; i < w.scale2->ndim; ++i) n = size_mul(n, size_t(w.scale2->shape[i]));
    if (n != 1 && n != size_t(w.t->shape[0])) throw std::runtime_error("invalid cold expert scale count");
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

void ColdMatrix::prepare(float* x, bool vector_nvfp4) const {
  if (q8) {
    for (int i = 0; i < cols; i += 32) {
      float max_abs = 0;
      for (int j = 0; j < 32; ++j) max_abs = std::max(max_abs, std::fabs(x[i + j]));
      const float d = max_abs / 127.f;
      for (int j = 0; j < 32; ++j) x[i + j] = d > 0 ? std::nearbyint(x[i + j] / d) * d : 0;
    }
  } else if (a16h) {
    ops::round_a16h(x, cols);
  } else if (nvfp4 && !vector_nvfp4) {
    for (int i = 0; i < cols; ++i) x[i] = float(ops::round_bf16(x[i]));
  }
}

float ColdMatrix::product(int expert, int row, const float* x) const {
  size_t e = size_t(expert - first);
  return cold_dot(dtype, data.data() + (e * rows + row) * stride, x, cols) * (scales.empty() ? 1.f : scales[e]);
}

struct ColdWorkspace {
  RowPool pool{RowPool::default_threads()};
  std::mutex mutex;
  int* ids;
  float *input, *cpu_slots, *gpu_slots, *ones;
  std::vector<float> act;
  std::vector<int> cold;
  cudaEvent_t ready = nullptr;
  bool verbose = false;
  uint64_t visits = 0;
  double route_seconds = 0;
  ~ColdWorkspace() {
    if (ready) cudaEventDestroy(ready);
    if (verbose && visits)
      std::fprintf(stderr, "flyx-cuda: hybrid expert route/activation wait %.1f us per visit (%llu visits)\n",
                   route_seconds * 1e6 / visits, static_cast<unsigned long long>(visits));
  }
};

void init_cold_experts(flyx_bmodel* m) {
  if (m->opt.gpu_experts == INT_MAX) return;
  ColdShape shape;
  for (const Block& b : m->blocks) {
    auto f = std::get_if<MoeBlock>(&b);
    if (f && f->cold_gu) shape.include(f->hidden, f->inter, f->top_k);
  }
  if (!shape.pairs) return;
  auto w = std::make_shared<ColdWorkspace>();
  w->verbose = m->verbose;
  w->ids = m->host_pinned<int>(shape.pairs);
  w->input = m->host_pinned<float>(shape.inputs);
  w->cpu_slots = m->host_pinned<float>(shape.outputs);
  w->gpu_slots = m->alloc<float>(shape.outputs);
  w->ones = m->alloc<float>(shape.pairs);
  std::vector<float> ones(shape.pairs, 1.f);
  CK(cudaMemcpy(w->ones, ones.data(), shape.pairs * sizeof(float), cudaMemcpyHostToDevice));
  w->act.resize(shape.activations);
  w->cold.reserve(shape.pairs);
  m->host_bytes += shape.activations * sizeof(float) + shape.pairs * sizeof(int);
  CK(cudaEventCreateWithFlags(&w->ready, cudaEventDisableTiming));
  m->cold = std::move(w);
  m->opt.graphs = false;
}

void hybrid_experts(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate, bool fused) {
  auto& ws = *ps.s->m->cold;
  std::lock_guard lock(ws.mutex);
  const Launch& L = ps.v.L;
  const auto& b = ps.v.b;
  int H = f.hidden, I = f.inter, K = f.top_k;
  const bool vec = opts(L).moe_vec;
  // A different session can own the previous visit. Order reuse of both DMA buffers and device slots on its event.
  CK(cudaStreamWaitEvent(L.st, ws.ready));
  ++ws.visits;
  try {
    for (int base = 0; base < ps.rows; base += ColdShape::chunk) {
      int rows = std::min(ColdShape::chunk, ps.rows - base);
      auto* ids = b.moe_idx + size_t(base) * K;
      float* input = b.h + size_t(base) * H;
      auto start = ws.verbose ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
      CK(cudaMemcpyAsync(ws.ids, ids, rows * K * sizeof(int), cudaMemcpyDeviceToHost, L.st));
      CK(cudaMemcpyAsync(ws.input, input, size_t(rows) * H * sizeof(float), cudaMemcpyDeviceToHost, L.st));
      CK(cudaEventRecord(ws.ready, L.st));
      CK(cudaEventSynchronize(ws.ready));
      if (ws.verbose)
        ws.route_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      ws.cold.clear();
      for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < K;) {
          int slot = r * K + k;
          int e = ws.ids[slot];
          if (e < 0 || e >= f.n_experts) throw std::runtime_error("invalid MoE router expert ID");
          if (e >= f.cold_gu->first) {
            ws.cold.push_back(slot);
            ++k;
            continue;
          }
          // Adjacent hot slots share the input and a gate/up launch. Each down row writes its unweighted slot.
          int count = 1;
          while (k + count < K && ws.ids[slot + count] >= 0 && ws.ids[slot + count] < f.cold_gu->first) ++count;
          moe_gate_up(input + size_t(r) * H, H, f.gate_up, ids + slot, 1, count, I, H, f.situ_beta, f.situ_linear_beta,
                      fused, fused ? b.moe_act : b.moe_gu, fused ? I : 2 * I, sc, L);
          if (!fused) situ_rows(b.moe_gu, 2 * I, I, f.situ_beta, f.situ_linear_beta, b.moe_act, I, count, L.st);
          moe_down(b.moe_act, I, f.down, ids + slot, ws.ones, count, 1, H, I, ws.gpu_slots + size_t(slot) * H, H, false,
                   sc, L);
          k += count;
        }
      }
      if (!ws.cold.empty()) {
        for (int r = 0; r < rows; ++r) f.cold_gu->prepare(ws.input + size_t(r) * H, vec);
        ws.pool.run(ws.cold.size() * I, [&](size_t a, size_t end) {
          for (size_t i = a; i < end; ++i) {
            int slot = ws.cold[i / I], j = int(i % I), e = ws.ids[slot];
            const float* x = ws.input + size_t(slot / K) * H;
            float g = f.cold_gu->product(e, j, x), u = f.cold_gu->product(e, I + j, x);
            ws.act[size_t(slot) * I + j] =
                float(f.situ_beta > 0 ? ops::situ(g, u, f.situ_beta, f.situ_linear_beta) : ops::silu(g) * u);
          }
        });
        for (int slot : ws.cold) f.cold_down->prepare(ws.act.data() + size_t(slot) * I, vec);
        ws.pool.run(ws.cold.size() * H, [&](size_t a, size_t end) {
          for (size_t i = a; i < end; ++i) {
            int slot = ws.cold[i / H], j = int(i % H);
            ws.cpu_slots[size_t(slot) * H + j] =
                f.cold_down->product(ws.ids[slot], j, ws.act.data() + size_t(slot) * I);
          }
        });
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
      moe_sum_slots(ws.gpu_slots, b.moe_w + size_t(base) * K, rows, K, H, out + size_t(base) * H, accumulate, L.st);
    }
    CK(cudaEventRecord(ws.ready, L.st));
  } catch (...) {
    cudaStreamSynchronize(L.st);  // any queued DMA must end before a second session can own the staging buffers
    throw;
  }
}
}  // namespace flyx::cuda
