// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/layer_ring.h"

#include "backends/cuda/loader/layout.h"

namespace flyx::cuda {
std::vector<char> streamed_blocks(const flyx_model_desc& d, const std::vector<char>& draft, int resident) {
  if (resident == INT_MAX) return {};
  std::vector<char> result(size_t(d.n_blocks), 0);
  int layer = -1;
  for (int i = 0; i < d.n_blocks; ++i) {
    if (draft[size_t(i)]) continue;
    const auto kind = d.blocks[i].kind;
    if (kind == FLYX_BLOCK_ATTN || kind == FLYX_BLOCK_GDN)
      ++layer;
    else if (kind != FLYX_BLOCK_MLP)
      continue;
    result[size_t(i)] = layer >= resident;
  }
  // DV's CONTEXT nodes execute the target projections directly.
  for (int i = 0; i < d.n_blocks; ++i) {
    const auto& b = d.blocks[i];
    if (b.kind != FLYX_BLOCK_DV_LAYER) continue;
    for (int base : {b.dv_layer.attn, b.dv_layer.mlp}) result.at(size_t(base)) = 0;
  }
  if (std::none_of(result.begin(), result.end(), [](char streamed) { return streamed; })) result.clear();
  return result;
}

// Size the same homogeneous runs as WeightLoader::lin_stack. A stack with a floating member expands to BF16.
StreamEstimate estimate_stream(const flyx_model_desc& d, const std::vector<char>& draft,
                               const BackendOptions& options) {
  StreamEstimate out;
  const auto& stream = out.blocks = streamed_blocks(d, draft, options.gpu_layers);
  if (stream.empty()) return out;
  out.host_weight.resize(size_t(d.n_weights));
  std::vector<char> resident(size_t(d.n_weights), 0);
  size_t next = 0, slot[2]{};
  for (int i = 0; i < d.n_blocks; ++i) {
    const auto& b = d.blocks[i];
    size_t image = 0;
    auto plane = [&](size_t bytes) {
      if (bytes) image = size_add(checkpoint_align(image), bytes);
    };
    auto stack = [&](std::initializer_list<int> weights, size_t expected_rows, size_t expected_cols) {
      bool packed = options.packed_blocks;
      size_t nrows = 0, cols = 0;
      for (int w : weights) {
        if (w < 0 || w >= d.n_weights || !d.weights[w].t) throw std::runtime_error("invalid streamed linear weight");
        check_linear_payload(d.weights[w]);
        const auto& t = *d.weights[w].t;
        if (t.ndim != 2 || t.shape[0] <= 0 || t.shape[1] <= 0 || (cols && cols != size_t(t.shape[1])))
          throw std::runtime_error("invalid streamed linear dimensions");
        (stream[size_t(i)] ? out.host_weight : resident)[size_t(w)] = 1;
        cols = size_t(t.shape[1]);
        nrows = size_add(nrows, size_t(t.shape[0]));
        if (nrows > INT_MAX) throw std::runtime_error("streamed stack exceeds row limit");
        packed = packed && blocks_supported(t.dtype, int(cols));
      }
      if (nrows != expected_rows || cols != expected_cols || expected_rows > INT_MAX || expected_cols > INT_MAX)
        throw std::runtime_error("streamed projection does not match block");
      if (!stream[size_t(i)]) return;
      if (!packed) {
        const auto dtype = d.weights[*weights.begin()].t->dtype;
        const size_t values = size_mul(nrows, cols);
        auto fp4 = [](int type) { return type == FLYX_DT_NVFP4 || type == FLYX_DT_GGML_NVFP4; };
        for (int w : weights)
          if (fp4(d.weights[w].t->dtype) != fp4(dtype)) throw std::runtime_error("mixed FP4 formats in layer stack");
        if (fp4(dtype)) {
          plane(values / 2);
          plane(values / 16);
          plane(nrows * sizeof(float));
        } else
          plane(size_mul(values, sizeof(bf16)));
        return;
      }
      for (auto it = weights.begin(); it != weights.end();) {
        const int dtype = d.weights[*it].t->dtype;
        const auto fmt = blocks_layout(dtype);
        size_t rows = 0;
        do {
          rows = size_add(rows, size_t(d.weights[*it++].t->shape[0]));
        } while (it != weights.end() && d.weights[*it].t->dtype == dtype);
        const size_t blocks = size_mul(rows, cols / size_t(fmt.values));
        plane(size_mul(blocks, fmt.stride));
        plane(size_mul(blocks, fmt.scales * sizeof(uint16_t)));
      }
    };
    if (b.kind == FLYX_BLOCK_ATTN) {
      const auto& a = b.attn;
      const size_t q = size_mul(size_t(a.n_heads), size_t(a.head_dim));
      const size_t kv = size_mul(size_t(a.n_kv), size_t(a.head_dim));
      stack({a.q, a.k, a.v}, size_add(size_mul(a.gate == FLYX_GATE_NONE ? 1 : 2, q), size_mul(2, kv)), a.hidden);
      stack({a.o}, a.hidden, q);
    } else if (b.kind == FLYX_BLOCK_GDN) {
      const auto& g = b.gdn;
      const size_t k = size_mul(size_t(g.nk), size_t(g.dk)), v = size_mul(size_t(g.nv), size_t(g.dv));
      stack({g.qkv, g.z, g.b, g.a}, size_mul(2, size_add(k, size_add(v, size_t(g.nv)))), g.hidden);
      stack({g.out}, g.hidden, v);
    } else if (b.kind == FLYX_BLOCK_MLP) {
      stack({b.mlp.gate, b.mlp.up}, size_mul(2, size_t(b.mlp.inter)), b.mlp.hidden);
      stack({b.mlp.down}, b.mlp.hidden, b.mlp.inter);
    } else if (b.kind == FLYX_BLOCK_EMBED)
      resident.at(size_t(b.embed.table)) = 1;
    else if (b.kind == FLYX_BLOCK_HEAD)
      resident.at(size_t(b.head.w)) = 1;
    if (image) {
      auto& peak = slot[next++ % 2];
      peak = std::max(peak, image);
      out.host_bytes = size_add(out.host_bytes, image);
    }
  }
  for (size_t w = 0; w < resident.size(); ++w)
    if (resident[w]) out.host_weight[w] = 0;
  out.ring_bytes = size_add(slot[0], slot[1]);
  return out;
}

void LayerRing::finish_samples() {
  if (ring_.stream() && cudaStreamSynchronize(ring_.stream()) != cudaSuccess) return;
  for (auto& image : images) collect(image);
}

void LayerRing::archive(flyx_bmodel* m, int index) {
  auto& image = images.at(size_t(index));
  image.slot = int(next++ % 2);
  struct Plane {
    const void* src;
    size_t offset, bytes;
  };
  std::vector<Plane> planes;
  auto plane = [&](auto& ptr, size_t bytes) {
    if (!ptr) return;
    auto it = m->allocs.find(const_cast<void*>(static_cast<const void*>(ptr)));
    if (it == m->allocs.end() || it->second != bytes)
      throw std::runtime_error("streamed layer plane differs from its allocation");
    size_t offset = 0;
    const auto same = std::find_if(planes.begin(), planes.end(), [&](const Plane& p) { return p.src == ptr; });
    if (same == planes.end()) {
      offset = checkpoint_align(image.bytes);
      image.bytes = size_add(offset, bytes);
      planes.push_back({ptr, offset, bytes});
    } else {
      if (bytes != same->bytes) throw std::runtime_error("inconsistent shared layer plane size");
      offset = same->offset;
    }
    image.fixups.emplace_back([field = &ptr, offset](uint8_t* base) {
      using Pointer = std::remove_reference_t<decltype(*field)>;
      *field = reinterpret_cast<Pointer>(base + offset);
    });
  };
  auto linear = [&](Lin& l, int rows, int cols) {
    const size_t n = size_mul(size_t(rows), size_t(cols));
    plane(l.w, size_mul(n, sizeof(bf16)));
    plane(l.q, n / 2);
    plane(l.s, n / 16);
    plane(l.rs, size_t(rows) * sizeof(float));
    for (auto& b : l.blk) {
      const auto fmt = blocks_layout(b.dtype);
      const size_t blocks = size_mul(size_t(b.rows), size_t(cols / fmt.values));
      plane(b.data, size_mul(blocks, fmt.stride));
      plane(b.scales, size_mul(blocks, fmt.scales * sizeof(uint16_t)));
    }
  };
  Block& block = m->blocks.at(size_t(index));
  if (auto a = std::get_if<AttnBlock>(&block)) {
    linear(a->w_in, a->n_in(), a->hidden);
    linear(a->w_out, a->hidden, a->nh * a->hd);
  } else if (auto g = std::get_if<GdnBlock>(&block)) {
    linear(g->w_in, g->n_in(), g->hidden);
    linear(g->w_out, g->hidden, g->v_width());
  } else {
    auto& f = std::get<MlpBlock>(block);
    linear(f.w_gu, 2 * f.inter, f.hidden);
    linear(f.w_down, f.hidden, f.inter);
  }
  if (bounce_) {
    image.pageable.resize(image.bytes);
    image.host = image.pageable.data();
    m->host_bytes = size_add(m->host_bytes, image.bytes);
  } else {
    image.host = m->host_pinned<uint8_t>(image.bytes);
  }
  for (const auto& p : planes) {
    CK(cudaMemcpy(image.host + p.offset, p.src, p.bytes, cudaMemcpyDeviceToHost));
    m->release(const_cast<void*>(p.src), p.bytes);
  }
  capacities[image.slot] = std::max(capacities[image.slot], image.bytes);
  total = size_add(total, image.bytes);
}

void LayerRing::allocate(flyx_bmodel* m) {
  ring_.allocate(m, capacities, bounce_);
  const size_t ring_bytes = size_add(capacities[0], capacities[1]);
  for (auto& image : images) {
    for (auto& fix : image.fixups) fix(ring_.device(image.slot));
    image.fixups.clear();
    if (image.host && m->verbose) {
      image.sample_start.create(cudaEventDefault);
      image.sample_end.create(cudaEventDefault);
    }
  }
  if (m->verbose)
    std::fprintf(stderr, "flyx-cuda: layer streaming: %zu host bytes per full pass, %zu device slot bytes\n", total,
                 ring_bytes);
}

// Time only completed samples. Overlapping submissions may skip a sample without making the host wait.
bool LayerRing::collect(Image& image) {
  if (!image.sample_pending) return true;
  float elapsed = 0;
  if (cudaEventElapsedTime(&elapsed, image.sample_start, image.sample_end) != cudaSuccess) return false;
  auto& phase = stats_.copies[image.sample_round];
  phase.milliseconds += elapsed;
  phase.bytes += image.bytes;
  ++phase.samples;
  image.sample_pending = false;
  return true;
}

StagingRing<2>::Lease LayerRing::lease_node(int block, bool round, cudaStream_t compute) {
  std::lock_guard lock(stats_mutex_);
  auto& image = images.at(size_t(block));
  const bool sample = image.sample_start && collect(image);
  if (sample) {
    image.sample_round = round;
    image.sample_pending = true;
  }
  return {ring_,
          size_t(image.slot),
          image.host,
          image.bytes,
          compute,
          sample ? cudaEvent_t(image.sample_start) : nullptr,
          sample ? cudaEvent_t(image.sample_end) : nullptr};
}
}  // namespace flyx::cuda
