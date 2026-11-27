// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/layer_ring.h"

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
  if (layer < resident) result.clear();
  return result;
}

// Size the same homogeneous runs as WeightLoader::lin_stack. A stack with a floating member expands to BF16.
StreamEstimate estimate_stream(const flyx_model_desc& d, const std::vector<char>& draft,
                               const BackendOptions& options) {
  StreamEstimate out;
  const auto stream = streamed_blocks(d, draft, options.gpu_layers);
  if (stream.empty()) return out;
  out.host_weight.resize(size_t(d.n_weights));
  std::vector<char> resident(size_t(d.n_weights), 0);
  size_t next = 0, slot[2]{};
  for (int i = 0; i < d.n_blocks; ++i) {
    const auto& b = d.blocks[i];
    size_t image = 0;
    auto plane = [&](size_t bytes) { image = size_add(checkpoint_align(image), bytes); };
    auto stack = [&](std::initializer_list<int> weights) {
      bool packed = options.packed_blocks;
      size_t nrows = 0, cols = 0;
      for (int w : weights) {
        if (w < 0 || w >= d.n_weights || !d.weights[w].t) throw std::runtime_error("invalid streamed linear weight");
        const auto& t = *d.weights[w].t;
        if (t.ndim != 2 || t.shape[0] <= 0 || t.shape[1] <= 0 || (cols && cols != size_t(t.shape[1])))
          throw std::runtime_error("invalid streamed linear dimensions");
        (stream[size_t(i)] ? out.host_weight : resident)[size_t(w)] = 1;
        cols = size_t(t.shape[1]);
        nrows = size_add(nrows, size_t(t.shape[0]));
        packed = packed && blocks_supported(t.dtype, int(cols));
      }
      if (!stream[size_t(i)]) return;
      if (!packed) {
        const auto dtype = d.weights[*weights.begin()].t->dtype;
        const size_t values = size_mul(nrows, cols);
        if (dtype == FLYX_DT_NVFP4 || dtype == FLYX_DT_GGML_NVFP4) {
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
      stack({b.attn.q, b.attn.k, b.attn.v});
      stack({b.attn.o});
    } else if (b.kind == FLYX_BLOCK_GDN) {
      stack({b.gdn.qkv, b.gdn.z, b.gdn.b, b.gdn.a});
      stack({b.gdn.out});
    } else if (b.kind == FLYX_BLOCK_MLP) {
      stack({b.mlp.gate, b.mlp.up});
      stack({b.mlp.down});
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

LayerRing::~LayerRing() {
  if (copy) cudaStreamSynchronize(copy);
  for (auto& slot : slots) {
    if (slot.consumed) cudaEventSynchronize(slot.consumed);
    if (slot.consumed) cudaEventDestroy(slot.consumed);
    if (slot.ready) cudaEventDestroy(slot.ready);
  }
  if (copy) cudaStreamDestroy(copy);
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
  image.host = m->host_pinned<uint8_t>(image.bytes);
  for (const auto& p : planes) {
    CK(cudaMemcpy(image.host + p.offset, p.src, p.bytes, cudaMemcpyDeviceToHost));
    m->release(const_cast<void*>(p.src), p.bytes);
  }
  slots[image.slot].capacity = std::max(slots[image.slot].capacity, image.bytes);
  total = size_add(total, image.bytes);
}

void LayerRing::allocate(flyx_bmodel* m) {
  CK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
  size_t ring_bytes = 0;
  for (auto& slot : slots) {
    if (!slot.capacity) continue;
    slot.device = m->alloc<uint8_t>(slot.capacity);
    CK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&slot.consumed, cudaEventDisableTiming));
    ring_bytes += slot.capacity;
  }
  for (auto& image : images) {
    for (auto& fix : image.fixups) fix(slots[image.slot].device);
    image.fixups.clear();
  }
  if (m->verbose)
    std::fprintf(stderr, "flyx-cuda: layer streaming: %zu host bytes per full pass, %zu device slot bytes\n", total,
                 ring_bytes);
}

// Both waits are device dependencies. The caller holds the ring mutex only while submitting a block.
void LayerRing::begin(int block, cudaStream_t compute) {
  const auto& image = images[size_t(block)];
  auto& slot = slots[image.slot];
  CK(cudaStreamWaitEvent(copy, slot.consumed));
  CK(cudaMemcpyAsync(slot.device, image.host, image.bytes, cudaMemcpyHostToDevice, copy));
  CK(cudaEventRecord(slot.ready, copy));
  CK(cudaStreamWaitEvent(compute, slot.ready));
}
void LayerRing::end(int block, cudaStream_t compute) {
  CK(cudaEventRecord(slots[images[size_t(block)].slot].consumed, compute));
}
}  // namespace flyx::cuda
