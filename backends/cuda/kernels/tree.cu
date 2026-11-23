// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// After a tree round's accept (docs/design/tree-verification.md): the accepted path's rows moved to rows 0 .. n, so
// the commit, the drafters' context and the next rounds see what a chain round leaves.  A path's rows come in order
// (taken[d] >= d), so moving them in that order reads every row before anything lands on it.
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

// grid (ceil(most_row_bytes / 16 / 128), caches); block 128: a thread one 16-byte word of a position's keys and values.
__global__ void k_compact_kv(const KvRows* table, const DevParams* P, const TreeRows* tree) {
  pdl_trigger();
  if (!tree_on(tree) || P->noop) return;
  const KvRows c = table[blockIdx.y];
  const int i = blockIdx.x * blockDim.x + threadIdx.x, words = c.row_bytes / 16, base = P->start;
  uint4* K = static_cast<uint4*>(const_cast<void*>(c.where.k));
  uint4* V = static_cast<uint4*>(const_cast<void*>(c.where.v));
  for (int d = 1; d <= tree->n_taken; ++d) {
    const int from = c.where.slot_of(base + tree->taken[d]), to = c.where.slot_of(base + d);
    if (from == to) continue;
    if (i < words) {
      K[size_t(to) * words + i] = K[size_t(from) * words + i];
      V[size_t(to) * words + i] = V[size_t(from) * words + i];
    }
    if (c.where.fp8() && i < c.nkv) {
      c.where.kscales(to)[size_t(to) * c.nkv + i] = c.where.kscales(from)[size_t(from) * c.nkv + i];
      c.where.vscales(to)[size_t(to) * c.nkv + i] = c.where.vscales(from)[size_t(from) * c.nkv + i];
    }
  }
}

// grid ceil(words / 256); block 256.
__global__ void k_compact_rows(uint32_t* rows, size_t ld, int words, const DevParams* P, const TreeRows* tree) {
  pdl_trigger();
  if (!tree_on(tree) || P->noop) return;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= words) return;
  for (int d = 1; d <= tree->n_taken; ++d) {
    const int from = tree->taken[d];
    if (from != d) rows[size_t(d) * ld + i] = rows[size_t(from) * ld + i];
  }
}

}  // namespace

void compact_kv(const KvRows* table, int n, int most_row_bytes, const DevParams* P, const TreeRows* tree,
                cudaStream_t st) {
  if (n <= 0) return;
  k_compact_kv<<<dim3(ceil_div(most_row_bytes / 16, 128), n), 128, 0, st>>>(table, P, tree);
}

void compact_rows(void* rows, size_t ld_words, int words, const DevParams* P, const TreeRows* tree, cudaStream_t st) {
  if (!rows || words <= 0) return;
  k_compact_rows<<<ceil_div(words, 256), 256, 0, st>>>(static_cast<uint32_t*>(rows), ld_words, words, P, tree);
}

}  // namespace flyx::cuda
