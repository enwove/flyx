// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__device__ int axis(int pair, MRoPE rope) {
  if (rope.interleaved) {
    if (pair % 3 == 1 && pair < 3 * rope.sec[1]) return 1;
    if (pair % 3 == 2 && pair < 3 * rope.sec[2]) return 2;
    return 0;
  }
  return pair < rope.sec[0] ? 0 : pair < rope.sec[0] + rope.sec[1] ? 1 : 2;
}

__device__ float prepare(float raw, float* sh, SparseIndex ix, const float* weights, const int* coords) {
  const int d = threadIdx.x;
  const float scale = rsqrtf(block_sum(raw * raw) / ix.dim + ix.eps);
  const float v = d < ix.dim ? raw * scale * (ix.offset + weights[d]) : 0.f;
  sh[d] = v;
  __syncthreads();
  if (d >= ix.rot) return v;
  const int half = ix.rot / 2, i = d % half;
  float c, s;
  rope_cs(coords[axis(i, ix.rope)], i, ix.rot, ix.theta, &c, &s);
  return d < half ? v * c - sh[i + half] * s : v * c + sh[i] * s;
}

__global__ void write_index(SparseIndex ix, float* proj, float* keys, const DevParams* p, const TreeRows* tree) {
  extern __shared__ float sh[];
  const int r = blockIdx.y, h = blockIdx.x, d = threadIdx.x;
  float* src = proj + (size_t(r) * (ix.heads + 1) + h) * ix.dim;
  const int pos = p->position + tree_depth(tree, r) + p->rope_delta;
  int coords[3];
  for (int a = 0; a < 3; ++a) coords[a] = ix.rope.pos_3d ? ix.rope.pos_3d[3 * r + a] : pos;
  if (h == ix.heads) {
    float* dst = keys + size_t(p->position + r) * (ix.dim + 3);
    if (d < ix.dim) dst[d] = src[d];
    if (d < 3) dst[ix.dim + d] = __int_as_float(coords[d]);
  } else {
    const float v = prepare(d < ix.dim ? src[d] : 0.f, sh, ix, ix.qn, coords);
    if (d < ix.dim) src[d] = v;
  }
}

__global__ void score_groups(SparseIndex ix, const float* proj, const float* keys, int first, float* scores, int stride,
                             const DevParams* p, const TreeRows* tree) {
  extern __shared__ float sh[];
  const int d = threadIdx.x, r = first + blockIdx.y, base = p->position;
  const int n = (base + tree_depth(tree, r) + 1) / ix.group;
  if (n <= ix.keep) return;
  const float* q = proj + size_t(r) * (ix.heads + 1) * ix.dim;
  for (int g = blockIdx.x; g < n; g += gridDim.x) {
    float sum = 0;
    int coords[3];
    for (int j = 0; j < ix.group; ++j) {
      const float* key = keys + size_t(tree_slot(tree, r, base, g * ix.group + j)) * (ix.dim + 3);
      if (d < ix.dim) sum += key[d];
      if (j == 0)
        for (int a = 0; a < 3; ++a) coords[a] = __float_as_int(key[ix.dim + a]);
    }
    const float k = prepare(sum / ix.group, sh, ix, ix.kn, coords);
    float score = 0;
    for (int h = 0; h < ix.heads; ++h) score += fmaxf(0.f, block_sum(d < ix.dim ? k * q[h * ix.dim + d] : 0.f));
    if (!d) scores[size_t(blockIdx.y) * stride + g] = score * rsqrtf(float(ix.dim));
    __syncthreads();
  }
}

// Find the kth nonnegative score with four radix passes. Ties go to the earliest group.
__global__ void select_groups(SparseIndex ix, const float* scores, int stride, int* selected, int first,
                              const DevParams* p, const TreeRows* tree) {
  const int t = threadIdx.x, r = first + blockIdx.x, n = (p->position + tree_depth(tree, r) + 1) / ix.group;
  if (n <= ix.keep) return;
  scores += size_t(blockIdx.x) * stride;
  selected += size_t(blockIdx.x) * ix.keep;
  __shared__ unsigned hist[256], prefix, mask, rank;
  if (!t) {
    prefix = 0;
    mask = 0;
    rank = ix.keep;
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[t] = 0;
    __syncthreads();
    for (int i = t; i < n; i += 256) {
      const unsigned key = __float_as_uint(scores[i]);
      if ((key & mask) == prefix) atomicAdd(hist + ((key >> shift) & 255), 1u);
    }
    __syncthreads();
    if (!t) {
      for (int b = 255; b >= 0; --b) {
        if (rank > hist[b])
          rank -= hist[b];
        else {
          prefix |= unsigned(b) << shift;
          break;
        }
      }
      mask |= 255u << shift;
    }
    __syncthreads();
  }
  if (!t) {
    int kept = 0;
    for (int i = 0; i < n && kept < ix.keep; ++i) {
      const unsigned key = __float_as_uint(scores[i]);
      if (key > prefix) selected[kept++] = i;
      if (key == prefix && rank) {
        selected[kept++] = i;
        --rank;
      }
    }
  }
}

__global__ void selected_attention(SparseIndex ix, const int* selected, const float* q, KvRef kv, int nh, int nkv,
                                   int hd, const float* gate, int ld, float* out, int first, const DevParams* p,
                                   const TreeRows* tree) {
  const int d = threadIdx.x, h = blockIdx.x, r = first + blockIdx.y, base = p->position;
  const int visible = base + tree_depth(tree, r) + 1, groups = visible / ix.group;
  const int complete = min(groups, ix.keep) * ix.group, count = complete + visible % ix.group;
  const size_t at = (size_t(r) * nh + h) * hd + d;
  const float query = d < hd ? q[at] : 0.f;
  float maximum = -INFINITY, denom = 0.f, output = 0.f;
  for (int i = 0; i < count; ++i) {
    int j = i;
    if (groups > ix.keep)
      j = i < complete ? selected[size_t(blockIdx.y) * ix.keep + i / ix.group] * ix.group + i % ix.group
                       : groups * ix.group + i - complete;
    const int pos = tree_slot(tree, r, base, j);
    const size_t cell = size_t(pos) * nkv + h / (nh / nkv), k = cell * hd + d;
    float kval = 0, vval = 0;
    if (d < hd) {
      if (kv.fp8()) {
        kval = e4m3_to_float(static_cast<uint8_t*>(kv.k)[k]) * kv.ks[cell];
        vval = e4m3_to_float(static_cast<uint8_t*>(kv.v)[k]) * kv.vs[cell];
      } else {
        kval = __half2float(static_cast<kv16*>(kv.k)[k]);
        vval = __half2float(static_cast<kv16*>(kv.v)[k]);
      }
    }
    const float z = block_sum(query * kval) * rsqrtf(float(hd)), next = fmaxf(maximum, z);
    const float a = expf(maximum - next), b = expf(z - next);
    output = output * a + b * vval;
    denom = denom * a + b;
    maximum = next;
  }
  if (d < hd) out[at] = (output / denom) * (gate ? 1.f / (1.f + expf(-gate[size_t(r) * ld + h * hd + d])) : 1.f);
}

__global__ void move_index(float* keys, int width, const DevParams* p, const TreeRows* tree) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (!tree_on(tree) || p->noop || d >= width) return;
  keys += size_t(p->start) * width + d;
  for (int i = 0; i <= tree->n_taken; ++i) keys[size_t(i) * width] = keys[size_t(tree->taken[i]) * width];
}
}  // namespace

void indexed_attention(SparseIndex ix, float* proj, float* keys, float* scores, int stride, int* selected,
                       const float* q, KvRef kv, int nh, int nkv, int hd, const float* gate, int ld, float* out,
                       int rows, const DevParams* p, const TreeRows* tree, cudaStream_t st) {
  const int threads = (ix.dim + 31) / 32 * 32;
  write_index<<<dim3(ix.heads + 1, rows), threads, threads * sizeof(float), st>>>(ix, proj, keys, p, tree);
  for (int first = 0; first < rows; first += kIndexTile) {
    const int n = std::min(kIndexTile, rows - first);
    score_groups<<<dim3(std::min(stride, 128), n), threads, threads * sizeof(float), st>>>(ix, proj, keys, first,
                                                                                           scores, stride, p, tree);
    select_groups<<<n, 256, 0, st>>>(ix, scores, stride, selected, first, p, tree);
    selected_attention<<<dim3(nh, n), (hd + 31) / 32 * 32, 0, st>>>(ix, selected, q, kv, nh, nkv, hd, gate, ld, out,
                                                                    first, p, tree);
  }
}

void compact_index(float* keys, int width, const DevParams* p, const TreeRows* tree, cudaStream_t st) {
  move_index<<<(width + 255) / 256, 256, 0, st>>>(keys, width, p, tree);
}
}  // namespace flyx::cuda
