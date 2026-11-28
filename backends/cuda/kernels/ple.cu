// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__global__ void hash_rows(flyx_ple_block b, PleWindow w, const int* tokens, int* ids, const DevParams* p,
                          const TreeRows* tree) {
  const int r = blockIdx.y, head = blockIdx.x * blockDim.x + threadIdx.x, heads = (b.ngram - 1) * b.heads_per_order;
  if (head >= heads) return;
  const int depth = tree_depth(tree, r), width = b.streams * b.hidden, hist = (b.taps - 1) * b.dilation;
  const float* old = w.win + size_t(slot_of(p, kSlotCur)) * w.w_stride;
  uint64_t hash = uint64_t(tokens[r]) * b.multipliers[0];
  bool ended = false;
  for (int i = 1; i <= head / b.heads_per_order + 1; ++i) {
    const int at = depth - i;
    int id = b.eos;
    if (!ended && p->position + at >= 0)
      id = at < 0 ? __float_as_int(old[size_t(hist + at) * (width + 1) + width]) : tokens[tree_row(tree, r, at)];
    ended = ended || id == b.eos;
    hash ^= uint64_t(id) * b.multipliers[i];
  }
  ids[size_t(r) * heads + head] = int(b.offsets[head] + hash % b.moduli[head]);
}

__device__ float unpack(const uint8_t* src, int dtype, int d) {
  if (dtype == FLYX_DT_F32) return reinterpret_cast<const float*>(src)[d];
  if (dtype == FLYX_DT_F16) return __half2float(reinterpret_cast<const __half*>(src)[d]);
  if (dtype == FLYX_DT_BF16) return __bfloat162float(reinterpret_cast<const bf16*>(src)[d]);
  const int lane = d % 32;
  const bool q8 = dtype == FLYX_DT_GGML_Q8_0, q5 = dtype == FLYX_DT_GGML_Q5_0;
  src += size_t(d / 32) * (q8 ? 34 : q5 ? 22 : 18);
  const float delta = __half2float(*reinterpret_cast<const __half*>(src));
  if (q8) return delta * reinterpret_cast<const int8_t*>(src + 2)[lane];
  const uint8_t packed = src[(q5 ? 6 : 2) + lane % 16];
  const int low = lane < 16 ? packed & 15 : packed >> 4;
  const int high = q5 ? ((src[2 + lane / 8] >> (lane % 8)) & 1) << 4 : 0;
  return delta * (low + high - (q5 ? 16 : 8));
}

__global__ void decode_rows(const uint8_t* table, size_t stride, int dtype, int width, const int* ids, int count,
                            float* out) {
  for (size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x; at < size_t(count) * width;
       at += size_t(gridDim.x) * blockDim.x) {
    const int slot = int(at / width);
    out[at] = unpack(table + size_t(ids ? ids[slot] : slot) * stride, dtype, int(at % width));
  }
}

__global__ void gates(const float* k, const float* q, const float* v, int h, int streams, float* y) {
  const int r = blockIdx.y, c = blockIdx.x;
  const size_t off = (size_t(r) * streams + c) * h;
  float total = 0;
  for (int i = threadIdx.x; i < h; i += blockDim.x) total += k[off + i] * q[off + i];
  float a = block_sum(total) * rsqrtf(float(h));
  if (a != 0.f) a = copysignf(sqrtf(fmaxf(fabsf(a), 1e-6f)), a);
  const float scale = 1.f / (1.f + expf(-a));
  for (int i = threadIdx.x; i < h; i += blockDim.x) y[off + i] = scale * v[size_t(r) * h + i];
}

__global__ void convolve(flyx_ple_block b, PleWindow w, const float* weights, const float* y, float* u, float* wide,
                         const int* tokens, const DevParams* p, const TreeRows* tree, bool round) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y, W = b.hidden * b.streams, S = W + 1;
  if (d > W) return;
  const int depth = tree_depth(tree, r), hist = (b.taps - 1) * b.dilation;
  const float* old = w.win + size_t(slot_of(p, kSlotCur)) * w.w_stride;
  const float val = d == W ? __int_as_float(tokens[r]) : u[size_t(r) * S + d];
  if (d == W)
    u[size_t(r) * S + d] = val;
  else {
    float sum = 0;
    for (int tap = 0; tap < b.taps; ++tap) {
      const int at = depth - hist + tap * b.dilation;
      const float v = at < 0 ? old[size_t(hist + at) * S + d] : u[size_t(tree_row(tree, r, at)) * S + d];
      sum += weights[size_t(d) * b.taps + tap] * v;
    }
    wide[size_t(r) * W + d] += y[size_t(r) * W + d] + silu(sum);
  }
  if (round && !p->stopped) w.replay[size_t(replay_slot_of(p, kSlotCur)) * w.r_stride + size_t(r) * S + d] = val;
}

// Each lane owns one channel. Ascending destinations preserve the input when prefill updates its slot in place.
__global__ void save_window(PleWindow w, const float* u, int stride, int hist, int n, int sel, const DevParams* p,
                            bool round) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= stride || (round && p->stopped)) return;
  float* dst = w.win + size_t(slot_of(p, sel)) * w.w_stride;
  const float* old = w.win + size_t(slot_of(p, kSlotCur)) * w.w_stride;
  for (int k = 0; k < hist; ++k) {
    const int row = n - hist + k;
    dst[size_t(k) * stride + d] = row < 0 ? old[size_t(hist + row) * stride + d] : u[size_t(row) * stride + d];
  }
}

__global__ void compact_replay(PleWindow w, int channels, const DevParams* p, const TreeRows* tree) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= channels || !tree_on(tree) || p->noop) return;
  float* row = w.replay + size_t(replay_slot_of(p, kSlotLast)) * w.r_stride + d;
  for (int j = 0; j <= tree->n_taken; ++j) row[size_t(j) * channels] = row[size_t(tree->taken[j]) * channels];
}
}  // namespace

void ple_hash(flyx_ple_block b, PleWindow w, const int* tokens, int* ids, int rows, const DevParams* p,
              const TreeRows* tree, cudaStream_t st) {
  hash_rows<<<dim3(((b.ngram - 1) * b.heads_per_order + 31) / 32, rows), 32, 0, st>>>(b, w, tokens, ids, p, tree);
}
void ple_decode(const uint8_t* table, size_t stride, int dtype, int width, const int* ids, int count, float* out,
                cudaStream_t st) {
  const int grid = int(std::min<size_t>(65535, (size_t(count) * width + 255) / 256));
  decode_rows<<<grid, 256, 0, st>>>(table, stride, dtype, width, ids, count, out);
}
void ple_gates(const float* k, const float* q, const float* v, int h, int streams, int rows, float* y,
               cudaStream_t st) {
  gates<<<dim3(streams, rows), 256, 0, st>>>(k, q, v, h, streams, y);
}
void ple_convolve(flyx_ple_block b, PleWindow w, const float* weights, const float* y, float* u, float* wide,
                  const int* tokens, int rows, const DevParams* p, const TreeRows* tree, bool round, int next, int snap,
                  int snap_rows, cudaStream_t st) {
  const int stride = b.hidden * b.streams + 1, blocks = (stride + 255) / 256, hist = (b.taps - 1) * b.dilation;
  convolve<<<dim3(blocks, rows), 256, 0, st>>>(b, w, weights, y, u, wide, tokens, p, tree, round);
  if (snap >= 0) save_window<<<blocks, 256, 0, st>>>(w, u, stride, hist, snap_rows, snap, p, false);
  if (next >= 0) save_window<<<blocks, 256, 0, st>>>(w, u, stride, hist, rows, next, p, round);
}
void ple_compact(PleWindow w, int channels, const DevParams* p, const TreeRows* tree, cudaStream_t st) {
  compact_replay<<<(channels + 255) / 256, 256, 0, st>>>(w, channels, p, tree);
}
}  // namespace flyx::cuda
