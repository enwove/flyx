// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What cuda_bench's benchmarks and its tuner (tune.cu) share: timing, random weights, the launches' scratch and
// options, the GGML block types and layers they measure, and the routes and buffers of their MoE layers.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include "backends/cuda/kernels/kernels.cuh"

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);    \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

namespace flyx::bench {

using namespace flyx::cuda;

struct Shape {
  const char* name;
  int N, K;
};

// Random-looking weights the benchmarks can rebuild from their index and seed.
static __device__ __forceinline__ uint32_t weight_hash(size_t i, uint32_t seed) {
  uint32_t h = uint32_t(i) * 2654435761u ^ seed;
  h ^= h >> 15;
  h *= 2246822519u;
  return h ^ (h >> 13);
}

static __global__ void k_fill(uint8_t* p, size_t n, uint32_t seed, uint8_t lo, uint8_t span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = weight_hash(i, seed);
    p[i] = span ? uint8_t(lo + h % span) : uint8_t(h);
  }
}

// Binary16 scales with a random sign and significand, magnitude in [1/128, 1/64).
static __global__ void k_fill_block_scales(uint16_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    p[i] = uint16_t(0x2000u | (weight_hash(i, seed) & 0x83ffu));
}
// Floats in [lo, lo + span).
static __global__ void k_fill_f(float* p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = weight_hash(i, seed);
    p[i] = lo + span * float(h >> 8) * (1.f / 16777216.f);
  }
}
// bf16 values in [lo, lo + span).
static __global__ void k_fill_bf16(bf16* p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint32_t h = weight_hash(i, seed);
    p[i] = __float2bfloat16(lo + span * float(h >> 8) * (1.f / 16777216.f));
  }
}

// Launches of the benchmarks: scratch for prefill rows up to 2048 and K up to 16384, the default options with this
// GPU's tuning table (cuda_bench's main), so what a launch picks is what the engine picks here.
extern Scratch g_bench_scratch;
extern KernelOptions g_bench_opts;
inline Launch BL(cudaStream_t st) { return Launch{st, &g_bench_scratch, &g_bench_opts}; }

// ms a call of f: three calls first, then the mean of `reps`.
template <class F>
double time_ms(F&& f, int reps) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) f();
  CK(cudaEventRecord(a));
  for (int i = 0; i < reps; ++i) f();
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms / reps;
}

inline void header() {
  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, 0));
  std::printf("%s, %d SMs, L2 %d MB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20);
}

// GGML block types and the layers their benchmarks run: Qwen3.8-27B's, then Qwen3.5-4B's.
struct BlockType {
  int dtype;
  const char* name;
};
inline const BlockType kBlockTypes[] = {{FLYX_DT_GGML_IQ2_XXS, "IQ2_XXS"}, {FLYX_DT_GGML_IQ2_S, "IQ2_S"},
                                        {FLYX_DT_GGML_IQ1_S, "IQ1_S"},     {FLYX_DT_GGML_IQ2_XS, "IQ2_XS"},
                                        {FLYX_DT_GGML_IQ3_XXS, "IQ3_XXS"}, {FLYX_DT_GGML_IQ3_S, "IQ3_S"},
                                        {FLYX_DT_GGML_Q4_K, "Q4_K"},       {FLYX_DT_GGML_IQ1_M, "IQ1_M"},
                                        {FLYX_DT_GGML_Q2_K, "Q2_K"},       {FLYX_DT_GGML_IQ4_XS, "IQ4_XS"},
                                        {FLYX_DT_GGML_Q8_0, "Q8_0"},       {FLYX_DT_GGML_Q3_K, "Q3_K"},
                                        {FLYX_DT_GGML_Q5_K, "Q5_K"},       {FLYX_DT_GGML_Q6_K, "Q6_K"},
                                        {FLYX_DT_GGML_Q4_0, "Q4_0"},       {FLYX_DT_GGML_IQ4_NL, "IQ4_NL"},
                                        {FLYX_DT_GGML_Q4_1, "Q4_1"},       {FLYX_DT_GGML_Q5_0, "Q5_0"},
                                        {FLYX_DT_GGML_Q5_1, "Q5_1"},       {FLYX_DT_GGML_MXFP4, "MXFP4"}};
inline const char* block_type_name(int dtype) {
  for (const BlockType& t : kBlockTypes)
    if (t.dtype == dtype) return t.name;
  return nullptr;
}
inline const Shape kBlockShapes[] = {{"mlp gate, up", 17408, 5120}, {"mlp down", 5120, 17408}, {"gdn out", 5120, 6144},
                                     {"attention q", 12288, 5120},  {"gdn qkv", 10240, 5120},  {"gdn z", 6144, 5120},
                                     {"attention k", 1024, 5120},   {"gdn a", 48, 5120},       {"head", 248320, 5120},
                                     {"4b gate, up", 9216, 2560},   {"4b down", 2560, 9216},   {"4b qkv", 8192, 2560},
                                     {"4b out", 2560, 4096},        {"4b gdn a", 32, 2560}};

// Copies of a block matrix of `dtype` and N x K, more than L2 holds together (one of a head of more than 100000
// columns), from random bytes: valid blocks of every type.
inline std::vector<BlockMat> block_copies(int dtype, int N, int K) {
  const BlockLayout l = blocks_layout(dtype);
  const size_t blocks = size_t(N) * K / l.values, bytes = blocks * (l.stride + 2 * l.scales);
  const size_t copies = N > 100000 ? 1 : (512ull << 20) / bytes + 1;
  std::vector<BlockMat> w;
  for (size_t c = 0; c < copies; ++c) {
    uint8_t* data;
    uint16_t* scales;
    CK(cudaMalloc(&data, blocks * l.stride));
    CK(cudaMalloc(&scales, blocks * l.scales * 2));
    k_fill<<<1024, 256>>>(data, blocks * l.stride, 101u + uint32_t(c), 0, 0);
    k_fill_block_scales<<<1024, 256>>>(scales, blocks * l.scales, 7u + uint32_t(c));
    w.push_back({data, scales, dtype, N});
  }
  return w;
}
inline std::vector<BlockMat> block_copies(const BlockType& t, const Shape& s) { return block_copies(t.dtype, s.N, s.K); }
inline void free_copies(std::vector<BlockMat>& w) {
  for (auto& m : w) {
    cudaFree(const_cast<uint8_t*>(m.data));
    cudaFree(const_cast<uint16_t*>(m.scales));
  }
  w.clear();
}

// The experts kMoeRouteSets calls of `rows` rows take from a bank of E, top a row (top <= E): consecutive
// rows keep two experts (or top if smaller) and slide the rest four a row; consecutive calls rotate sets over the
// bank by a permutation (stride >= 37 coprime to E).  With 128 experts and 8 a row, 32 calls complete a whole number
// of rotations at every measured row count, including the wrap from one graph replay to the next.  Returns the
// distinct experts a call reads.
constexpr int kMoeRouteSets = 32;
inline int make_moe_routes(int* indices, int rows, int E, int top) {
  const int pool = std::max(top, E / 2);
  const int distinct = std::min(pool, top + 4 * (rows - 1));
  const int keep = std::min(2, top);
  const int rest = pool - keep;
  int stride = 37;
  while (E > 1 && std::gcd(stride, E) != 1) ++stride;
  std::vector<int> routes(size_t(kMoeRouteSets) * rows * top);
  for (int i = 0; i < kMoeRouteSets; ++i)
    for (int r = 0; r < rows; ++r)
      for (int slot = 0; slot < top; ++slot) {
        const int expert = slot < keep || rest <= 0 ? slot : keep + (4 * r + slot - keep) % rest;
        routes[(size_t(i) * rows + r) * top + slot] = int((int64_t(stride) * (expert + int64_t(i) * distinct)) % E);
      }
  CK(cudaMemcpy(indices, routes.data(), routes.size() * sizeof(int), cudaMemcpyHostToDevice));
  return distinct;
}

// Device buffers freed together.
struct MoeBenchBuffers {
  std::vector<void*> buffers;
  template <class T>
  T* alloc(size_t count) {
    T* p;
    CK(cudaMalloc(&p, count * sizeof(T)));
    buffers.push_back(p);
    return p;
  }
  ~MoeBenchBuffers() {
    for (void* p : buffers) CK(cudaFree(p));
  }
};

// cuda_bench tune [FAMILIES] [FILE...] (tune.cu): this GPU's tuning table.
int tune(int argc, char** argv);
void bench_graphs(int argc, char** argv);  // graphs.cu
int bench_paging(int argc, char** argv);   // paging.cu

}  // namespace flyx::bench
