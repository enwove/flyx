// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel timings for the shapes of Qwen3.5-4B (the numbers behind the tuning tables).
//
//   cuda_bench gemm        weight bandwidth of each GEMM path by row count
//   cuda_bench skinny      every compiled skinny configuration, per shape and row count
//   cuda_bench fp4         the same for NVFP4 weights
//   cuda_bench tune [FAMILIES] [FILE...]  measure every family, print this GPU's tuning table (TOML; tune.cu)
//   cuda_bench gdn [27b]   a GDN layer between its projections: four kernels or one (the 4B's, or the 27B's)
//   cuda_bench decode      the NVFP4 decode GEMMs in graphs, against a read of their bytes
//   cuda_bench tc          the prefill GEMM's configurations by shape and rows
//   cuda_bench attn        decode attention by context length, against a read of the K/V
//   cuda_bench flash       prefill attention of one long prompt in 1024-row passes, per pass and in all
//   cuda_bench strata a round's strata attention over a long context, ms a layer
//   cuda_bench w12         the output head in bf16 and in W12 (the same bits), per configuration and row count
//   cuda_bench pick        the drafts' two-stage head (1-bit rough GEMM, draft_pick) against the NVFP4 head
//   cuda_bench host_read the GPU reading pinned host memory in place, mapped and write-combined
//   cuda_bench w4a4        NVFP4 prefill GEMMs: bf16 activations against NVFP4 ones (W4A4, sm_120a)
//   cuda_bench mlp a round's MLP as four kernels and as one persistent kernel
//   cuda_bench blocks [T [configs]]  the GEMMs on packed GGML block weights, per type (or type T), on the layers
//                          of Qwen3.8-27B and Qwen3.5-4B; with `configs`, every compiled configuration
//   cuda_bench q8runs [T [table]]  the prefill GEMM in 8 bits on those layers: every configuration, K in 1..16 runs
//                          in turn and side by side, at 32, 64, 128 and 512 rows (the numbers behind blocks_q8_runs);
//                          with `table`, each configuration with the table's runs only (behind blocks_q8)
//   cuda_bench rounds [T]  rounds of 1, 2, 4 and 8 rows in 8 bits on dp4a on those layers: every compiled
//                          configuration (columns a warp, groups ahead; the numbers behind blocks_round)
//   cuda_bench moe         Inkling-2-Mini's fused gate|up and weighted down projections, per dp4a variant and rows
//   cuda_bench round T LAYER ROWS [KERNEL [CFG]]  one round on one layer, 43 launches (for a profiler)
//   cuda_bench pass T LAYER ROWS  one prefill pass's GEMM (inputs in 8 bits) on one layer (for a profiler)
//   cuda_bench stack ROWS K T:N [T:N ...]  a stack of projections of one input, one launch per run of a block type,
//                          against each run alone
//   cuda_bench mma        peak rate of each tensor-core instruction the GEMMs could use (ours: f16 -> f32;
//                          llama.cpp's batches on GGML blocks: s8 -> s32), and the host's clock against the GPU's
//   cuda_bench stream      reading memory: GB/s against the bytes an SM has on their way (in registers, in a
//                          cp.async ring), all SMs reading or 60% of them
//   cuda_bench graphs [K]  what an instantiated graph takes of the device's memory, by the nodes it holds (graphs.cu)
//   cuda_bench paging KIND STEP [PIECE [FILL_TO]]  where the driver stops keeping allocations of a size on the GPU
//                          (Windows pages them to host memory), one cudaMalloc or VMM pieces a step (paging.cu)
//   cuda_bench mt [ROWS ...] [NxK ...]  bf16 GEMMs of 17..64 rows (by default 17, 32, 48 and 64) on the shapes given
//                          (by default MiniCPM5-1B's layers, DV's adapters and head): the prefill kernel, the 1..16-row
//                          kernel a tile of 16 rows at a time, and every k_gemm_skinny_mt configuration (bf16_mt), us
//                          and GB/s of weights
//
// Weights rotate through enough copies to exceed the L2 cache, as in a real
// forward pass where every weight is read once per round.  Kernels launch as
// programmatic dependents where the GPU's table says so (FLYX_BENCH_PDL=0|1
// otherwise), as the engine launches them.
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "backends/cuda/kernels/common/device.cuh"
#include "bench/microbench/bench.cuh"
#include "bench/microbench/mlp_persistent.h"

using namespace flyx::cuda;
using namespace flyx::bench;

namespace flyx::bench {
Scratch g_bench_scratch;
KernelOptions g_bench_opts;
}  // namespace flyx::bench

namespace {

const Shape kShapes[] = {{"attn qkv", 10240, 2560}, {"gdn in", 12352, 2560},    {"mlp gate|up", 18432, 2560},
                         {"mlp down", 2560, 9216},  {"attn/gdn out", 2560, 4096}, {"lm_head", 248320, 2560}};

// K/V cache values (binary16) in [lo, lo + span).
__global__ void k_fill_kv(kv16* p, size_t n, uint32_t seed, float lo, float span) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = __float2half_rn(lo + span * float(h >> 8) * (1.f / 16777216.f));
  }
}

// Random 32-bit words (codes of a low-bit copy).
__global__ void k_fill_words(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    p[i] = h;
  }
}

// Copies of a weight matrix, more than L2 holds together, so a GEMM reads its
// weights from DRAM: zeros, or random values.
struct Weights {
  std::vector<bf16*> copies;
  size_t i = 0;
  Weights(const Shape& s, bool random = false) {
    const size_t bytes = size_t(s.N) * s.K * 2;
    const size_t n = std::max<size_t>(1, (512ull << 20) / bytes + 1);  // > 512 MB in total
    for (size_t k = 0; k < n; ++k) {
      bf16* w;
      CK(cudaMalloc(&w, bytes));
      if (random) k_fill_bf16<<<1024, 256>>>(w, bytes / 2, 101u + uint32_t(k), -0.05f, 0.1f);
      else CK(cudaMemset(w, 0, bytes));
      copies.push_back(w);
    }
  }
  ~Weights() {
    for (auto* w : copies) cudaFree(w);
  }
  bf16* next() { return copies[i++ % copies.size()]; }
};

constexpr int kMaxBenchRows = 16;

void bench_gemm() {
  header();
  std::printf("%-14s %6s %5s      | %-20s %-20s %-20s\n", "shape", "N", "K", "rows path GB/s (us)", "skinny GB/s (us)",
              "mma GB/s (us)");
  for (const auto& s : kShapes) {
    Weights W(s);
    float *x, *y;
    CK(cudaMalloc(&x, size_t(2048) * s.K * 4));
    CK(cudaMemset(x, 0, size_t(2048) * s.K * 4));
    CK(cudaMalloc(&y, size_t(2048) * s.N * 4));
    const double gb = double(s.N) * s.K * 2 / 1e9;
    for (int rows : {1, 2, 4, 8, 16, 64, 256, 512, 1024, 2048}) {
      auto fmt = [&](double ms) {
        char buf[64];
        if (rows > 16)  // compute bound: report TFLOPS
          std::snprintf(buf, sizeof(buf), "%6.1f (%8.1f)", 2.0 * rows * s.N * s.K / (ms / 1e3) / 1e12, ms * 1e3);
        else
          std::snprintf(buf, sizeof(buf), "%6.0f (%8.1f)", gb / (ms / 1e3), ms * 1e3);
        return std::string(buf);
      };
      std::string a = "-", b = "-", c = "-";
      if (rows <= 16) {
        a = fmt(time_ms([&] { gemm_rows(x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, 0); }, 20));
        b = fmt(time_ms([&] { gemm_skinny(x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, BL(0)); }, 20));
      }
      if (rows >= 16) c = fmt(time_ms([&] { gemm_tc(x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, BL(0)); }, 10));
      std::printf("%-14s %6d %5d r%-3d | %-20s %-20s %-20s\n", s.name, s.N, s.K, rows, a.c_str(), b.c_str(), c.c_str());
    }
    CK(cudaFree(x));
    CK(cudaFree(y));
  }
}

void bench_skinny() {
  header();
  std::printf("GB/s of weights streamed, per skinny configuration\n%-14s %-5s", "shape", "rows");
  for (int c = 0; c < kSkinnyConfigs; ++c) std::printf("  cfg%-3d", c);
  std::printf("\n");
  for (const auto& s : kShapes) {
    Weights W(s);
    float *x, *y;
    CK(cudaMalloc(&x, size_t(16) * s.K * 4));
    CK(cudaMemset(x, 0, size_t(16) * s.K * 4));
    CK(cudaMalloc(&y, size_t(16) * s.N * 4));
    const double gb = double(s.N) * s.K * 2 / 1e9;
    for (int rows : {1, 4, 8, 16}) {
      std::printf("%-14s r%-4d", s.name, rows);
      for (int c = 0; c < kSkinnyConfigs; ++c) {
        if (!bench_skinny_cfg_fits(c, s.N, false)) {
          std::printf("  %6s", "-");
          continue;
        }
        double ms = time_ms([&] { gemm_skinny_cfg(c, x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, BL(0)); }, 20);
        std::printf("  %6.0f", gb / (ms / 1e3));
      }
      std::printf("\n");
    }
    CK(cudaFree(x));
    CK(cudaFree(y));
  }
}

// Inkling-2-Mini's MoE layers: a bank of 128 experts, 8 a token (make_moe_routes).
constexpr int kMoeExperts = 128, kMoeTopK = 8;

void bench_moe_projection(int dtype, bool down) {
  constexpr int H = 2304, I = 1152, E = kMoeExperts, top = kMoeTopK, max_pairs = kMaxBenchRows * top;
  const bool nvfp4 = dtype == FLYX_DT_GGML_NVFP4;
  const int N = down ? H : 2 * I;
  const int K = down ? (nvfp4 ? I : 1280) : H;  // Q4_K down needs whole 256-value blocks
  const size_t input_elems = size_t(down ? max_pairs : kMaxBenchRows) * K;
  MoeBenchBuffers mem;
  ExpertMat W{};
  W.n_experts = E;
  W.N = N;
  float* rs = mem.alloc<float>(E);
  k_fill_f<<<1, 256>>>(rs, E, 7u, 0.5f, 0.5f);
  W.rs = rs;
  size_t expert_bytes = sizeof(float);  // include the per-expert scale
  std::vector<BlockMat> packed;
  if (nvfp4) {
    const size_t values = size_t(E) * N * K;
    uint8_t* q = mem.alloc<uint8_t>(values / 2);
    uint8_t* s = mem.alloc<uint8_t>(values / 16);
    k_fill<<<1024, 256>>>(q, values / 2, 101u, 0, 0);
    k_fill<<<1024, 256>>>(s, values / 16, 103u, 16, 32);
    W.q = q;
    W.s = s;
    expert_bytes += size_t(N) * (K / 2 + K / 16);
  } else {
    packed = block_copies(dtype, E * N, K);
    W.blk = packed.front();
    W.blk.q8 = true;
    const BlockLayout l = blocks_layout(dtype);
    expert_bytes += size_t(N) * (K / l.values) * (l.stride + sizeof(uint16_t) * l.scales);
  }

  MoeScratch sc{};
  sc.chunk_rows = kMaxBenchRows;
  sc.max_experts = E;
  sc.max_top_k = top;
  sc.pairs = mem.alloc<int>(max_pairs);
  sc.tasks = mem.alloc<MoeTask>(max_pairs);
  sc.n_tasks = mem.alloc<int>(1);
  sc.stage_elems = input_elems;
  sc.q8_a = mem.alloc<int8_t>(input_elems);
  sc.q8_d = mem.alloc<float>(input_elems / 32);
  sc.q8_s = mem.alloc<float>(input_elems / 16);
  sc.down_buf = mem.alloc<float>(size_t(max_pairs) * H);
  float* x = mem.alloc<float>(input_elems);
  k_fill_f<<<64, 256>>>(x, input_elems, 31u, -1.f, 2.f);
  bf16* act = mem.alloc<bf16>(std::max(size_t(max_pairs) * I, nvfp4 ? input_elems : size_t(0)));
  if (nvfp4) {
    k_fill_bf16<<<64, 256>>>(act, input_elems, 31u, -1.f, 2.f);
    sc.bf16_a = act;
  } else if (!down) {
    moe_stage_q8(x, K, K, kMaxBenchRows, sc, moe_q8_mins(W), 0);
  }
  float* y = mem.alloc<float>(size_t(kMaxBenchRows) * H);
  int* indices = mem.alloc<int>(kMoeRouteSets * max_pairs);
  float* ew = mem.alloc<float>(max_pairs);
  k_fill_f<<<1, 256>>>(ew, max_pairs, 23u, 0.1f, 0.05f);
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  KernelOptions options = g_bench_opts;
  const Launch launch{st, nullptr, &options};
  for (int rows : {1, 2, 4, 6, 8, 16}) {
    const int distinct = make_moe_routes(indices, rows, E, top);
    const size_t bytes = distinct * expert_bytes;
    // -1: the rules (default); then every configuration this build compiles for the type, op and row band
    for (int v = -1; v < (nvfp4 ? 0 : kMoeDp4aConfigCount); ++v) {
      if (v >= 0 && !moe_dp4a_compiled(dtype, v, !down, rows)) continue;
      options.moe_dp4a = v;
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
      for (int i = 0; i < kMoeRouteSets; ++i) {
        const int* eidx = indices + i * rows * top;
        if (down)
          moe_down(x, K, W, eidx, ew, rows, top, H, K, y, H, false, &sc, launch, nvfp4 ? act : nullptr);
        else
          moe_gate_up(x, K, W, eidx, rows, top, I, K, 6.f, 8.f, true, nullptr, I, &sc, launch, true, act);
      }
      CK(cudaStreamEndCapture(st, &graph));
      CK(cudaGraphInstantiate(&exec, graph, 0));
      const double ms = time_ms([&] { CK(cudaGraphLaunch(exec, 0)); }, 5) / kMoeRouteSets;
      CK(cudaGraphExecDestroy(exec));
      CK(cudaGraphDestroy(graph));
      std::printf("%-7s %-8s %4d %4d %4d %7d %12.3f %-8s %10.2f %9.1f\n", down ? "down" : "gate|up",
                  nvfp4 ? "NVFP4" : block_type_name(dtype), N, K, rows, distinct, bytes / 1e6,
                  nvfp4 ? "control" : moe_dp4a_name(options.moe_dp4a), ms * 1e3, bytes / (ms * 1e6));
    }
    std::fflush(stdout);
  }
  CK(cudaStreamDestroy(st));
  free_copies(packed);
}

void bench_moe() {
  header();
  int driver;
  CK(cudaDriverGetVersion(&driver));
  std::printf("Inkling-2-Mini, E=128 top=8 H=2304 I=1152, CUDA driver API %d\n"
              "32 calls per graph, rotating expert sets, 3 warmup replays and 5 timed replays. "
              "Effective GB/s uses the distinct packed expert bytes per call.\n"
              "Gate|up reads pre-staged Q8 and writes fused SiTU in BF16. Down includes the weighted sum and reads "
              "staged BF16 (NVFP4) or stages Q8 (Q4_K). Each call builds its grouping when rows > 1.\n"
              "Q4_K down uses K=1280, the next whole 256-value block above 1152. NVFP4 down uses K=1152.\n"
              "op      format      N    K rows experts  distinct_MB variant     us/call  eff_GB/s\n", driver);
  blocks_init();
  for (int dtype : kMoeDp4aTypes) bench_moe_projection(dtype, false);
  bench_moe_projection(FLYX_DT_GGML_NVFP4, true);
  bench_moe_projection(FLYX_DT_GGML_Q4_K, true);
}

// NVFP4 weights: bytes streamed are K/2 + K/16 (+4) per output column.
void bench_fp4() {
  header();
  std::printf("NVFP4 GB/s of weights streamed, per skinny configuration (and the prefill path)\n%-14s %-5s",
              "shape", "rows");
  for (int c = 0; c < kSkinnyConfigs; ++c) std::printf("  cfg%-3d", c);
  std::printf("  mma\n");
  for (const auto& s : kShapes) {
    if (s.N == 248320) continue;  // the head stays bf16
    const size_t qb = size_t(s.N) * s.K / 2, sb = size_t(s.N) * s.K / 16;
    const size_t copies = std::max<size_t>(1, (512ull << 20) / (qb + sb) + 1);
    std::vector<uint8_t*> Q(copies), S(copies);
    for (size_t i = 0; i < copies; ++i) {
      CK(cudaMalloc(&Q[i], qb));
      CK(cudaMemset(Q[i], 0x11, qb));
      CK(cudaMalloc(&S[i], sb));
      CK(cudaMemset(S[i], 0x38, sb));
    }
    float *rs, *x, *y;
    CK(cudaMalloc(&rs, size_t(s.N) * 4));
    CK(cudaMemset(rs, 0, size_t(s.N) * 4));
    CK(cudaMalloc(&x, size_t(256) * s.K * 4));
    CK(cudaMemset(x, 0, size_t(256) * s.K * 4));
    CK(cudaMalloc(&y, size_t(256) * s.N * 4));
    const double gb = double(qb + sb) / 1e9;
    size_t it = 0;
    for (int rows : {1, 4, 8, 16, 256}) {
      std::printf("%-14s r%-4d", s.name, rows);
      for (int c = 0; c < kSkinnyConfigs; ++c) {
        if (rows > 16) {
          std::printf("  %6s", "-");
          continue;
        }
        double ms = time_ms([&] {
          const size_t i = it++ % copies;
          gemm_fp4_skinny_cfg(c, x, s.K, Q[i], S[i], rs, s.N, s.K, y, s.N, rows, false, BL(0));
        }, 20);
        std::printf("  %6.0f", gb / (ms / 1e3));
      }
      double ms = time_ms([&] {
        const size_t i = it++ % copies;
        gemm_fp4_mma(x, s.K, Q[i], S[i], rs, s.N, s.K, y, s.N, rows, false, 0);
      }, 10);
      std::printf("  %6.0f\n", gb / (ms / 1e3));
    }
    for (size_t i = 0; i < copies; ++i) {
      cudaFree(Q[i]);
      cudaFree(S[i]);
    }
    cudaFree(rs);
    cudaFree(x);
    cudaFree(y);
  }
}

void bench_gdn_prefill();

// One GDN layer's round path between the projections (Qwen3.5-4B shapes): the
// four kernels against the one-kernel path, replayed as CUDA graphs of 24
// layers (so launch gaps count as in a round).
void bench_gdn(bool big) {
  header();
  // Qwen3.5-4B's gated delta nets (24 layers), or with `27b` Qwen3.8-27B's (48 layers, 48 value heads)
  const GdnDims d{16, big ? 48 : 32, 128, 128, 4};
  const int C = d.C(), vd = d.nv * d.dv, ld = C + vd + 2 * d.nv, layers = big ? 48 : 24;
  std::printf("%s: %d key heads, %d value heads of %d x %d, %d layers; plain decoding reads and writes the state: "
              "%.2f MB a layer\n", big ? "Qwen3.8-27B" : "Qwen3.5-4B", d.nk, d.nv, d.dk, d.dv, layers,
              2.0 * d.nv * d.dv * d.dk * 4 / 1e6);
  const size_t s_stride = size_t(d.nv) * d.dv * d.dk, w_stride = size_t(d.ck - 1) * C;
  std::vector<float*> S(layers), W(layers), B(layers);
  GdnBufs bf = GdnBufs::round(16, d);
  for (int i = 0; i < layers; ++i) {
    CK(cudaMalloc(&S[i], 4 * s_stride * 4));
    CK(cudaMemset(S[i], 0, 4 * s_stride * 4));
    CK(cudaMalloc(&W[i], 4 * w_stride * 4));
    CK(cudaMemset(W[i], 0, 4 * w_stride * 4));
    CK(cudaMalloc(&B[i], 4 * bf.slot_stride * 4));
  }
  float *proj, *cw, *al, *dt, *gw, *o;
  CK(cudaMalloc(&proj, size_t(16) * ld * 4));
  CK(cudaMemset(proj, 0, size_t(16) * ld * 4));
  CK(cudaMalloc(&cw, size_t(C) * d.ck * 4));
  CK(cudaMemset(cw, 0, size_t(C) * d.ck * 4));
  CK(cudaMalloc(&al, d.nv * 4));
  CK(cudaMemset(al, 0, d.nv * 4));
  CK(cudaMalloc(&dt, d.nv * 4));
  CK(cudaMemset(dt, 0, d.nv * 4));
  CK(cudaMalloc(&gw, d.dv * 4));
  CK(cudaMemset(gw, 0, d.dv * 4));
  CK(cudaMalloc(&o, size_t(16) * vd * 4));
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  CK(cudaMemset(P, 0, sizeof(DevParams)));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 50);  // time_ms records on stream 0
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / layers;
  };
  std::printf("us per layer (graph of %d layers): rows | four kernels | one kernel | plain (next slot): four | one\n",
              layers);
  for (int rows : {1, 2, 4, 8, 16}) {
    auto four = [&](int next) {
      return graph_us([&] {
        for (int i = 0; i < layers; ++i) {
          GdnBufs b = bf;
          b.base = B[i];
          gdn_conv(proj, ld, cw, W[i], w_stride, kSlotCur, next, kSkipIfStopped, b, kSlotCur, P, rows, st);
          gdn_prep(b, kSlotCur, proj + C + vd, proj + C + vd + d.nv, ld, al, dt, P, rows, st);
          gdn_recur(S[i], s_stride, kSlotCur, next, b, kSlotCur, P, o, rows, nullptr,
                    next >= 0 ? kSkipIfStopped : kSkipNever, st);
          gdn_gated_norm_split(o, proj + C, ld, gw, 1e-6f, rows, d, BL(st));
        }
      });
    };
    bool ran = true;  // false: rows the one-kernel path does not take
    auto one = [&](int next) {
      return graph_us([&] {
        for (int i = 0; i < layers; ++i) {
          GdnBufs b = bf;
          b.base = B[i];
          ran &= gdn_round(proj, ld, cw, W[i], w_stride, S[i], s_stride, next, al, dt, gw, 1e-6f, b, P, rows, nullptr,
                           BL(st));
        }
      });
    };
    const double a = four(-1), b = one(-1);
    if (!ran) {
      std::printf("  r%-2d | %6.2f |      - |\n", rows, a);
    } else if (rows == 1) {
      const double c = four(kSlotNext), e = one(kSlotNext), mb = 2.0 * d.nv * d.dv * d.dk * 4 / 1e6;
      std::printf("  r%-2d | %6.2f | %6.2f | %6.2f | %6.2f (the state at %.0f GB/s)\n", rows, a, b, c, e, mb / e * 1e3);
    } else {
      std::printf("  r%-2d | %6.2f | %6.2f |\n", rows, a, b);
    }
  }
  if (!big) bench_gdn_prefill();
}

// A GDN layer's prefill pass between its projections, 1024 rows, kernel by
// kernel (graphs of 24 layers, states and buffers per layer).
void bench_gdn_prefill() {
  const GdnDims d{16, 32, 128, 128, 4};
  const int rows = 1024, C = d.C(), vd = d.nv * d.dv, ld = C + vd + 2 * d.nv, layers = 24;
  const size_t s_stride = size_t(d.nv) * d.dv * d.dk, w_stride = size_t(d.ck - 1) * C;
  GdnBufs bf = GdnBufs::prefill(rows, d);
  std::vector<float*> S(layers), W(layers), B(layers), proj(layers);
  for (int i = 0; i < layers; ++i) {
    CK(cudaMalloc(&S[i], 4 * s_stride * 4));
    CK(cudaMemset(S[i], 0, 4 * s_stride * 4));
    CK(cudaMalloc(&W[i], 4 * w_stride * 4));
    CK(cudaMemset(W[i], 0, 4 * w_stride * 4));
    CK(cudaMalloc(&B[i], bf.floats() * 4));
    CK(cudaMalloc(&proj[i], size_t(rows) * ld * 4));
    CK(cudaMemset(proj[i], 0, size_t(rows) * ld * 4));
  }
  float *cw, *al, *dt, *gw, *o, *yv;
  CK(cudaMalloc(&cw, size_t(C) * d.ck * 4));
  CK(cudaMemset(cw, 0, size_t(C) * d.ck * 4));
  CK(cudaMalloc(&al, d.nv * 4));
  CK(cudaMemset(al, 0, d.nv * 4));
  CK(cudaMalloc(&dt, d.nv * 4));
  CK(cudaMemset(dt, 0, d.nv * 4));
  CK(cudaMalloc(&gw, d.dv * 4));
  CK(cudaMemset(gw, 0, d.dv * 4));
  CK(cudaMalloc(&o, size_t(rows) * vd * 4));
  CK(cudaMalloc(&yv, size_t(rows) * vd * 4));
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  CK(cudaMemset(P, 0, sizeof(DevParams)));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < layers; ++i) body(i);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 10);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / layers;
  };
  auto bufs = [&](int i) {
    GdnBufs b = bf;
    b.base = B[i];
    return b;
  };
  const double conv = graph_us([&](int i) {
    gdn_conv(proj[i], ld, cw, W[i], w_stride, kSlotCur, kSlotCur, kSkipNever, bufs(i), kSlotCur, P, rows, st);
  });
  const double prep = graph_us([&](int i) {
    gdn_prep(bufs(i), kSlotCur, proj[i] + C + vd, proj[i] + C + vd + d.nv, ld, al, dt, P, rows, st);
  });
  const double recur = graph_us([&](int i) {
    gdn_recur(S[i], s_stride, kSlotCur, kSlotCur, bufs(i), kSlotCur, P, o, rows, nullptr, kSkipNever, st);
  });
  const double gnorm = graph_us([&](int i) { gdn_gated_norm(o, proj[i] + C, ld, gw, 1e-6f, yv, rows, d, st); });
  bool chunk_ok = true;
  const double chunked =
      graph_us([&](int i) { chunk_ok &= gdn_chunk(S[i], s_stride, kSlotCur, bufs(i), P, 0, o, rows, st); });
  std::printf("prefill, %d rows, us per layer: conv %.1f | prep %.1f | recurrence %.1f | gated norm %.1f | total %.1f\n",
              rows, conv, prep, recur, gnorm, conv + prep + recur + gnorm);
  if (chunk_ok)
    std::printf("  chunked recurrence (gdn_chunk) %.1f | total with it %.1f\n", chunked, conv + prep + chunked + gnorm);
  // The layer as a prefill pass runs it (each kernel reads what the one before wrote).
  auto layer = [&](int i, bool chunk) {
    gdn_conv(proj[i], ld, cw, W[i], w_stride, kSlotCur, kSlotCur, kSkipNever, bufs(i), kSlotCur, P, rows, st);
    gdn_prep(bufs(i), kSlotCur, proj[i] + C + vd, proj[i] + C + vd + d.nv, ld, al, dt, P, rows, st);
    if (!chunk || !gdn_chunk(S[i], s_stride, kSlotCur, bufs(i), P, 0, o, rows, st))
      gdn_recur(S[i], s_stride, kSlotCur, kSlotCur, bufs(i), kSlotCur, P, o, rows, nullptr, kSkipNever, st);
    gdn_gated_norm(o, proj[i] + C, ld, gw, 1e-6f, yv, rows, d, st);
  };
  const double seq_layer = graph_us([&](int i) { layer(i, false); });
  const double chunk_layer = graph_us([&](int i) { layer(i, true); });
  std::printf("  layer in order, us: row by row %.1f | chunked %.1f\n", seq_layer, chunk_layer);
}

// The prefill GEMM's configurations (gemm_tc cfg) by shape and rows, TFLOPS,
// bf16 and NVFP4 weights.  Each call includes the float32 -> bf16 rounding of
// the activations (k_to_bf16), which a prefill pass mostly leaves to the
// producer kernels (gemm_tc_staged): the trace of a real pass shows the GEMM
// alone (bench/trace).
// cuda_bench tc [NxK ...]: the shapes given (default kShapes), 24..2048 rows.
void bench_tc(int argc, char** argv) {
  header();
  std::printf("prefill GEMM TFLOPS per configuration (bf16 | nvfp4), random weights and activations\n");
  std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
  if (argc > 2) {
    shapes.clear();
    for (int i = 2; i < argc; ++i) {
      int n = 0, k = 0;
      if (std::sscanf(argv[i], "%dx%d", &n, &k) == 2) shapes.push_back({argv[i], n, k});
    }
  }
  for (const auto& s : shapes) {
    if (s.N == 248320) continue;
    Weights W(s, true);
    const size_t qb = size_t(s.N) * s.K / 2, sb = size_t(s.N) * s.K / 16;
    uint8_t *q, *sc;
    float *rs, *x, *y;
    CK(cudaMalloc(&q, qb));
    k_fill<<<1024, 256>>>(q, qb, 7u, 0, 0);
    CK(cudaMalloc(&sc, sb));
    k_fill<<<1024, 256>>>(sc, sb, 13u, 0x28, 0x18);  // e4m3 scales around 0.125 .. 1
    CK(cudaMalloc(&rs, size_t(s.N) * 4));
    k_fill_f<<<64, 256>>>(rs, size_t(s.N), 17u, 0.01f, 0.01f);
    CK(cudaMalloc(&x, size_t(2048) * s.K * 4));
    k_fill_f<<<1024, 256>>>(x, size_t(2048) * s.K, 19u, -2.f, 4.f);
    CK(cudaMalloc(&y, size_t(2048) * s.N * 4));
    for (int rows : {24, 32, 48, 64, 96, 128, 256, 512, 1024, 2048}) {
      std::printf("%-14s r%-5d", s.name, rows);
      const double tf = 2.0 * rows * s.N * s.K / 1e12;
      for (int fp4 = 0; fp4 < 2; ++fp4) {
        for (int c = 0; c < kTcConfigs; ++c) {
          const double ms = time_ms([&] {
            if (fp4) gemm_fp4_tc(x, s.K, q, sc, rs, s.N, s.K, y, s.N, rows, true, BL(0), c);
            else gemm_tc(x, s.K, W.next(), s.N, s.K, y, s.N, rows, true, BL(0), c);
          }, 10);
          std::printf(" cfg%d %5.1f", c, tf / (ms / 1e3));
        }
        std::printf(fp4 ? "\n" : "  |");
      }
    }
    cudaFree(q);
    cudaFree(sc);
    cudaFree(rs);
    cudaFree(x);
    cudaFree(y);
  }
}

// NVFP4 prefill GEMMs: bf16 activations on bf16 tensor cores (gemm_fp4_tc,
// configuration 0, the one used) against NVFP4 activations on FP4 tensor
// cores (quantize_rows_nvfp4 + gemm_w4a4), TFLOPS.
void bench_w4a4() {
  header();
  if (!gemm_w4a4_supported()) {
    std::printf("no W4A4 GEMM on this device or build\n");
    return;
  }
  std::printf("NVFP4 prefill GEMM TFLOPS: bf16 activations | W4A4 with quantization | W4A4 GEMM alone\n");
  const Shape shapes[] = {{"gdn in", 12352, 2560},      {"attn in", 10240, 2560},     {"mlp gate|up", 18432, 2560},
                          {"mlp down", 2560, 9216},      {"out", 2560, 4096},          {"27B gdn in", 16480, 5120},
                          {"27B attn in", 14336, 5120},  {"27B gate|up", 34816, 5120}, {"27B down", 5120, 17408},
                          {"27B out", 5120, 6144}};
  constexpr int kMaxRows = 4096;
  for (const auto& sh : shapes) {
    const size_t qb = size_t(sh.N) * sh.K / 2, sb = size_t(sh.N) * sh.K / 16;
    uint8_t *q, *sc, *aq, *as;
    float *rs, *x, *y, *ainv;
    CK(cudaMalloc(&q, qb));
    CK(cudaMemset(q, 0x37, qb));
    CK(cudaMalloc(&sc, sb));
    CK(cudaMemset(sc, 0x38, sb));
    CK(cudaMalloc(&rs, size_t(sh.N) * 4));
    CK(cudaMemset(rs, 0, size_t(sh.N) * 4));
    CK(cudaMalloc(&x, size_t(kMaxRows) * sh.K * 4));
    CK(cudaMemset(x, 0x3f, size_t(kMaxRows) * sh.K * 4));  // 0.747 each
    CK(cudaMalloc(&aq, size_t(kMaxRows) * sh.K / 2));
    CK(cudaMalloc(&as, size_t(kMaxRows) * sh.K / 16));
    CK(cudaMalloc(&ainv, kMaxRows * 4));
    CK(cudaMalloc(&y, size_t(kMaxRows) * sh.N * 4));
    for (int rows : {512, 1024, 2048, kMaxRows}) {
      const double tf = 2.0 * rows * sh.N * sh.K / 1e12;
      // The bf16 path's scratch holds 2048 rows of the longest K.
      const double a = rows <= 2048 ? time_ms([&] {
        gemm_fp4_tc(x, sh.K, q, sc, rs, sh.N, sh.K, y, sh.N, rows, true, BL(0), 0);
      }, 10) : 1e30;
      const double b = time_ms([&] {
        quantize_rows_nvfp4(x, sh.K, rows, sh.K, aq, as, ainv, 0);
        gemm_w4a4(aq, as, ainv, q, sc, rs, sh.N, sh.K, y, sh.N, rows, true, 0);
      }, 10);
      std::printf("%-12s r%-5d %6.1f | %6.1f |", sh.name, rows, tf / (a / 1e3), tf / (b / 1e3));
      for (int cfg = 0; cfg < 5; ++cfg) {
        const double c = time_ms([&] { gemm_w4a4(aq, as, ainv, q, sc, rs, sh.N, sh.K, y, sh.N, rows, true, 0, cfg); }, 10);
        std::printf(" cfg%d %6.1f", cfg, tf / (c / 1e3));
      }
      std::printf("   (%.0f -> %.0f us)\n", a * 1e3, b * 1e3);
    }
    for (void* p : {(void*)q, (void*)sc, (void*)rs, (void*)x, (void*)aq, (void*)as, (void*)ainv, (void*)y})
      cudaFree(p);
  }
}

// Reads n 16-byte words and keeps a value that depends on all of them (the read ceiling).
__global__ void k_read(const uint4* __restrict__ p, size_t n, unsigned* sink) {
  unsigned a = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcs(p + i);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0x9e3779b9u) *sink = a;
}

// The decode GEMMs as a round runs them (NVFP4, input already split, one
// row), each shape as a CUDA graph of 24 calls on rotating weights, against a
// streaming read of the same bytes in a graph: what a call costs beyond its
// bytes.
void bench_decode() {
  header();
  struct D {
    const char* name;
    int N, K;
  };
  const D shapes[] = {{"gdn in", 12352, 2560},      {"attn in", 10240, 2560},     {"mlp gate|up", 18432, 2560},
                      {"mlp down", 2560, 9216},      {"out", 2560, 4096},          {"27B gdn in", 16480, 5120},
                      {"27B attn in", 14336, 5120},  {"27B gate|up", 34816, 5120}, {"27B down", 5120, 17408},
                      {"27B out", 5120, 6144}};
  const int calls = 24;
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 30);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / calls;
  };
  std::printf("us per call (graph of %d), rows 1     |  GEMM    GB/s |  read    GB/s | GEMM - read\n", calls);
  for (const auto& d : shapes) {
    const size_t qb = size_t(d.N) * d.K / 2, sb = size_t(d.N) * d.K / 16;
    std::vector<uint8_t*> Q(calls), S(calls);
    for (int i = 0; i < calls; ++i) {
      CK(cudaMalloc(&Q[i], qb + sb));
      CK(cudaMemset(Q[i], 0x11, qb));
      S[i] = Q[i] + qb;
      CK(cudaMemset(S[i], 0x38, sb));
    }
    float *rs, *y;
    CK(cudaMalloc(&rs, size_t(d.N) * 4));
    CK(cudaMemset(rs, 0, size_t(d.N) * 4));
    CK(cudaMalloc(&y, size_t(16) * d.N * 4));
    const double gemm = graph_us([&] {
      for (int i = 0; i < calls; ++i) gemm_fp4_presplit(Q[i], S[i], rs, d.N, d.K, y, d.N, 1, true, BL(st));
    });
    const double read = graph_us([&] {
      for (int i = 0; i < calls; ++i)
        k_read<<<sms * 8, 256, 0, st>>>(reinterpret_cast<const uint4*>(Q[i]), (qb + sb) / 16, sink);
    });
    const double gb = double(qb + sb) / 1e3;  // bytes / 1e3 / us = GB/s
    std::printf("%-12s %6d x %5d %5.1f MB | %5.1f %5.0f  | %5.1f %5.0f  | %+5.1f\n", d.name, d.N, d.K,
                (qb + sb) / 1e6, gemm, gb / gemm, read, gb / read, gemm - read);
    for (auto* q : Q) cudaFree(q);
    cudaFree(rs);
    cudaFree(y);
  }
  cudaFree(sink);
}

// Every NVFP4 configuration of one shape as decoding runs it: a graph of 24 calls on rotating weights (so none
// stays in L2), launched as the engine launches them on this GPU (its tuning table's options: programmatic dependent
// launch on the RTX 5090), rows 1, 2, 4, 8, 16: GB/s.  cuda_bench fp4cfgs N K
void bench_fp4cfgs(int argc, char** argv) {
  header();
  if (argc < 4) {
    std::printf("usage: cuda_bench fp4cfgs N K\n");
    return;
  }
  const int N = std::atoi(argv[2]), K = std::atoi(argv[3]), calls = 24;
  const int rows_list[] = {1, 2, 4, 8, 16};
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const size_t qb = size_t(N) * K / 2, sb = size_t(N) * K / 16;
  std::vector<uint8_t*> Q(calls), S(calls);
  for (int i = 0; i < calls; ++i) {
    CK(cudaMalloc(&Q[i], qb + sb));
    CK(cudaMemset(Q[i], 0x11, qb));
    S[i] = Q[i] + qb;
    CK(cudaMemset(S[i], 0x38, sb));
  }
  float *rs, *x, *y;
  CK(cudaMalloc(&rs, size_t(N) * 4));
  CK(cudaMemset(rs, 0, size_t(N) * 4));
  CK(cudaMalloc(&x, size_t(16) * K * 4));
  CK(cudaMemset(x, 0, size_t(16) * K * 4));
  CK(cudaMalloc(&y, size_t(16) * N * 4));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 30);  // time_ms's events are on stream 0
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / calls;
  };
  const double gb = double(qb + sb) / 1e3;
  std::printf("%d x %d (%.1f MB), GB/s by rows; span chunks nt warps unroll\n", N, K, (qb + sb) / 1e6);
  for (int c = 0; c < kFp4ConfigCount; ++c) {
    const Fp4Config& f = kFp4Configs[c];
    if (K % f.span) continue;
    std::printf("cfg %2d: %3d %2d %d %2d %d |", c, f.span, f.chunks, f.nt, f.warps, f.unroll);
    for (int rows : rows_list) {
      const double us = graph_us([&] {
        for (int i = 0; i < calls; ++i)
          gemm_fp4_skinny_cfg(c, nullptr, K, Q[i], S[i], rs, N, K, y, N, rows, false, BL(st));  // staged input
      });
      CK(cudaGetLastError());
      std::printf(" r%-2d %5.0f", rows, gb / us);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  for (auto* q : Q) cudaFree(q);
  cudaFree(rs);
  cudaFree(x);
  cudaFree(y);
}

// A round's NVFP4 GEMM of many rows (gemm_fp4_rows: up to four 16-row tiles a weight decode, phase 2 of tree rounds)
// against 16 rows at a time, as decoding runs them (a graph of 24 calls on rotating weights): us a call and GB/s of
// weights.  cuda_bench fp4rows N K
void bench_fp4rows(int argc, char** argv) {
  header();
  if (argc < 4) {
    std::printf("usage: cuda_bench fp4rows N K\n");
    return;
  }
  const int N = std::atoi(argv[2]), K = std::atoi(argv[3]), calls = 24;
  cudaStream_t st;  // captured on, launched on stream 0 (time_ms's events)
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const size_t qb = size_t(N) * K / 2, sb = size_t(N) * K / 16;
  std::vector<uint8_t*> Q(calls), S(calls);
  for (int i = 0; i < calls; ++i) {
    CK(cudaMalloc(&Q[i], qb + sb));
    CK(cudaMemset(Q[i], 0x11, qb));
    S[i] = Q[i] + qb;
    CK(cudaMemset(S[i], 0x38, sb));
  }
  float *rs, *x, *y;
  CK(cudaMalloc(&rs, size_t(N) * 4));
  CK(cudaMemset(rs, 0, size_t(N) * 4));
  CK(cudaMalloc(&x, size_t(64) * K * 4));
  CK(cudaMemset(x, 0, size_t(64) * K * 4));
  CK(cudaMalloc(&y, size_t(64) * N * 4));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 30);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / calls;
  };
  const double gb = double(qb + sb) / 1e3;
  std::printf("%d x %d (%.1f MB): us a call (GB/s of weights)\n", N, K, (qb + sb) / 1e6);
  // W4A4 (NVFP4 activations on the FP4 tensor cores, sm_120a): the rows quantized, then the GEMM
  uint8_t *aq = nullptr, *as = nullptr;
  float* ainv = nullptr;
  const bool w4a4 = gemm_w4a4_supported();
  if (w4a4) {
    CK(cudaMalloc(&aq, size_t(64) * K / 2));
    CK(cudaMalloc(&as, size_t(64) * K / 16));
    CK(cudaMalloc(&ainv, 64 * 4));
  }
  std::printf("rows | many rows       | 16 at a time    | W4A4 (rows in 4 bits)\n");
  for (int rows : {1, 16, 24, 32, 40, 48, 64}) {
    const double many = graph_us([&] {
      for (int i = 0; i < calls; ++i) gemm_fp4_rows(x, K, Q[i], S[i], rs, N, K, y, N, rows, false, BL(st));
    });
    const double slices = graph_us([&] {
      for (int i = 0; i < calls; ++i)
        for (int r0 = 0; r0 < rows; r0 += 16)
          gemm_fp4(x + size_t(r0) * K, K, Q[i], S[i], rs, N, K, y + size_t(r0) * N, N, std::min(16, rows - r0), false,
                   BL(st));
    });
    const double a4 = !w4a4 ? 0.0 : graph_us([&] {
      for (int i = 0; i < calls; ++i) {
        quantize_rows_nvfp4(x, K, rows, K, aq, as, ainv, st);
        gemm_w4a4(aq, as, ainv, Q[i], S[i], rs, N, K, y, N, rows, false, st);
      }
    });
    CK(cudaGetLastError());
    std::printf("%4d | %7.1f (%5.0f) | %7.1f (%5.0f) | %7.1f (%5.0f)\n", rows, many, gb / many, slices, gb / slices, a4,
                a4 > 0 ? gb / a4 : 0.0);
    std::fflush(stdout);
  }
  for (void* p : {(void*)aq, (void*)as, (void*)ainv}) cudaFree(p);
  for (auto* q : Q) cudaFree(q);
  cudaFree(rs);
  cudaFree(x);
  cudaFree(y);
}

// Decode attention (one row, Qwen3.5-4B: 16 query heads, 4 KV heads, head dim
// 256, 32 key splits) over 8 layers' caches in a graph, by context length,
// against a streaming read of the same K/V bytes.
// cuda_bench attn [NH [SPLITS [CTX]]]: NH query heads over 4 KV heads of 256 (16: Qwen3.5-4B; 24: Qwen3.8-27B), SPLITS
// the partials' slots (32; rounds run 64, kRoundSplits); CTX: only the short-context part, 1 row at that context (its
// last launch is the one a profiler takes).
void bench_attn(int nh, int splits, int only_ctx) {
  header();
  const int nkv = 4, hd = 256, layers = 8;
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  float *q, *out, *part;
  CK(cudaMalloc(&q, size_t(nh) * hd * 4));
  CK(cudaMemset(q, 0, size_t(nh) * hd * 4));
  CK(cudaMalloc(&out, size_t(nh) * hd * 4));
  CK(cudaMalloc(&part, attention_partial_floats(1, nh, hd, splits) * 4));
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 20);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / layers;
  };
  std::printf("decode attention, us per layer (graph of %d) | K/V MB | attention GB/s | read GB/s\n", layers);
  for (int ctx : {1024, 4096, 8192, 16384, 32768}) {
    if (only_ctx) break;
    const size_t n = size_t(ctx) * nkv * hd;  // values per cache (binary16)
    std::vector<kv16*> K(layers), V(layers);
    for (int i = 0; i < layers; ++i) {
      CK(cudaMalloc(&K[i], n * 2));
      CK(cudaMemset(K[i], 0, n * 2));
      CK(cudaMalloc(&V[i], n * 2));
      CK(cudaMemset(V[i], 0, n * 2));
    }
    DevParams hp{};
    hp.position = ctx - 1;  // the row at position ctx - 1 sees ctx keys
    CK(cudaMemcpy(P, &hp, sizeof(hp), cudaMemcpyHostToDevice));
    const double att = graph_us([&] {
      for (int i = 0; i < layers; ++i)
        attention(q, K[i], V[i], nh, nkv, hd, 1, true, 0, 0.0625f, P, nullptr, 0, out, nh * hd, part, splits, BL(st), 0);
    });
    const double rd = graph_us([&] {
      for (int i = 0; i < layers; ++i) {
        k_read<<<sms * 8, 256, 0, st>>>(reinterpret_cast<const uint4*>(K[i]), n * 2 / 16, sink);
        k_read<<<sms * 8, 256, 0, st>>>(reinterpret_cast<const uint4*>(V[i]), n * 2 / 16, sink);
      }
    });
    const double mb = 2.0 * n * 2 / 1e6;
    std::printf("  ctx %5d | %6.1f us | %6.1f | %5.0f | %5.0f\n", ctx, att, mb, mb * 1e3 / att, mb * 1e3 / rd);
    for (int i = 0; i < layers; ++i) {
      cudaFree(K[i]);
      cudaFree(V[i]);
    }
  }
  // Short contexts as decoding meets them: the cache comes from DRAM (a 96 MB
  // write before each layer evicts the 48 MB L2, as the weights do in a
  // round); us per layer = (write + attention) - write alone.  Rows 1 (plain
  // decoding) and 4 (a verification round).
  void* flush;
  const size_t flush_bytes = size_t(96) << 20;
  CK(cudaMalloc(&flush, flush_bytes));
  float* q4;
  CK(cudaMalloc(&q4, size_t(4) * nh * hd * 4));
  k_fill_f<<<256, 256>>>(q4, size_t(4) * nh * hd, 29u, -1.f, 2.f);
  float* out4;
  CK(cudaMalloc(&out4, size_t(4) * nh * hd * 4));
  float* part4;
  CK(cudaMalloc(&part4, attention_partial_floats(4, nh, hd, splits) * 4));
  std::printf("short contexts, cache from DRAM: us per layer, rows 1 | rows 4\n");
  for (int ctx : {64, 128, 256, 512, 1024, 2048}) {
    if (only_ctx) ctx = only_ctx;
    const size_t n = size_t(ctx) * nkv * hd;
    std::vector<kv16*> K(layers), V(layers);
    for (int i = 0; i < layers; ++i) {
      CK(cudaMalloc(&K[i], n * 2));
      k_fill_kv<<<256, 256>>>(K[i], n, 31u + i, -1.f, 2.f);
      CK(cudaMalloc(&V[i], n * 2));
      k_fill_kv<<<256, 256>>>(V[i], n, 37u + i, -1.f, 2.f);
    }
    DevParams hp{};
    hp.position = ctx - 4;  // rows 4: the last row sees ctx keys
    CK(cudaMemcpy(P, &hp, sizeof(hp), cudaMemcpyHostToDevice));
    const double base = graph_us([&] {
      for (int i = 0; i < layers; ++i) CK(cudaMemsetAsync(flush, i, flush_bytes, st));
    });
    std::printf("  ctx %5d |", ctx);
    for (int rows : {1, 4}) {
      if (only_ctx && rows != 1) break;
      const double t = graph_us([&] {
        for (int i = 0; i < layers; ++i) {
          CK(cudaMemsetAsync(flush, i, flush_bytes, st));
          attention(q4, K[i], V[i], nh, nkv, hd, rows, true, 0, 0.0625f, P, nullptr, 0, out4, nh * hd, part4, splits,
                    BL(st), 0);
        }
      });
      std::printf(" %6.1f us", t - base);
    }
    std::printf("\n");
    if (only_ctx) {  // the outputs' FNV-1a hash (rows 1 and 4 on the first layer's cache): builds compare their bits
      std::printf("  checksum");
      for (int rows : {1, 4}) {
        attention(q4, K[0], V[0], nh, nkv, hd, rows, true, 0, 0.0625f, P, nullptr, 0, out4, nh * hd, part4, splits,
                  BL(st), 0);
        CK(cudaStreamSynchronize(st));
        std::vector<uint8_t> o(size_t(rows) * nh * hd * 4);
        CK(cudaMemcpy(o.data(), out4, o.size(), cudaMemcpyDeviceToHost));
        uint64_t h = 14695981039346656037ull;
        for (uint8_t b : o) h = (h ^ b) * 1099511628211ull;
        std::printf(" rows %d %016llx", rows, static_cast<unsigned long long>(h));
      }
      std::printf("\n");
    }
    for (int i = 0; i < layers; ++i) {
      cudaFree(K[i]);
      cudaFree(V[i]);
    }
    if (only_ctx) break;
  }
  for (void* p : {flush, (void*)q4, (void*)out4, (void*)part4}) cudaFree(p);
}


// A round's MLP (Qwen3.5-4B shapes, NVFP4) as the four kernels a fused region
// runs and as one persistent kernel (mlp_persistent.cu): same
// bits?  Time per layer in a graph of 32 layers on distinct weights, against a
// streaming read of their bytes.  MLP_PREFETCH=1, MLP_MODE=1|2 (one phase
// alone), MLP_BPS=n (blocks per SM) vary the persistent kernel.
void bench_mlp() {
  header();
  const int H = 2560, I = 9216, layers = 32;
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  KernelOptions opt;
  int arch_cc = 0;
  opt.tuning = gemm_tuning_auto(prop.name, prop.major * 10 + prop.minor, compiled_arch(), &arch_cc);
  std::printf("tuning table: %s\n", gemm_tuning_label(opt.tuning, arch_cc, prop.major * 10 + prop.minor).c_str());
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  const Launch L{st, &g_bench_scratch, &opt};
  struct Layer {
    uint8_t *qgu, *sgu, *qd, *sd;
    float *rsgu, *rsd, *nw;
  };
  const size_t qgu = size_t(2 * I) * H / 2, sgu = size_t(2 * I) * H / 16;
  const size_t qd = size_t(H) * I / 2, sd = size_t(H) * I / 16;
  std::vector<Layer> Ls(layers);
  for (int l = 0; l < layers; ++l) {
    auto& w = Ls[l];
    CK(cudaMalloc(&w.qgu, qgu + sgu));
    CK(cudaMalloc(&w.qd, qd + sd));
    w.sgu = w.qgu + qgu;
    w.sd = w.qd + qd;
    CK(cudaMalloc(&w.rsgu, size_t(2 * I) * 4));
    CK(cudaMalloc(&w.rsd, size_t(H) * 4));
    CK(cudaMalloc(&w.nw, size_t(H) * 4));
    k_fill<<<1024, 256>>>(w.qgu, qgu, 11u + l, 0, 0);
    k_fill<<<1024, 256>>>(w.sgu, sgu, 23u + l, 0x28, 0x18);  // e4m3 scales around 0.125 .. 1
    k_fill<<<1024, 256>>>(w.qd, qd, 37u + l, 0, 0);
    k_fill<<<1024, 256>>>(w.sd, sd, 41u + l, 0x28, 0x18);
    k_fill_f<<<64, 256>>>(w.rsgu, size_t(2 * I), 53u + l, 0.01f, 0.01f);
    k_fill_f<<<64, 256>>>(w.rsd, size_t(H), 59u + l, 0.002f, 0.002f);
    k_fill_f<<<64, 256>>>(w.nw, size_t(H), 61u + l, -0.3f, 0.6f);
  }
  float *x, *x2, *x0, *proj;
  bf16* act;
  unsigned* sync;
  CK(cudaMalloc(&act, size_t(kMaxBenchRows) * I * 2));
  CK(cudaMalloc(&sync, 64));
  CK(cudaMemset(sync, 0, 64));
  auto bench_env_int = [](const char* primary, const char* legacy, int fallback) {
    const char* v = std::getenv(primary);
    if (!v || !*v) v = std::getenv(legacy);
    return v && *v ? std::atoi(v) : fallback;
  };
  MlpPersistentOptions po;
  po.prefetch = bench_env_int("FLYX_BENCH_MLP_PREFETCH", "MLP_PREFETCH", 0) != 0;
  po.mode = bench_env_int("FLYX_BENCH_MLP_MODE", "MLP_MODE", 3);
  po.blocks_per_sm = bench_env_int("FLYX_BENCH_MLP_BPS", "MLP_BPS", 0);
  int bps = 0;
  CK(cudaMalloc(&x, size_t(kMaxBenchRows) * H * 4));
  CK(cudaMalloc(&x2, size_t(kMaxBenchRows) * H * 4));
  CK(cudaMalloc(&x0, size_t(kMaxBenchRows) * H * 4));
  CK(cudaMalloc(&proj, size_t(kMaxBenchRows) * 2 * I * 4));
  k_fill_f<<<64, 256>>>(x0, size_t(kMaxBenchRows) * H, 71u, -2.f, 4.f);
  CK(cudaDeviceSynchronize());
  auto generic = [&](const Layer& w, float* xr, int rows) {
    rmsnorm_split(xr, H, w.nw, 1.f, 1e-6f, rows, H, L);
    gemm_fp4_presplit(w.qgu, w.sgu, w.rsgu, 2 * I, H, proj, 2 * I, rows, false, L);
    swiglu_split(proj, 2 * I, I, rows, L);
    gemm_fp4_presplit(w.qd, w.sd, w.rsd, H, I, xr, H, rows, true, L);
  };
  auto fused = [&](const Layer& w, float* xr, int rows) {
    bps = mlp_persistent(xr, H, w.nw, 1.f, 1e-6f, w.qgu, w.sgu, w.rsgu, w.qd, w.sd, w.rsd, I, rows, act, sync,
                         opt.tuning, po, st);
    if (!bps) {
      std::printf("the persistent kernel does not take rows %d\n", rows);
      std::exit(1);
    }
  };
  auto graph_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    body();
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    const double ms = time_ms([&] { CK(cudaGraphLaunch(ge, 0)); }, 30);
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    return ms * 1e3 / layers;
  };
  unsigned* sink;
  CK(cudaMalloc(&sink, 4));
  const double read = graph_us([&] {
    for (const auto& w : Ls) {
      k_read<<<prop.multiProcessorCount * 8, 256, 0, st>>>(reinterpret_cast<const uint4*>(w.qgu), (qgu + sgu) / 16,
                                                           sink);
      k_read<<<prop.multiProcessorCount * 8, 256, 0, st>>>(reinterpret_cast<const uint4*>(w.qd), (qd + sd) / 16, sink);
    }
  });
  const double mb = double(qgu + sgu + qd + sd) / 1e6;
  std::printf("MLP %d x %d, %.1f MB of weights per layer; streaming read %.1f us (%.0f GB/s)\n", H, I, mb, read,
              mb * 1e3 / read);
  std::printf("persistent kernel: prefetch %d, phases %d%s\n", po.prefetch ? 1 : 0, po.mode,
              po.mode != 3 ? " (not the whole MLP: its bits differ)" : "");
  std::printf("rows | same bits (%d layers) | four kernels us | one kernel us | saved | blocks per SM\n", layers);
  for (int rows : {1, 2, 4, 8}) {
    const size_t bytes = size_t(rows) * H * 4;
    CK(cudaMemcpy(x, x0, bytes, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(x2, x0, bytes, cudaMemcpyDeviceToDevice));
    for (const auto& w : Ls) {
      generic(w, x, rows);
      fused(w, x2, rows);
    }
    CK(cudaStreamSynchronize(st));
    std::vector<float> a(size_t(rows) * H), b(size_t(rows) * H);
    CK(cudaMemcpy(a.data(), x, bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), x2, bytes, cudaMemcpyDeviceToHost));
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += std::memcmp(&a[i], &b[i], 4) != 0;
    const double t4 = graph_us([&] {
      for (const auto& w : Ls) generic(w, x, rows);
    });
    const double t1 = graph_us([&] {
      for (const auto& w : Ls) fused(w, x2, rows);
    });
    std::printf("%4d | %s (%zu of %zu differ) | %6.1f (%4.0f GB/s) | %6.1f (%4.0f GB/s) | %+.1f us | %d\n", rows,
                diff ? "NO " : "yes", diff, a.size(), t4, mb * 1e3 / t4, t1, mb * 1e3 / t1, t1 - t4, bps);
  }
  for (auto& w : Ls) {
    cudaFree(w.qgu);
    cudaFree(w.qd);
    cudaFree(w.rsgu);
    cudaFree(w.rsd);
    cudaFree(w.nw);
  }
  cudaFree(x);
  cudaFree(x2);
  cudaFree(x0);
  cudaFree(proj);
  cudaFree(act);
  cudaFree(sync);
  cudaFree(sink);
}


// Prefill attention of one long prompt in 1024-row passes (by default Qwen3.5-4B: 16 query heads over 4 KV heads,
// head dim 256, the 5760-token prefill of the perf records; `cuda_bench flash NH NKV TOKENS CHUNK` another: the 27B
// 24 4), random queries, keys and values, over a binary16 cache and an FP8 one (e4m3, a scale per position and
// head: the default); ms per pass and in all.
void bench_flash(int argc, char** argv) {
  header();
  const int nh = argc > 2 ? std::atoi(argv[2]) : 16, nkv = argc > 3 ? std::atoi(argv[3]) : 4, hd = 256;
  const int total = argc > 4 ? std::atoi(argv[4]) : 5760, chunk = argc > 5 ? std::atoi(argv[5]) : 1024;
  float *q, *out, *gate, *ks, *vs;
  kv16 *K, *V;
  uint8_t *K8, *V8;
  CK(cudaMalloc(&q, size_t(chunk) * nh * hd * 4));
  k_fill_f<<<1024, 256>>>(q, size_t(chunk) * nh * hd, 3u, -2.f, 4.f);
  CK(cudaMalloc(&gate, size_t(chunk) * nh * hd * 4));
  k_fill_f<<<1024, 256>>>(gate, size_t(chunk) * nh * hd, 5u, -2.f, 4.f);
  CK(cudaMalloc(&out, size_t(chunk) * nh * hd * 4));
  CK(cudaMalloc(&K, size_t(total) * nkv * hd * 2));
  k_fill_kv<<<1024, 256>>>(K, size_t(total) * nkv * hd, 7u, -2.f, 4.f);
  CK(cudaMalloc(&V, size_t(total) * nkv * hd * 2));
  k_fill_kv<<<1024, 256>>>(V, size_t(total) * nkv * hd, 11u, -2.f, 4.f);
  CK(cudaMalloc(&K8, size_t(total) * nkv * hd));
  k_fill<<<1024, 256>>>(K8, size_t(total) * nkv * hd, 13u, 0x20, 0x40);  // positive e4m3 values, no NaN
  CK(cudaMalloc(&V8, size_t(total) * nkv * hd));
  k_fill<<<1024, 256>>>(V8, size_t(total) * nkv * hd, 17u, 0x20, 0x40);
  CK(cudaMalloc(&ks, size_t(total) * nkv * 4));
  k_fill_f<<<64, 256>>>(ks, size_t(total) * nkv, 19u, 0.01f, 0.02f);
  CK(cudaMalloc(&vs, size_t(total) * nkv * 4));
  k_fill_f<<<64, 256>>>(vs, size_t(total) * nkv, 23u, 0.01f, 0.02f);
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  CK(cudaMemset(P, 0, sizeof(DevParams)));
  const float scale = 1.f / 16.f;
  const KvRef caches[2] = {KvRef{K, V, nullptr, nullptr}, KvRef{K8, V8, ks, vs}};
  double sum[2] = {0, 0};
  std::printf("prefill attention, %d tokens in %d-row passes: ms per pass (TFLOPS), binary16 | FP8 cache\n", total,
              chunk);
  for (int p0 = 0; p0 < total; p0 += chunk) {
    const int rows = std::min(chunk, total - p0);
    const double pairs = double(rows) * p0 + double(rows) * (rows + 1) / 2;  // causal (query, key) pairs
    std::printf("  rows %5d at %5d:", rows, p0);
    for (int c = 0; c < 2; ++c) {
      const double ms = time_ms([&] {
        attention_flash(q, caches[c], nh, nkv, hd, rows, scale, P, gate, nh * hd, out, nh * hd, BL(0), p0);
      }, 10);
      std::printf(" %s %7.3f ms (%5.1f TFLOPS)", c ? "|" : "", ms, 4.0 * pairs * hd * nh / (ms * 1e9));
      sum[c] += ms;
    }
    std::printf("\n");
  }
  const double pairs = double(total) * (total + 1) / 2;
  std::printf("  all: %.3f ms (%.1f TFLOPS) | %.3f ms (%.1f TFLOPS) a layer\n", sum[0],
              4.0 * pairs * hd * nh / (sum[0] * 1e9), sum[1], 4.0 * pairs * hd * nh / (sum[1] * 1e9));
  for (void* p : {(void*)q, (void*)gate, (void*)out, (void*)K, (void*)V, (void*)K8, (void*)V8, (void*)ks, (void*)vs,
                  (void*)P})
    cudaFree(p);
}
// A round's exact attention over an FP8 cache in page-locked host memory: `cuda_bench attnhost
// NH NKV CTX` (by default the 27B: 24 query heads over 4 KV heads, head dim 256, 116,999 positions).  For 1 to 16
// rows: the attention reading the cache from device memory, reading it from host memory over the bus, and the copy
// of its host memory to a device staging (kv_stage, 256 blocks, as an attention that waits for it); us and GB/s.
void bench_attn_host(int argc, char** argv) {
  header();
  const int nh = argc > 2 ? std::atoi(argv[2]) : 24, nkv = argc > 3 ? std::atoi(argv[3]) : 4, hd = 256;
  const int ctx = argc > 4 ? std::atoi(argv[4]) : 116999;
  const size_t bytes = size_t(ctx) * nkv * hd;  // of K, and of V
  const cudaStream_t st = 0;  // time_ms's
  float *q, *out, *part, *ks, *vs;
  uint8_t *Kd, *Vd, *Kh, *Vh, *Ks, *Vs;
  CK(cudaMalloc(&q, size_t(16) * nh * hd * 4));
  k_fill_f<<<256, 256>>>(q, size_t(16) * nh * hd, 3u, -2.f, 4.f);
  CK(cudaMalloc(&out, size_t(16) * nh * hd * 4));
  CK(cudaMalloc(&part, attention_partial_floats(16, nh, hd, 64) * 4));
  CK(cudaMalloc(&ks, size_t(ctx) * nkv * 4));
  CK(cudaMalloc(&vs, size_t(ctx) * nkv * 4));
  k_fill_f<<<256, 256>>>(ks, size_t(ctx) * nkv, 5u, 0.01f, 0.02f);
  k_fill_f<<<256, 256>>>(vs, size_t(ctx) * nkv, 7u, 0.01f, 0.02f);
  for (uint8_t** p : {&Kd, &Vd, &Ks, &Vs}) {
    CK(cudaMalloc(p, bytes));
    CK(cudaMemset(*p, 0x38, bytes));  // e4m3 1.0
  }
  for (uint8_t** p : {&Kh, &Vh}) {
    CK(cudaHostAlloc(p, bytes, cudaHostAllocMapped));
    std::memset(*p, 0x38, bytes);
  }
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  const double gb = 2.0 * bytes / 1e9;
  std::printf("round attention over %d positions (FP8, %d over %d heads): us | GB/s of K and V (%.2f GB)\n", ctx, nh,
              nkv, gb);
  std::printf("  rows | device memory   | host memory     | staging copy\n");
  for (int rows : {1, 4, 8, 16}) {
    DevParams hp{};
    hp.position = ctx - rows;
    CK(cudaMemcpy(P, &hp, sizeof(hp), cudaMemcpyHostToDevice));
    auto attn = [&](uint8_t* K, uint8_t* V) {
      return 1e3 * time_ms([&] {
        attention(q, KvRef{K, V, ks, vs}, nh, nkv, hd, rows, true, 0, 0.0625f, P, nullptr, 0, out, nh * hd, part, 64,
                  BL(st), 0);
      }, 10);
    };
    const double dev = attn(Kd, Vd), host = attn(Kh, Vh);
    const double copy = 1e3 * time_ms([&] {
      kv_stage(Kh, Ks, Vh, Vs, size_t(nkv) * hd, 0, ctx, P, 0, rows, kStageAll, 256, st);
    }, 10);
    std::printf("  %4d | %7.0f %6.1f  | %7.0f %6.1f  | %7.0f %6.1f\n", rows, dev, gb * 1e6 / dev, host, gb * 1e6 / host,
                copy, gb * 1e6 / copy);
  }
  for (void* p : {(void*)q, (void*)out, (void*)part, (void*)ks, (void*)vs, (void*)Kd, (void*)Vd, (void*)Ks, (void*)Vs,
                  (void*)P})
    cudaFree(p);
  cudaFreeHost(Kh);
  cudaFreeHost(Vh);
}

// Strata attention of one round over a long context: `cuda_bench strata NH NKV CTX ROWS BITS` (by
// default the 27B: 24 query heads over 4 KV heads, head dim 256, 531,072 positions, 7 rows, a 3-bit copy), random
// queries, copy and FP8 cache (every position on the device), 1 pair in 64 refined, at most 256 a row; ms a round.
// Under the CUPTI tracer the scan, selection and correction show apart.  FLYX_BENCH_STRATA=check also compares the
// tensor-core scan with the float32 one (strata_scan=cores): the heads' combined outputs and the pairs' log-sum-exps.
void bench_strata(int argc, char** argv) {
  header();
  const int nh = argc > 2 ? std::atoi(argv[2]) : 24, nkv = argc > 3 ? std::atoi(argv[3]) : 4, hd = 256;
  const int ctx = argc > 4 ? std::atoi(argv[4]) : 531072, rows = argc > 5 ? std::atoi(argv[5]) : 7;
  const int bits = argc > 6 ? std::atoi(argv[6]) : 3;
  if (!strata_supported(nh, nkv, hd, bits)) {
    std::printf("strata: not compiled for %d over %d at %d bits\n", nh, nkv, bits);
    return;
  }
  const int cap = ctx + rows + 16, words = strata_words(hd, bits), slots = 64;
  const size_t pos = size_t(cap) * nkv;
  float *q, *out, *part, *ks, *vs;
  uint8_t *K, *V;
  uint32_t *kq, *vq;
  __half2 *kqs, *vqs;
  CK(cudaMalloc(&q, size_t(rows) * nh * hd * 4));
  k_fill_f<<<256, 256>>>(q, size_t(rows) * nh * hd, 3u, -2.f, 4.f);
  CK(cudaMalloc(&out, size_t(rows) * nh * hd * 4));
  CK(cudaMalloc(&part, attention_partial_floats(rows, nh, hd, slots) * 4));
  CK(cudaMalloc(&K, pos * hd));
  CK(cudaMalloc(&V, pos * hd));
  CK(cudaMemset(K, 0x30, pos * hd));  // e4m3 0.75: finite
  CK(cudaMemset(V, 0x38, pos * hd));
  CK(cudaMalloc(&ks, pos * 4));
  CK(cudaMalloc(&vs, pos * 4));
  k_fill_f<<<1024, 256>>>(ks, pos, 5u, 0.01f, 0.02f);
  k_fill_f<<<1024, 256>>>(vs, pos, 6u, 0.01f, 0.02f);
  const size_t pos_copy = size_t(strata_positions(cap)) * nkv;  // whole tiles
  CK(cudaMalloc(&kq, pos_copy * words * 4));
  CK(cudaMalloc(&vq, pos_copy * words * 4));
  k_fill_words<<<1024, 256>>>(kq, pos_copy * words, 7u);
  k_fill_words<<<1024, 256>>>(vq, pos_copy * words, 8u);
  CK(cudaMalloc(&kqs, pos * 4));
  CK(cudaMalloc(&vqs, pos * 4));
  {
    std::vector<__half2> sl(pos, __halves2half2(__float2half(-1.5f), __float2half(3.f / (1 << bits))));
    CK(cudaMemcpy(kqs, sl.data(), pos * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(vqs, sl.data(), pos * 4, cudaMemcpyHostToDevice));
  }
  KvRef kv{K, V, ks, vs, kq, vq, kqs, vqs, bits, words, 0};
  StrataScratch sc;
  sc.how.denom = 64;
  sc.how.least = 32;
  sc.how.most = 256;
  sc.lse_stride = ((cap + 1) / 2 + 7) / 8 * 8;
  sc.sel_stride = std::max(1, sc.how.groups(cap));
  sc.chunks = std::clamp((sc.lse_stride + g_bench_opts.strata_chunk - 1) / g_bench_opts.strata_chunk, 1,
                         kStrataSelChunks);
  CK(cudaMalloc(&sc.lse, size_t(rows) * nh * sc.lse_stride * 2));
  CK(cudaMalloc(&sc.sel, size_t(rows) * nkv * sc.sel_stride * 4));
  CK(cudaMalloc(&sc.sel_n, size_t(rows) * nkv * 4));
  CK(cudaMalloc(&sc.cand, size_t(rows) * nkv * sc.chunks * sc.sel_stride * 4));
  CK(cudaMalloc(&sc.cand_key, size_t(rows) * nkv * sc.chunks * sc.sel_stride * 4));
  CK(cudaMalloc(&sc.cand_n, size_t(rows) * nkv * sc.chunks * 4));
  CK(cudaMalloc(&sc.gathered, size_t(rows) * nkv * sc.chunks * sc.sel_stride * 4));
  CK(cudaMalloc(&sc.gathered_key, size_t(rows) * nkv * sc.chunks * sc.sel_stride * 4));
  CK(cudaMalloc(&sc.delta, size_t(rows) * nkv * kStrataSlices * (nh / nkv) * (2 * hd + 1) * 4));
  CK(cudaMalloc(&sc.qr, size_t(rows) * nh * hd * 4));
  DevParams* P;
  CK(cudaMalloc(&P, sizeof(DevParams)));
  CK(cudaMemset(P, 0, sizeof(DevParams)));
  CK(cudaMemcpy(&P->position, &ctx, sizeof(int), cudaMemcpyHostToDevice));
  if (const char* e = std::getenv("FLYX_BENCH_STRATA"); e && std::string(e) == "check") {
    const size_t np = attention_partial_floats(rows, nh, hd, slots), nl = size_t(rows) * nh * sc.lse_stride;
    std::vector<float> pa(np), pr(np);
    std::vector<uint16_t> la(nl), lr(nl);
    for (int cores = 0; cores < 2; ++cores) {
      g_bench_opts.strata_mma = cores == 0;
      CK(cudaMemset(sc.lse, 0x7f, nl * 2));
      attention_strata(q, kv, sc, nh, nkv, hd, rows, 1.f / 16.f, P, nullptr, 0, out, nh * hd, part, slots, BL(0), 0);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(cores ? pr.data() : pa.data(), part, np * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(cores ? lr.data() : la.data(), sc.lse, nl * 2, cudaMemcpyDeviceToHost));
    }
    g_bench_opts.strata_mma = true;
    auto combined = [&](const std::vector<float>& pp, int splits, int r, int h, std::vector<double>& o) {
      const int used = attn_split_used(ctx + r + 1, splits);
      double M = -std::numeric_limits<double>::infinity(), L = 0;
      for (int sp = 0; sp < used; ++sp) M = std::max(M, double(pp[((size_t(r) * nh + h) * slots + sp) * (hd + 2)]));
      o.assign(hd, 0.0);
      for (int sp = 0; sp < used; ++sp) {
        const float* x = &pp[((size_t(r) * nh + h) * slots + sp) * (hd + 2)];
        if (std::isinf(x[0])) continue;
        const double f = std::exp(double(x[0]) - M);
        L += x[1] * f;
        for (int d = 0; d < hd; ++d) o[d] += x[2 + d] * f;
      }
      for (int d = 0; d < hd; ++d) o[d] /= L;
    };
    double worst = 0, mean = 0, lworst = 0, lmean = 0;
    size_t n = 0, nn = 0;
    std::vector<double> oa, orf;
    for (int r = 0; r < rows; ++r)
      for (int h = 0; h < nh; ++h) {
        combined(pa, slots, r, h, oa);
        combined(pr, kAttnCoresSplits, r, h, orf);
        double big = 0, d = 0;
        for (int i = 0; i < hd; ++i) big = std::max(big, std::fabs(orf[i])), d = std::max(d, std::fabs(oa[i] - orf[i]));
        worst = std::max(worst, d / big), mean += d / big, ++n;
      }
    for (size_t i = 0; i < nl; ++i) {
      if (lr[i] == 0x7f7f) continue;
      const double x = __half2float(__ushort_as_half(lr[i])), y = __half2float(__ushort_as_half(la[i]));
      if (!std::isfinite(x)) continue;
      lworst = std::max(lworst, std::fabs(x - y)), lmean += std::fabs(x - y), ++nn;
    }
    std::printf("tensor-core scan against the float32 one: the heads' outputs %.2e worst, %.2e mean (relative to the "
                "largest value); the pairs' log-sum-exps %.2e worst, %.2e mean\n",
                worst, mean / n, lworst, lmean / nn);
  }
  const double ms = time_ms([&] {
    attention_strata(q, kv, sc, nh, nkv, hd, rows, 1.f / 16.f, P, nullptr, 0, out, nh * hd, part, slots, BL(0), 0);
  }, 10);
  std::printf("strata attention: %d over %d heads, %d positions, %d rows, %d bits: %.3f ms a round (a layer)\n", nh,
              nkv, ctx, rows, bits, ms);
  for (void* p : {(void*)q, (void*)out, (void*)part, (void*)K, (void*)V, (void*)ks, (void*)vs, (void*)kq, (void*)vq,
                  (void*)kqs, (void*)vqs, (void*)sc.lse, (void*)sc.sel, (void*)sc.sel_n,
                  (void*)sc.cand, (void*)sc.cand_key, (void*)sc.cand_n, (void*)sc.gathered, (void*)sc.gathered_key,
                  (void*)sc.delta, (void*)sc.qr, (void*)P})
    cudaFree(p);
}
// The output head (248320 x 2560) in bf16 and in W12 (the same bits), per
// skinny configuration and row count: us per GEMM.
void bench_w12() {
  header();
  const int N = 248320, K = 2560;
  bf16* w;
  CK(cudaMalloc(&w, size_t(N) * K * 2));
  k_fill_bf16<<<1024, 256>>>(w, size_t(N) * K, 101u, -0.05f, 0.1f);
  W12 c;
  CK(cudaMalloc(&c.hi, size_t(N) * K));
  CK(cudaMalloc(&c.lo, size_t(N) * K / 2));
  CK(cudaMalloc(&c.e0, N));
  CK(cudaMalloc(&c.esc_start, (size_t(N) + 1) * 4));
  std::vector<int> starts(size_t(N) + 1);
  const int esc = w12_encode(w, N, K, &c, starts.data(), 0);
  CK(cudaMalloc(&c.esc_k, std::max(esc, 1) * 2));
  CK(cudaMalloc(&c.esc_v, std::max(esc, 1) * 2));
  const int bad = w12_finish(w, N, K, &c, 0);
  float *x, *y;
  CK(cudaMalloc(&x, size_t(16) * K * 4));
  k_fill_f<<<64, 256>>>(x, size_t(16) * K, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(16) * N * 4));
  std::printf("head %d x %d: %d escapes, %d bad groups; us per GEMM, bf16 | w12 (GB/s of bf16 / w12 bytes)\n", N, K,
              esc, bad);
  for (int cfg = 0; cfg < kSkinnyConfigs; ++cfg) {
    const auto& b = kBf16Configs[cfg];
    if (!bench_skinny_cfg_fits(cfg, N, true)) continue;
    std::printf("cfg%d nt%d w%2d u%2d", cfg, b.nt, b.warps, b.unroll);
    for (int rows : {1, 4, 8, 16}) {
      const double a = time_ms([&] { gemm_skinny_cfg(cfg, x, K, w, N, K, y, N, rows, false, BL(0)); }, 20);
      const double d = time_ms([&] { gemm_w12_skinny_cfg(cfg, x, K, c, N, K, y, N, rows, false, BL(0)); }, 20);
      std::printf(" | r%-2d %6.0f %6.0f (%4.0f %4.0f)", rows, a * 1e3, d * 1e3, size_t(N) * K * 2 / (a * 1e6),
                  w12_plane_bytes(N, K) / (d * 1e6));
    }
    std::printf("\n");
  }
  for (void* p : {(void*)w, (void*)c.hi, (void*)c.lo, (void*)c.e0, (void*)c.esc_start, (void*)c.esc_k, (void*)c.esc_v,
                  (void*)x, (void*)y})
    cudaFree(p);
}

// GEMMs on packed GGML block weights, per block type, on the
// shapes of Qwen3.8-27B: us per GEMM and GB/s of the packed bytes it reads,
// for 1, 4 and 16 rows (the input staged once; weights from DRAM).  Random
// bytes are valid blocks of every type; only the head is timed in one type.
// `cuda_bench blocks [TYPE [configs]]`: the default configuration, or with `configs` every compiled one.
void bench_blocks(const char* only, bool configs) {
  header();
  const int k_max = 17408;
  void* mem = nullptr;
  const int prefill_rows = 512;  // a prefill pass: the other kernel
  const size_t bytes = scratch_bytes(prefill_rows, k_max);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, prefill_rows, k_max);
  const Launch L{0, &scratch, &g_bench_opts};
  float *x, *y;
  CK(cudaMalloc(&x, size_t(prefill_rows) * k_max * 4));
  k_fill_f<<<64, 256>>>(x, size_t(prefill_rows) * k_max, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(prefill_rows) * 17408 * 4));  // (the head runs 16 rows at most: 16 x 248320 fits too)
  for (const BlockType& t : kBlockTypes) {
    if (only && std::string(only) != t.name) continue;
    const BlockLayout l = blocks_layout(t.dtype);
    for (const Shape& s : kBlockShapes) {
      const bool head = s.N > 100000;
      if (head && t.dtype != FLYX_DT_GGML_Q4_K && !only) continue;
      const size_t bytes = size_t(s.N) * s.K / l.values * (l.stride + 2 * l.scales);
      std::vector<BlockMat> w = block_copies(t, s);
      for (int cfg = -1; cfg < (configs ? kBlocksConfigCount : 0); ++cfg) {
        if (cfg < 0) std::printf("%-8s %-13s %6d x %5d  %5.1f MB", t.name, s.name, s.N, s.K, bytes / 1e6);
        else std::printf("    nt %d nw %-2d kw %d                          ", kBlocksConfigs[cfg].nt, kBlocksConfigs[cfg].nw, kBlocksConfigs[cfg].kw);
        for (int rows : {1, 4, 8, 16}) {
          blocks_stage(x, k_max, s.K, rows, L);
          size_t i = 0;
          bool fits = true;
          const double ms = time_ms(
              [&] {
                const BlockMat& m = w[i++ % w.size()];
                if (cfg < 0) gemm_blocks_staged(m, s.K, y, s.N, rows, false, L);
                else fits = gemm_blocks_cfg(cfg, m, s.K, y, s.N, rows, false, L);
              },
              30);
          if (fits) std::printf(" | r%-2d %7.0f us %4.0f GB/s", rows, ms * 1e3, bytes / (ms * 1e6));
          else std::printf(" | r%-2d       no room     ", rows);
        }
        if (cfg < 0) {  // rounds with inputs in 8 bits (FLYX_ACT_Q8): GB/s of the dp4a kernel / the tensor cores'
                        // with the rows in the MMA's first operand / with the columns there
          std::printf(" | in 8 bits, dp4a / mma / cols");
          for (int rows : {1, 2, 4, 8, 16}) {
            blocks_stage(x, k_max, s.K, rows, L, true);
            std::printf(" r%-2d", rows);
            for (int kernel : {0, 1, 2}) {
              size_t i = 0;
              const double ms = time_ms(
                  [&] {
                    BlockMat b = w[i++ % w.size()];
                    b.q8 = true;
                    gemm_blocks_round_q8_kernel(kernel, b, s.K, y, s.N, rows, false, L);
                  },
                  30);
              std::printf("%s%4.0f", kernel ? " / " : " ", bytes / (ms * 1e6));
            }
          }
        }
        if (cfg < 0 && !head) {  // the prefill kernel: TFLOPS of 2 M N K
          blocks_stage(x, k_max, s.K, prefill_rows, L);
          size_t i = 0;
          const double ms = time_ms([&] { gemm_blocks_staged(w[i++ % w.size()], s.K, y, s.N, prefill_rows, false, L); }, 10);
          std::printf(" | r%d %6.2f ms %5.1f TFLOPS", prefill_rows, ms, 2.0 * prefill_rows * s.N * s.K / (ms * 1e9));
          blocks_stage(x, k_max, s.K, prefill_rows, L, true);  // the same rows in 8 bits (FLYX_ACT_Q8)
          const double ms8 = time_ms([&] {
            BlockMat b = w[i++ % w.size()];
            b.q8 = true;
            gemm_blocks_staged(b, s.K, y, s.N, prefill_rows, false, L);
          }, 10);
          std::printf(" | in 8 bits %6.2f ms %5.1f TOPS", ms8, 2.0 * prefill_rows * s.N * s.K / (ms8 * 1e9));
          if (const int runs = gemm_blocks_q8_runs(g_bench_opts.tuning, s.N, s.K); runs > 1) {  // in turn / side by side
            std::printf(" | %d runs", runs);
            for (int spread : {0, 1}) {
              const double msr = time_ms([&] {
                BlockMat b = w[i++ % w.size()];
                b.q8 = true;
                gemm_blocks_q8_cfg(gemm_blocks_q8_default(g_bench_opts.tuning, t.dtype, s.N, prefill_rows), b, s.K, y,
                                   s.N, prefill_rows, false, L, spread);
              }, 10);
              std::printf("%s%.3f ms", spread ? " / " : " ", msr);
            }
          }
          for (int c = 0; configs && c < kBlocksQ8ConfigCount; ++c) {
            const double msc = time_ms([&] {
              BlockMat b = w[i++ % w.size()];
              b.q8 = true;
              gemm_blocks_q8_cfg(c, b, s.K, y, s.N, prefill_rows, false, L);
            }, 10);
            std::printf(" bn%d/qk%d/bm%d %.3f", kBlocksQ8Configs[c].bn, kBlocksQ8Configs[c].qk, kBlocksQ8Configs[c].bm,
                        msc);
          }
        }
        std::printf("\n");
      }
      free_copies(w);
    }
  }
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
}

// Rounds with inputs in 8 bits on the layers above (type T, else every type): GB/s of every compiled configuration of
// the dp4a kernel (variants blocks_round: columns a warp x groups ahead, 0 by the type's unit) whose kernel holds the
// round's 1, 2, 4, 6 or 8 rows (and the kernel of 2 rows for 1 row: a round of 1 row can run it), and of the two
// tensor-core kernels (mma: the rows in the MMA's first operand, cols: the columns there); * the defaults'.  The
// numbers behind blocks_round (bench/microbench/round_rules.py).
void bench_rounds(const char* only) {
  header();
  const int k_max = 17408, max_rows = 16;
  void* mem = nullptr;
  const size_t bytes = scratch_bytes(max_rows, k_max);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, max_rows, k_max);
  const Launch L{0, &scratch, &g_bench_opts};
  float *x, *y;
  CK(cudaMalloc(&x, size_t(max_rows) * k_max * 4));
  k_fill_f<<<64, 256>>>(x, size_t(max_rows) * k_max, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(max_rows) * 248320 * 4));
  std::printf("GB/s of the weights a round; rRxCxU: the kernel of R rows, C columns a warp, U groups ahead (0: by the "
              "type's unit); * the default\n");
  for (const BlockType& t : kBlockTypes) {
    if (only && std::string(only) != t.name) continue;
    const BlockLayout l = blocks_layout(t.dtype);
    for (const Shape& s : kBlockShapes) {
      const bool head_type = t.dtype == FLYX_DT_GGML_Q4_K || t.dtype == FLYX_DT_GGML_Q6_K;
      if (s.N > 100000 && !head_type && !only) continue;  // the head: the types files give it
      const size_t wbytes = size_t(s.N) * s.K / l.values * (l.stride + 2 * l.scales);
      std::vector<BlockMat> w = block_copies(t, s);
      for (int rows : {1, 2, 4, 6, 8}) {
        blocks_stage(x, k_max, s.K, rows, L, true);
        std::printf("%-8s %-13s %6d x %5d r%d", t.name, s.name, s.N, s.K, rows);
        const int def_kernel = gemm_blocks_round_default_kernel(g_bench_opts.tuning, t.dtype, s.N, rows);
        const int def = def_kernel == 0 ? gemm_blocks_round_default(g_bench_opts.tuning, t.dtype, s.N, rows) : -1;
        const int held = rows <= 1 ? 1 : rows <= 2 ? 2 : rows <= 4 ? 4 : 8;
        // the candidates: dp4a configurations (>= 0) and the tensor-core kernels (-1 mma, -2 cols, -3 split)
        std::vector<int> cands;
        for (int c = 0; c < kBlocksRoundConfigCount; ++c)
          if (kBlocksRoundConfigs[c].rows == held || (rows == 1 && kBlocksRoundConfigs[c].rows == 2)) cands.push_back(c);
        cands.push_back(-1);
        cands.push_back(-2);
        cands.push_back(-3);
        if (rows <= 8) cands.push_back(-4);
        size_t i = 0;
        auto timed = [&](int c, int reps) {
          return time_ms(
              [&] {
                BlockMat b = w[i++ % w.size()];
                b.q8 = true;
                if (c >= 0) gemm_blocks_round_q8_cfg(c, b, s.K, y, s.N, rows, false, L);
                else gemm_blocks_round_q8_kernel(-c, b, s.K, y, s.N, rows, false, L);
              },
              reps);
        };
        // The clocks settle first; then every candidate twice, in order and in reverse, the faster of the two.
        timed(def_kernel == 0 ? def : -def_kernel, 200);
        std::vector<double> best(cands.size(), 1e30);
        for (int pass = 0; pass < 2; ++pass)
          for (size_t j = 0; j < cands.size(); ++j) {
            const size_t k = pass ? cands.size() - 1 - j : j;
            best[k] = std::min(best[k], timed(cands[k], 100));
          }
        for (size_t k = 0; k < cands.size(); ++k) {
          const int c = cands[k];
          const double gbs = wbytes / (best[k] * 1e6);
          if (c >= 0) {
            const BlocksRoundConfig& cf = kBlocksRoundConfigs[c];
            std::printf(" | r%dx%dx%d%s%s%s %4.0f", cf.rows, cf.columns, cf.ahead,
                        cf.staged == 2 ? (cf.warps == 16 ? "w16" : "w") : cf.staged ? (cf.warps == 16 ? "s16" : "s") : "", cf.parts > 1 ? (cf.parts == 2 ? "p2" : "p4") : "",
                        def_kernel == 0 && c == def ? "*" : " ", gbs);
          } else {
            std::printf(" | %s%s %4.0f", c == -1 ? "mma" : c == -2 ? "cols" : c == -3 ? "split" : "splitc", def_kernel == -c ? "*" : " ", gbs);
          }
        }
        std::printf("\n");
      }
      free_copies(w);
    }
  }
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
}

// One round on one layer, for a profiler: cuda_bench round TYPE LAYER ROWS [KERNEL [CFG]], LAYER an index into the
// benchmarks' layers (0: the 27B's mlp gate, up) or NxK, KERNEL 0 dp4a, 1 the rows in the MMA's first operand, 2 the
// columns there, 3 the rows there and the streams split among blocks, 4 the columns there and the streams split (up to
// 8 rows), 5 the streams split among a block's warps (the types it takes; others: 3), -1 the rules' (the default), CFG
// the dp4a configuration (-1 the rules').  23 launches to settle, then 20 timed; us and GB/s; then one more on the
// first copy of the weights, whose outputs' FNV-1a hash it prints (a profiler that replaces that launch checks it gave
// the same bits).  Under Nsight Compute the 31st, say:
//   ncu --kernel-name regex:k_round_blocks --launch-skip 30 --launch-count 1 --set full cuda_bench round Q4_K 0 8 2
// Under eprofilier's region probes the last:
//   eprofilier run sass-regions --binary cuda_bench --kernel k_round_blocks --occurrence -1 --check checksum
//     --region ... --output DIR -- cuda_bench round Q4_K 0 8 2
void bench_round(int argc, char** argv) {
  const BlockType* t = nullptr;
  for (const BlockType& b : kBlockTypes)
    if (argc > 2 && std::string(argv[2]) == b.name) t = &b;
  Shape s{"NxK", 0, 0};
  if (argc > 3 && std::sscanf(argv[3], "%dx%d", &s.N, &s.K) != 2) {
    const int layer = std::atoi(argv[3]);
    if (layer >= 0 && layer < int(std::size(kBlockShapes))) s = kBlockShapes[layer];
  }
  const int rows = argc > 4 ? std::atoi(argv[4]) : 0;
  const int kernel = argc > 5 ? std::atoi(argv[5]) : -1, cfg = argc > 6 ? std::atoi(argv[6]) : -1;
  if (!t || s.N < 1 || s.K < 256 || s.K > 17408 || s.K % 256 || s.N > 248320 || rows < 1 || rows > 16 || kernel > 5 ||
      cfg >= kBlocksRoundConfigCount || (cfg >= 0 && rows > kBlocksRoundConfigs[cfg].rows)) {
    std::printf("usage: cuda_bench round TYPE LAYER|NxK ROWS [KERNEL [CFG]]  (ROWS 1..16, at most CFG's; K %% 256 == 0, "
                "at most 17408)\n");
    std::exit(2);
  }
  header();
  const int k_max = 17408, max_rows = 16;
  void* mem = nullptr;
  const size_t bytes = scratch_bytes(max_rows, k_max);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, max_rows, k_max);
  const Launch L{0, &scratch, &g_bench_opts};
  float *x, *y;
  CK(cudaMalloc(&x, size_t(max_rows) * k_max * 4));
  k_fill_f<<<64, 256>>>(x, size_t(max_rows) * k_max, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(max_rows) * s.N * 4));
  blocks_stage(x, k_max, s.K, rows, L, true);
  const BlockLayout l = blocks_layout(t->dtype);
  const size_t wbytes = size_t(s.N) * s.K / l.values * (l.stride + 2 * l.scales);
  std::vector<BlockMat> w = block_copies(*t, s);
  size_t i = 0;
  auto run = [&] {
    BlockMat b = w[i++ % w.size()];
    b.q8 = true;
    if (cfg >= 0) gemm_blocks_round_q8_cfg(cfg, b, s.K, y, s.N, rows, false, L);
    else if (kernel >= 0) gemm_blocks_round_q8_kernel(kernel, b, s.K, y, s.N, rows, false, L);
    else gemm_blocks_staged(b, s.K, y, s.N, rows, false, L);
  };
  for (int k = 0; k < 20; ++k) run();
  const double ms = time_ms(run, 20);
  std::printf("%s %s %d x %d, %d rows, kernel %d, configuration %d: %.1f us, %.0f GB/s\n", t->name, s.name, s.N, s.K,
              rows, kernel, cfg, ms * 1e3, wbytes / (ms * 1e6));
  i = 0;
  run();
  std::vector<float> out(size_t(rows) * s.N);
  const size_t output_bytes = out.size() * sizeof(float);
  CK(cudaMemcpy(out.data(), y, output_bytes, cudaMemcpyDeviceToHost));
  if (!std::all_of(out.begin(), out.end(), [](float v) { return std::isfinite(v); })) {
    std::fprintf(stderr, "non-finite output in %s round benchmark\n", t->name);
    std::exit(2);
  }
  const auto* bytes_out = reinterpret_cast<const uint8_t*>(out.data());
  uint64_t hash = 14695981039346656037ull;
  for (size_t j = 0; j < output_bytes; ++j) hash = (hash ^ bytes_out[j]) * 1099511628211ull;
  std::printf("checksum %016llx\n", static_cast<unsigned long long>(hash));
  free_copies(w);
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
}

// A round's MLP gate|up GEMM and its SwiGLU in 8 bits: cuda_bench glu GATE[:UP] I K ROWS.  us for the GEMM alone, the
// GEMM and swiglu_split, and the GEMM writing the SwiGLU itself (Launch::glu), whose 8-bit rows it checks against
// swiglu_split's, bit for bit.  UP: the up projection's own type (a launch each), else one run of 2 I columns.
void bench_glu(int argc, char** argv) {
  const BlockType *tg = nullptr, *tu = nullptr;
  const std::string types = argc > 2 ? argv[2] : "";
  const size_t colon = types.find(':');
  for (const BlockType& b : kBlockTypes) {
    if (types.substr(0, colon) == b.name) tg = &b;
    if (colon != std::string::npos && types.substr(colon + 1) == b.name) tu = &b;
  }
  const int I = argc > 3 ? std::atoi(argv[3]) : 0, K = argc > 4 ? std::atoi(argv[4]) : 0;
  const int rows = argc > 5 ? std::atoi(argv[5]) : 1;
  if (!tg || (colon != std::string::npos && !tu) || I < 64 || I % 64 || I > 17408 || K < 256 || K % 256 ||
      K > 17408 || rows < 1 || rows > 16) {
    std::printf("usage: cuda_bench glu GATE[:UP] I K ROWS  (I %% 64 == 0, K %% 256 == 0, both at most 17408)\n");
    std::exit(2);
  }
  header();
  const Launch L{0, &g_bench_scratch, &g_bench_opts};
  const Scratch& sc = g_bench_scratch;
  float *x, *y;
  CK(cudaMalloc(&x, size_t(rows) * K * 4));
  k_fill_f<<<64, 256>>>(x, size_t(rows) * K, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(rows) * 2 * I * 4));
  const int n = tu ? 2 : 1;
  std::vector<std::vector<BlockMat>> w(static_cast<size_t>(n));  // copies of each run, more than L2 holds
  const BlockType* ts[2] = {tg, tu};
  for (int r = 0; r < n; ++r) w[size_t(r)] = block_copies(*ts[r], Shape{"glu", n == 1 ? 2 * I : I, K});
  size_t i = 0;
  auto runs = [&] {
    std::vector<BlockMat> W;
    for (int r = 0; r < n; ++r) {
      BlockMat b = w[size_t(r)][i % w[size_t(r)].size()];
      b.q8 = true;
      W.push_back(b);
    }
    ++i;
    return W;
  };
  const RoundGlu glu{y, 2 * I, I};
  Launch fused = L;
  fused.glu = &glu;
  {  // the engine takes it at one row (gemm_blocks_round_glu_takes); the kernel, wherever the rules give a
     // configuration compiled with it
    const std::vector<BlockMat> W = runs();
    for (const BlockMat& b : W) {
      const int tuning = g_bench_opts.tuning;
      const BlocksRoundConfig& c = kBlocksRoundConfigs[std::max(0, gemm_blocks_round_default(tuning, b.dtype, b.rows,
                                                                                             rows))];
      if (gemm_blocks_round_default_kernel(tuning, b.dtype, b.rows, rows) != 0 || c.rows > 4 || c.parts != 1 ||
          c.staged == 2 || (n == 2 && I <= 2048)) {
        std::printf("the rules do not run this GEMM in a dp4a configuration compiled with a SwiGLU\n");
        std::exit(1);
      }
    }
    if (!gemm_blocks_round_glu_takes(W.data(), n, rows, I, L)) std::printf("(the engine takes it at one row only)\n");
  }
  auto gemm = [&](const Launch& on) {
    blocks_stage(x, K, K, rows, L, true);
    const std::vector<BlockMat> W = runs();
    gemm_blocks_staged_runs(W.data(), n, K, y, 2 * I, rows, false, on);
  };
  auto staging = [&] { blocks_stage(x, K, K, rows, L, true); };
  auto alone = [&] { gemm(L); };
  auto apart = [&] {
    gemm(L);
    swiglu_split(y, 2 * I, I, rows, L, true);
  };
  auto together = [&] { gemm(fused); };
  for (int k = 0; k < 20; ++k) apart(), together();
  const double t_stage = time_ms(staging, 50), t_alone = time_ms(alone, 50) - t_stage,
               t_apart = time_ms(apart, 50) - t_stage, t_fused = time_ms(together, 50) - t_stage;
  std::printf("%s%s%s I %d K %d, %d rows: GEMM %.1f us, GEMM + swiglu %.1f us, fused %.1f us (%+.1f)\n", tg->name,
              tu ? ":" : "", tu ? tu->name : "", I, K, rows, t_alone * 1e3, t_apart * 1e3, t_fused * 1e3,
              (t_fused - t_apart) * 1e3);
  auto q8_bytes = [&](const int8_t* q, const float* d, const float* s) {
    std::vector<uint8_t> b(size_t(rows) * I + size_t(rows) * I / 32 * 4 + size_t(rows) * I / 16 * 4);
    CK(cudaMemcpy(b.data(), q, size_t(rows) * I, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data() + size_t(rows) * I, d, size_t(rows) * I / 32 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data() + size_t(rows) * I * 9 / 8, s, size_t(rows) * I / 16 * 4, cudaMemcpyDeviceToHost));
    return b;
  };
  i = 0;
  apart();
  const std::vector<uint8_t> want = q8_bytes(sc.q8, sc.q8_scale, sc.q8_sums);
  i = 0;
  together();
  const bool same = q8_bytes(sc.q8_glu, sc.q8_glu_scale, sc.q8_glu_sums) == want;
  std::printf("%s: the fused SwiGLU's 8-bit rows are swiglu_split's\n", same ? "same bits" : "DIFFERENT");
  for (auto& c : w) free_copies(c);
  for (void* q : {(void*)x, (void*)y}) cudaFree(q);
  if (!same) std::exit(1);
}

// A stack of projections of one input (rows of different block types side by side) on the rows staged once, as the
// engine runs it (regions/linear.cpp gemm_packed: gemm_blocks_staged_runs): cuda_bench stack ROWS K TYPE:N [TYPE:N
// ...].  us for each run alone; for the stack a launch a run (KernelOptions::stacks off), what the runs after the
// first add to it against what their bytes take at the first run's rate (one launch for the whole stack could cost no
// less); and the stack as the engine runs it, with a run and the next in one launch where that pair is compiled,
// whose outputs it checks against the first's, bit for bit.
void bench_stack(int argc, char** argv) {
  struct Run {
    const BlockType* t = nullptr;
    int N = 0;
    std::vector<BlockMat> w;
    size_t bytes = 0;
  };
  const int rows = argc > 2 ? std::atoi(argv[2]) : 0, K = argc > 3 ? std::atoi(argv[3]) : 0;
  std::vector<Run> runs;
  bool ok = rows >= 1 && rows <= 16 && K >= 256 && K <= 17408 && K % 256 == 0 && argc > 4;
  for (int a = 4; ok && a < argc; ++a) {
    const std::string arg = argv[a];
    const size_t colon = arg.find(':');
    Run r;
    for (const BlockType& b : kBlockTypes)
      if (colon != std::string::npos && arg.substr(0, colon) == b.name) r.t = &b;
    r.N = colon == std::string::npos ? 0 : std::atoi(arg.c_str() + colon + 1);
    ok = r.t && r.N >= 1 && r.N <= 65536;
    runs.push_back(std::move(r));
  }
  if (!ok) {
    std::printf("usage: cuda_bench stack ROWS K TYPE:N [TYPE:N ...]  (ROWS 1..16; K %% 256 == 0, at most 17408)\n");
    std::exit(2);
  }
  header();
  const int k_max = 17408, max_rows = 16;
  void* mem = nullptr;
  const size_t bytes = scratch_bytes(max_rows, k_max);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, max_rows, k_max);
  const Launch L{0, &scratch, &g_bench_opts};
  int ldy = 0;
  for (Run& r : runs) {
    const BlockLayout l = blocks_layout(r.t->dtype);
    r.bytes = size_t(r.N) * K / l.values * (l.stride + 2 * l.scales);
    r.w = block_copies(r.t->dtype, r.N, K);
    ldy += r.N;
  }
  float *x, *y;
  CK(cudaMalloc(&x, size_t(max_rows) * k_max * 4));
  k_fill_f<<<64, 256>>>(x, size_t(max_rows) * k_max, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(max_rows) * ldy * 4));
  blocks_stage(x, k_max, K, rows, L, true);
  size_t i = 0;
  auto one = [&](size_t r, int n0) {
    BlockMat b = runs[r].w[i % runs[r].w.size()];
    b.q8 = true;
    gemm_blocks_staged(b, K, y + n0, ldy, rows, false, L);
  };
  std::vector<BlockMat> mats(runs.size());
  auto engine = [&] {
    for (size_t r = 0; r < runs.size(); ++r) {
      mats[r] = runs[r].w[i % runs[r].w.size()];
      mats[r].q8 = true;
    }
    gemm_blocks_staged_runs(mats.data(), int(mats.size()), K, y, ldy, rows, false, L);
    ++i;
  };
  auto outputs = [&](bool stacks) {  // the first copies' outputs
    g_bench_opts.stacks = stacks;
    CK(cudaMemset(y, 0xFF, size_t(rows) * ldy * 4));
    i = 0;
    engine();
    std::vector<float> out(size_t(rows) * ldy);
    CK(cudaMemcpy(out.data(), y, out.size() * 4, cudaMemcpyDeviceToHost));
    return out;
  };
  const std::vector<float> apart = outputs(false), merged_out = outputs(true);
  const bool same = std::memcmp(apart.data(), merged_out.data(), apart.size() * 4) == 0;
  std::vector<double> alone;
  for (size_t r = 0; r < runs.size(); ++r) {
    auto run = [&] {
      one(r, 0);
      ++i;
    };
    for (int k = 0; k < 20; ++k) run();
    alone.push_back(time_ms(run, 40) * 1e3);
  }
  auto stack = [&] {
    int n0 = 0;
    for (size_t r = 0; r < runs.size(); ++r) {
      one(r, n0);
      n0 += runs[r].N;
    }
    ++i;
  };
  for (int k = 0; k < 20; ++k) stack();
  const double together = time_ms(stack, 40) * 1e3;
  g_bench_opts.stacks = true;
  for (int k = 0; k < 20; ++k) engine();
  const double merged = time_ms(engine, 40) * 1e3;
  size_t rest = 0;
  double sum = 0;
  for (size_t r = 0; r < runs.size(); ++r) {
    std::printf("  %s:%d alone %.2f us, %.0f GB/s\n", runs[r].t->name, runs[r].N, alone[r],
                runs[r].bytes / (alone[r] * 1e3));
    sum += alone[r];
    if (r) rest += runs[r].bytes;
  }
  const double at_first = rest / (runs[0].bytes / alone[0]);
  std::printf("stack, %d rows, K %d: %.2f us (alone, summed: %.2f); the runs after the first add %.2f us, their bytes "
              "at the first's rate %.2f; as the engine runs it %.2f us (%s bits)\n", rows, K, together, sum,
              together - alone[0], at_first, merged, same ? "the same" : "OTHER");
  for (Run& r : runs) free_copies(r.w);
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
  if (!same) std::exit(1);
}

// One prefill pass's GEMM on one layer with inputs in 8 bits, for a profiler: cuda_bench pass TYPE LAYER|NxK ROWS
// (LAYER as for `round`, ROWS 17..8192): 10 runs to settle, 10 timed (ms and int8 TOPS of 2 M N K), then one more on
// the first copy of the weights whose outputs' FNV-1a hash it prints, as `round` does.
void bench_pass(int argc, char** argv) {
  const BlockType* t = nullptr;
  for (const BlockType& b : kBlockTypes)
    if (argc > 2 && std::string(argv[2]) == b.name) t = &b;
  Shape s{"NxK", 0, 0};
  if (argc > 3 && std::sscanf(argv[3], "%dx%d", &s.N, &s.K) != 2) {
    const int layer = std::atoi(argv[3]);
    if (layer >= 0 && layer < int(std::size(kBlockShapes))) s = kBlockShapes[layer];
  }
  const int rows = argc > 4 ? std::atoi(argv[4]) : 0;
  if (!t || s.N < 1 || s.K < 256 || s.K > 17408 || s.K % 256 || rows < 17 || rows > 8192) {
    std::printf("usage: cuda_bench pass TYPE LAYER|NxK ROWS  (ROWS 17..8192; K %% 256 == 0, at most 17408)\n");
    std::exit(2);
  }
  header();
  void* mem = nullptr;
  const size_t bytes = scratch_bytes(rows, s.K);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, rows, s.K);
  const Launch L{0, &scratch, &g_bench_opts};
  float *x, *y;
  CK(cudaMalloc(&x, size_t(rows) * s.K * 4));
  k_fill_f<<<64, 256>>>(x, size_t(rows) * s.K, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(rows) * s.N * 4));
  blocks_stage(x, s.K, s.K, rows, L, true);
  std::vector<BlockMat> w = block_copies(*t, s);
  size_t i = 0;
  auto run = [&] {
    BlockMat b = w[i++ % w.size()];
    b.q8 = true;
    gemm_blocks_staged(b, s.K, y, s.N, rows, false, L);
  };
  for (int k = 0; k < 10; ++k) run();
  const double ms = time_ms(run, 10);
  std::printf("%s %s %d x %d, %d rows: %.3f ms, %.1f TOPS\n", t->name, s.name, s.N, s.K, rows, ms,
              2.0 * rows * s.N * s.K / (ms * 1e9));
  i = 0;
  run();
  std::vector<uint8_t> out(size_t(rows) * s.N * 4);
  CK(cudaMemcpy(out.data(), y, out.size(), cudaMemcpyDeviceToHost));
  uint64_t hash = 14695981039346656037ull;
  for (uint8_t b : out) hash = (hash ^ b) * 1099511628211ull;
  std::printf("checksum %016llx\n", static_cast<unsigned long long>(hash));
  free_copies(w);
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
}

// The prefill GEMM with inputs in 8 bits on the layers above (type T, else IQ2_XXS): every configuration with K in
// 1, 2, 4, 8 and 16 runs, in turn / side by side, at passes of 32, 64, 128 and 512 rows; ms a GEMM.  The numbers behind
// variants' blocks_q8 and blocks_q8_runs: runs other than the table's give other bits, and are only timed here.
// table_only: each configuration with the table's runs, run as the launch chooses (the numbers behind blocks_q8).
void bench_q8runs(const char* only, bool table_only) {
  header();
  const int k_max = 17408, max_rows = 512;
  void* mem = nullptr;
  const size_t bytes = scratch_bytes(max_rows, k_max);
  CK(cudaMalloc(&mem, bytes));
  const Scratch scratch = scratch_in(mem, bytes, max_rows, k_max);
  const Launch L{0, &scratch, &g_bench_opts};
  float *x, *y;
  CK(cudaMalloc(&x, size_t(max_rows) * k_max * 4));
  k_fill_f<<<64, 256>>>(x, size_t(max_rows) * k_max, 7u, -1.f, 2.f);
  CK(cudaMalloc(&y, size_t(max_rows) * 17408 * 4));
  std::printf("ms a GEMM; runs R: in turn/side by side (/= the scratch cannot hold the parts: in turn);\n"
              "* the defaults' configuration, [R] the table's runs\n");
  for (const BlockType& t : kBlockTypes) {
    if (only ? std::string(only) != t.name : t.dtype != FLYX_DT_GGML_IQ2_XXS) continue;
    for (const Shape& s : kBlockShapes) {
      if (s.N > 100000) continue;  // the head runs as rounds
      std::vector<BlockMat> w = block_copies(t, s);
      size_t i = 0;
      const int table = gemm_blocks_q8_runs(g_bench_opts.tuning, s.N, s.K), windows = (s.K + 255) / 256;
      for (int rows : {32, 64, 128, 512}) {
        blocks_stage(x, k_max, s.K, rows, L, true);
        for (int c = 0; c < kBlocksQ8ConfigCount; ++c) {
          std::printf("%-8s %-13s %6d x %5d r%-4d bn%-3d qk%-3d bm%-3d%s", t.name, s.name, s.N, s.K, rows,
                      kBlocksQ8Configs[c].bn, kBlocksQ8Configs[c].qk, kBlocksQ8Configs[c].bm,
                      c == gemm_blocks_q8_default(g_bench_opts.tuning, t.dtype, s.N, rows) ? "*" : " ");
          if (table_only) {
            const double ms = time_ms([&] {
              BlockMat b = w[i++ % w.size()];
              b.q8 = true;
              gemm_blocks_q8_cfg(c, b, s.K, y, s.N, rows, false, L);
            }, 10);
            std::printf(" | the table's %d runs %.3f\n", table, ms);
            continue;
          }
          for (int runs : {1, 2, 4, 8, 16}) {
            if (runs > windows) break;
            std::printf(" |%s%d%s", runs == table ? "[" : " ", runs, runs == table ? "]" : " ");
            for (int spread : {0, 1}) {
              if (runs == 1 && spread) break;
              if (spread && size_t(runs) * rows * s.N > scratch.q8_parts_cap) {
                std::printf("/=    ");
                break;
              }
              const double ms = time_ms([&] {
                BlockMat b = w[i++ % w.size()];
                b.q8 = true;
                gemm_blocks_q8_cfg(c, b, s.K, y, s.N, rows, false, L, spread, runs);
              }, 10);
              std::printf("%s%.3f", spread ? "/" : " ", ms);
            }
          }
          std::printf("\n");
        }
      }
      free_copies(w);
    }
  }
  for (void* q : {(void*)x, (void*)y, mem}) cudaFree(q);
}

// The drafts' two-stage head on Qwen3.5's head shape: the 1-bit rough GEMM
// and the whole draft_pick, against the NVFP4 head GEMM they replace; us.
// The drafts' exact candidate scores on a packed head (dots_blocks, draft_pick_blocks' second stage): the 27B's
// Q4_K head (248320 x 5120), 512 candidates a row at random, rows 1 and 7 (an MTP draft; a DFlash2 block's); us.
void bench_dots() {
  header();
  const int N = 248320, K = 5120, cap = kDraftCandCap;
  std::vector<BlockMat> w = block_copies(FLYX_DT_GGML_Q4_K, N, K);
  const size_t ld = 3 * cap;
  std::vector<int> buf(16 * ld, 0);
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < cap; ++c) buf[r * ld + c] = int((uint32_t(r * 7919 + c) * 2654435761u) % uint32_t(N));
    buf[r * ld + 2 * cap] = cap;
  }
  int* dbuf;
  CK(cudaMalloc(&dbuf, buf.size() * 4));
  CK(cudaMemcpy(dbuf, buf.data(), buf.size() * 4, cudaMemcpyHostToDevice));
  float* x;
  CK(cudaMalloc(&x, size_t(16) * K * 4));
  k_fill_f<<<64, 256>>>(x, size_t(16) * K, 7u, -1.f, 2.f);
  std::printf("dots_blocks, the 27B's Q4_K head, %d candidates a row, us:", cap);
  for (int rows : {1, 7}) {
    const double ms = time_ms([&] { dots_blocks(w[0], K, dbuf, dbuf + 2 * cap, reinterpret_cast<float*>(dbuf + cap),
                                                ld, cap, x, K, rows, 0); }, 20);
    std::printf(" %d rows %.1f%s", rows, ms * 1e3, rows == 7 ? "\n" : ",");
  }
  free_copies(w);
  for (void* q : {(void*)dbuf, (void*)x}) cudaFree(q);
}

void bench_pick() {
  header();
  const int N = 248320, K = 2560;
  bf16* w;
  CK(cudaMalloc(&w, size_t(N) * K * 2));
  k_fill_bf16<<<1024, 256>>>(w, size_t(N) * K, 101u, -0.05f, 0.1f);
  SignHead h;
  CK(cudaMalloc(&h.bits, sign_head_bytes(N, K)));
  CK(cudaMalloc(&h.scales, sign_head_bytes(N, K)));
  make_sign_head(w, N, K, &h, 0);
  uint8_t *q, *s;
  float* rs;
  CK(cudaMalloc(&q, size_t(N) * K / 2));
  CK(cudaMalloc(&s, size_t(N) * K / 16));
  CK(cudaMalloc(&rs, size_t(N) * 4));
  quantize_nvfp4(w, N, K, q, s, rs, 0);
  float *x, *rough;
  CK(cudaMalloc(&x, size_t(16) * K * 4));
  k_fill_f<<<64, 256>>>(x, size_t(16) * K, 7u, -1.f, 2.f);
  CK(cudaMalloc(&rough, size_t(16) * N * 4));
  int* idx;
  float* val;
  CK(cudaMalloc(&idx, 16 * 16 * 4));
  CK(cudaMalloc(&val, 16 * 16 * 4));
  std::printf("draft head %d x %d, us: NVFP4 head GEMM | 1-bit rough GEMM | draft_pick (rough + pick)\n", N, K);
  for (int rows : {1, 4, 7, 15}) {
    const double a = time_ms([&] { gemm_fp4(x, K, q, s, rs, N, K, rough, N, rows, false, BL(0)); }, 20);
    const double b = time_ms([&] { gemm_sign_rough(x, K, h, N, K, rough, N, rows, BL(0)); }, 20);
    const double c = time_ms([&] { draft_pick(x, K, h, w, N, K, rows, rows == 1 ? 1 : 16, idx, val, rough, BL(0)); }, 20);
    std::printf("  rows %2d: %7.1f | %7.1f | %7.1f\n", rows, a * 1e3, b * 1e3, c * 1e3);
  }
  {  // the 27B's hidden size (its head packed: the rough GEMM alone, on a 1-bit copy of random signs)
    const int K2 = 5120;
    SignHead h2;
    CK(cudaMalloc(&h2.bits, sign_head_bytes(N, K2)));
    CK(cudaMalloc(&h2.scales, sign_head_bytes(N, K2)));
    k_fill_f<<<1024, 256>>>(reinterpret_cast<float*>(h2.bits), sign_head_bytes(N, K2) / 4, 11u, -1.f, 2.f);
    k_fill_bf16<<<1024, 256>>>(h2.scales, sign_head_bytes(N, K2) / 2, 13u, 0.f, 0.01f);
    float* x2;
    CK(cudaMalloc(&x2, size_t(16) * K2 * 4));
    k_fill_f<<<64, 256>>>(x2, size_t(16) * K2, 7u, -1.f, 2.f);
    std::printf("1-bit rough GEMM %d x %d, us:", N, K2);
    for (int rows : {1, 3, 7, 15})
      std::printf(" %d rows %.1f%s", rows,
                  1e3 * time_ms([&] { gemm_sign_rough(x2, K2, h2, N, K2, rough, N, rows, BL(0)); }, 20),
                  rows == 15 ? "\n" : ",");
    for (void* p : {(void*)h2.bits, (void*)h2.scales, (void*)x2}) cudaFree(p);
  }
  for (void* p : {(void*)w, (void*)h.bits, (void*)h.scales, (void*)q, (void*)s, (void*)rs, (void*)x, (void*)rough,
                  (void*)idx, (void*)val})
    cudaFree(p);
}

// ---- peak rates of the tensor cores' instructions ------------------------------------------------
// One instruction at a time, issued as the GEMMs issue it (mma.sync, operands in registers, nothing from memory):
// 16 warps a block, a block an SM, kMmaChains independent sums a warp so that no instruction waits for the one
// before.  An instruction is 2 m n k operations.  Each block reads its SM's cycle counter, so operations per clock
// and SM do not depend on the clock the GPU runs at; the times come from the GPU's events.
enum MmaKind { kMmaF16F32, kMmaBf16F32, kMmaF16F16, kMmaS8S32, kMmaE4m3F32 };
constexpr int kMmaChains = 8, kMmaWarps = 16;

// Operand bits: random signs and mantissas, magnitudes small enough that no sum overflows.
template <int KIND>
__device__ uint32_t mma_operand(uint32_t x) {
  x ^= x >> 16, x *= 0x7feb352du, x ^= x >> 15, x *= 0x846ca68bu, x ^= x >> 16;
  if (KIND == kMmaF16F32 || KIND == kMmaF16F16) {  // two binary16 values of 2^-8 .. 2^-5
    const auto half = [](uint32_t h) { return (h & 0x83FFu) | ((7u + ((h >> 10) & 3u)) << 10); };
    return half(x & 0xFFFFu) | half(x >> 16) << 16;
  }
  if (KIND == kMmaBf16F32) {
    const auto bf = [](uint32_t h) { return (h & 0x807Fu) | ((119u + ((h >> 7) & 3u)) << 7); };
    return bf(x & 0xFFFFu) | bf(x >> 16) << 16;
  }
  if (KIND == kMmaE4m3F32) {  // four e4m3 values of 2^-4 .. 2^-1
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const uint32_t b = (x >> (8 * i)) & 0xFFu;
      v |= ((b & 0x87u) | ((3u + ((b >> 3) & 3u)) << 3)) << (8 * i);
    }
    return v;
  }
  return x;  // four int8 values
}

template <int KIND>
__global__ void __launch_bounds__(32 * kMmaWarps) k_mma_peak(int iters, uint32_t seed, float* sink, long long* cycles,
                                                             int* sm) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4], b[2];
#pragma unroll
  for (int i = 0; i < 4; ++i) a[i] = mma_operand<KIND>(seed + 6 * id + i);
#pragma unroll
  for (int i = 0; i < 2; ++i) b[i] = mma_operand<KIND>(seed + 6 * id + 4 + i);
  float cf[kMmaChains][4] = {};
  uint32_t cu[kMmaChains][4] = {};
  __syncthreads();
  const long long t0 = clock64();
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < kMmaChains; ++c) {
      if constexpr (KIND == kMmaF16F32)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(cf[c][0]), "+f"(cf[c][1]), "+f"(cf[c][2]), "+f"(cf[c][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else if constexpr (KIND == kMmaBf16F32)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(cf[c][0]), "+f"(cf[c][1]), "+f"(cf[c][2]), "+f"(cf[c][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else if constexpr (KIND == kMmaF16F16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                     : "+r"(cu[c][0]), "+r"(cu[c][1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else if constexpr (KIND == kMmaS8S32)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+r"(cu[c][0]), "+r"(cu[c][1]), "+r"(cu[c][2]), "+r"(cu[c][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
        __trap();  // e4m3 operands are sm_89's: bench_mma skips the row below it
#else
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};"
                     : "+f"(cf[c][0]), "+f"(cf[c][1]), "+f"(cf[c][2]), "+f"(cf[c][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      }
    }
  }
  __syncthreads();
  const long long t1 = clock64();
  if (threadIdx.x == 0) {
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    cycles[blockIdx.x] = t1 - t0;
    sm[blockIdx.x] = int(s);
  }
  float keep = 0.f;
#pragma unroll
  for (int c = 0; c < kMmaChains; ++c)
#pragma unroll
    for (int e = 0; e < 4; ++e) keep += cf[c][e] + __uint_as_float(cu[c][e]);
  if (keep == 1234.5f) sink[id] = keep;  // the sums stay live (practically never stored)
}

template <int KIND>
void launch_mma_peak(int blocks, int iters, float* sink, long long* cycles, int* sm) {
  k_mma_peak<KIND><<<blocks, 32 * kMmaWarps>>>(iters, 12345u, sink, cycles, sm);
}

void bench_mma() {
  header();
  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount, iters = 1 << 16, runs = 5;
  float* sink;
  long long* cycles;
  int* sm;
  CK(cudaMalloc(&sink, size_t(sms) * 32 * kMmaWarps * sizeof(float)));
  CK(cudaMalloc(&cycles, sms * sizeof(long long)));
  CK(cudaMalloc(&sm, sms * sizeof(int)));
  const struct {
    const char* name;
    int k;
    void (*run)(int, int, float*, long long*, int*);
    int min_cc;  // compute capability the instruction needs, as 10 * major + minor
  } kinds[] = {
      {"f16 x f16 -> f32    m16n8k16  (ours)", 16, launch_mma_peak<kMmaF16F32>, 80},
      {"bf16 x bf16 -> f32  m16n8k16", 16, launch_mma_peak<kMmaBf16F32>, 80},
      {"f16 x f16 -> f16    m16n8k16", 16, launch_mma_peak<kMmaF16F16>, 80},
      {"s8 x s8 -> s32      m16n8k32  (llama.cpp's batches)", 32, launch_mma_peak<kMmaS8S32>, 80},
      {"e4m3 x e4m3 -> f32  m16n8k32", 32, launch_mma_peak<kMmaE4m3F32>, 89},
  };
  std::printf("tensor-core instructions, mma.sync, operands in registers: %d blocks x %d warps x %d chains, "
              "best of %d runs\n", sms, kMmaWarps, kMmaChains, runs);
  std::printf("%-52s %9s %13s %9s %9s\n", "instruction", "T(FL)OPS", "ops/clk/SM", "vs ours", "SM clock");
  double ours = 0;
  std::vector<long long> host_cycles(sms);
  std::vector<int> host_sm(sms);
  for (const auto& kind : kinds) {
    if (10 * p.major + p.minor < kind.min_cc) {
      std::printf("%-52s (needs sm_%d)\n", kind.name, kind.min_cc);
      continue;
    }
    const double ops_per_block = 2.0 * 16 * 8 * kind.k * kMmaChains * kMmaWarps * double(iters);
    double best_ms = 1e30, best_cycles = 0;
    for (int r = 0; r < runs + 1; ++r) {  // the first run warms up
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a));
      CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      kind.run(sms, iters, sink, cycles, sm);
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      CK(cudaGetLastError());
      float ms = 0;
      CK(cudaEventElapsedTime(&ms, a, b));
      cudaEventDestroy(a);
      cudaEventDestroy(b);
      CK(cudaMemcpy(host_cycles.data(), cycles, sms * sizeof(long long), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(host_sm.data(), sm, sms * sizeof(int), cudaMemcpyDeviceToHost));
      std::vector<int> seen = host_sm;
      std::sort(seen.begin(), seen.end());
      if (std::unique(seen.begin(), seen.end()) != seen.end()) std::printf("  (two blocks shared an SM in this run)\n");
      std::vector<long long> c = host_cycles;
      std::sort(c.begin(), c.end());
      if (r > 0 && ms < best_ms) best_ms = ms, best_cycles = double(c[c.size() / 2]);
    }
    const double rate = ops_per_block * sms / (best_ms * 1e9), per_clock = ops_per_block / best_cycles;
    if (ours == 0) ours = per_clock;
    std::printf("%-52s %9.1f %13.0f %8.2fx %6.2f GHz\n", kind.name, rate, per_clock, per_clock / ours,
                best_cycles / (best_ms * 1e6));
  }
  // The host's clock against the GPU's over the same stretch (WSL's clock was seen running fast against Windows').
  {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    CK(cudaDeviceSynchronize());
    const auto h0 = std::chrono::steady_clock::now();
    CK(cudaEventRecord(a));
    for (int i = 0; i < 20; ++i) kinds[0].run(sms, iters, sink, cycles, sm);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    const auto h1 = std::chrono::steady_clock::now();
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    const double host_ms = std::chrono::duration<double, std::milli>(h1 - h0).count();
    std::printf("host clock over GPU clock, %.0f ms of GPU time: %.4f\n", ms, host_ms / ms);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  cudaFree(sink);
  cudaFree(cycles);
  cudaFree(sm);
}

// ---- reading host memory in place ----------------------------------------------
// A kernel sums a buffer of pinned, mapped host memory where it lies: what the GPU reads over the bus, and
// whether a buffer's device address is its host address (with UVA it is; write-combined memory under WSL is not).
__global__ void k_sum_words(const uint32_t* p, size_t n, unsigned long long* out) {
  unsigned long long s = 0;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) s += p[i];
  atomicAdd(out, s);
}

void bench_host_read() {
  unsigned long long* out = nullptr;
  CK(cudaMalloc(&out, 8));
  for (const unsigned flags : {unsigned(cudaHostAllocMapped), unsigned(cudaHostAllocMapped | cudaHostAllocWriteCombined)})
    for (const size_t mb : {size_t(64), size_t(512)}) {
      const size_t bytes = mb << 20, n = bytes / 4;
      void* h = nullptr;
      CK(cudaHostAlloc(&h, bytes, flags));
      auto* w = static_cast<uint32_t*>(h);
      unsigned long long want = 0;
      for (size_t i = 0; i < n; ++i) {  // written once, never read back (slow on write-combined memory)
        const uint32_t v = uint32_t(i & 0xFF);
        w[i] = v;
        want += v;
      }
      void* d = nullptr;
      CK(cudaHostGetDevicePointer(&d, h, 0));
      const auto* p = static_cast<const uint32_t*>(d);
      unsigned long long got = 0;
      CK(cudaMemset(out, 0, 8));
      k_sum_words<<<280, 256>>>(p, n, out);
      CK(cudaMemcpy(&got, out, 8, cudaMemcpyDeviceToHost));
      const double ms = time_ms([&] { k_sum_words<<<280, 256>>>(p, n, out); }, 4);
      std::printf("%-22s %4zu MB: %5.1f GB/s, sum %s, device address %s the host's\n",
                  flags & cudaHostAllocWriteCombined ? "mapped, write-combined" : "mapped", mb, bytes / (ms * 1e6),
                  got == want ? "right" : "WRONG", d == h ? "is" : "is not");
      CK(cudaFreeHost(h));
    }
  CK(cudaFree(out));
}

// ---- memory: bandwidth against the bytes an SM has on their way ----------------------------------
// Little's law: bandwidth = bytes in flight / latency.  How many bytes an SM must have on their way for the GPU to
// read at its full bandwidth, kept two ways: in registers (every thread U loads of 16 bytes, then their use, as the
// round kernels hold their weights) and in shared memory (every warp a ring of S slots of 512 bytes filled by
// cp.async, S - 1 on their way, nothing held in registers); with all the SMs reading or only 60% of them (a kernel's
// last wave).  One block an SM (its shared memory keeps others off), 512 MB read once a launch (L2 holds 96 MB).
template <int U>
__global__ void k_stream_regs(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  extern __shared__ uint4 keep_off[];  // (only its size counts)
  uint32_t acc = 0;
  const size_t stride = size_t(gridDim.x) * blockDim.x;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i + (U - 1) * stride < n16; i += U * stride) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = __ldcs(p + i + u * stride);
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x9e3779b9u) *out = acc + keep_off[0].x;  // (practically never: the loads stay)
}

template <int S>
__global__ void k_stream_ring(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  extern __shared__ uint4 ring[];  // [warps][S][32]
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  uint4* mine = ring + size_t(warp) * S * 32;
  const size_t w = blockIdx.x * size_t(warps) + warp, nw = size_t(gridDim.x) * warps, chunks = n16 / 32;
  // the warp's chunks of 512 bytes: w, w + nw, w + 2 nw, ...
#pragma unroll
  for (int s = 0; s < S - 1; ++s) {
    const size_t c = w + s * nw;
    if (c < chunks) cp_async16(&mine[s * 32 + lane], p + c * 32 + lane);
    cp_async_commit();
  }
  uint32_t acc = 0;
  for (size_t k = 0; w + k * nw < chunks; ++k) {
    const size_t c = w + (k + S - 1) * nw;
    if (c < chunks) cp_async16(&mine[((k + S - 1) % S) * 32 + lane], p + c * 32 + lane);
    cp_async_commit();
    cp_async_wait<S - 1>();  // chunk k is in (each lane reads only what it copied)
    const uint4 v = mine[(k % S) * 32 + lane];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *out = acc;
}

void bench_stream() {
  header();
  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, 0));
  const int sms = p.multiProcessorCount;
  const size_t bytes = size_t(512) << 20, n16 = bytes / 16;
  uint4* buf;
  uint32_t* out;
  CK(cudaMalloc(&buf, bytes));
  CK(cudaMalloc(&out, 4));
  k_fill<<<1024, 256>>>(reinterpret_cast<uint8_t*>(buf), bytes, 5u, 0, 0);
  const size_t keep_off = 64 << 10;  // shared memory a block: one block an SM
  auto run = [&](auto kernel, int warps, int blocks, size_t smem) {
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    const double ms = time_ms([&] { kernel<<<blocks, 32 * warps, smem>>>(buf, n16, out); }, 10);
    CK(cudaGetLastError());
    return bytes / (ms * 1e6);
  };
  std::printf("GB/s reading 512 MB; KB: bytes on their way an SM; all %d SMs / %d of them\n", sms, sms * 6 / 10);
  for (int warps : {4, 8, 16, 32}) {
    std::printf("in registers, %2d warps an SM:", warps);
    auto regs = [&](auto kernel, int u) {
      const double a = run(kernel, warps, sms, keep_off), b = run(kernel, warps, sms * 6 / 10, keep_off);
      std::printf(" | U%d %4.0f KB %5.0f / %5.0f", u, warps * 32 * 16 * u / 1024.0, a, b);
    };
    regs(k_stream_regs<1>, 1);
    regs(k_stream_regs<2>, 2);
    regs(k_stream_regs<4>, 4);
    regs(k_stream_regs<8>, 8);
    std::printf("\n");
  }
  for (int warps : {4, 8, 16, 32}) {
    std::printf("cp.async ring, %2d warps an SM:", warps);
    auto ring = [&](auto kernel, int s) {
      const size_t smem = std::max(keep_off, size_t(warps) * s * 512);
      if (smem > size_t(p.sharedMemPerBlockOptin)) return;
      const double a = run(kernel, warps, sms, smem), b = run(kernel, warps, sms * 6 / 10, smem);
      std::printf(" | S%-2d %4.0f KB %5.0f / %5.0f", s, warps * (s - 1) * 512 / 1024.0, a, b);
    };
    ring(k_stream_ring<2>, 2);
    ring(k_stream_ring<4>, 4);
    ring(k_stream_ring<8>, 8);
    ring(k_stream_ring<16>, 16);
    std::printf("\n");
  }
  cudaFree(buf);
  cudaFree(out);
}

// MiniCPM5-1B's bf16 GEMMs and DV's rank-256 adapters on them (dv.md): what a tree round's verify pass
// and a canvas pass of 32 slots run over 17..64 rows.
const Shape kRowsShapes[] = {{"w_in q|k|v", 2560, 1536}, {"w_out", 1536, 2048},    {"w_gu", 9216, 1536},
                             {"w_down", 1536, 4608},     {"head", 130560, 1536},    {"a_qkv", 768, 1536},
                             {"a_o", 256, 2048},         {"a_gu", 512, 1536},       {"a_down", 256, 4608},
                             {"b_q", 2048, 256},         {"b_k, b_v", 256, 256},    {"b_o, b_down", 1536, 256},
                             {"b_gate, b_up", 4608, 256}};

void bench_mt(int argc, char** argv) {
  header();
  std::vector<int> rows_list;
  std::vector<Shape> shapes;
  for (int i = 2; i < argc; ++i) {
    int n = 0, k = 0;
    if (std::sscanf(argv[i], "%dx%d", &n, &k) == 2) shapes.push_back({argv[i], n, k});
    else if (const int r = std::atoi(argv[i]); r >= 1 && r <= FLYX_MAX_WIDE_ROWS) rows_list.push_back(r);
  }
  if (rows_list.empty()) rows_list = {17, 32, 48, 64};
  if (shapes.empty()) shapes.assign(std::begin(kRowsShapes), std::end(kRowsShapes));
  std::printf("us (GB/s of weights) per GEMM: prefill = gemm_tc, tiles = the 1..16-row kernel 16 rows at a time; "
              "mt c = bf16_mt configuration c (nt, chunks, unroll, parts)\n");
  for (int c = 0; c < kMtConfigCount; ++c)
    std::printf("  mt %2d: nt %d, chunks %2d, unroll %d, parts %d\n", c, kMtConfigs[c].nt, kMtConfigs[c].chunks,
                kMtConfigs[c].unroll, kMtConfigs[c].parts);
  for (const auto& s : shapes) {
    Weights W(s, true);
    float *x, *y;
    CK(cudaMalloc(&x, size_t(64) * s.K * 4));
    k_fill_f<<<64, 256>>>(x, size_t(64) * s.K, 7u, -1.f, 2.f);
    CK(cudaMalloc(&y, size_t(64) * s.N * 4));
    const double gb = double(s.N) * s.K * 2 / 1e9;
    for (int rows : rows_list) {
      auto cell = [&](double ms) { std::printf(" %7.1f (%3.0f)", ms * 1e3, gb / (ms / 1e3)); };
      std::printf("%-13s %6d x %5d r%-3d prefill", s.name, s.N, s.K, rows);
      cell(time_ms([&] { gemm_tc(x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, BL(0)); }, 20));
      std::printf(" tiles");
      cell(time_ms([&] {
        bf16* w = W.next();
        for (int r0 = 0; r0 < rows; r0 += 16)
          gemm_skinny(x + size_t(r0) * s.K, s.K, w, s.N, s.K, y + size_t(r0) * s.N, s.N, std::min(16, rows - r0),
                      false, BL(0));
      }, 20));
      std::printf("\n%-38s", "");
      for (int c = 0; c < kMtConfigCount; ++c) {
        std::printf(" mt%d", c);
        cell(time_ms([&] { gemm_skinny_mt_cfg(c, x, s.K, W.next(), s.N, s.K, y, s.N, rows, false, BL(0)); }, 20));
        if (c % 6 == 5 && c + 1 < kMtConfigCount) std::printf("\n%-38s", "");
      }
      std::printf("\n");
      std::fflush(stdout);
    }
    CK(cudaFree(x));
    CK(cudaFree(y));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string what = argc > 1 ? argv[1] : "gemm";
  // This tuning family uses the recorded decode timings from real model runs.
  if (what == "tune" && argc > 2 && std::string(argv[2]) == "strata_placement") return tune(argc - 2, argv + 2);
#if !FLYX_TUNE_CANDIDATES
  if (what == "skinny" || what == "fp4" || what == "fp4cfgs" || what == "fp4rows" || what == "tune" ||
      what == "w12" || (what == "blocks" && argc > 3) || what == "q8runs" || what == "rounds" || what == "glu") {
    std::fprintf(stderr, "cuda_bench %s requires all tuning candidates; reconfigure with -DFLYX_TUNE_CANDIDATES=ON\n",
                 what.c_str());
    return 2;
  }
#endif
  {
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, 0));
    // The table the engine takes for this device (backend.cpp, option tuning=table).
    g_bench_opts.tuning = gemm_tuning_auto(prop.name, prop.major * 10 + prop.minor, compiled_arch());
    // Kernels launch as the engine launches them on this GPU; FLYX_BENCH_PDL=0|1 says otherwise.
    g_bench_opts.pdl = gemm_tuning_pdl(g_bench_opts.tuning);
    if (const char* e = std::getenv("FLYX_BENCH_PDL")) g_bench_opts.pdl = std::string(e) == "1";
  }
  if (what == "moe") {  // uses its own routing and staging scratch
    bench_moe();
    return 0;
  }
  if (what == "paging") return bench_paging(argc, argv);  // the device's memory as it is, no scratch beside it
  {
    void* mem = nullptr;
    // K up to the 27B's MLP down projection (17408); split GEMMs of up to kBenchSplitColumns columns.
    const size_t bytes = scratch_bytes(2048, 17408, true, false, true, true, kBenchSplitColumns);
    CK(cudaMalloc(&mem, bytes));
    g_bench_scratch = scratch_in(mem, bytes, 2048, 17408, true, false, kBenchSplitColumns);
  }
  if (what == "gemm") bench_gemm();
  else if (what == "skinny") bench_skinny();
  else if (what == "fp4") bench_fp4();
  else if (what == "tune") return tune(argc - 2, argv + 2);
  else if (what == "gdn") bench_gdn(argc > 2 && std::string(argv[2]) == "27b");
  else if (what == "decode") bench_decode();
  else if (what == "fp4cfgs") bench_fp4cfgs(argc, argv);
  else if (what == "fp4rows") bench_fp4rows(argc, argv);
  else if (what == "tc") bench_tc(argc, argv);
  else if (what == "attn") bench_attn(argc > 2 ? std::atoi(argv[2]) : 16, argc > 3 ? std::atoi(argv[3]) : 32,
                                       argc > 4 ? std::atoi(argv[4]) : 0);
  else if (what == "mlp") bench_mlp();
  else if (what == "w4a4") bench_w4a4();
  else if (what == "flash") bench_flash(argc, argv);
  else if (what == "strata") bench_strata(argc, argv);
  else if (what == "attnhost") bench_attn_host(argc, argv);
  else if (what == "w12") bench_w12();
  else if (what == "pick") bench_pick();
  else if (what == "dots") bench_dots();
  else if (what == "blocks") bench_blocks(argc > 2 ? argv[2] : nullptr, argc > 3);
  else if (what == "q8runs") bench_q8runs(argc > 2 ? argv[2] : nullptr, argc > 3 && std::string(argv[3]) == "table");
  else if (what == "rounds") bench_rounds(argc > 2 ? argv[2] : nullptr);
  else if (what == "round") bench_round(argc, argv);
  else if (what == "glu") bench_glu(argc, argv);
  else if (what == "pass") bench_pass(argc, argv);
  else if (what == "stack") bench_stack(argc, argv);
  else if (what == "mma") bench_mma();
  else if (what == "host_read") bench_host_read();
  else if (what == "stream") bench_stream();
  else if (what == "graphs") bench_graphs(argc, argv);
  else if (what == "mt") bench_mt(argc, argv);
  else {
    std::printf("usage: cuda_bench gemm|skinny|fp4|fp4cfgs|fp4rows|tune|gdn|decode|tc|attn|flash|strata|mlp|w4a4|w12|pick|dots|blocks|q8runs|rounds|"
                "round|moe|glu|pass|stack|mma|host_read|stream|graphs|paging|mt\n");
    return 2;
  }
  return 0;
}
