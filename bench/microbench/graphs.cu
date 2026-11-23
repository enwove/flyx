// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// cuda_bench graphs [K]: what an instantiated CUDA graph takes of the device's memory, by the nodes it holds (the
// numbers behind backends/cuda/backend.h's kGraph*Bytes; a node count takes in a conditional node's body, "if body":
// as DV's canvas pass, regions/layers.cpp).  Each graph is captured on a stream and instantiated K times (default 64;
// one with a conditional node is captured anew each time, as its handle is its own) between two cudaMemGetInfo calls,
// three times; the median over K is printed.  Every instantiation is
// kept to the end: the driver keeps a destroyed graph's memory for the next, which would then take none (the last
// line: what destroying them all gives back, and what as many graphs take after).  The driver takes the memory in
// pieces of 2 MiB: a figure is good to 2 MiB / K.  Other programs' allocations move it too (on Windows cudaMemGetInfo
// is the video memory manager's budget): a run beside a busy desktop or another test is noisy.
#include <cuda.h>

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "bench/microbench/bench.cuh"

namespace flyx::bench {

namespace {

template <int Bytes>
struct Params {
  float* p;
  char pad[Bytes > 16 ? Bytes - 8 : 8];
};

template <int Bytes>
__global__ void k_params(Params<Bytes> a) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && a.p) a.p[0] += float(a.pad[1]);
}

size_t free_bytes() {
  size_t f = 0, t = 0;
  CK(cudaDeviceSynchronize());
  CK(cudaMemGetInfo(&f, &t));
  return f;
}

// A graph's nodes, those in its conditional nodes' bodies too (cuGraphNodeGetParams, from CUDA 13.2, through the
// runtime gives a body's graph); -1 where the driver does not give a conditional node's body.
long long all_nodes(cudaGraph_t g) {
  static_assert(std::is_same_v<CUgraph, cudaGraph_t> && std::is_same_v<CUgraphNode, cudaGraphNode_t>);
  using GetParams = CUresult(CUDAAPI*)(CUgraphNode, CUgraphNodeParams*);
  static const GetParams get_params = [] {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q{};
    if (cudaGetDriverEntryPointByVersion("cuGraphNodeGetParams", &p, 13020, cudaEnableDefault, &q) != cudaSuccess ||
        q != cudaDriverEntryPointSuccess) {
      cudaGetLastError();
      return GetParams(nullptr);
    }
    return reinterpret_cast<GetParams>(p);
  }();
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  long long all = 0;
  for (cudaGraphNode_t node : nodes) {
    ++all;
    cudaGraphNodeType type{};
    CK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeConditional) continue;
    CUgraphNodeParams p{};
    if (!get_params || get_params(node, &p) != CUDA_SUCCESS || p.type != CU_GRAPH_NODE_TYPE_CONDITIONAL ||
        !p.conditional.phGraph_out) {
      return -1;
    }
    for (unsigned b = 0; b < p.conditional.size; ++b) {
      const long long in = all_nodes(p.conditional.phGraph_out[b]);
      if (in < 0) return -1;
      all += in;
    }
  }
  return all;
}

// A kernel launched as the backend launches most of a round's: programmatic dependent launch (launch.h).
void launch_pdl(int grid, cudaStream_t st, float* dev) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(32);
  cfg.stream = st;
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &at;
  cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, k_params<16>, Params<16>{dev, {}}));
}

// n launches of `kind` captured in a chain on one stream (`fan`: every 8, four on a second stream forked and joined).
// "tiers": a round reading K/V in host memory as runtime/tiers.cpp stages it: n kernels launched with programmatic
// dependent launch, and before every 8 of them four copies from pinned host memory on a stream of the greatest
// priority, forked from the round's stream and joined back before the kernels that read them; the result ring's copy
// to the host at the end (runner.cpp).
cudaGraph_t capture(const std::string& kind, int n, int grid, bool fan, float* dev, float* host, float* mapped) {
  cudaStream_t st, side;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  int least = 0, greatest = 0;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  CK(cudaStreamCreateWithPriority(&side, cudaStreamNonBlocking, kind == "tiers" ? greatest : least));
  cudaEvent_t fork, join;
  CK(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  if (kind == "if body") {
    // A kernel (`fan`: n of them), then a conditional node (an if) whose body holds the n kernels, captured into it on
    // a second stream, as regions/layers.cpp makes DV's canvas pass the body of one (dv_context_unless_noop).
    for (int i = 0; i < (fan ? n : 1); ++i) k_params<16><<<grid, 32, 0, st>>>({dev, {}});
    cudaStreamCaptureStatus capturing{};
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    size_t n_deps = 0;
    CK(cudaStreamGetCaptureInfo(st, &capturing, nullptr, &graph, &deps, nullptr, &n_deps));
    cudaGraphConditionalHandle live{};
    CK(cudaGraphConditionalHandleCreate(&live, graph, 0, 0));
    cudaGraphNodeParams cond{};
    cond.type = cudaGraphNodeTypeConditional;
    cond.conditional.handle = live;
    cond.conditional.type = cudaGraphCondTypeIf;
    cond.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
    CK(cudaGraphAddNode(&node, graph, deps, nullptr, n_deps, &cond));
    CK(cudaStreamUpdateCaptureDependencies(st, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
    CK(cudaStreamBeginCaptureToGraph(side, cond.conditional.phGraph_out[0], nullptr, nullptr, 0,
                                     cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; ++i) k_params<16><<<grid, 32, 0, side>>>({dev, {}});  // a body takes no PDL launch
    cudaGraph_t body = nullptr;
    CK(cudaStreamEndCapture(side, &body));
  }
  for (int i = 0; i < n && kind == "tiers"; ++i) {
    if (i % 8 == 0) {
      CK(cudaEventRecord(fork, st));
      CK(cudaStreamWaitEvent(side, fork, 0));
      for (int c = 0; c < 4; ++c)
        CK(cudaMemcpyAsync(dev + 64 + 64 * c, host + 64 * c, 256, cudaMemcpyHostToDevice, side));
      CK(cudaEventRecord(join, side));
      CK(cudaStreamWaitEvent(st, join, 0));
    }
    launch_pdl(grid, st, dev);
    if (i == n - 1) CK(cudaMemcpyAsync(host + 512, dev, 256, cudaMemcpyDeviceToHost, st));
  }
  for (int i = 0; i < n && kind != "tiers" && kind != "if body"; ++i) {
    cudaStream_t on = st;
    if (fan && i % 8 == 4) {
      CK(cudaEventRecord(fork, st));
      CK(cudaStreamWaitEvent(side, fork, 0));
    }
    if (fan && i % 8 >= 4) on = side;
    if (kind == "kernel 16 B") k_params<16><<<grid, 32, 0, on>>>({dev, {}});
    else if (kind == "kernel PDL") launch_pdl(grid, on, dev);
    else if (kind == "kernel 272 B") k_params<272><<<grid, 32, 0, on>>>({dev, {}});
    else if (kind == "kernel 1 KB") k_params<1024><<<grid, 32, 0, on>>>({dev, {}});
    else if (kind == "kernel 4 KB") k_params<4096><<<grid, 32, 0, on>>>({dev, {}});
    else if (kind == "copy D2D") CK(cudaMemcpyAsync(dev + 64, dev, 256, cudaMemcpyDeviceToDevice, on));
    else if (kind == "copy H2D") CK(cudaMemcpyAsync(dev + 64, host, 256, cudaMemcpyHostToDevice, on));
    else if (kind == "copy D2H") CK(cudaMemcpyAsync(host + 512, dev, 256, cudaMemcpyDeviceToHost, on));
    else if (kind == "copy mapped") CK(cudaMemcpyAsync(dev + 64, mapped, 256, cudaMemcpyDefault, on));
    else if (kind == "memset") CK(cudaMemsetAsync(dev + 64, 0, 256, on));
    if (fan && i % 8 == 7) {
      CK(cudaEventRecord(join, side));
      CK(cudaStreamWaitEvent(st, join, 0));
    }
  }
  cudaGraph_t g = nullptr;
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaEventDestroy(fork));
  CK(cudaEventDestroy(join));
  CK(cudaStreamDestroy(side));
  CK(cudaStreamDestroy(st));
  return g;
}

}  // namespace

void bench_graphs(int argc, char** argv) {
  const int K = argc > 2 ? std::max(8, std::atoi(argv[2])) : 64;
  float *dev = nullptr, *host = nullptr, *mapped_host = nullptr, *mapped = nullptr;
  CK(cudaMalloc(&dev, 4096));
  CK(cudaMallocHost(&host, 4096));
  CK(cudaHostAlloc(&mapped_host, 4096, cudaHostAllocMapped));  // host memory the device reads in place (state.cpp)
  CK(cudaHostGetDevicePointer(&mapped, mapped_host, 0));
  struct Case {
    const char* kind;
    int n, grid;
    bool fan;
  };
  const Case cases[] = {
      {"kernel 16 B", 1, 1, false},    {"kernel 16 B", 100, 1, false},  {"kernel 16 B", 400, 1, false},
      {"kernel 16 B", 1000, 1, false}, {"kernel 16 B", 100, 1, true},   {"kernel 16 B", 100, 4096, false},
      {"kernel 272 B", 100, 1, false}, {"kernel 1 KB", 100, 1, false},  {"kernel 4 KB", 100, 1, false},
      {"copy D2D", 100, 1, false},     {"copy H2D", 100, 1, false},     {"memset", 100, 1, false},
      {"kernel PDL", 100, 1, false},   {"kernel PDL", 400, 1, false},   {"copy D2H", 100, 1, false},
      {"copy mapped", 100, 1, false},  {"tiers", 100, 1, false},        {"tiers", 400, 1, false},
      {"if body", 1, 1, false},        {"if body", 25, 1, false},       {"if body", 50, 1, false},
      {"if body", 100, 1, false},      {"if body", 200, 1, false},      {"if body", 400, 1, false},
      {"kernel 16 B", 25, 1, false},   {"kernel 16 B", 50, 1, false},   {"kernel 16 B", 200, 1, false},
      {"if body", 60, 1, true},        {"kernel 16 B", 120, 1, false},  {"if body", 200, 1, true},
  };
  std::printf("graphs instantiated %d times each, three times; the median (KiB): a graph, a node\n", K);
  std::vector<cudaGraphExec_t> kept;
  for (const Case& c : cases) {
    cudaGraph_t g = capture(c.kind, c.n, c.grid, c.fan, dev, host, mapped);
    const long long counted = all_nodes(g);  // a conditional node's body's nodes too
    if (counted < 0) std::printf("  (the driver does not give a conditional node's body)\n");
    const size_t nodes = size_t(std::max(1ll, counted));
    std::vector<double> per;
    for (int t = 0; t < 3; ++t) {
      const size_t before = free_bytes();
      for (int k = 0; k < K; ++k) {
        // A graph with a conditional node is instantiated once (its handle is its node's): captured anew each time.
        const bool again = std::string(c.kind) == "if body" && (t > 0 || k > 0);
        cudaGraph_t one = again ? capture(c.kind, c.n, c.grid, c.fan, dev, host, mapped) : g;
        cudaGraphExec_t e = nullptr;
        CK(cudaGraphInstantiate(&e, one, 0));
        if (one != g) CK(cudaGraphDestroy(one));
        kept.push_back(e);
      }
      per.push_back((double(before) - double(free_bytes())) / K);
    }
    CK(cudaGraphDestroy(g));
    std::sort(per.begin(), per.end());
    const char* shape = !c.fan ? "chain" : std::string(c.kind) == "if body" ? "n+if" : "forked";
    std::printf("  %-13s x%-5d grid %-5d %-6s %5zu nodes: %8.1f KiB a graph, %6.2f a node (runs %.1f %.1f %.1f)\n",
                c.kind, c.n, c.grid, shape, nodes, per[1] / 1024, per[1] / 1024 / double(nodes), per[0] / 1024,
                per[1] / 1024, per[2] / 1024);
  }
  // What destroyed graphs give back: nothing, at once or later; the driver keeps it for the process's next graphs.
  const size_t held = free_bytes();
  for (cudaGraphExec_t e : kept) CK(cudaGraphExecDestroy(e));
  kept.clear();
  const size_t destroyed = free_bytes();
  CK(cudaDeviceGraphMemTrim(0));
  const size_t trimmed = free_bytes();
  cudaGraph_t g = capture("kernel 16 B", 400, 1, false, dev, host, mapped);
  for (int k = 0; k < K; ++k) {
    cudaGraphExec_t e = nullptr;
    CK(cudaGraphInstantiate(&e, g, 0));
    kept.push_back(e);
  }
  const size_t again = free_bytes();
  std::printf("all destroyed: %.1f MiB back; after cudaDeviceGraphMemTrim %.1f; %d graphs of 400 nodes made again "
              "then took %.1f MiB\n", (double(destroyed) - double(held)) / 1048576,
              (double(trimmed) - double(held)) / 1048576, K, (double(trimmed) - double(again)) / 1048576);
  for (cudaGraphExec_t e : kept) CK(cudaGraphExecDestroy(e));
  CK(cudaGraphDestroy(g));
  CK(cudaFreeHost(mapped_host));
  CK(cudaFreeHost(host));
  CK(cudaFree(dev));
}

}  // namespace flyx::bench
