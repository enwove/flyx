// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// cuda_bench paging KIND STEP [PIECE [FILL_TO]]: where the driver stops keeping a process's device allocations on the
// GPU (Windows pages them out to host memory instead of refusing them), by the allocation's size: the numbers behind
// backends/cuda/runtime/device_stats.cpp and kPiecesFloor.
//
//   KIND malloc   one cudaMalloc of STEP MiB a step
//        pieces   one address range of STEP MiB a step, mapped from cuMemCreate pieces of PIECE MiB (default 8)
//        one      one cuMemCreate of STEP MiB a step
//   FILL_TO       first cudaMalloc 64 MiB buffers until NVML's free memory is at most this many MiB (default 1600)
//
// Each step prints NVML's free memory before and after, how much of the buffer its used memory took, the process's
// local and non-local (shared) video memory as Windows counts them (DXGI; 0 elsewhere), and the read bandwidth of the
// new buffer and of all of them: a buffer the driver placed in host memory reads at the bus's speed.  Until the driver
// refuses one, or 24 steps past the first that went to host memory.  Run it with the GPU otherwise idle: other
// programs' allocations move the free memory under it.
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bench/microbench/bench.cuh"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#else
#include <dlfcn.h>
#endif

namespace flyx::bench {

namespace {

struct Nvml2 {
  unsigned int version;
  unsigned long long total, reserved, free, used;
};
struct Nvml1 {
  unsigned long long total, free, used;
};
int (*g_mem1)(void*, Nvml1*) = nullptr;
int (*g_mem2)(void*, Nvml2*) = nullptr;
void* g_dev = nullptr;
#if defined(_WIN32)
IDXGIAdapter3* g_adapter = nullptr;
#endif

template <class F>
bool sym(void* lib, const char* name, F* f) {
#if defined(_WIN32)
  *f = reinterpret_cast<F>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  *f = reinterpret_cast<F>(dlsym(lib, name));
#endif
  return *f != nullptr;
}

bool open_counters() {
#if defined(_WIN32)
  void* lib = reinterpret_cast<void*>(LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#else
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
  int (*init)() = nullptr;
  int (*by_bus)(const char*, void**) = nullptr;
  if (!lib || !sym(lib, "nvmlInit_v2", &init) || !sym(lib, "nvmlDeviceGetHandleByPciBusId_v2", &by_bus) ||
      !sym(lib, "nvmlDeviceGetMemoryInfo", &g_mem1))
    return false;
  sym(lib, "nvmlDeviceGetMemoryInfo_v2", &g_mem2);
  char bus[64] = {};
  CK(cudaDeviceGetPCIBusId(bus, sizeof(bus), 0));
  if (init() != 0 || by_bus(bus, &g_dev) != 0) return false;
#if defined(_WIN32)
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  IDXGIFactory1* f = nullptr;
  if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&f)))) {
    IDXGIAdapter1* a = nullptr;
    for (UINT i = 0; !g_adapter && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
      DXGI_ADAPTER_DESC1 d{};
      if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, prop.luid, sizeof(LUID)) == 0)
        a->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&g_adapter));
      a->Release();
    }
    f->Release();
  }
#endif
  return true;
}

struct Sample {
  double free = 0, used = 0, reserved = 0, local = 0, nonlocal = 0;
};
Sample sample() {
  CK(cudaDeviceSynchronize());
  constexpr double kMiB = 1048576.0;
  Sample s;
  Nvml1 m{};
  if (g_mem1(g_dev, &m) == 0) s.free = m.free / kMiB, s.used = m.used / kMiB;
  Nvml2 m2{};
  m2.version = unsigned(sizeof(Nvml2)) | (2u << 24);
  if (g_mem2 && g_mem2(g_dev, &m2) == 0) s.reserved = m2.reserved / kMiB;
#if defined(_WIN32)
  DXGI_QUERY_VIDEO_MEMORY_INFO l{}, n{};
  if (g_adapter && SUCCEEDED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &l)) &&
      SUCCEEDED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &n)))
    s.local = l.CurrentUsage / kMiB, s.nonlocal = n.CurrentUsage / kMiB;
#endif
  return s;
}

__global__ void k_read(const uint4* p, size_t n, unsigned* out) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) *out = acc;
}

// GB/s reading `bytes` at p: the best of three after one untimed.
double read_gbs(const void* p, size_t bytes, unsigned* out) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  float best = 1e30f;
  for (int r = 0; r < 4; ++r) {
    CK(cudaEventRecord(a));
    k_read<<<1024, 256>>>(static_cast<const uint4*>(p), bytes / 16, out);
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    if (r > 0) best = std::min(best, ms);
  }
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return bytes / (best * 1e-3) / 1e9;
}

template <class F>
F driver(const char* name) {
  void* p = nullptr;
  cudaDriverEntryPointQueryResult q{};
  if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
      q != cudaDriverEntryPointSuccess)
    return nullptr;
  return reinterpret_cast<F>(p);
}

}  // namespace

int bench_paging(int argc, char** argv) {
  const std::string kind = argc > 2 ? argv[2] : "";
  if (kind != "malloc" && kind != "pieces" && kind != "one") {
    std::fprintf(stderr, "usage: cuda_bench paging malloc|pieces|one STEP_MIB [PIECE_MIB [FILL_TO_MIB]]\n");
    return 2;
  }
  const long long step_mib = argc > 3 ? std::atoll(argv[3]) : 32, piece_mib = argc > 4 ? std::atoll(argv[4]) : 8;
  const double fill_to = argc > 5 ? std::atof(argv[5]) : 1600;
  // Sizes the driver maps: whole multiples of 2 MiB, its granularity on every GPU flyx runs on.
  if (step_mib < 2 || piece_mib < 2 || step_mib % 2 || piece_mib % 2 || fill_to < 0) {
    std::fprintf(stderr, "cuda_bench paging: STEP and PIECE are whole even numbers of MiB, FILL_TO not negative\n");
    return 2;
  }
  const size_t step = size_t(step_mib) << 20, piece = size_t(piece_mib) << 20;
  CK(cudaFree(nullptr));
  if (!open_counters()) {
    std::fprintf(stderr, "cuda_bench paging: no NVML\n");
    return 1;
  }
  const auto create = driver<decltype(&cuMemCreate)>("cuMemCreate");
  const auto reserve = driver<decltype(&cuMemAddressReserve)>("cuMemAddressReserve");
  const auto map = driver<decltype(&cuMemMap)>("cuMemMap");
  const auto access = driver<decltype(&cuMemSetAccess)>("cuMemSetAccess");
  const auto granularity = driver<decltype(&cuMemGetAllocationGranularity)>("cuMemGetAllocationGranularity");
  if (kind != "malloc" && !(create && reserve && map && access)) {
    std::fprintf(stderr, "cuda_bench paging: the driver maps no memory (VMM)\n");
    return 1;
  }
  int dev = 0;
  CK(cudaGetDevice(&dev));
  CUmemAllocationProp dp{};
  dp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, dev};
  CUmemAccessDesc rw{};
  rw.location = {CU_MEM_LOCATION_TYPE_DEVICE, dev};
  rw.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  size_t gran = 0;
  if (kind != "malloc" && (!granularity || granularity(&gran, &dp, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
                           !gran || step % gran || piece % gran)) {
    std::fprintf(stderr, "cuda_bench paging: STEP and PIECE must be multiples of the granularity (%zu bytes)\n", gran);
    return 2;
  }
  unsigned* out = nullptr;
  CK(cudaMalloc(&out, sizeof(unsigned)));

  const Sample s0 = sample();
  std::printf("NVML free %.0f MiB, used %.0f (of it %.0f reserved by the driver); this process: local %.0f, "
              "non-local %.0f MiB\n", s0.free, s0.used, s0.reserved, s0.local, s0.nonlocal);
  std::vector<void*> fill;
  while (sample().free > fill_to + 64) {
    void* p = nullptr;
    if (cudaMalloc(&p, size_t(64) << 20) != cudaSuccess) {
      cudaGetLastError();
      break;
    }
    CK(cudaMemset(p, 0, size_t(64) << 20));
    fill.push_back(p);
  }
  const Sample s1 = sample();
  std::printf("%zu buffers of 64 MiB first: NVML free %.0f MiB, local %.0f, non-local %.0f\n", fill.size(), s1.free,
              s1.local, s1.nonlocal);
  std::printf("%-4s %9s %9s %9s %9s %9s %9s %9s\n", "step", "free", "free_aft", "on_gpu", "local", "nonlocal",
              "new_GB/s", "all_GB/s");
  struct Buf {
    void* p;
    size_t bytes;
  };
  std::vector<Buf> bufs;
  int past = -1;  // steps since the first buffer that went to host memory
  for (int i = 0; i < 1000 && past < 24; ++i) {
    const Sample b = sample();
    void* p = nullptr;
    if (kind == "malloc") {
      if (cudaMalloc(&p, step) != cudaSuccess) {
        cudaGetLastError();
        std::printf("cudaMalloc refused\n");
        break;
      }
    } else {
      CUdeviceptr va = 0;
      const size_t pc = kind == "one" ? step : piece;
      CUresult r = reserve(&va, step, 0, 0, 0);
      for (size_t o = 0; r == CUDA_SUCCESS && o < step; o += pc) {
        CUmemGenericAllocationHandle h = 0;
        const size_t n = std::min(pc, step - o);
        r = create(&h, n, &dp, 0);
        if (r == CUDA_SUCCESS) r = map(va + o, n, 0, h, 0);
      }
      if (r == CUDA_SUCCESS) r = access(va, step, &rw, 1);
      if (r == CUDA_ERROR_OUT_OF_MEMORY) {
        std::printf("cuMemCreate refused\n");
        break;
      }
      if (r != CUDA_SUCCESS) {
        std::fprintf(stderr, "cuda_bench paging: a VMM call failed (%d)\n", int(r));
        return 1;
      }
      p = reinterpret_cast<void*>(va);
    }
    CK(cudaMemset(p, 0, step));
    bufs.push_back({p, step});
    const Sample a = sample();
    const double gbs = read_gbs(p, step, out);
    double bytes = 0, ms = 0;
    for (const Buf& x : bufs) {
      bytes += double(x.bytes);
      ms += double(x.bytes) / (read_gbs(x.p, x.bytes, out) * 1e9) * 1e3;
    }
    const double on_gpu = a.used - b.used;
    if (past >= 0 || on_gpu < step / 1048576.0 - 1) ++past;
    std::printf("%-4d %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f %9.0f\n", i, b.free, a.free, on_gpu, a.local - b.local,
                a.nonlocal - b.nonlocal, gbs, bytes / (ms * 1e-3) / 1e9);
    std::fflush(stdout);
  }
  // The process's memory goes with it; the buffers are not freed one by one.
  return 0;
}

}  // namespace flyx::bench
