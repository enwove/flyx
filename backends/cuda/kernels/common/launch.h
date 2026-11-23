// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Host side the kernel families share: options and scratch of a launch, where
// a fused producer leaves a GEMM's input, programmatic dependent launch.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <mutex>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// The launch's options (the defaults when it has none) and scratch (aborts when it has none).
const KernelOptions& opts(const Launch& L);
const Scratch& scratch(const Launch& L);
// Aborts with a message when n elements do not fit (a sizing bug in the backend, not a condition to recover from).
void check_fits(size_t n, size_t cap, const char* what);
// Where a fused producer leaves `rows` rows of K values for gemm_presplit / gemm_fp4_presplit.
Out split_out(const Launch& L, int rows, int K, bool q8 = false);
// A sequence's slice starting at `row_offset` inside a `total_rows` presplit buffer (batched rounds).
Out split_out_slice(const Launch& L, int total_rows, int row_offset, int K, bool q8 = false);
// Thread block clusters and programmatic dependent launch: compute capability 9.0+, and this build's code for the
// device compiled for 9.0+ (not an older architecture's PTX compiled at load).
bool sm90_or_newer();
// Cached device hardware queries (single-GPU backend).
int device_sm_count();
size_t device_max_smem_optin();
// Pack float32 rows [rows][ldx] into contiguous bf16 rows [rows][K] (K even).
void rows_to_bf16(const float* x, int ldx, int rows, int K, bf16* out, cudaStream_t st);

// Opt in to the maximum dynamic shared memory a kernel's block may request beside its static shared memory.
template <class Kernel>
int optin_dynamic_smem(Kernel kernel) {
  int optin = int(device_max_smem_optin());
  cudaFuncAttributes fa{};
  if (cudaFuncGetAttributes(&fa, kernel) == cudaSuccess)
    optin = std::max(0, optin - int(fa.sharedSizeBytes));
  else
    cudaGetLastError();
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, optin) != cudaSuccess) {
    cudaGetLastError();
    optin = 48 * 1024 - int(fa.sharedSizeBytes);
  }
  return std::max(0, optin);
}

// Request `bytes` of dynamic shared memory (and optionally max shared carveout) on a kernel.
template <class Kernel>
bool set_dynamic_smem(Kernel kernel, size_t bytes, bool max_carveout = false) {
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (max_carveout) {
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                             int(cudaSharedmemCarveoutMaxShared)) != cudaSuccess)
      cudaGetLastError();
  }
  return true;
}

// Active blocks of `kernel` resident across the GPU at once.
template <class Kernel>
int resident_blocks(Kernel kernel, int threads, size_t smem = 0) {
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, smem);
  return std::max(per_sm, 1) * device_sm_count();
}

// Wave-balanced persistent grid size: fewest blocks that take the same number of passes as `resident`.
inline int balanced_grid(int work_items, int per_block, int resident) {
  const int grid = std::min(ceil_div(work_items, per_block), resident);
  const int passes = ceil_div(work_items, grid * per_block);
  return ceil_div(work_items, per_block * passes);
}

// Lock-free-read cache for resident_blocks() keyed by dynamic shared memory size (a handful of K values per model).
struct ResidentCache {
  struct Entry {
    size_t smem = 0;
    int resident = 0;
    std::atomic<bool> ready{false};
  };
  Entry slots[8]{};
  std::mutex mu;

  template <class Kernel>
  int get(Kernel kernel, int threads, size_t smem) {
    for (int i = 0; i < 8; ++i) {
      if (!slots[i].ready.load(std::memory_order_acquire)) break;
      if (slots[i].smem == smem) return slots[i].resident;
    }
    const int resident = resident_blocks(kernel, threads, smem);
    std::lock_guard<std::mutex> lock(mu);
    for (int i = 0; i < 8; ++i) {
      if (slots[i].ready.load(std::memory_order_relaxed)) {
        if (slots[i].smem == smem) return slots[i].resident;
        continue;
      }
      slots[i].smem = smem;
      slots[i].resident = resident;
      slots[i].ready.store(true, std::memory_order_release);
      break;
    }
    return resident;
  }
};

// A launch as a programmatic dependent of the kernel before it (KernelOptions::pdl, sm_90+): it may start while
// that one finishes and waits (pdl_wait) before reading what it wrote. On by GPU: the RTX 5090's
// table turns it on (Qwen3.8-27B: plain +0.9%, MTP +2.3%); elsewhere off (Qwen3.5-4B NVFP4 in 2026-11: +0.7% plain
// decode, 16-row rounds 4% slower).
template <class... KArgs, class... Args>
void launch_dependent_smem(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, const Launch& L, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = L.st;
  cudaLaunchAttribute at{};
  at.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &at;
  cfg.numAttrs = opts(L).pdl && sm90_or_newer() ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...);
}

template <class... KArgs, class... Args>
void launch_dependent(void (*kernel)(KArgs...), dim3 grid, dim3 block, const Launch& L, Args... args) {
  launch_dependent_smem(kernel, grid, block, 0, L, args...);
}

}  // namespace flyx::cuda
