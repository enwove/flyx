// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Live figures of the GPU (flyx_backend_api.device_stats) from NVML, the
// driver's management library, loaded at run time: it ships with the driver
// (also under WSL2), and a machine without it only loses the figures. Only
// the calls below are used, declared from NVML's documented C API.
//
// Any thread: NVML is thread-safe, and nothing here touches a model, a
// sequence or the backend's error string.
//
// What this process holds of the GPU (flyx_memory_info.process) comes from Windows, which counts video memory by
// process (DXGI, the adapter that is the CUDA device by its LUID); NVML does not tell a process's share there, nor
// under WSL2.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")  // CreateDXGIFactory1
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "backends/cuda/backend.h"

namespace flyx::cuda {
namespace {

struct NvmlDevice;  // opaque nvmlDevice_t
struct NvmlUtilization {
  unsigned int gpu, memory;
};
struct NvmlMemory {
  unsigned long long total, free, used;
};
// nvmlMemory_v2_t (driver 510 on): its used memory is nvidia-smi's, without what the driver reserves of the device
// for itself (its firmware, the display), which is in neither the used nor the free memory.
struct NvmlMemory2 {
  unsigned int version;
  unsigned long long total, reserved, free, used;
};
constexpr unsigned int kNvmlMemory2 = unsigned(sizeof(NvmlMemory2)) | (2u << 24);  // NVML_STRUCT_VERSION(Memory, 2)
constexpr int kTemperatureGpu = 0;  // NVML_TEMPERATURE_GPU
constexpr int kClockSm = 1;         // NVML_CLOCK_SM

struct Nvml {
  int (*init)();
  int (*by_bus_id)(const char*, NvmlDevice**);
  int (*utilization)(NvmlDevice*, NvmlUtilization*);
  int (*memory)(NvmlDevice*, NvmlMemory*);
  int (*temperature)(NvmlDevice*, int, unsigned int*);
  int (*power)(NvmlDevice*, unsigned int*);        // milliwatts
  int (*power_limit)(NvmlDevice*, unsigned int*);  // milliwatts
  int (*clock)(NvmlDevice*, int, unsigned int*);   // MHz
  int (*pcie_gen)(NvmlDevice*, unsigned int*);
  int (*pcie_width)(NvmlDevice*, unsigned int*);
  int (*fan)(NvmlDevice*, unsigned int*);          // percent
  int (*reasons)(NvmlDevice*, unsigned long long*) = nullptr;  // clock event reasons (optional)
  int (*memory2)(NvmlDevice*, NvmlMemory2*) = nullptr;         // nvmlDeviceGetMemoryInfo_v2 (optional)
  NvmlDevice* dev = nullptr;
};

constexpr int kClockMem = 2;  // NVML_CLOCK_MEM

// NVML's clock event reason bits (nvmlClocksEventReasons) as ours.
uint32_t clock_limits(unsigned long long r) {
  uint32_t out = 0;
  if (r & 0x1) out |= FLYX_CLOCK_IDLE;                 // GpuIdle
  if (r & 0x2) out |= FLYX_CLOCK_APP_SETTING;          // ApplicationsClocksSetting
  if (r & (0x4 | 0x80)) out |= FLYX_CLOCK_POWER_CAP;   // SwPowerCap, HwPowerBrakeSlowdown
  if (r & 0x8) out |= FLYX_CLOCK_HW_SLOWDOWN;          // HwSlowdown
  if (r & 0x10) out |= FLYX_CLOCK_SYNC_BOOST;          // SyncBoost
  if (r & (0x20 | 0x40)) out |= FLYX_CLOCK_THERMAL;    // SwThermalSlowdown, HwThermalSlowdown
  if (r & 0x100) out |= FLYX_CLOCK_DISPLAY;            // DisplayClockSetting
  return out;
}

std::mutex g_nvml_mu;
std::string g_bus_id;  // of the device probe() chose
#if defined(_WIN32)
LUID g_luid{};                       // of that device, as Windows names its adapter
bool g_has_luid = false;
IDXGIAdapter3* g_adapter = nullptr;  // that adapter, found once and kept for the process
bool g_adapter_tried = false;

// The DXGI adapter with g_luid, or null.  g_nvml_mu held.
IDXGIAdapter3* find_adapter() {
  IDXGIFactory1* factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) return nullptr;
  IDXGIAdapter3* found = nullptr;
  IDXGIAdapter1* a = nullptr;
  for (UINT i = 0; !found && factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
    DXGI_ADAPTER_DESC1 d{};
    if (SUCCEEDED(a->GetDesc1(&d)) && std::memcmp(&d.AdapterLuid, &g_luid, sizeof(LUID)) == 0)
      a->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&found));
    a->Release();
  }
  factory->Release();
  return found;
}
#endif
bool g_tried = false;
Nvml g_nvml;
bool g_ready = false;

// NVML's library, opened once and kept open for the process (nvml.dll comes with the Windows driver, in System32).
void* open_nvml_library() {
#if defined(_WIN32)
  return reinterpret_cast<void*>(LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#else
  return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

template <typename F>
bool bind(void* lib, const char* name, F* f) {
#if defined(_WIN32)
  *f = reinterpret_cast<F>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  *f = reinterpret_cast<F>(dlsym(lib, name));
#endif
  return *f != nullptr;
}

void open_nvml() {
  if (g_bus_id.empty()) return;
  void* lib = open_nvml_library();
  if (!lib) return;
  Nvml& n = g_nvml;
  const bool all = bind(lib, "nvmlInit_v2", &n.init) &&
                   bind(lib, "nvmlDeviceGetHandleByPciBusId_v2", &n.by_bus_id) &&
                   bind(lib, "nvmlDeviceGetUtilizationRates", &n.utilization) &&
                   bind(lib, "nvmlDeviceGetMemoryInfo", &n.memory) &&
                   bind(lib, "nvmlDeviceGetTemperature", &n.temperature) &&
                   bind(lib, "nvmlDeviceGetPowerUsage", &n.power) &&
                   bind(lib, "nvmlDeviceGetEnforcedPowerLimit", &n.power_limit) &&
                   bind(lib, "nvmlDeviceGetClockInfo", &n.clock) &&
                   bind(lib, "nvmlDeviceGetCurrPcieLinkGeneration", &n.pcie_gen) &&
                   bind(lib, "nvmlDeviceGetCurrPcieLinkWidth", &n.pcie_width) &&
                   bind(lib, "nvmlDeviceGetFanSpeed", &n.fan);
  if (!bind(lib, "nvmlDeviceGetCurrentClocksEventReasons", &n.reasons))
    bind(lib, "nvmlDeviceGetCurrentClocksThrottleReasons", &n.reasons);  // its name before driver 535
  bind(lib, "nvmlDeviceGetMemoryInfo_v2", &n.memory2);
  g_ready = all && n.init() == 0 && n.by_bus_id(g_bus_id.c_str(), &n.dev) == 0;
}

}  // namespace

void remember_device(int device) {
  char bus[64] = {};
  if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) == cudaSuccess) {
    std::lock_guard<std::mutex> lock(g_nvml_mu);
    g_bus_id = bus;
  }
#if defined(_WIN32)
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    std::lock_guard<std::mutex> lock(g_nvml_mu);
    static_assert(sizeof(prop.luid) == sizeof(LUID), "cudaDeviceProp::luid is a LUID's bytes");
    std::memcpy(&g_luid, prop.luid, sizeof(LUID));
    g_has_luid = true;
  }
#endif
}

namespace {

// The process's video memory of a segment group on the device, as Windows counts it; 0 elsewhere or unknown.
size_t process_video_memory([[maybe_unused]] int group) {
#if defined(_WIN32)
  std::lock_guard<std::mutex> lock(g_nvml_mu);
  if (!g_adapter_tried && g_has_luid) {
    g_adapter_tried = true;
    g_adapter = find_adapter();
  }
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (g_adapter && SUCCEEDED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP(group), &info)))
    return size_t(info.CurrentUsage);
#endif
  return 0;
}

// The host memory the process holds page-locked for the device (flyx_device_stats.process_host): Windows' non-local
// segment group.  Measured on an RTX 5070 Ti (driver 610.88): a 1 GiB cudaHostAlloc adds 1,024 MiB to it, to the
// working set and to the private bytes; 1 GiB of host memory from cuMemCreate (the K/V past the device's part) adds
// 1,024 MiB to it and nothing to either of the others.
size_t process_host_memory() {
#if defined(_WIN32)
  return process_video_memory(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
#else
  return 0;
#endif
}

}  // namespace

// The process's dedicated video memory on the device, as Windows counts it: 11,164 MiB with the 27B's 2-bit file
// loaded and no sequence (RTX 5070 Ti, driver 610.88), the "GPU Process Memory" counter's 11,168, where the weights
// and the kernels' workspace are 10,653: the CUDA context and its kernels' modules are the rest.
size_t gpu_process_memory() {
#if defined(_WIN32)
  return process_video_memory(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
#else
  return 0;
#endif
}

// Remaining page-locked (WDDM non-local) host memory budget for this device, less a safety reserve so DWM, driver
// pools and lazy checkpoints stay inside WDDM's 50% shared-memory cap.  UINT64_MAX where no WDDM cap applies.
uint64_t host_pinned_budget() {
#if defined(_WIN32)
  std::lock_guard<std::mutex> lock(g_nvml_mu);
  if (!g_adapter_tried && g_has_luid) {
    g_adapter_tried = true;
    g_adapter = find_adapter();
  }
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (g_adapter && SUCCEEDED(g_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &info)) &&
      info.Budget > 0) {
    const uint64_t reserve = std::max<uint64_t>(512ull << 20, info.Budget / 16);
    const uint64_t used = uint64_t(info.CurrentUsage) + reserve;
    return uint64_t(info.Budget) > used ? uint64_t(info.Budget) - used : 0;
  }
#endif
  return UINT64_MAX;
}


// On an RTX 5060 Ti 8 GB under Windows 11 (driver 610.88), buffers allocated one after another: cudaMemGetInfo's free
// memory is NVML's to 5.1 GB held, 650 MB below it at 5.4, 0 at 6.7 GB held with 645 MiB free by NVML.  Buffers of
// 32 MiB stayed on the GPU until NVML's free memory was 239 and 247 MiB (two runs); of 256 MiB to 1 GiB, one that
// would leave 560 MiB or less went partly to host memory (NVML's used memory grew by a half to nine tenths of it, the
// buffers read at 19 to 168 GB/s, not 435), one asked with 690 MiB or less free wholly.
//
// On an RTX 5070 Ti 16 GB (Windows 11 26300, driver 610.88), the same with buffers of each size until the driver
// refused one, beside the process's own 64 MiB buffers or another process's: the NVML free memory a buffer still went
// wholly to the GPU at (each run twice, or in both cases), and past it the buffers read at 51 to 54 GB/s, not 820 to
// 1,000 (DXGI counts such a buffer as the process's local memory, not its shared memory):
//   one cudaMalloc or cuMemCreate of 256 MiB  417 to 633 MiB (none whole past 452)
//   one cudaMalloc of 32 MiB                  212 to 270 MiB
//   one cudaMalloc of 8 MiB                   129 to 137 MiB
//   pieces of 8 MiB (cuMemCreate, one range)  121 to 137 MiB
//   pieces of 2 MiB (cuMemCreate, one range)   61 to  73 MiB
// So Windows places an allocation by its own size, and memory mapped from small pieces (device_pieces) as it does
// small allocations; past the floor, beside another process, it moved that process's memory out instead.  The budget
// reads NVML's free memory there and keeps flyx's reserves from it (plan/budget.cpp: 384 MB, total / 40 on GPUs of
// more than 15 GB); device_malloc keeps kPagedFloor (768 MiB) free, memory in pieces kPiecesFloor (256 MiB).
//
// The total is what programs share there: NVML's total less what the driver reserves of the device for itself (308
// MiB of the 5070 Ti's 16,303), which no program holds and none can allocate: counted in, the budget took it for
// another program's (flyx_memory_info: total = free + what processes hold).
void memory_of(const NvmlFigures& n, size_t* free, size_t* total) {
  if (n.v2 && n.total > n.reserved && n.total - n.reserved >= n.free) {
    *free = size_t(n.free);
    *total = size_t(n.total - n.reserved);
  } else if (n.v1 || n.v2) {
    *free = size_t(n.free);
  }
}

bool gpu_memory(size_t* free, size_t* total) {
  size_t fr = 0, tot = 0;
  if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return false;
#if defined(_WIN32)
  bool ready = false;
  {
    std::lock_guard<std::mutex> lock(g_nvml_mu);  // NVML opened once, as device_stats() opens it
    if (!g_tried && !g_bus_id.empty()) {
      g_tried = true;
      open_nvml();
    }
    ready = g_ready;
  }
  NvmlFigures n;
  NvmlMemory2 m2{};
  m2.version = kNvmlMemory2;
  NvmlMemory m{};
  if (ready && g_nvml.memory2 && g_nvml.memory2(g_nvml.dev, &m2) == 0) {
    n.v2 = true;
    n.total = m2.total, n.reserved = m2.reserved, n.free = m2.free;
  } else if (ready && g_nvml.memory(g_nvml.dev, &m) == 0) {
    n.v1 = true;
    n.total = m.total, n.free = m.free;
  }
  memory_of(n, &fr, &tot);
#endif
  *free = fr;
  *total = tot;
  return true;
}

int device_stats(flyx_device_stats* out) {
  *out = {};
  out->util = out->mem_util = out->temp_c = out->sm_clock_mhz = out->pcie_gen = out->pcie_width = out->fan =
      out->mem_clock_mhz = -1;
  out->power_w = out->power_limit_w = -1;
  {
    std::lock_guard<std::mutex> lock(g_nvml_mu);
    if (!g_tried && !g_bus_id.empty()) {
      g_tried = true;
      open_nvml();
    }
    if (!g_ready) return -1;
  }
  const Nvml& n = g_nvml;
  unsigned int v = 0;
  NvmlUtilization u{};
  if (n.utilization(n.dev, &u) == 0) out->util = int(u.gpu), out->mem_util = int(u.memory);
  NvmlMemory m{};
  if (n.memory(n.dev, &m) == 0) out->mem_used = m.used, out->mem_total = m.total;
  if (n.temperature(n.dev, kTemperatureGpu, &v) == 0) out->temp_c = int(v);
  if (n.power(n.dev, &v) == 0) out->power_w = v / 1000.0;
  if (n.power_limit(n.dev, &v) == 0) out->power_limit_w = v / 1000.0;
  if (n.clock(n.dev, kClockSm, &v) == 0) out->sm_clock_mhz = int(v);
  if (n.pcie_gen(n.dev, &v) == 0) out->pcie_gen = int(v);
  if (n.pcie_width(n.dev, &v) == 0) out->pcie_width = int(v);
  if (n.fan(n.dev, &v) == 0) out->fan = int(v);
  if (n.clock(n.dev, kClockMem, &v) == 0) out->mem_clock_mhz = int(v);
  unsigned long long reasons = 0;
  if (n.reasons && n.reasons(n.dev, &reasons) == 0) out->clock_limits = clock_limits(reasons);
  out->process_host = process_host_memory();
  return 0;
}

}  // namespace flyx::cuda
