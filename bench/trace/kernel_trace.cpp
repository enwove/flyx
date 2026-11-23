// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A kernel timeline of an unchanged CUDA program, through CUPTI's activity API.
// The driver loads this library at cuInit when told to:
//
//   CUDA_INJECTION64_PATH=<build>/bin/libflyx-trace.so FLYX_TRACE=out.tsv <program> ...
// On Windows set CUDA_INJECTION64_PATH to the absolute path to libflyx-trace.dll,
// and put the CUPTI runtime DLL on PATH. Windows writes completed records at
// device synchronization, profiler stop or stream/context teardown. An explicit
// device sync at the end includes the final asynchronous work.
//
// Every kernel (and memcpy / memset) the program runs, graph nodes included,
// goes to out.tsv when the program exits: device start and end (ns), the
// launch it belongs to (a graph launch's correlation id is shared by all its
// nodes), the graph, grid, block and demangled name.  bench/trace/gaps.py
// reads it.  Concurrent-kernel records keep the program's overlap (PDL)
// as it is; tracing adds a little time per kernel, so compare the traced
// speed with an untraced run before reading small differences.
//
// Programs that run CUDA in more than one process (servers with an engine
// process: vLLM, SGLang): %p in FLYX_TRACE is the process id, a file a
// process; a process that ran no kernel writes none (else a front end that
// only initialised CUDA would overwrite its engine's trace at the same path).
#include <cupti.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#ifdef FLYX_TRACE_CUFILT
#include <nv_decode.h>
#endif
#define FLYX_TRACE_EXPORT __declspec(dllexport)
#else
#include <cxxabi.h>
#include <unistd.h>
#define FLYX_TRACE_EXPORT __attribute__((visibility("default")))
#endif

namespace {

// FLYX_TRACE is a native path. Windows' narrow environment loses characters that
// the active code page cannot represent.
std::filesystem::path trace_path() {
#ifdef _WIN32
  const wchar_t* env = _wgetenv(L"FLYX_TRACE");
  std::wstring path = env && *env ? env : L"flyx_trace.tsv";
  const auto at = path.find(L"%p");
  if (at != path.npos) path.replace(at, 2, std::to_wstring(_getpid()));
#else
  const char* env = std::getenv("FLYX_TRACE");
  std::string path = env && *env ? env : "flyx_trace.tsv";
  const auto at = path.find("%p");
  if (at != path.npos) path.replace(at, 2, std::to_string(getpid()));
#endif
  return path;
}

struct Row {
  char kind;  // K kernel, M memcpy, S memset
  uint64_t start, end;
  uint32_t corr, graph, stream;
  int32_t gx, gy, gz, bx;
  std::string name;
};

std::mutex g_mu;
std::vector<Row> g_rows;
bool g_started = false;

void CUPTIAPI buffer_requested(uint8_t** buf, size_t* size, size_t* max_records) {
  *size = size_t(16) << 20;
  *buf = static_cast<uint8_t*>(std::malloc(*size));  // malloc satisfies CUPTI's 8-byte alignment.
  *max_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  std::lock_guard<std::mutex> lock(g_mu);
  CUpti_Activity* rec = nullptr;
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      const auto* k = reinterpret_cast<const CUpti_ActivityKernel10*>(rec);
      g_rows.push_back({'K', k->start, k->end, k->correlationId, k->graphId, k->streamId, k->gridX, k->gridY,
                        k->gridZ, k->blockX, k->name ? k->name : "?"});
    } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      const auto* c = reinterpret_cast<const CUpti_ActivityMemcpy6*>(rec);
      g_rows.push_back({'M', c->start, c->end, c->correlationId, c->graphId, c->streamId, int32_t(c->copyKind), 0,
                        0, 0, "memcpy " + std::to_string(c->bytes)});
    } else if (rec->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
      const auto* c = reinterpret_cast<const CUpti_ActivityMemset4*>(rec);
      g_rows.push_back({'S', c->start, c->end, c->correlationId, c->graphId, c->streamId, 0, 0, 0, 0,
                        "memset " + std::to_string(c->bytes)});
    }
  }
  std::free(buf);
}

std::string demangle(const std::string& s) {
  int status = 0;
  char* d = nullptr;
#if defined(FLYX_TRACE_CUFILT)
  d = __cu_demangle(s.c_str(), nullptr, nullptr, &status);
#elif !defined(_WIN32)
  d = abi::__cxa_demangle(s.c_str(), nullptr, nullptr, &status);
#endif
  std::string out = (status == 0 && d) ? d : s;
  std::free(d);
  for (char& c : out)
    if (c == '\t' || c == '\n') c = ' ';
  return out;
}

void write_rows() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_rows.empty()) return;
  if (!g_started) {
    bool kernels = false;
    for (const auto& r : g_rows) kernels |= r.kind == 'K';
    if (!kernels) return;
  }
  static const auto path = trace_path();
#ifdef _WIN32
  FILE* f = _wfopen(path.c_str(), g_started ? L"a" : L"w");
#else
  FILE* f = std::fopen(path.c_str(), g_started ? "a" : "w");
#endif
  if (!f) return;
  if (!g_started) std::fprintf(f, "kind\tstart\tend\tcorr\tgraph\tstream\tgx\tgy\tgz\tbx\tname\n");
  for (const auto& r : g_rows)
    std::fprintf(f, "%c\t%llu\t%llu\t%u\t%u\t%u\t%d\t%d\t%d\t%d\t%s\n", r.kind, (unsigned long long)r.start,
                 (unsigned long long)r.end, r.corr, r.graph, r.stream, r.gx, r.gy, r.gz, r.bx,
                 r.kind == 'K' ? demangle(r.name).c_str() : r.name.c_str());
  std::fclose(f);
  if (!g_started) {
    const auto utf8 = path.u8string();
    std::fprintf(stderr, "flyx-trace: %.*s\n", int(utf8.size()), reinterpret_cast<const char*>(utf8.data()));
  }
  g_started = true;
  g_rows.clear();
}

#ifndef _WIN32
void finish() {
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  write_rows();
}
#endif

using Callback = std::pair<CUpti_CallbackDomain, CUpti_CallbackId>;

constexpr Callback kDestroy[] = {
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxReset_v2},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuDevicePrimaryCtxRelease_v2},
};
#ifdef _WIN32
constexpr Callback kFlush[] = {
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize},
#if CUPTI_API_VERSION >= 130000
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize_v2},
#endif
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuStreamDestroy_v2},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuProfilerStop},
};
#endif

// Flush on the API thread before destruction. Resource callbacks cannot flush.
// Windows also drains completed records before DLL detach can stop CUPTI's worker. Stream and event syncs can be on the token path.
void CUPTIAPI on_api(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* args) {
  const bool destroy = std::find(std::begin(kDestroy), std::end(kDestroy), Callback{domain, id}) != std::end(kDestroy);
  if (static_cast<const CUpti_CallbackData*>(args)->callbackSite != (destroy ? CUPTI_API_ENTER : CUPTI_API_EXIT)) return;
  cuptiActivityFlushAll(destroy ? CUPTI_ACTIVITY_FLAG_FLUSH_FORCED : 0);
#ifdef _WIN32
  write_rows();
#endif
}

}  // namespace

extern "C" FLYX_TRACE_EXPORT int InitializeInjection(void) {
  static CUpti_SubscriberHandle sub;
  if (cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed) != CUPTI_SUCCESS) return 0;
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
  const bool subscribed = cuptiSubscribe(&sub, on_api, nullptr) == CUPTI_SUCCESS;
  if (subscribed)
    for (const auto& [domain, id] : kDestroy) cuptiEnableCallback(1, sub, domain, id);
#ifdef _WIN32
  if (!subscribed) return 0;
  for (const auto& [domain, id] : kFlush) cuptiEnableCallback(1, sub, domain, id);
#else
  std::atexit(finish);
#endif
  return 1;
}
