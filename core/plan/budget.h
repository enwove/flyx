// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Device memory accounting for one engine: the budget, the leases of the
// weights and the kernels' workspace, and what one session will lease.
#pragma once

#include <cstdint>
#include <string>

#include "core/resources/ledger.h"
#include "flyx/backend_abi.h"

namespace flyx::plan {

// What holds the GPU besides the model's weights and workspace, as the budget found it: other programs with the
// driver, and the process's own runtime (its CUDA context, its kernels' modules; a second model's memory, did the
// process hold two).  Where the system counts a process's share (flyx_memory_info::process) the two are told apart;
// where it does not, the runtime is in `others` and runtime_known false.
struct GpuHeld {
  uint64_t total = 0;  // the GPU's memory (0: the backend reports none)
  uint64_t others = 0, runtime = 0;
  bool runtime_known = false;
};

// A context shorter than the one wanted, where account() fitted it to memory: what a longer one did not fit in,
// the GPU or the host memory K/V may take.
enum class CtxLimit { kNone, kGpu, kHost };

struct Budget {
  Lease weights, workspace;
  uint64_t session_bytes = 0;  // one session's device memory at the engine's context
  uint64_t sessions_room = 0;  // of the budget after the weights and the workspace: what the sessions share
  GpuHeld held;
  CtxLimit ctx_limit = CtxLimit::kNone;
  // Where a session's K/V caches go, the host memory that takes, and how much may be taken.
  flyx_seq_layout layout{0, INT32_MAX, FLYX_KV_READ_ALL, 0, 0};
  uint64_t session_host_bytes = 0, host_budget = 0;
  // Of session_host_bytes: what is mapped for the device only, in no process memory figure (flyx_seq_size.host_mapped).
  uint64_t session_host_mapped = 0;
  std::string tier_note;  // why K/V that did not fit (or past tiers.gpu) stayed on the GPU
  // FLYX_KV_AUTO placed with every position exact: its copy did not fit beside the K/V (why: copy_note).
  bool auto_exact = false;
  std::string copy_note;
  // True when account() returned false due to insufficient device memory (not a non-OOM backend error).
  bool oom = false;
};

// The tiers a session's K/V caches may use (EngineOptions::kv_gpu, kv_ram): positions on the GPU at most (-1: as
// many as fit), bytes of host memory (-1: what the machine can spare, less `reserved`; 0: none).  copy_gpu
// (EngineOptions::kv_copy_gpu): the layers whose low-bit copy stays on the GPU at most, the others' in host memory
// (-1: as many as fit).
struct KvTiers {
  int gpu = -1;
  int64_t ram = -1;
  int copy_gpu = -1;
  uint64_t reserved = 0;
  int method = FLYX_KV_READ_ALL, method_arg = 0;  // how rounds read the K/V (flyx_seq_layout)
  bool gpu_paged = false;  // the GPU's memory is Windows' to page (os::gpu_memory_paged): reserve at least 832 MiB
  bool allow_auto_exact = true;  // a locked AUTO switch must survive placement and retries
};

// The host memory kv_ram's auto leaves to the system and other programs, of a machine with `total_ram` bytes: a
// sixteenth of it, at least 2 GiB (32 GB: 2 GiB; 64 GB: 4; 128 GB: 8).  The K/V's host memory is page-locked, so
// nothing else can have it back while the model is served.
uint64_t host_keep(uint64_t total_ram);

// Device VRAM held back for the CUDA context, fragmentation, and desktop stability (at least 832 MiB on paged WDDM).
uint64_t device_reserve(uint64_t total_vram, bool gpu_paged);

// Sets the ledger's budget (what the device has now plus what the model already
// took, less a reserve for the CUDA context, fragmentation and other programs;
// a session's round graphs are in seq_bytes), leases
// the weights and workspace, and sizes a session at `*max_ctx`.  When that
// session's K/V caches do not fit the GPU, their positions past the ones that do
// go to host memory, as `tiers` allow; when they do not fit there
// either and `min_ctx` > 0, *max_ctx shrinks to the longest context that fits the
// GPU, a multiple of 1024 of at least min_ctx. `longest` (--ctx
// max): when *max_ctx does not fit with its K/V in tiers either, *max_ctx becomes the
// longest context the GPU and host memory hold together, a multiple of 1024.
// false, with the options that would fit in *err (no drafter among them when
// `has_drafter`; NVFP4 weights when the matrices are `mostly_bf16`, else a
// smaller quantization; closing other programs when they hold enough of the GPU to matter), when not even one
// session fits (then *err says what the shortest session there is keeps on the GPU and how much is missing),
// or the budget has no room for the weights themselves.  A second call (after the
// backend gave memory back, the ABI's trim) starts from *out's leases given back.
// Where *max_ctx came out shorter than asked, out->ctx_limit says what a longer one did not fit in.
bool account(const flyx_backend_api* api, flyx_bmodel* bm, bool has_drafter, bool mostly_bf16, int* max_ctx,
             int min_ctx, const KvTiers& tiers, ResourceLedger* ledger, Budget* out, std::string* err,
             bool longest = false);

// What holds the GPU besides the model, in words ("other programs and the driver hold 3.77 GB of its 17.09 GB,
// flyx's own runtime 0.54 GB"); "" when nothing does or the backend reports no memory.
std::string held_note(const GpuHeld& h);
// Whether closing other programs is worth naming: they hold a quarter of a gigabyte or more, a gigabyte where the
// process's own runtime (a few hundred megabytes) cannot be told from them.
bool closable(const GpuHeld& h);
// Why the context is no longer (b.ctx_limit) and what would give more, for the load's log; "" when nothing cut it.
std::string ctx_limit_note(const Budget& b);

// Try to fit two sessions at the context and attention precision account() selected for one.
// The model and workspace are already charged to the ledger. Keep the original layout if both cannot fit.
void fit_second_session(const flyx_backend_api* api, flyx_bmodel* bm, const KvTiers& tiers,
                        const ResourceLedger& ledger, Budget* budget);

}  // namespace flyx::plan
