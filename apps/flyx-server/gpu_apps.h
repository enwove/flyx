// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// On-demand listing and safe local closing of applications holding GPU memory.
//
// Windows: matches the CUDA GPU by DXGI adapter description/LUID, queries per-process Local and Dedicated GPU memory
// via PDH English counters, reads live VRAM totals from NVML/DXGI, and resolves friendly app names from version info.
// Linux: queries NVML compute and graphics running processes and reads /proc/<pid> metadata.
//
// Safety invariants for closing:
// - Never trusts client-supplied PIDs or paths: `id` must match an entry in a fresh live GPU scan.
// - Never closes the current flyx process, any ancestor process of flyx (terminal/IDE that spawned it), OS/desktop
//   compositor processes, or processes owned by another user.
// - Graceful close first (WM_CLOSE to visible top-level windows on Windows, SIGTERM on Linux); force-termination only
//   when explicitly requested after user confirmation.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace flyx::server {

struct GpuApp {
  std::string id;            // opaque stable key derived from the normalized executable path / process name
  std::string name;          // user-facing application name (e.g. "Google Chrome", "Discord")
  std::string process_name;  // executable basename without extension (e.g. "chrome", "Discord")
  uint64_t bytes = 0;        // primary VRAM figure shown (dedicated on Windows user apps, NVML used on Linux)
  uint64_t local_bytes = 0;  // resident local VRAM bytes
  int processes = 0;         // number of GPU-holding PIDs grouped into this app
  bool closable = false;     // true only for same-user, non-system, non-ancestor applications
  std::string reason;        // "" | "system" | "runs_flyx" | "other_user"
};

struct GpuAppSnapshot {
  bool supported = false;
  std::string device_name;
  uint64_t mem_total = 0;
  uint64_t mem_used = 0;
  uint64_t mem_free = 0;
  uint64_t own_bytes = 0;     // VRAM held by this flyx process
  uint64_t others_bytes = 0;  // mem_used minus own_bytes (clamped)
  uint64_t system_bytes = 0;  // local VRAM held by OS desktop/compositor processes (read-only summary)
  std::string idle_igpu;      // optional name of an idle integrated GPU if present
  // The device the readings are of is the one the hint names and no other could be: the only GPU of its name (the
  // only one, without a hint).  Identical GPUs are told apart by no name, and the backend reports no PCI or UUID
  // identity (flyx_device_info), so readings of one of several are not taken for the device a model is on.
  bool device_unique = false;
  std::vector<GpuApp> apps;   // non-self, non-system apps sorted by bytes descending
};

enum class CloseStatus {
  kClosed,        // app released GPU / exited
  kStillRunning,  // signaled (WM_CLOSE / SIGTERM) but still holding GPU (e.g. tray or unsaved prompt)
  kNoWindow,      // Windows: background/tray app with no visible top-level window to send WM_CLOSE to
  kNotFound,      // app id is no longer in the live GPU list
  kForbidden,     // system, ancestor ("runs_flyx"), or other-user process
  kAccessDenied,  // OS denied termination (e.g. elevated process)
  kUnsupported,   // platform/driver does not expose GPU processes
};

struct GpuCloseResult {
  CloseStatus status = CloseStatus::kUnsupported;
  int windows_signaled = 0;
  std::string app_name;
  GpuAppSnapshot snapshot;
};

// Inspects applications currently occupying GPU memory on the device matching `device_hint` (empty: best discrete GPU).
GpuAppSnapshot list_gpu_apps(const std::string& device_hint);

// Safely closes (force=false) or terminates (force=true) an application currently occupying GPU memory.
GpuCloseResult close_gpu_app(const std::string& device_hint, const std::string& id, bool force);

// Inspects applications currently occupying host system RAM (grouped by executable).
GpuAppSnapshot list_ram_apps();

// Safely closes (force=false) or terminates (force=true) an application currently occupying host system RAM.
GpuCloseResult close_ram_app(const std::string& id, bool force);

// Pure helpers exposed for unit testing.
// Which of the GPUs NVML lists (`names` by index; nullopt where its handle or name was not read) a hint names: the
// first it names, else the first read (-1: none read); `unique`: readings of it are surely the hinted device's, the
// one GPU there is or the only one the hint names with every GPU's name read (one not read could be its twin).
struct DeviceMatch {
  int index = -1;
  bool unique = false;
};
DeviceMatch match_device(const std::vector<std::optional<std::string>>& names, const std::string& hint);
bool is_system_gpu_process(const std::string& proc_name, const std::string& exe_path);
std::string clean_gpu_app_name(const std::string& proc_name, const std::string& product_name,
                               const std::string& file_desc);
std::string gpu_app_group_id(const std::string& normalized_key);

}  // namespace flyx::server
