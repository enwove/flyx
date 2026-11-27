// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared VRAM policy helpers used by core/plan, core/exec, and backends so OOM thresholds and advice stay consistent.
#pragma once

#include <cstdint>
#include <string>

namespace flyx::interfaces {

// Whether other programs hold enough GPU memory to suggest closing them: 256 MiB when this process's CUDA runtime is
// separated out, else 1 GiB so the process's own runtime (~0.3-0.7 GB) is not blamed on other apps.
inline constexpr uint64_t kClosableOthersKnownBytes = 256ull << 20;
inline constexpr uint64_t kClosableOthersUnknownBytes = 1ull << 30;

inline bool closable_others(uint64_t others, bool runtime_known) {
  return others >= (runtime_known ? kClosableOthersKnownBytes : kClosableOthersUnknownBytes);
}

// Single source of truth for the actionable suffix appended to VRAM OOM diagnostics.
inline std::string oom_fix_advice(bool has_others, bool has_drafter) {
  if (has_others) {
    return std::string("Close other apps") + (has_drafter ? ", disable speculative decoding" : "") +
           ", or use a smaller quantization.";
  }
  return has_drafter ? "Disable speculative decoding or use a smaller quantization."
                     : "Use a smaller quantization.";
}

}  // namespace flyx::interfaces
