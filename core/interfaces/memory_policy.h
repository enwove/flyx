// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared VRAM policy threshold constants used by core/plan and backends so OOM thresholds stay consistent.
#pragma once

#include <cstdint>

namespace flyx::interfaces {

// Whether other programs hold enough GPU memory to suggest closing them: 256 MiB when this process's CUDA runtime is
// separated out, else 1 GiB so the process's own runtime (~0.3-0.7 GB) is not blamed on other apps.
inline constexpr uint64_t kClosableOthersKnownBytes = 256ull << 20;
inline constexpr uint64_t kClosableOthersUnknownBytes = 1ull << 30;

// Upper charge per low-bit head for placement: packed codes plus two binary16 grid values.
// Three-bit codes can need packing padding. CUDA's supported layouts are checked against this bound.
constexpr uint64_t strata_copy_head_bound(int head_dim, int bits) {
  return uint64_t(head_dim) * (bits == 2 ? 2 : 4) / 8 + 4;
}

}  // namespace flyx::interfaces
