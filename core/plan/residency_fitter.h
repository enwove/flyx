// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <cstddef>
#include <cstdint>

#include "core/plan/budget.h"

namespace flyx::plan {

// Prefer changing a single axis. Probe each legal cap because changing a staging bank's size can increase VRAM.
class ResidencyFitter {
 public:
  using Caps = std::array<int, 2>;
  // ram_reserve: EngineOptions::ram_reserve, bytes kept free in place of host_keep() (-1: host_keep()).
  static uint64_t host_room(uint64_t total, uint64_t available, uint64_t reserved, int64_t ram_reserve = -1) {
    if (!total || !available) return UINT64_MAX;
    available -= std::min(available, ram_reserve >= 0 ? uint64_t(ram_reserve) : host_keep(total));
    return available - std::min(available, reserved);
  }

  template <class Probe, class Room, class Reclaim>
  std::optional<Caps> fit_weights(Caps before, Caps limits, bool retry, Probe probe, Room room, Reclaim reclaim) {
    for (int pass = 0; pass < 2; ++pass) {
      const uint64_t available = room();
      bool host_limited = false;
      auto cap = fit(before, limits, retry, [&](Caps trial) {
        const auto estimate = probe(trial);
        if (!estimate) return false;
        if (estimate->offload_pinned_bytes <= available &&
            estimate->offload_pageable_bytes <= available - estimate->offload_pinned_bytes)
          return true;
        host_limited = true;
        return false;
      });
      if (cap || !host_limited || pass) return cap;
      reclaim();
    }
    return std::nullopt;
  }

  template <class Fits>
  std::optional<Caps> fit(Caps before, Caps limits, bool retry, Fits fits) {
    for (size_t axis = 0; axis < before.size(); ++axis) {
      if (!limits[axis]) continue;
      Caps trial = before;
      const int high = before[axis] - 1;
      const int start = retry ? std::max(0, before[axis] - backoff_[axis]) : high;
      for (int pass = 0; pass < 2; ++pass) {
        const int end = pass ? start + 1 : 0;
        for (int i = pass ? high : start; i >= end; --i) {
          trial[axis] = i;
          if (!fits(trial)) continue;
          if (retry) backoff_[axis] += std::min(backoff_[axis], limits[axis] - backoff_[axis]);
          return trial;
        }
      }
    }
    if (limits[0] && limits[1]) {
      for (int l = before[1] - 1; l >= 0; --l)
        for (int e = before[0] - 1; e >= 0; --e)
          if (Caps trial{e, l}; fits(trial)) return trial;
    }
    return std::nullopt;
  }

 private:
  Caps backoff_{1, 1};
};
}  // namespace flyx::plan
