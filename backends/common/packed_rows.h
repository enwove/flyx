// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "flyx/backend_abi.h"

namespace flyx::backends {

// One immutable table per instance. All tiers return original packed rows in caller order.
class PackedRows {
 public:
  PackedRows(const flyx_tensor_view& table, int placement, uint64_t hot_bytes, uint64_t host_room) : table_(table) {
    if (table.ndim != 2 || table.shape[0] <= 0 || table.shape[1] <= 0 || !table.nbytes ||
        table.nbytes % table.shape[0] || table.nbytes > SIZE_MAX)
      throw std::runtime_error("PLE: invalid packed table");
    stride_ = size_t(table.nbytes / table.shape[0]);
    if (placement == FLYX_PLE_AUTO)
      placement = table.nbytes <= host_room || !table.read_at ? FLYX_PLE_HOST
                  : hot_bytes                                 ? FLYX_PLE_HYBRID
                                                              : FLYX_PLE_DISK;
    if (placement == FLYX_PLE_GPU || placement == FLYX_PLE_HOST) {
      if (!table.data) throw std::runtime_error("PLE: resident table has no source");
      const auto* src = static_cast<const uint8_t*>(table.data);
      resident_.assign(src, src + table.nbytes);
      return;
    }
    if (placement != FLYX_PLE_DISK && placement != FLYX_PLE_HYBRID) throw std::runtime_error("PLE: invalid placement");
    if (!table.read_at || !table.file || table.file_offset > UINT64_MAX - table.nbytes)
      throw std::runtime_error("PLE: disk placement requires a positioned file reader");
    if (placement == FLYX_PLE_HYBRID) {
      const uint64_t bytes = sizeof(Set) + kWays * stride_;
      sets_ = size_t(std::min<uint64_t>(hot_bytes / bytes, (uint64_t(table.shape[0]) + kWays - 1) / kWays));
      if (sets_) {
        cache_ = std::make_unique<Set[]>(sets_);
        payload_.resize(sets_ * kWays * stride_);
      }
    }
  }

  size_t stride() const { return stride_; }
  size_t storage_bytes() const { return resident_.size() + payload_.size() + sets_ * sizeof(Set); }

  void gather(const int32_t* rows, size_t n, uint8_t* dst) {
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return rows[a] < rows[b]; });
    for (size_t i = 0; i < n;) {
      const int32_t id = rows[order[i]];
      if (id < 0 || id >= table_.shape[0]) throw std::runtime_error("PLE: row outside table");
      auto* row = dst + order[i++] * stride_;
      if (!resident_.empty())
        std::memcpy(row, resident_.data() + size_t(id) * stride_, stride_);
      else if (!lookup(id, row)) {
        if (table_.read_at(table_.file, table_.file_offset + uint64_t(id) * stride_, row, stride_))
          throw std::runtime_error("PLE: failed to read packed row " + std::to_string(id));
        insert(id, row);
      }
      while (i < n && rows[order[i]] == id) std::memcpy(dst + order[i++] * stride_, row, stride_);
    }
  }

 private:
  static constexpr int kWays = 4;
  struct Set {
    std::mutex lock;
    int32_t id[kWays] = {-1, -1, -1, -1};
    bool used[kWays]{};
    unsigned hand = 0;
  };
  bool lookup(int32_t id, uint8_t* dst) {
    if (!sets_) return false;
    const size_t index = size_t(id) % sets_;
    auto& set = cache_[index];
    std::lock_guard guard(set.lock);
    for (int i = 0; i < kWays; ++i) {
      if (set.id[i] != id) continue;
      set.used[i] = true;
      std::memcpy(dst, payload_.data() + (index * kWays + i) * stride_, stride_);
      return true;
    }
    return false;
  }
  void insert(int32_t id, const uint8_t* src) {
    if (!sets_) return;
    const size_t index = size_t(id) % sets_;
    auto& set = cache_[index];
    std::lock_guard guard(set.lock);
    for (auto key : set.id)
      if (key == id) return;
    while (set.used[set.hand]) {
      set.used[set.hand] = false;
      set.hand = (set.hand + 1) % kWays;
    }
    auto slot = set.hand;
    std::memcpy(payload_.data() + (index * kWays + slot) * stride_, src, stride_);
    set.id[slot] = id;
    set.used[slot] = true;
    set.hand = (slot + 1) % kWays;
  }

  flyx_tensor_view table_;
  size_t stride_ = 0, sets_ = 0;
  std::vector<uint8_t> resident_, payload_;
  std::unique_ptr<Set[]> cache_;
};
}  // namespace flyx::backends
