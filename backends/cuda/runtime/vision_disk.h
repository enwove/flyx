// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Auto-deleted temporary file store for disk-offloaded vision encoder weights.
// Keeps 0 bytes of GPU VRAM and 0 bytes of locked host RAM; once flushed clean at load time,
// both Windows (Standby list) and Linux (reclaimable Page Cache) count the backing pages as
// available memory for host KV budgeting while still serving warm reads from OS cache.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace flyx::cuda {

class VisionDiskStore {
 public:
  VisionDiskStore() = default;
  ~VisionDiskStore();
  VisionDiskStore(const VisionDiskStore&) = delete;
  VisionDiskStore& operator=(const VisionDiskStore&) = delete;

  // Append `nbytes` aligned to 256 bytes, creating the auto-deleted temp file on first call.
  // Returns the 256-byte-aligned byte offset where `data` was written.
  uint64_t append(const void* data, size_t nbytes);

  // Flush dirty pages to disk so OS cache holds clean reclaimable Standby / Page Cache pages.
  void flush_clean();

  // Positional read at `offset`. Lazily maps the clean file read-only on first read so warm
  // OS Standby/Page Cache reads run at memcpy speed without per-image soft page-fault storms,
  // while remaining clean and evictable by the OS whenever host KV needs the physical RAM.
  void read_at(uint64_t offset, void* dst, size_t nbytes) const;

  void close();
  bool active() const { return handle_ != -1; }
  uint64_t bytes() const { return size_; }

 private:
  void ensure_open();
  void ensure_mapped() const;
  intptr_t handle_ = -1;
  uint64_t size_ = 0;
  mutable std::mutex map_mu_;
  mutable const uint8_t* map_base_ = nullptr;
  mutable size_t map_len_ = 0;
  mutable bool map_tried_ = false;
};

}  // namespace flyx::cuda
