// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/backend.h"
#include "backends/cuda/runtime/row_pool.h"

#include <array>
#include <utility>

namespace flyx::cuda {

class CudaStream {
 public:
  CudaStream() = default;
  ~CudaStream() {
    if (handle_) cudaStreamDestroy(handle_);
  }
  CudaStream(CudaStream&& rhs) noexcept : handle_(std::exchange(rhs.handle_, nullptr)) {}
  CudaStream(const CudaStream&) = delete;
  void create() { CK(cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking)); }
  operator cudaStream_t() const { return handle_; }

 private:
  cudaStream_t handle_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() = default;
  ~CudaEvent() {
    if (handle_) cudaEventDestroy(handle_);
  }
  CudaEvent(CudaEvent&& rhs) noexcept : handle_(std::exchange(rhs.handle_, nullptr)) {}
  CudaEvent(const CudaEvent&) = delete;
  void create(unsigned flags = cudaEventDisableTiming) { CK(cudaEventCreateWithFlags(&handle_, flags)); }
  operator cudaEvent_t() const { return handle_; }

 private:
  cudaEvent_t handle_ = nullptr;
};

template <size_t N>
class StagingRing {
 public:
  class Lease {
   public:
    Lease(StagingRing& ring, size_t index, const uint8_t* src, size_t bytes, cudaStream_t consumer,
          cudaEvent_t start = nullptr, cudaEvent_t end = nullptr)
        : ring_(ring), lock_(ring.mutex_), index_(index), consumer_(consumer) {
      auto& slot = ring_.slots_.at(index);
      if (bytes > slot.capacity) throw std::logic_error("staging image exceeds slot");
      if (slot.bounce) {
        CK(cudaEventSynchronize(slot.uploaded));
        ring_.copy_to_bounce(slot.bounce, src, bytes);
        src = slot.bounce;
      }
      try {
        CK(cudaStreamWaitEvent(ring_.copy_, slot.consumed));
        if (start) CK(cudaEventRecord(start, ring_.copy_));
        CK(cudaMemcpyAsync(slot.device, src, bytes, cudaMemcpyHostToDevice, ring_.copy_));
        if (end) CK(cudaEventRecord(end, ring_.copy_));
        CK(cudaEventRecord(slot.uploaded, ring_.copy_));
      } catch (...) {
        cudaStreamSynchronize(ring_.copy_);
        throw;
      }
    }
    // A device-addressed projection can have separate code and scale planes. Fill both under one lease.
    template <class Upload>
    Lease(StagingRing& ring, size_t index, cudaStream_t consumer, Upload upload)
        : ring_(ring), lock_(ring.mutex_), index_(index), consumer_(consumer) {
      auto& slot = ring_.slots_.at(index);
      CK(cudaStreamWaitEvent(ring_.copy_, slot.consumed));
      try {
        upload(slot.device, slot.capacity, cudaStream_t(ring_.copy_));
        CK(cudaEventRecord(slot.uploaded, ring_.copy_));
      } catch (...) {
        cudaStreamSynchronize(ring_.copy_);
        throw;
      }
    }
    Lease(const Lease&) = delete;
    ~Lease() {
      if (!released_) {
        cudaStreamSynchronize(ring_.copy_);
        cudaStreamSynchronize(consumer_);
      }
    }
    uint8_t* device() const { return ring_.slots_[index_].device; }
    void wait() { CK(cudaStreamWaitEvent(consumer_, ring_.slots_[index_].uploaded)); }
    void release() {
      CK(cudaEventRecord(ring_.slots_[index_].consumed, consumer_));
      released_ = true;
    }

   private:
    StagingRing& ring_;
    std::unique_lock<std::mutex> lock_;
    size_t index_;
    cudaStream_t consumer_;
    bool released_ = false;
  };

  void allocate(flyx_bmodel* model, const std::array<size_t, N>& capacities, bool bounce, RowPool* pool = nullptr) {
    pool_ = pool;
    copy_.create();
    for (size_t i = 0; i < N; ++i) {
      auto& slot = slots_[i];
      if (!(slot.capacity = capacities[i])) continue;
      slot.device = model->alloc<uint8_t>(slot.capacity);
      if (bounce) slot.bounce = model->host_pinned<uint8_t>(slot.capacity);
      slot.uploaded.create();
      slot.consumed.create();
    }
  }
  ~StagingRing() {
    if (copy_) cudaStreamSynchronize(copy_);
    for (auto& slot : slots_)
      if (slot.consumed) cudaEventSynchronize(slot.consumed);
  }
  uint8_t* device(size_t i) const { return slots_[i].device; }
  uint8_t* bounce(size_t i) const { return slots_[i].bounce; }
  cudaStream_t stream() const { return copy_; }

 private:
  struct Slot {
    uint8_t* device = nullptr;
    uint8_t* bounce = nullptr;
    size_t capacity = 0;
    CudaEvent uploaded, consumed;
  };
  void copy_to_bounce(uint8_t* dst, const uint8_t* src, size_t bytes) {
    // Contiguous, disjoint host ranges amortize worker wakeups on multi-MiB expert images.
    constexpr size_t kCopyGrain = 256 * 1024;
    const size_t ranges = bytes / kCopyGrain;
    if (!pool_ || ranges < 2) {
      std::memcpy(dst, src, bytes);
      return;
    }
    pool_->run(ranges, [&](size_t first, size_t end) {
      const size_t offset = first * kCopyGrain;
      const size_t limit = end == ranges ? bytes : end * kCopyGrain;
      std::memcpy(dst + offset, src + offset, limit - offset);
    });
  }
  RowPool* pool_ = nullptr;
  CudaStream copy_;
  std::array<Slot, N> slots_;
  std::mutex mutex_;
};

}  // namespace flyx::cuda
