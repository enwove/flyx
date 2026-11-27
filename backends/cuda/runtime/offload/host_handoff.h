// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/runtime/offload/staging_ring.h"

namespace flyx::cuda {
// Submit both D2H reads before resident GEMMs. WDDM can buffer commands until the stream is queried.
class HostHandoff {
 public:
  HostHandoff(cudaStream_t stream, cudaEvent_t ready, const int* ids, int* host_ids, size_t pairs, const float* input,
              float* host_input, size_t values, bool direct, uint32_t* ticket, uint32_t generation)
      : stream_(stream), ready_(ready), ticket_(ticket), generation_(generation) {
    if (!direct) {
      CK(cudaMemcpyAsync(host_ids, ids, pairs * sizeof(int), cudaMemcpyDeviceToHost, stream_));
      CK(cudaMemcpyAsync(host_input, input, values * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }
    if (!ticket_) CK(cudaEventRecord(ready_, stream_));
    submit();
  }
  void wait() const {
    submit();
    if (ticket_) {
      const std::atomic_ref<uint32_t> observed(*ticket_);
      for (int spin = 0; spin < 4096; ++spin) {
        if (observed.load(std::memory_order_acquire) == generation_) return;
#if defined(_M_X64) || defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
      }
      // Failed launches and resets reach the caller through the checked CUDA wait.
      CK(cudaEventRecord(ready_, stream_));
    }
    CK(cudaEventSynchronize(ready_));
  }

 private:
  void submit() const {
    const auto status = cudaStreamQuery(stream_);
    if (status != cudaErrorNotReady) CK(status);
  }
  cudaStream_t stream_;
  cudaEvent_t ready_;
  uint32_t* ticket_;
  uint32_t generation_;
};
}  // namespace flyx::cuda
