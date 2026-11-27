// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backends/cuda/runtime/offload/staging_ring.h"

namespace flyx::cuda {
// Submit both D2H reads before resident GEMMs. WDDM can buffer commands until the stream is queried.
class HostHandoff {
 public:
  HostHandoff(cudaStream_t stream, cudaEvent_t ready, const int* ids, int* host_ids, size_t pairs, const float* input,
              float* host_input, size_t values)
      : stream_(stream), ready_(ready) {
    CK(cudaMemcpyAsync(host_ids, ids, pairs * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CK(cudaMemcpyAsync(host_input, input, values * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CK(cudaEventRecord(ready_, stream_));
    submit();
  }
  void wait() const {
    submit();
    CK(cudaEventSynchronize(ready_));
  }

 private:
  void submit() const {
    const auto status = cudaStreamQuery(stream_);
    if (status != cudaErrorNotReady) CK(status);
  }
  cudaStream_t stream_;
  cudaEvent_t ready_;
};
}  // namespace flyx::cuda
