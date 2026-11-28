// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What a test expects of the GPU the CUDA backend runs on.
#pragma once

#include "flyx/backend_abi.h"

namespace flyx::test {

// Whether the backend's GDN prefill takes the chunked bf16 recurrence on this device, so that a hybrid GDN model's
// prefill passes are aligned to its chunk (prefill_align > 1): GA10x, Ada and consumer Blackwell (sm_86, sm_89, sm_120,
// sm_121), as backends/cuda/kernels/gdn_chunk.cu's gdn_chunk_device says; the datacenter parts (A100, H100, H200,
// B200, B300) keep the float32 recurrence and passes of any length (prefill_align 1).
inline bool gdn_chunked_device(const flyx_backend_api* api) {
  flyx_device_info d{};
  if (!api || !api->probe || api->probe(&d) != 0) return false;
  const int cc = 10 * d.sm_major + d.sm_minor;
  return cc == 86 || cc == 89 || cc == 120 || cc == 121;
}

}  // namespace flyx::test
