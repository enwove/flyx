// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdexcept>

namespace flyx::cuda {

struct CudaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct OutOfDeviceMemory : CudaError {
  using CudaError::CudaError;
};

}  // namespace flyx::cuda
