// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace flyx::cuda {

void blocks_unpack(int dtype, const uint8_t* payload, const uint16_t* scales, size_t n, uint8_t* file);

struct BlockLayout {
  int values, bytes, stride, scales;  // per block: weights, bytes in the file, payload bytes and scales on the device
};
BlockLayout blocks_layout(int dtype);  // values 0: not a type the kernels take

}  // namespace flyx::cuda
