// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

namespace flyx::cuda {

struct BlockLayout {
  int values, bytes, stride, scales;  // per block: weights, bytes in the file, payload bytes and scales on the device
};
BlockLayout blocks_layout(int dtype);  // values 0: not a type the kernels take

}  // namespace flyx::cuda
