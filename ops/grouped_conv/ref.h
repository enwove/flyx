// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DFlash2's grouped dynamic causal convolution over a draft block.
#pragma once

namespace flyx::ops {

// DFlash2 grouped dynamic causal conv over a block of rows (row index mod block
// is the position in the block).  x: [rows, H]; delta: [rows, taps, G]; base: [taps, H].
inline void grouped_conv(const double* x, const double* delta, const double* base, int rows, int H, int taps,
                         int group, int block, double* y) {
  int G = H / group;
  for (int t = 0; t < rows; ++t)
    for (int c = 0; c < H; ++c) {
      double acc = 0;
      for (int tap = 0; tap < taps; ++tap) {
        if ((t % block) < tap) continue;
        double coef = base[size_t(tap) * H + c] + delta[(size_t(t) * taps + tap) * G + c / group];
        acc += coef * x[size_t(t - tap) * H + c];
      }
      y[size_t(t) * H + c] = acc;
    }
}

}  // namespace flyx::ops
