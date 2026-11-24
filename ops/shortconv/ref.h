// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Causal short depthwise convolution with residual scaling and optional per-head dynamic taps (Inkling-2).
// History holds the last (conv - 1) input vectors in chronological order (slot 0 oldest, slot conv-2 newest).
// Static weights w are [C, conv] where w[c, 0] multiplies oldest (t - conv + 1) and w[c, conv - 1] multiplies x[t].
// Dynamic taps dyn (when non-null) are [rows, n_heads, conv] where lag 0 multiplies x[t] (newest) and
// lag (conv - 1) multiplies oldest, broadcast across each head's (C / n_heads) channels.
// Output: (x[t] + static_conv + dyn_conv) / sqrt((dyn ? 2 : 1) * conv + 1).
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace flyx::ops {

inline void shortconv(const double* x, const double* w, const double* dyn, double* state, int rows, int C, int conv,
                      int n_heads, double* out) {
  const int hist = conv - 1;
  const int head_c = dyn ? (C / n_heads) : 0;
  const double inv_scale = 1.0 / std::sqrt(double((dyn ? 2 : 1) * conv + 1));
  std::vector<double> cur_hist(state, state + size_t(hist) * size_t(C));
  for (int r = 0; r < rows; ++r) {
    const double* xr = x + size_t(r) * size_t(C);
    const double* dr = dyn ? (dyn + size_t(r) * size_t(n_heads) * size_t(conv)) : nullptr;
    double* yr = out + size_t(r) * size_t(C);
    for (int c = 0; c < C; ++c) {
      const double* wc = w + size_t(c) * size_t(conv);
      double acc = 0.0;
      for (int k = 0; k < hist; ++k) acc += cur_hist[size_t(k) * size_t(C) + size_t(c)] * wc[k];
      acc += xr[c] * wc[hist];
      if (dr) {
        const double* dh = dr + size_t(c / head_c) * size_t(conv);
        acc += xr[c] * dh[0];
        for (int lag = 1; lag < conv; ++lag)
          acc += cur_hist[size_t(hist - lag) * size_t(C) + size_t(c)] * dh[lag];
      }
      yr[c] = (xr[c] + acc) * inv_scale;
    }
    // Shift history left by 1 and push xr.
    for (int k = 0; k + 1 < hist; ++k)
      for (int c = 0; c < C; ++c) cur_hist[size_t(k) * size_t(C) + size_t(c)] = cur_hist[size_t(k + 1) * size_t(C) + size_t(c)];
    if (hist > 0)
      for (int c = 0; c < C; ++c) cur_hist[size_t(hist - 1) * size_t(C) + size_t(c)] = xr[c];
  }
  for (size_t i = 0; i < cur_hist.size(); ++i) state[i] = cur_hist[i];
}

}  // namespace flyx::ops
