// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Float64 reference ops for vision encoder blocks (pos_embed bilinear interpolation,
// 2D axial coordinate ordering, bias addition, 2D average pooling).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace flyx::ops {

// Add bias vector b [cols] to each row of y [rows, ld].
inline void add_bias(double* y, int ld, const double* b, int rows, int cols) {
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) y[size_t(r) * ld + c] += b[c];
}

// Patch (h, w) coordinates for all grids in spatial-merge block order:
// for each grid (T, H, W): for t in [0, T), bh in [0, H/m), bw in [0, W/m), mh in [0, m), mw in [0, m):
//   coord = (bh * m + mh, bw * m + mw).
inline void vision_spatial_coords(const int32_t* grid_thw, int n_grids, int merge_size,
                                  std::vector<int32_t>* h_out, std::vector<int32_t>* w_out) {
  h_out->clear();
  w_out->clear();
  const int m = std::max(1, merge_size);
  for (int g = 0; g < n_grids; ++g) {
    const int T = grid_thw[g * 3 + 0];
    const int H = grid_thw[g * 3 + 1];
    const int W = grid_thw[g * 3 + 2];
    const int bh_n = H / m, bw_n = W / m;
    for (int t = 0; t < T; ++t)
      for (int bh = 0; bh < bh_n; ++bh)
        for (int bw = 0; bw < bw_n; ++bw)
          for (int mh = 0; mh < m; ++mh)
            for (int mw = 0; mw < m; ++mw) {
              h_out->push_back(bh * m + mh);
              w_out->push_back(bw * m + mw);
            }
  }
}

// Bilinearly interpolate learned 2D pos_table [pos_h * pos_w, hidden] (align_corners = true)
// to each grid's (H, W) in spatial-merge block order and add into x [n_patches, hidden].
inline void vision_pos_embed_bilinear_add(double* x, const double* pos_table, int pos_h, int pos_w, int hidden,
                                          const int32_t* grid_thw, int n_grids, int merge_size) {
  const int m = std::max(1, merge_size);
  int row = 0;
  for (int g = 0; g < n_grids; ++g) {
    const int T = grid_thw[g * 3 + 0];
    const int H = grid_thw[g * 3 + 1];
    const int W = grid_thw[g * 3 + 2];
    const int bh_n = H / m, bw_n = W / m;
    const double sy = H > 1 ? double(pos_h - 1) / double(H - 1) : 0.0;
    const double sx = W > 1 ? double(pos_w - 1) / double(W - 1) : 0.0;
    for (int t = 0; t < T; ++t) {
      for (int bh = 0; bh < bh_n; ++bh) {
        for (int bw = 0; bw < bw_n; ++bw) {
          for (int mh = 0; mh < m; ++mh) {
            for (int mw = 0; mw < m; ++mw) {
              const int h = bh * m + mh;
              const int w = bw * m + mw;
              const double yf = h * sy;
              const double xf = w * sx;
              const int y0 = std::min(pos_h - 1, std::max(0, int(std::floor(yf))));
              const int x0 = std::min(pos_w - 1, std::max(0, int(std::floor(xf))));
              const int y1 = std::min(pos_h - 1, y0 + 1);
              const int x1 = std::min(pos_w - 1, x0 + 1);
              const double dy = yf - y0;
              const double dx = xf - x0;
              const double w00 = (1.0 - dy) * (1.0 - dx);
              const double w01 = (1.0 - dy) * dx;
              const double w10 = dy * (1.0 - dx);
              const double w11 = dy * dx;
              const double* p00 = pos_table + (size_t(y0) * pos_w + x0) * hidden;
              const double* p01 = pos_table + (size_t(y0) * pos_w + x1) * hidden;
              const double* p10 = pos_table + (size_t(y1) * pos_w + x0) * hidden;
              const double* p11 = pos_table + (size_t(y1) * pos_w + x1) * hidden;
              double* dst = x + size_t(row) * hidden;
              for (int c = 0; c < hidden; ++c)
                dst[c] += w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c];
              ++row;
            }
          }
        }
      }
    }
  }
}

// 2D average pool over spatial_merge_size x spatial_merge_size blocks (input in spatial-merge order:
// each consecutive merge_size^2 rows form one spatial block).
inline void vision_avg_pool_blocks(const double* x, int n_blocks, int group, int hidden, double* out) {
  const double inv = 1.0 / double(group);
  for (int b = 0; b < n_blocks; ++b) {
    double* dst = out + size_t(b) * hidden;
    for (int c = 0; c < hidden; ++c) dst[c] = 0.0;
    for (int g = 0; g < group; ++g) {
      const double* src = x + (size_t(b) * group + g) * hidden;
      for (int c = 0; c < hidden; ++c) dst[c] += src[c];
    }
    for (int c = 0; c < hidden; ++c) dst[c] *= inv;
  }
}

}  // namespace flyx::ops
