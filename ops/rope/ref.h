// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Rotary position embedding.
#pragma once

#include <cmath>

namespace flyx::ops {

// Rotate-half RoPE on the first rot_dim dims of one head.
inline void rope(double* x, int pos, double theta, int rot_dim) {
  int half = rot_dim / 2;
  for (int i = 0; i < half; ++i) {
    double inv = 1.0 / std::pow(theta, (2.0 * i) / rot_dim);
    double a = pos * inv, c = std::cos(a), s = std::sin(a);
    double x1 = x[i], x2 = x[i + half];
    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

// Which axis (0 = T, 1 = H, 2 = W) frequency pair `i` of [0, rot_dim/2) reads in 3D MRoPE.
// When sec is all zeros, always returns 0 (identical to 1D RoPE).
inline int mrope_axis(int i, const int sec[3], bool interleaved) {
  if (sec[0] == 0 && sec[1] == 0 && sec[2] == 0) return 0;
  if (interleaved) {
    if (i % 3 == 1 && i < 3 * sec[1]) return 1;
    if (i % 3 == 2 && i < 3 * sec[2]) return 2;
    return 0;
  }
  if (i < sec[0]) return 0;
  if (i < sec[0] + sec[1]) return 1;
  return 2;
}

// 3D Multimodal RoPE (interleaved or contiguous sections) on the first rot_dim dims of one head.
// When pt == ph == pw == pos, reduces identically to rope(x, pos, theta, rot_dim).
inline void rope_3d(double* x, int pt, int ph, int pw, double theta, int rot_dim, const int sec[3],
                    bool interleaved) {
  const int p[3] = {pt, ph, pw};
  const int half = rot_dim / 2;
  for (int i = 0; i < half; ++i) {
    const int pos = p[mrope_axis(i, sec, interleaved)];
    const double inv = 1.0 / std::pow(theta, (2.0 * i) / rot_dim);
    const double a = pos * inv, c = std::cos(a), s = std::sin(a);
    const double x1 = x[i], x2 = x[i + half];
    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

// 2D axial Vision RoPE (Qwen3-VL / Qwen3.5 / Qwen3.8): spatial_dim = head_dim / 2,
// quarter = head_dim / 4 frequency pairs for H (dims 0..quarter-1 paired with half..half+quarter-1)
// and quarter pairs for W (dims quarter..half-1 paired with half+quarter..head_dim-1).
inline void rope_2d_axial(double* x, int ph, int pw, double theta, int head_dim) {
  const int half = head_dim / 2;
  const int spatial_dim = half;
  const int quarter = spatial_dim / 2;
  for (int i = 0; i < half; ++i) {
    const bool is_w = i >= quarter;
    const int j = is_w ? i - quarter : i;
    const int coord = is_w ? pw : ph;
    const double inv = 1.0 / std::pow(theta, (2.0 * j) / spatial_dim);
    const double a = coord * inv, c = std::cos(a), s = std::sin(a);
    const double x1 = x[i], x2 = x[i + half];
    x[i] = x1 * c - x2 * s;
    x[i + half] = x2 * c + x1 * s;
  }
}

}  // namespace flyx::ops
