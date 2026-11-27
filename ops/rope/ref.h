// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Rotary position embedding.
#pragma once

#include <algorithm>
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

// YaRN's rotary factors (Peng et al., 2023; flyx_rope_scaling FLYX_ROPE_YARN), as gpt-oss and Hugging Face's
// _compute_yarn_parameters define them: pair i's factor between theta^(-2i/rot_dim) (extrapolated) and that / factor
// (interpolated), by a linear ramp from the dim where a wavelength makes beta_fast rotations over orig_ctx positions
// (all extrapolated below it) to the one where it makes beta_slow (all interpolated past it), the two rounded out to
// whole dims when truncate.  inv: rot_dim / 2 factors.
inline void yarn_inv_freq(double theta, int rot_dim, double factor, int orig_ctx, double beta_fast, double beta_slow,
                          bool truncate, double* inv) {
  const double pi = 3.14159265358979323846;
  auto dim_of = [&](double rotations) {  // the dim whose wavelength makes `rotations` turns over orig_ctx
    return rot_dim * std::log(orig_ctx / (rotations * 2 * pi)) / (2 * std::log(theta));
  };
  double lo = dim_of(beta_fast), hi = dim_of(beta_slow);
  if (truncate) lo = std::floor(lo), hi = std::ceil(hi);
  lo = std::max(lo, 0.0);
  hi = std::min(hi, double(rot_dim - 1));
  if (lo == hi) hi += 0.001;  // a ramp of no width: Hugging Face's guard against a division by zero
  for (int i = 0; i < rot_dim / 2; ++i) {
    const double extra = 1.0 / std::pow(theta, (2.0 * i) / rot_dim), inter = extra / factor;
    const double ramp = std::clamp((i - lo) / (hi - lo), 0.0, 1.0), keep = 1.0 - ramp;  // keep: extrapolated part
    inv[i] = inter * (1.0 - keep) + extra * keep;
  }
}

// The scale YaRN gives q and k after the rotary (its "attention factor", applied to cos and sin).
inline double yarn_mscale(double factor) { return factor <= 1.0 ? 1.0 : 0.1 * std::log(factor) + 1.0; }

// Rotate-half RoPE on the first rot_dim dims of one head with the factors given (rot_dim / 2 of them).
inline void rope_with(double* x, int pos, const double* inv, int rot_dim) {
  const int half = rot_dim / 2;
  for (int i = 0; i < half; ++i) {
    const double a = pos * inv[i], c = std::cos(a), s = std::sin(a);
    const double x1 = x[i], x2 = x[i + half];
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
