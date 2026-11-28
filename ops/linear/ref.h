// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Linear layers: y = x W^T, with W in any stored format.
#pragma once

#include "ops/dtype/ref.h"

namespace flyx::ops {

// y[r, n] = sum_k x[r, k] * W[n, k] for rows r; W is [N, K] row-major.  Each sum starts from zero and runs over k
// in order.  Four outputs are summed side by side, so the CPU overlaps four chains of additions instead of waiting
// on one; each output gets the bits it gets alone.
inline void linear(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  constexpr int kSideBySide = 4;
  std::vector<double> w(static_cast<size_t>(kSideBySide) * K, 0.0);  // rows n0 .. n0 + 3 of W
  for (int n0 = 0; n0 < N; n0 += kSideBySide) {
    const int m = std::min(kSideBySide, N - n0);
    for (int j = 0; j < m; ++j) load_run(W, size_t(n0 + j) * K, K, &w[size_t(j) * K]);
    for (int r = 0; r < rows; ++r) {
      const double* xr = x + size_t(r) * K;
      double acc[kSideBySide] = {};
      for (int k = 0; k < K; ++k)
        for (int j = 0; j < kSideBySide; ++j) acc[j] += xr[k] * w[size_t(j) * K + k];
      for (int j = 0; j < m; ++j) y[size_t(r) * N + n0 + j] = acc[j];
    }
  }
}

// linear() of bf16-rounded inputs.
inline void linear_a16(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  std::vector<double> xr(x, x + size_t(rows) * K);
  for (auto& v : xr) v = round_bf16(v);
  linear(xr.data(), rows, K, W, N, y);
}

// linear() of inputs rounded to bf16 and held as binary16 over each row's scale (round_a16h): GGML block weights.
inline void linear_a16h(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  std::vector<double> xr(x, x + size_t(rows) * K);
  for (int r = 0; r < rows; ++r) round_a16h(&xr[size_t(r) * K], K);
  linear(xr.data(), rows, K, W, N, y);
}

// linear() of inputs rounded as NVFP4 (round_nvfp4): NVFP4 weights whose GEMM takes more than FLYX_MAX_ROWS rows
// at once (FLYX_ACT_NVFP4_BF16).  K % 16 == 0.
inline void linear_a4(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  std::vector<double> xr(x, x + size_t(rows) * K);
  for (int r = 0; r < rows; ++r) round_nvfp4(&xr[size_t(r) * K], K);
  linear(xr.data(), rows, K, W, N, y);
}

// linear() of inputs rounded to MXFP8 (round_mxfp8): MXFP4 weights asked for it (FLYX_ACT_MXFP8).  K % 32 == 0.
inline void linear_amxfp8(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  std::vector<double> xr(x, x + size_t(rows) * K);
  for (int r = 0; r < rows; ++r) round_mxfp8(&xr[size_t(r) * K], K);
  linear(xr.data(), rows, K, W, N, y);
}

// linear() of inputs rounded to 8-bit integers with a scale per 32 values (round_q8): GGML block weights in every
// pass when the weight says so (FLYX_ACT_Q8; since ABI 18 rounds too).  Rows whose K is not a multiple of 32
// (e.g. quantized vision MLPs with padded weights) are zero-padded to a multiple of 32 before rounding.
inline void linear_a8(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  const int K_pad = (K + 31) / 32 * 32;
  std::vector<double> xr(size_t(rows) * K);
  std::vector<double> padded;
  if (K_pad != K) padded.assign(static_cast<size_t>(K_pad), 0.0);
  std::vector<int8_t> p(static_cast<size_t>(K_pad));
  std::vector<float> d(static_cast<size_t>(K_pad / 32));
  for (int r = 0; r < rows; ++r) {
    const double* row = x + size_t(r) * K;
    if (K_pad != K) {
      std::copy(row, row + K, padded.data());
      row = padded.data();
    }
    round_q8(row, K_pad, p.data(), d.data());
    for (int k = 0; k < K; ++k) xr[size_t(r) * K + k] = double(d[size_t(k / 32)]) * p[size_t(k)];
  }
  linear(xr.data(), rows, K, W, N, y);
}

// A low-rank adapter beside a linear (backend_abi.h flyx_lora): y += scale (x a^T) b^T, a [rank, K], b [N, rank];
// u = scale (x a^T) first, then b's sums over u, each with linear()'s order.
inline void linear_lora(const double* x, int rows, int K, const flyx_tensor_view* a, const flyx_tensor_view* b,
                        int rank, double scale, int N, double* y) {
  std::vector<double> u(size_t(rows) * rank), d(size_t(rows) * N);
  linear(x, rows, K, a, rank, u.data());
  for (double& v : u) v *= scale;
  linear(u.data(), rows, rank, b, N, d.data());
  for (size_t i = 0; i < d.size(); ++i) y[i] += d[i];
}

}  // namespace flyx::ops
