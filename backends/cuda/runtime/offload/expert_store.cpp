// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_store.h"
#include "ops/dtype/ref.h"
#include <cstring>
#include <cmath>

namespace flyx::cuda {
ColdMatrix::ColdMatrix(const flyx_weight& w, int hot, const ExpertLayout& layout)
    : first(hot),
      rows(int(w.t->shape[1])),
      cols(int(w.t->shape[2])),
      dtype(w.t->dtype),
      q8(layout.format == ExpertLayout::ggml_blocks && w.activations == FLYX_ACT_Q8),
      a16h(layout.format == ExpertLayout::ggml_blocks && !q8),
      nvfp4(layout.format == ExpertLayout::nvfp4) {
  const size_t count = size_t(w.t->shape[0] - hot), nrows = size_mul(count, size_t(rows));
  if (w.scale2) {
    if (!w.scale2->data) throw std::runtime_error("missing cold expert scales");
    size_t n = 1;
    for (int i = 0; i < w.scale2->ndim; ++i) n = size_mul(n, size_t(w.scale2->shape[i]));
    if (n != 1 && n != size_t(w.t->shape[0])) throw std::runtime_error("invalid cold expert scale count");
    scales.reserve(count);
    for (size_t i = 0; i < count; ++i) scales.push_back(float(ops::load(w.scale2, n == 1 ? 0 : hot + i)));
  }
  if (layout.format == ExpertLayout::bf16_rows && dtype != FLYX_DT_BF16) {
    dtype = FLYX_DT_BF16;
    stride = size_mul(size_t(cols), 2);
    data.resize(size_mul(nrows, stride));
    const size_t start = size_t(hot) * rows * cols;
    for (size_t i = 0; i < data.size() / 2; ++i) {
      uint16_t b = ops::bf16_bits(float(ops::load(w.t, start + i)));
      std::memcpy(data.data() + 2 * i, &b, 2);
    }
  } else {
    stride = dtype == FLYX_DT_BF16 ? size_mul(size_t(cols), 2)
                                   : size_mul(size_t(cols / flyx_block_values(dtype)), size_t(flyx_block_bytes(dtype)));
    if (size_mul(layout.rows, stride) != w.t->nbytes) throw std::runtime_error("invalid cold expert payload size");
    const auto* src = static_cast<const uint8_t*>(w.t->data) + size_t(hot) * rows * stride;
    data.assign(src, src + size_mul(nrows, stride));
    if (nvfp4) {
      for (size_t offset = 0; offset < data.size(); offset += 36)
        for (int g = 0; g < 4; ++g) {
          auto& scale = data[offset + g];
          if (scale == 255) throw std::runtime_error("invalid NVFP4 cold expert scale");
          scale = scale == 127 ? 0 : uint8_t(scale & 127);
        }
    }
  }
}

void ColdMatrix::prepare(float* x, bool vector_nvfp4) const {
  if (q8) {
    for (int i = 0; i < cols; i += 32) {
      ColdQ8 group;
      const int count = std::min(32, cols - i);
      ops::round_q8_group(x + i, count, group.q, group.d);
      for (int j = 0; j < count; ++j) x[i + j] = group.q[j] * group.d;
    }
  } else if (a16h) {
    ops::round_a16h(x, cols);
  } else if (nvfp4 && !vector_nvfp4) {
    for (int i = 0; i < cols; ++i) x[i] = float(ops::round_bf16(x[i]));
  }
}

namespace {
template <bool Quantized, bool Scalar>
void project_rows(const uint8_t* weights, size_t stride, int begin, int end, const ActRows& x, float* out,
                  size_t ld_out) {
  for (int r = 0; r < x.n_rows; ++r) {
    const size_t row = size_t(x.slots[r] / x.divisor);
    const float* values = x.values + row * x.cols;
    const ColdQ8* quant = x.quant + row * (x.cols / 32);
    for (int j = begin; j < end; ++j) {
      const auto* w = weights + size_t(j) * stride;
      float dot;
      if constexpr (Quantized)
        dot = x.quant_dot(w, quant, x.cols);
      else if constexpr (Scalar)
        dot = cold_dot_scalar(x.dtype, w, values, x.cols);
      else
        dot = x.float_dot(w, values, x.cols);
      out[size_t(r) * ld_out + j - begin] = x.gain * dot;
    }
  }
}
}  // namespace

void ColdMatrix::resolve(int simd, bool integer) {
  max_isa = cpu_isa_limit(simd);
  quant_dot = q8 && integer ? cold_q8_kernel(dtype, max_isa) : nullptr;
  float_dot = cold_float_kernel(dtype, cols, max_isa);
  kernel = quant_dot ? project_rows<true, false> : float_dot ? project_rows<false, false> : project_rows<false, true>;
}
}  // namespace flyx::cuda
