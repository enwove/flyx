// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/row_pool.h"
#include "ops/activation/ref.h"
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
    round_cold_q8(x, cols, max_isa);
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
  shared_kernel = nullptr;
  quant_dot = q8 && integer ? cold_q8_kernel(dtype, max_isa, &shared_kernel) : nullptr;
  float_dot = cold_float_kernel(dtype, cols, max_isa, quant_dot ? nullptr : &shared_kernel);
  kernel = quant_dot ? project_rows<true, false> : float_dot ? project_rows<false, false> : project_rows<false, true>;
}
void ColdMatrix::project(RowPool& pool, const Groups& groups, bool gate, float beta, float linear_beta,
                         const float* values, const ColdQ8* quant, float* output) const {
  const int width = gate ? rows / 2 : rows;
  constexpr int tile = 32;
  constexpr size_t rows_per_line = 64 / sizeof(float);
  const size_t lines = (size_t(width) + rows_per_line - 1) / rows_per_line;
  pool.run(size_mul(size_t(groups.count), lines), [&](size_t begin, size_t end) {
    float lo[ColdShape::chunk * tile], hi[ColdShape::chunk * tile];
    // Each worker owns a contiguous range of projection rows. Bind its activations once per expert.
    while (begin < end) {
      const size_t group = begin / lines;
      const size_t stop_line = std::min(end, (group + 1) * lines);
      const int j0 = int((begin % lines) * rows_per_line);
      const int j1 = int(std::min(size_t(width), (stop_line - group * lines) * rows_per_line));
      const int expert = groups.ids[groups.slots[groups.offsets[group]]] - first;
      const auto* w = data.data() + size_t(expert) * rows * stride;
      for (int offset = groups.offsets[group]; offset < groups.offsets[group + 1];) {
        const int count = std::min(ColdShape::chunk, groups.offsets[group + 1] - offset);
        ActRows x{groups.slots + offset,
                  count,
                  gate ? groups.top_k : 1,
                  cols,
                  dtype,
                  values,
                  quant,
                  scales.empty() ? 1.f : scales[expert],
                  float_dot,
                  quant_dot};
        const auto project_tile = count > 1 && shared_kernel ? shared_kernel : kernel;
        if (!gate && count == 1) {
          kernel(w, stride, j0, j1, x, output + size_t(*x.slots) * width + j0, width);
        } else {
          for (int j = j0; j < j1; j += tile) {
            const int stop = std::min(j + tile, j1);
            project_tile(w, stride, j, stop, x, lo, tile);
            if (gate) project_tile(w, stride, width + j, width + stop, x, hi, tile);
            for (int r = 0; r < count; ++r) {
              const size_t at = size_t(x.slots[r]) * width + j;
              if (gate) {
                for (int c = 0; c < stop - j; ++c) {
                  const float a = lo[r * tile + c], b = hi[r * tile + c];
                  output[at + c] = float(beta > 0 ? ops::situ(a, b, beta, linear_beta) : ops::silu(a) * b);
                }
              } else {
                std::copy_n(lo + r * tile, stop - j, output + at);
              }
            }
          }
        }
        offset += count;
      }
      begin = stop_line;
    }
  });
}
}  // namespace flyx::cuda
