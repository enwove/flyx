// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <random>
#include <vector>

#include "backends/cuda/runtime/cold_experts.h"
#include "ops/dtype/ref.h"
#include "tests/common/check.h"

int main() {
  using namespace flyx;
  std::mt19937 rng(91211);
  for (int dtype :
       {FLYX_DT_BF16, FLYX_DT_GGML_NVFP4, FLYX_DT_GGML_MXFP4, FLYX_DT_GGML_Q2_0, FLYX_DT_GGML_IQ1_S, FLYX_DT_GGML_IQ2_S,
        FLYX_DT_GGML_IQ2_XS, FLYX_DT_GGML_IQ2_XXS, FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_IQ4_NL, FLYX_DT_GGML_Q4_K}) {
    int per = dtype == FLYX_DT_BF16 ? 1 : flyx_block_values(dtype);
    int bytes = dtype == FLYX_DT_BF16 ? 2 : flyx_block_bytes(dtype);
    constexpr int cols = 512;
    std::vector<uint8_t> data(cols / per * bytes);
    for (auto& b : data) b = uint8_t(rng());
    for (size_t i = 0; i < data.size(); i += bytes) {
      if (dtype == FLYX_DT_GGML_NVFP4) {
        for (int g = 0; g < 4; ++g) data[i + g] &= 127;
      } else if (dtype == FLYX_DT_GGML_MXFP4)
        data[i] = uint8_t(120 + rng() % 12);
      else {
        data[i + 1] = dtype == FLYX_DT_BF16 ? 62 : 52;
        if (dtype == FLYX_DT_GGML_Q4_K) data[i + 3] = 48;
      }
    }
    flyx_tensor_view view{};
    view.dtype = dtype;
    view.data = data.data();
    std::vector<float> x(cols);
    double expected = 0, norm = 0;
    for (int k = 0; k < cols; ++k) {
      x[k] = float(int(rng() % 4097) - 2048) / 2048.f;
      const double term = ops::load(&view, k) * x[k];
      expected += term;
      norm += std::fabs(term);
    }
    for (auto fn : {cuda::cold_dot_scalar, cuda::cold_dot}) {
      test::check(std::fabs(fn(dtype, data.data(), x.data(), cols) - expected) <= 2e-6 * norm,
                  "compressed CPU GEMV vs float64 for dtype " + std::to_string(dtype));
    }
  }
  return test::finish("cold_dot");
}
