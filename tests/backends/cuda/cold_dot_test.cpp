// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <random>
#include <vector>

#include "backends/cuda/runtime/cold_experts.h"
#include "backends/cuda/runtime/expert_order.h"
#include "ops/dtype/ref.h"
#include "tests/common/check.h"

int main() {
  using namespace flyx;
  {
    cuda::ExpertOrder o(8);
    o.counts = {1, 1, 1, 1, 8, 7, 0, 0};
    o.observe();
    std::vector<std::pair<int, int>> swaps;
    auto exchange = [&](int resident, int cold) { swaps.emplace_back(resident, cold); };
    test::check(o.update(4, o.scores.data(), exchange) == 1 && swaps.front() == std::pair{3, 4},
                "bounded promotion chooses the most frequent challenger and breaks ties by logical ID");
    test::check(o.slot[4] == 3 && o.id[3] == 4 && o.slot[3] == 4, "both sides of an expert exchange");
    o.counts.assign(8, 0);
    o.observe();
    test::check(o.scores[4] == 4 && o.update(4, o.scores.data(), exchange) == 1 && o.slot[5] < 4,
                "decayed counts retain a frequent expert for the next epoch");
    test::check(o.update(0, o.scores.data(), exchange) == 0 && o.update(8, o.scores.data(), exchange) == 0,
                "empty and full resident sets have no swaps");
  }
  std::mt19937 rng(91211);
  for (int dtype :
       {FLYX_DT_BF16, FLYX_DT_GGML_NVFP4, FLYX_DT_GGML_MXFP4, FLYX_DT_GGML_Q2_0, FLYX_DT_GGML_IQ1_S, FLYX_DT_GGML_IQ2_S,
        FLYX_DT_GGML_IQ2_XS, FLYX_DT_GGML_IQ2_XXS, FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_IQ4_NL, FLYX_DT_GGML_Q2_K,
        FLYX_DT_GGML_Q3_K, FLYX_DT_GGML_Q4_K, FLYX_DT_GGML_Q5_K, FLYX_DT_GGML_Q6_K, FLYX_DT_GGML_Q8_0}) {
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
        if (dtype == FLYX_DT_GGML_Q4_K || dtype == FLYX_DT_GGML_Q5_K) data[i + 3] = 48;
        if (dtype == FLYX_DT_GGML_Q3_K || dtype == FLYX_DT_GGML_Q6_K) data[i + bytes - 1] = 52;
        if (dtype == FLYX_DT_GGML_Q2_K) {
          data[i + bytes - 3] = 52;
          data[i + bytes - 1] = 48;
        }
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
    std::vector<cuda::ColdQ8> quant(cols / 32);
    cuda::prepare_cold_q8(x.data(), quant.data(), cols);
    double expected_q8 = 0, norm_q8 = 0;
    for (int k = 0; k < cols; ++k) {
      const auto& group = quant[k / 32];
      const double term = ops::load(&view, k) * (double(group.d) * group.q[k % 32]);
      expected_q8 += term;
      norm_q8 += std::fabs(term);
    }
    for (int isa : {256, 512})
      if (auto dot = cuda::cold_q8_kernel(dtype, isa))
        test::check(std::fabs(dot(data.data(), quant.data(), cols) - expected_q8) <= 2e-6 * norm_q8,
                    "integer cold dot vs float64 for dtype " + std::to_string(dtype) + " ISA " + std::to_string(isa));
    for (auto fn : {cuda::cold_dot_scalar, cuda::cold_dot}) {
      test::check(std::fabs(fn(dtype, data.data(), x.data(), cols) - expected) <= 2e-6 * norm,
                  "compressed CPU GEMV vs float64 for dtype " + std::to_string(dtype));
    }
  }
  return test::finish("cold_dot");
}
