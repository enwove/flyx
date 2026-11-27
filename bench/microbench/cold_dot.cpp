// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/row_pool.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <random>
#include <string_view>

namespace flyx::bench {
namespace {
int positive(const char* arg, int maximum) {
  const std::string_view text(arg);
  int n = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
  return error == std::errc{} && end == text.data() + text.size() && n > 0 && n <= maximum ? n : 0;
}
} // namespace

int cold_dot_bench(int argc, char** argv) {
  using namespace flyx::cuda;
  if (argc != 4) {
    std::fprintf(stderr, "usage: cuda_bench cold_dot COLUMNS PROJECTION_ROWS THREADS ACTIVATION_ROWS\n");
    return 2;
  }
  const int cols = positive(argv[0], 65536), rows = positive(argv[1], 65536);
  const int threads = positive(argv[2], 256), count = positive(argv[3], 8);
  if (!cols || cols % 32 || !rows || !threads || !count) return 2;
  RowPool pool(threads, true);
  std::mt19937 rng(817);
  std::vector<float> input(size_t(cols) * count), output(size_t(rows) * count);
  for (auto& x : input) x = float(int(rng() % 1024) - 512) / 512.f;
  std::vector<ColdQ8> q(input.size() / 32);
  prepare_cold_q8(input.data(), q.data(), int(input.size()));
  int ids[] = {0, 1, 2, 3, 4, 5, 6, 7};
  struct Type {
    int dtype;
    const char* name;
  };
  const Type types[] = {{FLYX_DT_GGML_Q2_K, "Q2_K"},   {FLYX_DT_GGML_Q3_K, "Q3_K"},   {FLYX_DT_GGML_IQ3_XXS, "IQ3_XXS"},
                        {FLYX_DT_GGML_Q4_K, "Q4_K"},   {FLYX_DT_GGML_Q6_K, "Q6_K"},   {FLYX_DT_GGML_IQ2_S, "IQ2_S"},
                        {FLYX_DT_GGML_NVFP4, "NVFP4"}, {FLYX_DT_GGML_MXFP4, "MXFP4"}, {FLYX_DT_GGML_Q2_0, "Q2_0"}};
  std::puts("dtype,threads,activation_rows,columns,projection_rows,compressed_GB_per_s,checksum");
  for (const auto& type : types) {
    const size_t block = flyx_block_bytes(type.dtype);
    if (cols % flyx_block_values(type.dtype)) continue;
    const size_t stride = size_t(cols / flyx_block_values(type.dtype)) * block;
    const size_t expert = size_t(rows) * stride;
    const size_t banks = std::max<size_t>(1, (256u << 20) / expert);
    std::vector<uint8_t> weights(banks * expert);
    // Rotate a deterministic bank through a footprint larger than the test machine's last-level cache.
    for (size_t j = 0; j < expert; ++j) weights[j] = uint8_t(rng());
    for (size_t j = 0; j < expert; j += block) {
      // Keep per-block scales finite in each quantization layout.
      weights[j + 1] = weights[j + 3] = weights[j + block - 1] = weights[j + block - 3] = 0x30;
      if (type.dtype == FLYX_DT_GGML_MXFP4) weights[j] = 126;
      if (type.dtype == FLYX_DT_GGML_NVFP4) std::fill_n(weights.data() + j, 4, uint8_t(0x38));
    }
    for (size_t n = 1; n < banks; ++n) std::copy_n(weights.data(), expert, weights.data() + n * expert);
    ExpertKernel shared = nullptr;
    const auto qdot = cold_q8_kernel(type.dtype, CpuIsa::avx512, &shared);
    const auto fdot = cold_float_kernel(type.dtype, cols, CpuIsa::avx512, qdot ? nullptr : &shared);
    ActRows x{ids, count, 1, cols, type.dtype, input.data(), q.data(), 1.f, fdot, qdot};
    auto pass = [&](size_t bank) {
      auto* w = weights.data() + bank * expert;
      pool.run(rows, [&](size_t first, size_t last) {
        if (shared && count > 1) {
          shared(w, stride, int(first), int(last), x, output.data() + first, rows);
          return;
        }
        for (int r = 0; r < count; ++r)
          for (size_t j = first; j < last; ++j) {
            const auto* row = w + j * stride;
            const auto* values = input.data() + r * cols;
            output[size_t(r) * rows + j] = qdot   ? qdot(row, q.data() + r * (cols / 32), cols)
                                           : fdot ? fdot(row, values, cols)
                                                  : cold_dot_scalar(type.dtype, row, values, cols);
          }
      });
    };
    pass(0);
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < banks; ++i) pass(i);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%s,%d,%d,%d,%d,%.3f,%.9g\n", type.name, threads, count, cols, rows, weights.size() / sec / 1e9,
                double(output.back()));
  }
  return 0;
}
} // namespace flyx::bench
