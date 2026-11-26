// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The float64 reference of GGML block weights (ops/dtype/ggml.h) against the
// gguf Python package's decoder: random blocks of every type, every value
// (tests/core/modelfile/make_gguf.py wrote them with scales short enough for
// the package's float32 arithmetic to be exact).  And binary16, all of it.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "ops/dtype/ref.h"
#include "tests/common/check.h"

using namespace flyx;
using json = nlohmann::json;

namespace {

const std::map<std::string, int> kTypes = {
    {"Q4_0", FLYX_DT_GGML_Q4_0},       {"Q4_1", FLYX_DT_GGML_Q4_1},     {"Q5_0", FLYX_DT_GGML_Q5_0},
    {"Q5_1", FLYX_DT_GGML_Q5_1},       {"Q8_0", FLYX_DT_GGML_Q8_0},     {"Q2_K", FLYX_DT_GGML_Q2_K},
    {"Q3_K", FLYX_DT_GGML_Q3_K},       {"Q4_K", FLYX_DT_GGML_Q4_K},     {"Q5_K", FLYX_DT_GGML_Q5_K},
    {"Q6_K", FLYX_DT_GGML_Q6_K},       {"IQ2_XXS", FLYX_DT_GGML_IQ2_XXS}, {"IQ2_XS", FLYX_DT_GGML_IQ2_XS},
    {"IQ2_S", FLYX_DT_GGML_IQ2_S},     {"IQ3_XXS", FLYX_DT_GGML_IQ3_XXS}, {"IQ3_S", FLYX_DT_GGML_IQ3_S},
    {"IQ1_S", FLYX_DT_GGML_IQ1_S},     {"IQ1_M", FLYX_DT_GGML_IQ1_M},   {"IQ4_NL", FLYX_DT_GGML_IQ4_NL},
    {"IQ4_XS", FLYX_DT_GGML_IQ4_XS}, {"NVFP4", FLYX_DT_GGML_NVFP4}};

void blocks(const std::string& path) {
  std::ifstream f(path);
  size_t seen = 0;
  for (const auto& c : json::parse(f)) {
    const std::string type = c["type"];
    const auto data = c["bytes"].get<std::vector<uint8_t>>();
    const auto expected = c["expected"].get<std::vector<double>>();
    flyx_tensor_view v{};
    v.dtype = kTypes.at(type);
    const size_t per = size_t(flyx_block_values(v.dtype)), bytes = size_t(flyx_block_bytes(v.dtype));
    check(per == c["block"].get<size_t>() && data.size() % bytes == 0 && expected.size() == data.size() / bytes * per,
          type + ": block geometry");
    v.ndim = 2;
    v.shape[0] = int64_t(data.size() / bytes);
    v.shape[1] = int64_t(per);
    v.data = data.data();
    v.nbytes = data.size();
    size_t bad = 0, nonzero = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
      bad += ops::load(&v, i) != expected[i];
      nonzero += expected[i] != 0;
    }
    check(bad == 0 && nonzero > expected.size() / 4,
          type + ": " + std::to_string(expected.size()) + " values equal the gguf package's" +
              (bad ? " (" + std::to_string(bad) + " differ)" : ""));
    ++seen;
  }
  check(seen == kTypes.size(), "every block type has a case");
}

void halves() {
  using ops::ggml::half;
  check(half(0x0001) == std::ldexp(1.0, -24) && half(0x8000) == 0 && std::signbit(half(0x8000)) &&
            std::isinf(half(0x7c00)) && std::isnan(half(0x7fff)),
        "binary16 subnormal, signed zero, infinity and NaN");
  bool all = true;
  for (uint32_t code = 0; code <= 65535; ++code) {
    const int exponent = (code >> 10) & 31, fraction = code & 1023;
    double want = exponent == 0    ? std::ldexp(double(fraction), -24)
                  : exponent == 31 ? (fraction ? NAN : INFINITY)
                                   : std::ldexp(1 + fraction / 1024.0, exponent - 15);
    if (code & 0x8000) want = -want;
    const double got = half(uint16_t(code));
    all &= std::isnan(want) ? std::isnan(got) : got == want && std::signbit(got) == std::signbit(want);
  }
  check(all, "all 65536 binary16 encodings match the definition");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc != 2) return 2;
  blocks(std::string(argv[1]) + "/gguf_quant_cases.json");
  halves();
  return flyx::test::finish("ggml_test");
}
