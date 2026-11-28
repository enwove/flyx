// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "core/session/session.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

using namespace flyx;
using flyx::test::check;
namespace {
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void fixture(const std::string& stem, bool parent) {
  const auto golden = nlohmann::json::parse(std::ifstream(stem + ".json"));
  const auto prompt = golden.at("prompt").get<std::vector<int32_t>>();
  const auto wanted = golden.at("logits").get<std::vector<std::vector<float>>>();
  std::vector<float> baseline;
  for (int tier : {FLYX_PLE_GPU, FLYX_PLE_HOST, FLYX_PLE_DISK, FLYX_PLE_HYBRID}) {
    if (!parent && tier != FLYX_PLE_GPU) continue;
    EngineOptions opt;
    opt.backend = "cpu_ref";
    opt.prefill_chunk = 32;
    opt.max_ctx = 128;
    opt.ple_placement = tier;
    opt.ple_hot_bytes = 1024;
    Engine engine;
    std::string err;
    require(engine.load(stem + ".gguf", opt, &err), err);
    const auto* api = engine.api();
    std::vector<float> trace;
    double maximum = 0;
    for (int chunk : {1, 5, 32}) {
      Session s(&engine);
      require(s.init(&err), err);
      flyx_gen_params gen{};
      for (int i = 0; i < int(prompt.size()); i += chunk) {
        int n = std::min(chunk, int(prompt.size()) - i);
        require(api->prefill(engine.prefill_plan(), s.seq(), prompt.data() + i, n, &gen) == 0, api->last_error());
        flyx_round_result result{};
        require(api->read(s.seq(), &result) == 0, api->last_error());
        std::vector<float> logits(wanted.back().size());
        require(api->logits(s.seq(), n - 1, logits.data()) == 0, api->last_error());
        const auto& ref = wanted[i + n - 1];
        for (size_t j = 0; j < logits.size(); ++j) maximum = std::max(maximum, std::abs(double(logits[j]) - ref[j]));
        if (chunk == 1)
          trace.insert(trace.end(), logits.begin(), logits.end());
        else
          check(std::equal(logits.begin(), logits.end(), trace.begin() + size_t(i + n - 1) * logits.size()),
                "chunk and step logits match exactly");
      }
    }
    std::printf("%s tier %d max absolute logits error %.8g\n", stem.c_str(), tier, maximum);
    check(maximum < 0.0002, "eager Transformers logits");
    if (baseline.empty())
      baseline = trace;
    else
      check(trace == baseline, "placement preserves every logit at every position");
  }
}
}  // namespace
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    fixture(std::string(argv[1]) + "/qwen4exp_coder", false);
    fixture(std::string(argv[1]) + "/qwen4exp_parent", true);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "%s\n", ex.what());
    return 1;
  }
  return flyx::test::finish("qwen4exp");
}
