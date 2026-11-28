// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace flyx {

// Canonicalize signed zero so stored metadata and diagnostics agree.
inline float sampling_float(float value) { return value == 0.f ? 0.f : value; }

// An absent field inherits. An explicit zero can request greedy decoding or disable a filter.
struct SamplingFields {
  std::optional<float> temperature, top_p;
  std::optional<int> top_k;
  std::optional<float> presence_penalty, frequency_penalty;
};

struct ModelSampling {
  SamplingFields gguf, generation, common, thinking, non_thinking, thinking_coding;
  std::vector<std::string> warnings;
};

ModelSampling read_model_sampling(const nlohmann::json& target_header);

}  // namespace flyx
