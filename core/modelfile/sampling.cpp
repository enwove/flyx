// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/modelfile/sampling.h"

#include <climits>
#include <limits>
#include <string_view>

namespace flyx {
namespace {
using json = nlohmann::json;

bool disabled_sampler(std::string_view field, double value) {
  if (field == "min_p" || field == "penalty_freq" || field == "penalty_present" ||
      field == "mirostat" || field == "xtc_probability") return value == 0;
  if (field == "penalty_repeat" || field == "repetition_penalty" || field == "typical_p") return value == 1;
  return false;
}

void read_fields(const json& map, std::string path, SamplingFields& fields, std::vector<std::string>& warnings,
                 bool gguf = false) {
  if (!map.is_object()) {
    warnings.push_back(path + ": expected a sampling object");
    return;
  }
  for (const auto& [key, value] : map.items()) {
    std::string_view field = key;
    if (gguf) {
      if (!field.starts_with(path)) continue;
      field.remove_prefix(path.size());
    }
    const double v = value.is_number() ? value.get<double>() : std::numeric_limits<double>::quiet_NaN();
    if (disabled_sampler(field, v)) continue;
    if (gguf) {
      if (field == "temp") field = "temperature";
      else if (field != "top_k" && field != "top_p") field = {};
    }
    const std::string name = gguf ? key : path + "." + key;
    if (field == "temperature" && v >= 0 && v <= 100) fields.temperature = sampling_float(static_cast<float>(v));
    else if (field == "top_p" && v > 0 && v <= 1) fields.top_p = sampling_float(static_cast<float>(v));
    else if (field == "top_k" && value.is_number_integer() && v >= 0 && v <= INT_MAX) fields.top_k = int(v);
    else if (field == "presence_penalty" && v >= -2 && v <= 2) fields.presence_penalty = sampling_float(static_cast<float>(v));
    else if (field == "frequency_penalty" && v >= -2 && v <= 2) fields.frequency_penalty = sampling_float(static_cast<float>(v));
    else if (field == "do_sample" && value.is_boolean()) continue;
    else {
      bool supported = field == "temperature" || field == "top_k" || field == "top_p" || field == "do_sample" ||
                       field == "presence_penalty" || field == "frequency_penalty";
      warnings.push_back(name + (supported ? ": invalid value, ignored" : ": unsupported, ignored"));
    }
  }
  // The temperature sometimes accompanies a disabled HF sampler. Only this layer becomes greedy.
  if (!gguf && map.contains("do_sample") && map["do_sample"] == false) fields.temperature = 0.f;
}

const json* sampling_map(const json& h, const char* parent) {
  auto p = h.find(parent);
  if (p == h.end() || !p->is_object()) return nullptr;
  auto it = p->find("sampling");
  return it == p->end() ? nullptr : &*it;
}
}  // namespace

ModelSampling read_model_sampling(const json& h) {
  ModelSampling out;
  if (auto it = h.find("gguf"); it != h.end())
    read_fields(*it, "general.sampling.", out.gguf, out.warnings, true);
  if (auto map = sampling_map(h, "generation")) read_fields(*map, "generation.sampling", out.generation, out.warnings);
  if (auto modes = sampling_map(h, "recommended")) {
    if (!modes->is_object()) {
      out.warnings.emplace_back("recommended.sampling: expected an object");
      return out;
    }
    for (const auto& [mode, map] : modes->items()) {
      SamplingFields* fields = nullptr;
      if (mode == "default") fields = &out.common;
      else if (mode == "thinking") fields = &out.thinking;
      else if (mode == "non_thinking") fields = &out.non_thinking;
      else if (mode == "thinking_coding") fields = &out.thinking_coding;
      if (fields) read_fields(map, "recommended.sampling." + mode, *fields, out.warnings);
      else out.warnings.push_back("recommended.sampling." + mode + ": unknown mode, ignored");
    }
  }
  return out;
}
}  // namespace flyx
