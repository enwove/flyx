// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/sampling_defaults.h"

#include <algorithm>
#include <iterator>
#include <string_view>

namespace flyx::apps {
namespace {
using json = nlohmann::json;
using Origin = SamplingOrigin;

std::string str(const json& obj, const char* key) {
  auto it = obj.find(key);
  return it != obj.end() && it->is_string() ? it->get<std::string>() : "";
}

// Only the exact publisher checkpoint is evidence for its card. A GGUF conversion may
// identify that checkpoint by its own name and a matching upstream URL (as Unsloth does).
std::string profile_model(const json& h) {
  if (h.contains("lora") && !h["lora"].is_null()) return {};
  std::string repo;
  if (auto s = h.find("source"); s != h.end()) repo = str(*s, "repo");
  auto g = h.find("gguf");
  if (g != h.end() && !str(*g, "general.finetune").empty()) return {};
  if (g != h.end()) {
    const auto name = str(*g, "general.name");
    const auto source = str(*g, "general.source.repo_url");
    constexpr std::string_view hub = "https://huggingface.co/";
    if (!name.empty() && !repo.empty() && repo != "Qwen/" + name) return {};
    if (!source.empty()) {
      if (!source.starts_with(hub) || (!repo.empty() && source.substr(hub.size()) != repo)) return {};
      repo = source.substr(hub.size());
      if (!name.empty() && repo != "Qwen/" + name) return {};
    } else if (repo.empty()) {
      if (name.empty() || !g->contains("general.base_model.count") || (*g)["general.base_model.count"] != 1) return {};
      repo = "Qwen/" + name;
      if (str(*g, "general.base_model.0.repo_url") != std::string(hub) + repo) return {};
    }
  }
  static constexpr std::string_view q35[] = {
      "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
      "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-122B-A10B", "Qwen/Qwen3.5-397B-A17B"};
  static constexpr std::string_view q3[] = {
      "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
      "Qwen/Qwen3-32B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B"};
  const std::string arch = str(h, "arch");
  if ((arch == "qwen35" || arch == "qwen35moe") && std::find(std::begin(q35), std::end(q35), repo) != std::end(q35))
    return repo;
  if ((arch == "qwen3" || arch == "qwen3moe") && std::find(std::begin(q3), std::end(q3), repo) != std::end(q3))
    return repo;
  return {};
}

SamplingFields profile(const std::string& model, SamplingMode mode, bool images) {
  if (model.empty()) return {};
  const bool q35 = model.starts_with("Qwen/Qwen3.5-");
  const bool is_08b = model == "Qwen/Qwen3.5-0.8B";
  if (mode == SamplingMode::NonThinking) {
    if (is_08b && !images) return {1.f, 1.f, 20, 2.f, {}};
    return {0.7f, 0.8f, 20, q35 ? 1.5f : 0.f, {}};
  }
  if (!q35 || mode == SamplingMode::ThinkingCoding || (is_08b && images)) return {0.6f, 0.95f, 20, 0.f, {}};
  return {1.f, 0.95f, 20, 1.5f, {}};
}

template <class T>
json describe_field(const SamplingValue<T>& field) {
  static constexpr const char* names[] = {"fallback", "model_profile", "model_metadata", "model_recommended", "cli", "request"};
  return {{"value", field.value}, {"origin", names[static_cast<int>(field.origin)]}};
}
}  // namespace

void ResolvedSampling::overlay(const SamplingFields& f, SamplingOrigin origin) {
  if (f.temperature) temperature = {sampling_float(*f.temperature), origin};
  if (f.top_k) top_k = {*f.top_k, origin};
  if (f.top_p) top_p = {*f.top_p == 0.f ? 1.f : *f.top_p, origin};
  if (f.presence_penalty) presence_penalty = {sampling_float(*f.presence_penalty), origin};
  if (f.frequency_penalty) frequency_penalty = {sampling_float(*f.frequency_penalty), origin};
}

SamplingParams ResolvedSampling::params() const {
  return {temperature.value, top_p.value, top_k.value, 0, presence_penalty.value, frequency_penalty.value};
}

json ResolvedSampling::describe() const {
  return {{"temperature", describe_field(temperature)}, {"top_k", describe_field(top_k)}, {"top_p", describe_field(top_p)},
          {"presence_penalty", describe_field(presence_penalty)}, {"frequency_penalty", describe_field(frequency_penalty)}};
}

const ResolvedSampling& SamplingDefaults::get(SamplingMode mode, bool images) const {
  if (images && mode == SamplingMode::Thinking && thinking_vision) return *thinking_vision;
  if (images && mode == SamplingMode::NonThinking && non_thinking_vision) return *non_thinking_vision;
  if (mode == SamplingMode::ThinkingCoding) return thinking_coding;
  return mode == SamplingMode::Thinking ? thinking : non_thinking;
}

json SamplingDefaults::describe() const {
  json out = {{"thinking_profile", thinking_profile == SamplingMode::ThinkingCoding ? "thinking_coding" : "thinking"},
          {"profile_model", profile_model.empty() ? json(nullptr) : json(profile_model)}, {"warnings", warnings},
          {"thinking", thinking.describe()}, {"non_thinking", non_thinking.describe()}, {"thinking_coding", thinking_coding.describe()}};
  if (thinking_vision) out["thinking_vision"] = thinking_vision->describe();
  if (non_thinking_vision) out["non_thinking_vision"] = non_thinking_vision->describe();
  return out;
}

std::string SamplingDefaults::log_text() const {
  // An unknown GGUF key can include arbitrary bytes. A diagnostic must still be printable.
  return describe().dump(-1, ' ', false, json::error_handler_t::replace);
}

SamplingDefaults sampling_defaults(const json& header, const ChatSamplingOptions& cli, bool adapted) {
  SamplingDefaults out;
  auto file = read_model_sampling(header);
  if (!adapted) out.profile_model = profile_model(header);
  out.thinking_profile = cli.thinking_profile;
  out.warnings = std::move(file.warnings);
  auto resolve = [&](SamplingMode mode, const SamplingFields& recommendation, bool images = false) {
    ResolvedSampling r;
    if (mode == SamplingMode::NonThinking) r.temperature.value = 0.7f;
    r.overlay(file.gguf, Origin::ModelMetadata);
    r.overlay(file.generation, Origin::ModelMetadata);
    r.overlay(profile(out.profile_model, mode, images), Origin::ModelProfile);
    r.overlay(file.common, Origin::ModelRecommended);
    r.overlay(recommendation, Origin::ModelRecommended);
    r.overlay(cli, Origin::Cli);
    return r;
  };
  out.non_thinking = resolve(SamplingMode::NonThinking, file.non_thinking);
  out.thinking_coding = resolve(SamplingMode::ThinkingCoding, file.thinking_coding);
  const bool coding = out.thinking_profile == SamplingMode::ThinkingCoding;
  out.thinking = coding ? out.thinking_coding : resolve(SamplingMode::Thinking, file.thinking);
  if (out.profile_model == "Qwen/Qwen3.5-0.8B") {
    out.thinking_vision = coding ? out.thinking_coding : resolve(SamplingMode::Thinking, file.thinking, true);
    out.non_thinking_vision = resolve(SamplingMode::NonThinking, file.non_thinking, true);
  }
  return out;
}
}  // namespace flyx::apps
