// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "core/exec/generation.h"
#include "core/modelfile/sampling.h"

namespace flyx::apps {

enum class SamplingMode { Thinking, NonThinking, ThinkingCoding };
struct ChatSamplingOptions : SamplingFields {
  SamplingMode thinking_profile = SamplingMode::Thinking;
};

enum class SamplingOrigin { Fallback, ModelProfile, ModelMetadata, ModelRecommended, Cli, Request };

template <class T>
struct SamplingValue {
  T value;
  SamplingOrigin origin = SamplingOrigin::Fallback;
};

struct ResolvedSampling {
  SamplingValue<float> temperature{0.6f}, top_p{0.95f}, presence_penalty{0}, frequency_penalty{0};
  SamplingValue<int> top_k{20};

  void overlay(const SamplingFields& fields, SamplingOrigin origin);
  SamplingParams params() const;
  nlohmann::json describe() const;
};

struct SamplingDefaults {
  ResolvedSampling thinking, non_thinking, thinking_coding;
  std::optional<ResolvedSampling> thinking_vision, non_thinking_vision;
  SamplingMode thinking_profile = SamplingMode::Thinking;
  std::string profile_model;
  std::vector<std::string> warnings;

  const ResolvedSampling& get(SamplingMode mode, bool images = false) const;
  nlohmann::json describe() const;
  std::string log_text() const;
};

SamplingDefaults sampling_defaults(const nlohmann::json& header, const ChatSamplingOptions& cli, bool adapted = false);

}  // namespace flyx::apps
