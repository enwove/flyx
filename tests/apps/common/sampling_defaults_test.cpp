// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <limits>

#include "apps/common/sampling_defaults.h"
#include "tests/common/check.h"

using namespace flyx;
using namespace flyx::apps;
using json = nlohmann::json;
using Origin = SamplingOrigin;
using Mode = SamplingMode;

namespace {
json gguf(std::string arch, std::string name) {
  return {{"arch", arch}, {"gguf", {{"general.architecture", arch}, {"general.name", name},
                                    {"general.base_model.count", 1},
                                    {"general.base_model.0.repo_url", "https://huggingface.co/Qwen/" + name}}}};
}

void check_row(const ResolvedSampling& r, float t, float p, int k, float presence, const char* message) {
  check(r.temperature.value == t && r.top_p.value == p && r.top_k.value == k && r.presence_penalty.value == presence,
        message);
}

void identities() {
  auto h = gguf("qwen35", "Qwen3.5-0.8B");
  auto d = sampling_defaults(h, {});
  check_row(d.thinking, 1, .95f, 20, 1.5f, "0.8B general thinking");
  check_row(d.non_thinking, 1, 1, 20, 2, "0.8B text without thinking");
  check_row(d.get(Mode::NonThinking, true), .7f, .8f, 20, 1.5f, "0.8B image without thinking");
  check_row(d.get(Mode::Thinking, true), .6f, .95f, 20, 0, "0.8B image with thinking");
  check_row(d.thinking_coding, .6f, .95f, 20, 0, "explicit precise coding");
  check(d.thinking.top_k.origin == Origin::ModelProfile, "profile origin");
  check(sampling_defaults(h, {}, true).profile_model.empty(), "attached adapter changes model identity");
  h["gguf"]["general.name"] = "Domain-Qwen3.5-0.8B";
  check_row(sampling_defaults(h, {}).thinking, .6f, .95f, 20, 0, "a base model URL alone cannot select a profile");
  h = gguf("qwen35", "Qwen3.5-0.8B");
  h["gguf"]["general.finetune"] = "domain-adapted";
  check(sampling_defaults(h, {}).profile_model.empty(), "a declared fine-tune cannot inherit the publisher profile");
  h["gguf"].erase("general.finetune");
  h["gguf"]["general.source.repo_url"] = "https://huggingface.co/acme/tune";
  check(sampling_defaults(h, {}).profile_model.empty(), "a conflicting source wins over a base-model record");
  h["source"]["repo"] = "Qwen/Qwen3.5-0.8B";
  check(sampling_defaults(h, {}).profile_model.empty(), "wrapper cannot hide a conflicting GGUF source");
  h.erase("source");
  h["gguf"].erase("general.name");
  h["gguf"]["general.source.repo_url"] = "https://huggingface.co/Qwen/Qwen3.5-0.8B";
  check(sampling_defaults(h, {}).profile_model == "Qwen/Qwen3.5-0.8B", "direct official source identifies target");
  h = {{"arch", "qwen3"}, {"source", {{"repo", "Qwen/Qwen3-8B"}}}};
  h["generation"]["sampling"] = {{"temperature", .6}, {"top_p", .95}, {"top_k", 20}};
  d = sampling_defaults(h, {});
  check_row(d.thinking, .6f, .95f, 20, 0, "official Qwen3 thinking");
  check_row(d.non_thinking, .7f, .8f, 20, 0, "official Qwen3 non-thinking");
  for (auto arch : {"gpt-oss", "llama", "qwen35"}) {
    d = sampling_defaults({{"arch", arch}}, {});
    check_row(d.thinking, .6f, .95f, 20, 0, "unverified model keeps thinking truncation");
    check_row(d.non_thinking, .7f, .95f, 20, 0, "unverified model keeps non-thinking truncation");
    check(d.thinking.temperature.origin == Origin::Fallback, "fallback origin");
  }
}

void precedence() {
  auto h = gguf("qwen35", "Qwen3.5-27B");
  auto& g = h["gguf"];
  g["general.sampling.temp"] = .6;
  g["general.sampling.top_k"] = 20;
  g["general.sampling.top_p"] = .95;
  auto d = sampling_defaults(h, {});
  check_row(d.thinking, 1.f, .95f, 20, 1.5f, "27B general thinking takes the complete profile");
  check_row(d.thinking_coding, .6f, .95f, 20, 0, "27B precise coding");
  check_row(d.non_thinking, .7f, .8f, 20, 1.5f, "27B non-thinking");
  check(d.thinking.temperature.origin == Origin::ModelProfile, "verified profile wins over flat GGUF");
  h["generation"]["sampling"] = {{"temperature", .3}, {"do_sample", false}, {"top_k", 11}};
  d = sampling_defaults(h, {}, true);
  check_row(d.thinking, 0.f, .95f, 11, 0, "adapter inherits HF do_sample and top_k over flat GGUF");
  g["general.finetune"] = "domain";
  check_row(sampling_defaults(h, {}).non_thinking, 0.f, .95f, 11, 0, "fine-tune inherits its flat metadata");
  g.erase("general.finetune");
  auto& modes = h["recommended"]["sampling"];
  modes["default"] = {{"top_k", 0}, {"frequency_penalty", .5}, {"temperature", .4}};
  modes["thinking"] = {{"temperature", .8}, {"min_p", .08}};
  modes["non_thinking"] = {{"temperature", .9}};
  modes["thinking_coding"] = {{"temperature", .25}, {"presence_penalty", 0}};
  d = sampling_defaults(h, {});
  check_row(d.thinking, .8f, .95f, 0, 1.5f, "thinking recommendation and shared explicit zero");
  check_row(d.non_thinking, .9f, .8f, 0, 1.5f, "non-thinking recommendation");
  check_row(d.thinking_coding, .25f, .95f, 0, 0, "coding recommendation");
  check(d.thinking.frequency_penalty.value == .5f && d.thinking.frequency_penalty.origin == Origin::ModelRecommended &&
        d.warnings.size() == 1, "shared recommendation and unsupported setting diagnostic");
  ChatSamplingOptions cli;
  cli.thinking_profile = Mode::ThinkingCoding;
  auto coding = sampling_defaults(h, cli);
  check_row(coding.get(Mode::Thinking), .25f, .95f, 0, 0, "operator selects curated coding row");
  check_row(coding.get(Mode::NonThinking), .9f, .8f, 0, 1.5f, "operator leaves non-thinking selection alone");
  cli.temperature = .5f;
  auto r = sampling_defaults(h, cli).thinking;
  check(r.temperature.value == .5f && r.temperature.origin == Origin::Cli, "explicit CLI wins");
  SamplingFields request;
  request.temperature = -0.f;
  r.overlay(request, Origin::Request);
  check(r.params().temperature == 0 && !std::signbit(r.params().temperature) &&
        r.describe()["temperature"]["origin"] == "request", "explicit request wins");
  check(d.thinking.temperature.value == .8f, "request left the loaded defaults untouched");
}

void other_metadata() {
  auto h = gguf("qwen35", "Qwen3.8-27B");
  auto& g = h["gguf"];
  g["general.sampling.temp"] = 1;
  g["general.sampling.top_k"] = 20;
  g["general.sampling.top_p"] = .95;
  check_row(sampling_defaults(h, {}).thinking, 1, .95f, 20, 0, "Qwen3.8 has its own metadata on qwen35");
  h["arch"] = "inkling2_loop";
  g["general.name"] = "Inkling-2-Mini";
  g["general.sampling.top_k"] = 0;
  g["general.sampling.top_p"] = 1;
  g["general.sampling.min_p"] = 0;
  g["general.sampling.penalty_repeat"] = 1;
  h["generation"]["sampling"] = {{"repetition_penalty", 1}, {"typical_p", 1}, {"presence_penalty", -0.0}};
  check(sampling_defaults(h, {}).warnings.empty(), "disabled unsupported controls load quietly");
  check(!std::signbit(*read_model_sampling(h).generation.presence_penalty), "file stores positive zero");
  g["general.sampling.penalty_repeat"] = 1.05;
  g["general.sampling.penalty_last_n"] = 64;
  auto d = sampling_defaults(h, {});
  check_row(d.thinking, 1, 1, 0, 0, "Inkling disables top-k");
  check(d.warnings.size() == 2 && d.thinking.top_k.origin == Origin::ModelMetadata, "inactive fields are diagnostic");
  h["generation"]["sampling"] = {{"temperature", std::numeric_limits<double>::quiet_NaN()}, {"top_k", 2.5},
                                  {"top_p", 7}, {"do_sample", "false"}};
  h["recommended"]["sampling"]["default"] = "malformed";
  d = sampling_defaults(h, {});
  check_row(d.non_thinking, 1, 1, 0, 0, "invalid optional fields inherit lower values");
  check(d.warnings.size() == 7, "each invalid field or map produces one load warning");
}
}  // namespace

int main() {
  identities();
  precedence();
  other_metadata();
  return flyx::test::finish("sampling_defaults_test");
}
