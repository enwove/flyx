// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The engine's command-line options, for every command that loads an engine (flyx run, chat, bench, serve): one
// parser and one help text, so the commands cannot drift apart.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "apps/common/chat_options.h"
#include "core/exec/engine.h"
#include "core/exec/generation.h"

namespace flyx::apps {

interfaces::ojson chat_template_kwargs_flag(const std::string& value);

// Sampling flag overrides given on the command line (only present fields overwrite a target SamplingParams).
struct SamplingOverrides {
  std::optional<float> temperature, top_p, presence_penalty, frequency_penalty, accept_min_p, accept_typical;
  std::optional<int> top_k;
  std::optional<uint64_t> seed;

  void apply(SamplingParams* sp) const;
};

// Applies `arg` when it is an engine option, taking its value from `value()`; false when it is not one.
bool engine_flag(const std::string& arg, const std::function<std::string()>& value, EngineOptions* eo);

// Applies `arg` when it is a sampling option (--temperature, --top-p, --top-k, --presence-penalty,
// --frequency-penalty; plus --seed, --accept-min-p, --accept-typical when `cli_only` is true).
bool sampling_flag(const std::string& arg, const std::function<std::string()>& value, SamplingOverrides* out,
                   bool cli_only = false);

// The help of those options, a line or two each as the programs print them; `ctx`: the program's default context.
std::string engine_flags_help(int ctx);

}  // namespace flyx::apps
