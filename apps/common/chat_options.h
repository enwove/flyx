// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "core/interfaces/chat.h"

namespace flyx::apps {

inline bool valid_chat_kwargs(const interfaces::ojson& values, std::string* err) {
  if (!values.is_object()) return *err = "requires a valid JSON object", false;
  const auto thinking = values.find("enable_thinking");
  if (thinking != values.end() && !thinking->is_boolean()) return *err = "enable_thinking must be true or false", false;
  return true;
}

// Keep request values distinct until the mode is known. A conflicting inherited effort would
// mask the format's own variable for that mode (Inkling-2 uses the effort "none" for thinking off).
inline interfaces::ChatOptions resolve_chat_options(const interfaces::ojson& defaults, const interfaces::ojson& request) {
  interfaces::ChatOptions options;
  options.kwargs = defaults;
  options.kwargs.update(request);
  const auto effort = options.kwargs.find("reasoning_effort");
  const bool effort_enables_thinking = effort == options.kwargs.end() || *effort != "none";
  const bool explicit_effort = request.contains("reasoning_effort");
  bool thinking = defaults.value("enable_thinking", effort_enables_thinking);
  if (explicit_effort && !effort_enables_thinking) thinking = false;
  options.enable_thinking = request.value("enable_thinking", thinking);
  if (!explicit_effort && effort != options.kwargs.end() && effort_enables_thinking != options.enable_thinking)
    options.kwargs.erase(effort);
  options.kwargs.erase("enable_thinking");
  return options;
}

}  // namespace flyx::apps
