// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "core/interfaces/chat.h"

namespace flyx::apps {

// Overlay template variables by key. The renderer, reply grammar and sampling defaults all read
// enable_thinking from ChatOptions, so a lower-priority copy in kwargs must never mask it.
inline bool merge_chat_kwargs(const interfaces::ojson& values, interfaces::ChatOptions* options, std::string* err) {
  if (!values.is_object()) return *err = "requires a valid JSON object", false;
  const auto thinking = values.find("enable_thinking");
  if (thinking != values.end()) {
    if (!thinking->is_boolean()) return *err = "enable_thinking must be true or false", false;
    options->enable_thinking = thinking->get<bool>();
  }
  for (auto it = values.begin(); it != values.end(); ++it)
    if (it != thinking) options->kwargs[it.key()] = it.value();
  return true;
}

}  // namespace flyx::apps
