// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "core/os/env.h"

namespace flyx::apps {

inline std::string api_key_from_env() {
  std::string key = os::getenv_str(env::kApiKey);
  if (key.empty()) return os::getenv_str(env::kKeyAlias);
  return key;
}

}  // namespace flyx::apps
