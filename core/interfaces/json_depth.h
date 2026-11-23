// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Depth guard for untrusted JSON text before nlohmann::json parsing.
// nlohmann parses iteratively, but copying, dumping, and walking a parsed value
// recurse once per nesting level; rejecting deeper inputs bounds stack depth.
#pragma once

#include <string_view>

namespace flyx::interfaces {

inline constexpr int kMaxJsonDepth = 64;

// Whether no array or object in `s` is nested more than `max_depth` deep (brackets inside strings do not count).
inline bool json_depth_within(std::string_view s, int max_depth = kMaxJsonDepth) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (char c : s) {
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') {
      if (++depth > max_depth) return false;
    } else if (c == '}' || c == ']') {
      if (depth > 0) --depth;
    }
  }
  return true;
}

}  // namespace flyx::interfaces
