// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace flyx::os {

// The CLI, C API and HTTP use UTF-8. A native filesystem path uses UTF-16 on Windows.
inline std::filesystem::path to_path(std::string_view utf8) {
  return std::filesystem::path(std::u8string(utf8.begin(), utf8.end()));
}

inline std::string from_path(const std::filesystem::path& path) {
  const auto utf8 = path.u8string();
  return std::string(utf8.begin(), utf8.end());
}

}  // namespace flyx::os
