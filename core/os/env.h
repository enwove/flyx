// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Single registry of environment variable names and typed readers over core/os/os.h.
#pragma once

#include <algorithm>
#include <cctype>
#include <charconv>
#include <optional>
#include <string>

#include "core/os/os.h"

namespace flyx::env {

// Engine & backend selection
inline constexpr const char* kBackend = "FLYX_BACKEND";
inline constexpr const char* kBackendDir = "FLYX_BACKEND_DIR";
inline constexpr const char* kDrafter = "FLYX_DRAFTER";
inline constexpr const char* kActivations = "FLYX_ACTIVATIONS";
inline constexpr const char* kBackendOptions = "FLYX_BACKEND_OPTIONS";
inline constexpr const char* kChatNative = "FLYX_CHAT_NATIVE";
inline constexpr const char* kLookup = "FLYX_LOOKUP";
inline constexpr const char* kDraftTrees = "FLYX_DRAFT_TREES";
inline constexpr const char* kAdaptive = "FLYX_ADAPTIVE";
inline constexpr const char* kLearnCosts = "FLYX_LEARN_COSTS";
inline constexpr const char* kWarmPrefill = "FLYX_WARM_PREFILL";

// Server & CLI
inline constexpr const char* kApiKey = "FLYX_API_KEY";
inline constexpr const char* kKeyAlias = "FLYX_KEY";
inline constexpr const char* kModelsDir = "FLYX_MODELS_DIR";
inline constexpr const char* kNoColor = "NO_COLOR";
inline constexpr const char* kColumns = "COLUMNS";

// Hugging Face Hub (compatible with huggingface_hub)
inline constexpr const char* kHfHome = "HF_HOME";
inline constexpr const char* kHfHubCache = "HF_HUB_CACHE";
inline constexpr const char* kHfToken = "HF_TOKEN";
inline constexpr const char* kHfEndpoint = "HF_ENDPOINT";
inline constexpr const char* kHfHubOffline = "HF_HUB_OFFLINE";

// OS / desktop / standard directories
inline constexpr const char* kHome = "HOME";
inline constexpr const char* kUserProfile = "USERPROFILE";
inline constexpr const char* kAppData = "APPDATA";
inline constexpr const char* kXdgConfigHome = "XDG_CONFIG_HOME";
inline constexpr const char* kXdgCacheHome = "XDG_CACHE_HOME";
inline constexpr const char* kSshConnection = "SSH_CONNECTION";
inline constexpr const char* kSshTty = "SSH_TTY";
inline constexpr const char* kDisplay = "DISPLAY";
inline constexpr const char* kWaylandDisplay = "WAYLAND_DISPLAY";

}  // namespace flyx::env

namespace flyx::os {

inline std::optional<bool> env_bool(const char* name) {
  std::string v = getenv_str(name);
  if (v.empty()) return std::nullopt;
  for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
  if (v == "0" || v == "false" || v == "no" || v == "off") return false;
  return std::nullopt;
}

inline int env_int(const char* name, int fallback, int lo, int hi) {
  const std::string v = getenv_str(name);
  if (v.empty()) return fallback;
  int parsed = 0;
  const auto [ptr, ec] = std::from_chars(v.data(), v.data() + v.size(), parsed);
  if (ec != std::errc{} || ptr != v.data() + v.size()) return fallback;
  return std::clamp(parsed, lo, hi);
}

inline std::string home_dir() {
  const std::string h = getenv_str(env::kHome);
  return h.empty() ? getenv_str(env::kUserProfile) : h;
}

inline std::string config_dir() {
#if defined(_WIN32)
  std::string base = getenv_str(env::kAppData);
  if (base.empty()) {
    const std::string h = home_dir();
    if (!h.empty()) base = h + "/AppData/Roaming";
  }
  return base;
#else
  std::string base = getenv_str(env::kXdgConfigHome);
  if (base.empty()) {
    const std::string h = home_dir();
    if (!h.empty()) base = h + "/.config";
  }
  return base;
#endif
}

}  // namespace flyx::os
