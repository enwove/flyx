// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Typed configuration option enums and canonical string parsers shared by core, models, and apps.
#pragma once

#include <charconv>
#include <string>
#include <string_view>

namespace flyx::interfaces {

enum class DrafterMode { kAuto, kMtp, kDFlash2, kDv, kDSpark, kAll, kNone };

inline bool parse_drafter_mode(std::string_view value, DrafterMode* out) {
  if (value.empty()) {
    *out = DrafterMode::kAuto;
    return true;
  }
  if (value == "mtp") {
    *out = DrafterMode::kMtp;
    return true;
  }
  if (value == "dflash2") {
    *out = DrafterMode::kDFlash2;
    return true;
  }
  if (value == "dv") {
    *out = DrafterMode::kDv;
    return true;
  }
  if (value == "dspark") {
    *out = DrafterMode::kDSpark;
    return true;
  }
  if (value == "all") {
    *out = DrafterMode::kAll;
    return true;
  }
  if (value == "none") {
    *out = DrafterMode::kNone;
    return true;
  }
  return false;
}

enum class KvMethodKind { kExact, kAuto, kStrata };

struct KvMethodSpec {
  KvMethodKind kind = KvMethodKind::kExact;
  int bits = 0;  // 2, 3, or 4 when kind != kExact
};

// Parses "exact", "strata[:BITS]", or "auto[:BITS]" (BITS in 2..4; default: 2).
inline bool parse_kv_method(std::string_view value, KvMethodSpec* out, std::string* err) {
  if (value == "exact") {
    *out = {KvMethodKind::kExact, 0};
    return true;
  }
  const size_t colon = value.find(':');
  const std::string_view method = value.substr(0, colon);
  const bool strata = method == "strata";
  if (!strata && method != "auto") {
    if (err) *err = "--kv-method " + std::string(value) + ": exact, strata[:BITS] or auto[:BITS]";
    return false;
  }
  int bits = 2;
  if (colon != std::string_view::npos) {
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data() + colon + 1, end, bits);
    if (parsed.ec != std::errc{} || parsed.ptr != end || bits < 2 || bits > 4) {
      if (err) *err = "--kv-method " + std::string(value) + ": 2, 3 or 4 bits";
      return false;
    }
  }
  *out = {strata ? KvMethodKind::kStrata : KvMethodKind::kAuto, bits};
  return true;
}

}  // namespace flyx::interfaces
