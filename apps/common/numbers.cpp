// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/numbers.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include "apps/common/defaults.h"
#include "core/os/path.h"

namespace flyx::apps {

bool parse_whole(const std::string& text, long long lo, long long hi, long long* out) {
  long long v = 0;
  const char* end = text.data() + text.size();
  const auto r = std::from_chars(text.data(), end, v, 10);
  if (text.empty() || r.ec != std::errc() || r.ptr != end || v < lo || v > hi) return false;
  *out = v;
  return true;
}

bool parse_number(const std::string& text, double lo, double hi, double* out) {
  double v = 0;
  const char* end = text.data() + text.size();
  const auto r = std::from_chars(text.data(), end, v, std::chars_format::general);
  if (text.empty() || r.ec != std::errc() || r.ptr != end || !std::isfinite(v) || v < lo || v > hi) return false;
  *out = v;
  return true;
}

bool parse_seed(const std::string& text, uint64_t* out) {
  uint64_t v = 0;
  const char* end = text.data() + text.size();
  const auto r = std::from_chars(text.data(), end, v, 10);
  if (text.empty() || r.ec != std::errc() || r.ptr != end) return false;
  *out = v;
  return true;
}

bool gb_to_bytes(double gb, int64_t* bytes_out) {
  if (!std::isfinite(gb) || gb < 0.0 || gb > kMaxGb) return false;
  *bytes_out = int64_t(gb * 1e9);
  return true;
}

bool parse_gb_or_auto(std::string_view text, int64_t* bytes_out) {
  if (text == "auto") {
    *bytes_out = -1;
    return true;
  }
  double gb = 0.0;
  return parse_number(std::string(text), 0.0, kMaxGb, &gb) && gb_to_bytes(gb, bytes_out);
}

int whole_flag(const char* flag, const std::string& text, int lo, int hi) {
  long long v = 0;
  if (!parse_whole(text, lo, hi, &v)) {
    std::fprintf(stderr, "error: %s %s: takes a whole number from %d to %d\n", flag, text.c_str(), lo, hi);
    std::exit(2);
  }
  return static_cast<int>(v);
}

double number_flag(const char* flag, const std::string& text, double lo, double hi) {
  double v = 0;
  if (!parse_number(text, lo, hi, &v)) {
    std::fprintf(stderr, "error: %s %s: takes a number from %g to %g\n", flag, text.c_str(), lo, hi);
    std::exit(2);
  }
  return v;
}

uint64_t seed_flag(const char* flag, const std::string& text) {
  uint64_t v = 0;
  if (!parse_seed(text, &v)) {
    std::fprintf(stderr, "error: %s %s: takes a whole number from 0 to 18446744073709551615\n", flag, text.c_str());
    std::exit(2);
  }
  return v;
}

int64_t gb_or_auto_flag(const char* flag, const std::string& text) {
  int64_t bytes = -1;
  if (!parse_gb_or_auto(text, &bytes)) {
    std::fprintf(stderr, "error: %s %s: takes GB (0 to %g) or \"auto\"\n", flag, text.c_str(), kMaxGb);
    std::exit(2);
  }
  return bytes;
}

bool read_text_file(const std::string& path, std::string* out) {
  std::ifstream f(os::to_path(path), std::ios::binary);
  if (!f) return false;
  out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return !f.bad();
}

}  // namespace flyx::apps
