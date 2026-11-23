// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/numbers.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>

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

bool read_text_file(const std::string& path, std::string* out) {
  std::ifstream f(os::to_path(path), std::ios::binary);
  if (!f) return false;
  out->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return !f.bad();
}

}  // namespace flyx::apps
