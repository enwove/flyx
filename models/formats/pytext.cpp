// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/pytext.h"

#include <algorithm>
#include <cstdint>

namespace flyx::models::formats {
namespace {

// Length of the Unicode whitespace character (as Python's str.isspace sees
// it) starting at s[i], or 0.
size_t space_len(std::string_view s, size_t i) {
  unsigned char c = static_cast<unsigned char>(s[i]);
  if (c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F)) return 1;
  if (c < 0x80) return 0;
  uint32_t cp = 0;
  size_t n = 0;
  if ((c >> 5) == 6) cp = c & 0x1F, n = 2;
  else if ((c >> 4) == 14) cp = c & 0x0F, n = 3;
  else return 0;
  if (i + n > s.size()) return 0;
  for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  bool sp = cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
            cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
  return sp ? n : 0;
}

bool is_cont(unsigned char c) { return (c & 0xC0) == 0x80; }

bool append_plain_string(const std::string& s, std::string* out) {
  // Most tool schema keys and names need only the JSON quotes. Let the
  // serializer handle escaping, Unicode and invalid UTF-8 as before.
  if (std::any_of(s.begin(), s.end(), [](unsigned char c) {
        return c < 0x20 || c >= 0x7f || c == '"' || c == '\\';
      })) return false;
  *out += '"';
  *out += s;
  *out += '"';
  return true;
}

void dump(const ojson& j, std::string* out) {
  switch (j.type()) {
    case ojson::value_t::object: {
      if (j.empty()) {
        *out += "{}";
        return;
      }
      *out += '{';
      bool first = true;
      for (auto it = j.begin(); it != j.end(); ++it) {
        if (!first) *out += ", ";
        first = false;
        if (!append_plain_string(it.key(), out))
          *out += ojson(it.key()).dump(-1, ' ', false, ojson::error_handler_t::replace);
        *out += ": ";
        dump(it.value(), out);
      }
      *out += '}';
      return;
    }
    case ojson::value_t::array: {
      if (j.empty()) {
        *out += "[]";
        return;
      }
      *out += '[';
      bool first = true;
      for (const auto& v : j) {
        if (!first) *out += ", ";
        first = false;
        dump(v, out);
      }
      *out += ']';
      return;
    }
    case ojson::value_t::string:
      if (append_plain_string(j.get_ref<const std::string&>(), out)) return;
      [[fallthrough]];
    default: *out += j.dump(-1, ' ', false, ojson::error_handler_t::replace);
  }
}

}  // namespace

std::string_view py_strip(std::string_view s) {
  size_t b = 0;
  while (b < s.size()) {
    size_t n = space_len(s, b);
    if (!n) break;
    b += n;
  }
  size_t e = s.size();
  while (e > b) {
    size_t k = e - 1;  // start of the last character
    while (k > b && is_cont(static_cast<unsigned char>(s[k]))) --k;
    size_t n = space_len(s, k);
    if (!n || k + n != e) break;
    e = k;
  }
  return s.substr(b, e - b);
}

std::string py_json(const ojson& j) {
  std::string out;
  dump(j, &out);
  return out;
}

std::string py_str(const ojson& j) {
  switch (j.type()) {
    case ojson::value_t::string: return j.get<std::string>();
    case ojson::value_t::boolean: return j.get<bool>() ? "True" : "False";
    case ojson::value_t::null: return "None";
    case ojson::value_t::object:
    case ojson::value_t::array: return py_json(j);
    default: return j.dump();
  }
}

bool holds_client_text(const ojson& j) {
  if (j.is_string() || (j.is_object() && !j.empty())) return true;
  if (j.is_array())
    for (const auto& v : j)
      if (holds_client_text(v)) return true;
  return false;
}

}  // namespace flyx::models::formats
