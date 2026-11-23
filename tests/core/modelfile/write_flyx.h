// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A .flyx file written by a test (the layout python/convert/flyx_convert/fileformat.py writes): the header a
// test gives, then its data.  For tests that need a file of their own making, a wrong one included, and no Python.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace flyx::testing {

// The bytes of a .flyx file: header (JSON text), padding to the data's start, data.  header_length: what the file
// says its header's length is (a test of a file that lies gives another); version: its format (2: a header with
// "requires", core/modelfile/flyx_format.cpp).
inline std::string flyx_file(const std::string& header, const std::string& data, uint64_t header_length,
                             uint32_t version = 1) {
  constexpr size_t kAlignData = 4096;
  std::string f("FLYX\0\0\0\0", 8);
  const uint32_t reserved = 0;
  f.append(reinterpret_cast<const char*>(&version), 4);
  f.append(reinterpret_cast<const char*>(&reserved), 4);
  f.append(reinterpret_cast<const char*>(&header_length), 8);
  f += header;
  f.resize((f.size() + kAlignData - 1) / kAlignData * kAlignData, '\0');
  return f + data;
}

inline std::string flyx_file(const std::string& header, const std::string& data) {
  return flyx_file(header, data, header.size());
}

inline bool write_file(const std::string& path, const std::string& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return bool(f);
}

}  // namespace flyx::testing
