// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// SHA-256 (FIPS 180-4), to check fetched files against the hash the Hub lists.  x86-64 processors with the SHA
// extensions (AMD since Zen, Intel since Ice Lake) hash through them, about 2 GB/s a core; others in plain C++.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace flyx::hub {

class Sha256 {
 public:
  // portable: the plain C++ rounds even where the SHA instructions are (the test holds the two paths to each other).
  explicit Sha256(bool portable = false);
  void update(const void* data, size_t n);
  // The digest as 64 lowercase hex digits; the object is spent after it.
  std::string hex();
  // Whether this processor hashes with its SHA instructions.
  static bool accelerated();

 private:
  void blocks(const uint8_t* p, size_t n);  // n whole 64-byte blocks
  bool portable_;
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t total_ = 0;
};

}  // namespace flyx::hub
