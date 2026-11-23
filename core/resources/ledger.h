// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// ResourceLedger: device memory is accounted for before it is taken.
//
// The ledger only keeps accounts (category, owner, bytes, budget).  Backends
// allocate; a Lease connects the two: whoever wants memory first asks the
// ledger for a lease, which fails with an itemized explanation when the
// budget would be exceeded, and gives the bytes back when it is destroyed.
// Nothing is ever shrunk silently: a caller that does not get its lease
// reports why (and what would fit) instead of running with less.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace flyx {

class ResourceLedger;

class Lease {
 public:
  Lease() = default;
  ~Lease() { release(); }
  Lease(Lease&& o) noexcept { *this = std::move(o); }
  Lease& operator=(Lease&& o) noexcept;
  Lease(const Lease&) = delete;
  Lease& operator=(const Lease&) = delete;
  void release();
  uint64_t bytes() const { return bytes_; }
  explicit operator bool() const { return ledger_ != nullptr; }

 private:
  friend class ResourceLedger;
  ResourceLedger* ledger_ = nullptr;
  std::string key_;
  uint64_t bytes_ = 0;
};

class ResourceLedger {
 public:
  // budget 0: unknown (a CPU backend); every lease is granted.
  void set_budget(uint64_t bytes) { budget_ = bytes; }
  uint64_t budget() const { return budget_; }
  uint64_t used() const;
  uint64_t available() const;
  // A lease for `bytes` in `category` held by `owner`; on failure *why says
  // what is in use and what is missing.
  Lease reserve(const std::string& category, const std::string& owner, uint64_t bytes, std::string* why);
  // One line per account: "weights 3.53 GB, session 1 1.21 GB, ...".
  std::string report() const;

 private:
  friend class Lease;
  uint64_t used_locked() const;
  void give_back(const std::string& key, uint64_t bytes);

  mutable std::mutex mu_;
  uint64_t budget_ = 0;
  std::map<std::string, uint64_t> held_;  // "category: owner" -> bytes
};

std::string gb(uint64_t bytes);

}  // namespace flyx
