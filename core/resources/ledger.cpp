// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/resources/ledger.h"

#include <cstdio>

namespace flyx {

std::string gb(uint64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f GB", bytes / 1e9);
  return buf;
}

Lease& Lease::operator=(Lease&& o) noexcept {
  if (this != &o) {
    release();
    ledger_ = o.ledger_;
    key_ = std::move(o.key_);
    bytes_ = o.bytes_;
    o.ledger_ = nullptr;
    o.bytes_ = 0;
  }
  return *this;
}

void Lease::release() {
  if (ledger_) ledger_->give_back(key_, bytes_);
  ledger_ = nullptr;
  bytes_ = 0;
}

uint64_t ResourceLedger::used_locked() const {
  uint64_t u = 0;
  for (const auto& [k, v] : held_) u += v;
  return u;
}

uint64_t ResourceLedger::used() const {
  std::lock_guard<std::mutex> lock(mu_);
  return used_locked();
}

uint64_t ResourceLedger::available() const {
  std::lock_guard<std::mutex> lock(mu_);
  const uint64_t u = used_locked();
  return budget_ > u ? budget_ - u : 0;
}

Lease ResourceLedger::reserve(const std::string& category, const std::string& owner, uint64_t bytes,
                              std::string* why) {
  Lease l;
  const std::string key = category + ": " + owner;
  {
    std::lock_guard<std::mutex> lock(mu_);
    const uint64_t u = used_locked();
    if (budget_ && (bytes > budget_ || u > budget_ - bytes)) {
      if (why) {
        *why = key + " needs " + gb(bytes) + ", " + gb(budget_ > u ? budget_ - u : 0) + " left of " + gb(budget_) +
               " (in use:";
        for (const auto& [k, v] : held_) *why += " " + k + " " + gb(v) + ";";
        *why += ")";
      }
      return l;
    }
    held_[key] += bytes;
  }
  l.ledger_ = this;
  l.key_ = key;
  l.bytes_ = bytes;
  return l;
}

void ResourceLedger::give_back(const std::string& key, uint64_t bytes) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = held_.find(key);
  if (it == held_.end()) return;
  it->second -= std::min(it->second, bytes);
  if (it->second == 0) held_.erase(it);
}

std::string ResourceLedger::report() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::string s;
  uint64_t u = 0;
  for (const auto& [k, v] : held_) {
    s += (s.empty() ? "" : ", ") + k + " " + gb(v);
    u += v;
  }
  if (budget_) s += " | " + gb(u) + " of " + gb(budget_) + " budget";
  return s;
}

}  // namespace flyx
