// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The system log and GET /v1/logs?after=SEQ&limit=N (default 500, at most 2000, oldest first): for
// this machine, or with --api-key for the key's holder (may_manage), as the request log: it names files and requests.
#include "apps/flyx-server/system_log.h"

#include <algorithm>

#include "apps/flyx-server/server.h"
#include "core/os/os.h"

namespace flyx::server {

void SystemLog::add(const std::string& line) {
  const double t = os::unix_seconds();
  std::lock_guard<std::mutex> lock(mu_);
  lines_.push_back({++last_seq_, t, line.size() > kLineBytes ? RequestLog::clip(line, kLineBytes) : line});
  while (lines_.size() > kKept) lines_.pop_front();
}

void SystemLog::set_enabled(bool on) {
  std::lock_guard<std::mutex> lock(mu_);
  enabled_ = on;
}

std::vector<std::string> SystemLog::last(size_t n) const {
  std::lock_guard<std::mutex> lock(mu_);
  const size_t from = lines_.size() > n ? lines_.size() - n : 0;
  std::vector<std::string> out;
  out.reserve(lines_.size() - from);
  for (size_t i = from; i < lines_.size(); ++i) out.push_back(lines_[i].text);
  return out;
}

ojson SystemLog::since(uint64_t after, size_t limit) const {
  std::lock_guard<std::mutex> lock(mu_);
  ojson out = ojson::array();
  uint64_t next = std::min(after, last_seq_), missed = 0;
  if (!lines_.empty() && lines_.front().seq > 0 && after < lines_.front().seq - 1)
    missed = (lines_.front().seq - 1) - after;
  for (const Line& l : lines_) {
    if (l.seq <= after) continue;
    if (out.size() >= limit) break;
    out.push_back({{"seq", l.seq}, {"t", l.t}, {"text", l.text}});
    next = l.seq;
  }
  return {{"enabled", enabled_}, {"lines", std::move(out)}, {"next", next}, {"missed", missed}};
}

void add_system_log_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/logs", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "the system log")) return;
    const uint64_t after = query_u64(q, "after", 0);
    const size_t limit = size_t(query_int(q, "limit", 500, 0, int(SystemLog::kKept)));
    r.set_header("Cache-Control", "no-store");
    if (!s->logs) return send_json(r, 200, {{"enabled", false}, {"lines", ojson::array()}, {"next", 0}, {"missed", 0}});
    send_json(r, 200, s->logs->since(after, limit));
  });
}

}  // namespace flyx::server
