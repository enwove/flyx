// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The system log: the last 2000 lines the server process wrote to standard error (the server's own,
// the engine's, the backend's), numbered, for the dashboard's Logs page (GET /v1/logs).  Kept in memory only; the
// terminal still gets every line.  `flyx serve --no-system-log` keeps none and leaves standard error alone.
#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "core/exec/scheduler.h"

namespace flyx::server {

class SystemLog {
 public:
  static constexpr size_t kKept = 2000;
  static constexpr size_t kLineBytes = 8 << 10;  // a longer line is cut: 2000 lines stay under 16 MB

  // A line as it was written (no newline); from the capture's thread.
  void add(const std::string& line);
  // Whether the lines are being kept: the capture took standard error.
  void set_enabled(bool on);
  // The newest `n` lines' text, oldest first (for the terminal dashboard and shutdown tail).
  std::vector<std::string> last(size_t n) const;
  // The kept lines numbered past `after` (0: from the oldest kept), at most `limit` of them, oldest first:
  // {"enabled", "lines": [{"seq", "t", "text"}], "next": the last number given (to ask after it next),
  //  "missed": lines past `after` that are no longer kept}.
  ojson since(uint64_t after, size_t limit) const;

 private:
  struct Line {
    uint64_t seq;
    double t;  // unix seconds
    std::string text;
  };
  mutable std::mutex mu_;
  bool enabled_ = false;
  uint64_t last_seq_ = 0;
  std::deque<Line> lines_;  // oldest first
};

}  // namespace flyx::server
