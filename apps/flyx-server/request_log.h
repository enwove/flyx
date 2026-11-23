// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The request log: the last 50 finished requests of every API, what the client sent and what the
// model answered, for the dashboard's Requests page (GET /v1/requests).  Kept in the server's memory only, never
// written to disk; `flyx serve --no-request-log` keeps none.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/exec/scheduler.h"
#include "core/os/os.h"

namespace flyx::server {

class RequestLog;

// One request as it is answered: the handler shows it each event of its job as it reads it (see()), and it goes into
// the log once, when the handler is done with the job (finish()) or, should the handler never say so (a client gone
// before its stream began), as it is destroyed.  Used by one thread at a time: the one answering the request.
class Recording {
 public:
  Recording(RequestLog* log, ojson entry, double started_ms, std::shared_ptr<Job> job);
  ~Recording();
  Recording(const Recording&) = delete;
  Recording& operator=(const Recording&) = delete;

  void see(const JobEvent& ev);
  void finish();
  void set_client_hints(ojson hints) {
    if (!hints.empty()) entry_["client_hints"] = std::move(hints);
  }

 private:
  RequestLog* log_;
  std::shared_ptr<Job> job_;
  ojson entry_;  // id, api, model, started_at, stream, request: what was known when the job was submitted
  double started_ms_;
  double ttft_ms_ = -1;  // < 0 until the first reasoning, content or tool call
  std::string reasoning_, content_, error_;  // the text kept: a little past kStringBytes at most
  size_t reasoning_bytes_ = 0, content_bytes_ = 0;  // all the text that came
  ojson calls_ = ojson::array();
  // Until the job says how it ended: a handler that stops reading before then has lost its client.
  std::string status_ = "cancelled";
  int prompt_tokens_ = 0, cached_tokens_ = 0, output_tokens_ = 0;
  bool logged_ = false;
};

class RequestLog {
 public:
  static constexpr size_t kKept = 50;
  // Each request string, reasoning or answer keeps up to this many bytes plus its truncation note.
  // The serialized body can exceed the target when many short values or keys remain.
  static constexpr size_t kStringBytes = size_t(64) << 10;
  static constexpr size_t kRequestBytes = size_t(256) << 10;

  explicit RequestLog(bool enabled) : enabled_(enabled) {}

  // Capture when the handler receives the body, even if it has to wait for a session.
  struct Arrival {
    double monotonic_ms = os::now_ms();
    double unix_seconds = os::unix_seconds();
  };

  bool enabled() const { return enabled_; }
  // A request whose job was just submitted; nullptr when the log is off.  `api`: chat, completions, responses or
  // messages; `id`: the id the API answers with.
  std::shared_ptr<Recording> start(const char* api, const std::string& id, const std::string& model, bool stream,
                                   const ojson& body, Arrival arrival, const std::shared_ptr<Job>& job = {});
  // The last `limit` finished requests, newest first (each pre-serialized as compact JSON).
  struct Snapshot {
    uint64_t revision = 0;
    std::vector<std::shared_ptr<const std::string>> entries;
    std::string json_response(bool enabled) const;
  };
  Snapshot snapshot(size_t limit) const;

  // `s` cut to `max` bytes (at a character's start) with a note of how many were left out, of `total` when `s` is
  // what was kept of a longer text.
  static std::string clip(const std::string& s, size_t max = kStringBytes, size_t total = 0);

 private:
  friend class Recording;
  void add(ojson entry);

  const bool enabled_;
  mutable std::mutex mu_;  // requests finish on many HTTP threads
  uint64_t revision_ = 0;
  std::deque<std::shared_ptr<const std::string>> entries_;  // newest first
};

}  // namespace flyx::server
