// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx-server: OpenAI- (chat completions, Responses) and Anthropic-compatible HTTP API over one engine.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "apps/common/defaults.h"
#include "apps/common/numbers.h"
#include "apps/flyx-server/http_util.h"
#include "apps/flyx-server/model_host.h"
#include "apps/flyx-server/request_log.h"
#include "apps/flyx-server/system_log.h"
#include "core/exec/scheduler.h"
#include "httplib/httplib.h"

namespace flyx::server {

struct Config {
  ChatOptions chat_defaults;
  // Qwen3.5's recommended settings (its model card): with thinking, the ones
  // for precise coding (no penalty); without, the general ones, which come
  // with a presence penalty of 1.5.
  SamplingParams think_defaults{0.6f, 0.95f, 20, 0, 0.f, 0.f};
  SamplingParams plain_defaults{0.7f, 0.8f, 20, 0, 1.5f, 0.f};
  std::string api_key;
  // The origins whose pages a browser lets call the inference API (--cors): this machine's own pages unless that is
  // turned off, the origins named, or any.  The model panel, the request log, the logs and the telemetry are never
  // another origin's, whatever is set here.
  bool cors_loopback = true;
  bool cors_any = false;
  std::vector<std::string> cors_origins;  // as a browser writes them, lower case: "https://chat.example.com", "null"
  bool keep_billing_header = false;             // Anthropic: keep Claude Code's per-request billing line
  int verbose = 0;
};

struct ListenConfig {
  std::string host;
  int port = 0;
  bool lan = false;
  std::vector<std::string> accepted_hosts;
};

struct LanManage {
  bool from_cli = false;
  std::atomic<bool> saved{false};
  static bool requested_for(bool from_cli_v, bool saved_v, bool api_key_set) {
    return from_cli_v || (!api_key_set && saved_v);
  }
  static bool effective_for(bool from_cli_v, bool saved_v, const ListenConfig& l, bool api_key_set) {
    return l.lan && !api_key_set && requested_for(from_cli_v, saved_v, api_key_set);
  }
  bool requested(bool api_key_set) const {
    return requested_for(from_cli, saved.load(std::memory_order_acquire), api_key_set);
  }
  bool effective(const ListenConfig& l, bool api_key_set) const {
    return effective_for(from_cli, saved.load(std::memory_order_acquire), l, api_key_set);
  }
};

struct Server {
  ModelHost* host;
  Config cfg;
  EngineOptions base;  // the command line's engine options: what a model named in the dashboard starts from
  double started_at = 0;  // unix seconds
  ojson device_probe;     // the GPU found at start when no model was named (telemetry's device until one loads)
  uint64_t device_reserve_floor = 0;  // backend-reported reserve floor from start-up probe (for idle telemetry)
  std::atomic<uint64_t> probe_mem_used{0};  // refreshed by /v1/device/* so idle VRAM bars update after closing apps
  RequestLog* requests = nullptr;  // the last finished requests, for GET /v1/requests
  SystemLog* logs = nullptr;       // the last lines on standard error, for GET /v1/logs
  std::vector<std::string> models_dirs{};  // canonicalized --models-dir + FLYX_MODELS_DIR roots
  ListenConfig listener{};         // immutable listener bind identity constructed at startup
  LanManage lan_manage{};          // opt-in (--no-auth or saved setting) for keyless LAN management
};

// How long a request waits for its job to start before it answers.  A job refused at the start (its prompt longer than
// the context) gets an error status, as clients look for, not an error inside a stream already answered 200; a job
// still queued after it streams, its errors inside the stream.
constexpr int kStartWaitMs = apps::kStartWaitMs;

// Output of a finished job, collected (non-streaming responses).
struct Collected {
  bool ok = false;
  bool superseded = false;
  JobEvent failure;  // !ok: the job's error
  std::string reasoning, content;
  std::vector<ToolCall> calls;
  GenerateResult result;
};
// Consume a nonstreaming job, checking for disconnects both during prefill and while tokens are arriving.
// Nonterminal events go to consume; the finished or failed event is returned. The recording sees all of them.
JobEvent collect_events(const std::shared_ptr<Job>& job, const httplib::Request& req, Recording* rec,
                        const std::function<void(JobEvent&)>& consume);
Collected collect(const std::shared_ptr<Job>& job, const httplib::Request& req, Recording* rec);

// Check the connection every second while waiting for the scheduler, sending a keep-alive every five seconds.
template <class Poll>
bool wait_stream(httplib::DataSink& sink, const std::string& ping, Poll poll) {
  for (int quiet = 1; !poll(1000); ++quiet) {
    if (!sink.is_writable()) return false;
    if (quiet % 5 == 0 && !sink.write(ping.data(), ping.size())) return false;
  }
  return true;
}

// A job's events to the client as server-sent events (streaming responses), the part every API's stream shares.
// write(sink, next) writes the whole stream in the API's own events: next(&ev) gives the job's next event, sending
// `ping` every 5 s it waits (so proxies keep the connection), and is false once the client has gone; write is
// true when the stream is whole, false when the client went away.  A client gone, here or as the connection
// drops, cancels the job.  The stream holds the scheduler (`sched`) until it ends: a job reaches back into it.
// Every event next() gives is shown to `rec` (null: the request log is off), logged once the stream ends.
template <class Write>
void stream_events(httplib::Response& res, const std::shared_ptr<Job>& job, std::shared_ptr<Scheduler> sched,
                   std::shared_ptr<Recording> rec, std::string ping, Write write) {
  res.set_header("Cache-Control", "no-cache");
  res.set_chunked_content_provider(
      "text/event-stream",
      [job, sched, rec, ping = std::move(ping), write = std::move(write)](size_t, httplib::DataSink& sink) mutable {
        auto next = [&](JobEvent* ev) {
          if (!wait_stream(sink, ping, [&](int ms) { return job->next(ev, ms); })) return false;
          if (rec) rec->see(*ev);
          return true;
        };
        const bool whole = write(sink, next);
        if (rec) rec->finish();
        if (!whole) {
          job->cancel();
          return false;
        }
        sink.done();
        return true;
      },
      [job, sched](bool success) {
        if (!success) job->cancel();
      });
}

void add_openai_chat_routes(httplib::Server& http, Server* s);
void add_anthropic_messages_routes(httplib::Server& http, Server* s);
void add_openai_responses_routes(httplib::Server& http, Server* s);
void add_telemetry_routes(httplib::Server& http, Server* s);
void add_manage_routes(httplib::Server& http, Server* s);  // manage.cpp: the dashboard's model panel
ListenConfig make_listen_config(const std::string& bound_host, int port,
                                std::string_view requested_host = {});  // network.cpp
void add_network_routes(httplib::Server& http, Server* s, bool host_from_cli);
void add_request_log_routes(httplib::Server& http, Server* s);  // request_log.cpp: GET /v1/requests
void add_system_log_routes(httplib::Server& http, Server* s);   // system_log.cpp: GET /v1/logs

// The dashboard's files (apps/flyx-server/dashboard/ and the libraries it imports), built into the executable
// (cmake/embed_files.cmake): each by its path under /dashboard/.
struct EmbeddedFile {
  const char* path;
  const unsigned char* data;
  size_t size;
};
extern const EmbeddedFile kDashboardFiles[];
extern const size_t kDashboardFilesCount;

}  // namespace flyx::server
