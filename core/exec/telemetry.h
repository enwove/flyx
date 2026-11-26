// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What the scheduler is doing and has done, for monitoring (the server's
// GET /v1/telemetry, `flyx monitor`). The scheduler thread records; any
// thread takes a snapshot. Plain figures: the wire format is the server's.
#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "core/plan/budget.h"

namespace flyx {

// One finished request.
struct RequestRecord {
  double finished_at = 0;         // unix seconds
  std::string label;              // chat, responses, messages, completion
  std::string status;             // stop, length, tool_calls, cancelled, error
  std::string error;              // status error: what went wrong
  int session = 0;                // the resident session it ran in (0 = none)
  int prompt_tokens = 0, reused_tokens = 0, output_tokens = 0;
  double queue_ms = 0, prefill_ms = 0, decode_ms = 0;
  double ttft_ms = -1;            // receipt to first reasoning, content or tool event; -1 without output
  double tokens_per_round = 0;
  int drafted = 0, accepted_drafts = 0;  // drafts its rounds verified, and of them the ones accepted
  int tool_calls = 0, grammar_fixes = 0;
  bool superseded = false;        // a newer request for the conversation cancelled this one
  std::string response_id;
};

// The loaded model, as of its load.
struct ModelInfo {
  std::string device, backend, drafter, chat_format, memory;  // memory: Engine::memory_report()
  bool chat_ok = false, chat_native = false;
  bool vision = false;
  std::string vision_device, vision_quant;
  int vision_max_tokens = 0;
  int width = 0;                  // widest round
  int max_ctx = 0, max_sessions = 0;
  uint64_t ledger_budget = 0;     // device bytes the engine may use (0 = unknown)
  double load_s = 0;
  // Where memory cut the context under the one wanted (Engine::budget): which memory, the context wanted, whether
  // the one it has is short (kShortCtx), and the figures behind it: the GPU's part the sessions share and what holds
  // the rest, the host memory K/V may take.
  plan::CtxLimit ctx_limit = plan::CtxLimit::kNone;
  int wanted_ctx = 0;
  int kv_gpu_tokens = 0;          // K/V positions placed on the GPU (0..max_ctx)
  bool ctx_short = false;
  uint64_t sessions_room = 0, host_budget = 0, ctx_need_more = 0;
  plan::GpuHeld held;
};

struct TelemetrySnapshot {
  enum Phase { kUnloaded, kLoading, kIdle, kPrefill, kDecode };
  Phase phase = kUnloaded;
  // Requests running now.  Several take turns on the GPU thread; the fields
  // below are those of the one that started first.
  int running = 0;
  // The running request (phase prefill or decode).
  std::string label;
  std::string response_id;        // public id, when the caller supplied one
  int session = 0;
  int prompt_tokens = 0, reused_tokens = -1;  // -1: not known yet
  int prefilled = 0;              // prompt tokens submitted to the device so far
  int generated = 0, max_new_tokens = 0;
  double elapsed_ms = 0;          // since it started running
  double decode_tok_s = 0;        // tokens per second over the last two seconds, every running request
  double window_tokens_per_round = -1;  // tokens per round over the last two seconds, every running request (-1: none)
  double window_acceptance = -1;        // accepted / drafted over the same window (-1: none drafted)
  double request_decode_tok_s = -1;  // this request's mean since its first token delivery
  double request_prefill_tok_s = -1; // its prompt tokens computed over the time from its start to its first delivery
                                     // (or to now, while it prefills): as RequestRecord's prefill_ms counts it
  double tokens_per_round = 0;    // the reply so far (1 + accepted / rounds, or tokens per delivery before round 1)
  int rounds = 0, drafted = 0, accepted = 0;  // its rounds verified so far, their drafts, and those accepted
  int queued = 0;                 // requests waiting
  // The model and its sessions.
  bool loaded = false;
  ModelInfo model;
  uint64_t ledger_used = 0;       // device bytes leased (weights, workspace, sessions)
  uint64_t host_mapped = 0;       // host bytes the sessions' K/V hold mapped for the device only (Engine)
  struct Slot {
    int id = 0, tokens = 0;
  };
  std::vector<Slot> sessions;     // most recently used first
  // History.
  std::vector<int> output_per_second;  // tokens generated in each of the last 60 whole seconds, oldest first
  std::vector<int> prefill_per_second; // prompt tokens computed (not reused) in each of those seconds, oldest first:
                                       // a pass's tokens count in the second it went to the device
  // Rounds verified in each of those seconds (every running request's), the drafts they verified, and those accepted.
  std::vector<int> rounds_per_second, drafted_per_second, accepted_per_second;
  std::vector<RequestRecord> recent;   // newest first
  struct Totals {
    int64_t requests = 0, errors = 0, cancelled = 0, prompt_tokens = 0, reused_tokens = 0, output_tokens = 0,
            tool_calls = 0, grammar_fixes = 0;
  } totals;
  double started_at = 0;          // unix seconds the server started
};

// The names of FLYX_CLOCK_* bits ("power cap, thermal"); "" when none is set.
std::string clock_limit_names(uint32_t bits);

class Telemetry {
 public:
  Telemetry();
  void loading();
  void loaded(const ModelInfo& m, uint64_t ledger_used);
  void unloaded();
  void memory(uint64_t ledger_used, uint64_t host_mapped, std::vector<TelemetrySnapshot::Slot> sessions);
  // A request starts running; the id names it in the calls below.
  int start(const std::string& label, int session, int prompt_tokens, int max_new_tokens, std::string response_id = {});
  // Prompt tokens submitted so far (the first call: those reused, nothing submitted yet).
  void prefill(int id, int done, int total);
  // Tokens the reply grew by.
  void tokens(int id, int n);
  // A round verified: the drafts it verified (its width), and of them the ones accepted.
  void round(int id, int drafted, int accepted);
  // A request ended; id 0: one that never started running.
  void finish(int id, RequestRecord r);
  void snapshot(TelemetrySnapshot* out) const;

 private:
  static constexpr int kSeconds = 64;
  void advance(int64_t second);  // buckets up to `second`, emptied
  void computed(int n, double start_ms, double end_ms);  // n prompt tokens computed across [start_ms, end_ms]

  struct Running {
    int id = 0;
    std::string label;
    std::string response_id;
    int session = 0;
    int prompt_tokens = 0, reused_tokens = -1, prefilled = 0, generated = 0, max_new_tokens = 0;
    int pending_prefill = 0;  // latest submitted prefill pass, spread over its execution time once it finishes
    double started_ms = 0;  // steady clock
    double prefill_at_ms = 0;
    double first_delivery_ms = 0;
    int first_delivery_tokens = 0;
    bool decoding = false;
    int deliveries = 0;
    int rounds = 0, drafted = 0, accepted = 0;
  };
  // What a second saw: tokens generated, prompt tokens computed, rounds verified, their drafts, those accepted.
  struct Second {
    int out = 0, pre = 0, rounds = 0, drafted = 0, accepted = 0;
  };
  Second& second_now(double now_ms);  // the current second's, the seconds before it advanced
  Running* find(int id);

  mutable std::mutex mu_;
  TelemetrySnapshot s_;                    // phase while nothing runs, model, totals
  std::vector<Running> running_;           // oldest first
  int next_id_ = 1;
  int decoding_ = 0;                       // of the running requests
  std::deque<std::pair<double, int64_t>> points_;  // (time, tokens so far) over the last seconds, while any decodes
  struct RoundPoint {
    double time_ms = 0;
    int drafted = 0, accepted = 0;
  };
  std::deque<RoundPoint> round_points_;    // verified rounds over the last seconds, while any decodes
  int64_t produced_ = 0;                   // every token generated
  Second seconds_[kSeconds];               // a second each, the last kSeconds
  int64_t last_second_ = 0;
  std::deque<RequestRecord> recent_;
};

}  // namespace flyx
