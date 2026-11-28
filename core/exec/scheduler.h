// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// One thread owns the GPU; other threads (HTTP handlers) submit jobs and read
// their events.
//
// The thread is a loop over the requests that are running, each a Generation
// (core/exec/generation.h) on a session of its own.  A request alone is
// advanced step after step, waiting for the device as it always did.  Several
// take turns: each is advanced when the device has its next result, and none
// waits for another to end. Their rounds stay their own: what
// a request generates does not depend on what runs beside it.
//
// * Jobs wait in a bounded queue; a job waits at most queue_timeout_ms.  A
//   job starts as soon as the session it belongs on is free: at most
//   max_sessions run at once, and one whose session is running another
//   request waits for it without holding up the jobs behind it.
// * A new request for a conversation that is generating right now (a retry,
//   or a follow-up sent without waiting) cancels the running one.  Chat
//   messages say which conversation they continue; a raw prompt does not, so a
//   raw generation is replaced only by a request with its conversation_id.
// * Cancelling a job stops it at the next round, or between prefill passes.
// * Idle tiers: after idle_unload_ms without requests the sessions are
//   freed (weights stay; the next request recomputes its prompt); after
//   keep_alive_ms the model is unloaded too (the next request loads it
//   again).  Both report what they freed; a reload reports its time.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/exec/request.h"
#include "core/exec/telemetry.h"
#include "core/render/chat.h"
#include "core/session/prefix_cache.h"
#include "core/speculation/batch_schedule.h"

namespace flyx {

struct JobEvent {
  enum Kind { kReasoning, kContent, kToolCallStart, kToolCallDelta, kToolCall, kDone, kError } kind;
  std::string text;                  // reasoning / content delta, kToolCallDelta args fragment, error message
  ToolCall call;
  GenerateResult result;             // kDone
  int tool_calls = 0;                // kDone
  double queue_ms = 0, tokens_per_round = 0;  // kDone: this job's scheduler figures
  // kError for a prompt longer than the context (the request's fault, not the server's): its tokens, and the most
  // the context takes; 0 for any other error.
  int prompt_tokens = 0, prompt_limit = 0;
  bool superseded = false;  // a later request for the conversation cancelled this job
  // kError the request brought on itself besides its length: tools or a schema no grammar can be made from.
  bool invalid_request = false;
  // kError because the server stopped serving (the model unloaded or switched, a shutdown) or the request waited in
  // the queue past its limit: one to send again, neither the request's fault nor a failure of the server.
  bool unavailable = false;
};

class Scheduler;

class Job {
 public:
  // Waits up to timeout_ms for the next event; false on timeout.
  bool next(JobEvent* ev, int timeout_ms);
  // Waits up to timeout_ms for the job to start: its session given, its prompt counted.  False when it ended there
  // instead, *ev its error (a prompt longer than the context, ...); true when it runs, or still waits in the queue.
  // (A server answers a request refused at the start with an error status, before a stream's first bytes.)
  bool wait_start(JobEvent* ev, int timeout_ms);
  bool has_started() const;  // the session was granted and the prompt accepted
  // Wait for the number of prompt tokens and how many the session actually reused. A job that ends before
  // preparing its prompt returns zero counts; its terminal event is still read with next().
  bool wait_prompt_usage(flyx_usage* usage, int timeout_ms);
  void cancel();
  bool finished() const { return finished_.load(); }

 private:
  friend class Scheduler;
  struct CancelGate {
    explicit CancelGate(Scheduler* e) : sched(e) {}
    std::mutex mu;
    Scheduler* sched = nullptr;
  };
  void push(JobEvent ev);
  void started();
  void prompt_ready(int tokens, int reused);

  std::weak_ptr<CancelGate> gate_;
  Request in_;
  RenderedChat prompt_;              // rendered at submit; its text also recognises the same conversation
  double arrived_at_ = 0, queued_at_ = 0;
  double ttft_ms_ = -1;              // set by the scheduler when it emits its first output
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<JobEvent> events_;
  bool started_ = false;             // under mu_
  int prompt_tokens_ = 0, reused_tokens_ = 0;  // under mu_, set once before the first prefill pass
  std::atomic<bool> cancelled_{false}, finished_{false};
  std::atomic<bool> superseded_{false};  // set under Scheduler::mu_ while queued or running, read on removal
  std::atomic<bool> closed_{false};      // cancelled by Scheduler::close() while still live
  uint64_t resume_assignment_ = 0;  // the interrupted prewarm's slot while it still holds that work
};

struct SchedulerOptions {
  // Sessions kept on the GPU at most; 0 uses kDefaultSessions (never more than fit in memory).
  static constexpr int kDefaultSessions = 8;
  int max_sessions = 0;
  // Host memory for the checkpoints of conversations that gave their session to another (0: none).
  uint64_t host_cache_bytes = 8000000000ull;
  // A disk for them past that: files under this directory, disk_cache_bytes of them at most ("":
  // none).
  std::string disk_cache_dir;
  uint64_t disk_cache_bytes = 0;
  // Requests running at once share rounds, the weights read once for all of them; false: each round
  // holds one session's rows, the sessions taking turns.
  bool batch_rounds = true;
  int queue_limit = 16;
  int queue_timeout_ms = 600000;
  // Every request decoded, with its record (status: stop, length, tool_calls or cancelled), on the scheduler thread.
  std::function<void(const RequestRecord&)> on_finished;
  int idle_unload_ms = 0;            // free the sessions after this long idle; 0 = never
  int keep_alive_ms = 0;             // unload the model after this long idle; 0 = never
  int verbose = 0;
};

class Scheduler {
 public:
  // Loads the model (check ok()); unloads and reloads it per keep_alive_ms.  It counts in `telemetry`: a server's,
  // which outlives the models it serves one after another (their totals and recent requests add up), or its own.
  Scheduler(const std::string& model, const EngineOptions& eo, const SchedulerOptions& o,
            std::shared_ptr<Telemetry> telemetry = std::make_shared<Telemetry>());
  ~Scheduler();
  bool ok(std::string* err) const;

  // Renders and queues the job.  Returns nullptr (and *err, *status) when the
  // request is invalid (400) or the queue is full (503).  Any thread.
  // arrived_at is the caller's monotonic receipt time in ms, or submit time when omitted.
  std::shared_ptr<Job> submit(Request in, std::string* err, int* status, double arrived_at = -1);
  // Tokens of a request's prompt (Anthropic count_tokens; works while unloaded).
  int count_tokens(const Request& in, std::string* err);
  // Tokens of a piece of text, e.g. a reply's thinking (any thread).
  int count_text_tokens(const std::string& text) const { return static_cast<int>(tok_.encode(text).size()); }
  // The immutable target header used for HTTP tokenization, also available while idle-unloaded.
  const nlohmann::json& model_header() const { return file_.header(); }
  // The chat template's output language ("none" when the file has no usable template).
  const char* chat_format() const { return chat_.format(); }
  // The loaded engine, or nullptr while unloaded.  For start-up reports only:
  // the scheduler thread owns it.
  Engine* engine() const { return engine_.get(); }
  // Thread-safe check of whether the engine is currently loaded.
  bool is_loaded() const {
    std::shared_lock<std::shared_mutex> lock(engine_mu_);
    return engine_ != nullptr;
  }
  // Thread-safe check of whether this model has an active vision encoder (survives idle unload).
  bool has_vision() const { return has_vision_.load(); }
  // The context of a session: as asked for, or the longest that fit in GPU memory when it was not.
  int max_ctx() const { return ctx_ > 0 ? ctx_.load() : eo_.max_ctx; }
  // What the scheduler is doing and has done, and the device's live figures
  // (*device_ok false when there are none: no model loaded, or a CPU backend).  Any thread.
  void telemetry(TelemetrySnapshot* out, flyx_device_stats* device, bool* device_ok) const;
  int max_sessions() const {
    std::shared_lock<std::shared_mutex> lock(engine_mu_);
    return max_sessions_;
  }
  int queued() const;
  // Ends every request, queued and running (each at its next round), and refuses new ones (503): the server is about
  // to serve another model, and this one's memory is wanted back.  Any thread.
  void close();

 private:
  friend class Job;
  // A request on the GPU thread, from the session it was given to its last event.
  struct Running;
  enum class Start { kRunning, kWaits, kOver };
  void loop();
  // Starts the queued requests that can start (mu_ held; released while one is made ready).
  void admit(std::unique_lock<std::mutex>& lock);
  // Gives a job its session and makes it ready.  kWaits: that session is
  // running another request; kOver: the job ended here (its error is sent).
  Start start(const std::shared_ptr<Job>& job, std::unique_ptr<Running>* out);
  // One unit of work for every running request the device is ready for.
  void advance_all();
  bool launch_rounds();  // several running: their next rounds in rounds of several sessions
  // The last events, the log line and the record of a request that ended.
  void finish(Running& r);
  void failed(Running& r, JobEvent ev);
  void failed(Running& r, const std::string& msg) { failed(r, JobEvent{JobEvent::kError, msg, {}, {}, 0}); }
  // A reply's pieces as the job's events.
  static void forward(Job& job, std::vector<OutputEvent> events);
  void cancel(Job* job);
  void cancel_all_locked(bool closing = false);  // mu_ held: marks all queued and running jobs cancelled and cancels running sessions
  bool load(std::string* err);  // scheduler thread (or the constructor)
  void publish_model();  // the loaded model, to the telemetry
  void unload();

  std::string model_;
  EngineOptions eo_;
  SchedulerOptions opt_;
  int max_sessions_ = 1;  // the configured limit capped by the current load's device and host budgets
  // Set and reset by the scheduler thread (or the constructor) under unique engine_mu_; other threads hold a shared
  // lock while asking the engine for device figures or compiling a request's grammar so readers never block each other.
  std::unique_ptr<Engine> engine_;
  std::shared_ptr<const interfaces::VisionAdapter> vision_adapter_;  // kept across idle unload for count_tokens()
  std::atomic<bool> has_vision_{false};
  BatchCostModel batch_cost_;
  mutable std::shared_mutex engine_mu_;
  std::shared_ptr<Telemetry> telemetry_;
  double load_s_ = 0;   // the last load took
  std::atomic<int> ctx_{0};  // the loaded engine's context (eo_'s, or the longest that fit)
  // Before prefix_cache_: the snapshots outlive the prefix cache and the engine (a reload of the same model reads
  // them).
  SnapshotStore store_{opt_.host_cache_bytes};
  std::unique_ptr<PrefixCache> prefix_cache_;
  std::string load_error_;
  ModelFile file_;                     // tokenizer and template for HTTP threads, independent of loading
  Tokenizer tok_;
  ChatTemplate chat_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  // The scheduler thread changes it under mu_; other threads read it under mu_.
  std::vector<std::unique_ptr<Running>> running_;
  // Reusable scratch buffers for launch_rounds() (scheduler thread only), avoiding per-round heap allocations.
  std::vector<Generation*> batch_ready_;
  std::vector<std::vector<RoundCandidate>> batch_cands_;
  std::vector<Generation::NextRound> batch_nr_;
  std::vector<flyx_plan*> batch_plans_;
  std::vector<flyx_seq*> batch_seqs_;
  std::vector<const flyx_gen_params*> batch_gens_;
  bool admit_due_ = false;   // the queue or the sessions in use changed since admit() last looked
  bool closed_ = false;      // close(): no new requests
  double admitted_at_ = 0;
  bool stop_ = false;
  double last_active_ = 0;
  std::shared_ptr<Job::CancelGate> gate_ = std::make_shared<Job::CancelGate>(this);
  std::thread thread_;
};

}  // namespace flyx
