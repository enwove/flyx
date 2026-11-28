// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#include "core/os/log.h"
#include "core/os/os.h"

namespace flyx {
namespace {

bool same_message(const ChatMessage& a, const ChatMessage& b) {
  if (a.role != b.role || a.content != b.content || a.reasoning != b.reasoning ||
      a.tool_call_id != b.tool_call_id || a.parts.size() != b.parts.size() ||
      !std::equal(a.tool_calls.begin(), a.tool_calls.end(), b.tool_calls.begin(), b.tool_calls.end(),
                  [](const ToolCall& x, const ToolCall& y) {
                    return x.id == y.id && x.name == y.name && x.arguments == y.arguments;
                  }))
    return false;
  for (size_t i = 0; i < a.parts.size(); ++i) {
    const auto &pa = a.parts[i], &pb = b.parts[i];
    if (pa.kind != pb.kind) return false;
    if (pa.kind == ContentPart::kImage) {
      if (!pa.image != !pb.image) return false;
      if (pa.image && (pa.image->sha256 != pb.image->sha256 || pa.image->detail != pb.image->detail)) return false;
    } else if (pa.text != pb.text) {
      return false;
    }
  }
  return true;
}

bool same_rendered_images_prefix(const RenderedChat& a, const RenderedChat& b, size_t end) {
  if (a.images.empty() && b.images.empty()) return true;
  if (a.images.size() > b.images.size()) return false;
  for (size_t i = 0; i < a.images.size(); ++i) {
    const auto &ia = a.images[i], &ib = b.images[i];
    if (ia.byte_offset != std::string::npos && ia.byte_offset >= end) break;
    if (!ia.image || !ib.image || ia.byte_offset != ib.byte_offset || ia.byte_end != ib.byte_end ||
        ia.image->sha256 != ib.image->sha256 || ia.image->detail != ib.image->detail)
      return false;
  }
  // Ensure b does not inject an extra image before `end`.
  if (a.images.size() < b.images.size() && b.images[a.images.size()].byte_offset != std::string::npos &&
      b.images[a.images.size()].byte_offset < end)
    return false;
  return true;
}

const char* finish_name(int f) {
  switch (f) {
    case FLYX_FINISH_STOP: return "stop";
    case FLYX_FINISH_LENGTH: return "length";
    case FLYX_FINISH_CANCEL: return "cancel";
    default: return "error";
  }
}

std::vector<TelemetrySnapshot::Slot> session_slots(const PrefixCache& cache) {
  std::vector<TelemetrySnapshot::Slot> out;
  for (const auto& s : cache.slots()) out.push_back({s.id, s.tokens});
  return out;
}

}  // namespace

// ---- Job -------------------------------------------------------------------------------------

bool Job::next(JobEvent* ev, int timeout_ms) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return !events_.empty(); })) return false;
  *ev = std::move(events_.front());
  events_.pop_front();
  return true;
}

bool Job::wait_start(JobEvent* ev, int timeout_ms) {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return started_ || !events_.empty(); });
  if (started_ || events_.empty()) return true;
  *ev = std::move(events_.front());  // a job that never started has one event, its error
  events_.pop_front();
  return false;
}

void Job::started() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    started_ = true;
  }
  cv_.notify_all();
}

bool Job::has_started() const {
  std::lock_guard<std::mutex> lock(mu_);
  return started_;
}

bool Job::wait_prompt_usage(flyx_usage* usage, int timeout_ms) {
  std::unique_lock<std::mutex> lock(mu_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return prompt_tokens_ > 0 || finished(); }))
    return false;
  *usage = {};
  usage->prompt_tokens = prompt_tokens_;
  usage->reused_tokens = reused_tokens_;
  return true;
}

void Job::prompt_ready(int tokens, int reused) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    prompt_tokens_ = tokens;
    reused_tokens_ = reused;
  }
  cv_.notify_all();
}

void Job::push(JobEvent ev) {
  if (ev.kind == JobEvent::kDone || ev.kind == JobEvent::kError) finished_.store(true);
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (ttft_ms_ < 0 && (ev.kind == JobEvent::kToolCallStart || ev.kind == JobEvent::kToolCall ||
        (!ev.text.empty() && (ev.kind == JobEvent::kReasoning || ev.kind == JobEvent::kContent))))
      ttft_ms_ = os::now_ms() - arrived_at_;
    events_.push_back(std::move(ev));
  }
  cv_.notify_all();
}

void Job::cancel() {
  cancelled_.store(true);
  if (auto g = gate_.lock()) {
    std::lock_guard<std::mutex> lock(g->mu);
    if (g->sched) g->sched->cancel(this);
  }
}

// ---- Scheduler --------------------------------------------------------------------------------

struct Scheduler::Running {
  std::shared_ptr<Job> job;
  Session* session = nullptr;
  uint64_t assignment = 0;
  std::string why;         // which session it got and why, for the log
  int telemetry_id = 0;    // 0: it never started running
  RequestRecord rec;
  PreparedRequest prepared;
  std::unique_ptr<Generation> gen;
};

Scheduler::Scheduler(const std::string& model, const EngineOptions& eo, const SchedulerOptions& o)
    : model_(model), eo_(eo), opt_(o) {
  if (!opt_.disk_cache_dir.empty() && opt_.disk_cache_bytes > 0) {
    if (!store_.set_disk(opt_.disk_cache_dir, opt_.disk_cache_bytes, &load_error_)) return;
    if (store_.stale_freed() > 0)
      log::info("flyx: removed %s of snapshots in %s that ended flyx processes left behind",
                gb(store_.stale_freed()).c_str(), opt_.disk_cache_dir.c_str());
  }
  if (!file_.open(model, &load_error_, eo_.allow_path) || !tok_.load(file_, &load_error_)) return;
  std::string chat_err;  // as Engine::load: only a template given at start-up must work
  EngineOptions env = eo_;
  apply_environment(&env);
  if (!chat_.load(file_, eo_.chat_template, &chat_err, env.chat_native) && !eo_.chat_template.empty()) {
    load_error_ = chat_err;
    return;
  }
  if (!load(&load_error_)) return;
  last_active_ = os::now_ms();
  thread_ = std::thread([this] { loop(); });
}

bool Scheduler::ok(std::string* err) const {
  if (engine_ || load_error_.empty()) return true;
  *err = load_error_;
  return false;
}

bool Scheduler::load(std::string* err) {
  const double t0 = os::now_ms();
  telemetry_.loading();
  auto e = std::make_unique<Engine>();
  const int wanted = opt_.max_sessions > 0 ? opt_.max_sessions : SchedulerOptions::kDefaultSessions;
  if (!e->load(model_, eo_, err, opt_.max_sessions)) {
    telemetry_.unloaded();
    return false;
  }
  {
    std::unique_lock<std::shared_mutex> lock(engine_mu_);
    max_sessions_ = std::max(1, std::min(wanted, e->sessions_that_fit()));
    const bool has_v = e->has_vision();
    has_vision_.store(has_v);
    vision_adapter_ = has_v ? e->model().vision_shared() : nullptr;
    engine_ = std::move(e);
  }
  batch_cost_.build(engine_->widths(), engine_->round_drafters(), engine_->round_ms(), engine_->lookup_widths(),
                    engine_->lookup_ms());
  if (const int rows = engine_->budget().declined_pair_host_rows)
    log::info("flyx: using one session: two fit, but would read up to %d exact K/V rows from host memory per round "
              "before the switch; --sessions 2 allows that trade", rows);
  else if (max_sessions_ < wanted)
    log::info("flyx: %d sessions %s, %d fit in memory (%s of the GPU's each): using %d", wanted,
              opt_.max_sessions > 0 ? "requested" : "by default", max_sessions_, gb(engine_->session_bytes()).c_str(),
              max_sessions_);

  // Live K/V takes priority over parked checkpoints. An automatic host budget counts memory currently free,
  // so the pieces already held by the store do not need a second reservation after an idle reload.
  uint64_t cache = opt_.host_cache_bytes;
  const uint64_t kv = engine_->session_host_bytes() * uint64_t(max_sessions_);
  if (kv > 0) {
    const uint64_t spare = engine_->host_budget();
    const uint64_t left = spare > kv ? spare - kv : 0;
    cache = std::min(cache, left + (eo_.kv_ram < 0 ? store_.reserved() : 0));
    if (cache < opt_.host_cache_bytes)
      log::info("flyx: host cache %s instead of %s: the K/V of %d session%s of %d tokens take %s of host memory",
                gb(cache).c_str(), gb(opt_.host_cache_bytes).c_str(), max_sessions_, max_sessions_ > 1 ? "s" : "",
                engine_->options().max_ctx, gb(kv).c_str());
  }
  store_.set_budget(cache);
  prefix_cache_ = std::make_unique<PrefixCache>(engine_.get(), max_sessions_, &store_);
  ctx_ = engine_->options().max_ctx;
  load_s_ = (os::now_ms() - t0) / 1000;
  last_active_ = os::now_ms();
  publish_model();
  if (opt_.verbose || thread_.joinable())
    log::info("flyx: model loaded in %.2f s", (os::now_ms() - t0) / 1000);
  return true;
}

void Scheduler::publish_model() {
  ModelInfo m;
  m.device = engine_->device().name;
  m.backend = engine_->backend_name();
  m.drafter = engine_->drafter();
  m.draft_weights = engine_->draft_weights();
  m.chat_format = chat_.format();
  m.chat_ok = chat_.ok();
  m.chat_native = chat_.native();
  m.vision = engine_->has_vision();
  if (m.vision) {
    m.vision_device = engine_->options().vision_device;
    m.vision_quant = engine_->options().vision_quant;
    m.vision_max_tokens = engine_->options().vision_max_tokens;
  }
  m.memory = engine_->memory_report();
  m.width = engine_->width();
  m.max_ctx = max_ctx();
  m.max_sessions = max_sessions_;
  m.ledger_budget = engine_->ledger().budget();
  m.device_reserve = engine_->budget().device_reserve;
  m.load_s = load_s_;
  m.load_adjustments = engine_->load_adjustments();
  m.ctx_limit = engine_->budget().ctx_limit;
  m.wanted_ctx = engine_->wanted_ctx();
  m.kv_gpu_tokens = std::max(0, std::min(engine_->budget().layout.device_positions, m.max_ctx));
  m.ctx_short = engine_->ctx_short();
  m.sessions_room = engine_->budget().sessions_room;
  m.host_budget = engine_->host_budget();
  m.ctx_need_more = engine_->budget().ctx_need_more;
  m.held = engine_->budget().held;
  telemetry_.loaded(m, engine_->ledger().used());
}

void Scheduler::unload() {
  {
    std::unique_lock<std::shared_mutex> lock(engine_mu_);
    if (prefix_cache_) prefix_cache_->clear();  // the conversations' checkpoints stay in host memory for the next load
    prefix_cache_.reset();
    store_.own_memory();  // in memory of its own: the backend's goes with the engine
    engine_.reset();
  }
  telemetry_.unloaded();
}

void Scheduler::telemetry(TelemetrySnapshot* out, flyx_device_stats* device, bool* device_ok) const {
  telemetry_.snapshot(out);
  out->queued = queued();
  std::shared_lock<std::shared_mutex> lock(engine_mu_);
  *device_ok = engine_ && engine_->device_stats(device);
}

void Scheduler::cancel_all_locked(bool closing) {
  for (auto& j : queue_) {
    if (!j->cancelled_.exchange(true) && closing) j->closed_.store(true);
  }
  for (auto& r : running_) {
    if (!r->job->cancelled_.exchange(true) && closing) r->job->closed_.store(true);
    r->session->cancel();
  }
}

Scheduler::~Scheduler() {
  {
    std::lock_guard<std::mutex> glock(gate_->mu);
    gate_->sched = nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
    cancel_all_locked();
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void Scheduler::close() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    closed_ = true;
    cancel_all_locked(/*closing=*/true);
    admit_due_ = true;
  }
  cv_.notify_all();
}

int Scheduler::queued() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<int>(queue_.size());
}

void Scheduler::cancel(Job* job) {
  job->cancelled_.store(true);
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& r : running_)
      if (r->job.get() == job) r->session->cancel();
    admit_due_ = true;  // a queued job: its error goes out at the next look
  }
  cv_.notify_all();
}

int Scheduler::count_tokens(const Request& in, std::string* err) {
  RenderedChat rc;
  if (!render_request(chat_, tok_, in, &rc, err, false)) return -1;
  const std::vector<int32_t> base = tok_.encode(rc.text, &rc.literal);
  int total = static_cast<int>(base.size());
  if (!rc.images.empty()) {
    std::shared_ptr<const interfaces::VisionAdapter> va;
    int budget_ctx = max_ctx();
    int vmax = eo_.vision_max_tokens;
    {
      std::shared_lock<std::shared_mutex> lock(engine_mu_);
      va = vision_adapter_;
      if (engine_) {
        budget_ctx = engine_->options().max_ctx;
        vmax = engine_->options().vision_max_tokens;
      }
    }
    if (!va) {
      *err = "this model has no vision encoder loaded (pass --mmproj <mmproj.gguf>)";
      return -1;
    }
    bool exact_offsets = true;
    for (const auto& ri : rc.images)
      if (ri.byte_offset == std::string::npos || ri.byte_end == std::string::npos || ri.byte_end < ri.byte_offset ||
          ri.byte_end > rc.text.size())
        exact_offsets = false;
    if (!exact_offsets) {
      const int32_t ph = va->placeholder_token();
      size_t found = 0;
      for (int32_t id : base)
        if (id == ph) ++found;
      if (found != rc.images.size()) {
        *err = "chat template emitted " + std::to_string(found) + " image placeholders for " +
               std::to_string(rc.images.size()) + " images";
        return -1;
      }
    }
    for (const auto& ri : rc.images) {
      if (!ri.image) return *err = "missing image data in multimodal message", -1;
      interfaces::RgbImage rgb;
      if (!vision::decode_image_bytes(ri.image->bytes.data(), ri.image->bytes.size(), &rgb, err)) return -1;
      const int budget = Engine::image_token_budget_for(budget_ctx, vmax, ri.image->detail);
      interfaces::PreparedMedia pm;
      if (!va->preprocess(rgb, budget, &pm, err)) return -1;
      total += static_cast<int>(pm.tokens.size()) - 1;  // replaces the single placeholder token
    }
  }
  return total;
}

std::shared_ptr<Job> Scheduler::submit(Request in, std::string* err, int* status, double arrived_at) {
  if (arrived_at < 0) arrived_at = os::now_ms();
  auto job = std::make_shared<Job>();
  job->arrived_at_ = arrived_at;
  job->gate_ = gate_;
  if (!render_request(chat_, tok_, in, &job->prompt_, err)) {
    *status = 400;
    return nullptr;
  }
  job->in_ = std::move(in);
  const bool prewarm = job->in_.gen.prefill_only;
  // The reply's grammar, compiled on the caller, outside the thread serving other requests' rounds.  Whether a reply
  // has one is its format's to say (tools, a JSON answer, thinking switched off where only a grammar keeps it out):
  // warm_request asks it, and compiles nothing for a reply that has none.
  auto warm = [&] {
    std::shared_lock<std::shared_mutex> lock(engine_mu_);
    if (engine_) warm_request(*engine_, job->in_, job->prompt_);
  };
  if (!prewarm) warm();
  job->queued_at_ = os::now_ms();
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (closed_) {
      *err = "the server is changing models";
      *status = 503;
      return nullptr;
    }
    if (static_cast<int>(queue_.size()) >= opt_.queue_limit) {
      *err = "server busy: " + std::to_string(queue_.size()) + " requests waiting";
      *status = 503;
      return nullptr;
    }
    // The same conversation again (a retry, or a follow-up sent without
    // waiting for the reply): the older request's output is no longer wanted.
    // Where the template gives the end of the conversation, compare up to it. A template with a variable
    // generation prompt may not give that boundary: compare the actual turns in the same template context.
    auto same = [&](const std::shared_ptr<Job>& other) {
      const Request& a = other->in_;
      const Request& b = job->in_;
      const bool identified = !a.conversation_id.empty() && !b.conversation_id.empty();
      if (identified && a.conversation_id != b.conversation_id) return false;
      if (b.gen.prefill_only && !a.gen.prefill_only) return false;
      // A raw prompt is the caller's own text: its bytes say nothing of who sent it, so two clients whose prompts
      // are the same, or one of which extends the other's, sent two requests.  Without a shared conversation id a
      // raw prompt replaces only a raw prewarm (no reply anyone waits for; it takes over the prewarm's work), and
      // neither replaces nor is replaced by a chat request.
      if ((!a.chat || !b.chat) && !identified && (a.chat != b.chat || !a.gen.prefill_only)) return false;
      if (const size_t end = other->prompt_.conversation_end)
        return job->prompt_.text.compare(0, end, other->prompt_.text, 0, end) == 0 &&
               same_rendered_images_prefix(other->prompt_, job->prompt_, end);
      return a.chat && b.chat && a.messages.size() <= b.messages.size() && a.tools == b.tools &&
             a.chat_options.enable_thinking == b.chat_options.enable_thinking &&
             a.chat_options.add_generation_prompt == b.chat_options.add_generation_prompt &&
             a.chat_options.kwargs == b.chat_options.kwargs &&
             std::equal(a.messages.begin(), a.messages.end(), b.messages.begin(), same_message);
    };
    auto replace = [&](const std::shared_ptr<Job>& other) {
      if (other->cancelled_.load() || !same(other)) return false;
      if (other->cancelled_.exchange(true)) return false;  // a disconnect can cancel outside mu_ during the comparison
      other->superseded_.store(true);
      return true;
    };
    for (auto& r : running_)
      if (replace(r->job)) {
        r->session->cancel();
        if (r->job->in_.gen.prefill_only) job->resume_assignment_ = r->assignment;
        if (opt_.verbose)
          log::info("flyx: a new request for the same conversation cancels the running one");
      }
    for (auto& q : queue_)
      if (replace(q) && !job->resume_assignment_) job->resume_assignment_ = q->resume_assignment_;
    queue_.push_back(job);
    admit_due_ = true;
  }
  cv_.notify_all();
  // A prompt prewarm needs no reply grammar itself. Compile for its next turn while its prompt is running.
  if (prewarm) warm();
  return job;
}

void Scheduler::loop() {
  std::unique_lock<std::mutex> lock(mu_);
  while (true) {
    if (running_.empty()) {
      cv_.wait_for(lock, std::chrono::seconds(1), [&] { return stop_ || !queue_.empty(); });
      if (stop_) break;
      if (queue_.empty()) {
        const double idle = os::now_ms() - last_active_;
        if (opt_.idle_unload_ms > 0 && prefix_cache_ && prefix_cache_->size() > 0 && idle > opt_.idle_unload_ms) {
          const uint64_t before = engine_->ledger().used();
          prefix_cache_->clear();
          telemetry_.memory(engine_->ledger().used(), engine_->host_mapped_in_use(), {});
          log::info("flyx: idle for %d s: sessions freed (%s)", opt_.idle_unload_ms / 1000,
                    gb(before - engine_->ledger().used()).c_str());
        }
        if (opt_.keep_alive_ms > 0 && engine_ && idle > opt_.keep_alive_ms) {
          const uint64_t before = engine_->ledger().used();
          unload();
          log::info("flyx: idle for %d s: model unloaded (%s of GPU memory freed)",
                    opt_.keep_alive_ms / 1000, gb(before).c_str());
        }
        continue;
      }
      admit_due_ = true;
    }
    // Queued requests start when the queue or the sessions in use changed, and
    // now and then for the ones that have waited too long.  Shutting down:
    // nothing new starts; the running requests were cancelled and end at
    // their next step.
    if (!stop_ && (admit_due_ || os::now_ms() - admitted_at_ > 1000)) admit(lock);
    if (running_.empty()) continue;

    lock.unlock();
    advance_all();
    lock.lock();
    std::vector<std::unique_ptr<Running>> over;
    for (size_t i = 0; i < running_.size();) {
      if (running_[i]->gen->done()) {
        over.push_back(std::move(running_[i]));
        running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(i));
      } else {
        ++i;
      }
    }
    if (over.empty()) continue;
    lock.unlock();
    for (auto& r : over) finish(*r);
    over.clear();  // their sessions are free again
    lock.lock();
    admit_due_ = true;
    last_active_ = os::now_ms();
  }
  for (auto& j : queue_) {
    JobEvent ev{JobEvent::kError, "server shutting down", {}, {}, 0};
    ev.unavailable = true;
    j->push(std::move(ev));
  }
  queue_.clear();
}

void Scheduler::admit(std::unique_lock<std::mutex>& lock) {
  admit_due_ = false;
  admitted_at_ = os::now_ms();
  // In queue order.  A job whose session is running another request stays
  // where it is and does not hold up the ones behind it.  Only this thread
  // takes jobs out of the queue, so an index stays good while mu_ is released.
  for (size_t i = 0; i < queue_.size();) {
    std::shared_ptr<Job> job = queue_[i];
    const bool cancelled = job->cancelled_.load();
    const double waited_ms = os::now_ms() - job->queued_at_;
    if (cancelled || waited_ms > opt_.queue_timeout_ms) {
      queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(i));
      const bool closing = job->closed_.load();
      const char* reason =
          closing ? "the server is changing models" : cancelled ? "cancelled" : "timed out waiting in the queue";
      RequestRecord rec;
      rec.label = job->in_.label;
      rec.response_id = job->in_.response_id;
      rec.finished_at = os::unix_seconds();
      rec.queue_ms = waited_ms;
      rec.status = cancelled ? "cancelled" : "error";
      rec.superseded = cancelled && job->superseded_.load();
      if (!cancelled) rec.error = reason;
      JobEvent refused{JobEvent::kError, reason, {}, {}, 0};
      refused.superseded = rec.superseded;
      refused.unavailable = !cancelled || closing;
      telemetry_.finish(0, std::move(rec));
      job->push(std::move(refused));
      continue;
    }
    if (static_cast<int>(running_.size()) >= max_sessions_) {
      ++i;  // no slot to start this job, but later cancellations do not need one
      continue;
    }
    lock.unlock();
    std::unique_ptr<Running> r;
    const Start how = start(job, &r);
    lock.lock();
    if (how == Start::kWaits) {
      if (stop_) return;
      ++i;
      continue;
    }
    queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(i));
    if (how == Start::kRunning) {
      if (r->job->cancelled_.load()) r->session->cancel();
      running_.push_back(std::move(r));
    }
    if (stop_) return;
  }
}

void Scheduler::failed(Running& r, JobEvent ev) {
  r.rec.finished_at = os::unix_seconds();
  r.rec.ttft_ms = r.job->ttft_ms_;
  r.rec.status = "error";
  r.rec.error = ev.text;
  r.job->push(std::move(ev));
  telemetry_.finish(r.telemetry_id, r.rec);
}

void Scheduler::forward(Job& job, std::vector<OutputEvent> events) {
  for (auto& e : events) {
    JobEvent je{};
    switch (e.kind) {
      case OutputEvent::kReasoning: je.kind = JobEvent::kReasoning; break;
      case OutputEvent::kContent: je.kind = JobEvent::kContent; break;
      case OutputEvent::kToolCallStart: je.kind = JobEvent::kToolCallStart; break;
      case OutputEvent::kToolCallDelta: je.kind = JobEvent::kToolCallDelta; break;
      case OutputEvent::kToolCall: je.kind = JobEvent::kToolCall; break;
    }
    je.text = std::move(e.text);
    je.call = std::move(e.call);
    job.push(std::move(je));
  }
}

Scheduler::Start Scheduler::start(const std::shared_ptr<Job>& job, std::unique_ptr<Running>* out) {
  const Request& in = job->in_;
  auto r = std::make_unique<Running>();
  r->job = job;
  r->rec.label = in.label;
  r->rec.response_id = in.response_id;
  r->rec.queue_ms = os::now_ms() - job->queued_at_;
  std::string err;
  if (!engine_) {  // unloaded while idle: load again
    if (!load(&err)) return failed(*r, "loading the model: " + err), Start::kOver;
  }
  PrefixCache::Choice choice;
  Session* s = prefix_cache_->acquire(job->prompt_, &choice, &err, job->resume_assignment_);
  if (!s && choice.busy) return Start::kWaits;
  if (!s) return failed(*r, "no session: " + err), Start::kOver;
  r->session = s;
  r->assignment = choice.assignment;
  r->rec.session = choice.id;
  r->why = std::move(choice.reason);
  if (!prepare_request(*engine_, *s, in, job->prompt_, &r->prepared, &err)) {
    JobEvent ev{JobEvent::kError, err, {}, {}, 0};
    ev.invalid_request = r->prepared.invalid;
    return failed(*r, std::move(ev)), Start::kOver;
  }
  if (!r->prepared.unconstrained.empty())
    log::warn("flyx: %s: %s; tool calls are not constrained", in.label.c_str(),
              r->prepared.unconstrained.c_str());
  GenerateRequest& gen = r->prepared.gen;
  // A prompt the context cannot take is the request's error, told before anything runs (the generation would find the
  // same), with the numbers a client needs to shorten it.
  const int max_ctx = engine_->options().max_ctx, n = static_cast<int>(gen.prompt.size());
  if (!flyx_room(n, max_ctx)) {
    JobEvent ev{JobEvent::kError, prompt_too_long(n, max_ctx), {}, {}, 0};
    ev.prompt_tokens = n;
    ev.prompt_limit = prompt_limit(max_ctx);
    log::warn("flyx: %s refused: %s", in.label.c_str(), ev.text.c_str());
    return failed(*r, std::move(ev)), Start::kOver;
  }
  gen.park_previous = choice.park;
  if (max_sessions_ > 1) {
    const auto after = std::upper_bound(gen.cache_prefixes.begin(), gen.cache_prefixes.end(), gen.system_end);
    if (after != gen.cache_prefixes.begin()) gen.share_system_prefix = *(after - 1);
  }
  gen.cancel = &job->cancelled_;
  r->rec.prompt_tokens = static_cast<int>(gen.prompt.size());
  const int id = r->telemetry_id = telemetry_.start(in.label, r->rec.session, r->rec.prompt_tokens, gen.max_new_tokens,
                                                    in.response_id);
  // The session taken holds its device memory from now: monitors count it as flyx's, not another program's.
  telemetry_.memory(engine_->ledger().used(), engine_->host_mapped_in_use(), session_slots(*prefix_cache_));
  gen.on_prefill = [this, id, job, first = true](int done, int total) mutable {
    if (first) {
      job->prompt_ready(total, done);  // the first progress update counts the tokens the session reused
      first = false;
    }
    telemetry_.prefill(id, done, total);
  };
  gen.on_round = [this, id](int drafted, int accepted) { telemetry_.round(id, drafted, accepted); };
  Running* self = r.get();
  TextSink sink = [this, self](std::string_view text, const int32_t*, int n) {
    if (n > 0) telemetry_.tokens(self->telemetry_id, n);
    if (self->prepared.parser) forward(*self->job, self->prepared.parser->feed(text));
    else if (!text.empty()) self->job->push({JobEvent::kContent, std::string(text), {}, {}, 0});
    return !self->job->cancelled_.load();
  };
  r->gen = std::make_unique<Generation>(s, gen, std::move(sink));
  job->started();
  *out = std::move(r);
  return Start::kRunning;
}

void Scheduler::advance_all() {
  if (running_.size() == 1) {  // alone: wait for the device, as a request always did
    running_.front()->gen->set_batched(false);
    running_.front()->gen->set_alone(true);
    running_.front()->gen->advance(true);
    return;
  }
  // Several: the rounds they would launch now go out together, the weights read once for all of them;
  // then each does what the device is ready for, none waited for.
  for (auto& r : running_) r->gen->set_alone(false);  // the time between its reads is others' rounds' too
  bool worked = opt_.batch_rounds && launch_rounds();
  for (auto& r : running_)
    if (!r->gen->done() && r->gen->advance(false) != Generation::Progress::kBusy) worked = true;
  // The device is working for every one of them: look again shortly.
  if (!worked) os::sleep_us(100);  // on Windows a high-resolution timer, not the OS scheduler's 1 to 16 ms tick
}

// Every running generation's next round, jointly scheduled across sessions into batches of at most FLYX_MAX_ROWS
// rows (core/speculation/batch_schedule.h). In step: they go out when every generation in its round
// phase has one to launch (one whose rounds in flight are at its limit reads one soon), so they share rounds rather
// than each keeping its own two in flight at its own times.
bool Scheduler::launch_rounds() {
  for (auto& r : running_) {
    r->gen->set_batched(true);
    if (!r->gen->done() && r->gen->in_rounds() && !r->gen->round_ready()) return false;
  }
  batch_ready_.clear();
  size_t idx = 0;
  for (auto& r : running_) {
    if (r->gen->done()) continue;
    if (idx == batch_cands_.size()) batch_cands_.emplace_back();
    batch_cands_[idx].clear();
    if (r->gen->round_candidates(&batch_cands_[idx])) {
      batch_ready_.push_back(r->gen.get());
      ++idx;
    }
  }
  if (idx == 0) return false;
  batch_cands_.resize(idx);
  const auto batches = plan_batches(batch_cost_, batch_cands_);
  const auto* api = engine_->api();
  for (const auto& b : batches) {
    const size_t n = b.items.size();
    batch_nr_.resize(n);
    batch_plans_.resize(n);
    batch_seqs_.resize(n);
    batch_gens_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      Generation* g = batch_ready_[b.items[i].session];
      g->commit_round(b.items[i].choice, &batch_nr_[i]);
      batch_plans_[i] = batch_nr_[i].plan;
      batch_seqs_[i] = g->session()->seq();
      batch_gens_[i] = &batch_nr_[i].g;
    }
    // A batch the backend refuses (it checks before it runs anything): each round on its own.
    const bool ok = api->launch_batch(batch_plans_.data(), batch_seqs_.data(), batch_gens_.data(),
                                      static_cast<int32_t>(n)) == 0;
    for (size_t i = 0; i < n; ++i) {
      Generation* g = batch_ready_[b.items[i].session];
      if (ok || (n > 1 && api->launch_batch(&batch_plans_[i], &batch_seqs_[i], &batch_gens_[i], 1) == 0))
        g->launched(batch_nr_[i]);
      else
        g->launch_failed(api->last_error());
    }
  }
  return true;
}

void Scheduler::finish(Running& r) {
  const Request& in = r.job->in_;
  if (r.gen->failed()) {
    log::error("flyx: %s failed: %s", in.label.c_str(), r.gen->error().c_str());
    failed(r, r.gen->error());
    telemetry_.memory(engine_->ledger().used(), engine_->host_mapped_in_use(), session_slots(*prefix_cache_));
    return;
  }
  const GenerateResult& res = r.gen->result();
  if (!res.parked.empty()) r.why += "; its conversation: " + res.parked;
  OutputParser* parser = r.prepared.parser.get();
  if (parser) forward(*r.job, parser->finish());
  const int tool_calls = parser ? parser->tool_calls() : 0;
  const auto& u = res.usage;
  const std::string looked_up = res.lookup_rounds ? " (" + std::to_string(res.lookup_rounds) + " lookup rounds, " +
                                                        std::to_string(res.lookup_accepted) + " drafts accepted)"
                                                  : "";
  if (opt_.verbose || u.finish_reason == FLYX_FINISH_CANCEL)
    log::info("flyx: %s | %s | prompt %d, reused %d (%s%s%s) | prefill %.0f ms | %d tokens, %.1f tok/s, %.2f "
              "tokens/round%s | %s%s%s",
              in.label.c_str(), r.why.c_str(), u.prompt_tokens, u.reused_tokens, res.reuse.c_str(),
              r.prepared.tokenized.empty() ? "" : "; ", r.prepared.tokenized.c_str(), u.prefill_ms, u.new_tokens,
              u.decode_ms > 0 && u.new_tokens > 1 ? (u.new_tokens - 1) * 1000.0 / u.decode_ms : 0.0,
              u.rounds ? double(u.new_tokens - 1) / u.rounds : 0.0, looked_up.c_str(), finish_name(u.finish_reason),
              tool_calls ? (", " + std::to_string(tool_calls) + " tool calls").c_str() : "",
              r.prepared.grammar ? (" | grammar " + std::to_string(int(r.prepared.grammar_ms + 0.5)) + " ms" +
                                    (r.prepared.grammar_cached ? " (compiled ahead)" : "") + ", " +
                                    std::to_string(res.grammar_fixes) + " tokens corrected").c_str()
                                 : "");
  RequestRecord& rec = r.rec;
  rec.finished_at = os::unix_seconds();
  rec.status = u.finish_reason == FLYX_FINISH_CANCEL   ? "cancelled"
               : u.finish_reason == FLYX_FINISH_LENGTH ? "length"
               : tool_calls > 0                           ? "tool_calls"
                                                          : "stop";
  rec.prompt_tokens = u.prompt_tokens;
  rec.reused_tokens = u.reused_tokens;
  rec.output_tokens = u.new_tokens + (res.stop_token >= 0 ? 1 : 0);
  rec.ttft_ms = r.job->ttft_ms_;
  rec.prefill_ms = u.prefill_ms;
  rec.decode_ms = u.decode_ms;
  rec.tokens_per_round = u.rounds ? double(u.new_tokens - 1) / u.rounds : 0.0;
  rec.drafted = u.drafted;
  rec.accepted_drafts = u.accepted_drafts;
  rec.tool_calls = tool_calls;
  rec.grammar_fixes = res.grammar_fixes;
  rec.superseded = u.finish_reason == FLYX_FINISH_CANCEL && r.job->superseded_.load();
  telemetry_.finish(r.telemetry_id, rec);
  if (opt_.on_finished) opt_.on_finished(rec);
  telemetry_.memory(engine_->ledger().used(), engine_->host_mapped_in_use(), session_slots(*prefix_cache_));
  JobEvent done{};
  done.kind = JobEvent::kDone;
  done.result = res;
  done.tool_calls = tool_calls;
  done.queue_ms = rec.queue_ms;
  done.tokens_per_round = rec.tokens_per_round;
  done.superseded = rec.superseded;
  r.job->push(std::move(done));
}

}  // namespace flyx
