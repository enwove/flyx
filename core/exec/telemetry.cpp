// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/telemetry.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "core/os/os.h"
#include "flyx/backend_abi.h"

namespace flyx {
namespace {

constexpr size_t kRecent = 100;     // finished requests kept
constexpr double kWindowMs = 2000;  // the "now" speed

}  // namespace

std::string clock_limit_names(uint32_t bits) {
  static const std::pair<uint32_t, const char*> kNames[] = {
      {FLYX_CLOCK_POWER_CAP, "power cap"},     {FLYX_CLOCK_THERMAL, "thermal"},
      {FLYX_CLOCK_HW_SLOWDOWN, "hw slowdown"}, {FLYX_CLOCK_APP_SETTING, "app clocks"},
      {FLYX_CLOCK_SYNC_BOOST, "sync boost"},   {FLYX_CLOCK_DISPLAY, "display"},
      {FLYX_CLOCK_IDLE, "idle"}};
  std::string out;
  for (const auto& [bit, name] : kNames)
    if (bits & bit) out += (out.empty() ? "" : ", ") + std::string(name);
  return out;
}

Telemetry::Telemetry() {
  s_.started_at = os::unix_seconds();
  last_second_ = int64_t(os::now_ms() / 1000);
}

void Telemetry::advance(int64_t second) {
  if (second <= last_second_) return;
  for (int64_t k = last_second_ + 1; k <= second && k <= last_second_ + kSeconds; ++k) seconds_[k % kSeconds] = {};
  last_second_ = second;
}

void Telemetry::loading() {
  std::lock_guard<std::mutex> lock(mu_);
  s_.phase = TelemetrySnapshot::kLoading;
}

void Telemetry::loaded(const ModelInfo& m, uint64_t ledger_used) {
  std::lock_guard<std::mutex> lock(mu_);
  s_.loaded = true;
  s_.loaded_at = os::unix_seconds();
  s_.model = m;
  s_.ledger_used = ledger_used;
  if (s_.phase == TelemetrySnapshot::kUnloaded || s_.phase == TelemetrySnapshot::kLoading)
    s_.phase = TelemetrySnapshot::kIdle;
}

void Telemetry::unloaded() {
  std::lock_guard<std::mutex> lock(mu_);
  s_.loaded = false;
  s_.loaded_at = 0;
  s_.phase = TelemetrySnapshot::kUnloaded;
  s_.ledger_used = 0;
  s_.host_mapped = 0;
  s_.sessions.clear();
}

void Telemetry::memory(uint64_t ledger_used, uint64_t host_mapped, std::vector<TelemetrySnapshot::Slot> sessions) {
  std::lock_guard<std::mutex> lock(mu_);
  s_.ledger_used = ledger_used;
  s_.host_mapped = host_mapped;
  s_.sessions = std::move(sessions);
}

Telemetry::Running* Telemetry::find(int id) {
  for (auto& r : running_)
    if (r.id == id) return &r;
  return nullptr;
}

int Telemetry::start(const std::string& label, int session, int prompt_tokens, int max_new_tokens, std::string response_id) {
  std::lock_guard<std::mutex> lock(mu_);
  Running r;
  r.id = next_id_++;
  r.label = label;
  r.response_id = std::move(response_id);
  r.session = session;
  r.prompt_tokens = prompt_tokens;
  r.max_new_tokens = max_new_tokens;
  r.started_ms = os::now_ms();
  r.prefill_at_ms = r.started_ms;
  running_.push_back(std::move(r));
  return running_.back().id;
}

Telemetry::Second& Telemetry::second_now(double now_ms) {
  advance(int64_t(now_ms / 1000));
  return seconds_[last_second_ % kSeconds];
}

void Telemetry::computed(int n, double start_ms, double end_ms) {
  if (n <= 0) return;
  if (end_ms <= start_ms) {
    second_now(end_ms).pre += n;
    return;
  }
  const int64_t s0 = std::max<int64_t>(0, int64_t(start_ms / 1000.0));
  const int64_t s1 = std::max<int64_t>(0, int64_t(end_ms / 1000.0));
  advance(s1);
  if (s0 >= s1) {
    seconds_[s1 % kSeconds].pre += n;
    return;
  }
  // Spread the pass's tokens proportionally across the whole seconds it overlapped; cumulative rounding preserves sum n.
  const double span = end_ms - start_ms;
  int prev_cum = 0;
  for (int64_t s = s0; s <= s1; ++s) {
    const double hi = std::min(end_ms, double(s + 1) * 1000.0);
    const int cum = s == s1 ? n : std::clamp(int(std::lround(double(n) * (hi - start_ms) / span)), prev_cum, n);
    const int share = cum - prev_cum;
    prev_cum = cum;
    if (s > last_second_ - kSeconds) seconds_[s % kSeconds].pre += share;
  }
}

void Telemetry::prefill(int id, int done, int total) {
  const double now = os::now_ms();
  std::lock_guard<std::mutex> lock(mu_);
  Running* r = find(id);
  if (!r) return;
  if (r->reused_tokens < 0) {
    r->reused_tokens = done;
    r->prefill_at_ms = now;
  } else {
    if (r->pending_prefill > 0) computed(r->pending_prefill, r->prefill_at_ms, now);
    r->pending_prefill = std::max(0, done - r->prefilled);
    r->prefill_at_ms = now;
  }
  r->prefilled = done;
  r->prompt_tokens = total;
}

void Telemetry::tokens(int id, int n) {
  const double now = os::now_ms();
  std::lock_guard<std::mutex> lock(mu_);
  Running* r = find(id);
  if (!r) return;
  if (!r->decoding) {
    r->decoding = true;
    r->first_delivery_ms = now;
    r->first_delivery_tokens = n;
    const int extra = r->reused_tokens >= 0 ? std::max(0, r->prompt_tokens - r->prefilled) : 0;
    const int flush = r->pending_prefill + extra;
    if (flush > 0) computed(flush, r->prefill_at_ms > 0 ? r->prefill_at_ms : r->started_ms, now);
    r->pending_prefill = 0;
    r->prefilled = r->prompt_tokens;
    if (decoding_++ == 0) {  // the "now" speed starts over when decoding does
      points_.clear();
      points_.emplace_back(now, produced_);
    }
  }
  r->generated += n;
  ++r->deliveries;
  produced_ += n;
  second_now(now).out += n;
  points_.emplace_back(now, produced_);
  while (points_.size() > 2 && points_[1].first < now - kWindowMs) points_.pop_front();
}

void Telemetry::round(int id, int drafted, int accepted) {
  const double now = os::now_ms();
  std::lock_guard<std::mutex> lock(mu_);
  if (Running* r = find(id)) {
    ++r->rounds;
    r->drafted += drafted;
    r->accepted += accepted;
  }
  Second& s = second_now(now);
  ++s.rounds;
  s.drafted += drafted;
  s.accepted += accepted;
  round_points_.push_back({now, drafted, accepted});
  while (round_points_.size() > 1 && round_points_.front().time_ms < now - kWindowMs) round_points_.pop_front();
}

void Telemetry::finish(int id, RequestRecord r) {
  const double now = os::now_ms();
  std::lock_guard<std::mutex> lock(mu_);
  for (size_t i = 0; i < running_.size(); ++i)
    if (running_[i].id == id) {
      if (running_[i].pending_prefill > 0) {
        computed(running_[i].pending_prefill,
                 running_[i].prefill_at_ms > 0 ? running_[i].prefill_at_ms : running_[i].started_ms, now);
        running_[i].pending_prefill = 0;
      }
      if (running_[i].decoding && --decoding_ == 0) points_.clear();
      running_.erase(running_.begin() + static_cast<std::ptrdiff_t>(i));
      if (decoding_ == 0 &&
          std::none_of(running_.begin(), running_.end(), [](const Running& q) { return q.rounds > 0; }))
        round_points_.clear();
      break;
    }
  auto& t = s_.totals;
  ++t.requests;
  t.errors += r.status == "error";
  t.cancelled += r.status == "cancelled";
  t.prompt_tokens += r.prompt_tokens;
  t.reused_tokens += r.reused_tokens;
  t.output_tokens += r.output_tokens;
  t.tool_calls += r.tool_calls;
  t.grammar_fixes += r.grammar_fixes;
  recent_.push_front(std::move(r));
  if (recent_.size() > kRecent) recent_.pop_back();
}

void Telemetry::snapshot(TelemetrySnapshot* out) const {
  const double now = os::now_ms();
  std::lock_guard<std::mutex> lock(mu_);
  *out = s_;
  out->running = static_cast<int>(running_.size());
  out->elapsed_ms = 0;
  out->decode_tok_s = 0;
  out->window_tokens_per_round = -1;
  out->window_acceptance = -1;
  out->tokens_per_round = 0;
  if (!running_.empty()) {  // the request that started first
    const Running& r = running_.front();
    out->phase = r.decoding ? TelemetrySnapshot::kDecode : TelemetrySnapshot::kPrefill;
    out->label = r.label;
    out->response_id = r.response_id;
    out->session = r.session;
    out->prompt_tokens = r.prompt_tokens;
    out->reused_tokens = r.reused_tokens;
    out->prefilled = r.prefilled;
    out->generated = r.generated;
    out->max_new_tokens = r.max_new_tokens;
    out->elapsed_ms = now - r.started_ms;
    out->tokens_per_round = r.rounds > 0 ? 1.0 + double(r.accepted) / r.rounds
                                         : (r.deliveries > 0 ? double(r.generated) / r.deliveries : 0);
    out->rounds = r.rounds;
    out->drafted = r.drafted;
    out->accepted = r.accepted;
    if (r.deliveries > 1 && now > r.first_delivery_ms)
      out->request_decode_tok_s = (r.generated - r.first_delivery_tokens) * 1000.0 / (now - r.first_delivery_ms);
    const double until = r.decoding ? r.first_delivery_ms : now;
    if (r.reused_tokens >= 0 && r.prefilled > r.reused_tokens && until > r.started_ms)
      out->request_prefill_tok_s = (r.prefilled - r.reused_tokens) * 1000.0 / (until - r.started_ms);
    // Keep active sessions' token counts live as prefill restores/advances and decode emits tokens.
    for (const auto& run : running_) {
      if (run.reused_tokens < 0) continue;
      for (auto& sl : out->sessions)
        if (sl.id == run.session) {
          sl.tokens = run.prefilled + run.generated;
          break;
        }
    }
  }
  if (decoding_ > 0 && !points_.empty()) {
    // Tokens since the last point before the window, over the window (or the decode so far).
    const double from = std::max(now - kWindowMs, points_.front().first);
    int64_t base = points_.front().second;
    for (const auto& p : points_)
      if (p.first <= from) base = p.second;
    if (now - from > 50) out->decode_tok_s = (produced_ - base) * 1000.0 / (now - from);
  }
  if (decoding_ > 0 && !round_points_.empty()) {
    const double from = now - kWindowMs;
    int win_rounds = 0, win_drafted = 0, win_accepted = 0;
    for (const auto& p : round_points_) {
      if (p.time_ms < from) continue;
      ++win_rounds;
      win_drafted += p.drafted;
      win_accepted += p.accepted;
    }
    if (win_rounds == 0) {
      const auto& last = round_points_.back();
      win_rounds = 1;
      win_drafted = last.drafted;
      win_accepted = last.accepted;
    }
    out->window_tokens_per_round = 1.0 + double(win_accepted) / win_rounds;
    if (win_drafted > 0) out->window_acceptance = std::min(1.0, double(win_accepted) / win_drafted);
  }
  const int64_t current = int64_t(now / 1000);
  for (auto* v : {&out->output_per_second, &out->prefill_per_second, &out->rounds_per_second, &out->drafted_per_second,
                  &out->accepted_per_second})
    v->assign(60, 0);
  for (int i = 0; i < 60; ++i) {
    const int64_t sec = current - 60 + i;  // negative in a machine's first minute (the steady clock counts from boot)
    if (sec < 0 || sec > last_second_ || sec <= last_second_ - kSeconds) continue;
    const Second& s = seconds_[sec % kSeconds];
    const size_t at = size_t(i);
    out->output_per_second[at] = s.out;
    out->prefill_per_second[at] = s.pre;
    out->rounds_per_second[at] = s.rounds;
    out->drafted_per_second[at] = s.drafted;
    out->accepted_per_second[at] = s.accepted;
  }
  out->recent.assign(recent_.begin(), recent_.end());
}

}  // namespace flyx
