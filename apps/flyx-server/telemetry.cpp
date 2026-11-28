// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GET /v1/telemetry: what the server is doing and has done, for monitors
// (`flyx monitor`, scripts). Read-only, versioned by schema_version; a
// figure the server cannot know is null. ?recent=N: the last N finished
// requests (default 20, at most 100).
//
// Schema 1: server, model, state (phase idle | loading | unloaded | prefill
// | decode, how many requests are running, and of those the one that started
// first), rates (decode tokens per second, tokens per round and draft
// acceptance now over every running request; tokens out, prompt tokens
// computed, rounds verified, their drafts and the drafts accepted per second
// over the last minute),
// memory (the engine's ledger, flyx's own CUDA runtime beside it where the
// system counts a process's share, the K/V's host memory mapped for the device
// only, resident sessions), device (load, memory, temperature, power, clocks,
// PCIe, the process's page-locked host memory where the system counts it; null
// without a GPU), process (CPU seconds, resident bytes, its RAM in all), system (RAM, CPUs),
// totals, recent (finished requests, newest first), load (the model being fetched or loaded, state
// none | fetching | loading | ready | failed, progress, error; model is null while none is served).
#include <algorithm>

#include "apps/common/chat_options.h"
#include "apps/common/version.h"
#include "apps/flyx-server/agent_launch.h"
#include "apps/flyx-server/server.h"
#include "core/os/os.h"
#include "core/plan/budget.h"

namespace flyx::server {

namespace {

// What a request that names no thinking mode gets, as inference resolves it from the server's defaults
// (resolve_chat_options: --no-think, --chat-template-kwargs' enable_thinking, a reasoning_effort of "none"); null where
// the defaults say nothing of either, the chat template's own default then deciding.
ojson thinking_default(const ojson& defaults) {
  if (!defaults.is_object() || (!defaults.contains("enable_thinking") && !defaults.contains("reasoning_effort")))
    return nullptr;
  return apps::resolve_chat_options(defaults, ojson::object()).enable_thinking;
}

const char* phase_name(TelemetrySnapshot::Phase p) {
  switch (p) {
    case TelemetrySnapshot::kUnloaded: return "unloaded";
    case TelemetrySnapshot::kLoading: return "loading";
    case TelemetrySnapshot::kIdle: return "idle";
    case TelemetrySnapshot::kPrefill: return "prefill";
    case TelemetrySnapshot::kDecode: return "decode";
  }
  return "idle";
}

ojson known(double v) { return v < 0 ? ojson(nullptr) : ojson(v); }
ojson known(int v) { return v < 0 ? ojson(nullptr) : ojson(v); }

// The host memory the process holds: what its resident figure counts (`rss`: Linux's resident set, Windows' working
// set), and what it does not: the sessions' K/V mapped for the device only (`mapped`, from the engine's books).
// Linux's resident set holds the other page-locked memory (cudaHostAlloc: checkpoints, low-bit copies, parked
// conversations), so there it is rss + mapped.  Windows counts every page-locked byte the process holds for the
// device (`locked`), and that is the figure for them: its working set holds cudaHostAlloc's pages from their
// allocation until the working set is emptied (flyx does after a load) or trimmed, which keeps them locked but
// counts them no more (RTX 5070 Ti, a 1 GiB buffer: 1,024 MiB in the working set, untouched or not; 0 after
// SetProcessWorkingSetSize(-1, -1)).  So the working set past the other page-locked memory is taken as the pageable
// rest: short of flyx's real figure by at most that rest (MBs), where adding the whole working set could count GBs of
// page-locked memory twice.
uint64_t process_ram(uint64_t rss, uint64_t mapped, uint64_t locked) {
  if (locked < mapped) return rss + mapped;  // none counted (0), or a figure that cannot be all of it
  const uint64_t other_locked = locked - mapped;
  return locked + (rss > other_locked ? rss - other_locked : 0);
}

ojson snapshot_json(const Server& s, int recent) {
  TelemetrySnapshot t;
  flyx_device_stats d{};
  bool device_ok = false;
  const ModelHost::Served served = s.host->served();
  if (served.sched) {
    served.sched->telemetry(&t, &d, &device_ok);
  } else {  // no model: none named yet, or one being fetched or loaded (the "load" object says which)
    s.host->telemetry(&t);  // what the models before counted
    t.loaded = false;
    t.phase = TelemetrySnapshot::kUnloaded;
  }
  t.started_at = s.started_at;  // the server's, not the served model's
  const double now = os::unix_seconds();
  const auto& m = t.model;

  ojson model = {{"id", served.sched ? ojson(served.name) : ojson(nullptr)}, {"loaded", t.loaded},
                 {"context", served.sched ? ojson(served.sched->max_ctx()) : ojson(nullptr)}};
  if (t.loaded) {
    model["device"] = m.device;
    model["backend"] = m.backend;
    model["drafter"] = m.drafter;
    model["draft_weights"] = m.draft_weights.empty() ? ojson(nullptr) : ojson(m.draft_weights);
    model["max_width"] = m.width;
    model["chat_format"] = m.chat_format;
    model["vision"] = m.vision;
    if (m.vision) {
      model["vision_device"] = m.vision_device;
      model["vision_quant"] = m.vision_quant;
      if (m.vision_max_tokens > 0) model["vision_max_tokens"] = m.vision_max_tokens;
    }
    model["max_sessions"] = m.max_sessions;
    model["memory"] = m.memory;
    model["load_s"] = m.load_s;
    ojson deg = ojson::array();
    for (const LoadAdjustment& a : m.load_adjustments)
      deg.push_back({{"kind", a.kind}, {"from", a.from}, {"to", a.to}, {"detail", a.detail}});
    model["degraded"] = std::move(deg);
    if (m.ctx_limit != plan::CtxLimit::kNone) {  // memory cut the context under the one wanted
      const bool gpu = m.ctx_limit == plan::CtxLimit::kGpu;
      model["context_limit"] = {{"by", gpu ? "gpu" : "host"},
                                {"wanted", m.wanted_ctx},
                                {"short", m.ctx_short},
                                {"left", gpu ? m.sessions_room : m.host_budget},
                                {"need_more", m.ctx_need_more > 0 ? ojson(m.ctx_need_more) : ojson(nullptr)},
                                {"gpu_tokens", m.kv_gpu_tokens},
                                {"others", gpu ? ojson(m.held.others) : ojson(nullptr)},
                                {"closable", gpu && plan::closable(m.held)},
                                {"runtime", gpu && m.held.runtime_known ? ojson(m.held.runtime) : ojson(nullptr)}};
    }
  }
  const bool running = t.phase == TelemetrySnapshot::kPrefill || t.phase == TelemetrySnapshot::kDecode;
  ojson state = {{"phase", phase_name(t.phase)}, {"queued", t.queued}, {"running", t.running}};
  if (running) {
    state["label"] = t.label;
    if (!t.response_id.empty()) state["id"] = t.response_id;
    state["session"] = t.session;
    state["prompt_tokens"] = t.prompt_tokens;
    state["reused_tokens"] = known(t.reused_tokens);
    state["prefilled"] = t.prefilled;
    state["generated"] = t.generated;
    state["tokens_per_round"] = t.tokens_per_round;
    state["rounds"] = t.rounds;
    state["drafted"] = t.drafted;
    state["accepted"] = t.accepted;
    state["decode_tok_s"] = known(t.request_decode_tok_s);
    state["prefill_tok_s"] = known(t.request_prefill_tok_s);
    state["max_new_tokens"] = t.max_new_tokens;
    state["elapsed_s"] = t.elapsed_ms / 1000;
  }
  ojson sessions = ojson::array();
  for (const auto& sl : t.sessions) sessions.push_back({{"id", sl.id}, {"tokens", sl.tokens}});
  ojson device = served.sched ? ojson(nullptr) : s.device_probe;  // no model: what the start-up probe found
  if (!served.sched && device.is_object()) {
    const int64_t fresh = s.probe_mem_used.load(std::memory_order_relaxed);  // -1: no reading (0 is one)
    if (fresh >= 0) device["mem_used"] = fresh;
  }
  if (device_ok) {
    device = {{"name", m.device},
              {"util", known(d.util)},
              {"mem_util", known(d.mem_util)},
              {"mem_used", d.mem_total ? ojson(d.mem_used) : ojson(nullptr)},
              {"mem_total", d.mem_total ? ojson(d.mem_total) : ojson(nullptr)},
              {"temp_c", known(d.temp_c)},
              {"power_w", known(d.power_w)},
              {"power_limit_w", known(d.power_limit_w)},
              {"sm_clock_mhz", known(d.sm_clock_mhz)},
              {"pcie_gen", known(d.pcie_gen)},
              {"pcie_width", known(d.pcie_width)},
              {"fan", known(d.fan)},
              {"mem_clock_mhz", known(d.mem_clock_mhz)},
              {"clock_limits", clock_limit_names(d.clock_limits)},
              {"process_host", d.process_host ? ojson(d.process_host) : ojson(nullptr)}};
  }
  const os::ProcessUsage pu = os::process_usage();
  const os::SystemMemory sm = os::system_memory();
  ojson records = ojson::array();
  for (size_t i = 0; i < t.recent.size() && int(i) < recent; ++i) {
    const auto& r = t.recent[i];
    ojson j = {{"finished_at", r.finished_at},
               {"label", r.label},
               {"status", r.status},
               {"session", r.session},
               {"prompt_tokens", r.prompt_tokens},
               {"reused_tokens", r.reused_tokens},
               {"output_tokens", r.output_tokens},
               {"queue_ms", r.queue_ms},
               {"ttft_ms", known(r.ttft_ms)},
               {"prefill_ms", r.prefill_ms},
               {"decode_ms", r.decode_ms},
               {"decode_tok_s", apps::decode_rate(r.output_tokens, r.decode_ms)},
               {"tokens_per_round", r.tokens_per_round},
               {"drafted", r.drafted},
               {"accepted_drafts", r.accepted_drafts},
               {"tool_calls", r.tool_calls},
               {"grammar_fixes", r.grammar_fixes}};
    if (!r.response_id.empty()) j["id"] = r.response_id;
    if (!r.error.empty()) j["error"] = r.error;
    if (r.superseded) j["cancel_reason"] = "superseded";
    records.push_back(std::move(j));
  }
  const auto& tt = t.totals;
  ojson device_reserve = nullptr;
  if (t.loaded) {
    device_reserve = m.device_reserve;
  } else if (device.is_object() && device.contains("mem_total") && device["mem_total"].is_number()) {
    device_reserve = plan::effective_device_reserve(device["mem_total"].get<uint64_t>(), os::gpu_memory_paged(),
                                                    s.base.vram_reserve, s.device_reserve_floor);
  }
  return {{"schema_version", 1},
          {"sampled_at", now},
          {"server", {{"name", "flyx-server"}, {"version", apps::kVersion}, {"pid", os::process_id()}, {"started_at", t.started_at},
                      {"uptime_s", now - t.started_at}, {"api_key_required", !s.cfg.api_key.empty()},
                      {"agent_launch", os::shell_window_supported() ? ojson(launch_agents()) : ojson::array()},
                      {"enable_thinking", thinking_default(s.cfg.chat_defaults)},
                      {"cors", {{"loopback", s.cfg.cors_loopback}, {"any", s.cfg.cors_any}, {"origins", s.cfg.cors_origins}}}}},
          {"model", model},
          {"state", state},
          {"rates", {{"decode_tok_s", t.decode_tok_s},
                     {"tokens_per_round", known(t.window_tokens_per_round)},
                     {"acceptance", known(t.window_acceptance)},
                     {"output_per_second", t.output_per_second},
                     {"prefill_per_second", t.prefill_per_second}, {"rounds_per_second", t.rounds_per_second},
                     {"drafted_per_second", t.drafted_per_second}, {"accepted_per_second", t.accepted_per_second}}},
          {"memory", {{"ledger_used", t.ledger_used}, {"ledger_budget", t.loaded ? ojson(m.ledger_budget) : ojson(nullptr)},
                      {"device_reserve", device_reserve},
                      {"runtime", t.loaded && m.held.runtime_known ? ojson(m.held.runtime) : ojson(nullptr)},
                      {"host_mapped", t.host_mapped},
                      {"sessions", sessions}}},
          {"device", device},
          {"process", {{"cpu_s", pu.cpu_seconds}, {"rss", pu.rss_bytes},
                       {"ram", process_ram(pu.rss_bytes, t.host_mapped, device_ok ? d.process_host : 0)}}},
          {"system", {{"ram_total", sm.total}, {"ram_available", sm.available}, {"cpus", os::cpu_count()},
                      {"cpu_s", os::system_cpu_seconds()}}},
          {"totals", {{"requests", tt.requests}, {"errors", tt.errors}, {"cancelled", tt.cancelled},
                      {"prompt_tokens", tt.prompt_tokens}, {"reused_tokens", tt.reused_tokens},
                      {"output_tokens", tt.output_tokens}, {"tool_calls", tt.tool_calls},
                      {"grammar_fixes", tt.grammar_fixes}}},
          {"recent", records},
          {"load", s.host->status()}};
}

}  // namespace

void add_telemetry_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/telemetry", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    if (!require_access(*s, q, r, Access::kDashboardRead, "reading server telemetry")) return;
    const int recent = query_int(q, "recent", 20, 0, 100);
    r.set_header("Cache-Control", "no-store");
    send_json(r, 200, snapshot_json(*s, recent));
  });
}

}  // namespace flyx::server
