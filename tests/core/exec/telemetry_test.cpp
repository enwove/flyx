// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Telemetry (core/exec/telemetry.h): phases as a request runs, the "now"
// speed, the per-second history, finished requests and totals; two requests
// running at once.
#include <cmath>
#include <cstdio>
#include <string>

#include "core/exec/telemetry.h"
#include "core/os/os.h"
#include "tests/common/check.h"

using namespace flyx;

int main() {
  Telemetry t;
  TelemetrySnapshot s;
  t.snapshot(&s);
  check(s.phase == TelemetrySnapshot::kUnloaded && !s.loaded, "starts unloaded");

  t.loading();
  t.snapshot(&s);
  check(s.phase == TelemetrySnapshot::kLoading, "loading");
  ModelInfo m;
  m.device = "test device";
  m.max_ctx = 1024;
  const double before_load = os::unix_seconds();
  t.loaded(m, 123);
  t.memory(456, 789, {{2, 300}, {1, 40}});
  t.snapshot(&s);
  check(s.loaded_at >= before_load && s.loaded_at <= os::unix_seconds(), "loaded: when, for the last request's figures");
  check(s.phase == TelemetrySnapshot::kIdle && s.loaded && s.model.device == "test device" && s.ledger_used == 456 &&
            s.host_mapped == 789 &&
            s.sessions.size() == 2 && s.sessions[0].id == 2 && s.sessions[0].tokens == 300 && s.running == 0,
        "loaded: idle, the model, memory and sessions");

  const double before_chat = os::now_ms();
  const int chat = t.start("chat", 2, 500, 64, "chatcmpl-test");
  t.snapshot(&s);
  check(s.phase == TelemetrySnapshot::kPrefill && s.label == "chat" && s.session == 2 && s.prompt_tokens == 500 &&
            s.reused_tokens == -1 && s.running == 1 && s.response_id == "chatcmpl-test" && s.request_decode_tok_s < 0,
        "a request starts: prefill, reuse not known yet");
  t.prefill(chat, 400, 500);
  t.prefill(chat, 450, 500);
  t.snapshot(&s);
  check(s.reused_tokens == 400 && s.prefilled == 450, "prefill progress: the first call says what was reused");
  check(s.request_prefill_tok_s > 0, "prefilling: the request's prompt tokens computed so far, a second");
  check(s.sessions.size() == 2 && s.sessions[0].id == 2 && s.sessions[0].tokens == 450 && s.sessions[1].tokens == 40,
        "live session tokens follow prefill progress");

  // 40 tokens over about 400 ms: decoding.  The first delivery starts the clock, so the speed is the 35 tokens
  // after it over the time since, whatever a busy machine does to the sleeps: between 35 over the time measured
  // around the whole (start to after the snapshot) and 35 over the loop's time (start to before it).
  const double t0 = os::now_ms();
  for (int i = 0; i < 8; ++i) {
    t.tokens(chat, 5);
    os::sleep_ms(50);
  }
  const double t1 = os::now_ms();
  t.snapshot(&s);
  const double t2 = os::now_ms();
  check(s.phase == TelemetrySnapshot::kDecode && s.generated == 40 && s.prefilled == 500, "decoding");
  // Two rounds of 3 drafts, 2 of each accepted: the request's figures, the recent window's, and the second's.
  t.round(chat, 3, 2);
  t.round(chat, 3, 2);
  {
    TelemetrySnapshot r;
    t.snapshot(&r);
    check(r.rounds == 2 && r.drafted == 6 && r.accepted == 4, "the request's rounds, drafts and drafts accepted");
    check(std::fabs(r.tokens_per_round - 3.0) < 1e-9, "the request's tokens/round from verified rounds");
    check(std::fabs(r.window_tokens_per_round - 3.0) < 1e-9 && std::fabs(r.window_acceptance - 4.0 / 6.0) < 1e-9,
          "the recent window's tokens/round and acceptance");
  }
  // Its prefill rate stops at the first delivery: the 100 tokens not reused, over less than start to now.
  check(s.request_prefill_tok_s >= 100000.0 / (t2 - before_chat),
        "the prefill rate up to the first delivery: " + std::to_string(s.request_prefill_tok_s) + " tok/s");
  check(s.decode_tok_s >= 0.99 * 35000.0 / (t2 - t0) && s.decode_tok_s <= 1.01 * 35000.0 / (t1 - t0),
        "the now speed: " + std::to_string(s.decode_tok_s) + " tok/s, 35 tokens over " + std::to_string(t1 - t0) + " to " +
            std::to_string(t2 - t0) + " ms");

  // A second request starts while the first decodes: two running, the
  // figures still the first one's; the speed counts both.
  const int side = t.start("messages", 1, 90, 16);
  t.prefill(side, 0, 90);
  t.tokens(side, 3);
  t.round(side, 1, 0);
  const double before = os::now_ms();
  t.snapshot(&s);
  const double after = os::now_ms();
  check(s.running == 2 && s.label == "chat" && s.response_id == "chatcmpl-test" &&
            s.generated == 40 && s.phase == TelemetrySnapshot::kDecode,
        "a second request runs beside it: the figures are the first one's");
  check(s.sessions.size() == 2 && s.sessions[0].tokens == 540 && s.sessions[1].tokens == 93,
        "live session tokens reflect both active requests' prefilled + generated");
  check(std::fabs(s.tokens_per_round - 3.0) < 1e-9 && std::fabs(s.window_tokens_per_round - (1.0 + 4.0 / 3.0)) < 1e-9 &&
            std::fabs(s.window_acceptance - 4.0 / 7.0) < 1e-9,
        "windowed speculation aggregates across running requests while state stays the first request's");
  check(s.request_decode_tok_s >= 0.99 * 35000.0 / (after - t0) && s.request_decode_tok_s <= 1.01 * 35000.0 / (before - t0),
        "the first request's rate counts only its own tokens after its first delivery");

  RequestRecord r;
  r.label = "chat";
  r.status = "tool_calls";
  r.prompt_tokens = 500;
  r.reused_tokens = 400;
  r.output_tokens = 40;
  r.tool_calls = 1;
  r.grammar_fixes = 2;
  t.finish(chat, r);
  t.snapshot(&s);
  check(s.running == 1 && s.label == "messages" && s.session == 1 && s.prompt_tokens == 90 && s.generated == 3 &&
            s.reused_tokens == 0 && s.response_id.empty() && s.request_decode_tok_s < 0 && s.phase == TelemetrySnapshot::kDecode,
        "the first one ends: the figures are the second one's");
  r.label = "messages";
  r.status = "error";
  t.finish(side, r);
  t.snapshot(&s);
  check(s.phase == TelemetrySnapshot::kIdle && s.decode_tok_s == 0 && s.window_tokens_per_round < 0 &&
            s.window_acceptance < 0 && s.label.empty() && s.running == 0,
        "finished: idle again");
  check(s.recent.size() == 2 && s.recent[0].status == "error" && s.recent[1].status == "tool_calls",
        "recent requests, newest first");
  check(s.totals.requests == 2 && s.totals.errors == 1 && s.totals.output_tokens == 80 &&
            s.totals.reused_tokens == 800 && s.totals.tool_calls == 2 && s.totals.grammar_fixes == 4,
        "totals");
  // A request that failed before it ran (no session, a grammar that does not compile).
  r.status = "error";
  r.output_tokens = 0;
  t.finish(0, r);
  t.snapshot(&s);
  check(s.totals.requests == 3 && s.totals.errors == 2 && s.running == 0, "a request that never ran is counted");

  // The history holds whole seconds: once the second of the last token is over, the 43 tokens are in it (asked
  // again for up to 4 s, as a busy machine can wake a sleep late).  The prompt tokens computed: the chat's 50 after
  // its 400 reused, and the 50 its first delivery found unreported; the second request's 90.
  int sum = 0, computed = 0, rounds = 0, drafted = 0, accepted = 0;
  for (int tries = 0; tries < 40 && (sum != 43 || computed != 190 || rounds != 3); ++tries) {
    os::sleep_ms(100);
    t.snapshot(&s);
    sum = computed = rounds = drafted = accepted = 0;
    for (int v : s.output_per_second) sum += v;
    for (int v : s.prefill_per_second) computed += v;
    for (int v : s.rounds_per_second) rounds += v;
    for (int v : s.drafted_per_second) drafted += v;
    for (int v : s.accepted_per_second) accepted += v;
  }
  check(s.output_per_second.size() == 60 && sum == 43, "per-second history: " + std::to_string(sum) + " of 43");
  check(s.rounds_per_second.size() == 60 && rounds == 3 && drafted == 7 && accepted == 4,
        "per-second rounds (" + std::to_string(rounds) + " of 3), drafts (" + std::to_string(drafted) +
            " of 7) and drafts accepted (" + std::to_string(accepted) + " of 4)");
  check(s.prefill_per_second.size() == 60 && computed == 190,
        "per-second prompt tokens computed: " + std::to_string(computed) + " of 190");

  // A prefill pass spanning across a whole-second boundary distributes proportionally across both seconds
  // instead of lumping the entire chunk into one second (sawtooth).
  {
    Telemetry t2;
    const int req = t2.start("chat", 1, 2048, 16);
    t2.prefill(req, 0, 2048);
    t2.prefill(req, 1024, 2048);
    os::sleep_ms(1300);  // crosses at least one whole-second boundary
    t2.prefill(req, 2048, 2048);
    t2.finish(req, RequestRecord{});
    int total = 0, nonzero = 0, peak = 0;
    for (int tries = 0; tries < 40 && total != 2048; ++tries) {
      os::sleep_ms(100);
      t2.snapshot(&s);
      total = nonzero = peak = 0;
      for (int v : s.prefill_per_second) {
        total += v;
        if (v > 0) ++nonzero;
        if (v > peak) peak = v;
      }
    }
    check(total == 2048 && nonzero >= 2 && peak < 2048,
          "multi-second prefill pass splits across seconds: total=" + std::to_string(total) +
              " nonzero=" + std::to_string(nonzero) + " peak=" + std::to_string(peak));
  }

  t.unloaded();
  t.snapshot(&s);
  check(s.phase == TelemetrySnapshot::kUnloaded && !s.loaded && s.sessions.empty() && s.host_mapped == 0 &&
            s.totals.requests == 3 && s.loaded_at == 0,
        "unloaded: no sessions, totals kept");

  return flyx::test::finish("telemetry_test");
}
