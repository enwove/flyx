// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Regression tests for Generation and Session edge cases:
// 1. Grammar rollback near max_ctx must not launch a speculative round past flyx_room(at, max_ctx).
// 2. Failed Session::prepare() (rewind error) must reset device and host session state cleanly.
// 3. Session::tokenize() must not match an empty history_text_ against every subsequent prompt.
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/grammar/grammar.h"
#include "core/session/session.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

const flyx_backend_api* g_cpu = nullptr;
int32_t g_tracked_pos = 0;
int32_t g_max_ctx = 64;
bool g_oob_launch = false;
bool g_fail_rewind = false;

int hook_prefill(flyx_plan* p, flyx_seq* s, const int32_t* t, int32_t n, const flyx_gen_params* g) {
  g_tracked_pos += n;
  return g_cpu->prefill(p, s, t, n, g);
}

int hook_launch(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g) {
  if (!flyx_room(g_tracked_pos, g_max_ctx)) {
    g_oob_launch = true;
    return -1;
  }
  return g_cpu->launch(p, s, g);
}

int hook_read(flyx_seq* s, flyx_round_result* r) {
  const int rc = g_cpu->read(s, r);
  if (rc == 0) g_tracked_pos = r->position;
  return rc;
}

int hook_rewind(flyx_seq* s, int32_t pos, int32_t anchor) {
  if (g_fail_rewind) return -1;
  const int rc = g_cpu->rewind(s, pos, anchor);
  if (rc == 1) g_tracked_pos = pos;
  return rc;
}

void test_grammar_rollback_near_max_ctx(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  g_max_ctx = 64;
  g_tracked_pos = 0;
  g_oob_launch = false;
  g_fail_rewind = false;

  auto delegated = *cpu;
  delegated.prefill = hook_prefill;
  delegated.launch = hook_launch;
  delegated.read = hook_read;
  delegated.rewind = hook_rewind;
  flyx_test_delegate(&delegated);

  EngineOptions eo;
  eo.backend = "test-host-kv";
  eo.max_ctx = g_max_ctx;
  eo.speculative = true;
  eo.drafter = "mtp";
  eo.width = 3;
  eo.lookup = false;
  eo.adaptive = false;

  Engine engine;
  std::string err;
  if (!engine.load(model, eo, &err)) {
    check(false, "load engine for grammar rollback test: " + err);
    return;
  }

  // Prompt of length max_ctx - FLYX_MAX_ROWS (48): flyx_room(48, 64) is true, flyx_room(49, 64) is false.
  // Pick a prompt whose unconstrained greedy continuation [t0, t1] is valid non-empty UTF-8 so we can
  // construct a real XGrammar const_string grammar accepting t0, t1 and rejecting t2.
  const int prompt_len = g_max_ctx - FLYX_MAX_ROWS;
  std::vector<int32_t> prompt, greedy;
  std::string tag;
  {
    EngineOptions eo_plain = eo;
    eo_plain.max_ctx = 128;
    eo_plain.speculative = false;
    g_max_ctx = 128;
    Engine plain;
    if (!plain.load(model, eo_plain, &err)) {
      check(false, "load plain engine: " + err);
      return;
    }
    for (int attempt = 0; attempt < 64 && tag.empty(); ++attempt) {
      prompt.clear();
      while (static_cast<int>(prompt.size()) < prompt_len) {
        auto ids = engine.tokenizer().encode("hello " + std::to_string(attempt) + " world ");
        prompt.insert(prompt.end(), ids.begin(), ids.end());
      }
      prompt.resize(size_t(prompt_len));
      greedy.clear();
      Session sp(&plain);
      if (!sp.init(&err)) {
        check(false, "plain session init: " + err);
        return;
      }
      GenerateRequest rq;
      rq.prompt = prompt;
      rq.max_new_tokens = 4;
      rq.ignore_eos = true;
      GenerateResult res;
      generate(sp, rq,
               [&](std::string_view, const int32_t* t, int nt) {
                 greedy.insert(greedy.end(), t, t + nt);
                 return true;
               },
               &res, &err);
      if (greedy.size() < 3) continue;
      const std::string t0_str = engine.tokenizer().decode({greedy[0]});
      const std::string t1_str = engine.tokenizer().decode({greedy[1]});
      const std::string t2_str = engine.tokenizer().decode({greedy[2]});
      if (t0_str.empty() || t1_str.empty() || t2_str.rfind("~~~", 0) == 0) continue;
      try {
        const std::string target = t0_str + t1_str + "~~~NEVER_MATCH_SENTINEL~~~";
        const std::string cand = nlohmann::json{
            {"type", "structural_tag"},
            {"format", {{"type", "const_string"}, {"value", target}}},
        }.dump();
        std::string gerr;
        auto m = engine.grammar().matcher(cand, &gerr);
        if (m && m->accept(greedy[0]) && m->accept(greedy[1]) && !m->accept(greedy[2])) tag = cand;
      } catch (const std::exception&) {
      }
    }
  }
  g_max_ctx = 64;
  g_tracked_pos = 0;
  g_oob_launch = false;
  if (tag.empty()) {
    check(false, "could not find a valid UTF-8 greedy prefix for grammar rollback test");
    return;
  }

  auto matcher = engine.grammar().matcher(tag, &err);
  if (!matcher) {
    check(false, "compile const_string grammar: " + err);
    return;
  }

  Session s(&engine);
  if (!s.init(&err)) {
    check(false, "session init: " + err);
    return;
  }
  // Force the drafter oracle (indexed by absolute sequence position) so the speculative round at pos=48
  // verifies [greedy[1], greedy[2]] together and crosses max_ctx - FLYX_MAX_ROWS.
  std::vector<int32_t> oracle = prompt;
  oracle.insert(oracle.end(), greedy.begin(), greedy.end());
  engine.api()->set_oracle(s.seq(), oracle.data(), static_cast<int32_t>(oracle.size()));

  GenerateRequest rq;
  rq.prompt = prompt;
  rq.max_new_tokens = 0;  // unlimited -> relies on flyx_room(at, max_ctx) to stop near context limit
  rq.ignore_eos = true;
  rq.grammar = matcher.get();
  GenerateResult res;
  std::vector<int32_t> out;
  const int rc = generate(s, rq,
                          [&](std::string_view, const int32_t* t, int nt) {
                            out.insert(out.end(), t, t + nt);
                            return true;
                          },
                          &res, &err);
  check(rc == 0 && !g_oob_launch && res.usage.finish_reason == FLYX_FINISH_LENGTH && res.grammar_fixes >= 1,
        "grammar rollback past max_ctx - FLYX_MAX_ROWS finishes cleanly without OOB round launch (rc=" +
            std::to_string(rc) + ", oob=" + std::to_string(g_oob_launch) +
            ", fixes=" + std::to_string(res.grammar_fixes) + ", err=" + err + ")");
}

void test_prepare_failure_resets_session(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  g_max_ctx = 128;
  g_tracked_pos = 0;
  g_oob_launch = false;
  g_fail_rewind = false;

  auto delegated = *cpu;
  delegated.prefill = hook_prefill;
  delegated.launch = hook_launch;
  delegated.read = hook_read;
  delegated.rewind = hook_rewind;
  flyx_test_delegate(&delegated);

  EngineOptions eo;
  eo.backend = "test-host-kv";
  eo.max_ctx = g_max_ctx;
  eo.speculative = false;

  Engine engine;
  std::string err;
  if (!engine.load(model, eo, &err)) {
    check(false, "load engine for prepare reset test: " + err);
    return;
  }

  Session s(&engine);
  if (!s.init(&err)) {
    check(false, "session init: " + err);
    return;
  }

  const auto p1 = engine.tokenizer().encode("The quick brown fox jumps over the lazy dog.");
  const auto p2 = engine.tokenizer().encode("A completely different prompt that triggers rewind.");
  GenerateRequest rq1;
  rq1.prompt = p1;
  rq1.max_new_tokens = 4;
  rq1.ignore_eos = true;
  GenerateResult r1;
  check(generate(s, rq1, [](std::string_view, const int32_t*, int) { return true; }, &r1, &err) == 0 &&
            s.state_pos() > 0,
        "initial generation populates session state");

  // Inject a rewind error during Session::prepare() on a diverging prompt.
  g_fail_rewind = true;
  GenerateRequest rq2;
  rq2.prompt = p2;
  rq2.max_new_tokens = 4;
  rq2.ignore_eos = true;
  GenerateResult r2;
  const int rc2 = generate(s, rq2, [](std::string_view, const int32_t*, int) { return true; }, &r2, &err);
  g_fail_rewind = false;
  check(rc2 != 0 && s.state_pos() == 0 && s.tokens().empty(),
        "failed Session::prepare() resets state_pos and tokens (rc=" + std::to_string(rc2) +
            ", state_pos=" + std::to_string(s.state_pos()) + ", tokens=" + std::to_string(s.tokens().size()) + ")");

  // Subsequent generation on the same session must succeed from cold start and match a fresh session.
  std::vector<int32_t> after_toks, fresh_toks;
  GenerateResult r3, rf;
  const int rc3 = generate(s, rq2,
                           [&](std::string_view, const int32_t* t, int nt) {
                             after_toks.insert(after_toks.end(), t, t + nt);
                             return true;
                           },
                           &r3, &err);
  Session fresh(&engine);
  fresh.init(&err);
  const int rcf = generate(fresh, rq2,
                           [&](std::string_view, const int32_t* t, int nt) {
                             fresh_toks.insert(fresh_toks.end(), t, t + nt);
                             return true;
                           },
                           &rf, &err);
  check(rc3 == 0 && rcf == 0 && r3.usage.reused_tokens == 0 && after_toks == fresh_toks,
        "session recovers cleanly after prepare() failure and matches fresh session");
}

void test_empty_history_text_tokenize(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  flyx_test_delegate(cpu);

  EngineOptions eo;
  eo.backend = "test-host-kv";
  eo.max_ctx = 128;
  eo.speculative = false;
  Engine engine;
  std::string err;
  if (!engine.load(model, eo, &err)) {
    check(false, "load engine for empty history_text test: " + err);
    return;
  }

  // Find a token id in [0, vocab) whose decode() is empty (e.g. padding id >= n_tokens or empty piece).
  int32_t empty_id = -1;
  const int vocab = engine.model().desc().vocab;
  for (int32_t id = vocab - 1; id >= 0; --id) {
    if (engine.tokenizer().decode({id}).empty()) {
      empty_id = id;
      break;
    }
  }
  if (empty_id < 0) {  // the tiny model has one: without it the history text below would be tested on nothing
    check(false, "the tiny model has a token whose decode() is empty (a padding id or an empty piece)");
    return;
  }

  Session s(&engine);
  if (!s.init(&err)) {
    check(false, "session init: " + err);
    return;
  }
  s.publish({empty_id, empty_id}, {}, 0, -1);
  check(!s.history().empty() && s.history_text().empty(), "published history has non-empty ids with empty decoded text");

  const std::string next = "Hello world";
  const auto got = s.tokenize(next);
  const auto expected = engine.tokenizer().encode(next);
  check(got == expected, "Session::tokenize() does not prepend empty-text history tokens to a new text prompt");
}

// The costs the rounds read teach (Engine::round_costs, lookup_costs): text that repeats, so that rounds draft by
// lookup, one in flight (each launched with none ahead of it, timed from its launch), and MTP's rounds besides.
void test_round_costs_learned(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  flyx_test_delegate(cpu);
  EngineOptions eo;
  eo.backend = "test-host-kv";
  eo.max_ctx = 512;
  eo.speculative = true;
  eo.drafter = "mtp";
  eo.lookup = true;
  Engine engine;
  std::string err;
  if (!engine.load(model, eo, &err)) {
    check(false, "load engine for learned costs: " + err);
    return;
  }
  Session s(&engine);
  if (!s.init(&err)) {
    check(false, "session init: " + err);
    return;
  }
  GenerateRequest rq;
  for (int k = 0; k < 12; ++k) {
    const auto ids = engine.tokenizer().encode("one two three four five six seven eight ");
    rq.prompt.insert(rq.prompt.end(), ids.begin(), ids.end());
  }
  rq.max_new_tokens = 160;
  rq.ignore_eos = true;
  const int at = int(rq.prompt.size()) + rq.max_new_tokens;
  auto costs_of = [&](const RoundCosts* costs) {
    std::vector<double> c;
    for (size_t k = 0; costs && k < costs->size(); ++k) c.push_back(costs->cost(k, at));
    return c;
  };
  const std::vector<double> lookup_before = costs_of(engine.lookup_costs()), before = costs_of(engine.round_costs());
  GenerateResult res;
  const int rc = generate(s, rq, [](std::string_view, const int32_t*, int) { return true; }, &res, &err);
  check(rc == 0 && res.lookup_rounds > 1, "rounds drafted by lookup: " + std::to_string(res.lookup_rounds));
  auto list = [](const std::vector<double>& v) {
    std::string t;
    for (double x : v) t += (t.empty() ? "" : " ") + std::to_string(x);
    return t;
  };
  check(!lookup_before.empty() && costs_of(engine.lookup_costs()) != lookup_before,
        "a lookup round's cost learned from the rounds read (launched with none in flight): " + list(lookup_before) +
            " -> " + list(costs_of(engine.lookup_costs())));
  check(!before.empty() && costs_of(engine.round_costs()) != before, "a round's cost learned from the rounds read");

  // Two sessions of the engine generating at once (as the C API lets threads do, here in turn on one): neither's
  // rounds are timed, the other's work in its clock.
  Session a(&engine), b(&engine);
  if (!a.init(&err) || !b.init(&err)) {
    check(false, "two sessions: " + err);
    return;
  }
  const std::vector<double> lookup_shared = costs_of(engine.lookup_costs()), shared = costs_of(engine.round_costs());
  {
    TextSink none = [](std::string_view, const int32_t*, int) { return true; };
    Generation ga(&a, rq, none), gb(&b, rq, none);
    check(engine.generations() == 2, "two generations under way");
    bool a_done = false, b_done = false;
    while (!a_done || !b_done) {
      if (!a_done) a_done = ga.advance(true) == Generation::Progress::kDone;
      if (!b_done) b_done = gb.advance(true) == Generation::Progress::kDone;
    }
    check(!ga.failed() && !gb.failed() && ga.result().usage.new_tokens > 0 &&
              gb.result().usage.new_tokens == ga.result().usage.new_tokens,
          "both generate: " + std::to_string(ga.result().usage.new_tokens) + " and " +
              std::to_string(gb.result().usage.new_tokens) + " tokens" + (ga.failed() ? ", " + ga.error() : "") +
              (gb.failed() ? ", " + gb.error() : ""));
  }
  check(engine.generations() == 0, "none under way after");
  check(costs_of(engine.round_costs()) == shared && costs_of(engine.lookup_costs()) == lookup_shared,
        "two generations at once: no round's cost learned");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc != 2) {
    std::fprintf(stderr, "usage: generation_test TINY.flyx\n");
    return 2;
  }
  EngineOptions probe;
  probe.backend = "cpu_ref";
  std::unique_ptr<os::DynLib> cpu_lib;
  const flyx_backend_api* cpu = nullptr;
  flyx_device_info device{};
  std::string err;
  if (!open_backend(probe, &cpu_lib, &cpu, &device, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  test_grammar_rollback_near_max_ctx(argv[1], cpu);
  test_prepare_failure_resets_session(argv[1], cpu);
  test_empty_history_text_tokenize(argv[1], cpu);
  test_round_costs_learned(argv[1], cpu);
  return flyx::test::finish("generation_test");
}
