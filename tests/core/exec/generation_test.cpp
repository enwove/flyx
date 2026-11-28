// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Regression tests for Generation and Session edge cases:
// 1. Grammar rollback near max_ctx must not launch a speculative round past flyx_room(at, max_ctx).
// 2. Failed Session::prepare() (rewind error) must reset device and host session state cleanly.
// 3. Session::tokenize() must not match an empty history_text_ against every subsequent prompt.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

const Engine* sharing_during_launch = nullptr;
int interrupt_launch(flyx_plan* p, flyx_seq* s, const flyx_gen_params* g) {
  if (sharing_during_launch) {
    sharing_during_launch->generation_began();
    sharing_during_launch->generation_ended();
  }
  return g_cpu->launch(p, s, g);
}

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
  auto delegated = *cpu;
  delegated.launch = interrupt_launch;
  flyx_test_delegate(&delegated);
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
  for (const char* name : {"dspark", "dflash2"}) {
    Engine parallel;
    EngineOptions block = eo;
    block.drafter = name;
    const auto path = block.drafter == "dspark" ? std::filesystem::path(model).parent_path() / "tiny_dspark.flyx"
                                                : std::filesystem::path(model);
    if (!parallel.load(path.string(), block, &err)) { check(false, err); return; }
    RoundChoice round;
    round.drafter = 0;
    check(round.measures_fixed_cost(parallel.model()), block.drafter + " fixed rounds teach fixed costs");
    round.adaptive = true;
    check(!round.measures_fixed_cost(parallel.model()), block.drafter + " adaptive rounds pay for the full block");
  }
  RoundChoice sequential;
  sequential.drafter = 0;
  sequential.adaptive = true;
  check(sequential.measures_fixed_cost(engine.model()), "stopped MTP rounds teach the verified width's cost");
  // A wide round's calibrated price can stay at its narrower neighbour's floor even after an observation.
  // Start below the duration of a cpu_ref round so every timed sample changes the price deterministically.
  for (RoundCosts* costs : {engine.lookup_costs(), engine.round_costs()}) {
    if (!costs) { check(false, "missing learned cost table"); return; }
    costs->reset(std::vector<double>(costs->size(), 0.001), nullptr, 0);
  }
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
  sharing_during_launch = &engine;
  const int interrupted = generate(a, rq, [](std::string_view, const int32_t*, int) { return true; }, &res, &err);
  sharing_during_launch = nullptr;
  check(interrupted == 0 && res.usage.rounds > 2 && costs_of(engine.round_costs()) == shared &&
            costs_of(engine.lookup_costs()) == lookup_shared, "overlap entirely inside backend launch excludes its timing");
  check(engine.generations() == 0, "none under way after");
  check(costs_of(engine.round_costs()) == shared && costs_of(engine.lookup_costs()) == lookup_shared,
        "two generations at once: no round's cost learned");
}

// A request naming no limit (max_new_tokens 0): the engine's default (EngineOptions::max_new_tokens, serve's
// --max-tokens), else the end of the context; one that names a limit keeps it.
void test_default_max_new_tokens(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  flyx_test_delegate(cpu);
  auto run = [&](int engine_default, int asked, int* got) {
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 96;
    eo.speculative = false;
    eo.max_new_tokens = engine_default;
    Engine engine;
    std::string err;
    if (!engine.load(model, eo, &err)) return check(false, "load engine for the default reply limit: " + err), false;
    Session s(&engine);
    if (!s.init(&err)) return check(false, "session init: " + err), false;
    GenerateRequest rq;
    rq.prompt = engine.tokenizer().encode("one two three");
    rq.max_new_tokens = asked;
    rq.ignore_eos = true;
    GenerateResult res;
    if (generate(s, rq, [](std::string_view, const int32_t*, int) { return true; }, &res, &err) != 0)
      return check(false, "generate: " + err), false;
    *got = res.usage.new_tokens;
    return true;
  };
  int got = 0;
  if (run(7, 0, &got)) check(got == 7, "no limit asked: the engine's default, 7 tokens (" + std::to_string(got) + ")");
  if (run(7, 3, &got)) check(got == 3, "a limit asked: its own, 3 tokens (" + std::to_string(got) + ")");
  if (run(0, 0, &got))
    check(got > 7 && got <= 96, "no limit anywhere: up to the context's end (" + std::to_string(got) + " tokens)");
}

// A thinking budget (GenerateRequest::thinking_budget): the reply thinks that many tokens and then writes the closing
// marker, with a drafter as without one (the budget is held as a grammar is: a speculative round gives plain
// decoding's tokens).  The markers here are ordinary words, so the model writes them only when made to.
void test_thinking_budget(const char* model, const flyx_backend_api* cpu) {
  g_cpu = cpu;
  flyx_test_delegate(cpu);
  constexpr int kBudget = 9, kReply = 40;
  // A reply from `prompt`: under the budget (budget >= 0) or none; its tokens.
  auto run = [&](bool speculative, const std::vector<int32_t>* prompt, int budget, int max_new,
                 std::vector<int32_t>* out, std::vector<int32_t>* base, std::vector<int32_t>* close) {
    EngineOptions eo;
    eo.backend = "test-host-kv";
    eo.max_ctx = 160;
    eo.speculative = speculative;
    if (speculative) eo.drafter = "mtp";
    Engine engine;
    std::string err;
    if (!engine.load(model, eo, &err)) return check(false, "load engine for the thinking budget: " + err), false;
    Session s(&engine);
    if (!s.init(&err)) return check(false, "session init: " + err), false;
    GenerateRequest rq;
    if (prompt) {
      rq.prompt = *prompt;
    } else {
      for (int k = 0; k < 3; ++k) {
        const auto ids = engine.tokenizer().encode("one two three four five six seven eight ");
        rq.prompt.insert(rq.prompt.end(), ids.begin(), ids.end());
      }
    }
    if (base) *base = rq.prompt;
    rq.max_new_tokens = max_new;
    rq.ignore_eos = true;
    rq.thinking_budget = budget;
    rq.thinking_open = engine.tokenizer().encode(" open");
    rq.thinking_close = engine.tokenizer().encode(" lighthouse harbor");
    if (close) *close = rq.thinking_close;
    rq.thinking_inside = true;
    GenerateResult res;
    if (generate(s, rq, [](std::string_view, const int32_t*, int) { return true; }, &res, &err) != 0)
      return check(false, "generate: " + err), false;
    // The reply: the tokens after the prompt (a drafter's session may hold a round's past the reply's end).
    const auto from = s.tokens().begin() + long(rq.prompt.size());
    *out = std::vector<int32_t>(from, from + res.usage.new_tokens);
    return true;
  };
  std::vector<int32_t> plain, spec, base, close;
  if (!run(false, nullptr, kBudget, kReply, &plain, &base, &close) ||
      !run(true, nullptr, kBudget, kReply, &spec, nullptr, nullptr))
    return;
  const size_t marked = kBudget + close.size();
  const bool at_budget = plain.size() > marked && std::equal(close.begin(), close.end(), plain.begin() + kBudget);
  check(at_budget, "the closing marker written right after the " + std::to_string(kBudget) + " tokens of the budget");
  size_t differ = 0;
  while (differ < spec.size() && differ < plain.size() && spec[differ] == plain[differ]) ++differ;
  std::string around;
  for (size_t k = differ > 2 ? differ - 2 : 0; k < std::min(plain.size(), differ + 3); ++k)
    around += " " + std::to_string(plain[k]) + "/" + (k < spec.size() ? std::to_string(spec[k]) : "-");
  check(spec == plain, "a drafter's reply under a thinking budget is plain decoding's (" +
                           std::to_string(spec.size()) + " and " + std::to_string(plain.size()) +
                           " tokens; first difference at " + std::to_string(differ) + ", plain/drafted:" + around + ")");
  if (!at_budget) return;
  // After the marker the reply goes on as the model would from it: what follows the prompt, the thought and the
  // marker with no budget (two rounds in flight must not write a second marker, nor drop a token).
  std::vector<int32_t> head = base, after;
  head.insert(head.end(), plain.begin(), plain.begin() + long(marked));
  if (!run(false, &head, -1, kReply - int(marked), &after, nullptr, nullptr)) return;
  const std::vector<int32_t> tail(plain.begin() + long(marked), plain.end());
  check(tail == after, "after the marker, the model's own continuation (" + std::to_string(tail.size()) +
                           " tokens against " + std::to_string(after.size()) + ")");
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
  test_default_max_new_tokens(argv[1], cpu);
  test_thinking_budget(argv[1], cpu);
  return flyx::test::finish("generation_test");
}
