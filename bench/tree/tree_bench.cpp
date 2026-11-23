// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What a round that verifies a tree of drafts costs (docs/design/tree-verification.md), by width and shape, against
// plain decoding's round and a chain of given drafts, at a context:
//
//   tree_bench MODEL [--ctx-at P] [--widths 1,2,...] [--rounds N] [--backend-options O] [--kv 16|fp8] [--check T]
//   tree_bench MODEL --drafted T [--drafter D] [--drafter-file F] [--backend-options O] [--kv 16|fp8]
//
// Widths up to FLYX_MAX_WIDE_ROWS - 1 (63): trees of more than 16 rows on the backend's wide rows (backend option
// wide=1, added where a width past 15 is asked and the options name none).  --kv: the K/V cache's format (the
// model's default without it).
//
// --check T: instead, T tokens of tree rounds that hide plain decoding's greedy continuation on random paths of random
// trees: exit 0 when every token is plain decoding's and every round accepts its hidden path (rule 4 on a real model
// at a context).
//
// --drafted T: instead, the drafter's own rounds (D, default dflash2: its chains and its trees, FLYX_DRAFT_TREE)
// taking turns on `flyx bench`'s three prompts, T tokens each, greedy: per round, the drafts it accepted on average,
// its tokens a round, and the tokens a millisecond that gives at its cost measured at load, against plain decoding's
// (what the speculation controller weighs; the text is the same whichever round runs).  Then, paired: only the
// widest tree drafts, and every round says what the tree of its first W drafts would have accepted at the same point
// (the trees grow greedily, so the tree of W drafts is the widest's first W; up to the drafter's depth, its chain of
// W): tokens a round by W on the same rounds, and tokens a ms where a round of W has a plan (and so a cost).
//
// The drafts are random tokens (nearly none accepted: every round advances one position), so the rows' work is the
// round's whole cost.  Every timing (the plain round's, each width's and shape's) starts from the prompt's P tokens,
// prefilled again before it (exit 1 when a timing's first round starts elsewhere): a width timed after others would
// else read their positions too (100 rounds of 4 shapes move the context by 400 a width), and the tree's attention
// grows with the keys.  Rounds go two in flight, as the engine runs them.  Prints ms a round per width and shape, and
// each as a multiple of the plain round: the tokens a round of that tree has to emit on average to beat plain decoding.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/exec/request.h"
#include "core/os/os.h"
#include "core/session/session.h"

using namespace flyx;

namespace {

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    v.push_back(std::atoi(s.substr(i, j - i).c_str()));
    i = j + 1;
  }
  return v;
}

// A tree of w drafts: parents[i] < i (-1: the anchor).
std::vector<int32_t> shape(const std::string& kind, int w, uint32_t* rng) {
  std::vector<int32_t> p(static_cast<size_t>(w));
  for (int i = 0; i < w; ++i) {
    if (kind == "chain") p[size_t(i)] = i - 1;
    else if (kind == "star") p[size_t(i)] = -1;
    else if (kind == "binary") p[size_t(i)] = i / 2 - 1;
    else {
      *rng = *rng * 1664525u + 1013904223u;
      p[size_t(i)] = int32_t((*rng >> 8) % uint32_t(i + 1)) - 1;
    }
  }
  return p;
}

// flyx bench's prompts (apps/flyx-cli).
const char* kPrompts[] = {
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss.",
    "Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, "
    "juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey.",
};

int drafted_rounds(const std::string& model, EngineOptions o, int tokens) {
  o.speculative = true;
  o.lookup = false;
  o.draft_trees = true;
  o.max_ctx = 4096;
  Engine e;
  std::string err;
  if (!e.load(model, o, &err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  std::vector<size_t> kinds;  // every drafted round, in turn
  size_t plain = 0;
  for (size_t i = 0; i < e.widths().size(); ++i) {
    if (e.round_drafters()[i] >= 0) kinds.push_back(i);
    if (e.widths()[i] == 0) plain = i;
  }
  std::vector<double> rounds(e.widths().size()), accepted(e.widths().size());
  for (const char* p : kPrompts) {
    Session s(&e);
    if (!s.init(&err)) return std::fprintf(stderr, "session: %s\n", err.c_str()), 1;
    Request rq;
    ChatMessage m;
    m.role = "user";
    m.content = p;
    rq.messages = {m};
    rq.chat_options.enable_thinking = false;
    rq.split_output = false;
    rq.gen.max_new_tokens = tokens;
    rq.gen.ignore_eos = true;
    RenderedChat rc;
    PreparedRequest pr;
    if (!render_request(e.chat(), e.tokenizer(), rq, &rc, &err) || !prepare_request(e, s, rq, rc, &pr, &err))
      return std::fprintf(stderr, "prompt: %s\n", err.c_str()), 1;
    s.rounds().cycle(kinds);
    GenerateResult res;
    if (generate(s, pr.gen, TextSink(), &res, &err) != 0) return std::fprintf(stderr, "generate: %s\n", err.c_str()), 1;
    for (size_t i = 0; i < res.round_choices.size(); ++i) {
      rounds[i] += res.round_choices[i];
      accepted[i] += res.round_accepted[i];
    }
  }
  const double plain_ms = e.round_ms()[plain];
  std::printf("%d tokens a prompt, the drafter's rounds in turn; plain round %.3f ms (at load)\n", tokens, plain_ms);
  std::printf("%-20s %8s %7s %9s %9s %8s %8s\n", "round", "ms", "rounds", "accepted", "tok/round", "tok/ms", "x plain");
  for (size_t i : kinds) {
    if (rounds[i] == 0) continue;
    const double ms = e.round_ms()[i], per = 1 + accepted[i] / rounds[i];
    std::printf("%-20s %8.3f %7.0f %9.3f %9.3f %8.4f %8.3f\n", e.round_name(i).c_str(), ms, rounds[i],
                accepted[i] / rounds[i], per, per / ms, per / ms * plain_ms);
  }

  // Paired: the widest tree alone; each round's accepted path (its drafts by the parents and the accepted tokens).
  size_t widest = e.widths().size();
  for (size_t i : kinds)
    if (e.round_trees()[i] && (widest == e.widths().size() || e.widths()[i] > e.widths()[widest])) widest = i;
  if (widest == e.widths().size()) return 0;
  const int W = e.widths()[widest];
  std::vector<double> got(size_t(W) + 1, 0.0);  // accepted drafts summed, by the tree's first w drafts
  int n_rounds = 0;
  for (const char* p : kPrompts) {
    Session s(&e);
    if (!s.init(&err)) return std::fprintf(stderr, "session: %s\n", err.c_str()), 1;
    Request rq;
    ChatMessage m;
    m.role = "user";
    m.content = p;
    rq.messages = {m};
    rq.chat_options.enable_thinking = false;
    rq.split_output = false;
    rq.gen.max_new_tokens = tokens;
    rq.gen.ignore_eos = true;
    RenderedChat rc;
    PreparedRequest pr;
    if (!render_request(e.chat(), e.tokenizer(), rq, &rc, &err) || !prepare_request(e, s, rq, rc, &pr, &err))
      return std::fprintf(stderr, "prompt: %s\n", err.c_str()), 1;
    s.rounds().cycle({widest});
    std::vector<flyx_round_result> trace;
    s.trace = &trace;
    GenerateResult res;
    if (generate(s, pr.gen, TextSink(), &res, &err) != 0) return std::fprintf(stderr, "generate: %s\n", err.c_str()), 1;
    for (const flyx_round_result& r : trace) {
      if (r.n_drafts != W || r.n_tokens == 0) continue;  // the prefill's entry, a round past the end
      ++n_rounds;
      std::vector<int> path;  // the accepted drafts, by index
      for (int j = 0, at = -1; j < r.n_accepted; ++j) {
        int child = -1;
        for (int i = 0; i < W && child < 0; ++i)
          if (r.parents[i] == at && r.drafts[i] == r.tokens[j]) child = i;
        if (child < 0) break;
        path.push_back(child);
        at = child;
      }
      for (int w = 1; w <= W; ++w) {
        int a = 0;
        while (a < int(path.size()) && path[size_t(a)] < w) ++a;
        got[size_t(w)] += a;
      }
    }
  }
  std::printf("\npaired: %d rounds of %s; by its first w drafts\n", n_rounds, e.round_name(widest).c_str());
  std::printf("%5s %9s %9s %10s %8s %8s\n", "w", "accepted", "tok/round", "round", "tok/ms", "x plain");
  for (int w = 1; w <= W; ++w) {
    const double per = 1 + got[size_t(w)] / std::max(1, n_rounds);
    size_t plan = e.widths().size();  // a round of w drafts of this drafter: its chain up to its depth, else a tree
    for (size_t i : kinds)
      if (e.widths()[i] == w && e.round_drafters()[i] == e.round_drafters()[widest] &&
          (plan == e.widths().size() || e.round_trees()[i] == (w > e.round_trunks()[widest])))
        plan = i;
    if (plan < e.widths().size())
      std::printf("%5d %9.3f %9.3f %10s %8.4f %8.3f\n", w, per - 1, per, e.round_name(plan).c_str(),
                  per / e.round_ms()[plan], per / e.round_ms()[plan] * plain_ms);
    else
      std::printf("%5d %9.3f %9.3f\n", w, per - 1, per);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: tree_bench MODEL [--ctx-at P] [--widths 1,2,...] [--rounds N] [--backend-options O] "
                         "[--kv 16|fp8] [--check T]\n");
    return 2;
  }
  const std::string model = argv[1];
  int at = 1024, rounds = 12, check = 0, drafted = 0;
  std::vector<int> widths = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  EngineOptions o;
  std::string drafter = "dflash2";
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--ctx-at") at = std::atoi(v.c_str());
    else if (k == "--widths") widths = parse_list(v);
    else if (k == "--rounds") rounds = std::atoi(v.c_str());
    else if (k == "--backend-options") o.backend_options = v;
    else if (k == "--kv") o.kv = v == "fp8" ? FLYX_KV_FP8 : FLYX_KV_16;
    else if (k == "--check") check = std::atoi(v.c_str());
    else if (k == "--drafted") drafted = std::atoi(v.c_str());
    else if (k == "--drafter") drafter = v;
    else if (k == "--drafter-file") o.drafter_files.push_back(v);
  }
  if (drafted > 0) {
    o.drafter = drafter;
    return drafted_rounds(model, o, drafted);
  }
  o.speculative = true;
  o.drafter = "none";
  o.lookup = true;  // the chain of given drafts to compare with
  o.max_ctx = at + rounds + 512 + 2 * check;  // a timing moves the context by about its rounds, then starts over
  const bool wide = std::any_of(widths.begin(), widths.end(), [](int w) { return w > FLYX_MAX_ROWS - 1; });
  if (wide && o.backend_options.find("wide=") == std::string::npos)
    o.backend_options += std::string(o.backend_options.empty() ? "" : ",") + "wide=1";
  Engine e;
  std::string err;
  if (!e.load(model, o, &err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return 1;
  }
  const auto* api = e.api();
  const int vocab = e.model().desc().vocab;
  // Plans: plain decoding, the chain of given drafts (15), and a tree plan for every width.
  flyx_plan* plain = nullptr;
  for (size_t i = 0; i < e.widths().size(); ++i)
    if (e.widths()[i] == 0) plain = e.round_plan(i);
  flyx_plan* chain15 = e.lookup_widths().empty() ? nullptr : e.lookup_plan(0);
  std::vector<flyx_plan*> trees;
  for (int w : widths) {
    auto rnd = e.model().round_program(w, -1, true);
    for (auto& n : rnd)
      if (n.kind == FLYX_NODE_GIVEN) n.arg1 = FLYX_GIVEN_TREE;
    flyx_program rp{FLYX_PHASE_ROUND, w, w + 1, static_cast<int32_t>(rnd.size()), rnd.data()};
    flyx_plan* p = nullptr;
    if (api->lower(e.bmodel(), &rp, &p) != 0) {
      std::fprintf(stderr, "lower tree %d: %s\n", w, api->last_error());
      return 1;
    }
    trees.push_back(p);
  }
  std::vector<int32_t> prompt(static_cast<size_t>(at));
  uint32_t rng = 99u;
  for (auto& t : prompt) {
    rng = rng * 1664525u + 1013904223u;
    t = int32_t((rng >> 8) % uint32_t(std::min(vocab, 30000)));
  }
  Session s(&e);
  if (!s.init(&err)) {
    std::fprintf(stderr, "session: %s\n", err.c_str());
    return 1;
  }
  // The state at the prompt's `at` tokens, its first token chosen.  The rounds timed below are launched on the
  // sequence directly, so the session does not know where they leave it: it starts over each time.
  auto prefill = [&] {
    s.reset_after_error();
    GenerateRequest rq;
    rq.prompt = prompt;
    rq.max_new_tokens = 1;
    rq.ignore_eos = true;
    GenerateResult res;
    if (generate(s, rq, TextSink{}, &res, &err) != 0) {
      std::fprintf(stderr, "prefill: %s\n", err.c_str());
      std::exit(1);
    }
  };
  prefill();
  if (check > 0) {
    // Plain decoding's continuation, then tree rounds that hide it on a random path of a random tree (the other
    // drafts never the token chosen at their depth): every round must emit its tokens and accept exactly that path.
    Session ref(&e);
    if (!ref.init(&err)) {
      std::fprintf(stderr, "session: %s\n", err.c_str());
      return 1;
    }
    std::vector<int32_t> want;
    GenerateRequest rq;
    rq.prompt = prompt;
    rq.max_new_tokens = check + FLYX_MAX_WIDE_ROWS;
    rq.ignore_eos = true;
    GenerateResult res;
    const TextSink collect = [&](std::string_view, const int32_t* t, int nt) {
      want.insert(want.end(), t, t + nt);
      return true;
    };
    if (generate(ref, rq, collect, &res, &err) != 0 || want.size() < size_t(check) + 2) {
      std::fprintf(stderr, "plain decoding: %s\n", err.c_str());
      return 1;
    }
    // s holds the prompt and its first token, which is want[0] (the same prefill).
    std::vector<int32_t> got = {want[0]};
    int n_rounds = 0, as_hidden = 0, off_chain = 0;
    uint32_t trng = 12345u;
    auto next = [&](uint32_t n) {
      trng = trng * 1664525u + 1013904223u;
      return (trng >> 8) % n;
    };
    while (got.size() < size_t(check)) {
      const size_t wi = next(uint32_t(widths.size()));
      const int W = widths[wi];
      const size_t nw = size_t(W);
      std::vector<int32_t> parents(nw), depth(nw), dr(nw);
      for (int i = 0; i < W; ++i) parents[size_t(i)] = int32_t(next(uint32_t(i + 1))) - 1;
      int deepest = 0;
      for (int i = 0; i < W; ++i) {
        depth[size_t(i)] = parents[size_t(i)] < 0 ? 1 : depth[size_t(parents[size_t(i)])] + 1;
        if (depth[size_t(i)] > depth[size_t(deepest)]) deepest = i;
      }
      const int x = next(2) ? deepest : int(next(uint32_t(W)));
      std::vector<bool> on(nw, false);
      for (int c = x; c >= 0; c = parents[size_t(c)]) on[size_t(c)] = true;
      const size_t base = got.size();
      for (int i = 0; i < W; ++i) {
        const int32_t truth = want[base + size_t(depth[size_t(i)]) - 1];
        int32_t t = int32_t(next(uint32_t(vocab)));
        if (t == truth) t = (t + 1) % vocab;
        dr[size_t(i)] = on[size_t(i)] ? truth : t;
        off_chain += on[size_t(i)] && i + 1 != depth[size_t(i)];
      }
      flyx_gen_params g{};
      g.top_p = 1.f;
      g.drafts = dr.data();
      g.n_drafts = W;
      g.drafts_at = static_cast<int32_t>(prompt.size() + got.size());
      g.parents = parents.data();
      flyx_round_result r{};
      if (api->launch(trees[wi], s.seq(), &g) != 0 || api->read(s.seq(), &r) != 0) {
        std::fprintf(stderr, "tree round: %s\n", api->last_error());
        return 1;
      }
      ++n_rounds;
      as_hidden += r.n_accepted == depth[size_t(x)];
      got.insert(got.end(), r.tokens, r.tokens + r.n_tokens);
    }
    got.resize(std::min(got.size(), want.size()));
    const bool same = std::equal(got.begin(), got.end(), want.begin());
    std::printf("check at context %d: %d tree rounds, %zu tokens %s plain decoding's; %d of %d rounds accepted their "
                "hidden path (%d path rows off the first chain)\n",
                at, n_rounds, got.size(), same ? "equal to" : "DIFFERENT from", as_hidden, n_rounds, off_chain);
    for (auto* p : trees) api->plan_free(p);
    return same && as_hidden == n_rounds ? 0 : 1;
  }
  int position = at;  // tokens in the state; the anchor is the next input
  std::vector<int32_t> drafts(FLYX_MAX_WIDE_ROWS);
  bool fresh = true;  // the state as prefill() left it
  int elsewhere = 0, furthest = at;  // timings whose first round started past `at`; the furthest position reached
  // ms a round of `plan` (drafts and parents given when w > 0), two in flight, from the prompt's `at` tokens
  auto time = [&](flyx_plan* plan, int w, const std::vector<int32_t>* parents) {
    if (!fresh) prefill();
    fresh = false;
    position = at;
    flyx_gen_params g{};
    g.top_p = 1.f;
    flyx_round_result r{};
    int in_flight = 0;
    bool first = true;
    auto read_one = [&] {
      if (api->read(s.seq(), &r) != 0) {
        std::fprintf(stderr, "read: %s\n", api->last_error());
        std::exit(1);
      }
      --in_flight;
      elsewhere += first && r.position - r.n_tokens != at;
      first = false;
      position = r.position;
      furthest = std::max(furthest, position);
    };
    double t0 = 0;
    const int warm = 3;
    for (int k = 0; k < warm + rounds; ++k) {
      if (k == warm) {
        while (in_flight > 0) read_one();
        t0 = os::now_ms();
      }
      for (int i = 0; i < w; ++i) {
        rng = rng * 1664525u + 1013904223u;
        drafts[size_t(i)] = int32_t((rng >> 8) % uint32_t(vocab));
      }
      if (w > 0) {
        g.drafts = drafts.data();
        g.n_drafts = w;
        g.drafts_at = position + in_flight + 1;  // nearly every round advances one position
        g.parents = parents ? parents->data() : nullptr;
      }
      if (api->launch(plan, s.seq(), &g) != 0) {
        std::fprintf(stderr, "launch: %s\n", api->last_error());
        std::exit(1);
      }
      if (++in_flight == FLYX_MAX_IN_FLIGHT) read_one();
    }
    while (in_flight > 0) read_one();
    return (os::now_ms() - t0) / rounds;
  };
  const double base = time(plain, 0, nullptr);
  std::printf("context %d, %d rounds each; plain round %.3f ms\n", at, rounds, base);
  if (chain15) {
    const double c = time(chain15, 15, nullptr);
    std::printf("chain of 15 given drafts (no tree): %.3f ms (%.3f x plain)\n", c, c / base);
  }
  const char* kinds[] = {"chain", "star", "binary", "random"};
  std::printf("%5s", "width");
  for (const char* k : kinds) std::printf(" %9s %6s", k, "x");
  std::printf("\n");
  for (size_t i = 0; i < widths.size(); ++i) {
    std::printf("%5d", widths[i]);
    for (const char* k : kinds) {
      uint32_t srng = 4242u + uint32_t(widths[i]);
      const auto parents = shape(k, widths[i], &srng);
      const double ms = time(trees[i], widths[i], &parents);
      std::printf(" %9.3f %6.3f", ms, ms / base);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  for (auto* p : trees) api->plan_free(p);
  std::printf("every timing's first round at %d tokens of context: %s (its rounds up to %d)\n", at,
              elsewhere ? "NO" : "yes", furthest);
  return elsewhere ? 1 : 0;
}
