// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV's rounds chosen by SpecController replayed on pinned runs' rounds: what the rule decodes at on a
// prompt set without the GPU, against DV of 15 or of 31 drafts every round and plain decoding.
//
//   dv_choice [NAME=VALUE ...] [--each] [-v] SET.txt ...
//
// SET.txt: a line a prompt, "P ; t1 t2 ... ; u1 u2 ...": its prompt's tokens, then the tokens each round of DV of 15
// drafts every round gave, and each round of DV of 31 every round (greedy: the same text), the rounds after the
// prefill's token; a line starting with `#` says what the set is.  bench/replay/dv_choice.py writes them from flyx
// bench --json and run --rounds records, and makes the derived sets.
//
// A round of width w at text position p gives the tokens the pinned round of w covering p gave from p on (it ends
// where that round ended: the drafts after p matched, the next did not); plain decoding and DV's warm-up give one
// token.  One round at a time (the engine has two in flight, so its choices land a round later), lookup off.  The sets
// run in turn on one engine (SharedTokens: the engine's means and probe credit), each prompt a session's reply, as
// `flyx bench --prompts` runs them; --each: a new engine for each set.  -v: each prompt's rounds, a letter each (p
// plain decoding, a DV of 15, b DV of 31, w its warm-up).
//
// NAME=VALUE: one of SpecController's public constants (kProbe=64, kTreePrior=4, ...: this program is built with
// them as variables, bench/replay/tunable_controller.cmake), a round's cost in ms at 64 tokens of context (plain,
// dv15, dv31, warm) or per token past it (plain/token, ...), or credit=full (the engine's probe credit full at load,
// not 0).  The default costs are MiniCPM5-1B's on the RTX 5060 Ti.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "core/speculation/controller.h"

using flyx::SharedTokens;
using flyx::SpecController;

namespace {

// The rounds, as the engine's plans hold them for a DV file of width 31: plain decoding, DV's trees of 15 and of 31
// (no trunk), its warm-up.
const std::vector<int> kWidths = {0, 15, 31, 0}, kDrafters = {-1, 0, 0, 0}, kTrunks = {0, -1, -1, 0};
const char* const kNames[] = {"plain", "dv15", "dv31", "warm"};
constexpr int kCostAt = 64;

struct Prompt {
  int context = 0;           // the prompt's tokens
  std::vector<int> ends[2];  // per text position: the tokens the pinned round of 15 (31) covering it gives from there
  int tokens = 0;            // the text both pinned runs wrote
};

std::vector<int> ends(const std::vector<int>& rounds) {
  std::vector<int> e;
  for (int t : rounds)
    for (int k = 0; k < t; ++k) e.push_back(t - k);
  return e;
}

bool load(const std::string& path, std::vector<Prompt>* out) {
  std::ifstream in(path);
  if (!in) return false;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::stringstream ss(line);
    Prompt p;
    std::string tok;
    ss >> p.context >> tok;
    std::vector<int> rounds[2];
    for (int w = 0; w < 2; ++w)
      while (ss >> tok && tok != ";") rounds[w].push_back(std::atoi(tok.c_str()));
    for (int w = 0; w < 2; ++w) p.ends[w] = ends(rounds[w]);
    p.tokens = static_cast<int>(std::min(p.ends[0].size(), p.ends[1].size()));
    out->push_back(p);
  }
  return true;
}

// Another revision's controller (bench/CMakeLists.txt: FLYX_DV_CHOICE_SRC) may lack what this one has: a reply's start
// was restart() before reply(), and an engine's probe credit came with the rule's revision.
template <typename C>
auto start_reply(C& c, int) -> decltype(c.reply(), void()) {
  c.reply();
}
template <typename C>
void start_reply(C& c, long) {
  c.restart();
}
template <typename S>
auto fill_credit(S& s, int) -> decltype(s.add_credit(0.0, 0.0), bool()) {
  s.add_credit(1e12, 1e12);  // the controller keeps it within its bound from its first round on
  return true;
}
template <typename S>
bool fill_credit(S&, long) {
  return false;
}

bool set_constant(const std::string& name, double value) {
#define FLYX_KNOB(type, k)                       \
  if (name == #k) {                              \
    SpecController::k = static_cast<type>(value); \
    return true;                                 \
  }
#include "dv_choice_knobs.inc"
#undef FLYX_KNOB
  return false;
}

void print_constants() {
  std::printf("constants:");
#define FLYX_KNOB(type, k) std::printf(" %s=%g", #k, static_cast<double>(SpecController::k));
#include "dv_choice_knobs.inc"
#undef FLYX_KNOB
  std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<double> cost = {4.40, 12.85, 15.25, 11.73}, per_token = {0.000083, 0.00157, 0.00215, 0.00093};
  bool each = false, verbose = false, credit_full = false;
  std::vector<std::string> files;
  for (int a = 1; a < argc; ++a) {
    const std::string s = argv[a];
    const size_t eq = s.find('=');
    if (s == "--each") {
      each = true;
    } else if (s == "-v") {
      verbose = true;
    } else if (s == "credit=full" || s == "credit=0") {
      credit_full = s == "credit=full";
    } else if (eq != std::string::npos) {
      const std::string name = s.substr(0, eq);
      const double value = std::atof(s.c_str() + eq + 1);
      bool known = set_constant(name, value);
      for (size_t i = 0; i < 4 && !known; ++i) {
        if (name == kNames[i]) cost[i] = value, known = true;
        if (name == std::string(kNames[i]) + "/token") per_token[i] = value, known = true;
      }
      if (!known) {
        std::fprintf(stderr, "dv_choice: %s: not a SpecController constant or a round's cost\n", name.c_str());
        return 2;
      }
    } else {
      files.push_back(s);
    }
  }
  if (files.empty()) {
    std::fprintf(stderr, "usage: dv_choice [NAME=VALUE ...] [--each] [-v] SET.txt ...\n");
    return 2;
  }
  print_constants();
  std::printf("the engine's probe credit at load: %s\n", credit_full ? "full" : "0");
  std::printf("costs (ms at %d tokens + per token past it):", kCostAt);
  for (size_t i = 0; i < 4; ++i) std::printf(" %s %.2f + %.6f", kNames[i], cost[i], per_token[i]);
  std::printf("\n");
  auto cost_at = [&](size_t i, int context) { return cost[i] + per_token[i] * std::max(0, context - kCostAt); };
  SharedTokens shared;
  auto new_engine = [&]() {  // its means none yet, its probe credit 0 (as the engine's at load) or full
    shared.resize(kWidths.size());
    return !credit_full || fill_credit(shared, 0);
  };
  if (!new_engine()) {
    std::fprintf(stderr, "dv_choice: credit=full: this controller keeps no engine's probe credit\n");
    return 2;
  }
  double all_ms = 0, all_tokens = 0, all_pinned[3] = {};
  for (const std::string& f : files) {
    std::vector<Prompt> prompts;
    if (!load(f, &prompts)) {
      std::fprintf(stderr, "dv_choice: cannot read %s\n", f.c_str());
      return 1;
    }
    if (each) (void)new_engine();
    long by[4] = {};
    double ms = 0, tokens = 0, pinned[3] = {};  // pinned: plain decoding, DV of 15, of 31 every round
    int with_dv = 0;
    for (size_t q = 0; q < prompts.size(); ++q) {
      const Prompt& p = prompts[q];
      SpecController c;
      c.reset(&kWidths, &kDrafters, &cost, &kTrunks, &per_token, kCostAt, &shared);
      start_reply(c, 0);
      int at = 0, context = p.context + 1;
      long dv = 0;
      double reply_ms = 0;
      std::string kinds;
      while (at < p.tokens) {
        const size_t i = static_cast<size_t>(c.choose(context));
        const int t = i == 1 || i == 2 ? std::min(p.ends[i - 1][size_t(at)], p.tokens - at) : 1;
        ++by[i];
        dv += i == 1 || i == 2;
        reply_ms += cost_at(i, context);
        c.observe(i, t - 1, 0);
        if (verbose) kinds += "pabw"[i];
        at += t;
        context += t;
      }
      ms += reply_ms;
      tokens += p.tokens;
      with_dv += dv > 0;
      for (int w = 0; w < 3; ++w) {
        int a = 0, x = p.context + 1;
        while (a < p.tokens) {
          const int t = w == 0 ? 1 : std::min(p.ends[w - 1][size_t(a)], p.tokens - a);
          pinned[w] += cost_at(size_t(w), x);
          a += t;
          x += t;
        }
      }
      if (verbose) std::printf("  %3zu (%4d): %7.1f ms %s\n", q, p.context, reply_ms, kinds.c_str());
    }
    std::printf("%-16s %7.1f tok/s | pinned 15 %7.1f, 31 %7.1f, plain %7.1f | rounds plain %ld, 15 %ld, 31 %ld, "
                "warm-up %ld | DV in %d of %zu\n",
                f.substr(f.find_last_of("/\\") + 1).c_str(), 1000 * tokens / ms, 1000 * tokens / pinned[1],
                1000 * tokens / pinned[2], 1000 * tokens / pinned[0], by[0], by[1], by[2], by[3], with_dv,
                prompts.size());
    all_ms += ms, all_tokens += tokens;
    for (int w = 0; w < 3; ++w) all_pinned[w] += pinned[w];
  }
  if (files.size() > 1)
    std::printf("%-16s %7.1f tok/s | pinned 15 %7.1f, 31 %7.1f, plain %7.1f\n", "all", 1000 * all_tokens / all_ms,
                1000 * all_tokens / all_pinned[1], 1000 * all_tokens / all_pinned[2],
                1000 * all_tokens / all_pinned[0]);
  return 0;
}
