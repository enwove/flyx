// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// plan::build (core/plan/plans.h) with a drafter that drafts nothing but trees and whose state follows its own rounds
// only (Model::drafts_trees, Model::restarts: DV): plain decoding, its widths the backend lowers, marked trees, its
// warm-up (a round of no drafts of it) and lookup rounds; a width pinned: the widest it lowers, at most that, alone
// (no plain or lookup round); every width refused said, those it lowers kept; where it lowers none, or not its
// warm-up, the rounds are the ones without it and the plans say why; out of GPU memory, the build fails (the load
// trims and builds again).  A drafter of the usual kind keeps plain decoding beside its widths.  A model and a backend
// that only describe and lower programs.
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "core/plan/plans.h"

struct flyx_plan {
  int width, drafter_block, tree;  // what the fake backend lowered
};

using namespace flyx;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

constexpr int kDrafterBlock = 7;
bool g_takes_drafter = true;  // the backend lowers the drafter's DRAFT nodes
bool g_takes_warm_up = true;  // ... and its CONTEXT node in a round of no drafts (its warm-up)
std::vector<int> g_refuses;   // ... but not of these widths
std::string g_why = "no rounds of this drafter here";
std::string g_error;

int lower(flyx_bmodel*, const flyx_program* p, flyx_plan** out) {
  int block = -1, tree = 0;
  for (int i = 0; i < p->n_nodes; ++i) {
    if (p->nodes[i].kind == FLYX_NODE_DRAFT) {
      block = p->nodes[i].block;
      tree = p->nodes[i].arg1;
      if (!g_takes_drafter || std::count(g_refuses.begin(), g_refuses.end(), p->width)) return g_error = g_why, -1;
    }
    if (p->nodes[i].kind == FLYX_NODE_CONTEXT && p->phase == FLYX_PHASE_ROUND && p->width == 0) {
      block = p->nodes[i].block;  // a warm-up
      if (!g_takes_drafter || !g_takes_warm_up) return g_error = g_why, -1;
    }
  }
  *out = new flyx_plan{p->width, block, tree};
  return 0;
}
void plan_free(flyx_plan* p) { delete p; }
const char* last_error() { return g_error.c_str(); }

// The lookup rounds a backend that lowers every round has: given_widths() and, wide, wide_given_widths().
std::vector<int> all_given() {
  std::vector<int> w = plan::given_widths();
  w.insert(w.end(), plan::wide_given_widths().begin(), plan::wide_given_widths().end());
  return w;
}

flyx_backend_api fake_backend() {
  flyx_backend_api api{};
  api.lower = lower;
  api.plan_free = plan_free;
  api.last_error = last_error;
  return api;
}

// A model with one drafter of widths {3, 7, 15}, or (sole) one that drafts trees of 15 (or of `widths`) and nothing
// else between.
class FakeModel final : public interfaces::Model {
 public:
  explicit FakeModel(bool sole, std::vector<int> widths = {15}) : sole_(sole), widths_(std::move(widths)) {}
  const flyx_model_desc& desc() const override { return desc_; }
  int drafters() const override { return 1; }
  const char* drafter_name(int) const override { return sole_ ? "dv" : "mtp"; }
  const char* drafter() const override { return drafter_name(0); }
  std::vector<int> draft_widths(int) const override {
    return sole_ ? widths_ : std::vector<int>{3, 7, 15};
  }
  int trained_ctx() const override { return 0; }
  int block_drafts(int) const override { return sole_ ? 0 : 15; }  // one pass: its widest chain; trees only: none
  bool drafts_trees(int) const override { return sole_; }
  bool restarts(int) const override { return sole_; }
  bool kv_fp8_by_default() const override { return false; }
  std::vector<flyx_node> prefill_program() const override { return {{FLYX_NODE_RUN, 0, 0, 0}}; }
  std::vector<flyx_node> round_program(int width, int d, bool given) const override {
    std::vector<flyx_node> p;
    if (given && width > 0) p.push_back({FLYX_NODE_GIVEN, -1, width, 0});
    if (!given && width > 0 && d >= 0)
      p.push_back({FLYX_NODE_DRAFT, kDrafterBlock, width, sole_ ? FLYX_DRAFT_TREE : 0});
    p.push_back({FLYX_NODE_RUN, 0, 0, 0});
    p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
    if (sole_ && !given && d >= 0) p.push_back({FLYX_NODE_CONTEXT, kDrafterBlock, 0, 0});
    return p;
  }

 private:
  bool sole_;
  std::vector<int> widths_;
  flyx_model_desc desc_{};
};

struct Built {
  plan::Plans plans;
  bool ok = false;
  std::string err;
};

Built build(const interfaces::Model& m, int fixed_width, bool given, bool draft_trees, bool takes,
            std::vector<int> refuses = {}, const std::string& why = "no rounds of this drafter here",
            bool warm_up = true) {
  g_takes_drafter = takes;
  g_takes_warm_up = warm_up;
  g_refuses = std::move(refuses);
  g_why = why;
  const flyx_backend_api api = fake_backend();
  Built b;
  b.ok = plan::build(&api, nullptr, m, 64, fixed_width, given, draft_trees, &b.plans, &b.err);
  return b;
}

std::string str(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
  return "{" + s + "}";
}

}  // namespace

int main() {
  const flyx_backend_api api = fake_backend();
  const FakeModel dv(true), chain(false);
  const std::vector<int> kWarm = {0, 15, 0}, kWho = {-1, 0, 0};
  {
    Built b = build(dv, -1, true, true, true);
    auto& p = b.plans;
    check(b.ok && p.widths == kWarm && p.drafters == kWho && p.drafter_trees == std::vector<bool>{false, true, false} &&
              p.rounds.size() == 3 && p.rounds[1]->tree == FLYX_DRAFT_TREE &&
              p.rounds[1]->drafter_block == kDrafterBlock && p.rounds[2]->width == 0 &&
              p.rounds[2]->drafter_block == kDrafterBlock && p.rounds[0]->drafter_block < 0,
          "a tree drafter the backend runs: plain decoding, its tree of 15, its warm-up " + str(p.widths));
    check(p.given_widths == all_given() && p.refused.empty() && p.refused_drafter < 0,
          "  lookup rounds beside them, nothing refused");
    check(p.tree_widths == std::vector<int>{3, 7, 15}, "  given trees still lowered (tests use them)");
    p.free(&api);
  }
  {
    Built b = build(dv, 15, true, false, true);
    check(b.ok && b.plans.widths == std::vector<int>{15} && b.plans.given.empty() && b.plans.rounds.size() == 1,
          "--width 15: its round alone, no warm-up or lookup round " + str(b.plans.widths));
    b.plans.free(&api);
    b = build(dv, 0, true, false, true);
    check(b.ok && b.plans.widths == std::vector<int>{0} && b.plans.drafters == std::vector<int>{-1} &&
              b.plans.given_widths == all_given() && b.plans.refused.empty(),
          "--width 0: plain decoding (and lookup), the drafter not tried");
    b.plans.free(&api);
    b = build(dv, 7, true, false, true);
    check(b.ok && b.plans.widths == std::vector<int>{7} && b.plans.drafters == std::vector<int>{0} &&
              b.plans.rounds.size() == 1 && b.plans.rounds[0]->width == 7 && b.plans.rounds[0]->tree,
          "--width 7: a tree of 7, below its widest " + str(b.plans.widths));
    b.plans.free(&api);
    b = build(dv, 31, true, false, true);
    check(b.ok && b.plans.widths == std::vector<int>{15}, "--width 31: its widest, 15 " + str(b.plans.widths));
    b.plans.free(&api);
  }
  {  // Two widths, the backend refusing the wider (rounds of more than 16 rows on a backend without them).
    const FakeModel dv2(true, {15, 31});
    Built b = build(dv2, -1, true, false, true);
    auto& p = b.plans;
    check(b.ok && p.widths == std::vector<int>{0, 15, 31, 0} && p.drafters == std::vector<int>{-1, 0, 0, 0} &&
              p.drafter_trees == std::vector<bool>{false, true, true, false} && p.rounds[2]->width == 31,
          "widths {15, 31}: plain decoding, both trees, the warm-up " + str(p.widths));
    p.free(&api);
    b = build(dv2, -1, true, false, true, {31}, "lower: 32 rows");
    check(b.ok && p.widths == kWarm && p.drafter_trees == std::vector<bool>{false, true, false} &&
              p.given_widths == all_given() && p.refused_drafter < 0,
          "widths {15, 31}, 31 refused: rounds of 15 and the warm-up " + str(p.widths));
    check(p.refused == "dv 31: lower round: lower: 32 rows", "  and the width refused said: " + p.refused);
    p.free(&api);
    b = build(dv2, 31, true, false, true, {31}, "lower: 32 rows");
    check(b.ok && p.widths == std::vector<int>{15} && p.drafters == std::vector<int>{0} && p.given.empty() &&
              p.refused == "dv 31: lower round: lower: 32 rows",
          "  --width 31, refused: its widest the backend runs, 15 " + str(p.widths) + " (" + p.refused + ")");
    p.free(&api);
    b = build(dv2, 20, true, false, true);
    check(b.ok && p.widths == std::vector<int>{20}, "  --width 20: a tree of 20 " + str(p.widths));
    p.free(&api);
    b = build(dv2, -1, true, false, false);
    check(b.ok && p.widths == std::vector<int>{0} && p.refused_drafter == 0 &&
              p.refused == "dv 15: lower round: no rounds of this drafter here; dv 31: lower round: no rounds of this "
                           "drafter here",
          "  both refused: as without it, both said (" + p.refused + ")");
    p.free(&api);
    b = build(dv2, -1, true, false, true, {}, "no canvas pass after plain decoding here", false);
    check(b.ok && p.widths == std::vector<int>{0} && p.drafters == std::vector<int>{-1} && p.refused_drafter == 0 &&
              p.given_widths == all_given() &&
              p.refused == "dv warm-up: lower round: no canvas pass after plain decoding here",
          "  its warm-up refused: as without it (its rounds after others' would draft nothing), said (" + p.refused +
              ")");
    p.free(&api);
  }
  {  // Out of GPU memory is no refusal: the load gives back what the backend keeps for speed, then builds again.
    Built b = build(dv, -1, true, false, false, {}, "out of GPU memory: 300 MiB asked");
    check(!b.ok && b.err.find("out of GPU memory") != std::string::npos && b.plans.refused_drafter < 0,
          "a tree drafter refused for want of GPU memory: the build fails (" + b.err + ")");
    b.plans.free(&api);
    b = build(dv, -1, true, false, true, {}, "out of GPU memory: 20 MiB asked", false);
    check(!b.ok && b.err.find("out of GPU memory") != std::string::npos,
          "  its warm-up refused for want of GPU memory: the build fails (" + b.err + ")");
    b.plans.free(&api);
  }
  {
    Built b = build(dv, -1, true, true, false);
    auto& p = b.plans;
    check(b.ok && p.widths == std::vector<int>{0} && p.drafters == std::vector<int>{-1} &&
              p.drafter_trees == std::vector<bool>{false} && p.rounds.size() == 1 && p.rounds[0]->drafter_block < 0,
          "a tree drafter the backend refuses: plain decoding " + str(p.widths));
    check(p.given_widths == all_given(), "  with lookup rounds, as without the drafter");
    check(p.refused == "dv 15: lower round: no rounds of this drafter here" && p.refused_drafter == 0,
          "  and why: " + p.refused);
    check(b.err.empty(), "  no error left behind");
    p.free(&api);
  }
  {
    Built b = build(chain, -1, true, false, true);
    check(b.ok && b.plans.widths == std::vector<int>{0, 3, 7, 15} && b.plans.given_widths == all_given() &&
              b.plans.refused.empty(),
          "a drafter of the usual kind: plain decoding, its widths and lookup " + str(b.plans.widths));
    b.plans.free(&api);
    b = build(chain, -1, true, false, false);
    check(!b.ok, "  one the backend refuses: the load fails (" + b.err + ")");
    b.plans.free(&api);
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
