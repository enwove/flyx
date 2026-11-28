// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/plan/plans.h"

#include <algorithm>
#include <optional>

namespace flyx::plan {

int Plans::widest() const { return widths.empty() ? 0 : *std::max_element(widths.begin(), widths.end()); }

void Plans::free(const flyx_backend_api* api) {
  for (auto* p : rounds) api->plan_free(p);
  for (auto* p : given) api->plan_free(p);
  for (auto* p : trees) api->plan_free(p);
  for (auto* p : adaptive)
    if (p) api->plan_free(p);
  if (prefill) api->plan_free(prefill);
  if (vision) api->plan_free(vision);
  rounds.clear();
  widths.clear();
  drafters.clear();
  drafter_trees.clear();
  given.clear();
  given_widths.clear();
  trees.clear();
  tree_widths.clear();
  adaptive.clear();
  adaptive_checkpoints.clear();
  refused.clear();
  refused_drafter = -1;
  oom = false;
  prefill = nullptr;
  vision = nullptr;
}

bool build(const flyx_backend_api* api, flyx_bmodel* bm, const interfaces::Model& model, int prefill_chunk,
           int fixed_width, bool given, bool draft_trees, Plans* out, std::string* err, bool adaptive) {
  out->oom = false;
  auto pre = model.prefill_program();
  flyx_program pp{FLYX_PHASE_PREFILL, 0, prefill_chunk, static_cast<int32_t>(pre.size()), pre.data()};
  if (const int rc = api->lower(bm, &pp, &out->prefill); rc != 0) {
    out->oom = rc == FLYX_ERR_OOM;
    *err = std::string("lower prefill: ") + api->last_error();
    return false;
  }
  if (auto vp = model.vision_program(); !vp.empty()) {
    constexpr int32_t kMaxVisionPatches = 16384;
    flyx_program vprog{FLYX_PHASE_VISION, 0, kMaxVisionPatches, static_cast<int32_t>(vp.size()), vp.data()};
    if (const int rc = api->lower(bm, &vprog, &out->vision); rc != 0) {
      out->oom = rc == FLYX_ERR_OOM;
      *err = std::string("lower vision: ") + api->last_error();
      return false;
    }
  }
  // Rounds to choose from: plain decoding and each drafter's draft counts (a fixed width: that many, every drafter).
  auto add = [&](int width, int drafter) {
    out->widths.push_back(width);
    out->drafters.push_back(drafter);
    out->drafter_trees.push_back(false);
  };
  auto lower_round = [&](int k, int drafter, bool given_drafts, std::vector<flyx_plan*>* to, bool tree = false,
                         bool* is_oom = nullptr) {
    if (is_oom) *is_oom = false;
    auto rnd = model.round_program(k, drafter, given_drafts);
    for (auto& n : rnd) {  // the tree is the round's (DV's say so themselves): the node of its drafts says so
      if (tree && n.kind == FLYX_NODE_GIVEN) n.arg1 = FLYX_GIVEN_TREE;
      if (tree && n.kind == FLYX_NODE_DRAFT) n.arg1 = FLYX_DRAFT_TREE;
    }
    flyx_program rp{FLYX_PHASE_ROUND, k, k + 1, static_cast<int32_t>(rnd.size()), rnd.data()};
    flyx_plan* plan = nullptr;
    if (const int rc = api->lower(bm, &rp, &plan); rc != 0) {
      if (is_oom) *is_oom = rc == FLYX_ERR_OOM;
      else out->oom = rc == FLYX_ERR_OOM;
      *err = std::string("lower round: ") + api->last_error();
      return false;
    }
    to->push_back(plan);
    return true;
  };
  // A round the engine can do without: lowered where the backend takes it, else the backend's refusal.  *err is
  // left as it was.
  struct ProbeErr {
    std::string msg;
    bool oom = false;
  };
  auto probe_round = [&](int k, int drafter, bool given_drafts, std::vector<flyx_plan*>* to,
                         bool tree) -> std::optional<ProbeErr> {
    std::string e;
    bool oom = false;
    std::swap(e, *err);
    const bool ok = lower_round(k, drafter, given_drafts, to, tree, &oom);
    std::swap(e, *err);
    if (ok) return std::nullopt;
    return ProbeErr{std::move(e), oom};
  };
  auto lower_given_trees = [&]() {
    for (int k : tree_widths()) {
      if (auto why = probe_round(k, -1, true, &out->trees, true)) {
        if (why->oom) return out->oom = true, *err = why->msg, false;
        break;
      }
      out->tree_widths.push_back(k);
    }
    return true;
  };
  // A drafter that drafts nothing but trees (DV): its widths where the backend lowers them, each refused said
  // (cpu_ref has no tree rounds); none lowered, the rounds are the ones without it.  Pinned (a width asked for), its
  // widest at most that many drafts the backend lowers, alone: no plain decoding or lookup round beside it.
  // Unpinned, one whose state follows its own rounds only (Model::restarts: DV's canvas) also has its warm-up,
  // round_program(0, d): plain decoding that brings the state along (the controller runs it before the drafter's
  // first round after any other).
  int pinned = -1;  // pinned to such a drafter: its rounds alone
  auto lower_trees = [&](int d) {
    const std::vector<int> drafts = model.draft_widths(d);
    std::vector<int> tries = drafts;
    if (fixed_width > 0) {  // widest first, capped at the width asked for, until one lowers
      tries.clear();
      for (auto w = drafts.rbegin(); w != drafts.rend(); ++w)
        if (std::find(tries.begin(), tries.end(), std::min(fixed_width, *w)) == tries.end())
          tries.push_back(std::min(fixed_width, *w));
    }
    std::vector<flyx_plan*> lowered;
    std::vector<int> widths;
    auto free_lowered = [&] {
      for (auto* p : lowered) api->plan_free(p);
      lowered.clear();
    };
    auto refuse = [&](int k, const std::string& e) {
      out->refused += (out->refused.empty() ? "" : "; ") + std::string(model.drafter_name(d)) + " " +
                      (k > 0 ? std::to_string(k) : std::string("warm-up")) + ": " + e;
    };
    for (int k : tries) {
      if (auto why = probe_round(k, d, false, &lowered, true)) {
        // Memory the load can give back (engine.cpp: trim) is not a reason to draft without it: the load lowers
        // the plans again once it has (the ones lowered here freed first).
        if (why->oom) return free_lowered(), out->oom = true, *err = why->msg, false;
        refuse(k, why->msg);
        continue;
      }
      widths.push_back(k);
      if (fixed_width > 0) break;
    }
    if (!widths.empty() && fixed_width < 0 && model.restarts(d)) {
      if (auto why = probe_round(0, d, false, &lowered, false)) {
        if (why->oom) return free_lowered(), out->oom = true, *err = why->msg, false;
        refuse(0, why->msg);  // without its warm-up the drafter is not run: its rounds after others' would draft nothing
        widths.clear();
      } else {
        widths.push_back(0);
      }
    }
    if (widths.empty()) {
      free_lowered();
      out->refused_drafter = d;
      return true;
    }
    for (size_t j = 0; j < widths.size(); ++j) {
      add(widths[j], d);
      out->drafter_trees.back() = widths[j] > 0;
      out->rounds.push_back(lowered[j]);
    }
    if (fixed_width > 0) pinned = d;
    return true;
  };
  auto add_round = [&](int k, int d) {  // a round of a drafter of chains (or plain decoding), lowered
    add(k, d);
    return lower_round(k, d, false, &out->rounds);
  };
  if (fixed_width < 0 && !add_round(0, -1)) return false;
  if (fixed_width != 0)
    for (int d = 0; d < model.drafters() && pinned < 0; ++d) {
      const std::vector<int> drafts = model.draft_widths(d);
      if (drafts.empty()) continue;
      if (model.drafts_trees(d)) {
        if (!lower_trees(d)) return false;
        if (pinned >= 0) {  // its rounds alone
          out->widths.assign(1, out->widths.back());
          out->drafters.assign(1, d);
          out->drafter_trees.assign(1, true);
          flyx_plan* keep = out->rounds.back();
          out->rounds.pop_back();
          for (auto* p : out->rounds) api->plan_free(p);
          out->rounds.assign(1, keep);
        }
        continue;
      }
      if (fixed_width > 0) {
        if (!add_round(std::min(fixed_width, drafts.back()), d)) return false;
      } else {
        for (int k : drafts)
          if (!add_round(k, d)) return false;
      }
    }
  if (out->widths.empty() && !add_round(0, -1)) return false;  // pinned, with no drafter to draft: plain decoding
  if (draft_trees && fixed_width < 0)
    for (int d = 0; d < model.drafters(); ++d)
      for (int k : draft_tree_widths()) {
        if (model.drafts_trees(d)) break;  // its rounds are trees already
        if (model.widest_trees(d) && k != FLYX_MAX_ROWS - 1) continue;
        if (auto why = probe_round(k, d, false, &out->rounds, true)) {
          if (why->oom) return out->oom = true, *err = why->msg, false;
          break;
        }
        add(k, d);
        out->drafter_trees.back() = true;
      }
  if (pinned >= 0) {  // a drafter's rounds alone, pinned: no lookup rounds beside them
    return lower_given_trees();
  }
  if (given) {
    for (int k : given_widths()) {
      if (!lower_round(k, -1, true, &out->given)) return false;
      out->given_widths.push_back(k);
    }
    for (int k : wide_given_widths()) {
      if (auto why = probe_round(k, -1, true, &out->given, false)) {
        if (why->oom) return out->oom = true, *err = why->msg, false;
        break;
      }
      out->given_widths.push_back(k);
    }
  }
  if (!lower_given_trees()) return false;
  if (adaptive && fixed_width < 0 && model.drafters() > 0) {
    out->adaptive.assign(size_t(model.drafters()), nullptr);
    out->adaptive_checkpoints.resize(size_t(model.drafters()));
    for (int d = 0; d < model.drafters(); ++d) {
      const std::vector<int> cps = model.adaptive_checkpoints(d);
      if (cps.size() < 2 || cps.size() > FLYX_MAX_STOP_BUCKETS) continue;
      auto rnd = model.round_program(cps.back(), d, false);
      flyx_program rp{FLYX_PHASE_ROUND, cps.back(), cps.back() + 1, static_cast<int32_t>(rnd.size()), rnd.data(),
                      static_cast<int32_t>(cps.size()), cps.data(), model.draft_steps_cheap(d) ? 1 : 0};
      flyx_plan* plan = nullptr;
      const int rc = api->lower(bm, &rp, &plan);
      if (rc == FLYX_ERR_OOM) return out->oom = true, *err = std::string("lower adaptive: ") + api->last_error(), false;
      if (rc == 0 && plan) {
        out->adaptive[size_t(d)] = plan;
        out->adaptive_checkpoints[size_t(d)] = cps;
      }
    }
  }
  return true;
}

}  // namespace flyx::plan
