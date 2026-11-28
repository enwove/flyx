// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/calibrate.h"

#include <algorithm>
#include <cstdio>
#include <memory>

#include "core/os/log.h"
#include "core/os/os.h"

namespace flyx::spec {

namespace {

// A scratch sequence brought to the end of `prompt` before each round: prefilled once and checkpointed there, then
// restored (a copy of the state on the device, where a prefill of the 27B's 64 tokens takes 57 ms on a 5070 Ti), or
// prefilled again where the backend's checkpoint lies before the prompt's end or it keeps none.
class Scratch {
 public:
  // Room past the prompt: the rounds a calibration runs (four, of up to FLYX_MAX_WIDE_ROWS tokens each) and a round's.
  static constexpr int kRoom = 512;
  Scratch(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill, int chunk,
          const std::vector<int32_t>& prompt)
      : api_(api), prefill_(prefill), chunk_(std::max(1, chunk)), prompt_(prompt) {
    if (api_->seq_new(bm, static_cast<int32_t>(prompt.size()) + kRoom, &seq_) != 0) seq_ = nullptr;
  }
  ~Scratch() {
    if (seq_) api_->seq_free(seq_);
  }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  flyx_seq* seq() const { return seq_; }

  // The sequence at the prompt's end with nothing in flight; false on an error.
  bool start(const flyx_gen_params& g) {
    const int n = static_cast<int>(prompt_.size());
    if (!seq_) return false;
    if (anchor_ >= 0) return api_->restore(seq_, kSlot, anchor_) == n;
    flyx_round_result r{};
    if (api_->rewind(seq_, 0, 0) < 0) return false;
    for (int at = 0; at < n; at += chunk_) {  // in passes of the prefill plan's rows, the last choosing the anchor
      const int len = std::min(chunk_, n - at);
      if (api_->prefill(prefill_, seq_, prompt_.data() + at, len, at + len == n ? &g : nullptr) != 0) return false;
    }
    if (api_->read(seq_, &r) != 0) return false;
    if (r.n_tokens > 0 && api_->checkpoint(seq_, kSlot) == n) anchor_ = r.tokens[0];
    return true;
  }

 private:
  static constexpr int32_t kSlot = 0;
  const flyx_backend_api* api_;
  flyx_plan* prefill_;
  int chunk_;
  const std::vector<int32_t>& prompt_;
  flyx_seq* seq_ = nullptr;
  int32_t anchor_ = -1;  // the token the prompt's prefill chose, once its state is checkpointed
};

}  // namespace

void time_rounds(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill, int chunk,
                 const std::vector<int32_t>& prompt, const std::vector<TimedGroup>& groups, int verbose) {
  for (const TimedGroup& grp : groups) grp.out->assign(grp.rounds->size(), 1.0);
  Scratch scratch(api, bm, prefill, chunk, prompt);
  flyx_seq* seq = scratch.seq();
  if (!seq) return;
  flyx_gen_params g{};
  g.limit = 0;
  bool ok = true;
  bool clocks_warm = false;
  flyx_round_result r{};

  for (const TimedGroup& grp : groups) {
    const auto& rounds = *grp.rounds;
    const auto& labels = *grp.labels;
    auto& ms = *grp.out;
    for (size_t i = 0; ok && i < rounds.size(); ++i) {
      // On the very first round after model load, GPUs often sit in an idle P-state while weights streamed from disk.
      // Run a short untimed burst (up to ~200 ms / 12 launches) so clocks settle before timing width 0.
      if (!clocks_warm) {
        ok = scratch.start(g);
        if (ok) ok = api->launch(rounds[i], seq, &g) == 0 && api->read(seq, &r) == 0;  // capture graph / load modules
        const double wake_t0 = os::now_ms();
        int in_flight = 0;
        for (int w = 0; ok && w < 12; ++w) {
          if (w >= 2 && os::now_ms() - wake_t0 >= 200.0) break;
          ok = api->launch(rounds[i], seq, &g) == 0;
          if (++in_flight == FLYX_MAX_IN_FLIGHT && ok) {
            ok = api->read(seq, &r) == 0;
            --in_flight;
          }
        }
        while (ok && in_flight > 0 && api->read(seq, &r) == 0) --in_flight;
        clocks_warm = true;
        if (!ok) {
          if (verbose) log::warn("flyx: calibrating %s failed: %s", labels[i].c_str(), api->last_error());
          break;
        }
      }
      // One round untimed (its graph captured, its kernels' modules loaded), then three timed: the device takes the
      // same time for the same work, to well under the 2 ms one more draft row costs the 27B.  The group's repeats,
      // the fastest kept.
      const int warm = 1, timed = 3;
      double best = 0;
      for (int rep = 0; ok && rep < std::max(1, grp.repeats); ++rep) {
        ok = scratch.start(g);
        double t0 = 0;
        int in_flight = 0;
        for (int k = 0; ok && k < warm + timed; ++k) {
          if (k == warm) {
            while (in_flight > 0 && api->read(seq, &r) == 0) --in_flight;
            t0 = os::now_ms();
          }
          ok = api->launch(rounds[i], seq, &g) == 0;
          if (++in_flight == FLYX_MAX_IN_FLIGHT && ok) {
            ok = api->read(seq, &r) == 0;
            --in_flight;
          }
        }
        while (ok && in_flight > 0 && api->read(seq, &r) == 0) --in_flight;
        const double t = (os::now_ms() - t0) / timed;
        if (ok && (rep == 0 || t < best)) best = t;
      }
      if (ok) ms[i] = best;
      else if (verbose) log::warn("flyx: calibrating %s failed: %s", labels[i].c_str(), api->last_error());
    }
    if (verbose && !rounds.empty()) {
      std::string line = std::string("flyx: ") + grp.what + " cost:";
      for (size_t i = 0; i < rounds.size(); ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), " %s=%.2f ms", labels[i].c_str(), ms[i]);
        line += buf;
      }
      log::info("%s", line.c_str());
    }
  }
}

bool costs_plausible(const std::vector<double>& at_short, const std::vector<double>& at_long, size_t plain,
                     double tolerance) {
  if (at_short.size() != at_long.size() || plain >= at_short.size()) return false;
  for (size_t i = 0; i < at_short.size(); ++i)
    if (at_long[i] < at_short[i] * (1 - tolerance) || at_short[plain] > at_short[i] * (1 + tolerance) ||
        at_long[plain] > at_long[i] * (1 + tolerance))
      return false;
  return true;
}

double time_batch(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill, int chunk,
                  const std::vector<int32_t>& prompt, flyx_plan* round, int n) {
  if (n < 2 || !api->launch_batch) return 0;
  std::vector<std::unique_ptr<Scratch>> scratch;
  std::vector<flyx_seq*> seqs;
  flyx_gen_params g{};
  g.limit = 0;
  for (int i = 0; i < n; ++i) {
    scratch.push_back(std::make_unique<Scratch>(api, bm, prefill, chunk, prompt));
    if (!scratch.back()->seq()) return 0;
    seqs.push_back(scratch.back()->seq());
  }
  const std::vector<flyx_plan*> plans(size_t(n), round);
  const std::vector<const flyx_gen_params*> gens(size_t(n), &g);
  flyx_round_result r{};
  // The batches in flight read back: every sequence's round of the oldest.
  auto read_one = [&] {
    bool ok = true;
    for (flyx_seq* s : seqs) ok = api->read(s, &r) == 0 && ok;
    return ok;
  };
  // As time_rounds' rounds: one batch untimed (its graph captured), then three timed, two in flight; twice, the
  // faster kept.
  const int warm = 1, timed = 3;
  double best = 0;
  for (int rep = 0; rep < 2; ++rep) {
    for (auto& sc : scratch)
      if (!sc->start(g)) return 0;
    double t0 = 0;
    int in_flight = 0;
    for (int k = 0; k < warm + timed; ++k) {
      if (k == warm) {
        for (; in_flight > 0; --in_flight)
          if (!read_one()) return 0;
        t0 = os::now_ms();
      }
      if (api->launch_batch(plans.data(), seqs.data(), gens.data(), n) != 0) return 0;
      if (++in_flight == FLYX_MAX_IN_FLIGHT) {
        if (!read_one()) return 0;
        --in_flight;
      }
    }
    for (; in_flight > 0; --in_flight)
      if (!read_one()) return 0;
    const double t = (os::now_ms() - t0) / timed;
    if (rep == 0 || t < best) best = t;
  }
  return best;
}

void warm_rounds(const flyx_backend_api* api, flyx_bmodel* bm, flyx_plan* prefill,
                 const std::vector<flyx_plan*>& rounds, const std::vector<int32_t>& prompt) {
  if (rounds.empty()) return;
  Scratch scratch(api, bm, prefill, static_cast<int>(prompt.size()), prompt);
  flyx_gen_params g{};
  g.limit = 0;
  flyx_round_result r{};
  for (flyx_plan* round : rounds)
    if (!scratch.start(g) || api->launch(round, scratch.seq(), &g) != 0 || api->read(scratch.seq(), &r) != 0)
      break;  // a warm-up only: a request that fails the same way says so
}

}  // namespace flyx::spec
