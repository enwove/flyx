// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV's trees and selector table (ops/draft/ref.h) against the Python reference (python/reference/flyx_ref/dv.py), on
// the cases tests/e2e/make_tiny.py writes to ASSETS/dv_cases.json:
//   * beam_tree and pool_tree on 1000 random tables each, the same drafts and parents: the tables (log-probabilities,
//     a fifth of them in ties) are made here again from each case's seed, bit for bit as make_tiny.py makes them, with
//     slots past the canvas, cov, beams wider than a depth's children, and every parent before its children; plus
//     fixed edge cases (DYN 31:8:cov and ST 31/7/1 on 32 slots, MASK drafts, padded beams, everything dead);
//   * slot_coverage (cov's term) from each slot's log-probabilities, also made from the seed, against the term the
//     Python reference adds to the slot's edges (written in the case; the beam trees take it from there, so that a
//     last-bit difference between two exp and log libraries cannot turn a tie), including slots under its floor;
//   * selector_table from the selector's weights within float64 rounding.
//     dv_tree_test ASSETS
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "ops/draft/ref.h"
#include "ops/linear/ref.h"
#include "tests/common/check.h"

namespace {

// make_tiny.py splitmix_uniform: (splitmix64(seed * 2^32 + i) >> 11) * 2^-53.
double uniform(uint64_t seed, uint64_t i) {
  uint64_t z = seed * (uint64_t(1) << 32) + i + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return double((z ^ (z >> 31)) >> 11) * 0x1p-53;
}

struct TreeCase {
  std::vector<double> table, logp;
  std::vector<int32_t> cand;
  std::vector<uint8_t> live;
};

// make_tiny.py dv_tree_table: the table [depths][k][k] in [-8, 0] (ties: multiples of 1/4), then each slot's k
// log-probabilities, multiples of 1/16 in (-19, -3], every third slot's 17 lower (its top-k mass under 1e-6).
TreeCase make_case(const nlohmann::json& c) {
  const int depths = c["depths"], k = c["k"], live_depths = c["live_depths"];
  const uint64_t seed = c["seed"];
  const size_t n = size_t(depths) * size_t(k) * size_t(k);
  TreeCase t;
  for (size_t i = 0; i < n; ++i) {
    const double u = uniform(seed, i);
    t.table.push_back(c["ties"].get<bool>() ? -std::floor(u * 8) / 4 : -8 * u);
  }
  for (int d = 0; d < depths; ++d) {
    t.live.push_back(d < live_depths);
    for (int j = 0; j < k; ++j) {
      const double u = uniform(seed, n + size_t(d) * size_t(k) + size_t(j));
      t.logp.push_back(-std::floor(u * 256) / 16 - 3 - (d % 3 == 2 ? 17 : 0));
      t.cand.push_back(1000 + d * k + j);
    }
  }
  return t;
}

void trees(const nlohmann::json& cases) {
  int n[2] = {}, same[2] = {}, dead[2] = {}, masks[2] = {}, ties[2] = {}, padded = 0, ordered = 0;
  int cov_slots = 0, floored = 0;
  double cov_worst = 0;
  for (const auto& c : cases) {
    const bool beam = c["tree"] == "beam";
    const int width = c["width"], depths = c["depths"], k = c["k"];
    const int32_t mask = c["mask"];
    const TreeCase t = make_case(c);
    auto lp = [&](int i, int d, int j) { return t.table[(size_t(d) * size_t(k) + size_t(i)) * size_t(k) + size_t(j)]; };
    std::vector<int32_t> drafts(static_cast<size_t>(width)), parents(static_cast<size_t>(width));
    std::vector<double> coverage;
    if (beam && c["cov"].get<bool>()) {
      coverage = c["coverage"].get<std::vector<double>>();
      for (int d = 0; d < depths; ++d) {
        const double mine = flyx::ops::slot_coverage(t.logp.data() + size_t(d) * size_t(k), k);
        cov_worst = std::max(cov_worst, std::abs(mine - coverage[size_t(d)]));
        floored += mine == std::log(1e-6);
        ++cov_slots;
      }
    }
    if (beam)
      flyx::ops::beam_tree(width, c["beam"], depths, k, t.cand.data(), t.live.data(),
                           coverage.empty() ? nullptr : coverage.data(), lp, mask, drafts.data(), parents.data());
    else
      flyx::ops::pool_tree(width, c["ranks"], depths, k, t.cand.data(), t.live.data(), lp, mask, drafts.data(),
                           parents.data());
    const bool ok = drafts == c["drafts"].get<std::vector<int32_t>>() &&
                    parents == c["parents"].get<std::vector<int32_t>>();
    bool first = true;
    for (int i = 0; i < width; ++i) first &= parents[size_t(i)] < i;
    if (!ok && flyx::test::failures < 20)
      std::printf("  %s case seed %d differs\n", beam ? "beam" : "pool", int(c["seed"]));
    const int b = beam ? 0 : 1;
    ++n[b];
    same[b] += ok;
    ordered += first;
    dead[b] += c["live_depths"].get<int>() < depths;
    masks[b] += std::find(drafts.begin(), drafts.end(), mask) != drafts.end();
    ties[b] += c["ties"].get<bool>();
    padded += beam && c["beam"].get<int>() > k;
  }
  char line[320];
  std::snprintf(line, sizeof line, "beam_tree: %d/%d trees are the Python reference's (%d with slots past the canvas, "
                "%d with MASK drafts, %d in ties, %d beams wider than the anchor's children)", same[0], n[0], dead[0],
                masks[0], ties[0], padded);
  check(n[0] >= 1000 && same[0] == n[0], line);
  std::snprintf(line, sizeof line, "pool_tree: %d/%d trees are the Python reference's (%d with slots past the canvas, "
                "%d with MASK drafts, %d in ties)", same[1], n[1], dead[1], masks[1], ties[1]);
  check(n[1] >= 1000 && same[1] == n[1], line);
  check(ordered == n[0] + n[1], "every draft's parent comes before it");
  std::snprintf(line, sizeof line, "slot_coverage: %d slots of the cov trees, %d of them at the floor of 1e-6, "
                "largest difference from the Python reference's term %.2e", cov_slots, floored, cov_worst);
  check(cov_slots > 0 && floored > 0 && cov_worst <= 1e-14, line);
}

flyx_tensor_view view(const std::vector<double>& v, int64_t rows, int64_t cols) {
  return flyx_tensor_view{"", FLYX_DT_F64, 2, {rows, cols, 0, 0}, v.data(), v.size() * sizeof(double)};
}

void tables(const nlohmann::json& cases) {
  double worst = 0;
  int n = 0;
  for (const auto& c : cases) {
    const int V = c["vocab"], D = c["hidden"], r = c["rank"], S = c["slots"], k = c["k"];
    const auto E = c["embed"].get<std::vector<double>>(), A = c["A"].get<std::vector<double>>();
    const auto H = c["H"].get<std::vector<double>>(), B = c["B"].get<std::vector<double>>();
    const auto h = c["h"].get<std::vector<double>>(), logp = c["logp"].get<std::vector<double>>();
    const auto want = c["table"].get<std::vector<double>>();
    const auto cand = c["cand"].get<std::vector<int32_t>>();
    const flyx_tensor_view va = view(A, r, D), vh = view(H, r, D), vb = view(B, r, D);
    std::vector<double> pred(size_t(V) * size_t(r)), succ(pred.size()), hid(size_t(S) * size_t(r));
    flyx::ops::linear(E.data(), V, D, &va, r, pred.data());  // A E[t] for every token
    flyx::ops::linear(E.data(), V, D, &vb, r, succ.data());
    flyx::ops::linear(h.data(), S, D, &vh, r, hid.data());
    std::vector<double> table(want.size());
    flyx::ops::selector_table(S, k, r, c["anchor"], cand.data(), logp.data(), pred.data(), succ.data(), hid.data(),
                              c["alpha"], c["temp"], table.data());
    for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(table[i] - want[i]));
    ++n;
  }
  char line[160];
  std::snprintf(line, sizeof line, "selector_table: %d tables, largest difference from the Python reference %.2e", n,
                worst);
  check(n > 0 && worst <= 1e-12, line);
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 2) return std::fprintf(stderr, "usage: dv_tree_test ASSETS\n"), 2;
  std::ifstream f(std::string(argv[1]) + "/dv_cases.json");
  if (!f) return std::fprintf(stderr, "no %s/dv_cases.json (tests/e2e/make_tiny.py writes it)\n", argv[1]), 1;
  const nlohmann::json cases = nlohmann::json::parse(f);
  trees(cases["trees"]);
  tables(cases["tables"]);
  return flyx::test::finish("dv_tree_test");
}
