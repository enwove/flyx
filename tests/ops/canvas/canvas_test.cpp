// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV's canvas (ops/canvas/ref.h) against the Python reference (python/reference/flyx_ref/dv.py), on the cases
// tests/e2e/make_tiny.py writes to ASSETS/dv_cases.json: canvas_input (MASK slots, new slots with z = 0),
// canvas_stats (the top 16 with ties to the lower id, the logits of half the cases on a grid of 1/4) and slide (every
// committed count, fill thresholds that keep all, some or no slot's guess).  Ids, tokens and which slots hold
// candidates must be equal; values within float64 rounding.
//     canvas_test ASSETS
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "ops/canvas/ref.h"

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

using Vec = std::vector<double>;

flyx_tensor_view view(const Vec& v, int64_t rows, int64_t cols) {
  return flyx_tensor_view{"", FLYX_DT_F64, 2, {rows, cols, 0, 0}, v.data(), v.size() * sizeof(double)};
}

double worst(const Vec& got, const Vec& want) {
  if (got.size() != want.size()) return INFINITY;
  double w = 0;
  for (size_t i = 0; i < got.size(); ++i)
    w = std::max(w, std::abs(got[i] - want[i]) / std::max(1.0, std::abs(want[i])));
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return std::fprintf(stderr, "usage: canvas_test ASSETS\n"), 2;
  std::ifstream f(std::string(argv[1]) + "/dv_cases.json");
  if (!f) return std::fprintf(stderr, "no %s/dv_cases.json (tests/e2e/make_tiny.py writes it)\n", argv[1]), 1;
  const nlohmann::json all = nlohmann::json::parse(f);
  double in_err = 0, stat_err = 0, slide_err = 0;
  int n = 0, ids_same = 0, slid_same = 0, ties = 0, masked = 0, kept = 0;
  for (const auto& c : all["canvases"]) {
    const int V = c["vocab"], D = c["hidden"], N = c["slots"], k = c["k"], width = c["width"], mask = c["mask"];
    const Vec E = c["embed"], mask_emb = c["mask_emb"], res_in = c["res_in"], res_out = c["res_out"], z = c["z"];
    const std::vector<int32_t> tokens = c["tokens"];
    const flyx_tensor_view ve = view(E, V, D), vm = view(mask_emb, 1, D), vi = view(res_in, 2 * D, D),
                           vo = view(res_out, D, D);
    Vec x(size_t(N) * size_t(D));
    flyx::ops::canvas_input(N, D, tokens.data(), mask, z.data(), &ve, &vm, &vi, &vo, x.data());
    in_err = std::max(in_err, worst(x, c["x"]));
    for (int32_t t : tokens) masked += t == mask;

    const Vec logits = c["logits"];
    std::vector<int32_t> ids(size_t(N) * size_t(k));
    Vec p(ids.size()), lse(static_cast<size_t>(N)), mass(static_cast<size_t>(N)), zn(size_t(N) * size_t(D));
    for (int w = 0; w < N; ++w)
      flyx::ops::canvas_stats(logits.data() + size_t(w) * size_t(V), V, k, &ve, D, ids.data() + size_t(w) * size_t(k),
                              p.data() + size_t(w) * size_t(k), &lse[size_t(w)], &mass[size_t(w)],
                              zn.data() + size_t(w) * size_t(D));
    ids_same += ids == c["ids"].get<std::vector<int32_t>>();
    stat_err = std::max({stat_err, worst(p, c["p"]), worst(lse, c["lse"]), worst(mass, c["mass"]), worst(zn, c["zn"])});
    for (int w = 0; w < N; ++w) {  // a tie at the edge of the top k or inside it
      const double* l = logits.data() + size_t(w) * size_t(V);
      const double edge = l[ids[size_t(w) * size_t(k) + size_t(k) - 1]];
      int at = 0;
      for (int v = 0; v < V; ++v) at += l[v] == edge;
      ties += at > 1;
    }

    // slide on the reference's own statistics, so it is checked alone
    const std::vector<int32_t> rids = c["ids"];
    const Vec rp = c["p"], rzn = c["zn"], hid = c["hid"];
    std::vector<int32_t> cv(static_cast<size_t>(N)), cand(size_t(N) * size_t(k));
    Vec sz(size_t(N) * size_t(D)), logp(cand.size()), shid(size_t(N) * size_t(width));
    std::vector<uint8_t> live(static_cast<size_t>(N));
    flyx::ops::slide(N, k, D, width, c["n"], c["fill"], mask, rids.data(), rp.data(), rzn.data(), hid.data(),
                     cv.data(), sz.data(), cand.data(), logp.data(), live.data(), shid.data());
    const auto& s = c["slid"];
    slid_same += cv == s["cv"].get<std::vector<int32_t>>() && cand == s["cand"].get<std::vector<int32_t>>() &&
                 live == s["live"].get<std::vector<uint8_t>>();
    slide_err = std::max({slide_err, worst(sz, s["z"]), worst(logp, s["logp"]), worst(shid, s["hid"])});
    for (int w = 0; w < N; ++w) kept += cv[size_t(w)] != mask;
    ++n;
  }
  char line[240];
  std::snprintf(line, sizeof line, "canvas_input: %d canvases (%d MASK slots), largest relative difference %.2e", n,
                masked, in_err);
  check(n > 0 && in_err <= 1e-12, line);
  std::snprintf(line, sizeof line, "canvas_stats: top k ids the reference's in %d/%d canvases (%d slots tie at or "
                "inside the k-th), largest relative difference %.2e", ids_same, n, ties, stat_err);
  check(ids_same == n && ties > 0 && stat_err <= 1e-12, line);
  std::snprintf(line, sizeof line, "slide: inputs, candidates and live slots the reference's in %d/%d canvases (%d "
                "guesses kept), largest relative difference %.2e", slid_same, n, kept, slide_err);
  check(slid_same == n && kept > 0 && slide_err <= 1e-12, line);
  std::printf("canvas: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
