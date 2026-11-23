// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: DV (dv.cu): the selector's token tables, a canvas pass's input rows and what it leaves, the slide, the
// round's selector table, the trees (bit for bit the float instantiations of ops/draft/ref.h on the same table), and
// a canvas pass's attention (attn_prep and attention with CanvasKeys).
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

namespace {

int* up_i(const std::vector<int>& v) {
  int* p = dev<int>(v.size());
  CK(cudaMemcpy(p, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
  return p;
}
void put_i(int* p, int v) { CK(cudaMemcpy(p, &v, sizeof(int), cudaMemcpyHostToDevice)); }
// Inputs a kernel reads as float32, as double (the reference reads the same values).
Vec floats(Vec v) {
  for (auto& x : v) x = double(float(x));
  return v;
}

DevParams* params_at(int position, int start, int anchor = 0, int oracle_len = 0) {
  DevParams p{};
  p.position = position;
  p.start = start;
  p.anchor = anchor;
  p.oracle_len = oracle_len;
  DevParams* d = dev<DevParams>(1);
  CK(cudaMemcpy(d, &p, sizeof(p), cudaMemcpyHostToDevice));
  return d;
}

// A canvas of N slots on the device.
DvCanvas canvas_on_device(int N, int H, int rank) {
  DvCanvas c{};
  c.cv = dev<int>(size_t(N));
  c.z = dev<float>(size_t(N) * H);
  c.cand = dev<int>(size_t(N) * kDvTop);
  c.logp = dev<float>(size_t(N) * kDvTop);
  c.cov = dev<float>(size_t(N));
  c.live = dev<int>(size_t(N));
  c.hid = dev<float>(size_t(N) * rank);
  c.next_start = dev<int>(1);
  return c;
}
void free_canvas(const DvCanvas& c) {
  for (void* p : std::initializer_list<void*>{c.cv, c.z, c.cand, c.logp, c.cov, c.live, c.hid, c.next_start})
    CK(cudaFree(p));
}

void report(bool ok, const std::string& what) {
  std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
  g_fail += !ok;
}

}  // namespace

// TA = A E^T, TB = B E^T against float64 (the float32 chain over hidden: its rounding), and bit for bit the plain
// fma loop over hidden on sampled entries (the tiles do not change a value's bits).
void test_dv_selector_tables(int vocab, int hidden, int rank) {
  seed("dv selector tables", vocab, hidden, rank);
  const Vec A = floats(randv(size_t(rank) * hidden, 0.05)), B = floats(randv(size_t(rank) * hidden, 0.05));
  Vec E = randv(size_t(vocab) * hidden);
  round_bf16(E);
  float *dA = up(A), *dB = up(B), *dTA = dev<float>(size_t(vocab) * rank), *dTB = dev<float>(size_t(vocab) * rank);
  bf16* dE = up_bf16(E);
  dv_selector_tables(dA, dB, rank, dE, vocab, hidden, dTA, dTB, 0);
  const std::string tag = "vocab " + std::to_string(vocab) + " hidden " + std::to_string(hidden) + " rank " +
                          std::to_string(rank);
  for (int which = 0; which < 2; ++which) {
    const Vec& M = which ? B : A;
    const Vec got = down(which ? dTB : dTA, size_t(vocab) * rank);
    Vec want(got.size());
    for (int t = 0; t < vocab; ++t)
      for (int j = 0; j < rank; ++j) {
        double acc = 0;
        for (int k = 0; k < hidden; ++k) acc += M[size_t(j) * hidden + k] * E[size_t(t) * hidden + k];
        want[size_t(t) * rank + j] = acc;
      }
    // one float32 chain of `hidden` terms a value
    expect(std::string(which ? "TB" : "TA") + " = M E^T vs float64 " + tag, got, want, 1e-5);
    std::uniform_int_distribution<int> pick_t(0, vocab - 1), pick_j(0, rank - 1);
    Vec chain, mine;
    for (int s = 0; s < 2000; ++s) {
      const int t = s < 4 ? (s & 1) * (vocab - 1) : pick_t(rng), j = s < 4 ? (s >> 1) * (rank - 1) : pick_j(rng);
      float acc = 0.f;
      for (int k = 0; k < hidden; ++k)
        acc = std::fmaf(float(E[size_t(t) * hidden + k]), float(M[size_t(j) * hidden + k]), acc);
      chain.push_back(acc);
      mine.push_back(got[size_t(t) * rank + j]);
    }
    expect(std::string(which ? "TB" : "TA") + ": the fma loop over hidden, same bits " + tag, mine, chain, 0);
  }
  for (void* p : std::initializer_list<void*>{dA, dB, dTA, dTB, dE}) CK(cudaFree(p));
}

// A canvas pass's input rows against ops::canvas_input (the products' inputs as three bf16 parts: float32 sums and
// silu's rounding only); a slot with z = 0 is its embedding exactly; a round that is not the canvas's starts from
// mask_emb; the rows past the first 16 alone give the bits they get with them (16-row GEMMs, class R).
void test_canvas_input(int slots, int hidden) {
  seed("canvas input", slots, hidden);
  const int V = 500, mask = V - 1, start = 300;
  Vec E = randv(size_t(V) * hidden), memb = randv(size_t(hidden)), rin = randv(size_t(2 * hidden) * hidden, 0.05),
      rout = randv(size_t(hidden) * hidden, 0.05);
  Vec z = floats(randv(size_t(slots) * hidden, 0.5));
  std::vector<int> cv(static_cast<size_t>(slots));
  std::uniform_int_distribution<int> tok(0, V - 2);
  for (int w = 0; w < slots; ++w) {
    cv[size_t(w)] = w % 3 == 1 ? mask : tok(rng);
    if (w % 5 == 4) std::fill(z.begin() + size_t(w) * hidden, z.begin() + size_t(w + 1) * hidden, 0.0);  // a new slot
  }
  std::vector<uint16_t> sE, sM, sI, sO;
  const flyx_tensor_view vE = view_of(E, &sE, V, hidden), vM = view_of(memb, &sM, 1, hidden),
                         vI = view_of(rin, &sI, 2 * hidden, hidden), vO = view_of(rout, &sO, hidden, hidden);
  round_bf16(E);
  round_bf16(memb);
  DvCanvas c = canvas_on_device(slots, hidden, 1);
  CK(cudaMemcpy(c.cv, cv.data(), cv.size() * 4, cudaMemcpyHostToDevice));
  {
    const std::vector<float> zf(z.begin(), z.end());
    CK(cudaMemcpy(c.z, zf.data(), zf.size() * 4, cudaMemcpyHostToDevice));
  }
  bf16 *dE = up_bf16(E), *dM = up_bf16(memb), *dI = up_bf16(rin), *dO = up_bf16(rout);
  float *ug = dev<float>(size_t(slots) * 2 * hidden), *x = dev<float>(size_t(slots) * hidden);
  DevParams* P = params_at(start + 4, start);
  put_i(c.next_start, start);
  canvas_input(P, c, slots, mask, dE, dM, dI, dO, hidden, ug, x, TL);
  const Vec got = down(x, size_t(slots) * hidden);
  Vec want(got.size());
  std::vector<int32_t> cv32(cv.begin(), cv.end());
  ops::canvas_input(slots, hidden, cv32.data(), mask, z.data(), &vE, &vM, &vI, &vO, want.data());
  const std::string tag = "slots " + std::to_string(slots) + " hidden " + std::to_string(hidden);
  expect("canvas input vs float64 " + tag, got, want, 2e-5);
  Vec emb_rows, got_rows;  // the new slots: their embedding, bit for bit
  for (int w = 4; w < slots; w += 5)
    for (int i = 0; i < hidden; ++i) {
      got_rows.push_back(got[size_t(w) * hidden + i]);
      emb_rows.push_back(cv[size_t(w)] == mask ? memb[size_t(i)] : E[size_t(cv[size_t(w)]) * hidden + i]);
    }
  expect("canvas input, slots with z = 0: their embedding exactly " + tag, got_rows, emb_rows, 0);
  if (slots > 16) {
    DvCanvas tail = c;
    tail.cv += 16;
    tail.z += size_t(16) * hidden;
    float* x2 = dev<float>(size_t(slots - 16) * hidden);
    canvas_input(P, tail, slots - 16, mask, dE, dM, dI, dO, hidden, ug, x2, TL);
    expect("canvas input, the slots past 16 alone: same bits " + tag, down(x2, size_t(slots - 16) * hidden),
           Vec(got.begin() + size_t(16) * hidden, got.end()), 0);
    CK(cudaFree(x2));
  }
  put_i(c.next_start, -1);  // a round after a reset: every slot MASK, no residual
  canvas_input(P, c, slots, mask, dE, dM, dI, dO, hidden, ug, x, TL);
  Vec fresh;
  for (int w = 0; w < slots; ++w) fresh.insert(fresh.end(), memb.begin(), memb.end());
  expect("canvas input, a canvas of another round: mask_emb " + tag, down(x, size_t(slots) * hidden), fresh, 0);
  for (void* p : std::initializer_list<void*>{dE, dM, dI, dO, ug, x, P}) CK(cudaFree(p));
  free_canvas(c);
}

// What a canvas pass leaves against ops::canvas_stats and H h / sqrt(rank) in float64: the top 16 exactly (ties to
// the lower id: logits on a grid of 1/4 in half the rows, and a run of equal largest values); lse within 1e-6, p, mass
// and zn within 1e-5; the same bits again.
void test_canvas_stats(int slots, int vocab, int hidden, int rank) {
  seed("canvas stats", slots, vocab, hidden, rank);
  Vec lg = randv(size_t(slots) * vocab, 3.0);
  for (int r = 0; r < slots; ++r) {
    double* row = &lg[size_t(r) * vocab];
    if (r % 2) for (int v = 0; v < vocab; ++v) row[v] = std::round(row[v] * 4) / 4;  // ties everywhere
    if (r % 4 == 2) for (int v = 7; v < vocab; v += vocab / 20) row[v] = 15.0;      // 20 equal largest
  }
  lg = floats(lg);
  Vec E = randv(size_t(vocab) * hidden), h = floats(randv(size_t(slots) * hidden)),
      Hm = floats(randv(size_t(rank) * hidden, 0.05));
  std::vector<uint16_t> sE;
  const flyx_tensor_view vE = view_of(E, &sE, vocab, hidden);
  round_bf16(E);
  DvStats s{dev<int>(size_t(slots) * kDvTop), dev<float>(size_t(slots) * kDvTop), dev<float>(size_t(slots)),
            dev<float>(size_t(slots)), dev<float>(size_t(slots) * hidden), dev<float>(size_t(slots) * rank),
            dev<int>(size_t(slots) * kDvParts * kDvTop), dev<float>(size_t(slots) * kDvParts * kDvTop)};
  float *dl = up(lg), *dh = up(h), *dH = up(Hm);
  bf16* dE = up_bf16(E);
  canvas_stats(dl, vocab, vocab, dE, dh, hidden, hidden, dH, rank, slots, s, 0);
  std::vector<int> want_ids(size_t(slots) * kDvTop);
  Vec want_p(size_t(slots) * kDvTop), want_lse(static_cast<size_t>(slots)), want_mass(static_cast<size_t>(slots)),
      want_zn(size_t(slots) * hidden), want_hid(size_t(slots) * rank);
  for (int r = 0; r < slots; ++r) {
    ops::canvas_stats(&lg[size_t(r) * vocab], vocab, kDvTop, &vE, hidden, &want_ids[size_t(r) * kDvTop],
                      &want_p[size_t(r) * kDvTop], &want_lse[size_t(r)], &want_mass[size_t(r)],
                      &want_zn[size_t(r) * hidden]);
    for (int j = 0; j < rank; ++j) {
      double acc = 0;
      for (int i = 0; i < hidden; ++i) acc += Hm[size_t(j) * hidden + i] * h[size_t(r) * hidden + i];
      want_hid[size_t(r) * rank + j] = acc / std::sqrt(double(rank));
    }
  }
  const std::string tag = "slots " + std::to_string(slots) + " vocab " + std::to_string(vocab) + " hidden " +
                          std::to_string(hidden) + " rank " + std::to_string(rank);
  const std::vector<int> ids = down_i(s.ids, want_ids.size());
  report(ids == want_ids, "canvas stats: top 16 ids, ties to the lower id " + tag);
  const Vec p = down(s.p, want_p.size()), lse = down(s.lse, size_t(slots)), mass = down(s.mass, size_t(slots)),
            zn = down(s.zn, want_zn.size()), hid = down(s.hid, want_hid.size());
  expect("canvas stats: lse " + tag, lse, want_lse, 1e-6);
  expect("canvas stats: p " + tag, p, want_p, 1e-5);
  expect("canvas stats: mass " + tag, mass, want_mass, 1e-5);
  expect("canvas stats: zn " + tag, zn, want_zn, 1e-5);
  expect("canvas stats: hid = H h / sqrt(rank) " + tag, hid, want_hid, 1e-5);
  canvas_stats(dl, vocab, vocab, dE, dh, hidden, hidden, dH, rank, slots, s, 0);
  bool same = down_i(s.ids, want_ids.size()) == ids && down(s.p, p.size()) == p && down(s.lse, lse.size()) == lse &&
              down(s.mass, mass.size()) == mass && down(s.zn, zn.size()) == zn && down(s.hid, hid.size()) == hid;
  report(same, "canvas stats: the same bits again " + tag);
  for (void* q : std::initializer_list<void*>{dl, dh, dH, dE, s.ids, s.p, s.lse, s.mass, s.zn, s.hid, s.part_i,
                                              s.part_v})
    CK(cudaFree(q));
}

// The slide against ops::slide, every committed count from 1 to past the canvas: ids, inputs, candidates and which
// slots hold them exactly; z and hid the pass's bits; logp and cov (ops::slot_coverage of logp) within float32's log
// and exp; next_start the round's position.  Probabilities under 1e-9 (the floor), slots whose 16 sum under 1e-6 (cov's
// floor) or just above it with 15 under 1e-9 (cov counts them at 1e-9), and top probabilities at the fill threshold
// exactly (kept) and just below it.
void test_dv_slide(int slots, int hidden, int rank) {
  seed("dv slide", slots, hidden, rank);
  const int mask = 77, start = 500;
  const float fill = 0.9f;
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::vector<int> ids(size_t(slots) * kDvTop);
  Vec p(size_t(slots) * kDvTop);
  for (int j = 0; j < slots; ++j)
    for (int k = 0; k < kDvTop; ++k) {
      ids[size_t(j) * kDvTop + k] = 1000 + 37 * j + k;
      const double x = u01(rng);
      p[size_t(j) * kDvTop + k] = j % 7 == 3 ? 1e-12 * x                                      // the 16 under 1e-6
                                  : j % 7 == 5 ? (k == 0 ? 2e-6 : 1e-11 * x)                 // just above it
                                  : k == 0 ? (j % 4 == 0 ? double(fill)                      // at the threshold
                                              : j % 4 == 1 ? double(std::nextafter(fill, 0.f)) : x)
                                  : k > 12 ? 1e-10 * x                                       // under the floor
                                           : x * 0.05;
    }
  p = floats(p);
  const Vec zn = floats(randv(size_t(slots) * hidden)), hid = floats(randv(size_t(slots) * rank));
  DvStats s{up_i(ids), up(p), nullptr, nullptr, up(zn), up(hid)};
  DvCanvas c = canvas_on_device(slots, hidden, rank);
  bool cv_ok = true, cand_ok = true, live_ok = true, copies_ok = true, next_ok = true;
  double logp_err = 0, cov_err = 0;
  for (int n : {1, 2, 3, slots / 2, slots - 1, slots, slots + 3}) {
    DevParams* P = params_at(start + n, start);
    dv_slide(P, s, slots, hidden, rank, fill, mask, c, 0);
    std::vector<int32_t> wcv(static_cast<size_t>(slots)), wcand(size_t(slots) * kDvTop);
    std::vector<uint8_t> wlive(static_cast<size_t>(slots));
    Vec wz(size_t(slots) * hidden), wlogp(size_t(slots) * kDvTop), whid(size_t(slots) * rank);
    std::vector<int32_t> ids32(ids.begin(), ids.end());
    ops::slide(slots, kDvTop, hidden, rank, n, double(fill), mask, ids32.data(), p.data(), zn.data(), hid.data(),
               wcv.data(), wz.data(), wcand.data(), wlogp.data(), wlive.data(), whid.data());
    const std::vector<int> gcv = down_i(c.cv, size_t(slots)), gcand = down_i(c.cand, wcand.size()),
                           glive = down_i(c.live, size_t(slots)), gnext = down_i(c.next_start, 1);
    const Vec gz = down(c.z, wz.size()), glogp = down(c.logp, wlogp.size()), gcov = down(c.cov, size_t(slots)),
              ghid = down(c.hid, whid.size());
    cv_ok &= std::equal(gcv.begin(), gcv.end(), wcv.begin());
    cand_ok &= std::equal(gcand.begin(), gcand.end(), wcand.begin());
    for (int w = 0; w < slots; ++w) {
      live_ok &= glive[size_t(w)] == int(wlive[size_t(w)]);
      const double want_cov = ops::slot_coverage(&wlogp[size_t(w) * kDvTop], kDvTop);
      cov_err = std::max(cov_err, std::abs(gcov[size_t(w)] - want_cov) / std::max(1.0, std::abs(want_cov)));
    }
    copies_ok &= gz == wz && ghid == whid;
    for (size_t i = 0; i < wlogp.size(); ++i)
      logp_err = std::max(logp_err, std::abs(glogp[i] - wlogp[i]) / std::max(1.0, std::abs(wlogp[i])));
    next_ok &= gnext[0] == start + n;
    CK(cudaFree(P));
  }
  // A no-op round (the reply had stopped: P->noop, start and position where the last round left them, and a pass
  // that followed nothing): the canvas as the last slide left it, every field and next_start.
  bool noop_ok = true;
  {
    auto bytes_of = [&]() {
      std::vector<int> v = down_i(c.cv, size_t(slots));
      for (const auto& part : {down_i(c.cand, size_t(slots) * kDvTop), down_i(c.live, size_t(slots)),
                               down_i(c.next_start, 1)})
        v.insert(v.end(), part.begin(), part.end());
      Vec f = down(c.z, size_t(slots) * hidden);
      for (const auto& part : {down(c.logp, size_t(slots) * kDvTop), down(c.cov, size_t(slots)),
                               down(c.hid, size_t(slots) * rank)})
        f.insert(f.end(), part.begin(), part.end());
      return std::make_pair(v, f);
    };
    const auto before = bytes_of();
    DevParams p{};
    p.position = start + 3;
    p.start = start;
    p.noop = 1;
    DevParams* P = dev<DevParams>(1);
    CK(cudaMemcpy(P, &p, sizeof(p), cudaMemcpyHostToDevice));
    dv_slide(P, s, slots, hidden, rank, fill, mask, c, 0);
    noop_ok = bytes_of() == before;
    CK(cudaFree(P));
  }
  const std::string tag = " slots " + std::to_string(slots);
  report(cv_ok, "slide: next inputs (guesses from p >= fill, at the threshold kept)" + tag);
  report(cand_ok && live_ok, "slide: candidates and live slots" + tag);
  report(copies_ok, "slide: z and hid the pass's rows (0 past it)" + tag);
  report(next_ok, "slide: next_start = the round's position" + tag);
  report(noop_ok, "slide: a no-op round leaves the canvas and next_start as they were" + tag);
  char line[160];
  std::snprintf(line, sizeof line, "slide: logp rel err %.2e, cov rel err %.2e (tol 1e-6)%s", logp_err, cov_err,
                tag.c_str());
  report(logp_err <= 1e-6 && cov_err <= 1e-6, line);
  for (void* q : std::initializer_list<void*>{s.ids, s.p, s.zn, s.hid}) CK(cudaFree(q));
  free_canvas(c);
}

// The round's selector table against ops::selector_table (float64, the same TA, TB, hid, logp; rank a power of 4 so
// the reference's sqrt(rank) is exact) at two temperatures; slot 0's 16 rows the same bits; nothing written in a round
// not at the canvas's position.
void test_dv_selector_table(int depths, int rank, int vocab) {
  seed("dv selector table", depths, rank, vocab);
  const int N = depths + 1, start = 900, anchor = 4321 % vocab;
  const Vec TA = floats(randv(size_t(vocab) * rank, 0.6)), TB = floats(randv(size_t(vocab) * rank, 0.6)),
            hid = floats(randv(size_t(N) * rank, 0.6));
  std::uniform_int_distribution<int> tok(0, vocab - 1);
  std::uniform_real_distribution<double> lp(-20.7, 0.0);
  std::vector<int> cand(size_t(N) * kDvTop);
  Vec logp(cand.size());
  for (size_t i = 0; i < cand.size(); ++i) cand[i] = tok(rng), logp[i] = lp(rng);
  logp = floats(logp);
  DvCanvas c = canvas_on_device(N, 1, rank);
  CK(cudaMemcpy(c.cand, cand.data(), cand.size() * 4, cudaMemcpyHostToDevice));
  {
    const std::vector<float> lf(logp.begin(), logp.end()), hf(hid.begin(), hid.end());
    CK(cudaMemcpy(c.logp, lf.data(), lf.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.hid, hf.data(), hf.size() * 4, cudaMemcpyHostToDevice));
  }
  put_i(c.next_start, start);
  float *dTA = up(TA), *dTB = up(TB), *table = dev<float>(size_t(depths) * kDvTop * kDvTop);
  DevParams* P = params_at(start, start - 3, anchor);
  const std::string tag = "depths " + std::to_string(depths) + " rank " + std::to_string(rank);
  Vec hid_ref = hid;
  for (auto& x : hid_ref) x *= std::sqrt(double(rank));
  std::vector<int32_t> cand32(cand.begin(), cand.end());
  for (const float temp : {1.0f, 0.8f}) {
    const float alpha = 0.7f;
    dv_selector_table(P, c, dTA, dTB, rank, alpha, temp, depths, table, 0);
    const Vec got = down(table, size_t(depths) * kDvTop * kDvTop);
    Vec want(got.size());
    ops::selector_table(depths, kDvTop, rank, anchor, cand32.data(), logp.data(), TA.data(), TB.data(), hid_ref.data(),
                        double(alpha), double(temp), want.data());
    char t[32];
    std::snprintf(t, sizeof t, " temp %.1f", temp);
    expect("selector table vs float64 " + tag + t, got, want, 2e-6);
    bool rows_same = true;
    for (int i = 1; i < kDvTop; ++i)
      for (int b = 0; b < kDvTop; ++b) rows_same &= got[size_t(i) * kDvTop + b] == got[size_t(b)];
    report(rows_same, "selector table: slot 0's rows (after the anchor) the same bits " + tag + t);
  }
  CK(cudaMemset(table, 0xff, size_t(depths) * kDvTop * kDvTop * 4));  // NaNs
  put_i(c.next_start, start - 1);
  dv_selector_table(P, c, dTA, dTB, rank, 0.7f, 1.0f, depths, table, 0);
  const Vec untouched = down(table, size_t(depths) * kDvTop * kDvTop);
  report(std::all_of(untouched.begin(), untouched.end(), [](double v) { return std::isnan(v); }),
         "selector table: nothing written for a canvas of another round " + tag);
  for (void* q : std::initializer_list<void*>{dTA, dTB, table, P}) CK(cudaFree(q));
  free_canvas(c);
}

namespace {

// A tree case: its table [depths][16][16], each slot's cov term, which slots hold candidates (slot d's candidate c is
// token 1000 + 16 d + c).
struct TreeCase {
  bool beam = true, cov = false, ties = false, above = false;
  int width = 0, wide = 0 /* beam or ranks */, depths = 0, live_depths = 0;
  std::vector<float> table, bonus;
  std::vector<uint8_t> live;
  std::vector<int32_t> cand;
};

// A case's table, candidates, live slots and cov terms for its shape.  Rows of the table: a log-softmax of random
// scores in float (as the table kernel leaves them), or for a tie case values on a grid of 1/4 in [-1.75, 0] (many
// equal candidates, equal sums).  A slot's cov term: the log of its top 16's mass, in (0, 1] as a canvas's is, some a
// float sum's few ulps above 1, some at the 1e-6 floor.  `above` (a tie case): edges above the depth's step, out of
// DV's range (the grid raised by 2^-10, cov terms up to 1e-3), so that a child is worth more than its parent and may
// be drafted without it; the trees must still be the reference's.
void fill_tree_case(TreeCase& t, int n) {
  std::uniform_int_distribution<int> any(0, 1 << 20);
  std::normal_distribution<float> gauss(0.f, 2.f);
  std::uniform_real_distribution<float> u01(0.f, 1.f);
  const float lift = t.above ? 1.f / 1024 : 0.f;
  for (int d = 0; d < t.depths; ++d) {
    t.live.push_back(d < t.live_depths);
    if (n % 11 == 5 && d < t.live_depths) t.live.back() = any(rng) % 4 != 0;  // dead slots anywhere
    for (int i = 0; i < kDvTop; ++i) {
      float s[kDvTop], m = -INFINITY, z = 0.f;
      for (int c = 0; c < kDvTop; ++c) m = std::max(m, s[c] = gauss(rng));
      for (int c = 0; c < kDvTop; ++c) z += std::exp(s[c] - m);
      for (int c = 0; c < kDvTop; ++c)
        t.table.push_back(t.ties ? lift - std::floor(u01(rng) * 8.f) / 4.f : (s[c] - m) - std::log(z));
    }
    for (int c = 0; c < kDvTop; ++c) t.cand.push_back(1000 + d * kDvTop + c);
    float mass = d % 5 == 3 ? 1e-7f * u01(rng) : 1e-3f + 0.999f * u01(rng);
    if (d % 7 == 2) mass = 1.f + 1e-6f * u01(rng);
    if (t.above) mass = 1.f + 1e-3f * u01(rng);
    t.bonus.push_back(std::log(std::max(mass, 1e-6f)));
  }
}

TreeCase random_tree_case(bool beam, int n) {
  TreeCase t;
  t.beam = beam;
  std::uniform_int_distribution<int> any(0, 1 << 20);
  if (beam) {
    static const int beams[] = {1, 2, 3, 4, 6, 8, 8, 8, 12, 16, 20, 24, 32};
    t.wide = beams[any(rng) % 13];
    t.depths = 1 + any(rng) % (n % 3 == 0 ? 63 : 31);
    t.width = 1 + any(rng) % std::min(kDvMaxWidth, t.depths * t.wide);
    t.cov = any(rng) % 3 != 0;
  } else {
    t.wide = n % 10 == 0 ? 8 + any(rng) % 8 : 1 + any(rng) % 7;  // ranks
    t.depths = 1 + any(rng) % (t.wide > 7 ? 31 : n % 3 == 0 ? 40 : 31);
    const int pool = t.depths * (2 + (t.depths + 1) * t.wide) / 2;
    t.width = 1 + any(rng) % std::min(kDvMaxWidth, pool);
  }
  if (n % 7 == 1) t.width = std::min(t.width, FLYX_MAX_ROWS - 1);
  t.above = n % 13 == 4;
  t.ties = t.above || n % 5 == 2;
  t.live_depths = n % 3 == 1 ? any(rng) % (t.depths + 1) : t.depths;
  fill_tree_case(t, n);
  return t;
}

}  // namespace

// Both trees against ops::beam_tree<float> / ops::pool_tree<float> on the same table, bit for bit: 1000 random cases
// each and then some (widths up to 63, depths up to 63 for the beam, beams wider than a depth's 16 children, ranks up
// to 15; slots without candidates at the end or anywhere, MASK drafts, tables in ties, edges above the depth's step),
// every parent before its child, the round's TreeRows for trees of up to 63 drafts (a canvas's draft at its slot's
// depth); DYN 31:8:cov and ST 31/7/1 on 32 slots as the round runs them; for each rank count the deepest pool the
// launcher takes (its launch goes through: the block's whole shared memory counted); DV's first tree and the test's
// chain.
void test_dv_trees() {
  std::printf("dv trees\n");
  seed("dv trees");
  const int mask = 7, start = 200, position = 203;
  DvCanvas c = canvas_on_device(kDvMaxDepths + 1, 1, 1);
  float* dtable = dev<float>(size_t(kDvMaxDepths) * kDvTop * kDvTop);
  int *ddrafts = dev<int>(kDvMaxWidth), *dparents = dev<int>(kDvMaxWidth);
  TreeRows* dtree = dev<TreeRows>(1);
  DevParams* P = params_at(position, start, 55);
  put_i(c.next_start, position);
  int n[2] = {}, same[2] = {}, dead[2] = {}, masked[2] = {}, ties[2] = {}, above[2] = {}, orphaned[2] = {};
  int rows[2] = {}, off_double[2] = {};
  int padded = 0, ordered = 0, rows_ok = 0;
  auto run = [&](const TreeCase& t) {
    std::vector<int> live(t.live.begin(), t.live.end());
    CK(cudaMemcpy(dtable, t.table.data(), t.table.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.live, live.data(), live.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.cov, t.bonus.data(), t.bonus.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c.cand, t.cand.data(), t.cand.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(ddrafts, 0x7f, kDvMaxWidth * sizeof(int)));  // no tree's: a launch that writes nothing differs
    CK(cudaMemset(dparents, 0x7f, kDvMaxWidth * sizeof(int)));
    const bool with_rows = t.width < FLYX_MAX_WIDE_ROWS;  // every tree a round verifies (its wide rows)
    if (t.beam)
      dv_tree_beam(P, c, dtable, t.width, t.wide, t.cov, t.depths, mask, nullptr, ddrafts, dparents,
                   with_rows ? dtree : nullptr, 0);
    else
      dv_tree_pool(P, c, dtable, t.width, t.wide, t.depths, mask, nullptr, ddrafts, dparents,
                   with_rows ? dtree : nullptr, 0);
    std::vector<int32_t> wd(size_t(t.width)), wp(size_t(t.width)), dd(wd.size()), dp(wd.size());
    auto lp = [&](int i, int d, int k) { return t.table[(size_t(d) * kDvTop + size_t(i)) * kDvTop + size_t(k)]; };
    auto lp64 = [&](int i, int d, int k) { return double(lp(i, d, k)); };
    std::vector<double> bonus64(t.bonus.begin(), t.bonus.end());
    if (t.beam) {
      ops::beam_tree<float>(t.width, t.wide, t.depths, kDvTop, t.cand.data(), t.live.data(),
                            t.cov ? t.bonus.data() : nullptr, lp, mask, wd.data(), wp.data());
      ops::beam_tree<double>(t.width, t.wide, t.depths, kDvTop, t.cand.data(), t.live.data(),
                             t.cov ? bonus64.data() : nullptr, lp64, mask, dd.data(), dp.data());
    } else {
      ops::pool_tree<float>(t.width, t.wide, t.depths, kDvTop, t.cand.data(), t.live.data(), lp, mask, wd.data(),
                            wp.data());
      ops::pool_tree<double>(t.width, t.wide, t.depths, kDvTop, t.cand.data(), t.live.data(), lp64, mask, dd.data(),
                             dp.data());
    }
    const std::vector<int> gd = down_i(ddrafts, size_t(t.width)), gp = down_i(dparents, size_t(t.width));
    const bool ok = std::equal(gd.begin(), gd.end(), wd.begin()) && std::equal(gp.begin(), gp.end(), wp.begin());
    const int b = t.beam ? 0 : 1;
    if (!ok && n[b] - same[b] < 5)
      std::printf("       %s tree of %d (beam/ranks %d, depths %d, live %d%s%s) differs\n", t.beam ? "beam" : "pool",
                  t.width, t.wide, t.depths, t.live_depths, t.cov ? ", cov" : "", t.above ? ", above the step" : "");
    ++n[b];
    same[b] += ok;
    off_double[b] += wd != dd || wp != dp;
    bool first = true;
    for (int i = 0; i < t.width; ++i) first &= gp[size_t(i)] < i;
    ordered += first;
    dead[b] += std::find(t.live.begin(), t.live.end(), 0) != t.live.end();
    masked[b] += std::find(gd.begin(), gd.end(), mask) != gd.end();
    ties[b] += t.ties;
    above[b] += t.above;
    bool orphan = false;  // a draft of a slot past the first under the anchor: its parent not drafted
    for (int i = 0; i < t.width; ++i) orphan |= wp[size_t(i)] < 0 && wd[size_t(i)] >= 1000 + kDvTop;
    orphaned[b] += orphan;
    padded += t.beam && t.wide > kDvTop;
    if (with_rows) {
      TreeRows tr{};
      CK(cudaMemcpy(&tr, dtree, sizeof(tr), cudaMemcpyDeviceToHost));
      bool rok = tr.n == t.width + 1 && tr.depth[0] == 0 && tr.n_taken == 0;
      for (int i = 0; i < t.width && rok; ++i) {
        const int r = i + 1, p = gp[size_t(i)] + 1;
        rok = tr.depth[r] == tr.depth[p] + 1 && tr.path[r][tr.depth[r]] == r;
        for (int d = 0; rok && d <= tr.depth[p]; ++d) rok = tr.path[r][d] == tr.path[p][d];
        if (rok && !t.above && gd[size_t(i)] != mask) rok = tr.depth[r] == (gd[size_t(i)] - 1000) / kDvTop + 1;
      }
      ++rows[b];
      rows_ok += rok;
    }
    return ok;
  };
  // The trees a round of MiniCPM5-1B's DV runs: 32 slots (depths 31), every slot live, then 3 slid past the canvas.
  for (int k = 0; k < 4; ++k) {
    TreeCase t;
    t.beam = t.cov = k < 2;
    t.depths = t.width = 31;
    t.wide = t.beam ? 8 : 7;
    t.live_depths = k % 2 ? 28 : 31;
    fill_tree_case(t, 0);
    run(t);
  }
  {  // nothing live: every draft MASK
    TreeCase t = random_tree_case(true, 0);
    t.live.assign(t.live.size(), 0);
    t.live_depths = 0;
    run(t);
    t = random_tree_case(false, 0);
    t.live.assign(t.live.size(), 0);
    t.live_depths = 0;
    run(t);
  }
  for (int i = 0; i < 2200; ++i) run(random_tree_case(i % 2 == 0, i / 2));
  char line[320];
  std::snprintf(line, sizeof line, "beam tree: %d/%d trees are ops::beam_tree<float>'s (%d with dead slots, %d with "
                "MASK drafts, %d in ties, %d beams wider than 16, %d above the step, %d drafting a node without its "
                "parent; %d differ from the double instantiation's)", same[0], n[0], dead[0], masked[0], ties[0],
                padded, above[0], orphaned[0], off_double[0]);
  report(n[0] >= 1000 && same[0] == n[0] && orphaned[0] > 0, line);
  std::snprintf(line, sizeof line, "pool tree: %d/%d trees are ops::pool_tree<float>'s (%d with dead slots, %d with "
                "MASK drafts, %d in ties, %d above the step, %d drafting a node without its parent; %d differ from "
                "the double instantiation's)", same[1], n[1], dead[1], masked[1], ties[1], above[1], orphaned[1],
                off_double[1]);
  report(n[1] >= 1000 && same[1] == n[1] && orphaned[1] > 0, line);
  report(ordered == n[0] + n[1], "every draft's parent before it");
  std::snprintf(line, sizeof line, "TreeRows of %d beam and %d pool trees of up to 63 drafts follow their parents (a "
                "canvas's draft at its slot's depth)", rows[0], rows[1]);
  report(rows_ok == rows[0] + rows[1] && rows[0] > 0 && rows[1] > 0, line);
  // The pools at the edge of a block's shared memory: for each rank count the deepest the launcher takes (a probe
  // of one draft finds it), as many drafts as it holds.
  int edges = 0, edges_ok = 0;
  CK(cudaGetLastError());
  for (int ranks = 1; ranks < kDvTop; ++ranks) {
    TreeCase t;
    t.beam = false;
    t.wide = ranks;
    for (t.depths = kDvMaxDepths; t.depths > 1; --t.depths) {
      try {
        dv_tree_pool(P, c, dtable, 1, ranks, t.depths, mask, nullptr, ddrafts, nullptr, nullptr, 0);
        break;
      } catch (const std::runtime_error&) {
      }
    }
    t.live_depths = t.depths;
    t.width = std::min(kDvMaxWidth, t.depths * (2 + (t.depths + 1) * ranks) / 2);
    fill_tree_case(t, ranks);
    const bool ok = run(t);
    const cudaError_t launch = cudaGetLastError();
    if (!ok || launch != cudaSuccess)
      std::printf("       pool %d/%d of %d drafts: %s\n", t.depths, ranks, t.width, cudaGetErrorString(launch));
    ++edges;
    edges_ok += ok && launch == cudaSuccess;
  }
  std::snprintf(line, sizeof line, "pool trees at the edge of shared memory: %d/%d rank counts' deepest pool the "
                "launcher takes launches and is ops::pool_tree<float>'s", edges_ok, edges);
  report(edges_ok == edges, line);
  // DV's first tree (a canvas of another round) and the test's chain, both kinds.
  std::vector<int> oracle(size_t(position) + 40);
  for (size_t i = 0; i < oracle.size(); ++i) oracle[i] = int(5000 + i);
  int* doracle = up_i(oracle);
  for (int kind = 0; kind < 2; ++kind)
    for (int mode = 0; mode < 2; ++mode) {
      const int width = 15;
      DevParams* Pm = params_at(position, start, 55, mode ? int(oracle.size()) : 0);
      put_i(c.next_start, mode ? position : position - 1);
      if (kind == 0)
        dv_tree_beam(Pm, c, dtable, width, 8, true, 31, mask, doracle, ddrafts, dparents, dtree, 0);
      else
        dv_tree_pool(Pm, c, dtable, width, 7, 31, mask, doracle, ddrafts, dparents, dtree, 0);
      const std::vector<int> gd = down_i(ddrafts, width), gp = down_i(dparents, width);
      TreeRows tr{};
      CK(cudaMemcpy(&tr, dtree, sizeof(tr), cudaMemcpyDeviceToHost));
      bool ok = tr.n == width + 1;
      for (int i = 0; i < width; ++i) {
        const int r = i + 1;
        if (mode) ok &= gd[size_t(i)] == oracle[size_t(position + 1 + i)] && gp[size_t(i)] == i - 1 &&
                        tr.depth[r] == r && tr.path[r][r] == r && (r < 2 || tr.path[r][r - 1] == r - 1);
        else ok &= gd[size_t(i)] == mask && gp[size_t(i)] == -1 && tr.depth[r] == 1 && tr.path[r][1] == r &&
                   tr.path[r][0] == 0;
      }
      report(ok, std::string(kind ? "pool" : "beam") + (mode ? ": the test's drafts as a chain (P->oracle_len)"
                                                             : ": DV's first tree for a canvas of another round"));
      CK(cudaFree(Pm));
    }
  for (void* q : std::initializer_list<void*>{dtable, ddrafts, dparents, dtree, P, doracle}) CK(cudaFree(q));
  free_canvas(c);
}

// A canvas pass's attention: attn_prep rotates row w to P->start + 1 + w (P->position elsewhere) and writes its K/V
// to slot s0 + w only, the same bits when it preps the rows one at a time (pos_add = w); the attention of row w over
// the text [0, P->start] and the canvas's slots (all, or up to its own when causal), never the round's tree slots
// between, against ops::attend on those keys listed explicitly (the caches as stored); the same bits again, through
// attention() (no prefill kernel for its rows, causal or not), with tensor cores asked for every row (the canvas takes
// the float32 kernels), and a row alone; the per-head kernel too.
void test_canvas_attention(int hd, int nh, int nkv, int start, int N, bool causal, bool fp8, double theta) {
  seed("canvas attention", hd, nh, nkv, start, N, causal ? 1 : 0, fp8 ? 1 : 0, theta);
  const int rot = hd, s0 = start + 1 + FLYX_MAX_ROWS + 3, cap = s0 + N + 5;
  const int ld = nh * hd + 2 * nkv * hd, k_off = nh * hd, v_off = nh * hd + nkv * hd;
  const size_t kv_n = size_t(cap) * nkv * hd;
  const Vec qkv = floats(randv(size_t(N) * ld));
  KvRef kv{};
  std::vector<uint8_t> K8(kv_n), V8(kv_n);
  std::vector<float> Ks(size_t(cap) * nkv), Vs(Ks.size());
  Vec Kc, Vc;
  if (fp8) {
    std::uniform_int_distribution<int> code(0, 255);
    std::uniform_real_distribution<float> scale(0.001f, 0.01f);
    for (size_t i = 0; i < kv_n; ++i) {
      do K8[i] = uint8_t(code(rng)); while ((K8[i] & 0x7f) == 0x7f);  // no NaN codes
      do V8[i] = uint8_t(code(rng)); while ((V8[i] & 0x7f) == 0x7f);
    }
    for (size_t i = 0; i < Ks.size(); ++i) Ks[i] = scale(rng), Vs[i] = scale(rng);
    kv.k = dev<uint8_t>(kv_n), kv.v = dev<uint8_t>(kv_n), kv.ks = dev<float>(Ks.size()), kv.vs = dev<float>(Vs.size());
    CK(cudaMemcpy(kv.k, K8.data(), kv_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kv.v, V8.data(), kv_n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kv.ks, Ks.data(), Ks.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kv.vs, Vs.data(), Vs.size() * 4, cudaMemcpyHostToDevice));
  } else {
    Kc = randv(kv_n), Vc = randv(kv_n);
    round_f16(Kc);
    round_f16(Vc);
    kv.k = up_f16(Kc), kv.v = up_f16(Vc);
  }
  // The caches as stored, in float64 (FP8: codes times their scales).
  auto stored = [&](Vec* K, Vec* V) {
    if (!fp8) {
      *K = down_f16(static_cast<kv16*>(kv.k), kv_n);
      *V = down_f16(static_cast<kv16*>(kv.v), kv_n);
      return;
    }
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> k8(kv_n), v8(kv_n);
    std::vector<float> ks(Ks.size()), vs(Vs.size());
    CK(cudaMemcpy(k8.data(), kv.k, kv_n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(v8.data(), kv.v, kv_n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ks.data(), kv.ks, ks.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vs.data(), kv.vs, vs.size() * 4, cudaMemcpyDeviceToHost));
    K->resize(kv_n), V->resize(kv_n);
    for (size_t i = 0; i < kv_n; ++i) {
      (*K)[i] = ops::e4m3_to_double(k8[i]) * ks[i / size_t(hd)];
      (*V)[i] = ops::e4m3_to_double(v8[i]) * vs[i / size_t(hd)];
    }
  };
  // The caches' first contents back on the device.
  auto put_initial = [&] {
    if (fp8) {
      CK(cudaMemcpy(kv.k, K8.data(), kv_n, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(kv.v, V8.data(), kv_n, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(kv.ks, Ks.data(), Ks.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(kv.vs, Vs.data(), Vs.size() * 4, cudaMemcpyHostToDevice));
      return;
    }
    std::vector<uint16_t> k16(kv_n), v16(kv_n);
    for (size_t i = 0; i < kv_n; ++i) k16[i] = to_f16(Kc[i]), v16[i] = to_f16(Vc[i]);
    CK(cudaMemcpy(kv.k, k16.data(), kv_n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(kv.v, v16.data(), kv_n * 2, cudaMemcpyHostToDevice));
  };
  Vec K0, V0;
  stored(&K0, &V0);
  DevParams* P = params_at(start + 4, start);  // the round committed 4: the canvas still sits after its anchor
  float *dqkv = up(qkv), *dq = dev<float>(size_t(N) * nh * hd), *dout = dev<float>(size_t(N) * nh * hd);
  const CanvasKeys ck{s0};
  attn_prep(dqkv, ld, 0, k_off, v_off, nullptr, nullptr, 0.f, 1e-6f, theta, rot, nh, nkv, hd, N, P, kPosPosition,
            false, dq, kv, 0, 0, nullptr, 0, ck);
  const std::string tag = "hd " + std::to_string(hd) + " heads " + std::to_string(nh) + "/" + std::to_string(nkv) +
                          " text " + std::to_string(start + 1) + " slots " + std::to_string(N) +
                          (causal ? " causal" : "") + (fp8 ? " fp8" : "");
  Vec K1, V1;  // the caches after the prep
  {  // the prep: q and K at P->start + 1 + w, K/V at s0 + w and nowhere else
    Vec q(size_t(N) * nh * hd), Kw, Vw;
    for (int w = 0; w < N; ++w)
      for (int h = 0; h < nh; ++h)
        ops::prep_head(&qkv[size_t(w) * ld + h * hd], nullptr, hd, 1e-6, 0.0, start + 1 + w, theta, rot,
                       &q[(size_t(w) * nh + h) * hd]);
    expect("canvas q prep at P->start + 1 + w " + tag, down(dq, q.size()), q, 1e-5);
    stored(&K1, &V1);
    bool others = true;
    for (size_t i = 0; i < kv_n; ++i) {
      const int slot = int(i / (size_t(nkv) * hd));
      if (slot >= s0 && slot < s0 + N) continue;
      others &= K1[i] == K0[i] && V1[i] == V0[i];
    }
    report(others, "canvas K/V prep: no slot but s0 .. s0 + N - 1 written " + tag);
    if (!fp8) {
      Vec kgot, kwant, vgot, vwant;
      for (int w = 0; w < N; ++w)
        for (int j = 0; j < nkv; ++j) {
          Vec k(static_cast<size_t>(hd));
          ops::prep_head(&qkv[size_t(w) * ld + k_off + j * hd], nullptr, hd, 1e-6, 0.0, start + 1 + w, theta, rot,
                         k.data());
          for (int d = 0; d < hd; ++d) {
            const size_t at = (size_t(s0 + w) * nkv + j) * hd + d;
            kgot.push_back(K1[at]), kwant.push_back(from_f16(to_f16(k[size_t(d)])));
            vgot.push_back(V1[at]), vwant.push_back(from_f16(to_f16(qkv[size_t(w) * ld + v_off + j * hd + d])));
          }
        }
      expect("canvas K at slot s0 + w " + tag, kgot, kwant, 2e-3);
      expect("canvas V at slot s0 + w " + tag, vgot, vwant, 2e-3);
    }
  }
  {  // a row alone (pos_add = w), from the caches as they were: its q, and its K/V in its slot, the same bits
    put_initial();
    float* dq1 = dev<float>(size_t(N) * nh * hd);
    for (int w = 0; w < N; ++w)
      attn_prep(dqkv + size_t(w) * ld, ld, 0, k_off, v_off, nullptr, nullptr, 0.f, 1e-6f, theta, rot, nh, nkv, hd, 1,
                P, kPosPosition, false, dq1 + size_t(w) * nh * hd, kv, 0, w, nullptr, 0, ck);
    expect("canvas q prep row by row: same bits " + tag, down(dq1, size_t(N) * nh * hd), down(dq, size_t(N) * nh * hd),
           0);
    Vec K2, V2;
    stored(&K2, &V2);
    report(K2 == K1 && V2 == V1, "canvas K/V prep row by row: the same bits in the same slots " + tag);
    CK(cudaFree(dq1));
  }
  const int splits = 5;
  const float scale = float(1.0 / std::sqrt(hd));
  float* part = dev<float>(attention_partial_floats(N, nh, hd, splits));
  attention_split(dq, kv, nh, nkv, hd, N, causal, N, scale, P, nullptr, 0, dout, nh * hd, part, splits, TL, 0, 0,
                  nullptr, ck);
  const Vec got = down(dout, size_t(N) * nh * hd);
  Vec want(got.size()), K, V;
  stored(&K, &V);
  const Vec qd = down(dq, size_t(N) * nh * hd);
  for (int w = 0; w < N; ++w) {
    std::vector<int> keys;  // the text and the anchor, then the canvas's slots
    for (int j = 0; j <= start; ++j) keys.push_back(j);
    for (int j = 0; j < (causal ? w + 1 : N); ++j) keys.push_back(s0 + j);
    Vec Kg(keys.size() * nkv * hd), Vg(Kg.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      std::copy_n(&K[size_t(keys[i]) * nkv * hd], size_t(nkv) * hd, &Kg[i * nkv * hd]);
      std::copy_n(&V[size_t(keys[i]) * nkv * hd], size_t(nkv) * hd, &Vg[i * nkv * hd]);
    }
    for (int h = 0; h < nh; ++h)
      ops::attend(&qd[(size_t(w) * nh + h) * hd], Kg.data(), Vg.data(), int(keys.size()), nkv, hd, h / (nh / nkv),
                  1.0 / std::sqrt(hd), &want[(size_t(w) * nh + h) * hd]);
  }
  const double tol = start > 1000 ? 1e-4 : 2e-5;
  expect("canvas attention vs explicit keys " + tag, got, want, tol);
  auto again = [&](const Launch& L, bool dispatch, const char* what) {
    float* o = dev<float>(got.size());
    if (dispatch)
      attention(dq, kv, nh, nkv, hd, N, causal, N, scale, P, nullptr, 0, o, nh * hd, part, splits, L, 0, 0, nullptr,
                ck);
    else
      attention_split(dq, kv, nh, nkv, hd, N, causal, N, scale, P, nullptr, 0, o, nh * hd, part, splits, L, 0, 0,
                      nullptr, ck);
    const Vec r = down(o, got.size());
    CK(cudaFree(o));
    expect(std::string("canvas attention, ") + what + ": same bits " + tag, r, got, 0);
  };
  again(TL, false, "again");
  again(TL, true, "through attention()");
  KernelOptions mma;
  mma.rows_mma = true;
  mma.rows_mma_from = 0;
  Launch LM = TL;
  LM.opt = &mma;
  again(LM, false, "tensor cores asked for every row");
  {  // a row alone (its keys' splits follow its own key count)
    float* o = dev<float>(got.size());
    for (int w = 0; w < N; ++w)
      attention_split(dq + size_t(w) * nh * hd, kv, nh, nkv, hd, 1, causal, N - w, scale, P, nullptr, 0,
                      o + size_t(w) * nh * hd, nh * hd, part, splits, TL, w, 0, nullptr, ck);
    expect("canvas attention row by row: same bits " + tag, down(o, got.size()), got, 0);
    CK(cudaFree(o));
  }
  if (!fp8) {  // the per-head kernel (option attention=perhead)
    KernelOptions ph;
    ph.attention = kAttnPerHead;
    Launch LP = TL;
    LP.opt = &ph;
    float* o = dev<float>(got.size());
    attention_split(dq, kv, nh, nkv, hd, N, causal, N, scale, P, nullptr, 0, o, nh * hd, part, splits, LP, 0, 0,
                    nullptr, ck);
    const Vec r = down(o, got.size());
    expect("canvas attention, per-head kernel vs explicit keys " + tag, r, want, tol);
    attention_split(dq, kv, nh, nkv, hd, N, causal, N, scale, P, nullptr, 0, o, nh * hd, part, splits, LP, 0, 0,
                    nullptr, ck);
    expect("canvas attention, per-head kernel again: same bits " + tag, down(o, got.size()), r, 0);
    CK(cudaFree(o));
  }
  for (void* p : std::initializer_list<void*>{kv.k, kv.v, kv.ks, kv.vs, dqkv, dq, dout, part, P})
    if (p) CK(cudaFree(p));
}

void test_dv() {
  std::printf("dv\n");
  test_dv_selector_tables(1000, 200, 72);    // tiles cut by vocab, hidden and rank
  test_dv_selector_tables(2048, 1536, 256);  // MiniCPM5-1B's hidden and selector rank
  test_canvas_input(32, 256);
  test_canvas_input(9, 1536);                // MiniCPM5-1B's hidden, one GEMM tile
  test_canvas_stats(32, 5000, 256, 64);
  test_canvas_stats(5, 130560, 64, 256);     // MiniCPM5-1B's vocabulary and selector rank
  test_canvas_stats(3, 3000, 1536, 256);     // ... and hidden
  test_dv_slide(32, 64, 32);
  test_dv_slide(9, 40, 16);
  test_dv_selector_table(31, 256, 3000);
  test_dv_selector_table(7, 64, 500);
  test_dv_trees();
  std::printf("canvas attention\n");
  test_canvas_attention(128, 16, 2, 300, 32, false, false, 5e6);  // MiniCPM5-1B's heads, 32 slots
  test_canvas_attention(128, 16, 2, 300, 32, true, false, 5e6);   // ... causal
  test_canvas_attention(128, 16, 2, 1500, 32, false, false, 5e6);  // deep in the context
  test_canvas_attention(128, 16, 2, 300, 32, false, true, 5e6);   // FP8 cache
  test_canvas_attention(64, 8, 4, 100, 7, true, false, 1e4);      // G = 2
  test_canvas_attention(64, 8, 4, 0, 5, false, false, 1e4);       // the anchor at position 0
}

}  // namespace flyx::optest
