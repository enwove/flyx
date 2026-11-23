// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: gated delta nets: the round kernel, the chunked prefill recurrence, the grouped conv.
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

void test_gdn(const GdnDims d, const int rows, const double tol) {
  std::printf("gated delta net (dk %d, dv %d, %d rows)\n", d.dk, d.dv, rows);
  seed("gated delta net", d.nk, d.nv, d.dk, d.dv, rows);
  const int C = d.C(), vd = d.nv * d.dv, ld = C + vd + 2 * d.nv;
  // proj rows: [mixed (C) | z (vd) | b (nv) | a (nv)]
  Vec proj = randv(size_t(rows) * ld), convw = randv(size_t(C) * d.ck, 0.5), alog = randv(d.nv, 0.5),
      dtb = randv(d.nv, 0.5), gw = randv(d.dv, 0.3), S0 = randv(size_t(d.nv) * d.dv * d.dk, 0.2),
      W0 = randv(size_t(d.ck - 1) * C);
  for (auto& g : gw) g += 1.0;
  // reference: rows in order from (S0, W0)
  Vec S = S0, win = W0, y(size_t(rows) * vd);
  const int kd = d.nk * d.dk, rep = d.nv / d.nk;
  for (int r = 0; r < rows; ++r) {
    const double* mixed = &proj[size_t(r) * ld];
    Vec conv(C);
    ops::conv_row(win.data(), mixed, convw.data(), C, d.ck, conv.data());
    ops::conv_push(win.data(), mixed, C, d.ck);
    for (int kh = 0; kh < d.nk; ++kh) {
      ops::l2norm(&conv[kh * d.dk], d.dk, 1e-6);
      ops::l2norm(&conv[kd + kh * d.dk], d.dk, 1e-6);
      for (int i = 0; i < d.dk; ++i) conv[kh * d.dk + i] /= std::sqrt(double(d.dk));
    }
    for (int h = 0; h < d.nv; ++h) {
      const int kh = h / rep;
      double beta = ops::sigmoid(proj[size_t(r) * ld + C + vd + h]);
      double g = -std::exp(alog[h]) * ops::softplus(proj[size_t(r) * ld + C + vd + d.nv + h] + dtb[h]);
      Vec o(d.dv), on(d.dv);
      ops::gdn_row(&S[size_t(h) * d.dv * d.dk], &conv[kh * d.dk], &conv[kd + kh * d.dk], &conv[2 * kd + h * d.dv], beta, g,
                   d.dk, d.dv, o.data());
      ops::rmsnorm(o.data(), gw.data(), d.dv, 1e-6, 0.0, on.data());
      for (int j = 0; j < d.dv; ++j) y[size_t(r) * vd + h * d.dv + j] = on[j] * ops::silu(proj[size_t(r) * ld + C + h * d.dv + j]);
    }
  }
  // device: slot 1 holds the start state; rounds read it (round = 1).
  const size_t s_stride = S0.size(), w_stride = W0.size();
  Vec S4(4 * s_stride, 0.0), W4(4 * w_stride, 0.0);
  std::copy(S0.begin(), S0.end(), S4.begin() + s_stride);
  std::copy(W0.begin(), W0.end(), W4.begin() + w_stride);
  float *dS = up(S4), *dW = up(W4), *dproj = up(proj), *dcw = up(convw), *dal = up(alog), *ddt = up(dtb), *dgw = up(gw);
  float *dout = dev<float>(size_t(rows) * vd), *dy = dev<float>(size_t(rows) * vd);
  GdnBufs bf = GdnBufs::round(std::max(16, rows), d);
  bf.base = dev<float>(4 * bf.slot_stride);
  DevParams* P = params(0, 1);
  // verify-style pass: outputs, state untouched
  gdn_conv(dproj, ld, dcw, dW, w_stride, kSlotCur, -1, kSkipNever, bf, kSlotCur, P, rows, 0);
  gdn_prep(bf, kSlotCur, dproj + C + vd, dproj + C + vd + d.nv, ld, dal, ddt, P, rows, 0);
  gdn_recur(dS, s_stride, kSlotCur, -1, bf, kSlotCur, P, dout, rows, nullptr, false, 0);
  gdn_gated_norm(dout, dproj + C, ld, dgw, 1e-6f, dy, rows, d, 0);
  expect("outputs", down(dy, y.size()), y, tol);
  Vec Sdev = down(dS, 4 * s_stride);
  expect("verify leaves the state alone", Vec(Sdev.begin() + s_stride, Sdev.begin() + 2 * s_stride), S0, 1e-7);
  // The round path in one kernel (rows <= 16): the bits of the four kernels,
  // outputs and replay buffers; with one row and a next slot (plain decoding),
  // also the advanced window and state.
  if (rows <= 16) {
    const Vec y4 = down(dy, y.size()), bufs4 = down(bf.base + bf.slot_stride, bf.floats()), S4dev = Sdev;
    float* dy1 = dev<float>(y.size());
    CK(cudaMemset(bf.base + bf.slot_stride, 0, bf.floats() * 4));
    if (gdn_round(dproj, ld, dcw, dW, w_stride, dS, s_stride, -1, dal, ddt, dgw, 1e-6f, bf, P, rows, dy1, TL)) {
      expect("one kernel: same output bits", down(dy1, y.size()), y4, 0);
      expect("one kernel: same replay buffer bits", down(bf.base + bf.slot_stride, bf.floats()), bufs4, 0);
      expect("one kernel: state alone", down(dS, 4 * s_stride), S4dev, 0);
      if (d.dv % 32 == 0) {  // its output in 8 bits (block weights that take them): the rows blocks_stage leaves
        blocks_stage(dy1, vd, vd, rows, TL, true);
        const Q8Rows staged = q8_rows(rows, vd);
        gdn_round(dproj, ld, dcw, dW, w_stride, dS, s_stride, -1, dal, ddt, dgw, 1e-6f, bf, P, rows, nullptr, TL, true);
        const bool same8 = q8_rows(rows, vd) == staged;
        std::printf("  %-4s %-48s\n", same8 ? "ok" : "FAIL", "one kernel in 8 bits: blocks_stage's rows");
        g_fail += !same8;
      }
      float *dSa = up(S4), *dWa = up(W4), *dSb = up(S4), *dWb = up(W4);
      float *dya = dev<float>(vd), *dyb = dev<float>(vd);
      gdn_round(dproj, ld, dcw, dWa, w_stride, dSa, s_stride, kSlotNext, dal, ddt, dgw, 1e-6f, bf, P, 1, dya, TL);
      gdn_conv(dproj, ld, dcw, dWb, w_stride, kSlotCur, kSlotNext, kSkipIfStopped, bf, kSlotCur, P, 1, 0);
      gdn_prep(bf, kSlotCur, dproj + C + vd, dproj + C + vd + d.nv, ld, dal, ddt, P, 1, 0);
      gdn_recur(dSb, s_stride, kSlotCur, kSlotNext, bf, kSlotCur, P, dout, 1, nullptr, kSkipIfStopped, 0);
      gdn_gated_norm(dout, dproj + C, ld, dgw, 1e-6f, dyb, 1, d, 0);
      expect("one kernel, plain decoding: same output bits", down(dya, vd), down(dyb, vd), 0);
      expect("one kernel, plain decoding: same next state bits", down(dSa, 4 * s_stride), down(dSb, 4 * s_stride), 0);
      expect("one kernel, plain decoding: same next window bits", down(dWa, 4 * w_stride), down(dWb, 4 * w_stride), 0);
    } else {
      std::printf("  (one-kernel round path: not for these shapes)\n");
      const std::vector<float> f(bufs4.begin(), bufs4.end());
      CK(cudaMemcpy(bf.base + bf.slot_stride, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    }
  }
  // fold all rows from slot 1 into slot 3 (explicit slots)
  gdn_recur(dS, s_stride, kSlotExplicit + 1, kSlotExplicit + 3, bf, kSlotExplicit + 1, P, nullptr, rows, nullptr, false, 0);
  gdn_conv_fold(dW, w_stride, kSlotExplicit + 1, kSlotExplicit + 3, bf, kSlotExplicit + 1, P, rows, nullptr, false, 0);
  Sdev = down(dS, 4 * s_stride);
  Vec Wdev = down(dW, 4 * w_stride);
  expect("fold: state", Vec(Sdev.begin() + 3 * s_stride, Sdev.begin() + 4 * s_stride), S, tol);
  expect("fold: conv window", Vec(Wdev.begin() + 3 * w_stride, Wdev.begin() + 4 * w_stride), win, 1e-6);
  // prefill-style pass: in place on slot 1, must give the same state
  gdn_conv(dproj, ld, dcw, dW, w_stride, kSlotCur, kSlotCur, kSkipNever, bf, kSlotCur, P, rows, 0);
  gdn_prep(bf, kSlotCur, dproj + C + vd, dproj + C + vd + d.nv, ld, dal, ddt, P, rows, 0);
  gdn_recur(dS, s_stride, kSlotCur, kSlotCur, bf, kSlotCur, P, dout, rows, nullptr, false, 0);
  Sdev = down(dS, 4 * s_stride);
  Wdev = down(dW, 4 * w_stride);
  expect("in place: state", Vec(Sdev.begin() + s_stride, Sdev.begin() + 2 * s_stride), S, tol);
  // A fold (no outputs) leaves the bits of the pass that wrote outputs: plain
  // decoding folds, speculation keeps the verification's rows.
  expect("fold and in place: same state bits", Vec(Sdev.begin() + 3 * s_stride, Sdev.begin() + 4 * s_stride),
         Vec(Sdev.begin() + s_stride, Sdev.begin() + 2 * s_stride), 0);
  expect("in place: conv window", Vec(Wdev.begin() + w_stride, Wdev.begin() + 2 * w_stride), win, 1e-6);
  // The same rows one at a time (decode) give the bits of all rows at once (prefill).
  const Vec all_out = down(dout, size_t(rows) * vd);
  Vec S4b(4 * s_stride, 0.0), W4b(4 * w_stride, 0.0);
  std::copy(S0.begin(), S0.end(), S4b.begin() + 2 * s_stride);
  std::copy(W0.begin(), W0.end(), W4b.begin() + 2 * w_stride);
  float *dS2 = up(S4b), *dW2 = up(W4b), *dout1 = dev<float>(size_t(rows) * vd);
  const int one = kSlotExplicit + 2;
  for (int r = 0; r < rows; ++r) {
    const float* pr = dproj + size_t(r) * ld;
    gdn_conv(pr, ld, dcw, dW2, w_stride, one, one, kSkipNever, bf, one, P, 1, 0);
    gdn_prep(bf, one, pr + C + vd, pr + C + vd + d.nv, ld, dal, ddt, P, 1, 0);
    gdn_recur(dS2, s_stride, one, one, bf, one, P, dout1 + size_t(r) * vd, 1, nullptr, false, 0);
  }
  expect("row by row: same output bits", down(dout1, size_t(rows) * vd), all_out, 0);
  const Vec S2 = down(dS2, 4 * s_stride), W2 = down(dW2, 4 * w_stride);
  expect("row by row: same state bits", Vec(S2.begin() + 2 * s_stride, S2.begin() + 3 * s_stride),
         Vec(Sdev.begin() + s_stride, Sdev.begin() + 2 * s_stride), 0);
  expect("row by row: same window bits", Vec(W2.begin() + 2 * w_stride, W2.begin() + 3 * w_stride),
         Vec(Wdev.begin() + w_stride, Wdev.begin() + 2 * w_stride), 0);
}

// The chunked prefill recurrence (gdn_chunk) against the float64 rows, and the
// bits it promises: a pass that starts at a multiple of the alignment gives the
// bits of one pass over the same positions, and a row's output does not depend
// on where its pass ends.  pos0: the position of the first row.
void test_gdn_chunk(const GdnDims d, const int rows, const int pos0, const double tol) {
  std::printf("chunked gated delta net (nk %d, nv %d, dk %d, dv %d, %d rows from position %d)\n", d.nk, d.nv, d.dk,
              d.dv, rows, pos0);
  if (!gdn_chunk_device()) {
    std::printf("  (not on this GPU: the recurrence stays float32 there, as llama.cpp's)\n");
    return;
  }
  seed("chunked gated delta net", d.nk, d.nv, d.dk, d.dv, rows, pos0);
  const int kd = d.nk * d.dk, rep = d.nv / d.nk, C = d.C(), vd = d.nv * d.dv;
  // per-row inputs: normalised q (with 1/sqrt(dk)) and k per key head, v per value head, beta, g
  Vec q(size_t(rows) * kd), k(size_t(rows) * kd), v = randv(size_t(rows) * vd), beta(size_t(rows) * d.nv),
      g(size_t(rows) * d.nv);
  for (int r = 0; r < rows; ++r)
    for (int kh = 0; kh < d.nk; ++kh) {
      Vec a = randv(d.dk), b = randv(d.dk);
      ops::l2norm(a.data(), d.dk, 1e-6);
      ops::l2norm(b.data(), d.dk, 1e-6);
      for (int i = 0; i < d.dk; ++i) {
        q[(size_t(r) * d.nk + kh) * d.dk + i] = a[i] / std::sqrt(double(d.dk));
        k[(size_t(r) * d.nk + kh) * d.dk + i] = b[i];
      }
    }
  std::uniform_real_distribution<double> ub(0.05, 0.95), ug(-1.5, -0.01);
  for (auto& x : beta) x = ub(rng);
  for (auto& x : g) x = ug(rng);
  const Vec S0 = randv(size_t(d.nv) * d.dv * d.dk, 0.2);
  // reference, row by row
  Vec S = S0, o(size_t(rows) * vd);
  for (int r = 0; r < rows; ++r)
    for (int h = 0; h < d.nv; ++h) {
      const int kh = h / rep;
      ops::gdn_row(&S[size_t(h) * d.dv * d.dk], &q[(size_t(r) * d.nk + kh) * d.dk], &k[(size_t(r) * d.nk + kh) * d.dk],
                   &v[size_t(r) * vd + h * d.dv], beta[size_t(r) * d.nv + h], g[size_t(r) * d.nv + h], d.dk, d.dv,
                   &o[size_t(r) * vd + h * d.dv]);
    }
  // A prefill set holding rows [r0, r1) of the inputs, the state in slot 1 (round 1).
  const size_t s_stride = S0.size();
  struct Pass {
    float *S, *out;
    GdnBufs bf;
  };
  auto pass = [&](int r0, int r1, float* dS) {
    const int n = r1 - r0;
    GdnBufs bf = GdnBufs::prefill(std::max(n, gdn_chunk_rows()), d);  // a prefill set is never smaller
    std::vector<float> host(bf.floats(), 0.f);
    for (int r = 0; r < n; ++r) {
      for (int i = 0; i < kd; ++i) {
        host[bf.o_qn() + size_t(r) * kd + i] = float(q[size_t(r0 + r) * kd + i]);
        host[bf.o_kn() + size_t(r) * kd + i] = float(k[size_t(r0 + r) * kd + i]);
      }
      for (int i = 0; i < vd; ++i) host[bf.o_v() + size_t(r) * vd + i] = float(v[size_t(r0 + r) * vd + i]);
      for (int h = 0; h < d.nv; ++h) {
        host[bf.o_beta() + size_t(r) * d.nv + h] = float(beta[size_t(r0 + r) * d.nv + h]);
        host[bf.o_g() + size_t(r) * d.nv + h] = float(g[size_t(r0 + r) * d.nv + h]);
      }
    }
    bf.base = dev<float>(host.size());
    CK(cudaMemcpy(bf.base, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    float* dout = dev<float>(size_t(n) * vd);
    if (!gdn_chunk(dS, s_stride, kSlotCur, bf, params(pos0 + r0, 1), pos0 + r0, dout, n, 0)) {
      std::printf("  FAIL gdn_chunk does not take these shapes\n");
      ++g_fail;
    }
    return Pass{dS, dout, bf};
  };
  Vec S4(4 * s_stride, 0.0);
  std::copy(S0.begin(), S0.end(), S4.begin() + s_stride);
  const Pass one = pass(0, rows, up(S4));
  const Vec out1 = down(one.out, size_t(rows) * vd), state1 = down(one.S, 4 * s_stride);
  expect("outputs", out1, o, tol);
  expect("state", Vec(state1.begin() + s_stride, state1.begin() + 2 * s_stride), S, tol);
  // Two passes split where an aligned chunk starts: the bits of one pass.
  const int A = gdn_chunk_align();
  const int cut = (pos0 / A + 1) * A - pos0 + A;  // the second aligned position after pos0
  if (cut < rows) {
    float* dS = up(S4);
    const Pass a = pass(0, cut, dS), b = pass(cut, rows, dS);
    Vec out2 = down(a.out, size_t(cut) * vd);
    const Vec tail = down(b.out, size_t(rows - cut) * vd);
    out2.insert(out2.end(), tail.begin(), tail.end());
    expect("passes split at an aligned position: same output bits", out2, out1, 0);
    expect("passes split at an aligned position: same state bits", down(dS, 4 * s_stride), state1, 0);
  }
  // A shorter pass: the same bits for its rows.
  const int n1 = std::max(1, rows / 2 + 7);
  if (n1 < rows) {
    const Pass s = pass(0, n1, up(S4));
    expect("a row's output does not depend on the pass end", down(s.out, size_t(n1) * vd),
           Vec(out1.begin(), out1.begin() + size_t(n1) * vd), 0);
  }
}

void test_grouped_conv() {
  std::printf("grouped conv\n");
  seed("grouped conv");
  const int B = 16, H = 64, taps = 2, group = 16, G = H / group, ld_delta = 2 * taps * G;
  Vec x = randv(size_t(B) * H), delta = randv(size_t(B) * ld_delta, 0.3), base = randv(size_t(taps) * H, 0.5),
      y0 = randv(size_t(B) * H);
  Vec d0(size_t(B) * taps * G);
  for (int r = 0; r < B; ++r)
    for (int i = 0; i < taps * G; ++i) d0[size_t(r) * taps * G + i] = delta[size_t(r) * ld_delta + taps * G + i];
  Vec want(x.size());
  ops::grouped_conv(x.data(), d0.data(), base.data(), B, H, taps, group, B, want.data());
  for (size_t i = 0; i < want.size(); ++i) want[i] += y0[i];
  float *dx = up(x), *dd = up(delta), *db = up(base), *dy = up(y0);
  grouped_conv(dx, H, dd + taps * G, ld_delta, db, H, taps, group, dy, H, B, true, 0);
  expect("side 1, accumulate", down(dy, want.size()), want, 1e-6);
}

}  // namespace flyx::optest
