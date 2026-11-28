// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: gated delta nets: the round kernel, the chunked prefill recurrence, the grouped conv.
#include "tests/backends/cuda/op_test.cuh"
#include "ops/ple/ref.h"
#include "ops/qsa/ref.h"

namespace flyx::optest {

template <class T>
static T* upload_values(const T* values, size_t n) {
  T* result = dev<T>(n);
  CK(cudaMemcpy(result, values, n * sizeof(T), cudaMemcpyHostToDevice));
  return result;
}

void test_sparse_index() {
  constexpr int R = 13, D = 32, G = 4, Q = 2;
  seed("sparse index", R, D);
  Vec packed = randv(R * (Q + 1) * D), q(R * D, 0), values = randv(R * D), expected(R * D);
  round_f16(values);
  auto* ones = up(Vec(D, 1));
  SparseIndex ix{D, Q, G, 2, 0, 10000.0, {}, ones, ones, 1e-6f, 0.f};
  KvRef kv{};
  kv.k = up_f16(q);
  kv.v = up_f16(values);
  float *out = dev<float>(R * D), *raw = dev<float>(R * (D + 3));
  auto* p = params(0);
  indexed_attention(ix, up(packed), raw, dev<float>(kIndexTile * 5), 5, dev<int>(kIndexTile * 2), up(q), kv, 1, 1, D,
                    nullptr, 0, out, R, p, nullptr, 0);
  for (int r = 0; r < R; ++r) {
    const int groups = (r + 1) / G;
    Vec nq(Q * D), pooled(D), scores(groups), weights(D, 1);
    for (int h = 0; h < Q; ++h)
      ops::rmsnorm(&packed[(r * (Q + 1) + h) * D], weights.data(), D, 1e-6, 0, nq.data() + h * D);
    for (int g = 0; g < groups; ++g) {
      std::fill(pooled.begin(), pooled.end(), 0);
      for (int j = 0; j < G; ++j)
        for (int d = 0; d < D; ++d) pooled[d] += packed[((g * G + j) * (Q + 1) + Q) * D + d] / G;
      ops::rmsnorm(pooled.data(), weights.data(), D, 1e-6, 0, pooled.data());
      scores[g] = ops::qsa_score(nq.data(), pooled.data(), Q, D);
    }
    const auto pos = ops::qsa_positions(scores, r + 1, G, 2 * G);
    for (int t : pos)
      for (int d = 0; d < D; ++d) expected[r * D + d] += values[t * D + d] / pos.size();
  }
  expect("sparse selected values", down(out, R * D), expected, 2e-6);
  TreeRows tree{};
  tree.n = 4;
  tree.n_taken = 2;
  tree.taken[1] = 2;
  tree.taken[2] = 3;
  const auto before = down(raw, R * (D + 3));
  compact_index(raw, D + 3, p, upload_values(&tree, 1), 0);
  const auto after = down(raw, R * (D + 3));
  for (int r = 0; r < 3; ++r)
    for (int d = 0; d < D; ++d)
      if (before[tree.taken[r] * (D + 3) + d] != after[r * (D + 3) + d]) {
        ++g_fail;
        return;
      }
}

void test_ple() {
  constexpr int R = 13, H = 32, C = 2, W = H * C, S = W + 1, N = 3;
  uint64_t m[]{3, 5, 7}, of[]{0, 23}, mod[]{23, 29};
  const int ids[R] = {42, 43, 44, 2, 1, 6, 15, 13, 9, 4, 7, 8, 42};
  seed("PLE", R, H, C);
  flyx_ple_block b{};
  b.hidden = H;
  b.streams = C;
  b.ngram = N;
  b.heads_per_order = 1;
  b.row_width = H;
  b.eos = 9;
  b.taps = 4;
  b.dilation = 3;
  b.multipliers = upload_values(m, N);
  b.moduli = upload_values(mod, 2);
  b.offsets = upload_values(of, 2);
  PleWindow w{dev<float>(FLYX_ROUND_SLOTS * 9 * S), dev<float>(kGdnReplaySlots * R * S), 9 * S, R * S};
  auto* tokens = upload_values(ids, R);
  auto* keys = dev<int>(R * 2);
  auto* p = params(0);
  ple_hash(b, w, tokens, keys, R, p, nullptr, 0);
  int hits[R * 2];
  CK(cudaMemcpy(hits, keys, sizeof hits, cudaMemcpyDeviceToHost));
  b.multipliers = m;
  b.moduli = mod;
  b.offsets = of;
  Vec embed = randv(52 * H), gather(R * 2 * H);
  for (int r = 0; r < R; ++r) {
    int prior[2] = {r ? ids[r - 1] : 9, r > 1 ? ids[r - 2] : 9}, want[2];
    ops::ple_indices(ids[r], prior, std::min(r, 2), b, want);
    for (int j = 0; j < 2; ++j) {
      if (want[j] != hits[2 * r + j]) ++g_fail;
      std::copy_n(embed.data() + want[j] * H, H, gather.data() + (2 * r + j) * H);
    }
  }
  auto* emb = dev<float>(gather.size());
  ple_decode(reinterpret_cast<uint8_t*>(up(embed)), H * sizeof(float), FLYX_DT_F32, H, keys, R * 2, emb, 0);
  expect("packed PLE gather", down(emb, gather.size()), gather, 1e-7);
  for (const int dtype : {FLYX_DT_BF16, FLYX_DT_F16, FLYX_DT_GGML_Q8_0,
                          FLYX_DT_GGML_Q5_0, FLYX_DT_GGML_Q4_0}) {
    const int block = flyx_block_bytes(dtype), stride = block ? block : 2 * H;
    std::vector<uint8_t> payload(52 * stride), staged(2 * R * stride);
    for (int i = 0; i < 52; ++i) {
      uint8_t *row = payload.data() + i * stride;
      for (int j = 0; j < stride; ++j)
        row[j] = uint8_t(17 * i + 31 * j);
      for (int j = 0; j < (block ? 1 : H); ++j)
        row[2 * j + 1] = uint8_t(0x38 + (j + i) % 8);
    }
    flyx_tensor_view source{};
    source.dtype = dtype;
    source.data = payload.data();
    for (int i = 0; i < 2 * R; ++i) {
      std::copy_n(payload.data() + hits[i] * stride, stride,
                  staged.data() + i * stride);
      for (int j = 0; j < H; ++j)
        gather[i * H + j] = ops::load(&source, hits[i] * H + j);
    }
    ple_decode(upload_values(payload.data(), payload.size()), stride, dtype, H,
               keys, 2 * R, emb, 0);
    const auto resident = down(emb, gather.size());
    ple_decode(upload_values(staged.data(), staged.size()), stride, dtype, H,
               nullptr, 2 * R, emb, 0);
    if (resident != down(emb, gather.size()) || resident != gather)
      ++g_fail;
  }
  Vec k = randv(R * W), q = randv(R * W), v = randv(R * H), branch(R * W), u = randv(R * S), cw = randv(W * 4),
      wide(R * W);
  auto* y = dev<float>(R * W);
  ple_gates(up(k), up(q), up(v), H, C, R, y, 0);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c) {
      const int at = (r * C + c) * H;
      const double gate = ops::ple_gate(k.data() + at, q.data() + at, H);
      for (int j = 0; j < H; ++j) branch[at + j] = v[r * H + j] * gate;
    }
  expect("PLE gate", down(y, branch.size()), branch, 2e-6);
  auto* output = dev<float>(wide.size());
  ple_convolve(b, w, up(cw), y, up(u), output, tokens, R, p, nullptr, true, -1, -1, -1, 0);
  for (int r = 0; r < R; ++r)
    for (int d = 0; d < W; ++d) {
      double a = 0;
      for (int j = 0; j < 4; ++j)
        if (r >= 9 - 3 * j) a += cw[4 * d + j] * u[(r - 9 + 3 * j) * S + d];
      wide[r * W + d] = branch[r * W + d] + ops::silu(a);
    }
  expect("PLE convolution", down(output, wide.size()), wide, 3e-6);
  TreeRows tree{};
  tree.n = 4;
  tree.n_taken = 2;
  tree.taken[1] = 2;
  tree.taken[2] = 3;
  ple_compact(w, S, p, upload_values(&tree, 1), 0);
  const auto compacted = down(w.replay, 3 * S);
  for (int r = 0; r < 3; ++r)
    for (int d = 0; d < W; ++d)
      if (float(u[tree.taken[r] * S + d]) != compacted[r * S + d]) {
        ++g_fail;
        return;
      }
}

void test_hc() {
  constexpr int R = 3, C = 4, H = 32;
  seed("hyper connection", R, C, H);
  Vec x = randv(R * H), u = randv(R * C * H), a = randv(R * C * H), j = randv(R * C), router = randv(H);
  auto* dx = up(x);
  auto* dw = dev<float>(u.size());
  auto* du = up(u);
  auto* da = up(a);
  auto* dj = up(j);
  Vec wide(u.size()), mixed(x.size(), 0);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i) {
        const size_t k = (r * C + c) * H + i;
        wide[k] = x[r * H + i];
        mixed[r * H + i] += u[k] * ops::sigmoid(a[k]) / C;
      }
  hc_rows(FLYX_HC_EXPAND, dx, dw, nullptr, nullptr, nullptr, R, C, H, 0);
  expect("expand", down(dw, wide.size()), wide, 1e-7);
  hc_rows(FLYX_HC_MIX, dx, dw, du, dj, da, R, C, H, 0);
  expect("mix", down(dx, x.size()), mixed, 1e-6);
  hc_rows(FLYX_HC_SCATTER, dx, dw, nullptr, dj, nullptr, R, C, H, 0);
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      for (int i = 0; i < H; ++i) wide[(r * C + c) * H + i] += 2 * ops::sigmoid(j[r * C + c] / C) * mixed[r * H + i];
  expect("scatter", down(dw, wide.size()), wide, 1e-6);
  hc_down_silu(du, u.size(), C, 0);
  for (auto& v : u) v = ops::silu(v / C);
  expect("scaled SiLU", down(du, u.size()), u, 1e-6);
  shared_expert_add(dx, up(x), up(router), da, R, H, 0);
  for (int r = 0; r < R; ++r) {
    double v = 0;
    for (int i = 0; i < H; ++i) v += x[r * H + i] * router[i];
    for (int i = 0; i < H; ++i) mixed[r * H + i] += ops::sigmoid(v) * a[r * H + i];
  }
  expect("shared expert gate", down(dx, mixed.size()), mixed, 1e-6);
}

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
      for (int j = 0; j < d.dv; ++j) {
        const double z = proj[size_t(r) * ld + C + h * d.dv + j];
        y[size_t(r) * vd + h * d.dv + j] = on[j] * (d.sigmoid_output ? ops::sigmoid(z) : ops::silu(z));
      }
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
      // Into the split (y null), `also` takes the output in float32 beside it, the rows a GDN block's out adapter
      // reads: the bits of y, whether the split holds bf16 parts or 8 bits; for a batch's slice (row_offset) at the
      // slice's rows, those before it untouched.
      const int off = 3, total = rows + off;
      float* dalso = dev<float>(size_t(total) * vd);
      for (const bool q8 : {false, true}) {
        if (q8 && d.dv % 32 != 0) continue;
        for (const int at : {0, off}) {
          CK(cudaMemset(dalso, 0xff, size_t(total) * vd * 4));  // NaN: what is not written stays so
          gdn_round(dproj, ld, dcw, dW, w_stride, dS, s_stride, -1, dal, ddt, dgw, 1e-6f, bf, P, rows, nullptr, TL, q8,
                    rows + at, at, dalso + size_t(at) * vd);
          const std::string how = std::string(q8 ? "8 bits" : "bf16 parts") + (at ? ", a slice" : "");
          expect("one kernel, also beside the split (" + how + "): the output's bits",
                 down(dalso + size_t(at) * vd, y.size()), y4, 0);
          const Vec before = down(dalso, size_t(at) * vd);
          const bool untouched = std::all_of(before.begin(), before.end(), [](double v) { return std::isnan(v); });
          std::printf("  %-4s %-48s\n", untouched ? "ok" : "FAIL", ("also (" + how + "): no row outside it").c_str());
          g_fail += !untouched;
        }
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
