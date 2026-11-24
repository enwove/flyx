// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// CUDA kernel tests for Inkling-2 target ops (softcap, AttnRes, loop control, ShortConv,
// relative-position-bias attention, sparse MoE) and DSpark drafter kernels against float64 refs.
#include "backends/cuda/loader/exact_bf16.h"
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

namespace {

bool bits_equal(const Vec& a, const Vec& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    float fa = float(a[i]), fb = float(b[i]);
    uint32_t ua = 0, ub = 0;
    std::memcpy(&ua, &fa, 4);
    std::memcpy(&ub, &fb, 4);
    if (ua != ub) return false;
  }
  return true;
}

void expect_bits(const std::string& what, const Vec& a, const Vec& b) {
  const bool ok = bits_equal(a, b);
  std::printf("  %-4s %-48s %s\n", ok ? "ok" : "FAIL", what.c_str(), ok ? "bit-identical" : "BIT MISMATCH");
  if (!ok) ++g_fail;
}

double round_fp8_e4m3_vec(Vec& v) {
  double amax = 0.0;
  for (double x : v) amax = std::max(amax, std::fabs(x));
  const float sc = float(amax) / 448.f;
  for (double& x : v) {
    if (sc > 0.f) {
      const uint8_t code = __nv_cvt_float_to_fp8(float(x) / sc, __NV_SATFINITE, __NV_E4M3);
      x = double(ops::e4m3_to_double(code)) * double(sc);
    } else {
      x = 0.0;
    }
  }
  return double(sc);
}

void round_q8_inplace(double* row, int K) {
  std::vector<int8_t> p(static_cast<size_t>(K));
  std::vector<float> d(static_cast<size_t>(K / 32));
  ops::round_q8(row, K, p.data(), d.data());
  for (int i = 0; i < K; ++i) row[i] = double(p[size_t(i)]) * double(d[size_t(i / 32)]);
}

}  // namespace

void test_inkling2_softcap() {
  std::printf("inkling2 softcap_rows\n");
  seed("inkling2_softcap", 7, 1029);
  const int R = 7, V = 1029;
  const float cap = 30.f;
  Vec x = randv(size_t(R) * V, 25.0);
  x[0] = -std::numeric_limits<double>::infinity();
  x[size_t(R - 1) * V + V - 1] = -std::numeric_limits<double>::infinity();

  float* d = up(x);
  softcap_rows(d, V, V, R, cap, 0);
  Vec got = down(d, size_t(R) * V);
  cudaFree(d);

  bool neg_inf_ok = std::isinf(got[0]) && got[0] < 0 && std::isinf(got.back()) && got.back() < 0;
  std::printf("  %-4s %-48s\n", neg_inf_ok ? "ok" : "FAIL", "softcap preserves -inf (head_cert)");
  if (!neg_inf_ok) ++g_fail;

  Vec got_fin, want_fin;
  for (size_t i = 0; i < x.size(); ++i) {
    if (!std::isfinite(x[i])) continue;
    got_fin.push_back(got[i]);
    want_fin.push_back(ops::softcap(x[i], double(cap)));
  }
  expect("softcap_rows finite", got_fin, want_fin, 2e-6);
}

void test_inkling2_attn_res() {
  std::printf("inkling2 attn_res_mix\n");
  for (int nb : {0, 1, 5}) {
    for (int nq : {1, 3, 4}) {
      const int R = 9, H = 512;
      seed("inkling2_attn_res", nb, nq, R, H);
      Vec bank = randv(size_t(nb) * R * H, 1.0);
      Vec cur = randv(size_t(R) * H, 1.0);
      Vec queries = randv(size_t(nq) * H, 0.5);

      float* db = nb ? up(bank) : nullptr;
      float* dc = up(cur);
      float* dq = up(queries);
      float* dout = dev<float>(size_t(nq) * R * H);
      attn_res_mix(db, nb, dc, dq, nq, R, H, dout, 0);
      Vec got = down(dout, size_t(nq) * R * H);

      // Reference (query-major [nq, R, H] matching CUDA layout).
      Vec want(size_t(nq) * R * H, 0.0);
      Vec tmp(size_t(nq) * H);
      std::vector<const double*> ptrs(static_cast<size_t>(nb));
      for (int r = 0; r < R; ++r) {
        for (int b = 0; b < nb; ++b) ptrs[size_t(b)] = bank.data() + (size_t(b) * R + r) * H;
        ops::attn_res_mix(ptrs.data(), nb, cur.data() + size_t(r) * H, queries.data(), nq, H, tmp.data());
        for (int q = 0; q < nq; ++q)
          std::copy_n(tmp.data() + size_t(q) * H, H, want.data() + (size_t(q) * R + r) * H);
      }
      expect("attn_res_mix nb=" + std::to_string(nb) + " nq=" + std::to_string(nq), got, want, 1e-5);

      // Row-count independence: row 3 alone vs in 9 rows.
      const int rr = 3;
      Vec bank1(size_t(nb) * H);
      for (int b = 0; b < nb; ++b)
        std::copy_n(bank.data() + (size_t(b) * R + rr) * H, H, bank1.data() + size_t(b) * H);
      Vec cur1(cur.begin() + ptrdiff_t(rr) * H, cur.begin() + ptrdiff_t(rr + 1) * H);
      float* db1 = nb ? up(bank1) : nullptr;
      float* dc1 = up(cur1);
      float* dout1 = dev<float>(size_t(nq) * H);
      attn_res_mix(db1, nb, dc1, dq, nq, 1, H, dout1, 0);
      Vec got1 = down(dout1, size_t(nq) * H);
      Vec slice_r(size_t(nq) * H);
      for (int q = 0; q < nq; ++q)
        std::copy_n(got.data() + (size_t(q) * R + rr) * H, H, slice_r.data() + size_t(q) * H);
      expect_bits("attn_res_mix R=1 vs R=9 nb=" + std::to_string(nb) + " nq=" + std::to_string(nq), got1, slice_r);

      if (db) cudaFree(db);
      if (db1) cudaFree(db1);
      cudaFree(dc);
      cudaFree(dc1);
      cudaFree(dq);
      cudaFree(dout);
      cudaFree(dout1);
    }
  }
}

void test_inkling2_loop_ctrl() {
  std::printf("inkling2 loop_ctrl\n");
  const int R = 9, H = 256, rank = 64;
  seed("inkling2_loop_ctrl", R, H, rank);
  Vec cur = randv(size_t(R) * H, 1.0);
  Vec msg = randv(size_t(R) * H, 1.0);
  Vec w_down = randv(size_t(rank) * 2 * H, 0.05);
  Vec w_gain = randv(size_t(H) * rank, 0.1);
  Vec w_forget = randv(size_t(H) * rank, 0.1);
  round_bf16(w_down);
  round_bf16(w_gain);
  round_bf16(w_forget);

  std::vector<uint16_t> sd, sg, sf;
  flyx_tensor_view vd = view_of(w_down, &sd, rank, 2 * H);
  flyx_tensor_view vg = view_of(w_gain, &sg, H, rank);
  flyx_tensor_view vf = view_of(w_forget, &sf, H, rank);
  Vec want(size_t(R) * H, 0.0);
  ops::loop_control(cur.data(), msg.data(), R, H, rank, &vd, &vg, &vf, want.data());

  float* d_cur = up(cur);
  float* d_msg = up(msg);
  bf16* d_wd = up_bf16(w_down);
  bf16* d_wg = up_bf16(w_gain);
  bf16* d_wf = up_bf16(w_forget);
  float* d_cat = dev<float>(size_t(R) * 2 * H);
  float* d_gate = dev<float>(size_t(R) * rank);
  float* d_gf = dev<float>(size_t(R) * 2 * H);
  float* d_out = dev<float>(size_t(R) * H);

  loop_ctrl_prep(d_cur, d_msg, R, H, d_cat, 0);
  gemm(d_cat, 2 * H, d_wd, rank, 2 * H, d_gate, rank, R, false, TL);
  gemm(d_gate, rank, d_wg, H, rank, d_gf, 2 * H, R, false, TL);
  gemm(d_gate, rank, d_wf, H, rank, d_gf + H, 2 * H, R, false, TL);
  loop_ctrl_apply(d_cur, d_msg, d_gf, 2 * H, d_gf + H, 2 * H, R, H, d_out, 0);
  Vec got = down(d_out, size_t(R) * H);
  expect("loop_ctrl_prep + apply", got, want, 2e-5);

  // In-place apply (out == cur) must be bit-identical.
  loop_ctrl_apply(d_cur, d_msg, d_gf, 2 * H, d_gf + H, 2 * H, R, H, d_cur, 0);
  Vec got_ip = down(d_cur, size_t(R) * H);
  expect_bits("loop_ctrl_apply in-place", got_ip, got);

  cudaFree(d_cur);
  cudaFree(d_msg);
  cudaFree(d_wd);
  cudaFree(d_wg);
  cudaFree(d_wf);
  cudaFree(d_cat);
  cudaFree(d_gate);
  cudaFree(d_gf);
  cudaFree(d_out);
}

namespace {

// A float32 matrix [N][K] as a model file holds one (the reference reads these very values).
flyx_tensor_view view_f32(const std::vector<float>& w, int N, int K) {
  flyx_tensor_view t{};
  t.dtype = FLYX_DT_F32;
  t.ndim = 2;
  t.shape[0] = N;
  t.shape[1] = K;
  t.data = w.data();
  t.nbytes = w.size() * sizeof(float);
  return t;
}

std::vector<float> randf(size_t n, double scale) {
  const Vec v = randv(n, scale);
  return std::vector<float>(v.begin(), v.end());
}

float* up_f(const std::vector<float>& v) {
  float* p = dev<float>(v.size());
  CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  return p;
}

void expect_device_bits(const std::string& what, const float* got, const float* want, size_t count) {
  std::vector<uint32_t> a(count), b(count);
  CK(cudaMemcpy(a.data(), got, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b.data(), want, count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const auto mismatch = std::mismatch(a.begin(), a.end(), b.begin());
  if (mismatch.first == a.end()) {
    std::printf("  ok   %-48s bit-identical\n", what.c_str());
  } else {
    std::printf("  FAIL %-48s word %zu: %08x != %08x\n", what.c_str(), size_t(mismatch.first - a.begin()),
                unsigned(*mismatch.first), unsigned(*mismatch.second));
    ++g_fail;
  }
}

// The reference and both device matrices use exactly the same weight values.
struct ExactTestMatrix {
  std::vector<float> values;
  flyx_tensor_view view;
  float* full;
  bf16* compact;

  ExactTestMatrix(int N, int K, double scale) {
    Vec w = randv(size_t(N) * K, scale);
    round_bf16(w);
    values.assign(w.begin(), w.end());
    view = view_f32(values, N, K);
    full = up(w);
    compact = up_bf16(w);
  }
  ~ExactTestMatrix() {
    cudaFree(full);
    cudaFree(compact);
  }
  ExactTestMatrix(const ExactTestMatrix&) = delete;
  ExactTestMatrix& operator=(const ExactTestMatrix&) = delete;
};

void test_compact_loop_control() {
  const int rows = 16, H = 259, rank = 35;
  seed("inkling2_compact_loop_control", rows, H, rank);
  const auto cur = randf(size_t(rows) * H, 1.0), msg = randf(cur.size(), 0.5);
  ExactTestMatrix wd(rank, 2 * H, 0.05), wg(H, rank, 0.1), wf(H, rank, 0.1);
  const Vec cur32(cur.begin(), cur.end()), msg32(msg.begin(), msg.end());
  Vec want(cur.size());
  ops::loop_control(cur32.data(), msg32.data(), rows, H, rank, &wd.view, &wg.view, &wf.view, want.data());

  float* d_cur = up_f(cur);
  float* d_msg = up_f(msg);
  float* out_full = dev<float>(cur.size());
  float* out_compact = dev<float>(kLoopGates * cur.size());
  const size_t gate_size = rows * loop_row_floats(rank);
  float* gate_full = dev<float>(gate_size);
  float* gate_compact = dev<float>(kLoopGates * gate_size);
  LoopGate full{d_cur, d_msg, {wd.full}, {wg.full}, {wf.full}, out_full};
  LoopGate compact{d_cur, d_msg, {nullptr, wd.compact}, {nullptr, wg.compact}, {nullptr, wf.compact}, out_compact};
  LoopGate batch = compact;
  batch.out += cur.size();
  loop_control(&batch, 1, rows, H, rank, gate_compact + gate_size, 0);
  for (int R : {1, 2, 3, 4, 5, 8, 16}) {
    const std::string at = " R=" + std::to_string(R);
    loop_control(&full, 1, R, H, rank, gate_full, 0);
    CK(cudaMemset(gate_compact, 0xff, R * loop_row_floats(rank) * sizeof(float)));
    CK(cudaMemset(out_compact, 0xff, size_t(R) * H * sizeof(float)));
    loop_control(&compact, 1, R, H, rank, gate_compact, 0);
    expect_device_bits("compact loop gate = f32" + at, gate_compact, gate_full, R * loop_row_floats(rank));
    expect_device_bits("compact loop apply = f32" + at, out_compact, out_full, size_t(R) * H);
    expect_device_bits("compact loop_control = 16-row prefix" + at, out_compact, batch.out, size_t(R) * H);
    expect("compact loop_control" + at, down(out_compact, size_t(R) * H), Vec(want.begin(), want.begin() + R * H),
           2e-5);
  }

  // Each controller keeps a different one of its three matrices in float32 in the same launch.
  LoopGate mixed[kLoopGates] = {compact, compact, compact};
  mixed[0].down = full.down;
  mixed[1].gain = full.gain;
  mixed[2].forget = full.forget;
  for (int c = 0; c < kLoopGates; ++c) mixed[c].out = out_compact + c * cur.size();
  CK(cudaMemset(gate_compact, 0xff, kLoopGates * gate_size * sizeof(float)));
  CK(cudaMemset(out_compact, 0xff, kLoopGates * cur.size() * sizeof(float)));
  loop_control(mixed, kLoopGates, rows, H, rank, gate_compact, 0);
  for (int c = 0; c < kLoopGates; ++c) {
    const std::string at = " controller " + std::to_string(c);
    expect_device_bits("mixed loop gate = f32" + at, gate_compact + c * gate_size, gate_full, gate_size);
    expect_device_bits("mixed loop apply = f32" + at, mixed[c].out, out_full, cur.size());
  }
  cudaFree(d_cur);
  cudaFree(d_msg);
  cudaFree(out_full);
  cudaFree(out_compact);
  cudaFree(gate_full);
  cudaFree(gate_compact);
}

void test_compact_linear_f32() {
  const int rows = 16, N = 37, K = 101;
  seed("inkling2_compact_linear_f32", rows, N, K);
  const auto x = randf(size_t(rows) * K, 1.0);
  const Vec x32(x.begin(), x.end());
  ExactTestMatrix w(N, K, 0.2);
  Vec want(size_t(rows) * N);
  ops::linear(x32.data(), rows, K, &w.view, N, want.data());
  float* dx = up_f(x);
  float* full = dev<float>(want.size());
  float* compact = dev<float>(want.size());
  float* batch = dev<float>(want.size());
  linear_f32(dx, K, {nullptr, w.compact}, N, K, batch, N, rows, false, 0);
  for (int R : {1, 2, 3, 4, 5, 8, 16}) {
    const std::string at = " R=" + std::to_string(R);
    const size_t count = size_t(R) * N;
    linear_f32(dx, K, {w.full}, N, K, full, N, R, false, 0);
    linear_f32(dx, K, {nullptr, w.compact}, N, K, compact, N, R, false, 0);
    expect_device_bits("compact linear_f32 = f32" + at, compact, full, count);
    expect_device_bits("compact linear_f32 = 16-row prefix" + at, compact, batch, count);
    expect("compact linear_f32" + at, down(compact, count), Vec(want.begin(), want.begin() + R * N), 2e-5);
    linear_f32(dx, K, {w.full}, N, K, full, N, R, true, 0);
    linear_f32(dx, K, {nullptr, w.compact}, N, K, compact, N, R, true, 0);
    expect_device_bits("compact linear_f32 accumulate = f32" + at, compact, full, count);
  }
  cudaFree(dx);
  cudaFree(full);
  cudaFree(compact);
  cudaFree(batch);
}

void test_one_inexact_weight() {
  // Signed zero, a subnormal, infinities and NaN payloads are exact when their low halves are zero.
  const uint32_t words[] = {0x80000000u, 0x00010000u, 0x7f800000u, 0xff800000u, 0x7f810000u, 0xffc30000u};
  bool has_nan = false;
  const bool exact_words = all_exact_bf16(words, sizeof(words) / sizeof(words[0]), &has_nan) && has_nan;
  std::printf("  %-4s %-48s\n", exact_words ? "ok" : "FAIL", "all_exact_bf16 accepts exact special values");
  if (!exact_words) ++g_fail;

  const int K = 65;
  std::vector<float> w(K, 0.f);
  w.front() = -0.f;
  w.back() = 1.f;
  const bool exact = all_exact_bf16(w.data(), w.size(), &has_nan) && !has_nan;
  w.back() = std::nextafter(1.f, 2.f);  // only bit zero differs from the exact tensor
  const bool compact = all_exact_bf16(w.data(), w.size());
  const bool ok = exact && !compact;
  std::printf("  %-4s %-48s\n", ok ? "ok" : "FAIL", "all_exact_bf16 detects one low bit in last weight");
  if (!ok) ++g_fail;

  Vec x(K, 0.);
  x.back() = 1.;
  float* dx = up(x);
  float* dw = up_f(w);
  bf16* db = up_bf16(Vec(w.begin(), w.end()));
  float* out = dev<float>(1);
  const RoundWeight selected = compact ? RoundWeight{nullptr, db} : RoundWeight{dw};
  linear_f32(dx, K, selected, 1, K, out, 1, 1, false, 0);
  expect_device_bits("linear_f32 keeps the single non-bf16 weight", out, dw + K - 1, 1);
  cudaFree(dx);
  cudaFree(dw);
  cudaFree(db);
  cudaFree(out);
}

}  // namespace

// loop_control against ops::loop_control on float32 weights the reference reads as they are: one controller and
// three in one launch, the real model's shape and sizes that fill no warp or block, a row alone and among others,
// out beside cur and on it.
void test_inkling2_loop_control() {
  std::printf("inkling2 loop_control\n");
  struct Shape {
    int R, H, rank;
  };
  for (const Shape sh : {Shape{9, 256, 64}, Shape{3, 2304, 64}, Shape{20, 200, 12}, Shape{1, 2304, 64}}) {
    const int R = sh.R, H = sh.H, rank = sh.rank, n = kLoopGates;
    seed("inkling2_loop_control", R, H, rank);
    const std::string at = " R=" + std::to_string(R) + " H=" + std::to_string(H) + " rank=" + std::to_string(rank);
    const size_t RH = size_t(R) * H;
    std::vector<Vec> cur(n), msg(n), want(n);
    std::vector<float*> d_cur(n), d_msg(n), d_down(n), d_gain(n), d_forget(n);
    float* d_out = dev<float>(size_t(n) * RH);
    float* d_gate = dev<float>(size_t(n) * R * loop_row_floats(rank));
    LoopGate gates[kLoopGates];
    for (int c = 0; c < n; ++c) {
      cur[size_t(c)] = randv(RH, 1.0 + c);
      msg[size_t(c)] = randv(RH, 0.5 + c);
      const std::vector<float> w_down = randf(size_t(rank) * 2 * H, 0.05);
      const std::vector<float> w_gain = randf(size_t(H) * rank, 0.1), w_forget = randf(size_t(H) * rank, 0.1);
      const flyx_tensor_view vd = view_f32(w_down, rank, 2 * H), vg = view_f32(w_gain, H, rank);
      const flyx_tensor_view vf = view_f32(w_forget, H, rank);
      // the kernel's inputs are float32: the reference from the same values
      Vec cur32(RH), msg32(RH);
      for (size_t i = 0; i < RH; ++i) {
        cur32[i] = double(float(cur[size_t(c)][i]));
        msg32[i] = double(float(msg[size_t(c)][i]));
      }
      want[size_t(c)].assign(RH, 0.0);
      ops::loop_control(cur32.data(), msg32.data(), R, H, rank, &vd, &vg, &vf, want[size_t(c)].data());
      d_cur[size_t(c)] = up(cur[size_t(c)]);
      d_msg[size_t(c)] = up(msg[size_t(c)]);
      d_down[size_t(c)] = up_f(w_down);
      d_gain[size_t(c)] = up_f(w_gain);
      d_forget[size_t(c)] = up_f(w_forget);
      gates[c] = {d_cur[size_t(c)], d_msg[size_t(c)], {d_down[size_t(c)]}, {d_gain[size_t(c)]}, {d_forget[size_t(c)]},
                  d_out + size_t(c) * RH};
    }

    loop_control(gates, n, R, H, rank, d_gate, 0);
    std::vector<Vec> got(n);
    for (int c = 0; c < n; ++c) {
      got[size_t(c)] = down(d_out + size_t(c) * RH, RH);
      expect("loop_control, three at once: controller " + std::to_string(c) + at, got[size_t(c)], want[size_t(c)],
             2e-5);
    }

    // One controller a launch: the bits it has among three.
    float* d_one = dev<float>(RH);
    for (int c = 0; c < n; ++c) {
      LoopGate one = gates[c];
      one.out = d_one;
      loop_control(&one, 1, R, H, rank, d_gate, 0);
      expect_bits("loop_control alone = among three: controller " + std::to_string(c) + at, down(d_one, RH),
                  got[size_t(c)]);
    }

    // A row alone: the bits it has among the others (rule 4: every row count).
    const int rr = R / 2;
    {
      LoopGate one = gates[1];
      one.cur += size_t(rr) * H;
      one.msg += size_t(rr) * H;
      one.out = d_one;
      loop_control(&one, 1, 1, H, rank, d_gate, 0);
      const Vec& all = got[1];
      expect_bits("loop_control a row alone = among " + std::to_string(R) + at, down(d_one, size_t(H)),
                  Vec(all.begin() + ptrdiff_t(rr) * H, all.begin() + ptrdiff_t(rr + 1) * H));
    }

    // out on its own cur (the three at once, each in place).
    for (int c = 0; c < n; ++c) gates[c].out = d_cur[size_t(c)];
    loop_control(gates, n, R, H, rank, d_gate, 0);
    for (int c = 0; c < n; ++c)
      expect_bits("loop_control in place: controller " + std::to_string(c) + at, down(d_cur[size_t(c)], RH),
                  got[size_t(c)]);

    for (int c = 0; c < n; ++c) {
      cudaFree(d_cur[size_t(c)]);
      cudaFree(d_msg[size_t(c)]);
      cudaFree(d_down[size_t(c)]);
      cudaFree(d_gain[size_t(c)]);
      cudaFree(d_forget[size_t(c)]);
    }
    cudaFree(d_out);
    cudaFree(d_gate);
    cudaFree(d_one);
  }
  test_compact_loop_control();
}

// linear_f32 against ops::linear on a float32 matrix: a router's and a sink's shapes of the real model, sizes that
// fill no warp, accumulation, and a row alone against the same row among others.
void test_inkling2_linear_f32() {
  std::printf("inkling2 linear_f32\n");
  struct Shape {
    int R, N, K;
  };
  for (const Shape sh : {Shape{1, 128, 2304}, Shape{8, 128, 2304}, Shape{20, 24, 2304}, Shape{5, 37, 100},
                         Shape{33, 4, 64}}) {
    const int R = sh.R, N = sh.N, K = sh.K;
    seed("inkling2_linear_f32", R, N, K);
    const std::string at = " " + std::to_string(R) + "x" + std::to_string(N) + "x" + std::to_string(K);
    const Vec x = randv(size_t(R) * K, 1.0);
    const std::vector<float> w = randf(size_t(N) * K, 0.2);
    const flyx_tensor_view vw = view_f32(w, N, K);
    Vec x32(x.size());
    for (size_t i = 0; i < x.size(); ++i) x32[i] = double(float(x[i]));
    Vec want(size_t(R) * N, 0.0);
    ops::linear(x32.data(), R, K, &vw, N, want.data());

    float* dx = up(x);
    float* dw = up_f(w);
    const int ldy = N + 3;  // rows apart by more than their length
    float* dy = dev<float>(size_t(R) * ldy);
    linear_f32(dx, K, {dw}, N, K, dy, ldy, R, false, 0);
    const Vec all = down(dy, size_t(R) * ldy);
    Vec got(size_t(R) * N);
    for (int r = 0; r < R; ++r) std::copy_n(all.begin() + ptrdiff_t(r) * ldy, N, got.begin() + ptrdiff_t(r) * N);
    expect("linear_f32" + at, got, want, 2e-5);

    linear_f32(dx, K, {dw}, N, K, dy, ldy, R, true, 0);  // y = y + y
    const Vec twice = down(dy, size_t(R) * ldy);
    Vec got2(size_t(R) * N), want2(size_t(R) * N);
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < N; ++c) {
        got2[size_t(r) * N + c] = twice[size_t(r) * ldy + c];
        want2[size_t(r) * N + c] = double(float(got[size_t(r) * N + c]) + float(got[size_t(r) * N + c]));
      }
    expect_bits("linear_f32 accumulate" + at, got2, want2);

    const int rr = R / 2;
    float* d1 = dev<float>(size_t(N));
    linear_f32(dx + size_t(rr) * K, K, {dw}, N, K, d1, N, 1, false, 0);
    expect_bits("linear_f32 a row alone = among " + std::to_string(R) + at, down(d1, size_t(N)),
                Vec(got.begin() + ptrdiff_t(rr) * N, got.begin() + ptrdiff_t(rr + 1) * N));
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dy);
    cudaFree(d1);
  }
  test_compact_linear_f32();
  test_one_inexact_weight();
}

void test_inkling2_shortconv() {
  std::printf("inkling2 shortconv_pass + fold\n");
  for (int with_dyn : {0, 1}) {
    const int conv = 4, hist = conv - 1, nh = with_dyn ? 4 : 0;
    const int C = 128, ch_off = 64, ch_total = 256, R = 37;
    seed("inkling2_shortconv", with_dyn, C, R);
    Vec x = randv(size_t(R) * C, 1.0);
    Vec w = randv(size_t(C) * conv, 0.3);
    Vec dyn = with_dyn ? randv(size_t(R) * nh * conv, 0.2) : Vec{};
    Vec st0 = randv(size_t(hist) * ch_total, 0.5);
    Vec base_out = randv(size_t(R) * C, 0.7);

    // Reference on slice [ch_off, ch_off + C).
    Vec sub_st(size_t(hist) * C);
    for (int k = 0; k < hist; ++k)
      std::copy_n(st0.data() + size_t(k) * ch_total + ch_off, C, sub_st.data() + size_t(k) * C);
    Vec want_out(size_t(R) * C, 0.0);
    Vec sub_st_ref = sub_st;
    ops::shortconv(x.data(), w.data(), with_dyn ? dyn.data() : nullptr, sub_st_ref.data(), R, C, conv, nh,
                   want_out.data());

    // Prefill in-place update (multi-tile R = 37, in_sel == next_sel == kSlotCur).
    DevParams* P = params(0, 0);
    const size_t w_stride = size_t(hist) * ch_total;
    float* d_win = dev<float>(FLYX_ROUND_SLOTS * w_stride);
    std::vector<float> h_st0(st0.begin(), st0.end());
    CK(cudaMemcpy(d_win, h_st0.data(), w_stride * sizeof(float), cudaMemcpyHostToDevice));
    float* d_x = up(x);
    float* d_w = up(w);
    float* d_dyn = with_dyn ? up(dyn) : nullptr;
    float* d_out = up(base_out);

    shortconv_pass(d_x, C, d_w, d_dyn, d_win, w_stride, kSlotCur, kSlotCur, kSkipNever, nullptr, 0, kSlotCur, ch_off,
                   ch_total, C, conv, nh, P, R, d_out, C, 0, false);
    Vec got_out = down(d_out, size_t(R) * C);
    expect(std::string("shortconv_pass dyn=") + std::to_string(with_dyn), got_out, want_out, 1e-5);

    // Verify updated window slice matches ref state.
    Vec got_win = down(d_win, w_stride);
    Vec got_sub(size_t(hist) * C);
    for (int k = 0; k < hist; ++k)
      std::copy_n(got_win.data() + size_t(k) * ch_total + ch_off, C, got_sub.data() + size_t(k) * C);
    expect(std::string("shortconv_pass state dyn=") + std::to_string(with_dyn), got_sub, sub_st_ref, 1e-6);

    // Verify accumulate=true fuses residual add cleanly.
    CK(cudaMemcpy(d_win, h_st0.data(), w_stride * sizeof(float), cudaMemcpyHostToDevice));
    float* d_acc = up(base_out);
    shortconv_pass(d_x, C, d_w, d_dyn, d_win, w_stride, kSlotCur, kSlotCur, kSkipNever, nullptr, 0, kSlotCur, ch_off,
                   ch_total, C, conv, nh, P, R, d_acc, C, 0, true);
    Vec got_acc = down(d_acc, size_t(R) * C);
    Vec want_acc(size_t(R) * C);
    for (size_t i = 0; i < want_acc.size(); ++i) want_acc[i] = base_out[i] + want_out[i];
    expect(std::string("shortconv_pass accumulate dyn=") + std::to_string(with_dyn), got_acc, want_acc, 1e-5);

    // Verify round replay + shortconv_fold matches direct prefill on accepted prefix (bit-identical, class P).
    const int R_round = 8, accepted = 5;
    const size_t r_stride = size_t(R_round) * ch_total;
    float* d_rep = dev<float>(kGdnReplaySlots * r_stride);
    CK(cudaMemcpy(d_win, h_st0.data(), w_stride * sizeof(float), cudaMemcpyHostToDevice));
    shortconv_pass(d_x, C, d_w, d_dyn, d_win, w_stride, kSlotCur, -1, kSkipNever, d_rep, r_stride, kSlotCur, ch_off,
                   ch_total, C, conv, nh, P, R_round, d_out, C, 0, false);
    ShortConvLayer layer_h{d_win, d_rep};
    ShortConvLayer* d_layer = dev<ShortConvLayer>(1);
    CK(cudaMemcpy(d_layer, &layer_h, sizeof(layer_h), cudaMemcpyHostToDevice));
    shortconv_fold(d_layer, 1, w_stride, r_stride, kSlotExplicit + 0, kSlotExplicit + 1, kSlotExplicit + 0, ch_total,
                   conv, P, accepted, nullptr, false, 0);
    Vec folded_win = down(d_win + w_stride, w_stride);
    Vec folded_sub(size_t(hist) * C);
    for (int k = 0; k < hist; ++k)
      std::copy_n(folded_win.data() + size_t(k) * ch_total + ch_off, C, folded_sub.data() + size_t(k) * C);

    // Direct run of `accepted` rows.
    CK(cudaMemcpy(d_win, h_st0.data(), w_stride * sizeof(float), cudaMemcpyHostToDevice));
    shortconv_pass(d_x, C, d_w, d_dyn, d_win, w_stride, kSlotCur, kSlotCur, kSkipNever, nullptr, 0, kSlotCur, ch_off,
                   ch_total, C, conv, nh, P, accepted, d_out, C, 0, false);
    Vec direct_win = down(d_win, w_stride);
    Vec direct_sub(size_t(hist) * C);
    for (int k = 0; k < hist; ++k)
      std::copy_n(direct_win.data() + size_t(k) * ch_total + ch_off, C, direct_sub.data() + size_t(k) * C);
    expect_bits(std::string("shortconv_fold vs direct dyn=") + std::to_string(with_dyn), folded_sub, direct_sub);

    cudaFree(P);
    cudaFree(d_win);
    cudaFree(d_x);
    cudaFree(d_w);
    if (d_dyn) cudaFree(d_dyn);
    cudaFree(d_out);
    cudaFree(d_acc);
    cudaFree(d_rep);
    cudaFree(d_layer);
  }
}

void test_inkling2_rel_attn() {
  std::printf("inkling2 rel_attn_prep + rel_attention\n");
  struct Case {
    const char* name;
    int nh, nkv, hd, d_rel, rel_extent, window, ring, ctx, rows;
    bool fp8;
  };
  const Case cases[] = {
      {"swa ring f16 hd128 R=9", 8, 2, 128, 16, 16, 16, 64, 90, 9, false},
      {"swa ring f16 hd128 R=37", 12, 4, 128, 16, 32, 32, 96, 110, 37, false},
      {"global f16 hd128 R=37", 8, 2, 128, 16, 128, 0, 0, 50, 37, false},
      {"global fp8 hd128 R=37", 12, 4, 128, 16, 128, 0, 0, 50, 37, true},
      {"swa fp8 hd64 R=5", 8, 4, 64, 8, 16, 16, 64, 70, 5, true},
      {"global f16 hd32 R=8", 4, 2, 32, 8, 16, 0, 0, 40, 8, false},
      {"global fp8 hd32 R=1", 4, 2, 32, 8, 32, 0, 0, 40, 1, true},
  };
  for (const Case& c : cases) {
    seed("inkling2_rel_attn", c.hd, c.window, c.rows, int(c.fp8));
    const int total = c.ctx + c.rows;
    const int q_dim = c.nh * c.hd, kv_dim = c.nkv * c.hd;
    const float qk_clip = 6.f, rel_clip = 4.f, v_eps = 1e-6f;
    const int log_floor = 32;
    const float log_alpha = 0.25f;
    const bool is_swa = c.window > 0;

    Vec K_all = randv(size_t(total) * kv_dim, 0.5);
    Vec V_all = randv(size_t(total) * kv_dim, 0.5);
    Vec q_in = randv(size_t(c.rows) * q_dim, 1.0);
    Vec r_in = randv(size_t(c.rows) * c.nh * c.d_rel, 1.0);
    Vec gate = randv(size_t(c.rows) * q_dim, 1.0);
    Vec sink = randv(size_t(c.rows) * c.nh, 0.5);
    Vec q_cw = randv(size_t(q_dim), 0.5);
    Vec r_cw = randv(size_t(c.nh) * c.d_rel, 0.5);
    Vec rel_bias = randv(size_t(c.rel_extent) * c.d_rel, 0.3);

    // Prepare history K/V in reference (rounded to f16 or fp8 as stored) and upload to device KV (with ring!).
    const int kv_slots = c.ring > 0 ? c.ring : total;
    void *d_k = nullptr, *d_v = nullptr;
    float *d_ks = nullptr, *d_vs = nullptr;
    if (c.fp8) {
      std::vector<uint8_t> hk(size_t(kv_slots) * kv_dim, 0), hv(size_t(kv_slots) * kv_dim, 0);
      std::vector<float> hks(size_t(kv_slots) * c.nkv, 0.f), hvs(size_t(kv_slots) * c.nkv, 0.f);
      for (int p = 0; p < c.ctx; ++p) {
        const int slot = c.ring > 0 ? (p % c.ring) : p;
        for (int kh = 0; kh < c.nkv; ++kh) {
          Vec kvec(c.hd), vvec(c.hd);
          for (int d = 0; d < c.hd; ++d) {
            kvec[d] = K_all[(size_t(p) * c.nkv + kh) * c.hd + d];
            vvec[d] = V_all[(size_t(p) * c.nkv + kh) * c.hd + d];
          }
          const float ksc = float(round_fp8_e4m3_vec(kvec));
          const float vsc = float(round_fp8_e4m3_vec(vvec));
          hks[size_t(slot) * c.nkv + kh] = ksc;
          hvs[size_t(slot) * c.nkv + kh] = vsc;
          for (int d = 0; d < c.hd; ++d) {
            K_all[(size_t(p) * c.nkv + kh) * c.hd + d] = kvec[d];
            V_all[(size_t(p) * c.nkv + kh) * c.hd + d] = vvec[d];
            hk[(size_t(slot) * c.nkv + kh) * c.hd + d] =
                ksc > 0.f ? __nv_cvt_float_to_fp8(float(kvec[d]) / ksc, __NV_SATFINITE, __NV_E4M3) : 0;
            hv[(size_t(slot) * c.nkv + kh) * c.hd + d] =
                vsc > 0.f ? __nv_cvt_float_to_fp8(float(vvec[d]) / vsc, __NV_SATFINITE, __NV_E4M3) : 0;
          }
        }
      }
      d_k = dev<uint8_t>(hk.size());
      d_v = dev<uint8_t>(hv.size());
      d_ks = dev<float>(hks.size());
      d_vs = dev<float>(hvs.size());
      CK(cudaMemcpy(d_k, hk.data(), hk.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_v, hv.data(), hv.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_ks, hks.data(), hks.size() * 4, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_vs, hvs.data(), hvs.size() * 4, cudaMemcpyHostToDevice));
    } else {
      std::vector<uint16_t> hk(size_t(kv_slots) * kv_dim, 0), hv(size_t(kv_slots) * kv_dim, 0);
      for (int p = 0; p < c.ctx; ++p) {
        const int slot = c.ring > 0 ? (p % c.ring) : p;
        for (int i = 0; i < kv_dim; ++i) {
          const uint16_t ku = to_f16(K_all[size_t(p) * kv_dim + i]);
          const uint16_t vu = to_f16(V_all[size_t(p) * kv_dim + i]);
          K_all[size_t(p) * kv_dim + i] = from_f16(ku);
          V_all[size_t(p) * kv_dim + i] = from_f16(vu);
          hk[size_t(slot) * kv_dim + i] = ku;
          hv[size_t(slot) * kv_dim + i] = vu;
        }
      }
      d_k = dev<kv16>(hk.size());
      d_v = dev<kv16>(hv.size());
      CK(cudaMemcpy(d_k, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_v, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));
    }
    KvRef kv{d_k, d_v, d_ks, d_vs};
    kv.ring = c.ring;

    // Extract raw K/V for the new rows [ctx, ctx + rows).
    Vec k_new(K_all.begin() + ptrdiff_t(c.ctx) * kv_dim, K_all.end());
    Vec v_new(V_all.begin() + ptrdiff_t(c.ctx) * kv_dim, V_all.end());

    // Reference prep + attend_rel + gate.
    Vec q_ref(size_t(c.rows) * q_dim), r_ref(size_t(c.rows) * c.nh * c.d_rel);
    for (int r = 0; r < c.rows; ++r) {
      const int pos = c.ctx + r;
      const double tau = is_swa ? 1.0 : ops::log_pos_scale(pos, log_floor, log_alpha);
      for (int hh = 0; hh < c.nh; ++hh) {
        ops::l2_clip(&q_in[(size_t(r) * c.nh + hh) * c.hd], &q_cw[size_t(hh) * c.hd], c.hd, qk_clip,
                     &q_ref[(size_t(r) * c.nh + hh) * c.hd]);
        for (int d = 0; d < c.hd; ++d) q_ref[(size_t(r) * c.nh + hh) * c.hd + d] *= tau;
        ops::l2_clip(&r_in[(size_t(r) * c.nh + hh) * c.d_rel], &r_cw[size_t(hh) * c.d_rel], c.d_rel, rel_clip,
                     &r_ref[(size_t(r) * c.nh + hh) * c.d_rel]);
        for (int d = 0; d < c.d_rel; ++d) r_ref[(size_t(r) * c.nh + hh) * c.d_rel + d] *= tau;
      }
      for (int kh = 0; kh < c.nkv; ++kh) {
        Vec ko(c.hd), vo(c.hd);
        ops::l2_clip(&k_new[(size_t(r) * c.nkv + kh) * c.hd], nullptr, c.hd, qk_clip, ko.data());
        ops::rmsnorm(&v_new[(size_t(r) * c.nkv + kh) * c.hd], nullptr, c.hd, v_eps, 0.0, vo.data());
        if (c.fp8) {
          round_fp8_e4m3_vec(ko);
          round_fp8_e4m3_vec(vo);
        } else {
          round_f16(ko);
          round_f16(vo);
        }
        std::copy(ko.begin(), ko.end(), &K_all[(size_t(pos) * c.nkv + kh) * c.hd]);
        std::copy(vo.begin(), vo.end(), &V_all[(size_t(pos) * c.nkv + kh) * c.hd]);
      }
    }
    Vec want_o(size_t(c.rows) * q_dim, 0.0);
    const double scale = 1.0 / std::sqrt(double(c.hd));
    for (int r = 0; r < c.rows; ++r) {
      for (int hh = 0; hh < c.nh; ++hh) {
        ops::attend_rel(&q_ref[(size_t(r) * c.nh + hh) * c.hd], &r_ref[(size_t(r) * c.nh + hh) * c.d_rel], K_all.data(),
                        V_all.data(), rel_bias.data(), sink[size_t(r) * c.nh + hh], c.ctx + r, c.window, c.nkv, c.hd,
                        c.d_rel, c.rel_extent, hh / (c.nh / c.nkv), scale, &want_o[(size_t(r) * c.nh + hh) * c.hd]);
      }
    }
    for (size_t i = 0; i < want_o.size(); ++i) want_o[i] *= ops::sigmoid(gate[i]);

    // CUDA prep + attention.
    DevParams* P = params(c.ctx, 0);
    float* d_qin = up(q_in);
    float* d_kin = up(k_new);
    float* d_vin = up(v_new);
    float* d_rin = up(r_in);
    float* d_qcw = up(q_cw);
    float* d_rcw = up(r_cw);
    float* d_qout = dev<float>(size_t(c.rows) * q_dim);
    float* d_rout = dev<float>(size_t(c.rows) * c.nh * c.d_rel);
    float* d_rb = up(rel_bias);
    float* d_sink = up(sink);
    float* d_gate = up(gate);
    float* d_out = dev<float>(size_t(c.rows) * q_dim);
    const int splits = 16;
    float* d_part = dev<float>(attention_partial_floats(c.rows, c.nh, c.hd, splits));

    rel_attn_prep(d_qin, q_dim, d_kin, kv_dim, d_vin, kv_dim, d_rin, c.nh * c.d_rel, d_qcw, d_rcw, qk_clip, rel_clip,
                  v_eps, log_floor, log_alpha, c.nh, c.nkv, c.hd, c.d_rel, c.rows, P, kPosPosition, is_swa, d_qout,
                  d_rout, kv, 0);
    rel_attention(d_qout, d_rout, kv, d_rb, d_sink, c.nh, d_gate, q_dim, c.nh, c.nkv, c.hd, c.d_rel, c.rel_extent,
                  c.window, c.rows, P, d_out, q_dim, d_part, splits, TL);
    Vec got_o = down(d_out, size_t(c.rows) * q_dim);
    const bool is_flash = c.rows > FLYX_MAX_ROWS;
    expect(std::string("rel_attn ") + c.name, got_o, want_o, is_flash ? (c.fp8 ? 3e-3 : 2e-3) : (c.fp8 ? 5e-4 : 3e-5));

    // When rows > FLYX_MAX_ROWS, also verify exact float32 kAttnSplit path (GQA + per-head) and R=1 bit-identity.
    if (is_flash) {
      KernelOptions kopt{};
      kopt.attention = kAttnSplit;
      Launch Ls = TL;
      Ls.opt = &kopt;
      float* d_os = dev<float>(size_t(c.rows) * q_dim);
      rel_attention(d_qout, d_rout, kv, d_rb, d_sink, c.nh, d_gate, q_dim, c.nh, c.nkv, c.hd, c.d_rel, c.rel_extent,
                    c.window, c.rows, P, d_os, q_dim, d_part, splits, Ls);
      Vec got_os = down(d_os, size_t(c.rows) * q_dim);
      expect(std::string("rel_attn split ") + c.name, got_os, want_o, c.fp8 ? 5e-4 : 3e-5);
      for (int rr : {0, c.rows - 1}) {
        DevParams* P1 = params(c.ctx + rr, 0);
        float* d_o1 = dev<float>(size_t(q_dim));
        rel_attention(d_qout + size_t(rr) * q_dim, d_rout + size_t(rr) * c.nh * c.d_rel, kv, d_rb,
                      d_sink + size_t(rr) * c.nh, c.nh, d_gate + size_t(rr) * q_dim, q_dim, c.nh, c.nkv, c.hd, c.d_rel,
                      c.rel_extent, c.window, 1, P1, d_o1, q_dim, d_part, splits, Ls);
        Vec got_1 = down(d_o1, size_t(q_dim));
        Vec got_rr(got_os.begin() + ptrdiff_t(rr) * q_dim, got_os.begin() + ptrdiff_t(rr + 1) * q_dim);
        expect_bits(std::string("rel_attn split R=1 vs multi r=") + std::to_string(rr) + " " + c.name, got_1, got_rr);
        cudaFree(P1);
        cudaFree(d_o1);
      }
      cudaFree(d_os);
    } else if (c.rows > 1) {
      // Round-sized rows: verify Gate 4 row-count bit-identity (GQA R=1 vs multi) and per-head fallback parity.
      for (int rr : {0, c.rows - 1}) {
        DevParams* P1 = params(c.ctx + rr, 0);
        float* d_o1 = dev<float>(size_t(q_dim));
        rel_attention(d_qout + size_t(rr) * q_dim, d_rout + size_t(rr) * c.nh * c.d_rel, kv, d_rb,
                      d_sink + size_t(rr) * c.nh, c.nh, d_gate + size_t(rr) * q_dim, q_dim, c.nh, c.nkv, c.hd, c.d_rel,
                      c.rel_extent, c.window, 1, P1, d_o1, q_dim, d_part, splits, TL);
        Vec got_1 = down(d_o1, size_t(q_dim));
        Vec got_rr(got_o.begin() + ptrdiff_t(rr) * q_dim, got_o.begin() + ptrdiff_t(rr + 1) * q_dim);
        expect_bits(std::string("rel_attn R=1 vs multi r=") + std::to_string(rr) + " " + c.name, got_1, got_rr);
        cudaFree(P1);
        cudaFree(d_o1);
      }
    }

    // Windowed prefill rel_attention (RelWindow) across 3 windows must match float64 ref and single-pass output.
    if (c.window == 0 && total >= 24) {
      float* d_wstate = dev<float>(size_t(c.rows) * c.nh * kFlashStateFloats(c.hd));
      float* d_wout = dev<float>(size_t(c.rows) * q_dim);
      const int w1 = total / 3, w2 = (2 * total) / 3;
      const int bounds[4] = {0, w1, w2, total};
      for (int wi = 0; wi < 3; ++wi) {
        RelWindow rw{bounds[wi], bounds[wi + 1], d_wstate, wi == 0, wi == 2};
        rel_attention(d_qout, d_rout, kv, d_rb, d_sink, c.nh, d_gate, q_dim, c.nh, c.nkv, c.hd, c.d_rel, c.rel_extent,
                      c.window, c.rows, P, d_wout, q_dim, d_part, splits, TL, 0, rw);
      }
      Vec got_w = down(d_wout, size_t(c.rows) * q_dim);
      expect(std::string("rel_attn windowed(3) vs ref ") + c.name, got_w, want_o,
             is_flash ? (c.fp8 ? 3e-3 : 2e-3) : (c.fp8 ? 5e-4 : 3e-5));
      expect(std::string("rel_attn windowed(3) vs single ") + c.name, got_w, got_o, is_flash ? 5e-4 : 2e-5);
      cudaFree(d_wstate);
      cudaFree(d_wout);
    }

    // Fused split_out -> gemm_presplit == unfused -> gemm (Gate 2) for round-sized rows.
    if (c.rows <= 16 && q_dim % 64 == 0) {
      const int H_out = 128;
      Vec wo = randv(size_t(H_out) * q_dim, 0.2);
      round_bf16(wo);
      bf16* d_wo = up_bf16(wo);
      float* d_y0 = dev<float>(size_t(c.rows) * H_out);
      float* d_y1 = dev<float>(size_t(c.rows) * H_out);
      gemm(d_out, q_dim, d_wo, H_out, q_dim, d_y0, H_out, c.rows, false, TL);
      rel_attention(d_qout, d_rout, kv, d_rb, d_sink, c.nh, d_gate, q_dim, c.nh, c.nkv, c.hd, c.d_rel, c.rel_extent,
                    c.window, c.rows, P, nullptr, q_dim, d_part, splits, TL);
      gemm_presplit(d_wo, H_out, q_dim, d_y1, H_out, c.rows, false, TL);
      expect_bits(std::string("rel_attn fused==unfused ") + c.name, down(d_y1, size_t(c.rows) * H_out),
                  down(d_y0, size_t(c.rows) * H_out));
      cudaFree(d_wo);
      cudaFree(d_y0);
      cudaFree(d_y1);
    }

    cudaFree(P);
    cudaFree(d_k);
    cudaFree(d_v);
    if (d_ks) cudaFree(d_ks);
    if (d_vs) cudaFree(d_vs);
    cudaFree(d_qin);
    cudaFree(d_kin);
    cudaFree(d_vin);
    cudaFree(d_rin);
    cudaFree(d_qcw);
    cudaFree(d_rcw);
    cudaFree(d_qout);
    cudaFree(d_rout);
    cudaFree(d_rb);
    cudaFree(d_sink);
    cudaFree(d_gate);
    cudaFree(d_out);
    cudaFree(d_part);
  }
}

void test_inkling2_rel_strata() {
  std::printf("inkling2 rel_bias_table + attention_strata (RelStrata)\n");
  struct Case {
    int hd, nh, nkv, d_rel, rel_extent, ctx, rows, bits;
    bool fp8;
  };
  const Case cases[] = {
      {128, 12, 4, 16, 64, 260, 8, 2, true},   // Inkling-2 shape (hd=128, G=3), 2-bit FP8 + hot cache
      {128, 12, 4, 16, 64, 260, 8, 2, false},  // Inkling-2 shape (hd=128, G=3), 2-bit binary16
      {128, 12, 4, 16, 64, 260, 6, 3, true},   // Inkling-2 shape (hd=128, G=3), 3-bit packed CTA FP8
      {128, 12, 4, 16, 64, 190, 5, 4, false},  // Inkling-2 shape (hd=128, G=3), 4-bit binary16
      {64, 6, 2, 8, 32, 140, 5, 2, true},      // hd=64, G=3, 2-bit FP8
  };
  for (const Case& c : cases) {
    const std::string tag = "hd" + std::to_string(c.hd) + " G=" + std::to_string(c.nh / c.nkv) + " b" +
                            std::to_string(c.bits) + (c.fp8 ? " fp8" : " f16");
    seed("inkling2_rel_strata", c.hd, c.nh, c.bits, int(c.fp8));
    const int total = c.ctx + c.rows, cap = total + 16, slots = 64;
    const int q_dim = c.nh * c.hd, kv_dim = c.nkv * c.hd;
    const int words = strata_words(c.hd, c.bits);
    const float qk_clip = 6.f, rel_clip = 4.f, v_eps = 1e-6f, log_alpha = 0.25f;
    const int log_floor = 32;
    const double scale = 1.0 / std::sqrt(double(c.hd));

    Vec q_all = randv(size_t(total) * q_dim, 0.8);
    Vec k_all = randv(size_t(total) * kv_dim, 0.8);
    Vec v_all = randv(size_t(total) * kv_dim, 0.8);
    Vec r_all = randv(size_t(total) * c.nh * c.d_rel, 0.8);
    Vec q_cw = randv(size_t(q_dim), 0.5);
    Vec r_cw = randv(size_t(c.nh) * c.d_rel, 0.5);
    Vec rel_bias = randv(size_t(c.rel_extent) * c.d_rel, 0.3);
    Vec sink = randv(size_t(c.rows) * c.nh, 0.8);
    Vec gate = randv(size_t(c.rows) * q_dim, 1.0);

    const size_t nv = size_t(cap) * kv_dim;
    void* dK = c.fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
    void* dV = c.fp8 ? static_cast<void*>(dev<uint8_t>(nv)) : static_cast<void*>(dev<kv16>(nv));
    float* dKs = c.fp8 ? dev<float>(size_t(cap) * c.nkv) : nullptr;
    float* dVs = c.fp8 ? dev<float>(size_t(cap) * c.nkv) : nullptr;
    const size_t ncode = size_t(strata_positions(cap)) * c.nkv * words;
    uint32_t *dkq = dev<uint32_t>(ncode), *dvq = dev<uint32_t>(ncode);
    __half2 *dkqs = dev<__half2>(size_t(cap) * c.nkv), *dvqs = dev<__half2>(size_t(cap) * c.nkv);
    KvRef kv{dK, dV, dKs, dVs, dkq, dvq, dkqs, dvqs, c.bits, words, 0};

    float *dq_in = up(q_all), *dk_in = up(k_all), *dv_in = up(v_all), *dr_in = up(r_all);
    float *d_qcw = up(q_cw), *d_rcw = up(r_cw), *d_rb = up(rel_bias), *d_sink = up(sink), *d_gate = up(gate);
    float* dq_out = dev<float>(size_t(total) * q_dim);
    float* dr_out = dev<float>(size_t(total) * c.nh * c.d_rel);
    DevParams* P0 = params(0, 0);
    rel_attn_prep(dq_in, q_dim, dk_in, kv_dim, dv_in, kv_dim, dr_in, c.nh * c.d_rel, d_qcw, d_rcw, qk_clip, rel_clip,
                  v_eps, log_floor, log_alpha, c.nh, c.nkv, c.hd, c.d_rel, total, P0, kPosPosition, false, dq_out,
                  dr_out, kv, 0);
    CK(cudaDeviceSynchronize());

    // 1. Staged prefill Strata write (strata_base = 16) must match direct writes byte-for-byte.
    {
      const int base = 16, nr = 16;
      const size_t sw = size_t(nr) * c.nkv * words, sg = size_t(nr) * c.nkv;
      KvRef staged = kv;
      staged.kq = dev<uint32_t>(sw);
      staged.vq = dev<uint32_t>(sw);
      staged.kqs = dev<__half2>(sg);
      staged.vqs = dev<__half2>(sg);
      CK(cudaMemset(staged.kq, 0xcd, sw * sizeof(uint32_t)));
      CK(cudaMemset(staged.vq, 0xcd, sw * sizeof(uint32_t)));
      CK(cudaMemset(staged.kqs, 0xcd, sg * sizeof(__half2)));
      CK(cudaMemset(staged.vqs, 0xcd, sg * sizeof(__half2)));
      DevParams* Pb = params(base, 0);
      rel_attn_prep(dq_in + size_t(base) * q_dim, q_dim, dk_in + size_t(base) * kv_dim, kv_dim,
                    dv_in + size_t(base) * kv_dim, kv_dim, dr_in + size_t(base) * c.nh * c.d_rel, c.nh * c.d_rel,
                    d_qcw, d_rcw, qk_clip, rel_clip, v_eps, log_floor, log_alpha, c.nh, c.nkv, c.hd, c.d_rel, nr, Pb,
                    kPosPosition, false, dq_out + size_t(base) * q_dim, dr_out + size_t(base) * c.nh * c.d_rel,
                    staged, 0, 0, base);
      std::vector<uint32_t> kq_all(ncode), vq_all(ncode), skq(sw), svq(sw);
      std::vector<__half2> kqs_all(size_t(cap) * c.nkv), vqs_all(size_t(cap) * c.nkv), skqs(sg), svqs(sg);
      CK(cudaMemcpy(kq_all.data(), dkq, ncode * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vq_all.data(), dvq, ncode * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(kqs_all.data(), dkqs, kqs_all.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vqs_all.data(), dvqs, vqs_all.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(skq.data(), staged.kq, sw * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(svq.data(), staged.vq, sw * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(skqs.data(), staged.kqs, sg * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(svqs.data(), staged.vqs, sg * 4, cudaMemcpyDeviceToHost));
      const size_t woff = strata_words_at(base, 0, c.nkv, words), goff = size_t(base) * c.nkv;
      const bool eq = std::equal(skq.begin(), skq.end(), kq_all.begin() + ptrdiff_t(woff)) &&
                      std::equal(svq.begin(), svq.end(), vq_all.begin() + ptrdiff_t(woff)) &&
                      std::memcmp(skqs.data(), kqs_all.data() + goff, sg * sizeof(__half2)) == 0 &&
                      std::memcmp(svqs.data(), vqs_all.data() + goff, sg * sizeof(__half2)) == 0;
      std::printf("  %-4s rel_attn_prep strata_base=16 staged==direct (%s)\n", eq ? "ok" : "FAIL", tag.c_str());
      if (!eq) ++g_fail;
      cudaFree(Pb);
      cudaFree(staged.kq);
      cudaFree(staged.vq);
      cudaFree(staged.kqs);
      cudaFree(staged.vqs);
    }

    // 2. rel_bias_table vs float64 dot product.
    float* d_btab = dev<float>(size_t(c.rows) * c.nh * c.rel_extent);
    rel_bias_table(dr_out + size_t(c.ctx) * c.nh * c.d_rel, d_rb, c.nh, c.d_rel, c.rel_extent, c.rows, d_btab, 0);
    Vec got_bt = down(d_btab, size_t(c.rows) * c.nh * c.rel_extent);
    Vec rq_h = down(dr_out + size_t(c.ctx) * c.nh * c.d_rel, size_t(c.rows) * c.nh * c.d_rel);
    Vec want_bt(got_bt.size(), 0.0);
    for (int r = 0; r < c.rows; ++r)
      for (int h = 0; h < c.nh; ++h)
        for (int dist = 0; dist < c.rel_extent; ++dist) {
          double acc = 0.0;
          for (int m = 0; m < c.d_rel; ++m)
            acc += rel_bias[size_t(dist) * c.d_rel + m] * rq_h[(size_t(r) * c.nh + h) * c.d_rel + m];
          want_bt[(size_t(r) * c.nh + h) * c.rel_extent + dist] = acc;
        }
    expect("rel_bias_table " + tag, got_bt, want_bt, 1e-5);

    // Prepare exact float64 reference attend_rel over the stored cache (binary16 or FP8).
    Vec Kr(nv), Vr(nv);
    if (c.fp8) {
      std::vector<uint8_t> hk(nv), hv(nv);
      Vec hks = down(dKs, size_t(cap) * c.nkv), hvs = down(dVs, size_t(cap) * c.nkv);
      CK(cudaMemcpy(hk.data(), dK, nv, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(hv.data(), dV, nv, cudaMemcpyDeviceToHost));
      for (int p = 0; p < total; ++p)
        for (int kh = 0; kh < c.nkv; ++kh) {
          const double ks = hks[size_t(p) * c.nkv + kh], vs = hvs[size_t(p) * c.nkv + kh];
          for (int d = 0; d < c.hd; ++d) {
            const size_t idx = (size_t(p) * c.nkv + kh) * c.hd + d;
            Kr[idx] = double(ops::e4m3_to_double(hk[idx])) * ks;
            Vr[idx] = double(ops::e4m3_to_double(hv[idx])) * vs;
          }
        }
    } else {
      Kr = down_f16(static_cast<kv16*>(dK), nv);
      Vr = down_f16(static_cast<kv16*>(dV), nv);
    }
    Vec q_h = down(dq_out + size_t(c.ctx) * q_dim, size_t(c.rows) * q_dim);
    Vec want_exact(size_t(c.rows) * q_dim, 0.0);
    for (int r = 0; r < c.rows; ++r)
      for (int h = 0; h < c.nh; ++h)
        ops::attend_rel(&q_h[(size_t(r) * c.nh + h) * c.hd], &rq_h[(size_t(r) * c.nh + h) * c.d_rel], Kr.data(),
                        Vr.data(), rel_bias.data(), sink[size_t(r) * c.nh + h], c.ctx + r, 0, c.nkv, c.hd, c.d_rel,
                        c.rel_extent, h / (c.nh / c.nkv), scale, &want_exact[(size_t(r) * c.nh + h) * c.hd]);
    for (size_t i = 0; i < want_exact.size(); ++i) want_exact[i] *= ops::sigmoid(gate[i]);

    // Allocate StrataScratch.
    StrataScratch sc{};
    sc.lse_stride = ((cap + 1) / 2 + 7) / 8 * 8;
    sc.sel_stride = (cap + 1) / 2;
    sc.chunks = 2;
    sc.lse = dev<__half>(size_t(c.rows) * c.nh * sc.lse_stride);
    sc.cand = dev<int>(size_t(c.rows) * c.nkv * sc.chunks * sc.sel_stride);
    sc.cand_key = dev<uint32_t>(size_t(c.rows) * c.nkv * sc.chunks * sc.sel_stride);
    sc.cand_n = dev<int>(size_t(c.rows) * c.nkv * sc.chunks);
    sc.gathered = dev<int>(size_t(c.rows) * c.nkv * sc.chunks * sc.sel_stride);
    sc.gathered_key = dev<uint32_t>(size_t(c.rows) * c.nkv * sc.chunks * sc.sel_stride);
    sc.sel = dev<int>(size_t(c.rows) * c.nkv * sc.sel_stride);
    sc.sel_n = dev<int>(size_t(c.rows) * c.nkv);
    sc.qr = dev<float>(size_t(c.rows) * c.nh * c.hd);
    sc.delta = dev<float>(size_t(c.rows) * c.nkv * kStrataSlices * (2 * size_t(c.nh / c.nkv * c.hd) + 8));

    float* d_part = dev<float>(attention_partial_floats(c.rows, c.nh, c.hd, slots));
    float* d_out = dev<float>(size_t(c.rows) * q_dim);
    DevParams* P = params(c.ctx, 0);
    const RelStrata rel{d_btab, c.rel_extent, d_sink, c.nh};

    // 3. Every pair refined (how.least = 100000) == exact attend_rel float64 ref (both MMA and cores scan).
    sc.how = {16, 100000, 16, 0};
    for (bool mma : {true, false}) {
      KernelOptions kopt{};
      kopt.strata_mma = mma;
      Launch L = TL;
      L.opt = &kopt;
      attention_strata(dq_out + size_t(c.ctx) * q_dim, kv, sc, c.nh, c.nkv, c.hd, c.rows, float(scale), P, d_gate,
                       q_dim, d_out, q_dim, d_part, slots, L, 0, nullptr, nullptr, rel);
      expect(std::string("rel_strata all-refined ") + (mma ? "mma " : "cores ") + tag,
             down(d_out, size_t(c.rows) * q_dim), want_exact, 6e-3);
    }

    // 4. Partial refinement: Gate 4 bit-identity (R=1 vs R=rows) + FP8 hot-cache bit-identity.
    sc.how = {8, 4, 16, 64};
    for (bool mma : {true, false}) {
      KernelOptions kopt{};
      kopt.strata_mma = mma;
      Launch L = TL;
      L.opt = &kopt;
      attention_strata(dq_out + size_t(c.ctx) * q_dim, kv, sc, c.nh, c.nkv, c.hd, c.rows, float(scale), P, d_gate,
                       q_dim, d_out, q_dim, d_part, slots, L, 0, nullptr, nullptr, rel);
      Vec multi = down(d_out, size_t(c.rows) * q_dim);
      for (int rr : {0, c.rows - 1}) {
        DevParams* P1 = params(c.ctx + rr, 0);
        float* d_o1 = dev<float>(size_t(q_dim));
        const RelStrata rel1{d_btab + size_t(rr) * c.nh * c.rel_extent, c.rel_extent, d_sink + size_t(rr) * c.nh, c.nh};
        attention_strata(dq_out + size_t(c.ctx + rr) * q_dim, kv, sc, c.nh, c.nkv, c.hd, 1, float(scale), P1,
                         d_gate + size_t(rr) * q_dim, q_dim, d_o1, q_dim, d_part, slots, L, 0, nullptr, nullptr, rel1);
        Vec got1 = down(d_o1, size_t(q_dim));
        Vec slice(multi.begin() + ptrdiff_t(rr) * q_dim, multi.begin() + ptrdiff_t(rr + 1) * q_dim);
        expect_bits(std::string("rel_strata R=1 vs multi r=") + std::to_string(rr) + (mma ? " mma " : " cores ") + tag,
                    got1, slice);
        cudaFree(P1);
        cudaFree(d_o1);
      }

      if (c.fp8 && mma) {
        KvRef hkv = kv;
        hkv.hot_bits = 7;  // 128 slots per KV head
        hkv.host_from = 0;
        const size_t htags = size_t(c.nkv) << hkv.hot_bits;
        hkv.hot_tag = dev<int32_t>(htags);
        hkv.hot_claim = dev<int32_t>(htags);
        hkv.hot = dev<uint8_t>(htags * 4 * size_t(c.hd));
        hkv.hot_scale = dev<float>(htags * 4);
        std::vector<int32_t> neg1(htags, -1), freev(htags, kHotFree);
        CK(cudaMemcpy(hkv.hot_tag, neg1.data(), htags * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(hkv.hot_claim, freev.data(), htags * 4, cudaMemcpyHostToDevice));
        // Cold pass (claims & fills hot cache), then warm pass (reads from hot cache).
        attention_strata(dq_out + size_t(c.ctx) * q_dim, hkv, sc, c.nh, c.nkv, c.hd, c.rows, float(scale), P, d_gate,
                         q_dim, d_out, q_dim, d_part, slots, L, 0, nullptr, nullptr, rel);
        Vec cold = down(d_out, size_t(c.rows) * q_dim);
        attention_strata(dq_out + size_t(c.ctx) * q_dim, hkv, sc, c.nh, c.nkv, c.hd, c.rows, float(scale), P, d_gate,
                         q_dim, d_out, q_dim, d_part, slots, L, 0, nullptr, nullptr, rel);
        Vec warm = down(d_out, size_t(c.rows) * q_dim);
        expect_bits("rel_strata fp8 hot-cache cold==uncached " + tag, cold, multi);
        expect_bits("rel_strata fp8 hot-cache warm==uncached " + tag, warm, multi);
        cudaFree(hkv.hot_tag);
        cudaFree(hkv.hot_claim);
        cudaFree(hkv.hot);
        cudaFree(hkv.hot_scale);
      }
    }

    cudaFree(P0);
    cudaFree(P);
    cudaFree(dK);
    cudaFree(dV);
    if (dKs) cudaFree(dKs);
    if (dVs) cudaFree(dVs);
    cudaFree(dkq);
    cudaFree(dvq);
    cudaFree(dkqs);
    cudaFree(dvqs);
    cudaFree(dq_in);
    cudaFree(dk_in);
    cudaFree(dv_in);
    cudaFree(dr_in);
    cudaFree(d_qcw);
    cudaFree(d_rcw);
    cudaFree(d_rb);
    cudaFree(d_sink);
    cudaFree(d_gate);
    cudaFree(dq_out);
    cudaFree(dr_out);
    cudaFree(d_btab);
    cudaFree(sc.lse);
    cudaFree(sc.cand);
    cudaFree(sc.cand_key);
    cudaFree(sc.cand_n);
    cudaFree(sc.gathered);
    cudaFree(sc.gathered_key);
    cudaFree(sc.sel);
    cudaFree(sc.sel_n);
    cudaFree(sc.qr);
    cudaFree(sc.delta);
    cudaFree(d_part);
    cudaFree(d_out);
  }
}

void test_inkling2_moe() {
  std::printf("inkling2 moe (route, situ, gate_up, down)\n");
  // 1. moe_route: sigmoid+bias and softmax.
  for (int sig : {1, 0}) {
    const int R = 11, E = 64, top_k = 4;
    seed("inkling2_moe_route", sig, R, E, top_k);
    Vec logits = randv(size_t(R) * E, 1.5);
    Vec bias = sig ? randv(size_t(E), 0.2) : Vec{};
    float* dl = up(logits);
    float* db = sig ? up(bias) : nullptr;
    int* di = dev<int>(size_t(R) * top_k);
    float* dw = dev<float>(size_t(R) * top_k);
    moe_route(dl, db, R, E, top_k, sig != 0, true, 1e-20f, 2.5f, di, dw, 0);
    auto got_i = down_i(di, size_t(R) * top_k);
    Vec got_w = down(dw, size_t(R) * top_k);

    std::vector<int> want_i(size_t(R) * top_k);
    Vec want_w(size_t(R) * top_k);
    for (int r = 0; r < R; ++r)
      ops::moe_route(&logits[size_t(r) * E], sig ? bias.data() : nullptr, E, top_k, sig != 0, true, 1e-20, 2.5,
                     &want_i[size_t(r) * top_k], &want_w[size_t(r) * top_k]);
    bool idx_ok = (got_i == want_i);
    std::printf("  %-4s %-48s\n", idx_ok ? "ok" : "FAIL", sig ? "moe_route sigmoid+bias indices" : "moe_route softmax indices");
    if (!idx_ok) ++g_fail;
    expect(sig ? "moe_route sigmoid+bias weights" : "moe_route softmax weights", got_w, want_w, 1e-5);
    cudaFree(dl);
    if (db) cudaFree(db);
    cudaFree(di);
    cudaFree(dw);
  }

  // 2. situ_rows: SiTU and SwiGLU fallback.
  {
    const int R = 13, I = 192;
    seed("inkling2_situ", R, I);
    Vec gu = randv(size_t(R) * 2 * I, 2.0);
    float* d_gu = up(gu);
    float* d_out = dev<float>(size_t(R) * I);
    for (float beta : {6.f, 0.f}) {
      const float lb = beta > 0.f ? 8.f : 0.f;
      situ_rows(d_gu, 2 * I, I, beta, lb, d_out, I, R, 0);
      Vec got = down(d_out, size_t(R) * I);
      Vec want(size_t(R) * I);
      for (int r = 0; r < R; ++r)
        for (int i = 0; i < I; ++i) {
          const double g = gu[size_t(r) * 2 * I + i], u = gu[size_t(r) * 2 * I + I + i];
          want[size_t(r) * I + i] = beta > 0.f ? ops::situ(g, u, beta, lb) : ops::silu(g) * u;
        }
      expect(beta > 0.f ? "situ_rows SiTU" : "situ_rows SwiGLU fallback", got, want, 1e-5);
    }
    cudaFree(d_gu);
    cudaFree(d_out);
  }

  // 3. moe_gate_up + moe_down across NVFP4 (per-expert rs), GGML blocks (MXFP4, Q2_0, Q4_K, Q8_0 in q8 & a16h), BF16.
  const int R = 37, E = 8, top_k = 2, H = 256, I = 256;
  const float beta = 6.f, lb = 8.f;
  seed("inkling2_moe_gemm", R, E, top_k, H, I);
  Vec x = randv(size_t(R) * H, 1.0);
  Vec r_logits = randv(size_t(R) * E, 1.0);
  std::vector<int> eidx(size_t(R) * top_k);
  Vec ew(size_t(R) * top_k);
  for (int r = 0; r < R; ++r)
    ops::moe_route(&r_logits[size_t(r) * E], nullptr, E, top_k, true, true, 1e-20, 1.0, &eidx[size_t(r) * top_k],
                   &ew[size_t(r) * top_k]);

  float* d_x = up(x);
  int* d_eidx = dev<int>(eidx.size());
  CK(cudaMemcpy(d_eidx, eidx.data(), eidx.size() * 4, cudaMemcpyHostToDevice));
  float* d_ew = up(ew);
  MoeScratch msc{};
  msc.counts = dev<int>(E);
  msc.offsets = dev<int>(E + 1);
  msc.pairs = dev<int>(size_t(R) * top_k);
  msc.tasks = dev<MoeTask>(size_t(R) * top_k);
  msc.n_tasks = dev<int>(1);
  msc.down_buf = dev<float>(size_t(R) * top_k * H);
  msc.stage_elems = std::max(size_t(R) * H, size_t(R) * top_k * I);
  const size_t q8_a_b = (msc.stage_elems + 255) / 256 * 256;
  const size_t q8_d_b = (((msc.stage_elems + 31) / 32) * sizeof(float) + 255) / 256 * 256;
  const size_t q8_s_b = (((msc.stage_elems + 15) / 16) * sizeof(float) + 255) / 256 * 256;
  const size_t stg_b = std::max(msc.stage_elems * sizeof(bf16), q8_a_b + q8_d_b + q8_s_b);
  uint8_t* msc_stage = dev<uint8_t>(stg_b);
  msc.q8_a = reinterpret_cast<int8_t*>(msc_stage);
  msc.q8_d = reinterpret_cast<float*>(msc_stage + q8_a_b);
  msc.q8_s = reinterpret_cast<float*>(msc_stage + q8_a_b + q8_d_b);
  msc.bf16_a = reinterpret_cast<bf16*>(msc_stage);
  msc.chunk_rows = R;
  msc.max_experts = E;
  msc.max_top_k = top_k;

  auto run_and_check = [&](const std::string& label, const ExpertMat& w_gu, const ExpertMat& w_dn, const Vec& gu_f64,
                           const Vec& dn_f64, bool round_a16, bool round_q8, bool round_a16h, double tol) {
    // Reference in float64 with the exact input rounding contract of this weight format.
    Vec want_gu(size_t(R) * top_k * 2 * I, 0.0);
    Vec want_act(size_t(R) * top_k * I, 0.0);
    Vec want_out(size_t(R) * H, 0.0);
    for (int r = 0; r < R; ++r) {
      Vec xr(x.begin() + ptrdiff_t(r) * H, x.begin() + ptrdiff_t(r + 1) * H);
      if (round_a16) round_bf16(xr);
      else if (round_q8) round_q8_inplace(xr.data(), H);
      else if (round_a16h) ops::round_a16h(xr.data(), H);
      for (int k = 0; k < top_k; ++k) {
        const int e = eidx[size_t(r) * top_k + k];
        Vec gu(static_cast<size_t>(2 * I), 0.0);
        ops::linear_expert_f64(xr.data(), H, gu_f64.data(), e, 2 * I, gu.data());
        std::copy(gu.begin(), gu.end(), want_gu.begin() + ptrdiff_t(r * top_k + k) * 2 * I);
        Vec a(static_cast<size_t>(I));
        for (int i = 0; i < I; ++i) a[i] = ops::situ(gu[i], gu[I + i], beta, lb);
        std::copy(a.begin(), a.end(), want_act.begin() + ptrdiff_t(r * top_k + k) * I);
        Vec a_in = a;
        if (round_a16) round_bf16(a_in);
        else if (round_q8) round_q8_inplace(a_in.data(), I);
        else if (round_a16h) ops::round_a16h(a_in.data(), I);
        Vec dn(static_cast<size_t>(H), 0.0);
        ops::linear_expert_f64(a_in.data(), I, dn_f64.data(), e, H, dn.data());
        for (int c = 0; c < H; ++c) want_out[size_t(r) * H + c] += ew[size_t(r) * top_k + k] * dn[c];
      }
    }

    // Prefill grouped fused (R = 37, sc = &msc).
    float* d_act_fused = dev<float>(size_t(R) * top_k * I);
    float* d_gu_raw = dev<float>(size_t(R) * top_k * 2 * I);
    float* d_act_unfused = dev<float>(size_t(R) * top_k * I);
    float* d_y_grouped = dev<float>(size_t(R) * H);
    float* d_y_direct = dev<float>(size_t(R) * H);

    moe_gate_up(d_x, H, w_gu, d_eidx, R, top_k, I, H, beta, lb, true, d_act_fused, I, &msc, TL);
    moe_gate_up(d_x, H, w_gu, d_eidx, R, top_k, I, H, beta, lb, false, d_gu_raw, 2 * I, &msc, TL);
    situ_rows(d_gu_raw, 2 * I, I, beta, lb, d_act_unfused, I, R * top_k, 0);
    Vec got_gu = down(d_gu_raw, size_t(R) * top_k * 2 * I);
    Vec got_act_f = down(d_act_fused, size_t(R) * top_k * I);
    Vec got_act_u = down(d_act_unfused, size_t(R) * top_k * I);
    expect("moe_gate_up raw " + label, got_gu, want_gu, tol);
    // situ clamps output to [-beta*lb, +beta*lb] even when random_blocks (e.g. Q6_K) raw dots reach ~1000.
    expect("moe_gate_up " + label, got_act_f, want_act, std::max(tol, 1.5e-4));
    expect_bits("moe_gate_up fused==unfused " + label, got_act_f, got_act_u);

    float* d_want_act = up(want_act);
    moe_down(d_want_act, I, w_dn, d_eidx, d_ew, R, top_k, H, I, d_y_grouped, H, false, &msc, TL);
    Vec got_y_grp_clean = down(d_y_grouped, size_t(R) * H);
    expect("moe_down " + label, got_y_grp_clean, want_out, tol);
    cudaFree(d_want_act);

    // Gate 4: R=1 (sc=nullptr), R=1 (sc=&msc), R=8 grouped (sc=&msc), and R=37 grouped (sc=&msc) MUST BE BIT-IDENTICAL!
    moe_down(d_act_fused, I, w_dn, d_eidx, d_ew, R, top_k, H, I, d_y_grouped, H, false, &msc, TL);
    Vec got_y_grp = down(d_y_grouped, size_t(R) * H);
    const int rr = 19;
    float* d_act_1 = dev<float>(size_t(top_k) * I);
    float* d_y_1 = dev<float>(size_t(H));
    moe_gate_up(d_x + size_t(rr) * H, H, w_gu, d_eidx + size_t(rr) * top_k, 1, top_k, I, H, beta, lb, true, d_act_1, I,
                nullptr, TL);
    moe_down(d_act_1, I, w_dn, d_eidx + size_t(rr) * top_k, d_ew + size_t(rr) * top_k, 1, top_k, H, I, d_y_1, H, false,
             nullptr, TL);
    Vec got_y1 = down(d_y_1, size_t(H));
    Vec got_ygrp_rr(got_y_grp.begin() + ptrdiff_t(rr) * H, got_y_grp.begin() + ptrdiff_t(rr + 1) * H);
    expect_bits("moe R=1 direct == R=37 grouped " + label, got_y1, got_ygrp_rr);

    moe_gate_up(d_x + size_t(rr) * H, H, w_gu, d_eidx + size_t(rr) * top_k, 1, top_k, I, H, beta, lb, true, d_act_1, I,
                &msc, TL);
    moe_down(d_act_1, I, w_dn, d_eidx + size_t(rr) * top_k, d_ew + size_t(rr) * top_k, 1, top_k, H, I, d_y_1, H, false,
             &msc, TL);
    expect_bits("moe R=1 staged == R=37 grouped " + label, down(d_y_1, size_t(H)), got_ygrp_rr);

    const int r8_0 = 16;  // covers rr = 19 at offset 3
    float* d_act_8 = dev<float>(size_t(8) * top_k * I);
    float* d_y_8 = dev<float>(size_t(8) * H);
    moe_gate_up(d_x + size_t(r8_0) * H, H, w_gu, d_eidx + size_t(r8_0) * top_k, 8, top_k, I, H, beta, lb, true, d_act_8,
                I, &msc, TL);
    moe_down(d_act_8, I, w_dn, d_eidx + size_t(r8_0) * top_k, d_ew + size_t(r8_0) * top_k, 8, top_k, H, I, d_y_8, H,
             false, &msc, TL);
    Vec got_y8 = down(d_y_8, size_t(8) * H);
    Vec got_y8_rr(got_y8.begin() + ptrdiff_t(rr - r8_0) * H, got_y8.begin() + ptrdiff_t(rr - r8_0 + 1) * H);
    expect_bits("moe R=8 grouped == R=37 grouped " + label, got_y8_rr, got_ygrp_rr);
    cudaFree(d_act_8);
    cudaFree(d_y_8);

    // Also verify accumulate=true on moe_down.
    Vec bias_y = randv(size_t(H), 0.5);
    float* d_y_acc = up(bias_y);
    moe_down(d_act_1, I, w_dn, d_eidx + size_t(rr) * top_k, d_ew + size_t(rr) * top_k, 1, top_k, H, I, d_y_acc, H, true,
             nullptr, TL);
    Vec got_acc = down(d_y_acc, size_t(H));
    Vec want_acc(static_cast<size_t>(H));
    for (int c = 0; c < H; ++c) want_acc[c] = bias_y[c] + got_y1[c];
    expect("moe_down accumulate " + label, got_acc, want_acc, 1e-6);

    cudaFree(d_act_fused);
    cudaFree(d_gu_raw);
    cudaFree(d_act_unfused);
    cudaFree(d_y_grouped);
    cudaFree(d_y_direct);
    cudaFree(d_act_1);
    cudaFree(d_y_1);
    cudaFree(d_y_acc);
  };

  // (a) BF16 experts.
  {
    Vec gu = randv(size_t(E) * 2 * I * H, 0.1);
    Vec dn = randv(size_t(E) * H * I, 0.1);
    round_bf16(gu);
    round_bf16(dn);
    ExpertMat w_gu{}, w_dn{};
    w_gu.n_experts = E;
    w_gu.N = 2 * I;
    w_gu.w = up_bf16(gu);
    w_dn.n_experts = E;
    w_dn.N = H;
    w_dn.w = up_bf16(dn);
    run_and_check("bf16", w_gu, w_dn, gu, dn, false, false, false, 3e-5);
    cudaFree(const_cast<bf16*>(w_gu.w));
    cudaFree(const_cast<bf16*>(w_dn.w));
  }

  // (b) NVFP4 experts with distinct per-expert F32 scales!
  {
    Vec gu_raw = randv(size_t(E) * 2 * I * H, 0.2);
    Vec dn_raw = randv(size_t(E) * H * I, 0.2);
    auto make_nvfp4_experts = [&](const Vec& raw, int N, int K, ExpertMat* out, Vec* f64_out) {
      std::vector<uint8_t> q(size_t(E) * N * (K / 2)), s(size_t(E) * N * (K / 16));
      std::vector<float> rs(static_cast<size_t>(E));
      f64_out->resize(size_t(E) * N * K);
      for (int e = 0; e < E; ++e) {
        const float rscale = 0.5f + 0.35f * float(e);
        Vec e_f(size_t(N) * K);
        for (size_t i = 0; i < size_t(N) * K; ++i) e_f[i] = raw[size_t(e) * N * K + i];
        bf16* d_bf = up_bf16(e_f);
        uint8_t* dq = dev<uint8_t>(size_t(N) * (K / 2));
        uint8_t* ds = dev<uint8_t>(size_t(N) * (K / 16));
        float* drs = dev<float>(1);
        quantize_nvfp4(d_bf, N, K, dq, ds, drs, 0);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> hq(size_t(N) * (K / 2)), hs(size_t(N) * (K / 16));
        float hrs = 1.f;
        CK(cudaMemcpy(hq.data(), dq, hq.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hs.data(), ds, hs.size(), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&hrs, drs, 4, cudaMemcpyDeviceToHost));
        rs[size_t(e)] = hrs * rscale;
        std::copy(hq.begin(), hq.end(), q.begin() + ptrdiff_t(e) * N * (K / 2));
        std::copy(hs.begin(), hs.end(), s.begin() + ptrdiff_t(e) * N * (K / 16));
        flyx_tensor_view qv{}, sv{}, s2v{};
        qv.dtype = FLYX_DT_U8;
        qv.ndim = 2;
        qv.shape[0] = N;
        qv.shape[1] = K;
        qv.data = hq.data();
        sv.dtype = FLYX_DT_U8;
        sv.ndim = 2;
        sv.shape[0] = N;
        sv.shape[1] = K / 16;
        sv.data = hs.data();
        s2v.dtype = FLYX_DT_F32;
        s2v.ndim = 1;
        s2v.shape[0] = 1;
        s2v.data = &rs[size_t(e)];
        Vec eq = ops::dequant_nvfp4(&qv, &sv, &s2v);
        std::copy(eq.begin(), eq.end(), f64_out->begin() + ptrdiff_t(e) * N * K);
        cudaFree(d_bf);
        cudaFree(dq);
        cudaFree(ds);
        cudaFree(drs);
      }
      uint8_t* d_q = dev<uint8_t>(q.size());
      uint8_t* d_s = dev<uint8_t>(s.size());
      float* d_rs = dev<float>(rs.size());
      CK(cudaMemcpy(d_q, q.data(), q.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_s, s.data(), s.size(), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(d_rs, rs.data(), rs.size() * 4, cudaMemcpyHostToDevice));
      out->n_experts = E;
      out->N = N;
      out->q = d_q;
      out->s = d_s;
      out->rs = d_rs;
    };
    ExpertMat w_gu{}, w_dn{};
    Vec gu_f64, dn_f64;
    make_nvfp4_experts(gu_raw, 2 * I, H, &w_gu, &gu_f64);
    make_nvfp4_experts(dn_raw, H, I, &w_dn, &dn_f64);
    run_and_check("nvfp4", w_gu, w_dn, gu_f64, dn_f64, true, false, false, 3e-5);
    cudaFree(const_cast<uint8_t*>(w_gu.q));
    cudaFree(const_cast<uint8_t*>(w_gu.s));
    cudaFree(const_cast<float*>(w_gu.rs));
    cudaFree(const_cast<uint8_t*>(w_dn.q));
    cudaFree(const_cast<uint8_t*>(w_dn.s));
    cudaFree(const_cast<float*>(w_dn.rs));
  }

  // (c) Packed GGML block experts: MXFP4, Q2_0, Q4_K, Q6_K, Q8_0, IQ2_XXS, IQ2_S, IQ3_XXS, IQ4_NL (Q8 and a16h).
  const int ggml_types[] = {FLYX_DT_GGML_MXFP4,  FLYX_DT_GGML_Q2_0,   FLYX_DT_GGML_Q4_K,
                            FLYX_DT_GGML_Q6_K,   FLYX_DT_GGML_Q8_0,   FLYX_DT_GGML_IQ2_XXS,
                            FLYX_DT_GGML_IQ2_S,  FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_IQ4_NL};
  for (int dt : ggml_types) {
    const BlockType& bt = block_type(dt);
    const int k_blk = flyx_block_values(dt);
    if (I % k_blk != 0 || H % k_blk != 0) continue;
    auto make_blk_experts = [&](int N, int K, bool use_q8, ExpertMat* out, Vec* f64_out) {
      const size_t n_blocks = size_t(E) * N * (K / k_blk);
      std::vector<uint8_t> raw = random_blocks(bt, n_blocks);
      f64_out->resize(size_t(E) * N * K);
      for (size_t idx = 0; idx < f64_out->size(); ++idx) (*f64_out)[idx] = ops::ggml::value(dt, raw.data(), idx);
      out->n_experts = E;
      out->N = N;
      out->blk = up_blocks(dt, raw, E * N);
      out->blk.q8 = use_q8;
    };
    for (int use_q8 : {1, 0}) {
      ExpertMat w_gu{}, w_dn{};
      Vec gu_f64, dn_f64;
      make_blk_experts(2 * I, H, use_q8 != 0, &w_gu, &gu_f64);
      make_blk_experts(H, I, use_q8 != 0, &w_dn, &dn_f64);
      run_and_check(std::string(bt.name) + (use_q8 ? "_q8" : "_a16h"), w_gu, w_dn, gu_f64, dn_f64, false,
                    use_q8 != 0, use_q8 == 0, 5e-5);
      cudaFree(const_cast<uint8_t*>(w_gu.blk.data));
      cudaFree(const_cast<uint8_t*>(w_dn.blk.data));
    }
  }

  // (d) Multi-span non-power-of-2 K (Inkling-2 dimensions: H = 2304 -> 3 spans tail 8; I = 1152 -> 2 spans tail 4).
  {
    const int RM = 19, EM = 4, KM = 2, HM = 2304, IM = 1152;
    seed("inkling2_moe_multispan", RM, EM, KM, HM, IM);
    Vec xm = randv(size_t(RM) * HM, 0.5);
    Vec rlm = randv(size_t(RM) * EM, 1.0);
    std::vector<int> eim(size_t(RM) * KM);
    Vec ewm(size_t(RM) * KM);
    for (int r = 0; r < RM; ++r)
      ops::moe_route(&rlm[size_t(r) * EM], nullptr, EM, KM, true, true, 1e-20, 1.0, &eim[size_t(r) * KM],
                     &ewm[size_t(r) * KM]);
    float* dxm = up(xm);
    int* deim = dev<int>(eim.size());
    CK(cudaMemcpy(deim, eim.data(), eim.size() * 4, cudaMemcpyHostToDevice));
    float* dewm = up(ewm);
    MoeScratch sm{};
    sm.counts = dev<int>(EM);
    sm.offsets = dev<int>(EM + 1);
    sm.pairs = dev<int>(size_t(RM) * KM);
    sm.tasks = dev<MoeTask>(size_t(RM) * KM);
    sm.n_tasks = dev<int>(1);
    sm.down_buf = dev<float>(size_t(RM) * KM * HM);
    sm.stage_elems = std::max(size_t(RM) * HM, size_t(RM) * KM * IM);
    const size_t qa = (sm.stage_elems + 255) / 256 * 256;
    const size_t qd = (((sm.stage_elems + 31) / 32) * sizeof(float) + 255) / 256 * 256;
    const size_t qs = (((sm.stage_elems + 15) / 16) * sizeof(float) + 255) / 256 * 256;
    uint8_t* sbuf = dev<uint8_t>(std::max(sm.stage_elems * sizeof(bf16), qa + qd + qs));
    sm.q8_a = reinterpret_cast<int8_t*>(sbuf);
    sm.q8_d = reinterpret_cast<float*>(sbuf + qa);
    sm.q8_s = reinterpret_cast<float*>(sbuf + qa + qd);
    sm.bf16_a = reinterpret_cast<bf16*>(sbuf);
    sm.chunk_rows = RM;
    sm.max_experts = EM;
    sm.max_top_k = KM;
    float* d_act_m = dev<float>(size_t(RM) * KM * IM);
    float* d_y_m = dev<float>(size_t(RM) * HM);
    float* d_act_1 = dev<float>(size_t(KM) * IM);
    float* d_y_1 = dev<float>(size_t(HM));
    float* d_act_8 = dev<float>(size_t(8) * KM * IM);
    float* d_y_8 = dev<float>(size_t(8) * HM);
    for (int gu_dt : {FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_Q4_K}) {
      const BlockType& gbt = block_type(gu_dt);
      const BlockType& dbt = block_type(FLYX_DT_GGML_IQ4_NL);
      std::vector<uint8_t> graw = random_blocks(gbt, size_t(EM) * 2 * IM * (HM / flyx_block_values(gu_dt)));
      std::vector<uint8_t> draw = random_blocks(dbt, size_t(EM) * HM * (IM / flyx_block_values(FLYX_DT_GGML_IQ4_NL)));
      ExpertMat wgu{}, wdn{};
      wgu.n_experts = EM;
      wgu.N = 2 * IM;
      wgu.blk = up_blocks(gu_dt, graw, EM * 2 * IM);
      wgu.blk.q8 = true;
      wdn.n_experts = EM;
      wdn.N = HM;
      wdn.blk = up_blocks(FLYX_DT_GGML_IQ4_NL, draw, EM * HM);
      wdn.blk.q8 = true;
      moe_gate_up(dxm, HM, wgu, deim, RM, KM, IM, HM, beta, lb, true, d_act_m, IM, &sm, TL);
      moe_down(d_act_m, IM, wdn, deim, dewm, RM, KM, HM, IM, d_y_m, HM, false, &sm, TL);
      Vec ym = down(d_y_m, size_t(RM) * HM);
      const int rr = 11;
      Vec yref(ym.begin() + ptrdiff_t(rr) * HM, ym.begin() + ptrdiff_t(rr + 1) * HM);
      moe_gate_up(dxm + size_t(rr) * HM, HM, wgu, deim + size_t(rr) * KM, 1, KM, IM, HM, beta, lb, true, d_act_1, IM,
                  &sm, TL);
      moe_down(d_act_1, IM, wdn, deim + size_t(rr) * KM, dewm + size_t(rr) * KM, 1, KM, HM, IM, d_y_1, HM, false, &sm,
               TL);
      expect_bits(std::string("moe multispan 2304x1152 R=1==R=19 ") + gbt.name, down(d_y_1, size_t(HM)), yref);
      moe_gate_up(dxm + size_t(8) * HM, HM, wgu, deim + size_t(8) * KM, 8, KM, IM, HM, beta, lb, true, d_act_8, IM, &sm,
                  TL);
      moe_down(d_act_8, IM, wdn, deim + size_t(8) * KM, dewm + size_t(8) * KM, 8, KM, HM, IM, d_y_8, HM, false, &sm, TL);
      Vec y8 = down(d_y_8, size_t(8) * HM);
      expect_bits(std::string("moe multispan 2304x1152 R=8==R=19 ") + gbt.name,
                  Vec(y8.begin() + ptrdiff_t(rr - 8) * HM, y8.begin() + ptrdiff_t(rr - 7) * HM), yref);
      cudaFree(const_cast<uint8_t*>(wgu.blk.data));
      cudaFree(const_cast<uint8_t*>(wdn.blk.data));
    }
    cudaFree(dxm);
    cudaFree(deim);
    cudaFree(dewm);
    cudaFree(sm.counts);
    cudaFree(sm.offsets);
    cudaFree(sm.pairs);
    cudaFree(sm.tasks);
    cudaFree(sm.n_tasks);
    cudaFree(sm.down_buf);
    cudaFree(sbuf);
    cudaFree(d_act_m);
    cudaFree(d_y_m);
    cudaFree(d_act_1);
    cudaFree(d_y_1);
    cudaFree(d_act_8);
    cudaFree(d_y_8);
  }

  cudaFree(d_x);
  cudaFree(d_eidx);
  cudaFree(d_ew);
  cudaFree(msc.counts);
  cudaFree(msc.offsets);
  cudaFree(msc.pairs);
  cudaFree(msc.tasks);
  cudaFree(msc.n_tasks);
  cudaFree(msc.down_buf);
  cudaFree(msc_stage);
}

void test_dspark() {
  std::printf("dspark drafter kernels\n");
  // 1. dspark_ctx_mix_norm.
  {
    const int R = 11, H = 512, nt = 6;
    seed("dspark_ctx_mix_norm", R, H, nt);
    Vec feats = randv(size_t(R) * nt * H, 1.0);
    Vec mix = randv(size_t(nt), 0.5);
    Vec nw = randv(size_t(H), 1.0);
    const float eps = 1e-6f;
    float* df = up(feats);
    float* dm = up(mix);
    float* dnw = up(nw);
    float* dout = dev<float>(size_t(R) * H);
    dspark_ctx_mix_norm(df, nt * H, dm, dnw, eps, nt, R, H, dout, 0);
    Vec got = down(dout, size_t(R) * H);
    Vec want(size_t(R) * H);
    Vec mixed(H);
    for (int r = 0; r < R; ++r) {
      std::fill(mixed.begin(), mixed.end(), 0.0);
      for (int t = 0; t < nt; ++t)
        for (int i = 0; i < H; ++i) mixed[i] += mix[t] * feats[(size_t(r) * nt + t) * H + i];
      ops::rmsnorm(mixed.data(), nw.data(), H, eps, 0.0, &want[size_t(r) * H]);
    }
    expect("dspark_ctx_mix_norm", got, want, 1e-5);
    cudaFree(df);
    cudaFree(dm);
    cudaFree(dnw);
    cudaFree(dout);
  }

  // 2. dspark_prep_block.
  {
    const int B = 8, H = 256, Rk = 64;
    seed("dspark_prep_block", B, H, Rk);
    Vec anc = randv(size_t(H), 1.0);
    Vec tn = randv(size_t(H), 1.0);
    Vec masks = randv(size_t(B - 1) * H, 0.5);
    DevParams hP{};
    hP.position = 42;
    hP.anchor = 1234;
    DevParams* P = dev<DevParams>(1);
    CK(cudaMemcpy(P, &hP, sizeof(hP), cudaMemcpyHostToDevice));
    float* d_anc = up(anc);
    float* d_tn = up(tn);
    float* d_m = up(masks);
    float* d_dx = dev<float>(size_t(B) * H);
    int* d_prev = dev<int>(1);
    Vec init_st(Rk, 7.0);
    float* d_st = up(init_st);
    dspark_prep_block(P, d_anc, d_tn, 1e-6f, d_m, B, H, d_dx, d_prev, d_st, Rk, 0);
    Vec got_dx = down(d_dx, size_t(B) * H);
    Vec want_dx(size_t(B) * H);
    ops::rmsnorm(anc.data(), tn.data(), H, 1e-6, 0.0, want_dx.data());
    std::copy(masks.begin(), masks.end(), want_dx.begin() + H);
    expect("dspark_prep_block dx", got_dx, want_dx, 1e-5);
    auto got_prev = down_i(d_prev, 1);
    Vec got_st = down(d_st, Rk);
    bool st_ok = got_prev[0] == 1234 && std::all_of(got_st.begin(), got_st.end(), [](double v) { return v == 0.0; });
    std::printf("  %-4s %-48s\n", st_ok ? "ok" : "FAIL", "dspark_prep_block prev & zero state");
    if (!st_ok) ++g_fail;
    cudaFree(P);
    cudaFree(d_anc);
    cudaFree(d_tn);
    cudaFree(d_m);
    cudaFree(d_dx);
    cudaFree(d_prev);
    cudaFree(d_st);
  }

  // 3. dspark_block_attention (both SWA window and Full window=0).
  for (int window : {16, 0}) {
    const int nh = 8, nkv = 2, hd = 64, B = 8, ctx = 45;
    seed("dspark_block_attention", window, hd, ctx);
    Vec q = randv(size_t(B) * nh * hd, 0.5);
    Vec K = randv(size_t(ctx + B) * nkv * hd, 0.5);
    Vec V = randv(size_t(ctx + B) * nkv * hd, 0.5);
    round_f16(K);
    round_f16(V);
    DevParams* P = params(ctx, 0);
    float* dq = up(q);
    kv16* dK = up_f16(K);
    kv16* dV = up_f16(V);
    float* dout = dev<float>(size_t(B) * nh * hd);
    const int splits = 16;
    float* dpart = dev<float>(attention_partial_floats(B, nh, hd, splits));
    const float scale = 1.f / std::sqrt(float(hd));
    dspark_block_attention(dq, dK, dV, nh, nkv, hd, B, window, scale, P, dout, nh * hd, dpart, splits, TL);
    Vec got = down(dout, size_t(B) * nh * hd);

    const int lo = window > 0 ? std::max(0, ctx - window + 1) : 0;
    const int nk = ctx + B - lo;
    Vec want(size_t(B) * nh * hd, 0.0);
    for (int r = 0; r < B; ++r)
      for (int hh = 0; hh < nh; ++hh)
        ops::attend(&q[(size_t(r) * nh + hh) * hd], &K[size_t(lo) * nkv * hd], &V[size_t(lo) * nkv * hd], nk, nkv, hd,
                    hh / (nh / nkv), double(scale), &want[(size_t(r) * nh + hh) * hd]);
    expect(window > 0 ? "dspark_block_attention SWA" : "dspark_block_attention Full", got, want, 3e-5);
    if (window > 0) {
      // Ring wraparound: ctx = 45 > ring = 32, verify bit-identical to linear KV.
      const int ring = 32;
      Vec Kr(size_t(ring) * nkv * hd, 0.0), Vr(size_t(ring) * nkv * hd, 0.0);
      for (int p = 0; p < ctx + B; ++p) {
        const int cell = p % ring;
        std::copy_n(&K[size_t(p) * nkv * hd], nkv * hd, &Kr[size_t(cell) * nkv * hd]);
        std::copy_n(&V[size_t(p) * nkv * hd], nkv * hd, &Vr[size_t(cell) * nkv * hd]);
      }
      kv16* dKr = up_f16(Kr);
      kv16* dVr = up_f16(Vr);
      float* doutr = dev<float>(size_t(B) * nh * hd);
      dspark_block_attention(dq, dKr, dVr, nh, nkv, hd, B, window, scale, P, doutr, nh * hd, dpart, splits, TL, ring);
      expect_bits("dspark_block_attention SWA ring==linear", down(doutr, size_t(B) * nh * hd), got);
      cudaFree(dKr);
      cudaFree(dVr);
      cudaFree(doutr);
    }
    cudaFree(P);
    cudaFree(dq);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dout);
    cudaFree(dpart);
  }

  // 4. dspark_markov_pack, dspark_markov_gate, dspark_combine_logits, dspark_pick_step.
  {
    const int Rk = 64, H = 256, V = 1024;
    seed("dspark_markov", Rk, H, V);
    Vec st = randv(Rk, 0.5);
    Vec w1 = randv(Rk, 0.5);
    Vec hid = randv(H, 0.5);
    Vec hring = randv(size_t(kHring) * H, 0.5);
    const int ctx = 75;
    DevParams* P = params(ctx, 0);
    float* d_st = up(st);
    float* d_w1 = up(w1);
    float* d_hid = up(hid);
    float* d_hr = up(hring);
    float* d_joint = dev<float>(size_t(2 * Rk + 2 * H));
    dspark_markov_pack(P, d_st, d_w1, d_hid, d_hr, Rk, H, d_joint, 0);
    Vec got_j = down(d_joint, size_t(2 * Rk + 2 * H));
    Vec want_j(size_t(2 * Rk + 2 * H));
    std::copy(st.begin(), st.end(), want_j.begin());
    std::copy(w1.begin(), w1.end(), want_j.begin() + Rk);
    std::copy(hid.begin(), hid.end(), want_j.begin() + 2 * Rk);
    std::copy_n(&hring[size_t((ctx - 1) % kHring) * H], H, want_j.begin() + 2 * Rk + H);
    expect("dspark_markov_pack", got_j, want_j, 1e-6);

    Vec jout = randv(3 * Rk, 1.0);
    Vec jbias = randv(3 * Rk, 0.3);
    float* d_jo = up(jout);
    float* d_jb = up(jbias);
    float* d_mk = dev<float>(Rk);
    dspark_markov_gate(d_jo, d_jb, Rk, d_st, d_mk, 0);
    Vec got_st = down(d_st, Rk);
    Vec got_mk = down(d_mk, Rk);
    Vec want_st = st, want_mk(Rk), jo_b(3 * Rk);
    for (int i = 0; i < 3 * Rk; ++i) jo_b[i] = jout[i] + jbias[i];
    ops::dspark_markov_gate(jo_b.data(), Rk, want_st.data(), want_mk.data());
    expect("dspark_markov_gate state", got_st, want_st, 1e-5);
    expect("dspark_markov_gate markov", got_mk, want_mk, 1e-5);

    Vec base = randv(V, 15.0);
    Vec bias = randv(V, 5.0);
    Vec w2b = randv(V, 1.0);
    const float cap = 30.f;
    float* d_base = up(base);
    float* d_bias = up(bias);
    float* d_w2b = up(w2b);
    dspark_combine_logits(d_base, d_bias, d_w2b, V, cap, d_base, 0);
    Vec got_l = down(d_base, V);
    Vec want_l(V);
    for (int i = 0; i < V; ++i) want_l[i] = ops::softcap(base[i] + bias[i] + w2b[i], cap);
    expect("dspark_combine_logits", got_l, want_l, 2e-6);

    // dspark_pick_step without and with oracle.
    int* d_prev = dev<int>(1);
    int* d_drafts = dev<int>(4);
    int picked = 77;
    CK(cudaMemcpy(d_prev, &picked, 4, cudaMemcpyHostToDevice));
    dspark_pick_step(P, nullptr, 1, d_prev, d_drafts, 0);
    auto dr0 = down_i(d_drafts, 4);
    DevParams hPo{};
    hPo.position = 10;
    hPo.oracle_len = 20;
    CK(cudaMemcpy(P, &hPo, sizeof(hPo), cudaMemcpyHostToDevice));
    std::vector<int> oracle(20);
    for (int i = 0; i < 20; ++i) oracle[i] = 100 + i;
    int* d_or = dev<int>(20);
    CK(cudaMemcpy(d_or, oracle.data(), 20 * 4, cudaMemcpyHostToDevice));
    dspark_pick_step(P, d_or, 2, d_prev, d_drafts, 0);
    auto dr1 = down_i(d_drafts, 4);
    auto pr1 = down_i(d_prev, 1);
    bool pick_ok = dr0[1] == 77 && dr1[2] == 113 && pr1[0] == 113;
    std::printf("  %-4s %-48s\n", pick_ok ? "ok" : "FAIL", "dspark_pick_step (normal + oracle)");
    if (!pick_ok) ++g_fail;

    cudaFree(P);
    cudaFree(d_st);
    cudaFree(d_w1);
    cudaFree(d_hid);
    cudaFree(d_hr);
    cudaFree(d_joint);
    cudaFree(d_jo);
    cudaFree(d_jb);
    cudaFree(d_mk);
    cudaFree(d_base);
    cudaFree(d_bias);
    cudaFree(d_w2b);
    cudaFree(d_prev);
    cudaFree(d_drafts);
    cudaFree(d_or);
  }
}

}  // namespace flyx::optest
