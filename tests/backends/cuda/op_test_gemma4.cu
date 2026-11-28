// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gemma 4's pieces against the float64 reference: a norm after a projection added to the residual and the layer's
// scalar (norm_add), the embedding's scale (scale_rows), GELU's tanh form in the gated rows (swiglu), and attn_prep
// on its global layers (v as k's projection, normed without a weight; proportional rotary from the table its bind
// makes: a quarter of the head turned at the whole head's frequencies) and its sliding ones (v normed).
#include "tests/backends/cuda/op_test.cuh"

#include "ops/activation/ref.h"
#include "ops/norm/ref.h"
#include "ops/rope/ref.h"

namespace flyx::optest {
namespace {

constexpr int kPrefills = 8;  // attention()'s splits in a prefill pass (backend.h kPrefillSplits)

// x = (x + rmsnorm(y) w) * s over rows 1 and 20 (a row's bits whatever the rows), with and without the scalar.
void norm_add_case() {
  const int n = 3840, rows = 20;
  seed("gemma4_norm_add", n, rows);
  const Vec x = randv(size_t(rows) * n, 4.0), y = randv(size_t(rows) * n, 30.0), w = randv(n, 0.5);
  const double s = 0.6182;
  float* dw = up(w);
  float* ds = up(Vec{s});
  for (const bool scaled : {false, true}) {
    Vec want(x.size()), normed(n);
    for (int r = 0; r < rows; ++r) {
      ops::rmsnorm(&y[size_t(r) * n], w.data(), n, 1e-6, 0.0, normed.data());
      for (int c = 0; c < n; ++c)
        want[size_t(r) * n + c] = (float(x[size_t(r) * n + c]) + normed[size_t(c)]) * (scaled ? float(s) : 1.0);
    }
    float *dx = up(x), *dy = up(y), *dx1 = up(Vec(x.begin(), x.begin() + n));
    norm_add(dx, n, dy, n, dw, 0.f, 1e-6f, scaled ? ds : nullptr, rows, n, 0);
    norm_add(dx1, n, dy, n, dw, 0.f, 1e-6f, scaled ? ds : nullptr, 1, n, 0);
    const Vec got = down(dx, x.size());
    expect(std::string("norm_add") + (scaled ? ", times the layer's scalar" : ""), got, want, 1e-5);
    expect(std::string("norm_add") + (scaled ? ", times the layer's scalar" : "") + ": a row alone, the same bits",
           down(dx1, size_t(n)), Vec(got.begin(), got.begin() + n), 0);
    for (void* p : {static_cast<void*>(dx), static_cast<void*>(dy), static_cast<void*>(dx1)}) CK(cudaFree(p));
  }
  CK(cudaFree(dw));
  CK(cudaFree(ds));
}

// x *= s: each value the float32 product.
void scale_rows_case() {
  const int n = 2560, rows = 7, ld = n + 5;
  seed("gemma4_scale_rows", n, rows);
  Vec x = randv(size_t(rows) * ld, 2.0);
  for (double& v : x) v = float(v);
  const float s = 50.5f;  // bf16(sqrt(2560))
  Vec want = x;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < n; ++c) want[size_t(r) * ld + c] = double(float(x[size_t(r) * ld + c]) * s);
  float* dx = up(x);
  scale_rows(dx, ld, n, s, rows, 0);
  expect("scale_rows: the float32 products, the columns past n kept", down(dx, x.size()), want, 0);
  CK(cudaFree(dx));
}

// act(gate) * up, both activations, against float64 (the float32 functions' error).
void glu_act_case() {
  const int I = 1536, rows = 3, ld = 2 * I + 8;
  seed("gemma4_glu_act", I, rows);
  const Vec gu = randv(size_t(rows) * ld, 3.0);
  float* dgu = up(gu);
  float* dout = dev<float>(size_t(rows) * I);
  for (const int act : {0, 1}) {
    Vec want(size_t(rows) * I);
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < I; ++i) {
        const double g = float(gu[size_t(r) * ld + i]), u = float(gu[size_t(r) * ld + I + i]);
        want[size_t(r) * I + i] = (act ? ops::gelu_tanh(g) : ops::silu(g)) * u;
      }
    swiglu(dgu, ld, I, dout, I, rows, 0, act);
    expect(act ? "GELU (tanh) gated rows" : "SiLU gated rows", down(dout, want.size()), want, 2e-6);
  }
  CK(cudaFree(dgu));
  CK(cudaFree(dout));
}

// attn_prep as Gemma 4's layers run it.  global: q|k rows only (v is k's projection), proportional rotary of rot_dim
// over hd from its table, q and k normed with weights; sliding: q|k|v, plain rotary.  v normed without a weight at
// k's eps either way.  Positions near 10^5.
void prep_case(bool global) {
  const int nh = 4, nkv = global ? 1 : 2, hd = global ? 512 : 256, rot = global ? 128 : hd, R = 3;
  const double theta = global ? 1e6 : 1e4, eps = 1e-6;
  seed("gemma4_prep", global, nh, nkv, hd);
  const int ld = (nh + (global ? 1 : 2) * nkv) * hd, k_off = nh * hd, v_off = global ? k_off : (nh + nkv) * hd;
  const Vec qkv = randv(size_t(R) * ld, 2.0), qw = randv(hd, 0.3), kw = randv(hd, 0.3);
  std::vector<double> inv(size_t(hd / 2));
  if (global) ops::proportional_inv_freq(theta, hd, rot, inv.data());
  else for (int i = 0; i < hd / 2; ++i) inv[size_t(i)] = 1.0 / std::pow(theta, 2.0 * i / hd);
  const int base = 100000;
  Vec want_q(size_t(R) * nh * hd), want_k(size_t(R) * nkv * hd), want_v(size_t(R) * nkv * hd);
  for (int r = 0; r < R; ++r) {
    for (int h = 0; h < nh + nkv; ++h) {
      const bool q = h < nh;
      const double* src = &qkv[size_t(r) * ld + (q ? 0 : k_off) + (q ? h : h - nh) * hd];
      Vec y(hd);
      ops::rmsnorm(src, q ? qw.data() : kw.data(), hd, eps, 0.0, y.data());
      ops::rope_with(y.data(), base + r, inv.data(), hd);
      std::copy(y.begin(), y.end(), q ? want_q.begin() + ptrdiff_t(r * nh + h) * hd
                                      : want_k.begin() + ptrdiff_t(r * nkv + h - nh) * hd);
    }
    for (int j = 0; j < nkv; ++j)
      ops::rmsnorm(&qkv[size_t(r) * ld + v_off + j * hd], nullptr, hd, eps, 0.0, &want_v[(size_t(r) * nkv + j) * hd]);
  }
  double* d_inv = nullptr;
  if (global) {
    d_inv = dev<double>(inv.size());
    CK(cudaMemcpy(d_inv, inv.data(), inv.size() * sizeof(double), cudaMemcpyHostToDevice));
  }
  float *d_qkv = up(qkv), *d_qw = up(qw), *d_kw = up(kw), *d_q = dev<float>(want_q.size());
  const size_t cells = size_t(base + R) * nkv * hd;
  kv16 *k16 = dev<kv16>(cells), *v16 = dev<kv16>(cells);
  DevParams P{};
  P.position = base;
  DevParams* dP = dev<DevParams>(1);
  CK(cudaMemcpy(dP, &P, sizeof(P), cudaMemcpyHostToDevice));
  KvRef kv{k16, v16, nullptr, nullptr};
  if (!global) rope_register(theta, hd);
  attn_prep(d_qkv, ld, 0, k_off, v_off, d_qw, d_kw, 0.f, float(eps), theta, hd, nh, nkv, hd, R, dP, kPosPosition,
            false, d_q, kv, 0, 0, nullptr, 0, CanvasKeys{}, MRoPE{}, d_inv, nullptr, float(eps));
  const std::string what = global ? "global layer (proportional rotary, v from k)" : "sliding layer";
  expect(what + ": q", down(d_q, want_q.size()), want_q, 2e-6);
  auto cache = [&](const kv16* c) {
    std::vector<kv16> h(size_t(R) * nkv * hd);
    CK(cudaMemcpy(h.data(), c + size_t(base) * nkv * hd, h.size() * sizeof(kv16), cudaMemcpyDeviceToHost));
    Vec v(h.size());
    for (size_t i = 0; i < h.size(); ++i) v[i] = __half2float(h[i]);
    return v;
  };
  expect(what + ": k (binary16 cache)", cache(k16), want_k, 1e-3);
  expect(what + ": v normed without a weight (binary16 cache)", cache(v16), want_v, 1e-3);
  if (d_inv) CK(cudaFree(d_inv));
  for (void* p : {static_cast<void*>(d_qkv), static_cast<void*>(d_qw), static_cast<void*>(d_kw),
                  static_cast<void*>(d_q), static_cast<void*>(k16), static_cast<void*>(v16), static_cast<void*>(dP)})
    CK(cudaFree(p));
}

// A prefill pass at head dim 512 (attention: the split kernel 16 rows at a time): each row's output the bits of that
// row alone, as decoding it gives.
void prefill_512_case() {
  const int nh = 16, nkv = 1, hd = 512, rows = 40, ctx = 90;
  seed("gemma4_prefill_512", nh, nkv, hd, rows, ctx);
  const int cap = ctx + rows + 16;
  Vec q = randv(size_t(rows) * nh * hd), Kc = randv(size_t(cap) * nkv * hd), Vc = randv(size_t(cap) * nkv * hd);
  round_f16(Kc);
  round_f16(Vc);
  float *dq = up(q), *dout = dev<float>(q.size()), *done = dev<float>(q.size());
  kv16 *dK = up_f16(Kc), *dV = up_f16(Vc);
  DevParams P{};
  P.position = ctx;
  DevParams* dP = dev<DevParams>(1);
  CK(cudaMemcpy(dP, &P, sizeof(P), cudaMemcpyHostToDevice));
  float* part = dev<float>(attention_partial_floats(FLYX_MAX_ROWS, nh, hd, kPrefills));
  const KvRef kv{dK, dV, nullptr, nullptr};
  attention(dq, kv, nh, nkv, hd, rows, true, 0, 1.f, dP, nullptr, 0, dout, nh * hd, part, kPrefills, TL);
  for (int r = 0; r < rows; ++r)  // a row at a time, at its own position: what decoding it gives
    attention(dq + size_t(r) * nh * hd, kv, nh, nkv, hd, 1, true, 0, 1.f, dP, nullptr, 0, done + size_t(r) * nh * hd,
              nh * hd, part, kPrefills, TL, r);
  expect("head dim 512, a prefill pass of 40 rows: each row decoding's bits", down(dout, q.size()),
         down(done, q.size()), 0);
  for (void* p : {static_cast<void*>(dq), static_cast<void*>(dout), static_cast<void*>(done), static_cast<void*>(dK),
                  static_cast<void*>(dV), static_cast<void*>(dP), static_cast<void*>(part)})
    CK(cudaFree(p));
}

}  // namespace

void test_gemma4() {
  std::printf("gemma4\n");
  norm_add_case();
  scale_rows_case();
  glu_act_case();
  prep_case(true);
  prep_case(false);
  prefill_512_case();
}

}  // namespace flyx::optest
