// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The float64 references sum several outputs side by side (ops/linear/ref.h, ops/gdn/ref.h) only so that the CPU
// overlaps their chains of additions: every output must get the bits of its own sum computed alone, in order.
// Random inputs, bf16 and float64 weights, one row and several, sizes that leave a remainder.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ops/gdn/ref.h"
#include "ops/linear/ref.h"

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// What linear() means: one output at a time.
void linear_alone(const double* x, int rows, int K, const flyx_tensor_view* W, int N, double* y) {
  for (int r = 0; r < rows; ++r)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += x[size_t(r) * K + k] * flyx::ops::load(W, size_t(n) * K + k);
      y[size_t(r) * N + n] = acc;
    }
}

// What gdn_row() means: one row of the state at a time.
void gdn_row_alone(double* S, const double* q, const double* k, const double* v, double beta, double g, int dk,
                   int dv, double* out) {
  const double decay = std::exp(g);
  for (int j = 0; j < dv; ++j) {
    double* s = S + size_t(j) * dk;
    double mem = 0;
    for (int i = 0; i < dk; ++i) {
      s[i] *= decay;
      mem += s[i] * k[i];
    }
    const double delta = (v[j] - mem) * beta;
    double o = 0;
    for (int i = 0; i < dk; ++i) {
      s[i] += k[i] * delta;
      o += s[i] * q[i];
    }
    out[j] = o;
  }
}

uint16_t bf16_bits(double v) {
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1, 1);
  for (int K : {64, 96, 256})
    for (int N : {512, 37, 3})
      for (int rows : {1, 5})
        for (int dtype : {FLYX_DT_BF16, FLYX_DT_F64}) {
          std::vector<double> wd(size_t(N) * K);
          std::vector<uint16_t> wb(wd.size());
          for (size_t i = 0; i < wd.size(); ++i) wd[i] = U(rng), wb[i] = bf16_bits(wd[i]);
          flyx_tensor_view W{};
          W.dtype = dtype;
          W.data = dtype == FLYX_DT_BF16 ? static_cast<void*>(wb.data()) : static_cast<void*>(wd.data());
          W.ndim = 2;
          W.shape[0] = N;
          W.shape[1] = K;
          std::vector<double> x(size_t(rows) * K), got(size_t(rows) * N), want(got.size());
          for (auto& e : x) e = 3 * U(rng);
          flyx::ops::linear(x.data(), rows, K, &W, N, got.data());
          linear_alone(x.data(), rows, K, &W, N, want.data());
          check(same_bits(got, want), "linear " + std::to_string(rows) + "x" + std::to_string(K) + "x" +
                                          std::to_string(N) + (dtype == FLYX_DT_BF16 ? " bf16" : " f64") +
                                          ": the bits of one output at a time");
        }
  for (int dk : {128, 64})
    for (int dv : {128, 66, 3}) {
      std::vector<double> S(size_t(dv) * dk), q(dk), k(dk), v(dv), got(dv), want(dv);
      for (auto& e : S) e = 0.1 * U(rng);
      for (int i = 0; i < dk; ++i) q[i] = U(rng), k[i] = 0.2 * U(rng);
      for (auto& e : v) e = U(rng);
      std::vector<double> S_alone = S;
      bool same = true;
      for (int step = 0; step < 5; ++step) {
        flyx::ops::gdn_row(S.data(), q.data(), k.data(), v.data(), 0.4, -0.03, dk, dv, got.data());
        gdn_row_alone(S_alone.data(), q.data(), k.data(), v.data(), 0.4, -0.03, dk, dv, want.data());
        same = same && same_bits(got, want) && same_bits(S, S_alone);
      }
      check(same, "gdn_row " + std::to_string(dv) + "x" + std::to_string(dk) +
                      ", 5 steps: the state and outputs of one row at a time");
    }
  std::printf(failures ? "FAILED (%d failures)\n" : "PASSED (0 failures)\n", failures);
  return failures != 0;
}
