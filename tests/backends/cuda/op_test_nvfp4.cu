// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: NVFP4: quantization, the decoder, the GEMMs (W4A4 included).
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

void test_quantize_nvfp4() {
  std::printf("nvfp4 quantization\n");
  seed("nvfp4 quantization");
  const int N = 37, K = 96;
  Vec w = randv(size_t(N) * K);
  for (size_t i = 0; i < w.size(); i += 7) w[i] *= 20.0;  // a spread of block maxima
  w[5 * K + 3] = 0.0;
  for (int k = 0; k < K; ++k) w[9 * K + k] = 0.0;         // an all-zero row
  round_bf16(w);
  bf16* dw = up_bf16(w);
  uint8_t *dq = dev<uint8_t>(size_t(N) * K / 2), *ds = dev<uint8_t>(size_t(N) * K / 16);
  float* drs = dev<float>(N);
  quantize_nvfp4(dw, N, K, dq, ds, drs, 0);
  std::vector<uint8_t> q(size_t(N) * K / 2), sc(size_t(N) * K / 16);
  std::vector<float> rs(N);
  CK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sc.data(), ds, sc.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rs.data(), drs, rs.size() * 4, cudaMemcpyDeviceToHost));
  // Every value lands within half a step of the grid at its block's scale:
  // at most (6 - 4) / 2 = 1 scale unit, plus the e4m3 rounding of the scale.
  double worst = 0;
  for (int n = 0; n < N; ++n)
    for (int b = 0; b < K / 16; ++b) {
      double bmax = 0;
      for (int e = 0; e < 16; ++e) bmax = std::max(bmax, std::fabs(w[size_t(n) * K + b * 16 + e]));
      const double scale = ops::e4m3_to_double(sc[size_t(n) * (K / 16) + b]) * rs[n];
      for (int e = 0; e < 16; ++e) {
        const int k = b * 16 + e;
        const uint8_t byte = q[size_t(n) * (K / 2) + k / 2];
        const double v = ops::e2m1_to_double((k & 1) ? byte >> 4 : byte & 0xF) * scale;
        if (bmax > 0) worst = std::max(worst, std::fabs(v - w[size_t(n) * K + k]) / bmax);
        else worst = std::max(worst, std::fabs(v));
      }
    }
  const double limit = 1.0 / 6 * 1.07;
  const bool ok = worst <= limit;
  std::printf("  %-4s %-48s %.3f (limit %.3f)\n", ok ? "ok" : "FAIL", "round trip: |error| / block max", worst, limit);
  if (!ok) ++g_fail;
}

void test_gemm_w4a4(int M, int N, int K) {
  std::printf("W4A4 prefill GEMM (%d x %d x %d)\n", M, N, K);
  seed("w4a4", M, N, K);
  if (!gemm_w4a4_supported()) {
    std::printf("  (not on this device or build)\n");
    return;
  }
  const float E2M1v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  auto e2m1v = [&](uint8_t c) { return (c & 8 ? -1.0 : 1.0) * E2M1v[c & 7]; };
  auto e4m3v = [](uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    return e == 0 ? m * std::ldexp(1.0, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
  };
  // activations as the float32 rows the kernels hold (rounded once to NVFP4), weights as random NVFP4;
  // the last two rows reach e4m3's subnormal block scales (blocks a million times smaller than the row's largest)
  // and spread over 30 binades
  Vec x = randv(size_t(M) * K);
  for (int k = 0; k < K; ++k) {
    if ((k / 16) % 2) x[size_t(M - 1) * K + k] *= 1e-6;
    x[size_t(M - 2) * K + k] *= std::ldexp(1.0, -(k % 30));
  }
  std::vector<float> xf(x.size());
  for (size_t i = 0; i < x.size(); ++i) xf[i] = float(x[i]), x[i] = xf[i];
  std::vector<uint8_t> Qh(size_t(N) * K / 2), Sh(size_t(N) * K / 16);
  std::vector<float> RSh(N);
  std::uniform_int_distribution<int> byte(0, 255), sc(0x28, 0x40);
  for (auto& b : Qh) b = uint8_t(byte(rng));
  for (auto& b : Sh) b = uint8_t(sc(rng));
  std::uniform_real_distribution<float> rs(0.001f, 0.01f);
  for (auto& v : RSh) v = rs(rng);
  // host quantization, as quantize_rows_nvfp4 does it
  std::vector<uint8_t> qh(size_t(M) * K / 2, 0), sh(size_t(M) * K / 16);
  std::vector<float> invh(M);
  for (int r = 0; r < M; ++r) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(float(x[size_t(r) * K + k])));
    const float gs = amax > 0.f ? kNvfp4RowRange / amax : 1.f;
    invh[r] = 1.f / gs;
    for (int b = 0; b < K / 16; ++b) {
      float bmax = 0.f;
      for (int e = 0; e < 16; ++e) bmax = std::max(bmax, std::fabs(float(x[size_t(r) * K + 16 * b + e])));
      const uint8_t s = nvfp4_block_scale(bmax, gs);
      sh[size_t(r) * (K / 16) + b] = s;
      const float sv = float(e4m3v(s)), f = sv > 0.f ? gs / sv : 0.f;
      for (int e = 0; e < 16; ++e) {
        const uint8_t c = e2m1_nearest(float(x[size_t(r) * K + 16 * b + e]) * f);
        qh[size_t(r) * (K / 2) + 8 * b + e / 2] |= uint8_t(c << (4 * (e & 1)));
      }
    }
  }
  // device
  float* dx = dev<float>(xf.size());
  CK(cudaMemcpy(dx, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
  uint8_t *dq = dev<uint8_t>(qh.size()), *ds = dev<uint8_t>(sh.size()), *dQ = dev<uint8_t>(Qh.size()),
          *dS = dev<uint8_t>(Sh.size());
  float *dinv = dev<float>(M), *dRS = dev<float>(N), *dy = dev<float>(size_t(M) * N);
  CK(cudaMemcpy(dQ, Qh.data(), Qh.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS, Sh.data(), Sh.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dRS, RSh.data(), N * 4, cudaMemcpyHostToDevice));
  quantize_rows_nvfp4(dx, K, M, K, dq, ds, dinv, 0);
  gemm_w4a4(dq, ds, dinv, dQ, dS, dRS, N, K, dy, N, M, false, 0);
  std::vector<uint8_t> qd(qh.size()), sd(sh.size());
  std::vector<float> invd(M);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(qd.data(), dq, qd.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sd.data(), ds, sd.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(invd.data(), dinv, M * 4, cudaMemcpyDeviceToHost));
  const bool same = qd == qh && sd == sh && std::memcmp(invd.data(), invh.data(), M * 4) == 0;
  std::printf("  %-4s %-48s\n", same ? "ok" : "FAIL", "activations: the host's codes, scales, row factors");
  if (!same) ++g_fail;
  {  // the reference's rounding (ops/dtype/ref.h: round_nvfp4, what cpu_ref computes with) is the device's
    bool same_ref = true;
    for (int r = 0; r < M; ++r) {
      Vec row(x.begin() + size_t(r) * K, x.begin() + size_t(r + 1) * K);
      ops::round_nvfp4(row.data(), K);
      for (int k = 0; k < K; ++k) {
        const uint8_t c = uint8_t((qd[size_t(r) * (K / 2) + k / 2] >> (4 * (k & 1))) & 15u);
        same_ref = same_ref && row[size_t(k)] == e2m1v(c) * e4m3v(sd[size_t(r) * (K / 16) + k / 16]) * double(invd[r]);
      }
    }
    std::printf("  %-4s %-48s\n", same_ref ? "ok" : "FAIL", "activations: the reference's values (round_nvfp4)");
    if (!same_ref) ++g_fail;
  }
  {  // from bf16 rows with their largest |value| given (the SwiGLU rows): the codes the float32 path gives them
    std::vector<uint16_t> xb(xf.size());
    std::vector<float> xbf(xf.size()), amax(M, 0.f);
    for (size_t i = 0; i < xf.size(); ++i) {
      xb[i] = to_bf16(xf[i]);
      uint32_t u = uint32_t(xb[i]) << 16;
      std::memcpy(&xbf[i], &u, 4);
      amax[i / K] = std::max(amax[i / K], std::fabs(xbf[i]));
    }
    float* dxf = dev<float>(xbf.size());
    CK(cudaMemcpy(dxf, xbf.data(), xbf.size() * 4, cudaMemcpyHostToDevice));
    bf16* dxb = dev<bf16>(xb.size());
    CK(cudaMemcpy(dxb, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice));
    float* damax = dev<float>(M);
    CK(cudaMemcpy(damax, amax.data(), M * 4, cudaMemcpyHostToDevice));
    uint8_t *dq2 = dev<uint8_t>(qh.size()), *ds2 = dev<uint8_t>(sh.size());
    float* dinv2 = dev<float>(M);
    quantize_rows_nvfp4(dxf, K, M, K, dq, ds, dinv, 0);
    quantize_rows_nvfp4_bf16(dxb, K, M, K, damax, dq2, ds2, dinv2, 0);
    std::vector<uint8_t> qa(qh.size()), qb(qh.size()), sa(sh.size()), sb(sh.size());
    std::vector<float> ia(M), ib(M);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(qa.data(), dq, qa.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(qb.data(), dq2, qb.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sa.data(), ds, sa.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sb.data(), ds2, sb.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ia.data(), dinv, M * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ib.data(), dinv2, M * 4, cudaMemcpyDeviceToHost));
    const bool same_bf16 = qa == qb && sa == sb && std::memcmp(ia.data(), ib.data(), M * 4) == 0;
    std::printf("  %-4s %-48s\n", same_bf16 ? "ok" : "FAIL", "bf16 rows, maxima given: the float32 path's codes");
    if (!same_bf16) ++g_fail;
    quantize_rows_nvfp4(dx, K, M, K, dq, ds, dinv, 0);  // the rows below
  }
  // references: the quantized operands, and the activations as they were
  Vec want(size_t(M) * N), plain(size_t(M) * N);
  std::vector<double> wdq(size_t(N) * K), adq(size_t(M) * K);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      wdq[size_t(n) * K + k] = e2m1v((Qh[size_t(n) * K / 2 + k / 2] >> (4 * (k & 1))) & 15) *
                               e4m3v(Sh[size_t(n) * K / 16 + k / 16]) * RSh[n];
  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k)
      adq[size_t(r) * K + k] = e2m1v((qh[size_t(r) * K / 2 + k / 2] >> (4 * (k & 1))) & 15) *
                               e4m3v(sh[size_t(r) * K / 16 + k / 16]) * invh[r];
  for (int r = 0; r < M; ++r)
    for (int n = 0; n < N; ++n) {
      double a = 0, b = 0;
      for (int k = 0; k < K; ++k) {
        a += adq[size_t(r) * K + k] * wdq[size_t(n) * K + k];
        b += x[size_t(r) * K + k] * wdq[size_t(n) * K + k];
      }
      want[size_t(r) * N + n] = a;
      plain[size_t(r) * N + n] = b;
    }
  const Vec got = down(dy, size_t(M) * N);
  expect("product of the quantized operands", got, want, 1e-5);
  for (int cfg = 0; cfg <= 4; ++cfg) {  // every tile configuration: the default's bits
    float* dy2 = dev<float>(size_t(M) * N);
    gemm_w4a4(dq, ds, dinv, dQ, dS, dRS, N, K, dy2, N, M, false, 0, cfg);
    expect("tile configuration " + std::to_string(cfg) + ": same bits", down(dy2, size_t(M) * N), got, 0);
    bf16* dy16 = dev<bf16>(size_t(M) * N);  // bf16 rows: the float32 product rounded once
    gemm_w4a4(dq, ds, dinv, dQ, dS, dRS, N, K, nullptr, N, M, false, 0, cfg, nullptr, nullptr, dy16);
    Vec want16(got.size());
    for (size_t i = 0; i < got.size(); ++i) want16[i] = from_bf16(to_bf16(got[i]));
    expect("tile configuration " + std::to_string(cfg) + ": bf16 rows, the float32 rows rounded",
           down_bf16(dy16, size_t(M) * N), want16, 0);
    // accumulating into bf16 rows (the residual): the float32 sum of the row's value and the product, rounded once
    std::vector<uint16_t> base(got.size());
    Vec want_acc(got.size());
    for (size_t i = 0; i < got.size(); ++i) {
      base[i] = to_bf16(0.25 * std::sin(double(i)));
      want_acc[i] = from_bf16(to_bf16(double(float(from_bf16(base[i])) + float(got[i]))));
    }
    CK(cudaMemcpy(dy16, base.data(), base.size() * 2, cudaMemcpyHostToDevice));
    gemm_w4a4(dq, ds, dinv, dQ, dS, dRS, N, K, nullptr, N, M, true, 0, cfg, nullptr, nullptr, dy16);
    expect("tile configuration " + std::to_string(cfg) + ": bf16 rows accumulated, the float32 sum rounded",
           down_bf16(dy16, size_t(M) * N), want_acc, 0);
  }
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < got.size(); ++i) err = std::max(err, std::fabs(got[i] - plain[i])), scale = std::max(scale, std::fabs(plain[i]));
  std::printf("       against bf16 activations: rel err %.2e (the activations' rounding to NVFP4)\n", err / scale);
}

// NVFP4 (ops::dequant_nvfp4 is the reference): decoder, both GEMM paths,
// and bit-identical results across row counts and configurations.
void test_fp4() {
  std::printf("nvfp4\n");
  seed("nvfp4");
  {  // decoder: every code in every position
    std::vector<uint32_t> in;
    for (uint32_t c = 0; c < 16; ++c)
      for (int pos = 0; pos < 8; ++pos) in.push_back((c << (4 * pos)) | (((c + 5) & 15) << (4 * ((pos + 1) & 7))));
    for (int i = 0; i < 1000; ++i) in.push_back(uint32_t(rng()));
    uint32_t *din, *dout;
    CK(cudaMalloc(&din, in.size() * 4));
    CK(cudaMalloc(&dout, in.size() * 16));
    CK(cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    fp4_decode_test(din, dout, int(in.size()), 0);
    std::vector<uint32_t> out(in.size() * 4);
    CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < in.size(); ++i)
      for (int v = 0; v < 8; ++v) {
        const uint16_t got = uint16_t(out[i * 4 + v / 2] >> (16 * (v & 1)));
        const double want = ops::e2m1_to_double((in[i] >> (4 * v)) & 15);
        uint16_t w = to_bf16(want);
        if (want == 0 && std::signbit(want)) w = 0x8000;
        ok &= got == w;
      }
    std::printf("  %-4s e2m1 -> bf16 decoder (%zu words)\n", ok ? "ok" : "FAIL", in.size());
    g_fail += !ok;
  }
  struct S { int rows, N, K; };
  for (S s : {S{1, 200, 256}, S{5, 72, 64}, S{16, 300, 512}, S{17, 130, 96}, S{64, 64, 2560}, S{3, 520, 2560},
              S{24, 300, 512}, S{40, 136, 1024}, S{70, 72, 512}}) {
    const size_t QB = size_t(s.N) * s.K / 2, SB = size_t(s.N) * s.K / 16;
    std::vector<uint8_t> q(QB), sc(SB);
    for (auto& b : q) b = uint8_t(rng());
    for (auto& b : sc) b = uint8_t(0x28 + rng() % 0x30);  // e4m3 roughly 2^-2 .. 2^4, no NaN
    const float s2 = 0.0123f;
    std::vector<float> rs(s.N, s2);
    flyx_tensor_view tq{"q", FLYX_DT_NVFP4, 2, {s.N, s.K}, q.data(), QB};
    flyx_tensor_view ts{"s", FLYX_DT_F8E4M3, 2, {s.N, s.K / 16}, sc.data(), SB};
    flyx_tensor_view t2{"s2", FLYX_DT_F32, 1, {1}, &s2, 4};
    Vec w = ops::dequant_nvfp4(&tq, &ts, &t2), x = randv(size_t(s.rows) * s.K), want(size_t(s.rows) * s.N);
    // linear_a16 of dequant_nvfp4: every NVFP4 path takes its inputs in bf16, so the reference rounds them first.
    flyx_tensor_view dense{"w", FLYX_DT_F64, 2, {s.N, s.K}, w.data(), w.size() * sizeof(double)};
    ops::linear_a16(x.data(), s.rows, s.K, &dense, s.N, want.data());
    uint8_t *dq = dev<uint8_t>(QB), *ds = dev<uint8_t>(SB);
    CK(cudaMemcpy(dq, q.data(), QB, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ds, sc.data(), SB, cudaMemcpyHostToDevice));
    float* drs = up(Vec(rs.begin(), rs.end()));
    float* dx = up(x);
    float* dy = dev<float>(want.size());
    const std::string tag = std::to_string(s.rows) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);
    if (s.K % 32 == 0) {
      gemm_fp4_mma(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, 0);
      expect("nvfp4 tensor-core path " + tag, down(dy, want.size()), want, 2e-5);
      gemm_fp4_tc(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL);
      const Vec tc0 = down(dy, want.size());
      expect("nvfp4 pipelined tensor-core path " + tag, tc0, want, 2e-5);
      for (int c = 1; c < kTcConfigs; ++c) {
        gemm_fp4_tc(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL, c);
        expect("nvfp4 pipelined configuration " + std::to_string(c) + ": same bits " + tag, down(dy, want.size()), tc0,
               0);
      }
    }
    if (s.rows <= 16 && s.K % 64 == 0) {
      gemm_fp4(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL);
      Vec ref = down(dy, want.size());
      expect("nvfp4 skinny path (bf16 activations) " + tag, ref, want, 2e-5);
      bool same = true;
      for (int r = 0; r < s.rows; ++r) {
        gemm_fp4(dx + size_t(r) * s.K, s.K, dq, ds, drs, s.N, s.K, dy, s.N, 1, false, TL);
        Vec one = down(dy, s.N);
        same &= std::equal(one.begin(), one.end(), ref.begin() + size_t(r) * s.N);
      }
      // Within one K partition (span, chunks), every compiled configuration and row count gives the same bits.
      for (int cfg = 0; cfg < kFp4ConfigCount; ++cfg) {
        const auto& c = kFp4Configs[cfg];
        if (s.K % c.span || !gemm_fp4_cfg_compiled(cfg, s.rows)) continue;
        int first = cfg;  // the first compiled configuration with this partition
        for (int f = 0; f < cfg; ++f)
          if (kFp4Configs[f].span == c.span && kFp4Configs[f].chunks == c.chunks &&
              gemm_fp4_cfg_compiled(f, s.rows)) {
            first = f;
            break;
          }
        gemm_fp4_skinny_cfg(first, dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL);
        Vec base = down(dy, want.size());
        gemm_fp4_skinny_cfg(cfg, dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL);
        same &= down(dy, want.size()) == base;
        if (gemm_fp4_cfg_compiled(cfg, 1))
          for (int r : {0, s.rows - 1}) {
            gemm_fp4_skinny_cfg(cfg, dx + size_t(r) * s.K, s.K, dq, ds, drs, s.N, s.K, dy, s.N, 1, false, TL);
            Vec one = down(dy, s.N);
            same &= std::equal(one.begin(), one.end(), base.begin() + size_t(r) * s.N);
          }
      }
      std::printf("  %-4s nvfp4 skinny %s: same bits for every row count, and per K partition for every configuration\n",
                  same ? "ok" : "FAIL", tag.c_str());
      g_fail += !same;
    }
    if (s.rows > 16 && s.K % 64 == 0) {  // a round of many rows: the bits of 16 rows at a time, every configuration
      gemm_fp4_rows(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL);
      expect("nvfp4 many-row round path " + tag, down(dy, want.size()), want, 2e-5);
      bool same = true;
      int tried = 0;
      for (int cfg = 0; cfg < kFp4ConfigCount; ++cfg) {
        if (s.K % kFp4Configs[cfg].span || !gemm_fp4_cfg_compiled(cfg, 16)) continue;
        gemm_fp4_rows(dx, s.K, dq, ds, drs, s.N, s.K, dy, s.N, s.rows, false, TL, cfg);
        const Vec many = down(dy, want.size());
        for (int r0 = 0; r0 < s.rows; r0 += 16) {
          const int n = std::min(16, s.rows - r0);
          if (!gemm_fp4_cfg_compiled(cfg, n)) continue;
          gemm_fp4_skinny_cfg(cfg, dx + size_t(r0) * s.K, s.K, dq, ds, drs, s.N, s.K, dy, s.N, n, false, TL);
          const Vec part = down(dy, size_t(n) * s.N);
          same &= std::equal(part.begin(), part.end(), many.begin() + size_t(r0) * s.N);
        }
        ++tried;
      }
      std::printf("  %-4s nvfp4 many-row rounds %s: the bits of 16 rows at a time (%d configurations)\n",
                  same ? "ok" : "FAIL", tag.c_str(), tried);
      g_fail += !same;
    }
  }
}

}  // namespace flyx::optest
