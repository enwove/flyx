// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: the output head: certified, FP8, W12, the drafts' pick.
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

// The FP8 output head (option head_fp8): codes as the host rounds them, the
// product against float64 of the rounded weights, the same bits for a row
// whatever rows run with it (1..16).
// The certified head: after gemm_fp8_skinny and head_certify,
// every finite entry holds gemm_skinny's bits and each row's largest entry
// (lowest index on ties) is gemm_skinny's over the whole vocabulary.  The
// weights hold exact duplicate rows (ties) and rows one bf16 step apart
// (near ties), and some activation rows point at them; the bound held at
// every entry (|exact - copy| <= ||x|| coef).
void test_head_cert(int V, int K) {
  std::printf("certified head (%d x %d)\n", V, K);
  seed("certified head", V, K);
  Vec w = randv(size_t(V) * K, 0.05);
  std::vector<uint16_t> wb(w.size());
  for (size_t i = 0; i < w.size(); ++i) wb[i] = to_bf16(w[i]);
  for (int d = 0; d < 32; ++d) {  // rows 64 d + 1 = 64 d (a tie); rows 64 d + 3 = 64 d + 2 but one step in one weight
    const size_t a = size_t(64 * d) * K;
    std::copy(wb.begin() + a, wb.begin() + a + K, wb.begin() + a + K);
    std::copy(wb.begin() + a + 2 * K, wb.begin() + a + 3 * K, wb.begin() + a + 3 * K);
    wb[a + 3 * K + d % K] += 1;
  }
  bf16* dw = dev<bf16>(wb.size());
  CK(cudaMemcpy(dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
  uint8_t* dq = dev<uint8_t>(size_t(V) * K);
  float *drs = dev<float>(V), *coef = dev<float>(V);
  quantize_fp8_rows(dw, V, K, dq, drs, 0);
  head_cert_coef(dw, dq, drs, V, K, coef, 0);
  const std::vector<float> cf = [&] {
    Vec c = down(coef, V);
    return std::vector<float>(c.begin(), c.end());
  }();
  float* hn = dev<float>(16);
  int *key = dev<int>(16), *cols = dev<int>(V), *count = dev<int>(1);
  for (int rows : {1, 5, 16}) {
    Vec x = randv(size_t(rows) * K);
    for (int r = 0; r < rows; r += 2)  // point at a tied or nearly tied pair: their scores lead
      for (int k = 0; k < K; ++k) x[size_t(r) * K + k] += 40.0 * from_bf16(wb[size_t(64 * (r / 2) + (r & 2)) * K + k]);
    float *dx = up(x), *full = dev<float>(size_t(rows) * V), *got = dev<float>(size_t(rows) * V);
    gemm_skinny(dx, K, dw, V, K, full, V, rows, false, TL);
    gemm_fp8_skinny(dx, K, dq, drs, V, K, got, V, rows, false, TL);
    const Vec approx = down(got, size_t(rows) * V);
    head_certify(dx, K, dw, coef, V, K, got, V, rows, hn, key, cols, count, TL);
    const Vec f = down(full, size_t(rows) * V), c = down(got, size_t(rows) * V);
    const int survivors = down_i(count, 1)[0];
    bool bound = true, bits = true, choice = true;
    for (int r = 0; r < rows; ++r) {
      double n2 = 0;
      for (int k = 0; k < K; ++k) n2 += x[size_t(r) * K + k] * x[size_t(r) * K + k];
      int bf = 0, bc = 0;
      for (int n = 0; n < V; ++n) {
        const size_t i = size_t(r) * V + n;
        bound &= std::fabs(f[i] - approx[i]) <= std::sqrt(n2) * cf[n];
        const float fv = float(f[i]), cv = float(c[i]);
        bits &= std::isinf(cv) ? cv < 0 : std::memcmp(&fv, &cv, 4) == 0;
        if (f[i] > f[size_t(r) * V + bf]) bf = n;
        if (c[i] > c[size_t(r) * V + bc]) bc = n;
      }
      choice &= bf == bc;
    }
    const std::string tag = std::to_string(rows) + " rows, " + std::to_string(survivors) + " survivors";
    std::printf("  %-4s %-48s\n", bound ? "ok" : "FAIL", ("the bound holds at every entry, " + tag).c_str());
    std::printf("  %-4s %-48s\n", bits ? "ok" : "FAIL", "survivors: gemm_skinny's bits, the others -inf");
    std::printf("  %-4s %-48s\n", choice ? "ok" : "FAIL", "every row's choice is gemm_skinny's");
    g_fail += !bound + !bits + !choice;
  }
}

void test_gemm_fp8(int N, int K) {
  std::printf("FP8 head GEMM (%d x %d)\n", N, K);
  seed("fp8 head", N, K);
  Vec w = randv(size_t(N) * K, 0.05);
  std::vector<uint16_t> wb(w.size());
  for (size_t i = 0; i < w.size(); ++i) wb[i] = to_bf16(w[i]), w[i] = from_bf16(wb[i]);
  bf16* dw = dev<bf16>(wb.size());
  CK(cudaMemcpy(dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
  uint8_t* dq = dev<uint8_t>(size_t(N) * K);
  float* drs = dev<float>(N);
  quantize_fp8_rows(dw, N, K, dq, drs, 0);
  std::vector<uint8_t> q(size_t(N) * K);
  std::vector<float> rs(N);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(q.data(), dq, q.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rs.data(), drs, N * 4, cudaMemcpyDeviceToHost));
  bool same = true;
  std::vector<double> wq(size_t(N) * K);
  for (int n = 0; n < N; ++n) {
    float amax = 0.f;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(float(w[size_t(n) * K + k])));
    const float r = amax > 0.f ? amax / 448.f : 1.f;
    same = same && std::memcmp(&r, &rs[n], 4) == 0;
    for (int k = 0; k < K; ++k) {
      const uint8_t c = uint8_t(__nv_cvt_float_to_fp8(float(w[size_t(n) * K + k]) / r, __NV_SATFINITE, __NV_E4M3));
      same = same && c == q[size_t(n) * K + k];
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(c), __NV_E4M3);
      wq[size_t(n) * K + k] = double(__half2float(__half(hr))) * r;
    }
  }
  std::printf("  %-4s %-48s\n", same ? "ok" : "FAIL", "codes and row scales as the host rounds them");
  if (!same) ++g_fail;
  Vec all;
  for (int rows : {1, 5, 16}) {
    Vec x = randv(size_t(rows) * K), want(size_t(rows) * N);
    // where K % 256 == 0 the rows enter as bf16 (k_gemm_fp8_sliced), else as three bf16 parts (exact)
    const bool bf_rows = K % 256 == 0;
    for (int r = 0; r < rows; ++r)
      for (int n = 0; n < N; ++n) {
        double a = 0;
        for (int k = 0; k < K; ++k) {
          const double xv = bf_rows ? from_bf16(to_bf16(x[size_t(r) * K + k])) : x[size_t(r) * K + k];
          a += xv * wq[size_t(n) * K + k];
        }
        want[size_t(r) * N + n] = a;
      }
    float *dx = up(x), *dy = dev<float>(want.size());
    gemm_fp8_skinny(dx, K, dq, drs, N, K, dy, N, rows, false, TL);
    const Vec got = down(dy, want.size());
    expect("rows " + std::to_string(rows) + ": product of the rounded weights", got, want, 2e-5);
    if (rows == 16) {  // row r alone gives the bits it got with 15 others
      float* dy1 = dev<float>(N);
      gemm_fp8_skinny(dx + size_t(5) * K, K, dq, drs, N, K, dy1, N, 1, false, TL);
      expect("a row alone: the bits it gets among 16", down(dy1, N), Vec(got.begin() + 5 * N, got.begin() + 6 * N), 0);
    }
  }
}

// W12 (the head's lossless 12-bit copy): every weight decodes to its bf16
// bits, escapes included (zeros, denormals, exponents outside the row's
// window, in-window weights whose code is the marker), and the GEMM gives the
// bits of the bf16 skinny GEMM for every row count and configuration.
void test_gemm_w12(int N, int K) {
  std::printf("W12 head GEMM (%d x %d)\n", N, K);
  seed("w12 head", N, K);
  Vec w = randv(size_t(N) * K, 0.05);
  std::vector<uint16_t> wb(w.size());
  for (size_t i = 0; i < w.size(); ++i) wb[i] = to_bf16(w[i]);
  // the window the encoder picks (the most weights in 16 binades, the lowest first)
  auto window = [&](int n) {
    int hist[256] = {};
    for (int k = 0; k < K; ++k) ++hist[(wb[size_t(n) * K + k] >> 7) & 0xFF];
    int best = -1, e0 = 1;
    for (int e = 1; e <= 239; ++e) {
      int c = 0;
      for (int j = e; j < e + 16; ++j) c += hist[j];
      if (c > best) best = c, e0 = e;
    }
    return e0;
  };
  int specials = 0;
  for (int n = 0; n < N; n += 7) {  // every 7th row: a zero, a denormal, a huge, a tiny and a marker-coded weight
    const size_t r = size_t(n) * K;
    const int e0 = window(n);
    wb[r + (n * 13) % K] = 0x0000;
    wb[r + (n * 17 + 1) % K] = 0x8003;                            // -denormal
    wb[r + (n * 19 + 2) % K] = uint16_t(((e0 + 20) & 0xFF) << 7 | 0x15);   // above the window
    wb[r + (n * 23 + 3) % K] = uint16_t(0x8000 | std::max(1, e0 - 9) << 7);  // below it
    wb[r + (n * 29 + 4) % K] = uint16_t((e0 + 15) << 7 | 0x7F);  // in the window, the marker's code
    specials += 5;
  }
  bf16* dw = dev<bf16>(wb.size());
  CK(cudaMemcpy(dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
  W12 c;
  c.hi = dev<uint8_t>(size_t(N) * K);
  c.lo = dev<uint8_t>(size_t(N) * K / 2);
  c.e0 = dev<uint8_t>(N);
  c.esc_start = dev<int>(size_t(N) + 1);
  std::vector<int> starts(size_t(N) + 1);
  const int esc = w12_encode(dw, N, K, &c, starts.data(), 0);
  c.esc_k = dev<uint16_t>(std::max(esc, 1));
  c.esc_v = dev<uint16_t>(std::max(esc, 1));
  const int bad = w12_finish(dw, N, K, &c, 0);
  int want_esc = 0;  // the host's count: outside the row's window, or the marker's code
  for (int n = 0; n < N; ++n) {
    const int e0 = window(n);
    for (int k = 0; k < K; ++k) {
      const int b = wb[size_t(n) * K + k], e = (b >> 7) & 0xFF, off = e - e0;
      want_esc += !(off >= 0 && off < 16 && !(off == 15 && (b & 0x7F) == 0x7F));
    }
  }
  const bool ok = bad == 0 && esc == want_esc;
  std::printf("  %-4s %-48s (%d escapes, %d expected, %d special weights placed)\n", ok ? "ok" : "FAIL",
              "every weight decodes to its bf16 bits", esc, want_esc, specials);
  if (!ok) ++g_fail;
  for (int cfg = 0; cfg < kSkinnyConfigs; ++cfg)
    for (int rows : {1, 3, 9, 16}) {
      if (!gemm_skinny_cfg_compiled(cfg, rows, false) || !gemm_skinny_cfg_compiled(cfg, rows, true)) continue;
      const Vec x = randv(size_t(rows) * K);
      float *dx = up(x), *dy = dev<float>(size_t(rows) * N), *dz = dev<float>(size_t(rows) * N);
      gemm_skinny_cfg(cfg, dx, K, dw, N, K, dy, N, rows, false, TL);
      gemm_w12_skinny_cfg(cfg, dx, K, c, N, K, dz, N, rows, false, TL);
      expect("configuration " + std::to_string(cfg) + ", rows " + std::to_string(rows) + ": the bf16 GEMM's bits",
             down(dz, size_t(rows) * N), down(dy, size_t(rows) * N), 0);
    }
}

// The drafts' two-stage head (draft_pick): the 1-bit copy as the host makes
// it, rough scores against float64 of its weights, and the picked top k equal
// to the exact top k over the whole vocabulary when the best tokens stand out
// (planted rows) or when every token is a candidate (V <= kDraftCands).  pick false: the rough scores only.
void test_draft_pick(int V, int K, bool plant, bool pick) {
  std::printf("draft_pick (%d x %d%s%s)\n", V, K, plant ? ", planted best rows" : "", pick ? "" : ", rough scores");
  seed("draft_pick", V, K, plant ? 1 : 0);
  Vec w = randv(size_t(V) * K, 0.05);
  const int rows = 5, k = 16;
  Vec x = randv(size_t(rows) * K);
  if (plant)  // for each row, 16 tokens whose weights follow the row's x (including 4 consecutive IDs in one float4)
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < k; ++j) {
        const int n = j < 4 ? (r * 976 + 8 + j) % V : (r * 977 + j * 131 + 7) % V;
        for (int q = 0; q < K; ++q) w[size_t(n) * K + q] = (0.2 + 0.01 * j) * x[size_t(r) * K + q] / 3.0;
      }
  std::vector<uint16_t> wb(w.size());
  for (size_t i = 0; i < w.size(); ++i) wb[i] = to_bf16(w[i]), w[i] = from_bf16(wb[i]);
  bf16* dw = dev<bf16>(wb.size());
  CK(cudaMemcpy(dw, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
  SignHead h;
  const size_t tiled = size_t(V + 255) / 256 * 256;
  h.bits = dev<uint32_t>(tiled * K / 32);
  h.scales = dev<bf16>(tiled * K / 64);
  make_sign_head(dw, V, K, &h, 0);
  std::vector<uint32_t> bits(tiled * K / 32);
  std::vector<uint16_t> sc(tiled * K / 64);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(bits.data(), h.bits, bits.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sc.data(), h.scales, sc.size() * 2, cudaMemcpyDeviceToHost));
  bool same = true;
  std::vector<double> wq(w.size());
  for (int n = 0; n < V; ++n)
    for (int g = 0; g < K / 64; ++g) {
      float s = 0.f;
      for (int j = 0; j < 64; ++j) s += std::fabs(float(w[size_t(n) * K + 64 * g + j]));
      const uint16_t sb = to_bf16(s / 64);
      same = same && sb == sc[sign_scale_at(n, g, K)];
      for (int j = 0; j < 64; ++j) {
        const size_t i = size_t(n) * K + 64 * g + j;
        const bool pos = (wb[i] >> 15) == 0;
        const int k = 64 * g + j;
        same = same && pos == bool((bits[sign_word_at(n, k / 32, K)] >> (k % 32)) & 1);
        wq[i] = (pos ? 1.0 : -1.0) * from_bf16(sb);
      }
    }
  std::printf("  %-4s %-48s\n", same ? "ok" : "FAIL", "signs and scales as the host makes them");
  if (!same) ++g_fail;
  float *dx = up(x), *rough = dev<float>(size_t(rows) * std::max(V, kDraftPickFloats));
  gemm_sign_rough(dx, K, h, V, K, rough, V, rows, TL);
  Vec want(size_t(rows) * V);
  for (int r = 0; r < rows; ++r)
    for (int n = 0; n < V; ++n) {
      double a = 0;
      for (int q = 0; q < K; ++q) a += from_bf16(to_bf16(x[size_t(r) * K + q])) * wq[size_t(n) * K + q];
      want[size_t(r) * V + n] = a;
    }
  expect("rough scores, 5 rows (bf16 activations, tensor cores)", down(rough, want.size()), want, 1e-4);
  {  // 2 to 8 rows (the head's columns as the tensor cores' 16 rows): each row the bits the 9-16-row kernel gives it
    const int big = 12;
    const Vec xb = randv(size_t(big) * K);
    float *dxb = up(xb), *yb = dev<float>(size_t(big) * V), *ys = dev<float>(size_t(big) * V);
    gemm_sign_rough(dxb, K, h, V, K, yb, V, big, TL);
    bool bits_same = true;
    for (int r : {2, 7, 8}) {
      gemm_sign_rough(dxb, K, h, V, K, ys, V, r, TL);
      std::vector<uint32_t> a(size_t(r) * V), b(size_t(r) * V);
      CK(cudaMemcpy(a.data(), ys, a.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost));
      bits_same = bits_same && a == b;
    }
    std::printf("  %-4s %-48s\n", bits_same ? "ok" : "FAIL", "rough scores of 2, 7, 8 rows: the 12-row kernel's bits");
    g_fail += !bits_same;
    for (float* p : {dxb, yb, ys}) CK(cudaFree(p));
  }
  {  // one row: float32 activations on CUDA cores (sums from 4-value tables), and multi-row exact_1row bit identity
    Vec want1(V);
    for (int n = 0; n < V; ++n) {
      double a = 0;
      for (int q = 0; q < K; ++q) a += x[q] * wq[size_t(n) * K + q];
      want1[n] = a;
    }
    gemm_sign_rough(dx, K, h, V, K, rough, V, 1, TL);
    expect("rough scores, 1 row (float32 activations)", down(rough, V), want1, 1e-4);

    // Multi-row exact_1row (2, 3, 4, 5 rows) must produce the exact bits of running rows == 1 on each row.
    std::vector<uint32_t> ref_bits(size_t(rows) * V), got_bits(size_t(rows) * V);
    for (int r = 0; r < rows; ++r) {
      gemm_sign_rough(dx + size_t(r) * K, K, h, V, K, rough, V, 1, TL);
      CK(cudaMemcpy(ref_bits.data() + size_t(r) * V, rough, size_t(V) * 4, cudaMemcpyDeviceToHost));
    }
    bool exact_same = true;
    for (int mr : {2, 3, 4, rows}) {
      gemm_sign_rough(dx, K, h, V, K, rough, V, mr, TL, true);
      CK(cudaMemcpy(got_bits.data(), rough, size_t(mr) * V * 4, cudaMemcpyDeviceToHost));
      exact_same = exact_same && std::equal(got_bits.begin(), got_bits.begin() + size_t(mr) * V, ref_bits.begin());
    }
    std::printf("  %-4s %-48s\n", exact_same ? "ok" : "FAIL", "rough scores, exact_1row (2..5 rows): 1-row bits");
    g_fail += !exact_same;
  }
  if (!pick) return;
  int* di = dev<int>(size_t(rows) * k);
  float* dv = dev<float>(size_t(rows) * k);
  float* dp = dev<float>(size_t(rows));
  float* dm = dev<float>(size_t(rows));
  draft_pick(dx, K, h, dw, V, K, rows, k, di, dv, rough, TL, false, dp, dm);
  std::vector<int> gi(size_t(rows) * k);
  CK(cudaMemcpy(gi.data(), di, gi.size() * 4, cudaMemcpyDeviceToHost));
  const Vec gv = down(dv, size_t(rows) * k);
  const Vec gp = down(dp, size_t(rows));
  const Vec gm = down(dm, size_t(rows));
  {  // Partitioned select (with Scratch::q8) must match single-block k_draft_select (no q8) bit-for-bit.
    Scratch no_q8 = *TL.scratch;
    no_q8.q8 = nullptr;
    no_q8.q8_round_cap = 0;
    Launch no_sc = TL;
    no_sc.scratch = &no_q8;
    int* di2 = dev<int>(size_t(rows) * k);
    float* dv2 = dev<float>(size_t(rows) * k);
    draft_pick(dx, K, h, dw, V, K, rows, k, di2, dv2, rough, no_sc, false);
    std::vector<int> gi2(size_t(rows) * k);
    CK(cudaMemcpy(gi2.data(), di2, gi2.size() * 4, cudaMemcpyDeviceToHost));
    const bool sel_same = gi == gi2 && down(dv2, size_t(rows) * k) == gv;
    std::printf("  %-4s %-48s\n", sel_same ? "ok" : "FAIL", "partitioned draft_select matches single-block bit-for-bit");
    g_fail += !sel_same;
    CK(cudaFree(di2));
    CK(cudaFree(dv2));
  }
  bool order = true, prob_ok = true;
  Vec wantv(size_t(rows) * k);
  Vec wantm(static_cast<size_t>(rows)), wantp(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    std::vector<std::pair<double, int>> ex(V);
    for (int n = 0; n < V; ++n) {
      double a = 0;
      for (int q = 0; q < K; ++q) a += x[size_t(r) * K + q] * w[size_t(n) * K + q];
      ex[n] = {-a, n};
    }
    std::partial_sort(ex.begin(), ex.begin() + k, ex.end());
    for (int j = 0; j < k; ++j) {
      order = order && gi[size_t(r) * k + j] == ex[j].second;
      wantv[size_t(r) * k + j] = -ex[j].first;
    }
    wantm[size_t(r)] = (-ex[0].first) - (-ex[1].first);
    double z = 0;
    for (int n = 0; n < V; ++n) z += std::exp((-ex[n].first) - (-ex[0].first));
    wantp[size_t(r)] = 1.0 / z;
    prob_ok = prob_ok && gp[size_t(r)] > 0.0 && gp[size_t(r)] <= 1.0 + 1e-6 &&
              gp[size_t(r)] >= wantp[size_t(r)] - 1e-4;
  }
  std::printf("  %-4s %-48s\n", order ? "ok" : "FAIL", "the exact top 16 over the whole vocabulary, in order");
  if (!order) ++g_fail;
  expect("their exact logits", gv, wantv, 1e-5);
  expect("top-1 / top-2 margin", gm, wantm, 1e-4);
  if (V <= kDraftCandCap) expect("truncated softmax p_max (all rows scored)", gp, wantp, 1e-5);
  else {
    std::printf("  %-4s %-48s\n", prob_ok ? "ok" : "FAIL", "truncated softmax p_max bounds");
    g_fail += !prob_ok;
  }

  // Shared additive/softcap two-stage candidate-head primitives: gemm_sign_rough_add, draft_score_bf16 (accum),
  // draft_combine_cands, and draft_pick_topk (run once on the small test case).
  if (V <= kDraftCandCap) {
    auto up_raw = [](const std::vector<float>& v) {
      float* p = dev<float>(v.size());
      CK(cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
      return p;
    };
    auto down_raw = [](const float* p, size_t n) {
      CK(cudaDeviceSynchronize());
      std::vector<float> v(n);
      CK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost));
      return v;
    };
    auto softcap_ref = [](double z, double cap) { return cap > 0.0 ? cap * std::tanh(z / cap) : z; };
    auto test_score = [&](int Rk_c, bool accum, bool with_bias, float cap_c, const char* tag) {
      constexpr int Vc = 1024, C = 173;
      std::vector<int> c_ids(C);
      for (int c = 0; c < C; ++c) c_ids[c] = (c * 5 + 3) % Vc;
      Vec c_base = randv(C, 2.0), mk_c = randv(Rk_c, 0.8);
      Vec w2_c = randv(size_t(Vc) * Rk_c, 0.2), w2b_c = randv(Vc, 0.3);
      Vec w2_q(size_t(Vc) * Rk_c);
      for (size_t i = 0; i < w2_c.size(); ++i) w2_q[i] = from_bf16(to_bf16(w2_c[i]));
      std::vector<float> h_row(kDraftPickFloats, 0.f);
      int* h_ci = reinterpret_cast<int*>(h_row.data());
      for (int c = 0; c < C; ++c) {
        h_ci[c] = c_ids[c];
        h_row[kDraftCandCap + c] = float(c_base[c]);
      }
      h_ci[2 * kDraftCandCap] = C;
      float* d_row = up_raw(h_row);
      float* d_mk = up(mk_c);
      bf16* d_w2 = up_bf16(w2_c);
      float* d_w2b = with_bias ? up(w2b_c) : nullptr;
      draft_score_bf16(d_mk, Rk_c, d_w2, d_w2b, Rk_c, 1, d_row, kDraftPickFloats, accum, cap_c, 0);
      auto got_row = down_raw(d_row, kDraftPickFloats);
      Vec got_v(C), want_v(C);
      for (int c = 0; c < C; ++c) {
        const int vid = c_ids[c];
        double dot = 0.0;
        for (int q = 0; q < Rk_c; ++q) dot += w2_q[size_t(vid) * Rk_c + q] * double(float(mk_c[q]));
        const double z = (accum ? double(float(c_base[c])) : 0.0) + dot + (with_bias ? double(float(w2b_c[vid])) : 0.0);
        got_v[c] = got_row[kDraftCandCap + c];
        want_v[c] = softcap_ref(z, cap_c);
      }
      expect(tag, got_v, want_v, 3e-5);
      CK(cudaFree(d_row));
      CK(cudaFree(d_mk));
      CK(cudaFree(d_w2));
      if (d_w2b) CK(cudaFree(d_w2b));
    };
    test_score(256, true, true, 30.f, "draft_score_bf16 (K=256 accum + bias + cap)");
    test_score(36, true, false, 0.f, "draft_score_bf16 (K=36 scalar accum)");
    test_score(36, false, false, 0.f, "draft_score_bf16 (K=36 scalar overwrite)");

    {  // Multi-row draft_combine_cands (rows=3, with add+bias+cap and add=nullptr softcap-only).
      constexpr int Vc = 512, C = 64, R_c = 3;
      std::vector<float> h_rows(size_t(R_c) * kDraftPickFloats, 0.f);
      Vec c_base = randv(size_t(R_c) * C, 1.5), c_add = randv(size_t(R_c) * C, 1.0), w2b_c = randv(Vc, 0.25);
      for (int r = 0; r < R_c; ++r) {
        float* hr = h_rows.data() + size_t(r) * kDraftPickFloats;
        int* h_ci = reinterpret_cast<int*>(hr);
        for (int c = 0; c < C; ++c) {
          h_ci[c] = (r * 37 + c * 5 + 3) % Vc;
          hr[kDraftCandCap + c] = float(c_base[size_t(r) * C + c]);
        }
        h_ci[2 * kDraftCandCap] = C;
      }
      float* d_rows = up_raw(h_rows);
      float* d_add = up(c_add);
      float* d_w2b = up(w2b_c);
      draft_combine_cands(d_add, C, d_w2b, 20.f, R_c, d_rows, kDraftPickFloats, 0);
      auto got_m = down_raw(d_rows, h_rows.size());
      Vec got_v(size_t(R_c) * C), want_v(size_t(R_c) * C);
      for (int r = 0; r < R_c; ++r) {
        const float* gr = got_m.data() + size_t(r) * kDraftPickFloats;
        const int* g_ci = reinterpret_cast<const int*>(gr);
        for (int c = 0; c < C; ++c) {
          got_v[size_t(r) * C + c] = gr[kDraftCandCap + c];
          want_v[size_t(r) * C + c] = softcap_ref(
              double(float(c_base[size_t(r) * C + c])) + double(float(c_add[size_t(r) * C + c])) +
                  double(float(w2b_c[g_ci[c]])),
              20.0);
        }
      }
      expect("draft_combine_cands (rows=3 + bias + cap)", got_v, want_v, 2e-6);

      // Softcap-only mode (add == nullptr, bias == nullptr).
      CK(cudaMemcpy(d_rows, h_rows.data(), h_rows.size() * sizeof(float), cudaMemcpyHostToDevice));
      draft_combine_cands(nullptr, 0, nullptr, 15.f, R_c, d_rows, kDraftPickFloats, 0);
      auto got_sc = down_raw(d_rows, h_rows.size());
      for (int r = 0; r < R_c; ++r) {
        const float* gr = got_sc.data() + size_t(r) * kDraftPickFloats;
        for (int c = 0; c < C; ++c) {
          got_v[size_t(r) * C + c] = gr[kDraftCandCap + c];
          want_v[size_t(r) * C + c] = softcap_ref(double(float(c_base[size_t(r) * C + c])), 15.0);
        }
      }
      expect("draft_combine_cands (rows=3 softcap-only)", got_v, want_v, 2e-6);
      CK(cudaFree(d_rows));
      CK(cudaFree(d_add));
      CK(cudaFree(d_w2b));
    }

    {  // draft_combine_cands + draft_pick_topk (argmax, tie-break, prob, margin).
      constexpr int Vc = 512, C = 97;
      std::vector<int> c_ids(C);
      for (int c = 0; c < C; ++c) c_ids[c] = (c * 3 + 7) % Vc;
      Vec c_base = randv(C, 1.5), c_add = randv(C, 1.0), w2b_c = randv(Vc, 0.25);
      std::vector<float> h_row(kDraftPickFloats, 0.f);
      int* h_ci = reinterpret_cast<int*>(h_row.data());
      for (int c = 0; c < C; ++c) {
        h_ci[c] = c_ids[c];
        h_row[kDraftCandCap + c] = float(c_base[c]);
      }
      h_ci[2 * kDraftCandCap] = C;
      float* d_row = up_raw(h_row);
      float* d_add = up(c_add);
      float* d_w2b = up(w2b_c);
      draft_combine_cands(d_add, 0, d_w2b, 25.f, 1, d_row, kDraftPickFloats, 0);
      auto got_row = down_raw(d_row, kDraftPickFloats);
      Vec got_v(C), want_v(C);
      for (int c = 0; c < C; ++c) {
        const int vid = c_ids[c];
        got_v[c] = got_row[kDraftCandCap + c];
        want_v[c] = softcap_ref(double(float(c_base[c])) + double(float(c_add[c])) + double(float(w2b_c[vid])), 25.0);
      }
      expect("draft_combine_cands (bias + softcap)", got_v, want_v, 2e-6);

      int best_c = -1, best_id = INT_MAX;
      double m1 = -1e30, m2 = -1e30;
      for (int c = 0; c < C; ++c) {
        if (got_v[c] > m1 || (got_v[c] == m1 && c_ids[c] < best_id)) m1 = got_v[c], best_id = c_ids[c], best_c = c;
      }
      double zsum = 0.0;
      for (int c = 0; c < C; ++c) {
        zsum += std::exp(got_v[c] - m1);
        if (c != best_c && got_v[c] > m2) m2 = got_v[c];
      }
      int* d_idx = dev<int>(1);
      float* d_p = dev<float>(1);
      float* d_m = dev<float>(1);
      draft_pick_topk(d_row, kDraftPickFloats, 1, 1, d_idx, nullptr, 0, d_p, d_m);
      int got_id = -1;
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&got_id, d_idx, 4, cudaMemcpyDeviceToHost));
      const Vec gp1 = down(d_p, 1), gm1 = down(d_m, 1);

      // Tie-break: lower vocab id wins and margin == 0.
      std::vector<float> t_row = got_row;
      int* t_ci = reinterpret_cast<int*>(t_row.data());
      t_ci[5] = 400;
      t_ci[12] = 120;
      t_row[kDraftCandCap + 5] = 50.f;
      t_row[kDraftCandCap + 12] = 50.f;
      CK(cudaMemcpy(d_row, t_row.data(), t_row.size() * sizeof(float), cudaMemcpyHostToDevice));
      draft_pick_topk(d_row, kDraftPickFloats, 1, 1, d_idx, nullptr, 0, d_p, d_m);
      int tb_id = -1;
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&tb_id, d_idx, 4, cudaMemcpyDeviceToHost));
      const Vec tb_m = down(d_m, 1);
      const bool ok_pk = got_id == best_id && std::abs(gp1[0] - 1.0 / zsum) < 1e-5 &&
                         std::abs(gm1[0] - (m1 - m2)) < 1e-5 && tb_id == 120 && tb_m[0] == 0.0;
      std::printf("  %-4s %-48s\n", ok_pk ? "ok" : "FAIL", "draft_pick_topk (argmax/prob/margin/tie)");
      g_fail += !ok_pk;
      CK(cudaFree(d_row));
      CK(cudaFree(d_add));
      CK(cudaFree(d_w2b));
      CK(cudaFree(d_idx));
      CK(cudaFree(d_p));
      CK(cudaFree(d_m));
    }

    auto test_sign_add = [&](int K_s, bool with_bias, const char* tag) {
      constexpr int Ns = 777;
      Vec W_s = randv(size_t(Ns) * K_s, 0.4), x_s = randv(K_s, 0.9);
      Vec init_y = randv(Ns, 2.0), b_s = randv(Ns, 0.5);
      bf16* d_W = up_bf16(W_s);
      float* d_x = up(x_s);
      float* d_y = up(init_y);
      float* d_b = with_bias ? up(b_s) : nullptr;
      const size_t tiled = size_t(Ns + 255) / 256 * 256;
      SignHead sh{dev<uint32_t>(tiled * size_t(K_s) / 32), dev<bf16>(tiled * size_t(K_s) / 64)};
      make_sign_head(d_W, Ns, K_s, &sh, 0);
      gemm_sign_rough_add(d_x, K_s, sh, d_b, Ns, K_s, d_y, Ns, TL);
      Vec got = down(d_y, Ns);
      float* d_ref = dev<float>(Ns);
      gemm_sign_rough(d_x, K_s, sh, Ns, K_s, d_ref, Ns, 1, TL);
      Vec ref_rough = down(d_ref, Ns), want(Ns);
      for (int n = 0; n < Ns; ++n)
        want[n] = double(float(init_y[n])) + ref_rough[n] + (with_bias ? double(float(b_s[n])) : 0.0);
      expect(tag, got, want, 2e-6);
      CK(cudaFree(d_W));
      CK(cudaFree(d_x));
      CK(cudaFree(d_y));
      if (d_b) CK(cudaFree(d_b));
      CK(cudaFree(d_ref));
      CK(cudaFree(sh.bits));
      CK(cudaFree(sh.scales));
    };
    test_sign_add(256, true, "gemm_sign_rough_add (K=256 + bias)");
    test_sign_add(512, false, "gemm_sign_rough_add (K=512 no bias)");
  }
}

}  // namespace flyx::optest
