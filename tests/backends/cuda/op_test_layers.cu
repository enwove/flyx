// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: the rest of a layer: rmsnorm and the fused producers.
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

void test_rmsnorm() {
  std::printf("rmsnorm\n");
  seed("rmsnorm");
  const int rows = 5, n = 2560;
  Vec x = randv(size_t(rows) * n), w = randv(n, 0.3);
  for (double off : {0.0, 1.0}) {
    Vec want(x.size());
    for (int r = 0; r < rows; ++r) ops::rmsnorm(&x[size_t(r) * n], w.data(), n, 1e-6, off, &want[size_t(r) * n]);
    float *dx = up(x), *dw = up(w), *dy = dev<float>(x.size());
    rmsnorm(dx, n, dw, float(off), 1e-6f, dy, n, rows, n, 0);
    expect("offset " + std::to_string(int(off)), down(dy, x.size()), want, 1e-5);
  }
  {  // bf16 rows (a prefill pass's bf16 residual), 1 and 40 rows: the float32 kernels' bits on the same values
    Vec xb = x;
    round_bf16(xb);
    float *dw = up(w), *dxf = up(xb), *y1 = dev<float>(xb.size()), *y2 = dev<float>(xb.size());
    bf16* dxb = up_bf16(xb);
    for (int r : {1, rows}) {
      rmsnorm(dxf, n, dw, 1.f, 1e-6f, y1, n, r, n, 0);
      rmsnorm(dxb, n, dw, 1.f, 1e-6f, y2, n, r, n, 0);
      expect("bf16 rows (" + std::to_string(r) + "): the float32 rows' bits", down(y2, size_t(r) * n),
             down(y1, size_t(r) * n), 0);
    }
  }
}

// Rule 2 at the kernel level: a fused producer (writing the next GEMM's input
// already rounded) followed by the GEMM that reads it gives the bits of the
// generic producer and GEMM, for decode rows (split input) and prefill rows
// (bf16 input), with bf16 and NVFP4 weights.
void test_fused_producers() {
  std::printf("fused producers\n");
  seed("fused producers");
  const int H = 256, I = 512, N = 192;
  const GdnDims gd{2, 4, 32, 64, 4};
  const int vd = gd.nv * gd.dv;
  Vec wn = randv(H, 0.3), wg = randv(size_t(N) * H, 1.0 / std::sqrt(H)), wi = randv(size_t(N) * I, 1.0 / std::sqrt(I)),
      wv = randv(size_t(N) * vd, 1.0 / std::sqrt(vd)), gw = randv(gd.dv, 0.3);
  round_bf16(wg);
  round_bf16(wi);
  round_bf16(wv);
  float *dwn = up(wn), *dgw = up(gw);
  bf16 *bg = up_bf16(wg), *bi = up_bf16(wi), *bv = up_bf16(wv);
  struct Fp4 {
    uint8_t *q, *s;
    float* rs;
  };
  auto fp4_of = [&](const bf16* w, int K) {
    Fp4 f{dev<uint8_t>(size_t(N) * K / 2), dev<uint8_t>(size_t(N) * K / 16), dev<float>(N)};
    quantize_nvfp4(w, N, K, f.q, f.s, f.rs, 0);
    return f;
  };
  const Fp4 fg = fp4_of(bg, H), fi = fp4_of(bi, I), fv = fp4_of(bv, vd);
  // GGML block weights whose inputs are 8 bits (Q4_K): the producers write the rows in 8 bits themselves.
  const BlockType& q4k = block_type(FLYX_DT_GGML_Q4_K);
  auto q8_of = [&](int K) {
    BlockMat b = up_blocks(q4k.dtype, random_blocks(q4k, size_t(N) * K / flyx_block_values(q4k.dtype)), N);
    b.q8 = true;
    return b;
  };
  const BlockMat qg = q8_of(H), qi = q8_of(I), qv = q8_of(vd);
  for (int rows : {1, 5, 16, 40}) {
    const std::string tag = " (" + std::to_string(rows) + " rows)";
    float *y1 = dev<float>(size_t(rows) * N), *y2 = dev<float>(size_t(rows) * N);
    auto same = [&](const std::string& what) {
      expect(what + tag, down(y2, size_t(rows) * N), down(y1, size_t(rows) * N), 0);
    };
    {  // rmsnorm -> a projection
      float *x = up(randv(size_t(rows) * H)), *h = dev<float>(size_t(rows) * H);
      rmsnorm(x, H, dwn, 1.f, 1e-6f, h, H, rows, H, 0);
      gemm(h, H, bg, N, H, y1, N, rows, false, TL);
      rmsnorm_split(x, H, dwn, 1.f, 1e-6f, rows, H, TL);
      gemm_presplit(bg, N, H, y2, N, rows, false, TL);
      same("rmsnorm, bf16 GEMM");
      gemm_fp4(h, H, fg.q, fg.s, fg.rs, N, H, y1, N, rows, false, TL);
      rmsnorm_split(x, H, dwn, 1.f, 1e-6f, rows, H, TL);
      gemm_fp4_presplit(fg.q, fg.s, fg.rs, N, H, y2, N, rows, false, TL);
      same("rmsnorm, NVFP4 GEMM");
      blocks_stage(h, H, H, rows, TL, true);
      const Q8Rows staged = q8_rows(rows, H);
      gemm_blocks_staged(qg, H, y1, N, rows, false, TL);
      rmsnorm_split(x, H, dwn, 1.f, 1e-6f, rows, H, TL, true);
      const bool same8 = q8_rows(rows, H) == staged;
      std::printf("  %-4s %-48s\n", same8 ? "ok" : "FAIL", ("rmsnorm in 8 bits: blocks_stage's rows" + tag).c_str());
      g_fail += !same8;
      gemm_blocks_staged(qg, H, y2, N, rows, false, TL);
      same("rmsnorm in 8 bits, Q4_K GEMM");
    }
    {  // SwiGLU -> the down projection
      float *gu = up(randv(size_t(rows) * 2 * I)), *act = dev<float>(size_t(rows) * I);
      swiglu(gu, 2 * I, I, act, I, rows, 0);
      gemm(act, I, bi, N, I, y1, N, rows, false, TL);
      swiglu_split(gu, 2 * I, I, rows, TL);
      gemm_presplit(bi, N, I, y2, N, rows, false, TL);
      same("SwiGLU, bf16 GEMM");
      gemm_fp4(act, I, fi.q, fi.s, fi.rs, N, I, y1, N, rows, false, TL);
      swiglu_split(gu, 2 * I, I, rows, TL);
      gemm_fp4_presplit(fi.q, fi.s, fi.rs, N, I, y2, N, rows, false, TL);
      same("SwiGLU, NVFP4 GEMM");
      blocks_stage(act, I, I, rows, TL, true);
      const Q8Rows staged = q8_rows(rows, I);
      gemm_blocks_staged(qi, I, y1, N, rows, false, TL);
      swiglu_split(gu, 2 * I, I, rows, TL, true);
      const bool same8 = q8_rows(rows, I) == staged;
      std::printf("  %-4s %-48s\n", same8 ? "ok" : "FAIL", ("SwiGLU in 8 bits: blocks_stage's rows" + tag).c_str());
      g_fail += !same8;
      gemm_blocks_staged(qi, I, y2, N, rows, false, TL);
      same("SwiGLU in 8 bits, Q4_K GEMM");
    }
    {  // the GDN gated norm -> the out projection
      float *o = up(randv(size_t(rows) * vd)), *z = up(randv(size_t(rows) * vd)), *g = dev<float>(size_t(rows) * vd);
      gdn_gated_norm(o, z, vd, dgw, 1e-6f, g, rows, gd, 0);
      CK(cudaMemset(y2, 0, size_t(rows) * N * 4));
      CK(cudaMemset(y1, 0, size_t(rows) * N * 4));
      gemm(g, vd, bv, N, vd, y1, N, rows, true, TL);
      gdn_gated_norm_split(o, z, vd, dgw, 1e-6f, rows, gd, TL);
      gemm_presplit(bv, N, vd, y2, N, rows, true, TL);
      same("gated norm, bf16 GEMM (accumulate)");
      gemm_fp4(g, vd, fv.q, fv.s, fv.rs, N, vd, y1, N, rows, false, TL);
      gdn_gated_norm_split(o, z, vd, dgw, 1e-6f, rows, gd, TL);
      gemm_fp4_presplit(fv.q, fv.s, fv.rs, N, vd, y2, N, rows, false, TL);
      same("gated norm, NVFP4 GEMM");
      blocks_stage(g, vd, vd, rows, TL, true);
      const Q8Rows staged = q8_rows(rows, vd);
      gemm_blocks_staged(qv, vd, y1, N, rows, false, TL);
      gdn_gated_norm_split(o, z, vd, dgw, 1e-6f, rows, gd, TL, true);
      const bool same8 = q8_rows(rows, vd) == staged;
      std::printf("  %-4s %-48s\n", same8 ? "ok" : "FAIL", ("gated norm in 8 bits: blocks_stage's rows" + tag).c_str());
      g_fail += !same8;
      gemm_blocks_staged(qv, vd, y2, N, rows, false, TL);
      same("gated norm in 8 bits, Q4_K GEMM");
    }
  }
}

// A linear with a low-rank adapter (ABI 43, a V-LoRA: lora_down, then gemm on its product, accumulating): against
// ops::linear and ops::linear_lora in float64 (1..16 rows: the inputs as three bf16 parts, float32's; a prefill pass's:
// bf16 inputs on the tensor cores, which the reference rounds first, u included).  Scale 1 (bifill-dv's) and another.
void test_lora() {
  std::printf("adapters (lora_down)\n");
  const int H = 256, N = 192;
  for (const int r : {64, 16})
    for (const double scale : {1.0, 0.375}) {
      seed("lora", r, int(scale * 1000));
      const std::string tag = " (rank " + std::to_string(r) + ", scale " + std::to_string(scale).substr(0, 5) + ")";
      Vec wn = randv(H, 0.3), w = randv(size_t(N) * H, 1.0 / std::sqrt(H));
      const Vec a = randv(size_t(r) * H, 1.0 / std::sqrt(H)), b = randv(size_t(N) * r, 0.5 / std::sqrt(r));
      for (double& v : wn) v += 1.0;
      std::vector<uint16_t> sw, sa, sb;
      const flyx_tensor_view tw = view_of(w, &sw, N, H), ta = view_of(a, &sa, r, H), tb = view_of(b, &sb, N, r);
      float* dwn = up(wn);
      bf16 *dw = up_bf16(w), *da = up_bf16(a), *db = up_bf16(b);
      for (const int rows : {1, 5, 16, 40}) {
        const std::string rt = tag + ", " + std::to_string(rows) + " rows";
        const Vec x = randv(size_t(rows) * H);
        float *dx = up(x), *h = dev<float>(size_t(rows) * H), *y1 = dev<float>(size_t(rows) * N);
        float* u1 = dev<float>(size_t(rows) * r);
        rmsnorm(dx, H, dwn, 0.f, 1e-6f, h, H, rows, H, 0);
        gemm(h, H, dw, N, H, y1, N, rows, false, TL);
        lora_down(h, H, da, r, H, float(scale), u1, r, rows, TL);
        gemm(u1, r, db, N, r, y1, N, rows, true, TL);
        // the float64 reference on the normed rows the GPU made (rounded to bf16 where the tensor cores take them)
        Vec hn = down(h, size_t(rows) * H), want(size_t(rows) * N);
        const bool tc = rows > 16;
        if (tc) round_bf16(hn);
        ops::linear(hn.data(), rows, H, &tw, N, want.data());
        Vec u(size_t(rows) * r), d(want.size());
        ops::linear(hn.data(), rows, H, &ta, r, u.data());
        for (double& v : u) v = double(float(v)) * scale;  // u as the GPU holds it: float32, then the scale
        if (tc) {
          // The tensor cores take u in bf16: the reference rounds the u the GPU made, not its own.  A float32 sum
          // in another order can sit on the other side of a bf16 tie (one product 8e-5 off on sm_80 to sm_89, where
          // sm_90 and sm_100 summed the other way); u itself is held to float64 first.
          const Vec gu = down(u1, u.size());
          expect("the adapter's product u, against float64" + rt, gu, u, 2e-5);
          u = gu;
          round_bf16(u);
        }
        ops::linear(u.data(), rows, r, &tb, N, d.data());
        for (size_t i = 0; i < want.size(); ++i) want[i] += d[i];
        expect("a linear and its adapter, against float64" + rt, down(y1, want.size()), want, 2e-5);
        if (!tc) {  // the reference's own: linear_lora on the same rows
          Vec ref(want.size());
          ops::linear(hn.data(), rows, H, &tw, N, ref.data());
          ops::linear_lora(hn.data(), rows, H, &ta, &tb, r, scale, N, ref.data());
          expect("ops::linear_lora" + rt, down(y1, want.size()), ref, 2e-5);
        }
      }
    }
}

// An adapter's own kernels (linear_lora on float32 rows): lora_down_round and lora_down_rows (u = x a^T, unscaled)
// and lora_up / lora_up_rows (y += (u scale) b^T) against float64; a round's kernels a row's bits alone (R);
// lora_down_rows split over K (part, rows <= part_rows) the bits of its one-pass grid; the SwiGLU taking the second
// products (LoraGlu) lora_up's bits added before swiglu_split.  K a multiple of 256 and one past it by 64.
void test_lora_kernels() {
  std::printf("adapters (lora_down_round, lora_down_rows, lora_up, lora_up_rows, swiglu_split with LoraGlu)\n");
  for (const int K : {512, 320})
    for (const int r : {16, 64}) {
      seed("lora kernels", K, r);
      const int n = 2 * r, N = 192, I = 128;  // two linears' a's stacked (as an MLP's gate and up)
      const float scale = 0.75f;
      const std::string tag = " (K " + std::to_string(K) + ", rank " + std::to_string(r) + ")";
      Vec a = randv(size_t(n) * K, 1.0 / std::sqrt(K)), b = randv(size_t(N) * r, 0.5 / std::sqrt(r));
      round_bf16(a);
      round_bf16(b);
      bf16 *da = up_bf16(a), *db = up_bf16(b);
      const int part_rows = 128;
      float* part = dev<float>(size_t(lora_down_part_floats(n, K, part_rows)));
      for (const int rows : {1, 3, 16, 40, 128, 200}) {
        const std::string rt = tag + ", " + std::to_string(rows) + " rows";
        const Vec x = randv(size_t(rows) * K);
        float *dx = up(x), *u1 = dev<float>(size_t(rows) * n), *u2 = dev<float>(size_t(rows) * n),
              *u3 = dev<float>(size_t(rows) * n);
        Vec want(size_t(rows) * n, 0.0);
        for (int row = 0; row < rows; ++row)
          for (int j = 0; j < n; ++j) {
            double t = 0;
            for (int k = 0; k < K; ++k) t += x[size_t(row) * K + k] * a[size_t(j) * K + k];
            want[size_t(row) * n + j] = t;
          }
        lora_down_round(dx, K, da, n, K, u1, n, rows, 0);
        expect("lora_down_round, against float64" + rt, down(u1, want.size()), want, 2e-5);
        lora_down_rows(dx, K, da, n, K, u2, n, rows, part, part_rows, 0);
        lora_down_rows(dx, K, da, n, K, u3, n, rows, nullptr, 0, 0);
        expect("lora_down_rows, against float64" + rt, down(u2, want.size()), want, 2e-5);
        expect("lora_down_rows: split over K the bits of one pass" + rt, down(u2, want.size()),
               down(u3, want.size()), 0);
        {  // a round's rows: each the bits it has alone
          const Vec all = down(u1, want.size());
          bool same = true;
          for (int row = 0; row < rows; ++row) {
            lora_down_round(dx + size_t(row) * K, K, da, n, K, u3, n, 1, 0);
            const Vec one = down(u3, size_t(n));
            same &= std::equal(one.begin(), one.end(), all.begin() + size_t(row) * n);
          }
          std::printf("  %-4s lora_down_round%s: every row the bits it has alone\n", same ? "ok" : "FAIL", rt.c_str());
          g_fail += !same;
        }
        // the second product on the first linear's u (columns 0..r), into y over a base y0
        const Vec u = down(u1, want.size()), y0 = randv(size_t(rows) * N);
        Vec wy = y0;
        for (int row = 0; row < rows; ++row)
          for (int o = 0; o < N; ++o) {
            double t = 0;
            for (int k = 0; k < r; ++k)
              t += double(float(u[size_t(row) * n + k] * scale)) * b[size_t(o) * r + k];
            wy[size_t(row) * N + o] += t;
          }
        LoraUp lu;
        lu.b[0] = db;
        lu.n[0] = N;
        lu.linears = 1;
        float *y1 = up(y0), *y2 = up(y0), *y3 = dev<float>(N);
        lora_up(u1, n, r, scale, lu, y1, N, rows, TL);
        expect("lora_up, against float64" + rt, down(y1, wy.size()), wy, 2e-5);
        if (rows > 16) {
          lora_up_rows(u1, n, r, scale, lu, y2, N, rows, TL);
          expect("lora_up_rows, against float64" + rt, down(y2, wy.size()), wy, 2e-5);
        }
        {
          const Vec all = down(y1, wy.size());
          bool same = true;
          for (int row = 0; row < rows; ++row) {
            const Vec base(y0.begin() + size_t(row) * N, y0.begin() + size_t(row + 1) * N);
            CK(cudaMemcpy(y3, std::vector<float>(base.begin(), base.end()).data(), sizeof(float) * N,
                          cudaMemcpyHostToDevice));
            lora_up(u1 + size_t(row) * n, n, r, scale, lu, y3, N, 1, TL);
            const Vec one = down(y3, size_t(N));
            same &= std::equal(one.begin(), one.end(), all.begin() + size_t(row) * N);
          }
          std::printf("  %-4s lora_up%s: every row the bits it has alone\n", same ? "ok" : "FAIL", rt.c_str());
          g_fail += !same;
        }
        // an MLP's: gate|up rows [rows][2 I], the adapters' second products taken by the SwiGLU (LoraGlu) against
        // lora_up's added first, then swiglu_split without them
        Vec bg = randv(size_t(I) * r, 0.5 / std::sqrt(r)), bu = randv(size_t(I) * r, 0.5 / std::sqrt(r));
        round_bf16(bg);
        round_bf16(bu);
        bf16 *dbg = up_bf16(bg), *dbu = up_bf16(bu);
        const Vec gu = randv(size_t(rows) * 2 * I);
        float *g1 = up(gu), *g2 = up(gu), *o1 = dev<float>(size_t(rows) * I), *o2 = dev<float>(size_t(rows) * I);
        LoraUp two;
        two.b[0] = dbg;
        two.b[1] = dbu;
        two.n[0] = two.n[1] = I;
        two.u_col[1] = r;
        two.y_col[1] = I;
        two.linears = 2;
        lora_up(u1, n, r, scale, two, g2, 2 * I, rows, TL);
        swiglu_split(g2, 2 * I, I, rows, TL, false, o2);
        LoraGlu lg;
        lg.u = u1;
        lg.ldu = n;
        lg.gate = dbg;
        lg.up = dbu;
        lg.rank = r;
        lg.scale = scale;
        swiglu_split(g1, 2 * I, I, rows, TL, false, o1, lg);
        expect("swiglu_split with LoraGlu: lora_up's bits, then the SwiGLU" + rt, down(o1, size_t(rows) * I),
               down(o2, size_t(rows) * I), 0);
        for (void* p : std::initializer_list<void*>{dx, u1, u2, u3, y1, y2, y3, dbg, dbu, g1, g2, o1, o2})
          CK(cudaFree(p));
      }
      for (void* p : std::initializer_list<void*>{da, db, part}) CK(cudaFree(p));
    }
}

}  // namespace flyx::optest
