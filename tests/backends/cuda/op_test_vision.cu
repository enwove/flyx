// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// CUDA vision encoder and MRoPE kernel tests against float64 reference ops (ops/ops.h).
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

void test_layernorm() {
  std::printf("layernorm\n");
  for (int n : {1152, 6000}) {  // 1152: Qwen3.8 ViT (in register cache); 6000: past cache spill path
    for (int wb = 0; wb < 2; ++wb) {
      seed("layernorm", n, wb);
      const int rows = 19;
      const Vec x = randv(size_t(rows) * n);
      const Vec w = wb ? randv(size_t(n)) : Vec(size_t(n), 1.0);
      const Vec b = wb ? randv(size_t(n)) : Vec(size_t(n), 0.0);
      const double eps = 1e-6;
      Vec want(x.size());
      for (int r = 0; r < rows; ++r)
        ops::layernorm(x.data() + size_t(r) * n, w.data(), b.data(), n, eps, want.data() + size_t(r) * n);
      float *dx = up(x), *dw = wb ? up(w) : nullptr, *db = wb ? up(b) : nullptr, *dy = dev<float>(x.size());
      layernorm(dx, n, dw, db, float(eps), dy, n, rows, n, 0);
      expect("layernorm n=" + std::to_string(n) + (wb ? " affine" : " plain"), down(dy, x.size()), want, 1e-5);
      cudaFree(dx);
      if (dw) cudaFree(dw);
      if (db) cudaFree(db);
      cudaFree(dy);
    }
  }
}

void test_add_bias() {
  std::printf("add_bias + vision_residual_add\n");
  seed("add_bias", 37, 1152);
  const int rows = 37, cols = 1152;
  const Vec x = randv(size_t(rows) * cols);
  const Vec b = randv(size_t(cols));
  Vec want = x;
  ops::add_bias(want.data(), cols, b.data(), rows, cols);
  float *dy = up(x), *db = up(b);
  add_bias(dy, cols, db, rows, cols, 0);
  expect("add_bias 37x1152", down(dy, x.size()), want, 1e-6);

  const Vec y = randv(size_t(rows) * cols);
  const Vec sc = randv(size_t(cols));
  Vec want_plain = x, want_ls = x;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      want_plain[size_t(r) * cols + c] += y[size_t(r) * cols + c];
      want_ls[size_t(r) * cols + c] += y[size_t(r) * cols + c] * sc[size_t(c)];
    }
  float *dx1 = up(x), *dx2 = up(x), *dy2 = up(y), *dsc = up(sc);
  vision_residual_add(dx1, dy2, nullptr, rows, cols, 0);
  vision_residual_add(dx2, dy2, dsc, rows, cols, 0);
  expect("vision_residual_add plain", down(dx1, x.size()), want_plain, 1e-6);
  expect("vision_residual_add layer_scale", down(dx2, x.size()), want_ls, 1e-6);
  cudaFree(dy);
  cudaFree(db);
  cudaFree(dx1);
  cudaFree(dx2);
  cudaFree(dy2);
  cudaFree(dsc);
}

void test_vact() {
  std::printf("vact_apply + vact_gated\n");
  const size_t n = 4304 * 5;
  for (int act : {FLYX_VACT_GELU_TANH, FLYX_VACT_GELU, FLYX_VACT_SILU, FLYX_VACT_QUICK_GELU}) {
    seed("vact", act);
    const Vec g = randv(n, 2.0), u = randv(n, 1.5);
    Vec want1(n), want2(n);
    for (size_t i = 0; i < n; ++i) {
      want1[i] = ops::vact(g[i], act);
      want2[i] = want1[i] * u[i];
    }
    float *dg = up(g), *du = up(u), *dout = dev<float>(n);
    vact_gated(dg, du, dout, n, act, 0);
    vact_apply(dg, n, act, 0);
    expect("vact_apply kind=" + std::to_string(act), down(dg, n), want1, 1e-5);
    expect("vact_gated kind=" + std::to_string(act), down(dout, n), want2, 1e-5);
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
  }
  // vision_zero_tail: zero trailing columns [valid_cols, ld) per row, preserving [0, valid_cols).
  {
    const int rows = 13, valid = 144, ld = 192;
    Vec mat(size_t(rows) * ld);
    for (size_t i = 0; i < mat.size(); ++i) mat[i] = double(float(double(i % 97) * 0.25 - 12.0));
    Vec want = mat;
    for (int r = 0; r < rows; ++r)
      for (int c = valid; c < ld; ++c) want[size_t(r) * ld + c] = 0.0;
    float* dm = up(mat);
    vision_zero_tail(dm, rows, valid, ld, 0);
    expect("vision_zero_tail 13x144->192", down(dm, mat.size()), want, 0.0);
    cudaFree(dm);
  }
}

void test_gemm_f32() {
  std::printf("gemm_f32\n");
  struct Shape {
    int M, N, K;
  };
  for (Shape sh : {Shape{73, 1152, 1536}, Shape{19, 115, 588}}) {  // Qwen3.8 patch stem & SigLIP non-mult-8 stem
    seed("gemm_f32", sh.M, sh.N, sh.K);
    const Vec x = randv(size_t(sh.M) * sh.K, 0.5);
    const Vec W = randv(size_t(sh.N) * sh.K, 0.5);
    const Vec y0 = randv(size_t(sh.M) * sh.N, 0.5);
    std::vector<float> wf(W.begin(), W.end());
    flyx_tensor_view wv{};
    wv.dtype = FLYX_DT_F32;
    wv.ndim = 2;
    wv.shape[0] = sh.N;
    wv.shape[1] = sh.K;
    wv.data = wf.data();
    Vec want(size_t(sh.M) * sh.N, 0.0);
    ops::linear(x.data(), sh.M, sh.K, &wv, sh.N, want.data());
    Vec want_acc = want;
    for (size_t i = 0; i < want_acc.size(); ++i) want_acc[i] += y0[i];

    float *dx = up(x), *dW = up(W), *dy = dev<float>(want.size()), *dya = up(y0);
    gemm_f32(dx, sh.K, dW, sh.N, sh.K, dy, sh.N, sh.M, false, 0);
    gemm_f32(dx, sh.K, dW, sh.N, sh.K, dya, sh.N, sh.M, true, 0);
    expect("gemm_f32 " + std::to_string(sh.M) + "x" + std::to_string(sh.N) + "x" + std::to_string(sh.K),
           down(dy, want.size()), want, 2e-5);
    expect("gemm_f32 acc " + std::to_string(sh.M) + "x" + std::to_string(sh.N) + "x" + std::to_string(sh.K),
           down(dya, want.size()), want_acc, 2e-5);
    cudaFree(dx);
    cudaFree(dW);
    cudaFree(dy);
    cudaFree(dya);
  }
}

void test_vision_pos_embed() {
  std::printf("vision_pos_embed\n");
  seed("vision_pos_embed", 48, 1152);
  const int H = 128, pos_h = 48, pos_w = 48, merge = 2;
  const int32_t grids[6] = {1, 12, 16, 1, 8, 10};
  const int N = 12 * 16 + 8 * 10;
  const Vec x = randv(size_t(N) * H);
  const Vec pos = randv(size_t(pos_h * pos_w) * H);

  Vec want_bi = x;
  ops::vision_pos_embed_bilinear_add(want_bi.data(), pos.data(), pos_h, pos_w, H, grids, 2, merge);

  std::vector<int32_t> ch, cw, rgh, rgw;
  ops::vision_spatial_coords(grids, 2, merge, &ch, &cw);
  for (int i = 0; i < 12 * 16; ++i) rgh.push_back(12), rgw.push_back(16);
  for (int i = 0; i < 8 * 10; ++i) rgh.push_back(8), rgw.push_back(10);

  float *dx = up(x), *dpos = up(pos);
  int32_t *dch = dev<int32_t>(N), *dcw = dev<int32_t>(N), *dgh = dev<int32_t>(N), *dgw = dev<int32_t>(N);
  CK(cudaMemcpy(dch, ch.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcw, cw.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dgh, rgh.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dgw, rgw.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
  vision_pos_embed_bilinear_add(dx, dpos, pos_h, pos_w, H, dch, dcw, dgh, dgw, N, 0);
  expect("vision_pos_embed_bilinear_add 2 grids", down(dx, x.size()), want_bi, 1e-5);

  // Fixed position embedding modulo pos_rows
  const int pos_rows = 64;
  Vec want_fix = x;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < H; ++c) want_fix[size_t(r) * H + c] += pos[size_t(r % pos_rows) * H + c];
  float* dx2 = up(x);
  vision_pos_embed_add(dx2, dpos, pos_rows, H, N, 0);
  expect("vision_pos_embed_add fixed", down(dx2, x.size()), want_fix, 1e-6);

  cudaFree(dx);
  cudaFree(dx2);
  cudaFree(dpos);
  cudaFree(dch);
  cudaFree(dcw);
  cudaFree(dgh);
  cudaFree(dgw);
}

void test_vision_rope_2d() {
  std::printf("vision_rope_2d\n");
  for (int hd : {72, 64}) {  // 72: Qwen3.8-27B ViT; 64: standard
    seed("vision_rope_2d", hd);
    const int nh = 16, pw = nh * hd, ld = 3 * pw;
    const int32_t grid[3] = {1, 8, 12};
    const int N = 8 * 12;
    const double theta = 10000.0;
    std::vector<int32_t> ch, cw;
    ops::vision_spatial_coords(grid, 1, 2, &ch, &cw);
    const Vec qkv = randv(size_t(N) * ld);
    Vec want = qkv;
    for (int r = 0; r < N; ++r) {
      for (int h = 0; h < nh; ++h) {
        ops::rope_2d_axial(&want[size_t(r) * ld + size_t(h) * hd], ch[size_t(r)], cw[size_t(r)], theta, hd);
        ops::rope_2d_axial(&want[size_t(r) * ld + pw + size_t(h) * hd], ch[size_t(r)], cw[size_t(r)], theta, hd);
      }
    }
    float* dqkv = up(qkv);
    int32_t *dch = dev<int32_t>(N), *dcw = dev<int32_t>(N);
    CK(cudaMemcpy(dch, ch.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dcw, cw.data(), size_t(N) * 4, cudaMemcpyHostToDevice));
    vision_rope_2d(dqkv, ld, 0, pw, nh, hd, theta, dch, dcw, N, 0);
    expect("vision_rope_2d hd=" + std::to_string(hd), down(dqkv, qkv.size()), want, 1e-5);
    cudaFree(dqkv);
    cudaFree(dch);
    cudaFree(dcw);
  }
}

void test_vision_attention() {
  std::printf("vision_attention\n");
  for (int hd : {72, 192}) {  // 72: Qwen3.8 ViT (BK=32); 192: large hd branch (BK=16)
    seed("vision_attention", hd);
    const int nh = 8, pw = nh * hd, ld = 3 * pw;
    const int32_t cu[3] = {0, 41, 96};  // two unequal segments spanning across 8-row block boundaries
    const int N = cu[2];
    const double scale = 1.0 / std::sqrt(double(hd));
    const Vec qkv = randv(size_t(N) * ld, 0.5);
    Vec K(size_t(N) * pw), V(size_t(N) * pw);
    for (int r = 0; r < N; ++r) {
      std::copy_n(&qkv[size_t(r) * ld + pw], pw, &K[size_t(r) * pw]);
      std::copy_n(&qkv[size_t(r) * ld + 2 * pw], pw, &V[size_t(r) * pw]);
    }
    Vec want(size_t(N) * pw, 0.0);
    for (int seg = 0; seg < 2; ++seg) {
      for (int r = cu[seg]; r < cu[seg + 1]; ++r)
        for (int h = 0; h < nh; ++h)
          ops::attend_range(&qkv[size_t(r) * ld + size_t(h) * hd], K.data(), V.data(), cu[seg], cu[seg + 1], nh, hd, h,
                            scale, &want[size_t(r) * pw + size_t(h) * hd]);
    }
    float *dqkv = up(qkv), *dout = dev<float>(want.size());
    int32_t* dcu = dev<int32_t>(3);
    CK(cudaMemcpy(dcu, cu, sizeof(cu), cudaMemcpyHostToDevice));
    vision_attention(dqkv, ld, 0, pw, 2 * pw, nh, hd, dcu, 2, N, float(scale), dout, pw, 0);
    expect("vision_attention 2-seg hd=" + std::to_string(hd), down(dout, want.size()), want, 1e-5);
    cudaFree(dqkv);
    cudaFree(dout);
    cudaFree(dcu);
  }
}

void test_vision_avg_pool() {
  std::printf("vision_avg_pool_blocks\n");
  seed("vision_avg_pool", 24, 4, 1152);
  const int nb = 24, grp = 4, H = 1152;
  const Vec x = randv(size_t(nb) * grp * H);
  Vec want(size_t(nb) * H);
  ops::vision_avg_pool_blocks(x.data(), nb, grp, H, want.data());
  float *dx = up(x), *dout = dev<float>(want.size());
  vision_avg_pool_blocks(dx, nb, grp, H, dout, 0);
  expect("vision_avg_pool_blocks 24x4x1152", down(dout, want.size()), want, 1e-6);
  cudaFree(dx);
  cudaFree(dout);
}

void test_embed_scatter() {
  std::printf("embed_scatter\n");
  seed("embed_scatter", 32, 256);
  const int rows = 32, H = 256, ne = 7;
  const int32_t erows[ne] = {2, 3, 4, 15, 16, 20, 31};
  const Vec base = randv(size_t(rows) * H);
  const Vec emb = randv(size_t(ne) * H);
  Vec want = base;
  for (int i = 0; i < ne; ++i)
    for (int c = 0; c < H; ++c) want[size_t(erows[i]) * H + c] = emb[size_t(i) * H + c];

  Vec want32 = want;
  for (double& v : want32) v = double(float(v));
  float *dx = up(base), *de = up(emb);
  int32_t* dr = dev<int32_t>(ne);
  CK(cudaMemcpy(dr, erows, sizeof(erows), cudaMemcpyHostToDevice));
  embed_scatter(dx, nullptr, H, de, dr, ne, 0);
  expect("embed_scatter f32", down(dx, base.size()), want32, 0.0);

  Vec want16 = want;
  round_bf16(want16);
  bf16* dx16 = up_bf16(base);
  embed_scatter(nullptr, dx16, H, de, dr, ne, 0);
  expect("embed_scatter bf16", down_bf16(dx16, base.size()), want16, 0.0);

  cudaFree(dx);
  cudaFree(dx16);
  cudaFree(de);
  cudaFree(dr);
}

void test_mrope() {
  std::printf("mrope (attn_prep)\n");
  const int hd = 256, rot = 64, nh = 8, nkv = 2, rows = 12, ctx = 20;
  const double theta = 1e7, eps = 1e-6;
  const int sec[3] = {11, 11, 10};
  rope_register(theta, rot);
  for (int interleaved : {1, 0}) {
    seed("mrope", interleaved);
    const int ld = (nh + 2 * nkv) * hd;
    const Vec qkv = randv(size_t(rows) * ld);
    const Vec qn = randv(size_t(hd), 0.2), kn = randv(size_t(hd), 0.2);
    std::vector<int32_t> p3(size_t(rows) * 3);
    for (int r = 0; r < rows; ++r) {
      p3[size_t(r) * 3 + 0] = ctx + (r < 4 ? r : 4);
      p3[size_t(r) * 3 + 1] = ctx + (r < 4 ? r : 4 + (r - 4) / 2);
      p3[size_t(r) * 3 + 2] = ctx + (r < 4 ? r : 4 + (r - 4) % 2);
    }
    Vec want_q(size_t(rows) * nh * hd), want_k(size_t(rows) * nkv * hd);
    for (int r = 0; r < rows; ++r) {
      const int pt = p3[size_t(r) * 3 + 0], ph = p3[size_t(r) * 3 + 1], pw = p3[size_t(r) * 3 + 2];
      for (int h = 0; h < nh; ++h)
        ops::prep_head_3d(&qkv[size_t(r) * ld + size_t(h) * hd], qn.data(), hd, eps, 1.0, pt, ph, pw, theta, rot, sec,
                          interleaved != 0, &want_q[(size_t(r) * nh + h) * hd]);
      for (int j = 0; j < nkv; ++j)
        ops::prep_head_3d(&qkv[size_t(r) * ld + (nh + j) * hd], kn.data(), hd, eps, 1.0, pt, ph, pw, theta, rot, sec,
                          interleaved != 0, &want_k[(size_t(r) * nkv + j) * hd]);
    }
    float *dqkv = up(qkv), *dqn = up(qn), *dkn = up(kn), *dq = dev<float>(want_q.size());
    kv16 *Kc = dev<kv16>(size_t(ctx + rows) * nkv * hd), *Vc = dev<kv16>(size_t(ctx + rows) * nkv * hd);
    int32_t* dp3 = dev<int32_t>(p3.size());
    CK(cudaMemcpy(dp3, p3.data(), p3.size() * 4, cudaMemcpyHostToDevice));
    DevParams* dP = params(ctx);
    MRoPE mr{{sec[0], sec[1], sec[2]}, interleaved, dp3};
    attn_prep(dqkv, ld, 0, nh * hd, (nh + nkv) * hd, dqn, dkn, 1.f, float(eps), theta, rot, nh, nkv, hd, rows, dP,
              kPosPosition, false, dq, KvRef{Kc, Vc, nullptr, nullptr}, 0, 0, nullptr, 0, CanvasKeys{}, mr);
    expect(std::string("mrope 3D Q ") + (interleaved ? "interleaved" : "contiguous"), down(dq, want_q.size()), want_q,
           1e-5);
    expect(std::string("mrope 3D K ") + (interleaved ? "interleaved" : "contiguous"),
           down_f16(Kc + size_t(ctx) * nkv * hd, want_k.size()), want_k, 2e-3);

    // Scalar rope_delta path (decode after multimodal prefill: pos_3d == nullptr, P->rope_delta = -7)
    DevParams phost{};
    phost.position = ctx;
    phost.rope_delta = -7;
    CK(cudaMemcpy(dP, &phost, sizeof(phost), cudaMemcpyHostToDevice));
    Vec want_qd(size_t(rows) * nh * hd);
    for (int r = 0; r < rows; ++r) {
      const int sp = ctx + r - 7;
      for (int h = 0; h < nh; ++h)
        ops::prep_head(&qkv[size_t(r) * ld + size_t(h) * hd], qn.data(), hd, eps, 1.0, sp, theta, rot,
                       &want_qd[(size_t(r) * nh + h) * hd]);
    }
    MRoPE mr_scalar{{sec[0], sec[1], sec[2]}, interleaved, nullptr};
    attn_prep(dqkv, ld, 0, nh * hd, (nh + nkv) * hd, dqn, dkn, 1.f, float(eps), theta, rot, nh, nkv, hd, rows, dP,
              kPosPosition, false, dq, KvRef{Kc, Vc, nullptr, nullptr}, 0, 0, nullptr, 0, CanvasKeys{}, mr_scalar);
    expect(std::string("mrope rope_delta Q ") + (interleaved ? "interleaved" : "contiguous"), down(dq, want_qd.size()),
           want_qd, 1e-5);

    cudaFree(dqkv);
    cudaFree(dqn);
    cudaFree(dkn);
    cudaFree(dq);
    cudaFree(Kc);
    cudaFree(Vc);
    cudaFree(dp3);
    cudaFree(dP);
  }
}

}  // namespace flyx::optest
