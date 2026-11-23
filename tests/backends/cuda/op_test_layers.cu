// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: the rest of a layer: rmsnorm and the fused producers.
#include <functional>

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

namespace {
// The kernels run() launches, by name: captured from a stream of its own into a graph, whose kernel nodes are read.
std::vector<std::string> kernels_launched(const std::function<void(const Launch&)>& run) {
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  Launch L = TL;
  L.st = st;
  cudaGraph_t g;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  run(L);
  CK(cudaStreamEndCapture(st, &g));
  size_t n = 0;
  CK(cudaGraphGetNodes(g, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  CK(cudaGraphGetNodes(g, nodes.data(), &n));
  std::vector<std::string> names;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type;
    CK(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel) continue;
    cudaKernelNodeParams p{};
    CK(cudaGraphKernelNodeGetParams(node, &p));
    const char* name = nullptr;
    CK(cudaFuncGetName(&name, p.func));
    names.emplace_back(name ? name : "?");
  }
  CK(cudaGraphDestroy(g));
  CK(cudaStreamDestroy(st));
  return names;
}

// A V-LoRA's GEMMs in a round's rows past 16 as an attention block's adapters run them (regions/layers.cpp lora_in):
// q|k|v's a stacked [3 rank, H] in one GEMM, their products side by side in u (3 rank apart), each b reading its
// rank columns of u into its own columns of y.  Every row the bits it has alone, and every GEMM whose K is a multiple
// of 32 (the a's; the b's at a rank such as lora_g_b2h's 64) one launch of k_gemm_skinny_mt, the weights read once
// for every 16-row tile, not a launch a tile.
void lora_round_layout(int r, float scale, const std::string& tag) {
  const int H = 256, out[3] = {128, 64, 64}, ldy = 256, ldu = 3 * r;
  seed("lora round layout", r, int(scale * 1000));
  bf16* da = up_bf16(randv(size_t(ldu) * H, 1.0 / std::sqrt(H)));
  bf16* db[3];
  for (int i = 0; i < 3; ++i) db[i] = up_bf16(randv(size_t(out[i]) * r, 0.5 / std::sqrt(r)));
  auto adapters = [&](const float* x, float* u, float* y, int rows, const Launch& L) {
    lora_down(x, H, da, ldu, H, scale, u, ldu, rows, L);
    for (int i = 0, col = 0; i < 3; col += out[i], ++i)
      gemm(u + i * r, ldu, db[i], out[i], r, y + col, ldy, rows, true, L);
  };
  Launch LR = TL;
  LR.round_rows = true;
  for (const int rows : {17, 32, 64}) {
    const Vec x = randv(size_t(rows) * H), y0 = randv(size_t(rows) * ldy);
    float *dx = up(x), *y = up(y0), *start = up(y0), *u = dev<float>(size_t(rows) * ldu), *y1 = dev<float>(ldy),
          *u1 = dev<float>(ldu);
    adapters(dx, u, y, rows, LR);
    const Vec all = down(y, size_t(rows) * ldy);
    bool same = true;
    for (int row = 0; row < rows; ++row) {
      CK(cudaMemcpy(y1, start + size_t(row) * ldy, ldy * 4, cudaMemcpyDeviceToDevice));
      adapters(dx + size_t(row) * H, u1, y1, 1, TL);
      const Vec one = down(y1, ldy);
      same &= std::equal(one.begin(), one.end(), all.begin() + size_t(row) * ldy);
    }
    const std::string rt = tag + ", a round of " + std::to_string(rows) + " rows";
    std::printf("  %-4s q|k|v's adapters as lora_in runs them%s: every row the bits it has alone\n",
                same ? "ok" : "FAIL", rt.c_str());
    g_fail += !same;
    int multi = 0, other = 0;
    const auto kernels = kernels_launched([&](const Launch& L) {
      Launch round = L;
      round.round_rows = true;
      adapters(dx, u, y, rows, round);
    });
    for (const std::string& k : kernels) {
      if (k.find("k_gemm_skinny_mt") != std::string::npos) ++multi;
      else if (k.find("k_scale_rows") == std::string::npos) ++other;  // the scale's own kernel is no GEMM's
    }
    const int want = 1 + (r % 32 == 0 ? 3 : 0);  // the a's GEMM; the b's where their K, the rank, takes the kernel
    const bool routed = multi == want && (r % 32 != 0 || other == 0);
    std::printf("  %-4s %d of its 4 GEMMs one launch of k_gemm_skinny_mt each%s (%d other kernels)\n",
                routed ? "ok" : "FAIL", multi, rt.c_str(), other);
    g_fail += !routed;
    for (void* p : std::initializer_list<void*>{dx, y, start, u, y1, u1}) CK(cudaFree(p));
  }
  CK(cudaFree(da));
  for (bf16* p : db) CK(cudaFree(p));
}
}  // namespace

// A linear with a low-rank adapter (ABI 43, a V-LoRA: lora_down, then gemm on its product, accumulating): against
// ops::linear and ops::linear_lora in float64 (1..16 rows: the inputs as three bf16 parts, float32's; a prefill pass's:
// bf16 inputs on the tensor cores, which the reference rounds first, u included); a fused producer's staged input
// gives the generic producer's bits (F) for bf16 and NVFP4 weights; a round's rows past 16 (Launch::round_rows) each the
// bits it has alone (R), the adapter's two GEMMs included.  Scale 1 (bifill-dv's) and another.
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
      uint8_t *fq = dev<uint8_t>(size_t(N) * H / 2), *fs = dev<uint8_t>(size_t(N) * H / 16);
      float* frs = dev<float>(N);
      quantize_nvfp4(dw, N, H, fq, fs, frs, 0);
      for (const int rows : {1, 5, 16, 40}) {
        const std::string rt = tag + ", " + std::to_string(rows) + " rows";
        const Vec x = randv(size_t(rows) * H);
        float *dx = up(x), *h = dev<float>(size_t(rows) * H), *y1 = dev<float>(size_t(rows) * N),
              *y2 = dev<float>(size_t(rows) * N);
        float *u1 = dev<float>(size_t(rows) * r), *u2 = dev<float>(size_t(rows) * r);
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
        // fused: the producer stages the input; the linear's GEMM and the adapter's first read it
        rmsnorm_split(dx, H, dwn, 0.f, 1e-6f, rows, H, TL);
        gemm_presplit(dw, N, H, y2, N, rows, false, TL);
        lora_down(nullptr, H, da, r, H, float(scale), u2, r, rows, TL);
        gemm(u2, r, db, N, r, y2, N, rows, true, TL);
        expect("a fused producer: the adapter's product the generic one's" + rt, down(u2, size_t(rows) * r),
               down(u1, size_t(rows) * r), 0);
        expect("a fused producer: the generic bits, bf16 weights" + rt, down(y2, want.size()), down(y1, want.size()),
               0);
        gemm_fp4(h, H, fq, fs, frs, N, H, y1, N, rows, false, TL);
        lora_down(h, H, da, r, H, float(scale), u1, r, rows, TL);
        gemm(u1, r, db, N, r, y1, N, rows, true, TL);
        rmsnorm_split(dx, H, dwn, 0.f, 1e-6f, rows, H, TL);
        gemm_fp4_presplit(fq, fs, frs, N, H, y2, N, rows, false, TL);
        lora_down(nullptr, H, da, r, H, float(scale), u2, r, rows, TL);
        gemm(u2, r, db, N, r, y2, N, rows, true, TL);
        expect("a fused producer: the generic bits, NVFP4 weights" + rt, down(y2, want.size()), down(y1, want.size()),
               0);
      }
      // A round's rows past 16: every row, through both of the adapter's GEMMs, the bits it has alone.
      Launch LR = TL;
      LR.round_rows = true;
      for (const int rows : {17, 32, 64}) {
        const Vec x = randv(size_t(rows) * H);
        float *dx = up(x), *y = dev<float>(size_t(rows) * N), *u = dev<float>(size_t(rows) * r), *y1 = dev<float>(N),
              *u1 = dev<float>(r);
        gemm(dx, H, dw, N, H, y, N, rows, false, LR);
        lora_down(dx, H, da, r, H, float(scale), u, r, rows, LR);
        gemm(u, r, db, N, r, y, N, rows, true, LR);
        const Vec all = down(y, size_t(rows) * N);
        bool same = true;
        for (int row = 0; row < rows; ++row) {
          gemm(dx + size_t(row) * H, H, dw, N, H, y1, N, 1, false, TL);
          lora_down(dx + size_t(row) * H, H, da, r, H, float(scale), u1, r, 1, TL);
          gemm(u1, r, db, N, r, y1, N, 1, true, TL);
          const Vec one = down(y1, N);
          same &= std::equal(one.begin(), one.end(), all.begin() + size_t(row) * N);
        }
        std::printf("  %-4s a round of %d rows%s: every row the bits it has alone\n", same ? "ok" : "FAIL", rows,
                    tag.c_str());
        g_fail += !same;
      }
      lora_round_layout(r, float(scale), tag);
    }
}

}  // namespace flyx::optest
