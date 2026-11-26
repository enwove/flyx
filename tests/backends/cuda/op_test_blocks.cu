// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Kernel tests: GGML block weights, packed: every type and kernel against their float64 values.
#include "tests/backends/cuda/op_test.cuh"

namespace flyx::optest {

// ---- GGML block weights kept packed (gemm_blocks.cu) -------------------------------------------
const BlockType kBlockTypes[21] = {
    {FLYX_DT_GGML_Q4_0, "Q4_0", {0}},       {FLYX_DT_GGML_Q4_1, "Q4_1", {0, 2}},
    {FLYX_DT_GGML_Q5_0, "Q5_0", {0}},       {FLYX_DT_GGML_Q5_1, "Q5_1", {0, 2}},
    {FLYX_DT_GGML_Q8_0, "Q8_0", {0}},       {FLYX_DT_GGML_MXFP4, "MXFP4", {}},
    {FLYX_DT_GGML_Q2_0, "Q2_0", {0}},       {FLYX_DT_GGML_Q2_K, "Q2_K", {80, 82}},
    {FLYX_DT_GGML_Q3_K, "Q3_K", {108}},     {FLYX_DT_GGML_Q4_K, "Q4_K", {0, 2}},
    {FLYX_DT_GGML_Q5_K, "Q5_K", {0, 2}},    {FLYX_DT_GGML_Q6_K, "Q6_K", {208}},
    {FLYX_DT_GGML_IQ2_XXS, "IQ2_XXS", {0}}, {FLYX_DT_GGML_IQ2_XS, "IQ2_XS", {0}},
    {FLYX_DT_GGML_IQ2_S, "IQ2_S", {0}},     {FLYX_DT_GGML_IQ3_XXS, "IQ3_XXS", {0}},
    {FLYX_DT_GGML_IQ3_S, "IQ3_S", {0}},     {FLYX_DT_GGML_IQ1_S, "IQ1_S", {0}},
    {FLYX_DT_GGML_IQ1_M, "IQ1_M", {}},      {FLYX_DT_GGML_IQ4_NL, "IQ4_NL", {0}},
    {FLYX_DT_GGML_IQ4_XS, "IQ4_XS", {0}}};

const BlockType& block_type(int dtype) {
  for (const BlockType& bt : kBlockTypes)
    if (bt.dtype == dtype) return bt;
  std::printf("no block type %d in the tests' list\n", dtype);
  std::exit(2);
}

// n blocks as a file stores them: every payload bit random, the scales finite (1/128 to 1/8, both signs).
std::vector<uint8_t> random_blocks(const BlockType& bt, size_t n) {
  const size_t bytes = size_t(flyx_block_bytes(bt.dtype));
  std::vector<uint8_t> file(n * bytes);
  for (auto& b : file) b = uint8_t(rng());
  auto scale = [] { return uint16_t((rng() & 0x83FFu) | ((8u + rng() % 4u) << 10)); };
  for (size_t i = 0; i < n; ++i) {
    uint8_t* b = file.data() + i * bytes;
    for (int at : bt.scales_at) {
      const uint16_t h = scale();
      std::memcpy(b + at, &h, 2);
    }
    if (bt.dtype == FLYX_DT_GGML_IQ1_M) {  // its scale: the top nibbles of the four scale words
      const uint16_t h = scale();
      for (int j = 0; j < 4; ++j) b[49 + 2 * j] = uint8_t((b[49 + 2 * j] & 0x0F) | (((h >> (4 * j)) & 0xF) << 4));
    }
    if (bt.dtype == FLYX_DT_GGML_MXFP4) {  // E8M0 exponent: finite scales ~1/128..1/8 (e in 120..124)
      b[0] = uint8_t(120u + (rng() % 5u));
    }
  }
  return file;
}

// The file's blocks on the device as the kernels read them; host: in pinned, mapped host memory the kernels read
// over the bus, as the backend keeps an embedding no head reads (freed with cudaFreeHost).
BlockMat up_blocks(int dtype, const std::vector<uint8_t>& file, int rows, bool host) {
  const BlockLayout l = blocks_layout(dtype);
  const size_t n = file.size() / size_t(l.bytes);
  std::vector<uint8_t> payload(n * size_t(l.stride));
  std::vector<uint16_t> scales(n * size_t(l.scales));
  blocks_pack(dtype, file.data(), n, payload.data(), scales.data());
  uint8_t* dp = nullptr;
  uint16_t* ds = nullptr;
  if (host) {  // the device's pointers (the host's are the same here: mapped, not write-combined)
    CK(cudaHostAlloc(&dp, payload.size(), cudaHostAllocMapped));
    CK(cudaHostAlloc(&ds, scales.size() * 2, cudaHostAllocMapped));
    void *d1 = nullptr, *d2 = nullptr;
    CK(cudaHostGetDevicePointer(&d1, dp, 0));
    CK(cudaHostGetDevicePointer(&d2, ds, 0));
    if (d1 != dp || d2 != ds) {
      std::printf("mapped host memory has other addresses on the device here\n");
      std::exit(2);
    }
  } else {
    dp = dev<uint8_t>(payload.size());
    ds = dev<uint16_t>(scales.size());
  }
  CK(cudaMemcpy(dp, payload.data(), payload.size(), cudaMemcpyDefault));
  CK(cudaMemcpy(ds, scales.data(), scales.size() * 2, cudaMemcpyDefault));
  return {dp, ds, dtype, rows};
}

// Every type: the weights the kernels decode are the reference's, bit for bit;
// the GEMMs against the float64 reference on activations rounded as the
// kernels hold them (linear_a16h); a row's result whatever rows run with it
// and whatever configuration runs it.
void test_blocks() {
  std::printf("GGML block weights, packed\n");
  seed("blocks");
  for (const BlockType& bt : kBlockTypes) {
    const int per = flyx_block_values(bt.dtype), N = 40, K = 3 * per;  // three blocks a row: uneven chunks, a partial tile
    const auto file = random_blocks(bt, size_t(N) * 3);
    flyx_tensor_view view{};
    view.dtype = bt.dtype;
    view.ndim = 2;
    view.shape[0] = N;
    view.shape[1] = K;
    view.data = file.data();
    view.nbytes = file.size();
    const BlockMat W = up_blocks(bt.dtype, file, N);
    const std::string tag = std::string(bt.name) + " ";
    const bool takes = blocks_supported(bt.dtype, K) && !blocks_supported(bt.dtype, K + 16) && !blocks_supported(FLYX_DT_BF16, K);
    std::printf("  %-4s %sis a type the kernels take, in whole blocks\n", takes ? "ok" : "FAIL", tag.c_str());
    g_fail += !takes;

    {  // rows of the table, out of order
      std::vector<int> tokens(N);
      for (int i = 0; i < N; ++i) tokens[i] = (i * 7 + 3) % N;
      int* dt = dev<int>(N);
      CK(cudaMemcpy(dt, tokens.data(), N * sizeof(int), cudaMemcpyHostToDevice));
      float* dx = dev<float>(size_t(N) * K);
      embed_blocks(dt, W, K, dx, N, 0);
      Vec want(size_t(N) * K);
      for (int r = 0; r < N; ++r)
        for (int k = 0; k < K; ++k) want[size_t(r) * K + k] = double(float(ops::load(&view, size_t(tokens[r]) * K + k)));
      expect(tag + "weights: the reference's, to the bit", down(dx, want.size()), want, 0);
      const BlockMat H = up_blocks(bt.dtype, file, N, true);
      CK(cudaMemset(dx, 0xFF, size_t(N) * K * sizeof(float)));
      embed_blocks(dt, H, K, dx, N, 0);
      CK(cudaGetLastError());
      expect(tag + "weights from a table in host memory: the same bits", down(dx, want.size()), want, 0);
      CK(cudaFreeHost(const_cast<uint8_t*>(H.data)));
      CK(cudaFreeHost(const_cast<uint16_t*>(H.scales)));
      CK(cudaFree(dt));
      CK(cudaFree(dx));
    }

    const int R = 70;
    const Vec x = randv(size_t(R) * K), y0 = randv(size_t(R) * N);
    Vec want(size_t(R) * N);
    ops::linear_a16h(x.data(), R, K, &view, N, want.data());
    float* dx = up(x);
    float* dy = dev<float>(want.size());
    auto run = [&](int rows, bool accumulate = false) {
      if (accumulate) CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), size_t(rows) * N * 4, cudaMemcpyHostToDevice));
      gemm_blocks(dx, K, W, K, dy, N, rows, accumulate, TL);
      return down(dy, size_t(rows) * N);
    };
    auto first = [&](const Vec& v, int rows) { return Vec(v.begin(), v.begin() + size_t(rows) * N); };
    const Vec y16 = run(16), y70 = run(R);
    expect(tag + "16 rows", y16, first(want, 16), 2e-5);
    expect(tag + "70 rows", y70, want, 2e-5);
    for (int rows : {1, 2, 8, 9}) expect(tag + std::to_string(rows) + " rows: the bits of 16", run(rows), first(y16, rows), 0);
    for (int rows : {17, 64, 65}) expect(tag + std::to_string(rows) + " rows: the bits of 70", run(rows), first(y70, rows), 0);
    Vec want_acc = want;
    for (size_t i = 0; i < want.size(); ++i) want_acc[i] += y0[i];
    expect(tag + "accumulate, 5 rows", run(5, true), first(want_acc, 5), 2e-5);
    expect(tag + "accumulate, 33 rows", run(33, true), first(want_acc, 33), 2e-5);
    {  // Inputs in 8 bits (FLYX_ACT_Q8), every pass, against linear_a8: a round's rows (1..16) with the same
       // bits whatever rows run with them, a prefill pass's whatever its length.
      BlockMat W8 = W;
      W8.q8 = true;
      Vec want8(size_t(R) * N);
      ops::linear_a8(x.data(), R, K, &view, N, want8.data());
      auto run8 = [&](int rows, bool accumulate = false) {
        if (accumulate) CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.end()).data(), size_t(rows) * N * 4, cudaMemcpyHostToDevice));
        else CK(cudaMemset(dy, 0xFF, size_t(rows) * N * 4));
        gemm_blocks(dx, K, W8, K, dy, N, rows, accumulate, TL);
        CK(cudaGetLastError());
        return down(dy, size_t(rows) * N);
      };
      const Vec y8 = run8(R), r8 = run8(16);
      expect(tag + "70 rows, inputs in 8 bits", y8, want8, 2e-5);
      expect(tag + "16 rows, inputs in 8 bits", r8, first(want8, 16), 2e-5);
      for (int rows : {1, 2, 3, 5, 8, 9})
        expect(tag + std::to_string(rows) + " rows, inputs in 8 bits: the bits of 16", run8(rows), first(r8, rows), 0);
      {  // every round kernel (dp4a, the integer tensor cores either way round, the streams split among blocks either way
         // round, among a block's warps) at every row count: those bits
        bool same = true;
        for (int kernel : {0, 1, 2, 3, 4, 5})
          for (int rows : {1, 3, 8, 9, 16}) {
            if (!gemm_blocks_round_kernel_compiled(W8.dtype, kernel, rows)) continue;
            CK(cudaMemset(dy, 0xFF, size_t(rows) * N * 4));
            blocks_stage(dx, K, K, rows, TL, true);
            gemm_blocks_round_q8_kernel(kernel, W8, K, dy, N, rows, false, TL);
            CK(cudaGetLastError());
            same = same && down(dy, size_t(rows) * N) == first(r8, rows);
          }
        std::printf("  %-4s %sinputs in 8 bits: every round kernel gives the bits of 16 rows\n", same ? "ok" : "FAIL",
                    tag.c_str());
        g_fail += !same;
        same = true;  // every compiled configuration of the dp4a kernel (columns a warp, groups ahead), at the rows it holds
        for (int c = 0; c < kBlocksRoundConfigCount; ++c) {
          if (!gemm_blocks_round_cfg_compiled(W8.dtype, c)) continue;
          for (int rows : {1, 2, 3, 5, 8, 9, 16}) {
            if (rows > kBlocksRoundConfigs[c].rows) continue;
            CK(cudaMemset(dy, 0xFF, size_t(rows) * N * 4));
            blocks_stage(dx, K, K, rows, TL, true);
            gemm_blocks_round_q8_cfg(c, W8, K, dy, N, rows, false, TL);
            CK(cudaGetLastError());
            same = same && down(dy, size_t(rows) * N) == first(r8, rows);
          }
        }
        std::printf("  %-4s %sinputs in 8 bits: every round configuration gives the bits of 16 rows\n",
                    same ? "ok" : "FAIL", tag.c_str());
        g_fail += !same;
      }
      for (int rows : {17, 64, 65})
        expect(tag + std::to_string(rows) + " rows, inputs in 8 bits: the bits of 70", run8(rows), first(y8, rows), 0);
      Vec want8_acc = want8;
      for (size_t i = 0; i < want8.size(); ++i) want8_acc[i] += y0[i];
      expect(tag + "accumulate, 5 rows, inputs in 8 bits", run8(5, true), first(want8_acc, 5), 2e-5);
      expect(tag + "accumulate, 33 rows, inputs in 8 bits", run8(33, true), first(want8_acc, 33), 2e-5);
    }
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    CK(cudaFree(const_cast<uint8_t*>(W.data)));
    CK(cudaFree(const_cast<uint16_t*>(W.scales)));

    // Every compiled configuration, on a longer row: 11 blocks of 256 (runs of K of one and of two windows), or 24
    // blocks of 32 (three whole windows, whose scales are loaded eight at once).
    {
      const int N2 = 100, nb = per == 32 ? 24 : 11, K2 = nb * per, R2 = 11;
      const auto file2 = random_blocks(bt, size_t(N2) * nb);
      const BlockMat W2 = up_blocks(bt.dtype, file2, N2);
      flyx_tensor_view view2 = view;
      view2.shape[0] = N2, view2.shape[1] = K2, view2.data = file2.data(), view2.nbytes = file2.size();
      const Vec x2 = randv(size_t(R2) * K2);
      Vec want2(size_t(R2) * N2);
      ops::linear_a16h(x2.data(), R2, K2, &view2, N2, want2.data());
      float* dx2 = up(x2);
      float* dy2 = dev<float>(want2.size());
      blocks_stage(dx2, K2, K2, R2, TL);
      Vec y_first;
      bool same = true;
      for (int cfg = 0; cfg < kBlocksConfigCount; ++cfg) {
        if (!gemm_blocks_cfg_compiled(cfg)) continue;
        CK(cudaMemset(dy2, 0xFF, want2.size() * sizeof(float)));  // NaNs: a configuration that did not run shows
        if (!gemm_blocks_cfg(cfg, W2, K2, dy2, N2, R2, false, TL)) continue;  // no room for 11 rows on this device
        CK(cudaGetLastError());
        const Vec got = down(dy2, want2.size());
        if (y_first.empty()) y_first = got;
        else same = same && got == y_first;
      }
      expect(tag + "11 rows of a longer row", y_first, want2, 2e-5);
      std::printf("  %-4s %sthe same bits from every configuration\n", same ? "ok" : "FAIL", tag.c_str());
      g_fail += !same;
      {  // a round with inputs in 8 bits on it (spans of 32 groups, the last one short), against linear_a8
        BlockMat W8 = W2;
        W8.q8 = true;
        Vec want8(size_t(R2) * N2);
        ops::linear_a8(x2.data(), R2, K2, &view2, N2, want8.data());
        auto round8 = [&](int rows) {
          CK(cudaMemset(dy2, 0xFF, want8.size() * sizeof(float)));
          gemm_blocks(dx2, K2, W8, K2, dy2, N2, rows, false, TL);
          CK(cudaGetLastError());
          return down(dy2, size_t(rows) * N2);
        };
        const Vec r11 = round8(R2);
        expect(tag + "11 rows of a longer row, inputs in 8 bits", r11, want8, 2e-5);
        for (int rows : {1, 4})
          expect(tag + std::to_string(rows) + " rows of a longer row, inputs in 8 bits: the bits of 11", round8(rows),
                 Vec(r11.begin(), r11.begin() + size_t(rows) * N2), 0);
        bool same = true;  // every round kernel
        for (int kernel : {0, 1, 2, 3, 4, 5})
          for (int rows : {2, R2}) {
            if (!gemm_blocks_round_kernel_compiled(W8.dtype, kernel, rows)) continue;
            CK(cudaMemset(dy2, 0xFF, want8.size() * sizeof(float)));
            blocks_stage(dx2, K2, K2, rows, TL, true);
            gemm_blocks_round_q8_kernel(kernel, W8, K2, dy2, N2, rows, false, TL);
            CK(cudaGetLastError());
            same = same && down(dy2, size_t(rows) * N2) == Vec(r11.begin(), r11.begin() + size_t(rows) * N2);
          }
        std::printf("  %-4s %sa longer row, inputs in 8 bits: every round kernel gives the same bits\n",
                    same ? "ok" : "FAIL", tag.c_str());
        g_fail += !same;
        same = true;  // every configuration of the dp4a kernel; 100 columns: a warp's last columns past N
        for (int c = 0; c < kBlocksRoundConfigCount; ++c) {
          if (!gemm_blocks_round_cfg_compiled(W8.dtype, c)) continue;
          for (int rows : {1, 4, R2}) {
            if (rows > kBlocksRoundConfigs[c].rows) continue;
            CK(cudaMemset(dy2, 0xFF, want8.size() * sizeof(float)));
            blocks_stage(dx2, K2, K2, rows, TL, true);
            gemm_blocks_round_q8_cfg(c, W8, K2, dy2, N2, rows, false, TL);
            CK(cudaGetLastError());
            same = same && down(dy2, size_t(rows) * N2) == Vec(r11.begin(), r11.begin() + size_t(rows) * N2);
          }
        }
        std::printf("  %-4s %sa longer row, inputs in 8 bits: every round configuration gives the same bits\n",
                    same ? "ok" : "FAIL", tag.c_str());
        g_fail += !same;
      }
      CK(cudaFree(dx2));
      CK(cudaFree(dy2));

      // A prefill pass over several tiles of rows, of columns and of K, against the reference (the weights decoded
      // once for it), and the same bits for the rows of a shorter pass.
      {
        const int R3 = 150;
        const Vec w2 = ops::to_vec(&view2);
        flyx_tensor_view dense = view2;
        dense.dtype = FLYX_DT_F64, dense.data = w2.data(), dense.nbytes = w2.size() * sizeof(double);
        const Vec x3 = randv(size_t(R3) * K2);
        Vec want3(size_t(R3) * N2);
        ops::linear_a16h(x3.data(), R3, K2, &dense, N2, want3.data());
        float* dx3 = up(x3);
        float* dy3 = dev<float>(want3.size());
        auto pass = [&](int rows) {
          CK(cudaMemset(dy3, 0xFF, want3.size() * sizeof(float)));
          gemm_blocks(dx3, K2, W2, K2, dy3, N2, rows, false, TL);
          CK(cudaGetLastError());
          return down(dy3, size_t(rows) * N2);
        };
        const Vec y150 = pass(R3);
        expect(tag + "150 rows of a longer row", y150, want3, 2e-5);
        for (int rows : {17, 129})
          expect(tag + std::to_string(rows) + " rows of it: the bits of 150", pass(rows),
                 Vec(y150.begin(), y150.begin() + size_t(rows) * N2), 0);
        BlockMat W8 = W2;
        W8.q8 = true;
        ops::linear_a8(x3.data(), R3, K2, &dense, N2, want3.data());
        auto pass8 = [&](int rows) {
          CK(cudaMemset(dy3, 0xFF, want3.size() * sizeof(float)));
          gemm_blocks(dx3, K2, W8, K2, dy3, N2, rows, false, TL);
          CK(cudaGetLastError());
          return down(dy3, size_t(rows) * N2);
        };
        const Vec y8 = pass8(R3);
        expect(tag + "150 rows of a longer row, inputs in 8 bits", y8, want3, 2e-5);
        // Every compiled configuration, its runs of K in turn or side by side (their sums through the scratch, or with
        // no room there folded onto y in order), onto y or not: one set of bits.  For the table's runs (the bits of the
        // pass above), and for two and a run a window (other bits, by the same rules).
        const int windows = (K2 + 255) / 256;
        Scratch no_parts = *TL.scratch;
        no_parts.q8_parts_cap = 0;
        Launch TL_in_order = TL;
        TL_in_order.scratch = &no_parts;
        const Vec y0 = randv(want3.size());
        Vec want_acc(want3.size());
        for (size_t i = 0; i < want3.size(); ++i) want_acc[i] = double(float(y0[i])) + want3[i];
        for (int want_runs : {0, 2, windows}) {
          const int runs =
              want_runs ? std::min(want_runs, windows) : gemm_blocks_q8_runs(TL.opt ? TL.opt->tuning : -1, N2, K2);
          const std::string in_runs = "K in " + std::to_string(runs) + (want_runs ? " runs" : " runs (the table's)");
          Vec base = want_runs ? Vec() : y8, acc8;
          bool same8 = true, same_acc = true;
          for (int cfg = 0; cfg < kBlocksQ8ConfigCount; ++cfg) {
            if (!gemm_blocks_q8_cfg_compiled(W8.dtype, cfg)) continue;
            for (int way : {0, 1, 2}) {  // in turn; side by side; side by side, in order
              const int spread = way ? 1 : 0;
              const Launch& L = way == 2 ? TL_in_order : TL;
              CK(cudaMemset(dy3, 0xFF, want3.size() * sizeof(float)));
              gemm_blocks_q8_cfg(cfg, W8, K2, dy3, N2, R3, false, L, spread, want_runs);
              CK(cudaGetLastError());
              const Vec got = down(dy3, want3.size());
              if (base.empty()) base = got;
              same8 = same8 && got == base;
              CK(cudaMemcpy(dy3, std::vector<float>(y0.begin(), y0.end()).data(), want3.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
              gemm_blocks_q8_cfg(cfg, W8, K2, dy3, N2, R3, true, L, spread, want_runs);
              CK(cudaGetLastError());
              const Vec on_y = down(dy3, want3.size());
              if (acc8.empty()) acc8 = on_y;
              same_acc = same_acc && on_y == acc8;
            }
          }
          std::printf("  %-4s %sinputs in 8 bits, %s: the same bits from every configuration, in turn or side by "
                      "side\n", same8 ? "ok" : "FAIL", tag.c_str(), in_runs.c_str());
          g_fail += !same8;
          if (want_runs) expect(tag + "150 rows, inputs in 8 bits, " + in_runs, base, want3, 2e-5);
          expect(tag + "150 rows onto y, inputs in 8 bits, " + in_runs, acc8, want_acc, 2e-5);
          std::printf("  %-4s %sonto y, %s: the same bits from every configuration and spread\n",
                      same_acc ? "ok" : "FAIL", tag.c_str(), in_runs.c_str());
          g_fail += !same_acc;
        }
        for (int rows : {17, 129})
          expect(tag + std::to_string(rows) + " rows of it, inputs in 8 bits: the bits of 150", pass8(rows),
                 Vec(y8.begin(), y8.begin() + size_t(rows) * N2), 0);
        CK(cudaFree(dx3));
        CK(cudaFree(dy3));
      }
      CK(cudaFree(const_cast<uint8_t*>(W2.data)));
      CK(cudaFree(const_cast<uint16_t*>(W2.scales)));
    }
  }

  // A row with values far below its largest: binary16 over the row's scale drops bits of them (the reference
  // rounds the same way), and a row whose values are all tiny or all huge loses nothing.
  {
    const BlockType& bt = kBlockTypes[10];  // IQ2_XXS
    const int N = 24, K = 512, R = 4;
    const auto file = random_blocks(bt, size_t(N) * 2);
    const BlockMat W = up_blocks(bt.dtype, file, N);
    flyx_tensor_view view{};
    view.dtype = bt.dtype, view.ndim = 2, view.shape[0] = N, view.shape[1] = K;
    view.data = file.data(), view.nbytes = file.size();
    Vec x = randv(size_t(R) * K);
    for (int k = 0; k < K; ++k) {
      x[k] *= std::ldexp(1.0, -(k % 48));       // row 0: 48 binades
      x[size_t(K) + k] *= 1e-30;                // row 1: all tiny
      x[2 * size_t(K) + k] *= 1e30;             // row 2: all huge
      if (k % 7) x[3 * size_t(K) + k] = 0;      // row 3: mostly zeros
    }
    x[5] = 3e4;
    Vec want(size_t(R) * N);
    ops::linear_a16h(x.data(), R, K, &view, N, want.data());
    float* dx = up(x);
    float* dy = dev<float>(want.size());
    gemm_blocks(dx, K, W, K, dy, N, R, false, TL);
    const Vec got = down(dy, want.size());
    for (int r = 0; r < R; ++r)
      expect(std::string("rows over their scales: ") + (r == 0 ? "48 binades" : r == 1 ? "tiny" : r == 2 ? "huge" : "sparse"),
             Vec(got.begin() + size_t(r) * N, got.begin() + size_t(r + 1) * N),
             Vec(want.begin() + size_t(r) * N, want.begin() + size_t(r + 1) * N), 2e-5);
    CK(cudaFree(dx));
    CK(cudaFree(dy));
    CK(cudaFree(const_cast<uint8_t*>(W.data)));
    CK(cudaFree(const_cast<uint16_t*>(W.scales)));
  }
}

// The DFlash2 drafter's Q8_0 copy of its bf16 projections (quantize_q8_0): every code and scale as the
// host computes them from the bf16 rows, laid out as blocks_pack lays out a GGUF's Q8_0 blocks; a block of zeros
// takes d = 0 and codes 0.
// A stack's runs in one round launch (gemm_blocks_ops.h: round_stack_ops_<group>): for every stack of two or three
// types compiled together, every configuration of the dp4a kernel compiled for it, at the rows it holds, gives each
// run's columns the bits of the run alone; so does a stack as the engine runs it (a run before the stack's), and
// accumulating.
void test_blocks_round_stacks() {
  std::printf("GGML block weights, a stack's runs in one round launch\n");
  seed("blocks round stacks");
  const int K = 11 * 256, R = 16, widths[3] = {70, 37, 29};  // every run ends in a warp's columns past N
  const int ld = 70 + 37 + 29 + 70;                         // room for a run before the stack
  const Vec x = randv(size_t(R) * K), y0 = randv(size_t(R) * ld);
  float* dx = up(x);
  float* dy = dev<float>(size_t(R) * ld);
  auto fill = [&](int rows, bool accumulate) {
    if (accumulate)
      CK(cudaMemcpy(dy, std::vector<float>(y0.begin(), y0.begin() + size_t(rows) * ld).data(), size_t(rows) * ld * 4,
                    cudaMemcpyHostToDevice));
    else CK(cudaMemset(dy, 0xFF, size_t(rows) * ld * 4));
    blocks_stage(dx, K, K, rows, TL, true);
  };
  auto same_rows = [&](const Vec& got, const Vec& want, int columns, bool accumulate) {
    for (size_t i = 0; i < want.size(); ++i) {
      if (int(i % ld) < columns) {
        if (!std::isfinite(got[i]) || got[i] != want[i]) return false;
      } else if (accumulate) {
        if (got[i] != float(y0[i]) || want[i] != float(y0[i])) return false;
      } else if (!std::isnan(got[i]) || !std::isnan(want[i])) {
        return false;  // neither launch may write a padding column
      }
    }
    return true;
  };
  int stacks = 0;
  const int nt = int(std::size(kBlockTypes));
  for (int n : {2, 3})
    for (int code = 0; code < (n == 2 ? nt * nt : nt * nt * nt); ++code) {
      const BlockType* t[3] = {&kBlockTypes[code % nt], &kBlockTypes[code / nt % nt], &kBlockTypes[code / nt / nt % nt]};
      const int dtype[3] = {t[0]->dtype, t[1]->dtype, t[2]->dtype};
      if (!gemm_blocks_round_stack_compiled(dtype, n)) continue;
      ++stacks;
      BlockMat W[4];  // the stack's runs, then (W[n]) the first again, the run before it in the engine's stack
      for (int r = 0; r <= n; ++r) {
        const BlockType& bt = *t[r < n ? r : 0];
        W[r] = up_blocks(bt.dtype, random_blocks(bt, size_t(widths[r < n ? r : 0]) * K / flyx_block_values(bt.dtype)),
                         widths[r < n ? r : 0]);
        W[r].q8 = true;
      }
      int cols = 0;
      for (int r = 0; r < n; ++r) cols += W[r].rows;
      std::string tag;
      for (int r = 0; r < n; ++r) tag += std::string(r ? " then " : "") + t[r]->name;
      tag += ": ";
      auto alone = [&](const BlockMat* runs, int count, int rows, bool accumulate) {  // a launch a run
        fill(rows, accumulate);
        for (int r = 0, n0 = 0; r < count; n0 += runs[r].rows, ++r)
          gemm_blocks_staged(runs[r], K, dy + n0, ld, rows, accumulate, TL);
        CK(cudaGetLastError());
        return down(dy, size_t(rows) * ld);
      };
      bool same = true;
      int ran = 0;
      for (int rows : {1, 2, 3, 5, 8, 9, 16}) {
        const Vec want = alone(W, n, rows, false);
        for (int c = 0; c < kBlocksRoundConfigCount; ++c) {
          if (rows > kBlocksRoundConfigs[c].rows) continue;
          fill(rows, false);
          if (!gemm_blocks_round_q8_stack_cfg(c, W, n, K, dy, ld, rows, false, TL)) continue;
          CK(cudaGetLastError());
          ++ran;
          same = same && same_rows(down(dy, size_t(rows) * ld), want, cols, false);
        }
      }
      same = same && ran > 0;
      std::printf("  %-4s %severy configuration compiled for it (%d launches) gives the bits of a launch a run\n",
                  same ? "ok" : "FAIL", tag.c_str(), ran);
      g_fail += !same;
      BlockMat engine[4] = {W[n]};  // the first type's run, then the stack
      for (int r = 0; r < n; ++r) engine[r + 1] = W[r];
      for (bool accumulate : {false, true})
        for (int rows : {1, 4, 8}) {
          const Vec want = alone(engine, n + 1, rows, accumulate);
          fill(rows, accumulate);
          gemm_blocks_staged_runs(engine, n + 1, K, dy, ld, rows, accumulate, TL);
          CK(cudaGetLastError());
          const bool match = same_rows(down(dy, size_t(rows) * ld), want, W[n].rows + cols, accumulate);
          std::printf("  %-4s %safter a run of the first type, as the engine runs it, %d rows%s\n",
                      match ? "ok" : "FAIL", tag.c_str(), rows, accumulate ? ", accumulating" : "");
          g_fail += !match;
        }
      for (int r = 0; r <= n; ++r) {
        CK(cudaFree(const_cast<uint8_t*>(W[r].data)));
        CK(cudaFree(const_cast<uint16_t*>(W[r].scales)));
      }
    }
  std::printf("  %-4s stacks compiled: %d\n", stacks ? "ok" : "FAIL", stacks);
  g_fail += !stacks;
  CK(cudaFree(dx));
  CK(cudaFree(dy));
}

// A round's SwiGLU in its gate|up GEMM (Launch::glu): the 8-bit rows swiglu_split leaves, bit for bit (integers,
// scales, sums), and the gate|up rows as without it, from every dp4a configuration compiled with a SwiGLU that holds
// the round's rows, the gate and up of one type (one run, one launch) or of two (a launch each); and as the engine
// runs it where gemm_blocks_round_glu_takes says it can.
void test_blocks_round_glu() {
  std::printf("GGML block weights, a round's SwiGLU in its gate|up GEMM\n");
  seed("blocks round glu");
  const int K = 4 * 256, I = 2560, ld = 2 * I + 64;  // I over 2048: the up run is never stacked after the gate's
  const Scratch& sc = *TL.scratch;
  const Vec x = randv(size_t(16) * K);
  float* dx = up(x);
  float* dy = dev<float>(size_t(16) * ld);
  struct Bytes {
    std::vector<int8_t> q;
    std::vector<float> d, s;
    std::vector<float> y;
    bool operator==(const Bytes& o) const {
      return q == o.q && !std::memcmp(d.data(), o.d.data(), d.size() * 4) && d.size() == o.d.size() &&
             s.size() == o.s.size() && !std::memcmp(s.data(), o.s.data(), s.size() * 4) && y.size() == o.y.size() &&
             !std::memcmp(y.data(), o.y.data(), y.size() * 4);
    }
  };
  auto read = [&](const int8_t* q, const float* d, const float* s, int rows) {
    Bytes b;
    b.q.resize(size_t(rows) * I);
    b.d.resize(size_t(rows) * I / 32);
    b.s.resize(size_t(rows) * I / 16);
    CK(cudaMemcpy(b.q.data(), q, b.q.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.d.data(), d, b.d.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.s.data(), s, b.s.size() * 4, cudaMemcpyDeviceToHost));
    const Vec all = down(dy, size_t(rows) * ld);
    for (int r = 0; r < rows; ++r)
      b.y.insert(b.y.end(), all.begin() + size_t(r) * ld, all.begin() + size_t(r) * ld + 2 * I);
    return b;
  };
  auto start = [&](int rows) {  // the round's staged rows, the outputs unwritten
    blocks_stage(dx, K, K, rows, TL, true);
    CK(cudaMemset(dy, 0xFF, size_t(rows) * ld * 4));
    CK(cudaMemset(sc.q8_glu, 0x55, size_t(rows) * I));
    CK(cudaMemset(sc.q8_glu_scale, 0x55, size_t(rows) * I / 32 * 4));
    CK(cudaMemset(sc.q8_glu_sums, 0x55, size_t(rows) * I / 16 * 4));
  };
  const int pairs[][2] = {{FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_IQ3_XXS},
                          {FLYX_DT_GGML_IQ2_XXS, FLYX_DT_GGML_IQ3_S},
                          {FLYX_DT_GGML_Q4_K, FLYX_DT_GGML_Q6_K},
                          {FLYX_DT_GGML_IQ1_S, FLYX_DT_GGML_Q4_0}};
  for (const auto& pr : pairs) {
    const BlockType &bg = block_type(pr[0]), &bu = block_type(pr[1]);
    const int n = pr[0] == pr[1] ? 1 : 2;
    BlockMat W[2];
    if (n == 1) {
      W[0] = up_blocks(bg.dtype, random_blocks(bg, size_t(2) * I * K / flyx_block_values(bg.dtype)), 2 * I);
    } else {
      W[0] = up_blocks(bg.dtype, random_blocks(bg, size_t(I) * K / flyx_block_values(bg.dtype)), I);
      W[1] = up_blocks(bu.dtype, random_blocks(bu, size_t(I) * K / flyx_block_values(bu.dtype)), I);
    }
    for (int r = 0; r < n; ++r) W[r].q8 = true;
    const std::string tag = std::string(bg.name) + (n == 1 ? " gate|up: " : std::string(" gate, ") + bu.name + " up: ");
    bool same = true;
    int ran = 0, engine = 0;
    for (int rows : {1, 2, 3, 4}) {
      start(rows);
      gemm_blocks_staged_runs(W, n, K, dy, ld, rows, false, TL);
      swiglu_split(dy, ld, I, rows, TL, true);
      CK(cudaGetLastError());
      const Bytes want = read(sc.q8, sc.q8_scale, sc.q8_sums, rows);
      const RoundGlu glu{dy, ld, I};
      Launch L = TL;
      L.glu = &glu;
      for (int c = 0; c < kBlocksRoundConfigCount; ++c) {
        const BlocksRoundConfig& k = kBlocksRoundConfigs[c];
        if (rows > k.rows || k.rows > 4 || k.parts != 1 || k.staged == 2) continue;
        const bool compiled = n == 1 ? gemm_blocks_round_glu_compiled(W[0].dtype, c)
                                     : (gemm_blocks_round_cfg_compiled(W[0].dtype, c) &&
                                        gemm_blocks_round_glu_compiled(W[1].dtype, c));
        if (!compiled) continue;
        start(rows);
        for (int r = 0, n0 = 0; r < n; n0 += W[r].rows, ++r)
          gemm_blocks_round_q8_cfg(c, W[r], K, dy + n0, ld, rows, false, L);
        CK(cudaGetLastError());
        ++ran;
        const bool ok = read(sc.q8_glu, sc.q8_glu_scale, sc.q8_glu_sums, rows) == want;
        if (!ok && same) std::printf("  (first difference: %d rows, configuration %d)\n", rows, c);
        same = same && ok;
      }
      if (gemm_blocks_round_glu_takes(W, n, rows, I, TL)) {
        start(rows);
        gemm_blocks_staged_runs(W, n, K, dy, ld, rows, false, L);
        CK(cudaGetLastError());
        ++engine;
        const bool ok = read(sc.q8_glu, sc.q8_glu_scale, sc.q8_glu_sums, rows) == want;
        if (!ok && same) std::printf("  (first difference: %d rows, as the engine runs it)\n", rows);
        same = same && ok;
      }
    }
    same = same && ran > 0;
    std::printf("  %-4s %sswiglu_split's 8-bit rows and the gate|up rows (%d launches, %d as the engine)\n",
                same ? "ok" : "FAIL", tag.c_str(), ran, engine);
    g_fail += !same;
    for (int r = 0; r < n; ++r) {
      CK(cudaFree(const_cast<uint8_t*>(W[r].data)));
      CK(cudaFree(const_cast<uint16_t*>(W[r].scales)));
    }
  }
  CK(cudaFree(dx));
  CK(cudaFree(dy));
}

// A stack's narrow launches on a side stream beside its wide ones (Launch::side): the bits of one stream, with the
// defaults' rules and with this GPU's table (where its rounds of 9-16 rows take the split kernel, whose sums and
// counters on the side stream are its own: side_scratch).
void test_blocks_round_side() {
  std::printf("GGML block weights, a stack's narrow launches on a side stream\n");
  seed("blocks round side");
  const int K = 4 * 256, n = 3, widths[n] = {3072, 96, 1024};  // a wide run, then narrow ones of other types
  const int dtypes[n] = {FLYX_DT_GGML_IQ3_XXS, FLYX_DT_GGML_Q8_0, FLYX_DT_GGML_IQ4_XS};
  const int ld = 3072 + 96 + 1024;
  BlockMat W[n];
  for (int r = 0; r < n; ++r) {
    const BlockType& bt = block_type(dtypes[r]);
    W[r] = up_blocks(bt.dtype, random_blocks(bt, size_t(widths[r]) * K / flyx_block_values(bt.dtype)), widths[r]);
    W[r].q8 = true;
  }
  const Vec x = randv(size_t(16) * K);
  float* dx = up(x);
  float* dy = dev<float>(size_t(16) * ld);
  SideStream side;
  CK(cudaStreamCreateWithFlags(&side.st, cudaStreamNonBlocking));
  CK(cudaEventCreateWithFlags(&side.fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&side.join, cudaEventDisableTiming));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  for (int tuning : {-1, gemm_tuning_for(prop.name, prop.major * 10 + prop.minor)}) {
    KernelOptions o;
    o.tuning = tuning;
    o.pdl = tuning >= 0 && gemm_tuning_pdl(tuning);
    Launch one = TL, two = TL;
    one.opt = two.opt = &o;
    two.side = &side;
    for (int rows : {1, 4, 8, 9, 16}) {
      auto run = [&](const Launch& on) {
        blocks_stage(dx, K, K, rows, on, true);
        CK(cudaMemset(dy, 0xFF, size_t(rows) * ld * 4));
        gemm_blocks_staged_runs(W, n, K, dy, ld, rows, false, on);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        return down(dy, size_t(rows) * ld);
      };
      const Vec want = run(one), got = run(two);
      expect(std::string("tuning ") + std::to_string(tuning) + ", " + std::to_string(rows) +
                 " rows: the bits of one stream",
             got, want, 0);
    }
  }
  CK(cudaEventDestroy(side.fork));
  CK(cudaEventDestroy(side.join));
  CK(cudaStreamDestroy(side.st));
  for (int r = 0; r < n; ++r) {
    CK(cudaFree(const_cast<uint8_t*>(W[r].data)));
    CK(cudaFree(const_cast<uint16_t*>(W[r].scales)));
  }
  CK(cudaFree(dx));
  CK(cudaFree(dy));
}

void test_quantize_q8_0() {
  std::printf("Q8_0 copy of bf16 rows\n");
  seed("quantize_q8_0");
  const int N = 37, K = 256, nb = N * (K / 32);
  Vec w = randv(size_t(N) * K, 3.0);
  for (int e = 0; e < 32; ++e) w[size_t(3) * K + e] = 0.0;  // row 3's first block: zeros
  bf16* dw = up_bf16(w);
  uint8_t* dpay = dev<uint8_t>(size_t(nb) * 32);
  uint16_t* dsc = dev<uint16_t>(size_t(nb));
  quantize_q8_0(dw, N, K, dpay, dsc, 0);
  CK(cudaDeviceSynchronize());
  std::vector<uint8_t> pay(size_t(nb) * 32), file(size_t(nb) * 34), want_pay(pay.size());
  std::vector<uint16_t> sc(nb), want_sc(nb);
  CK(cudaMemcpy(pay.data(), dpay, pay.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sc.data(), dsc, sc.size() * 2, cudaMemcpyDeviceToHost));
  for (int b = 0; b < nb; ++b) {  // the host's blocks as a GGUF holds them: binary16 d, then 32 codes
    float v[32], amax = 0.f;
    for (int e = 0; e < 32; ++e) {
      v[e] = float(from_bf16(to_bf16(w[size_t(b) * 32 + e])));
      amax = std::max(amax, std::fabs(v[e]));
    }
    const uint16_t d16 = to_f16(double(amax / 127.f));
    const float d = float(from_f16(d16)), inv = d > 0.f ? 1.f / d : 0.f;
    std::memcpy(&file[size_t(b) * 34], &d16, 2);
    for (int e = 0; e < 32; ++e) {
      const int q = std::max(-127, std::min(127, int(std::nearbyint(v[e] * inv))));
      file[size_t(b) * 34 + 2 + e] = uint8_t(int8_t(q));
    }
  }
  blocks_pack(FLYX_DT_GGML_Q8_0, file.data(), size_t(nb), want_pay.data(), want_sc.data());
  const bool same = pay == want_pay && sc == want_sc;
  std::printf("  %-4s %-60s\n", same ? "ok" : "FAIL", "codes and scales as the host's, as blocks_pack lays them out");
  g_fail += !same;

  // Row-chunked streaming quantization on a stream must match whole-matrix quantization bit for bit.
  uint8_t* dpay2 = dev<uint8_t>(size_t(nb) * 32);
  uint16_t* dsc2 = dev<uint16_t>(size_t(nb));
  cudaStream_t st = nullptr;
  CK(cudaStreamCreate(&st));
  for (int r0 = 0; r0 < N; r0 += 7) {
    const int nr = std::min(7, N - r0);
    quantize_q8_0(dw + size_t(r0) * K, nr, K, dpay2 + size_t(r0) * K, dsc2 + size_t(r0) * (K / 32), st);
  }
  CK(cudaStreamSynchronize(st));
  CK(cudaStreamDestroy(st));
  std::vector<uint8_t> pay2(pay.size());
  std::vector<uint16_t> sc2(nb);
  CK(cudaMemcpy(pay2.data(), dpay2, pay2.size(), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sc2.data(), dsc2, sc2.size() * 2, cudaMemcpyDeviceToHost));
  const bool same_chunk = pay == pay2 && sc == sc2;
  std::printf("  %-4s %-60s\n", same_chunk ? "ok" : "FAIL", "chunked streaming == whole matrix (bit-identical)");
  g_fail += !same_chunk;
  for (void* p : std::initializer_list<void*>{dw, dpay, dsc, dpay2, dsc2}) CK(cudaFree(p));
}

// The drafts' two-stage head on a packed head (draft_pick_blocks): the 1-bit copy made from the rows
// embed_blocks decodes, in two runs of rows as the backend makes it a run at a time, against the host's from the
// rows' values; with every token a candidate (V <= kDraftCands) the exact top 16 over the vocabulary, in order, and
// their logits against float64.  A K-quant and an IQ type, K = 512 (two blocks a row, four spans of 128).
void test_draft_pick_blocks() {
  std::printf("draft_pick on packed blocks\n");
  seed("draft_pick_blocks");
  const int V = 200, K = 512, k = 16;
  for (const BlockType& bt : kBlockTypes) {
    if (bt.dtype != FLYX_DT_GGML_Q4_K && bt.dtype != FLYX_DT_GGML_IQ2_S) continue;
    const std::string tag = std::string(bt.name) + " ";
    const auto file = random_blocks(bt, size_t(V) * (K / flyx_block_values(bt.dtype)));
    flyx_tensor_view view{};
    view.dtype = bt.dtype, view.ndim = 2, view.shape[0] = V, view.shape[1] = K;
    view.data = file.data(), view.nbytes = file.size();
    const BlockMat W = up_blocks(bt.dtype, file, V);
    std::vector<float> w(size_t(V) * K);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(ops::load(&view, i));

    std::vector<int> ids(V);
    for (int i = 0; i < V; ++i) ids[size_t(i)] = i;
    int* dids = dev<int>(V);
    CK(cudaMemcpy(dids, ids.data(), sizeof(int) * V, cudaMemcpyHostToDevice));
    float* drows = dev<float>(size_t(V) * K);
    embed_blocks(dids, W, K, drows, V, 0);
    SignHead h;
    const size_t tiled = size_t(V + 255) / 256 * 256;
    h.bits = dev<uint32_t>(tiled * K / 32);
    h.scales = dev<bf16>(tiled * K / 64);
    make_sign_head_rows(drows, 0, V / 2, K, &h, 0);
    make_sign_head_rows(drows + size_t(V / 2) * K, V / 2, V - V / 2, K, &h, 0);
    std::vector<uint32_t> bits(tiled * K / 32);
    std::vector<uint16_t> sc(tiled * K / 64);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(bits.data(), h.bits, bits.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(sc.data(), h.scales, sc.size() * 2, cudaMemcpyDeviceToHost));
    bool same = true;
    for (int n = 0; n < V; ++n)
      for (int g = 0; g < K / 64; ++g) {
        float s = 0.f;
        for (int j = 0; j < 64; ++j) s += std::fabs(w[size_t(n) * K + 64 * g + j]);
        same = same && to_bf16(s / 64) == sc[sign_scale_at(n, g, K)];
        for (int j = 0; j < 64; ++j) {
          const int q = 64 * g + j;
          const bool pos = !std::signbit(w[size_t(n) * K + q]);
          same = same && pos == bool((bits[sign_word_at(n, q / 32, K)] >> (q % 32)) & 1);
        }
      }
    std::printf("  %-4s %s%-44s\n", same ? "ok" : "FAIL", tag.c_str(), "the 1-bit copy as the host makes it");
    g_fail += !same;

    // Three rows (a DFlash2 block's drafts; an MTP step has one).
    const int rows = 3;
    const Vec x = randv(size_t(rows) * K);
    float* dx = up(x);
    {  // dots_blocks: bit for bit the dots of the rows embed_blocks decodes, read 4 values a lane in order, then the
       // warp's tree; rows of fewer dots than the cap (nothing past them written), indices in any order
      const int cap = 24, counts[rows] = {cap, 17, 1};
      const size_t ld = 3 * cap;  // a row: indices [0, cap), dots [cap, 2 cap), its count at 2 cap
      std::vector<int> buf(size_t(rows) * ld, 0);
      for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cap; ++c) buf[size_t(r) * ld + c] = (r * 37 + c * 53 + 11) % V;
        buf[size_t(r) * ld + 2 * cap] = counts[r];
      }
      int* dbuf = dev<int>(buf.size());
      CK(cudaMemcpy(dbuf, buf.data(), sizeof(int) * buf.size(), cudaMemcpyHostToDevice));
      dots_blocks(W, K, dbuf, dbuf + 2 * cap, reinterpret_cast<float*>(dbuf + cap), ld, cap, dx, K, rows, 0);
      std::vector<uint32_t> got(buf.size());
      std::vector<float> wf(size_t(V) * K), xf(x.begin(), x.end());
      CK(cudaMemcpy(got.data(), dbuf, sizeof(int) * got.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(wf.data(), drows, sizeof(float) * wf.size(), cudaMemcpyDeviceToHost));
      bool exact = true;
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cap; ++c) {
          float want_bits = 0.f;  // past the count: as it was
          if (c < counts[r]) {
            const float* wr = wf.data() + size_t(buf[size_t(r) * ld + c]) * K;
            const float* xr = xf.data() + size_t(r) * K;
            float s[32];
            for (int l = 0; l < 32; ++l) {
              s[l] = 0.f;
              for (int q = l; q < K / 4; q += 32)
                for (int e = 0; e < 4; ++e) s[l] = std::fmaf(wr[4 * q + e], xr[4 * q + e], s[l]);
            }
            for (int o = 16; o > 0; o >>= 1) {
              float t[32];
              for (int l = 0; l < 32; ++l) t[l] = s[l] + s[l ^ o];
              std::copy(t, t + 32, s);
            }
            want_bits = s[0];
          }
          uint32_t want;
          std::memcpy(&want, &want_bits, 4);
          exact = exact && got[size_t(r) * ld + cap + c] == want;
        }
      std::printf("  %-4s %s%-44s\n", exact ? "ok" : "FAIL", tag.c_str(), "dots_blocks: the decoded rows' dots, bit for bit");
      g_fail += !exact;
      CK(cudaFree(dbuf));
    }
    float* rough = dev<float>(size_t(rows) * std::max(V, kDraftPickFloats));
    int* di = dev<int>(size_t(rows) * k);
    float* dv = dev<float>(size_t(rows) * k);
    draft_pick_blocks(dx, K, h, W, V, K, rows, k, di, dv, rough, TL);
    std::vector<int> gi(size_t(rows) * k);
    CK(cudaMemcpy(gi.data(), di, sizeof(int) * gi.size(), cudaMemcpyDeviceToHost));
    bool order = true;
    Vec want(size_t(rows) * k);
    for (int r = 0; r < rows; ++r) {
      std::vector<std::pair<double, int>> ex(V);
      for (int n = 0; n < V; ++n) {
        double a = 0;
        for (int q = 0; q < K; ++q) a += x[size_t(r) * K + q] * double(w[size_t(n) * K + q]);
        ex[size_t(n)] = {-a, n};
      }
      std::partial_sort(ex.begin(), ex.begin() + k, ex.end());
      for (int j = 0; j < k; ++j) {
        order = order && gi[size_t(r) * k + j] == ex[size_t(j)].second;
        want[size_t(r) * k + j] = -ex[size_t(j)].first;
      }
    }
    std::printf("  %-4s %s%-44s\n", order ? "ok" : "FAIL", tag.c_str(),
                "the exact top 16 over the vocabulary, in order, 3 rows");
    g_fail += !order;
    expect(tag + "their exact logits", down(dv, size_t(rows) * k), want, 1e-5);
    for (void* p : std::initializer_list<void*>{dids, drows, h.bits, h.scales, dx, rough, di, dv}) CK(cudaFree(p));
    CK(cudaFree(const_cast<uint8_t*>(W.data)));
    CK(cudaFree(const_cast<uint16_t*>(W.scales)));
  }
}

}  // namespace flyx::optest
