// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// CUDA kernel tests for Inkling-2 target ops (softcap, AttnRes, loop control, ShortConv,
// relative-position-bias attention, sparse MoE) and DSpark drafter kernels against float64 refs.
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
      {"global f16 hd128 R=37", 8, 2, 128, 16, 128, 0, 0, 50, 37, false},
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
    expect(std::string("rel_attn ") + c.name, got_o, want_o, c.fp8 ? 5e-4 : 3e-5);

    // Row-count independence: check BOTH row 0 (inside a full RT=4 tile) and last row (run alone vs multi).
    if (c.rows > 1) {
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
  msc.down_buf = dev<float>(size_t(R) * top_k * H);
  msc.chunk_rows = R;
  msc.max_experts = E;
  msc.max_top_k = top_k;

  auto run_and_check = [&](const std::string& label, const ExpertMat& w_gu, const ExpertMat& w_dn, const Vec& gu_f64,
                           const Vec& dn_f64, bool round_a16, bool round_q8, bool round_a16h, double tol) {
    // Reference in float64 with the exact input rounding contract of this weight format.
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
    Vec got_act_f = down(d_act_fused, size_t(R) * top_k * I);
    Vec got_act_u = down(d_act_unfused, size_t(R) * top_k * I);
    expect("moe_gate_up " + label, got_act_f, want_act, tol);
    expect_bits("moe_gate_up fused==unfused " + label, got_act_f, got_act_u);

    float* d_want_act = up(want_act);
    moe_down(d_want_act, I, w_dn, d_eidx, d_ew, R, top_k, H, I, d_y_grouped, H, false, &msc, TL);
    Vec got_y_grp_clean = down(d_y_grouped, size_t(R) * H);
    expect("moe_down " + label, got_y_grp_clean, want_out, tol);
    cudaFree(d_want_act);

    // Gate 4: direct single-row decode (R=1, sc=nullptr) vs grouped prefill (R=37, sc=&msc) MUST BE BIT-IDENTICAL!
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

  // (c) Packed GGML block experts: MXFP4, Q2_0, Q4_K, Q8_0 (both Q8 and exact a16h).
  const int ggml_types[] = {FLYX_DT_GGML_MXFP4, FLYX_DT_GGML_Q2_0, FLYX_DT_GGML_Q4_K, FLYX_DT_GGML_Q8_0};
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

  cudaFree(d_x);
  cudaFree(d_eidx);
  cudaFree(d_ew);
  cudaFree(msc.counts);
  cudaFree(msc.offsets);
  cudaFree(msc.pairs);
  cudaFree(msc.down_buf);
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
