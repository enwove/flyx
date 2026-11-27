// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// gpt-oss's pieces against the float64 reference: its experts (gate | up apart in the file, MXFP4, each expert's
// biases, the clamped activation: fused in the gate | up GEMM, or moe_act_rows after it; the down biases in moe_down)
// at its shapes and the tiny model's,
// and its attention (a learned sink a head; a sliding window over a ring of slots; 8 query heads of 64 dims a KV head;
// YaRN rotary factors in attn_prep).
#include "tests/backends/cuda/op_test.cuh"

#include "ops/activation/ref.h"
#include "ops/attention/ref.h"
#include "ops/dtype/ggml.h"
#include "ops/moe/ref.h"
#include "ops/rope/ref.h"

namespace flyx::optest {
namespace {

constexpr int kRounds = 64, kPrefills = 8;  // attention()'s splits in a round and a prefill (backend.h)

void round_q8_row(double* row, int K) {
  std::vector<int8_t> p(static_cast<size_t>(K));
  std::vector<float> d(static_cast<size_t>(K / 32));
  ops::round_q8(row, K, p.data(), d.data());
  for (int i = 0; i < K; ++i) row[i] = double(p[size_t(i)]) * double(d[size_t(i / 32)]);
}

MoeScratch moe_scratch(int R, int top_k, int E, int H, int I) {
  MoeScratch msc{};
  msc.pairs = dev<int>(size_t(R) * top_k);
  msc.tasks = dev<MoeTask>(size_t(R) * top_k);
  msc.n_tasks = dev<int>(1);
  msc.down_buf = dev<float>(size_t(R) * top_k * std::max(H, 2 * I));
  msc.stage_elems = std::max(size_t(R) * H, size_t(R) * top_k * I);
  const size_t q8_a = (msc.stage_elems + 255) / 256 * 256;
  const size_t q8_d = (((msc.stage_elems + 31) / 32) * sizeof(float) + 255) / 256 * 256;
  const size_t q8_s = (((msc.stage_elems + 15) / 16) * sizeof(float) + 255) / 256 * 256;
  uint8_t* stage = dev<uint8_t>(std::max(msc.stage_elems * sizeof(bf16), q8_a + q8_d + q8_s));
  msc.q8_a = reinterpret_cast<int8_t*>(stage);
  msc.q8_d = reinterpret_cast<float*>(stage + q8_a);
  msc.q8_s = reinterpret_cast<float*>(stage + q8_a + q8_d);
  msc.bf16_a = reinterpret_cast<bf16*>(stage);
  msc.chunk_rows = R;
  msc.max_experts = E;
  msc.max_top_k = top_k;
  return msc;
}

// The experts at (E, top_k, H, I), R rows, MXFP4 with 8-bit or binary16-pair inputs.
void moe_case(int E, int top_k, int H, int I, int R, bool q8) {
  const float alpha = 1.702f, limit = 7.f;
  seed("gpt_oss_moe", E, top_k, H, I, R, int(q8));
  const BlockType& bt = block_type(FLYX_DT_GGML_MXFP4);
  auto experts = [&](int N, int K, ExpertMat* out, Vec* f64) {
    const std::vector<uint8_t> raw = random_blocks(bt, size_t(E) * N * (K / 32));
    f64->resize(size_t(E) * N * K);
    for (size_t i = 0; i < f64->size(); ++i) (*f64)[i] = ops::ggml::value(FLYX_DT_GGML_MXFP4, raw.data(), i);
    out->n_experts = E;
    out->N = N;
    out->blk = up_blocks(FLYX_DT_GGML_MXFP4, raw, E * N);
    out->blk.q8 = q8;
  };
  ExpertMat w_gu{}, w_dn{};
  Vec gu, dn;
  experts(2 * I, H, &w_gu, &gu);  // an expert's gate rows, then its up rows (expert_mat's stack of the two)
  experts(H, I, &w_dn, &dn);
  // MXFP4's random blocks hold values of up to 6 times a scale of up to 2^something: inputs small enough that the
  // activation's clamp both bites and does not.
  const Vec x = randv(size_t(R) * H, 0.02);
  const Vec gu_b = randv(size_t(E) * 2 * I, 1.0), dn_b = randv(size_t(E) * H, 1.0);
  const Vec logits = randv(size_t(R) * E, 1.0);
  std::vector<int> eidx(size_t(R) * top_k);
  Vec ew(size_t(R) * top_k);
  for (int r = 0; r < R; ++r)
    ops::moe_route(&logits[size_t(r) * E], nullptr, E, top_k, false, true, 0.0, 1.0, &eidx[size_t(r) * top_k],
                   &ew[size_t(r) * top_k]);

  Vec want_act(size_t(R) * top_k * I), want_y(size_t(R) * H, 0.0);
  for (int r = 0; r < R; ++r) {
    Vec xr(x.begin() + ptrdiff_t(r) * H, x.begin() + ptrdiff_t(r + 1) * H);
    q8 ? round_q8_row(xr.data(), H) : ops::round_a16h(xr.data(), H);
    for (int k = 0; k < top_k; ++k) {
      const int e = eidx[size_t(r) * top_k + k];
      Vec g(size_t(2) * I);
      ops::linear_expert_f64(xr.data(), H, gu.data(), e, 2 * I, g.data());
      Vec a(static_cast<size_t>(I));
      for (int i = 0; i < I; ++i)
        a[size_t(i)] = ops::swiglu_oai(g[size_t(i)] + gu_b[size_t(e) * 2 * I + i],
                                        g[size_t(I + i)] + gu_b[size_t(e) * 2 * I + I + i], alpha, limit);
      std::copy(a.begin(), a.end(), want_act.begin() + ptrdiff_t(r * top_k + k) * I);
      q8 ? round_q8_row(a.data(), I) : ops::round_a16h(a.data(), I);
      Vec y(static_cast<size_t>(H));
      ops::linear_expert_f64(a.data(), I, dn.data(), e, H, y.data());
      for (int c = 0; c < H; ++c)
        want_y[size_t(r) * H + c] += ew[size_t(r) * top_k + k] * (y[size_t(c)] + dn_b[size_t(e) * H + c]);
    }
  }

  float* d_x = up(x);
  float* d_gub = up(gu_b);
  float* d_dnb = up(dn_b);
  float* d_ew = up(ew);
  int* d_eidx = dev<int>(eidx.size());
  CK(cudaMemcpy(d_eidx, eidx.data(), eidx.size() * sizeof(int), cudaMemcpyHostToDevice));
  float* d_gu = dev<float>(size_t(R) * top_k * 2 * I);
  float* d_act = dev<float>(size_t(R) * top_k * I);
  float* d_fused = dev<float>(size_t(R) * top_k * I);
  float* d_y = dev<float>(size_t(R) * H);
  MoeScratch msc = moe_scratch(R, top_k, E, H, I);
  const MoeAct act{FLYX_MOE_SWIGLU_OAI, alpha, limit, d_gub};
  moe_gate_up(d_x, H, w_gu, d_eidx, R, top_k, I, H, act, false, d_gu, 2 * I, &msc, TL);
  moe_act_rows(d_gu, 2 * I, I, act, d_eidx, d_act, I, R * top_k, 0);
  const int grouped = moe_gate_up(d_x, H, w_gu, d_eidx, R, top_k, I, H, act, true, d_fused, I, &msc, TL);
  moe_down(d_fused, I, w_dn, d_eidx, d_ew, R, top_k, H, I, d_y, H, false, &msc, TL, nullptr, grouped, d_dnb);
  const std::string tag = "E " + std::to_string(E) + " k " + std::to_string(top_k) + " H " + std::to_string(H) +
                          " I " + std::to_string(I) + " R " + std::to_string(R) + (q8 ? " q8" : " a16h");
  const Vec unfused = down(d_act, want_act.size()), fused = down(d_fused, want_act.size());
  expect("gpt-oss experts' activation " + tag, unfused, want_act, 5e-5);
  expect("gpt-oss experts' activation fused in the GEMM, as moe_act_rows' " + tag, fused, unfused, 0.0);
  expect("gpt-oss experts' output " + tag, down(d_y, want_y.size()), want_y, 5e-5);
  for (void* p : {static_cast<void*>(d_x), static_cast<void*>(d_gub), static_cast<void*>(d_dnb),
                  static_cast<void*>(d_ew), static_cast<void*>(d_eidx), static_cast<void*>(d_gu),
                  static_cast<void*>(d_act), static_cast<void*>(d_fused), static_cast<void*>(d_y)})
    CK(cudaFree(p));
}

// One head's attention as the engine runs a round's rows: the rows at positions base .. base + R - 1 over a cache
// of keys [0, base + R) (window 0) or a ring of `ring` slots holding the last positions (window > 0), with sinks.
void attention_case(int nh, int nkv, int hd, int base, int R, int window, int ring, bool sinks, int splits) {
  seed("gpt_oss_attention", nh, nkv, hd, base, R, window, ring, int(sinks), splits);
  const int T = base + R;
  const Vec q = randv(size_t(R) * nh * hd, 1.0);
  Vec K = randv(size_t(T) * nkv * hd, 1.0), V = randv(size_t(T) * nkv * hd, 1.0);
  for (double& v : K) v = double(__half2float(__float2half_rn(float(v))));  // a binary16 cache
  for (double& v : V) v = double(__half2float(__float2half_rn(float(v))));
  const Vec sink = randv(size_t(nh), 2.0);
  const double scale = 1.0 / std::sqrt(double(hd));
  Vec want(size_t(R) * nh * hd);
  for (int r = 0; r < R; ++r)
    for (int h = 0; h < nh; ++h) {
      const int pos = base + r, from = window > 0 ? std::max(0, pos + 1 - window) : 0;
      double* o = &want[(size_t(r) * nh + h) * hd];
      if (sinks)
        ops::attend_range_sink(&q[(size_t(r) * nh + h) * hd], K.data(), V.data(), from, pos + 1, nkv, hd,
                               h / (nh / nkv), scale, sink[size_t(h)], o);
      else
        ops::attend_range(&q[(size_t(r) * nh + h) * hd], K.data(), V.data(), from, pos + 1, nkv, hd, h / (nh / nkv),
                          scale, o);
    }
  const int slots = ring > 0 ? ring : T;
  std::vector<kv16> k16(size_t(slots) * nkv * hd), v16(size_t(slots) * nkv * hd);
  for (int p = ring > 0 ? std::max(0, T - ring) : 0; p < T; ++p)
    for (int i = 0; i < nkv * hd; ++i) {
      const size_t s = size_t(ring > 0 ? p % ring : p) * nkv * hd + i;
      k16[s] = __float2half_rn(float(K[size_t(p) * nkv * hd + i]));
      v16[s] = __float2half_rn(float(V[size_t(p) * nkv * hd + i]));
    }
  kv16* dk = dev<kv16>(k16.size());
  kv16* dv = dev<kv16>(v16.size());
  CK(cudaMemcpy(dk, k16.data(), k16.size() * sizeof(kv16), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, v16.data(), v16.size() * sizeof(kv16), cudaMemcpyHostToDevice));
  KvRef kv{dk, dv, nullptr, nullptr};
  kv.ring = ring;
  float* dq = up(q);
  float* ds = sinks ? up(sink) : nullptr;
  float* d_out = dev<float>(want.size());
  float* d_part = dev<float>(attention_partial_floats(R, nh, hd, splits));
  DevParams P{};
  P.position = base;
  DevParams* dP = dev<DevParams>(1);
  CK(cudaMemcpy(dP, &P, sizeof(P), cudaMemcpyHostToDevice));
  attention(dq, kv, nh, nkv, hd, R, true, 0, float(scale), dP, nullptr, 0, d_out, nh * hd, d_part, splits, TL, 0,
            window, nullptr, CanvasKeys{}, ds);
  const std::string tag = std::to_string(nh) + "/" + std::to_string(nkv) + "x" + std::to_string(hd) + " pos " +
                          std::to_string(base) + " rows " + std::to_string(R) +
                          (window ? " window " + std::to_string(window) + " ring " + std::to_string(ring) : "") +
                          (sinks ? " sinks" : "");
  // The kernels round q and the probabilities through binary16 on the tensor-core paths: their tolerance.
  expect("attention " + tag, down(d_out, want.size()), want, R > FLYX_MAX_ROWS ? 4e-3 : 2e-4);
  for (void* p : {static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(dq), static_cast<void*>(d_out),
                  static_cast<void*>(d_part), static_cast<void*>(dP)})
    CK(cudaFree(p));
  if (ds) CK(cudaFree(ds));
}

// moe_route with a bias on the router's logits (gpt-oss's): the softmax over the k best of logits + bias.
void route_case() {
  const int R = 9, E = 32, top_k = 4;
  seed("gpt_oss_route", R, E, top_k);
  const Vec logits = randv(size_t(R) * E, 2.0), lb = randv(size_t(E), 1.0);
  std::vector<int> want_i(size_t(R) * top_k);
  Vec want_w(size_t(R) * top_k);
  for (int r = 0; r < R; ++r) {
    Vec l(logits.begin() + ptrdiff_t(r) * E, logits.begin() + ptrdiff_t(r + 1) * E);
    for (int e = 0; e < E; ++e) l[size_t(e)] += lb[size_t(e)];
    ops::moe_route(l.data(), nullptr, E, top_k, false, true, 0.0, 1.0, &want_i[size_t(r) * top_k],
                   &want_w[size_t(r) * top_k]);
  }
  float* dl = up(logits);
  float* db = up(lb);
  int* di = dev<int>(want_i.size());
  float* dw = dev<float>(want_w.size());
  moe_route(dl, nullptr, R, E, top_k, false, true, 0.f, 1.f, di, dw, TL, db);
  std::vector<int> got_i(want_i.size());
  CK(cudaMemcpy(got_i.data(), di, got_i.size() * sizeof(int), cudaMemcpyDeviceToHost));
  const bool same = got_i == want_i;
  std::printf("  %-4s %-48s\n", same ? "ok" : "FAIL", "moe_route with the router's logit bias: the experts");
  g_fail += !same;
  expect("moe_route with the router's logit bias: the weights", down(dw, want_w.size()), want_w, 1e-5);
  for (void* p : {static_cast<void*>(dl), static_cast<void*>(db), static_cast<void*>(di), static_cast<void*>(dw)})
    CK(cudaFree(p));
}

// attn_prep's rotary with YaRN's factors (gpt-oss's: theta 150000, factor 32 over 4096, betas 32 and 1, the ramp not
// rounded) for q and k heads at far positions, against ops::rope_with on the float64 factors.
void yarn_case() {
  const int nh = 4, nkv = 2, hd = 64, R = 3;
  seed("gpt_oss_yarn", nh, nkv, hd, R);
  std::vector<double> inv(size_t(hd / 2));
  ops::yarn_inv_freq(150000.0, hd, 32.0, 4096, 32.0, 1.0, false, inv.data());
  const Vec qkv = randv(size_t(R) * (nh + 2 * nkv) * hd, 1.0);
  const Vec bias = randv(size_t(nh + 2 * nkv) * hd, 0.5);  // q|k|v's biases, added as attn_prep reads the rows
  const int base = 100000;
  Vec want_q(size_t(R) * nh * hd), want_k(size_t(R) * nkv * hd);
  for (int r = 0; r < R; ++r) {
    for (int h = 0; h < nh + nkv; ++h) {
      Vec y(qkv.begin() + ptrdiff_t(r * (nh + 2 * nkv) + h) * hd,
            qkv.begin() + ptrdiff_t(r * (nh + 2 * nkv) + h + 1) * hd);
      for (int d = 0; d < hd; ++d) y[size_t(d)] += bias[size_t(h) * hd + d];
      ops::rope_with(y.data(), base + r, inv.data(), hd);
      std::copy(y.begin(), y.end(), h < nh ? want_q.begin() + ptrdiff_t(r * nh + h) * hd
                                           : want_k.begin() + ptrdiff_t(r * nkv + h - nh) * hd);
    }
  }
  double* d_inv = dev<double>(inv.size());
  CK(cudaMemcpy(d_inv, inv.data(), inv.size() * sizeof(double), cudaMemcpyHostToDevice));
  float* d_qkv = up(qkv);
  float* d_bias = up(bias);
  float* d_q = dev<float>(want_q.size());
  std::vector<kv16> zeros(size_t(base + R) * nkv * hd);
  kv16* k16 = dev<kv16>(zeros.size());
  kv16* v16 = dev<kv16>(zeros.size());
  DevParams P{};
  P.position = base;
  DevParams* dP = dev<DevParams>(1);
  CK(cudaMemcpy(dP, &P, sizeof(P), cudaMemcpyHostToDevice));
  KvRef kv{k16, v16, nullptr, nullptr};
  const int ld = (nh + 2 * nkv) * hd;
  attn_prep(d_qkv, ld, 0, nh * hd, (nh + nkv) * hd, nullptr, nullptr, 0.f, 0.f, 150000.0, hd, nh, nkv, hd, R, dP,
            kPosPosition, false, d_q, kv, 0, 0, nullptr, 0, CanvasKeys{}, MRoPE{}, d_inv, d_bias);
  expect("YaRN rotary of q (attn_prep's table)", down(d_q, want_q.size()), want_q, 2e-6);
  std::vector<kv16> got16(size_t(R) * nkv * hd);
  CK(cudaMemcpy(got16.data(), k16 + size_t(base) * nkv * hd, got16.size() * sizeof(kv16), cudaMemcpyDeviceToHost));
  Vec got_k(got16.size());
  for (size_t i = 0; i < got16.size(); ++i) got_k[i] = __half2float(got16[i]);
  expect("YaRN rotary of k (binary16 cache)", got_k, want_k, 1e-3);
  Vec want_v(size_t(R) * nkv * hd), got_v(size_t(R) * nkv * hd);
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < nkv * hd; ++i)
      want_v[size_t(r) * nkv * hd + i] = qkv[size_t(r) * ld + (nh + nkv) * hd + i] + bias[size_t(nh + nkv) * hd + i];
  CK(cudaMemcpy(got16.data(), v16 + size_t(base) * nkv * hd, got16.size() * sizeof(kv16), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < got16.size(); ++i) got_v[i] = __half2float(got16[i]);
  expect("q, k and v biases as attn_prep reads the rows (v in the cache)", got_v, want_v, 1e-3);
  CK(cudaFree(d_bias));
  for (void* p : {static_cast<void*>(d_inv), static_cast<void*>(d_qkv), static_cast<void*>(d_q),
       static_cast<void*>(k16), static_cast<void*>(v16), static_cast<void*>(dP)})
    CK(cudaFree(p));
}

}  // namespace

void test_gpt_oss() {
  std::printf("gpt-oss (experts with biases and the clamped activation, attention with sinks and windows, YaRN)\n");
  for (bool q8 : {true, false}) {
    for (auto [H, I] : {std::pair{64, 96}, {64, 160}, {256, 192}}) {
      for (int R : {1, 5, 37}) moe_case(4, 2, H, I, R, q8);
    }
    for (int R : {1, 3, 40}) moe_case(32, 4, 2880, 2880, R, q8);  // gpt-oss-20b's experts
  }
  // Rounds and prefills, with and without sinks, the grouped kernel's 8 heads of 64 a KV head, and a sliding window of
  // 128 over a ring of 128 + a round's slots.
  for (bool sinks : {false, true}) {
    for (int R : {1, 4, 16}) attention_case(64, 8, 64, 300, R, 0, 0, sinks, kRounds);
    attention_case(64, 8, 64, 0, 77, 0, 0, sinks, kPrefills);
    attention_case(64, 8, 64, 200, 96, 0, 0, sinks, kPrefills);
    for (int R : {1, 4}) attention_case(64, 8, 64, 500, R, 128, 128 + FLYX_MAX_ROWS, sinks, kRounds);
    attention_case(64, 8, 64, 0, 40, 128, 128 + 64, sinks, kPrefills);
    // prefill passes the window bites in (k_flash_prefill's sliding window), and whose keys wrap the ring
    attention_case(64, 8, 64, 100, 96, 32, 160, sinks, kPrefills);
    attention_case(64, 8, 64, 300, 40, 128, 128 + 64, sinks, kPrefills);
  }
  yarn_case();
  route_case();
}

}  // namespace flyx::optest
