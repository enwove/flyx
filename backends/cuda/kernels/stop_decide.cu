// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Device-side intra-round confidence optimal stopping decisions for adaptive CUDA conditional graphs.
// Works identically for sequential drafters (MTP, EAGLE-4) via stop_decide_step and parallel block
// drafters (DFlash2, DSpark) via stop_decide_block, using only drafter-agnostic (prob, margin) and
// core/speculation/'s flyx_stop_policy.
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__device__ inline float d_stop_draft_prob(float prob, float margin, int pos, const flyx_stop_policy& pol) {
  if (prob <= 0.f) return pol.p_pos[pos < FLYX_MAX_ROWS ? pos : FLYX_MAX_ROWS - 1];
  const float p = fminf(fmaxf(prob, 1e-5f), 1.f - 1e-5f);
  const float lp = fminf(fmaxf(logf(p / (1.f - p)), -6.f), 10.f);
  const float mg = fminf(fmaxf(margin, 0.f), 12.f);
  const float z = fminf(fmaxf(pol.w_bias + pol.w_margin * mg + pol.w_logit_p * lp, -20.f), 20.f);
  return 1.f / (1.f + expf(-z));
}

__global__ void k_stop_decide_step(const float* __restrict__ prob, const float* __restrict__ margin,
                                   const flyx_stop_policy* __restrict__ pol, DevStopState* __restrict__ state,
                                   int bucket, int prev_w, int cur_w, unsigned long long h_next_draft,
                                   unsigned long long h_ver) {
  const flyx_stop_policy P = *pol;
  const int n_buckets = P.n_buckets < 1 ? 1 : (P.n_buckets > FLYX_MAX_STOP_BUCKETS ? FLYX_MAX_STOP_BUCKETS : P.n_buckets);
  const int lo = prev_w < 0 ? 0 : (prev_w > FLYX_MAX_ROWS ? FLYX_MAX_ROWS : prev_w);
  const int hi = cur_w < lo ? lo : (cur_w > FLYX_MAX_ROWS ? FLYX_MAX_ROWS : cur_w);
  float S = bucket == 0 ? 1.f : state->survival;
  int cont = 0;
  if (!P.enabled) {
    cont = (bucket + 1 < n_buckets) ? 1 : 0;
  } else {
    for (int k = lo; k < hi; ++k) S *= d_stop_draft_prob(prob[k], margin[k], k, P);
    if (bucket + 1 < n_buckets) {
      const float c_cur = P.costs[bucket < n_buckets ? bucket : n_buckets - 1];
      for (int nxt = bucket + 1; nxt < n_buckets; ++nxt) {
        const int w_nxt = P.widths[nxt] < hi ? hi : (P.widths[nxt] > FLYX_MAX_ROWS ? FLYX_MAX_ROWS : P.widths[nxt]);
        float exp_more = 0.f;
        float prod_p = S;
        for (int j = hi; j < w_nxt; ++j) {
          prod_p *= P.p_pos[j];
          exp_more += prod_p;
        }
        const float dc = P.costs[nxt] - c_cur;
        if (exp_more > P.lambda * dc) {
          cont = 1;
          break;
        }
      }
    }
  }
  if (h_ver && !h_next_draft) cont = 0;
  state->survival = S;
  state->decision = cont;
#if CUDART_VERSION >= 12040
  if (cont) {
    if (h_next_draft) cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_next_draft), 1u);
  } else {
    if (h_ver) cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_ver), 1u);
  }
#else
  (void)h_next_draft;
  (void)h_ver;
#endif
}

__global__ void k_stop_decide_block(const float* __restrict__ prob, const float* __restrict__ margin,
                                    const flyx_stop_policy* __restrict__ pol, DevStopState* __restrict__ state,
                                    StopCondHandles h_ver) {
  const flyx_stop_policy P = *pol;
  int n = P.n_buckets < 1 ? 1 : (P.n_buckets > FLYX_MAX_STOP_BUCKETS ? FLYX_MAX_STOP_BUCKETS : P.n_buckets);
  if (h_ver.h[0]) {
    while (n > 1 && !h_ver.h[n - 1]) --n;
  }
  int best = n - 1;
  float last_s = 1.f;
  if (n > 1 && P.enabled) {
    int w_max = P.widths[n - 1];
    if (w_max < 0) w_max = 0;
    if (w_max > FLYX_MAX_ROWS) w_max = FLYX_MAX_ROWS;
    float S[FLYX_MAX_ROWS];
    float prod = 1.f;
    for (int k = 0; k < w_max; ++k) {
      prod *= d_stop_draft_prob(prob[k], margin[k], k, P);
      S[k] = prod;
    }
    last_s = prod;
    best = 0;
    float best_val = -1e30f;
    const float c0 = P.costs[0];
    for (int b = 0; b < n; ++b) {
      int wb = P.widths[b];
      if (wb < 0) wb = 0;
      if (wb > w_max) wb = w_max;
      float etok = 1.f;
      for (int k = 0; k < wb; ++k) etok += S[k];
      const float val = etok - P.lambda * (P.costs[b] - c0);
      if (val > best_val) {
        best_val = val;
        best = b;
      }
    }
  }
  state->survival = last_s;
  state->decision = best;
#if CUDART_VERSION >= 12040
  if (best >= 0 && best < FLYX_MAX_STOP_BUCKETS && h_ver.h[best])
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_ver.h[best]), 1u);
#else
  (void)h_ver;
#endif
}

}  // namespace

void stop_decide_step(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                      int bucket, int prev_w, int cur_w, unsigned long long h_next_draft, unsigned long long h_ver,
                      cudaStream_t st) {
  k_stop_decide_step<<<1, 1, 0, st>>>(prob, margin, pol, state, bucket, prev_w, cur_w, h_next_draft, h_ver);
}

void stop_decide_block(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                       StopCondHandles h_ver, cudaStream_t st) {
  k_stop_decide_block<<<1, 1, 0, st>>>(prob, margin, pol, state, h_ver);
}

}  // namespace flyx::cuda
