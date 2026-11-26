// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Device-side intra-round confidence optimal stopping decisions for adaptive CUDA conditional graphs.
// Works identically for step-wise drafters (MTP, EAGLE-4, DSpark Markov head) via stop_decide_step and parallel block
// drafters (DFlash2) via stop_decide_block, using only drafter-agnostic (prob, margin) and core/speculation/'s
// flyx_stop_policy.  A step that stops verifies what has been drafted, or, where the caller says a draft step costs
// little beside a verify row (earlier_prefix), the prefix that pays: the policy's costs are whole rounds', so two
// checkpoints differ by their draft steps too, which a stop has already paid.
#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {
namespace {

__device__ inline float d_stop_draft_prob(float prob, float margin, int pos, const flyx_stop_policy& pol) {
  const float p_pos = pol.p_pos[pos < FLYX_MAX_ROWS ? pos : FLYX_MAX_ROWS - 1];
  if (prob <= 0.f) return p_pos;
  const float p_base = fminf(fmaxf(p_pos, 1e-4f), 1.f - 1e-4f);
  const float base_logit = fminf(fmaxf(logf(p_base / (1.f - p_base)), -6.f), 6.f);
  const float p = fminf(fmaxf(prob, 1e-5f), 1.f - 1e-5f);
  const float lp = fminf(fmaxf(logf(p / (1.f - p)), -6.f), 10.f);
  const float mg = fminf(fmaxf(margin, 0.f), 12.f);
  const float z = fminf(fmaxf(base_logit + pol.w_bias + pol.w_margin * mg + pol.w_logit_p * lp, -20.f), 20.f);
  return 1.f / (1.f + expf(-z));
}

__global__ void k_stop_decide_step(const float* __restrict__ prob, const float* __restrict__ margin,
                                   const flyx_stop_policy* __restrict__ pol, DevStopState* __restrict__ state,
                                   int bucket, int prev_w, int cur_w, unsigned long long h_next_draft,
                                   StopCondHandles h_ver, bool earlier_prefix) {
  (void)prev_w;
  const flyx_stop_policy P = *pol;
  int n_buckets = P.n_buckets < 1 ? 1 : (P.n_buckets > FLYX_MAX_STOP_BUCKETS ? FLYX_MAX_STOP_BUCKETS : P.n_buckets);
  if (h_ver.h[0]) {
    while (n_buckets > 1 && !h_ver.h[n_buckets - 1]) --n_buckets;
  }
  const int b_cur = bucket < 0 ? 0 : (bucket >= n_buckets ? n_buckets - 1 : bucket);
  const int hi = cur_w < 0 ? 0 : (cur_w > FLYX_MAX_ROWS ? FLYX_MAX_ROWS : cur_w);
  float prod = 1.f;
  int best = b_cur;
  int cont = 0;
  if (!P.enabled) {
    cont = (b_cur + 1 < n_buckets) ? 1 : 0;
  } else {
    float S[FLYX_MAX_ROWS];
    for (int k = 0; k < hi; ++k) {
      prod *= d_stop_draft_prob(prob[k], margin[k], k, P);
      S[k] = prod;
    }
    const float c0 = P.costs[0];
    best = earlier_prefix ? 0 : b_cur;
    float best_val = -1e30f;
    float etok_cur = 1.f;
    for (int b = best; b <= b_cur; ++b) {
      int wb = P.widths[b];
      if (wb < 0) wb = 0;
      if (wb > hi) wb = hi;
      float etok = 1.f;
      for (int k = 0; k < wb; ++k) etok += S[k];
      if (b == b_cur) etok_cur = etok;
      const float val = etok - P.lambda * (P.costs[b] - c0);
      if (val > best_val) {
        best_val = val;
        best = b;
      }
    }
    if (b_cur + 1 < n_buckets) {
      for (int nxt = b_cur + 1; nxt < n_buckets; ++nxt) {
        const int w_nxt = P.widths[nxt] < hi ? hi : (P.widths[nxt] > FLYX_MAX_ROWS ? FLYX_MAX_ROWS : P.widths[nxt]);
        float exp_more = 0.f;
        float prod_p = prod;
        for (int j = hi; j < w_nxt; ++j) {
          prod_p *= P.p_pos[j];
          exp_more += prod_p;
        }
        const float val_nxt = (etok_cur + exp_more) - P.lambda * (P.costs[nxt] - c0);
        if (val_nxt > best_val) {
          cont = 1;
          break;
        }
      }
    }
  }
  if (h_ver.h[0] && !h_next_draft) cont = 0;
  state->survival = prod;
  state->decision = cont;
  state->verify_bucket = best;
#if CUDART_VERSION >= 12040
  if (cont) {
    if (h_next_draft) cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_next_draft), 1u);
  } else if (best >= 0 && best < FLYX_MAX_STOP_BUCKETS && h_ver.h[best]) {
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_ver.h[best]), 1u);
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
  state->verify_bucket = best;
#if CUDART_VERSION >= 12040
  if (best >= 0 && best < FLYX_MAX_STOP_BUCKETS && h_ver.h[best])
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(h_ver.h[best]), 1u);
#else
  (void)h_ver;
#endif
}

}  // namespace

void stop_decide_step(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                      int bucket, int prev_w, int cur_w, unsigned long long h_next_draft, StopCondHandles h_ver,
                      bool earlier_prefix, cudaStream_t st) {
  k_stop_decide_step<<<1, 1, 0, st>>>(prob, margin, pol, state, bucket, prev_w, cur_w, h_next_draft, h_ver,
                                      earlier_prefix);
}

void stop_decide_block(const float* prob, const float* margin, const flyx_stop_policy* pol, DevStopState* state,
                       StopCondHandles h_ver, cudaStream_t st) {
  k_stop_decide_block<<<1, 1, 0, st>>>(prob, margin, pol, state, h_ver);
}

}  // namespace flyx::cuda
