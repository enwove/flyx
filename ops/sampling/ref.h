// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Token choice in float64: argmax, sampling as ops/sampling/sampling.h
// defines it, and top-k.
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "flyx/backend_abi.h"
#include "ops/sampling/sampling.h"

namespace flyx::ops {

inline int argmax(const double* v, int n, const uint32_t* mask = nullptr) {
  int best = -1;
  for (int i = 0; i < n; ++i)
    if (allowed(mask, i) && (best < 0 || v[i] > v[best])) best = i;
  return best < 0 ? 0 : best;
}

// Token choice for one row of logits (see ops/sampling/sampling.h for the definition).
// `pos` is the absolute position the chosen token will occupy; `mask` the
// tokens allowed there (null: all); `counts[v]` how often token v was chosen
// earlier in the reply (null: a reply without penalties, or its first token).
inline int sample_token(const double* logits, int V, const flyx_gen_params& g, int64_t pos,
                        const uint32_t* mask = nullptr, const int* counts = nullptr) {
  std::vector<double> penalized;
  if (counts && (g.presence_penalty != 0.f || g.frequency_penalty != 0.f)) {
    penalized.assign(logits, logits + V);
    for (int v = 0; v < V; ++v)
      if (counts[v] > 0) penalized[v] -= penalty(double(g.presence_penalty), double(g.frequency_penalty), counts[v]);
    logits = penalized.data();
  }
  if (!(g.temperature > 0)) return argmax(logits, V, mask);
  double m = logits[0];
  for (int v = 1; v < V; ++v) m = std::max(m, logits[v]);
  std::vector<std::pair<double, int>> c;
  for (int v = 0; v < V; ++v) {
    double z = (logits[v] - m) / g.temperature;
    if (z > kSampleZMin) c.push_back({z, v});
  }
  std::sort(c.begin(), c.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
  size_t keep = c.size();
  auto keep_at_least = [&](double x) {
    size_t k = 0;
    while (k < keep && c[k].first >= x) ++k;
    keep = k;
  };
  if (g.top_k > 0 && size_t(g.top_k) < keep) keep_at_least(c[g.top_k - 1].first);
  if (g.top_p < 1.f) {
    double total = 0, cum = 0;
    for (size_t i = 0; i < keep; ++i) total += std::exp(c[i].first);
    for (size_t i = 0; i < keep; ++i) {
      cum += std::exp(c[i].first);
      if (cum >= double(g.top_p) * total) {
        keep_at_least(c[i].first);
        break;
      }
    }
  }
  int best = -1;
  double best_s = -INFINITY;
  for (size_t i = 0; i < keep; ++i) {
    if (!allowed(mask, c[i].second)) continue;
    double u = sample_uniform(g.seed, pos, c[i].second);
    double s = c[i].first - std::log(-std::log(u));
    if (best < 0 || s > best_s || (s == best_s && c[i].second < best)) {
      best_s = s;
      best = c[i].second;
    }
  }
  if (best >= 0) return best;
  // No kept token is allowed: the best allowed token of all, by the same noisy score.
  // Center at an allowed maximum so the noise on equally likely allowed tokens survives a very small temperature.
  if (mask) m = logits[argmax(logits, V, mask)];
  for (int v = 0; v < V; ++v) {
    if (!allowed(mask, v)) continue;
    double u = sample_uniform(g.seed, pos, v);
    double s = (logits[v] - m) / g.temperature - std::log(-std::log(u));
    if (best < 0 || s > best_s || (s == best_s && v < best)) {
      best_s = s;
      best = v;
    }
  }
  return best < 0 ? 0 : best;
}

// Top-k indices, highest first; ties go to the lower index.
inline void topk(const double* v, int n, int k, int* idx, double* val) {
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
    return v[a] > v[b] || (v[a] == v[b] && a < b);
  });
  for (int i = 0; i < k; ++i) {
    idx[i] = order[i];
    val[i] = v[order[i]];
  }
}

}  // namespace flyx::ops
