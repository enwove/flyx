// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace flyx::ops {

inline double qsa_score(const double* queries, const double* pooled_key, int heads, int dim) {
  double total = 0;
  for (int h = 0; h < heads; ++h) {
    double dot = 0;
    for (int i = 0; i < dim; ++i) dot += queries[size_t(h) * dim + i] * pooled_key[i];
    total += std::max(0.0, dot);
  }
  return total / std::sqrt(double(dim));
}

// Equal scores choose the earlier group. Gather in logical KV order so the selected set alone determines the sum.
inline std::vector<int> qsa_positions(const std::vector<double>& scores, int visible, int ratio, int budget) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  const size_t keep = std::min(order.size(), size_t(budget / ratio));
  if (keep < order.size()) {
    std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                      [&](int a, int b) { return scores[a] != scores[b] ? scores[a] > scores[b] : a < b; });
    order.resize(keep);
    std::sort(order.begin(), order.end());
  }
  std::vector<int> positions;
  for (int group : order)
    for (int i = 0; i < ratio; ++i) positions.push_back(group * ratio + i);
  for (int t = int(scores.size()) * ratio; t < visible; ++t) positions.push_back(t);
  return positions;
}

}  // namespace flyx::ops
