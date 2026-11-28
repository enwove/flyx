// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace flyx::cuda {

struct ExpertOrder {
  explicit ExpertOrder(size_t n) : slot(n), id(n), scores(n), counts(n) {
    std::iota(slot.begin(), slot.end(), 0);
    id = slot;
  }
  std::vector<int> slot, id;
  std::vector<uint64_t> scores;
  std::vector<uint32_t> counts;

  void observe() {
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = scores[i] / 2 + counts[i];
  }

  // Swaps are between whole sessions. Prefer the existing resident on ties and use logical ID to break other ties.
  template <class Exchange>
  size_t update(int hot, const uint64_t* score, Exchange exchange, bool all = false) {
    size_t swaps = 0;
    for (size_t limit = all ? id.size() : (id.size() + 7) / 8; swaps < limit; ++swaps) {
      auto less = [&](int a, int b) { return score[a] != score[b] ? score[a] < score[b] : a > b; };
      auto victim = std::min_element(id.begin(), id.begin() + hot, less);
      auto incoming = std::max_element(id.begin() + hot, id.end(), less);
      if (victim == id.begin() + hot || incoming == id.end() || score[*incoming] <= score[*victim]) break;
      exchange(slot[*victim], slot[*incoming]);
      std::swap(slot[*victim], slot[*incoming]);
      std::iter_swap(victim, incoming);
    }
    return swaps;
  }
};

}  // namespace flyx::cuda
