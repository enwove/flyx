// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "flyx/backend_abi.h"
#include "ops/activation/ref.h"

namespace flyx::ops {

// prior[0] is the nearest ancestor, clipped to the sequence boundary. The current EOS still reads its history.
inline void ple_indices(int32_t token, const int32_t* prior, int available, const flyx_ple_block& p, int32_t* out) {
  uint64_t hash = uint64_t(token) * p.multipliers[0];
  bool ended = false;
  for (int distance = 1; distance < p.ngram; ++distance) {
    ended = ended || distance > available || prior[distance - 1] == p.eos;
    const int32_t id = ended ? p.eos : prior[distance - 1];
    hash ^= uint64_t(id) * p.multipliers[distance];
    for (int head = 0; head < p.heads_per_order; ++head) {
      const int h = (distance - 1) * p.heads_per_order + head;
      out[h] = int32_t(p.offsets[h] + hash % p.moduli[h]);
    }
  }
}

inline double ple_gate(const double* key, const double* query, int hidden) {
  double score = 0;
  for (int i = 0; i < hidden; ++i) score += key[i] * query[i];
  score /= std::sqrt(double(hidden));
  return sigmoid(score == 0 ? 0 : std::copysign(std::sqrt(std::max(std::abs(score), 1e-6)), score));
}

// The window contains the oldest (taps-1)*dilation through the most recent activation.
inline double ple_shortconv(const double* window, const double* current, const double* w, int channel, int channels,
                            int taps, int dilation) {
  double sum = 0;
  for (int j = 0; j < taps - 1; ++j)
    sum += window[size_t(j) * dilation * channels + channel] * w[size_t(channel) * taps + j];
  return silu(sum + current[channel] * w[size_t(channel) * taps + taps - 1]);
}

}  // namespace flyx::ops
