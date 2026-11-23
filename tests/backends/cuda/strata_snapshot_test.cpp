// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A snapshot piece can cut through the shared bits of a word or span two different head tiles.
#include "backends/cuda/runtime/strata_snapshot.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <random>

namespace {
template <bool Key>
bool check_pieces() {
  constexpr size_t compact_words = 384, file_words = 416, tiles = 3;
  constexpr size_t tile_bytes = file_words * sizeof(uint32_t);
  std::array<uint32_t, tiles * compact_words> source{}, restored{};
  std::mt19937 random(1729);
  for (auto& word : source) word = random();
  std::array<uint32_t, tiles * file_words> canonical{};
  for (size_t t = 0; t < tiles; ++t)
    flyx::cuda::strata_snapshot_tile<Key, false>(source.data() + t * compact_words, canonical.data() + t * file_words);

  auto* bytes = reinterpret_cast<uint8_t*>(canonical.data());
  std::array<uint8_t, sizeof(canonical)> expected{}, actual{};
  const auto piece = [&](size_t offset, size_t count) {
    flyx::cuda::strata_snapshot_span<Key>(restored.data(), bytes + offset, offset, count, false);
    std::memcpy(expected.data() + offset, bytes + offset, count);
    flyx::cuda::strata_snapshot_span<Key>(restored.data(), actual.data(), 0, actual.size(), true);
    return actual == expected;
  };
  // Read the middle of the shared words first, then a single byte and a cross-tile range.
  if (!piece(compact_words * sizeof(uint32_t) + 1, 13) || !piece(0, 1) ||
      !piece(tile_bytes - 3, tile_bytes + 7) || !piece(1, tile_bytes - 1) ||
      !piece(2 * tile_bytes, tile_bytes) || restored != source) return false;

  actual.fill(0);
  for (size_t offset = 0; offset < actual.size();) {
    const size_t count = std::min(size_t(257), actual.size() - offset);
    flyx::cuda::strata_snapshot_span<Key>(restored.data(), actual.data() + offset, offset, count, true);
    offset += count;
  }
  return actual == expected;
}
}  // namespace

int main() {
  if (check_pieces<true>() && check_pieces<false>()) return 0;
  std::fputs("three-bit Strata snapshot piece changed a code or an untouched byte\n", stderr);
  return 1;
}
