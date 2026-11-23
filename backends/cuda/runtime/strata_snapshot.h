// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Canonical snapshot words of a 256d, three-bit Strata head tile and its compact host transport.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace flyx::cuda {
template <bool Key, bool Pack>
inline void strata_snapshot_tile(const uint32_t* in, uint32_t* out) {
  constexpr int own = Key ? 6 : 3, panels = Key ? 2 : 4;
  if constexpr (!Pack) std::fill(out + 384, out + 416, 0u);
  for (int panel = 0; panel < panels; ++panel) {
    for (int lane = 0; lane < 32; ++lane) {
      const int g = lane >> 2, t = lane & 3;
      const int shared = Key ? 16 * panel + 2 * g + (t >> 1) : (panel * 4 + t) * 2 + g / 5;
      const int shift = Key ? 6 * (t & 1) : 3 * (g % 5);
      uint32_t tail = Pack ? in[384 + shared] >> shift : 0;
      for (int i = 0; i < own; ++i) {
        const int j = (i * panels + panel) * 32 + lane;
        if constexpr (Pack) out[j] = (in[j] & 0x7fff7fffu) | (((tail >> i) & 0x10001u) << 15);
        else {
          out[j] = in[j] & 0x7fff7fffu;
          tail |= ((in[j] >> 15) & 0x10001u) << i;
        }
      }
      if constexpr (!Pack) out[384 + shared] |= tail << shift;
    }
  }
}

// Snapshot pieces may start and stop inside a word or tile. Merge partial loads with what a prior piece supplied.
template <bool Key>
inline void strata_snapshot_span(uint32_t* packed, uint8_t* host, uint64_t offset, uint64_t length, bool save) {
  uint32_t original[416];
  while (length) {
    const size_t inside = size_t(offset % sizeof(original));
    const size_t count = std::min<uint64_t>(length, sizeof(original) - inside);
    uint32_t* tile = packed + (offset / sizeof(original)) * 384;
    if (save || inside || count < sizeof(original)) strata_snapshot_tile<Key, false>(tile, original);
    auto* part = reinterpret_cast<uint8_t*>(original) + inside;
    if (save) std::memcpy(host, part, count);
    else {
      std::memcpy(part, host, count);
      strata_snapshot_tile<Key, true>(original, tile);
    }
    offset += count;
    host += count;
    length -= count;
  }
}
}  // namespace flyx::cuda
