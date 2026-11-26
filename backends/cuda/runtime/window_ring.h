// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Caches whose rows read their last `window` positions kept as a ring of slots that every checkpoint copies to host
// memory (KvBuf::swa_ring: the target's and DSpark's sliding-window layers, and past 8 of option mtp_window's windows
// DSpark's Full layer and the MTP layer): how many slots they take, and from which position on they still hold the
// window a row reads.  Host only (tests/backends/cuda/window_ring_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>

#include "flyx/backend_abi.h"

namespace flyx::cuda {

// The ring's slots: its first `sink` positions kept for good (KvRef::slot_of), then the window plus the rows one
// prefill pass writes at once (rows_cap: a checkpoint inside the pass, prefill_snapshot's, then still finds its
// window) or the rows of the rounds a rewind reaches back across (FLYX_ROUND_SLOTS rounds and a wide round's), in
// whole 64s.  0 (a slot a position) without a window or where the ring would not be fewer slots than the cache's
// `cap` positions.  In 64 bits: a window may be INT_MAX.
inline int window_ring_slots(int64_t window, int64_t sink, int rows_cap, int wide_rows, int cap) {
  if (window <= 0 || cap <= 0) return 0;
  const int64_t span = std::max<int64_t>(int64_t(FLYX_ROUND_SLOTS) * FLYX_MAX_ROWS + wide_rows, rows_cap);
  const int64_t ring = (std::max<int64_t>(sink, 0) + window + span + 63) / 64 * 64;
  return ring < cap ? int(ring) : 0;
}

// The first position whose window the ring holds, positions before `ring_hi` the last written (and past the sink, of
// those the ring - sink latest): the row there reads [pos - window + 1, pos], its own written by its round.  The MTP
// layer keeps its entries a position behind (a pass from p writes entries p - 1 on; its draft step at p reads
// [p - window, p - 1] and writes p - 1): the same floor.
inline int64_t window_ring_floor(int64_t ring_hi, int ring, int sink, int64_t window) {
  return ring_hi - (ring - sink) + window - 1;
}

}  // namespace flyx::cuda
