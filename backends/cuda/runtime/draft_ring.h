// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A sliding-window DFlash2 drafter's context K/V kept as a ring of slots (KvBuf::ring): how many slots it takes, and
// which positions the ring still holds for the drafter's window, so that a rewind, a restore or a snapshot never hands
// the drafter slots a later position overwrote.  Host only (tests/backends/cuda/draft_ring_test.cpp).
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>

#include "flyx/backend_abi.h"

namespace flyx::cuda {

// The ring's slots for a drafter attending its last `window` positions: the window, plus the rows a prefill pass
// writes at once (rows_cap) or the rounds a rewind reaches back across write past it (FLYX_ROUND_SLOTS rounds of a
// block's or a wide round's rows), in whole 64s.  0 (a slot a position) with the ring off, without a window, or where
// the ring would not be fewer slots than the cache's `cap` positions.  In 64 bits: a window may be INT_MAX.
inline int dflash_ring_slots(bool on, int64_t window, int block, int rows_cap, int wide_rows, int cap) {
  if (!on || window <= 0 || cap <= 0) return 0;
  const int64_t rows = std::max<int64_t>(block, FLYX_MAX_ROWS);
  const int64_t extra = std::max<int64_t>(rows_cap, int64_t(FLYX_ROUND_SLOTS) * std::max<int64_t>(rows, wide_rows));
  const int64_t ring = (window + extra + 63) / 64 * 64;
  return ring < cap ? int(ring) : 0;
}

// What a ring of `slots` slots still holds for a drafter that, at position X, reads the `reach` positions before X.
// Position p is in slot p % slots, so writing position q overwrites q - slots.  The floor: every position in
// [floor, X) is still there (X the sequence's position).  A checkpoint keeps the first position its window needs and
// whether that window is still there: a write since that reached into it (a later position in one of its slots, or a
// sequence that went back before it) takes it away.  Without a ring (slots 0) everything is held.
class DraftRing {
 public:
  void init(int slots, int reach) {
    slots_ = std::max(0, slots);
    reach_ = std::max(0, reach);
    floor_ = 0;
    for (Kept& k : kept_) k = {};
  }
  int slots() const { return slots_; }
  // The first position the drafter at pos reads.
  int low(int pos) const { return std::max(0, pos - reach_); }
  // The drafter's window at pos (at most the sequence's position) is still in the ring.
  bool holds(int pos) const { return slots_ == 0 || low(pos) >= floor_; }
  // Positions [from, to) were written, or may have been (to: past the last row a pass may write, rejected rows and
  // a draft block's rows included).
  void wrote(int from, int to) {
    if (slots_ == 0 || to <= from) return;
    floor_ = std::max(floor_, to - slots_);
    for (Kept& k : kept_)
      if (k.ok && (from < k.end || to - slots_ > k.low)) k.ok = false;
  }
  // The sequence back at position 0: nothing before it to hold.  Checkpoints keep what they hold.
  void reset() { floor_ = 0; }
  // Checkpoint c taken at pos: its window, if the ring holds it now (else the checkpoint keeps none).
  bool keep(int c, int pos) {
    kept_[c] = {holds(pos), low(pos), pos};
    return kept_[c].ok;
  }
  // Checkpoint c's window is still in the ring: it may be restored or saved as a snapshot.
  bool kept(int c) const { return slots_ == 0 || kept_[c].ok; }
  // The sequence restored to checkpoint c: the ring holds its window, nothing before it.
  void restored(int c) {
    if (slots_) floor_ = kept_[c].low;
  }
  // Every slot about to be replaced (a snapshot's load): no checkpoint's window survives, and nothing is held until
  // loaded() says what the slots hold.
  void replacing() {
    if (slots_ == 0) return;
    for (Kept& k : kept_) k.ok = false;
    floor_ = INT_MAX;
  }
  // The slots replaced by a snapshot of the state at pos: they hold its window (snapshots are saved only of
  // checkpoints whose window is kept).
  void loaded(int pos) {
    if (slots_) floor_ = low(pos);
  }

 private:
  struct Kept {
    bool ok = false;
    int low = 0, end = 0;
  };
  int slots_ = 0, reach_ = 0, floor_ = 0;
  Kept kept_[FLYX_CHECKPOINTS]{};
};

}  // namespace flyx::cuda
