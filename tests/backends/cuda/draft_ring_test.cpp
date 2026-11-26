// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A sliding-window DFlash2 drafter's context ring (backends/cuda/runtime/draft_ring.h): its slots at the edges of the
// cache and for the largest window a drafter file may give, and the positions it still holds for the drafter's window
// through prefill passes, rounds, rewinds, checkpoints, restores and snapshots, as runtime/runner.cpp tells it.
#include "backends/cuda/runtime/draft_ring.h"

#include <climits>
#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

using flyx::cuda::dflash_ring_slots;
using flyx::cuda::DraftRing;

void slots() {
  // gguf_test's: a window of 8, a block of 16, prefill passes of 64 rows, rounds of 16 rows: 8 + 64 in whole 64s.
  const int block = 16, rows_cap = 64, wide = 16;
  check(dflash_ring_slots(true, 8, block, rows_cap, wide, 544) == 128, "a small window: 128 slots");
  check(dflash_ring_slots(true, 8, block, rows_cap, wide, 129) == 128, "  a cache one position longer: the ring");
  check(dflash_ring_slots(true, 8, block, rows_cap, wide, 128) == 0, "  a cache of as many positions: flat");
  check(dflash_ring_slots(true, 8, block, rows_cap, wide, 96) == 0, "  a cache of fewer positions: flat");
  check(dflash_ring_slots(false, 8, block, rows_cap, wide, 544) == 0, "  the ring off (dflash_ring=0): flat");
  check(dflash_ring_slots(true, 0, block, rows_cap, wide, 544) == 0, "no window: flat");
  check(dflash_ring_slots(true, 8, block, 16, 64, 1 << 20) == (8 + FLYX_ROUND_SLOTS * 64 + 63) / 64 * 64,
        "the rows of the rounds a rewind reaches back across, where they are more than a prefill pass's");
  // models/drafters/dflash2.cpp takes windows up to INT_MAX: in int, window + rows wrapped negative, passed
  // `ring < cap` and sized the cache by it.
  check(dflash_ring_slots(true, INT_MAX, block, rows_cap, wide, INT_MAX) == 0, "a window of INT_MAX: flat");
  check(dflash_ring_slots(true, INT_MAX - 64, block, 4096, 64, INT_MAX) == 0, "  INT_MAX - 64: flat");
  check(dflash_ring_slots(true, int64_t(1) << 30, block, 4096, 64, INT_MAX) == (1 << 30) + 4096,
        "a window of 2^30 beside a cache of INT_MAX positions: its ring");
}

void history() {
  // A ring of 128 slots for a window of 8 (the drafter at X reads [X - 8, X)).
  DraftRing none;
  none.wrote(0, 1 << 20);
  check(none.holds(0) && none.keep(0, 5) && none.kept(0), "no ring: every position held");

  DraftRing r;
  r.init(128, 8);
  r.wrote(0, 64);  // a prefill pass of 64 rows
  check(r.holds(64) && r.keep(0, 64), "a checkpoint after the first pass: its window held");
  r.wrote(64, 128);
  r.wrote(128, 184);  // position 183 overwrote 55, before the window [56, 64)
  check(r.kept(0), "  while later positions overwrite only the slots before its window");
  r.wrote(184, 185);  // 184 overwrites 56
  check(!r.kept(0), "  not once one overwrites its window's first slot: neither restored nor saved");
  check(r.holds(185) && r.keep(1, 185), "the position the sequence is at: held");
  r.wrote(185, 305);  // 304 overwrote 176, before its window [177, 185)
  check(r.kept(1), "  a checkpoint there kept until its window's first slot is overwritten");
  r.wrote(305, 306);  // 305 overwrites 177
  check(!r.kept(1), "  then refused");

  r.init(128, 8);
  r.wrote(0, 64);
  r.keep(0, 64);
  r.keep(1, 60);
  r.restored(1);
  r.wrote(60, 64);
  check(!r.kept(0) && r.kept(1), "divergent rows [60, 64) overwrite only the checkpoint at 64 without wrapping");

  // Restore: the floor is the checkpoint's window; the positions before it are no longer held.
  r.init(128, 8);
  r.wrote(0, 64);
  r.keep(2, 64);
  r.wrote(64, 120);
  check(r.kept(2), "a checkpoint the ring still holds");
  r.reset();  // rewind(0): the session's way back to a checkpoint
  r.restored(2);
  check(r.holds(64) && !r.holds(63), "  restored: its window held, nothing before it");
  r.wrote(64, 80);
  check(r.kept(2), "  and kept while the rows after it are written");

  // Rounds after a prefill, then prefill passes past them (one turn's reply, the next turn's prompt): a rewind into
  // the rounds needs the drafter's window there, which the passes overwrote.
  r.init(128, 8);
  r.wrote(0, 64);
  r.wrote(64, 80);  // a round of 16 rows from 64 (its rejected rows, its draft block's)
  r.wrote(68, 84);  // the next from 68
  check(r.holds(70), "rewind into the last rounds: their window held");
  DraftRing one = r;
  one.wrote(72, 80);  // one short pass
  check(one.holds(70), "  past one short prefill pass: held");
  r.wrote(72, 136);
  r.wrote(136, 200);  // 199 overwrote 71
  check(!r.holds(70) && !r.holds(72), "  past consecutive prefill passes that overwrote it: refused");
  check(r.holds(200), "  the position the passes reached: held");

  // A prefill pass of rows_cap rows keeps the window at its end and at a checkpoint inside it (prefill_snapshot).
  r.init(dflash_ring_slots(true, 8, 16, 64, 16, 1 << 20), 8);
  r.wrote(0, 1000);
  r.wrote(1000, 1064);
  check(r.holds(1064) && r.holds(1000), "a pass of the most rows: the window at its end and at its start held");
  // Rounds of the widest rows, every row accepted: the rewind window (FLYX_ROUND_SLOTS - 1 rounds back) still held.
  r.init(dflash_ring_slots(true, 8, 16, 16, 64, 1 << 20), 8);
  for (int start = 0; start < 1024; start += 64) r.wrote(start, start + 64);
  check(r.holds(1024 - (FLYX_ROUND_SLOTS - 1) * 64), "wide rounds: as far back as a rewind reaches, held");

  // A snapshot's load replaces every slot: no checkpoint's window survives; the loaded one's is held.
  r.init(128, 8);
  r.wrote(0, 64);
  r.keep(3, 64);
  r.replacing();
  check(!r.kept(3) && !r.holds(64), "a snapshot loading: no checkpoint kept, nothing held");
  r.loaded(300);
  check(r.holds(300) && !r.kept(3), "  loaded at 300: its window held");
  r.reset();
  check(r.holds(0), "  rewound to 0: the empty window held");

  // A sequence that went back before a checkpoint's window and wrote again: refused (another history may be there).
  r.init(128, 8);
  r.wrote(0, 64);
  r.keep(0, 64);
  r.reset();
  r.wrote(0, 16);
  check(!r.kept(0), "a checkpoint whose positions were written again from before its window: refused");
}
}  // namespace

int main() {
  slots();
  history();
  return failures ? 1 : 0;
}
