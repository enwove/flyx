// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Caches kept as a ring of slots that every checkpoint copies to host memory (backends/cuda/runtime/window_ring.h):
// their slots at the cache's edges and for the windows options may give, and, on a model of the device's ring
// (KvRef::slot_of: a sink kept for good, the other positions in turn), what they hold through prefill passes, rounds,
// checkpoints, restores and rewinds as runtime/runner.cpp drives them.  The MTP layer's entries (a position behind the
// sequence's) as a sliding-window layer's rows.
#include "backends/cuda/runtime/window_ring.h"
#include "tests/common/check.h"

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

namespace {

using flyx::cuda::window_ring_floor;
using flyx::cuda::window_ring_slots;

// What a ring of `ring` slots holds, slot by slot, as the device writes it: positions [0, sink) in their own slots,
// the others in turn in the rest.
struct Ring {
  int ring = 0, sink = 0;
  std::vector<int> held;  // the position in each slot (-1: none yet)
  Ring(int r, int s) : ring(r), sink(s), held(size_t(r), -1) {}
  int slot(int p) const { return p < sink ? p : sink + (p - sink) % (ring - sink); }
  void write(int from, int to) {
    for (int p = std::max(0, from); p < to; ++p) held[size_t(slot(p))] = p;
  }
  bool has(int from, int to) const {
    for (int p = std::max(0, from); p < to; ++p)
      if (held[size_t(slot(p))] != p) return false;
    return true;
  }
  // A checkpoint's copy (copy_checkpoint_state): the slots before min(pos, ring), as they are; a restore puts them
  // back.
  std::vector<int> copy(int pos) const { return {held.begin(), held.begin() + std::min(pos, ring)}; }
  void restore(const std::vector<int>& c) { std::copy(c.begin(), c.end(), held.begin()); }
};

// A layer the ring holds, from the sequence's side.  A sliding-window layer's row at p reads [p - window + 1, p], a
// pass from p writing p on.  The MTP layer keeps (hidden at j, token j + 1) as entry j: a pass from p writes entries
// p - 1 on, and its first draft step at p reads its sink and [p - window, p - 1], writing entry p - 1 itself
// (regions/layers.cpp: mtp_context, mtp_draft_attention through kv_window).  `lag`: 1 for the MTP layer's.
struct Layer {
  int window = 0, sink = 0, lag = 0;
  // A pass of n rows from p, or a round of n rows from p (its rejected rows and the MTP layer's draft steps inside).
  void wrote(Ring& r, int p, int n) const { r.write(p - lag, p + n - lag); }
  // The row at p (the MTP layer's: its first draft step) finds what it reads from before it in the ring: its own entry
  // it writes.
  bool reads(const Ring& r, int p) const {
    const int q = p - lag, first = std::max(0, q - window + 1);
    const int head = first > sink ? sink : 0, from = first > sink ? first : 0;
    return r.has(0, head) && r.has(from, q);
  }
};

// gguf_test's sizes and the defaults: mtp_window 4096 beside prefill passes of 4096 rows (16 GB free) or 1024.
void slots() {
  check(window_ring_slots(4096, 0, 4096, 16, 1 << 18) == 8192, "MTP's window of 4096, passes of 4096 rows: 8192 slots");
  check(window_ring_slots(4096, 64, 4096, 16, 1 << 18) == 8256, "  with the 64 positions it keeps for good: 8256");
  check(window_ring_slots(4096, 0, 1024, 16, 1 << 18) == 5120, "  passes of 1024 rows: 5120");
  check(window_ring_slots(4096, 0, 16, 64, 1 << 18) == 4096 + 128,
        "  passes shorter than the rounds' rows: the rounds'");
  // The sliding-window layers' (no sink) as they were sized before the MTP layer's joined them.
  bool as_before = true;
  for (const int window : {1, 8, 1025, 4096})
    for (const int rows_cap : {16, 64, 256, 4096})
      for (const int wide : {16, 64}) {
        const long long span = std::max(FLYX_ROUND_SLOTS * FLYX_MAX_ROWS + wide, rows_cap);
        const int ring = int((window + span + 63) / 64 * 64);
        as_before = as_before && window_ring_slots(window, 0, rows_cap, wide, 1 << 20) == ring;
      }
  check(as_before, "  without a sink: a sliding-window layer's slots as before");
  check(window_ring_slots(96, 16, 128, 16, 833) == 256 && window_ring_slots(96, 16, 128, 16, 256) == 0 &&
            window_ring_slots(96, 16, 128, 16, 255) == 0,
        "a cache one position longer than the ring: the ring; as long or shorter: a slot a position");
  check(window_ring_slots(0, 16, 128, 16, 1 << 20) == 0, "no window: a slot a position");
  check(window_ring_slots(4096, (1 << 18) - 4096, 4096, 16, 1 << 18) == 0,
        "a sink as long as the context leaves (mtp_sink clamped): a slot a position");
  check(window_ring_slots(INT_MAX, 0, 4096, 64, INT_MAX) == 0 &&
            window_ring_slots(INT_MAX - 64, 64, 4096, 64, INT_MAX) == 0,
        "a window of INT_MAX: a slot a position (no int overflow past the cap)");
}

// A model of the MTP layer's ring at the defaults: a prompt in passes of 4096 rows, checkpoints in and after them, a
// reply in rounds, the next turn's prompt; each checkpoint restored after the positions past it went round the ring.
void mtp_checkpoints() {
  const int window = 4096, rows = 4096, wide = 16;
  for (const int sink : {0, 64}) {
    const std::string at = sink ? " (a sink of 64)" : " (no sink)";
    const int n = window_ring_slots(window, sink, rows, wide, 1 << 18);
    const Layer mtp{window, sink, 1};
    Ring r(n, sink);
    // The ring the MTP layer had: its sink and window and two rounds' rows, or a pass's rows (its window inside them).
    const int had = int((sink + std::max<long long>(window + 2 * FLYX_MAX_ROWS, rows) + 63) / 64 * 64);
    Ring old(had, sink);
    for (int p = 0; p < 2 * rows; p += rows) {
      mtp.wrote(r, p, rows);
      mtp.wrote(old, p, rows);
    }
    const std::vector<int> after_pass = r.copy(2 * rows);  // a checkpoint where the passes end
    // A pass of 4096 rows from 8192 that keeps the state at its first boundary past it (prefill_snapshot: a message
    // boundary inside the pass): the ring's at its end.
    const int mid = 2 * rows + 64;
    mtp.wrote(r, 2 * rows, rows);
    mtp.wrote(old, 2 * rows, rows);
    check(mtp.reads(r, mid) && !mtp.reads(old, mid),
          "a checkpoint inside a pass of 4096 rows: its window held" + at + "; the ring the MTP layer had lost it");
    const std::vector<int> inside = r.copy(mid);
    // The reply's rounds (16 rows from wherever the last accepted), then the next turn's prompt: past the ring.
    int pos = 3 * rows;
    for (int i = 0; i < 300; ++i) {
      mtp.wrote(r, pos, 16);
      pos += 1 + i % 16;
    }
    mtp.wrote(r, pos, rows);
    pos += rows;
    check(mtp.reads(r, pos) && !mtp.reads(r, 2 * rows) && !mtp.reads(r, mid),
          "  the positions past them gone round the ring: the window at the sequence's position held, theirs not");
    // Restored (copy_checkpoint_state back): each one's window, the newer first, then the older, then again.
    r.restore(inside);
    check(mtp.reads(r, mid), "  the one inside the pass restored: its window from its copy");
    mtp.wrote(r, mid, 600);  // rows after it, then the older one
    r.restore(after_pass);
    const bool older = mtp.reads(r, 2 * rows);
    r.restore(inside);
    check(older && mtp.reads(r, mid), "  the older restored after it, then it again: each one's window");
    check(r.has(0, sink), "  its first positions kept through every wrap");
  }
}

// Rewinds (rewind_floor): ring_hi one past the highest position written since a reset or restore; a rewind to a
// position at or past the floor finds the window there in the ring, and one before it does not.  Sliding-window and
// MTP layers, with and without a sink, after rounds alone and past prefill passes.
void rewinds() {
  struct Case {
    Layer layer;
    int rows_cap;
    const char* what;
  };
  const Case cases[] = {{{96, 0, 1}, 128, "MTP, window 96"},
                        {{96, 16, 1}, 128, "MTP, window 96 and a sink of 16"},
                        {{4096, 0, 1}, 1024, "MTP, window 4096, passes of 1024 rows"},
                        {{1025, 0, 0}, 4096, "a sliding-window layer of 1025"},
                        {{8, 0, 0}, 64, "a sliding-window layer of 8"}};
  for (const Case& c : cases) {
    const Layer& L = c.layer;
    const int wide = 16, n = window_ring_slots(L.window, L.sink, c.rows_cap, wide, 1 << 20);
    Ring r(n, L.sink);
    int pos = 0, hi = 0;
    auto pass = [&](int rows) {
      L.wrote(r, pos, rows);
      pos += rows;
      hi = std::max(hi, pos);
    };
    auto floor = [&] { return int(window_ring_floor(hi, n, L.sink, L.window)); };
    for (int i = 0; i < 4; ++i) pass(c.rows_cap);
    // Rounds of 16 rows, each accepting 1 to 16: as far back as a rewind reaches (FLYX_ROUND_SLOTS - 1 rounds) held.
    std::vector<int> starts;
    for (int i = 0; i < 40; ++i) {
      L.wrote(r, pos, 16);
      hi = std::max(hi, pos + 16);
      starts.push_back(pos);
      pos += 1 + (i * 7) % 16;
    }
    const int back = starts[starts.size() - (FLYX_ROUND_SLOTS - 1)];
    bool held = true;
    for (int p = back; p <= pos; ++p) held = held && p >= floor() && L.reads(r, p);
    check(held, std::string(c.what) + ": rounds alone, every rewind target at or past the floor, its window held");
    // A prefill pass after them (the next turn's prompt) of a few rows, then of a whole pass: what the floor says.
    for (const int rows : {16, c.rows_cap}) {
      Ring kept = r;
      const int kept_pos = pos, kept_hi = hi;
      pass(rows);
      bool safe = true;
      for (int p = std::max(0, back - 2 * L.window); p <= pos; ++p)
        if (p >= floor()) safe = safe && L.reads(r, p);
      const int f = floor();
      const bool tight = f <= L.window + L.sink || (L.reads(r, f) && !L.reads(r, f - 1));
      check(safe && tight, std::string("  past a pass of ") + std::to_string(rows) +
                               " rows: at or past the floor (" + std::to_string(f) +
                               ") the window held, before it not");
      r = kept;
      pos = kept_pos;
      hi = kept_hi;
    }
    // A restore: the ring as the checkpoint copied it, ring_hi its position.  Rewinds into the rounds after it held.
    const int at = pos;
    const std::vector<int> ckp = r.copy(at);
    pass(c.rows_cap);
    pass(c.rows_cap);
    r.restore(ckp);
    pos = hi = at;
    for (int i = 0; i < 8; ++i) {
      L.wrote(r, pos, 16);
      hi = std::max(hi, pos + 16);
      pos += 16;
    }
    held = true;
    for (int p = at; p <= pos; ++p) held = held && (p < floor() || L.reads(r, p));
    for (int p = pos - (FLYX_ROUND_SLOTS - 1) * 16; p <= pos; ++p) held = held && p >= floor();
    check(held, "  restored, then rounds: since the checkpoint, at or past the floor its window held, and as far "
                "back as a rewind reaches at or past it");
  }
}
}  // namespace

int main() {
  flyx::test::print_passes = true;
  slots();
  mtp_checkpoints();
  rewinds();
  return flyx::test::finish("window_ring_test");
}
