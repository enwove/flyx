// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Drafts looked up in the sequence itself (prompt lookup decoding,
// SuffixDecoding): when the last n tokens (n = kMaxN down to
// kMinN) occurred before, the tokens that followed their latest earlier
// occurrence are the draft. Agents copy text from their context (file
// contents into edits, paths, tool output), and those runs are drafted whole.
// Verification decides every token, so a wrong draft costs only its row.
//
// A cursor follows the copy (as YASAKANI's copy index does): while the reply
// keeps to the source it drafts on from there with no lookup; where the reply
// leaves it (an edit), kLocalN tokens that agree again within kLocalSpan of the
// place it left put the cursor back (the nearest such place: look-alike lines
// are not mixed up), before the n-grams of the whole sequence are looked up.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flyx {

class LookupDrafter {
 public:
  // Matched n-grams of 6 to 8 tokens: a shorter match drafts more wrong rows
  // than right ones (bench/replay/lookup_coverage.py).
  static constexpr int kMinN = 6, kMaxN = 8;
  // Re-anchoring near the place the reply left its source: a short match there is rarely a chance one.
  static constexpr int kLocalN = 4, kLocalSpan = 256;

  void reset();
  // The sequence was cut back to n tokens: entries past it would hide earlier
  // occurrences of their n-grams, so the index starts over (on the next sync).
  void cut(size_t n);
  // Index the sequence up to (not including) its last token, after it grew.
  void sync(const std::vector<int32_t>& seq);
  // The draft after the sequence's end, `max` tokens (a repeat goes on as one); returns the
  // length of the matched n-gram (0: no earlier occurrence, *out empty).
  int propose(const std::vector<int32_t>& seq, int max, std::vector<int32_t>* out);

 private:
  // Flat open-addressed table (16 B/slot): avoids hundreds of thousands of node allocations on long contexts,
  // and the generation tag makes reset()/cut() O(1) without freeing or zeroing the backing array.
  struct Slot {
    uint64_t key = 0;
    int32_t pos = 0;
    uint32_t gen = 0;
  };
  static uint64_t key(const int32_t* tokens, int n);
  static size_t bucket(uint64_t k, size_t mask) { return size_t(k ^ (k >> 32)) & mask; }
  void reserve(size_t additional);
  void put(uint64_t k, int32_t pos);
  int32_t find(uint64_t k) const;

  std::vector<Slot> slots_;  // power-of-two capacity when non-empty
  size_t count_ = 0;         // live entries in the current generation
  uint32_t gen_ = 1;         // slots with slot.gen == gen_ are occupied
  size_t indexed_ = 0;       // n-grams ending at positions < indexed_ are in slots_
  // kLocalN-grams: the latest end position (in slots_, key of length kLocalN) and, per end position, the one before
  // with the same tokens (-1: none), so every occurrence is reachable from the latest.
  std::vector<int32_t> prev_local_;
  // The cursor: the reply's position at_ (the sequence's length when it was set) copies the source from src_ on;
  // left_: the source position where the reply last left it (-1: none).
  int64_t src_ = -1, left_ = -1;
  size_t at_ = 0;
  // Draft seq[src..] (a repeat reaching the end goes on from its own start), the cursor at src.
  void draft_from(const std::vector<int32_t>& seq, int src, int max, std::vector<int32_t>* out);
};

}  // namespace flyx
