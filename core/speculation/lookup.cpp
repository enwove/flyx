// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/speculation/lookup.h"

#include <algorithm>
#include <cstring>

namespace flyx {

void LookupDrafter::reset() {
  if (++gen_ == 0) {
    std::fill(slots_.begin(), slots_.end(), Slot{});
    gen_ = 1;
  }
  count_ = 0;
  indexed_ = 0;
  prev_local_.clear();
  src_ = left_ = -1;
  at_ = 0;
}

uint64_t LookupDrafter::key(const int32_t* tokens, int n) {
  uint64_t h = 0xcbf29ce484222325ull ^ uint64_t(n);  // FNV-1a over the tokens, the length mixed in
  for (int i = 0; i < n; ++i) {
    h ^= uint32_t(tokens[i]);
    h *= 0x100000001b3ull;
  }
  return h;
}

void LookupDrafter::reserve(size_t additional) {
  const size_t need = count_ + additional;
  if (!slots_.empty() && need * 10 <= slots_.size() * 7) return;
  size_t cap = std::max<size_t>(64, slots_.size() * 2);
  while (need * 10 > cap * 7) cap *= 2;
  std::vector<Slot> grown(cap);
  const size_t mask = cap - 1;
  if (count_ > 0) {
    for (const Slot& s : slots_) {
      if (s.gen != gen_) continue;
      size_t i = bucket(s.key, mask);
      while (grown[i].gen == 1) i = (i + 1) & mask;
      grown[i] = {s.key, s.pos, 1};
    }
  }
  slots_ = std::move(grown);
  gen_ = 1;
}

void LookupDrafter::put(uint64_t k, int32_t pos) {
  const size_t mask = slots_.size() - 1;
  size_t i = bucket(k, mask);
  for (;;) {
    Slot& s = slots_[i];
    if (s.gen != gen_) {
      s = {k, pos, gen_};
      ++count_;
      return;
    }
    if (s.key == k) {
      s.pos = pos;
      return;
    }
    i = (i + 1) & mask;
  }
}

int32_t LookupDrafter::find(uint64_t k) const {
  if (count_ == 0 || slots_.empty()) return -1;
  const size_t mask = slots_.size() - 1;
  size_t i = bucket(k, mask);
  for (;;) {
    const Slot& s = slots_[i];
    if (s.gen != gen_) return -1;
    if (s.key == k) return s.pos;
    i = (i + 1) & mask;
  }
}

void LookupDrafter::cut(size_t n) {
  if (n < indexed_) {
    reset();
  } else {
    src_ = left_ = -1;
    at_ = 0;
  }
}

void LookupDrafter::sync(const std::vector<int32_t>& seq) {
  // The n-grams ending at the last token stay out: propose() looks for an earlier occurrence of them.
  const size_t end = seq.empty() ? 0 : seq.size() - 1;
  if (end < indexed_) reset();  // cut back without cut(): start over
  if (end <= indexed_) return;
  const size_t added = (end - indexed_) * size_t(kMaxN - kMinN + 2);
  reserve(added);
  prev_local_.resize(end + 1, -1);
  for (size_t e = indexed_ + 1; e <= end; ++e) {
    for (int n = kMinN; n <= kMaxN && size_t(n) <= e; ++n) put(key(seq.data() + e - n, n), int32_t(e));
    if (size_t(kLocalN) <= e) {
      const uint64_t k = key(seq.data() + e - kLocalN, kLocalN);
      prev_local_[e] = find(k);
      put(k, int32_t(e));
    }
  }
  indexed_ = end;
}

void LookupDrafter::draft_from(const std::vector<int32_t>& seq, int src, int max, std::vector<int32_t>* out) {
  const int size = static_cast<int>(seq.size());
  // a copy that reaches the end of the sequence goes on from its own start (as in LZ77), so a repeat of period
  // size - src is drafted whole
  const int period = size - src;
  for (int j = 0; j < max; ++j) out->push_back(j < period ? seq[size_t(src + j)] : (*out)[size_t(j - period)]);
  src_ = src;
  at_ = seq.size();
}

int LookupDrafter::propose(const std::vector<int32_t>& seq, int max, std::vector<int32_t>* out) {
  out->clear();
  const int size = static_cast<int>(seq.size());
  // The cursor: the tokens written since it was set still copy the source, and the source goes on before the end.
  if (src_ >= 0 && at_ <= seq.size()) {
    const size_t adv = seq.size() - at_;
    size_t i = 0;
    while (i < adv && size_t(src_) + i < at_ && seq[size_t(src_) + i] == seq[at_ + i]) ++i;
    if (i == adv && size_t(src_) + adv < seq.size()) {
      draft_from(seq, int(src_ + int64_t(adv)), max, out);
      return kMaxN;
    }
    left_ = src_ + int64_t(i);  // the reply left the source there
    src_ = -1;
  }
  // Back to the source near where the reply left it: the nearest place at or after it whose kLocalN tokens are the
  // last ones (an edit's replacement written, the source agrees again).
  if (left_ >= 0 && size > kLocalN && size_t(size - 1) <= prev_local_.size()) {
    const int32_t* tail = seq.data() + size - kLocalN;
    int best = -1;
    for (int e = find(key(tail, kLocalN)), steps = 0; e >= 0 && steps < 512; e = prev_local_[size_t(e)], ++steps) {
      if (e < left_) break;  // the chain runs back in position: none nearer after it
      if (e <= int(left_) + kLocalSpan && e < size &&
          std::memcmp(seq.data() + e - kLocalN, tail, sizeof(int32_t) * size_t(kLocalN)) == 0)
        best = e;
    }
    if (best >= 0) {
      draft_from(seq, best, max, out);
      return kLocalN;
    }
  }
  for (int n = std::min(kMaxN, size - 1); n >= kMinN; --n) {
    const int32_t* tail = seq.data() + size - n;
    const int e = find(key(tail, n));
    if (e < 0) continue;
    if (e >= size || e < n || std::memcmp(seq.data() + e - n, tail, sizeof(int32_t) * size_t(n)) != 0) continue;
    draft_from(seq, e, max, out);  // what followed it there
    return n;
  }
  return 0;
}

}  // namespace flyx
