// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>

namespace flyx::cuda {

// A failed graph can be followed by more graphs using the same staging slots.
// Retire the entire stream before returning those slots or resetting the latch.
template <class Sequence, class Sync>
bool drain_failed_ple(Sequence* s, Sync sync) {
  if (!s->ple_failed.load(std::memory_order_acquire)) return false;
  sync(s);
  s->pending.clear();
  s->read_no = s->seq_no;
  return true;
}

}  // namespace flyx::cuda
