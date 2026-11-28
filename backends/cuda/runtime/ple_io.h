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

// The scheduler retries a rejected batch one sequence at a time. Validate every member
// before submitting the first, including when each round will use its own stream.
template <class Validate, class Submit>
int launch_serial_batch(int n, Validate validate, Submit submit) {
  for (int i = 0; i < n; ++i)
    if (const int rc = validate(i)) return rc;
  for (int i = 0; i < n; ++i)
    if (const int rc = submit(i)) return rc;
  return 0;
}

}  // namespace flyx::cuda
