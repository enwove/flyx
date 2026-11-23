// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared binary header, layout hash helper, and host-piece cursor for checkpoint snapshots,
// used by both the reference and CUDA backends.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

#include "flyx/backend_abi.h"

namespace flyx::backends {

struct SnapshotHeader {
  uint32_t magic, version;
  uint64_t layout;
  int32_t pos, begin;
  uint64_t bytes;
};

constexpr uint32_t kSnapshotMagic = 0x504E5346u;  // "FSNP"
constexpr uint32_t kSnapshotVersion = 1;
constexpr uint32_t kDeltaVersion = 2;

constexpr uint64_t kFnv1aOffset = 1469598103934665603ull;
constexpr uint64_t kFnv1aPrime = 1099511628211ull;

inline void fnv1a64_mix(uint64_t* h, uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    *h ^= (v >> (8 * i)) & 0xFFu;
    *h *= kFnv1aPrime;
  }
}

// A position in host pieces (flyx_host_pieces): bytes visited piece by piece, skipping null pieces (ABI 31).
struct HostCursor {
  const flyx_host_pieces* p;
  uint64_t at = 0;

  template <class F>  // f(piece_span, offset_within_n, length)
  void take(uint64_t n, F&& f) {
    for (uint64_t done = 0; done < n;) {
      const uint64_t off = at % p->piece_bytes, len = std::min(n - done, p->piece_bytes - off);
      if (void* piece = p->pieces[at / p->piece_bytes]) f(static_cast<uint8_t*>(piece) + off, done, len);
      done += len;
      at += len;
    }
  }

  void put(const void* from, uint64_t n) {
    take(n, [&](uint8_t* dst, uint64_t o, uint64_t len) {
      std::memcpy(dst, static_cast<const uint8_t*>(from) + o, len);
    });
  }

  void get(void* to, uint64_t n) {
    take(n, [&](uint8_t* src, uint64_t o, uint64_t len) {
      std::memcpy(static_cast<uint8_t*>(to) + o, src, len);
    });
  }
};

inline bool pieces_hold(const flyx_host_pieces* p, uint64_t bytes) {
  return p && p->pieces && p->n > 0 && p->pieces[0] && p->piece_bytes >= 4096 &&
         uint64_t(p->n) * p->piece_bytes >= bytes;
}

// Reads and validates a snapshot header from host pieces before loading its payload.
// On success, fills *out and returns true; on failure, writes a human-readable error into *err and returns false.
template <class SizeFn>
inline bool validate_snapshot_header(const flyx_host_pieces* src, uint64_t bytes, uint64_t expected_layout,
                                     int max_ctx, int cur_pos, SizeFn&& expected_size, SnapshotHeader* out,
                                     std::string* err) {
  SnapshotHeader h{};
  if (bytes < sizeof h || !pieces_hold(src, bytes)) {
    *err = "not a snapshot";
    return false;
  }
  HostCursor{src}.get(&h, sizeof h);
  if (h.magic != kSnapshotMagic || h.version != (h.begin ? kDeltaVersion : kSnapshotVersion) ||
      h.layout != expected_layout) {
    *err = "a snapshot of another model, layout or version";
    return false;
  }
  if (h.begin < 0 || h.pos < h.begin || h.pos > max_ctx || h.bytes != expected_size(h.begin, h.pos) ||
      bytes < h.bytes) {
    *err = "a snapshot that does not fit this sequence";
    return false;
  }
  if (h.begin && cur_pos != h.begin && cur_pos != h.pos) {
    *err = "snapshot delta loaded without its parent";
    return false;
  }
  *out = h;
  return true;
}

}  // namespace flyx::backends
