// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Resident sessions for clients that do not name one (OpenAI / Anthropic
// HTTP APIs): a request goes to the session whose history it continues
// furthest; a request that continues none gets a new session.  When all
// slots are taken it takes the place of a sibling (a conversation it shares
// a real prefix with: the same system prompt and tools), else of the least
// recently used. Among equally close siblings, it reuses the oldest idle slot.
//
// Several requests can run at once, each on its own session.  A request
// that continues a session, or would take a sibling's place, waits while
// that session is running another request: it is not given a session that
// fits it less, and no other resident conversation is pushed out for it to
// run sooner.  A request with no such tie takes an idle session.
//
// With a SnapshotStore, a session that takes another
// conversation keeps its own one's checkpoints there first, and every
// session loads from it a snapshot that holds more of a prompt than it does.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/chat.h"
#include "core/session/session.h"

namespace flyx {

class PrefixCache {
 public:
  PrefixCache(Engine* e, int max_sessions, SnapshotStore* store = nullptr)
      : e_(e), max_(max_sessions < 1 ? 1 : max_sessions), store_(store) {}

  struct Choice {
    std::string reason;
    int id = 0;
    uint64_t assignment = 0;  // identifies this use of the slot until it goes to another request
    bool busy = false;
    bool park = false;  // save the displaced conversation once the incoming prompt is known
  };
  // The session whose history the rendered prompt continues. The system and the first user
  // boundaries distinguish a continuation from a conversation sharing only instructions. Null with busy set when that
  // session is running another request (nothing changes: ask again when a
  // request has ended); null with *err when no session can be made.
  Session* acquire(const interfaces::RenderedChat& prompt, Choice* choice, std::string* err,
                   uint64_t resume_assignment = 0);
  int size() const { return static_cast<int>(slots_.size()); }
  // The resident sessions (id, tokens in the state), most recently used first.
  struct SlotInfo {
    int id = 0, tokens = 0;
  };
  std::vector<SlotInfo> slots() const;
  // Frees every session (idle unload), its checkpoints kept in the store first.
  void clear();

 private:
  struct Slot {
    std::unique_ptr<Session> s;
    uint64_t used = 0;
    int id = 0;
  };
  Engine* e_;
  int max_;
  SnapshotStore* store_ = nullptr;
  std::vector<Slot> slots_;
  uint64_t clock_ = 0;
  int next_id_ = 1;
};

}  // namespace flyx
