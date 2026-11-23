// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Centralized CLI and HTTP server defaults shared across apps/.
#pragma once

#include <cstddef>
#include <cstdint>

namespace flyx::apps {

inline constexpr const char* kDefaultHost = "127.0.0.1";
inline constexpr int kDefaultPort = 8080;
inline constexpr const char* kDefaultServerUrl = "http://127.0.0.1:8080";
// One worker holds a connection for as long as its request lives, queued or running: more of them than the
// sessions and the queue together (8 and 16 by default), so that /health, the dashboard and a refusal of what the
// queue does not hold still find one.
inline constexpr int kDefaultHttpThreads = 64;
inline constexpr int kServerDefaultCtx = 32768;
inline constexpr uint64_t kDefaultDiskCacheBytes = 64'000'000'000ull;
inline constexpr uint64_t kHttpMaxPayloadBytes = 64ull << 20;
inline constexpr int kHttpTimeoutSec = 600;
// What a client may take between two pieces of its request (line, headers, body): a connection that sends nothing
// gives its worker back after this, not after the ten minutes a reply may wait in the queue.
inline constexpr int kHttpReadTimeoutSec = 30;
inline constexpr int kStartWaitMs = 10000;
inline constexpr int kMinTerminalCols = 60;
inline constexpr int kMaxCtx = 1 << 24;
inline constexpr int kMaxDraftWidth = 64;
inline constexpr double kMaxKvRamGb = 1e6;
// What the dashboard's Hub list shows before a search: repositories named for the family whose architectures this
// build runs (models/: qwen3, qwen35).  Most of the Hub's other GGUFs are of architectures it does not open.
inline constexpr const char* kHubFamily = "Qwen3";

}  // namespace flyx::apps
