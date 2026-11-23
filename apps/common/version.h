// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The version this build is of: CMake's project version, given to the programs as FLYX_VERSION.  `flyx --version`
// prints it and the server's telemetry carries it, for a bug report to name.
#pragma once

#ifndef FLYX_VERSION
#define FLYX_VERSION "unknown"
#endif

namespace flyx::apps {

inline constexpr const char* kVersion = FLYX_VERSION;

}  // namespace flyx::apps
