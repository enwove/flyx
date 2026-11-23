// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Centralized engine logging sink. Defaults to writing lines to stderr so CLI, server, and stderr capture
// keep identical output, while embedders may redirect logs via set_sink().
#pragma once

#include <string_view>

namespace flyx::log {

enum class Level { kInfo, kWarn, kError };

using Sink = void (*)(Level level, std::string_view message);

// Replaces the log sink (nullptr restores the default stderr sink). Thread-safe.
void set_sink(Sink sink);

// Emits a single log line (without trailing newline) through the active sink.
void write(Level level, std::string_view message);

#if defined(__GNUC__) || defined(__clang__)
#define FLYX_PRINTF_FMT(fmt_idx, arg_idx) __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#define FLYX_PRINTF_FMT(fmt_idx, arg_idx)
#endif

void info(const char* fmt, ...) FLYX_PRINTF_FMT(1, 2);
void warn(const char* fmt, ...) FLYX_PRINTF_FMT(1, 2);
void error(const char* fmt, ...) FLYX_PRINTF_FMT(1, 2);

#undef FLYX_PRINTF_FMT

}  // namespace flyx::log
