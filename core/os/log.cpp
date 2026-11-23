// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/os/log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace flyx::log {

namespace {

std::atomic<Sink> g_sink{nullptr};
std::mutex g_stderr_mu;

void write_default(std::string_view message) {
  std::lock_guard<std::mutex> lock(g_stderr_mu);
  if (!message.empty()) std::fwrite(message.data(), 1, message.size(), stderr);
  std::fputc('\n', stderr);
}

void vwrite(Level level, const char* fmt, std::va_list ap) {
  std::va_list ap2;
  va_copy(ap2, ap);
  char stack_buf[512];
  const int n = std::vsnprintf(stack_buf, sizeof(stack_buf), fmt, ap);
  if (n < 0) {
    va_end(ap2);
    return;
  }
  if (size_t(n) < sizeof(stack_buf)) {
    va_end(ap2);
    std::string_view msg(stack_buf, size_t(n));
    while (!msg.empty() && msg.back() == '\n') msg.remove_suffix(1);
    write(level, msg);
    return;
  }
  std::vector<char> heap_buf(size_t(n) + 1);
  std::vsnprintf(heap_buf.data(), heap_buf.size(), fmt, ap2);
  va_end(ap2);
  std::string_view msg(heap_buf.data(), size_t(n));
  while (!msg.empty() && msg.back() == '\n') msg.remove_suffix(1);
  write(level, msg);
}

}  // namespace

void set_sink(Sink sink) { g_sink.store(sink, std::memory_order_release); }

void write(Level level, std::string_view message) {
  if (const Sink s = g_sink.load(std::memory_order_acquire)) s(level, message);
  else write_default(message);
}

void info(const char* fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  vwrite(Level::kInfo, fmt, ap);
  va_end(ap);
}

void warn(const char* fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  vwrite(Level::kWarn, fmt, ap);
  va_end(ap);
}

void error(const char* fmt, ...) {
  std::va_list ap;
  va_start(ap, fmt);
  vwrite(Level::kError, fmt, ap);
  va_end(ap);
}

}  // namespace flyx::log
