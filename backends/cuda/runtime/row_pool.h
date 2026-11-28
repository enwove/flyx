// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace flyx::cuda {
// One caller submits contiguous, disjoint row ranges and computes the first range itself.
class RowPool {
 public:
  static size_t default_threads() {
    static const size_t n = std::max<size_t>(1, std::min<size_t>(16, std::thread::hardware_concurrency()));
    return n;
  }

  explicit RowPool(size_t max_threads, bool spin = false)
      : max_threads_(std::max<size_t>(1, max_threads)), spin_(spin) {
    workers_.reserve(max_threads_ - 1);
    try {
      for (size_t idx = 1; idx < max_threads_; ++idx) workers_.emplace_back([this, idx] { worker_loop(idx); });
    } catch (...) {
      shutdown();
      throw;
    }
  }

  ~RowPool() { shutdown(); }
  size_t threads() const { return max_threads_; }
  RowPool(const RowPool&) = delete;
  RowPool& operator=(const RowPool&) = delete;

  void run(size_t n, const std::function<void(size_t, size_t)>& fn) {
    const size_t parts = std::min(max_threads_, n);
    if (parts <= 1) {
      if (n) fn(0, n);
      return;
    }
    n_ = n;
    parts_ = parts;
    fn_ = &fn;
    error_ = nullptr;
    pending_.store(workers_.size(), std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    std::exception_ptr caller_error;
    try {
      fn(0, n / parts);
    } catch (...) {
      caller_error = std::current_exception();
    }
    for (auto pending = pending_.load(std::memory_order_acquire); pending; pending = pending_.load(std::memory_order_acquire))
      wait_while(pending_, pending);
    fn_ = nullptr;
    if (caller_error) std::rethrow_exception(caller_error);
    if (error_) std::rethrow_exception(error_);
  }

 private:
  template <class T>
  void wait_while(const std::atomic<T>& value, T seen) const {
    if (spin_) {
      for (int i = 0; i < 512; ++i) {
        if (value.load(std::memory_order_relaxed) != seen) return;
#if defined(_M_X64) || defined(__x86_64__)
        _mm_pause();
#else
        std::this_thread::yield();
#endif
      }
    }
    value.wait(seen, std::memory_order_relaxed);
  }

  void worker_loop(size_t idx) {
    uint64_t seen = 0;
    for (;;) {
      wait_while(generation_, seen);
      seen = generation_.load(std::memory_order_acquire);
      if (stop_) return;
      try {
        if (idx < parts_) (*fn_)(n_ * idx / parts_, n_ * (idx + 1) / parts_);
      } catch (...) {
        std::lock_guard lock(error_mu_);
        if (!error_) error_ = std::current_exception();
      }
      // Even an empty shard acknowledges the generation before the caller can replace its bounds and function.
      if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) pending_.notify_one();
    }
  }

  void shutdown() noexcept {
    stop_ = true;
    generation_.fetch_add(1, std::memory_order_release);
    generation_.notify_all();
    for (auto& th : workers_) if (th.joinable()) th.join();
  }

  size_t max_threads_;
  bool spin_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<size_t> pending_{0};
  std::atomic<bool> stop_{false};
  size_t n_ = 0, parts_ = 0;
  const std::function<void(size_t, size_t)>* fn_ = nullptr;
  std::mutex error_mu_;
  std::exception_ptr error_;
  std::vector<std::thread> workers_;
};

}  // namespace flyx::cuda
