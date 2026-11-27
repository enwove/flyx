// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace flyx::cuda {
class RowPool {
 public:
  static size_t default_threads() {
    static const size_t n = std::max<size_t>(1, std::min<size_t>(16, std::thread::hardware_concurrency()));
    return n;
  }

  explicit RowPool(size_t max_threads) : max_threads_(std::max<size_t>(1, max_threads)) {
    workers_.reserve(max_threads_ - 1);
    try {
      for (size_t idx = 1; idx < max_threads_; ++idx) workers_.emplace_back([this, idx] { worker_loop(idx); });
    } catch (...) {
      shutdown();
      throw;
    }
  }

  ~RowPool() { shutdown(); }
  RowPool(const RowPool&) = delete;
  RowPool& operator=(const RowPool&) = delete;

  void run(size_t n, const std::function<void(size_t, size_t)>& fn) {
    const size_t parts = std::min(max_threads_, n);
    if (parts <= 1) {
      if (n) fn(0, n);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      n_ = n;
      parts_ = parts;
      fn_ = &fn;
      error_ = nullptr;
      active_ = parts - 1;
      ++gen_;
    }
    wake_cv_.notify_all();
    std::exception_ptr caller_err;
    try {
      fn(0, n / parts);
    } catch (...) {
      caller_err = std::current_exception();
    }
    std::exception_ptr err;
    {
      std::unique_lock<std::mutex> lock(mu_);
      done_cv_.wait(lock, [&] { return active_ == 0; });
      err = caller_err ? caller_err : error_;
      error_ = nullptr;
      fn_ = nullptr;
    }
    if (err) std::rethrow_exception(err);
  }

 private:
  void worker_loop(size_t idx) {
    uint64_t seen = 0;
    for (;;) {
      size_t a = 0, b = 0;
      const std::function<void(size_t, size_t)>* fn = nullptr;
      {
        std::unique_lock<std::mutex> lock(mu_);
        wake_cv_.wait(lock, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
        if (idx >= parts_) continue;
        a = n_ * idx / parts_;
        b = n_ * (idx + 1) / parts_;
        fn = fn_;
      }
      std::exception_ptr err;
      try {
        (*fn)(a, b);
      } catch (...) {
        err = std::current_exception();
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (err && !error_) error_ = err;
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  void shutdown() noexcept {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_cv_.notify_all();
    for (auto& th : workers_)
      if (th.joinable()) th.join();
  }

  size_t max_threads_ = 1;
  std::mutex mu_;
  std::condition_variable wake_cv_;
  std::condition_variable done_cv_;
  uint64_t gen_ = 0;
  size_t active_ = 0;
  size_t n_ = 0;
  size_t parts_ = 0;
  const std::function<void(size_t, size_t)>* fn_ = nullptr;
  std::exception_ptr error_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
};

}  // namespace flyx::cuda
