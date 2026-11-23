// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A short sleep for the waits on the device (runner.cpp).  On Windows std::this_thread::sleep_for ends on the
// scheduler's tick, 1 to 16 ms after it is asked to (asked for 20 us: 15.1 ms, the median of 200 on an RTX 5060 Ti
// machine), longer than a round of most models; a high-resolution
// waitable timer (Windows 10 1803 and later) ends about 0.5 ms after (531 us).  Its own file: windows.h's macros stay
// out of the backend's other sources.
#include <chrono>
#include <cstdint>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace flyx::cuda {

#if defined(_WIN32)
void sleep_us(int64_t us) {
  struct Timer {
    HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    ~Timer() {
      if (h) CloseHandle(h);
    }
  };
  thread_local Timer timer;  // one per waiting thread: sessions on two threads wait at once
  LARGE_INTEGER due;
  due.QuadPart = -us * 10;  // relative, in 100 ns
  if (timer.h && SetWaitableTimer(timer.h, &due, 0, nullptr, nullptr, FALSE)) {
    WaitForSingleObject(timer.h, INFINITE);
    return;
  }
  std::this_thread::sleep_for(std::chrono::microseconds(us));  // before Windows 10 1803: the tick
}
#else
void sleep_us(int64_t us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }
#endif

}  // namespace flyx::cuda
