// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Two callers opening sessions on the same engine. The first waits inside the backend allocator while the second
// asks for the last slot. The CPU backend uses the tiny model, and its adapter reports a host budget of one session.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

#include "core/session/session.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {
const flyx_backend_api* cpu = nullptr;
std::promise<void> entered, resume;
std::atomic<int> allocations{0};
bool fail_next = false;

int allocate(flyx_bmodel* model, const flyx_seq_layout* layout, flyx_seq** seq) {
  if (++allocations == 1) {
    entered.set_value();
    resume.get_future().wait();
  }
  auto requested = *layout;
  if (fail_next) requested.method = -1;  // the CPU backend returns an allocation error
  return cpu->seq_new_in(model, &requested, seq);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  EngineOptions options;
  options.backend = "cpu_ref";
  options.max_ctx = 64;
  options.speculative = false;
  options.kv_gpu = 0;
  options.kv_ram = options.max_ctx;
  options.kv_method = "exact";
  std::unique_ptr<os::DynLib> cpu_lib;
  flyx_device_info device{};
  std::string err;
  if (!open_backend(options, &cpu_lib, &cpu, &device, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  auto delegated = *cpu;
  delegated.seq_size = [](flyx_bmodel*, const flyx_seq_layout* layout, flyx_seq_size* size) {
    const int dev = std::min(layout->device_positions, layout->max_ctx);
    *size = {8, uint64_t(layout->max_ctx - dev), dev, 0, 0};
    return 0;
  };
  delegated.seq_new_in = allocate;
  flyx_test_delegate(&delegated);
  options.backend = "test-host-kv";
  Engine engine;
  if (!engine.load(argv[1], options, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  int failures = 0;
  auto check = [&](bool ok, const char* message) {
    std::printf("%s %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
  };
  check(engine.sessions_that_fit() == 1, "one session fits the host even when device memory is unknown");
  const auto idle_device_bytes = engine.ledger().used();
  {
    Session first(&engine), second(&engine);
    std::string first_error, second_error;
    bool first_ok = false;
    std::thread worker([&] { first_ok = first.init(&first_error); });
    const bool pending = entered.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    check(pending, "first session reached the allocator");
    if (pending) {
      check(engine.sessions_that_fit() == 0, "the pending allocation holds the only host slot");
      check(!second.init(&second_error) && second_error.find("--kv-ram") != std::string::npos && allocations == 1,
            "another caller is refused before entering the allocator");
    }
    resume.set_value();
    worker.join();
    check(first_ok && engine.sessions_that_fit() == 0, "the session holds its slot through its lifetime");
  }
  check(engine.sessions_that_fit() == 1 && engine.ledger().used() == idle_device_bytes,
        "closing the session gives both budgets back");
  Session retry(&engine);
  fail_next = true;
  check(!retry.init(&err) && err.find("K/V layout") != std::string::npos &&
            engine.sessions_that_fit() == 1 && engine.ledger().used() == idle_device_bytes,
        "a backend allocation failure gives both budgets back");
  fail_next = false;
  check(retry.init(&err) && engine.sessions_that_fit() == 0, "the failed allocation can be retried");
  return failures != 0;
}
