// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Two callers opening sessions on the same engine. The first waits inside the backend allocator while the second
// asks for the last slot. The CPU backend uses the tiny model, and its adapter reports a host budget of one session.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

#include "core/exec/scheduler.h"
#include "core/session/session.h"
#include "tests/common/check.h"

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

// Exercise the server's session option through Scheduler::load and Engine::load. The available 4 MiB
// holds one whole session, or two sessions with 2048 device positions each. AUTO switches at 3072.
void check_scheduler_placement(const char* path) {
  auto profile = *cpu;
  profile.memory = [](flyx_bmodel*, flyx_memory_info* out) {
    *out = {0, 0, 4u << 20, 4u << 20, 0};
    return 0;
  };
  profile.seq_size = [](flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) {
    const int dev = std::min(l->max_ctx, l->device_positions);
    *out = {uint64_t(dev) * 1024, uint64_t(l->max_ctx - dev) * 1024, dev, 3072,
            l->method == FLYX_KV_READ_ALL ? 0 : 8};
    return 0;
  };
  flyx_test_delegate(&profile);
  EngineOptions eo;
  eo.backend = "test-host-kv";
  eo.max_ctx = 4096;
  eo.ctx_given = true;
  eo.speculative = false;
  eo.vram_reserve = 0;
  SchedulerOptions so;
  so.host_cache_bytes = 0;
  auto run = [&](const char* method, int sessions, int ram, int expected, int rows) {
    eo.kv_method = method;
    eo.kv_ram = ram;
    so.max_sessions = sessions;
    Scheduler sched(path, eo, so);
    std::string err;
    const std::string options = std::string(method) + ", sessions=" + std::to_string(sessions) +
                                ", host bytes=" + std::to_string(ram);
    if (!sched.ok(&err)) {
      check(false, options + ": " + err);
      return;
    }
    const auto& b = sched.engine()->budget();
    check(sched.max_sessions() == expected && b.declined_pair_host_rows == rows &&
              std::min(b.layout.device_positions, eo.max_ctx) == eo.max_ctx / expected,
          options + " selects the expected concurrency and exact prefix");
  };
  constexpr int ram = 16 << 20;
  run("auto", 0, ram, 1, 1024);
  run("auto", 1, ram, 1, 0);
  run("auto", 2, ram, 2, 0);
  run("auto", 0, 0, 1, 0);
  run("exact", 0, ram, 2, 0);
  run("strata", 0, ram, 2, 0);
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
  check_scheduler_placement(argv[1]);
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
  return test::finish("session_budget_test");
}
