// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <deque>
#include <string>
#include <vector>

#include "backends/cuda/loader/ple_budget.h"
#include "backends/cuda/runtime/ple_io.h"
#include "core/exec/generation.h"
#include "core/session/session.h"
#include "tests/common/check.h"

using namespace flyx;
using flyx::test::check;
extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {
const flyx_backend_api* backend;
flyx_tensor_view table;
decltype(table.read_at) read_row;
std::atomic<bool> fail_next{false};
bool observed_reset = false;
std::atomic<bool> read_failed{false};

int open_table(const flyx_model_desc* d, const flyx_open_options* opts, flyx_bmodel** model) {
  auto desc = *d;
  std::vector<flyx_weight> weights(d->weights, d->weights + d->n_weights);
  for (int i = 0; i < d->n_blocks; ++i) {
    if (d->blocks[i].kind != FLYX_BLOCK_PLE) continue;
    auto& weight = weights[d->blocks[i].ple.table];
    table = *weight.t;
    read_row = table.read_at;
    table.read_at = [](const void* file, uint64_t at, void* dst, size_t bytes) {
      if (fail_next.exchange(false)) {
        read_failed = true;
        return -1;
      }
      return read_row(file, at, dst, bytes);
    };
    weight.t = &table;
  }
  desc.weights = weights.data();
  return backend->open(&desc, opts, model);
}

int rewind_checked(flyx_seq* seq, int32_t pos, int32_t anchor) {
  const int result = backend->rewind(seq, pos, anchor);
  if (pos == 0 && result == 1 && read_failed) observed_reset = true;
  return result;
}

void queued_failure() {
  struct Sequence {
    std::atomic<bool> ple_failed{true};
    std::deque<int> pending{41, 42};
    unsigned read_no = 40, seq_no = 42;
  } seq;
  int waits = 0;
  auto finish_callbacks = [&](Sequence* s) {
    ++waits;
    check(s->pending.size() == 2 && s->read_no == 40, "callback storage lives until both rounds finish");
    s->ple_failed = true;  // the second queued callback also failed
  };
  check(cuda::drain_failed_ple(&seq, finish_callbacks), "failed graph is reported");
  check(waits == 1 && seq.pending.empty() && seq.read_no == seq.seq_no && seq.ple_failed,
        "all failed rounds retired, with the error latched until reset");
  seq.ple_failed = false;
  check(!cuda::drain_failed_ple(&seq, finish_callbacks) && waits == 1, "healthy round has no stream wait");
}

void budget_boundaries() {
  constexpr uint64_t gib = 1ull << 30, mib = 1ull << 20;
  for (uint64_t total : {8 * gib, 80 * gib}) {
    const uint64_t weights = total / 2;
    const uint64_t required = weights + std::max(384 * mib, total / 40) + cuda::min_runtime(weights) + 8192 * 131072ull;
    check(cuda::ple_fits_gpu(required, total, weights, 131072, 8192), "GPU fits weights, both reserves and full K/V");
    check(!cuda::ple_fits_gpu(required - 1, total, weights, 131072, 8192), "one byte short falls back");
  }
  check(!cuda::ple_fits_gpu(UINT64_MAX, UINT64_MAX, 1, UINT64_MAX, 2), "oversized K/V does not wrap");
}

bool session_recovery(const char* file, const char* backend_name, std::string* err) {
  EngineOptions options;
  options.backend = backend_name;
  std::unique_ptr<os::DynLib> lib;
  flyx_device_info device{};
  if (!open_backend(options, &lib, &backend, &device, err)) return false;
  auto delegated = *backend;
  delegated.open = open_table;
  delegated.rewind = rewind_checked;
  flyx_test_delegate(&delegated);
  options.backend = "test-host-kv";
  options.max_ctx = 128;
  options.speculative = false;
  options.ple_placement = FLYX_PLE_DISK;
  Engine engine;
  if (!engine.load(file, options, err)) return false;
  Session session(&engine), fresh(&engine);
  if (!session.init(err) || !fresh.init(err)) return false;
  GenerateRequest request;
  request.prompt = {42, 43, 44};
  request.max_new_tokens = 6;
  request.ignore_eos = true;
  auto run = [&](Session& s, std::vector<int32_t>& tokens, bool fail_decode = false) {
    GenerateResult r;
    return generate(s, request, [&](std::string_view, const int32_t* t, int n) {
      tokens.insert(tokens.end(), t, t + n);
      if (fail_decode && n) fail_next = true;
      return true;
    }, &r, err);
  };
  std::vector<int32_t> failed, recovered, expected;
  const auto* original_seq = session.seq();
  observed_reset = false;
  fail_next = true;
  check(run(session, failed) != 0 && err->find("PLE") != std::string::npos, "positioned read error reaches the caller");
  check(!fail_next && failed.empty() && observed_reset && session.state_pos() == 0,
        "failed read yields no token and Generation resets the session");
  read_failed = false;
  observed_reset = false;
  check(run(session, failed, true) != 0 && err->find("PLE") != std::string::npos,
        "transient failure after emitting the prefill anchor reaches the caller");
  check(read_failed && observed_reset && !failed.empty() && failed.size() < 6,
        "failed decode drops remaining rounds and resets the same sequence");
  if (run(session, recovered) || run(fresh, expected)) return false;
  check(session.seq() == original_seq && recovered.size() == 6 && recovered == expected,
        "a subsequent request on the original sequence matches a fresh session");
  return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  queued_failure();
  budget_boundaries();
  std::string err;
  const bool ok = session_recovery(argv[1], argv[2], &err);
  check(ok, err);
  return test::finish("ple_recovery");
}
