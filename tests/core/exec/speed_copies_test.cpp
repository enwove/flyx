// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The K/V of the context asked for before the copies a backend keeps only for speed (Engine::plan_memory,
// flyx_memory_info::speed, ABI 54): where a session of that context fits on the GPU without them, the load gives them
// back, a step at a time, rather than put its positions past what fits beside them in host memory; where it would not
// fit without them either, or --kv-gpu limits the GPU's part, the copies stay.  The CPU backend runs the tiny model;
// its adapter reports a GPU of 8 GiB whose room after the weights the test sets, a session's bytes by its positions
// on it, and the copies trim() gives back.
#include <algorithm>
#include <cstdio>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/os/os.h"
#include "core/plan/budget.h"
#include "core/session/session.h"
#include "tests/common/check.h"

using namespace flyx;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

constexpr uint64_t kMiB = 1ull << 20;
constexpr uint64_t kTotal = 8ull << 30, kWeights = 1ull << 30, kFixed = 16 * kMiB, kRow = 64 * 1024;
constexpr int kCtx = 8192;

const flyx_backend_api* g_cpu = nullptr;
uint64_t g_room = 0;                // the sessions' room beside the weights and every copy
std::vector<uint64_t> g_copies;     // the copies kept for speed, in the order trim() gives them back
uint64_t g_freed = 0;
int g_trims = 0;

uint64_t speed() { return std::accumulate(g_copies.begin(), g_copies.end(), uint64_t(0)); }

int fake_memory(flyx_bmodel*, flyx_memory_info* out) {
  *out = {};
  out->weights = kWeights + speed();
  out->total = kTotal;
  out->free = plan::device_reserve(kTotal, os::gpu_memory_paged()) + g_room + g_freed;
  out->speed = speed();
  return 0;
}

uint64_t fake_trim(flyx_bmodel*, uint64_t bytes) {
  uint64_t freed = 0;
  while (!g_copies.empty() && freed < bytes) {
    freed += g_copies.front();
    g_copies.erase(g_copies.begin());
  }
  g_freed += freed;
  g_trims += freed > 0;
  return freed;
}

// A position on the GPU takes a row; one in host memory a quarter of one on the GPU (its staging) and a row there.
int fake_size(flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) {
  const int dev = std::min(l->device_positions, l->max_ctx);
  const uint64_t host = uint64_t(l->max_ctx - dev);
  *out = {};
  out->device = kFixed + uint64_t(dev) * kRow + host * kRow / 4;
  out->host = host * kRow;
  out->device_positions = dev;
  out->strata_from = l->max_ctx;
  return 0;
}

struct Loaded {
  bool ok = false;
  int device_positions = 0, ctx = 0, trims = 0;
  uint64_t speed_left = 0;
  std::vector<int32_t> tokens;
};

Loaded load(const std::string& model, uint64_t room, std::vector<uint64_t> copies, int kv_gpu = -1) {
  g_room = room;
  g_copies = std::move(copies);
  g_freed = 0;
  g_trims = 0;
  EngineOptions o;
  o.backend = "test-host-kv";
  o.max_ctx = kCtx;
  o.ctx_given = true;
  o.speculative = false;
  o.kv_method = "exact";
  o.kv_gpu = kv_gpu;
  o.kv_ram = int64_t(2) << 30;
  Loaded r;
  Engine e;
  std::string err;
  if (!e.load(model, o, &err)) {
    std::fprintf(stderr, "load: %s\n", err.c_str());
    return r;
  }
  r.ok = true;
  r.device_positions = std::min(e.kv_layout().device_positions, e.kv_layout().max_ctx);
  r.ctx = e.kv_layout().max_ctx;
  r.trims = g_trims;
  r.speed_left = speed();
  Session s(&e);
  GenerateRequest rq;
  rq.prompt = e.tokenizer().encode("The quick brown fox");
  rq.max_new_tokens = 8;
  rq.ignore_eos = true;
  GenerateResult res;
  TextSink sink = [&](std::string_view, const int32_t* t, int n) {
    r.tokens.insert(r.tokens.end(), t, t + n);
    return true;
  };
  if (!s.init(&err) || generate(s, rq, sink, &res, &err) != 0) std::fprintf(stderr, "generate: %s\n", err.c_str());
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 2) {
    std::fprintf(stderr, "usage: speed_copies_test TINY_MODEL.flyx\n");
    return 2;
  }
  EngineOptions probe;
  probe.backend = "cpu_ref";
  std::unique_ptr<os::DynLib> lib;
  flyx_device_info info{};
  std::string err;
  if (!open_backend(probe, &lib, &g_cpu, &info, &err)) {
    std::fprintf(stderr, "cannot open cpu_ref: %s\n", err.c_str());
    return 1;
  }
  flyx_backend_api delegated = *g_cpu;
  delegated.memory = fake_memory;
  delegated.trim = fake_trim;
  delegated.seq_size = fake_size;
  flyx_test_delegate(&delegated);
  const std::string model = argv[1];
  const uint64_t whole = kFixed + kCtx * kRow;  // 528 MiB

  // Beside the copy 300 MiB: 3072 positions on the GPU; without it 940 MiB, all of them.
  const Loaded given = load(model, 300 * kMiB, {640 * kMiB});
  check(given.ok && given.device_positions == kCtx && given.ctx == kCtx && given.trims == 1 && given.speed_left == 0,
        "a copy in the way of the context's K/V is given back: " + std::to_string(given.device_positions) +
            " positions on the GPU, " + std::to_string(given.trims) + " trims");
  check(given.tokens.size() == 8, "and the engine decodes after it");

  // The first of two copies makes room: the second stays.
  const Loaded first = load(model, 300 * kMiB, {300 * kMiB, 340 * kMiB});
  check(first.ok && first.device_positions == kCtx && first.trims == 1 && first.speed_left == 340 * kMiB,
        "a step at a time: the copy that makes room goes, the next one stays (" +
            std::to_string(first.speed_left / kMiB) + " MiB left)");

  // The whole session fits beside the copy: it stays.
  const Loaded beside = load(model, whole + 8 * kMiB, {640 * kMiB});
  check(beside.ok && beside.device_positions == kCtx && beside.trims == 0 && beside.speed_left == 640 * kMiB,
        "a copy beside the whole context stays");

  // Without the copy the context would not fit the GPU either: the copy stays, the K/V in tiers as before.
  const Loaded tiers = load(model, 300 * kMiB, {100 * kMiB});
  check(tiers.ok && tiers.device_positions == 3072 && tiers.trims == 0 && tiers.speed_left == 100 * kMiB,
        "a copy whose bytes would not bring the context on the GPU stays: " +
            std::to_string(tiers.device_positions) + " positions there");

  // --kv-gpu: the GPU's part is the user's; the copy stays.
  const Loaded limited = load(model, 300 * kMiB, {640 * kMiB}, 2048);
  check(limited.ok && limited.device_positions == 2048 && limited.trims == 0,
        "with --kv-gpu the copy stays: " + std::to_string(limited.device_positions) + " positions on the GPU");

  // Not even the K/V in tiers fits beside the copy (as before ABI 54): it goes, and the context is on the GPU.
  const Loaded none = load(model, 100 * kMiB, {640 * kMiB});
  check(none.ok && none.device_positions == kCtx && none.trims == 1, "a copy no session fits beside goes, as before");

  // Same tokens whether the copy went or stayed (the copies only run faster; here the CPU backend has none).
  check(given.tokens == beside.tokens && given.tokens == tiers.tokens, "the same tokens either way");
  return flyx::test::finish("speed_copies_test");
}
