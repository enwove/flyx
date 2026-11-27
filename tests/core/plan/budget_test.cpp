// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// plan::account (core/plan/budget.h): a session that fits keeps its context; one that does not shrinks to the
// longest context that fits, in steps of 1024 and not below the floor given, or, with no floor or
// nothing at or above it, the load is refused naming what would fit. With host memory for K/V, a
// context that does not fit keeps its length: its first positions' K/V on the GPU, the rest in host memory, as much
// as the host budget allows.  A backend that only reports memory and sizes.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/plan/budget.h"
#include "tests/common/check.h"

using namespace flyx;

namespace {

constexpr uint64_t kMiB = 1ull << 20, kGiB = 1ull << 30;
// 16 GiB, 10 of them weights, 0.5 the workspace; a session 1 GiB plus 64 KiB a token.  g_weights: what the weights
// take after the backend gave some back (trim); g_others: what other programs hold of the GPU; g_runtime: what the
// process's own runtime holds besides the model, which the system counts as the process's where g_split.
uint64_t g_weights = 10 * kGiB, g_others = 0, g_workspace = kGiB / 2, g_runtime = 0;
bool g_split = false;
int memory(flyx_bmodel*, flyx_memory_info* out) {
  *out = {g_weights, g_workspace, 16 * kGiB - g_weights - g_workspace - g_others - g_runtime, 16 * kGiB,
          g_split ? g_weights + g_workspace + g_runtime : 0};
  return 0;
}
int seq_bytes(flyx_bmodel*, int32_t max_ctx, uint64_t* out) {
  *out = kGiB + uint64_t(max_ctx) * 64 * 1024;
  return 0;
}

// The same session with its first device_positions positions on the device and the rest in host memory.
int seq_size(flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) {
  const int dev = l->device_positions < l->max_ctx ? l->device_positions : l->max_ctx;
  *out = {kGiB + uint64_t(dev) * 64 * 1024, uint64_t(l->max_ctx - dev) * 64 * 1024, dev, 0, 0};
  return 0;
}

flyx_backend_api fake() {
  flyx_backend_api api{};
  api.memory = memory;
  api.seq_bytes = seq_bytes;
  api.seq_size = seq_size;
  return api;
}

// With a low-bit copy: 16 layers' copies, 1 KiB a position each, on the device unless copy_host of
// them are in host memory, read through stagings of a layer each (two at most).
constexpr int kCopies = 16;
int seq_size_copy(flyx_bmodel* bm, const flyx_seq_layout* l, flyx_seq_size* out) {
  seq_size(bm, l, out);
  const int host = l->copy_host < kCopies ? l->copy_host : kCopies;
  const int dev = kCopies - host + (host < 2 ? host : 2);
  out->device += uint64_t(dev) * l->max_ctx * 1024;
  out->host += uint64_t(host) * l->max_ctx * 1024;
  out->copies = kCopies;
  return 0;
}

// FLYX_KV_AUTO: the copy as above; every position exact (its fallback, FLYX_KV_READ_ALL): none.
int seq_size_auto(flyx_bmodel* bm, const flyx_seq_layout* l, flyx_seq_size* out) {
  return l->method == FLYX_KV_READ_ALL ? seq_size(bm, l, out) : seq_size_copy(bm, l, out);
}

const plan::KvTiers kGpuOnly{-1, 0};

void check_shortfall_replay(flyx_backend_api api, plan::KvTiers tiers, const plan::Budget& before, int wanted,
                            int min_ctx, bool longest = false) {
  if (!before.ctx_need_more) return;
  flyx_memory_info memory{};
  api.memory(nullptr, &memory);
  uint64_t room = before.sessions_room;
  if (before.ctx_limit == plan::CtxLimit::kGpu) {
    // A larger fake GPU with exactly the quoted extra session room after its reserve.
    room += before.ctx_need_more;
    memory.total = 64 * kGiB;
    memory.free = memory.total / 40 + room;
  } else {
    tiers.ram = int64_t(before.host_budget + before.ctx_need_more);
  }
  api.memory = [](flyx_bmodel* bm, flyx_memory_info* out) {
    *out = *reinterpret_cast<flyx_memory_info*>(bm);
    return 0;
  };
  ResourceLedger ledger;
  plan::Budget after;
  int ctx = wanted;
  std::string err;
  const bool ok = plan::account(&api, reinterpret_cast<flyx_bmodel*>(&memory), false, false, &ctx, min_ctx, tiers,
                                &ledger, &after, &err, longest);
  check(ok && ctx == wanted && after.layout.max_ctx == wanted && after.sessions_room == room &&
            after.host_budget == uint64_t(tiers.ram) &&
            after.session_bytes <= after.sessions_room && after.session_host_bytes <= after.host_budget,
        "adding the reported " + std::to_string(before.ctx_need_more) + " bytes holds " + std::to_string(wanted) +
            " positions: " + err);
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  const flyx_backend_api api = fake();
  // The budget: 16 GiB less the reserve (16 GiB / 40) = 15.6 GiB; 5.1 GiB after weights and workspace, so a context
  // of up to (5.1 GiB - 1 GiB) / 64 KiB = 65,536 tokens' worth less a little.
  const uint64_t left = 16 * kGiB - 16 * kGiB / 40 - 10 * kGiB - kGiB / 2;
  const int longest = int((left - kGiB) / (64 * 1024)) / 1024 * 1024;
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 8192;
    check(plan::account(&api, nullptr, false, false, &ctx, 4096, kGpuOnly, &ledger, &b, &err) && ctx == 8192 &&
              b.ctx_limit == plan::CtxLimit::kNone && plan::ctx_limit_note(b).empty(),
          "a session that fits keeps its context");
  }
  {  // One 64K session fits on the GPU. Two need host K/V; each must fit half of both remaining budgets.
    for (int host_gib : {0, 4, 5}) {
      ResourceLedger ledger;
      plan::Budget b;
      std::string err;
      const plan::KvTiers tiers{-1, host_gib * int64_t(kGiB)};
      int ctx = 65536;
      check(plan::account(&api, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err) &&
                b.session_bytes == 5 * kGiB && !b.session_host_bytes,
            "the single-session layout fits all 65536 positions on the GPU");
      plan::fit_second_session(&api, nullptr, tiers, ledger, &b);
      if (host_gib == 5) {
        check(b.layout.max_ctx == ctx && b.layout.device_positions == 24576 && b.session_bytes <= left / 2 &&
                  b.session_host_bytes * 2 == b.host_budget && ledger.used() == 10 * kGiB + kGiB / 2,
              "two full contexts fit, with weights and workspace charged once");
      } else {
        check(b.layout.max_ctx == ctx && b.session_bytes == 5 * kGiB && !b.session_host_bytes,
              "too little host memory for the pair keeps the original single-session layout");
      }
    }
  }
  {  // AUTO's low-bit switch can move when device_positions changes. Preserve which positions are read exactly.
    auto automatic = fake();
    automatic.seq_size = [](flyx_bmodel* bm, const flyx_seq_layout* l, flyx_seq_size* s) {
      seq_size(bm, l, s);
      s->strata_from = l->method == FLYX_KV_AUTO ? std::min(l->max_ctx, s->device_positions + 32768) : l->max_ctx;
      return 0;
    };
    for (int ctx : {8192, 65536, 131072}) {
      ResourceLedger ledger;
      plan::Budget b;
      plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
      tiers.method = FLYX_KV_AUTO;
      std::string err;
      check(plan::account(&automatic, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err), "AUTO budget: " + err);
      const int dev = b.layout.device_positions;
      plan::fit_second_session(&automatic, nullptr, tiers, ledger, &b);
      if (ctx == 65536) {
        check(b.layout.method == FLYX_KV_READ_ALL && b.session_bytes <= left / 2 && b.session_host_bytes > 0,
              "two sessions keep an originally exact context exact even when moving K/V to host");
      } else {
        check(b.layout.method == FLYX_KV_AUTO && b.layout.device_positions == dev,
              ctx == 8192 ? "two sessions already fit: keep the original layout"
                          : "keep the original layout when two would switch to low-bit reads earlier");
      }
    }
  }
  {  // Two caches with different mapping granularities report their common device boundary.
    auto rounded = fake();
    rounded.memory = [](flyx_bmodel*, flyx_memory_info* m) {
      *m = {0, 0, 384 * kMiB + 6344, kGiB, 0};
      return 0;
    };
    rounded.seq_size = [](flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* s) {
      const int asked = std::min(l->max_ctx, l->device_positions);
      const int a = asked / 1024 * 1024, b = asked / 4096 * 4096;
      *s = {uint64_t(100 + a + b), uint64_t(2 * l->max_ctx - a - b), std::min(a, b), l->max_ctx, 0};
      return 0;
    };
    const plan::KvTiers tiers{-1, 26624};
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 8192;
    check(plan::account(&rounded, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err) &&
              b.layout.device_positions == 0 && b.session_host_bytes == 16384,
          "a single session is charged for the selected layout after rounding");
    plan::fit_second_session(&rounded, nullptr, tiers, ledger, &b);
    check(b.session_bytes == 100 && b.session_host_bytes == 16384 && b.host_budget / b.session_host_bytes == 1,
          "two rounded layouts cannot claim the host bytes of two larger device prefixes");
  }
  {  // Keeping AUTO's switch can require moving one more layer's copy to host than a GPU-only search would choose.
    auto automatic = fake();
    automatic.seq_size = [](flyx_bmodel* bm, const flyx_seq_layout* l, flyx_seq_size* s) {
      seq_size_auto(bm, l, s);
      s->strata_from = std::min(l->max_ctx, s->device_positions + 32768);
      return 0;
    };
    ResourceLedger ledger;
    plan::Budget b;
    plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
    tiers.method = FLYX_KV_AUTO;
    int ctx = 262144;
    std::string err;
    check(plan::account(&automatic, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err), "AUTO copy budget: " + err);
    const int dev = b.layout.device_positions;
    plan::fit_second_session(&automatic, nullptr, tiers, ledger, &b);
    check(dev == 1024 && b.layout.device_positions == dev && b.layout.copy_host == 13 && b.session_bytes <= left / 2,
          "two sessions keep AUTO's switch and bits with thirteen host copies");
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 4096, kGpuOnly, &ledger, &b, &err);
    check(ok && ctx == longest && b.session_bytes <= ledger.available() && b.layout.max_ctx == ctx &&
              b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more > 0,
          "one that does not shrinks to the longest multiple of 1024 that fits (" + std::to_string(ctx) + ", want " +
              std::to_string(longest) + "), and says the GPU cut it: " + plan::ctx_limit_note(b));
    uint64_t wanted_gpu = 0;
    seq_bytes(nullptr, 262144, &wanted_gpu);
    check(b.ctx_need_more == wanted_gpu - ledger.available(),
          "GPU-only shortfall matches exact bytes needed for max context");
    check_shortfall_replay(api, kGpuOnly, b, 262144, 4096);
    uint64_t next = 0;
    seq_bytes(nullptr, ctx + 1024, &next);
    check(next > ledger.available(), "and 1024 more would not fit");
  }
  {  // Under memory pressure, no fixed allocation may disappear from the accounting.
    auto low = fake();
    low.memory = [](flyx_bmodel* bm, flyx_memory_info* out) {
      *out = *reinterpret_cast<const flyx_memory_info*>(bm);
      return 0;
    };
    low.seq_size = [](flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) {
      *out = {16 * kMiB, 0, l->max_ctx, 0, 0};
      return 0;
    };
    for (auto reported : {flyx_memory_info{8 * kMiB, 4 * kMiB, 64 * kMiB, 16 * kGiB, 0},
                          flyx_memory_info{10 * kGiB, 64 * kMiB, 32 * kMiB, 16 * kGiB, 0},
                          flyx_memory_info{4 * kGiB, kGiB, 32 * kMiB, 16 * kGiB, 0}}) {
      ResourceLedger ledger;
      plan::Budget b;
      std::string err;
      int ctx = 8192;
      const bool ok = plan::account(&low, reinterpret_cast<flyx_bmodel*>(&reported), false, false, &ctx, 0,
                                    kGpuOnly, &ledger, &b, &err);
      check(!ok && !err.empty(), "refuse a load that cannot cover the reserve, weights or workspace");
    }
  }
  {  // where Windows pages GPU memory out of the backend's sight (WSL2), reserve is max(base, 832 MiB), not base + 512
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    plan::KvTiers paged = kGpuOnly;
    paged.gpu_paged = true;
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 4096, paged, &ledger, &b, &err);
    const uint64_t paged_left = 16 * kGiB - 832 * kMiB - 10 * kGiB - kGiB / 2;
    const int want = int((paged_left - kGiB) / (64 * 1024)) / 1024 * 1024;
    check(ok && ctx == want, "GPU memory Windows pages unseen (WSL2): 832 MiB reserve floor (" + std::to_string(ctx) + ", want " +
                                 std::to_string(want) + ")");
    check_shortfall_replay(api, paged, b, 262144, 4096);
  }
  {  // on a large GPU where base reserve (total / 40) already exceeds 832 MiB, gpu_paged adds nothing extra
    auto large = fake();
    large.memory = [](flyx_bmodel*, flyx_memory_info* out) {
      *out = {10 * kGiB, kGiB / 2, 37 * kGiB + kGiB / 2, 48 * kGiB, 0};
      return 0;
    };
    ResourceLedger l0, l1;
    plan::Budget b0, b1;
    std::string e0, e1;
    int c0 = 8192, c1 = 8192;
    plan::KvTiers paged = kGpuOnly;
    paged.gpu_paged = true;
    check(plan::account(&large, nullptr, false, false, &c0, 0, kGpuOnly, &l0, &b0, &e0) &&
              plan::account(&large, nullptr, false, false, &c1, 0, paged, &l1, &b1, &e1) &&
              b0.sessions_room == b1.sessions_room,
          "large GPU (>33 GiB) does not stack paged reserve on top of total/40");
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    check(!plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err) && ctx == 262144 &&
              err.find("Max supported context:") != std::string::npos,
          "a context asked for is refused, naming one that fits: " + err);
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    check(!plan::account(&api, nullptr, true, false, &ctx, 131072, kGpuOnly, &ledger, &b, &err) && ctx == 262144 &&
              err.find("Disable speculative decoding") != std::string::npos,
          "nothing at or above the floor fits: refused: " + err);
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 4096, tiers, &ledger, &b, &err);
    const int dev = b.layout.device_positions;
    check(ok && ctx == 262144 && dev == longest && b.session_bytes <= ledger.available() &&
              b.session_host_bytes == uint64_t(262144 - dev) * 64 * 1024 && b.host_budget == 64 * kGiB,
          "with host memory it keeps its context: the GPU holds " + std::to_string(dev) + " positions (want " +
              std::to_string(longest) + "), host memory the rest");
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const plan::KvTiers tiers{-1, int64_t(kGiB)};
    check(plan::account(&api, nullptr, false, false, &ctx, 4096, tiers, &ledger, &b, &err) && ctx == longest &&
              b.layout.device_positions >= ctx && !b.session_host_bytes,
          "too little host memory for the rest: it shrinks as before (" + std::to_string(ctx) + ")");
    check(b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more == 0,
          "1 GiB of host RAM can cover K/V on reload: the GPU-only shortfall is unknown");
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 8192;
    const plan::KvTiers tiers{2048, int64_t(64 * kGiB)};
    check(plan::account(&api, nullptr, false, false, &ctx, 4096, tiers, &ledger, &b, &err) && ctx == 8192 &&
              b.layout.device_positions == 2048 && b.session_host_bytes == uint64_t(8192 - 2048) * 64 * 1024,
          "--kv-gpu 2048 keeps 2048 positions on the GPU though all would fit");
  }
  {
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 8192;
    const plan::KvTiers tiers{0, int64_t(64 * kGiB)};
    check(plan::account(&api, nullptr, false, false, &ctx, 4096, tiers, &ledger, &b, &err) &&
              b.layout.device_positions == 0 && b.session_host_bytes == uint64_t(8192) * 64 * 1024,
          "--kv-gpu 0: every position in host memory");
  }
  {  // too little host memory for the rest of a context asked for: refused, naming the longest that the GPU and the
     // host memory hold together (here 4 GiB of host memory beside the GPU's 4.1 GiB: about 131,000 positions)
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const plan::KvTiers tiers{-1, int64_t(4 * kGiB)};
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err);
    const size_t at = err.find("Max supported context: ");
    int named = at == std::string::npos ? 0 : std::atoi(err.c_str() + at + 23);
    check(!ok && named > 120000 && named < 140000,
          "a context host memory cannot take is refused, naming one the GPU and host memory hold (" +
              std::to_string(named) + "): " + err);
    // --ctx max: the same machine takes that context instead of refusing; one step longer would not fit.
    ResourceLedger ledger2;
    plan::Budget b2;
    std::string err2;
    int longest = 262144;
    const bool ok2 = plan::account(&api, nullptr, false, false, &longest, 0, tiers, &ledger2, &b2, &err2, true);
    ResourceLedger ledger3;
    plan::Budget b3;
    std::string err3;
    int one_more = longest + 1024;
    check(ok2 && longest == named && b2.layout.max_ctx == longest && b2.session_host_bytes <= uint64_t(4 * kGiB) &&
              !plan::account(&api, nullptr, false, false, &one_more, 0, tiers, &ledger3, &b3, &err3),
          "--ctx max: the longest context the GPU and host memory hold (" + std::to_string(longest) + ") " + err2);
    const std::string why = plan::ctx_limit_note(b2);
    check(b2.ctx_limit == plan::CtxLimit::kHost && b2.ctx_need_more > 0 &&
              why.find("does not fit host memory: 4.29 GB of it can be spared") != std::string::npos &&
              why.find("more RAM needed for max context") != std::string::npos,
          "and says host memory cut it with max-context shortfall: " + why);
    check_shortfall_replay(api, tiers, b2, 262144, 0, true);
  }
  {  // --ctx max with --kv-method auto: 14 GiB of host memory hold 262,144 positions every one exact (each step then
     // reads the K/V past the GPU's part over the bus), not with the copy: --ctx 262144 takes the first, --ctx max a
     // shorter context with the copy
    flyx_backend_api api_auto = fake();
    api_auto.seq_size = seq_size_auto;
    plan::KvTiers tiers{-1, int64_t(14 * kGiB)};
    tiers.method = FLYX_KV_AUTO;
    ResourceLedger l1, l2;
    plan::Budget asked, most;
    std::string e1, e2;
    int ctx = 262144, most_ctx = 262144;
    const bool ok1 = plan::account(&api_auto, nullptr, false, false, &ctx, 0, tiers, &l1, &asked, &e1);
    const bool ok2 = plan::account(&api_auto, nullptr, false, false, &most_ctx, 0, tiers, &l2, &most, &e2, true);
    check(ok1 && ctx == 262144 && asked.auto_exact, "--ctx 262144 with auto: every position exact " + e1);
    check(ok2 && most_ctx > 131072 && most_ctx < 262144 && !most.auto_exact &&
              most.session_host_bytes <= uint64_t(14 * kGiB),
          "--ctx max with auto: the longest with the copy (" + std::to_string(most_ctx) + ") " + e2);
    check_shortfall_replay(api_auto, tiers, most, 262144, 0, true);
    tiers.allow_auto_exact = false;
    ResourceLedger locked_ledger;
    plan::Budget locked;
    check(!plan::account(&api_auto, nullptr, false, false, &ctx, 0, tiers, &locked_ledger, &locked, &e1) &&
              locked.oom && !locked.auto_exact && e1.find("AUTO attention switch") != std::string::npos,
          "a locked target switch refuses an exact-only fallback");
  }
  {  // the copy: 300,000 positions' copies take 4.6 GiB on the GPU, more than the 4.1 GiB left beside
     // the fixed state; with 4 in host memory the other 12 and two stagings take 4.0 GiB, with 3 4.3 GiB
    flyx_backend_api api_copy = fake();
    api_copy.seq_size = seq_size_copy;
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 300000;
    plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
    tiers.method = FLYX_KV_STRATA;
    const bool ok = plan::account(&api_copy, nullptr, false, false, &ctx, 4096, tiers, &ledger, &b, &err);
    flyx_seq_size three{};
    const flyx_seq_layout l3{ctx, 0, FLYX_KV_STRATA, 0, 3};
    seq_size_copy(nullptr, &l3, &three);
    check(ok && ctx == 300000 && b.layout.copy_host == 4 && b.session_bytes <= ledger.available() &&
              three.device > ledger.available() && b.layout.device_positions == 1024,
          "a copy the GPU does not hold: the fewest layers' copies in host memory (" +
              std::to_string(b.layout.copy_host) + ", want 4), the GPU's room left to K/V positions (" +
              std::to_string(b.layout.device_positions) + ", want 1024)");
    ResourceLedger ledger2;
    plan::Budget b2;  // its leases are ledger2's: after it
    tiers.copy_gpu = 10;
    ctx = 300000;
    check(plan::account(&api_copy, nullptr, false, false, &ctx, 4096, tiers, &ledger2, &b2, &err) &&
              b2.layout.copy_host == 6,
          "--kv-copy-gpu 10: the copies of the other 6 layers in host memory");
  }
  {  // the engine's second call, after the backend gave memory back (trim): the leases of the first are not counted
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 131072;  // 1 GiB + 8 GiB: no room beside 10 GiB of weights
    const bool first = plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err);
    check(!first && ledger.report().find("weights: model 10.74 GB") != std::string::npos,
          "a session that does not fit beside 10 GiB of weights is refused: " + ledger.report());
    g_weights -= 4 * kGiB;  // a copy kept for speed given back
    const bool second = plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err);
    check(second && ctx == 131072 && ledger.used() == 6 * kGiB + kGiB / 2 &&
              ledger.report().find("weights: model 6.44 GB") != std::string::npos,
          "after 4 GiB given back it fits, the weights counted once: " + ledger.report());
    g_weights += 4 * kGiB;
  }
  {  // the GPU's free memory below the reserve: the weights' lease is not held, which is not enough for one session
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 4096;
    g_others = 16 * kGiB - 10 * kGiB - kGiB / 2 - 64 * kMiB;  // 64 MiB free
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err);
    check(!ok && err.find("Not enough VRAM: 5.84 / 17.18 GB used by other apps.") != std::string::npos &&
              err.find("Close other apps to free at least 0.36 GB VRAM") != std::string::npos,
          "no room for the weights' own lease: refused, not granted with the weights uncounted: " + err);
    g_others = 0;
  }
  {  // free memory, weights and workspace together under the reserve: refused, naming what other programs hold
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 4096;
    g_weights = 128 * kMiB, g_workspace = 64 * kMiB;
    g_others = 16 * kGiB - g_weights - g_workspace - 64 * kMiB;  // 64 MiB free: 256 MiB in all, the reserve 410
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err);
    check(!ok && err.find("Not enough VRAM: 16.91 / 17.18 GB used by other apps.") != std::string::npos &&
              err.find("Close other apps to free at least 0.36 GB VRAM") != std::string::npos,
          "under the reserve altogether: refused, saying what other programs hold: " + err);
    g_weights = 10 * kGiB, g_workspace = kGiB / 2, g_others = 0;
  }
  {  // what holds the GPU besides the model: the process's own runtime told from other programs where the system
     // counts the process's share, in one figure with them where it does not
    g_others = 3 * kGiB, g_runtime = kGiB / 2;
    for (bool split : {false, true}) {
      g_split = split;
      ResourceLedger ledger;
      plan::Budget b;
      std::string err;
      int ctx = 262144;
      const bool ok = plan::account(&api, nullptr, false, false, &ctx, 0, kGpuOnly, &ledger, &b, &err);
      check(!ok && err.find("Close other apps, or use a smaller quantization.") != std::string::npos &&
                b.held.runtime_known == split && b.held.others == 3 * kGiB + (split ? 0 : kGiB / 2) &&
                b.held.runtime == (split ? kGiB / 2 : 0),
            std::string(split ? "the process's share counted" : "no share counted") + ": " + err);
    }
    g_others = g_runtime = 0, g_split = false;
  }
  {  // not even the shortest session fits: the refusal says what that one keeps on the GPU and how much is missing,
     // not what the context asked for would take
    g_split = true;
    g_others = left - kGiB + 100 * kMiB;  // the sessions' room 100 MiB under the fixed gigabyte
    const plan::KvTiers host{-1, int64_t(64 * kGiB)};
    for (bool longest : {false, true}) {
      ResourceLedger ledger;
      plan::Budget b;
      std::string err;
      int ctx = 262144;
      const bool ok =
          plan::account(&api, nullptr, true, false, &ctx, longest ? 1024 : 0, host, &ledger, &b, &err, longest);
      check(!ok &&
                err.find("used by other apps. Loading needs at least ") != std::string::npos &&
                err.find("1.07 GB session") != std::string::npos &&
                err.find("Close other apps to free at least 0.10 GB VRAM, disable speculative decoding") != std::string::npos &&
                err.find("-token context") == std::string::npos,
            std::string(longest ? "--ctx max" : "a context asked for") + ", no session fits: " + err);
    }
    // FLYX_KV_AUTO with Strata overhead at 1024 must still report the exact fallback floor in the OOM error.
    auto strata_auto = fake();
    strata_auto.seq_size = [](flyx_bmodel* bm, const flyx_seq_layout* l, flyx_seq_size* out) {
      seq_size(bm, l, out);
      if (l->method == FLYX_KV_AUTO) out->device += 256 * kMiB;  // Strata overhead when kStrataHostExact == 0
      return 0;
    };
    {
      ResourceLedger ledger;
      plan::Budget b;
      std::string err;
      int ctx = 262144;
      plan::KvTiers host_auto = host;
      host_auto.method = FLYX_KV_AUTO;
      const bool ok = plan::account(&strata_auto, nullptr, true, false, &ctx, 1024, host_auto, &ledger, &b, &err, true);
      check(!ok && err.find("1.07 GB session") != std::string::npos &&
                err.find("free at least 0.10 GB VRAM") != std::string::npos,
            "FLYX_KV_AUTO OOM floor uses exact fallback size, not Strata overhead: " + err);
    }
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const bool ok = plan::account(&api, nullptr, false, false, &ctx, 4096, kGpuOnly, &ledger, &b, &err);
    check(!ok && err.find("1.14 GB session") != std::string::npos &&
              err.find("Close other apps to free at least 0.17 GB VRAM") != std::string::npos,
          "no session fits the GPU alone: " + err);
    g_others = 0, g_split = false;
  }
  {  // --ctx max where what stays on the GPU grows with the context (two stagings of the copy): the longest the GPU
     // holds, the load saying it was the GPU and what would give more
    flyx_backend_api api_copy = fake();
    api_copy.seq_size = seq_size_copy;
    g_split = true;
    g_others = left - kGiB - kGiB / 4;  // the sessions' room: a quarter of a gigabyte past the fixed state
    plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
    tiers.method = FLYX_KV_STRATA;
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const bool ok = plan::account(&api_copy, nullptr, false, false, &ctx, 1024, tiers, &ledger, &b, &err, true);
    const std::string why = plan::ctx_limit_note(b);
    check(ok && ctx > 120000 && ctx <= 131072 && b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more == kGiB / 4 &&
              b.sessions_room == kGiB + kGiB / 4 &&
              why.find("does not fit the GPU: the sessions have 1.34 GB of it after the weights; other programs and "
                       "the driver hold") != std::string::npos &&
              why.find("more VRAM needed for max context") != std::string::npos &&
              why.find("(closing other programs gives a longer context)") != std::string::npos,
          "--ctx max cut by the GPU (" + std::to_string(ctx) + "): " + why + " " + err);
    check_shortfall_replay(api_copy, tiers, b, 262144, 1024, true);
    for (int host_gib : {15, 1}) {
      ctx = 262144;
      tiers.ram = int64_t(host_gib * kGiB);
      const auto limit = host_gib == 15 ? plan::CtxLimit::kGpu : plan::CtxLimit::kHost;
      check(plan::account(&api_copy, nullptr, false, false, &ctx, 1024, tiers, &ledger, &b, &err, true) &&
                ctx < 262144 && b.ctx_limit == limit && b.ctx_need_more == 0,
            "262144 positions with 1.25 GiB of GPU room and " + std::to_string(host_gib) +
                " GiB of host RAM: the other side cannot hold the candidate layout");
    }
    g_others = 0, g_split = false;
  }
  {  // FLYX_KV_AUTO falls back to exact at a short context (when Strata has a fixed staging floor), yet at max_ctx
     // Strata needs far less VRAM than exact: need_more_for must report Strata's smaller shortfall, not exact's.
    flyx_backend_api api_auto = fake();
    api_auto.seq_size = [](flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) -> int {
      const int dev = std::min(l->device_positions, l->max_ctx);
      out->device_positions = dev;
      out->copies = kCopies;
      if (l->method == FLYX_KV_AUTO) {
        // Strata: fixed 1.5 GiB floor (fails at 1.2 GiB room even at 1024 ctx), small growth to 2.0 GiB at 262144.
        out->device = kGiB + 512 * kMiB + uint64_t(dev) * 64 * 1024 + uint64_t(l->max_ctx) * 2048;
        out->host = uint64_t(l->max_ctx - dev) * 64 * 1024;
        out->strata_from = dev;
      } else {
        // Exact: 1.0 GiB floor (fits at 1.2 GiB room for short ctx), but non-windowed staging grows to 5.0 GiB at 262144.
        out->device = kGiB + uint64_t(dev) * 64 * 1024 + uint64_t(l->max_ctx) * 16 * 1024;
        out->host = uint64_t(l->max_ctx - dev) * 64 * 1024;
        out->strata_from = l->max_ctx;
      }
      return 0;
    };
    g_split = true;
    g_others = left - (kGiB + 256 * kMiB);  // 1.25 GiB sessions_room: exact fits ~16K, Strata needs ~2.0 GiB at 262K
    plan::KvTiers tiers{-1, int64_t(64 * kGiB)};
    tiers.method = FLYX_KV_AUTO;
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 262144;
    const bool ok = plan::account(&api_auto, nullptr, false, false, &ctx, 1024, tiers, &ledger, &b, &err, true);
    const uint64_t want_strata_need = (kGiB + 512 * kMiB + uint64_t(262144) * 2048) - (kGiB + 256 * kMiB);
    check(ok && ctx > 0 && ctx < 262144 && b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more == want_strata_need,
          "FLYX_KV_AUTO exact-fallback shortfall reports Strata VRAM need (" + std::to_string(b.ctx_need_more) +
              " vs want " + std::to_string(want_strata_need) + ")");
    check_shortfall_replay(api_auto, tiers, b, 262144, 1024, true);
    for (int wanted : {32768, 36864}) {
      // At 32768 the exact quote keeps Strata below its staging floor. At 36864 it would admit a shorter Strata
      // context, which --ctx max would select, so only the Strata quote can deliver wanted.
      ctx = wanted;
      const uint64_t need = wanted == 32768 ? 256 * kMiB : 256 * kMiB + uint64_t(wanted) * 2048;
      check(plan::account(&api_auto, nullptr, false, false, &ctx, 1024, tiers, &ledger, &b, &err, true) &&
                ctx < wanted && b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more == need,
            "AUTO reports the replayable shortfall for " + std::to_string(wanted) + " positions");
      check_shortfall_replay(api_auto, tiers, b, wanted, 1024, true);
    }
    ctx = 262144;
    tiers.ram = int64_t(15 * kGiB);
    check(plan::account(&api_auto, nullptr, false, false, &ctx, 1024, tiers, &ledger, &b, &err, true) &&
              ctx < 262144 && b.ctx_limit == plan::CtxLimit::kGpu && b.ctx_need_more == 0,
          "AUTO reports no GPU shortfall when both Strata and exact candidates exceed the host budget");
    g_others = 0, g_split = false;
  }
  // What --kv-ram auto leaves to the rest of the machine: 2 GiB up to 32 GiB of RAM (a 32 GB machine shows 31.2 GiB
  // to Windows), then a sixteenth.
  check(plan::host_keep(16 * kGiB) == 2 * kGiB && plan::host_keep(31 * kGiB + kGiB / 5) == 2 * kGiB &&
            plan::host_keep(64 * kGiB) == 4 * kGiB && plan::host_keep(128 * kGiB) == 8 * kGiB,
        "host memory kv_ram auto leaves: 2 GiB at least, a sixteenth of the RAM");

  // seq_size() failure must fail plan::account() with the backend's error instead of succeeding with session_bytes == 0,
  // and distinguish FLYX_ERR_OOM from generic non-OOM failures.
  {
    flyx_backend_api api_fail = fake();
    api_fail.seq_size = [](flyx_bmodel*, const flyx_seq_layout*, flyx_seq_size*) -> int { return FLYX_ERR_OOM; };
    api_fail.last_error = []() -> const char* { return "sizing the round graphs: out of GPU memory"; };
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 8192;
    const bool ok = plan::account(&api_fail, nullptr, false, false, &ctx, 4096, kGpuOnly, &ledger, &b, &err);
    check(!ok && b.oom && err.find("sizing the round graphs") != std::string::npos,
          "seq_size OOM fails plan::account with b.oom=true and backend error: " + err);
    api_fail.seq_size = [](flyx_bmodel*, const flyx_seq_layout*, flyx_seq_size*) -> int { return -1; };
    api_fail.last_error = []() -> const char* { return "unsupported layout"; };
    err.clear();
    const bool ok2 = plan::account(&api_fail, nullptr, false, false, &ctx, 4096, kGpuOnly, &ledger, &b, &err);
    check(!ok2 && !b.oom && err.find("unsupported layout") != std::string::npos,
          "seq_size non-OOM fails plan::account with b.oom=false: " + err);
  }

  // fit_second_session() must preserve non-1024-aligned device_positions (e.g. 512-token VMM granularity -> 1536)
  // rather than truncating min_device_positions / pair.gpu down to 1024.
  {
    flyx_backend_api api_vmm = fake();
    api_vmm.seq_size = [](flyx_bmodel*, const flyx_seq_layout* l, flyx_seq_size* out) -> int {
      // 512-token VMM rounding, 16 copies of 160 MiB each (at 64K): 1 session fits 1536 dev + 16 dev copies (3.6 GiB),
      // 2 sessions have 2.55 GiB/session so preserving dev=1536 requires moving 11 copies to host (<= 2.55 GiB).
      const int dev = (std::min(l->device_positions, l->max_ctx) / 512) * 512;
      const int host_c = std::min(l->copy_host, kCopies);
      const int dev_c = kCopies - host_c + std::min(host_c, 2);
      out->device = kGiB + uint64_t(dev) * 64 * 1024 + uint64_t(dev_c) * 160 * kMiB;
      out->host = uint64_t(l->max_ctx - dev) * 64 * 1024 + uint64_t(host_c) * 160 * kMiB;
      out->device_positions = dev;
      out->strata_from = dev;
      out->copies = kCopies;
      return 0;
    };
    ResourceLedger ledger;
    plan::Budget b;
    std::string err;
    int ctx = 65536;
    plan::KvTiers tiers{1536, int64_t(64 * kGiB)};
    tiers.method = FLYX_KV_AUTO;
    const bool ok = plan::account(&api_vmm, nullptr, false, false, &ctx, 0, tiers, &ledger, &b, &err);
    check(ok && b.layout.device_positions == 1536 && b.layout.copy_host == 0,
          "non-1024-aligned --kv-gpu 1536 keeps 1536 device positions (" + std::to_string(b.layout.device_positions) +
              ")");
    plan::fit_second_session(&api_vmm, nullptr, tiers, ledger, &b);
    check(b.layout.device_positions == 1536 && b.layout.copy_host > 0 && b.session_bytes <= ledger.available() / 2,
          "fit_second_session preserves non-1024-aligned device_positions=1536 via copy_host=" +
              std::to_string(b.layout.copy_host) + ", dev=" + std::to_string(b.layout.device_positions));
  }

  // Explicit --vram-reserve and --ram-reserve overrides: 0 lets a dedicated host use all available VRAM/RAM.
  {
    ResourceLedger ledger_auto, ledger_zero, ledger_big;
    plan::Budget b_auto, b_zero, b_big;
    std::string err;
    int ctx_auto = 8192, ctx_zero = 8192, ctx_big = 8192;
    plan::KvTiers t_auto = kGpuOnly;
    plan::KvTiers t_zero = kGpuOnly;
    t_zero.vram_reserve = 0;
    plan::KvTiers t_big = kGpuOnly;
    t_big.vram_reserve = int64_t(2 * kGiB);
    check(plan::account(&api, nullptr, false, false, &ctx_auto, 4096, t_auto, &ledger_auto, &b_auto, &err),
          "auto vram_reserve accounts: " + err);
    check(plan::account(&api, nullptr, false, false, &ctx_zero, 4096, t_zero, &ledger_zero, &b_zero, &err),
          "zero vram_reserve accounts: " + err);
    check(plan::account(&api, nullptr, false, false, &ctx_big, 4096, t_big, &ledger_big, &b_big, &err),
          "explicit 2 GiB vram_reserve accounts: " + err);
    check(ledger_zero.budget() == 16 * kGiB && ledger_zero.budget() > ledger_auto.budget(),
          "vram_reserve=0 unlocks full device VRAM (" + std::to_string(ledger_zero.budget()) + " > " +
              std::to_string(ledger_auto.budget()) + ")");
    check(ledger_big.budget() == 14 * kGiB && ledger_big.budget() < ledger_auto.budget(),
          "explicit vram_reserve=2 GiB reserves exactly 2 GiB (" + std::to_string(ledger_big.budget()) + ")");
  }
  {
    // Platform VRAM floor (min_vram_reserve, e.g. native Windows 256 MiB WDDM pieces floor) clamps lower overrides.
    const uint64_t floor = 256 * kMiB;
    check(plan::effective_device_reserve(16 * kGiB, false, 0, 0) == 0,
          "effective_device_reserve honors 0 when min_floor=0 (Linux)");
    check(plan::effective_device_reserve(16 * kGiB, false, 0, floor) == floor,
          "effective_device_reserve clamps 0 to 256 MiB floor");
    check(plan::effective_device_reserve(16 * kGiB, false, int64_t(100 * kMiB), floor) == floor,
          "effective_device_reserve clamps 100 MiB to 256 MiB floor");
    check(plan::effective_device_reserve(16 * kGiB, false, int64_t(512 * kMiB), floor) == 512 * kMiB,
          "effective_device_reserve preserves 512 MiB above 256 MiB floor");
    ResourceLedger l_zero, l_small, l_above;
    plan::Budget b_zero, b_small, b_above;
    std::string err;
    int c0 = 8192, c1 = 8192, c2 = 8192;
    plan::KvTiers t_zero = kGpuOnly;
    t_zero.vram_reserve = 0;
    t_zero.min_vram_reserve = floor;
    plan::KvTiers t_small = kGpuOnly;
    t_small.vram_reserve = int64_t(100 * kMiB);
    t_small.min_vram_reserve = floor;
    plan::KvTiers t_above = kGpuOnly;
    t_above.vram_reserve = int64_t(512 * kMiB);
    t_above.min_vram_reserve = floor;
    check(plan::account(&api, nullptr, false, false, &c0, 4096, t_zero, &l_zero, &b_zero, &err) &&
              l_zero.budget() == 16 * kGiB - floor,
          "min_vram_reserve=256 MiB clamps vram_reserve=0 in account()");
    check(plan::account(&api, nullptr, false, false, &c1, 4096, t_small, &l_small, &b_small, &err) &&
              l_small.budget() == 16 * kGiB - floor,
          "min_vram_reserve=256 MiB clamps vram_reserve=100 MiB in account()");
    check(plan::account(&api, nullptr, false, false, &c2, 4096, t_above, &l_above, &b_above, &err) &&
              l_above.budget() == 16 * kGiB - 512 * kMiB,
          "min_vram_reserve=256 MiB preserves vram_reserve=512 MiB in account()");
  }
  {
    // Auto host RAM budget (--kv-ram auto) honors --ram-reserve: 0 >= auto > large explicit reserve.
    ResourceLedger l0, l1, l2;
    plan::Budget b0, b1, b2;
    std::string err;
    int c0 = 65536, c1 = 65536, c2 = 65536;
    plan::KvTiers t_zero{-1, -1};
    t_zero.ram_reserve = 0;
    plan::KvTiers t_auto{-1, -1};
    plan::KvTiers t_large{-1, -1};
    t_large.ram_reserve = int64_t(4 * kGiB);
    check(plan::account(&api, nullptr, false, false, &c0, 0, t_zero, &l0, &b0, &err), "ram_reserve=0: " + err);
    check(plan::account(&api, nullptr, false, false, &c1, 0, t_auto, &l1, &b1, &err), "ram_reserve=auto: " + err);
    check(plan::account(&api, nullptr, false, false, &c2, 0, t_large, &l2, &b2, &err), "ram_reserve=4GiB: " + err);
    if (b0.host_budget > 0) {
      check(b0.host_budget > b1.host_budget,
            "ram_reserve=0 grants more host KV budget than auto (" + std::to_string(b0.host_budget) + " > " +
                std::to_string(b1.host_budget) + ")");
      check(b0.host_budget >= b2.host_budget + 3 * kGiB || b2.host_budget == 0,
            "explicit ram_reserve reduces host KV budget");
    }
  }
  {
    // Without per-process accounting (Linux), ~0.65 GiB of CUDA context is not misreported as closable other apps.
    const flyx_memory_info linux_ctx{0, 0, 16 * kGiB - 650 * kMiB, 16 * kGiB, 0};
    const flyx_memory_info linux_other{0, 0, 16 * kGiB - 1500 * kMiB, 16 * kGiB, 0};
    const flyx_memory_info win_other{10 * kGiB, 0, 16 * kGiB - 10 * kGiB - 500 * kMiB - 512 * kMiB, 16 * kGiB,
                                     10 * kGiB + 500 * kMiB};
    check(!plan::closable(plan::held_of(linux_ctx)),
          "0.65 GiB without process accounting is not blamed on other apps");
    check(plan::closable(plan::held_of(linux_other)),
          "1.5 GiB without process accounting is reported as other apps");
    check(plan::closable(plan::held_of(win_other)),
          "0.5 GiB with process accounting is reported as other apps");
  }

  return flyx::test::finish("budget_test");
}
