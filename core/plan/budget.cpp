// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/plan/budget.h"

#include <algorithm>
#include <cstdio>

#include "core/os/os.h"

namespace flyx::plan {

uint64_t host_keep(uint64_t total_ram) { return std::max<uint64_t>(2ull << 30, total_ram / 16); }

uint64_t device_reserve(uint64_t total_vram, bool gpu_paged) {
  // Windows pages a process's GPU memory out rather than refuse it, and does so before the GPU is full, the earlier
  // the larger the allocation (backends/cuda/runtime/device_stats.cpp): a 27B session filling the budget on a 16 GB
  // card under WSL2 (15.8 of 16.3 GB in use) prefilled 12 times slower, one with 0.15 GB less ran as fast as ever;
  // allocations of 256 MiB to 1 GiB start paging around 420-690 MiB free.  Under WSL2 (gpu_paged) the backend sees
  // none of it: hold at least 768 MiB + a 64 MiB desktop cushion there, without stacking on top of base.  Natively the
  // CUDA backend refuses an allocation past Windows' floor for its size, and makes what it allocates once the weights
  // are in (a session's arena and K/V, the prefill and staging pools) of pieces of 8 MiB at most, which stay on the
  // GPU to 120-140 MiB free: base keeps 250 MiB past that, as the 832 MiB did past the large allocations' floor.
  const uint64_t base = std::max<uint64_t>(384ull << 20, total_vram / 40);
  constexpr uint64_t kPagedReserve = (768ull + 64ull) << 20;
  return gpu_paged ? std::max(base, kPagedReserve) : base;
}

namespace {

// The host memory K/V may take: as asked, or what the machine can spare (its available memory less host_keep(), and
// less what the process takes besides).
uint64_t host_budget(const KvTiers& tiers) {
  if (tiers.ram >= 0) return uint64_t(tiers.ram);
  const os::SystemMemory m = os::system_memory();
  const uint64_t keep = host_keep(m.total) + tiers.reserved;
  return m.available > keep ? m.available - keep : 0;
}

// The layout of a session's K/V at `ctx` with as many positions on the GPU as `available` bytes hold (in steps of
// 1024), at most tiers.gpu; false when not even its fixed state fits (*why: the backend's error, if it gave one).  When
// none fits with the low-bit copy of every layer on the GPU, the copies of the fewest layers that make it fit go to
// host memory too, or as many as tiers.copy_gpu leaves.
bool tiered(const flyx_backend_api* api, flyx_bmodel* bm, int ctx, const KvTiers& tiers, uint64_t available,
            flyx_seq_layout* layout, flyx_seq_size* size, std::string* why, int min_device_positions = 0) {
  auto size_of = [&](int dev, int copy_host, flyx_seq_size* s) {
    const flyx_seq_layout l{ctx, dev, tiers.method, tiers.method_arg, copy_host};
    *s = {};
    const int r = api->seq_size(bm, &l, s);
    if (r != 0) *why = api->last_error();
    return r == 0;
  };
  auto fits = [&](int dev, int copy_host, flyx_seq_size* s) {
    return size_of(dev, copy_host, s) && s->device <= available;
  };
  // The most positions on the GPU, in steps of 1024 (plus non-1024-aligned endpoints so a VMM boundary or
  // explicit --kv-gpu cap is never truncated below min_device_positions), with copy_host copies in host memory.
  int sized_positions = 0;
  auto most = [&](int copy_host, flyx_seq_size* out) {
    const int cap = std::min(ctx, tiers.gpu >= 0 ? tiers.gpu : ctx);
    if (min_device_positions > cap) return false;
    int lo = (min_device_positions + 1023) / 1024, hi = cap / 1024, best_pos = -1;
    while (lo <= hi) {
      const int mid = (lo + hi) / 2;
      flyx_seq_size s{};
      if (fits(mid * 1024, copy_host, &s)) {
        best_pos = mid * 1024;
        *out = s;
        sized_positions = best_pos;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    if (cap > best_pos && (cap % 1024) != 0) {
      flyx_seq_size s{};
      if (fits(cap, copy_host, &s)) {
        best_pos = cap;
        *out = s;
        sized_positions = cap;
      }
    }
    if (best_pos < 0 && min_device_positions < cap && (min_device_positions % 1024) != 0) {
      flyx_seq_size s{};
      if (fits(min_device_positions, copy_host, &s)) {
        best_pos = min_device_positions;
        *out = s;
        sized_positions = min_device_positions;
      }
    }
    return best_pos >= 0;
  };
  const bool copy = tiers.method != FLYX_KV_READ_ALL;
  flyx_seq_size all{};  // every copy in host memory: how many there are
  int copy_host = 0;
  if (copy && tiers.copy_gpu >= 0 && size_of(0, INT32_MAX, &all)) copy_host = std::max(0, all.copies - tiers.copy_gpu);
  flyx_seq_size best_size{};
  bool placed = most(copy_host, &best_size);
  if (!placed && copy && tiers.copy_gpu < 0 && fits(min_device_positions, INT32_MAX, &all) && all.copies > 0) {
    int lo = 1, hi = all.copies;  // the fewest host copies that leave room for the required device positions
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      flyx_seq_size s{};
      if (fits(min_device_positions, mid, &s)) hi = mid;
      else lo = mid + 1;
    }
    copy_host = lo;
    placed = most(copy_host, &best_size);
  }
  if (!placed) return false;
  *layout = {ctx, best_size.device_positions, tiers.method, tiers.method_arg, std::min(copy_host, best_size.copies)};
  // A backend can round different caches at different granularities. Asking every cache for the common
  // boundary can change the host bytes and AUTO's switch from the probe that produced that boundary.
  if (layout->device_positions != sized_positions || layout->copy_host != copy_host)
    return fits(layout->device_positions, layout->copy_host, size);
  *size = best_size;
  return true;
}

// FLYX_KV_AUTO when its copy does not fit beside what stays on the GPU: every position read exactly.
KvTiers exact_of(KvTiers t) {
  t.method = FLYX_KV_READ_ALL;
  t.method_arg = 0;
  return t;
}

// A session's device bytes at `ctx` with every K/V position on the GPU, read by the tiers' method.
bool device_bytes(const flyx_backend_api* api, flyx_bmodel* bm, int ctx, const KvTiers& tiers, uint64_t* out,
                  std::string* err = nullptr, bool* oom = nullptr) {
  if (oom) *oom = false;
  const flyx_seq_layout l{ctx, INT32_MAX, tiers.method, tiers.method_arg, 0};
  flyx_seq_size s{};
  if (const int rc = api->seq_size(bm, &l, &s); rc != 0) {
    if (oom) *oom = rc == FLYX_ERR_OOM;
    if (err) {
      const char* msg = api->last_error ? api->last_error() : nullptr;
      *err = (msg && *msg) ? msg : "could not size a session";
    }
    return false;
  }
  *out = s.device;
  return true;
}

// What holds the GPU besides the model (budget.h).  A process's share the system counts below the model's own (its
// memory partly paged out) tells nothing: then as where it counts none.
GpuHeld held_of(const flyx_memory_info& mi) {
  GpuHeld h;
  h.total = mi.total;
  const uint64_t model = mi.weights + mi.workspace;
  if (mi.process >= model && mi.total >= mi.free + mi.process) {
    h.runtime_known = true;
    h.runtime = mi.process - model;
    h.others = mi.total - mi.free - mi.process;
  } else if (mi.total > mi.free + model) {
    h.others = mi.total - mi.free - model;
  }
  return h;
}

// "3.88" for 3.88e9 bytes (matching gb()'s decimal GB without the unit, for "3.88 / 17.09 GB").
std::string gb_num(uint64_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.2f", bytes / 1e9);
  return buf;
}

// One short sentence of actionable options when VRAM is short.
std::string vram_fix(const GpuHeld& held, uint64_t missing, bool has_drafter, bool mostly_bf16) {
  std::string alt;
  if (has_drafter) alt += ", disable speculative decoding";
  alt += mostly_bf16 ? ", or use NVFP4 weights" : ", or use a smaller quantization";
  if (closable(held)) {
    if (missing) return "Close other apps to free at least " + gb(missing) + " VRAM" + alt + ".";
    return "Close other apps" + alt + ".";
  }
  if (has_drafter)
    return std::string("Disable speculative decoding or use ") +
           (mostly_bf16 ? "NVFP4 weights." : "a smaller quantization.");
  return mostly_bf16 ? "Use NVFP4 weights." : "Use a smaller quantization.";
}

// Short, itemized VRAM refusal message: other-app usage (or GPU size), minimum needed breakdown, and fix.
std::string vram_short_error(const GpuHeld& held, uint64_t weights, uint64_t session, uint64_t overhead,
                             uint64_t missing, bool has_drafter, bool mostly_bf16) {
  const uint64_t need = weights + session + overhead;
  std::string msg;
  if (closable(held) && held.total > 0)
    msg = "Not enough VRAM: " + gb_num(held.others) + " / " + gb(held.total) + " used by other apps. ";
  else if (held.total > 0)
    msg = "Not enough VRAM: " + gb(held.total) + " GPU, " + gb(missing) + " short. ";
  else
    msg = "Not enough VRAM: " + gb(missing) + " short. ";
  if (session > 0)
    msg += "Loading needs at least " + gb(need) + " (" + gb(weights) + " weights + " + gb(session) + " session + " +
           gb(overhead) + " runtime/reserve). ";
  else
    msg += "Loading needs at least " + gb(need) + " (" + gb(weights) + " weights + " + gb(overhead) +
           " runtime/reserve). ";
  return msg + vram_fix(held, missing, has_drafter, mostly_bf16);
}

bool init_device_budget(const flyx_backend_api* api, flyx_bmodel* bm, const KvTiers& tiers, bool has_drafter,
                        bool mostly_bf16, ResourceLedger* ledger, Budget* out, flyx_memory_info* mi,
                        uint64_t* reserve_out, std::string* err) {
  *mi = {};
  *reserve_out = 0;
  const bool reported = api->memory(bm, mi) == 0 && mi->total > 0;
  out->held = held_of(*mi);
  uint64_t reserve = 0;
  auto fail_vram = [&] {
    out->oom = true;
    const uint64_t overhead = mi->workspace + out->held.runtime + reserve;
    const uint64_t missing = reserve > mi->free ? reserve - mi->free : 0;
    *err = vram_short_error(out->held, mi->weights, 0, overhead, missing, has_drafter, mostly_bf16);
    return false;
  };
  if (reported) {
    reserve = device_reserve(mi->total, tiers.gpu_paged);
    *reserve_out = reserve;
    const uint64_t have = mi->free + mi->weights + mi->workspace;
    // Zero is the ledger's sentinel for a backend that does not report a budget.
    if (have <= reserve) return fail_vram();
    ledger->set_budget(have - reserve);
  }
  std::string why;
  out->weights = ledger->reserve("weights", "model", mi->weights, &why);
  if (!out->weights) return fail_vram();
  out->workspace = ledger->reserve("workspace", "kernels", mi->workspace, &why);
  if (!out->workspace) return fail_vram();
  out->sessions_room = ledger->budget() ? ledger->available() : 0;
  return true;
}

// Places a session's K/V caches across GPU and host memory.
struct TierPlacer {
  const flyx_backend_api* api;
  flyx_bmodel* bm;
  const KvTiers& tiers;
  uint64_t room;
  Budget* out;

  // *by, where given: which memory did not hold it, when it is not placed.
  bool place(int ctx, flyx_seq_layout* l, flyx_seq_size* s, std::string* note, bool exact_ok = true,
             CtxLimit* by = nullptr) const {
    std::string e;
    if (by) *by = CtxLimit::kNone;
    bool placed = tiered(api, bm, ctx, tiers, room, l, s, &e);
    out->auto_exact = false;
    if (tiers.allow_auto_exact && exact_ok && (!placed || s->host > out->host_budget) && tiers.method == FLYX_KV_AUTO) {
      out->copy_note = !placed ? "it does not fit on the GPU"
                               : "it would take " + gb(s->host) + " of host memory, " + gb(out->host_budget) +
                                     " can be spared";
      flyx_seq_layout le{};
      flyx_seq_size se{};
      if (tiered(api, bm, ctx, exact_of(tiers), room, &le, &se, &e) && (!placed || se.host <= out->host_budget))
        *l = le, *s = se, placed = true, out->auto_exact = true;
    }
    if (!placed) {
      *note = !e.empty() ? "the backend cannot place it there: " + e
                         : std::string("what stays on the GPU does not fit there (the fixed state, the "
                                       "positions' scales and the staging, which grow with the context)");
      if (by) *by = CtxLimit::kGpu;
      return false;
    }
    if (s->host > out->host_budget) {
      *note = "it would take " + gb(s->host) + " of host memory, " + gb(out->host_budget) +
              " can be spared (--kv-ram GB gives more)";
      if (by) *by = CtxLimit::kHost;
      return false;
    }
    return true;
  }

  // What the next context past `ctx` (1024 more, `max_ctx` at most) does not fit in, placed as `ctx` was; *out's
  // notes stay those of `ctx`.
  CtxLimit limit_past(int ctx, int max_ctx, bool exact_ok) const {
    const bool auto_exact = out->auto_exact;
    const std::string copy_note = out->copy_note;
    flyx_seq_layout l{};
    flyx_seq_size s{};
    std::string note;
    CtxLimit by = CtxLimit::kNone;
    place(std::min(ctx + 1024, max_ctx), &l, &s, &note, exact_ok, &by);
    out->auto_exact = auto_exact;
    out->copy_note = copy_note;
    return by;
  }

  // Extra bytes on one side that would place wanted_ctx under --ctx max, or zero if no candidate does.
  uint64_t need_more_for(int wanted_ctx, CtxLimit by, bool exact_ok) const {
    if (by == CtxLimit::kNone || wanted_ctx <= 0) return 0;
    if (by == CtxLimit::kGpu && room == UINT64_MAX) return 0;
    auto shortfall = [&](const KvTiers& t) -> uint64_t {
      uint64_t device = room;
      flyx_seq_layout l{};
      flyx_seq_size s{};
      std::string why;
      if (by == CtxLimit::kGpu) {
        int copy_host = t.method == FLYX_KV_READ_ALL ? 0 : INT32_MAX;
        if (copy_host && t.copy_gpu >= 0) {
          flyx_seq_size all{};
          const flyx_seq_layout la{wanted_ctx, 0, t.method, t.method_arg, INT32_MAX};
          if (api->seq_size(bm, &la, &all) != 0) return 0;
          copy_host = std::max(0, all.copies - t.copy_gpu);
        }
        l = {wanted_ctx, 0, t.method, t.method_arg, copy_host};
        if (api->seq_size(bm, &l, &s) != 0 || s.host > out->host_budget || s.device <= room) return 0;
        device = s.device;
      }
      if (!tiered(api, bm, wanted_ctx, t, device, &l, &s, &why)) return 0;
      if (by == CtxLimit::kGpu) return s.host <= out->host_budget ? device - room : 0;
      return s.host > out->host_budget ? s.host - out->host_budget : 0;
    };
    const uint64_t need = shortfall(tiers);
    if (!exact_ok || tiers.method != FLYX_KV_AUTO) return need;
    const uint64_t exact = shortfall(exact_of(tiers));
    if (!exact || (need && need <= exact)) return need;

    // More memory can enable a shorter Strata context, which --ctx max prefers to exact at wanted_ctx.
    Budget probe;
    probe.host_budget = out->host_budget + (by == CtxLimit::kHost ? exact : 0);
    const TierPlacer replay{api, bm, tiers, room + (by == CtxLimit::kGpu ? exact : 0), &probe};
    flyx_seq_layout l{};
    flyx_seq_size s{};
    std::string why;
    if (!replay.place(wanted_ctx, &l, &s, &why, false) && replay.longest_placed(wanted_ctx, &l, &s, false)) return need;
    return exact;
  }

  // The longest context below max_ctx, a multiple of 1024, that the GPU and host memory hold together (0: none): a
  // binary search, then the next 16 steps one by one (placing is not monotone everywhere: past auto's switch a copy
  // appears).  Placed again at the end, so *out's notes are that context's.
  int longest_placed(int max_ctx, flyx_seq_layout* l, flyx_seq_size* s, bool exact_ok = true) const {
    int lo = 1, hi = (max_ctx - 1) / 1024, best = 0;
    std::string note;
    while (lo <= hi) {
      const int mid = (lo + hi) / 2;
      if (place(mid * 1024, l, s, &note, exact_ok)) best = mid, lo = mid + 1;
      else hi = mid - 1;
    }
    for (int k = best + 1; k <= std::min(best + 16, (max_ctx - 1) / 1024); ++k)
      if (place(k * 1024, l, s, &note, exact_ok)) best = k;
    return best > 0 && place(best * 1024, l, s, &note, exact_ok) ? best * 1024 : 0;
  }
};

// The longest GPU-only context in [min_ctx, max_ctx) that fits in `available` bytes, in steps of 1024.
bool fit_gpu_only(const flyx_backend_api* api, flyx_bmodel* bm, const KvTiers& tiers, uint64_t available, int max_ctx,
                  int min_ctx, int* fitted_ctx, Budget* out) {
  int lo = (min_ctx + 1023) / 1024, hi = (max_ctx - 1) / 1024, best = 0;
  uint64_t best_bytes = 0;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    uint64_t b = 0;
    if (device_bytes(api, bm, mid * 1024, tiers, &b) && b <= available) {
      best = mid, best_bytes = b;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (best <= 0) return false;
  *fitted_ctx = best * 1024;
  out->layout.max_ctx = *fitted_ctx;  // the sessions' context too
  out->session_bytes = best_bytes;
  return true;
}

// The device bytes of the shortest session there is (kLeastCtx tokens): `in_host`, with its K/V and their copies in
// host memory, which is the least a session keeps on the GPU; else wholly on the GPU.  0: the backend sizes none.
constexpr int kLeastCtx = 1024;
uint64_t least_session(const TierPlacer& placer, bool in_host) {
  uint64_t bytes = 0;
  if (!in_host) return device_bytes(placer.api, placer.bm, kLeastCtx, placer.tiers, &bytes) ? bytes : 0;
  // FLYX_KV_AUTO falls back to exact when Strata copies do not fit, so the true floor is exact.
  const bool exact = placer.tiers.allow_auto_exact && placer.tiers.method == FLYX_KV_AUTO;
  const KvTiers t = exact ? exact_of(placer.tiers) : placer.tiers;
  const flyx_seq_layout l{kLeastCtx, 0, t.method, t.method_arg, INT32_MAX};
  flyx_seq_size s{};
  return placer.api->seq_size(placer.bm, &l, &s) == 0 ? s.device : 0;
}

std::string format_oom_error(const TierPlacer& placer, const flyx_memory_info& mi, uint64_t reserve, int max_ctx,
                             bool has_drafter, bool mostly_bf16, bool tiers_on) {
  const GpuHeld& held = placer.out->held;
  const uint64_t left = placer.room;
  int fit_ctx = 0;
  if (tiers_on) {  // the longest the GPU and host memory hold together
    flyx_seq_layout l{};
    flyx_seq_size placed{};
    fit_ctx = placer.longest_placed(max_ctx, &l, &placed);
  }
  if (!fit_ctx) {
    for (int ctx = max_ctx / 2; ctx >= 1024; ctx /= 2) {
      uint64_t b = 0;
      if (device_bytes(placer.api, placer.bm, ctx, placer.tiers, &b) && b <= left) {
        fit_ctx = ctx;
        break;
      }
    }
  }
  // No context fits: the context asked for is not what is in the way.  What the shortest session keeps on the GPU
  // is, and how much of it is missing says how far off the load is.
  const uint64_t least = !fit_ctx ? least_session(placer, tiers_on) : 0;
  const uint64_t missing = least > left ? least - left : 0;
  if (missing) {
    const uint64_t overhead = mi.workspace + held.runtime + reserve;
    return vram_short_error(held, mi.weights, least, overhead, missing, has_drafter, mostly_bf16);
  }
  std::string msg = "Not enough VRAM for " + std::to_string(max_ctx) + "-token context (needs " +
                    gb(placer.out->session_bytes) + ", " + gb(left) + " free after weights). ";
  if (fit_ctx > 0) msg += "Max supported context: " + std::to_string(fit_ctx) + " tokens. ";
  return msg + vram_fix(held, 0, has_drafter, mostly_bf16);
}

}  // namespace

std::string held_note(const GpuHeld& h) {
  if (!h.total || !h.others) return "";
  if (!h.runtime_known)
    return "other programs, the driver and flyx's own runtime hold " + gb(h.others) + " of its " + gb(h.total);
  return "other programs and the driver hold " + gb(h.others) + " of its " + gb(h.total) +
         (h.runtime ? ", flyx's own runtime " + gb(h.runtime) : "");
}

bool closable(const GpuHeld& h) { return h.others >= (h.runtime_known ? 256ull << 20 : 1ull << 30); }

std::string ctx_limit_note(const Budget& b) {
  if (b.ctx_limit == CtxLimit::kHost)
    return "a longer context does not fit host memory: " + gb(b.host_budget) +
           " of it can be spared for K/V (--kv-ram GB gives more)" +
           (b.ctx_need_more ? " (" + gb(b.ctx_need_more) + " more RAM needed for max context)" : "");
  if (b.ctx_limit != CtxLimit::kGpu) return "";
  const std::string others = held_note(b.held);
  return "a longer context does not fit the GPU: the sessions have " + gb(b.sessions_room) +
         " of it after the weights" + (others.empty() ? "" : "; " + others) +
         (b.ctx_need_more ? "; " + gb(b.ctx_need_more) + " more VRAM needed for max context" : "") +
         (closable(b.held) ? " (closing other programs gives a longer context)" : "");
}

bool account(const flyx_backend_api* api, flyx_bmodel* bm, bool has_drafter, bool mostly_bf16, int* fitted_ctx,
             int min_ctx, const KvTiers& tiers, ResourceLedger* ledger, Budget* out, std::string* err, bool longest) {
  const int max_ctx = *fitted_ctx;
  *out = Budget{};  // a call before held leases: reserved again beside them, the weights would count twice
  out->layout = {max_ctx, INT32_MAX, tiers.method, tiers.method_arg, 0};
  out->host_budget = host_budget(tiers);
  flyx_memory_info mi{};
  uint64_t reserve = 0;
  if (!init_device_budget(api, bm, tiers, has_drafter, mostly_bf16, ledger, out, &mi, &reserve, err)) return false;
  uint64_t sb = 0;
  if (!device_bytes(api, bm, max_ctx, tiers, &sb, err, &out->oom)) {
    // Not even a session's fixed state fits beside the weights (a round's graphs are sized on one): what makes room.
    if (out->oom)
      *err = "Not enough VRAM for a session beside the weights (" + *err + "). " +
             vram_fix(out->held, 0, has_drafter, mostly_bf16);
    return false;
  }
  out->session_bytes = sb;

  const TierPlacer placer{api, bm, tiers, ledger->budget() ? ledger->available() : UINT64_MAX, out};
  const bool tiers_on = tiers.ram != 0 && out->host_budget > 0;
  const bool limited = tiers.gpu >= 0 && tiers.gpu < max_ctx;
  if (limited || out->session_bytes > placer.room) {
    flyx_seq_layout l{};
    flyx_seq_size s{};
    if (tiers.ram == 0) {
      out->tier_note = "--kv-ram 0";
    } else if (out->host_budget == 0) {
      out->tier_note = "no host memory to spare for it (--kv-ram GB gives some)";
    } else if (placer.place(max_ctx, &l, &s, &out->tier_note, !longest)) {
      out->layout = l;
      out->session_bytes = s.device;
      out->session_host_bytes = s.host;
      out->session_host_mapped = s.host_mapped;
      return true;
    } else if (longest) {
      // --ctx max, in place of a refusal: the longest the two hold with auto's copy (every position exact past the
      // GPU's part would have each step read all of it over the bus: a deep conversation at the bus's speed); only
      // when the copy fits nowhere, every position exact.
      bool exact_ok = false;
      int ctx = placer.longest_placed(max_ctx, &l, &s, exact_ok);
      if (ctx == 0 && tiers.allow_auto_exact) {
        exact_ok = true;
        ctx = placer.place(max_ctx, &l, &s, &out->tier_note) ? max_ctx : placer.longest_placed(max_ctx, &l, &s);
      }
      if (ctx > 0) {
        *fitted_ctx = ctx;
        out->layout = l;
        out->session_bytes = s.device;
        out->session_host_bytes = s.host;
        out->session_host_mapped = s.host_mapped;
        out->tier_note.clear();
        if (ctx < max_ctx) {
          out->ctx_limit = placer.limit_past(ctx, max_ctx, exact_ok);
          out->ctx_need_more = placer.need_more_for(max_ctx, out->ctx_limit, exact_ok);
        }
        return true;
      }
    }
  }
  if (out->session_bytes > placer.room) {
    if (min_ctx > 0 && fit_gpu_only(api, bm, tiers, placer.room, max_ctx, min_ctx, fitted_ctx, out)) {
      out->ctx_limit = CtxLimit::kGpu;
      // With host RAM, reloading the wanted context tries tiered placement before this GPU-only fallback.
      if (!tiers_on) out->ctx_need_more = sb - placer.room;
      return true;
    }
    out->oom = true;
    *err = format_oom_error(placer, mi, reserve, max_ctx, has_drafter, mostly_bf16, tiers_on);
    if (tiers.method == FLYX_KV_AUTO && !tiers.allow_auto_exact)
      *err += " The requested context must keep the target model's AUTO attention switch.";
    return false;
  }
  return true;
}

void fit_second_session(const flyx_backend_api* api, flyx_bmodel* bm, const KvTiers& tiers,
                        const ResourceLedger& ledger, Budget* budget) {
  const uint64_t device = ledger.budget() ? ledger.available() / 2 : UINT64_MAX;
  const uint64_t host = budget->host_budget / 2;
  if (!host || (budget->session_bytes <= device && budget->session_host_bytes <= host)) return;

  const int ctx = budget->layout.max_ctx;
  KvTiers pair = tiers;
  pair.method = budget->layout.method;
  pair.method_arg = budget->layout.method_arg;
  flyx_seq_size original{};
  if (pair.method == FLYX_KV_AUTO) {
    if (api->seq_size(bm, &budget->layout, &original) != 0) return;
    // Moving the device boundary can start using a low-bit copy earlier. Keep a wholly exact context exact.
    if (original.strata_from >= ctx) pair = exact_of(pair);
  }
  flyx_seq_layout layout{};
  flyx_seq_size size{};
  std::string why;
  if (!tiered(api, bm, ctx, pair, device, &layout, &size, &why)) return;
  if (pair.method == FLYX_KV_AUTO && size.strata_from != original.strata_from) {
    // Keep the old device boundary and move more layers' copies instead.
    pair.gpu = original.device_positions;
    if (!original.copies || !tiered(api, bm, ctx, pair, device, &layout, &size, &why, original.device_positions) ||
        size.strata_from != original.strata_from) return;
  }
  if (size.host > host) return;
  budget->layout = layout;
  budget->session_bytes = size.device;
  budget->session_host_bytes = size.host;
  budget->session_host_mapped = size.host_mapped;
  budget->tier_note.clear();
}

}  // namespace flyx::plan
