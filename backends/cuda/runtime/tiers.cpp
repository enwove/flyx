// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// K/V caches in two tiers: one address range whose first rows are device memory and whose rest is
// page-locked host memory, mapped with the driver's virtual memory management.  A kernel reads and writes the range
// as it does any cache (the host rows go over the bus: 52 GB/s on PCIe 5 x16, against 800 GB/s from the device); the
// graphs keep their addresses.  The driver's functions come through the runtime (cudaGetDriverEntryPointByVersion):
// the backend links no driver library.
#include <cuda.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "backends/cuda/backend.h"

using namespace flyx::cuda;

namespace {

struct Vmm {
  decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
  decltype(&cuMemAddressReserve) reserve = nullptr;
  decltype(&cuMemAddressFree) address_free = nullptr;
  decltype(&cuMemCreate) create = nullptr;
  decltype(&cuMemRelease) release = nullptr;
  decltype(&cuMemMap) map = nullptr;
  decltype(&cuMemUnmap) unmap = nullptr;
  decltype(&cuMemSetAccess) set_access = nullptr;
  size_t gran = 0;  // the larger of the device's and the host's granularity
  int device = 0;
  std::string missing;
};

const Vmm& vmm() {
  static Vmm v;
  static std::once_flag once;
  std::call_once(once, [] {
    auto get = [&](const char* name, auto* fn) {
      void* p = nullptr;
      cudaDriverEntryPointQueryResult q{};
      if (cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q) != cudaSuccess ||
          q != cudaDriverEntryPointSuccess || !p) {
        if (v.missing.empty()) v.missing = std::string("the driver has no ") + name;
        return;
      }
      *fn = reinterpret_cast<std::remove_pointer_t<decltype(fn)>>(p);
    };
    get("cuMemGetAllocationGranularity", &v.granularity);
    get("cuMemAddressReserve", &v.reserve);
    get("cuMemAddressFree", &v.address_free);
    get("cuMemCreate", &v.create);
    get("cuMemRelease", &v.release);
    get("cuMemMap", &v.map);
    get("cuMemUnmap", &v.unmap);
    get("cuMemSetAccess", &v.set_access);
    if (!v.missing.empty()) return;
    cudaGetDevice(&v.device);
    CUmemAllocationProp dev{}, host{};
    dev.type = host.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    dev.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
    host.location = {CU_MEM_LOCATION_TYPE_HOST, 0};
    size_t gd = 0, gh = 0;
    if (v.granularity(&gd, &dev, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
        v.granularity(&gh, &host, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || !gd || !gh) {
      v.missing = "the driver maps no host memory into a device's address range";
      return;
    }
    v.gran = std::max(gd, gh);
  });
  return v;
}

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) throw std::runtime_error(std::string("K/V in host memory: ") + what + " failed (" +
                                                  std::to_string(int(r)) + ")");
}

size_t round_up(size_t n, size_t g) { return (n + g - 1) / g * g; }

}  // namespace

namespace flyx::cuda {

size_t vmm_granularity() {
  const Vmm& v = vmm();
  return v.missing.empty() && v.gran > 0 ? v.gran : (2ull << 20);
}

VmmSlab::~VmmSlab() {
  if (handles.empty()) return;
  const Vmm& v = vmm();
  if (!v.missing.empty()) return;
  for (unsigned long long h : handles)
    if (h) v.release(CUmemGenericAllocationHandle(h));
}

void prefill_vmm_free(flyx_bmodel* m) {
  if (!m) return;
  const Vmm& v = vmm();
  if (m->prefill_va && v.missing.empty() && m->prefill_slab) {
    const size_t g = m->prefill_slab->gran;
    for (size_t o = 0; o < m->prefill_va_bytes; o += g) v.unmap(CUdeviceptr(m->prefill_va) + o, g);
    v.address_free(CUdeviceptr(m->prefill_va), m->prefill_va_bytes);
  }
  m->prefill_va = 0;
  m->prefill_va_bytes = 0;
  m->prefill_slab.reset();
  m->prefill_zero_ptr = nullptr;
  m->prefill_zero_bytes = 0;
  m->prefill_dirty = false;
}

void* prefill_vmm_alloc(flyx_bmodel* m, size_t bytes, size_t* actual_bytes) {
  const Vmm& v = vmm();
  if (!v.missing.empty() || bytes == 0) return nullptr;
  const size_t total = round_up(bytes, v.gran);
  const size_t pieces = total / v.gran;
  auto slab = std::make_shared<VmmSlab>();
  slab->gran = v.gran;
  slab->bytes = total;
  slab->handles.reserve(pieces);
  CUmemAllocationProp dp{};
  dp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  CUmemAccessDesc access{};
  access.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUdeviceptr va = 0;
  if (v.reserve(&va, total, v.gran, 0, 0) != CUDA_SUCCESS) return nullptr;
  size_t mapped_n = 0;
  auto cleanup = [&] {
    for (size_t i = 0; i < mapped_n; ++i) v.unmap(va + i * v.gran, v.gran);
    v.address_free(va, total);
  };
  for (size_t i = 0; i < pieces; ++i) {
    CUmemGenericAllocationHandle h = 0;
    if (v.create(&h, v.gran, &dp, 0) != CUDA_SUCCESS) {
      cleanup();
      return nullptr;
    }
    slab->handles.push_back(h);
    if (v.map(va + i * v.gran, v.gran, 0, h, 0) != CUDA_SUCCESS) {
      cleanup();
      return nullptr;
    }
    ++mapped_n;
  }
  if (v.set_access(va, total, &access, 1) != CUDA_SUCCESS) {
    cleanup();
    return nullptr;
  }
  CK(cudaMemset(reinterpret_cast<void*>(va), 0, bytes));
  m->prefill_va = va;
  m->prefill_va_bytes = total;
  m->prefill_slab = std::move(slab);
  if (actual_bytes) *actual_bytes = total;
  return reinterpret_cast<void*>(va);
}

}  // namespace flyx::cuda

void* flyx_seq::pageable_alloc_raw(size_t bytes) {
  if (dry) {
    dry_host_bytes += bytes;
    return reinterpret_cast<void*>(uintptr_t(256));
  }
  if (bytes == 0) return nullptr;
  void* p = nullptr;
#if defined(_WIN32)
  p = VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (!p)
    throw std::runtime_error("K/V in pageable host memory: VirtualAlloc failed (" + std::to_string(GetLastError()) +
                             ")");
#else
  p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) throw std::runtime_error("K/V in pageable host memory: mmap failed");
#endif
  pageable_allocs.push_back({p, bytes});
  return p;
}

flyx_seq::Tiered flyx_seq::tiered_alloc(size_t positions, size_t row, int role, int parity, int scale_floats,
                                        bool pageable, int ring_tail, bool even_dev) {
  Tiered t;
  const size_t want = positions * row;
  if (device_positions < 0 || size_t(device_positions) >= positions) {
    t.at = alloc<uint8_t>(want);
    return t;
  }
  const Vmm& v = vmm();
  if (!v.missing.empty()) throw std::runtime_error("K/V in host memory: " + v.missing);
  const size_t total = round_up(want, v.gran);
  // The device part rounds down to the granularity: the rows past it go to the host with the rest.
  // Strata pageable layers require an even device_rows split so no Strata pair straddles device and pageable RAM;
  // enforcing the same even rounding for all page-eligible layers keeps device_positions and staging slots identical
  // across pinned and pageable layers in hybrid auto mode.
  // alloc_state asks every cache for positions that are whole rows here (tiered_device_positions), so K and V of a
  // layer, and every layer, get the same device_rows.
  const size_t dev = tiered_device_bytes(device_positions, row, v.gran, pageable || even_dev);
  const size_t host = total - dev;
  t.device_rows = int(dev / row);
  // The staging the view's rest maps: one for the caches of a role, size and parity, which attentions read in turn,
  // in the pool (pool_finish).  As large as the host part; with strata attention, stage_window positions and what a
  // round reading the caches reaches.
  size_t stage = host, held = host / row;  // the staging's bytes, and the positions past the device part it holds
  if (stage_window > 0) {
    const long long want_w = std::max<long long>(stage_window, (long long)stage_reach - t.device_rows);
    const size_t whole = size_t((want_w + kFlashWindowAlign - 1) / kFlashWindowAlign * kFlashWindowAlign);
    if (whole * row < host) stage = round_up(whole * row, v.gran), held = whole;  // the rest of the piece unused
  }
  t.stage_rows = int(std::min<size_t>(held, INT32_MAX));
  int slot = -1;
  for (size_t i = 0; i < staging.size(); ++i)
    if (staging[i].role == role && staging[i].bytes == stage && staging[i].parity == parity) slot = int(i);
  if (slot < 0) {
    slot = int(staging.size());
    staging.push_back({role, parity, stage, staging_bytes, row, t.device_rows});
    staging_bytes += stage;  // a multiple of the granularity, as each offset
    if (scale_floats) staging.back().scales = alloc<float>(stage / row * size_t(scale_floats));
  }
  t.slot = slot;
  t.scales = staging[size_t(slot)].scales;

  if (pageable) {
    // When dev is not a multiple of row, slot t.device_rows straddles the VMM boundary dev and its prefix bytes live
    // in the device handle `hd` that `view` also maps for staged reads. Skip that straddling slot (ring_pad = 1) so
    // every tail-ring slot lies strictly inside the private ring handle `hr` (va + dev) and staging into `view` at
    // device_positions can never clobber or race with a wrapped tail-ring token.
    const int ring_pad = (dev > 0 && (dev % row) != 0) ? 1 : 0;
    t.page_ring_base = t.device_rows + ring_pad;
    t.page_ring = t.page_ring_base + ring_tail;
    const size_t host_want = (positions - size_t(t.device_rows)) * row;
    t.page = static_cast<uint8_t*>(pageable_alloc_raw(host_want));
    const size_t ring_bytes = size_t(ring_tail) * row;
    const size_t ring_vmm_bytes = size_t(ring_tail + ring_pad) * row;
    device_positions = std::min(device_positions, t.device_rows);
    if (dry) {
      if (dev == 0) {
        t.at = alloc<uint8_t>(ring_bytes);
      } else {
        dry_bytes += dev + round_up(ring_vmm_bytes, v.gran);
        t.at = reinterpret_cast<void*>(uintptr_t(256));
      }
      t.view = reinterpret_cast<void*>(uintptr_t(256));
      return t;
    }
    CUmemAllocationProp dp{};
    dp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
    CUmemAccessDesc access{};
    access.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    mapped.push_back(Mapped{});
    Mapped& m = mapped.back();
    m.dev = dev;
    m.bytes = total;
    m.stage = stage;
    m.slot = slot;
    CUdeviceptr vw = 0;
    check(v.reserve(&vw, total, v.gran, 0, 0), "reserving addresses for a view");
    m.view = vw;
    if (dev == 0) {
      m.va = 0;
      m.va_bytes = 0;
      t.at = alloc<uint8_t>(ring_bytes);
    } else {
      const size_t ring_vmm = round_up(ring_vmm_bytes, v.gran);
      const size_t va_sz = dev + ring_vmm;
      CUdeviceptr va = 0;
      check(v.reserve(&va, va_sz, v.gran, 0, 0), "reserving addresses for a pageable ring");
      m.va = va;
      m.va_bytes = va_sz;
      CUmemGenericAllocationHandle hd = 0, hr = 0;
      check(v.create(&hd, dev, &dp, 0), "allocating device memory");
      m.handles[0] = hd;
      check(v.map(va, dev, 0, hd, 0), "mapping device memory");
      check(v.map(vw, dev, 0, hd, 0), "mapping a view's device memory");
      check(v.create(&hr, ring_vmm, &dp, 0), "allocating device tail ring");
      m.handles[1] = hr;
      check(v.map(va + dev, ring_vmm, 0, hr, 0), "mapping device tail ring");
      check(v.set_access(va, va_sz, &access, 1), "granting the device access");
      CK(cudaMemsetAsync(reinterpret_cast<void*>(va), 0, dev + ring_vmm_bytes, st));
      t.at = reinterpret_cast<void*>(va);
    }
    t.view = reinterpret_cast<void*>(vw);
    return t;
  }

  device_positions = std::min(device_positions, t.device_rows);  // what every cache keeps on the device, at most
  if (dry) {
    dry_bytes += dev;
    dry_host_bytes += host;
    dry_mapped_bytes += host;
    t.at = t.view = reinterpret_cast<void*>(uintptr_t(256));
    return t;
  }
  CUmemAllocationProp dp{}, hp{};
  dp.type = hp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  hp.location = {CU_MEM_LOCATION_TYPE_HOST, 0};
  CUmemAccessDesc access{};
  access.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  mapped.push_back(Mapped{});
  Mapped& m = mapped.back();
  m.dev = dev;
  m.bytes = total;
  m.va_bytes = total;
  m.stage = stage;
  m.slot = slot;
  CUdeviceptr va = 0, vw = 0;
  check(v.reserve(&va, total, v.gran, 0, 0), "reserving addresses");
  m.va = va;
  check(v.reserve(&vw, total, v.gran, 0, 0), "reserving addresses for a view");
  m.view = vw;
  CUmemGenericAllocationHandle hd = 0, hh = 0;
  if (dev) {
    check(v.create(&hd, dev, &dp, 0), "allocating device memory");
    m.handles[0] = hd;
    check(v.map(va, dev, 0, hd, 0), "mapping device memory");
    check(v.map(vw, dev, 0, hd, 0), "mapping a view's device memory");
  }
  check(v.create(&hh, host, &hp, 0), "allocating host memory");
  m.handles[1] = hh;
  check(v.map(va + dev, host, 0, hh, 0), "mapping host memory");
  check(v.set_access(va, total, &access, 1), "granting the device access");
  // the view's staging, and its access, once the pool is there (pool_finish)
  CK(cudaMemsetAsync(reinterpret_cast<void*>(va), 0, want, st));  // as alloc's
  t.at = reinterpret_cast<void*>(va);
  t.view = reinterpret_cast<void*>(vw);
  return t;
}

void flyx_seq::pool_finish() {
  if (staging.empty()) return;  // no cache in tiers: the copy, if any, has only its device home
  const Vmm& v = vmm();
  if (!v.missing.empty()) throw std::runtime_error("K/V in host memory: " + v.missing);
  // The copies in host memory read through stagings after the copies on the device: the K/V stagings
  // and (when available) the model's VMM prefill slab share the tail of the pool.  During prefill, K/V stagings occupy
  // [owned_pool_bytes - staging_bytes, owned_pool_bytes) and prefill_bufs occupies [owned_pool_bytes, pool_bytes),
  // disjoint from each other; during strata-only decode both are idle and the whole pool holds resident copies + slots.
  copy_slot_bytes = 0;
  for (const flyx::cuda::KvBuf& b : kv)  // its codes and its grids
    if (b.copy_slot >= 0) copy_slot_bytes = std::max(copy_slot_bytes, 2 * b.q_span() + 2 * b.grid_span());
  copy_stage_off = copy_bytes;
  copy_bytes += size_t(copy_slots) * copy_slot_bytes;
  const size_t copy_need = round_up(copy_bytes, v.gran);
  const std::shared_ptr<flyx::cuda::VmmSlab> slab = m ? m->prefill_slab : nullptr;
  const size_t prefill_avail = slab ? slab->bytes : 0;
  if (copy_need > staging_bytes && prefill_avail > 0) {
    prefill_shared = std::min(prefill_avail, copy_need - staging_bytes);
    owned_pool_bytes = copy_need - prefill_shared;  // >= staging_bytes
    pool_bytes = owned_pool_bytes + prefill_shared;
  } else {
    prefill_shared = 0;
    owned_pool_bytes = std::max(staging_bytes, copy_need);
    pool_bytes = owned_pool_bytes;
  }
  // Resident copies reaching into [staged_from, pool_bytes) (the K/V stagings or the shared prefill slab) keep a host
  // home so they can park there during prefill / exact rounds and DMA back on strata decode.
  const size_t staged_from = owned_pool_bytes - staging_bytes;
  for (flyx::cuda::KvBuf& b : kv)
    if (b.copy_slot >= 0) {  // in host memory for good
      b.kq = b.kq_host = host_alloc<uint32_t>(b.q_bytes / sizeof(uint32_t));
      b.vq = b.vq_host = host_alloc<uint32_t>(b.q_bytes / sizeof(uint32_t));
    } else if (b.q_bytes && b.copy_off + 2 * b.q_span() > staged_from) {
      b.kq_host = host_alloc<uint32_t>(b.q_bytes / sizeof(uint32_t));
      b.vq_host = host_alloc<uint32_t>(b.q_bytes / sizeof(uint32_t));
      b.kq = b.kq_host;
      b.vq = b.vq_host;
      copy_pooled = true;
    }
  if (dry) {
    dry_bytes += owned_pool_bytes;  // prefill_shared is already accounted in model workspace (prefill_bytes)
    return;
  }
  CUmemAllocationProp dp{};
  dp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  dp.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  CUmemAccessDesc access{};
  access.location = {CU_MEM_LOCATION_TYPE_DEVICE, v.device};
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  // The pool in pieces of the granularity: owned_pieces first, then shared prefill pieces at the tail.
  const size_t owned_pieces = owned_pool_bytes / v.gran;
  const size_t shared_pieces = prefill_shared / v.gran;
  pool_pieces.reserve(owned_pieces);
  CUdeviceptr va = 0;
  check(v.reserve(&va, pool_bytes, v.gran, 0, 0), "reserving the stagings' addresses");
  pool_va = va;
  for (size_t i = 0; i < owned_pieces; ++i) {
    CUmemGenericAllocationHandle h = 0;
    check(v.create(&h, v.gran, &dp, 0), "allocating the stagings");
    pool_pieces.push_back(h);
    check(v.map(va + i * v.gran, v.gran, 0, h, 0), "mapping the stagings");
  }
  if (shared_pieces > 0) {
    shared_prefill = slab;
    for (size_t i = 0; i < shared_pieces; ++i)
      check(v.map(va + (owned_pieces + i) * v.gran, v.gran, 0,
                  CUmemGenericAllocationHandle(shared_prefill->handles[i]), 0),
            "mapping shared prefill memory");
  }
  check(v.set_access(va, pool_bytes, &access, 1), "granting the device access to the stagings");
  for (Staging& x : staging) x.offset += owned_pool_bytes - staging_bytes;  // stagings at the end of owned_pool_bytes
  for (const Mapped& m : mapped) {
    if (!m.view) continue;
    const size_t first = staging[size_t(m.slot)].offset / v.gran;
    for (size_t i = 0; i < m.stage / v.gran; ++i)
      check(v.map(CUdeviceptr(m.view) + m.dev + i * v.gran, v.gran, 0,
                  CUmemGenericAllocationHandle(pool_pieces[first + i]), 0),
            "mapping a view's staging");
    check(v.set_access(CUdeviceptr(m.view), m.dev + m.stage, &access, 1), "granting the device access to a view");
  }
  // The stagings of each parity a prefill's windows of a cache's keys take: its role's and size's.
  for (flyx::cuda::KvBuf& b : kv) {
    if (!b.vk) continue;
    const Staging& own = staging[size_t(b.slot)];
    for (size_t i = 0; i < staging.size(); ++i) {
      const Staging& x = staging[i];
      if (x.bytes != own.bytes || x.row != own.row) continue;
      uint8_t* at = reinterpret_cast<uint8_t*>(pool_va + x.offset);
      if (x.role == 0) b.wslot[x.parity] = int(i), b.wk[x.parity] = at, b.wks[x.parity] = x.scales;
      else b.wv[x.parity] = at, b.wvs[x.parity] = x.scales;
    }
  }
  for (flyx::cuda::KvBuf& b : kv)
    if (b.copy_slot >= 0) {
      uint8_t* at = reinterpret_cast<uint8_t*>(pool_va + copy_stage_off + size_t(b.copy_slot) * copy_slot_bytes);
      b.kq_stage = reinterpret_cast<uint32_t*>(at);
      b.vq_stage = reinterpret_cast<uint32_t*>(at + b.q_span());
      b.kqs_stage = reinterpret_cast<__half2*>(at + 2 * b.q_span());
      b.vqs_stage = reinterpret_cast<__half2*>(at + 2 * b.q_span() + b.grid_span());
    } else if (b.q_bytes) {  // the copy at the pool's start
      b.kq_dev = reinterpret_cast<uint32_t*>(pool_va + b.copy_off);
      b.vq_dev = reinterpret_cast<uint32_t*>(pool_va + b.copy_off + b.q_span());
      if (!b.kq_host) b.kq = b.kq_dev, b.vq = b.vq_dev;  // never reached by the stagings: always there
    }
}

void flyx_seq::copy_home(bool dev) {
  if ((!copy_pooled && prefill_shared == 0) || copy_dev == dev) return;
  if (copy_pooled) {
    // The positions rounds in flight may have written are below write_hi (atomic for cross-thread eviction).
    const int hi = std::min(cap, write_hi.load(std::memory_order_acquire));
    const int lo = std::min(copy_dirty_from, hi);
    auto move = [&](uint32_t* host, uint32_t* devp, size_t from, size_t to) {  // bytes [from, to) of a part
      if (from >= to) return;
      void* d = dev ? static_cast<void*>(reinterpret_cast<uint8_t*>(devp) + from)
                    : static_cast<void*>(reinterpret_cast<uint8_t*>(host) + from);
      const void* src = dev ? static_cast<const void*>(reinterpret_cast<const uint8_t*>(host) + from)
                            : static_cast<const void*>(reinterpret_cast<const uint8_t*>(devp) + from);
      CK(cudaMemcpyAsync(d, src, to - from, cudaMemcpyDefault, st));
    };
    for (flyx::cuda::KvBuf& b : kv) {
      if (!b.kq_host || b.copy_slot >= 0) continue;  // a copy in host memory for good stays there
      // the 16s of positions written since the copy last moved
      const size_t f = size_t(lo / 16) * b.q_group, t = std::min(b.q_bytes, size_t((hi + 15) / 16) * b.q_group);
      move(b.kq_host, b.kq_dev, f, t);
      move(b.vq_host, b.vq_dev, f, t);
      if (!dev) continue;
      // Restore intervals clobbered by stagings or the shared prefill slab, skipping [f, t) already copied above.
      auto restore_range = [&](size_t s0, size_t s1) {
        for (int part = 0; part < 2; ++part) {
          const size_t p0 = b.copy_off + size_t(part) * b.q_span(), p1 = p0 + t;
          const size_t a = std::max(s0, p0), z = std::min(s1, p1);
          if (a >= z) continue;
          const size_t u = a - p0, v_end = z - p0;
          uint32_t* h = part ? b.vq_host : b.kq_host;
          uint32_t* dp = part ? b.vq_dev : b.kq_dev;
          if (u < f) move(h, dp, u, std::min(v_end, f));
          if (v_end > t) move(h, dp, std::max(u, t), v_end);
        }
      };
      for (const Staging& x : staging) {
        const size_t rows = staged_to > x.device_rows ? size_t(staged_to - x.device_rows) : 0;
        const size_t s0 = x.offset, s1 = x.offset + std::min(x.bytes, rows * x.row);
        if (s0 < s1) restore_range(s0, s1);
      }
      if (prefill_clobbered && prefill_shared > 0)
        restore_range(owned_pool_bytes, owned_pool_bytes + prefill_shared);
    }
    for (flyx::cuda::KvBuf& b : kv)
      if (b.kq_host && b.copy_slot < 0) {
        b.kq = dev ? b.kq_dev : b.kq_host;
        b.vq = dev ? b.vq_dev : b.vq_host;
      }
  }
  copy_dev = dev;
  copy_dirty_from = INT32_MAX;
  if (dev) {
    staged_to = 0;
    prefill_clobbered = false;
  }
}

// The stage stream runs at the device's highest priority: its few blocks start as soon as a block of the pass ends,
// and then keep the bus busy while the pass's kernels take the rest of the GPU.
void flyx_seq::create_stage_stream() {
  if (dry || staging.empty() || stage_side.st) return;
  int least = 0, greatest = 0;
  CK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
  CK(cudaStreamCreateWithPriority(&stage_side.st, cudaStreamNonBlocking, greatest));
  CK(cudaEventCreateWithFlags(&stage_side.fork, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&stage_side.join, cudaEventDisableTiming));
  for (auto* evs : {&stage_ready, &stage_free}) {
    const int early_events = evs == &stage_ready && copy_scan_end > 0 ? copy_slots : 0;
    evs->resize(staging.size() + size_t(copy_slots + early_events));
    for (auto& e : *evs) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  }
}

namespace {

constexpr int kStageBlocksAll = 256;  // a copy the attention waits for: as fast as the bus goes

}  // namespace

void flyx_seq::stage_begin(std::vector<StageItem> order, cudaStream_t s) {
  stage_on = false;
  if (!m->opt.kernels.stage_ahead || order.empty() || !stage_side.st) return;
  stage_order = std::move(order);
  stage_next = 0;
  stage_on = true;
  CK(cudaEventRecord(stage_side.fork, s));
  CK(cudaStreamWaitEvent(stage_side.st, stage_side.fork, 0));
  std::vector<bool> started(stage_ready.size());
  for (const StageItem& x : stage_order) {
    const size_t e = stage_event(x);
    if (started[e]) continue;
    started[e] = true;
    stage_ahead(x);
    CK(cudaEventRecord(stage_ready[e], stage_side.st));
  }
}

// What the item's attention reads that was written before the pass, on the stage stream.
void flyx_seq::stage_ahead(const StageItem& x) {
  const KvBuf& b = kv[size_t(x.kv)];
  const int blocks = m->opt.kernels.stage_blocks;
  if (x.copy) {
    if (x.from > 0) {
      auto move = [&](int from, int to) {
        const size_t code = size_t(from / 16) * b.q_group / sizeof(uint32_t);
        const size_t grid = size_t(from) * b.nkv;
        const size_t codes = size_t((to - from) / 16) * b.q_group;
        const size_t grids = size_t(to - from) * b.nkv * sizeof(__half2);
        CK(cudaMemcpyAsync(b.kq_stage + code, b.kq_host + code, codes, cudaMemcpyHostToDevice, stage_side.st));
        CK(cudaMemcpyAsync(b.vq_stage + code, b.vq_host + code, codes, cudaMemcpyHostToDevice, stage_side.st));
        CK(cudaMemcpyAsync(b.kqs_stage + grid, b.kqs + grid, grids, cudaMemcpyHostToDevice, stage_side.st));
        CK(cudaMemcpyAsync(b.vqs_stage + grid, b.vqs + grid, grids, cudaMemcpyHostToDevice, stage_side.st));
      };
      if (x.to > 0) {
        move(0, x.to);
        CK(cudaEventRecord(stage_ready[stage_event(x) + copy_slots], stage_side.st));
      }
      move(x.to, x.from);
    } else {
      b.copy_stage(P, 0, kStageBefore, blocks, stage_side.st);
    }
  } else if (x.parity >= 0) {
    // Prefill starts with an idle sequence: this whole historical window is already written.
    if (b.pageable) {
      window_stage(b, x.from, x.to, x.parity, 0, kStageBefore, blocks, stage_side.st);
    } else if (m->opt.stage_dma && x.from >= b.device_positions && x.to <= position) {
      const size_t n = size_t(x.to - x.from), offset = size_t(x.from) * b.row;
      CK(cudaMemcpyAsync(b.wk[x.parity], static_cast<const uint8_t*>(b.k) + offset, n * b.row,
                         cudaMemcpyDefault, stage_side.st));
      CK(cudaMemcpyAsync(b.wv[x.parity], static_cast<const uint8_t*>(b.v) + offset, n * b.row,
                         cudaMemcpyDefault, stage_side.st));
      if (b.ks_host) {
        const size_t at = size_t(x.from) * b.nkv, bytes = n * b.nkv * sizeof(float);
        CK(cudaMemcpyAsync(b.wks[x.parity], b.ks_host + at, bytes, cudaMemcpyHostToDevice, stage_side.st));
        CK(cudaMemcpyAsync(b.wvs[x.parity], b.vs_host + at, bytes, cudaMemcpyHostToDevice, stage_side.st));
      }
    } else {
      window_stage(b, x.from, x.to, x.parity, 0, kStageBefore, blocks, stage_side.st);
    }
  } else {
    if (b.pageable) {
      const int h0 = b.device_positions;
      const int h1 = std::min(b.staged_cap(), position);
      if (h1 > h0) {
        const size_t n = size_t(h1 - h0);
        CK(cudaMemcpyAsync(static_cast<uint8_t*>(b.vk) + size_t(h0) * b.row, b.k_page, n * b.row,
                           cudaMemcpyHostToDevice, stage_side.st));
        CK(cudaMemcpyAsync(static_cast<uint8_t*>(b.vv) + size_t(h0) * b.row, b.v_page, n * b.row,
                           cudaMemcpyHostToDevice, stage_side.st));
        if (b.ks_page) {
          const size_t sb = n * size_t(b.nkv) * sizeof(float);
          CK(cudaMemcpyAsync(b.vks, b.ks_page, sb, cudaMemcpyHostToDevice, stage_side.st));
          CK(cudaMemcpyAsync(b.vvs, b.vs_page, sb, cudaMemcpyHostToDevice, stage_side.st));
        }
      }
    } else {
      b.stage(P, 0, 0, kStageBefore, blocks, stage_side.st);
    }
  }
}

namespace {

// A window's staging, addressed at its keys' own positions: key w0 at its start.
void* shifted(uint8_t* staging, int w0, size_t row) {
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(staging) - size_t(w0) * row);
}

}  // namespace

// The keys [w0, w1) of a prefill's window to that parity's stagings: those written before the pass
// (kStageBefore), the pass's own (kStagePass), or all of them (kStageAll).
void flyx_seq::window_stage(const KvBuf& b, int w0, int w1, int parity, int rows, KvStagePart part, int blocks,
                            cudaStream_t s) const {
  if (b.pageable) {
    if (part != kStagePass) {
      // Any sub-window prefix below device_positions (when device_positions isn't window-aligned) lives on GPU.
      if (w0 < b.device_positions) {
        const int d1 = std::min(w1, b.device_positions);
        if (d1 > w0) {
          const size_t n = size_t(d1 - w0);
          CK(cudaMemcpyAsync(b.wk[parity], static_cast<const uint8_t*>(b.k) + size_t(w0) * b.row, n * b.row,
                             cudaMemcpyDeviceToDevice, s));
          CK(cudaMemcpyAsync(b.wv[parity], static_cast<const uint8_t*>(b.v) + size_t(w0) * b.row, n * b.row,
                             cudaMemcpyDeviceToDevice, s));
          if (b.ks_page) {
            const size_t sb = n * size_t(b.nkv) * sizeof(float);
            CK(cudaMemcpyAsync(b.wks[parity], b.ks + size_t(w0) * b.nkv, sb, cudaMemcpyDeviceToDevice, s));
            CK(cudaMemcpyAsync(b.wvs[parity], b.vs + size_t(w0) * b.nkv, sb, cudaMemcpyDeviceToDevice, s));
          }
        }
      }
      // Historical host positions [max(w0, device_positions), min(w1, position)) live in pageable RAM.
      const int h0 = std::max(w0, b.device_positions);
      const int h1 = std::min(w1, position);
      if (h1 > h0) {
        const size_t n = size_t(h1 - h0);
        const size_t off_dst = size_t(h0 - w0);
        const size_t off_src = size_t(h0 - b.device_positions);
        CK(cudaMemcpyAsync(b.wk[parity] + off_dst * b.row, b.k_page + off_src * b.row, n * b.row,
                           cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(b.wv[parity] + off_dst * b.row, b.v_page + off_src * b.row, n * b.row,
                           cudaMemcpyHostToDevice, s));
        if (b.ks_page) {
          const size_t sb = n * size_t(b.nkv) * sizeof(float);
          CK(cudaMemcpyAsync(b.wks[parity] + off_dst * b.nkv, b.ks_page + off_src * b.nkv, sb,
                             cudaMemcpyHostToDevice, s));
          CK(cudaMemcpyAsync(b.wvs[parity] + off_dst * b.nkv, b.vs_page + off_src * b.nkv, sb,
                             cudaMemcpyHostToDevice, s));
        }
      }
    }
    if (part != kStageBefore) {
      KvStageScales sc;
      if (b.ks_page) sc = {b.ks, b.vs, b.scales_from(b.wks[parity], w0), b.scales_from(b.wvs[parity], w0), b.nkv};
      kv_stage_ring(b.k, shifted(b.wk[parity], w0, b.row), b.v, shifted(b.wv[parity], w0, b.row), b.row,
                    b.device_positions, b.page_ring_base, b.page_ring, w0, w1, P, 0, rows, kStagePass, blocks, s, sc);
    }
    return;
  }
  KvStageScales sc;  // and their scales
  if (b.ks_host) sc = {b.ks_host, b.vs_host, b.scales_from(b.wks[parity], w0), b.scales_from(b.wvs[parity], w0), b.nkv};
  kv_stage(b.k, shifted(b.wk[parity], w0, b.row), b.v, shifted(b.wv[parity], w0, b.row), b.row, w0, w1, P, 0, rows,
           part, blocks, s, 1, sc);
}

int flyx_seq::windows(const KvBuf& kb, int rows) const {
  if (!kb.vk || stage_window <= 0 || (!kb.pageable && rows <= FLYX_MAX_ROWS)) return 0;
  const long long end = (long long)position + rows;  // the pass's keys: [0, end)
  if (end <= kb.staged_cap()) return 0;               // the view holds them
  const int from = window_from(kb), len = window_len(kb);
  return int((end - from + len - 1) / len);
}

flyx::cuda::KvRef flyx_seq::window_before(const KvBuf& kb, int w0, int w1, int rows, cudaStream_t s) {
  int parity = 0;
  if (!stage_on) {  // copied now, one window after the other in the one staging
    window_stage(kb, w0, w1, parity, rows, kStageAll, kStageBlocksAll, s);
  } else {
    const StageItem* x = stage_next < stage_order.size() ? &stage_order[stage_next] : nullptr;
    if (!x || &kv[size_t(x->kv)] != &kb || x->copy || x->from != w0 || x->to != w1 || x->parity < 0)
      throw std::logic_error("cuda: a window of keys out of the order its copies were started in");
    parity = x->parity;
    CK(cudaStreamWaitEvent(s, stage_ready[stage_event(*x)], 0));
    if (w1 > position) window_stage(kb, w0, w1, parity, rows, kStagePass, kStageBlocksAll, s);
  }
  KvRef r{shifted(kb.wk[parity], w0, kb.row), shifted(kb.wv[parity], w0, kb.row), kb.ks, kb.vs};
  if (kb.ks_host || kb.ks_page) {  // the window's scales past the device part, staged with it
    r.ks_far = kb.scales_from(kb.wks[parity], w0), r.vs_far = kb.scales_from(kb.wvs[parity], w0);
    r.scale_split = kb.device_positions;
  }
  return r;
}

// Before the item's attention: its copy started ahead done (the attentions come in the order stage_begin was given).
void flyx_seq::stage_wait(const KvBuf& kb, bool copy, cudaStream_t s, bool first_segment) {
  if (stage_next >= stage_order.size() || &kv[size_t(stage_order[stage_next].kv)] != &kb ||
      stage_order[stage_next].copy != copy)
    throw std::logic_error("cuda: an attention out of the order its copies were started in");
  const size_t e = stage_event(stage_order[stage_next]) + (first_segment ? copy_slots : 0);
  CK(cudaStreamWaitEvent(s, stage_ready[e], 0));
}

void flyx_seq::stage_before(const KvBuf& kb, int pos_add, int span, cudaStream_t s) {
  if (!kb.vk) return;
  if (kb.pageable) {
    if (!stage_on) {
      const int h0 = kb.device_positions;
      const int h1 = std::min(kb.staged_cap(), position);
      if (h1 > h0) {
        const size_t n = size_t(h1 - h0);
        CK(cudaMemcpyAsync(static_cast<uint8_t*>(kb.vk) + size_t(h0) * kb.row, kb.k_page, n * kb.row,
                           cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(static_cast<uint8_t*>(kb.vv) + size_t(h0) * kb.row, kb.v_page, n * kb.row,
                           cudaMemcpyHostToDevice, s));
        if (kb.ks_page) {
          const size_t sb = n * size_t(kb.nkv) * sizeof(float);
          CK(cudaMemcpyAsync(kb.vks, kb.ks_page, sb, cudaMemcpyHostToDevice, s));
          CK(cudaMemcpyAsync(kb.vvs, kb.vs_page, sb, cudaMemcpyHostToDevice, s));
        }
      }
    } else {
      if (pos_add != 0) throw std::logic_error("cuda: a staged attention's rows past the pass's");
      stage_wait(kb, false, s);
    }
    KvStageScales sc;
    if (kb.ks_page)
      sc = {kb.ks, kb.vs, kb.scales_from(kb.vks, kb.device_positions), kb.scales_from(kb.vvs, kb.device_positions),
            kb.nkv};
    // Copy both any in-flight committed rows [position, P->position) and the pass's own rows from the GPU tail ring.
    kv_stage_ring(kb.k, kb.vk, kb.v, kb.vv, kb.row, kb.device_positions, kb.page_ring_base, kb.page_ring,
                  std::max(kb.device_positions, position), kb.staged_cap(), P, pos_add, span, kStageAll,
                  kStageBlocksAll, s, sc);
    return;
  }
  if (!stage_on) {
    kb.stage(P, pos_add, span, kStageAll, kStageBlocksAll, s);
    return;
  }
  if (pos_add != 0) throw std::logic_error("cuda: a staged attention's rows past the pass's");  // copies made at 0
  stage_wait(kb, false, s);
  kb.stage(P, pos_add, span, kStagePass, kStageBlocksAll, s);
}

cudaEvent_t flyx_seq::copy_before(const KvBuf& kb, int span, cudaStream_t s, bool scan_halves) {
  if (!kb.kq_stage) return nullptr;
  if (!stage_on) {
    kb.copy_stage(P, span, kStageAll, kStageBlocksAll, s);
    return nullptr;
  }
  const StageItem& x = stage_order.at(stage_next);
  if (scan_halves && x.to > 0) {
    // The tail and the committed DMA prefix start on separate 16-position tiles. The previous reader
    // of this slot and the current layer's writer precede the tail on this stream.
    kb.copy_stage(P, span, kStageAll, kStageBlocksAll, s, x.from);
    stage_wait(kb, true, s, true);
    return stage_ready[stage_event(x)];
  }
  stage_wait(kb, true, s);
  // The writer has finished. With a DMA prefix the unread history and the new rows now form one range.
  // Copy it here so successive H2D transfers on the stage stream have no kernel between them.
  const int prefix = x.from;
  kb.copy_stage(P, span, prefix ? kStageAll : kStagePass, kStageBlocksAll, s, prefix);
  return nullptr;
}

void flyx_seq::stage_after(const KvBuf& kb, cudaStream_t s) {
  if (!stage_on || stage_next >= stage_order.size() || &kv[size_t(stage_order[stage_next].kv)] != &kb) return;
  const size_t e = stage_event(stage_order[stage_next]);
  CK(cudaEventRecord(stage_free[e], s));
  for (size_t j = stage_next + 1; j < stage_order.size(); ++j) {
    if (stage_event(stage_order[j]) != e) continue;
    CK(cudaStreamWaitEvent(stage_side.st, stage_free[e], 0));
    stage_ahead(stage_order[j]);
    CK(cudaEventRecord(stage_ready[e], stage_side.st));
    break;
  }
  ++stage_next;
}

void flyx_seq::stage_text(const KvBuf& kb, cudaStream_t s) {
  if (!kb.vk) return;
  // After every attention of the round's order: no copy of the stage stream still writes a staging.
  if (stage_on && stage_next < stage_order.size())
    throw std::logic_error("cuda: the text staged for a pass before the round's attentions are done");
  // The whole text and the canvas's slots past it (DvRows::s0) go through the staging: one as large as the host part.
  // Strata attention's stagings of a fixed size hold fewer, but DV's trees never read through them
  // (the engine reads every position for trees).
  if (stage_window > 0) throw std::logic_error("cuda: DV's canvas pass over stagings of a fixed size (strata's)");
  if (kb.pageable) {
    const int h0 = kb.device_positions;
    const int h1 = std::min(kb.staged_cap(), position);
    if (h1 > h0) {
      const size_t n = size_t(h1 - h0);
      CK(cudaMemcpyAsync(static_cast<uint8_t*>(kb.vk) + size_t(h0) * kb.row, kb.k_page, n * kb.row,
                         cudaMemcpyHostToDevice, s));
      CK(cudaMemcpyAsync(static_cast<uint8_t*>(kb.vv) + size_t(h0) * kb.row, kb.v_page, n * kb.row,
                         cudaMemcpyHostToDevice, s));
      if (kb.ks_page) {
        const size_t sb = n * size_t(kb.nkv) * sizeof(float);
        CK(cudaMemcpyAsync(kb.vks, kb.ks_page, sb, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(kb.vvs, kb.vs_page, sb, cudaMemcpyHostToDevice, s));
      }
    }
    KvStageScales sc;
    if (kb.ks_page)
      sc = {kb.ks, kb.vs, kb.scales_from(kb.vks, kb.device_positions), kb.scales_from(kb.vvs, kb.device_positions),
            kb.nkv};
    kv_stage_ring(kb.k, kb.vk, kb.v, kb.vv, kb.row, kb.device_positions, kb.page_ring_base, kb.page_ring,
                  std::max(kb.device_positions, position), kb.staged_cap(), P, 0, 0, kStageAll, kStageBlocksAll, s, sc);
    return;
  }
  kb.stage(P, 0, 0, kStageAll, kStageBlocksAll, s);
}

void flyx_seq::stage_end(cudaStream_t s) {
  if (!stage_on) return;
  stage_on = false;
  CK(cudaEventRecord(stage_side.join, stage_side.st));
  CK(cudaStreamWaitEvent(s, stage_side.join, 0));
}

void flyx_seq::release_prefill_pool() {
  if (!m || !shared_prefill) return;
  auto* bm = const_cast<flyx_bmodel*>(m);
  std::lock_guard<std::mutex> lock(bm->prefill_mu);
  if (bm->prefill_owner == this) bm->prefill_owner = nullptr;
}

void flyx_seq::flush_page_prefill(int start, int n) {
  if (page_layers.empty() || n <= 0) return;
  for (int idx : page_layers) {
    const KvBuf& b = kv[size_t(idx)];
    const int lo = std::max(b.device_positions, start);
    const int hi = start + n;
    if (lo >= hi) continue;
    const int base = b.page_ring_base;
    const int tail = b.page_ring - base;
    for (int p = lo; p < hi;) {
      const int slot = base + (p - b.device_positions) % tail;
      const int chunk = std::min(hi - p, b.page_ring - slot);
      const size_t dst = size_t(p - b.device_positions);
      CK(cudaMemcpyAsync(b.k_page + dst * b.row, static_cast<const uint8_t*>(b.k) + size_t(slot) * b.row,
                         size_t(chunk) * b.row, cudaMemcpyDeviceToHost, st));
      CK(cudaMemcpyAsync(b.v_page + dst * b.row, static_cast<const uint8_t*>(b.v) + size_t(slot) * b.row,
                         size_t(chunk) * b.row, cudaMemcpyDeviceToHost, st));
      if (b.ks_page) {
        const size_t sb = size_t(chunk) * size_t(b.nkv) * sizeof(float);
        CK(cudaMemcpyAsync(b.ks_page + dst * b.nkv, b.ks + size_t(slot) * b.nkv, sb, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(b.vs_page + dst * b.nkv, b.vs + size_t(slot) * b.nkv, sb, cudaMemcpyDeviceToHost, st));
      }
      p += chunk;
    }
  }
}

void flyx_seq::drain_page_host(unsigned long long seq, int pos, int n_tokens) {
  if (page_layers.empty() || n_tokens <= 0) return;
  const int rslot = int(seq % kRing);
  const int max_rows = m->wide_rows;
  for (size_t li = 0; li < page_layers.size(); ++li) {
    const KvBuf& b = kv[size_t(page_layers[li])];
    const PageDrainLayer& dl = page_drain_host[li];
    for (int r = 0; r < n_tokens; ++r) {
      const int p = pos - n_tokens + r;
      if (p < b.device_positions) continue;
      const size_t dst = size_t(p - b.device_positions);
      const size_t src_row = size_t(rslot) * size_t(max_rows) + size_t(r);
      std::memcpy(b.k_page + dst * b.row, dl.drain_k_host + src_row * b.row, b.row);
      std::memcpy(b.v_page + dst * b.row, dl.drain_v_host + src_row * b.row, b.row);
      if (b.ks_page) {
        const size_t sb = size_t(b.nkv) * sizeof(float);
        std::memcpy(b.ks_page + dst * b.nkv, dl.drain_ks_host + src_row * b.nkv, sb);
        std::memcpy(b.vs_page + dst * b.nkv, dl.drain_vs_host + src_row * b.nkv, sb);
      }
    }
  }
}

void flyx_seq::reload_page_ring(int pos) {
  if (page_layers.empty() || pos <= 0) return;
  for (int idx : page_layers) {
    const KvBuf& b = kv[size_t(idx)];
    const int base = b.page_ring_base;
    const int tail = b.page_ring - base;
    const int lo = std::max(b.device_positions, pos - tail);
    const int hi = pos;
    if (lo >= hi) continue;
    for (int p = lo; p < hi;) {
      const int slot = base + (p - b.device_positions) % tail;
      const int chunk = std::min(hi - p, b.page_ring - slot);
      const size_t src = size_t(p - b.device_positions);
      CK(cudaMemcpyAsync(static_cast<uint8_t*>(b.k) + size_t(slot) * b.row, b.k_page + src * b.row,
                         size_t(chunk) * b.row, cudaMemcpyHostToDevice, st));
      CK(cudaMemcpyAsync(static_cast<uint8_t*>(b.v) + size_t(slot) * b.row, b.v_page + src * b.row,
                         size_t(chunk) * b.row, cudaMemcpyHostToDevice, st));
      if (b.ks_page) {
        const size_t sb = size_t(chunk) * size_t(b.nkv) * sizeof(float);
        CK(cudaMemcpyAsync(b.ks + size_t(slot) * b.nkv, b.ks_page + src * b.nkv, sb, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(b.vs + size_t(slot) * b.nkv, b.vs_page + src * b.nkv, sb, cudaMemcpyHostToDevice, st));
      }
      p += chunk;
    }
  }
}

void flyx_seq::release_mapped() {
  if (st) cudaStreamSynchronize(st);
  if (!mapped.empty() || !staging.empty()) {
    const Vmm& v = vmm();
    for (const Mapped& m : mapped) {
      if (m.va) {
        if (m.dev) v.unmap(CUdeviceptr(m.va), m.dev);
        if (m.va_bytes > m.dev) v.unmap(CUdeviceptr(m.va) + m.dev, m.va_bytes - m.dev);
        v.address_free(CUdeviceptr(m.va), m.va_bytes);
      }
      if (m.view) {
        if (m.dev) v.unmap(CUdeviceptr(m.view), m.dev);
        for (size_t o = m.dev; o < m.dev + m.stage; o += v.gran) v.unmap(CUdeviceptr(m.view) + o, v.gran);
        v.address_free(CUdeviceptr(m.view), m.bytes);
      }
    }
    for (const Mapped& m : mapped)
      for (auto h : m.handles)
        if (h) v.release(CUmemGenericAllocationHandle(h));
    if (pool_va) {
      for (size_t o = 0; o < pool_bytes; o += v.gran) v.unmap(CUdeviceptr(pool_va) + o, v.gran);
      v.address_free(CUdeviceptr(pool_va), pool_bytes);
    }
    for (unsigned long long h : pool_pieces) v.release(CUmemGenericAllocationHandle(h));
    pool_pieces.clear();
    shared_prefill.reset();
    pool_va = 0;
    mapped.clear();
    staging.clear();
  }
  for (const auto& [p, bytes] : pageable_allocs) {
#if defined(_WIN32)
    (void)bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
  }
  pageable_allocs.clear();
}
