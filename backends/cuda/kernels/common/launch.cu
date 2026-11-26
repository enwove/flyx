// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Options and scratch of a launch (common/launch.h).
#include "backends/cuda/kernels/common/launch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace flyx::cuda {

// ---- scratch and options ----------------------------------------------------------------
std::atomic<bool>& stream_pdl() {
  static std::atomic<bool> on{false};
  return on;
}
void set_stream_pdl(bool on) { stream_pdl().store(on, std::memory_order_relaxed); }

const KernelOptions& opts(const Launch& L) {
  static const KernelOptions defaults;
  return L.opt ? *L.opt : defaults;
}

const Scratch& scratch(const Launch& L) {
  if (!L.scratch) {
    std::fprintf(stderr, "flyx-cuda: a GEMM input was to be staged without scratch\n");
    std::abort();
  }
  return *L.scratch;
}

namespace {
// The 1..16 rows of a GEMM on block weights, in pairs of rows (gemm_blocks.cu): up to 8 rows take twice their
// values (the absent partner of each is zeros); more take 16 rows' worth.
size_t pairs_values(int rows, int k_max) { return size_t(rows < 8 ? 2 * rows : 16) * k_max; }
// A prefill GEMM's runs of K side by side: 24 MB of their sums (the 4B's 2560-column layers at 512 rows in four runs
// take 21 MB; a pass that needs more folds its runs onto y in order, fold_runs), and the tiles' counters.
constexpr size_t kQ8PartsFloats = size_t(6) << 20;
// A round's GEMM whose streams are split among blocks: the most columns a launch takes (a wider layer, an output head,
// takes slices of them in turn): the 27B's MLP gate and up in one run, 34816.
constexpr int kRoundPartsColumns = 36864;
constexpr uintptr_t kScratchAlign = 256;

// Buffers one after another from an address, each at its alignment.
struct Carve {
  uintptr_t at;
  template <class T>
  T* take(size_t n, uintptr_t align = alignof(T)) {
    at = (at + align - 1) / align * align;
    T* p = reinterpret_cast<T*>(at);
    at += n * sizeof(T);
    return p;
  }
};

// The scratch's layout from `base` (a multiple of kScratchAlign): the one description that both sizes the scratch
// (from 0) and carves it.  Returns where it ends.
// Counters stay dedicated (kernels expect them zeroed between launches); the active GEMM staging areas are mutually
// exclusive per launch (1..16-row round staging vs >16-row 8-bit block GEMMs vs >16-row bf16/NVFP4/non-Q8 GEMMs) and
// share one union base.
uintptr_t lay_out(uintptr_t base, int rows, int k_max, bool q8_parts, bool f32_rows, bool need_glu, bool need_q4,
                  Scratch* s) {
  Carve c{base};
  // Dedicated zeroed tile counters and per-row scales (tiny, shared across paths).
  s->row_scale = c.take<float>(size_t(std::max(rows, 16) + 3) / 4 * 4, 16);
  // Shared by split Q8 round GEMMs (k_round_blocks_split) and fused relative-attention round combine (k_rel_round_gqa).
  s->round_count_cap = kRoundPartsColumns / 8;
  s->round_count = c.take<int>(s->round_count_cap, 16);
  if (rows > 16 && q8_parts) s->q8_count = c.take<int>(kQ8Tiles, 16);
  if (q8_parts) {
    s->side_count_cap = kSideColumns / 8;
    s->side_count = c.take<int>(s->side_count_cap, 16);
  }
  const uintptr_t work = (c.at + kScratchAlign - 1) / kScratchAlign * kScratchAlign;

  // Branch 1: 8-bit block-weight staging (q8, q8_scale) followed by either >16-row q8_parts OR <=16-row round staging
  // (split, pairs, q8_sums, q8_glu, round_parts, side_parts).  A single launch never runs both >16 and <=16 rows.
  Carve cq{work};
  const size_t q8 = size_t(std::max(rows, 16)) * k_max;
  s->q8 = cq.take<int8_t>(q8, 256);
  s->q8_scale = cq.take<float>(q8 / 32, 16);  // 16-byte aligned: k_round_blocks_warps reads 4 groups' at once
  s->q8_round_cap = size_t(16) * k_max;
  const uintptr_t q_tail = (cq.at + kScratchAlign - 1) / kScratchAlign * kScratchAlign;

  Carve cb{q_tail};
  if (rows > 16 && q8_parts) {
    s->q8_parts_cap = kQ8PartsFloats;
    s->q8_parts = cb.take<float>(kQ8PartsFloats, 256);
  }

  Carve ca{q_tail};
  s->split_cap = size_t(std::min(rows, 16)) * k_max;
  s->split = ca.take<bf16>(3 * s->split_cap, 256);
  s->pairs_cap = pairs_values(rows, k_max);
  s->pairs = ca.take<uint16_t>(s->pairs_cap, 256);
  s->q8_sums = ca.take<float>(size_t(k_max), 16);  // a round's: 16 rows, K / 16 each
  // a round's SwiGLU in 8 bits, beside the gate|up GEMM's input (Launch::glu)
  s->q8_glu_scale = ca.take<float>(size_t(16) * k_max / 32, 256);
  s->q8_glu_sums = ca.take<float>(size_t(k_max), 256);
  s->q8_glu = ca.take<int8_t>(size_t(16) * k_max, 256);
  if (q8_parts) {  // a round's split streams (k_round_blocks_split): 4 sets of 16 rows, kRoundPartsColumns a launch
    s->round_parts_cap = size_t(4) * 16 * kRoundPartsColumns;
    s->round_parts = ca.take<float>(s->round_parts_cap, 256);
    s->side_parts_cap = size_t(4) * 16 * kSideColumns;  // a launch on the side stream (side_scratch)
    s->side_parts = ca.take<float>(s->side_parts_cap, 256);
  }
  uintptr_t end = std::max(cb.at, ca.at);

  // Branch 2: >16-row bf16 / NVFP4 / non-Q8 block staging (mutually exclusive with Branch 1).
  s->abuf_cap = rows > 16 ? size_t(rows) * k_max : 0;
  if (s->abuf_cap) {
    Carve cc{work};
    s->abuf = cc.take<bf16>(s->abuf_cap, 256);
    const uintptr_t c_tail = (cc.at + kScratchAlign - 1) / kScratchAlign * kScratchAlign;
    // Non-Q8 GGML block weights use run_sums; bf16/NVFP4 use glu/q4/af32/glu_amax (never together).
    Carve cblk{c_tail};
    s->run_sums = cblk.take<float>(s->abuf_cap / 16, 16);
    Carve cfp{c_tail};
    if (need_q4 || f32_rows) {
      // prefill rows in NVFP4 (gemm_w4a4): K / 2 code bytes, K / 16 scale bytes and a float per row
      s->q4 = cfp.take<uint8_t>(s->abuf_cap / 2, 256);
      s->q4_scale = cfp.take<uint8_t>(s->abuf_cap / 16, 256);
      s->q4_inv = cfp.take<float>(size_t(rows), 256);
    }
    if (need_glu || f32_rows) {
      // SwiGLU rows (gemm_glu_staged), I <= k_max
      s->glu_cap = s->abuf_cap;
      s->glu = cfp.take<bf16>(s->glu_cap, 256);
    }
    if (f32_rows) {  // NVFP4 inputs rounded once from float32
      s->af32 = cfp.take<float>(s->abuf_cap, 256);
      s->glu_amax = cfp.take<float>(size_t(rows), 256);
    }
    end = std::max(end, std::max(cblk.at, cfp.at));
  }
  return end;
}
}  // namespace

size_t scratch_bytes(int rows, int k_max, bool q8_parts, bool f32_rows, bool need_glu, bool need_q4) {
  Scratch s;
  return lay_out(0, rows, k_max, q8_parts, f32_rows, need_glu, need_q4, &s);
}

Scratch scratch_carve(void* mem, size_t bytes, int rows, int k_max, bool q8_parts, bool f32_rows, bool need_glu,
                      bool need_q4) {
  Scratch s;
  const uintptr_t base = reinterpret_cast<uintptr_t>(mem);
  const uintptr_t end = lay_out(base, rows, k_max, q8_parts, f32_rows, need_glu, need_q4, &s);
  if (base % kScratchAlign || end - base > bytes) {  // a sizing bug in the caller, not a condition to recover from
    std::fprintf(stderr, "flyx-cuda: scratch for %d rows of %d needs %zu bytes from a %zu-byte boundary, has %zu\n",
                 rows, k_max, size_t(end - base), size_t(kScratchAlign), bytes);
    std::abort();
  }
  return s;
}

Scratch scratch_in(void* mem, size_t bytes, int rows, int k_max, bool q8_parts, bool f32_rows) {
  Scratch s = scratch_carve(mem, bytes, rows, k_max, q8_parts, f32_rows);
  // The tiles' counters start at zero; every GEMM leaves them so.  Waited for: callers launch on streams that do
  // not wait for the default one.
  if ((s.q8_count && cudaMemset(s.q8_count, 0, kQ8Tiles * sizeof(int)) != cudaSuccess) ||
      (s.round_count && cudaMemset(s.round_count, 0, s.round_count_cap * sizeof(int)) != cudaSuccess) ||
      (s.side_count && cudaMemset(s.side_count, 0, s.side_count_cap * sizeof(int)) != cudaSuccess) ||
      ((s.q8_count || s.round_count) && cudaDeviceSynchronize() != cudaSuccess)) {
    std::fprintf(stderr, "flyx-cuda: scratch_in: %s\n", cudaGetErrorString(cudaGetLastError()));
    std::abort();
  }
  return s;
}

Scratch glu_inputs(const Scratch& s) {
  Scratch g = s;
  g.q8 = s.q8_glu;
  g.q8_scale = s.q8_glu_scale;
  g.q8_sums = s.q8_glu_sums;
  return g;
}

Scratch side_scratch(const Scratch& s) {
  Scratch g = s;
  g.round_parts = s.side_parts;
  g.round_parts_cap = s.side_parts_cap;
  g.round_count = s.side_count;
  g.round_count_cap = s.side_count_cap;
  return g;
}

// A staging request larger than the scratch is a sizing bug in the backend, not a condition to recover from.
void check_fits(size_t n, size_t cap, const char* what) {
  if (n <= cap) return;
  std::fprintf(stderr, "flyx-cuda: %s needs %zu elements of scratch, has %zu\n", what, n, cap);
  std::abort();
}

// Where a fused producer leaves `rows` rows of K values for gemm_presplit /
// gemm_fp4_presplit: split for the skinny GEMMs (up to 16 rows), else rounded
// to bf16 in the prefill GEMM's input buffer, or in float32 where the scratch
// has room for them (NVFP4 inputs: the W4A4 kernels round once);
// q8 (GGML block weights with 8-bit inputs): in 8 bits where the block
// kernels read them, a round's layout with its sums up to 16 rows, a prefill
// pass's beyond.
Out split_out(const Launch& L, int rows, int K, bool q8) {
  const Scratch& sc = scratch(L);
  if (q8) {
    check_fits(size_t(rows) * K, rows <= 16 ? sc.q8_round_cap : sc.abuf_cap, "a fused producer in 8 bits");
    Out o{nullptr, nullptr, nullptr, nullptr};
    o.q8 = sc.q8;
    o.q8_d = sc.q8_scale;
    o.q8_sums = rows <= 16 ? sc.q8_sums : nullptr;
    o.q8_k = K;
    return o;
  }
  if (rows <= 16) {
    check_fits(size_t(rows) * K, sc.split_cap, "a fused producer");
    return Out{nullptr, sc.split, sc.split + sc.split_cap, sc.split + 2 * sc.split_cap};
  }
  check_fits(size_t(rows) * K, sc.abuf_cap, "a fused producer");
  if (sc.af32) {
    Out o{nullptr, nullptr, nullptr, nullptr};
    o.f32 = sc.af32;
    return o;
  }
  return Out{nullptr, sc.abuf, nullptr, nullptr};
}

Out split_out_slice(const Launch& L, int total_rows, int row_offset, int K, bool q8) {
  Out o = split_out(L, total_rows, K, q8);
  if (row_offset > 0) {
    const size_t off = size_t(row_offset) * size_t(K);
    if (o.q8) {
      o.q8 += off;
      o.q8_d += off / 32;
      if (o.q8_sums) o.q8_sums += off / 16;
    } else if (o.f32) {
      o.f32 += off;
    } else if (o.hi) {
      o.hi += off;
      if (o.mid) {
        o.mid += off;
        o.lo += off;
      }
    }
  }
  return o;
}

namespace {
__global__ void k_compiled_for() {}
}  // namespace

// Thread block clusters and programmatic dependent launch: the device has them and the code that runs on it was
// compiled for them.  A build without sm_90 code runs on an H100 or a B200 from an older architecture's PTX,
// compiled when it loads (__CUDA_ARCH__ 890 for a build of 80;86;89;120): there the GDN round is a stub that traps
// (gdn.cu) and pdl_wait() waits for nothing (device.cuh), so neither may be launched.  Every kernel of the library
// is compiled for the same architectures, so one kernel's PTX version is theirs (llama.cpp checks its highest
// compiled architecture the same way, ggml_cuda_highest_compiled_arch).
int compiled_arch() {
  static const int arch = [] {
    cudaFuncAttributes a{};
    const bool ok = cudaFuncGetAttributes(&a, k_compiled_for) == cudaSuccess;
    cudaGetLastError();  // clear the error a missing kernel image leaves behind
    return ok ? a.ptxVersion : 0;
  }();
  return arch;
}

bool sm90_or_newer() {
  static const bool yes = [&] {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 9 && compiled_arch() >= 90;
  }();
  return yes;
}

int device_sm_count() {
  static const int sms = [] {
    int dev = 0, n = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    return n > 0 ? n : 1;
  }();
  return sms;
}

size_t device_max_smem_optin() {
  static const size_t limit = [] {
    int dev = 0, bytes = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    return bytes > 0 ? size_t(bytes) : size_t(48 * 1024);
  }();
  return limit;
}

namespace {

template <bool Aligned>
__global__ void k_rows_to_bf16(const float* __restrict__ x, int ldx, int K, int total, __nv_bfloat162* __restrict__ out) {
  pdl_trigger();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // one pair
  if (i >= total) return;
  const int r = i / (K / 2), c = 2 * (i - r * (K / 2));
  if constexpr (Aligned) {
    const float2 v = *reinterpret_cast<const float2*>(x + size_t(r) * ldx + c);
    out[i] = __floats2bfloat162_rn(v.x, v.y);
  } else {
    out[i] = __floats2bfloat162_rn(x[size_t(r) * ldx + c], x[size_t(r) * ldx + c + 1]);
  }
}

}  // namespace

void rows_to_bf16(const float* x, int ldx, int rows, int K, bf16* out, cudaStream_t st) {
  const int total = int(size_t(rows) * K / 2);
  auto* dst = reinterpret_cast<__nv_bfloat162*>(out);
  const bool aligned = (ldx % 2 == 0) && (reinterpret_cast<uintptr_t>(x) % 8 == 0);
  if (aligned) k_rows_to_bf16<true><<<ceil_div(total, 256), 256, 0, st>>>(x, ldx, K, total, dst);
  else k_rows_to_bf16<false><<<ceil_div(total, 256), 256, 0, st>>>(x, ldx, K, total, dst);
}

}  // namespace flyx::cuda
