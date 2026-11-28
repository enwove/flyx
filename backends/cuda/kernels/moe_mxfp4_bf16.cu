// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// MXFP4 experts with bf16 inputs (FLYX_ACT_BF16, gpt-oss's experts as vLLM runs them) on the bf16 tensor cores:
// mma.sync m16n8k16, A the inputs rounded once to bf16 (k_moe_stage_mxfp4_bf16), B the experts' e2m1 codes times their
// blocks' scales (exact in bf16), float32 sums.
//
// Tasks of 16 or 32 rows of one expert (moe_mxfp4_bm: 32 where the experts average more than 14 rows), column sets of
// 8 columns of 4 warps; spans of 16 units of K (a unit a GGML block of 32 values), the rows, the weights and their
// scales staged in shared memory a span ahead (cp.async), warp wi of a set its units 4 wi .. 4 wi + 3 of each span:
// one float32 sum a lane, MMA after MMA in K's order, then the four warps' sums in a fixed tree.  Every row gets the
// same MMAs in the same order whatever other rows its task holds and whichever task size: the same bits at every row
// count (rounds and passes take these kernels alike).
//
// A unit's 16 bytes: byte b's low nibble value b, its high nibble value b + 16.  Lane t4 takes bytes 4 t4 .. 4 t4 + 3
// of its column's block, eight codes whose bf16 pairs (mxfp4x8_to_bf16) are values (4 t4 + i, 4 t4 + 16 + i),
// i = 0..3: B's pairs of K for two MMAs as they come.  The staged inputs hold each block's values in
// the same pairs (position 2 c and 2 c + 1: values c and c + 16), so a lane's 16 bytes of a row are A's pairs for those
// MMAs: a dot product of the same pairs.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_act.cuh"

namespace flyx::cuda {
namespace {

constexpr int kSetWarps = 4;

// Rows of float32 values as bf16 (nearest), each block of 32 values interleaved: positions 2 c, 2 c + 1 values c,
// c + 16.  A block a row and 256 threads; a thread a pair.
__global__ void __launch_bounds__(256) k_moe_stage_mxfp4_bf16(const float* x, int ldx, int K,
                                                          bf16* __restrict__ out) {
  pdl_trigger();
  pdl_wait();  // x: the kernel before
  const int r = int(blockIdx.x);
  const float* src = x + size_t(r) * ldx;
  uint32_t* dst = reinterpret_cast<uint32_t*>(out + size_t(r) * K);
  for (int i = threadIdx.x; i < (K >> 1); i += blockDim.x) {
    const int b = i >> 4, c = i & 15;
    const __nv_bfloat162 v = __floats2bfloat162_rn(src[32 * b + c], src[32 * b + c + 16]);
    dst[i] = *reinterpret_cast<const uint32_t*>(&v);
  }
}

// Eight e2m1 codes of a word (nibble i: value i) as bf16 pairs of twice their values (GGML's MXFP4 values: 2 e2m1),
// pair p values 2 p (low half), 2 p + 1: fp4x8_to_bf16 with tables of 0, 1, 2, 3, 4, 6, 8, 12.
__device__ __forceinline__ void mxfp4x8_to_bf16(uint32_t w, uint32_t out[4]) {
  constexpr uint32_t kHi0 = 0x40403F00u, kHi1 = 0x41414040u, kLo0 = 0x40008000u, kLo1 = 0x4000C080u;
  const uint32_t m = w & 0x77777777u;  // 3-bit selectors (bit 3 of a prmt selector would sign-extend)
  const uint32_t hb0 = __byte_perm(kHi0, kHi1, m & 0xFFFFu), lb0 = __byte_perm(kLo0, kLo1, m & 0xFFFFu);
  const uint32_t hb1 = __byte_perm(kHi0, kHi1, m >> 16), lb1 = __byte_perm(kLo0, kLo1, m >> 16);
  const uint32_t E = (w << 4) & 0x80808080u;  // byte b bit 7: sign of value 2b
  const uint32_t O = w & 0x80808080u;         // byte b bit 7: sign of value 2b + 1
  out[0] = __byte_perm(lb0, hb0, 0x5140) | (__byte_perm(E, O, 0x4000) & 0x80008000u);
  out[1] = __byte_perm(lb0, hb0, 0x7362) | (__byte_perm(E, O, 0x5010) & 0x80008000u);
  out[2] = __byte_perm(lb1, hb1, 0x5140) | (__byte_perm(E, O, 0x6020) & 0x80008000u);
  out[3] = __byte_perm(lb1, hb1, 0x7362) | (__byte_perm(E, O, 0x7030) & 0x80008000u);
}

// Spans of 16 units, the rows, the weights and their scales staged in shared memory a span ahead (cp.async): warp wi
// of a set takes units 4 wi .. 4 wi + 3 of each span.  BM rows a task (16, or 32: two tiles of 16 decoding each unit
// of weights once), SETS column sets of 8 columns a block (the bits do not depend on either: a column's MMAs and tree
// are the same).
constexpr int kSpan = 16;
constexpr int kRow = kSpan * 32 + 32;   // bf16 a row: rows 16 banks apart
constexpr int kWcol = kSpan * 16 + 16;  // bytes a column: 68 words, 4 banks apart (a warp's 8 columns x 4 lanes)
constexpr int kScol = kSpan + 4;        // scales a column (u16): 10 words
template <int BM, int SETS>
struct Smem {
  struct Stage {
    alignas(16) bf16 rows[BM][kRow];
    alignas(16) uint8_t w[8 * SETS][kWcol];
    alignas(16) uint16_t s[8 * SETS][kScol];
  };
  union {
    Stage st[2];
    float tree[SETS][kSetWarps][32][BM / 4];  // after a pass's spans
  };
  int pids[BM];
  int rids[BM];
};

template <bool FUSE_ACT, bool IS_DOWN, int BM, int SETS>
__global__ void __launch_bounds__(32 * kSetWarps * SETS)
    k_moe_mxfp4_bf16_mma(const bf16* xa, const uint8_t* __restrict__ W, const uint16_t* __restrict__ D,
                     const float* __restrict__ RS, int N_exp, int I_or_H, int K, const MoeTask* tasks,
                     const int* n_tasks, const int* pairs, int top_k, MoeAct mact,
                     float* __restrict__ out, int ld_out) {
  constexpr int T = BM / 16, NT = 32 * kSetWarps * SETS, NC = 8 * SETS;
  pdl_trigger();
  pdl_wait();  // the staged rows and the tasks: the kernels before
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<Smem<BM, SETS>*>(smem_raw);
  const int tid = threadIdx.x;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, wi = warp % kSetWarps, ct = warp / kSetWarps;
  const int g = lane >> 2, t4 = lane & 3;
  const int units = K >> 5, spans = (units + kSpan - 1) / kSpan;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = int(blockIdx.x) * NC;
  const float rs = RS ? RS[e] : 1.f;
  const bool two = T == 2 && nb > 16;    // the second tile holds rows
  const int rows_st = two ? 32 : 16;     // the rows staged

  // span j of the rows and of the block's columns starting at row0 of the expert (a short last span: its units alone,
  // the rest never read)
  auto stage = [&](int row0, int j) {
    auto& S = sm.st[j & 1];
    const int n_s = min(kSpan, units - kSpan * j);
    for (int i = tid; i < rows_st * 4 * kSpan; i += NT) {  // 16 bytes a thread
      const int r = i / (4 * kSpan), c = i % (4 * kSpan);
      const bool ok = r < nb && c < 4 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * (kSpan * 32) + 8 * c) : 0;
      cp_async16(&S.rows[r][8 * c], xa + src, ok);
    }
    for (int i = tid; i < NC * kSpan; i += NT) {  // a unit's 16 bytes a thread
      const int c = i / kSpan, u = i % kSpan;
      const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
      const bool ok = u < n_s;
      cp_async16(&S.w[c][16 * u], W + (r * size_t(units) + size_t(kSpan) * j + (ok ? u : 0)) * 16, ok);
    }
    if ((units & 1) == 0) {  // their scales two at a time (4 bytes, aligned where a row's units are even)
      for (int i = tid; i < NC * (kSpan / 2); i += NT) {
        const int c = i / (kSpan / 2), u = 2 * (i % (kSpan / 2));
        const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
        const bool ok = u < n_s;
        cp_async4(&S.s[c][u], D + r * size_t(units) + size_t(kSpan) * j + (ok ? u : 0), ok);
      }
    } else {
      for (int i = tid; i < NC * n_s; i += NT) {
        const int c = i / n_s, u = i - c * n_s;
        const size_t r = size_t(e) * N_exp + min(row0 + c0 + c, N_exp - 1);
        S.s[c][u] = D[r * size_t(units) + size_t(kSpan) * j + u];
      }
    }
    cp_async_commit();
  };
  float acc[T][4];
  auto mul = [&](int j) {
    const auto& S = sm.st[j & 1];
    const int cl = ct * 8 + g, n_w = min(4, units - kSpan * j - 4 * wi);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int ul = 4 * wi + s;
      if (s >= n_w) break;
      const uint32_t q = *reinterpret_cast<const uint32_t*>(&S.w[cl][16 * ul + 4 * t4]);
      // the stored scale 0.5 2^(e - 127) on both halves: times GGML's values, 2 e2m1
      const uint32_t sb = __byte_perm(uint32_t(S.s[cl][ul]), 0u, 0x1010);
      const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(&sb);
      uint32_t bw[4];
      mxfp4x8_to_bf16(q, bw);
#pragma unroll
      for (int i = 0; i < 4; ++i) bw[i] = bf16x2_mul(bw[i], s2);
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t == 1 && !two) break;
        const uint4 x0 = *reinterpret_cast<const uint4*>(&S.rows[16 * t + g][32 * ul + 8 * t4]);
        const uint4 x8 = *reinterpret_cast<const uint4*>(&S.rows[16 * t + g + 8][32 * ul + 8 * t4]);
        const uint32_t a0[4] = {x0.x, x8.x, x0.y, x8.y}, b0[2] = {bw[0], bw[1]};
        mma_bf16(acc[t], a0, b0);
        const uint32_t a1[4] = {x0.z, x8.z, x0.w, x8.w}, b1[2] = {bw[2], bw[3]};
        mma_bf16(acc[t], a1, b1);
      }
    }
  };
  auto pass = [&](int row0) {
#pragma unroll
    for (int t = 0; t < T; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
    stage(row0, 0);
    for (int j = 0; j < spans; ++j) {
      cp_async_wait<0>();
      __syncthreads();
      if (j + 1 < spans) stage(row0, j + 1);
      mul(j);
    }
    __syncthreads();  // the stages free: the tree overlays them
  };
  auto to_tree = [&] {
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int e4 = 0; e4 < 4; ++e4) sm.tree[ct][wi][lane][4 * t + e4] = acc[t][e4];
  };
  auto from_tree = [&](int k) {
    return ((sm.tree[ct][0][lane][k] + sm.tree[ct][1][lane][k]) + (sm.tree[ct][2][lane][k] + sm.tree[ct][3][lane][k])) *
           rs;
  };
  const int n0 = c0 + ct * 8;
  pass(0);
  to_tree();
  __syncthreads();
  float vg[4 * T];
  if (wi == 0)
#pragma unroll
    for (int k = 0; k < 4 * T; ++k) vg[k] = from_tree(k);
  if constexpr (FUSE_ACT) {
    __syncthreads();
    pass(I_or_H);
    to_tree();
    __syncthreads();
  }
  if (wi == 0) {
#pragma unroll
    for (int k = 0; k < 4 * T; ++k) {
      const int r = 16 * (k >> 2) + g + 8 * ((k & 3) >> 1), col = n0 + 2 * t4 + (k & 1);
      if (r >= nb || col >= N_work) continue;
      float v = vg[k];
      if constexpr (FUSE_ACT) v = moe_act_val(mact, v, from_tree(k), e, col, I_or_H);
      out[size_t(sm.pids[r]) * ld_out + col] = v;
    }
  }
}

// 16 rows a task: 8 column sets (1024 threads); 32: 6 (768 threads, the stages' larger rows in shared memory).
// gpt-oss-20b's experts on an RTX PRO 6000, us an expert GEMM (gate|up and down averaged), 16-row tasks / 32-row:
// 91 rows 172 / 181, 131 rows 199 / 188, 512 rows 526 / 435.
template <bool FUSE_ACT, bool IS_DOWN, int BM>
void launch(const bf16* xa, const ExpertMat& W, int N_work, int I_or_H, int K, const MoeScratch& sc, int max_tiles,
            int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  constexpr int SETS = BM == 32 ? 6 : 8;
  auto* kernel = k_moe_mxfp4_bf16_mma<FUSE_ACT, IS_DOWN, BM, SETS>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(Smem<BM, SETS>));
  if (!ready) throw std::runtime_error("moe_mxfp4_bf16_mma: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, 8 * SETS), max_tiles), dim3(32 * kSetWarps * SETS),
                        sizeof(Smem<BM, SETS>), L, xa, W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks,
                        sc.n_tasks, sc.pairs, top_k, mact, out, ld_out);
}

}  // namespace

void moe_mxfp4_bf16_stage(const float* x, int ldx, int K, int rows, bf16* out, const Launch& L) {
  if (K % 32) throw std::runtime_error("moe_mxfp4_bf16_stage: K a multiple of 32 (whole MXFP4 blocks)");
  launch_dependent(k_moe_stage_mxfp4_bf16, dim3(rows), dim3(256), L, x, ldx, K, out);
}

int moe_mxfp4_bm(int pairs, int n_experts) { return pairs > 14 * n_experts ? 32 : 16; }

void moe_mxfp4_bf16_mma(bool fuse_act, bool is_down, int bm, const bf16* xa, const ExpertMat& W, int N_work, int I_or_H,
                    int K, const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out,
                    const Launch& L) {
  if (bm != 16 && bm != 32) throw std::runtime_error("moe_mxfp4_bf16_mma: tasks of 16 or 32 rows");
  auto go = [&](auto bmc) {
    constexpr int B = decltype(bmc)::value;
    if (is_down) launch<false, true, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act) launch<true, false, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch<false, false, B>(xa, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (bm == 32) go(std::integral_constant<int, 32>{});
  else go(std::integral_constant<int, 16>{});
}

}  // namespace flyx::cuda
