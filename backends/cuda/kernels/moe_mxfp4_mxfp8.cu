// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// MXFP4 experts with MXFP8 inputs (FLYX_ACT_MXFP8, gpt-oss's experts as SGLang runs them) on the block-scaled tensor
// cores of sm_120a: mma.sync m16n8k32 kind::mxf8f6f4, A the inputs' e4m3 values with a ue8m0 scale per 32
// (k_moe_stage_mxfp8 in moe.cu), B the experts' e2m1 codes (in the middle 4 bits of a byte each) with their blocks'
// E8M0 scales, float32 sums.  Built for sm_120a only (its own library); a stub elsewhere.
//
// moe_mxfp4_bf16.cu's tiles and pipeline (tasks of 16 or 32 rows of one expert, moe_mxfp4_bm; 64 columns a block of
// 8 column sets of 4 warps; spans of 16 units of K, the rows, their scales, the weights and theirs staged in shared
// memory a span ahead; warp wi of a set its units 4 wi .. 4 wi + 3 of each span), its operand layouts (a lane's 8
// bytes of a row at 8 t4 of a unit, the codes of its column at the same values of K), the scales in hardware: one
// float32 sum a lane, MMA after MMA in K's order, then the four warps' sums in a fixed tree.  Every row gets the same
// MMAs in the same order whatever other rows its task holds and whichever task size: the same bits at every row count
// (rounds and passes take these kernels alike).
//
// Scale operands (PTX ISA, block scaling for mma.sync, scale_vec::1X): thread-id-a 0, the scale of A's row g from lane
// 4 g, of row g + 8 from lane 4 g + 1; thread-id-b 0, the scale of B's column g from lane 4 g; byte-id 0, the low byte.
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/kernels.cuh"
#include "backends/cuda/kernels/moe_act.cuh"

namespace flyx::cuda {
namespace {

constexpr int kSetWarps = 4, kSpan = 16;
constexpr int kRow = kSpan * 32 + 32;   // e4m3 bytes a row: rows 8 banks apart
constexpr int kXs = kSpan + 8;          // a row's scales of a span: from the 4-byte word holding its first one
constexpr int kWcol = kSpan * 16 + 16;  // bytes a column: 68 words, 4 banks apart (a warp's 8 columns x 4 lanes)
constexpr int kScol = kSpan + 4;        // scales a column (u16): 10 words

template <int BM, int SETS>
struct Smem {
  struct Stage {
    alignas(16) uint8_t rows[BM][kRow];
    alignas(16) uint8_t xs[BM][kXs];
    alignas(16) uint8_t w[8 * SETS][kWcol];
    alignas(16) uint16_t s[8 * SETS][kScol];
  };
  union {
    Stage st[2];
    float tree[SETS][kSetWarps][32][BM / 4];  // after a pass's spans
  };
  int pids[BM];
  int rids[BM];
  int xoff[BM];  // where a row's scales start in its 4-byte word
};

__device__ __forceinline__ void mma_mx8(float* c, const uint32_t* a, const uint32_t* b, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(uint16_t(0)),
        "h"(uint16_t(0)), "r"(sfb), "h"(uint16_t(0)), "h"(uint16_t(0)));
}

// A block's E8M0 exponent from the bf16 bits of its stored scale 0.5 2^(e - 127) (blocks_pack: GGML's values are
// 2 e2m1): the exponent field plus one, the two smallest (2^-128 and 2^-127, bf16 subnormals) from the mantissa.
__device__ __forceinline__ uint32_t e8m0_of(uint32_t bits) {
  const uint32_t f = (bits >> 7) & 0xFFu;
  return f ? f + 1 : ((bits & 0x40u) ? 1u : 0u);
}

template <bool FUSE_ACT, bool IS_DOWN, int BM, int SETS>
__global__ void __launch_bounds__(32 * kSetWarps * SETS)
    k_moe_mxfp4_mxfp8_mma(const uint8_t* xa, const uint8_t* xsc, const uint8_t* __restrict__ W,
                    const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                    const MoeTask* tasks, const int* n_tasks, const int* pairs, int top_k, MoeAct mact,
                    float* __restrict__ out, int ld_out) {
  constexpr int T = BM / 16, NT = 32 * kSetWarps * SETS, NC = 8 * SETS, RB = kSpan * 32;
  pdl_trigger();
  pdl_wait();  // the staged rows and the tasks: the kernels before
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<Smem<BM, SETS>*>(smem_raw);
  const int tid = threadIdx.x;
  const int units = K >> 5, spans = (units + kSpan - 1) / kSpan;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
    sm.xoff[tid] = int((size_t(sm.rids[tid]) * units) & 3);
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, wi = warp % kSetWarps, ct = warp / kSetWarps;
  const int g = lane >> 2, t4 = lane & 3;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = int(blockIdx.x) * NC;
  const float rs = RS ? RS[e] : 1.f;
  const bool two = T == 2 && nb > 16;  // the second tile holds rows
  const int rows_st = two ? 32 : 16;   // the rows staged (past the task's rows: zeros, their scales never read)

  // span j of the rows and of the block's columns starting at row0 of the expert (a short last span: its units alone,
  // the rest never read)
  auto stage = [&](int row0, int j) {
    auto& S = sm.st[j & 1];
    const int n_s = min(kSpan, units - kSpan * j);
    for (int i = tid; i < rows_st * (RB / 16); i += NT) {  // 16 bytes a thread
      const int r = i / (RB / 16), c = i % (RB / 16);
      const bool ok = r < nb && c < 2 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * RB + 16 * c) : 0;
      cp_async16(&S.rows[r][16 * c], xa + src, ok);
    }
    // the rows' scales: the 4-byte words that hold a span's (K / 32 bytes a row, any alignment; a word or so past a
    // row's last scale stays inside the scratch, which holds 4 bytes a unit)
    for (int i = tid; i < nb * (kXs / 4); i += NT) {
      const int r = i / (kXs / 4), c = i % (kXs / 4);
      const size_t at = (size_t(sm.rids[r]) * units + size_t(kSpan) * j) & ~size_t(3);
      cp_async4(&S.xs[r][4 * c], xsc + at + 4 * c);
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
    const int cl = ct * 8 + g, n_w = min(4, units - kSpan * j - 4 * wi), sh = 4 * (t4 >> 1);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int ul = 4 * wi + s;
      if (s >= n_w) break;
      // the lane's 8 values of the block, 8 t4 .. 8 t4 + 7: bytes 8 (t4 & 1) on, their low nibbles for t4 < 2
      const uint2 q = *reinterpret_cast<const uint2*>(&S.w[cl][16 * ul + 8 * (t4 & 1)]);
      const uint32_t b[2] = {((q.x >> sh) & 0x0F0F0F0Fu) << 2, ((q.y >> sh) & 0x0F0F0F0Fu) << 2};
      const uint32_t sfb = e8m0_of(S.s[cl][ul]);
#pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t == 1 && !two) break;
        const int ra = 16 * t + g, rb = ra + 8, rsf = (t4 & 1) ? rb : ra;
        const uint2 x0 = *reinterpret_cast<const uint2*>(&S.rows[ra][32 * ul + 8 * t4]);
        const uint2 x8 = *reinterpret_cast<const uint2*>(&S.rows[rb][32 * ul + 8 * t4]);
        const uint32_t a[4] = {x0.x, x8.x, x0.y, x8.y};
        // past the task's rows: no scales staged, their products never stored
        const uint32_t sfa = rsf < nb ? S.xs[rsf][sm.xoff[rsf] + ul] : 127u;
        mma_mx8(acc[t], a, b, sfa, sfb);
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

template <bool FUSE_ACT, bool IS_DOWN, int BM>
void launch(const uint8_t* xa, const uint8_t* xs, const ExpertMat& W, int N_work, int I_or_H, int K,
            const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  constexpr int SETS = 8;
  auto* kernel = k_moe_mxfp4_mxfp8_mma<FUSE_ACT, IS_DOWN, BM, SETS>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(Smem<BM, SETS>));
  if (!ready) throw std::runtime_error("moe_mxfp4_mxfp8_mma: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, 8 * SETS), max_tiles), dim3(32 * kSetWarps * SETS),
                        sizeof(Smem<BM, SETS>), L, xa, xs, W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks,
                        sc.n_tasks, sc.pairs, top_k, mact, out, ld_out);
}

// ---- Prefill passes (more than FLYX_MAX_ROWS rows): swap-AB tiles ----------------------------------------------
// A the experts' weight rows (e2m1, their blocks' E8M0 scales), B the task's rows (e4m3, ue8m0), C[weight row][row]:
// one float32 sum an output, MMA after MMA in K's order, so every tile shape gives the same bits (a pass's rows do not
// depend on the pass's length: numerics.md K) and the shape follows the rows an expert has (moe_mxfp4_mxfp8_pass_rows).
// Rounds take the K-split kernels above (their own bits, these within float32 sums of them, as the dense GEMMs').
// A block: 16 MT WMW weight rows (fused gate|up: an m16 tile 8 columns' gate rows then their up rows, the activation's
// pair in one thread) x 8 NT WT rows of one expert; warps WMW x WT, a warp MT m16 x NT n8 tiles; spans of SU units,
// the rows, their scales, the weights and theirs staged a span ahead (cp.async), NS stages.
template <int WROWS, int TOK, int NS, int SU>
struct PassSmem {
  struct Stage {
    alignas(16) uint8_t x[TOK][SU * 32 + 32];    // rows: 8 banks apart
    alignas(16) uint8_t xs[TOK][SU + 4];         // the words holding a row's span of scales
    alignas(16) uint8_t w[WROWS][SU * 16 + 16];  // codes
    alignas(16) uint16_t ws[WROWS][SU + 2];      // a weight row's span of scales (bf16 bits)
  };
  Stage st[NS];
  int pids[TOK];
  int rids[TOK];
  int xoff[TOK];  // where a row's scales start in its 4-byte word
};

template <bool FUSE_ACT, bool IS_DOWN, int MT, int NT, int WMW, int WT, int NS, int SU>
__global__ void __launch_bounds__(32 * WMW * WT)
    k_moe_mxfp4_mxfp8_pass(const uint8_t* xa, const uint8_t* xsc, const uint8_t* __restrict__ W,
                           const uint16_t* __restrict__ D, const float* __restrict__ RS, int N_exp, int I_or_H, int K,
                           const MoeTask* tasks, const int* n_tasks, const int* pairs, int top_k, MoeAct mact,
                           float* __restrict__ out, int ld_out) {
  constexpr int WROWS = 16 * MT * WMW, TOK = 8 * NT * WT, NTH = 32 * WMW * WT, XW = (SU + 4) / 4;
  constexpr int COLS = FUSE_ACT ? WROWS / 2 : WROWS;  // output columns a block
  using S_t = PassSmem<WROWS, TOK, NS, SU>;
  pdl_trigger();
  pdl_wait();  // the staged rows and the tasks: the kernels before
  const int tid_y = int(blockIdx.y);
  if (tid_y >= *n_tasks) return;
  const MoeTask tk = tasks[tid_y];
  const int e = tk.expert, nb = tk.nb;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  auto& sm = *reinterpret_cast<S_t*>(smem_raw);
  const int tid = threadIdx.x;
  const int units = K >> 5, spans = (units + SU - 1) / SU;
  if (tid < nb) {
    const int pid = pairs[tk.p0 + tid];
    sm.pids[tid] = pid;
    sm.rids[tid] = IS_DOWN ? pid : (pid / top_k);
    sm.xoff[tid] = int((size_t(sm.rids[tid]) * units) & 3);
  }
  __syncthreads();
  const int lane = tid & 31, warp = tid >> 5, ww = warp / WT, wt = warp % WT;
  const int g = lane >> 2, t4 = lane & 3;
  const int N_work = FUSE_ACT ? I_or_H : N_exp;
  const int c0 = int(blockIdx.x) * COLS;
  const float rs = RS ? RS[e] : 1.f;
  const int rows_st = min(TOK, (nb + 7) & ~7);
  // smem weight row s -> the expert's row
  auto wrow = [&](int s) {
    if constexpr (FUSE_ACT) {
      const int col = min(c0 + 8 * (s >> 4) + (s & 7), N_work - 1);
      return (s & 8) ? I_or_H + col : col;
    } else {
      return min(c0 + s, N_exp - 1);
    }
  };
  auto stage = [&](int j) {
    if (j >= spans) {
      cp_async_commit();
      return;
    }
    auto& S = sm.st[j % NS];
    const int n_s = min(SU, units - SU * j);
    for (int i = tid; i < rows_st * (SU * 2); i += NTH) {  // 16 bytes a thread
      const int r = i / (SU * 2), c = i % (SU * 2);
      const bool ok = r < nb && c < 2 * n_s;
      const size_t src = ok ? (size_t(sm.rids[r]) * K + size_t(j) * (SU * 32) + 16 * c) : 0;
      cp_async16(&S.x[r][16 * c], xa + src, ok);
    }
    // the rows' scales: the 4-byte words that hold a span's (K / 32 bytes a row; a word past a row's last one stays
    // inside the scratch, which holds 4 bytes a unit)
    for (int i = tid; i < nb * XW; i += NTH) {
      const int r = i / XW, c = i % XW;
      const size_t at = (size_t(sm.rids[r]) * units + size_t(SU) * j) & ~size_t(3);
      cp_async4(&S.xs[r][4 * c], xsc + at + 4 * c);
    }
    for (int i = tid; i < WROWS * SU; i += NTH) {  // a unit's 16 bytes a thread
      const int s = i / SU, u = i % SU;
      const size_t r = size_t(e) * N_exp + wrow(s);
      const bool ok = u < n_s;
      cp_async16(&S.w[s][16 * u], W + (r * size_t(units) + size_t(SU) * j + (ok ? u : 0)) * 16, ok);
    }
    if ((units & 1) == 0) {  // their scales two at a time (4 bytes, aligned where a row's units are even)
      for (int i = tid; i < WROWS * (SU / 2); i += NTH) {
        const int s = i / (SU / 2), u = 2 * (i % (SU / 2));
        const size_t r = size_t(e) * N_exp + wrow(s);
        const bool ok = u < n_s;
        cp_async4(&S.ws[s][u], D + r * size_t(units) + size_t(SU) * j + (ok ? u : 0), ok);
      }
    } else {
      for (int i = tid; i < WROWS * n_s; i += NTH) {
        const int s = i / n_s, u = i - s * n_s;
        S.ws[s][u] = D[(size_t(e) * N_exp + wrow(s)) * size_t(units) + size_t(SU) * j + u];
      }
    }
    cp_async_commit();
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
  const bool hi = (t4 >> 1) != 0;  // the lane's codes: the bytes' high nibbles (values 16..31 of a block)
  auto mul = [&](int j) {
    const auto& S = sm.st[j % NS];
    const int n_s = min(SU, units - SU * j);
#pragma unroll
    for (int ul = 0; ul < SU; ++ul) {
      if (ul >= n_s) break;
      uint32_t a[MT][4], sfa[MT], b[NT][2], sfb[NT];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {  // weight rows 16 t + g and + 8: values 8 t4 .. 8 t4 + 7, in bits 5..2
        const int ra = 16 * (MT * ww + mt) + g, rb = ra + 8;
        const uint2 qa = *reinterpret_cast<const uint2*>(&S.w[ra][16 * ul + 8 * (t4 & 1)]);
        const uint2 qb = *reinterpret_cast<const uint2*>(&S.w[rb][16 * ul + 8 * (t4 & 1)]);
        a[mt][0] = (hi ? (qa.x >> 2) : (qa.x << 2)) & 0x3C3C3C3Cu;
        a[mt][1] = (hi ? (qb.x >> 2) : (qb.x << 2)) & 0x3C3C3C3Cu;
        a[mt][2] = (hi ? (qa.y >> 2) : (qa.y << 2)) & 0x3C3C3C3Cu;
        a[mt][3] = (hi ? (qb.y >> 2) : (qb.y << 2)) & 0x3C3C3C3Cu;
        sfa[mt] = e8m0_of(S.ws[(t4 & 1) ? rb : ra][ul]);
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {  // row 8 t + g: bytes 8 t4 .. 8 t4 + 7 of the unit
        const int tr = 8 * (NT * wt + nt) + g;
        const uint2 x = *reinterpret_cast<const uint2*>(&S.x[tr][32 * ul + 8 * t4]);
        b[nt][0] = x.x, b[nt][1] = x.y;
        sfb[nt] = tr < nb ? S.xs[tr][sm.xoff[tr] + ul] : 127u;  // past the task's rows: never stored
      }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if (8 * (NT * wt + nt) >= nb) break;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
          asm volatile(
              "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
              : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
              : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b[nt][0]), "r"(b[nt][1]),
                "r"(sfa[mt]), "h"(uint16_t(0)), "h"(uint16_t(0)), "r"(sfb[nt]), "h"(uint16_t(0)), "h"(uint16_t(0)));
      }
    }
  };
#pragma unroll
  for (int p = 0; p < NS - 1; ++p) stage(p);
  for (int j = 0; j < spans; ++j) {
    cp_async_wait<NS - 2>();
    __syncthreads();
    stage(j + NS - 1);
    mul(j);
  }
  // C[weight row][row]: the lane's weight rows 16 t + g (+ 8), rows 2 t4 (+ 1) of each n8 tile
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const int tr = 8 * (NT * wt + nt) + 2 * t4 + (k & 1);
        if (tr >= nb) continue;
        if constexpr (FUSE_ACT) {
          if (k >= 2) continue;  // the up row's sum: k + 2
          const int col = c0 + 8 * (MT * ww + mt) + g;
          if (col >= N_work) continue;
          out[size_t(sm.pids[tr]) * ld_out + col] =
              moe_act_val(mact, acc[mt][nt][k] * rs, acc[mt][nt][k + 2] * rs, e, col, I_or_H);
        } else {
          const int col = c0 + 16 * (MT * ww + mt) + g + 8 * (k >> 1);
          if (col >= N_work) continue;
          out[size_t(sm.pids[tr]) * ld_out + col] = acc[mt][nt][k] * rs;
        }
      }
}

// The three shapes, by the rows an expert has: 16 rows a task, 128 weight rows (8 warps, spans of 16); 64, 128
// (8 warps 4 x 2, spans of 8); 128, 256 (16 warps 4 x 4, spans of 4).  gpt-oss-20b's experts on an RTX PRO 6000, us
// an expert GEMM (gate|up and down averaged): 91 rows 156 (the K-split kernels: 165), 256 rows 196 (227), 512 rows
// 257 (371), 4096 rows 1093 (2175).
template <bool FUSE_ACT, bool IS_DOWN, int TOK>
void launch_pass(const uint8_t* xa, const uint8_t* xs, const ExpertMat& W, int N_work, int I_or_H, int K,
                 const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  constexpr int MT = TOK == 16 ? 1 : TOK == 64 ? 2 : 4, NT = TOK == 16 ? 2 : 4;
  constexpr int WMW = TOK == 16 ? 8 : 4, WT = TOK == 16 ? 1 : TOK == 64 ? 2 : 4;
  constexpr int NS = 2, SU = TOK == 16 ? 16 : TOK == 64 ? 8 : 4;
  constexpr int WROWS = 16 * MT * WMW, COLS = FUSE_ACT ? WROWS / 2 : WROWS;
  static_assert(8 * NT * WT == TOK);
  using S_t = PassSmem<WROWS, TOK, NS, SU>;
  auto* kernel = k_moe_mxfp4_mxfp8_pass<FUSE_ACT, IS_DOWN, MT, NT, WMW, WT, NS, SU>;
  static const bool ready = set_dynamic_smem(kernel, sizeof(S_t));
  if (!ready) throw std::runtime_error("moe_mxfp4_mxfp8_pass: the device refused its shared memory");
  launch_dependent_smem(kernel, dim3(ceil_div(N_work, COLS), max_tiles), dim3(32 * WMW * WT), sizeof(S_t), L, xa, xs,
                        W.blk.data, W.blk.scales, W.rs, W.N, I_or_H, K, sc.tasks, sc.n_tasks, sc.pairs, top_k, mact,
                        out, ld_out);
}

}  // namespace

bool moe_mxfp4_mxfp8_supported() {
  static const bool ok = [] {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major == 12 && minor == 0;  // the kernels are sm_120a machine code (strictly sm_120)
  }();
  return ok;
}

void moe_mxfp4_mxfp8_mma(bool fuse_act, bool is_down, int bm, const uint8_t* xa, const uint8_t* xs, const ExpertMat& W,
                         int N_work, int I_or_H, int K, const MoeScratch& sc, int max_tiles, int top_k, MoeAct mact,
                         float* out, int ld_out, const Launch& L) {
  if (!moe_mxfp4_mxfp8_supported())
    throw std::runtime_error("moe_mxfp4_mxfp8_mma: MXFP8 inputs need sm_120's tensor cores");
  if (bm != 16 && bm != 32) throw std::runtime_error("moe_mxfp4_mxfp8_mma: tasks of 16 or 32 rows");
  auto go = [&](auto bmc) {
    constexpr int B = decltype(bmc)::value;
    if (is_down) launch<false, true, B>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act) launch<true, false, B>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch<false, false, B>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (bm == 32) go(std::integral_constant<int, 32>{});
  else go(std::integral_constant<int, 16>{});
}

void moe_mxfp4_mxfp8_pass(bool fuse_act, bool is_down, int tok, const uint8_t* xa, const uint8_t* xs,
                          const ExpertMat& W, int N_work, int I_or_H, int K, const MoeScratch& sc, int max_tiles,
                          int top_k, MoeAct mact, float* out, int ld_out, const Launch& L) {
  if (!moe_mxfp4_mxfp8_supported())
    throw std::runtime_error("moe_mxfp4_mxfp8_pass: MXFP8 inputs need sm_120's tensor cores");
  if (tok != 16 && tok != 64 && tok != 128)
    throw std::runtime_error("moe_mxfp4_mxfp8_pass: tasks of 16, 64 or 128 rows");
  auto go = [&](auto tc) {
    constexpr int T = decltype(tc)::value;
    if (is_down) launch_pass<false, true, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else if (fuse_act)
      launch_pass<true, false, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
    else launch_pass<false, false, T>(xa, xs, W, N_work, I_or_H, K, sc, max_tiles, top_k, mact, out, ld_out, L);
  };
  if (tok == 128) go(std::integral_constant<int, 128>{});
  else if (tok == 64) go(std::integral_constant<int, 64>{});
  else go(std::integral_constant<int, 16>{});
}

}  // namespace flyx::cuda
