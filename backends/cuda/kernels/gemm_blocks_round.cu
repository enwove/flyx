// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Rounds (1..16 rows: decoding, verifying drafts) on GGML block weights whose inputs are 8 bits (FLYX_ACT_Q8):
// which kernel, which configuration of the dp4a one (variants/candidates.toml, blocks_round), and
// the dispatch to the kernels of the weights' group of block types (gemm_blocks_round_impl.cuh: the kernels and
// how they add up a column, the same bits whatever the kernel, configuration or rows).
#include <cstdio>
#include <cstdlib>

#include "backends/cuda/kernels/common/launch.h"
#include "backends/cuda/kernels/gemm_blocks_ops.h"
#include "backends/cuda/kernels/gemm_internal.h"
#include "flyx/gemm_tables.inc"

#ifndef FLYX_TUNE_CANDIDATES
#define FLYX_TUNE_CANDIDATES 1
#endif

namespace flyx::cuda {

namespace {

// The rows a configuration's kernel holds.
constexpr int kRoundRows[] = {
#define X(i, R, C, U, NW, S, P) R,
    FLYX_BLOCKS_ROUND(X)
#undef X
};
// The parts of the rows a configuration's warps hold apart.
constexpr int kRoundParts[] = {
#define X(i, R, C, U, NW, S, P) P,
    FLYX_BLOCKS_ROUND(X)
#undef X
};
// Whether a configuration's rows go through shared memory a window of K at a time (one run a launch, no stack).
constexpr bool kRoundWindows[] = {
#define X(i, R, C, U, NW, S, P) S == 2,
    FLYX_BLOCKS_ROUND(X)
#undef X
};

// The rules a round on a GPU of table `tuning` follows: the first that matches the layer's type `dtype`, the round's
// rows (1..16) and the layer's N columns, among this GPU's table's (or the one it inherits) or else the defaults
// (variants/candidates.toml); `dp4a`: only the rules that run the dp4a kernel.
const tables::BlocksRoundRule* round_rule(int tuning, int dtype, int N, int rows, bool dp4a = false) {
  const int s = tables::rules_sku(tuning, tables::kFamilyBlocksRound);
  for (const tables::BlocksRoundRule& r : tables::kBlocksRound)
    if (r.sku == s && (r.dtype == dtype || r.dtype < 0) && rows >= r.lo && rows <= r.hi && N > r.n_min &&
        N <= r.n_max && (!dp4a || r.kernel == 0))
      return &r;
  return nullptr;
}

const tables::BlocksRoundRule& default_round_rule(int tuning, int dtype, int N, int rows) {
  if (const tables::BlocksRoundRule* r = round_rule(tuning, dtype, N, rows)) return *r;
  std::fprintf(stderr, "flyx-cuda: no blocks_round rule for %d rows (variants)\n", rows);
  std::abort();
}

// The launcher of a stack's n runs of these types in one launch; nullptr where they are not compiled together.
RoundStackFn round_stack_ops(const int* dtype, int n) {
  for (RoundStackFn (*group)(const int*, int) : {round_stack_ops_legacy, round_stack_ops_k, round_stack_ops_iq_low,
                                                  round_stack_ops_iq_high})
    if (const RoundStackFn launch = group(dtype, n)) return launch;
  return nullptr;
}

// The widest run after the first that a launch takes: those are taken a column a turn, and a wide one loses what its
// own configuration shares among a warp's columns (a row's values loaded once for them): the GDN's z (6144 columns)
// after qkv, of another type, went 19.1 -> 20.2 us at 1 row and 30.3 -> 33.9 at 8 on the 27B's 2-bit file, where
// attention's k and v (1024 columns each) after q went 22.1 -> 14.0 and 40.2 -> 24.0.
constexpr int kStackNextColumns = 2048;

// A stack's n runs in one launch: its launcher and the first run's dp4a configuration by the rules (the first run's
// columns go as they would alone); -1 where the types are not compiled together, a run after the first is wider than
// kStackNextColumns, or the rules run the first run on the tensor cores.
int stack_cfg(const BlockMat* W, int n, int rows, const Launch& L, RoundStackFn* launch) {
  if (n < 2 || n > 3 || rows < 1 || rows > 16 || L.glu || W[0].rows <= 2048 || W[0].rows > 65536) return -1;
  int dtype[3];
  for (int i = 0; i < n; ++i) {
    if (!W[i].q8 || (i > 0 && W[i].rows > kStackNextColumns)) return -1;
    dtype[i] = W[i].dtype;
  }
  *launch = round_stack_ops(dtype, n);
  if (!*launch) return -1;
  const tables::BlocksRoundRule& rule = default_round_rule(opts(L).tuning, W[0].dtype, W[0].rows, rows);
  return rule.kernel == 0 && !kRoundWindows[rule.cfg] ? rule.cfg : -1;
}

// A round's kernel: 0 dp4a, 1 the tensor cores with the rows in the MMA's first operand, 2 with the columns there
// (all give the same bits), -1 the rules'.  The dp4a kernel's configuration: `cfg`, or the rules'.
void round_q8(int kernel, int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
              const Launch& L) {
  if (rows < 1 || rows > 16 || K % 32 || (cfg >= 0 && (cfg >= FLYX_BLOCKS_ROUND_COUNT || rows > kRoundRows[cfg]))) {
    std::fprintf(stderr, "flyx-cuda: a round on block weights takes 1..16 rows (at most a configuration's) and "
                         "K %% 32 == 0 (rows %d, K %d, configuration %d)\n", rows, K, cfg);
    std::abort();
  }
  const int tuning = opts(L).tuning;
  if (kernel < 0) {
    const tables::BlocksRoundRule& rule = default_round_rule(tuning, W.dtype, W.rows, rows);
    kernel = rule.kernel;
    if (cfg < 0) cfg = rule.cfg;
  }
  if (kernel == 0 && cfg < 0) {  // dp4a asked for, the rules a tensor-core kernel: the first dp4a rule that matches
    if (const tables::BlocksRoundRule* r = round_rule(tuning, W.dtype, W.rows, rows, true)) {
      if (gemm_blocks_round_cfg_compiled(W.dtype, r->cfg)) cfg = r->cfg;
    }
    if (cfg < 0) {  // a compiled configuration that holds the round's rows
      for (int i = 0; i < FLYX_BLOCKS_ROUND_COUNT; ++i)
        if (kRoundRows[i] >= rows && gemm_blocks_round_cfg_compiled(W.dtype, i) &&
            (cfg < 0 || kRoundRows[i] < kRoundRows[cfg]))
          cfg = i;
    }
  }
  if (L.glu) {
    if (!(kernel == 0 && !kRoundWindows[cfg] && kRoundParts[cfg] == 1 && kRoundRows[cfg] <= kRoundGluRows)) {
      // gemm_blocks_round_glu_takes said so
      std::fprintf(stderr, "flyx-cuda: a round's SwiGLU in a GEMM whose kernel does not write it (kernel %d, "
                           "configuration %d)\n", kernel, cfg);
      std::abort();
    }
    if (round_glu(cfg, W, K, y, ldy, rows, accumulate, L)) return;
  }
  for (RoundFn (*group)(int) : {round_ops_legacy, round_ops_k, round_ops_iq_low, round_ops_iq_high})
    if (const RoundFn launch = group(W.dtype)) return launch(kernel, cfg, W, K, y, ldy, rows, accumulate, L);
  std::fprintf(stderr, "flyx-cuda: a round on dtype %d, which is not a GGML block type the kernels take\n", W.dtype);
  std::abort();
}

}  // namespace

const BlocksRoundConfig kBlocksRoundConfigs[] = {
#define X(i, R, C, U, NW, S, P) {R, C, U, NW, S, P},
    FLYX_BLOCKS_ROUND(X)
#undef X
};
const int kBlocksRoundConfigCount = FLYX_BLOCKS_ROUND_COUNT;

int gemm_blocks_round_default(int tuning, int dtype, int N, int rows) {
  return default_round_rule(tuning, dtype, N, rows).cfg;
}
int gemm_blocks_round_default_kernel(int tuning, int dtype, int N, int rows) {
  return default_round_rule(tuning, dtype, N, rows).kernel;
}

void gemm_blocks_round_q8(const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  round_q8(-1, -1, W, K, y, ldy, rows, accumulate, L);
}

void gemm_blocks_round_q8_kernel(int kernel, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                                 const Launch& L) {
  round_q8(kernel, -1, W, K, y, ldy, rows, accumulate, L);
}

void gemm_blocks_round_q8_cfg(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                              const Launch& L) {
  round_q8(0, cfg, W, K, y, ldy, rows, accumulate, L);
}

bool gemm_blocks_round_warps_takes(int dtype) {
  return round_warps_legacy(dtype) || round_warps_k(dtype) || round_warps_iq_low(dtype) || round_warps_iq_high(dtype);
}

bool gemm_blocks_round_cfg_compiled(int dtype, int cfg) {
#if FLYX_TUNE_CANDIDATES
  (void)dtype, (void)cfg;
  return true;
#else
  const int arch = compiled_arch();
  const int di = tables::ggml_type_index(dtype);
  return arch > 0 && di >= 0 && cfg >= 0 && cfg < FLYX_BLOCKS_ROUND_COUNT &&
         (tables::kReachRoundDp4a[tables::arch_bucket(arch)][di] & (1u << cfg)) != 0;
#endif
}

bool gemm_blocks_round_glu_compiled(int dtype, int cfg) {
#if FLYX_TUNE_CANDIDATES
  (void)dtype, (void)cfg;
  return true;
#else
  const int arch = compiled_arch();
  const int di = tables::ggml_type_index(dtype);
  return arch > 0 && di >= 0 && cfg >= 0 && cfg < FLYX_BLOCKS_ROUND_COUNT &&
         (tables::kReachRoundGlu[tables::arch_bucket(arch)][di] & (1u << cfg)) != 0;
#endif
}

bool gemm_blocks_round_kernel_compiled(int dtype, int kernel, int rows) {
#if FLYX_TUNE_CANDIDATES
  (void)dtype, (void)kernel, (void)rows;
  return true;
#else
  const int arch = compiled_arch();
  const int di = tables::ggml_type_index(dtype);
  if (arch <= 0 || di < 0) return false;
  const int b = tables::arch_bucket(arch);
  if (kernel == 0) {
    const unsigned m = tables::kReachRoundDp4a[b][di];
    for (int i = 0; i < FLYX_BLOCKS_ROUND_COUNT; ++i)
      if ((m & (1u << i)) && kRoundRows[i] >= rows) return true;
    return false;
  }
  const unsigned tc = tables::kReachRoundTc[b][di];
  if (kernel == 5) {
    if (gemm_blocks_round_warps_takes(dtype)) return (tc & (1u << (rows <= 8 ? 6 : 7))) != 0;
    kernel = 3;
  }
  if (kernel == 4) return (tc & (1u << (rows <= 8 ? 5 : 4))) != 0;
  if (kernel == 3) return (tc & (1u << (rows <= 8 ? 3 : 4))) != 0;
  if (kernel == 2) return (tc & (1u << (rows <= 8 ? 2 : 1))) != 0;
  if (kernel == 1) return (tc & (1u << (rows <= 8 ? 0 : 1))) != 0;
  return false;
#endif
}

bool gemm_blocks_round_q8_stack_takes(const BlockMat* W, int n, int rows, const Launch& L) {
  RoundStackFn launch = nullptr;
  return stack_cfg(W, n, rows, L, &launch) >= 0;
}

void gemm_blocks_round_q8_stack(const BlockMat* W, int n, int K, float* y, int ldy, int rows, bool accumulate,
                                const Launch& L) {
  RoundStackFn launch = nullptr;
  const int cfg = stack_cfg(W, n, rows, L, &launch);
  if (cfg < 0 || K % 32 || !launch(cfg, W, K, y, ldy, rows, accumulate, L)) {
    std::fprintf(stderr, "flyx-cuda: a stack of %d runs that one round launch does not take (rows %d, K %d)\n", n,
                 rows, K);
    std::abort();
  }
}

bool gemm_blocks_round_glu_takes(const BlockMat* W, int n, int rows, int I, const Launch& L) {
  // One row (decoding): a group a block lost to the GEMM and k_swiglu apart at 3 and 4 rows for most block types and
  // at 2 for some (Q6_K 2 rows 106.7 against 95.9 us), and drafted rounds of 2 to 4 rows lost 1-2.5% on the 27B;
  // the kernel takes up to kRoundGluRows.
  if (!opts(L).round_glu || !L.scratch || n < 1 || rows != 1 || I % 64 || size_t(I) > L.scratch->q8_round_cap)
    return false;
  int columns = 0;
  for (int i = 0; i < n; ++i) {
    // a run after the first that could share a launch with the one before (gemm_blocks_staged_runs): none
    if (!W[i].q8 || (i > 0 && W[i].rows <= kStackNextColumns)) return false;
    const tables::BlocksRoundRule& r = default_round_rule(opts(L).tuning, W[i].dtype, W[i].rows, rows);
    if (r.kernel != 0 || kRoundWindows[r.cfg] || kRoundParts[r.cfg] != 1 || kRoundRows[r.cfg] > kRoundGluRows ||
        rows > kRoundRows[r.cfg])
      return false;
    columns += W[i].rows;
  }
  return columns == 2 * I;
}

bool gemm_blocks_round_stack_compiled(const int* dtype, int n) {
  return n >= 2 && n <= 3 && round_stack_ops(dtype, n) != nullptr;
}

bool gemm_blocks_round_q8_stack_cfg(int cfg, const BlockMat* W, int n, int K, float* y, int ldy, int rows,
                                    bool accumulate, const Launch& L) {
  if (rows < 1 || rows > 16 || K % 32 || cfg < 0 || cfg >= FLYX_BLOCKS_ROUND_COUNT || rows > kRoundRows[cfg]) {
    std::fprintf(stderr, "flyx-cuda: a round on a stack takes 1..16 rows (at most the configuration's) and K %% 32 "
                         "== 0 (rows %d, K %d, configuration %d)\n", rows, K, cfg);
    std::abort();
  }
  int dtype[3] = {0, 0, 0};
  for (int i = 0; i < n && i < 3; ++i) dtype[i] = W[i].dtype;
  const RoundStackFn launch = n >= 2 && n <= 3 ? round_stack_ops(dtype, n) : nullptr;
  return launch && launch(cfg, W, K, y, ldy, rows, accumulate, L);
}

}  // namespace flyx::cuda
