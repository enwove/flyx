// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A round's SwiGLU in the dp4a gate and up GEMM (Launch::glu): the kernel's GLU variant for every block
// type, in a translation unit of its own, so the groups' round units compile as they did without it (with it inside,
// each took 15-20% longer and the longest, the K types', set the build's length).
#include "backends/cuda/kernels/gemm_blocks_round_impl.cuh"

namespace flyx::cuda {

namespace {

// The launch of type F in configuration cfg that also writes the SwiGLU; false, nothing run, for the gate run's own
// launch of a layer whose gate and up are runs of their own (the plain kernel's, the up run's writes it).
template <class F>
bool launch_glu(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  const RoundGluArgs glu = round_glu_args(L, W, y, ldy, rows, accumulate);
  if (!glu.gu) return false;
  constexpr int di = tables::ggml_type_index(F::kDtype);
  switch (cfg) {
#define X(i, R, C, U, NW, S, P)                                                                                  \
  case i:                                                                                                       \
    if constexpr (P == 1 && S != 2 && R <= kRoundGluRows &&                                                     \
                  (FLYX_TUNE_CANDIDATES || (di >= 0 && (tables::kReachRoundGlu[tables::kArchBucketAny][di] &    \
                                                        (1u << i))))) {                                         \
      launch_round_glu<F, R, kRoundColumns<F, C>, kUnitsAhead<F, C, U>, NW, S>(W, K, y, ldy, rows, L, glu);     \
      return true;                                                                                              \
    }                                                                                                           \
    break;
    FLYX_BLOCKS_ROUND(X)
#undef X
    default:
      break;
  }
  std::fprintf(stderr, "flyx-cuda: a round's SwiGLU in configuration %d, which has none\n", cfg);
  std::abort();
}

using GluFn = bool (*)(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate,
                       const Launch& L);
template <class... Fs>
GluFn glu_of(int dtype) {
  GluFn found = nullptr;
  ((found = !found && dtype == Fs::kDtype ? &launch_glu<Fs> : found), ...);
  return found;
}

}  // namespace

bool round_glu(int cfg, const BlockMat& W, int K, float* y, int ldy, int rows, bool accumulate, const Launch& L) {
  const GluFn launch = glu_of<FQ4_0, FQ4_1, FQ5_0, FQ5_1, FQ8_0, FQ2_K, FQ3_K, FQ4_K, FQ5_K, FQ6_K, FIQ1_S, FIQ1_M,
                              FIQ2_XXS, FIQ2_XS, FIQ2_S, FIQ3_XXS, FIQ3_S, FIQ4_NL, FIQ4_XS>(W.dtype);
  if (!launch) {
    std::fprintf(stderr, "flyx-cuda: a round's SwiGLU on dtype %d, not a GGML block type\n", W.dtype);
    std::abort();
  }
  return launch(cfg, W, K, y, ldy, rows, accumulate, L);
}

}  // namespace flyx::cuda
