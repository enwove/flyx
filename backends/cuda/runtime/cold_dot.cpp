// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/cold_experts.h"
#include "ops/dtype/ref.h"

#include <array>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#define FLYX_AVX2 __declspec(noinline)
#else
#define FLYX_AVX2 __attribute__((target("avx2,fma")))
#endif
#endif

namespace flyx::cuda {
using namespace ops::ggml;

float cold_dot_scalar(int dtype, const uint8_t* row, const float* x, int cols) {
  flyx_tensor_view v{};
  v.dtype = dtype;
  v.data = row;
  float sum = 0;
  for (int i = 0; i < cols; ++i) sum = std::fma(float(ops::load(&v, i)), x[i], sum);
  return sum;
}

#ifdef FLYX_AVX2
namespace {

bool has_avx2() {
#ifdef _MSC_VER
  int r[4];
  __cpuid(r, 0);
  if (r[0] < 7) return false;
  __cpuidex(r, 1, 0);
  constexpr int flags = (1 << 12) | (1 << 27) | (1 << 28);
  if ((r[2] & flags) != flags || (_xgetbv(0) & 6) != 6) return false;
  __cpuidex(r, 7, 0);
  return (r[1] & (1 << 5)) != 0;
#else
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

float half_scale(uint16_t h) {
  static const auto table = [] {
    std::array<float, 65536> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(half(uint16_t(i)));
    return values;
  }();
  return table[h];
}

// Unpack eight neighboring weights into registers. A codebook lookup serves eight products at once.
template <int Type>
FLYX_AVX2 float vector_dot(const uint8_t* row, const float* x, int cols) {
  __m256 acc = _mm256_setzero_ps();
  const __m256i lanes = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  for (int k = 0; k < cols; k += 8) {
    __m256 w;
    if constexpr (Type == FLYX_DT_BF16) {
      __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 2 * size_t(k)));
      w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16));
    } else if constexpr (Type == FLYX_DT_GGML_NVFP4 || Type == FLYX_DT_GGML_MXFP4) {
      constexpr bool nv = Type == FLYX_DT_GGML_NVFP4;
      const uint8_t* b = row + size_t(k / (nv ? 64 : 32)) * (nv ? 36 : 17);
      const int group = k % 64 / 16;
      const uint8_t* packed = nv ? b + 4 + group * 8 : b + 1 + (k & 8);
      uint64_t bytes;
      std::memcpy(&bytes, packed, 8);
      __m256i codes = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(bytes));
      if (k & (nv ? 8 : 16)) codes = _mm256_srli_epi32(codes, 4);
      codes = _mm256_and_si256(codes, _mm256_set1_epi32(15));
      static constexpr float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
      // MXFP4 stores an E8M0 scale; NVFP4 stores one E4M3 scale per 16.
      static const auto scales = [] {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i) {
          int v = i & 127;
          table[i] = nv ? (v == 127 ? 0.f
                           : v < 8  ? std::ldexp(float(v), -9)
                                    : std::ldexp(float(8 + (v & 7)), (v >> 3) - 10))
                        : std::ldexp(1.f, i - 127);
        }
        return table;
      }();
      w = _mm256_mul_ps(_mm256_i32gather_ps(values, codes, 4), _mm256_set1_ps(scales[b[nv ? group : 0]]));
    } else if constexpr (Type == FLYX_DT_GGML_IQ4_NL) {
      auto b = block_at<block_iq4_nl>(row, k / 32);
      uint64_t bytes;
      std::memcpy(&bytes, b.qs + (k & 8), 8);
      auto idx = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(bytes));
      if (k & 16) idx = _mm256_srli_epi32(idx, 4);
      idx = _mm256_and_si256(idx, _mm256_set1_epi32(15));
      static const auto magnitudes = [] {
        std::array<float, 16> values{};
        std::copy_n(kvalues_iq4nl, 16, values.begin());
        return values;
      }();
      w = _mm256_mul_ps(_mm256_i32gather_ps(magnitudes.data(), idx, 4), _mm256_set1_ps(half_scale(b.d)));
    } else if constexpr (Type == FLYX_DT_GGML_Q2_0) {
      auto b = block_at<block_q2_0>(row, k / 64);
      uint16_t word;
      std::memcpy(&word, b.qs + (k % 64) / 4, 2);
      auto code = _mm256_srlv_epi32(_mm256_set1_epi32(word), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
      code = _mm256_sub_epi32(_mm256_and_si256(code, _mm256_set1_epi32(3)), _mm256_set1_epi32(1));
      w = _mm256_mul_ps(_mm256_cvtepi32_ps(code), _mm256_set1_ps(half_scale(b.d)));
    } else {
      const int j = k % 256, g = j / 8, t = g % 4;
      uint64_t grid;
      int signs = 0;
      float scale, bias = 0;
      if constexpr (Type == FLYX_DT_GGML_IQ1_S) {
        auto b = block_at<block_iq1_s>(row, k / 256);
        const unsigned h = b.qh[g / 4];
        grid = iq1s_grid[b.qs[g] | (((h >> (3 * t)) & 7) << 8)];
        scale = half_scale(b.d) * (2 * ((h >> 12) & 7) + 1);
        bias = (h & 0x8000) ? -.125f : .125f;
      } else if constexpr (Type == FLYX_DT_GGML_IQ2_XXS) {
        auto b = block_at<block_iq2_xxs>(row, k / 256);
        const auto* p = reinterpret_cast<const uint8_t*>(b.qs) + 8 * (g / 4);
        const auto aux = u32_at(p + 4);
        signs = ksigns_iq2xs[(aux >> (7 * t)) & 127];
        grid = iq2xxs_grid[p[t]];
        scale = half_scale(b.d) * (.5f + (aux >> 28)) * .25f;
      } else if constexpr (Type == FLYX_DT_GGML_IQ2_XS) {
        auto b = block_at<block_iq2_xs>(row, k / 256);
        grid = iq2xs_grid[b.qs[g] & 511];
        signs = ksigns_iq2xs[b.qs[g] >> 9];
        scale = half_scale(b.d) * (.5f + ((b.scales[g / 4] >> (4 * (t / 2))) & 15)) * .25f;
      } else if constexpr (Type == FLYX_DT_GGML_IQ2_S) {
        auto b = block_at<block_iq2_s>(row, k / 256);
        grid = iq2s_grid[b.qs[g] | (((b.qh[g / 4] >> (2 * t)) & 3) << 8)];
        signs = b.qs[32 + g];
        scale = half_scale(b.d) * (.5f + ((b.scales[g / 4] >> (4 * (t / 2))) & 15)) * .25f;
      } else {
        auto b = block_at<block_iq3_xxs>(row, k / 256);
        auto aux = u32_at(b.qs + 64 + 4 * (g / 4));
        grid = uint64_t(iq3xxs_grid[b.qs[2 * g]]) | (uint64_t(iq3xxs_grid[b.qs[2 * g + 1]]) << 32);
        signs = ksigns_iq2xs[(aux >> (7 * t)) & 127];
        scale = half_scale(b.d) * (.5f + (aux >> 28)) * .5f;
      }
      const auto bytes = _mm_cvtsi64_si128(grid);
      if constexpr (Type == FLYX_DT_GGML_IQ1_S)
        w = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)), _mm256_set1_ps(bias));
      else {
        auto n = _mm256_cvtepu8_epi32(bytes);
        auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(lanes, _mm256_set1_epi32(signs)), lanes);
        w = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_xor_si256(n, mask), mask));
      }
      w = _mm256_mul_ps(w, _mm256_set1_ps(scale));
    }
    acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + k), acc);
  }
  alignas(32) float sums[8];
  _mm256_store_ps(sums, acc);
  float sum = 0;
  for (float v : sums) sum += v;
  return sum;
}
}  // namespace
#endif

float cold_dot(int dtype, const uint8_t* row, const float* input, int cols) {
#ifdef FLYX_AVX2
  static const bool simd = has_avx2();
  if (simd && cols % 8 == 0) {
    switch (dtype) {
#define FLYX_DOT(T) \
  case T:           \
    return vector_dot<T>(row, input, cols)
      FLYX_DOT(FLYX_DT_BF16);
      FLYX_DOT(FLYX_DT_GGML_NVFP4);
      FLYX_DOT(FLYX_DT_GGML_MXFP4);
      FLYX_DOT(FLYX_DT_GGML_Q2_0);
      FLYX_DOT(FLYX_DT_GGML_IQ4_NL);
      FLYX_DOT(FLYX_DT_GGML_IQ1_S);
      FLYX_DOT(FLYX_DT_GGML_IQ2_S);
      FLYX_DOT(FLYX_DT_GGML_IQ2_XS);
      FLYX_DOT(FLYX_DT_GGML_IQ2_XXS);
      FLYX_DOT(FLYX_DT_GGML_IQ3_XXS);
#undef FLYX_DOT
    }
  }
#endif
  return cold_dot_scalar(dtype, row, input, cols);
}
}  // namespace flyx::cuda
