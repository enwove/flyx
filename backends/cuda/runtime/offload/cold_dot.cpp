// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_store.h"
#include "ops/dtype/ref.h"

#include <array>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define FLYX_AVX512 __declspec(noinline)
#define FLYX_VNNI __declspec(noinline)
#define FLYX_AVX2 __declspec(noinline)
#define FLYX_AVX_INLINE __forceinline
#else
#define FLYX_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,fma")))
#define FLYX_VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vnni,fma")))
#define FLYX_AVX2 __attribute__((target("avx2,fma")))
#define FLYX_AVX_INLINE inline __attribute__((target("avx2,fma"), always_inline))
#endif
#endif

namespace flyx::cuda {
using namespace ops::ggml;

void prepare_cold_q8(const float* input, ColdQ8* out, int cols) {
  for (int i = 0; i < cols; i += 32, ++out) {
    ops::round_q8_group(input + i, std::min(32, cols - i), out->q, out->d);
  }
}

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

const auto kFp4Codes = [] {
  std::array<float, 16> codes{};
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = .5f * kvalues_mxfp4[i];
  return codes;
}();

template <bool Nv>
const float* fp4_scales() {
  static const auto scales = [] {
    std::array<float, 256> table{};
    for (int i = 0; i < 256; ++i) {
      const int v = i & 127;
      if constexpr (Nv)
        table[i] = v == 127 ? 0.f : v < 8 ? std::ldexp(float(v), -9) : std::ldexp(float(8 + (v & 7)), (v >> 3) - 10);
      else
        table[i] = std::ldexp(1.f, i - 128);  // half scale keeps exponent 255 representable
    }
    return table;
  }();
  return scales.data();
}

bool has_avx2() {
#if defined(_MSC_VER) && !defined(__clang__)
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

const float* half_scales() {
  static const auto table = [] {
    std::array<float, 65536> values{};
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(half(uint16_t(i)));
    return values;
  }();
  return table.data();
}

bool has_avx512() {
#if defined(_MSC_VER) && !defined(__clang__)
  if (!has_avx2() || (_xgetbv(0) & 0xe6) != 0xe6) return false;
  int r[4];
  __cpuidex(r, 7, 0);
  constexpr unsigned needed = (1u << 16) | (1u << 17) | (1u << 30);
  return (unsigned(r[1]) & needed) == needed;
#else
  return has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512dq");
#endif
}

bool has_vnni() {
#if defined(_MSC_VER) && !defined(__clang__)
  int r[4];
  __cpuidex(r, 7, 0);
  return has_avx512() && (r[2] & (1 << 11));
#else
  return has_avx512() && __builtin_cpu_supports("avx512vnni");
#endif
}

uint64_t signed_grid(uint64_t grid, uint8_t signs) {
  static const auto masks = [] {
    std::array<uint64_t, 256> m{};
    for (int s = 0; s < 256; ++s)
      for (int i = 0; i < 8; ++i)
        if (s & (1 << i)) m[s] |= uint64_t(255) << (8 * i);
    return m;
  }();
  // All IQ2 and IQ3 grid magnitudes are positive, so the bytewise negations cannot carry.
  const uint64_t mask = masks[signs];
  return (grid ^ mask) + (mask & 0x0101010101010101ull);
}

uint64_t ternary_grid(unsigned index, bool minus) {
  static const auto table = [] {
    std::array<std::array<uint64_t, 2048>, 2> t{};
    for (int neg = 0; neg < 2; ++neg)
      for (int i = 0; i < 2048; ++i)
        for (int j = 0; j < 8; ++j) {
          int v = int(int8_t(iq1s_grid[i] >> (8 * j))) * 8 + (neg ? -1 : 1);
          t[neg][i] |= uint64_t(uint8_t(v)) << (j * 8);
        }
    return t;
  }();
  return table[minus][index];
}

struct Q8Weights {
  uint64_t codes[4];
  float scale[2];
};

template <int Type, class B>
inline Q8Weights q8_weights(const B& b, int group, const float* half) {
  Q8Weights w;
  constexpr bool halves = Type == FLYX_DT_GGML_IQ2_S || Type == FLYX_DT_GGML_IQ2_XS;
  const float d = half[b.d];
  if constexpr (halves) {
    const auto factors = b.scales[group];
    w.scale[0] = (.125f * d) * (2 * (factors & 15) + 1);
    w.scale[1] = (.125f * d) * (2 * (factors >> 4) + 1);
  }
  if constexpr (Type == FLYX_DT_GGML_IQ1_S) {
    const unsigned h = b.qh[group];
    w.scale[0] = w.scale[1] = (.125f * d) * (2 * ((h >> 12) & 7) + 1);
    for (int j = 0; j < 4; ++j) w.codes[j] = ternary_grid(b.qs[group * 4 + j] | (((h >> (3 * j)) & 7) << 8), h >> 15);
  } else if constexpr (Type == FLYX_DT_GGML_IQ2_S || Type == FLYX_DT_GGML_IQ2_XS) {
    for (int j = 0; j < 4; ++j) {
      int g = 4 * group + j;
      if constexpr (Type == FLYX_DT_GGML_IQ2_S)
        w.codes[j] = signed_grid(iq2s_grid[b.qs[g] | (((b.qh[group] >> (2 * j)) & 3) << 8)], b.qs[32 + g]);
      else
        w.codes[j] = signed_grid(iq2xs_grid[b.qs[g] & 511], ksigns_iq2xs[b.qs[g] >> 9]);
    }
  } else {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(b.qs);
    constexpr bool iq3 = Type == FLYX_DT_GGML_IQ3_XXS;
    const uint32_t aux = u32_at(p + (iq3 ? 64 + 4 * group : 8 * group + 4));
    w.scale[0] = w.scale[1] = (iq3 ? .25f : .125f) * d * (2 * (aux >> 28) + 1);
    for (int j = 0; j < 4; ++j) {
      uint64_t grid;
      if constexpr (iq3) {
        const uint8_t* ij = p + 8 * group + 2 * j;
        grid = uint64_t(iq3xxs_grid[ij[0]]) | (uint64_t(iq3xxs_grid[ij[1]]) << 32);
      } else
        grid = iq2xxs_grid[p[8 * group + j]];
      w.codes[j] = signed_grid(grid, ksigns_iq2xs[(aux >> (7 * j)) & 127]);
    }
  }
  return w;
}

template <int Type>
float iq_tail_dot(const uint8_t* row, const ColdQ8* x, int cols) {
  flyx_tensor_view v{};
  v.dtype = Type;
  v.data = row;
  float sum = 0;
  for (int i = 0; i < cols; ++i) sum = std::fma(float(ops::load(&v, i)), x[i / 32].d * x[i / 32].q[i % 32], sum);
  return sum;
}

template <int Type, class B>
FLYX_AVX2 float iq_dot_avx2(const uint8_t* row, const ColdQ8* x, int cols) {
  if (cols % 256) return iq_tail_dot<Type>(row, x, cols);
  const float* half = half_scales();
  __m256 acc = _mm256_setzero_ps();
  for (int k = 0; k < cols; k += 256) {
    const B b = block_at<B>(row, size_t(k / 256));
    for (int g = 0; g < 8; ++g, ++x) {
      const auto w = q8_weights<Type>(b, g, half);
      const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w.codes));
      const auto a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x->q));
      const auto p = _mm256_maddubs_epi16(_mm256_abs_epi8(v), _mm256_sign_epi8(a, v));
      const auto sum = _mm256_madd_epi16(p, _mm256_set1_epi16(1));
      const auto scale = _mm256_set_m128(_mm_set1_ps(x->d * w.scale[1]), _mm_set1_ps(x->d * w.scale[0]));
      acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum), scale, acc);
    }
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  float sum = 0;
  for (float v : lanes) sum += v;
  return sum;
}

template <int Type, class B>
FLYX_VNNI float iq_dot_vnni(const uint8_t* row, const ColdQ8* x, int cols) {
  if (cols % 256) return iq_tail_dot<Type>(row, x, cols);
  const float* half = half_scales();
  __m512 acc = _mm512_setzero_ps();
  for (int k = 0; k < cols; k += 256) {
    const B b = block_at<B>(row, size_t(k / 256));
    for (int g = 0; g < 8; g += 2, x += 2) {
      const auto lo = q8_weights<Type>(b, g, half);
      const auto hi = q8_weights<Type>(b, g + 1, half);
      const auto v = _mm512_set_epi64(hi.codes[3], hi.codes[2], hi.codes[1], hi.codes[0], lo.codes[3], lo.codes[2],
                                      lo.codes[1], lo.codes[0]);
      const auto a =
          _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x->q))),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[1].q)), 1);
      auto signed_a = _mm512_mask_sub_epi8(a, _mm512_movepi8_mask(v), _mm512_setzero_si512(), a);
      auto sum = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(v), signed_a);
      const float scales[] = {lo.scale[0] * x->d, lo.scale[1] * x->d, hi.scale[0] * x[1].d, hi.scale[1] * x[1].d};
      const auto scale = _mm512_permutexvar_ps(_mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
                                               _mm512_castps128_ps512(_mm_loadu_ps(scales)));
      acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sum), scale, acc);
    }
  }
  return _mm512_reduce_add_ps(acc);
}
// The eight codes share a scale. Byte loads use memcpy because GGUF rows need only block alignment.
FLYX_AVX_INLINE __m256i bytes8(const void* src) {
  uint64_t bytes;
  std::memcpy(&bytes, src, sizeof(bytes));
  return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(bytes));
}

FLYX_AVX_INLINE __m256i bits8(const uint8_t* src, int shift, int mask) {
  return _mm256_and_si256(_mm256_srl_epi32(bytes8(src), _mm_cvtsi32_si128(shift)), _mm256_set1_epi32(mask));
}

template <int Type, class Block>
FLYX_AVX_INLINE __m256 k_weights(const Block& b, int k, const float* half) {
  const int j = k % 256, group = j / 32, lane = j % 32, part = j / 128;
  __m256i code;
  float scale, minimum = 0;
  if constexpr (Type == FLYX_DT_GGML_Q2_K) {
    code = bits8(b.qs + part * 32 + lane, 2 * (group % 4), 3);
    scale = half[b.data.data.d] * (b.scales[j / 16] & 15);
    minimum = half[b.data.data.dmin] * (b.scales[j / 16] >> 4);
  } else if constexpr (Type == FLYX_DT_GGML_Q3_K) {
    const int g = j / 16;
    const int s = ((b.scales[g % 8] >> (4 * (g / 8))) & 15) | (((b.scales[8 + g % 4] >> (2 * (g / 4))) & 3) << 4);
    const auto low = bits8(b.qs + part * 32 + lane, 2 * (group % 4), 3);
    const auto high = _mm256_slli_epi32(bits8(b.hmask + lane, group, 1), 2);
    code = _mm256_sub_epi32(_mm256_add_epi32(low, high), _mm256_set1_epi32(4));
    scale = half[b.d] * (s - 32);
  } else if constexpr (Type == FLYX_DT_GGML_Q4_K || Type == FLYX_DT_GGML_Q5_K) {
    int s, m;
    k_scale_min(b.scales, group, &s, &m);
    code = bits8(b.qs + group / 2 * 32 + lane, 4 * (group % 2), 15);
    if constexpr (Type == FLYX_DT_GGML_Q5_K)
      code = _mm256_or_si256(code, _mm256_slli_epi32(bits8(b.qh + lane, group, 1), 4));
    scale = half[b.data.data.d] * s;
    minimum = half[b.data.data.dmin] * m;
  } else {
    const auto low = bits8(b.ql + part * 64 + group % 2 * 32 + lane, (group % 4) / 2 * 4, 15);
    const auto high = _mm256_slli_epi32(bits8(b.qh + part * 32 + lane, (group % 4) * 2, 3), 4);
    code = _mm256_sub_epi32(_mm256_or_si256(low, high), _mm256_set1_epi32(32));
    scale = half[b.d] * b.scales[j / 16];
  }
  return _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(code), _mm256_set1_ps(scale)), _mm256_set1_ps(minimum));
}

// Load each GGUF block once, then unpack eight neighboring weights per vector.
template <int Type, class Block, int Span>
FLYX_AVX2 float vector_dot(const uint8_t* row, const float* x, int cols) {
  if (cols % Span) return cold_dot_scalar(Type, row, x, cols);
  __m256 acc = _mm256_setzero_ps();
  const __m256i lanes = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
  const float* half = nullptr;
  if constexpr (Type != FLYX_DT_BF16 && Type != FLYX_DT_GGML_NVFP4 && Type != FLYX_DT_GGML_MXFP4) half = half_scales();
  for (int base = 0; base < cols; base += Span) {
    const auto b = block_at<Block>(row, size_t(base / Span));
    for (int k = 0; k < Span; k += 8) {
      __m256 w;
      if constexpr (Type == FLYX_DT_BF16) {
        __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b.data()));
        w = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
      } else if constexpr (Type == FLYX_DT_GGML_Q8_0) {
        uint64_t bytes;
        std::memcpy(&bytes, b.qs + k % 32, 8);
        const auto codes = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(bytes));
        w = _mm256_mul_ps(_mm256_cvtepi32_ps(codes), _mm256_set1_ps(half[b.d]));
      } else if constexpr (Type == FLYX_DT_GGML_Q2_K || Type == FLYX_DT_GGML_Q3_K || Type == FLYX_DT_GGML_Q4_K ||
                           Type == FLYX_DT_GGML_Q5_K || Type == FLYX_DT_GGML_Q6_K) {
        w = k_weights<Type>(b, k, half);
      } else if constexpr (Type == FLYX_DT_GGML_NVFP4 || Type == FLYX_DT_GGML_MXFP4) {
        constexpr bool nv = Type == FLYX_DT_GGML_NVFP4;
        const int group = k % 64 / 16;
        const uint8_t* packed = b.qs + (nv ? group * 8 : (k & 8));
        uint64_t bytes;
        std::memcpy(&bytes, packed, 8);
        __m256i codes = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(bytes));
        if (k & (nv ? 8 : 16)) codes = _mm256_srli_epi32(codes, 4);
        codes = _mm256_and_si256(codes, _mm256_set1_epi32(15));
        auto* scales = fp4_scales<nv>();
        float scale;
        if constexpr (nv)
          scale = scales[b.d[group]];
        else
          scale = scales[b.e];
        auto magnitudes = _mm256_i32gather_ps(kFp4Codes.data(), codes, 4);
        if constexpr (!nv) magnitudes = _mm256_add_ps(magnitudes, magnitudes);
        w = _mm256_mul_ps(magnitudes, _mm256_set1_ps(scale));
      } else if constexpr (Type == FLYX_DT_GGML_IQ4_NL) {
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
        w = _mm256_mul_ps(_mm256_i32gather_ps(magnitudes.data(), idx, 4), _mm256_set1_ps(half[b.d]));
      } else if constexpr (Type == FLYX_DT_GGML_Q2_0) {
        uint16_t word;
        std::memcpy(&word, b.qs + (k % 64) / 4, 2);
        auto code = _mm256_srlv_epi32(_mm256_set1_epi32(word), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
        code = _mm256_sub_epi32(_mm256_and_si256(code, _mm256_set1_epi32(3)), _mm256_set1_epi32(1));
        w = _mm256_mul_ps(_mm256_cvtepi32_ps(code), _mm256_set1_ps(half[b.d]));
      } else {
        const int j = k % 256, g = j / 8, t = g % 4;
        uint64_t grid;
        int signs = 0;
        float scale, bias = 0;
        if constexpr (Type == FLYX_DT_GGML_IQ1_S) {
          const unsigned h = b.qh[g / 4];
          grid = iq1s_grid[b.qs[g] | (((h >> (3 * t)) & 7) << 8)];
          scale = half[b.d] * (2 * ((h >> 12) & 7) + 1);
          bias = (h & 0x8000) ? -.125f : .125f;
        } else if constexpr (Type == FLYX_DT_GGML_IQ2_XXS) {
          const auto* p = reinterpret_cast<const uint8_t*>(b.qs) + 8 * (g / 4);
          const auto aux = u32_at(p + 4);
          signs = ksigns_iq2xs[(aux >> (7 * t)) & 127];
          grid = iq2xxs_grid[p[t]];
          scale = half[b.d] * (.5f + (aux >> 28)) * .25f;
        } else if constexpr (Type == FLYX_DT_GGML_IQ2_XS) {
          grid = iq2xs_grid[b.qs[g] & 511];
          signs = ksigns_iq2xs[b.qs[g] >> 9];
          scale = half[b.d] * (.5f + ((b.scales[g / 4] >> (4 * (t / 2))) & 15)) * .25f;
        } else if constexpr (Type == FLYX_DT_GGML_IQ2_S) {
          grid = iq2s_grid[b.qs[g] | (((b.qh[g / 4] >> (2 * t)) & 3) << 8)];
          signs = b.qs[32 + g];
          scale = half[b.d] * (.5f + ((b.scales[g / 4] >> (4 * (t / 2))) & 15)) * .25f;
        } else {
          auto aux = u32_at(b.qs + 64 + 4 * (g / 4));
          grid = uint64_t(iq3xxs_grid[b.qs[2 * g]]) | (uint64_t(iq3xxs_grid[b.qs[2 * g + 1]]) << 32);
          signs = ksigns_iq2xs[(aux >> (7 * t)) & 127];
          scale = half[b.d] * (.5f + (aux >> 28)) * .5f;
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
      acc = _mm256_fmadd_ps(w, _mm256_loadu_ps(x + base + k), acc);
    }
  }
  alignas(32) float sums[8];
  _mm256_store_ps(sums, acc);
  float sum = 0;
  for (float v : sums) sum += v;
  return sum;
}
// Sixteen adjacent FP4 weights share the activation load and fit the register lookup table.
template <int Type, class B, int Span>
FLYX_AVX512 float vector_dot_512(const uint8_t* row, const float* x, int cols) {
  __m512 lookup;
  if constexpr (Type == FLYX_DT_GGML_IQ4_NL)
    lookup = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl))));
  else
    lookup = _mm512_loadu_ps(kFp4Codes.data());
  const float* half = half_scales();
  __m512 acc = _mm512_setzero_ps();
  for (int k = 0; k < cols; k += Span) {
    const B b = block_at<B>(row, size_t(k / Span));
    for (int j = 0; j < Span; j += 16) {
      __m512 weights;
      float scale;
      if constexpr (Type == FLYX_DT_GGML_Q2_0) {
        uint32_t bits;
        std::memcpy(&bits, b.qs + j / 4, 4);
        const auto codes = _mm512_srlv_epi32(
            _mm512_set1_epi32(bits), _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30));
        weights =
            _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_si512(codes, _mm512_set1_epi32(3)), _mm512_set1_epi32(1)));
        scale = half[b.d];
      } else {
        __m512i codes;
        if constexpr (Type == FLYX_DT_GGML_NVFP4) {
          const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b.qs + j / 2));
          codes = _mm512_cvtepu8_epi32(_mm_unpacklo_epi64(bytes, _mm_srli_epi16(bytes, 4)));
          scale = fp4_scales<true>()[b.d[j / 16]];
        } else {
          codes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b.qs)));
          if (j) codes = _mm512_srli_epi32(codes, 4);
          if constexpr (Type == FLYX_DT_GGML_MXFP4)
            scale = fp4_scales<false>()[b.e];
          else
            scale = half[b.d];
        }
        weights = _mm512_permutexvar_ps(codes, lookup);
        if constexpr (Type == FLYX_DT_GGML_MXFP4) weights = _mm512_add_ps(weights, weights);
      }
      acc = _mm512_fmadd_ps(_mm512_mul_ps(weights, _mm512_set1_ps(scale)), _mm512_loadu_ps(x + k + j), acc);
    }
  }
  return _mm512_reduce_add_ps(acc);
}
}  // namespace
#endif

QuantDot cold_q8_kernel(int dtype, CpuIsa max_isa) {
#ifdef FLYX_AVX2
  static const bool avx2 = has_avx2(), vnni = has_vnni();
  if (avx2 && max_isa >= CpuIsa::avx2) {
#define FLYX_IQ_DOT(T, B) \
  case T:                 \
    return vnni && max_isa >= CpuIsa::avx512 ? iq_dot_vnni<T, B> : iq_dot_avx2<T, B>
    switch (dtype) {
      FLYX_IQ_DOT(FLYX_DT_GGML_IQ1_S, block_iq1_s);
      FLYX_IQ_DOT(FLYX_DT_GGML_IQ2_XXS, block_iq2_xxs);
      FLYX_IQ_DOT(FLYX_DT_GGML_IQ2_XS, block_iq2_xs);
      FLYX_IQ_DOT(FLYX_DT_GGML_IQ2_S, block_iq2_s);
      FLYX_IQ_DOT(FLYX_DT_GGML_IQ3_XXS, block_iq3_xxs);
    }
#undef FLYX_IQ_DOT
  }
#endif
  return nullptr;
}

FloatDot cold_float_kernel(int dtype, int cols, CpuIsa max_isa) {
#ifdef FLYX_AVX2
  static const bool simd = has_avx2();
  static const bool wide = has_avx512();
  if (wide && max_isa >= CpuIsa::avx512 && cols % 64 == 0) {
    switch (dtype) {
      case FLYX_DT_GGML_NVFP4:
        return vector_dot_512<FLYX_DT_GGML_NVFP4, block_nvfp4, 64>;
      case FLYX_DT_GGML_Q2_0:
        return vector_dot_512<FLYX_DT_GGML_Q2_0, block_q2_0, 64>;
      case FLYX_DT_GGML_MXFP4:
        return vector_dot_512<FLYX_DT_GGML_MXFP4, block_mxfp4, 32>;
      case FLYX_DT_GGML_IQ4_NL:
        return vector_dot_512<FLYX_DT_GGML_IQ4_NL, block_iq4_nl, 32>;
    }
  }
  if (simd && max_isa >= CpuIsa::avx2 && cols % 8 == 0) {
    switch (dtype) {
#define FLYX_DOT(T, B, N) \
  case T:                 \
    return vector_dot<T, B, N>
      using Bf16Eight = std::array<uint16_t, 8>;
      FLYX_DOT(FLYX_DT_BF16, Bf16Eight, 8);
      FLYX_DOT(FLYX_DT_GGML_Q2_K, block_q2_K, 256);
      FLYX_DOT(FLYX_DT_GGML_Q3_K, block_q3_K, 256);
      FLYX_DOT(FLYX_DT_GGML_Q4_K, block_q4_K, 256);
      FLYX_DOT(FLYX_DT_GGML_Q5_K, block_q5_K, 256);
      FLYX_DOT(FLYX_DT_GGML_Q6_K, block_q6_K, 256);
      FLYX_DOT(FLYX_DT_GGML_Q8_0, block_q8_0, 32);
      FLYX_DOT(FLYX_DT_GGML_NVFP4, block_nvfp4, 64);
      FLYX_DOT(FLYX_DT_GGML_MXFP4, block_mxfp4, 32);
      FLYX_DOT(FLYX_DT_GGML_Q2_0, block_q2_0, 64);
      FLYX_DOT(FLYX_DT_GGML_IQ4_NL, block_iq4_nl, 32);
      FLYX_DOT(FLYX_DT_GGML_IQ1_S, block_iq1_s, 256);
      FLYX_DOT(FLYX_DT_GGML_IQ2_S, block_iq2_s, 256);
      FLYX_DOT(FLYX_DT_GGML_IQ2_XS, block_iq2_xs, 256);
      FLYX_DOT(FLYX_DT_GGML_IQ2_XXS, block_iq2_xxs, 256);
      FLYX_DOT(FLYX_DT_GGML_IQ3_XXS, block_iq3_xxs, 256);
#undef FLYX_DOT
    }
  }
#endif
  return nullptr;
}
float cold_dot_limited(int dtype, const uint8_t* row, const float* input, int cols, CpuIsa limit) {
  const auto dot = cold_float_kernel(dtype, cols, limit);
  return dot ? dot(row, input, cols) : cold_dot_scalar(dtype, row, input, cols);
}
float cold_dot(int dtype, const uint8_t* row, const float* input, int cols) {
  return cold_dot_limited(dtype, row, input, cols, CpuIsa::avx512);
}
}  // namespace flyx::cuda
