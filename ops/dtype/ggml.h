// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block weights as float64: what weight i of a tensor stored in one of
// GGUF's quantized types is.  The block layouts and the constant tables (the
// IQ codebooks, sign tables, the non-linear 4-bit values) are ggml's, read
// from its own header (third_party/ggml); the arithmetic below
// is written from those layouts and checked value by value against the gguf
// Python package's decoder (tests/core/modelfile).  Every product here is
// exact in double, so a value is the encoding's exact meaning.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "flyx/backend_abi.h"

namespace flyx::ops::ggml {

// ggml's header as its authors wrote it, kept out of the global namespace.
// (In C++ it names a block's (d, m) and (d, dmin) pairs `data.data`.)
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml/ggml-common.h"

// binary16 bits -> double (exact).
inline double half(uint16_t bits) {
  const int e = (bits >> 10) & 31, m = bits & 1023;
  const double v = e == 0 ? std::ldexp(double(m), -24) : e == 31 ? (m ? NAN : INFINITY) : std::ldexp(1024.0 + m, e - 25);
  return (bits & 0x8000) ? -v : v;
}

template <class B>
B block_at(const uint8_t* data, size_t index) {
  B b;
  std::memcpy(&b, data + index * sizeof(B), sizeof(B));
  return b;
}
inline uint32_t u32_at(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
// Byte j of a codebook row: weight j of the group it encodes.
inline int grid_u8(uint64_t row, int j) { return int((row >> (8 * j)) & 0xFF); }
inline int grid_i8(uint64_t row, int j) { return int(int8_t((row >> (8 * j)) & 0xFF)); }
inline double sign_of(unsigned signs, int j) { return ((signs >> j) & 1) ? -1.0 : 1.0; }

// The 6-bit scale and minimum of group g (32 weights) of a Q4_K / Q5_K block.
inline void k_scale_min(const uint8_t* scales, int g, int* scale, int* minimum) {
  if (g < 4) {
    *scale = scales[g] & 63;
    *minimum = scales[g + 4] & 63;
  } else {
    *scale = (scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4);
    *minimum = (scales[g + 4] >> 4) | ((scales[g] >> 6) << 4);
  }
}

// Weight i of a tensor of GGML block type `dtype` whose blocks start at data.
inline double value(int32_t dtype, const uint8_t* data, size_t i) {
  const size_t per = size_t(flyx_block_values(dtype));
  if (!per) throw std::runtime_error("dtype " + std::to_string(dtype) + " is not a GGML block type");
  const size_t at = i / per;
  const int j = int(i % per);
  switch (dtype) {
    case FLYX_DT_GGML_Q4_0: {
      const auto b = block_at<block_q4_0>(data, at);
      const int q = j < 16 ? b.qs[j] & 15 : b.qs[j - 16] >> 4;
      return half(b.d) * (q - 8);
    }
    case FLYX_DT_GGML_Q4_1: {
      const auto b = block_at<block_q4_1>(data, at);
      const int q = j < 16 ? b.qs[j] & 15 : b.qs[j - 16] >> 4;
      return half(b.data.data.d) * q + half(b.data.data.m);
    }
    case FLYX_DT_GGML_Q5_0: {
      const auto b = block_at<block_q5_0>(data, at);
      const int q = (j < 16 ? b.qs[j] & 15 : b.qs[j - 16] >> 4) | int((u32_at(b.qh) >> j) & 1) << 4;
      return half(b.d) * (q - 16);
    }
    case FLYX_DT_GGML_Q5_1: {
      const auto b = block_at<block_q5_1>(data, at);
      const int q = (j < 16 ? b.qs[j] & 15 : b.qs[j - 16] >> 4) | int((u32_at(b.qh) >> j) & 1) << 4;
      return half(b.data.data.d) * q + half(b.data.data.m);
    }
    case FLYX_DT_GGML_Q8_0: {
      const auto b = block_at<block_q8_0>(data, at);
      return half(b.d) * b.qs[j];
    }
    case FLYX_DT_GGML_Q2_K: {  // 16 groups of 16: a 4-bit scale and minimum each, 2-bit codes
      const auto b = block_at<block_q2_K>(data, at);
      const int code = (b.qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
      return half(b.data.data.d) * (b.scales[j / 16] & 15) * code - half(b.data.data.dmin) * (b.scales[j / 16] >> 4);
    }
    case FLYX_DT_GGML_Q3_K: {  // 16 groups of 16: a 6-bit scale each, codes of 2 bits and a high bit
      const auto b = block_at<block_q3_K>(data, at);
      const int g = j / 16;
      const int scale = ((b.scales[g % 8] >> (4 * (g / 8))) & 15) | ((b.scales[8 + g % 4] >> (2 * (g / 4))) & 3) << 4;
      const int low = (b.qs[j / 128 * 32 + j % 32] >> (2 * (j % 128 / 32))) & 3;
      const int code = low - (((b.hmask[j % 32] >> (j / 32)) & 1) ? 0 : 4);
      return half(b.d) * (scale - 32) * code;
    }
    case FLYX_DT_GGML_Q4_K: {  // 8 groups of 32: a 6-bit scale and minimum each, 4-bit codes
      const auto b = block_at<block_q4_K>(data, at);
      const int g = j / 32, l = j % 32;
      int scale, minimum;
      k_scale_min(b.scales, g, &scale, &minimum);
      const int code = (b.qs[g / 2 * 32 + l] >> (4 * (g % 2))) & 15;
      return half(b.data.data.d) * scale * code - half(b.data.data.dmin) * minimum;
    }
    case FLYX_DT_GGML_Q5_K: {
      const auto b = block_at<block_q5_K>(data, at);
      const int g = j / 32, l = j % 32;
      int scale, minimum;
      k_scale_min(b.scales, g, &scale, &minimum);
      const int code = ((b.qs[g / 2 * 32 + l] >> (4 * (g % 2))) & 15) | ((b.qh[l] >> g) & 1) << 4;
      return half(b.data.data.d) * scale * code - half(b.data.data.dmin) * minimum;
    }
    case FLYX_DT_GGML_Q6_K: {  // 16 groups of 16: an 8-bit scale each, 6-bit codes
      const auto b = block_at<block_q6_K>(data, at);
      const int part = j / 128, g = j % 128 / 32, l = j % 32;
      const int low = (b.ql[part * 64 + g % 2 * 32 + l] >> (g / 2 * 4)) & 15;
      const int high = (b.qh[part * 32 + l] >> (g * 2)) & 3;
      return half(b.d) * b.scales[j / 16] * ((low | high << 4) - 32);
    }
    // The IQ2 family: 8 weights are one row of a codebook (magnitudes), their
    // signs 7 bits through ksigns_iq2xs or a byte, a 4-bit scale per 32 or 16.
    case FLYX_DT_GGML_IQ2_XXS: {
      const auto b = block_at<block_iq2_xxs>(data, at);
      const auto* group = reinterpret_cast<const uint8_t*>(b.qs) + 8 * (j / 32);  // 4 rows, then signs and scale
      const uint32_t rows = u32_at(group), aux = u32_at(group + 4);
      const int l = j % 32 / 8, e = j % 8;
      const unsigned signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
      return half(b.d) * (0.5 + (aux >> 28)) * 0.25 * grid_u8(iq2xxs_grid[(rows >> (8 * l)) & 255], e) * sign_of(signs, e);
    }
    case FLYX_DT_GGML_IQ2_XS: {
      const auto b = block_at<block_iq2_xs>(data, at);
      const unsigned v = b.qs[j / 8];
      const int scale = (b.scales[j / 32] >> (4 * (j % 32 / 16))) & 15, e = j % 8;
      return half(b.d) * (0.5 + scale) * 0.25 * grid_u8(iq2xs_grid[v & 511], e) * sign_of(ksigns_iq2xs[v >> 9], e);
    }
    case FLYX_DT_GGML_IQ2_S: {  // qs: 32 row bytes, then 32 sign bytes; qh: two more row bits per group
      const auto b = block_at<block_iq2_s>(data, at);
      const int g = j / 8, e = j % 8;
      const int row = b.qs[g] | ((b.qh[j / 32] >> (2 * (g % 4))) & 3) << 8;
      const int scale = (b.scales[j / 32] >> (4 * (j % 32 / 16))) & 15;
      return half(b.d) * (0.5 + scale) * 0.25 * grid_u8(iq2s_grid[row], e) * sign_of(b.qs[32 + g], e);
    }
    // The IQ3 family: 4 weights are one row of a codebook.
    case FLYX_DT_GGML_IQ3_XXS: {  // qs: 64 row bytes, then per 32 weights 4 x 7 sign bits and a 4-bit scale
      const auto b = block_at<block_iq3_xxs>(data, at);
      const uint32_t aux = u32_at(b.qs + 64 + 4 * (j / 32));
      const unsigned signs = ksigns_iq2xs[(aux >> (7 * (j % 32 / 8))) & 127];
      return half(b.d) * (0.5 + (aux >> 28)) * 0.5 * grid_u8(iq3xxs_grid[b.qs[j / 4]], j % 4) * sign_of(signs, j % 8);
    }
    case FLYX_DT_GGML_IQ3_S: {
      const auto b = block_at<block_iq3_s>(data, at);
      const int g = j / 4;
      const int row = b.qs[g] | ((b.qh[j / 32] >> (g % 8)) & 1) << 8;
      const int scale = (b.scales[j / 64] >> (4 * (j / 32 % 2))) & 15;
      return half(b.d) * (1 + 2 * scale) * grid_u8(iq3s_grid[row], j % 4) * sign_of(b.signs[j / 8], j % 8);
    }
    // The IQ1 family: 8 weights are one row of a codebook of -1, 0, 1, all shifted by +-1/8.
    case FLYX_DT_GGML_IQ1_S: {  // qh per 32 weights: 4 x 3 more row bits, a 3-bit scale, the shift's sign
      const auto b = block_at<block_iq1_s>(data, at);
      const unsigned h = b.qh[j / 32];
      const int row = b.qs[j / 8] | ((h >> (3 * (j % 32 / 8))) & 7) << 8;
      const double shift = (h & 0x8000) ? -0.125 : 0.125;
      return half(b.d) * (2 * ((h >> 12) & 7) + 1) * (grid_i8(iq1s_grid[row], j % 8) + shift);
    }
    case FLYX_DT_GGML_IQ1_M: {  // the block scale in the top nibbles of the four scale words; 3-bit scales per 16
      const auto b = block_at<block_iq1_m>(data, at);
      uint16_t sc[4];
      std::memcpy(sc, b.scales, 8);
      const double d = half(uint16_t((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)));
      const int ib = j / 32, scale = (sc[ib / 2] >> (6 * (ib % 2) + 3 * (j % 32 / 16))) & 7;
      const unsigned h = b.qh[j / 16] >> (4 * (j % 16 / 8));  // this group's nibble: 3 row bits, the shift's sign
      const int row = b.qs[j / 8] | (h & 7) << 8;
      const double shift = (h & 8) ? -0.125 : 0.125;
      return d * (2 * scale + 1) * (grid_i8(iq1s_grid[row], j % 8) + shift);
    }
    // IQ4: 4-bit codes through a table of 16 values.
    case FLYX_DT_GGML_IQ4_NL: {
      const auto b = block_at<block_iq4_nl>(data, at);
      return half(b.d) * kvalues_iq4nl[j < 16 ? b.qs[j] & 15 : b.qs[j - 16] >> 4];
    }
    case FLYX_DT_GGML_IQ4_XS: {  // a 6-bit scale per 32 weights
      const auto b = block_at<block_iq4_xs>(data, at);
      const int ib = j / 32, l = j % 32;
      const int scale = ((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 15) | ((b.scales_h >> (2 * ib)) & 3) << 4;
      const int code = l < 16 ? b.qs[16 * ib + l] & 15 : b.qs[16 * ib + l - 16] >> 4;
      return half(b.d) * (scale - 32) * kvalues_iq4nl[code];
    }
    // NVFP4: e2m1 codes (ggml's table holds them doubled) and an unsigned e4m3 scale per 16 values, which ggml reads
    // halved, with its sign bit ignored and 0x7F (e4m3's NaN) as zero; no scale per tensor.  In a group of 16 the low
    // nibbles of its 8 bytes are values 0..7, the high ones 8..15.
    case FLYX_DT_GGML_NVFP4: {
      const auto b = block_at<block_nvfp4>(data, at);
      const int g = j / 16, l = j % 16;
      const uint8_t q = b.qs[8 * g + (l & 7)], x = b.d[g];
      const int e = (x >> 3) & 15, m = x & 7;
      const double scale = (x & 0x7F) == 0x7F ? 0.0
                           : e == 0           ? std::ldexp(double(m), -9)
                                              : std::ldexp(double(8 + m), e - 10);
      return kvalues_mxfp4[l < 8 ? q & 15 : q >> 4] * scale * 0.5;
    }
    // MXFP4: 32 doubled e2m1 codes with one E8M0 scale, halved (ggml_e8m0_to_fp32_half).
    case FLYX_DT_GGML_MXFP4: {
      const auto b = block_at<block_mxfp4>(data, at);
      const uint32_t bits = b.e < 2 ? (0x00200000u << b.e) : (uint32_t(b.e - 1) << 23);
      float d = 0.f;
      std::memcpy(&d, &bits, 4);
      const uint8_t q = b.qs[j & 15];
      return double(d) * kvalues_mxfp4[j < 16 ? (q & 15) : (q >> 4)];
    }
    // Q2_0: 64 2-bit codes (00=-1, 01=0, 10=+1, 11=+2) with one binary16 scale.
    case FLYX_DT_GGML_Q2_0: {
      const auto b = block_at<block_q2_0>(data, at);
      const int code = (b.qs[j >> 2] >> ((j & 3) * 2)) & 3;
      return half(b.d) * double(code - 1);
    }
    default: throw std::runtime_error("GGML block type " + std::to_string(dtype) + " has no reference decoder");
  }
}

}  // namespace flyx::ops::ggml
