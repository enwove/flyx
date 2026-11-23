// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGML block weights (GGUF files) kept packed on the device:
// the GEMMs read the blocks as the file stores them, 1.6 to 8.5 bits a
// weight, and decode them in registers.
//
// Every one of these types writes a weight as  d * s * q + minimum:  q a
// small integer (bit fields, or a row of a codebook with a sign per weight),
// s a few bits per group of 16 or 32 weights, d the block's binary16 scale.
// The product s * q is an integer of at most 11 significant bits for almost
// every type, which binary16 holds exactly; so the tensor cores multiply
// binary16 activations by the integers s * q, a block at a time from zero,
// and the block's d multiplies the sums in float32.  What does not fit one
// such product is a second one ("plane") with a coefficient of its own: a
// minimum (the group's minimum code as the weight, -dmin as the coefficient),
// and the two types whose s * q needs 12 bits (Q6_K, IQ4_XS: s in two parts).
//
// The activations: a row is rounded to bf16 as for every other weight
// format, then held as binary16 over a power of two that puts the row's
// largest value just under 2^16 (k_stage_*).  That is exact for every value
// within 2^-32 of the row's largest; smaller ones round to the subnormal
// grid.  The float64 reference rounds the same way (ops/linear/ref.h).
//
// A type is a struct below: what a lane of the tensor cores reads of a block
// (`load`, `share` / `lane`), eight weights of it as four words of binary16
// pairs (`words`), and the coefficients (`coef`).  The block layouts and the
// codebooks are ggml's own (third_party/ggml): the layouts are restated here
// only as offsets, each checked against its struct, and the codebooks are
// read on the host into the tables of gemm_blocks_tables.cu.  A kernel keeps
// the tables of its type in shared memory (global tables do not survive in
// the first-level cache next to the weights streaming through it).
//
// Shared by the kernels' files: gemm_blocks.cu (every pass with the inputs
// as binary16, and prefill passes with them in 8 bits) and
// gemm_blocks_round.cu (rounds with the inputs in 8 bits).
#pragma once

#include <cstddef>
#include <cstdint>

#include "backends/cuda/kernels/common/device.cuh"
#include "backends/cuda/kernels/gemm_internal.h"
#include "backends/cuda/kernels/kernels.cuh"

namespace flyx::cuda {

// ggml's header as its authors wrote it: here its block structs, to check the offsets below against.
namespace ggml {
#define GGML_COMMON_DECL_CUDA
#include "ggml/ggml-common.h"
}  // namespace ggml

namespace {

using namespace ggml;
using Tables = BlockTables;

// ---- loads, bytes, binary16 pairs ---------------------------------------------------------------
__device__ __forceinline__ uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ __forceinline__ uint2 ld64(const uint8_t* p) { return *reinterpret_cast<const uint2*>(p); }
__device__ __forceinline__ uint4 ld128(const uint8_t* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void put(uint32_t* w, uint2 v) { w[0] = v.x, w[1] = v.y; }
__device__ __forceinline__ void put(uint32_t* w, uint4 v) { w[0] = v.x, w[1] = v.y, w[2] = v.z, w[3] = v.w; }

// Four bytes picked from two words by a selector of four nibbles, each 0..7
// (the instruction itself: __byte_perm masks the selector first, and a
// selector from our tables never has bit 3 of a nibble set).
__device__ __forceinline__ uint32_t pick(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
  return d;
}
// Low bytes and high bytes of four binary16 values as two words of pairs.
__device__ __forceinline__ void interleave(uint32_t lo, uint32_t hi, uint32_t* w) {
  w[0] = __byte_perm(lo, hi, 0x5140u);
  w[1] = __byte_perm(lo, hi, 0x7362u);
}
__device__ __forceinline__ uint32_t byte_of(uint32_t w, int i) { return (w >> (8 * i)) & 0xFFu; }
// The low nibbles of bytes 0 and 1 of x as one byte (the first in its low nibble).
__device__ __forceinline__ uint32_t nibble_pair(uint32_t x) { return (x & 0x0Fu) | ((x >> 4) & 0xF0u); }
// Four 2-bit codes, one in the low bits of each byte of v (v & 0x03030303), as a byte (code i at bits 2 i).
__device__ __forceinline__ uint32_t two_bit_quad(uint32_t v) { return ((v * 0x41041u) >> 18) & 0xFFu; }
// Four bits, one in bit 0 of each byte of v (v & 0x01010101), as a nibble.
__device__ __forceinline__ uint32_t one_bit_quad(uint32_t v) { return (v * 0x01020408u) >> 24; }
// The bytes i, i + 2, i + 4, i + 6 (i = 0 or 1) of two words.
__device__ __forceinline__ uint32_t every_other_byte(uint32_t lo, uint32_t hi, int odd) {
  return __byte_perm(lo, hi, 0x6420u + 0x1111u * uint32_t(odd));
}
// binary16 pairs: sums, products (exact here: every result is an integer binary16 holds), one value twice.
__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  const __half2 r = __hadd2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t mul2(uint32_t a, uint32_t b) {
  const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&a), *reinterpret_cast<const __half2*>(&b));
  return *reinterpret_cast<const uint32_t*>(&r);
}
__device__ __forceinline__ uint32_t twice(uint32_t h) { return h * 0x10001u; }
// The pair (c if bit 0 of b, c if bit 8 of b), c a binary16 pattern; b & 0x0101.
__device__ __forceinline__ uint32_t pair_if_bytes(uint32_t b, uint32_t c) { return (b * (c * 0x101u)) & (c * 0x10001u); }
// The pair (c if bit 0 of b, c if bit 1 of b); b & 3.
__device__ __forceinline__ uint32_t pair_if_bits(uint32_t b, uint32_t c) { return ((b & 1u) | ((b & 2u) << 15)) * c; }
constexpr uint32_t kHalf16 = 0x4C00u, kHalfMinus16 = 0xCC00u, kOnes = 0x3C003C00u;  // 16, -16, (1, 1)

__device__ __forceinline__ float half_of(uint32_t bits) { return __half2float(__ushort_as_half(uint16_t(bits))); }
__device__ __forceinline__ float scale_at(const uint16_t* plane, size_t i) { return half_of(plane[i]); }

// D (+)= A B on mma.m16n8k16: binary16 inputs, float32 accumulators.
__device__ __forceinline__ void mma_f16(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D (+)= A B on mma.m16n8k32 and m16n8k16: int8 inputs, int32 sums (exact).
__device__ __forceinline__ void mma_s8_k32(int* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D = A B on mma.m16n8k32 from zero (the sums alone, no accumulator to set first): int8 inputs, int32 sums.
__device__ __forceinline__ void mma_s8_k32_zero(int* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(0));
}
// Integer sums of 8-bit products start from the bits of 1.5 * 2^23: what is added to them is then a float32's
// integer part (|sum| < 2^22), read back with one exact subtraction.  From sm_86 on a sum from zero is converted
// instead (I2FP, at the FMA pipe's rate there; sm_80's I2F runs at a quarter): the same float, exactly, since
// |sum| < 2^24 converts without rounding, and neither the accumulators' setting nor the subtraction is needed.
constexpr int kMagicBits = 0x4B400000;
constexpr float kMagic = 12582912.f;

// The rows of a round with inputs in 8 bits: round_q8_at (common/device.cuh).

__device__ __forceinline__ void mma_s8_k16(int* c, uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b));
}

// A table from global memory into the block's shared memory, all threads together (then a barrier).
template <class X>
__device__ __forceinline__ void fetch(X* local, const X* global, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) local[i] = global[i];
}

// int8 codes, four to a word (the kernel for inputs in 8 bits).
// Four values negated where `signs` has bit 7 of their byte set.
__device__ __forceinline__ uint32_t negate_where(uint32_t v, uint32_t signs) {
  const uint32_t m = ((signs >> 7) & 0x01010101u) * 0xFFu;
  return __vsub4(v ^ m, m);
}
// Four bits (bit i for byte i) as bit 4 of their bytes.
__device__ __forceinline__ uint32_t fifth_bits(uint32_t h) { return ((h * 0x00204081u) & 0x01010101u) << 4; }
// A word's four low nibbles (of its bytes) through a table of two values a byte of two nibbles.
__device__ __forceinline__ uint32_t nibbles_through(const uint16_t* pairs8, uint32_t x) {
  return uint32_t(pairs8[nibble_pair(x)]) | uint32_t(pairs8[nibble_pair(x >> 16)]) << 16;
}

// For a thread on its own (the rounds' kernel, inputs in 8 bits): 32 codes are eight words, word i weights
// 4 i .. 4 i + 3 of the group.
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
__device__ __forceinline__ uint32_t ld8(const uint8_t* p) { return *p; }
// Word i (a constant) of four.
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }
// 16 bytes of two nibbles, the low ones weights 0..15 and the high ones 16..31, less `bias` (a byte repeated).
__device__ __forceinline__ void nibble_codes(const uint4& q, uint32_t bias, uint32_t* w) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = __vsub4(word(q, i) & 0x0F0F0F0Fu, bias);
    w[4 + i] = __vsub4((word(q, i) >> 4) & 0x0F0F0F0Fu, bias);
  }
}
// ... with a fifth bit each (bit j of `high`: weight j), less `bias`.
__device__ __forceinline__ void fifth_codes(const uint32_t* q, uint32_t high, uint32_t bias, uint32_t* w) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    w[i] = __vsub4((q[i] & 0x0F0F0F0Fu) | fifth_bits((high >> (4 * i)) & 15u), bias);
    w[4 + i] = __vsub4(((q[i] >> 4) & 0x0F0F0F0Fu) | fifth_bits((high >> (16 + 4 * i)) & 15u), bias);
  }
}
// 32 bytes (weight j in byte j) of 2-bit codes at bit `shift`.
__device__ __forceinline__ void two_bit_codes(const uint4* q, int shift, uint32_t* w) {
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = (word(q[i >> 2], i & 3) >> shift) & 0x03030303u;
}
// 32 bytes of two nibbles (weight j in byte j), the nibble at bit `shift`.
__device__ __forceinline__ void nibble_half_codes(const uint4* q, int shift, uint32_t* w) {
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] = (word(q[i >> 2], i & 3) >> shift) & 0x0F0F0F0Fu;
}
// Q6_K's low bits, pair-major: four rows of 32 bytes (row c: groups c & 1 of half block c / 2), held so that a lane's
// 8 bytes of rows 2 h and 2 h + 1 (bytes 8 t .. 8 t + 7 of each) are 16 bytes together: byte j of row c at
// 64 (c / 2) + 16 (j / 8) + 8 (c % 2) + j % 8 (blocks_pack).  A lane of the tensor-core kernels loads a half block's
// in one load. (Q4_K's and Q5_K's codes, the same shape, gained nothing from it: RTX 5070 Ti.)
__host__ __device__ constexpr int k_pair_major(int c, int j) { return 64 * (c / 2) + 16 * (j / 8) + 8 * (c % 2) + j % 8; }
// Q4_K's and Q5_K's 6-bit scale and minimum of group G (0..7) from the block's 12 scale bytes (three words).
__device__ __forceinline__ void k_scale_min(const uint32_t* s, int G, uint32_t* scale, uint32_t* minimum) {
  const int at = 8 * (G & 3);
  const uint32_t x = (s[0] >> at) & 0xFFu, y = (s[1] >> at) & 0xFFu, z = (s[2] >> at) & 0xFFu;  // bytes G & 3 of 0.., 4.., 8..
  *scale = G < 4 ? (x & 63u) : ((z & 15u) | (x >> 6) << 4);
  *minimum = G < 4 ? (y & 63u) : ((z >> 4) | (y >> 6) << 4);
}

// ---- the block types ------------------------------------------------------------------------
// kValues weights a block; in the file kBytes bytes, of which the device keeps
// kStride from kSkip on (the payload: the block without the binary16 scales
// that lead or trail it, so that its fields are aligned; a few are padded
// for 16-byte loads) and the kScales binary16 values at kS0 (and kS1) in a
// plane of their own.  Offsets below are payload offsets.
//
// The tensor cores take a block in groups of 32 weights (two MMA steps): lane
// t of a column holds, of every group G, the 8 weights 32 G + 8 t ..
//   load(p, t)                what lane t loads itself of the block at p
//                             (loads only: the kernel issues them a block ahead)
//   share(raw, t, lane)       all that lane t reads of the block, the rest
//                             from its three neighbours' loads (shuffles)
//   lane(p, t)                the same for a thread on its own
//   words(L, ln, G, w0, w1)   its 8 weights of group G as four words of
//                             binary16 pairs: plane 0 and, with kPlanes == 2, plane 1
//   coef(p, d0, d1)           plane p's coefficient from the block's scales
// For the kernel whose inputs are 8-bit integers (FLYX_ACT_Q8):
//   codes(T, L, ln, G, w)     the lane's 8 codes c of group G as int8, four
//                             to a word, without their sub-group's scale
//   factors(ln, G, d0, d1, f, m)  the factors of the lane's sub-group of
//                             kSub weights: its weights are f c + m
// For a thread that takes a whole group on its own, inputs in 8 bits (the
// rounds' kernel, gemm_blocks_round.cu); G need not be a constant:
//   unit(p, G)                what the thread loads of the block at p for
//                             group G (loads only), a Unit
//   unit_codes(T, L, u, G, w) the group's 32 codes as int8: word i is
//                             weights 4 i .. 4 i + 3
//   unit_factors(u, G, d0, d1, f, m)  f[h], m[h] of its sub-groups of kSub
//                             weights (one of 32, or two of 16)
// kFlatRun: the second plane is one value per that many weights (a minimum,
// or ones under a minimum that is the plane's coefficient), else 0.
// Where the layout keeps a lane's bytes apart (the bit planes of the K
// types) `load` is the lane's own bytes; elsewhere it is a quarter of the
// block, and the four lanes of a column put the block together.

// Types whose lanes each load what they read.
#define FLYX_LANES_LOAD                                                                             \
  static __device__ __forceinline__ Lane share(const Raw& r, int t, int) { return build(r, t); }       \
  static __device__ __forceinline__ Lane lane(const uint8_t* p, int t) { return build(load(p, t), t); }

// A type with kHeadInScales keeps its block's head (the first 2 kScales bytes the file has: scales and their bits)
// in the scales plane and only its codes in the payload, which blocks_pack lays out for its loads (IQ4_XS: 16-byte
// aligned, lane-major); its load, lane and unit take the head's address as well.  The kernels reach every type's
// block `at` (of W, D) through these.
template <class F, class = void>
constexpr bool kHeadInScales = false;
template <class F>
constexpr bool kHeadInScales<F, std::void_t<decltype(F::kHeadInScales)>> = F::kHeadInScales;
template <class F>
__device__ __forceinline__ typename F::Raw block_load(const uint8_t* W, const uint16_t* D, size_t at, int t) {
  if constexpr (kHeadInScales<F>) return F::load(W + at * F::kStride, D + at * F::kScales, t);
  else return F::load(W + at * F::kStride, t);
}
template <class F>
__device__ __forceinline__ typename F::Lane block_lane(const uint8_t* W, const uint16_t* D, size_t at, int t) {
  if constexpr (kHeadInScales<F>) return F::build(F::load(W + at * F::kStride, D + at * F::kScales, t), t);
  else return F::lane(W + at * F::kStride, t);
}
template <class F>
__device__ __forceinline__ typename F::Unit block_unit(const uint8_t* W, const uint16_t* D, size_t at, int G) {
  if constexpr (kHeadInScales<F>) return F::unit(W + at * F::kStride, D + at * F::kScales, G);
  else return F::unit(W + at * F::kStride, G);
}
// Types whose lanes load a quarter of the block each: kOwn words (Raw::own), which `build` gets for all four
// quarters, quarter after quarter.
template <int N>
__device__ __forceinline__ void quarters(const uint32_t* own, int lane, uint32_t* all) {
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int i = 0; i < N; ++i) all[q * N + i] = __shfl_sync(0xffffffffu, own[i], (lane & ~3) | q);
}
#define FLYX_LANES_SHARE                                                                            \
  static __device__ __forceinline__ Lane share(const Raw& r, int t, int lane) {                        \
    uint32_t all[4 * kOwn];                                                                            \
    quarters<kOwn>(r.own, lane, all);                                                                  \
    return build(all, r, t);                                                                           \
  }                                                                                                    \
  static __device__ __forceinline__ Lane lane(const uint8_t* p, int t) {                               \
    uint32_t all[4 * kOwn];                                                                            \
    Raw r;                                                                                             \
    for (int q = 0; q < 4; ++q) {                                                                      \
      r = load(p, q);                                                                                  \
      for (int i = 0; i < kOwn; ++i) all[q * kOwn + i] = r.own[i];                                     \
    }                                                                                                  \
    return build(all, r, t);                                                                           \
  }

// ---- blocks of 32 with one scale ------------------------------------------------------------
// Weight j's 4-bit code is the low nibble of byte j for the first 16 weights,
// the high nibble of byte j - 16 for the rest.  Lane t: nibble t >> 1 of bytes 8 (t & 1) ..
struct Nibbles32 {
  uint32_t q[2];
  int shift;
};
__device__ __forceinline__ Nibbles32 nibbles32(const uint8_t* qs, int t) {
  const uint2 q = ld64(qs + 8 * (t & 1));
  return {{q.x, q.y}, 4 * (t >> 1)};
}
// Its 8 codes through a table of pairs (a byte of two codes -> their two values).
__device__ __forceinline__ void words_of_nibbles(const uint32_t* pairs, const Nibbles32& n, uint32_t* w) {
  const uint32_t x = n.q[0] >> n.shift, y = n.q[1] >> n.shift;
  w[0] = pairs[nibble_pair(x)];
  w[1] = pairs[nibble_pair(x >> 16)];
  w[2] = pairs[nibble_pair(y)];
  w[3] = pairs[nibble_pair(y >> 16)];
}
struct PairTable {
  uint32_t pairs[256];
};
struct Iq4Table {
  uint32_t pairs[256];
  uint16_t pairs8[256];  // the two values as int8
};
#define FLYX_NIBBLES32                                                                              \
  using Raw = Nibbles32;                                                                               \
  using Lane = Nibbles32;                                                                              \
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) { return nibbles32(p, t); }      \
  static __device__ __forceinline__ Lane build(const Raw& r, int) { return r; }                        \
  FLYX_LANES_LOAD

struct FQ4_0 {  // q = code - 8
  static constexpr int kDtype = FLYX_DT_GGML_Q4_0, kValues = 32, kBytes = 18, kSkip = 2, kStride = 16;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1;
  static constexpr int kFlatRun = 0;
  using Local = PairTable;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) { fetch(L.pairs, T.pair_s4, 256); }
  FLYX_NIBBLES32
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t*) {
    words_of_nibbles(L.pairs, n, w0);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int, uint32_t* w) {
    w[0] = __vsub4((n.q[0] >> n.shift) & 0x0F0F0F0Fu, 0x08080808u);
    w[1] = __vsub4((n.q[1] >> n.shift) & 0x0F0F0F0Fu, 0x08080808u);
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float, float* f, float* m) {
    *f = d0, *m = 0.f;
  }
  struct Unit {
    uint4 q;
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) { return {ld128(p)}; }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int, uint32_t* w) {
    nibble_codes(u.q, 0x08080808u, w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float, float* f, float* m) {
    f[0] = d0, m[0] = 0.f;
  }
};
static_assert(sizeof(block_q4_0) == 18 && offsetof(block_q4_0, qs) == 2, "block_q4_0");

struct FQ4_1 {  // q = code, and a minimum m per block: the second plane is all ones
  static constexpr int kDtype = FLYX_DT_GGML_Q4_1, kValues = 32, kBytes = 20, kSkip = 4, kStride = 16;
  static constexpr int kScales = 2, kS0 = 0, kS1 = 2, kPlanes = 2;
  static constexpr int kFlatRun = 32;  // the second plane: one value per 32 weights
  using Local = PairTable;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) { fetch(L.pairs, T.pair_u4, 256); }
  FLYX_NIBBLES32
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t* w1) {
    words_of_nibbles(L.pairs, n, w0);
    w1[0] = w1[1] = w1[2] = w1[3] = kOnes;
  }
  static __device__ __forceinline__ float coef(int p, float d0, float d1) { return p ? d1 : d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int, uint32_t* w) {
    w[0] = (n.q[0] >> n.shift) & 0x0F0F0F0Fu;
    w[1] = (n.q[1] >> n.shift) & 0x0F0F0F0Fu;
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float d1, float* f, float* m) {
    *f = d0, *m = d1;
  }
  struct Unit {
    uint4 q;
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) { return {ld128(p)}; }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int, uint32_t* w) {
    nibble_codes(u.q, 0u, w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float d1, float* f, float* m) {
    f[0] = d0, m[0] = d1;
  }
};
static_assert(sizeof(block_q4_1) == 20 && offsetof(block_q4_1, qs) == 4, "block_q4_1");

struct FIQ4_NL {  // q = the 16-value table's entry of the code
  static constexpr int kDtype = FLYX_DT_GGML_IQ4_NL, kValues = 32, kBytes = 18, kSkip = 2, kStride = 16;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1;
  static constexpr int kFlatRun = 0;
  using Local = Iq4Table;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.pairs, T.pair_iq4, 256);
    fetch(L.pairs8, T.pair_iq4_s8, 256);
  }
  FLYX_NIBBLES32
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t*) {
    words_of_nibbles(L.pairs, n, w0);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local& L, const Lane& n, int, uint32_t* w) {
    w[0] = nibbles_through(L.pairs8, n.q[0] >> n.shift);
    w[1] = nibbles_through(L.pairs8, n.q[1] >> n.shift);
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float, float* f, float* m) {
    *f = d0, *m = 0.f;
  }
  struct Unit {
    uint4 q;
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) { return {ld128(p)}; }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      w[i] = nibbles_through(L.pairs8, word(u.q, i));
      w[4 + i] = nibbles_through(L.pairs8, word(u.q, i) >> 4);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float, float* f, float* m) {
    f[0] = d0, m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq4_nl) == 18 && offsetof(block_iq4_nl, qs) == 2, "block_iq4_nl");

// 5-bit codes: the fifth bits in a word (bit j: weight j), then the nibbles as above.
struct Fifths32 {
  uint32_t q[2];
  uint32_t high;  // the fifth bits of the lane's 8 weights, from bit `at`
  int at, shift;
};
__device__ __forceinline__ Fifths32 fifths32(const uint8_t* p, int t) {  // the payload is 4-byte aligned: word loads
  const uint8_t* qs = p + 4 + 8 * (t & 1);
  return {{ld32(qs), ld32(qs + 4)}, ld32(p), 8 * t, 4 * (t >> 1)};
}
// code + 16 fifth: c = 16.  With `clear`, 16 fifth - 16 instead: c = -16 where the bit is clear.
__device__ __forceinline__ void words_of_fifths(const uint32_t* pairs, const Fifths32& n, bool clear, uint32_t c,
                                                uint32_t* w) {
  const uint32_t high = (n.high >> n.at) ^ (clear ? 0xFFu : 0u);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t x = n.q[i] >> n.shift, h = high >> (4 * i);
    w[2 * i] = add2(pairs[nibble_pair(x)], pair_if_bits(h & 3u, c));
    w[2 * i + 1] = add2(pairs[nibble_pair(x >> 16)], pair_if_bits((h >> 2) & 3u, c));
  }
}
// The 5-bit codes as int8, less `bias` (a byte repeated).
__device__ __forceinline__ void codes_of_fifths(const Fifths32& n, uint32_t bias, uint32_t* w) {
  const uint32_t high = n.high >> n.at;
#pragma unroll
  for (int i = 0; i < 2; ++i) w[i] = __vsub4(((n.q[i] >> n.shift) & 0x0F0F0F0Fu) | fifth_bits((high >> (4 * i)) & 15u), bias);
}
#define FLYX_FIFTHS32                                                                               \
  using Raw = Fifths32;                                                                                \
  using Lane = Fifths32;                                                                               \
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) { return fifths32(p, t); }       \
  static __device__ __forceinline__ Lane build(const Raw& r, int) { return r; }                        \
  FLYX_LANES_LOAD
struct FQ5_0 {  // q = code - 16
  static constexpr int kDtype = FLYX_DT_GGML_Q5_0, kValues = 32, kBytes = 22, kSkip = 2, kStride = 20;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1;
  static constexpr int kFlatRun = 0;
  using Local = PairTable;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) { fetch(L.pairs, T.pair_u4, 256); }
  FLYX_FIFTHS32
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t*) {
    words_of_fifths(L.pairs, n, true, kHalfMinus16, w0);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int, uint32_t* w) {
    codes_of_fifths(n, 0x10101010u, w);
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float, float* f, float* m) {
    *f = d0, *m = 0.f;
  }
  struct Unit {
    uint32_t high, q[4];
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) {  // the payload is 4-byte aligned: word loads
    return {ld32(p), {ld32(p + 4), ld32(p + 8), ld32(p + 12), ld32(p + 16)}};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int, uint32_t* w) {
    fifth_codes(u.q, u.high, 0x10101010u, w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float, float* f, float* m) {
    f[0] = d0, m[0] = 0.f;
  }
};
static_assert(sizeof(block_q5_0) == 22 && offsetof(block_q5_0, qh) == 2 && offsetof(block_q5_0, qs) == 6, "block_q5_0");
struct FQ5_1 {  // q = code, and a minimum
  static constexpr int kDtype = FLYX_DT_GGML_Q5_1, kValues = 32, kBytes = 24, kSkip = 4, kStride = 20;
  static constexpr int kScales = 2, kS0 = 0, kS1 = 2, kPlanes = 2;
  static constexpr int kFlatRun = 32;  // the second plane: one value per 32 weights
  using Local = PairTable;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) { fetch(L.pairs, T.pair_u4, 256); }
  FLYX_FIFTHS32
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t* w1) {
    words_of_fifths(L.pairs, n, false, kHalf16, w0);
    w1[0] = w1[1] = w1[2] = w1[3] = kOnes;
  }
  static __device__ __forceinline__ float coef(int p, float d0, float d1) { return p ? d1 : d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int, uint32_t* w) {
    codes_of_fifths(n, 0u, w);
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float d1, float* f, float* m) {
    *f = d0, *m = d1;
  }
  struct Unit {
    uint32_t high, q[4];
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) {
    return {ld32(p), {ld32(p + 4), ld32(p + 8), ld32(p + 12), ld32(p + 16)}};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int, uint32_t* w) {
    fifth_codes(u.q, u.high, 0u, w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float d1, float* f, float* m) {
    f[0] = d0, m[0] = d1;
  }
};
static_assert(sizeof(block_q5_1) == 24 && offsetof(block_q5_1, qh) == 4 && offsetof(block_q5_1, qs) == 8, "block_q5_1");

// 32 weights: a scale and a signed byte each.
struct FQ8_0 {
  static constexpr int kDtype = FLYX_DT_GGML_Q8_0, kValues = 32, kBytes = 34, kSkip = 2, kStride = 32;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint16_t value[256];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) { fetch(L.value, T.int8, 256); }
  struct Lane {
    uint32_t q[2];
  };
  using Raw = Lane;
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    const uint2 q = ld64(p + 8 * t);
    return {{q.x, q.y}};
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int) { return r; }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int, uint32_t* w0, uint32_t*) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t two = n.q[i >> 1] >> (16 * (i & 1));
      w0[i] = uint32_t(L.value[two & 0xFFu]) | uint32_t(L.value[(two >> 8) & 0xFFu]) << 16;
    }
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int, uint32_t* w) {
    w[0] = n.q[0], w[1] = n.q[1];
  }
  static __device__ __forceinline__ void factors(const Lane&, int, float d0, float, float* f, float* m) {
    *f = d0, *m = 0.f;
  }
  struct Unit {
    uint4 q[2];
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int) { return {{ld128(p), ld128(p + 16)}}; }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = word(u.q[i >> 2], i & 3);
  }
  static __device__ __forceinline__ void unit_factors(const Unit&, int, float d0, float, float* f, float* m) {
    f[0] = d0, m[0] = 0.f;
  }
};
static_assert(sizeof(block_q8_0) == 34 && offsetof(block_q8_0, qs) == 2, "block_q8_0");

// ---- the K types -------------------------------------------------------------------------------
// The bit planes of Q2_K, Q3_K and Q6_K: a byte of a half block (128 weights)
// holds weights l, l + 32, l + 64, l + 96 of it, 2 bits each.  Group G of the
// block is quarter G & 3 of half G >> 2; lane t: bits 2 (G & 3) of the half's
// bytes 8 t .. 8 t + 7, its own bytes.  A scale holds over 16 weights: of the
// block's 16, the lane's is 2 G + (t >> 1).

// Q2_K: a byte per 16 weights: a 4-bit scale and a 4-bit minimum; 2-bit codes.
struct FQ2_K {
  static constexpr int kDtype = FLYX_DT_GGML_Q2_K, kValues = 256, kBytes = 84, kSkip = 0, kStride = 80;
  static constexpr int kScales = 2, kS0 = 80, kS1 = 82, kPlanes = 2;
  static constexpr int kFlatRun = 16;  // the second plane: one value per 16 weights
  struct Local {
    uint2 quad[256];
    uint32_t twice[64];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.quad, T.quad_u2, 256);
    fetch(L.twice, T.twice_u6, 64);
  }
  struct Raw {
    uint32_t q[4], scales[4];
  };
  struct Lane {
    uint32_t q[2][2];
    uint32_t groups[2];  // byte i of [h]: the scale and minimum of group 4 h + i
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.q, ld64(p + 16 + 8 * t));
    put(r.q + 2, ld64(p + 48 + 8 * t));
    put(r.scales, ld128(p));
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int t) {
    return {{{r.q[0], r.q[1]}, {r.q[2], r.q[3]}},
            {every_other_byte(r.scales[0], r.scales[1], t >> 1), every_other_byte(r.scales[2], r.scales[3], t >> 1)}};
  }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t* w1) {
    const uint32_t g = byte_of(n.groups[G >> 2], G & 3), scale = L.twice[g & 15u], minimum = L.twice[g >> 4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint2 c = L.quad[two_bit_quad((n.q[G >> 2][i] >> (2 * (G & 3))) & 0x03030303u)];
      w0[2 * i] = mul2(c.x, scale);
      w0[2 * i + 1] = mul2(c.y, scale);
      w1[2 * i] = w1[2 * i + 1] = minimum;
    }
  }
  static __device__ __forceinline__ float coef(int p, float d0, float d1) { return p ? -d1 : d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int G, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 2; ++i) w[i] = (n.q[G >> 2][i] >> (2 * (G & 3))) & 0x03030303u;
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float d1, float* f, float* m) {
    const uint32_t g = byte_of(n.groups[G >> 2], G & 3);
    *f = d0 * float(g & 15u), *m = -d1 * float(g >> 4);
  }
  struct Unit {
    uint4 q[2];       // the 32 bytes of the group's half block
    uint32_t groups;  // the bytes of sub-groups 2 G and 2 G + 1: a scale and a minimum each
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint8_t* q = p + 16 + 32 * (G >> 2);
    return {{ld128(q), ld128(q + 16)}, ld16(p + 2 * G)};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int G, uint32_t* w) {
    two_bit_codes(u.q, 2 * (G & 3), w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float d1, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t g = (u.groups >> (8 * h)) & 0xFFu;
      f[h] = d0 * float(g & 15u), m[h] = -d1 * float(g >> 4);
    }
  }
};
static_assert(sizeof(block_q2_K) == 84 && offsetof(block_q2_K, scales) == 0 && offsetof(block_q2_K, qs) == 16, "block_q2_K");

// Q3_K: a 6-bit scale per 16 weights (less 32), codes of 2 bits and a mask bit (clear: 4 less; bit G of byte l
// of the 32 mask bytes is weight l of group G).  On the device its 108 payload bytes are kept in 112.
// Scale j of the block: its low 4 bits are nibble j >> 3 of byte j & 7 of the 12 scale bytes, its high 2 bits
// bits 2 (j >> 2) of byte 8 + (j & 3).
struct FQ3_K {
  static constexpr int kDtype = FLYX_DT_GGML_Q3_K, kValues = 256, kBytes = 110, kSkip = 0, kStride = 112;
  static constexpr int kScales = 1, kS0 = 108, kS1 = -1, kPlanes = 1;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint2 quad[256];
    uint2 masked[16];
    uint32_t twice[64];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.quad, T.quad_u2, 256);
    fetch(L.masked, T.quad_mask, 16);
    fetch(L.twice, T.twice_s6, 64);
  }
  struct Raw {
    uint32_t q[4], mask[2], low[2], high;
  };
  struct Lane {
    uint32_t q[2][2], mask[2];
    uint32_t low, high;  // of the lane's scales 2 G + (t >> 1): the bytes with their low bits, with their high bits
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.q, ld64(p + 32 + 8 * t));
    put(r.q + 2, ld64(p + 64 + 8 * t));
    put(r.mask, ld64(p + 8 * t));
    put(r.low, ld64(p + 96));
    r.high = ld32(p + 104);
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int t) {
    return {{{r.q[0], r.q[1]}, {r.q[2], r.q[3]}},
            {r.mask[0], r.mask[1]},
            every_other_byte(r.low[0], r.low[1], t >> 1),
            __byte_perm(r.high, r.high, 0x2020u + 0x1111u * uint32_t(t >> 1))};
  }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t s6 = ((n.low >> (8 * (G & 3) + 4 * (G >> 2))) & 15u) | ((n.high >> (8 * (G & 1) + 2 * (G >> 1))) & 3u) << 4;
    const uint32_t scale = L.twice[s6];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint2 c = L.quad[two_bit_quad((n.q[G >> 2][i] >> (2 * (G & 3))) & 0x03030303u)];
      const uint2 m = L.masked[one_bit_quad((n.mask[i] >> G) & 0x01010101u)];
      w0[2 * i] = mul2(add2(c.x, m.x), scale);
      w0[2 * i + 1] = mul2(add2(c.y, m.y), scale);
    }
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int G, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t c = (n.q[G >> 2][i] >> (2 * (G & 3))) & 0x03030303u, kept = (n.mask[i] >> G) & 0x01010101u;
      w[i] = __vsub4(c | (kept << 2), 0x04040404u);
    }
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    const uint32_t s6 = ((n.low >> (8 * (G & 3) + 4 * (G >> 2))) & 15u) | ((n.high >> (8 * (G & 1) + 2 * (G >> 1))) & 3u) << 4;
    *f = d0 * float(int(s6) - 32), *m = 0.f;
  }
  struct Unit {
    uint4 q[2], mask[2];
    uint32_t low, high;  // of scales 2 G and 2 G + 1: the bytes with their low bits, the bytes with their high bits
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint8_t* q = p + 32 + 32 * (G >> 2);
    return {{ld128(q), ld128(q + 16)}, {ld128(p), ld128(p + 16)}, ld16(p + 96 + 2 * (G & 3)), ld16(p + 104 + 2 * (G & 1))};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int G, uint32_t* w) {
    const int shift = 2 * (G & 3);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t c = (word(u.q[i >> 2], i & 3) >> shift) & 0x03030303u;
      const uint32_t kept = (word(u.mask[i >> 2], i & 3) >> G) & 0x01010101u;
      w[i] = __vsub4(c | (kept << 2), 0x04040404u);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t s6 = ((u.low >> (8 * h + 4 * (G >> 2))) & 15u) | ((u.high >> (8 * h + 2 * (G >> 1))) & 3u) << 4;
      f[h] = d0 * float(int(s6) - 32), m[h] = 0.f;
    }
  }
};
static_assert(sizeof(block_q3_K) == 110 && offsetof(block_q3_K, qs) == 32 && offsetof(block_q3_K, scales) == 96,
              "block_q3_K");

// Q4_K and Q5_K: 8 groups of 32 with a 6-bit scale and minimum each, packed in 12 bytes after the block's two
// binary16 scales (the whole block is kept: its fields are aligned).  Group G < 4: byte G and byte G + 4; else the
// nibbles of byte G + 4 with the top bits of bytes G - 4 and G.  Here for groups 2 c and 2 c + 1, a byte each:
// scale, scale, minimum, minimum.
__device__ __forceinline__ uint32_t k_groups(const uint32_t* head, int c) {  // head: the block's first 16 bytes
  uint32_t out = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int at = 8 * (2 * (c & 1) + i);
    const uint32_t x = head[1] >> at, y = head[2] >> at, z = head[3] >> at;  // bytes G & 3 of scales 0.., 4.., 8..
    const uint32_t scale = c < 2 ? (x & 63u) : ((z & 15u) | ((x >> 6) & 3u) << 4);
    const uint32_t minimum = c < 2 ? (y & 63u) : (((z >> 4) & 15u) | ((y >> 6) & 3u) << 4);
    out |= scale << (8 * i) | minimum << (16 + 8 * i);
  }
  return out;
}
struct KTables {
  uint32_t pairs[256];
  uint32_t twice[64];
};
// 4-bit codes: 64 weights are 32 bytes, the low nibbles then the high ones.  Group G: the nibbles G & 1 of bytes
// 32 (G >> 1) ..; lane t: bytes 8 t .. 8 t + 7 of those, its own.
struct FQ4_K {
  static constexpr int kDtype = FLYX_DT_GGML_Q4_K, kValues = 256, kBytes = 144, kSkip = 0, kStride = 144;
  static constexpr int kScales = 2, kS0 = 0, kS1 = 2, kPlanes = 2;
  static constexpr int kFlatRun = 32;  // the second plane: one value per 32 weights
  using Local = KTables;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.pairs, T.pair_u4, 256);
    fetch(L.twice, T.twice_u6, 64);
  }
  struct Raw {
    uint32_t q[8], head[4];
  };
  struct Lane {
    uint32_t q[8], groups[4];
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
#pragma unroll
    for (int c = 0; c < 4; ++c) put(r.q + 2 * c, ld64(p + 16 + 32 * c + 8 * t));
    put(r.head, ld128(p));
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int) {
    Lane n;
#pragma unroll
    for (int i = 0; i < 8; ++i) n.q[i] = r.q[i];
#pragma unroll
    for (int c = 0; c < 4; ++c) n.groups[c] = k_groups(r.head, c);
    return n;
  }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t* w1) {
    const uint32_t g = n.groups[G >> 1], scale = L.twice[byte_of(g, G & 1)], minimum = L.twice[byte_of(g, 2 + (G & 1))];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t x = n.q[2 * (G >> 1) + i] >> (4 * (G & 1));
      w0[2 * i] = mul2(L.pairs[nibble_pair(x)], scale);
      w0[2 * i + 1] = mul2(L.pairs[nibble_pair(x >> 16)], scale);
      w1[2 * i] = w1[2 * i + 1] = minimum;
    }
  }
  static __device__ __forceinline__ float coef(int p, float d0, float d1) { return p ? -d1 : d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int G, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 2; ++i) w[i] = (n.q[2 * (G >> 1) + i] >> (4 * (G & 1))) & 0x0F0F0F0Fu;
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float d1, float* f, float* m) {
    const uint32_t g = n.groups[G >> 1];
    *f = d0 * float(byte_of(g, G & 1)), *m = -d1 * float(byte_of(g, 2 + (G & 1)));
  }
  struct Unit {
    uint4 q[2];     // the 32 bytes whose low or high nibbles the group's codes are
    uint32_t s[3];  // the 12 scale bytes
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint8_t* q = p + 16 + 32 * (G >> 1);
    const uint4 head = ld128(p);
    return {{ld128(q), ld128(q + 16)}, {head.y, head.z, head.w}};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int G, uint32_t* w) {
    nibble_half_codes(u.q, 4 * (G & 1), w);
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float d1, float* f, float* m) {
    uint32_t scale, minimum;
    k_scale_min(u.s, G, &scale, &minimum);
    f[0] = d0 * float(scale), m[0] = -d1 * float(minimum);
  }
};
static_assert(sizeof(block_q4_K) == 144 && offsetof(block_q4_K, scales) == 4 && offsetof(block_q4_K, qs) == 16,
              "block_q4_K");

// As Q4_K with a fifth code bit (qh: bit G of byte l is weight l of group G of the block).
struct FQ5_K {
  static constexpr int kDtype = FLYX_DT_GGML_Q5_K, kValues = 256, kBytes = 176, kSkip = 0, kStride = 176;
  static constexpr int kScales = 2, kS0 = 0, kS1 = 2, kPlanes = 2;
  static constexpr int kFlatRun = 32;  // the second plane: one value per 32 weights
  using Local = KTables;
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.pairs, T.pair_u4, 256);
    fetch(L.twice, T.twice_u6, 64);
  }
  struct Raw {
    uint32_t q[8], high[2], head[4];
  };
  struct Lane {
    uint32_t q[8], high[2], groups[4];
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
#pragma unroll
    for (int c = 0; c < 4; ++c) put(r.q + 2 * c, ld64(p + 48 + 32 * c + 8 * t));
    put(r.high, ld64(p + 16 + 8 * t));
    put(r.head, ld128(p));
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int) {
    Lane n;
#pragma unroll
    for (int i = 0; i < 8; ++i) n.q[i] = r.q[i];
    n.high[0] = r.high[0], n.high[1] = r.high[1];
#pragma unroll
    for (int c = 0; c < 4; ++c) n.groups[c] = k_groups(r.head, c);
    return n;
  }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t* w1) {
    const uint32_t g = n.groups[G >> 1], scale = L.twice[byte_of(g, G & 1)], minimum = L.twice[byte_of(g, 2 + (G & 1))];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t x = n.q[2 * (G >> 1) + i] >> (4 * (G & 1)), h = (n.high[i] >> G) & 0x01010101u;
      w0[2 * i] = mul2(add2(L.pairs[nibble_pair(x)], pair_if_bytes(h & 0x0101u, kHalf16)), scale);
      w0[2 * i + 1] = mul2(add2(L.pairs[nibble_pair(x >> 16)], pair_if_bytes(h >> 16, kHalf16)), scale);
      w1[2 * i] = w1[2 * i + 1] = minimum;
    }
  }
  static __device__ __forceinline__ float coef(int p, float d0, float d1) { return p ? -d1 : d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int G, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
      w[i] = ((n.q[2 * (G >> 1) + i] >> (4 * (G & 1))) & 0x0F0F0F0Fu) | ((n.high[i] >> G) & 0x01010101u) << 4;
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float d1, float* f, float* m) {
    const uint32_t g = n.groups[G >> 1];
    *f = d0 * float(byte_of(g, G & 1)), *m = -d1 * float(byte_of(g, 2 + (G & 1)));
  }
  struct Unit {
    uint4 q[2], high[2];  // the 32 bytes of the group's nibbles; the 32 bytes of the fifth bits
    uint32_t s[3];        // the 12 scale bytes
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint8_t* q = p + 48 + 32 * (G >> 1);
    const uint4 head = ld128(p);
    return {{ld128(q), ld128(q + 16)}, {ld128(p + 16), ld128(p + 32)}, {head.y, head.z, head.w}};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int G, uint32_t* w) {
    nibble_half_codes(u.q, 4 * (G & 1), w);
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] |= ((word(u.high[i >> 2], i & 3) >> G) & 0x01010101u) << 4;
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float d1, float* f, float* m) {
    uint32_t scale, minimum;
    k_scale_min(u.s, G, &scale, &minimum);
    f[0] = d0 * float(scale), m[0] = -d1 * float(minimum);
  }
};
static_assert(sizeof(block_q5_K) == 176 && offsetof(block_q5_K, qh) == 16 && offsetof(block_q5_K, qs) == 48, "block_q5_K");

// Q6_K: a signed 8-bit scale per 16 weights; 6-bit codes (4 low bits, 2 high bits), q = code - 32.  The low bits'
// four 32-byte rows are held pair-major (k_pair_major, as Q4_K's chunks).
// scale * q can need 12 bits, so the scale goes in two parts: 16 hi + lo (hi -8..7, lo 0..15), a plane each.
// Quarter c of a half block: the nibbles c >> 1 of the half's ql bytes 32 (c & 1) .., bits 2 c of its qh bytes.
struct FQ6_K {
  static constexpr int kDtype = FLYX_DT_GGML_Q6_K, kValues = 256, kBytes = 210, kSkip = 0, kStride = 208;
  static constexpr int kScales = 1, kS0 = 208, kS1 = -1, kPlanes = 2;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint2 high[256];
    uint32_t pairs[256];
    uint32_t parts[256];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.high, T.quad_hi6, 256);
    fetch(L.pairs, T.pair_u4, 256);
    fetch(L.parts, T.parts_s8, 256);
  }
  struct Raw {
    uint32_t low[8], high[4], scales[4];
  };
  struct Lane {
    uint32_t low[2][2][2], high[2][2];
    uint32_t groups[2];  // byte i of [h]: the scale of group 4 h + i
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
#pragma unroll
    for (int h = 0; h < 2; ++h) put(r.low + 4 * h, ld128(p + 64 * h + 16 * t));  // rows 2 h, 2 h + 1: pair-major
    put(r.high, ld64(p + 128 + 8 * t));
    put(r.high + 2, ld64(p + 160 + 8 * t));
    put(r.scales, ld128(p + 192));
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int t) {
    Lane n;
#pragma unroll
    for (int j = 0; j < 8; ++j) n.low[j >> 2][(j >> 1) & 1][j & 1] = r.low[j];
#pragma unroll
    for (int j = 0; j < 4; ++j) n.high[j >> 1][j & 1] = r.high[j];
    n.groups[0] = every_other_byte(r.scales[0], r.scales[1], t >> 1);
    n.groups[1] = every_other_byte(r.scales[2], r.scales[3], t >> 1);
    return n;
  }
  FLYX_LANES_LOAD
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t* w1) {
    const int half = G >> 2, c = G & 3;
    const uint32_t parts = L.parts[byte_of(n.groups[half], c)], hi = twice(parts & 0xFFFFu), lo = twice(parts >> 16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t x = n.low[half][c & 1][i] >> (4 * (c >> 1));
      const uint2 h = L.high[two_bit_quad((n.high[half][i] >> (2 * c)) & 0x03030303u)];
      const uint32_t q0 = add2(L.pairs[nibble_pair(x)], h.x), q1 = add2(L.pairs[nibble_pair(x >> 16)], h.y);
      w0[2 * i] = mul2(q0, hi), w0[2 * i + 1] = mul2(q1, hi);
      w1[2 * i] = mul2(q0, lo), w1[2 * i + 1] = mul2(q1, lo);
    }
  }
  static __device__ __forceinline__ float coef(int p, float d0, float) { return p ? d0 : 16.f * d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables&, const Local&, const Lane& n, int G, uint32_t* w) {
    const int half = G >> 2, c = G & 3;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t x = (n.low[half][c & 1][i] >> (4 * (c >> 1))) & 0x0F0F0F0Fu;
      w[i] = __vsub4(x | ((n.high[half][i] >> (2 * c)) & 0x03030303u) << 4, 0x20202020u);
    }
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = d0 * float(int(int8_t(byte_of(n.groups[G >> 2], G & 3)))), *m = 0.f;
  }
  struct Unit {
    uint4 low[2], high[2];  // the 32 bytes of the group's low bits (a nibble), of its high bits (two)
    uint32_t scales;        // the signed scales of sub-groups 2 G and 2 G + 1
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {  // low row 2 (G >> 2) + (G & 1), pair-major
    const uint8_t* lo = p + 64 * (G >> 2) + 8 * (G & 1);
    const uint8_t* hi = p + 128 + 32 * (G >> 2);
    const uint2 a = ld64(lo), b = ld64(lo + 16), c = ld64(lo + 32), d = ld64(lo + 48);
    return {{make_uint4(a.x, a.y, b.x, b.y), make_uint4(c.x, c.y, d.x, d.y)},
            {ld128(hi), ld128(hi + 16)},
            ld16(p + 192 + 2 * G)};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local&, const Unit& u, int G, uint32_t* w) {
    const int c = G & 3;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t x = (word(u.low[i >> 2], i & 3) >> (4 * (c >> 1))) & 0x0F0F0F0Fu;
      w[i] = __vsub4(x | ((word(u.high[i >> 2], i & 3) >> (2 * c)) & 0x03030303u) << 4, 0x20202020u);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) f[h] = d0 * float(int(int8_t((u.scales >> (8 * h)) & 0xFFu))), m[h] = 0.f;
  }
};
static_assert(sizeof(block_q6_K) == 210 && offsetof(block_q6_K, qh) == 128 && offsetof(block_q6_K, scales) == 192,
              "block_q6_K");

// ---- the IQ types --------------------------------------------------------------------------------
// A codebook row is the selectors of its weights' magnitudes, a nibble each.
// Four selectors become two words through byte tables of the magnitudes
// times the group's scale (binary16, high and low bytes).
// Three magnitudes (IQ2), the sign in bit 2 of the selector: tb = {high bytes, high bytes negated, low bytes}.
__device__ __forceinline__ void words_of_signed(uint32_t sel, uint4 tb, uint32_t* w) {
  interleave(pick(tb.z, tb.z, sel), pick(tb.x, tb.y, sel), w);
}
// Eight magnitudes (IQ3), the signs as a mask of the high bytes: tb = {high 0..3, high 4..7, low 0..3, low 4..7}.
__device__ __forceinline__ void words_of_masked(uint32_t sel, uint4 tb, uint32_t signs, uint32_t* w) {
  interleave(pick(tb.z, tb.w, sel), pick(tb.x, tb.y, sel) ^ signs, w);
}
// -1, 0, 1 shifted by an eighth (IQ1): tb = {high bytes, low bytes} of the three values.
__device__ __forceinline__ void words_of_ternary(uint32_t sel, uint2 tb, uint32_t* w) {
  interleave(pick(tb.y, tb.y, sel), pick(tb.x, tb.x, sel), w);
}

// The IQ2 family: 8 weights are a codebook row of magnitudes with a sign each;
// a 4-bit scale s per 32 or 16 weights, the scale d (2 s + 1) / 8.
// IQ2_XXS, per 32 weights: 4 row bytes, then a word of 4 x 7 sign-code bits and s.
struct FIQ2_XXS {
  static constexpr int kDtype = FLYX_DT_GGML_IQ2_XXS, kValues = 256, kBytes = 66, kSkip = 2, kStride = 64;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 4;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint4 scaled[16];
    uint32_t rows[256], signs[128];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq2_scaled, 16);
    fetch(L.rows, T.iq2xxs, 256);
    fetch(L.signs, T.ksign_sel, 128);
  }
  struct Raw {
    uint32_t own[kOwn];
  };
  struct Lane {
    uint32_t q[16];  // per group: the rows, the signs and s
    int row, sign;   // where the lane's row byte and sign code start
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.own, ld128(p + 16 * t));
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw&, int t) {
    Lane n;
#pragma unroll
    for (int i = 0; i < 16; ++i) n.q[i] = all[i];
    n.row = 8 * t, n.sign = 7 * t;
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t aux = n.q[2 * G + 1];
    const uint32_t sel = L.rows[(n.q[2 * G] >> n.row) & 255u] | L.signs[(aux >> n.sign) & 127u];
    const uint4 tb = L.scaled[aux >> 28];
    words_of_signed(sel, tb, w0);
    words_of_signed(sel >> 16, tb, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.125f * d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t aux = n.q[2 * G + 1];
    const uint32_t sel = L.rows[(n.q[2 * G] >> n.row) & 255u] | L.signs[(aux >> n.sign) & 127u];
    w[0] = pick(T.iq2_s8[0][0], T.iq2_s8[0][1], sel);
    w[1] = pick(T.iq2_s8[0][0], T.iq2_s8[0][1], sel >> 16);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = 0.125f * d0 * float(2 * (n.q[2 * G + 1] >> 28) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows, aux;  // the group's four row bytes; its four sign codes and s
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint2 v = ld64(p + 8 * G);
    return {v.x, v.y};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t sel = L.rows[(u.rows >> (8 * j)) & 255u] | L.signs[(u.aux >> (7 * j)) & 127u];
      w[2 * j] = pick(T.iq2_s8[0][0], T.iq2_s8[0][1], sel);
      w[2 * j + 1] = pick(T.iq2_s8[0][0], T.iq2_s8[0][1], sel >> 16);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
    f[0] = 0.125f * d0 * float(2 * (u.aux >> 28) + 1), m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq2_xxs) == 66 && offsetof(block_iq2_xxs, qs) == 2, "block_iq2_xxs");

// IQ2_XS: per 8 weights 16 bits: a 9-bit row and a 7-bit sign code; then a byte per 32 weights: two scales.
struct FIQ2_XS {
  static constexpr int kDtype = FLYX_DT_GGML_IQ2_XS, kValues = 256, kBytes = 74, kSkip = 2, kStride = 72;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 4;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint4 scaled[16];
    uint32_t rows[512], signs[128];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq2_scaled + 16, 16);
    fetch(L.rows, T.iq2xs, 512);
    fetch(L.signs, T.ksign_sel, 128);
  }
  struct Raw {
    uint32_t own[kOwn], scales[2];
  };
  struct Lane {
    uint32_t code[8];    // the lane's 16 bits of each group
    uint32_t scales[2];  // a byte per group
    int scale;           // where the lane's nibble of it starts
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {  // the payload is 8-byte aligned
    Raw r;
    put(r.own, ld64(p + 16 * t));
    put(r.own + 2, ld64(p + 16 * t + 8));
    put(r.scales, ld64(p + 64));
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw& r, int t) {
    Lane n;
#pragma unroll
    for (int G = 0; G < 8; ++G) n.code[G] = (((t & 2) ? all[2 * G + 1] : all[2 * G]) >> (16 * (t & 1))) & 0xFFFFu;
    n.scales[0] = r.scales[0], n.scales[1] = r.scales[1];
    n.scale = 4 * (t >> 1);
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t v = n.code[G];
    const uint32_t sel = L.rows[v & 511u] | L.signs[v >> 9];
    const uint4 tb = L.scaled[(byte_of(n.scales[G >> 2], G & 3) >> n.scale) & 15u];
    words_of_signed(sel, tb, w0);
    words_of_signed(sel >> 16, tb, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.125f * d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t v = n.code[G];
    const uint32_t sel = L.rows[v & 511u] | L.signs[v >> 9];
    w[0] = pick(T.iq2_s8[1][0], T.iq2_s8[1][1], sel);
    w[1] = pick(T.iq2_s8[1][0], T.iq2_s8[1][1], sel >> 16);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = 0.125f * d0 * float(2 * ((byte_of(n.scales[G >> 2], G & 3) >> n.scale) & 15u) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t code[2];  // 16 bits per 8 weights: a 9-bit row, a 7-bit sign code
    uint32_t scales;   // the group's byte: the scales of its two sub-groups
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint2 v = ld64(p + 8 * G);
    return {{v.x, v.y}, ld8(p + 64 + G)};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t v = (u.code[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
      const uint32_t sel = L.rows[v & 511u] | L.signs[v >> 9];
      w[2 * j] = pick(T.iq2_s8[1][0], T.iq2_s8[1][1], sel);
      w[2 * j + 1] = pick(T.iq2_s8[1][0], T.iq2_s8[1][1], sel >> 16);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) f[h] = 0.125f * d0 * float(2 * ((u.scales >> (4 * h)) & 15u) + 1), m[h] = 0.f;
  }
};
static_assert(sizeof(block_iq2_xs) == 74 && offsetof(block_iq2_xs, qs) == 2 && offsetof(block_iq2_xs, scales) == 66,
              "block_iq2_xs");

// IQ2_S: per 8 weights a row byte (32 of them), then a sign byte (32); qh: two more row bits per 8 weights
// (a byte per 32); scales: a byte per 32 weights, two scales.
struct FIQ2_S {
  static constexpr int kDtype = FLYX_DT_GGML_IQ2_S, kValues = 256, kBytes = 82, kSkip = 2, kStride = 80;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 4;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint4 scaled[16];
    uint32_t rows[1024], signs[256];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq2_scaled + 32, 16);
    fetch(L.rows, T.iq2s, 1024);
    fetch(L.signs, T.sign_sel, 256);
  }
  struct Raw {
    uint32_t own[kOwn], tail[4];  // a quarter's row and sign bytes; the block's qh and scales
  };
  struct Lane {
    uint32_t rows[8], signs[8], more[2], scales[2];  // a word, or a byte, per group
    int byte, bits, scale;                           // where the lane's parts start
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.own, ld64(p + 8 * t));
    put(r.own + 2, ld64(p + 32 + 8 * t));
    put(r.tail, ld128(p + 64));
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw& r, int t) {
    Lane n;
#pragma unroll
    for (int G = 0; G < 8; ++G) n.rows[G] = all[4 * (G >> 1) + (G & 1)], n.signs[G] = all[4 * (G >> 1) + 2 + (G & 1)];
    n.more[0] = r.tail[0], n.more[1] = r.tail[1], n.scales[0] = r.tail[2], n.scales[1] = r.tail[3];
    n.byte = 8 * t, n.bits = 2 * t, n.scale = 4 * (t >> 1);
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t row = ((n.rows[G] >> n.byte) & 255u) | ((byte_of(n.more[G >> 2], G & 3) >> n.bits) & 3u) << 8;
    const uint32_t sel = L.rows[row] | L.signs[(n.signs[G] >> n.byte) & 255u];
    const uint4 tb = L.scaled[(byte_of(n.scales[G >> 2], G & 3) >> n.scale) & 15u];
    words_of_signed(sel, tb, w0);
    words_of_signed(sel >> 16, tb, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.125f * d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t row = ((n.rows[G] >> n.byte) & 255u) | ((byte_of(n.more[G >> 2], G & 3) >> n.bits) & 3u) << 8;
    const uint32_t sel = L.rows[row] | L.signs[(n.signs[G] >> n.byte) & 255u];
    w[0] = pick(T.iq2_s8[2][0], T.iq2_s8[2][1], sel);
    w[1] = pick(T.iq2_s8[2][0], T.iq2_s8[2][1], sel >> 16);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = 0.125f * d0 * float(2 * ((byte_of(n.scales[G >> 2], G & 3) >> n.scale) & 15u) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows, signs;  // the group's four row bytes, four sign bytes
    uint32_t more, scales; // its qh byte (two more row bits per 8 weights), its scale byte (two sub-groups)
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    return {ld32(p + 4 * G), ld32(p + 32 + 4 * G), ld8(p + 64 + G), ld8(p + 72 + G)};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t row = ((u.rows >> (8 * j)) & 255u) | ((u.more >> (2 * j)) & 3u) << 8;
      const uint32_t sel = L.rows[row] | L.signs[(u.signs >> (8 * j)) & 255u];
      w[2 * j] = pick(T.iq2_s8[2][0], T.iq2_s8[2][1], sel);
      w[2 * j + 1] = pick(T.iq2_s8[2][0], T.iq2_s8[2][1], sel >> 16);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) f[h] = 0.125f * d0 * float(2 * ((u.scales >> (4 * h)) & 15u) + 1), m[h] = 0.f;
  }
};
static_assert(sizeof(block_iq2_s) == 82 && offsetof(block_iq2_s, qs) == 2 && offsetof(block_iq2_s, qh) == 66 &&
                  offsetof(block_iq2_s, scales) == 74,
              "block_iq2_s");

// The lane's two row bytes of every group, of a block's 64 row bytes (a byte per 4 weights) as 16 words.
__device__ __forceinline__ void two_rows(const uint32_t* rows, int t, uint32_t* out) {
#pragma unroll
  for (int G = 0; G < 8; ++G) out[G] = ((t & 2) ? rows[2 * G + 1] : rows[2 * G]) >> (16 * (t & 1));
}

// IQ3_XXS: a row byte per 4 weights (64 bytes), then per 32 weights a word of 4 x 7 sign-code bits and s; the
// scale d (2 s + 1) / 4.
struct FIQ3_XXS {
  static constexpr int kDtype = FLYX_DT_GGML_IQ3_XXS, kValues = 256, kBytes = 98, kSkip = 2, kStride = 96;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 6;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint4 scaled[16];
    uint2 signs[128];
    uint16_t rows[256];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq3xxs_scaled, 16);
    fetch(L.signs, T.ksign_hi, 128);
    fetch(L.rows, T.iq3xxs, 256);
  }
  // The tables of the kernel that keeps the round's rows in dynamic shared memory (k_round_blocks_split): also a row
  // byte with its four sign bits (bit b: value b negative) -> the four values, signed: codes() through one load (16 KB;
  // the other kernels, dp4a's above all, keep their shared memory small).
  struct MmaLocal : Local {
    uint32_t signed_rows[4096];
  };
  static __device__ __forceinline__ void mma_tables(const Tables& T, MmaLocal& L) {
    tables(T, L);
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
      const uint32_t s = uint32_t(i) >> 8;
      const uint32_t m = (s & 1u) * 0xFFu | (s >> 1 & 1u) * 0xFF00u | (s >> 2 & 1u) * 0xFF0000u | (s >> 3) * 0xFF000000u;
      L.signed_rows[i] = __vsub4(pick(T.iq3_s8[0][0], T.iq3_s8[0][1], T.iq3xxs[i & 255]) ^ m, m);
    }
  }
  struct Raw {
    uint32_t own[kOwn];  // a quarter's 16 row bytes and two words of signs and s
  };
  struct Lane {
    uint32_t rows[8], aux[8];
    int sign;
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.own, ld128(p + 16 * t));
    put(r.own + 4, ld64(p + 64 + 8 * t));
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw&, int t) {
    Lane n;
    uint32_t rows[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) rows[i] = all[6 * (i >> 2) + (i & 3)];
    two_rows(rows, t, n.rows);
#pragma unroll
    for (int G = 0; G < 8; ++G) n.aux[G] = all[6 * (G >> 1) + 4 + (G & 1)];
    n.sign = 7 * t;
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t rows = n.rows[G], aux = n.aux[G];
    const uint2 signs = L.signs[(aux >> n.sign) & 127u];
    const uint4 tb = L.scaled[aux >> 28];
    words_of_masked(L.rows[rows & 0xFFu], tb, signs.x, w0);
    words_of_masked(L.rows[(rows >> 8) & 0xFFu], tb, signs.y, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.25f * d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t rows = n.rows[G];
    const uint2 signs = L.signs[(n.aux[G] >> n.sign) & 127u];
    w[0] = negate_where(pick(T.iq3_s8[0][0], T.iq3_s8[0][1], L.rows[rows & 0xFFu]), signs.x);
    w[1] = negate_where(pick(T.iq3_s8[0][0], T.iq3_s8[0][1], L.rows[(rows >> 8) & 0xFFu]), signs.y);
  }
  // The same values through signed_rows: the lane's sign code (7 bits and their parity: its 8 values' signs).
  static __device__ __forceinline__ void codes(const Tables&, const MmaLocal& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t rows = n.rows[G], code = (n.aux[G] >> n.sign) & 127u;
    const uint32_t signs = code | (__popc(code) & 1u) << 7;
    w[0] = L.signed_rows[(rows & 0xFFu) | (signs & 15u) << 8];
    w[1] = L.signed_rows[((rows >> 8) & 0xFFu) | (signs >> 4) << 8];
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = 0.25f * d0 * float(2 * (n.aux[G] >> 28) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows[2], aux;  // the group's eight row bytes (a byte per 4 weights); its four sign codes and s
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint2 v = ld64(p + 8 * G);
    return {{v.x, v.y}, ld32(p + 64 + 4 * G)};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t rows = u.rows[j >> 1] >> (16 * (j & 1));
      const uint2 signs = L.signs[(u.aux >> (7 * j)) & 127u];
      w[2 * j] = negate_where(pick(T.iq3_s8[0][0], T.iq3_s8[0][1], L.rows[rows & 0xFFu]), signs.x);
      w[2 * j + 1] = negate_where(pick(T.iq3_s8[0][0], T.iq3_s8[0][1], L.rows[(rows >> 8) & 0xFFu]), signs.y);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
    f[0] = 0.25f * d0 * float(2 * (u.aux >> 28) + 1), m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq3_xxs) == 98 && offsetof(block_iq3_xxs, qs) == 2, "block_iq3_xxs");

// IQ3_S: a row byte and (qh) a ninth row bit per 4 weights, a byte of signs per 8; s per 32 weights, the scale
// d (2 s + 1).  Its 108 payload bytes are kept in 112 (16-byte loads).
struct FIQ3_S {
  static constexpr int kDtype = FLYX_DT_GGML_IQ3_S, kValues = 256, kBytes = 110, kSkip = 2, kStride = 112;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 6;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint4 scaled[16];
    uint2 signs[256];
    uint16_t rows[512];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq3s_scaled, 16);
    fetch(L.signs, T.sign_hi, 256);
    fetch(L.rows, T.iq3s, 512);
  }
  // As FIQ3_XXS's: a 9-bit row with its four sign bits -> the four values, signed (32 KB: the split kernel takes it
  // where its rows leave room, else Local).
  struct MmaLocal : Local {
    uint32_t signed_rows[8192];
  };
  static __device__ __forceinline__ void mma_tables(const Tables& T, MmaLocal& L) {
    tables(T, L);
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
      const uint32_t s = uint32_t(i) >> 9;
      const uint32_t m = (s & 1u) * 0xFFu | (s >> 1 & 1u) * 0xFF00u | (s >> 2 & 1u) * 0xFF0000u | (s >> 3) * 0xFF000000u;
      L.signed_rows[i] = __vsub4(pick(T.iq3_s8[1][0], T.iq3_s8[1][1], T.iq3s[i & 511]) ^ m, m);
    }
  }
  struct Raw {
    uint32_t own[kOwn], ninth[2], scales;  // a quarter's 16 row bytes and 8 sign bytes; the block's qh and scales
  };
  struct Lane {
    uint32_t rows[8], signs[8], ninth[2], scales;
    int byte, bits;
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.own, ld128(p + 16 * t));
    put(r.own + 4, ld64(p + 72 + 8 * t));
    put(r.ninth, ld64(p + 64));
    r.scales = ld32(p + 104);
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw& r, int t) {
    Lane n;
    uint32_t rows[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) rows[i] = all[6 * (i >> 2) + (i & 3)];
    two_rows(rows, t, n.rows);
#pragma unroll
    for (int G = 0; G < 8; ++G) n.signs[G] = all[6 * (G >> 1) + 4 + (G & 1)];
    n.ninth[0] = r.ninth[0], n.ninth[1] = r.ninth[1], n.scales = r.scales;
    n.byte = 8 * t, n.bits = 2 * t;
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t rows = n.rows[G], ninth = byte_of(n.ninth[G >> 2], G & 3) >> n.bits;
    const uint2 signs = L.signs[(n.signs[G] >> n.byte) & 255u];
    const uint4 tb = L.scaled[(n.scales >> (4 * G)) & 15u];
    words_of_masked(L.rows[(rows & 0xFFu) | (ninth & 1u) << 8], tb, signs.x, w0);
    words_of_masked(L.rows[((rows >> 8) & 0xFFu) | (ninth & 2u) << 7], tb, signs.y, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t rows = n.rows[G], ninth = byte_of(n.ninth[G >> 2], G & 3) >> n.bits;
    const uint2 signs = L.signs[(n.signs[G] >> n.byte) & 255u];
    w[0] = negate_where(pick(T.iq3_s8[1][0], T.iq3_s8[1][1], L.rows[(rows & 0xFFu) | (ninth & 1u) << 8]), signs.x);
    w[1] = negate_where(pick(T.iq3_s8[1][0], T.iq3_s8[1][1], L.rows[((rows >> 8) & 0xFFu) | (ninth & 2u) << 7]),
                        signs.y);
  }
  static __device__ __forceinline__ void codes(const Tables&, const MmaLocal& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t rows = n.rows[G], ninth = byte_of(n.ninth[G >> 2], G & 3) >> n.bits;
    const uint32_t signs = (n.signs[G] >> n.byte) & 255u;
    w[0] = L.signed_rows[(rows & 0xFFu) | (ninth & 1u) << 8 | (signs & 15u) << 9];
    w[1] = L.signed_rows[((rows >> 8) & 0xFFu) | (ninth & 2u) << 7 | (signs >> 4) << 9];
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    *f = d0 * float(2 * ((n.scales >> (4 * G)) & 15u) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows[2], signs;  // the group's eight row bytes, four sign bytes
    uint32_t ninth, scales;   // its qh byte (a ninth row bit per 4 weights); the scale byte of its pair of groups
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    const uint2 v = ld64(p + 8 * G);
    return {{v.x, v.y}, ld32(p + 72 + 4 * G), ld8(p + 64 + G), ld8(p + 104 + (G >> 1))};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t rows = u.rows[j >> 1] >> (16 * (j & 1)), ninth = u.ninth >> (2 * j);
      const uint2 signs = L.signs[(u.signs >> (8 * j)) & 255u];
      w[2 * j] = negate_where(pick(T.iq3_s8[1][0], T.iq3_s8[1][1], L.rows[(rows & 0xFFu) | (ninth & 1u) << 8]), signs.x);
      w[2 * j + 1] =
          negate_where(pick(T.iq3_s8[1][0], T.iq3_s8[1][1], L.rows[((rows >> 8) & 0xFFu) | (ninth & 2u) << 7]), signs.y);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float, float* f, float* m) {
    f[0] = d0 * float(2 * ((u.scales >> (4 * (G & 1))) & 15u) + 1), m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq3_s) == 110 && offsetof(block_iq3_s, qs) == 2 && offsetof(block_iq3_s, qh) == 66 &&
                  offsetof(block_iq3_s, signs) == 74 && offsetof(block_iq3_s, scales) == 106,
              "block_iq3_s");

// IQ1: 8 weights are a row of -1, 0, 1 of a 2048-row codebook, all shifted by +-1/8: 8 times the weights are
// -9, -1, 7 or -7, 1, 9; times 2 s + 1 (s of 3 bits) still integers binary16 holds.  The tables are per shift and s.
// IQ1_S, per 32 weights: 4 row bytes; qh: 16 bits: 4 x 3 more row bits, s, the sign of the shift.
struct FIQ1_S {
  static constexpr int kDtype = FLYX_DT_GGML_IQ1_S, kValues = 256, kBytes = 50, kSkip = 2, kStride = 48;
  static constexpr int kScales = 1, kS0 = 0, kS1 = -1, kPlanes = 1, kOwn = 3;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint2 scaled[16];
    uint32_t rows[2048];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq1_scaled, 16);
    fetch(L.rows, T.iq1, 2048);
  }
  struct Raw {
    uint32_t own[kOwn];  // a quarter's 8 row bytes and two qh
  };
  struct Lane {
    uint32_t rows[8], more[4];  // a word per group; 16 bits per group
    int byte, bits;
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {
    Raw r;
    put(r.own, ld64(p + 8 * t));
    r.own[2] = ld32(p + 32 + 4 * t);
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw&, int t) {
    Lane n;
#pragma unroll
    for (int G = 0; G < 8; ++G) n.rows[G] = all[3 * (G >> 1) + (G & 1)];
#pragma unroll
    for (int q = 0; q < 4; ++q) n.more[q] = all[3 * q + 2];
    n.byte = 8 * t, n.bits = 3 * t;
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t h = (n.more[G >> 1] >> (16 * (G & 1))) & 0xFFFFu;
    const uint32_t sel = L.rows[((n.rows[G] >> n.byte) & 255u) | ((h >> n.bits) & 7u) << 8];
    const uint2 tb = L.scaled[h >> 12];
    words_of_ternary(sel, tb, w0);
    words_of_ternary(sel >> 16, tb, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.125f * d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t h = (n.more[G >> 1] >> (16 * (G & 1))) & 0xFFFFu;
    const uint32_t sel = L.rows[((n.rows[G] >> n.byte) & 255u) | ((h >> n.bits) & 7u) << 8], v = T.iq1_s8[h >> 15];
    w[0] = pick(v, v, sel);
    w[1] = pick(v, v, sel >> 16);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    const uint32_t h = (n.more[G >> 1] >> (16 * (G & 1))) & 0xFFFFu;
    *f = 0.125f * d0 * float(2 * ((h >> 12) & 7u) + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows, more;  // the group's four row bytes; its 16 bits: 4 x 3 more row bits, s, the shift's sign
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) { return {ld32(p + 4 * G), ld16(p + 32 + 2 * G)}; }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
    const uint32_t v = T.iq1_s8[u.more >> 15];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t sel = L.rows[((u.rows >> (8 * j)) & 255u) | ((u.more >> (3 * j)) & 7u) << 8];
      w[2 * j] = pick(v, v, sel);
      w[2 * j + 1] = pick(v, v, sel >> 16);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int, float d0, float, float* f, float* m) {
    f[0] = 0.125f * d0 * float(2 * ((u.more >> 12) & 7u) + 1), m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq1_s) == 50 && offsetof(block_iq1_s, qs) == 2 && offsetof(block_iq1_s, qh) == 34, "block_iq1_s");

// IQ1_M, per 8 weights: a row byte and (qh) a nibble: 3 more row bits and the sign of the shift; s (3 bits) per 16
// weights in the four 16-bit scale words, whose top nibbles together are the block's binary16 scale
// (blocks_pack puts it in the scales plane).
struct FIQ1_M {
  static constexpr int kDtype = FLYX_DT_GGML_IQ1_M, kValues = 256, kBytes = 56, kSkip = 0, kStride = 56;
  static constexpr int kScales = 1, kS0 = -1, kS1 = -1, kPlanes = 1, kOwn = 3;
  static constexpr int kFlatRun = 0;
  struct Local {
    uint2 scaled[16];
    uint32_t rows[2048];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.scaled, T.iq1_scaled, 16);
    fetch(L.rows, T.iq1, 2048);
  }
  struct Raw {
    uint32_t own[kOwn], scales[2];  // a quarter's 8 row bytes and 8 nibbles of qh; the block's scale words
  };
  struct Lane {
    uint32_t rows[8], more[4], scales[2];
    int byte, nibble, scale;
  };
  static __device__ __forceinline__ Raw load(const uint8_t* p, int t) {  // the payload is 8-byte aligned
    Raw r;
    put(r.own, ld64(p + 8 * t));
    r.own[2] = ld32(p + 32 + 4 * t);
    put(r.scales, ld64(p + 48));
    return r;
  }
  static __device__ __forceinline__ Lane build(const uint32_t* all, const Raw& r, int t) {
    Lane n;
#pragma unroll
    for (int G = 0; G < 8; ++G) n.rows[G] = all[3 * (G >> 1) + (G & 1)];
#pragma unroll
    for (int q = 0; q < 4; ++q) n.more[q] = all[3 * q + 2];
    n.scales[0] = r.scales[0], n.scales[1] = r.scales[1];
    n.byte = 8 * t, n.nibble = 4 * t, n.scale = 3 * (t >> 1);
    return n;
  }
  FLYX_LANES_SHARE
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t*) {
    const uint32_t h = (n.more[G >> 1] >> (16 * (G & 1) + n.nibble)) & 15u;
    const uint32_t sel = L.rows[((n.rows[G] >> n.byte) & 255u) | (h & 7u) << 8];
    const uint32_t s = (n.scales[G >> 2] >> (16 * ((G >> 1) & 1) + 6 * (G & 1) + n.scale)) & 7u;
    const uint2 tb = L.scaled[(h & 8u) | s];
    words_of_ternary(sel, tb, w0);
    words_of_ternary(sel >> 16, tb, w0 + 2);
  }
  static __device__ __forceinline__ float coef(int, float d0, float) { return 0.125f * d0; }
  static constexpr int kSub = 16;
  static __device__ __forceinline__ void codes(const Tables& T, const Local& L, const Lane& n, int G, uint32_t* w) {
    const uint32_t h = (n.more[G >> 1] >> (16 * (G & 1) + n.nibble)) & 15u;
    const uint32_t sel = L.rows[((n.rows[G] >> n.byte) & 255u) | (h & 7u) << 8], v = T.iq1_s8[h >> 3];
    w[0] = pick(v, v, sel);
    w[1] = pick(v, v, sel >> 16);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    const uint32_t s = (n.scales[G >> 2] >> (16 * ((G >> 1) & 1) + 6 * (G & 1) + n.scale)) & 7u;
    *f = 0.125f * d0 * float(2 * s + 1), *m = 0.f;
  }
  struct Unit {
    uint32_t rows, more, scales;  // four row bytes; four nibbles (3 more row bits, the shift's sign); a scale word
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, int G) {
    return {ld32(p + 4 * G), ld16(p + 32 + 2 * G), ld16(p + 48 + 2 * (G >> 1))};
  }
  static __device__ __forceinline__ void unit_codes(const Tables& T, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t h = (u.more >> (4 * j)) & 15u, v = T.iq1_s8[h >> 3];
      const uint32_t sel = L.rows[((u.rows >> (8 * j)) & 255u) | (h & 7u) << 8];
      w[2 * j] = pick(v, v, sel);
      w[2 * j + 1] = pick(v, v, sel >> 16);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float, float* f, float* m) {
#pragma unroll
    for (int h = 0; h < 2; ++h) f[h] = 0.125f * d0 * float(2 * ((u.scales >> (6 * (G & 1) + 3 * h)) & 7u) + 1), m[h] = 0.f;
  }
};
static_assert(sizeof(block_iq1_m) == 56 && offsetof(block_iq1_m, qh) == 32 && offsetof(block_iq1_m, scales) == 48,
              "block_iq1_m");

// IQ4_XS: 8 groups of 32 in the nibble layout of the blocks of 32, the 16-value table; a 6-bit scale per group
// (less 32): 4 low bits in scales_l, 2 high bits in scales_h.  (scale - 32) * value can need 12 bits, so the
// scale goes in two parts: 8 hi + lo (hi -4..3, lo 0..7), a plane each.  The whole block is kept.
// The payload is the block's 128 code bytes lane-major (blocks_pack): bytes 4 t .. 4 t + 3 of each group G at
// 32 t + 4 G, so a quad lane's bytes of every group are together and 16-byte aligned (k_round_blocks_warps loads a
// lane's four groups at once; `load` a lane's two quarters, `unit` a group's four words).  The block's first 8 bytes
// (d, the scales' high bits, their low bits) are its four scales (kHeadInScales).
struct FIQ4_XS {
  static constexpr int kDtype = FLYX_DT_GGML_IQ4_XS, kValues = 256, kBytes = 136, kSkip = 8, kStride = 128;
  static constexpr int kScales = 4, kS0 = 0, kS1 = -1, kPlanes = 2;
  static constexpr bool kHeadInScales = true;
  static constexpr int kFlatRun = 0;
  // A code byte's place in the payload: byte j of group G.
  static constexpr int lane_major(int G, int j) { return 32 * (j / 4) + 4 * G + j % 4; }
  struct Local {
    uint32_t pairs[256];
    uint32_t parts[64];
    uint16_t pairs8[256];
  };
  static __device__ __forceinline__ void tables(const Tables& T, Local& L) {
    fetch(L.pairs, T.pair_iq4, 256);
    fetch(L.parts, T.parts_s6, 64);
    fetch(L.pairs8, T.pair_iq4_s8, 256);
  }
  struct Lane {
    uint32_t q[16], head[2];  // of each group the lane's 8 bytes; the block's first 8 bytes
    int shift;
  };
  using Raw = Lane;
  // Lane t: bytes 8 (t & 1) .. 8 (t & 1) + 7 of each group (its nibbles at shift), the lane-major quarters 2 (t & 1)
  // and 2 (t & 1) + 1: 64 bytes together.  h: the head (the scales plane).
  static __device__ __forceinline__ Raw load(const uint8_t* p, const uint16_t* h, int t) {
    Raw r;
    const uint8_t* c = p + 64 * (t & 1);
    uint32_t lo[8], hi[8];
    put(lo, ld128(c));
    put(lo + 4, ld128(c + 16));
    put(hi, ld128(c + 32));
    put(hi + 4, ld128(c + 48));
#pragma unroll
    for (int G = 0; G < 8; ++G) r.q[2 * G] = lo[G], r.q[2 * G + 1] = hi[G];
    put(r.head, *reinterpret_cast<const uint2*>(h));
    r.shift = 4 * (t >> 1);
    return r;
  }
  static __device__ __forceinline__ Lane build(const Raw& r, int) { return r; }
  static __device__ __forceinline__ Lane share(const Raw& r, int t, int) { return build(r, t); }
  static __device__ __forceinline__ void words(const Local& L, const Lane& n, int G, uint32_t* w0, uint32_t* w1) {
    const uint32_t s6 = ((byte_of(n.head[1], G >> 1) >> (4 * (G & 1))) & 15u) | ((n.head[0] >> (16 + 2 * G)) & 3u) << 4;
    const uint32_t parts = L.parts[s6], hi = twice(parts & 0xFFFFu), lo = twice(parts >> 16);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t x = n.q[2 * G + i] >> n.shift;
      const uint32_t v0 = L.pairs[nibble_pair(x)], v1 = L.pairs[nibble_pair(x >> 16)];
      w0[2 * i] = mul2(v0, hi), w0[2 * i + 1] = mul2(v1, hi);
      w1[2 * i] = mul2(v0, lo), w1[2 * i + 1] = mul2(v1, lo);
    }
  }
  static __device__ __forceinline__ float coef(int p, float d0, float) { return p ? d0 : 8.f * d0; }
  static constexpr int kSub = 32;
  static __device__ __forceinline__ void codes(const Tables&, const Local& L, const Lane& n, int G, uint32_t* w) {
    w[0] = nibbles_through(L.pairs8, n.q[2 * G] >> n.shift);
    w[1] = nibbles_through(L.pairs8, n.q[2 * G + 1] >> n.shift);
  }
  static __device__ __forceinline__ void factors(const Lane& n, int G, float d0, float, float* f, float* m) {
    const uint32_t s6 = ((byte_of(n.head[1], G >> 1) >> (4 * (G & 1))) & 15u) | ((n.head[0] >> (16 + 2 * G)) & 3u) << 4;
    *f = d0 * float(int(s6) - 32), *m = 0.f;
  }
  struct Unit {
    uint32_t q[4], head[2];  // the group's 16 bytes; the block's first 8 (d, the scales' high bits, their low bits)
  };
  static __device__ __forceinline__ Unit unit(const uint8_t* p, const uint16_t* h, int G) {  // group G's 4 words
    const uint2 hd = *reinterpret_cast<const uint2*>(h);
    return {{ld32(p + 4 * G), ld32(p + 32 + 4 * G), ld32(p + 64 + 4 * G), ld32(p + 96 + 4 * G)}, {hd.x, hd.y}};
  }
  static __device__ __forceinline__ void unit_codes(const Tables&, const Local& L, const Unit& u, int, uint32_t* w) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      w[i] = nibbles_through(L.pairs8, u.q[i]);
      w[4 + i] = nibbles_through(L.pairs8, u.q[i] >> 4);
    }
  }
  static __device__ __forceinline__ void unit_factors(const Unit& u, int G, float d0, float, float* f, float* m) {
    const uint32_t s6 = ((byte_of(u.head[1], G >> 1) >> (4 * (G & 1))) & 15u) | ((u.head[0] >> (16 + 2 * G)) & 3u) << 4;
    f[0] = d0 * float(int(s6) - 32), m[0] = 0.f;
  }
};
static_assert(sizeof(block_iq4_xs) == 136 && offsetof(block_iq4_xs, scales_h) == 2 &&
                  offsetof(block_iq4_xs, scales_l) == 4 && offsetof(block_iq4_xs, qs) == 8 &&
                  FIQ4_XS::kSkip == offsetof(block_iq4_xs, qs) && 2 * FIQ4_XS::kScales == offsetof(block_iq4_xs, qs),
              "block_iq4_xs");

#define FLYX_BLOCK_TYPES(X)                                                                  \
  X(FQ4_0) X(FQ4_1) X(FQ5_0) X(FQ5_1) X(FQ8_0) X(FQ2_K) X(FQ3_K) X(FQ4_K) X(FQ5_K) X(FQ6_K) \
  X(FIQ2_XXS) X(FIQ2_XS) X(FIQ2_S) X(FIQ3_XXS) X(FIQ3_S) X(FIQ1_S) X(FIQ1_M) X(FIQ4_NL) X(FIQ4_XS)

}  // namespace
}  // namespace flyx::cuda
