// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The tables the block-weight kernels decode with (gemm_blocks.cu), built
// once on the host from ggml's own (third_party/ggml) and put on the device.
//
// The kernels multiply binary16 activations by integers: a weight's code
// times its group's scale.  So every table holds binary16 bit patterns of
// integers, and building them is where "binary16 holds it" is checked:
// half_bits refuses an integer of more than 11 significant bits, and
// check_products goes through the products the kernels form in registers.
//
// ggml stores a codebook row as the magnitudes of its weights, a byte each.
// A family has few distinct magnitudes (three in the IQ2 codebooks, eight in
// the IQ3 ones; the IQ1 rows hold -1, 0, 1), so a row becomes the selectors
// of its weights' magnitudes, a nibble each, and the magnitudes times each
// scale become byte tables: one byte permute picks four high bytes, another
// four low bytes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cuda/kernels/gemm_internal.h"

namespace flyx::cuda {

// ggml's header as its authors wrote it: here its tables, on the host.
namespace ggml_host {
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml/ggml-common.h"
}  // namespace ggml_host

namespace {

using namespace ggml_host;

[[noreturn]] void die(const char* what) {
  throw std::runtime_error(std::string("flyx-cuda: the block weights' tables: ") + what);
}

// Whether binary16 holds the integer v: at most 11 significant bits, below 2^16.
bool half_holds(int v) {
  unsigned m = unsigned(v < 0 ? -v : v);
  if (m >= 65536u) return false;
  while (m && !(m & 1u)) m >>= 1;
  return m < 2048u;
}
// The integer v as binary16 bits.
uint16_t half_bits(int v) {
  if (!half_holds(v)) die("an integer binary16 does not hold");
  if (v == 0) return 0;
  const unsigned m = unsigned(v < 0 ? -v : v);
  int e = 0;
  while ((m >> (e + 1)) != 0) ++e;
  const unsigned frac = e <= 10 ? (m << (10 - e)) & 0x3FFu : (m >> (e - 10)) & 0x3FFu;
  return uint16_t((v < 0 ? 0x8000u : 0u) | unsigned(e + 15) << 10 | frac);
}
uint32_t pair(int a, int b) { return uint32_t(half_bits(a)) | uint32_t(half_bits(b)) << 16; }
uint2 quad(int a, int b, int c, int d) { return make_uint2(pair(a, b), pair(c, d)); }

// Bytes from..from + 3 of a list of small integers as int8 in a word (sign times each; byte i: entry from + i).
uint32_t int8_word(const std::vector<int>& values, size_t from, int sign) {
  uint32_t w = 0;
  for (size_t i = 0; i < 4 && from + i < values.size(); ++i) {
    const int v = sign * values[from + i];
    if (v < -127 || v > 127) die("a value an int8 does not hold");
    w |= uint32_t(uint8_t(int8_t(v))) << (8 * i);
  }
  return w;
}

// Byte tables of the binary16 of values[i] * mult: entry i in byte i % 4 of word i / 4.
void byte_tables(const std::vector<int>& values, int mult, uint32_t hi[2], uint32_t lo[2]) {
  if (values.size() > 8) die("a codebook has more magnitudes than a selector takes");
  hi[0] = hi[1] = lo[0] = lo[1] = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const uint32_t b = half_bits(values[i] * mult);
    hi[i / 4] |= (b >> 8) << (8 * (i % 4));
    lo[i / 4] |= (b & 0xFFu) << (8 * (i % 4));
  }
}

// A codebook of `width` byte magnitudes a row: its distinct magnitudes in
// ascending order, and per row the selectors of its weights (nibble j: weight j).
template <class Row, class Sel>
std::vector<int> selectors(const Row* grid, size_t rows, int width, std::vector<Sel>* out) {
  bool seen[256] = {};
  for (size_t r = 0; r < rows; ++r)
    for (int j = 0; j < width; ++j) seen[(grid[r] >> (8 * j)) & 0xFF] = true;
  std::vector<int> values;
  int index[256];
  for (int v = 0; v < 256; ++v)
    if (seen[v]) {
      index[v] = int(values.size());
      values.push_back(v);
    }
  out->resize(rows);
  for (size_t r = 0; r < rows; ++r) {
    Sel s = 0;
    for (int j = 0; j < width; ++j) s = Sel(s | Sel(index[(grid[r] >> (8 * j)) & 0xFF]) << (4 * j));
    (*out)[r] = s;
  }
  return values;
}

template <class T>
const T* upload(const std::vector<T>& v) {
  void* p = nullptr;
  if (cudaMalloc(&p, v.size() * sizeof(T)) != cudaSuccess ||
      cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
    die("no device memory");
  return static_cast<const T*>(p);
}

// The products the kernels form in registers (a table's value times a group's scale, or plus a bias first):
// each must be an integer binary16 holds, or the multiplication would round.
void check_products() {
  auto all = [](int a0, int a1, int b0, int b1, const char* what) {
    for (int a = a0; a <= a1; ++a)
      for (int b = b0; b <= b1; ++b)
        if (!half_holds(a * b)) die(what);
  };
  all(0, 3, 0, 15, "Q2_K: code times scale");
  all(-4, 3, -32, 31, "Q3_K: code times scale");
  all(0, 15, 0, 63, "Q4_K: code times scale");
  all(0, 31, 0, 63, "Q5_K: code times scale");
  all(-32, 31, -8, 7, "Q6_K: code times the scale's high part");
  all(-32, 31, 0, 15, "Q6_K: code times the scale's low part");
  for (int i = 0; i < 16; ++i) {
    all(kvalues_iq4nl[i], kvalues_iq4nl[i], -4, 3, "IQ4_XS: value times the scale's high part");
    all(kvalues_iq4nl[i], kvalues_iq4nl[i], 0, 7, "IQ4_XS: value times the scale's low part");
  }
}

// A sign per weight (bit j: weight j is negative) as bit 2 of the weights' selectors.
uint32_t sign_selectors(uint32_t bits) {
  uint32_t s = 0;
  for (int j = 0; j < 8; ++j)
    if ((bits >> j) & 1u) s |= 4u << (4 * j);
  return s;
}
// ... as the top bits of the high bytes of weights 0..3 and 4..7.
uint2 sign_bytes(uint32_t bits) {
  uint2 s = make_uint2(0u, 0u);
  for (int j = 0; j < 4; ++j) {
    if ((bits >> j) & 1u) s.x |= 0x80u << (8 * j);
    if ((bits >> (j + 4)) & 1u) s.y |= 0x80u << (8 * j);
  }
  return s;
}

BlockTables build() {
  check_products();
  BlockTables t{};
  std::vector<uint32_t> s32;
  std::vector<uint16_t> s16;
  uint32_t hi[2], lo[2];

  // IQ2: three magnitudes; selector bit 2 picks the negated high byte.
  std::vector<uint4> iq2(3 * 16);
  auto iq2_family = [&](int family, const std::vector<int>& values) {
    if (values.size() > 4) die("an IQ2 codebook has more than four magnitudes");
    for (int s = 0; s < 16; ++s) {
      byte_tables(values, 2 * s + 1, hi, lo);
      iq2[size_t(family * 16 + s)] = make_uint4(hi[0], hi[0] | 0x80808080u, lo[0], 0u);
    }
    t.iq2_s8[family][0] = int8_word(values, 0, 1);
    t.iq2_s8[family][1] = int8_word(values, 0, -1);
  };
  iq2_family(0, selectors(iq2xxs_grid, 256, 8, &s32));
  t.iq2xxs = upload(s32);
  iq2_family(1, selectors(iq2xs_grid, 512, 8, &s32));
  t.iq2xs = upload(s32);
  iq2_family(2, selectors(iq2s_grid, 1024, 8, &s32));
  t.iq2s = upload(s32);
  t.iq2_scaled = upload(iq2);

  // IQ3: eight magnitudes.
  auto iq3_scaled = [&](int family, const std::vector<int>& values) {
    std::vector<uint4> v(16);
    for (int s = 0; s < 16; ++s) {
      byte_tables(values, 2 * s + 1, hi, lo);
      v[size_t(s)] = make_uint4(hi[0], hi[1], lo[0], lo[1]);
    }
    t.iq3_s8[family][0] = int8_word(values, 0, 1);
    t.iq3_s8[family][1] = int8_word(values, 4, 1);
    return upload(v);
  };
  t.iq3xxs_scaled = iq3_scaled(0, selectors(iq3xxs_grid, 256, 4, &s16));
  t.iq3xxs = upload(s16);
  std::vector<uint32_t> iq3xxs_signed_rows(16 * 256);
  for (unsigned i = 0; i < iq3xxs_signed_rows.size(); ++i) {
    const uint32_t row = iq3xxs_grid[i & 255u];
    for (int j = 0; j < 4; ++j) {
      const int magnitude = int((row >> (8 * j)) & 255u);
      const int value = (i & (1u << (8 + j))) ? -magnitude : magnitude;
      iq3xxs_signed_rows[i] |= uint32_t(uint8_t(value)) << (8 * j);
    }
  }
  t.iq3xxs_signed_rows = upload(iq3xxs_signed_rows);
  t.iq3s_scaled = iq3_scaled(1, selectors(iq3s_grid, 512, 4, &s16));
  t.iq3s = upload(s16);

  // IQ1: rows of -1, 0, 1 (selectors 0, 1, 2); 8 times a weight is 8 * row + 1 or 8 * row - 1.
  s32.assign(NGRID_IQ1S, 0);
  for (size_t r = 0; r < NGRID_IQ1S; ++r)
    for (int j = 0; j < 8; ++j) {
      const int v = int(int8_t((iq1s_grid[r] >> (8 * j)) & 0xFF));
      if (v < -1 || v > 1) die("an IQ1 codebook row is not of -1, 0, 1");
      s32[r] |= uint32_t(v + 1) << (4 * j);
    }
  t.iq1 = upload(s32);
  std::vector<uint2> iq1(16);
  for (int down = 0; down < 2; ++down) {
    const int shift = down ? -1 : 1;
    for (int s = 0; s < 8; ++s) {
      byte_tables({-8 + shift, shift, 8 + shift}, 2 * s + 1, hi, lo);
      iq1[size_t(down * 8 + s)] = make_uint2(hi[0], lo[0]);
    }
    t.iq1_s8[down] = int8_word({-8 + shift, shift, 8 + shift}, 0, 1);
  }
  t.iq1_scaled = upload(iq1);

  // Signs.
  std::vector<uint32_t> ksel(128), sel(256);
  std::vector<uint2> khi(128), shi(256);
  for (uint32_t c = 0; c < 128; ++c) ksel[c] = sign_selectors(ksigns_iq2xs[c]), khi[c] = sign_bytes(ksigns_iq2xs[c]);
  for (uint32_t b = 0; b < 256; ++b) sel[b] = sign_selectors(b), shi[b] = sign_bytes(b);
  t.ksign_sel = upload(ksel), t.sign_sel = upload(sel), t.ksign_hi = upload(khi), t.sign_hi = upload(shi);

  // Bit fields.
  std::vector<uint32_t> pu(256), ps(256), pi(256), pmx(256), parts8(256);
  std::vector<uint2> qu(256), qs2(256), qh(256), qm(16);
  std::vector<uint16_t> i8(256), pi8(256), pmx8(256);
  for (int b = 0; b < 256; ++b) {
    const int lo4 = b & 15, hi4 = b >> 4;
    pu[size_t(b)] = pair(lo4, hi4);
    ps[size_t(b)] = pair(lo4 - 8, hi4 - 8);
    pi[size_t(b)] = pair(kvalues_iq4nl[lo4], kvalues_iq4nl[hi4]);
    pi8[size_t(b)] = uint16_t(int8_word({kvalues_iq4nl[lo4], kvalues_iq4nl[hi4]}, 0, 1));
    pmx[size_t(b)] = pair(kvalues_mxfp4[lo4], kvalues_mxfp4[hi4]);
    pmx8[size_t(b)] = uint16_t(int8_word({kvalues_mxfp4[lo4], kvalues_mxfp4[hi4]}, 0, 1));
    const int c[4] = {b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3};
    qu[size_t(b)] = quad(c[0], c[1], c[2], c[3]);
    qs2[size_t(b)] = quad(c[0] - 1, c[1] - 1, c[2] - 1, c[3] - 1);
    qh[size_t(b)] = quad(16 * c[0] - 32, 16 * c[1] - 32, 16 * c[2] - 32, 16 * c[3] - 32);
    const int s = int(int8_t(b));
    i8[size_t(b)] = half_bits(s);
    const int low = s & 15, high = (s - low) / 16;  // s = 16 high + low
    parts8[size_t(b)] = pair(high, low);
  }
  for (int b = 0; b < 16; ++b) qm[size_t(b)] = quad(4 * (b & 1) - 4, 4 * ((b >> 1) & 1) - 4, 4 * ((b >> 2) & 1) - 4, 4 * (b >> 3) - 4);
  t.pair_u4 = upload(pu), t.pair_s4 = upload(ps), t.pair_iq4 = upload(pi), t.pair_iq4_s8 = upload(pi8);
  t.pair_mxfp4 = upload(pmx), t.pair_mxfp4_s8 = upload(pmx8);
  for (int w = 0; w < 4; ++w)
    t.iq4_s8[w] = int8_word({kvalues_iq4nl[4 * w], kvalues_iq4nl[4 * w + 1], kvalues_iq4nl[4 * w + 2],
                             kvalues_iq4nl[4 * w + 3]}, 0, 1);
  t.quad_u2 = upload(qu), t.quad_s2 = upload(qs2), t.quad_hi6 = upload(qh), t.quad_mask = upload(qm);
  t.int8 = upload(i8), t.parts_s8 = upload(parts8);

  // Scales of a group.
  std::vector<uint32_t> tu(64), ts(64), parts6(64);
  for (int v = 0; v < 64; ++v) {
    tu[size_t(v)] = pair(v, v);
    ts[size_t(v)] = pair(v - 32, v - 32);
    const int s = v - 32, low = s & 7, high = (s - low) / 8;  // s = 8 high + low
    parts6[size_t(v)] = pair(high, low);
  }
  t.twice_u6 = upload(tu), t.twice_s6 = upload(ts), t.parts_s6 = upload(parts6);
  return t;
}

}  // namespace

const BlockTables& block_tables() {
  static const BlockTables t = build();
  return t;
}

}  // namespace flyx::cuda
