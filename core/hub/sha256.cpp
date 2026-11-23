// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// SHA-256 as FIPS 180-4 defines it: a portable compression function, and one through the x86 SHA extensions
// (sha256rnds2 does two rounds on the state held as ABEF and CDGH; sha256msg1 / sha256msg2 extend the message).
#include "core/hub/sha256.h"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#define FLYX_SHA_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define FLYX_SHA_TARGET
#else
#include <cpuid.h>
#define FLYX_SHA_TARGET __attribute__((target("sha,sse4.1,ssse3")))
#endif
#endif

namespace flyx::hub {

namespace {

constexpr uint32_t kRound[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void blocks_portable(uint32_t* h, const uint8_t* p, size_t n) {
  for (; n > 0; --n, p += 64) {
    uint32_t w[64];
    for (int t = 0; t < 16; ++t)
      w[t] = uint32_t(p[4 * t]) << 24 | uint32_t(p[4 * t + 1]) << 16 | uint32_t(p[4 * t + 2]) << 8 | p[4 * t + 3];
    for (int t = 16; t < 64; ++t) {
      const uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
      const uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = w[t - 16] + s0 + w[t - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int t = 0; t < 64; ++t) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kRound[t] + w[t];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
}

#if FLYX_SHA_X86
bool has_sha_extensions() {
#if defined(_MSC_VER)
  int r1[4], r7[4];
  __cpuid(r1, 1);
  __cpuidex(r7, 7, 0);
  const unsigned ecx1 = unsigned(r1[2]), ebx7 = unsigned(r7[1]);
#else
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
  const unsigned ecx1 = c;
  if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
  const unsigned ebx7 = b;
#endif
  const bool ssse3 = ecx1 & (1u << 9), sse41 = ecx1 & (1u << 19), sha = ebx7 & (1u << 29);
  return ssse3 && sse41 && sha;
}

FLYX_SHA_TARGET void blocks_sha_ni(uint32_t* h, const uint8_t* p, size_t n) {
  const __m128i swap = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);  // big-endian words
  // The state as the instructions hold it: ABEF and CDGH (word 0 the last letter).
  __m128i dcba = _mm_loadu_si128(reinterpret_cast<const __m128i*>(h));
  __m128i hgfe = _mm_loadu_si128(reinterpret_cast<const __m128i*>(h + 4));
  const __m128i cdab = _mm_shuffle_epi32(dcba, 0xB1);
  const __m128i efgh = _mm_shuffle_epi32(hgfe, 0x1B);
  __m128i abef = _mm_alignr_epi8(cdab, efgh, 8);
  __m128i cdgh = _mm_blend_epi16(efgh, cdab, 0xF0);
  for (; n > 0; --n, p += 64) {
    const __m128i abef0 = abef, cdgh0 = cdgh;
    __m128i w[4];
    for (int i = 0; i < 4; ++i)
      w[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16 * i)), swap);
    for (int g = 0; g < 16; ++g) {  // rounds 4g .. 4g + 3 on message words W_g
      if (g >= 4) {  // W_g from W_{g-4} (slot g & 3), W_{g-3}, W_{g-2}, W_{g-1}
        __m128i x = _mm_sha256msg1_epu32(w[g & 3], w[(g + 1) & 3]);
        x = _mm_add_epi32(x, _mm_alignr_epi8(w[(g + 3) & 3], w[(g + 2) & 3], 4));
        w[g & 3] = _mm_sha256msg2_epu32(x, w[(g + 3) & 3]);
      }
      __m128i k = _mm_add_epi32(w[g & 3], _mm_loadu_si128(reinterpret_cast<const __m128i*>(kRound + 4 * g)));
      cdgh = _mm_sha256rnds2_epu32(cdgh, abef, k);  // two rounds: the new ABEF lands in cdgh's register
      k = _mm_shuffle_epi32(k, 0x0E);
      abef = _mm_sha256rnds2_epu32(abef, cdgh, k);  // and two more, back
    }
    abef = _mm_add_epi32(abef, abef0);
    cdgh = _mm_add_epi32(cdgh, cdgh0);
  }
  const __m128i feba = _mm_shuffle_epi32(abef, 0x1B);
  const __m128i dchg = _mm_shuffle_epi32(cdgh, 0xB1);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(h), _mm_blend_epi16(feba, dchg, 0xF0));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(h + 4), _mm_alignr_epi8(dchg, feba, 8));
}
#endif

}  // namespace

bool Sha256::accelerated() {
#if FLYX_SHA_X86
  static const bool yes = has_sha_extensions();
  return yes;
#else
  return false;
#endif
}

Sha256::Sha256(bool portable) : portable_(portable) {
  static constexpr uint32_t kStart[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::memcpy(h_, kStart, sizeof h_);
}

void Sha256::blocks(const uint8_t* p, size_t n) {
#if FLYX_SHA_X86
  if (!portable_ && accelerated()) return blocks_sha_ni(h_, p, n);
#endif
  blocks_portable(h_, p, n);
}

void Sha256::update(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  total_ += n;
  if (used_ > 0) {
    const size_t take = std::min(n, 64 - used_);
    std::memcpy(buf_ + used_, p, take);
    used_ += take, p += take, n -= take;
    if (used_ < 64) return;
    blocks(buf_, 1);
    used_ = 0;
  }
  if (n >= 64) {
    blocks(p, n / 64);
    p += n / 64 * 64, n %= 64;
  }
  std::memcpy(buf_, p, n);
  used_ = n;
}

std::string Sha256::hex() {
  const uint64_t bits = total_ * 8;
  const uint8_t one = 0x80, zero = 0;
  update(&one, 1);
  while (used_ != 56) update(&zero, 1);
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = uint8_t(bits >> (56 - 8 * i));
  update(len, 8);
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (uint32_t v : h_)
    for (int s = 28; s >= 0; s -= 4) out += digits[(v >> s) & 15];
  return out;
}

}  // namespace flyx::hub
