// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0

#include <array>
#include <vector>

#include "ops/ple/ref.h"
#include "ops/qsa/ref.h"
#include "tests/common/check.h"

using flyx::test::check;
using namespace flyx::ops;

int main() {
  constexpr uint64_t m[] = {23703573157769, 20109073645365, 8052911324071};
  constexpr uint64_t mod[] = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                              20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
  constexpr uint64_t off[] = {0,         20000003,  40000026,  60000059,  80000106,  100000165, 120000228, 140000297,
                              160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
  flyx_ple_block b{};
  b.ngram = 3;
  b.heads_per_order = 8;
  b.eos = 248044;
  b.multipliers = m;
  b.moduli = mod;
  b.offsets = off;
  auto hash = [&](int token, std::vector<int32_t> ancestors) {
    std::array<int32_t, 16> rows;
    ple_indices(token, ancestors.data(), int(ancestors.size()), b, rows.data());
    return rows;
  };
  using Rows = std::array<int32_t, 16>;
  const Rows start = {4064167,   21428206,  43851727,  63435096,  89826284,  106088217, 121229381, 156147803,
                      175176707, 188463481, 216828809, 228933352, 248448227, 269188956, 283041664, 314347972};
  check(hash(42, {}) == start, "start pads both orders with PLE EOS");
  check(hash(44, {43, 42}) == Rows{15934461, 32323911, 40518776, 71991788, 81825914, 105104047, 130021251, 156577564,
                                   173528038, 186094676, 214089569, 222648464, 248932794, 275217214, 290645427,
                                   307786149},
        "three consumed tokens");
  check(hash(42, {248044, 43}) == start, "previous PLE EOS resets every older ID");
  check(hash(42, {248046, 43}) == Rows{3992284, 21066846, 53345660, 78726454, 92943978, 105148023, 134202424, 141005001,
                                       166755275, 182786099, 206028157, 239370916, 256814368, 275155726, 281469046,
                                       314532156},
        "generation stop is an ordinary hash ID");
  check(hash('f', {'d', 'c'}) == Rows{5527960, 27586526, 58619162, 66068517, 95314331, 111730334, 126354973, 159189119,
                                      179335413, 195869619, 215162970, 230307016, 258581424, 266856363, 289614987,
                                      303408442},
        "path c,d,f excludes sibling e");
  for (int t = 0; t < 10; ++t) {
    double window[9]{}, current = t == 0 ? 1.0 : 0;
    if (t > 0) window[9 - t] = 1.0;
    const double weights[] = {1, 2, 3, 4};
    check(ple_shortconv(window, &current, weights, 0, 1, 4, 3) == (t % 3 ? 0 : silu(4 - t / 3)),
          "nine-step dilated convolution impulse response");
  }
  for (int visible = 8; visible <= 15; ++visible) {
    std::vector<double> scores(visible / 4, 0);
    scores.front() = -1;
    const auto selected = qsa_positions(scores, visible, 4, 8);
    std::vector<int> wanted;
    for (int i = visible < 12 ? 0 : 4; i < visible; ++i) wanted.push_back(i);
    check(selected == wanted, "complete groups and unconditionally visible trailing tokens in KV order");
  }
  check(qsa_positions({1, 1, 1}, 12, 4, 8) == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}), "stable tied groups");
  return flyx::test::finish("qwen4exp");
}
