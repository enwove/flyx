// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <limits>
#include <random>
#include <stdexcept>
#include "backends/cuda/runtime/row_pool.h"
#include <vector>

#include "backends/cuda/runtime/offload/expert_store.h"
#include "backends/cuda/runtime/offload/expert_order.h"
#include "backends/cuda/runtime/offload/expert_profile.h"
#include "ops/dtype/ref.h"
#include "tests/common/check.h"

int main() {
  using namespace flyx;
  {
    cuda::ExpertOrder o(8);
    o.counts = {1, 1, 1, 1, 8, 7, 0, 0};
    o.observe();
    std::vector<std::pair<int, int>> swaps;
    auto exchange = [&](int resident, int cold) { swaps.emplace_back(resident, cold); };
    test::check(o.update(4, o.scores.data(), exchange) == 1 && swaps.front() == std::pair{3, 4},
                "bounded promotion chooses the most frequent challenger and breaks ties by logical ID");
    test::check(o.slot[4] == 3 && o.id[3] == 4 && o.slot[3] == 4, "both sides of an expert exchange");
    o.counts.assign(8, 0);
    o.observe();
    test::check(o.scores[4] == 4 && o.update(4, o.scores.data(), exchange) == 1 && o.slot[5] < 4,
                "decayed counts retain a frequent expert for the next epoch");
    test::check(o.update(0, o.scores.data(), exchange) == 0 && o.update(8, o.scores.data(), exchange) == 0,
                "empty and full resident sets have no swaps");
  }
  // The expert profile: its bytes round-trip, and only a whole file of the same fingerprint and layers is read.
  {
    const std::vector<cuda::ProfileLayer> written{{{3, 4}, {9, 0, 7}}, {{5, 6}, {1, 2, 3, 4}}};
    const uint64_t print = 0x1234abcdull;
    const std::string bytes = cuda::encode_expert_profile(print, written);
    auto empty = [&] {
      std::vector<cuda::ProfileLayer> layers = written;
      for (auto& layer : layers) layer.scores.assign(layer.scores.size(), 0);
      return layers;
    };
    auto layers = empty();
    test::check(cuda::decode_expert_profile(bytes, print, &layers) && layers[0].scores == written[0].scores &&
                    layers[1].scores == written[1].scores,
                "profile round trip");
    auto refused = [&](const std::string& b, uint64_t p, std::vector<cuda::ProfileLayer> l) {
      const auto before = l;
      if (cuda::decode_expert_profile(b, p, &l)) return false;
      for (size_t i = 0; i < l.size(); ++i)
        if (l[i].scores != before[i].scores) return false;
      return true;
    };
    test::check(refused(bytes, print + 1, empty()), "another model's profile");
    test::check(refused(bytes.substr(0, bytes.size() - 1), print, empty()), "a cut profile");
    test::check(refused(bytes + '\0', print, empty()), "a profile with bytes after it");
    test::check(refused(bytes.substr(0, 20), print, empty()), "a damaged header");
    auto old_version = bytes;
    old_version[8] ^= 1;
    test::check(refused(old_version, print, empty()), "incompatible profile version");
    auto other = empty();
    other[1].weights.second = 7;
    test::check(refused(bytes, print, other), "a profile of other projections");
    other = empty();
    other[0].scores.resize(4);
    test::check(refused(bytes, print, other), "a profile of another expert count");
    test::check(refused(bytes, print, {empty()[0]}), "a profile of another layer count");

    const auto folder = std::filesystem::temp_directory_path() /
                        ("flyx-cold-dot-profile-" + std::to_string(std::random_device{}()));
    const auto file = cuda::expert_profile_file(folder, print);
    auto increments = written;
    test::check(!cuda::load_expert_profile(file, print, &increments), "cache miss");
    test::check(cuda::merge_expert_profile(file, print, &increments), "create local profile");
    increments = written;
    test::check(cuda::merge_expert_profile(file, print, &increments), "merge another engine's counts");
    layers = empty();
    test::check(cuda::load_expert_profile(file, print, &layers) && layers[0].scores[0] == 18 &&
                layers[1].scores[3] == 8, "two writers accumulate routes once each");
    auto writer = [&] {
      auto delta = written;
      return cuda::merge_expert_profile(file, print, &delta);
    };
    bool other_ok = false;
    std::thread another([&] { other_ok = writer(); });
    const bool own_ok = writer();
    another.join();
    test::check(own_ok && other_ok, "concurrent cache merge");
    test::check(cuda::load_expert_profile(file, print, &layers) && layers[0].scores[0] == 36,
                "overlapping writers preserve both increments");
    std::string corrupt = bytes;
    corrupt[corrupt.size() - 10] ^= 1;
    test::check(refused(corrupt, print, empty()), "checksum rejects modified scores");
    test::check(cuda::add_routes(UINT64_MAX - 2, 9) == UINT64_MAX, "route counts saturate");
    std::error_code ec;
    std::filesystem::remove_all(folder, ec);

    test::check(cuda::expert_profile_folder("0").empty(), "expert_profile=0 names no folder");
    test::check(cuda::expert_profile_folder("D:/p/q") == std::filesystem::path("D:/p/q"), "a folder given");
    const auto cache = cuda::expert_profile_folder("auto");
#ifdef _WIN32
    const auto* root = _wgetenv(L"LOCALAPPDATA");
#else
    const auto* root = std::getenv("XDG_CACHE_HOME");
#endif
    if (root && *root)
      test::check(cache == std::filesystem::path(root) / "flyx" / "expert_profiles", "user-local cache root");
    test::check(file.filename() == "v2-000000001234abcd.bin", "cache key includes the format and model");

    // The fingerprint: a router's every byte, a large projection's ends, every tensor's format and shape.
    std::vector<unsigned char> router(4096, 1), projection(1u << 20, 2);
    flyx_tensor_view r{"r", FLYX_DT_F32, 2, {32, 32}, router.data(), router.size()};
    flyx_tensor_view w{"w", FLYX_DT_GGML_Q8_0, 2, {1024, 1024}, projection.data(), projection.size()};
    auto print_of = [&] {
      return cuda::fingerprint_tensor(cuda::fingerprint_tensor(cuda::kFingerprintSeed, &r, true), &w, false);
    };
    const uint64_t first = print_of();
    router[2000] = 9;
    const uint64_t router_changed = print_of();
    router[2000] = 1;
    projection[projection.size() / 2] = 9;
    const uint64_t middle_changed = print_of();
    projection[projection.size() - 1] = 9;
    const uint64_t end_changed = print_of();
    w.dtype = FLYX_DT_GGML_Q4_K;
    test::check(router_changed != first && middle_changed == first && end_changed != first && print_of() != end_changed,
                "profile fingerprint: router bytes, projection ends, formats");
  }
  for (bool spin : {false, true}) {
    cuda::RowPool pool(4, spin);
    for (int repeat = 0; repeat < 12; ++repeat) {
      std::vector<int> rows(3 + repeat);
      pool.run(rows.size(), [&](size_t a, size_t b) {
        for (; a < b; ++a) ++rows[a];
      });
      test::check(std::all_of(rows.begin(), rows.end(), [](int n) { return n == 1; }), "row pool reuse");
      bool caught = false;
      try {
        pool.run(12, [&](size_t a, size_t) {
          if (a == size_t(repeat % 4) * 3) throw std::runtime_error("row");
        });
      } catch (const std::runtime_error&) {
        caught = true;
      }
      test::check(caught, "caller and worker exceptions reach the submitter");
    }
  }
  {
    float values[65]{};
    values[1] = INFINITY;
    values[32] = -INFINITY;
    values[64] = std::numeric_limits<float>::quiet_NaN();
    cuda::ColdQ8 groups[3]{};
    cuda::prepare_cold_q8(values, groups, 65);
    for (const auto& g : groups) test::check(g.d == 0.f && g.q[0] == 0, "non-finite group has a finite zero scale");
    values[1] = std::numeric_limits<float>::quiet_NaN();
    values[2] = -3.f;
    cuda::prepare_cold_q8(values, groups, 32);
    test::check(groups[0].q[1] == 0 && groups[0].q[2] == -127, "NaN and finite values in one Q8 group");
    for (auto dtype : {FLYX_DT_GGML_NVFP4, FLYX_DT_GGML_MXFP4}) {
      uint8_t w[36]{};
      float x[64]{};
      x[0] = 1.f;
      if (dtype == FLYX_DT_GGML_MXFP4) {
        w[0] = w[17] = 255;
        w[1] = 1;  // half of 2^128 is representable
      } else {
        w[0] = 127;
        w[1] = 255;
        w[4] = w[12] = 0x77;
        x[16] = 1.f;
      }
      const float expected = dtype == FLYX_DT_GGML_MXFP4 ? std::ldexp(1.f, 127) : 0.f;
      for (int bits : {0, 256, 512})
        test::check(cuda::cold_dot_limited(dtype, w, x, 64, cuda::cpu_isa_limit(bits)) == expected,
                    "FP4 extreme scale");
    }
  }
  std::mt19937 rng(91211);
  for (int dtype : {FLYX_DT_BF16, FLYX_DT_GGML_NVFP4, FLYX_DT_GGML_MXFP4, FLYX_DT_GGML_Q2_0, FLYX_DT_GGML_IQ1_S,
                    FLYX_DT_GGML_IQ2_S, FLYX_DT_GGML_IQ2_XS, FLYX_DT_GGML_IQ2_XXS, FLYX_DT_GGML_IQ3_XXS,
                    FLYX_DT_GGML_IQ4_NL, FLYX_DT_GGML_Q2_K, FLYX_DT_GGML_Q4_0, FLYX_DT_GGML_Q5_0, FLYX_DT_GGML_Q3_K,
                    FLYX_DT_GGML_Q4_K, FLYX_DT_GGML_Q5_K, FLYX_DT_GGML_Q6_K, FLYX_DT_GGML_Q8_0}) {
    int per = dtype == FLYX_DT_BF16 ? 1 : flyx_block_values(dtype);
    int bytes = dtype == FLYX_DT_BF16 ? 2 : flyx_block_bytes(dtype);
    constexpr int cols = 512;
    std::vector<uint8_t> data(cols / per * bytes);
    for (auto& b : data) b = uint8_t(rng());
    for (size_t i = 0; i < data.size(); i += bytes) {
      if (dtype == FLYX_DT_GGML_NVFP4) {
        for (int g = 0; g < 4; ++g) data[i + g] &= 127;
      } else if (dtype == FLYX_DT_GGML_MXFP4)
        data[i] = uint8_t(120 + rng() % 12);
      else {
        const uint8_t sign = uint8_t((i / bytes) % 2) << 7;
        data[i + 1] = uint8_t((dtype == FLYX_DT_BF16 ? 62 : 52) | sign);
        if (dtype == FLYX_DT_GGML_Q4_K || dtype == FLYX_DT_GGML_Q5_K) data[i + 3] = 48;
        if (dtype == FLYX_DT_GGML_Q3_K || dtype == FLYX_DT_GGML_Q6_K) data[i + bytes - 1] = 52;
        if (dtype == FLYX_DT_GGML_Q2_K) {
          data[i + bytes - 3] = 52;
          data[i + bytes - 1] = 48;
        }
      }
    }
    flyx_tensor_view view{};
    view.dtype = dtype;
    view.data = data.data();
    std::vector<float> x(cols);
    double expected = 0, norm = 0;
    for (int k = 0; k < cols; ++k) {
      x[k] = float(int(rng() % 4097) - 2048) / 2048.f;
      const double term = ops::load(&view, k) * x[k];
      expected += term;
      norm += std::fabs(term);
    }
    std::vector<cuda::ColdQ8> quant(cols / 32);
    cuda::prepare_cold_q8(x.data(), quant.data(), cols);
    std::vector<cuda::ColdQ8> scalar(cols / 32);
    cuda::prepare_cold_q8(x.data(), scalar.data(), cols, cuda::CpuIsa::scalar);
    auto rounded = x;
    cuda::round_cold_q8(rounded.data(), cols, cuda::CpuIsa::avx2);
    bool same = true;
    for (int i = 0; i < cols; ++i) {
      const auto& q = scalar[i / 32];
      same =
          same && q.d == quant[i / 32].d && q.q[i % 32] == quant[i / 32].q[i % 32] && rounded[i] == q.d * q.q[i % 32];
    }
    test::check(same, "CPU SIMD and scalar Q8 scales, codes and rounded values agree");
    double expected_q8 = 0, norm_q8 = 0;
    for (int k = 0; k < cols; ++k) {
      const auto& group = quant[k / 32];
      const double term = ops::load(&view, k) * (double(group.d) * group.q[k % 32]);
      expected_q8 += term;
      norm_q8 += std::fabs(term);
    }
    for (int isa : {256, 512})
      if (auto dot = cuda::cold_q8_kernel(dtype, cuda::cpu_isa_limit(isa)))
        test::check(std::fabs(dot(data.data(), quant.data(), cols) - expected_q8) <= 2e-6 * norm_q8,
                    "integer cold dot vs float64 for dtype " + std::to_string(dtype) + " ISA " + std::to_string(isa));
    for (int bits : {256, 512}) {
      const auto isa = cuda::cpu_isa_limit(bits);
      for (bool packed : {false, true}) {
        cuda::ExpertKernel shared = nullptr;
        auto qdot = packed ? cuda::cold_q8_kernel(dtype, isa, &shared) : nullptr;
        auto fdot = cuda::cold_float_kernel(dtype, cols, isa, packed ? nullptr : &shared);
        if (!shared) continue;
        std::vector<float> values(8 * cols);
        std::vector<cuda::ColdQ8> qs(8 * cols / 32);
        int slots[8] = {7, 0, 5, 2, 1, 6, 3, 4};
        for (int i = 0; i < 8; ++i) {
          auto* row = values.data() + i * cols;
          std::rotate_copy(x.begin(), x.begin() + i, x.end(), row);
          cuda::prepare_cold_q8(row, qs.data() + i * (cols / 32), cols, isa);
        }
        for (int n = 1; n <= 8; ++n) {
          cuda::ActRows activations{slots, n, 1, cols, dtype, values.data(), qs.data(), 0.5f, fdot, qdot};
          float result[16]{};
          shared(data.data(), data.size(), 0, 1, activations, result, 2);
          for (int r = 0; r < n; ++r) {
            const float want = packed ? qdot(data.data(), qs.data() + slots[r] * (cols / 32), cols)
                                      : fdot(data.data(), values.data() + slots[r] * cols, cols);
            test::check(result[2 * r] == 0.5f * want && result[2 * r + 1] == 0,
                        "shared expert dot preserves single-row bits");
          }
        }
      }
    }
    for (int n : {31, 47, 64, 96, 288}) {
      double sum = 0, abs_sum = 0;
      for (int i = 0; i < n; ++i) {
        const auto& q = quant[i / 32];
        const double term = ops::load(&view, i) * (double(q.d) * q.q[i % 32]);
        sum += term;
        abs_sum += std::fabs(term);
      }
      for (int bits : {256, 512}) {
        if (auto dot = cuda::cold_q8_kernel(dtype, cuda::cpu_isa_limit(bits)))
          test::check(std::fabs(dot(data.data(), quant.data(), n) - sum) <= 2e-6 * abs_sum, "partial integer dot");
      }
    }
    for (int isa : {0, 256, 512}) {
      test::check(std::fabs(cuda::cold_dot_limited(dtype, data.data(), x.data(), cols, cuda::cpu_isa_limit(isa)) -
                            expected) <= 2e-6 * norm,
                  "compressed CPU GEMV vs float64 for dtype " + std::to_string(dtype));
    }
  }
  return test::finish("cold_dot");
}
