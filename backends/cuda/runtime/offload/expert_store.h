// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <vector>

#include "backends/cuda/runtime/offload/expert_archive.h"

struct flyx_bmodel;
namespace flyx::cuda {
struct Pass;
struct MoeBlock;
struct MoeScratch;
struct ExpertMat;

struct ColdQ8 {
  int8_t q[32];
  float d;
};

// Bound the pinned buffers and scratch by chunk size, independently of prompt length.
struct ColdShape {
  static constexpr int chunk = FLYX_MAX_ROWS;
  size_t bank_bytes = 0;
  size_t inputs = 0, pairs = 0, outputs = 0, activations = 0;
  void include(int hidden, int inter, int top_k) {
    size_t count = size_mul(chunk, size_t(top_k));
    pairs = std::max(pairs, count);
    inputs = std::max(inputs, size_mul(chunk, size_t(hidden)));
    outputs = std::max(outputs, size_mul(count, size_t(hidden)));
    activations = std::max(activations, size_mul(count, size_t(inter)));
  }
  size_t device_bytes() const {
    return size_add(size_mul(size_add(outputs, size_mul(2, pairs)), sizeof(float)), size_mul(2, bank_bytes));
  }
  size_t quant_bytes() const { return size_mul((inputs + 31) / 32 + (activations + 31) / 32, sizeof(ColdQ8)); }
  size_t pinned_bytes() const {
    return size_add(size_mul(size_add(inputs, outputs), sizeof(float)),
                    size_add(size_mul(pairs, sizeof(int)), size_mul(2, bank_bytes)));
  }
  size_t pageable_bytes() const {
    return size_add(quant_bytes(), size_add(size_mul(activations, sizeof(float)), size_mul(pairs, sizeof(int))));
  }
};

enum class CpuIsa { scalar, avx2, avx512 };
inline CpuIsa cpu_isa_limit(int bits) {
  return bits >= 512 ? CpuIsa::avx512 : bits >= 256 ? CpuIsa::avx2 : CpuIsa::scalar;
}

using QuantDot = float (*)(const uint8_t*, const ColdQ8*, int);
QuantDot cold_q8_kernel(int dtype, CpuIsa max_isa = CpuIsa::avx512);
void prepare_cold_q8(const float* input, ColdQ8* out, int cols);

using FloatDot = float (*)(const uint8_t*, const float*, int);
FloatDot cold_float_kernel(int dtype, int cols, CpuIsa limit);

struct ActRows {
  const int* slots;
  int n_rows, divisor, cols, dtype;
  const float* values;
  const ColdQ8* quant;
  float gain;
  FloatDot float_dot;
  QuantDot quant_dot;
};
using ExpertKernel = void (*)(const uint8_t* weights, size_t stride, int begin, int end, const ActRows& x, float* out,
                              size_t ld_out);

// Pageable, model-owned copy. The ABI caller may unmap the source immediately after open.
struct ColdMatrix {
  int first, rows, cols, dtype;
  size_t stride;
  bool q8, a16h, nvfp4;
  CpuIsa max_isa = CpuIsa::avx512;
  QuantDot quant_dot = nullptr;
  FloatDot float_dot = nullptr;
  ExpertKernel kernel = nullptr;
  void resolve(int simd, bool integer);
  std::vector<uint8_t> data;
  std::vector<float> scales;
  ExpertBankLayout bank{};
  std::vector<uint8_t> archive;
  struct DevicePlanes {
    void* payload = nullptr;
    void* scale = nullptr;
    float* gains = nullptr;
  } resident;
  void exchange(int hot_slot, int cold_slot, int total, uint8_t* bounce);
  void pack_banks(const ExpertLayout& layout, size_t all);
  ColdMatrix(const flyx_weight& w, int hot, const ExpertLayout& layout);
  size_t bytes() const { return data.size() + scales.size() * sizeof(float) + archive.size(); }
  void prepare(float* input, bool vector_nvfp4) const;
};

float cold_dot_limited(int dtype, const uint8_t* row, const float* input, int cols, CpuIsa max_isa);
float cold_dot(int dtype, const uint8_t* row, const float* input, int cols);
float cold_dot_scalar(int dtype, const uint8_t* row, const float* input, int cols);
void hybrid_experts(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate, bool fused);

}  // namespace flyx::cuda
