// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <vector>

#include "backends/cuda/loader/layout.h"

struct flyx_bmodel;
namespace flyx::cuda {
struct Pass;
struct MoeBlock;
struct MoeScratch;

// Bound the pinned buffers and scratch by chunk size, independently of prompt length.
struct ColdShape {
  static constexpr int chunk = 8;
  size_t inputs = 0, pairs = 0, outputs = 0, activations = 0;
  void include(int hidden, int inter, int top_k) {
    size_t count = size_mul(chunk, size_t(top_k));
    pairs = std::max(pairs, count);
    inputs = std::max(inputs, size_mul(chunk, size_t(hidden)));
    outputs = std::max(outputs, size_mul(count, size_t(hidden)));
    activations = std::max(activations, size_mul(count, size_t(inter)));
  }
  size_t device_bytes() const { return size_mul(size_add(outputs, pairs), sizeof(float)); }
  size_t host_bytes() const {
    return size_add(size_mul(size_add(size_add(inputs, outputs), activations), sizeof(float)),
                    size_mul(2 * sizeof(int), pairs));
  }
};

// Pageable, model-owned copy. The ABI caller may unmap the source immediately after open.
struct ColdMatrix {
  int first, rows, cols, dtype;
  size_t stride;
  bool q8, a16h, nvfp4;
  std::vector<uint8_t> data;
  std::vector<float> scales;
  ColdMatrix(const flyx_weight& w, int hot, const ExpertLayout& layout);
  size_t bytes() const { return data.size() + scales.size() * sizeof(float); }
  float product(int expert, int row, const float* input) const;
  void prepare(float* input, bool vector_nvfp4) const;
};

float cold_dot(int dtype, const uint8_t* row, const float* input, int cols);
float cold_dot_scalar(int dtype, const uint8_t* row, const float* input, int cols);
void init_cold_experts(flyx_bmodel* m);
void hybrid_experts(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate, bool fused);

}  // namespace flyx::cuda
