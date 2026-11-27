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
struct ExpertMat;

// Contiguous CUDA planes for a batch of experts. The largest single expert also fits when it exceeds the DMA target.
struct ExpertBankLayout {
  size_t payload, scale, gains, count;
  static ExpertBankLayout from(const ExpertLayout& layout, size_t experts, size_t cold) {
    const size_t bytes = layout.bytes() / experts;
    return {layout.payload_bytes() / experts, layout.scale_bytes() / experts, layout.expert_scales ? sizeof(float) : 0,
            std::min(cold, std::max<size_t>(1, (16u << 20) / bytes))};
  }
  size_t scale_offset(size_t n) const { return checkpoint_align(size_mul(payload, n)); }
  size_t gain_offset(size_t n) const { return checkpoint_align(size_add(scale_offset(n), size_mul(scale, n))); }
  size_t bytes(size_t n) const { return size_add(gain_offset(n), size_mul(gains, n)); }
  size_t archive_bytes(size_t n) const { return size_add(size_mul(n / count, bytes(count)), bytes(n % count)); }
};

struct ColdQ8 {
  int8_t q[32];
  float d;
};

// Bound the pinned buffers and scratch by chunk size, independently of prompt length.
struct ColdShape {
  static constexpr int chunk = 8;
  size_t bank_bytes = 0;
  size_t inputs = 0, pairs = 0, outputs = 0, activations = 0;
  void include(int hidden, int inter, int top_k) {
    size_t count = size_mul(chunk, size_t(top_k));
    pairs = std::max(pairs, count);
    inputs = std::max(inputs, size_mul(chunk, size_t(hidden)));
    outputs = std::max(outputs, size_mul(count, size_t(hidden)));
    activations = std::max(activations, size_mul(count, size_t(inter)));
  }
  size_t device_bytes() const { return size_add(size_mul(size_add(outputs, size_mul(2, pairs)), sizeof(float)), size_mul(2, bank_bytes)); }
  size_t quant_bytes() const { return size_mul((inputs + 31) / 32 + (activations + 31) / 32, sizeof(ColdQ8)); }
  size_t host_bytes() const {
    return size_add(size_mul(size_add(size_add(inputs, outputs), activations), sizeof(float)),
                    size_add(quant_bytes(), size_add(size_mul(2 * sizeof(int), pairs), size_mul(2, bank_bytes))));
  }
};

using QuantDot = float (*)(const uint8_t*, const ColdQ8*, int);
QuantDot cold_q8_kernel(int dtype, int max_isa = 512);
void prepare_cold_q8(const float* input, ColdQ8* out, int cols);

// Pageable, model-owned copy. The ABI caller may unmap the source immediately after open.
struct ColdMatrix {
  int first, rows, cols, dtype;
  size_t stride;
  bool q8, a16h, nvfp4;
  QuantDot quant_dot = nullptr;
  std::vector<uint8_t> data;
  std::vector<float> scales;
  ExpertBankLayout bank{};
  std::vector<uint8_t> archive;
  void exchange(const ExpertMat& resident, int hot_slot, int cold_slot, int total, uint8_t* bounce);
  void pack_banks(const ExpertLayout& layout, size_t all);
  ColdMatrix(const flyx_weight& w, int hot, const ExpertLayout& layout);
  size_t bytes() const { return data.size() + scales.size() * sizeof(float) + archive.size(); }
  float product(int expert, int row, const float* input, const ColdQ8* quant) const;
  void prepare(float* input, bool vector_nvfp4) const;
};

void pack_expert_nvfp4(const uint8_t* in, size_t blocks, uint8_t* qo, uint8_t* so);
float cold_dot_limited(int dtype, const uint8_t* row, const float* input, int cols, int max_isa);
float cold_dot(int dtype, const uint8_t* row, const float* input, int cols);
float cold_dot_scalar(int dtype, const uint8_t* row, const float* input, int cols);
void cold_sequence_open(flyx_bmodel* m);
void cold_sequence_close(flyx_bmodel* m);
void init_cold_experts(flyx_bmodel* m);
void hybrid_experts(const Pass& ps, const MoeBlock& f, const MoeScratch* sc, float* out, bool accumulate, bool fused);

}  // namespace flyx::cuda
