// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Load-only weight uploader: maps description weights onto device or pinned host memory during open_model(),
// keeping transient dedup maps and double-buffered staging scoped to bind() instead of flyx_bmodel.
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "backends/cuda/backend.h"

namespace flyx::cuda {

// Page-locked buffers the packed rows go through to the device, two in turn: the rows of one are laid out while the
// other is copied (WeightLoader::pack_rows). Made at the first upload, freed when WeightLoader ends.
struct LoadStaging {
  static constexpr size_t kBytes = size_t(64) << 20;
  uint8_t* buf[2] = {nullptr, nullptr};
  cudaEvent_t done[2] = {nullptr, nullptr};  // the last copy out of each
  cudaStream_t st = nullptr;
  int next = 0;
  LoadStaging();
  ~LoadStaging();
};

class RowPool;

class WeightLoader {
 public:
  // The stored row that row r of an upload is, for a weight whose file keeps
  // its rows in another order than the kernels (empty: the same order).
  using RowOrder = std::function<size_t(size_t)>;

  WeightLoader(flyx_bmodel* model, const flyx_model_desc& desc);
  ~WeightLoader();
  WeightLoader(const WeightLoader&) = delete;
  WeightLoader& operator=(const WeightLoader&) = delete;

  flyx_bmodel* model() const { return m_; }
  const LoadTimes& times() const { return times_; }

  // Wait for any in-flight asynchronous packed-weight copies and release the staging buffers and worker pool.
  void finish();

  const flyx_tensor_view* t(int w) const;
  static size_t numel(const flyx_tensor_view* v);
  static bool numbers(const flyx_tensor_view* v);

  // Copy a [rows, K] matrix into dst (bf16, device or host_alloc memory), converting if stored in another type.
  void put_bf16(bf16* dst, int w, const RowOrder& order = {}, bool host = false);
  // Copy `bytes` from host (e.g. mmapped file) to device memory via double-buffered pinned staging for large tensors.
  void upload_staged(void* dst, const void* src, size_t bytes);
  bf16* mat(int w);
  // A matrix in host memory (host_alloc), for row gathers only: not shared through mats.
  bf16* mat_host(int w);
  // Several matrices with the same K stacked by rows; orders (optional): a row order per matrix.
  bf16* stack(const std::vector<int>& ws, int* rows_out = nullptr, const std::vector<RowOrder>& orders = {});

  // Whether a matrix stays packed: a GGML block type the kernels take (and option blocks is not bf16).
  bool packable(int w) const;
  // A packable matrix on the device as gemm_blocks reads it.
  BlockMat pack(int w, const RowOrder& order = {});
  // The same in host memory (host_alloc), for row gathers only: not shared through packs.
  BlockMat pack_host(int w) { return pack_run({w}, {}, true); }
  // Neighbouring tensors of one block type and width as one matrix of all their rows.
  BlockMat pack_run(const std::vector<int>& ws, const std::vector<RowOrder>& orders, bool host = false);
  // One tensor's rows at data and scales (device memory, or host_alloc memory when host).
  void pack_rows(int w, const RowOrder& order, uint8_t* data, uint16_t* scales, bool host);

  // A linear weight as stored: NVFP4 stays NVFP4 (with its scales), GGML blocks stay packed, anything else -> bf16.
  Lin lin(int w) { return lin_stack({w}); }
  // Several linears with the same K stacked by rows (one GEMM computes them all; packed blocks: one GEMM per run
  // of neighbouring tensors of one block type, each run one matrix, on the shared input).
  Lin lin_stack(const std::vector<int>& ws, int* rows_out = nullptr, const std::vector<RowOrder>& orders = {});

  std::vector<float> host_f32(int w) const;
  float* vec(int w) { return vecs({w}); }
  float* vecs(const std::vector<int>& ws);
  float* put(const std::vector<float>& values);

 private:
  void parallel_rows(size_t n, const std::function<void(size_t, size_t)>& fn);

  flyx_bmodel* m_ = nullptr;
  const flyx_weight* weights_ = nullptr;
  int n_weights_ = 0;
  std::unordered_map<int, bf16*> mats_;      // matrices uploaded once (a tied head is the embedding)
  std::unordered_map<int, BlockMat> packs_;  // block weights packed once
  std::unique_ptr<LoadStaging> staging_;
  std::unique_ptr<RowPool> pool_;            // destroyed before staging_ if unwinding without finish()
  LoadTimes times_{};
};

}  // namespace flyx::cuda
