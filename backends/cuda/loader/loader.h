// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Load-only weight uploader: maps description weights onto device or pinned host memory during open_model(),
// keeping transient dedup maps and double-buffered staging scoped to bind() instead of flyx_bmodel.
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "backends/cuda/backend.h"
#include "backends/cuda/loader/layout.h"

namespace flyx::cuda {

// Page-locked buffers the packed rows go through to the device, two in turn: the rows of one are laid out while the
// other is copied (WeightLoader::pack_rows). Made at the first upload, freed when WeightLoader ends.
// Also owns a small double-buffered device BF16 staging pair (2 x 16 MiB, allocated lazily) so drafter linears
// stream-quantize directly into NVFP4/Q8_0 on `st` without uploading a full BF16 copy first.
struct LoadStaging {
  static constexpr size_t kBytes = kLoadSlotBytes;
  static constexpr size_t kQuantBytes = size_t(16) << 20;
  struct Slot {
    uint8_t* buf = nullptr;
    cudaEvent_t done = nullptr;
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() {
      if (done) cudaEventDestroy(done);
      if (buf) cudaFreeHost(buf);
    }
  };
  Slot slots[2];
  bf16* d_quant[2] = {nullptr, nullptr};
  size_t d_quant_bytes = 0;
  cudaStream_t st = nullptr;
  int next = 0;
  LoadStaging();
  ~LoadStaging();
  void ensure_quant_staging(size_t min_bytes);
};

// Target quantization applied on the fly during WeightLoader::lin_stack_quant for drafter linears.
enum class DraftQuant { kNone, kNvfp4, kQ8 };

// What drafter linears hold once uploaded, for the active-precision report (flyx_bmodel::active_draft_weights):
// "nvfp4" (NVFP4 planes), "q8" (Q8_0 blocks, made at load or kept from the file) or "exact" (the file's own
// precision: bf16, or GGML blocks of another type kept as they are), whichever most of `lins` hold; "" when none
// was uploaded.  The policy (DraftQuant) alone does not say: lin_stack_quant keeps packed blocks unless nvfp4 is
// asked for explicitly and comes out smaller, so a Q8_0 drafter under auto beside an NVFP4 target stays Q8_0.
const char* held_precision(const std::vector<const Lin*>& lins);

// Reusable device scratch for WeightLoader::pack_quantize_nvfp4 (allocated lazily and freed in finish()/destructor).
struct PackedQuantScratch {
  int* dids = nullptr;
  uint8_t* d_data = nullptr;
  uint16_t* d_scales = nullptr;
  float* f32 = nullptr;
  bf16* b16 = nullptr;
  size_t cap_ids = 0, cap_data = 0, cap_scales = 0, cap_f32 = 0, cap_b16 = 0;
  ~PackedQuantScratch();
  void ensure(size_t run, size_t max_row_bytes, size_t max_row_scales, size_t K, cudaStream_t st);
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
  void finish_stream_block();

  const flyx_tensor_view* t(int w) const;
  // Whether a block's binding read weight w (anything the loader was asked for).  One that none read is a program
  // node's (FLYX_NODE_RES_MIX names its vector by weight index).
  bool asked(int w) const { return w >= 0 && size_t(w) < asked_.size() && asked_[size_t(w)]; }
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
  // Whether several 2D weights share format and activation rounding so lin_stack keeps them packed/NVFP4 together.
  bool can_stack(const std::vector<int>& ws) const;
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
  // A 3D stacked MoE expert weight [E, N, K] (NVFP4 with optional per-expert scale2, packed GGML blocks, or bf16).
  ExpertMat expert_mat(int w, std::shared_ptr<const ColdMatrix>& cold);
  // Drafter linear with zero-spike GPU streaming quantization directly into NVFP4 or Q8_0 when requested.
  Lin lin_quant(int w, DraftQuant q) { return lin_stack_quant({w}, q); }
  Lin lin_stack_quant(const std::vector<int>& ws, DraftQuant q, int* rows_out = nullptr,
                      const std::vector<RowOrder>& orders = {});

  std::vector<float> host_f32(int w) const;
  // A round's exact float32 values, sharing the row-major prefill matrix (or its slice) when bf16 holds them.
  RoundWeight round_weight(int w, const bf16* prefill);
  // A small control linear with exact float32 weights and activations in rounds for F32, F16 and F64 files.
  Lin lin_f32(int w);
  float* vec(int w);
  float* vecs(const std::vector<int>& ws);
  float* put(const std::vector<float>& values);

  // Vision encoder loaders: honor BackendOptions::vision (GPU vs pinned host RAM) and vision_quant (exact/q8/nvfp4).
  VisionVec vision_vec(int w) { return vision_put(host_f32(w)); }
  VisionVec vision_put(const std::vector<float>& values);
  VisionLin vision_lin(int w) { return vision_lin_stack({w}); }
  VisionLin vision_lin_stack(const std::vector<int>& ws, int* rows_out = nullptr);

 private:
  Lin pack_quantize_nvfp4(const std::vector<int>& ws, int* rows_out, const std::vector<RowOrder>& orders);
  void parallel_rows(size_t n, const std::function<void(size_t, size_t)>& fn);

  flyx_bmodel* m_ = nullptr;
  const flyx_weight* weights_ = nullptr;
  int n_weights_ = 0;
  mutable std::vector<bool> asked_;          // t(w) was called
  std::unordered_map<int, bf16*> mats_;      // matrices uploaded once (a tied head is the embedding)
  std::unordered_map<int, BlockMat> packs_;  // block weights packed once
  std::unordered_map<int, float*> vecs_;     // 1D/small float vectors uploaded once (shared across loop passes)
  std::unordered_map<int, RoundWeight> round_weights_;  // one decision and upload across all loop passes
  std::map<std::vector<int>, Lin> lins_;     // stacked/single linears uploaded once (shared across loop passes)
  std::unordered_map<int, std::shared_ptr<const ColdMatrix>> cold_experts_;
  std::unordered_map<int, ExpertMat> experts_;  // 3D expert tensors uploaded once (shared across loop passes)
  std::unique_ptr<PackedQuantScratch> packed_scratch_;
  std::unique_ptr<LoadStaging> staging_;     // destroyed before packed_scratch_ (syncs st before scratch frees)
  std::unique_ptr<RowPool> pool_;            // destroyed before staging_ if unwinding without finish()
  LoadTimes times_{};
};

}  // namespace flyx::cuda
