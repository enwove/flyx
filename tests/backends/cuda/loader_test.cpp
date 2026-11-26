// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Link the production binder and runtime with a backend.cpp whose device allocator can refuse a small request.
#include "backends/cuda/loader/loader.h"

namespace flyx::cuda {
namespace {
bool refuse_room = false;
cudaError_t allocation_error = cudaSuccess;

void require(bool ok, const char* what) {
  if (!ok) throw std::runtime_error(std::string(what) + ": " + g_error);
}

void oom_paths() {
  flyx_bmodel m;
  m.hidden = m.vocab = m.need.k_max = 64;
  m.opt.graphs = true;
  flyx_node node{FLYX_NODE_COMMIT, 0, 0};
  flyx_program prefill{FLYX_PHASE_PREFILL, 0, 32, 1, &node};
  flyx_plan* plan = nullptr;
  refuse_room = true;
  require(lower(&m, &prefill, &plan) == FLYX_ERR_OOM && !plan, "prefill slab OOM through lower");
  refuse_room = false;

  const flyx_seq_layout layout{64, INT32_MAX, FLYX_KV_READ_ALL, 0, 0};
  flyx_seq* seq = nullptr;
  allocation_error = cudaErrorMemoryAllocation;
  require(seq_new_in(&m, &layout, &seq) == FLYX_ERR_OOM && !seq, "device arena OOM through seq_new_in");
  flyx_plan round{};
  plan_lowered(&m, &round);  // graph_room must create its counting sequence
  flyx_seq_size size{};
  require(seq_size(&m, &layout, &size) == FLYX_ERR_OOM &&
              g_error.find("sizing the round graphs") != std::string::npos,
          "counting sequence arena OOM through seq_size");
  allocation_error = cudaErrorInvalidValue;
  require(seq_size(&m, &layout, &size) == -1, "other arena errors retain their status");
  allocation_error = cudaSuccess;
}

void staging_width() {
  const size_t limit = kLoadSlotBytes / sizeof(uint16_t);
  require(quant_row_bytes(limit, "matrix") == kLoadSlotBytes, "a BF16 row fits at the limit");
  flyx_bmodel m;
  flyx_tensor_view view{"matrix", FLYX_DT_BF16, 2, {1, int64_t(limit + 1)}};
  flyx_weight weight{&view};
  flyx_model_desc desc{};
  desc.n_weights = 1;
  desc.weights = &weight;
  WeightLoader loader(&m, desc);
  refuse_room = true;  // an accidental allocation fails before any input could be read
  for (int type : {FLYX_DT_BF16, FLYX_DT_F16, FLYX_DT_F32, FLYX_DT_F64}) {
    view.dtype = type;
    for (auto quant : {DraftQuant::kNvfp4, DraftQuant::kQ8}) {
      for (bool reorder : {false, true}) {
        std::vector<WeightLoader::RowOrder> orders;
        if (reorder) orders.emplace_back([](size_t i) { return i; });
        for (int extra : {1, 64}) {
          view.shape[1] = int64_t(limit + extra);
          bool refused = false;
          try {
            loader.lin_stack_quant({0}, quant, nullptr, orders);
          } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("64 MiB") != std::string::npos;
          }
          require(refused && m.bytes == 0, "wide quantized row refused before output allocation");
        }
      }
    }
  }
  refuse_room = false;
}

template <class T>
bool on_device(const T* ptr, const std::vector<T>& expected) {
  std::vector<T> actual(expected.size());
  CK(cudaMemcpy(actual.data(), ptr, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return actual == expected;
}

void quantized_bytes(DraftQuant quant) {
  constexpr int rows = 3, cols = 64;
  std::vector<float> values(rows * cols);
  for (int i = 0; i < rows * cols; ++i) values[i] = float(i % 31 - 15) * float(i / cols + 1);
  flyx_tensor_view view{"matrix", FLYX_DT_F32, 2, {rows, cols}, values.data(), values.size() * sizeof(float)};
  flyx_weight weight{&view};
  flyx_model_desc desc{};
  desc.n_weights = 1;
  desc.weights = &weight;
  flyx_bmodel m;
  WeightLoader loader(&m, desc);
  const auto reverse = [](size_t r) { return rows - 1 - r; };
  const Lin lin = loader.lin_stack_quant({0}, quant, nullptr, {reverse});
  loader.finish();
  if (quant == DraftQuant::kQ8) {
    std::vector<uint8_t> payload(rows * cols);
    std::vector<uint16_t> scales(rows * cols / 32);
    for (int r = 0; r < rows; ++r)
      ops::quantize_row_q8_0(values.data() + reverse(r) * cols, cols, cols, payload.data() + r * cols,
                             scales.data() + r * cols / 32);
    require(on_device(lin.blk.at(0).data, payload) && on_device(lin.blk[0].scales, scales), "Q8 loader planes");
  } else {
    std::vector<uint8_t> payload(rows * cols / 2), scales(rows * cols / 16);
    std::vector<float> rs(rows);
    for (int r = 0; r < rows; ++r)
      ops::quantize_row_nvfp4(values.data() + reverse(r) * cols, cols, cols, payload.data() + r * cols / 2,
                              scales.data() + r * cols / 16, rs.data() + r);
    require(on_device(lin.q, payload) && on_device(lin.s, scales) && on_device(lin.rs, rs), "NVFP4 loader planes");
  }
}

void vision_estimates() {
  flyx_tensor_view view{"vision", FLYX_DT_F32, 2, {3, 64}};
  flyx_weight weight{&view};
  flyx_block block{};
  block.kind = FLYX_BLOCK_VISION_MERGE;
  block.vision_merge.fc1 = 0;
  block.vision_merge.fc2 = block.vision_merge.fc1_b = block.vision_merge.fc2_b = -1;
  block.vision_merge.norm.w = block.vision_merge.norm.b = -1;
  flyx_model_desc desc{};
  desc.weights = &weight;
  desc.n_weights = 1;
  desc.blocks = &block;
  desc.n_blocks = 1;
  flyx_open_options opts{};
  opts.vision_device = FLYX_VISION_DEV_GPU;
  for (int cols : {64, 65}) {
    view.shape[1] = cols;
    view.nbytes = 3 * cols * sizeof(float);
    for (int quant : {FLYX_VISION_QUANT_Q8, FLYX_VISION_QUANT_NVFP4}) {
      opts.vision_quant = quant;
      flyx_load_estimate e{};
      require(estimate_load(&desc, &opts, 0, 0, &e) == 0, "estimate vision weights");
      const size_t row = quant == FLYX_VISION_QUANT_Q8 ? (cols == 64 ? 68 : 102) : (cols == 64 ? 40 : 76);
      require(e.weights_bytes == 3 * row, "F32 vision projection padded bytes");
    }
  }
  // A separate Q/K/V stack containing float and file blocks is quantized as a whole.
  view.shape[1] = 64;
  view.nbytes = 3 * 64 * sizeof(float);
  auto packed = view;
  packed.dtype = FLYX_DT_GGML_Q4_0;
  packed.nbytes = 3 * 2 * 18;
  flyx_weight qkv[3] = {{&packed}, {&view}, {&view}};
  block = {};
  block.kind = FLYX_BLOCK_VIT_ATTN;
  block.vit_attn.qkv = block.vit_attn.o = -1;
  block.vit_attn.q = 0;
  block.vit_attn.k = 1;
  block.vit_attn.v = 2;
  desc.weights = qkv;
  desc.n_weights = 3;
  opts.vision_quant = FLYX_VISION_QUANT_Q8;
  flyx_load_estimate e{};
  require(estimate_load(&desc, &opts, 0, 0, &e) == 0 && e.weights_bytes == 9 * 68,
          "mixed-format Q/K/V estimate uses the upload format of the stack");
}

void expert_estimates() {
  constexpr int e = 2, n = 3, k = 256;
  struct Case { int dtype, file_row, device_row; bool packed, scale; };
  for (const auto& c : {
           Case{FLYX_DT_GGML_Q4_K, 144, 148, true, false}, Case{FLYX_DT_GGML_Q4_K, 144, 148, true, true},
           Case{FLYX_DT_GGML_Q4_K, 144, 512, false, true}, Case{FLYX_DT_GGML_Q8_0, 272, 272, true, false},
           Case{FLYX_DT_GGML_NVFP4, 144, 144, true, false}, Case{FLYX_DT_GGML_NVFP4, 144, 144, false, true},
           Case{FLYX_DT_F32, 1024, 512, true, true}, Case{FLYX_DT_BF16, 512, 512, true, false}}) {
    std::vector<uint8_t> data(e * n * c.file_row);
    float scales[e] = {1.f, 2.f};
    flyx_tensor_view v{"experts", c.dtype, 3, {e, n, k}, data.data(), data.size()};
    flyx_tensor_view s{"scales", FLYX_DT_F32, 1, {e}, scales, sizeof(scales)};
    flyx_weight w{&v, nullptr, c.scale ? &s : nullptr};
    flyx_block b{};
    b.kind = FLYX_BLOCK_MOE;
    b.moe.gate_up = b.moe.down = 0;
    b.moe.router = -1;
    flyx_model_desc d{};
    d.weights = &w;
    d.n_weights = 1;
    d.blocks = &b;
    d.n_blocks = 1;
    flyx_open_options opts{};
    opts.options = c.packed ? "blocks=packed" : "blocks=bf16";
    flyx_load_estimate est{};
    const size_t expected = e * n * c.device_row + (c.scale || c.dtype == FLYX_DT_GGML_NVFP4 ? e * sizeof(float) : 0);
    require(estimate_load(&d, &opts, 0, 0, &est) == 0 && est.weights_bytes == expected, "exact expert estimate");
    flyx_bmodel m;
    m.opt.packed_blocks = c.packed;
    WeightLoader loader(&m, d);
    loader.expert_mat(0);
    loader.finish();
    require(m.bytes == expected, "actual expert upload matches estimate");
  }
}

void packed_drafter(bool compact) {
  constexpr int cols = 64;
  std::vector<uint8_t> data(68);  // two zero Q8_0 blocks
  flyx_tensor_view view{"matrix", FLYX_DT_GGML_Q8_0, 2, {1, cols}, data.data(), data.size()};
  flyx_weight weight{&view};
  flyx_model_desc desc{};
  desc.n_weights = 1;
  desc.weights = &weight;
  flyx_bmodel m;
  m.opt.draft_weights = compact ? interfaces::DraftWeightsMode::kNvfp4 : interfaces::DraftWeightsMode::kAuto;
  WeightLoader loader(&m, desc);
  const Lin lin = loader.lin_quant(0, DraftQuant::kNvfp4);  // DFlash2 beside an NVFP4 target
  loader.finish();
  require(compact ? lin.q && m.bytes == 40 : lin.packed() && m.bytes == 68, "packed drafter policy and bytes");
  // What is reported is what the drafter holds: a Q8_0 file kept packed under auto is q8, not the nvfp4 asked.
  require(std::string(held_precision({&lin})) == (compact ? "nvfp4" : "q8"), "a drafter's precision as it is held");
}

void held_labels() {
  Lin dense, fp4, q8, q4k, unbound;
  dense.w = reinterpret_cast<bf16*>(uintptr_t(16));  // only the pointers' presence is read
  fp4.q = reinterpret_cast<uint8_t*>(uintptr_t(16));
  q8.blk = {BlockMat{nullptr, nullptr, FLYX_DT_GGML_Q8_0, 1, true}};
  q4k.blk = {BlockMat{nullptr, nullptr, FLYX_DT_GGML_Q4_K, 1, true}};
  require(std::string(held_precision({})).empty() && std::string(held_precision({&unbound})).empty(),
          "no drafter linear");
  require(std::string(held_precision({&dense, &dense, &fp4})) == "exact", "mostly the file's bf16");
  require(std::string(held_precision({&q4k, &q4k})) == "exact", "GGML blocks of another type kept as they are");
  require(std::string(held_precision({&q8, &q8, &fp4})) == "q8", "mostly Q8_0 blocks");
  require(std::string(held_precision({&fp4, &q8})) == "nvfp4", "NVFP4 planes on a tie");
}
}  // namespace

void test_device_room(size_t) {
  if (refuse_room) throw OutOfDeviceMemory("injected device reservation refusal");
}

cudaError_t test_cuda_malloc(void** ptr, size_t bytes) {
  return allocation_error == cudaSuccess ? cudaMalloc(ptr, bytes) : allocation_error;
}
}  // namespace flyx::cuda

int main() {
  using namespace flyx::cuda;
  try {
    vision_estimates();
    expert_estimates();
    staging_width();
    oom_paths();
    quantized_bytes(DraftQuant::kQ8);
    quantized_bytes(DraftQuant::kNvfp4);
    packed_drafter(false);
    packed_drafter(true);
    held_labels();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
