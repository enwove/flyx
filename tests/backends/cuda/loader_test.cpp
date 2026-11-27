// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Link the production binder and runtime with a backend.cpp whose device allocator can refuse a small request (or
// fail its room check otherwise).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "backends/cuda/loader/loader.h"

namespace flyx::cuda {
namespace {
bool refuse_room = false;
bool room_error = false;  // the room check fails otherwise than for want of memory
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

  // A session's device arena, in pieces where the driver maps memory (device_pieces) or else one allocation, and the
  // counting sequence's single piece (device_alias): each asks device_room first.
  const flyx_seq_layout layout{64, INT32_MAX, FLYX_KV_READ_ALL, 0, 0};
  flyx_seq* seq = nullptr;
  refuse_room = true;
  allocation_error = cudaErrorMemoryAllocation;
  require(seq_new_in(&m, &layout, &seq) == FLYX_ERR_OOM && !seq, "device arena OOM through seq_new_in");
  flyx_plan round{};
  plan_lowered(&m, &round);  // graph_room must create its counting sequence
  flyx_seq_size size{};
  require(seq_size(&m, &layout, &size) == FLYX_ERR_OOM &&
              g_error.find("sizing the round graphs") != std::string::npos,
          "counting sequence arena OOM through seq_size");
  refuse_room = false;
  allocation_error = cudaSuccess;
  room_error = true;
  require(seq_size(&m, &layout, &size) == -1, "other arena errors retain their status");
  room_error = false;
}

// NVML's figures as gpu_memory makes them the device's free and total memory on Windows: the total less what the
// driver reserves of the device for itself where nvmlDeviceGetMemoryInfo_v2 tells it, the used memory nvidia-smi
// shows plus the free memory (RTX 5070 Ti, driver 610.88: 16,303 MiB, 308 reserved).  Without v2, the free memory
// alone; with figures that do not add up, cudaMemGetInfo's.
void memory_figures() {
  constexpr uint64_t kMiB = 1ull << 20;
  auto made = [](const NvmlFigures& n, size_t* free, size_t* total) {
    *free = 1, *total = 16303 * kMiB;  // cudaMemGetInfo's
    memory_of(n, free, total);
  };
  size_t free = 0, total = 0;
  NvmlFigures v2;
  v2.v2 = true, v2.total = 16303 * kMiB, v2.reserved = 308 * kMiB, v2.free = 10326 * kMiB;
  made(v2, &free, &total);
  require(free == 10326 * kMiB && total == 15995 * kMiB, "the reserved memory is no program's: out of the total");
  NvmlFigures v1;
  v1.v1 = true, v1.total = 16303 * kMiB, v1.free = 10326 * kMiB;
  made(v1, &free, &total);
  require(free == 10326 * kMiB && total == 16303 * kMiB, "without v2 the free memory alone");
  NvmlFigures odd = v2;
  odd.reserved = 16303 * kMiB;
  made(odd, &free, &total);
  require(free == 1 && total == 16303 * kMiB, "a reserve as large as the device: cudaMemGetInfo's figures");
  odd = v2;
  odd.free = 16000 * kMiB;
  made(odd, &free, &total);
  require(free == 1 && total == 16303 * kMiB, "more free than the device less its reserve: cudaMemGetInfo's");
  made(NvmlFigures{}, &free, &total);
  require(free == 1 && total == 16303 * kMiB, "no NVML: cudaMemGetInfo's figures");
}

// What this process holds of the device as Windows counts it (DXGI, the adapter of device 0); 0 where the system
// does not tell a process's share, and the checks of what the memory took are skipped: the device's free memory is
// every program's, and the GPU tests run beside each other.
int64_t held_now() {
  CK(cudaDeviceSynchronize());
  return int64_t(gpu_process_memory());
}

// device_pieces: device memory in pieces of kPieceBytes (or of the granularity, where it is larger) that hold what is
// written to each, and take what they map; device_alias: a range as long over one piece, where a byte written shows at
// the same offset in every other piece; both give their memory back when released (a session's arena and the counting
// sequence's: build_seq).
void pieces_and_alias() {
  constexpr size_t kMiB = 1ull << 20, kBytes = 64 * kMiB + 12345;
  CK(cudaSetDevice(0));
  remember_device(0);  // the adapter gpu_process_memory reads
  const size_t gran = vmm_granularity(), piece = std::max(gran, kPieceBytes / gran * gran);
  const int64_t before = held_now();
  const bool counted = before > 0;
  if (!counted) std::printf("  device_pieces, device_alias: no process memory figure, what they take not checked\n");
  {
    DevicePieces pieces;
    unsigned char* p = static_cast<unsigned char*>(device_pieces(kBytes, &pieces));
    if (!p) {
      std::printf("  skip device_pieces: the driver maps no memory\n");
      return;
    }
    require(pieces.bytes >= kBytes && pieces.bytes % gran == 0 && pieces.mapped == pieces.bytes &&
                pieces.handles.size() == (pieces.bytes + piece - 1) / piece,
            "device_pieces: the range in pieces");
    for (size_t i = 0; i < pieces.handles.size(); ++i) CK(cudaMemset(p + i * piece, int(i + 1), 16));
    std::vector<unsigned char> got(pieces.handles.size());
    for (size_t i = 0; i < got.size(); ++i) CK(cudaMemcpy(&got[i], p + i * piece, 1, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < got.size(); ++i) require(got[i] == (i + 1) % 256, "device_pieces: each piece its own");
    const int64_t took = held_now() - before;
    require(!counted || (took >= int64_t(64 * kMiB) && took <= int64_t(pieces.bytes + 4 * kMiB)),
            "device_pieces: what it maps");
  }
  require(!counted || held_now() <= before + int64_t(2 * kMiB), "device_pieces: the memory back when released");
  {
    DevicePieces alias;
    unsigned char* p = static_cast<unsigned char*>(device_alias(kBytes, &alias));
    require(p && alias.alias && alias.handles.size() == 1 && alias.bytes >= kBytes && alias.mapped == alias.bytes,
            "device_alias: one piece");
    const int64_t took = held_now() - before;
    require(!counted || took <= int64_t(gran + 2 * kMiB), "device_alias: one piece of memory under the range");
    CK(cudaMemset(p + 5, 0x5a, 1));
    unsigned char at[2] = {};
    CK(cudaMemcpy(&at[0], p + gran + 5, 1, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&at[1], p + alias.bytes - gran + 5, 1, cudaMemcpyDeviceToHost));
    require(at[0] == 0x5a && at[1] == 0x5a, "device_alias: every piece the same memory");
  }
  require(!counted || held_now() <= before + int64_t(2 * kMiB), "device_alias: the memory back when released");
}

void residual_query_extent() {
  RelAttnBlock a;
  a.hidden = 2304;
  a.window = 1024;
  require(a.res_query_bytes() == 2304 * sizeof(float), "SWA warms one residual query");
  a.window = 0;
  require(a.res_query_bytes() == 3 * 2304 * sizeof(float), "full attention warms three residual queries");
}

void drafter_dimensions() {
  flyx_bmodel m;
  DFlashBlock b;
  m.hidden = b.hidden = b.target_hidden = b.hd = 64;
  b.n_taps = b.nh = b.nkv = 1;
  b.block = b.sel_topk = 2;
  b.dspine.code = 256;
  b.dspine.rank = kDSpineMaxRank;
  std::string why;
  require(supported(&m, b, &why), "the widest portable DSpine rank opens");
  b.dspine.rank += 8;
  require(!supported(&m, b, &why), "a rank beyond portable shared memory is refused");
  b.dspine.rank = 128;
  DFlashLayer l;
  l.attn_proj.blk = {BlockMat{nullptr, nullptr, FLYX_DT_GGML_Q8_0, 1, true}};
  m.blocks = {b, l};
  require(row_needs(&m).k_max == 256 && row_needs(&m).q8, "scratch covers the code and Q8 convolution");
  std::swap(b.dspine.rank, b.dspine.code);
  m.blocks[0] = b;
  require(row_needs(&m).k_max == 256, "scratch covers the rank");

  flyx_block block{};
  block.kind = FLYX_BLOCK_DSPINE;
  block.dspine.n_taps = 1;
  flyx_model_desc desc{};
  desc.blocks = &block;
  desc.n_blocks = 1;
  desc.hidden = INT32_MAX - 63;
  require(kernel_limits(desc, &why), "largest padded DSpine tap row");
  ++desc.hidden;
  require(!kernel_limits(desc, &why), "DSpine tap padding would overflow");
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

// Three distinct floating projections: the target's MLP, DSpine's fc and a co-loaded DSpark layer's q.
void companion_estimates() {
  const flyx_tensor_view matrix{"matrix", FLYX_DT_BF16, 2, {3, 64}, nullptr, 3 * 128};
  const flyx_weight weights[] = {{}, {&matrix}, {&matrix}, {&matrix}};
  flyx_block blocks[4]{};
  blocks[0].kind = FLYX_BLOCK_MLP;
  blocks[0].mlp.gate = blocks[0].mlp.up = blocks[0].mlp.down = 1;
  blocks[1].kind = FLYX_BLOCK_DSPINE;
  auto& spine = blocks[1].dspine;
  spine.fc = 2;
  spine.hidden = 64;
  spine.rank = 16;
  spine.layers = 3;
  spine.head_dim = 32;
  blocks[2].kind = FLYX_BLOCK_DSPARK;
  blocks[3].kind = FLYX_BLOCK_DSPARK_LAYER;
  blocks[3].dspark_layer.q = 3;
  flyx_model_desc desc{};
  desc.blocks = blocks;
  desc.n_blocks = 4;
  desc.weights = weights;
  desc.n_weights = 4;
  flyx_open_options opts{};
  flyx_load_estimate est{};
  const size_t copies = 2 * spine.hidden * spine.rank + 4 * spine.layers * spine.head_dim;
  const size_t bf16 = 3 * 128, nvfp4 = 3 * 40;
  require(estimate_load(&desc, &opts, 0, 0, &est) == 0 && est.draft_weights_bytes == nvfp4 + bf16 + copies &&
              est.weights_bytes == bf16 + est.draft_weights_bytes && est.draft_compact_savings == bf16 - nvfp4,
          "AUTO beside a float target: DSpine in NVFP4 and DSpark in BF16, including DSpine's fixed copies");
  opts.draft_weights = FLYX_DRAFT_WEIGHTS_NVFP4;
  require(estimate_load(&desc, &opts, 0, 0, &est) == 0 && est.draft_weights_bytes == 2 * nvfp4 + copies &&
              est.draft_compact_savings == 0,
          "explicit NVFP4 applies to both companions");

  blocks[0] = {};
  blocks[0].kind = FLYX_BLOCK_EAGLE4;
  auto& eagle = blocks[0].eagle4;
  eagle.head = 1;
  eagle.cand_down = 2;
  eagle.cand_up = 3;
  desc.n_blocks = 1;
  for (auto mode : {FLYX_DRAFT_WEIGHTS_AUTO, FLYX_DRAFT_WEIGHTS_NVFP4}) {
    opts.draft_weights = mode;
    require(estimate_load(&desc, &opts, 0, 0, &est) == 0 && est.draft_weights_bytes == 3 * bf16 &&
                est.draft_compact_savings == 0, "EAGLE-4 heads stay BF16 in both load estimates");
  }
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
    b.moe.gate = b.moe.up = b.moe.gate_b = b.moe.up_b = b.moe.down_b = b.moe.router_logit_b = -1;
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
// The environment apply_env_aliases reads in env_aliases: these names and values, every other name unset.
std::vector<std::pair<const char*, const char*>> g_env;
const char* fake_env(const char* name) {
  for (const auto& [key, value] : g_env)
    if (std::strcmp(key, name) == 0) return value;
  return nullptr;
}

const char* const kAliasNames[] = {"FLYX_SC_CLUSTER", "FLYX_MOE_VEC", "FLYX_WARM", "FLYX_EXPERT_PREFETCH", "FLYX_GUESS",
                                   "FLYX_GUESS_K", "FLYX_SIDE_LOOP_MSG", "FLYX_SIDE_COMPANION", "FLYX_SIDE_KV"};

// The options a load gets from the legacy environment spellings and its options text (resolve_options).
BackendOptions resolved(const char* text, std::vector<std::pair<const char*, const char*>> env) {
  g_env = std::move(env);
  BackendOptions o;
  std::string err;
  const bool ok = resolve_options(text, &o, &err, fake_env);
  require(ok, err.c_str());
  return o;
}

bool kernel_defaults(const KernelOptions& k) {
  const KernelOptions def;
  return k.sc_cluster == 8 && def.sc_cluster == 8 && !k.moe_vec && k.warm == 23 && def.warm == 23 &&
         k.expert_prefetch == 2 && def.expert_prefetch == 2 && k.guess == 5 && def.guess == 5 && k.guess_k == 4 &&
         def.guess_k == 4 && k.side_loop_msg && k.side_companion && k.side_kv;
}

// The nine options' environment spellings: unset, empty or not an integer leave the defaults (an empty value read as 0
// turned warming, the expert guesses and the side streams off before); an integer keeps its old meaning; the options
// given (FLYX_BACKEND_OPTIONS, --backend-opt) override them; and a load reads them again, not once for the process.
void env_aliases() {
  require(kernel_defaults(resolved(nullptr, {}).kernels), "no variables: the defaults");
  std::vector<std::pair<const char*, const char*>> empty, junk, spaced;
  for (const char* name : kAliasNames) {
    empty.push_back({name, ""});
    junk.push_back({name, "on"});
    spaced.push_back({name, " 3"});
  }
  require(kernel_defaults(resolved(nullptr, empty).kernels), "empty variables: the defaults");
  require(kernel_defaults(resolved(nullptr, junk).kernels), "variables that are no integer: the defaults");
  require(kernel_defaults(resolved(nullptr, spaced).kernels), "an integer with a space before it: the defaults");
  require(kernel_defaults(resolved(nullptr, {{"FLYX_WARM", "7x"}, {"FLYX_GUESS", "0x5"}}).kernels),
          "a number followed by more: the defaults");
  require(describe(resolved(nullptr, empty)) == "defaults", "empty variables describe as the defaults");

  const KernelOptions set = resolved(nullptr, {{"FLYX_SC_CLUSTER", "4"}, {"FLYX_MOE_VEC", "1"}, {"FLYX_WARM", "0"},
                                               {"FLYX_EXPERT_PREFETCH", "0"}, {"FLYX_GUESS", "0"},
                                               {"FLYX_GUESS_K", "0"}, {"FLYX_SIDE_LOOP_MSG", "0"},
                                               {"FLYX_SIDE_COMPANION", "0"}, {"FLYX_SIDE_KV", "0"}})
                                .kernels;
  require(set.sc_cluster == 4 && set.moe_vec && set.warm == 0 && set.expert_prefetch == 0 && set.guess == 0 &&
              set.guess_k == 0 && !set.side_loop_msg && !set.side_companion && !set.side_kv,
          "every variable read");
  // The integers' old meanings, in the options' values.
  const KernelOptions old = resolved(nullptr, {{"FLYX_SC_CLUSTER", "2"}, {"FLYX_MOE_VEC", "-3"}, {"FLYX_WARM", "-1"},
                                               {"FLYX_EXPERT_PREFETCH", "5"}, {"FLYX_GUESS", "21"},
                                               {"FLYX_GUESS_K", "-2"}, {"FLYX_SIDE_LOOP_MSG", "2"}})
                                .kernels;
  require(old.sc_cluster == 1 && old.moe_vec && old.warm == kWarmAll && old.expert_prefetch == 1 && old.guess == 5 &&
              old.guess_k == 0 && old.side_loop_msg,
          "integers keep their old meanings");
  require(resolved(nullptr, {{"FLYX_SC_CLUSTER", "8"}, {"FLYX_GUESS", "-4"}}).kernels.guess == 0, "a negative guess");

  // The options given override the variables, whichever way.
  const BackendOptions over =
      resolved("sc_cluster=8,moe_vec=0,warm=16,expert_prefetch=2,guess=9,guess_k=3,side_kv=1",
               {{"FLYX_SC_CLUSTER", "1"}, {"FLYX_MOE_VEC", "1"}, {"FLYX_WARM", "0"}, {"FLYX_EXPERT_PREFETCH", "0"},
                {"FLYX_GUESS", "0"}, {"FLYX_GUESS_K", "0"}, {"FLYX_SIDE_KV", "0"}});
  const KernelOptions& k = over.kernels;
  require(k.sc_cluster == 8 && !k.moe_vec && k.warm == 16 && k.expert_prefetch == 2 && k.guess == 9 &&
              k.guess_k == 3 && k.side_kv,
          "options override the variables");
  require(describe(over) == "warm=16,guess=9,guess_k=3", "the options describe what differs from the defaults");
  const KernelOptions given = resolved("sc_cluster=1,warm=0,side_companion=0", empty).kernels;
  require(given.sc_cluster == 1 && given.warm == 0 && !given.side_companion, "options beside empty variables");

  // The options' own values: strict.
  for (const char* bad : {"sc_cluster=2", "moe_vec=2", "warm=8", "warm=-1", "expert_prefetch=3", "guess=16",
                          "guess_k=-1", "side_kv=yes", "warm="}) {
    BackendOptions o;
    std::string err;
    require(!parse_options(bad, &o, &err) && !err.empty(), bad);
  }

  // The process environment (no lookup given: std::getenv), read again at each load.
  auto put = [](const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);  // "" removes the variable
#else
    setenv(name, value, 1);  // "" keeps it, empty
#endif
  };
  for (const char* value : {"7", "", "19"}) {
    put("FLYX_WARM", value);
    BackendOptions o;
    std::string err;
    const bool ok = resolve_options("guess=3", &o, &err);
    require(ok, err.c_str());
    const int want = *value ? std::atoi(value) : 23;
    require(o.kernels.warm == want && o.kernels.guess == 3, "FLYX_WARM from the process at each load");
  }
  put("FLYX_WARM", "");
}
}  // namespace

void test_device_room(size_t) {
  if (refuse_room) throw OutOfDeviceMemory("injected device reservation refusal");
  if (room_error) throw std::runtime_error("injected device reservation error");
}

cudaError_t test_cuda_malloc(void** ptr, size_t bytes) {
  return allocation_error == cudaSuccess ? cudaMalloc(ptr, bytes) : allocation_error;
}
}  // namespace flyx::cuda

int main() {
  using namespace flyx::cuda;
  try {
    vision_estimates();
    companion_estimates();
    expert_estimates();
    residual_query_extent();
    drafter_dimensions();
    staging_width();
    oom_paths();
    memory_figures();
    pieces_and_alias();
    quantized_bytes(DraftQuant::kQ8);
    quantized_bytes(DraftQuant::kNvfp4);
    packed_drafter(false);
    packed_drafter(true);
    held_labels();
    env_aliases();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
