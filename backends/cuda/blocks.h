// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The model as the CUDA backend runs it: typed blocks with
// their dimensions, device weights and the state they keep, and the table of
// the state a sequence allocates for them.  loader/bind.cpp builds both from the
// description's tables (backend_abi.h), one block here per block there, with
// the weights laid out as the kernels want them (q|k|v and gate|up stacked);
// the regions, row buffers, sequence state, commit, rollback and checkpoints
// read only these.  supported() (regions/plan.cpp) says which blocks the kernels take.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "backends/cuda/kernels/kernels.cuh"
#include "flyx/backend_abi.h"

namespace flyx::cuda {

// A linear layer's weight [N, K]: bf16 (w), NVFP4 (q, s, rs; see kernels.cuh),
// or GGML blocks kept packed (blk: the tensors stacked in it, in row order;
// neighbouring tensors of one block type are one matrix, a GEMM each).
struct Lin {
  bf16* w = nullptr;
  uint8_t *q = nullptr, *s = nullptr;
  float* rs = nullptr;
  std::vector<BlockMat> blk{};
  bool packed() const { return !blk.empty(); }
  // Block weights whose inputs are 8 bits in every pass (FLYX_ACT_Q8): a fused producer writes them so.
  bool q8_inputs() const { return packed() && blk[0].q8; }
};

// RMSNorm: y = x / rms(x) * (offset + w).
struct Norm {
  float* w = nullptr;
  float offset = 0.f, eps = 0.f;
};

// A block's own low-rank adapters (backend_abi.h flyx_lora, ABI 43: a V-LoRA), y = W x + scale b (a x) on each of its
// linears, one rank: the a's of the linears that read one input stacked as their W's are (attention: a_in [3 rank,
// hidden] for q | k | v, a_out [rank, nh hd] for o; MLP: a_in [2 rank, hidden] for gate | up, a_out [rank, inter] for
// down), then each linear's b (attention q, k, v, o; MLP gate, up, down).  rank 0: none.
struct Adapters {
  int rank = 0;
  float scale = 1.f;
  bf16 *a_in = nullptr, *a_out = nullptr;
  bf16* b[4] = {};
  explicit operator bool() const { return rank > 0; }
};

// ---- state: what a sequence keeps ---------------------------------------------------------
// A K/V cache: positions x nkv x hd, `layers` caches stacked (a drafter's
// layers, written by one kernel).  Positional: rollback only moves the position.
struct KvSpec {
  int layers = 1, nkv = 0, hd = 0;
  bool fp8 = false;    // e4m3 with a scale per position and KV head (one cache only)
  bool draft = false;  // only drafting reads it (the MTP layer's): on the device, never in tiers
  int window = 0;      // >0: sliding-window ring cache kept on device (never tiered)
};
// Recurrent state of a depthwise ShortConv layer (Inkling-2): (conv - 1) x channels window
// in FLYX_ROUND_SLOTS slots, round replay inputs, and checkpoint copies.
struct ShortConvSpec {
  int conv = 0, channels = 0;
};
// Recurrent state of a gated delta net (GdnDims): state and conv window in
// FLYX_ROUND_SLOTS slots, the round's per-row inputs kept for the commit
// fold and rewinds, and a copy per checkpoint.
// Hidden ring: vectors of `ring_width` floats for the last kHring positions;
// a checkpoint keeps the one before its position.
struct StateTable {
  std::vector<KvSpec> kv;
  std::vector<GdnDims> gdn;
  std::vector<ShortConvSpec> sconv;
  int ring_width = 0;  // 0: none
};

// ---- blocks ---------------------------------------------------------------------------------
// Token rows -> residual stream rows of width `hidden`.
struct EmbedBlock {
  int hidden = 0, vocab = 0;
  bf16* table = nullptr;  // [vocab, hidden]; null when packed
  BlockMat blk;           // the table as packed GGML blocks (data null: bf16)
  bool has_norm = false;
  Norm norm{};            // optional post-embedding RMSNorm (Inkling-2)
};

// norm -> q (| output gate) | k | v -> q/k norms, rotary on the first `rot`
// dims -> attention over its cache -> (* sigmoid(gate)) -> output projection,
// added to the residual.
struct AttnBlock {
  int hidden = 0, nh = 0, nkv = 0, hd = 0, rot = 0;
  double theta = 0;
  bool gate = false;           // the q projection is [q (nh*hd) | gate (nh*hd)]
  Norm norm, q_norm, k_norm;   // q_norm and k_norm share offset and eps; w null: none (Llama's heads as projected)
  Lin w_in, w_out;             // w_in: the projections stacked as above
  Adapters lora;               // its own adapters on q, k, v and o (never with a gate)
  int kv = -1;                 // its cache (StateTable::kv)
  int q_width() const { return (gate ? 2 : 1) * nh * hd; }
  int n_in() const { return q_width() + 2 * nkv * hd; }
};

// norm -> [mixed (q|k|v) | z | b | a] -> causal conv, gated delta rule over
// its state -> gated norm (w, eps; offset 0) -> output projection, added to the residual.
struct GdnBlock {
  int hidden = 0;
  GdnDims gd{};
  Norm norm, gated;
  Lin w_in, w_out;
  float *conv_w = nullptr, *A_log = nullptr, *dt_bias = nullptr;
  int state = -1;              // StateTable::gdn
  int v_width() const { return gd.nv * gd.dv; }
  int n_in() const { return gd.C() + v_width() + 2 * gd.nv; }
};

// norm -> gate | up -> silu(gate) * up -> down, added to the residual.
struct MlpBlock {
  int hidden = 0, inter = 0;
  Norm norm;
  Lin w_gu, w_down;
  Adapters lora;               // its own adapters on gate, up and down
};

// Final norm -> logits -> the token choice per row.
struct HeadBlock {
  int hidden = 0, vocab = 0;   // vocab is vocab_out (unpadded) when vocab_out > 0
  float softcap = 0.f;         // >0: z = softcap * tanh(z / softcap)
  Norm norm;
  bf16* w = nullptr;           // [vocab, hidden]; the embedding table when tied; null when packed
  BlockMat wb;                 // the head as packed GGML blocks (data null: bf16); no copies below then
  Lin draft;                   // NVFP4 copy draft logits read; empty: w
  uint8_t* w8 = nullptr;       // option head_fp8 / head_cert: an e4m3 copy [vocab, hidden]
  float* rs8 = nullptr;        // its row scales
  W12 w12;                     // a lossless 12-bit copy the target's logits read (1..16 rows); hi null: none
  int w12_escapes = 0;         // its escape list's length as allocated (what trim() gives back)
  SignHead coarse;             // drafts' 1-bit copy (draft_pick); bits null: none
  float* cert = nullptr;       // option head_cert: the e4m3 copy's bound coefficient per row (head_cert_coef)
};

// Multi-token prediction: from (target hidden, next token) pairs one decoder
// layer predicts the token after.  Gather: [norm_emb(embed[token]) |
// norm_hidden(hidden)] with (1 + w) norms -> fc -> the attention and MLP
// blocks -> norm -> the head's draft logits.  Its context keeps the target's
// final-normed hidden in the ring.
struct MtpBlock {
  int embed = -1, head = -1, attn = -1, mlp = -1;  // blocks it runs or reads
  Norm norm_emb, norm_hidden, norm;
  Lin fc;                                          // [hidden, 2 * hidden]
};

// DFlash2: a block of `block` rows (the anchor, then mask tokens) through its
// layers attending to the context K/V, then unary top-k from the head and the
// selector walk.  Its layers wrap attention and MLP in grouped dynamic convs.
struct DFlashLayer {
  Norm norm, post_norm, q_norm, k_norm;
  Lin w_qkv, w_o, w_gu, w_down;
  Lin attn_proj, mlp_proj;                    // the convs' coefficient deltas
  float *attn_base = nullptr, *mlp_base = nullptr;  // and bases, [2][taps][hidden]
};
struct DFlashBlock {
  int hidden = 0, nh = 0, nkv = 0, hd = 0, inter = 0;
  double theta = 0;
  int block = 0, mask_token = 0;
  int window = 0;                             // sliding window of every layer (0: none)
  int n_taps = 0, target_hidden = 0;          // context features: n_taps target hidden rows
  int conv_taps = 0, conv_group = 0, sel_rank = 0, sel_topk = 0;
  int embed = -1, head = -1;
  int first_layer = -1, n_layers = 0;         // blocks (DFlashLayer)
  Lin fc, ctx, sel_hid;                       // ctx: every layer's context k | v stacked
  bf16 *sel_pred = nullptr, *sel_succ = nullptr;
  Norm hidden_norm, norm, k_norms;            // k_norms: every layer's, stacked
  int kv = -1;                                // context K/V, one cache per layer
  int layers() const { return n_layers; }
  int ctx_width() const { return layers() * 2 * nkv * hd; }
};

// DV (backend_abi.h flyx_dv_block, docs/design/dv.md): `slots` canvas rows run the target's attention and MLP blocks
// (their base weights) with a layer's own norms and a low-rank adapter on each linear, y = W x + b (a x).  The
// adapters that read one input are stacked: a_qkv [3 rank, hidden] for q | k | v, a_gu [2 rank, hidden] for gate | up.
struct DvLayer {
  int attn = -1, mlp = -1;                    // the target's blocks (AttnBlock, MlpBlock)
  int rank = 0;
  float scale = 1.f;                          // the adapters' (bifill-dv's: 1)
  Norm norm, post_norm;                       // ln1c, ln2c
  Lin a_qkv, a_o, a_gu, a_down;               // the adapters' a
  Lin b_q, b_k, b_v, b_o, b_gate, b_up, b_down;  // and b
};
// The drafter: the canvas's final norm, MASK embedding, residual context (res_in [2 hidden, hidden]: u then g;
// res_out [hidden, hidden]), the selector (A, H, B [sel_rank, hidden], float32) and the tree it drafts.  The
// selector's token tables A E and B E are made when a plan first drafts with them (make_dv_tables) and counted with
// the weights; a sequence keeps the canvas (flyx_seq::dv).
struct DvBlock {
  int hidden = 0, vocab = 0, slots = 0;
  int tree = FLYX_DV_TREE_BEAM, tree_width = 0, beam = 0, pool_ranks = 0;
  bool cov = false, causal = false;
  int topk = 0, res_k = 0, mask_token = 0;
  float fill = 0.f, sel_alpha = 0.f, sel_temp = 1.f;
  int embed = -1, head = -1;
  int first_layer = -1, n_layers = 0;         // blocks (DvLayer)
  Norm norm;                                  // fnormc
  const bf16* mask_emb = nullptr;             // [hidden], bf16 as stored (the canvas input reads it so)
  Lin res_in, res_out;
  int sel_rank = 0;
  float *sel_a = nullptr, *sel_h = nullptr, *sel_b = nullptr;
  float *sel_ta = nullptr, *sel_tb = nullptr;  // [vocab][sel_rank]: A E[t] and B E[t] for every token t
};

// Inkling-2 relative-position-bias attention (SWA or Global) with AttnRes, optional cross-pass loop control,
// Q/K/R L2 clipping, no-gamma V RMSNorm, per-query sink, and depthwise ShortConv on K/V/output.
struct RelAttnBlock {
  int hidden = 0, nh = 0, nkv = 0, hd = 0;
  int d_rel = 0, rel_extent = 0, window = 0, conv = 0, log_scale_floor = 0;
  float qk_clip = 0.f, rel_clip = 0.f, log_scale_alpha = 0.f, eps = 0.f;
  Norm q_norm_in, k_norm_in, v_norm_in;
  Lin w_qgr, w_q, w_gate, w_rel;  // stacked {q, gate, rel} when compatible, else separate
  bool stacked_qgr = false;
  Lin w_k, w_v, w_o, w_sink, w_k_conv_proj, w_v_conv_proj;
  float *rel_bias = nullptr, *q_clip_w = nullptr, *rel_clip_w = nullptr;
  float *k_conv = nullptr, *v_conv = nullptr, *post_conv = nullptr;
  float *res_query = nullptr, *loop_msg_query = nullptr;
  int res_push = 0, loop_rank = 0;
  Lin loop_ctrl_down[3], loop_ctrl_gf[3], loop_ctrl_gain[3], loop_ctrl_forget[3];
  bool loop_ctrl_stacked[3] = {};
  int kv = -1, sconv = -1;
};

// Inkling-2 sparse MoE block with AttnRes, optional cross-pass loop control, SiTU activation, and post ShortConv.
struct MoeBlock {
  int hidden = 0, inter = 0, n_experts = 0, top_k = 0, conv = 0;
  bool gating_sigmoid = true, norm_weights = true;
  float route_scale = 1.f, norm_eps = 1e-20f, situ_beta = 0.f, situ_linear_beta = 0.f;
  Norm norm;
  Lin router;
  float* router_bias = nullptr;
  ExpertMat gate_up, down;
  float *post_conv = nullptr, *res_query = nullptr;
  int loop_rank = 0;
  Lin loop_ctrl_down, loop_ctrl_gf, loop_ctrl_gain, loop_ctrl_forget;
  bool loop_ctrl_stacked = false;
  int sconv = -1, sconv_offset = 0;
};

// DSpark companion drafter (Inkling-2): 3-layer DFlash with block-uniform SWA/Full + Markov RNN head.
struct DSparkLayer {
  int window = 0;
  Norm norm, post_norm, q_norm, k_norm, k_ctx_norm, ctx_norm;
  Lin w_qkv, w_q, w_k, w_v, w_o;
  bool stacked_qkv = false;
  Lin w_ctx_kv, w_k_ctx, w_v_ctx;
  bool stacked_ctx_kv = false;
  Lin w_gu, w_gate, w_up, w_down;
  bool stacked_gu = false;
  float* ctx_mix = nullptr;  // [n_taps]
};
struct DSparkBlock {
  int hidden = 0, nh = 0, nkv = 0, hd = 0, inter = 0;
  int block = 0, n_taps = 0, markov_rank = 0, vocab_out = 0;
  double theta = 0;
  float softcap = 0.f;
  int embed = -1, head = -1, first_layer = -1, n_layers = 0;
  Norm tok_norm, norm, markov_ctx_norm;
  float* mask_embd = nullptr;       // [block - 1, hidden]
  float* markov_ctx_mix = nullptr;  // [n_taps]
  bf16* markov_w1 = nullptr;        // [vocab_out, markov_rank] (null when packed)
  BlockMat markov_w1_blk{};         // packed GGML blocks for markov_w1 (data null: bf16)
  Lin markov_w2;                    // [vocab_out, markov_rank]
  float* markov_w2_b = nullptr;     // [vocab_out] (nullable)
  Lin joint_proj;                   // [3 * markov_rank, 2 * hidden + 2 * markov_rank]
  float* joint_proj_b = nullptr;    // [3 * markov_rank]
  int kv = -1;                      // StateTable::kv
  int layers() const { return n_layers; }
};

using Block = std::variant<EmbedBlock, AttnBlock, GdnBlock, MlpBlock, HeadBlock, MtpBlock, DFlashBlock, DFlashLayer,
                           DvBlock, DvLayer, RelAttnBlock, MoeBlock, DSparkBlock, DSparkLayer>;

// ---- a sequence's state, as the table declares it -------------------------------------------
struct KvBuf {
  void *k = nullptr, *v = nullptr;  // [layers][cap][nkv][hd]: binary16 (kv16), or e4m3 (fp8)
  float *ks = nullptr, *vs = nullptr;
  size_t layer_stride = 0;          // elements between stacked caches
  // In tiers: positions [device_positions, cap) of k and v are page-locked host memory. vk and vv
  // are views of them, the same device memory before device_positions and past it a device staging the sequence's
  // caches share; stage() copies the host part there for the attention that reads the views next (null: one tier).
  void *vk = nullptr, *vv = nullptr;
  int device_positions = 0, cap = 0;
  size_t row = 0;  // bytes a position of k (and of v)
  int slot = -1;   // the staging k's view maps (flyx_seq::staging; v's is its own of the same parity)
  // The positions past device_positions the staging holds (fewer than the host part's with strata
  // attention); and the stagings of each parity a prefill's windows of keys take (k's index, k's and v's memory).
  int stage_rows = INT32_MAX;
  int wslot[2] = {-1, -1};
  uint8_t *wk[2] = {nullptr, nullptr}, *wv[2] = {nullptr, nullptr};
  // FP8 scales where the positions are: ks and vs those of [0, device_positions) on the device;
  // ks_host and vs_host every position's, the ones in host memory kept there; vks and vvs the views' stagings of them
  // (from device_positions on), wks and wvs the windows' (from the window's first position).  nkv: scales a position.
  float *ks_host = nullptr, *vs_host = nullptr, *vks = nullptr, *vvs = nullptr;
  float *wks[2] = {nullptr, nullptr}, *wvs[2] = {nullptr, nullptr};
  float* hot_scale = nullptr;  // the scales of the pairs kept on the device (KvRef::hot_scale)
  int nkv = 0;
  // A staging of scales addressed by position: position `first` at its start.
  float* scales_from(float* staging, int first) const {
    return reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(staging) - size_t(first) * nkv * sizeof(float));
  }
  // The low-bit copy strata attention reads (KvRef::kq); null: none. On the device; or, for caches
  // in tiers with a staging, at one of two homes: kq_dev and vq_dev in the device pool the stagings
  // share (while the rounds read only the copy), kq_host and vq_host in page-locked host memory (while passes read
  // the caches through the stagings).  copy_off: the pool's bytes before kq_dev; q_group: bytes of 16 positions of kq
  // (all heads: StrataLayout's unit); q_bytes: of kq (and of vq), the pool keeping vq at copy_off + q_span().
  uint32_t *kq = nullptr, *vq = nullptr;
  uint32_t *kq_dev = nullptr, *vq_dev = nullptr, *kq_host = nullptr, *vq_host = nullptr;
  size_t copy_off = 0, q_group = 0, q_bytes = 0;
  size_t q_span() const { return (q_bytes + 255) / 256 * 256; }
  // A copy in host memory (flyx_seq_layout::copy_host): kq and vq are always its host home, where the
  // passes write it; a round that reads only the copy reads kq_stage and vq_stage, a device staging in the pool that
  // copy_stage fills (copy_slot: which one; the copies take turns).  Null: the copy is on the device (above).
  uint32_t *kq_stage = nullptr, *vq_stage = nullptr;
  __half2 *kqs_stage = nullptr, *vqs_stage = nullptr;  // its grids' (kqs and vqs in host memory too)
  int copy_slot = -1;
  size_t grid_span() const { return (size_t(cap) * nkv * sizeof(__half2) + 255) / 256 * 256; }
  __half2 *kqs = nullptr, *vqs = nullptr;
  int bits = 0, words = 0, grid = 0;
  // The pairs in host memory strata attention re-read lately, on the device (KvRef::hot); null: none.
  int32_t *hot_tag = nullptr, *hot_claim = nullptr;
  uint8_t* hot = nullptr;
  int hot_bits = 0;
  size_t hot_tags = 0;  // slots, all heads
  // The MTP layer's: its draft steps attend to its first `sink` positions and its last `window` (0:
  // all), which a ring of `ring` slots holds (0: a slot a position).
  // SWA target layers (KvSpec::window > 0): fixed GPU ring buffer (`swa_ring = true`), checkpointed on host.
  int ring = 0, sink = 0, window = 0;
  bool swa_ring = false;
  void* ckp_k[FLYX_CHECKPOINTS]{};
  void* ckp_v[FLYX_CHECKPOINTS]{};
  float* ckp_ks[FLYX_CHECKPOINTS]{};
  float* ckp_vs[FLYX_CHECKPOINTS]{};
  KvRef ref() const {  // where a pass writes
    KvRef r{k, v, ks, vs, kq, vq, kqs, vqs, bits, words, grid, hot_tag, hot_claim, hot, hot_bits,
            vk ? device_positions : INT32_MAX, ring, sink};
    if (ks_host) r.ks_far = ks_host, r.vs_far = vs_host, r.scale_split = device_positions;
    r.hot_scale = hot_scale;
    return r;
  }
  // What an attention reads: the caches, or after stage() their views.
  KvRef staged() const {
    if (!vk) return ref();
    KvRef r{vk, vv, ks, vs};
    if (ks_host) {
      r.ks_far = scales_from(vks, device_positions), r.vs_far = scales_from(vvs, device_positions);
      r.scale_split = device_positions;
    }
    return r;
  }
  // What strata attention reads: the copy where it is, or, in a round that reads only the copy, after copy_stage()
  // the staging of a copy in host memory.
  KvRef copy_read(bool copy_only) const {
    KvRef r = ref();
    if (copy_only && kq_stage) r.kq = kq_stage, r.vq = vq_stage, r.kqs = kqs_stage, r.vqs = vqs_stage;
    return r;
  }
  // Copies (a part of) the positions an attention of rows [P->position + pos_add, + span) reads from host memory to
  // the views (nothing for one tier; kv_stage).  flyx_seq::stage_before decides which.
  void stage(const DevParams* P, int pos_add, int span, KvStagePart part, int blocks, cudaStream_t st) const {
    if (!vk) return;
    KvStageScales sc;
    if (ks_host) sc = {ks_host, vs_host, scales_from(vks, device_positions), scales_from(vvs, device_positions), nkv};
    kv_stage(k, vk, v, vv, row, device_positions, staged_cap(), P, pos_add, span, part, blocks, st, 1, sc);
  }
  // The positions the view holds: past them its staging ends.
  int staged_cap() const { return int(std::min<long long>(cap, (long long)device_positions + stage_rows)); }
  // The same for a copy in host memory: its 16s of positions to its staging (StrataLayout's unit, q_group bytes).
  void copy_stage(const DevParams* P, int span, KvStagePart part, int blocks, cudaStream_t st, int from = 0) const {
    if (!kq_stage) return;
    const KvStageScales grids{reinterpret_cast<const float*>(kqs), reinterpret_cast<const float*>(vqs),
                              reinterpret_cast<float*>(kqs_stage), reinterpret_cast<float*>(vqs_stage), nkv};
    kv_stage(kq, kq_stage, vq, vq_stage, q_group, from, cap, P, 0, span, part, blocks, st, 16, grids);
  }
  kv16* k16(int layer = 0) const { return static_cast<kv16*>(k) + size_t(layer) * layer_stride; }
  kv16* v16(int layer = 0) const { return static_cast<kv16*>(v) + size_t(layer) * layer_stride; }
};
struct GdnBuf {
  float *S = nullptr, *win = nullptr;  // FLYX_ROUND_SLOTS slots each
  size_t s_stride = 0, w_stride = 0;   // floats per slot
  GdnBufs replay{};                    // the round's rows, a slot each
  float* ckp_S[FLYX_CHECKPOINTS]{};  // the checkpoints' state and window, in host memory
  float* ckp_win[FLYX_CHECKPOINTS]{};
};
struct ShortConvBuf {
  int conv = 0, channels = 0;
  float *win = nullptr, *replay = nullptr;  // FLYX_ROUND_SLOTS and kGdnReplaySlots slots each
  size_t w_stride = 0, r_stride = 0;        // floats per slot
  float* ckp_win[FLYX_CHECKPOINTS]{};       // host checkpoints
};

}  // namespace flyx::cuda
