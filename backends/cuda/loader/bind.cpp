// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The binder: the description's tables (backend_abi.h) -> blocks and the
// state table (blocks.h), one block for each of the description's, with the
// weights uploaded as the kernels want them: projections that share an input
// stacked (q|k|v, gate|up, the GDN in-projection, every DFlash2 layer's
// context k|v, DV's adapters on one input), norms as float.  Knows block
// kinds, not models.
#include "backends/cuda/loader/loader.h"

#include <algorithm>
#include <cstring>
#include <optional>

namespace flyx::cuda {
namespace {

Norm norm(WeightLoader& L, const flyx_norm& n) { return {L.vec(n.w), float(n.offset), float(n.eps)}; }
// An attention block's q or k norm: none (weight -1, Norm::w null) where the block names none.
Norm head_norm(WeightLoader& L, const flyx_norm& n) { return n.w < 0 ? Norm{} : norm(L, n); }

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}

// The state indices of the description -> this backend's tables.
struct StateMap {
  std::vector<int> kv, gdn, sconv, index;  // per description state: index in StateTable::kv / ::gdn / ::sconv, else -1
  std::vector<bool> ring, canvas;
};

StateMap bind_states(flyx_bmodel* m, const flyx_model_desc& d) {
  StateMap sm;
  // The caches block drafters keep their context in (DFlash2's, DSpine's): their kernels write and read binary16, one
  // layer or more.
  std::vector<bool> dflash_kv(size_t(std::max(d.n_states, 0)), false);
  for (int b = 0; b < d.n_blocks; ++b) {
    const flyx_block& blk = d.blocks[b];
    // EAGLE-4's too: its draft steps attend over a window (attention's per-head kernel reads binary16)
    const int kv = blk.kind == FLYX_BLOCK_DFLASH2  ? blk.dflash2.kv
                   : blk.kind == FLYX_BLOCK_DSPINE ? blk.dspine.kv
                   : blk.kind == FLYX_BLOCK_EAGLE4 ? blk.eagle4.kv
                                                   : -1;
    if (kv >= 0 && kv < d.n_states) dflash_kv[size_t(kv)] = true;
  }
  for (int i = 0; i < d.n_states; ++i) {
    const flyx_state& s = d.states[i];
    sm.kv.push_back(-1);
    sm.index.push_back(-1);
    sm.gdn.push_back(-1);
    sm.sconv.push_back(-1);
    sm.ring.push_back(false);
    sm.canvas.push_back(false);
    switch (s.kind) {
      case FLYX_STATE_KV:
        sm.kv.back() = int(m->states.kv.size());
        // FP8 for the target's caches and the MTP layer's beside them (only drafting reads it, so its
        // rounding moves how many drafts are accepted, never the output); a stacked drafter's, and a block
        // drafter's of any number of layers, stay binary16.
        m->states.kv.push_back({s.layers, s.n_kv, s.head_dim,
                                m->kv_fp8 && s.layers == 1 && !(s.draft && m->opt.draft_kv16) && !dflash_kv[size_t(i)],
                                s.draft != 0, s.window});
        break;
      case FLYX_STATE_INDEXER:
        expect(s.head_dim > 0 && s.head_dim <= 256, "indexer key width");
        sm.index.back() = int(m->states.index_width.size());
        m->states.index_width.push_back(s.head_dim + 3);
        break;
      case FLYX_STATE_GDN:
        sm.gdn.back() = int(m->states.gdn.size());
        m->states.gdn.push_back({s.nk, s.nv, s.dk, s.dv, s.conv, s.head_order == FLYX_GDN_HEADS_TILED});
        break;
      case FLYX_STATE_PLE:
        expect(s.conv >= 2 && s.conv <= 64 && s.width < s.conv && s.channels < INT_MAX, "PLE history dimensions");
        sm.sconv.back() = int(m->states.sconv.size());
        m->states.sconv.push_back({s.conv, s.channels + 1});
        break;
      case FLYX_STATE_SHORTCONV:
        sm.sconv.back() = int(m->states.sconv.size());
        expect(s.conv >= 2 && s.channels > 0, "ShortConv state: invalid conv or channels");
        m->states.sconv.push_back({s.conv, s.channels});
        break;
      case FLYX_STATE_RING:
        expect(m->states.ring_width == 0, "more than one hidden ring");
        m->states.ring_width = s.width;
        sm.ring.back() = true;
        break;
      case FLYX_STATE_CANVAS:  // DV's: a sequence keeps one once DV's rounds run here (docs/design/dv.md)
        expect(s.slots >= 2 && s.width == d.hidden && s.topk == 16 && s.rank >= 1, "a DV canvas of these sizes");
        sm.canvas.back() = true;
        break;
      default: throw std::runtime_error("state kind " + std::to_string(s.kind) + ": not implemented by this backend");
    }
  }
  return sm;
}

int state_of(const std::vector<int>& map, int i, const char* kind) {
  expect(i >= 0 && i < int(map.size()) && map[size_t(i)] >= 0, std::string("block names no ") + kind + " state");
  return map[size_t(i)];
}

// One part of an adapter piece: weight w uploaded as bf16, or (w < 0) n zeros, for a linear of an adapted block that
// has no adapter of its own: the block's stacked GEMMs then run it as an adapter whose products are 0.  rows: the
// stored row each uploaded row is (an adapter whose rows are in another layout than its linear's weight); cols: the
// same of its columns (out's a beside a GDN whose value heads the adapter orders otherwise).
struct AdapterPart {
  int w = -1;
  size_t n = 0;  // elements
  WeightLoader::RowOrder rows{}, cols{};
  size_t width = 0;  // a b's columns as uploaded, past its stored rank's (zeros): 0, as stored
};
using AdapterPiece = std::vector<AdapterPart>;

AdapterPiece weights_piece(const WeightLoader& L, const std::vector<int>& ws) {
  AdapterPiece p;
  for (int w : ws) p.push_back({w, WeightLoader::numel(L.t(w)), {}, {}});
  return p;
}

// A DV layer's adapters as uploaded, in order: the a's that read one input stacked (q|k|v, gate|up), then each b.
std::vector<AdapterPiece> dv_adapters(const WeightLoader& L, const flyx_dv_layer_block& b) {
  const flyx_lora* a = b.lora;
  std::vector<AdapterPiece> pieces;
  for (const std::vector<int>& ws : std::vector<std::vector<int>>{
           {a[FLYX_LIN_Q].a, a[FLYX_LIN_K].a, a[FLYX_LIN_V].a},
           {a[FLYX_LIN_O].a},
           {a[FLYX_LIN_GATE].a, a[FLYX_LIN_UP].a},
           {a[FLYX_LIN_DOWN].a},
           {a[FLYX_LIN_Q].b}, {a[FLYX_LIN_K].b}, {a[FLYX_LIN_V].b}, {a[FLYX_LIN_O].b},
           {a[FLYX_LIN_GATE].b}, {a[FLYX_LIN_UP].b}, {a[FLYX_LIN_DOWN].b}})
    pieces.push_back(weights_piece(L, ws));
  return pieces;
}

// A block's adapted rank: its linears' with an adapter, one rank and scale (0: none on any), the scale to *scale.
int block_rank(const flyx_lora* l, int n, float* scale_out = nullptr) {
  int rank = 0;
  float scale = 0.f;
  for (int i = 0; i < n; ++i) {
    expect(l[i].rank >= 0, "an adapter of a negative rank");
    if (l[i].rank == 0) continue;  // none on this linear (a block built zeroed has none on any)
    if (!rank) rank = l[i].rank, scale = l[i].scale;
    expect(l[i].rank == rank && l[i].scale == scale, "a block's adapters of ranks or scales apart");
  }
  if (scale_out && rank) *scale_out = scale;
  return rank;
}

// The inputs and outputs of an attention block's q, k, v, o and of an MLP block's gate, up, down.
void attn_dims(const flyx_attn_block& b, int in[4], int out[4]) {
  const int q = b.n_heads * b.head_dim, kv = b.n_kv * b.head_dim, qg = b.gate != FLYX_GATE_NONE ? 2 * q : q;
  const int i[4] = {b.hidden, b.hidden, b.hidden, q}, o[4] = {qg, kv, kv, b.hidden};
  std::copy(i, i + 4, in);
  std::copy(o, o + 4, out);
}
// A gated delta net's qkv, z, b, a and out.
void gdn_dims(const flyx_gdn_block& b, int in[5], int out[5]) {
  const int v = b.nv * b.dv, C = 2 * b.nk * b.dk + v;
  const int i[5] = {b.hidden, b.hidden, b.hidden, b.hidden, v}, o[5] = {C, v, b.nv, b.nv, b.hidden};
  std::copy(i, i + 5, in);
  std::copy(o, o + 5, out);
}
void mlp_dims(const flyx_mlp_block& b, int in[3], int out[3]) {
  const int i[3] = {b.hidden, b.hidden, b.inter}, o[3] = {b.inter, b.inter, b.hidden};
  std::copy(i, i + 3, in);
  std::copy(o, o + 3, out);
}

// Where an adapter's rows are not its linear's weight's (backend_abi.h lora_gate, lora_head_order): per linear, the
// stored row of each of its b's uploaded rows (empty: the same), and of each of the last linear's a's columns.
struct AdapterOrder {
  std::vector<WeightLoader::RowOrder> b_rows;
  WeightLoader::RowOrder out_cols;
};

// An adapter's rank as uploaded: its stored rank's up to a multiple of 8 (the second GEMM's K: gemm() takes those),
// the a's rows past it and the b's columns zeros, so that B(Ax) is the same (r 1, 3 and 4 are on the Hub).
int padded_rank(int rank) { return (rank + 7) / 8 * 8; }

// A target block's own adapters as uploaded (ABI 43; n linears, the last one's input its own): the a's of the n - 1
// linears that read the block's input stacked, the last one's a, then each b, of the padded rank; a linear without an
// adapter in an adapted block as zeros of its shape (in[i] inputs, out[i] outputs).  None on any linear: nothing.
std::vector<AdapterPiece> block_adapters(const WeightLoader& L, const flyx_lora* l, int n, const int* in,
                                         const int* out, const AdapterOrder& order = {}) {
  const int stored = block_rank(l, n);
  if (!stored) return {};
  const size_t rank = size_t(padded_rank(stored)), more = rank - size_t(stored);
  auto part = [&](int i, bool a) {
    AdapterPiece p;
    const int w = a ? l[i].a : l[i].b;
    if (w < 0) return AdapterPiece{AdapterPart{-1, rank * size_t(a ? in[i] : out[i])}};
    if (a) {
      p.push_back(AdapterPart{w, WeightLoader::numel(L.t(w))});
      if (i == n - 1) p[0].cols = order.out_cols;
      if (more) p.push_back(AdapterPart{-1, more * size_t(in[i])});
      return p;
    }
    p.push_back(AdapterPart{w, rank * size_t(out[i])});
    if (size_t(i) < order.b_rows.size()) p[0].rows = order.b_rows[size_t(i)];
    if (more) p[0].width = rank;
    return p;
  };
  std::vector<AdapterPiece> pieces(2);
  for (int i = 0; i + 1 < n; ++i)
    for (const AdapterPart& q : part(i, true)) pieces[0].push_back(q);
  pieces[1] = part(n - 1, true);
  for (int i = 0; i < n; ++i) pieces.push_back(part(i, false));
  return pieces;
}

// Every adapter in one allocation, carved in order (each piece on 256 bytes, as cudaMalloc places them): past 1 MiB,
// Windows' driver gives an allocation whole 2 MiB, and DV's adapters of rank 256 are 1.1 to 2.3 MiB each (MiniCPM5-1B:
// one allocation each took 190 MiB more than their 0.37 GB, which memory() never counted); a V-LoRA's of rank 64 are
// 264 pieces of 32 KiB to 0.6 MiB.
struct AdapterCarve {
  bf16* next = nullptr;
  size_t left = 0;  // elements
  static size_t piece(const AdapterPiece& parts) {
    size_t n = 0;
    for (const AdapterPart& p : parts) n += p.n;
    return (n + 127) / 128 * 128;
  }
  bf16* take(WeightLoader& L, const AdapterPiece& parts) {
    const size_t n = piece(parts);
    expect(n <= left, "adapters: more than their allocation holds");
    bf16* p = next;
    size_t off = 0;
    for (const AdapterPart& part : parts) {
      if (part.width) {  // a b [out, rank] widened with zero columns on the host (its rows gathered, if ordered)
        const std::vector<float> f = L.host_f32(part.w);
        const size_t R = size_t(L.t(part.w)->shape[1]), O = f.size() / R;
        std::vector<uint16_t> h(O * part.width, 0);
        for (size_t o = 0; o < O; ++o) {
          const size_t from = part.rows ? part.rows(o) : o;
          for (size_t c = 0; c < R; ++c) h[o * part.width + c] = float_to_bf16(f[from * R + c]);
        }
        L.upload_staged(p + off, h.data(), h.size() * sizeof(uint16_t));
      } else if (part.cols) {  // a small [rank, K] matrix: its columns gathered on the host
        const std::vector<float> f = L.host_f32(part.w);
        const size_t K = size_t(L.t(part.w)->shape[1]), R = f.size() / K;
        std::vector<uint16_t> h(f.size());
        for (size_t r = 0; r < R; ++r)
          for (size_t c = 0; c < K; ++c) h[r * K + c] = float_to_bf16(f[r * K + part.cols(c)]);
        L.upload_staged(p + off, h.data(), h.size() * sizeof(uint16_t));
      } else if (part.w >= 0) {
        L.put_bf16(p + off, part.w, part.rows);
      } else {
        const std::vector<bf16> zeros(part.n);  // bf16's 0 is all zero bits
        L.upload_staged(p + off, zeros.data(), part.n * sizeof(bf16));
      }
      off += part.n;
    }
    next += n;
    left -= n;
    return p;
  }
};

// A target block's own adapters (ABI 43): one rank and scale on the linears that have one, zeros on the others; each
// of its linear's shape (in[i] inputs, out[i] outputs), stored bf16, f16 or f32 (uploaded as bf16); carved from
// `carve`.
Adapters block_lora(WeightLoader& L, const flyx_lora* l, int n, const int* in, const int* out, AdapterCarve& carve,
                    const AdapterOrder& order = {}) {
  Adapters a;
  const int stored = block_rank(l, n, &a.scale);
  if (!stored) return a;
  a.rank = padded_rank(stored);
  expect(std::isfinite(a.scale), "an adapter's scale is not finite");
  for (int i = 0; i < n; ++i) {
    if (l[i].rank <= 0) continue;
    const flyx_tensor_view *ta = L.t(l[i].a), *tb = L.t(l[i].b);
    auto floats = [](const flyx_tensor_view* t) {
      return t->dtype == FLYX_DT_BF16 || t->dtype == FLYX_DT_F16 || t->dtype == FLYX_DT_F32;
    };
    expect(floats(ta) && floats(tb) && ta->ndim == 2 && tb->ndim == 2 && ta->shape[0] == stored &&
               ta->shape[1] == in[i] && tb->shape[0] == out[i] && tb->shape[1] == stored,
           std::string(ta->name) + ": an adapter that is not bf16, f16 or f32 [rank, in] and [out, rank] of its linear");
  }
  const auto pieces = block_adapters(L, l, n, in, out, order);
  a.a_in = carve.take(L, pieces[0]);
  a.a_out = carve.take(L, pieces[1]);
  for (int i = 0; i < n; ++i) a.b[i] = carve.take(L, pieces[size_t(2 + i)]);
  return a;
}

AttnBlock attention(WeightLoader& L, const flyx_attn_block& b, const StateMap& sm, AdapterCarve& carve,
                    DraftQuant dq = DraftQuant::kNone) {
  AttnBlock a;
  a.hidden = b.hidden;
  a.nh = b.n_heads;
  a.nkv = b.n_kv;
  a.hd = b.head_dim;
  a.rot = b.rot_dim;
  a.theta = b.theta;
  a.mrope_section[0] = b.mrope_section[0];
  a.mrope_section[1] = b.mrope_section[1];
  a.mrope_section[2] = b.mrope_section[2];
  a.mrope_interleaved = b.mrope_interleaved;
  rope_register(a.theta, a.rot);  // attn_prep's rotary factors, once
  a.gate = b.gate != FLYX_GATE_NONE;
  a.body_only = b.body_only != 0;
  a.norm = a.body_only ? Norm{} : norm(L, b.norm);
  // The kernels read every head's q, then every head's gate: a file that keeps
  // a head's gate rows after its q rows is uploaded in that order (and so is an adapter's q, in its own layout).
  auto gate_rows = [&](int layout) {
    WeightLoader::RowOrder rows;
    if (layout == FLYX_GATE_PER_HEAD)
      rows = [q = size_t(b.n_heads) * b.head_dim, hd = size_t(b.head_dim)](size_t r) {
        return (r % q / hd * 2 + r / q) * hd + r % hd;
      };
    return rows;
  };
  const WeightLoader::RowOrder q_rows = gate_rows(b.gate);
  int rows = 0;
  a.w_in = L.lin_stack_quant({b.q, b.k, b.v}, dq, &rows, {q_rows});
  expect(rows == a.n_in(), "attention q|k|v rows do not match its dimensions");
  a.w_out = L.lin_quant(b.o, dq);
  a.q_norm = head_norm(L, b.q_norm);
  a.k_norm = head_norm(L, b.k_norm);
  a.kv = state_of(sm.kv, b.kv, "K/V");
  if (b.has_indexer) {
    expect(b.index_kv_heads == 1 && b.index_heads > 0 && b.group_size > 0 && b.token_budget >= b.group_size &&
               b.token_budget % b.group_size == 0 && b.index_dim >= a.rot,
           "sparse attention dimensions");
    a.index_state = state_of(sm.index, b.index_state, "indexer");
    a.index_dim = b.index_dim;
    a.index_heads = b.index_heads;
    a.group = b.group_size;
    a.budget = b.token_budget;
    a.index_q_norm = norm(L, b.index_q_norm);
    a.index_k_norm = norm(L, b.index_k_norm);
    a.index_in = L.lin_stack_quant({b.index_q, b.index_k}, dq, &rows);
  }
  int in[4], out[4];
  attn_dims(b, in, out);
  expect(!a.gate || b.lora[FLYX_LIN_Q].rank <= 0 || b.lora_gate == FLYX_GATE_AFTER_Q ||
             b.lora_gate == FLYX_GATE_PER_HEAD,
         "an adapter on a gated attention's q whose gate rows' layout the block does not say");
  AdapterOrder order;
  if (a.gate) order.b_rows = {gate_rows(b.lora_gate)};
  a.lora = block_lora(L, b.lora, 4, in, out, carve, order);
  // An adapter's GEMM reads its input as the linear's bf16 or NVFP4 GEMM does (three bf16 parts, or prefill's bf16
  // rows); beside GGML block weights, whose kernels stage theirs in 8 bits or binary16 pairs, the generic producers'
  // float32 rows (regions/layers.cpp attn_layer).  Its q adapter covers the gate's rows too, uploaded as the weight's.
  return a;
}

// A GDN adapter's rows (and out's a's columns) where its value heads are in another order than the weights'
// (lora_head_order, head_order): per value head of the weights' order, the adapter's.
AdapterOrder gdn_adapter_order(const flyx_gdn_block& b) {
  AdapterOrder o;
  if (b.lora_head_order == b.head_order) return o;
  const size_t nk = size_t(b.nk), rep = size_t(b.nv / b.nk), dv = size_t(b.dv), keys = 2 * nk * size_t(b.dk);
  const bool tiled = b.head_order == FLYX_GDN_HEADS_TILED;
  auto head = [=](size_t v) {  // the weights' value head v -> the adapter's: key head k's i-th value head
    const size_t k = tiled ? v % nk : v / rep, i = tiled ? v / nk : v % rep;
    return tiled ? k * rep + i : i * nk + k;
  };
  auto rows = [=](size_t r) { return head(r / dv) * dv + r % dv; };  // a value head's dv rows
  o.b_rows = {[=](size_t r) { return r < keys ? r : keys + rows(r - keys); }, rows, head, head};
  o.out_cols = rows;
  return o;
}

GdnBlock gdn(WeightLoader& L, const flyx_gdn_block& b, const StateMap& sm, AdapterCarve& carve) {
  GdnBlock g;
  g.hidden = b.hidden;
  g.gd = {b.nk, b.nv, b.dk, b.dv, b.conv, b.head_order == FLYX_GDN_HEADS_TILED};
  g.gd.sigmoid_output = b.out_gate_act != 0;
  g.body_only = b.body_only != 0;
  g.norm = g.body_only ? Norm{} : norm(L, b.norm);
  int rows = 0;
  g.w_in = L.lin_stack({b.qkv, b.z, b.b, b.a}, &rows);
  expect(rows == g.n_in(), "GDN qkv|z|b|a rows do not match its dimensions");
  g.w_out = L.lin(b.out);
  g.conv_w = L.vec(b.conv_w);
  if (b.decay == FLYX_GDN_DECAY_NEG_EXP) {  // the kernels take A_log: the file has -exp(A_log)
    std::vector<float> a = L.host_f32(b.A_log);
    for (float& v : a) {
      expect(v < 0.f && std::isfinite(v), "GDN decay -exp(A_log) must be negative");
      v = float(std::log(-double(v)));
    }
    g.A_log = L.put(a);
  } else {
    g.A_log = L.vec(b.A_log);
  }
  g.dt_bias = L.vec(b.dt_bias);
  g.gated = norm(L, b.out_norm);
  g.state = state_of(sm.gdn, b.state, "GDN");
  const GdnDims& sd = L.model()->states.gdn[size_t(g.state)];
  expect(sd.C() == g.gd.C() && sd.dk == g.gd.dk && sd.tiled == g.gd.tiled, "GDN block and its state differ in shape");
  expect(b.lora_head_order == FLYX_GDN_HEADS_GROUPED || b.lora_head_order == FLYX_GDN_HEADS_TILED,
         "GDN adapters in an unknown head order");
  int in[5], out[5];
  gdn_dims(b, in, out);
  g.lora = block_lora(L, b.lora, 5, in, out, carve, gdn_adapter_order(b));
  return g;
}

MlpBlock mlp(WeightLoader& L, const flyx_mlp_block& b, AdapterCarve& carve, DraftQuant dq = DraftQuant::kNone) {
  MlpBlock f;
  f.hidden = b.hidden;
  f.inter = b.inter;
  f.norm = norm(L, b.norm);
  int rows = 0;
  f.w_gu = L.lin_stack_quant({b.gate, b.up}, dq, &rows);
  expect(rows == 2 * f.inter, "MLP gate|up rows do not match its dimensions");
  f.w_down = L.lin_quant(b.down, dq);
  int in[3], out[3];
  mlp_dims(b, in, out);
  f.lora = block_lora(L, b.lora, 3, in, out, carve);
  return f;
}

DFlashLayer dflash_layer(WeightLoader& L, const flyx_dflash2_layer_block& b, DraftQuant dq) {
  DFlashLayer l;
  l.norm = norm(L, b.norm);
  l.post_norm = norm(L, b.post_norm);
  l.q_norm = norm(L, b.q_norm);
  l.k_norm = norm(L, b.k_norm);
  l.w_qkv = L.lin_stack_quant({b.q, b.k, b.v}, dq);
  l.w_o = L.lin_quant(b.o, dq);
  l.w_gu = L.lin_stack_quant({b.gate, b.up}, dq);
  l.w_down = L.lin_quant(b.down, dq);
  if (b.attn_conv_proj >= 0 || b.mlp_conv_proj >= 0 || b.attn_conv_base >= 0 || b.mlp_conv_base >= 0) {
    expect(b.attn_conv_proj >= 0 && b.mlp_conv_proj >= 0 && b.attn_conv_base >= 0 && b.mlp_conv_base >= 0,
           "a DFlash2 layer with some of its convolutions");
    l.attn_proj = L.lin_quant(b.attn_conv_proj, dq);
    l.mlp_proj = L.lin_quant(b.mlp_conv_proj, dq);
    l.attn_base = L.vec(b.attn_conv_base);
    l.mlp_base = L.vec(b.mlp_conv_base);
  }
  return l;
}

// EAGLE-4's layer as Eagle4Block runs it: its q|k|v stacked over the 2 hidden of a step's input (no norm of the
// attention's own, no q/k norms, rotary on every dim), gate|up, its norms, fc, its own head (bf16: the candidates are
// scored on its rows) and the candidate head's two factors (bf16: the rough scores' order is all they are for).
Eagle4Block eagle4(WeightLoader& L, const flyx_model_desc& d, const flyx_eagle4_block& b, const StateMap& sm,
                   DraftQuant dq) {
  Eagle4Block e;
  expect(b.n_taps >= 1 && b.n_taps <= FLYX_EAGLE4_MAX_TAPS && b.target_hidden == d.hidden && b.hidden == d.hidden &&
             b.window >= 0 && b.vocab > 0 && b.vocab <= d.vocab && b.cand_rank > 0 && b.n_cands > 0 &&
             b.n_cands <= b.vocab,
         "EAGLE-4 dimensions");
  e.hidden = b.hidden;
  e.n_taps = b.n_taps;
  e.target_hidden = b.target_hidden;
  e.window = b.window;
  e.vocab = b.vocab;
  e.cand_rank = b.cand_rank;
  e.n_cands = b.n_cands;
  e.embed = b.embed;
  AttnBlock& a = e.attn;
  a.hidden = b.hidden;
  a.nh = b.n_heads;
  a.nkv = b.n_kv;
  a.hd = b.head_dim;
  a.rot = b.head_dim;
  a.theta = b.theta;
  rope_register(a.theta, a.rot);
  int rows = 0;
  a.w_in = L.lin_stack_quant({b.q, b.k, b.v}, dq, &rows);
  expect(rows == a.n_in(), "EAGLE-4 q|k|v rows do not match its dimensions");
  a.w_out = L.lin_quant(b.o, dq);
  a.kv = state_of(sm.kv, b.kv, "K/V");
  MlpBlock& f = e.mlp;
  f.hidden = b.hidden;
  f.inter = b.inter;
  f.norm = norm(L, b.post_norm);
  f.w_gu = L.lin_stack_quant({b.gate, b.up}, dq, &rows);
  expect(rows == 2 * f.inter, "EAGLE-4 gate|up rows do not match its dimensions");
  f.w_down = L.lin_quant(b.down, dq);
  for (int i = 0; i < b.n_taps; ++i) e.fc_norm[i] = norm(L, b.fc_norm[i]);
  e.in_norm = norm(L, b.in_norm);
  e.hidden_norm = norm(L, b.hidden_norm);
  e.norm = norm(L, b.norm);
  e.fc = L.lin_quant(b.fc, dq);
  e.cand_down = L.lin_quant(b.cand_down, DraftQuant::kNone);
  e.cand_up = L.lin_quant(b.cand_up, DraftQuant::kNone);
  e.head = L.lin_quant(b.head, DraftQuant::kNone).w;
  expect(e.head && e.cand_down.w && e.cand_up.w, "EAGLE-4's head and candidate head in bf16");
  return e;
}

// What DFlash2's block and DSpine's bind alike (`title` names the drafter in refusals; layer(j): its layer block j,
// checked against the block): its dimensions, rotary, window, taps, embedding, head and layers, fc and norms, and its
// context: every layer's k|v in one GEMM, their k norms side by side, its binary16 cache.
template <class Block, class LayerOf>
DFlashBlock block_common(WeightLoader& L, const flyx_model_desc& d, const Block& b, const StateMap& sm, DraftQuant dq,
                         int layer_kind, const char* title, LayerOf layer) {
  const std::string t = title;
  DFlashBlock f;
  f.hidden = b.hidden;
  f.nh = b.n_heads;
  f.nkv = b.n_kv;
  f.hd = b.head_dim;
  f.inter = b.inter;
  f.theta = b.theta;
  rope_register(f.theta, f.hd);  // its layers' and its context keys' rotary factors, once
  f.block = b.block;
  f.mask_token = b.mask_token;
  expect(b.window >= 0, "a " + t + " window below zero");
  f.window = b.window;
  f.n_taps = b.n_taps;
  f.target_hidden = d.hidden;
  f.embed = b.embed;
  f.head = b.head;
  f.first_layer = b.first_layer;
  f.n_layers = b.layers;
  f.fc = L.lin_quant(b.fc, dq);
  f.hidden_norm = norm(L, b.hidden_norm);
  f.norm = norm(L, b.norm);
  std::vector<int> ctx, kn;
  double eps = 0;
  for (int j = 0; j < b.layers; ++j) {
    expect(b.first_layer >= 0 && b.first_layer + j < d.n_blocks && d.blocks[b.first_layer + j].kind == layer_kind,
           t + " layers are not its layer blocks");
    const auto& l = layer(j);
    ctx.push_back(l.k);
    ctx.push_back(l.v);
    kn.push_back(l.k_norm.w);
    expect(l.k_norm.offset == 0.0 && (j == 0 || l.k_norm.eps == eps), t + " context k norms: plain weights, one eps");
    eps = l.k_norm.eps;
  }
  int rows = 0;
  f.ctx = L.lin_stack_quant(ctx, dq, &rows);
  expect(rows == f.ctx_width(), t + " context k|v rows do not match its dimensions");
  f.k_norms = {L.vecs(kn), 0.f, float(eps)};
  f.kv = state_of(sm.kv, b.kv, "K/V");
  const KvSpec& k = L.model()->states.kv[size_t(f.kv)];
  expect(k.layers == f.layers() && k.nkv == f.nkv && k.hd == f.hd, t + " context cache differs from its layers");
  expect(!k.fp8, t + " context cache is not binary16");  // what draft_ctx_kv and the draft block write and read
  return f;
}

DFlashBlock dflash(WeightLoader& L, const flyx_model_desc& d, const flyx_dflash2_block& b, const StateMap& sm,
                   DraftQuant dq) {
  DFlashBlock f = block_common(L, d, b, sm, dq, FLYX_BLOCK_DFLASH2_LAYER, "DFlash2", [&](int j) -> const auto& {
    const auto& l = d.blocks[b.first_layer + j].dflash2_layer;
    expect((l.attn_conv_proj >= 0) == (b.conv_taps > 0), "DFlash2 layer convolutions differ from its conv_taps");
    return l;
  });
  expect(std::isfinite(b.softcap) && b.softcap >= 0.f, "a DFlash2 softcap out of range");
  expect(b.softcap_mode == FLYX_DFLASH2_SOFTCAP_BEFORE || b.softcap_mode == FLYX_DFLASH2_SOFTCAP_AFTER,
         "unsupported DFlash2 softcap mode");
  f.softcap = b.softcap;
  f.softcap_mode = b.softcap_mode;
  f.conv_taps = b.conv_taps;
  f.conv_group = b.conv_group;
  f.sel_rank = b.sel_rank;
  f.sel_topk = b.sel_topk;
  // ABI 52: without a selector (sel_rank 0) its weights are -1: the walk takes the head's candidates by their logits.
  expect(b.sel_rank > 0 || (b.sel_pred < 0 && b.sel_succ < 0 && b.sel_hid < 0), "selector weights of no selector");
  if (b.sel_rank > 0) {
    f.sel_pred = L.mat(b.sel_pred);
    f.sel_succ = L.mat(b.sel_succ);
  }
  expect(b.draft_from == 0 || b.draft_from == 1, "a block drafter's first drafting row is the anchor's or the next");
  f.draft_from = b.draft_from;
  if (b.sel_hid >= 0) f.sel_hid = L.lin_quant(b.sel_hid, dq);
  else if (b.sel_rank > 0)  // a unit gate
    f.unit_hid = L.put(std::vector<float>(size_t(FLYX_MAX_ROWS) * b.sel_rank, 1.f));
  return f;
}

// A DSpine layer: DFlash2's without the convs, the bridge's read and write (bf16).
DFlashLayer dspine_layer(WeightLoader& L, const flyx_dspine_layer_block& b, DraftQuant dq) {
  DFlashLayer l;
  l.norm = norm(L, b.norm);
  l.post_norm = norm(L, b.post_norm);
  l.q_norm = norm(L, b.q_norm);
  l.k_norm = norm(L, b.k_norm);
  l.w_qkv = L.lin_stack_quant({b.q, b.k, b.v}, dq);
  l.w_o = L.lin_quant(b.o, dq);
  l.w_gu = L.lin_stack_quant({b.gate, b.up}, dq);
  l.w_down = L.lin_quant(b.down, dq);
  l.read = Lin{L.mat(b.read)};
  l.write = Lin{L.mat(b.write)};
  return l;
}

// A [rows, cols] matrix transposed, [cols, rows] bf16 on the device (from bf16 values: no rounding).
bf16* transposed(WeightLoader& L, int w, int rows, int cols) {
  const std::vector<float> f = L.host_f32(w);
  expect(f.size() == size_t(rows) * size_t(cols), "a matrix to transpose of another size");
  std::vector<uint16_t> t(f.size());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) t[size_t(c) * rows + r] = float_to_bf16(f[size_t(r) * cols + c]);  // exact
  bf16* out = L.model()->alloc<bf16>(t.size());
  CK(cudaMemcpy(out, t.data(), t.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  return out;
}

// DSpine: DFlash2's block (its context the same), sel_topk its walk's candidates, and its bridge.
DFlashBlock dspine(WeightLoader& L, const flyx_model_desc& d, const flyx_dspine_block& b, const StateMap& sm,
                   DraftQuant dq) {
  DFlashBlock f = block_common(L, d, b, sm, dq, FLYX_BLOCK_DSPINE_LAYER, "DSpine",
                               [&](int j) -> const auto& { return d.blocks[b.first_layer + j].dspine_layer; });
  f.sel_topk = b.topk;
  // the walk scales the candidates' head rows by the final norm's weight alone (k_dspine_heads)
  expect(f.norm.offset == 0.f, "a DSpine final norm with an offset");
  int rows = 0;
  DSpineBridge& g = f.dspine;
  g.code = b.code;
  g.rank = b.rank;
  g.eps = float(b.bridge_eps);
  g.codebook = L.mat(b.codebook);
  g.packed = Lin{L.stack({b.gate, b.pred, b.prev}, &rows)};
  expect(rows == 3 * b.rank, "DSpine gate | pred | prev rows do not match its rank");
  g.pv = Lin{g.packed.w + size_t(b.rank) * size_t(b.code)};
  g.gate_bias = L.vec(b.gate_bias);
  g.write_t = Lin{transposed(L, d.blocks[b.first_layer + b.layers - 1].dspine_layer.write, b.hidden, b.rank)};
  expect(b.float_weights == FLYX_DRAFT_WEIGHTS_AUTO || b.float_weights == FLYX_DRAFT_WEIGHTS_Q8 ||
             b.float_weights == FLYX_DRAFT_WEIGHTS_NVFP4,
         "a DSpine drafter's weights beside a float target: the backend's own, Q8_0 or NVFP4");
  return f;
}

// A DV canvas layer: the target blocks it runs, its norms, its adapters (one rank, on every linear) uploaded in
// bf16 with the a's that read one input stacked, carved from `carve`.
DvLayer dv_layer(WeightLoader& L, const flyx_model_desc& d, const flyx_dv_layer_block& b, AdapterCarve& carve) {
  DvLayer l;
  expect(b.attn >= 0 && b.attn < d.n_blocks && d.blocks[b.attn].kind == FLYX_BLOCK_ATTN && b.mlp >= 0 &&
             b.mlp < d.n_blocks && d.blocks[b.mlp].kind == FLYX_BLOCK_MLP,
         "a DV layer names no attention and MLP block of the target");
  const flyx_attn_block& at = d.blocks[b.attn].attn;
  const flyx_mlp_block& f = d.blocks[b.mlp].mlp;
  const int q = at.n_heads * at.head_dim, kv = at.n_kv * at.head_dim;
  const int in[FLYX_LINS] = {at.hidden, at.hidden, at.hidden, q, f.hidden, f.hidden, f.inter};
  const int out[FLYX_LINS] = {q, kv, kv, at.hidden, f.inter, f.inter, f.hidden};
  l.attn = b.attn;
  l.mlp = b.mlp;
  l.rank = b.lora[0].rank;
  l.scale = b.lora[0].scale;
  expect(std::isfinite(l.scale), "a DV adapter's scale is not finite");
  for (int i = 0; i < FLYX_LINS; ++i) {
    const flyx_lora& a = b.lora[i];
    expect(a.a >= 0 && a.b >= 0 && a.rank == l.rank && l.rank >= 1 && a.scale == l.scale,
           "a DV layer without an adapter of one rank and scale on every linear");
    const flyx_tensor_view *ta = L.t(a.a), *tb = L.t(a.b);
    expect(ta->ndim == 2 && tb->ndim == 2 && ta->shape[0] == l.rank && ta->shape[1] == in[i] &&
               tb->shape[0] == out[i] && tb->shape[1] == l.rank,
           std::string(ta->name) + ": a DV adapter of another shape than its linear's");
  }
  l.norm = norm(L, b.norm);
  l.post_norm = norm(L, b.post_norm);
  Lin* to[] = {&l.a_qkv, &l.a_o, &l.a_gu, &l.a_down, &l.b_q, &l.b_k, &l.b_v, &l.b_o, &l.b_gate, &l.b_up, &l.b_down};
  const auto pieces = dv_adapters(L, b);
  for (size_t i = 0; i < pieces.size(); ++i) *to[i] = Lin{carve.take(L, pieces[i])};
  return l;
}

// DV: the canvas's final norm as float, its MASK embedding and residual context in bf16 (stored so: the canvas runs
// in the model's dtype), the selector in float32.
// The selector's tables [vocab, sel_rank] are made from these and the embedding when a plan first drafts with them
// (make_dv_tables).
DvBlock dv(WeightLoader& L, const flyx_model_desc& d, const flyx_dv_block& b, const StateMap& sm) {
  DvBlock v;
  expect(b.hidden == d.hidden && b.vocab == d.vocab, "a DV of another hidden size or vocabulary than the target's");
  v.hidden = b.hidden;
  v.vocab = b.vocab;
  v.slots = b.slots;
  v.tree = b.tree;
  v.tree_width = b.tree_width;
  v.beam = b.beam;
  v.cov = b.cov != 0;
  v.pool_ranks = b.pool_ranks;
  v.topk = b.topk;
  v.res_k = b.res_k;
  v.mask_token = b.mask_token;
  v.causal = b.causal != 0;
  v.fill = b.fill;
  v.sel_temp = b.sel_temp;
  v.embed = b.embed;
  v.head = b.head;
  v.first_layer = b.first_layer;
  v.n_layers = b.layers;
  for (int j = 0; j < b.layers; ++j)
    expect(b.first_layer >= 0 && b.first_layer + j < d.n_blocks &&
               d.blocks[b.first_layer + j].kind == FLYX_BLOCK_DV_LAYER,
           "DV's layers are not its layer blocks");
  v.norm = norm(L, b.norm);
  expect(L.t(b.mask_emb)->ndim == 1 && L.t(b.mask_emb)->shape[0] == b.hidden, "DV's MASK embedding is [hidden]");
  v.mask_emb = L.mat(b.mask_emb);
  v.res_in = Lin{L.mat(b.res_in)};
  v.res_out = Lin{L.mat(b.res_out)};
  v.sel_rank = b.sel_rank;
  for (int w : {b.sel_a, b.sel_h, b.sel_b})
    expect(L.t(w)->dtype == FLYX_DT_F32 && L.t(w)->ndim == 2 && L.t(w)->shape[0] == b.sel_rank &&
               L.t(w)->shape[1] == b.hidden,
           std::string(L.t(w)->name) + ": the DV selector's A, H and B are float32 [rank, hidden]");
  v.sel_a = L.vec(b.sel_a);
  v.sel_h = L.vec(b.sel_h);
  v.sel_b = L.vec(b.sel_b);
  const std::vector<float> alpha = L.host_f32(b.sel_alpha);
  expect(alpha.size() == 1, "the DV selector's alpha is one number");
  v.sel_alpha = alpha[0];
  expect(b.canvas >= 0 && b.canvas < int(sm.canvas.size()) && sm.canvas[size_t(b.canvas)], "DV names no canvas state");
  const flyx_state& c = d.states[b.canvas];
  expect(c.slots == b.slots && c.topk == b.topk && c.rank == b.sel_rank, "DV and its canvas differ in shape");
  return v;
}

LoopWeights loop_weights(WeightLoader& L, int down, int gain, int forget, int hidden, int rank) {
  auto shaped = [&](int w, int N, int K) {
    const flyx_tensor_view* v = L.t(w);
    return v->ndim == 2 && v->shape[0] == N && v->shape[1] == K;
  };
  expect(shaped(down, rank, 2 * hidden) && shaped(gain, hidden, rank) && shaped(forget, hidden, rank),
         "a loop gate controller of other shapes than [rank, 2 hidden], [hidden, rank], [hidden, rank]");
  LoopWeights weights;
  weights.down = L.lin(down);
  weights.stacked = L.can_stack({gain, forget});
  if (weights.stacked) {
    weights.gain_forget = L.lin_stack({gain, forget});
  } else {
    weights.gain = L.lin(gain);
    weights.forget = L.lin(forget);
  }
  // The dense stack keeps both [hidden, rank] matrices in row order, gain first.
  const bf16* gf = weights.gain_forget.w;
  weights.down.round = L.round_weight(down, weights.down.w);
  weights.gain.round = L.round_weight(gain, gf ? gf : weights.gain.w);
  weights.forget.round = L.round_weight(forget, gf ? gf + size_t(hidden) * rank : weights.forget.w);
  return weights;
}

RelAttnBlock rel_attn(WeightLoader& L, const flyx_rel_attn_block& b, const StateMap& sm) {
  RelAttnBlock a;
  a.hidden = b.hidden;
  a.nh = b.n_heads;
  a.nkv = b.n_kv;
  a.hd = b.head_dim;
  a.d_rel = b.d_rel;
  a.rel_extent = b.rel_extent;
  a.window = b.window;
  a.conv = b.conv;
  a.log_scale_floor = b.log_scale_floor;
  a.qk_clip = b.qk_clip;
  a.rel_clip = b.rel_clip;
  a.log_scale_alpha = b.log_scale_alpha;
  a.eps = b.eps;
  a.q_norm_in = norm(L, b.q_norm_in);
  a.k_norm_in = norm(L, b.k_norm_in);
  a.v_norm_in = norm(L, b.v_norm_in);
  if (L.can_stack({b.q, b.gate, b.rel})) {
    a.w_qgr = L.lin_stack({b.q, b.gate, b.rel});
    a.stacked_qgr = true;
  } else {
    a.w_q = L.lin(b.q);
    a.w_gate = L.lin(b.gate);
    a.w_rel = L.lin(b.rel);
  }
  a.w_k = L.lin(b.k);
  a.w_v = L.lin(b.v);
  a.w_o = L.lin(b.o);
  a.w_sink = L.lin_f32(b.sink_proj);
  if (b.k_conv_proj >= 0) a.w_k_conv_proj = L.lin_f32(b.k_conv_proj);
  if (b.v_conv_proj >= 0) a.w_v_conv_proj = L.lin_f32(b.v_conv_proj);
  a.rel_bias = L.vec(b.rel_bias);
  a.q_clip_w = L.vec(b.q_clip_w);
  a.rel_clip_w = L.vec(b.rel_clip_w);
  a.k_conv = L.vec(b.k_conv);
  a.v_conv = L.vec(b.v_conv);
  a.post_conv = L.vec(b.post_conv);
  if (b.res_query >= 0) a.res_query = L.vec(b.res_query);
  if (b.loop_msg_query >= 0) a.loop_msg_query = L.vec(b.loop_msg_query);
  a.res_push = b.res_push;
  a.loop_rank = b.loop_rank;
  if (b.loop_rank > 0) {
    for (int j = 0; j < 3; ++j)
      a.loop_ctrl[j] = loop_weights(L, b.loop_ctrl_down[j], b.loop_ctrl_gain[j], b.loop_ctrl_forget[j], b.hidden,
                                   b.loop_rank);
  }
  a.kv = state_of(sm.kv, b.kv, "K/V");
  a.sconv = state_of(sm.sconv, b.sconv, "ShortConv");
  return a;
}

MoeBlock moe(WeightLoader& L, const flyx_moe_block& b, const StateMap& sm) {
  MoeBlock f;
  f.hidden = b.hidden;
  f.inter = b.inter;
  f.n_experts = b.n_experts;
  f.top_k = b.top_k;
  f.conv = b.conv;
  f.gating_sigmoid = b.gating != 0;
  f.norm_weights = b.norm_weights != 0;
  f.route_scale = b.route_scale;
  f.norm_eps = b.norm_eps;
  f.situ_beta = b.situ_beta;
  f.situ_linear_beta = b.situ_linear_beta;
  f.body_only = b.body_only != 0;
  f.norm = f.body_only ? Norm{} : norm(L, b.norm);
  f.router = L.lin_f32(b.router);
  if (b.router_bias >= 0) f.router_bias = L.vec(b.router_bias);
  f.separate_gate_up = b.separate_gate_up != 0;
  if (f.separate_gate_up) {
    expect(f.inter % 2 == 0, "separate gate and up matrices require an even intermediate dimension");
    f.gate_up = L.expert_mat(b.gate);
    f.up = L.expert_mat(b.up);
  } else {
    f.gate_up = L.expert_mat(b.gate_up);
  }
  if (b.has_shared) {
    f.shared_inter = b.shared_inter;
    f.shared_gu = L.lin_stack({b.shared_gate, b.shared_up});
    f.shared_down = L.lin(b.shared_down);
    f.shared_router = L.vec(b.shared_router);
  }
  f.down = L.expert_mat(b.down);
  if (b.post_conv >= 0) f.post_conv = L.vec(b.post_conv);
  if (b.res_query >= 0) f.res_query = L.vec(b.res_query);
  f.loop_rank = b.loop_rank;
  if (b.loop_rank > 0)
    f.loop_ctrl = loop_weights(L, b.loop_ctrl_down, b.loop_ctrl_gain, b.loop_ctrl_forget, b.hidden, b.loop_rank);
  if (b.sconv >= 0) {
    f.sconv = state_of(sm.sconv, b.sconv, "ShortConv");
    f.sconv_offset = b.sconv_offset;
  }
  return f;
}

// A DSpark layer; its linears stream-quantized to `dq` as DFlash2's are (a file in GGML blocks keeps them unless
// nvfp4 is asked for and smaller).  Its block's Markov head and joint projection stay as stored: the head's 1-bit
// copy is made of them (make_draft_weights).
DSparkLayer dspark_layer(WeightLoader& L, const flyx_dspark_layer_block& b, const StateMap& sm, DraftQuant dq) {
  DSparkLayer l;
  l.window = b.window;
  l.kv = state_of(sm.kv, b.kv, "K/V");
  KvSpec& k = L.model()->states.kv[size_t(l.kv)];
  expect(k.layers == 1, "DSpark layer cache must have 1 layer");
  k.fp8 = false;  // DSpark draft kernels (draft_ctx_kv / dspark_block_attention) read binary16 KV
  l.norm = norm(L, b.norm);
  l.post_norm = norm(L, b.post_norm);
  l.q_norm = norm(L, b.q_norm);
  l.k_norm = norm(L, b.k_norm);
  l.k_ctx_norm = norm(L, b.k_ctx_norm);
  l.ctx_norm = norm(L, b.ctx_norm);
  if (L.can_stack({b.q, b.k, b.v})) {
    l.w_qkv = L.lin_stack_quant({b.q, b.k, b.v}, dq);
    l.stacked_qkv = true;
  } else {
    l.w_q = L.lin_quant(b.q, dq);
    l.w_k = L.lin_quant(b.k, dq);
    l.w_v = L.lin_quant(b.v, dq);
  }
  l.w_o = L.lin_quant(b.o, dq);
  if (L.can_stack({b.k_ctx, b.v_ctx})) {
    l.w_ctx_kv = L.lin_stack_quant({b.k_ctx, b.v_ctx}, dq);
    l.stacked_ctx_kv = true;
  } else {
    l.w_k_ctx = L.lin_quant(b.k_ctx, dq);
    l.w_v_ctx = L.lin_quant(b.v_ctx, dq);
  }
  if (L.can_stack({b.gate, b.up})) {
    l.w_gu = L.lin_stack_quant({b.gate, b.up}, dq);
    l.stacked_gu = true;
  } else {
    l.w_gate = L.lin_quant(b.gate, dq);
    l.w_up = L.lin_quant(b.up, dq);
  }
  l.w_down = L.lin_quant(b.down, dq);
  l.ctx_mix = L.vec(b.ctx_mix);
  return l;
}

DSparkBlock dspark(WeightLoader& L, const flyx_model_desc& d, const flyx_dspark_block& b, const StateMap& sm) {
  DSparkBlock f;
  f.hidden = b.hidden;
  f.nh = b.n_heads;
  f.nkv = b.n_kv;
  f.hd = b.head_dim;
  f.inter = b.inter;
  f.block = b.block;
  f.n_taps = b.n_taps;
  f.markov_rank = b.markov_rank;
  f.vocab_out = b.vocab_out;
  f.theta = b.theta;
  f.softcap = b.softcap;
  rope_register(f.theta, f.hd);
  f.embed = b.embed;
  f.head = b.head;
  f.first_layer = b.first_layer;
  f.n_layers = b.layers;
  for (int j = 0; j < b.layers; ++j)
    expect(b.first_layer >= 0 && b.first_layer + j < d.n_blocks &&
               d.blocks[b.first_layer + j].kind == FLYX_BLOCK_DSPARK_LAYER,
           "DSpark layers are not its layer blocks");
  f.tok_norm = norm(L, b.tok_norm);
  f.norm = norm(L, b.norm);
  f.markov_ctx_norm = norm(L, b.markov_ctx_norm);
  f.mask_embd = L.vec(b.mask_embd);
  f.markov_ctx_mix = L.vec(b.markov_ctx_mix);
  if (L.packable(b.markov_w1)) f.markov_w1_blk = L.pack(b.markov_w1);
  else f.markov_w1 = L.mat(b.markov_w1);
  f.markov_w2 = L.lin(b.markov_w2);
  if (b.markov_w2_b >= 0) f.markov_w2_b = L.vec(b.markov_w2_b);
  f.joint_proj = L.lin(b.joint_proj);
  f.joint_proj_b = L.vec(b.joint_proj_b);
  for (int j = 0; j < b.layers; ++j) {
    const int lkv = state_of(sm.kv, d.blocks[b.first_layer + j].dspark_layer.kv, "K/V");
    const KvSpec& k = L.model()->states.kv[size_t(lkv)];
    expect(k.layers == 1 && k.nkv == f.nkv && k.hd == f.hd, "DSpark layer context cache differs from its dims");
  }
  expect(b.ring >= 0 && b.ring < d.n_states && sm.ring[size_t(b.ring)], "DSpark names no hidden ring state");
  return f;
}

VNorm bind_vnorm(WeightLoader& L, const flyx_vnorm& n) {
  expect(!(n.kind == FLYX_NORM_RMS && n.w < 0 && n.b >= 0), "vnorm: RMSNorm with bias requires weight");
  return VNorm{n.kind, n.w >= 0 ? L.vision_vec(n.w) : VisionVec{}, n.b >= 0 ? L.vision_vec(n.b) : VisionVec{},
               float(n.offset), float(n.eps)};
}

VisionStemBlock vision_stem(WeightLoader& L, const flyx_vision_stem_block& b) {
  VisionStemBlock s;
  s.in_dim = b.in_dim;
  s.hidden = b.hidden;
  expect(s.in_dim > 0 && s.hidden > 0, "vision_stem: non-positive dimensions");
  if (b.patch_w2 >= 0) {
    expect(s.in_dim % 2 == 0, "vision_stem: split patch_w2 requires even in_dim");
    const size_t H = size_t(s.hidden), half = size_t(s.in_dim / 2);
    const std::vector<float> w0 = L.host_f32(b.patch_w);
    const std::vector<float> w1 = L.host_f32(b.patch_w2);
    expect(w0.size() == H * half && w1.size() == H * half, "vision_stem: patch_w / patch_w2 size mismatch");
    std::vector<float> cat(H * size_t(s.in_dim));
    for (size_t r = 0; r < H; ++r) {
      std::memcpy(cat.data() + r * size_t(s.in_dim), w0.data() + r * half, half * sizeof(float));
      std::memcpy(cat.data() + r * size_t(s.in_dim) + half, w1.data() + r * half, half * sizeof(float));
    }
    s.patch_w = L.vision_put(cat);
  } else {
    const std::vector<float> w = L.host_f32(b.patch_w);
    expect(w.size() == size_t(s.hidden) * size_t(s.in_dim), "vision_stem: patch_w size mismatch");
    s.patch_w = L.vision_put(w);
  }
  if (b.patch_b >= 0) s.patch_b = L.vision_vec(b.patch_b);
  s.pos_mode = b.pos_mode;
  s.pos_grid_h = b.pos_grid_h;
  s.pos_grid_w = b.pos_grid_w;
  s.merge_size = std::max(1, int(b.merge_size));
  if (b.pos_w >= 0) {
    const size_t n = WeightLoader::numel(L.t(b.pos_w));
    expect(n % size_t(s.hidden) == 0, "vision_stem: pos_w size not divisible by hidden");
    s.pos_rows = int(n / size_t(s.hidden));
    if (s.pos_mode == FLYX_VPOS_BILINEAR_2D) {
      expect(s.pos_grid_h > 0 && s.pos_grid_w > 0 &&
                 size_t(s.pos_grid_h) * size_t(s.pos_grid_w) <= size_t(s.pos_rows),
             "vision_stem: pos_grid_h * pos_grid_w exceeds pos_w rows");
    }
    s.pos_w = L.vision_vec(b.pos_w);
  }
  expect(b.cls_token < 0, "vision_stem: cls_token is not yet implemented by this backend");
  s.pre_norm = bind_vnorm(L, b.pre_norm);
  return s;
}

VitAttnBlock vit_attn(WeightLoader& L, const flyx_vit_attn_block& b) {
  VitAttnBlock a;
  a.hidden = b.hidden;
  a.nh = b.n_heads;
  a.hd = b.head_dim;
  a.rope_mode = b.rope_mode;
  a.theta = b.theta;
  a.merge_size = std::max(1, int(b.merge_size));
  a.norm = bind_vnorm(L, b.norm);
  a.q_norm = bind_vnorm(L, b.q_norm);
  a.k_norm = bind_vnorm(L, b.k_norm);
  const int pw = a.nh * a.hd;
  int rows = 0;
  if (b.qkv >= 0) {
    a.w_qkv = L.vision_lin(b.qkv);
    rows = a.w_qkv.N;
    if (b.qkv_b >= 0) a.qkv_b = L.vision_vec(b.qkv_b);
  } else {
    a.w_qkv = L.vision_lin_stack({b.q, b.k, b.v}, &rows);
    if (b.q_b >= 0 || b.k_b >= 0 || b.v_b >= 0) {
      std::vector<float> qb(size_t(3 * pw), 0.f);
      auto copy_b = [&](int w, int off) {
        if (w < 0) return;
        const std::vector<float> v = L.host_f32(w);
        expect(int(v.size()) == pw, "vit_attn: q/k/v bias size mismatch");
        std::memcpy(qb.data() + off, v.data(), size_t(pw) * sizeof(float));
      };
      copy_b(b.q_b, 0);
      copy_b(b.k_b, pw);
      copy_b(b.v_b, 2 * pw);
      a.qkv_b = L.vision_put(qb);
    }
  }
  expect(rows == 3 * pw, "vit_attn: qkv projection rows do not match 3 * nh * hd");
  a.w_o = L.vision_lin(b.o);
  if (b.o_b >= 0) a.o_b = L.vision_vec(b.o_b);
  if (b.layer_scale >= 0) a.layer_scale = L.vision_vec(b.layer_scale);
  return a;
}

VitMlpBlock vit_mlp(WeightLoader& L, const flyx_vit_mlp_block& b) {
  VitMlpBlock f;
  f.hidden = b.hidden;
  f.inter = b.inter;
  f.act = b.act;
  f.gated = b.gated != 0;
  f.norm = bind_vnorm(L, b.norm);
  f.w_fc1 = L.vision_lin(b.fc1);
  if (b.fc1_b >= 0) f.fc1_b = L.vision_vec(b.fc1_b);
  if (f.gated) {
    f.w_gate = L.vision_lin(b.gate);
    if (b.gate_b >= 0) f.gate_b = L.vision_vec(b.gate_b);
  }
  f.w_fc2 = L.vision_lin(b.fc2);
  if (b.fc2_b >= 0) f.fc2_b = L.vision_vec(b.fc2_b);
  if (b.layer_scale >= 0) f.layer_scale = L.vision_vec(b.layer_scale);
  return f;
}

VisionMergeBlock vision_merge(WeightLoader& L, const flyx_vision_merge_block& b) {
  VisionMergeBlock mg;
  mg.in_hidden = b.in_hidden;
  mg.merge_hidden = b.fc1_out;
  mg.out_hidden = b.out_hidden;
  mg.merge_mode = b.merge_mode;
  mg.merge_size = std::max(1, int(b.merge_size));
  mg.act = b.act;
  mg.post_norm = b.post_norm != 0;
  mg.norm = bind_vnorm(L, b.norm);
  if (b.fc1 >= 0) mg.w_fc1 = L.vision_lin(b.fc1);
  if (b.fc1_b >= 0) mg.fc1_b = L.vision_vec(b.fc1_b);
  if (b.fc2 >= 0) mg.w_fc2 = L.vision_lin(b.fc2);
  if (b.fc2_b >= 0) mg.fc2_b = L.vision_vec(b.fc2_b);
  return mg;
}

template <class B>
void put_block(flyx_bmodel* m, int i, B b) {
  m->blocks[size_t(i)].emplace<B>(std::move(b));
}

// Whether a head reads this table too (a tied head reads all of it every token).
bool head_reads(const flyx_model_desc& d, int table) {
  for (int i = 0; i < d.n_blocks; ++i)
    if (d.blocks[i].kind == FLYX_BLOCK_HEAD && d.blocks[i].head.w == table) return true;
  return false;
}

enum class TargetQuant { kFloat, kNvfp4, kPacked };

// Both preflight and the binder choose AUTO from the same resident target linears. A fixed stack (q|k|v,
// gate|up or the GDN input) falls back to BF16 when only some members can stay in packed GGML blocks.
TargetQuant target_quant(const flyx_model_desc& d, const std::vector<char>& is_draft, bool packed_blocks) {
  TargetQuant result = TargetQuant::kFloat;
  auto format = [&](int w) {
    if (w < 0 || w >= d.n_weights || !d.weights[w].t) return TargetQuant::kFloat;
    const auto& t = *d.weights[w].t;
    if (t.dtype == FLYX_DT_NVFP4 || t.dtype == FLYX_DT_GGML_NVFP4) return TargetQuant::kNvfp4;
    if (packed_blocks && (t.ndim == 2 || t.ndim == 3)) {
      const int64_t k = t.shape[t.ndim - 1];
      if (k > 0 && k <= INT_MAX && blocks_supported(t.dtype, int(k))) return TargetQuant::kPacked;
    }
    return TargetQuant::kFloat;
  };
  auto saw = [&](TargetQuant q) {
    if (q == TargetQuant::kNvfp4 || result == TargetQuant::kFloat) result = q;
  };
  auto linears = [&](std::initializer_list<int> ws) {
    for (int w : ws) saw(format(w));
  };
  auto stack = [&](std::initializer_list<int> ws) {
    const auto q = format(*ws.begin());
    if (std::all_of(ws.begin(), ws.end(), [&](int w) { return format(w) == q; })) saw(q);
  };
  for (int i = 0; i < d.n_blocks; ++i) {
    if (is_draft[size_t(i)]) continue;
    const auto& b = d.blocks[i];
    if (b.kind == FLYX_BLOCK_MLP) {
      stack({b.mlp.gate, b.mlp.up});
      linears({b.mlp.down});
    } else if (b.kind == FLYX_BLOCK_ATTN) {
      stack({b.attn.q, b.attn.k, b.attn.v});
      linears({b.attn.o});
    } else if (b.kind == FLYX_BLOCK_GDN) {
      stack({b.gdn.qkv, b.gdn.z, b.gdn.b, b.gdn.a});
      linears({b.gdn.out});
    } else if (b.kind == FLYX_BLOCK_MOE) {
      const auto& e = b.moe;
      if (e.separate_gate_up)
        linears({e.gate, e.up, e.down, e.router});
      else
        linears({e.gate_up, e.down, e.router});
      if (e.has_shared) linears({e.shared_gate, e.shared_up, e.shared_down});
      if (e.loop_rank > 0) linears({e.loop_ctrl_down, e.loop_ctrl_gain, e.loop_ctrl_forget});
    } else if (b.kind == FLYX_BLOCK_REL_ATTN) {
      // The binder splits q|gate|rel and loop gain|forget when their formats differ.
      const auto& r = b.rel_attn;
      linears({r.q, r.gate, r.rel, r.k, r.v, r.o, r.sink_proj, r.k_conv_proj, r.v_conv_proj});
      if (r.loop_rank > 0)
        for (int j = 0; j < 3; ++j) linears({r.loop_ctrl_down[j], r.loop_ctrl_gain[j], r.loop_ctrl_forget[j]});
    }
  }
  return result;
}

// Each block family chooses its own precision beside a floating target. A packed target sets the precision
// for all block drafters; an explicit draft_weights option overrides AUTO for every drafter, including MTP.
struct DraftQuants {
  DraftQuant mtp, dflash, dspine, dspark, eagle4;
};

DraftQuants resolve_draft_quant(const flyx_model_desc& d, interfaces::DraftWeightsMode mode, TargetQuant tq) {
  using Mode = interfaces::DraftWeightsMode;
  if (mode != Mode::kAuto) {
    const DraftQuant q = mode == Mode::kNvfp4 ? DraftQuant::kNvfp4
                         : mode == Mode::kQ8    ? DraftQuant::kQ8
                                                : DraftQuant::kNone;
    return {q, q, q, q, q};
  }
  const DraftQuant block = tq == TargetQuant::kNvfp4 ? DraftQuant::kNvfp4
                           : tq == TargetQuant::kPacked ? DraftQuant::kQ8
                                                        : DraftQuant::kNone;
  DraftQuants q{DraftQuant::kNvfp4, block, block, block, block};
  if (block == DraftQuant::kNone) {
    q.dspine = q.eagle4 = DraftQuant::kNvfp4;
    for (int i = 0; i < d.n_blocks; ++i) {
      const flyx_block& b = d.blocks[i];
      // ABI 55 descriptors can also request Q8 explicitly beside a float target.
      if (b.kind == FLYX_BLOCK_DSPINE && b.dspine.float_weights == FLYX_DRAFT_WEIGHTS_Q8) q.dspine = DraftQuant::kQ8;
      if (b.kind == FLYX_BLOCK_EAGLE4 && b.eagle4.float_weights == FLYX_DRAFT_WEIGHTS_Q8) q.eagle4 = DraftQuant::kQ8;
      if (b.kind != FLYX_BLOCK_DFLASH2) continue;
      // Ungated block projections (DFlash, DeepSpec's DSpark) use Q8 beside float target weights.
      if (b.dflash2.q8_weights || b.dflash2.sel_rank == 0 || b.dflash2.draft_from == 0)
        q.dflash = DraftQuant::kQ8;
    }
  }
  return q;
}

// The linears a drafter's draft steps read, as bound (an alternative left unbound, a split or stacked form, is skipped
// by held_precision).
std::vector<const Lin*> drafter_lins(const flyx_bmodel* m, int block) {
  std::vector<const Lin*> out;
  if (block < 0) return out;
  if (const auto* t = std::get_if<MtpBlock>(&m->blocks[size_t(block)])) {
    const auto& a = std::get<AttnBlock>(m->blocks[size_t(t->attn)]);
    const auto& f = std::get<MlpBlock>(m->blocks[size_t(t->mlp)]);
    out = {&t->fc, &a.w_in, &a.w_out, &f.w_gu, &f.w_down};
  } else if (const auto* d = std::get_if<DFlashBlock>(&m->blocks[size_t(block)])) {
    out = {&d->fc, &d->ctx, &d->sel_hid};
    for (int j = 0; j < d->layers(); ++j) {
      const auto& l = std::get<DFlashLayer>(m->blocks[size_t(d->first_layer + j)]);
      for (const Lin* x : {&l.w_qkv, &l.w_o, &l.w_gu, &l.w_down, &l.attn_proj, &l.mlp_proj}) out.push_back(x);
    }
  } else if (const auto* e = std::get_if<Eagle4Block>(&m->blocks[size_t(block)])) {
    out = {&e->fc, &e->attn.w_in, &e->attn.w_out, &e->mlp.w_gu, &e->mlp.w_down};
  } else if (const auto* d = std::get_if<DSparkBlock>(&m->blocks[size_t(block)])) {
    for (int j = 0; j < d->layers(); ++j) {
      const auto& l = std::get<DSparkLayer>(m->blocks[size_t(d->first_layer + j)]);
      for (const Lin* x : {&l.w_qkv, &l.w_q, &l.w_k, &l.w_v, &l.w_o, &l.w_ctx_kv, &l.w_k_ctx, &l.w_v_ctx, &l.w_gu,
                           &l.w_gate, &l.w_up, &l.w_down})
        out.push_back(x);
    }
  }
  return out;
}

}  // namespace

const char* held_precision(const std::vector<const Lin*>& lins) {
  int nvfp4 = 0, q8 = 0, exact = 0;
  for (const Lin* l : lins) {
    if (l->q) {
      ++nvfp4;
    } else if (l->packed()) {
      const bool all_q8 = std::all_of(l->blk.begin(), l->blk.end(),
                                      [](const BlockMat& b) { return b.dtype == FLYX_DT_GGML_Q8_0; });
      ++(all_q8 ? q8 : exact);
    } else if (l->w) {
      ++exact;
    }
  }
  if (nvfp4 + q8 + exact == 0) return "";
  return nvfp4 >= q8 && nvfp4 >= exact ? "nvfp4" : q8 >= exact ? "q8" : "exact";
}


void bind(flyx_bmodel* m, const flyx_model_desc& d, LoadTimes* times) {
  m->hidden = d.hidden;
  m->vocab = d.vocab;
  WeightLoader L(m, d);
  const StateMap sm = bind_states(m, d);
  m->blocks.resize(size_t(d.n_blocks));

  // Identify drafter blocks up front so Pass 1 binds all target, vision, and DV blocks first.
  // That establishes target precision and remaining GPU VRAM before Pass 2 stream-quantizes drafter weights.
  std::vector<char> is_mtp(size_t(d.n_blocks), 0), is_dflash(size_t(d.n_blocks), 0), is_dspark(size_t(d.n_blocks), 0);
  auto ref = [&](int owner, int j, int kind, const char* what) {
    expect(j >= 0 && j < d.n_blocks && d.blocks[j].kind == kind,
           "block " + std::to_string(owner) + " names no " + what + " block");
    return j;
  };
  for (int i = 0; i < d.n_blocks; ++i) {
    const flyx_block& b = d.blocks[i];
    if (b.kind == FLYX_BLOCK_MTP) {
      is_mtp[size_t(i)] = 1;
      is_mtp[size_t(ref(i, b.mtp.attn, FLYX_BLOCK_ATTN, "attention"))] = 1;
      is_mtp[size_t(ref(i, b.mtp.mlp, FLYX_BLOCK_MLP, "MLP"))] = 1;
    } else if (b.kind == FLYX_BLOCK_DFLASH2 || b.kind == FLYX_BLOCK_DFLASH2_LAYER || b.kind == FLYX_BLOCK_DSPINE ||
               b.kind == FLYX_BLOCK_DSPINE_LAYER) {
      is_dflash[size_t(i)] = 1;
    } else if (b.kind == FLYX_BLOCK_DSPARK || b.kind == FLYX_BLOCK_DSPARK_LAYER) {
      is_dspark[size_t(i)] = 1;
    } else if (b.kind == FLYX_BLOCK_EAGLE4) {
      is_dflash[size_t(i)] = 1;  // bound beside the block drafters, after the target
    }
  }
  std::vector<char> is_draft(size_t(d.n_blocks), 0);
  for (int i = 0; i < d.n_blocks; ++i)
    is_draft[size_t(i)] = is_mtp[size_t(i)] || is_dflash[size_t(i)] || is_dspark[size_t(i)];

  AdapterCarve carve;
  for (int i = 0; i < d.n_blocks; ++i) {
    const flyx_block& b = d.blocks[i];
    std::vector<AdapterPiece> pieces;
    int in[5], out[5];
    if (b.kind == FLYX_BLOCK_DV_LAYER) pieces = dv_adapters(L, b.dv_layer);
    if (b.kind == FLYX_BLOCK_ATTN) attn_dims(b.attn, in, out), pieces = block_adapters(L, b.attn.lora, 4, in, out);
    if (b.kind == FLYX_BLOCK_MLP) mlp_dims(b.mlp, in, out), pieces = block_adapters(L, b.mlp.lora, 3, in, out);
    if (b.kind == FLYX_BLOCK_GDN) gdn_dims(b.gdn, in, out), pieces = block_adapters(L, b.gdn.lora, 5, in, out);
    for (const auto& p : pieces) carve.left += AdapterCarve::piece(p);
  }
  if (carve.left) carve.next = m->alloc<bf16>(carve.left);

  // Pass 1: target, DV, and vision blocks.
  for (int i = 0; i < d.n_blocks; ++i) {
    if (is_draft[size_t(i)]) continue;
    const flyx_block& b = d.blocks[i];
    switch (b.kind) {
      case FLYX_BLOCK_EMBED: {
        EmbedBlock e;
        e.hidden = b.embed.hidden;
        e.vocab = b.embed.vocab;
        e.has_norm = b.embed.has_norm != 0;
        if (e.has_norm) e.norm = norm(L, b.embed.norm);
        // Every kernel that reads the table gathers the rows of a pass's tokens, a few kilobytes: a table no head
        // reads stays in host memory and frees its size on the GPU, unless the GPU has room (embed=auto, open_model).
        const bool host = m->opt.embed == 1 && !head_reads(d, b.embed.table);
        try {
          if (L.packable(b.embed.table)) e.blk = host ? L.pack_host(b.embed.table) : L.pack(b.embed.table);
          else e.table = host ? L.mat_host(b.embed.table) : L.mat(b.embed.table);
        } catch (const OutOfDeviceMemory&) {
          throw;  // Packed host embeddings can still require tables on the device.
        } catch (const CudaError& err) {
          if (!host) throw;
          throw std::runtime_error(std::string("the embedding in host memory: ") + err.what() +
                                   " (backend option embed=gpu keeps it on the GPU)");
        }
        put_block(m, i, std::move(e));
        break;
      }
      case FLYX_BLOCK_PLE: {
        PleBlock p;
        p.dims = b.ple;
        p.state = state_of(sm.sconv, b.ple.state, "PLE");
        const auto& t = *L.t(b.ple.table);
        p.dtype = t.dtype;
        p.stride = size_t(t.nbytes / t.shape[0]);
        expect(t.dtype == FLYX_DT_F32 || t.dtype == FLYX_DT_BF16 || t.dtype == FLYX_DT_F16 ||
                   t.dtype == FLYX_DT_GGML_Q4_0 || t.dtype == FLYX_DT_GGML_Q5_0 || t.dtype == FLYX_DT_GGML_Q8_0,
               "unsupported packed PLE table dtype");
        auto upload = [&](const uint64_t* src, int count) {
          auto* dst = m->alloc<uint64_t>(count);
          L.upload_staged(dst, src, size_t(count) * sizeof(uint64_t));
          return dst;
        };
        p.dims.multipliers = upload(b.ple.multipliers, b.ple.ngram);
        p.dims.moduli = upload(b.ple.moduli, p.heads());
        p.dims.offsets = upload(b.ple.offsets, p.heads());
        if (m->ple_placement == FLYX_PLE_GPU) {
          auto* data = m->alloc<uint8_t>(t.nbytes);
          L.upload_staged(data, t.data, t.nbytes);
          p.table = data;
        } else {
          const uint64_t limit = host_pinned_budget();
          if (m->ple_placement == FLYX_PLE_HOST && m->host_bytes < limit && t.nbytes <= limit - m->host_bytes) {
            void* host = nullptr;
            if (cudaHostAlloc(&host, t.nbytes, cudaHostAllocMapped) == cudaSuccess) {
              m->host_allocs.push_back(host);
              void* mapped = nullptr;
              CK(cudaHostGetDevicePointer(&mapped, host, 0));
              std::memcpy(host, t.data, t.nbytes);
              p.table = static_cast<uint8_t*>(mapped);
              m->host_bytes += t.nbytes;
            } else
              cudaGetLastError();
          }
          if (!p.table) {
            p.store = std::make_shared<backends::PackedRows>(t, m->ple_placement, m->ple_hot_bytes, 0);
            m->host_bytes += p.store->storage_bytes();
          }
        }
        p.key = L.lin(b.ple.key);
        p.value = L.lin(b.ple.value);
        p.nk = norm(L, b.ple.norm_key);
        p.nq = norm(L, b.ple.norm_query);
        p.nc = norm(L, b.ple.norm_conv);
        p.conv = L.vec(b.ple.conv_w);
        put_block(m, i, std::move(p));
        break;
      }
      case FLYX_BLOCK_HC: {
        HcBlock hc{};
        hc.phase = b.hc.phase;
        hc.hidden = b.hc.hidden;
        hc.streams = b.hc.streams;
        hc.rank = b.hc.rank;
        if (hc.phase == FLYX_HC_MIX || hc.phase == FLYX_HC_COLLAPSE) {
          hc.norm = norm(L, b.hc.norm);
          hc.down = L.lin(b.hc.down);
          hc.up = L.lin(b.hc.up);
          if (hc.phase == FLYX_HC_MIX) hc.inject = L.lin(b.hc.inject);
        }
        put_block(m, i, std::move(hc));
        break;
      }
      case FLYX_BLOCK_HEAD: {
        HeadBlock h;
        h.hidden = b.head.hidden;
        h.vocab = b.head.vocab_out > 0 ? b.head.vocab_out : b.head.vocab;
        h.softcap = b.head.softcap;
        h.norm = head_norm(L, b.head.norm);
        if (L.packable(b.head.w)) {
          h.wb = L.pack(b.head.w);  // a tied head: the embedding's blocks
          h.wb.rows = h.vocab;      // truncate padded output rows to vocab_out
        } else {
          h.w = L.mat(b.head.w);
        }
        put_block(m, i, std::move(h));
        break;
      }
      case FLYX_BLOCK_ATTN: {
        auto a = attention(L, b.attn, sm, carve);
        m->states.kv.at(a.kv).sparse = a.index_state >= 0;
        put_block(m, i, std::move(a));
        break;
      }
      case FLYX_BLOCK_GDN: put_block(m, i, gdn(L, b.gdn, sm, carve)); break;
      case FLYX_BLOCK_MLP: put_block(m, i, mlp(L, b.mlp, carve)); break;
      case FLYX_BLOCK_REL_ATTN: put_block(m, i, rel_attn(L, b.rel_attn, sm)); break;
      case FLYX_BLOCK_MOE: put_block(m, i, moe(L, b.moe, sm)); break;
      case FLYX_BLOCK_DV: {
        const int e = ref(i, b.dv.embed, FLYX_BLOCK_EMBED, "embedding");
        const int head = ref(i, b.dv.head, FLYX_BLOCK_HEAD, "head");
        expect(block<EmbedBlock>(m, e).table && block<HeadBlock>(m, head).w,
               "the DV drafter reads a bf16 embedding and head");
        expect(m->dv_block < 0, "more than one DV drafter");
        expect(m->head_block < 0 || m->head_block == head, "drafters that read different heads");
        m->dv_block = i;
        m->head_block = head;
        put_block(m, i, dv(L, d, b.dv, sm));
        break;
      }
      case FLYX_BLOCK_DV_LAYER: put_block(m, i, dv_layer(L, d, b.dv_layer, carve)); break;
      case FLYX_BLOCK_VISION_STEM: put_block(m, i, vision_stem(L, b.vision_stem)); break;
      case FLYX_BLOCK_VIT_ATTN: put_block(m, i, vit_attn(L, b.vit_attn)); break;
      case FLYX_BLOCK_VIT_MLP: put_block(m, i, vit_mlp(L, b.vit_mlp)); break;
      case FLYX_BLOCK_VISION_MERGE: put_block(m, i, vision_merge(L, b.vision_merge)); break;
      default: throw std::runtime_error("block kind " + std::to_string(b.kind) + ": not implemented by this backend");
    }
  }

  // Decide drafter quantization for Pass 2 from draft_weights policy, target precision, and remaining VRAM.
  const TargetQuant tq = target_quant(d, is_draft, m->opt.packed_blocks);
  const DraftQuants q = resolve_draft_quant(d, m->opt.draft_weights, tq);

  // Pass 2: MTP and block drafters, stream-quantized directly into final VRAM allocations. The
  // embedding and head they read were bound in Pass 1.
  const size_t pre_draft_bytes = m->bytes;
  for (int i = 0; i < d.n_blocks; ++i) {
    if (!is_draft[size_t(i)]) continue;
    const flyx_block& b = d.blocks[i];
    switch (b.kind) {
      case FLYX_BLOCK_ATTN: put_block(m, i, attention(L, b.attn, sm, carve, q.mtp)); break;
      case FLYX_BLOCK_MLP: put_block(m, i, mlp(L, b.mlp, carve, q.mtp)); break;
      case FLYX_BLOCK_MTP: {
        MtpBlock t;
        t.embed = ref(i, b.mtp.embed, FLYX_BLOCK_EMBED, "embedding");
        t.head = ref(i, b.mtp.head, FLYX_BLOCK_HEAD, "head");
        t.attn = ref(i, b.mtp.attn, FLYX_BLOCK_ATTN, "attention");
        t.mlp = ref(i, b.mtp.mlp, FLYX_BLOCK_MLP, "MLP");
        expect(!block_rank(d.blocks[t.attn].attn.lora, 4) && !block_rank(d.blocks[t.mlp].mlp.lora, 3),
               "an MTP layer with adapters (its draft steps run its blocks without them)");
        t.norm_emb = norm(L, b.mtp.norm_emb);
        t.norm_hidden = norm(L, b.mtp.norm_hidden);
        t.norm = norm(L, b.mtp.norm);
        t.fc = L.lin_quant(b.mtp.fc, q.mtp);
        expect(b.mtp.ring >= 0 && b.mtp.ring < d.n_states && sm.ring[size_t(b.mtp.ring)], "MTP block names no ring");
        expect(m->mtp_block < 0, "more than one MTP drafter");
        expect(m->head_block < 0 || m->head_block == t.head, "drafters that read different heads");
        m->mtp_block = i;
        m->head_block = t.head;
        put_block(m, i, std::move(t));
        break;
      }
      case FLYX_BLOCK_DFLASH2:
      case FLYX_BLOCK_DSPINE: {  // the block drafter (m->dflash_block): DFlash2's block, or DSpine's with its bridge
        const bool spine = b.kind == FLYX_BLOCK_DSPINE;
        const int e = ref(i, spine ? b.dspine.embed : b.dflash2.embed, FLYX_BLOCK_EMBED, "embedding");
        const int head = ref(i, spine ? b.dspine.head : b.dflash2.head, FLYX_BLOCK_HEAD, "head");
        const auto& E = block<EmbedBlock>(m, e);
        const auto& Hd = block<HeadBlock>(m, head);
        expect((E.table || E.blk.data) && (Hd.w || Hd.wb.data),
               std::string("the ") + (spine ? "DSpine" : "DFlash2") +
                   " drafter reads an embedding and a head of bf16 or packed GGML blocks");
        expect(m->dflash_block < 0, "more than one block drafter");
        expect(m->head_block < 0 || m->head_block == head, "drafters that read different heads");
        m->dflash_block = i;
        m->head_block = head;
        put_block(m, i, spine ? dspine(L, d, b.dspine, sm, q.dspine) : dflash(L, d, b.dflash2, sm, q.dflash));
        break;
      }
      case FLYX_BLOCK_DFLASH2_LAYER: put_block(m, i, dflash_layer(L, b.dflash2_layer, q.dflash)); break;
      case FLYX_BLOCK_DSPARK: {
        const int e = ref(i, b.dspark.embed, FLYX_BLOCK_EMBED, "embedding");
        const int head = ref(i, b.dspark.head, FLYX_BLOCK_HEAD, "head");
        const auto& E = block<EmbedBlock>(m, e);
        const auto& Hd = block<HeadBlock>(m, head);
        expect((E.table || E.blk.data) && (Hd.w || Hd.wb.data),
               "the DSpark drafter reads an embedding and a head of bf16 or packed GGML blocks");
        expect(m->dspark_block < 0, "more than one DSpark drafter");
        expect(m->head_block < 0 || m->head_block == head, "drafters that read different heads");
        m->dspark_block = i;
        m->head_block = head;
        put_block(m, i, dspark(L, d, b.dspark, sm));
        break;
      }
      case FLYX_BLOCK_DSPARK_LAYER: put_block(m, i, dspark_layer(L, b.dspark_layer, sm, q.dspark)); break;
      case FLYX_BLOCK_DSPINE_LAYER: put_block(m, i, dspine_layer(L, b.dspine_layer, q.dspine)); break;
      case FLYX_BLOCK_EAGLE4: {
        ref(i, b.eagle4.embed, FLYX_BLOCK_EMBED, "embedding");
        expect(m->eagle4_block < 0 && m->dflash_block < 0, "more than one drafter of the target's taps");
        expect(b.eagle4.ring >= 0 && b.eagle4.ring < d.n_states && sm.ring[size_t(b.eagle4.ring)],
               "EAGLE-4 block names no ring");
        m->eagle4_block = i;
        put_block(m, i, eagle4(L, d, b.eagle4, sm, q.eagle4));
        break;
      }
      default: throw std::runtime_error("unexpected draft block kind " + std::to_string(b.kind));
    }
  }
  // EAGLE-4 keeps its state where MTP keeps its own (the sequence's ring, its draft tokens and candidates): never both
  // (models/model.cpp leaves an MTP layer out beside EAGLE-4)
  expect(m->eagle4_block < 0 || m->mtp_block < 0, "an EAGLE-4 drafter beside an MTP layer");
  // The precision the drafts run at, as their linears hold it (not the policy asked: a packed file can stay as it is):
  // a block drafter's when there is one (DFlash2's before DSpark's), else MTP's.
  const std::string mtp_held = held_precision(drafter_lins(m, m->mtp_block));
  const std::string dflash_held = held_precision(drafter_lins(m, m->dflash_block));
  const std::string dspark_held = held_precision(drafter_lins(m, m->dspark_block));
  const std::string eagle4_held = held_precision(drafter_lins(m, m->eagle4_block));
  if (m->eagle4_block >= 0) m->active_draft_weights = eagle4_held;
  else if (m->dflash_block >= 0) m->active_draft_weights = dflash_held;
  else if (m->dspark_block >= 0) m->active_draft_weights = dspark_held;
  else if (m->mtp_block >= 0) m->active_draft_weights = mtp_held;
  else if (m->dv_block >= 0) m->active_draft_weights = "exact";
  if (m->verbose && (m->mtp_block >= 0 || m->dflash_block >= 0 || m->dspark_block >= 0 || m->eagle4_block >= 0)) {
    std::string what;
    auto say = [&](const char* name, const std::string& held) {
      what += (what.empty() ? "" : ", ") + std::string(name) + "=" + held;
    };
    if (m->mtp_block >= 0) say("mtp", mtp_held);
    if (m->dflash_block >= 0) say(block<DFlashBlock>(m, m->dflash_block).name(), dflash_held);
    if (m->dspark_block >= 0) say("dspark", dspark_held);
    if (m->eagle4_block >= 0) say("eagle4", eagle4_held);
    std::fprintf(stderr, "flyx-cuda: drafter weights %s (%.2f GB on GPU)\n", what.c_str(),
                 double(m->bytes - pre_draft_bytes) / 1e9);
  }
  // Assign disjoint per-row tap offsets so multiple tap-consuming drafters (e.g. DSpark + DFlash2) can coexist.
  int64_t tap_off = 0;  // kernel_limits checked the total before binding
  for (Block& blk : m->blocks) {
    if (auto* df = std::get_if<DFlashBlock>(&blk)) {
      df->feats_offset = int(tap_off);
      tap_off += int64_t(df->n_taps) * df->target_hidden;
    } else if (auto* ds = std::get_if<DSparkBlock>(&blk)) {
      ds->feats_offset = int(tap_off);
      tap_off += int64_t(ds->n_taps) * m->hidden;
    } else if (auto* e = std::get_if<Eagle4Block>(&blk)) {
      e->feats_offset = int(tap_off);
      tap_off += int64_t(e->n_taps) * e->target_hidden;
    }
  }
  // FLYX_NODE_RES_MIX names a [hidden] vector by its weight index (Inkling-2's output_res_query), and no block
  // owns it: the vectors of that shape that no block's binding read go on the device for it.  A block's own vectors
  // stay where the block put them (a vision block's in host memory or on disk when the options say so, a drafter's
  // in the layout its kernels read), so a model without such a node holds nothing more for it.
  for (int w = 0; w < d.n_weights; ++w) {
    const flyx_tensor_view* v = d.weights[w].t;
    if (v && v->ndim == 1 && v->shape[0] == d.hidden && WeightLoader::numbers(v) && !L.asked(w))
      m->vec_weights[w] = L.vec(w);
  }
  m->vision_disk.flush_clean();
  L.finish();
  if (times) *times = L.times();
}

int resolve_vision_device(const flyx_model_desc& desc, const BackendOptions& opt, uint64_t free_vram,
                          uint64_t* resident_bytes, std::vector<char>* vision_weights) {
  enum class Linear { none, native_packed, convert };
  std::vector<char> is_vis(size_t(desc.n_weights), 0);
  std::vector<Linear> linear(size_t(desc.n_weights), Linear::none);
  uint64_t padded_bias = 0;
  auto mv = [&](int w) {
    if (w >= 0 && w < desc.n_weights) is_vis[size_t(w)] = 1;
  };
  auto ml = [&](std::initializer_list<int> ws) {
    // vision_lin_stack keeps file blocks only when every member of the stack is packable.
    const bool packed = opt.packed_blocks && std::all_of(ws.begin(), ws.end(), [&](int w) {
      const auto* t = w >= 0 && w < desc.n_weights ? desc.weights[w].t : nullptr;
      return t && t->ndim == 2 && blocks_supported(t->dtype, int(t->shape[1]));
    });
    for (int w : ws) {
      if (w < 0 || w >= desc.n_weights) continue;
      mv(w);
      linear[size_t(w)] = packed ? Linear::native_packed : Linear::convert;
    }
  };
  auto mn = [&](const flyx_vnorm& n) { mv(n.w), mv(n.b); };
  for (int i = 0; i < desc.n_blocks; ++i) {
    const flyx_block& b = desc.blocks[i];
    if (b.kind == FLYX_BLOCK_VISION_STEM) {
      mv(b.vision_stem.patch_w), mv(b.vision_stem.patch_w2), mv(b.vision_stem.patch_b), mv(b.vision_stem.pos_w),
          mv(b.vision_stem.cls_token);
      mn(b.vision_stem.pre_norm);
    } else if (b.kind == FLYX_BLOCK_VIT_ATTN) {
      // Separate Q/K/V biases are uploaded as one vector, with missing lanes zero-filled.
      if (b.vit_attn.qkv < 0) {
        ml({b.vit_attn.q, b.vit_attn.k, b.vit_attn.v});
        const int biases = int(b.vit_attn.q_b >= 0) + int(b.vit_attn.k_b >= 0) + int(b.vit_attn.v_b >= 0);
        if (biases) padded_bias += uint64_t(3 - biases) * b.vit_attn.n_heads * b.vit_attn.head_dim * sizeof(float);
      } else {
        ml({b.vit_attn.qkv});
      }
      mn(b.vit_attn.norm), mn(b.vit_attn.q_norm), mn(b.vit_attn.k_norm);
      ml({b.vit_attn.o});
      mv(b.vit_attn.qkv_b), mv(b.vit_attn.q_b), mv(b.vit_attn.k_b), mv(b.vit_attn.v_b), mv(b.vit_attn.o_b),
          mv(b.vit_attn.layer_scale);
    } else if (b.kind == FLYX_BLOCK_VIT_MLP) {
      mn(b.vit_mlp.norm);
      ml({b.vit_mlp.fc1}), ml({b.vit_mlp.gate}), ml({b.vit_mlp.fc2});
      mv(b.vit_mlp.fc1_b), mv(b.vit_mlp.gate_b), mv(b.vit_mlp.fc2_b), mv(b.vit_mlp.layer_scale);
    } else if (b.kind == FLYX_BLOCK_VISION_MERGE) {
      mn(b.vision_merge.norm);
      ml({b.vision_merge.fc1}), ml({b.vision_merge.fc2});
      mv(b.vision_merge.fc1_b), mv(b.vision_merge.fc2_b);
    }
  }
  uint64_t llm = 0, vis = padded_bias;
  for (int i = 0; i < desc.n_weights; ++i) {
    const flyx_tensor_view* t = desc.weights[i].t;
    if (!t) continue;
    if (!is_vis[size_t(i)]) {
      llm += t->nbytes;
      continue;
    }
    uint64_t nb = WeightLoader::numel(t) * sizeof(float);  // stem, biases and norms are uploaded as float
    if (linear[size_t(i)] != Linear::none && t->ndim == 2) {
      const size_t rows = size_t(t->shape[0]), cols = size_t(t->shape[1]);
      if (linear[size_t(i)] == Linear::native_packed) {
        const auto lo = blocks_layout(t->dtype);
        nb = rows * (cols / lo.values) * (lo.stride + lo.scales * sizeof(uint16_t));
      } else if (t->dtype == FLYX_DT_NVFP4 || t->dtype == FLYX_DT_GGML_NVFP4) {
        nb = rows * (cols / 2 + cols / 16 + sizeof(float));
      } else if (opt.vision_quant == FLYX_VISION_QUANT_Q8) {
        nb = VisionQuantLayout::q8(rows, cols).bytes();
      } else if (opt.vision_quant == FLYX_VISION_QUANT_NVFP4) {
        nb = VisionQuantLayout::nvfp4(rows, cols).bytes();
      } else {
        nb = rows * cols * sizeof(uint16_t);
      }
    }
    vis += nb;
  }
  if (vision_weights) *vision_weights = std::move(is_vis);
  int placement = opt.vision;
  if (placement == FLYX_VISION_DEV_AUTO) {
    const bool roomy = free_vram >= llm && free_vram - llm >= vis &&
                       free_vram - llm - vis >= uint64_t(8) * 1000000000;
    placement = vis && roomy ? FLYX_VISION_DEV_GPU : FLYX_VISION_DEV_DISK;
  }
  if (resident_bytes) *resident_bytes = placement == FLYX_VISION_DEV_GPU ? vis : 0;
  return placement;
}

int resolve_ple(const flyx_model_desc& d, const flyx_open_options* opts, uint64_t free_bytes) {
  if (opts && opts->ple_placement) return opts->ple_placement;
  uint64_t bytes = 0, table = 0;
  for (int i = 0; i < d.n_weights; ++i)
    if (d.weights[i].t) bytes = size_add(bytes, d.weights[i].t->nbytes);
  for (int i = 0; i < d.n_blocks; ++i)
    if (d.blocks[i].kind == FLYX_BLOCK_PLE) table = size_add(table, d.weights[d.blocks[i].ple.table].t->nbytes);
  if (free_bytes >= bytes && free_bytes - bytes >= (uint64_t(2) << 30)) return FLYX_PLE_GPU;
  if (opts && table <= opts->ple_host_room) return FLYX_PLE_HOST;
  return opts && opts->ple_hot_bytes ? FLYX_PLE_HYBRID : FLYX_PLE_DISK;
}

int estimate_load(const flyx_model_desc* desc, const flyx_open_options* opts, uint64_t free_vram,
                  uint64_t total_vram, flyx_load_estimate* out) try {
  if (!desc || !out) return -1;
  *out = {};
  BackendOptions bo{};
  if (opts) {
    bo.vision = opts->vision_device;
    bo.vision_quant = opts->vision_quant;
    if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_NVFP4)
      bo.draft_weights = interfaces::DraftWeightsMode::kNvfp4;
    else if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_Q8)
      bo.draft_weights = interfaces::DraftWeightsMode::kQ8;
    else if (opts->draft_weights == FLYX_DRAFT_WEIGHTS_EXACT)
      bo.draft_weights = interfaces::DraftWeightsMode::kExact;
    else
      bo.draft_weights = interfaces::DraftWeightsMode::kAuto;
  }
  std::string err;
  if (!resolve_options(opts ? opts->options : nullptr, &bo, &err)) return -1;

  const int nw = desc->n_weights;
  std::vector<char> is_vis, is_drafter(size_t(nw), 0), is_dv_w(size_t(nw), 0), is_host_embed(size_t(nw), 0);
  std::vector<std::optional<DraftQuant>> linear_quant(static_cast<size_t>(nw));
  auto mark = [&](std::vector<char>& vec, int w) {
    if (w >= 0 && w < nw) vec[size_t(w)] = 1;
  };
  auto mark_norm = [&](std::vector<char>& vec, const flyx_norm& n) { mark(vec, n.w); };

  // Identify MTP's referenced attn/mlp blocks so they are accounted as drafter weights rather than target weights.
  std::vector<char> is_draft_blk(size_t(desc->n_blocks), 0), is_mtp_blk(size_t(desc->n_blocks), 0);
  for (int i = 0; i < desc->n_blocks; ++i) {
    const flyx_block& b = desc->blocks[i];
    if (b.kind == FLYX_BLOCK_MTP) {
      is_draft_blk[size_t(i)] = is_mtp_blk[size_t(i)] = 1;
      if (b.mtp.attn >= 0 && b.mtp.attn < desc->n_blocks)
        is_draft_blk[size_t(b.mtp.attn)] = is_mtp_blk[size_t(b.mtp.attn)] = 1;
      if (b.mtp.mlp >= 0 && b.mtp.mlp < desc->n_blocks)
        is_draft_blk[size_t(b.mtp.mlp)] = is_mtp_blk[size_t(b.mtp.mlp)] = 1;
    } else if (b.kind == FLYX_BLOCK_DFLASH2 || b.kind == FLYX_BLOCK_DFLASH2_LAYER || b.kind == FLYX_BLOCK_DSPARK ||
               b.kind == FLYX_BLOCK_DSPARK_LAYER || b.kind == FLYX_BLOCK_DSPINE || b.kind == FLYX_BLOCK_DSPINE_LAYER ||
               b.kind == FLYX_BLOCK_EAGLE4) {
      is_draft_blk[size_t(i)] = 1;
    }
  }

  const TargetQuant tq = target_quant(*desc, is_draft_blk, bo.packed_blocks);
  const DraftQuants q = resolve_draft_quant(*desc, bo.draft_weights, tq);
  auto mark_linear = [&](int w, DraftQuant quant) {
    if (w >= 0 && w < nw) {
      is_drafter[size_t(w)] = 1;
      linear_quant[size_t(w)] = quant;
    }
  };

  // Resolve embed=auto (same 32 GB rule as open_model) and vision=auto (same 8 GB rule as open_model).
  uint64_t raw_weights = 0;
  for (int i = 0; i < nw; ++i)
    if (desc->weights[i].t) raw_weights += desc->weights[i].t->nbytes;
  if (bo.embed == 0) {
    const bool roomy = free_vram > raw_weights && free_vram - raw_weights >= uint64_t(32) * 1000000000;
    bo.embed = roomy ? 2 : 1;
  }
  uint64_t vision_gpu_bytes = 0;
  out->vision_device = resolve_vision_device(*desc, bo, free_vram, &vision_gpu_bytes, &is_vis);

  std::vector<char> is_expert(size_t(nw), 0);
  uint64_t dv_extra = 0, block_extra = 0, ple_bytes = 0;
  std::vector<int> dflash_ctx_ws;  // stacked second copy of each DFlash2 or DSpine layer's k|v (f.ctx)
  for (int i = 0; i < desc->n_blocks; ++i) {
    const flyx_block& b = desc->blocks[i];
    if (b.kind == FLYX_BLOCK_EMBED) {
      if (bo.embed == 1 && !head_reads(*desc, b.embed.table)) mark(is_host_embed, b.embed.table);
    } else if (b.kind == FLYX_BLOCK_PLE) {
      mark(is_host_embed, b.ple.table);
      if (resolve_ple(*desc, opts, free_vram) == FLYX_PLE_GPU) ple_bytes += desc->weights[b.ple.table].t->nbytes;
      ple_bytes += sizeof(uint64_t) * (b.ple.ngram + 2ull * (b.ple.ngram - 1) * b.ple.heads_per_order);
    } else if (b.kind == FLYX_BLOCK_MTP) {
      mark_norm(is_drafter, b.mtp.norm_emb), mark_norm(is_drafter, b.mtp.norm_hidden), mark_norm(is_drafter, b.mtp.norm);
      mark_linear(b.mtp.fc, q.mtp);
    } else if (b.kind == FLYX_BLOCK_ATTN) {
      if (is_mtp_blk[size_t(i)]) {
        mark_norm(is_drafter, b.attn.norm), mark_norm(is_drafter, b.attn.q_norm), mark_norm(is_drafter, b.attn.k_norm);
        for (int w : {b.attn.q, b.attn.k, b.attn.v, b.attn.o}) mark_linear(w, q.mtp);
      }
    } else if (b.kind == FLYX_BLOCK_MLP) {
      if (is_mtp_blk[size_t(i)]) {
        mark_norm(is_drafter, b.mlp.norm);
        for (int w : {b.mlp.gate, b.mlp.up, b.mlp.down}) mark_linear(w, q.mtp);
      }
    } else if (b.kind == FLYX_BLOCK_MOE) {
      if (b.moe.separate_gate_up) {
        mark(is_expert, b.moe.gate);
        mark(is_expert, b.moe.up);
      } else
        mark(is_expert, b.moe.gate_up);
      mark(is_expert, b.moe.down);
    } else if (b.kind == FLYX_BLOCK_DSPARK) {
      // Its block's norms, vectors, Markov head and joint projection stay as stored (dspark(): not quantized).
      mark_norm(is_drafter, b.dspark.tok_norm), mark_norm(is_drafter, b.dspark.norm),
          mark_norm(is_drafter, b.dspark.markov_ctx_norm);
      for (int w : {b.dspark.mask_embd, b.dspark.markov_ctx_mix, b.dspark.markov_w1, b.dspark.markov_w2,
                    b.dspark.markov_w2_b, b.dspark.joint_proj, b.dspark.joint_proj_b})
        mark(is_drafter, w);
    } else if (b.kind == FLYX_BLOCK_DSPARK_LAYER) {
      const flyx_dspark_layer_block& l = b.dspark_layer;
      for (const flyx_norm& n : {l.norm, l.post_norm, l.q_norm, l.k_norm, l.k_ctx_norm, l.ctx_norm})
        mark_norm(is_drafter, n);
      mark(is_drafter, l.ctx_mix);
      for (int w : {l.q, l.k, l.v, l.o, l.k_ctx, l.v_ctx, l.gate, l.up, l.down})
        mark_linear(w, q.dspark);
    } else if (b.kind == FLYX_BLOCK_DFLASH2) {
      block_extra += uint64_t(b.dflash2.layers) * b.dflash2.head_dim * sizeof(float);  // stacked context K norms
      mark_norm(is_drafter, b.dflash2.hidden_norm), mark_norm(is_drafter, b.dflash2.norm);
      for (int w : {b.dflash2.fc, b.dflash2.sel_hid}) mark_linear(w, q.dflash);
      // sel_pred / sel_succ are uploaded via L.mat() as BF16 tables (never quantized).
      mark(is_drafter, b.dflash2.sel_pred), mark(is_drafter, b.dflash2.sel_succ);
    } else if (b.kind == FLYX_BLOCK_DFLASH2_LAYER) {
      mark_norm(is_drafter, b.dflash2_layer.norm), mark_norm(is_drafter, b.dflash2_layer.q_norm),
          mark_norm(is_drafter, b.dflash2_layer.k_norm), mark_norm(is_drafter, b.dflash2_layer.post_norm);
      mark(is_drafter, b.dflash2_layer.attn_conv_base), mark(is_drafter, b.dflash2_layer.mlp_conv_base);
      for (int w : {b.dflash2_layer.attn_conv_proj, b.dflash2_layer.q, b.dflash2_layer.k, b.dflash2_layer.v,
                    b.dflash2_layer.o, b.dflash2_layer.mlp_conv_proj, b.dflash2_layer.gate, b.dflash2_layer.up,
                    b.dflash2_layer.down})
        mark_linear(w, q.dflash);
      dflash_ctx_ws.push_back(b.dflash2_layer.k);
      dflash_ctx_ws.push_back(b.dflash2_layer.v);
    } else if (b.kind == FLYX_BLOCK_DSPINE) {
      block_extra += uint64_t(b.dspine.hidden) * b.dspine.rank * sizeof(bf16);  // transposed last write
      block_extra += uint64_t(b.dspine.layers) * b.dspine.head_dim * sizeof(float);  // stacked context K norms
      // Its fc is quantized as the block's; the bridge's tables stay bf16 (dspine()).
      mark_norm(is_drafter, b.dspine.hidden_norm), mark_norm(is_drafter, b.dspine.norm);
      mark_linear(b.dspine.fc, q.dspine);
      for (int w : {b.dspine.codebook, b.dspine.gate, b.dspine.pred, b.dspine.prev, b.dspine.gate_bias})
        mark(is_drafter, w);
    } else if (b.kind == FLYX_BLOCK_DSPINE_LAYER) {
      const flyx_dspine_layer_block& l = b.dspine_layer;
      for (const flyx_norm& n : {l.norm, l.post_norm, l.q_norm, l.k_norm}) mark_norm(is_drafter, n);
      for (int w : {l.q, l.k, l.v, l.o, l.gate, l.up, l.down}) mark_linear(w, q.dspine);
      mark(is_drafter, l.read), mark(is_drafter, l.write);
      dflash_ctx_ws.push_back(l.k);
      dflash_ctx_ws.push_back(l.v);
    } else if (b.kind == FLYX_BLOCK_EAGLE4) {
      // Its layer's linears and fc quantized as asked (eagle4()); its head and candidate head stay bf16.
      const flyx_eagle4_block& e = b.eagle4;
      for (int t = 0; t < e.n_taps && t < FLYX_EAGLE4_MAX_TAPS; ++t) mark_norm(is_drafter, e.fc_norm[t]);
      for (const flyx_norm& n : {e.in_norm, e.hidden_norm, e.post_norm, e.norm}) mark_norm(is_drafter, n);
      for (int w : {e.fc, e.q, e.k, e.v, e.o, e.gate, e.up, e.down}) mark_linear(w, q.eagle4);
      for (int w : {e.head, e.cand_down, e.cand_up}) mark(is_drafter, w);
    } else if (b.kind == FLYX_BLOCK_DV) {
      mark_norm(is_dv_w, b.dv.norm);
      for (int w : {b.dv.mask_emb, b.dv.res_in, b.dv.res_out, b.dv.sel_a, b.dv.sel_h, b.dv.sel_b, b.dv.sel_alpha})
        mark(is_dv_w, w);
      dv_extra += uint64_t(b.dv.vocab) * uint64_t(b.dv.sel_rank) * 2 * sizeof(float);
    } else if (b.kind == FLYX_BLOCK_DV_LAYER) {
      mark_norm(is_dv_w, b.dv_layer.norm), mark_norm(is_dv_w, b.dv_layer.post_norm);
      for (int l = 0; l < FLYX_LINS; ++l) mark(is_dv_w, b.dv_layer.lora[l].a), mark(is_dv_w, b.dv_layer.lora[l].b);
    }
  }

  // Exact device byte formula mirroring WeightLoader::lin_stack / lin_stack_quant / mat / vec.
  auto tensor_dev_bytes = [&](const flyx_tensor_view* v, bool is_lin, DraftQuant q, bool compact = false) -> uint64_t {
    if (!v) return 0;
    uint64_t nel = 1;
    for (int d = 0; d < v->ndim; ++d) nel *= uint64_t(v->shape[d]);
    if (nel == 0) return 0;
    if (v->dtype == FLYX_DT_NVFP4 || v->dtype == FLYX_DT_GGML_NVFP4) {
      if (v->ndim == 2) {
        const uint64_t R = uint64_t(v->shape[0]), K = uint64_t(v->shape[1]);
        return R * K / 2 + R * K / 16 + R * sizeof(float);
      }
      return v->nbytes;
    }
    if (v->ndim >= 3) return v->nbytes;
    if (is_lin && v->ndim == 2) {
      const uint64_t R = uint64_t(v->shape[0]), K = uint64_t(v->shape[1]);
      const uint64_t nvfp4_b = R * K / 2 + R * K / 16 + R * sizeof(float);
      const uint64_t q8_b = R * K + R * (K / 32) * sizeof(uint16_t);
      if (bo.packed_blocks && blocks_supported(v->dtype, int(K))) {
        const BlockLayout lo = blocks_layout(v->dtype);
        const uint64_t per_row = K / uint64_t(lo.values);
        const uint64_t packed_b = R * per_row * (uint64_t(lo.stride) + uint64_t(lo.scales) * sizeof(uint16_t));
        if (compact && K % 64 == 0 && packed_b * 10 >= nvfp4_b * 11) return nvfp4_b;
        return packed_b;
      }
      if (q == DraftQuant::kNvfp4 && K % 64 == 0) return nvfp4_b;
      if (q == DraftQuant::kQ8 && K % 32 == 0) return q8_b;
      return R * K * sizeof(uint16_t);
    }
    if (v->ndim == 1) return nel * sizeof(float);
    if (v->ndim == 2 && bo.packed_blocks && blocks_supported(v->dtype, int(v->shape[1]))) {
      const BlockLayout lo = blocks_layout(v->dtype);
      const uint64_t R = uint64_t(v->shape[0]), K = uint64_t(v->shape[1]);
      const uint64_t per_row = K / uint64_t(lo.values);
      return R * per_row * (uint64_t(lo.stride) + uint64_t(lo.scales) * sizeof(uint16_t));
    }
    return nel * sizeof(uint16_t);
  };

  uint64_t target_bytes = 0, dv_bytes = dv_extra, cur_draft_bytes = 0, nvfp4_draft_bytes = 0;
  const bool compact = bo.draft_weights == interfaces::DraftWeightsMode::kNvfp4;
  for (int i = 0; i < nw; ++i) {
    const flyx_tensor_view* v = desc->weights[i].t;
    if (!v || is_vis[size_t(i)]) continue;
    if (is_expert[size_t(i)]) {
      target_bytes = size_add(target_bytes, expert_layout(desc->weights[i], bo.packed_blocks).bytes());
    } else if (is_drafter[size_t(i)]) {
      const auto quant = linear_quant[size_t(i)];
      cur_draft_bytes += tensor_dev_bytes(v, quant.has_value(), quant.value_or(DraftQuant::kNone), compact);
      nvfp4_draft_bytes += tensor_dev_bytes(v, quant.has_value(), DraftQuant::kNvfp4, true);
    } else if (is_dv_w[size_t(i)]) {
      dv_bytes += tensor_dev_bytes(v, false, DraftQuant::kNone);
    } else if (!is_host_embed[size_t(i)]) {
      target_bytes += tensor_dev_bytes(v, v->ndim == 2, DraftQuant::kNone);
    }
  }
  for (int w : dflash_ctx_ws) {
    if (w < 0 || w >= nw || !desc->weights[w].t) continue;
    cur_draft_bytes += tensor_dev_bytes(desc->weights[w].t, true, *linear_quant[size_t(w)], compact);
    nvfp4_draft_bytes += tensor_dev_bytes(desc->weights[w].t, true, DraftQuant::kNvfp4, true);
  }
  for (int i = 0; i < desc->n_states; ++i) {
    const flyx_state& state = desc->states[i];
    if (state.kind == FLYX_STATE_INDEXER) out->target_kv_bytes_per_position += uint64_t(state.head_dim+3) * sizeof(float);
    if (state.kind != FLYX_STATE_KV || state.draft) continue;
    const bool fp8 = opts && opts->kv_fp8 && state.layers == 1;
    const uint64_t head_bytes = fp8 ? state.head_dim + sizeof(float) : state.head_dim * sizeof(uint16_t);
    out->target_kv_bytes_per_position += 2ull * state.layers * state.n_kv * head_bytes;
  }
  target_bytes = size_add(target_bytes, ple_bytes);
  out->draft_weights_bytes = cur_draft_bytes + dv_bytes + block_extra;
  out->draft_compact_savings = cur_draft_bytes > nvfp4_draft_bytes ? cur_draft_bytes - nvfp4_draft_bytes : 0;
  out->weights_bytes = target_bytes + vision_gpu_bytes + out->draft_weights_bytes;
  // Minimum non-weight device headroom: CUDA context + kernel modules + workspace + minimal KV/graphs.
  out->min_runtime_bytes = std::clamp<uint64_t>(out->weights_bytes / 20, 384ull << 20, 1ull << 30);
  // Extra headroom above hard_need to keep drafter weights at full auto precision (vs compacting to NVFP4).
  out->comfort_margin_bytes =
      total_vram > 0 ? std::clamp<uint64_t>(total_vram / 8, 1ull << 30, 2560ull << 20) : (1536ull << 20);
  return 0;
} catch (const std::exception& error) {
  g_error = error.what();
  return -1;
}

void make_dv_tables(flyx_bmodel* m, int i) {
  DvBlock& v = block<DvBlock>(m, i);
  if (v.sel_ta) return;
  const size_t n = size_t(v.vocab) * size_t(v.sel_rank);
  float* t = m->alloc<float>(2 * n);  // out of GPU memory: lowering fails, and the engine has the backend trim
  dv_selector_tables(v.sel_a, v.sel_b, v.sel_rank, block<EmbedBlock>(m, v.embed).table, v.vocab, v.hidden, t, t + n,
                     nullptr);
  CK(cudaDeviceSynchronize());  // before any session's stream (non-blocking) reads them
  v.sel_ta = t;
  v.sel_tb = t + n;
}

}  // namespace flyx::cuda
