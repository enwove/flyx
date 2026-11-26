// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The binder: the description's tables (backend_abi.h) -> blocks and the
// state table (blocks.h), one block for each of the description's, with the
// weights uploaded as the kernels want them: projections that share an input
// stacked (q|k|v, gate|up, the GDN in-projection, every DFlash2 layer's
// context k|v, DV's adapters on one input), norms as float.  Knows block
// kinds, not models.
#include "backends/cuda/loader/loader.h"

#include <cstring>

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
  std::vector<int> kv, gdn;  // per description state: index in StateTable::kv / ::gdn, else -1
  std::vector<bool> ring, canvas;
};

StateMap bind_states(flyx_bmodel* m, const flyx_model_desc& d) {
  StateMap sm;
  for (int i = 0; i < d.n_states; ++i) {
    const flyx_state& s = d.states[i];
    sm.kv.push_back(-1);
    sm.gdn.push_back(-1);
    sm.ring.push_back(false);
    sm.canvas.push_back(false);
    switch (s.kind) {
      case FLYX_STATE_KV:
        sm.kv.back() = int(m->states.kv.size());
        // FP8 for the target's caches and the MTP layer's beside them (only drafting reads it, so its
        // rounding moves how many drafts are accepted, never the output); a stacked drafter's stays binary16.
        m->states.kv.push_back({s.layers, s.n_kv, s.head_dim,
                                m->kv_fp8 && s.layers == 1 && !(s.draft && m->opt.draft_kv16), s.draft != 0});
        break;
      case FLYX_STATE_GDN:
        sm.gdn.back() = int(m->states.gdn.size());
        m->states.gdn.push_back({s.nk, s.nv, s.dk, s.dv, s.conv, s.head_order == FLYX_GDN_HEADS_TILED});
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

AttnBlock attention(WeightLoader& L, const flyx_attn_block& b, const StateMap& sm, AdapterCarve& carve) {
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
  a.norm = norm(L, b.norm);
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
  a.w_in = L.lin_stack({b.q, b.k, b.v}, &rows, {q_rows});
  expect(rows == a.n_in(), "attention q|k|v rows do not match its dimensions");
  a.w_out = L.lin(b.o);
  a.q_norm = head_norm(L, b.q_norm);
  a.k_norm = head_norm(L, b.k_norm);
  a.kv = state_of(sm.kv, b.kv, "K/V");
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
  g.norm = norm(L, b.norm);
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

MlpBlock mlp(WeightLoader& L, const flyx_mlp_block& b, AdapterCarve& carve) {
  MlpBlock f;
  f.hidden = b.hidden;
  f.inter = b.inter;
  f.norm = norm(L, b.norm);
  int rows = 0;
  f.w_gu = L.lin_stack({b.gate, b.up}, &rows);
  expect(rows == 2 * f.inter, "MLP gate|up rows do not match its dimensions");
  f.w_down = L.lin(b.down);
  int in[3], out[3];
  mlp_dims(b, in, out);
  f.lora = block_lora(L, b.lora, 3, in, out, carve);
  return f;
}

DFlashLayer dflash_layer(WeightLoader& L, const flyx_dflash2_layer_block& b) {
  DFlashLayer l;
  l.norm = norm(L, b.norm);
  l.post_norm = norm(L, b.post_norm);
  l.q_norm = norm(L, b.q_norm);
  l.k_norm = norm(L, b.k_norm);
  l.w_qkv = L.lin_stack({b.q, b.k, b.v});
  l.w_o = L.lin(b.o);
  l.w_gu = L.lin_stack({b.gate, b.up});
  l.w_down = L.lin(b.down);
  l.attn_proj = L.lin(b.attn_conv_proj);
  l.mlp_proj = L.lin(b.mlp_conv_proj);
  l.attn_base = L.vec(b.attn_conv_base);
  l.mlp_base = L.vec(b.mlp_conv_base);
  return l;
}

DFlashBlock dflash(WeightLoader& L, const flyx_model_desc& d, const flyx_dflash2_block& b, const StateMap& sm) {
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
  expect(b.window >= 0, "a DFlash2 window below zero");
  f.window = b.window;
  f.n_taps = b.n_taps;
  f.target_hidden = d.hidden;
  f.conv_taps = b.conv_taps;
  f.conv_group = b.conv_group;
  f.sel_rank = b.sel_rank;
  f.sel_topk = b.sel_topk;
  f.embed = b.embed;
  f.head = b.head;
  f.first_layer = b.first_layer;
  f.n_layers = b.layers;
  f.fc = L.lin(b.fc);
  f.hidden_norm = norm(L, b.hidden_norm);
  f.norm = norm(L, b.norm);
  f.sel_pred = L.mat(b.sel_pred);
  f.sel_succ = L.mat(b.sel_succ);
  f.sel_hid = L.lin(b.sel_hid);
  // Context K/V projections of every layer in one GEMM, their k norms side by side.
  std::vector<int> ctx, kn;
  double eps = 0;
  for (int j = 0; j < b.layers; ++j) {
    expect(b.first_layer >= 0 && b.first_layer + j < d.n_blocks &&
               d.blocks[b.first_layer + j].kind == FLYX_BLOCK_DFLASH2_LAYER,
           "DFlash2 layers are not its layer blocks");
    const auto& l = d.blocks[b.first_layer + j].dflash2_layer;
    ctx.push_back(l.k);
    ctx.push_back(l.v);
    kn.push_back(l.k_norm.w);
    expect(l.k_norm.offset == 0.0 && (j == 0 || l.k_norm.eps == eps), "DFlash2 context k norms: plain weights, one eps");
    eps = l.k_norm.eps;
  }
  int rows = 0;
  f.ctx = L.lin_stack(ctx, &rows);
  expect(rows == f.ctx_width(), "DFlash2 context k|v rows do not match its dimensions");
  f.k_norms = {L.vecs(kn), 0.f, float(eps)};
  f.kv = state_of(sm.kv, b.kv, "K/V");
  const KvSpec& k = L.model()->states.kv[size_t(f.kv)];
  expect(k.layers == f.layers() && k.nkv == f.nkv && k.hd == f.hd, "DFlash2 context cache differs from its layers");
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
void add(flyx_bmodel* m, B b) {
  m->blocks.emplace_back(std::in_place_type<B>, std::move(b));
}

// Whether a head reads this table too (a tied head reads all of it every token).
bool head_reads(const flyx_model_desc& d, int table) {
  for (int i = 0; i < d.n_blocks; ++i)
    if (d.blocks[i].kind == FLYX_BLOCK_HEAD && d.blocks[i].head.w == table) return true;
  return false;
}

}  // namespace

void bind(flyx_bmodel* m, const flyx_model_desc& d, LoadTimes* times) {
  m->hidden = d.hidden;
  m->vocab = d.vocab;
  WeightLoader L(m, d);
  const StateMap sm = bind_states(m, d);
  m->blocks.reserve(size_t(d.n_blocks));
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
  for (int i = 0; i < d.n_blocks; ++i) {
    const flyx_block& b = d.blocks[i];
    auto ref = [&](int j, int kind, const char* what) {
      expect(j >= 0 && j < d.n_blocks && d.blocks[j].kind == kind,
             "block " + std::to_string(i) + " names no " + what + " block");
      return j;
    };
    switch (b.kind) {
      case FLYX_BLOCK_EMBED: {
        EmbedBlock e;
        e.hidden = b.embed.hidden;
        e.vocab = b.embed.vocab;
        // Every kernel that reads the table gathers the rows of a pass's tokens, a few kilobytes: a table no head
        // reads stays in host memory and frees its size on the GPU, unless the GPU has room (embed=auto, open_model).
        const bool host = m->opt.embed == 1 && !head_reads(d, b.embed.table);
        try {
          if (L.packable(b.embed.table)) e.blk = host ? L.pack_host(b.embed.table) : L.pack(b.embed.table);
          else e.table = host ? L.mat_host(b.embed.table) : L.mat(b.embed.table);
        } catch (const CudaError& err) {
          if (!host) throw;
          throw std::runtime_error(std::string("the embedding in host memory: ") + err.what() +
                                   " (backend option embed=gpu keeps it on the GPU)");
        }
        add(m, e);
        break;
      }
      case FLYX_BLOCK_HEAD: {
        HeadBlock h;
        h.hidden = b.head.hidden;
        h.vocab = b.head.vocab;
        h.norm = norm(L, b.head.norm);
        if (L.packable(b.head.w)) h.wb = L.pack(b.head.w);  // a tied head: the embedding's blocks
        else h.w = L.mat(b.head.w);
        add(m, h);
        break;
      }
      case FLYX_BLOCK_ATTN: add(m, attention(L, b.attn, sm, carve)); break;
      case FLYX_BLOCK_GDN: add(m, gdn(L, b.gdn, sm, carve)); break;
      case FLYX_BLOCK_MLP: add(m, mlp(L, b.mlp, carve)); break;
      case FLYX_BLOCK_MTP: {
        MtpBlock t;
        t.embed = ref(b.mtp.embed, FLYX_BLOCK_EMBED, "embedding");
        t.head = ref(b.mtp.head, FLYX_BLOCK_HEAD, "head");
        t.attn = ref(b.mtp.attn, FLYX_BLOCK_ATTN, "attention");
        t.mlp = ref(b.mtp.mlp, FLYX_BLOCK_MLP, "MLP");
        expect(!block_rank(d.blocks[t.attn].attn.lora, 4) && !block_rank(d.blocks[t.mlp].mlp.lora, 3),
               "an MTP layer with adapters (its draft steps run its blocks without them)");
        t.norm_emb = norm(L, b.mtp.norm_emb);
        t.norm_hidden = norm(L, b.mtp.norm_hidden);
        t.norm = norm(L, b.mtp.norm);
        t.fc = L.lin(b.mtp.fc);
        expect(b.mtp.ring >= 0 && b.mtp.ring < d.n_states && sm.ring[size_t(b.mtp.ring)], "MTP block names no ring");
        expect(m->mtp_block < 0, "more than one MTP drafter");
        expect(m->head_block < 0 || m->head_block == t.head, "drafters that read different heads");
        m->mtp_block = i;
        m->head_block = t.head;
        add(m, t);
        break;
      }
      case FLYX_BLOCK_DFLASH2: {
        const int e = ref(b.dflash2.embed, FLYX_BLOCK_EMBED, "embedding");
        const int head = ref(b.dflash2.head, FLYX_BLOCK_HEAD, "head");
        const auto& E = block<EmbedBlock>(m, e);
        const auto& Hd = block<HeadBlock>(m, head);
        expect((E.table || E.blk.data) && (Hd.w || Hd.wb.data),
               "the DFlash2 drafter reads an embedding and a head of bf16 or packed GGML blocks");
        expect(m->dflash_block < 0, "more than one DFlash2 drafter");
        expect(m->head_block < 0 || m->head_block == head, "drafters that read different heads");
        m->dflash_block = i;
        m->head_block = head;
        add(m, dflash(L, d, b.dflash2, sm));
        break;
      }
      case FLYX_BLOCK_DFLASH2_LAYER: add(m, dflash_layer(L, b.dflash2_layer)); break;
      case FLYX_BLOCK_DV: {
        const int e = ref(b.dv.embed, FLYX_BLOCK_EMBED, "embedding");
        const int head = ref(b.dv.head, FLYX_BLOCK_HEAD, "head");
        expect(block<EmbedBlock>(m, e).table && block<HeadBlock>(m, head).w,
               "the DV drafter reads a bf16 embedding and head");
        expect(m->dv_block < 0, "more than one DV drafter");
        expect(m->head_block < 0 || m->head_block == head, "drafters that read different heads");
        m->dv_block = i;
        m->head_block = head;
        add(m, dv(L, d, b.dv, sm));
        break;
      }
      case FLYX_BLOCK_DV_LAYER: add(m, dv_layer(L, d, b.dv_layer, carve)); break;
      case FLYX_BLOCK_VISION_STEM: add(m, vision_stem(L, b.vision_stem)); break;
      case FLYX_BLOCK_VIT_ATTN: add(m, vit_attn(L, b.vit_attn)); break;
      case FLYX_BLOCK_VIT_MLP: add(m, vit_mlp(L, b.vit_mlp)); break;
      case FLYX_BLOCK_VISION_MERGE: add(m, vision_merge(L, b.vision_merge)); break;
      default: throw std::runtime_error("block kind " + std::to_string(b.kind) + ": not implemented by this backend");
    }
  }
  m->vision_disk.flush_clean();
  L.finish();
  if (times) *times = L.times();
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
