// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV from its file: the header's "dv" section (python/convert/flyx_convert/dv.py writes it) and the dv.* tensors,
// the names python/reference/flyx_ref/dv.py reads.  The description follows the earlier CUDA port's models/dv.cpp
// (docs/prior-art.md) in what it holds; the blocks are flyx's (backend_abi.h, ABI 42).
#include "models/dv.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace flyx::models::dv {
namespace {

using nlohmann::json;

constexpr const char* kLinears[FLYX_LINS] = {"q", "k", "v", "o", "gate", "up", "down"};
// A slot's candidates and those its expected embedding sums: 16 each, bifill-dv's canvases' and what the backends hold;
// the converter writes nothing else.
constexpr int kTopk = 16;

void expect(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("DV: " + what);
}

// Tuples of a pool tree over `depths` slots: the selector's best path and d * ranks deviations at depth d.
long pool_size(int depths, int ranks) {
  long n = 0;
  for (int d = 1; d <= depths; ++d) n += 1 + long(d) * ranks;
  return n;
}

}  // namespace

bool in_file(const ModelFile& file) {
  const json& h = file.header();
  return h.is_object() && h.contains("dv") && h.at("dv").is_object();
}

int describe(Tables& T, const ModelFile& file, const Target& t) {
  const json& j = file.header().at("dv");
  const json& tree = j.at("tree");
  const int H = t.hidden, q = t.n_heads * t.head_dim, kv = t.n_kv * t.head_dim, I = t.inter;
  const int layers = int(t.attn.size());
  expect(layers > 0 && t.mlp.size() == t.attn.size(), "a target of attention and MLP layers");
  flyx_dv_block d{};
  d.hidden = H;
  d.vocab = t.vocab;
  d.slots = j.at("slots");
  const std::string kind = tree.at("kind");
  expect(kind == "beam" || kind == "pool", "tree kind '" + kind + "' (beam or pool)");
  d.tree = kind == "beam" ? FLYX_DV_TREE_BEAM : FLYX_DV_TREE_POOL;
  d.tree_width = tree.at("width");
  d.beam = kind == "beam" ? tree.at("beam").get<int>() : 0;
  d.cov = kind == "beam" && tree.at("cov").get<bool>();
  d.pool_ranks = kind == "pool" ? tree.at("ranks").get<int>() : 0;
  d.topk = j.at("topk");
  d.res_k = j.at("res_k");
  d.mask_token = j.at("mask_token");
  d.causal = j.at("causal").get<bool>();
  d.fill = j.at("fill").get<float>();
  d.sel_temp = j.value("temp", 1.0f);
  d.sel_rank = j.at("sel_rank");
  const int rank = j.at("rank");
  // What the converter refuses, again: a file is read whoever wrote it.  Its dimensions bounded as a model's are
  // (models/model.h), so that no product of two of them wraps round.
  expect(d.slots <= kMaxHeads && d.tree_width <= kMaxHeads && d.beam <= kMaxHeads && rank <= kMaxHeadDim &&
             d.sel_rank <= kMaxHeadDim,
         "dimensions past any model's (slots, tree width and beam at most 4096, ranks at most 16384)");
  const int depths = std::min(d.slots - 1, d.tree_width);
  expect(d.slots >= 2 && d.tree_width >= 1 && depths >= 1, "a canvas of 2 slots or more and a tree of 1 draft or more");
  expect(d.topk == kTopk && d.res_k == kTopk, "topk and res_k 16 (a slot's candidates, those z sums)");
  expect(d.mask_token >= 0 && d.mask_token < t.vocab, "a mask token outside the vocabulary");
  expect(d.fill >= 0.f && d.fill <= 1.f, "fill is not a probability");
  expect(d.sel_temp > 0.f, "a selector temperature of 0 or less");
  expect(rank >= 1 && d.sel_rank >= 1, "ranks of 1 or more");
  if (d.tree == FLYX_DV_TREE_BEAM)
    expect(d.beam >= 1 && long(depths) * d.beam >= d.tree_width, "a beam tree that holds fewer nodes than its width");
  else
    expect(d.pool_ranks >= 1 && d.pool_ranks < d.topk && pool_size(depths, d.pool_ranks) >= d.tree_width,
           "a pool tree that cannot give its width (ranks 1 to topk - 1)");
  flyx_state canvas{};
  canvas.kind = FLYX_STATE_CANVAS;
  canvas.draft = 1;
  canvas.width = H;
  canvas.slots = d.slots;
  canvas.topk = d.topk;
  canvas.rank = d.sel_rank;
  d.canvas = T.state(canvas);
  d.embed = t.embed;
  d.head = t.head;
  // A layer's adapters: a [rank, in], b [out, rank] on q, k, v, o (attention) and gate, up, down (MLP).
  const int in[FLYX_LINS] = {H, H, H, q, H, H, I}, out[FLYX_LINS] = {q, kv, kv, H, I, I, H};
  d.first_layer = int(T.blocks.size());
  d.layers = layers;
  for (int l = 0; l < layers; ++l) {
    const std::string p = "dv.L" + std::to_string(l) + ".";
    flyx_block b{};
    b.kind = FLYX_BLOCK_DV_LAYER;
    auto& y = b.dv_layer;
    y.attn = t.attn[size_t(l)];
    y.mlp = t.mlp[size_t(l)];
    y.norm = T.norm(p + "ln1c", H, 0.0, t.eps);
    y.post_norm = T.norm(p + "ln2c", H, 0.0, t.eps);
    for (int i = 0; i < FLYX_LINS; ++i)
      y.lora[i] = {T.weight(p + kLinears[i] + ".lora_a", {rank, in[i]}),
                   T.weight(p + kLinears[i] + ".lora_b", {out[i], rank}), rank, 1.f};  // bifill-dv's: no scale
    T.block(b);
  }
  d.norm = T.norm("dv.fnormc", H, 0.0, t.eps);
  d.mask_emb = T.weight("dv.mask_emb", {H});
  d.res_in = T.weight("dv.res_in", {2 * H, H});
  d.res_out = T.weight("dv.res_out", {H, H});
  d.sel_a = T.weight("dv.sel.A", {d.sel_rank, H});
  d.sel_h = T.weight("dv.sel.H", {d.sel_rank, H});
  d.sel_b = T.weight("dv.sel.B", {d.sel_rank, H});
  d.sel_alpha = T.weight("dv.sel.alpha", {1});
  for (int w : {d.sel_a, d.sel_h, d.sel_b, d.sel_alpha})
    expect(T.weights[size_t(w)].t->dtype == FLYX_DT_F32, std::string(T.weights[size_t(w)].t->name) + " is not float32");
  flyx_block b{};
  b.kind = FLYX_BLOCK_DV;
  b.dv = d;
  return T.block(b);
}

std::vector<flyx_node> round_program(int block, int width, const std::vector<flyx_node>& verify) {
  std::vector<flyx_node> p;
  if (width > 0) p.push_back({FLYX_NODE_DRAFT, block, width, FLYX_DRAFT_TREE});
  p.insert(p.end(), verify.begin(), verify.end());
  p.push_back({FLYX_NODE_ACCEPT, -1, width, 0});
  p.push_back({FLYX_NODE_COMMIT, -1, 0, 0});
  p.push_back({FLYX_NODE_CONTEXT, block, 0, 0});
  return p;
}

}  // namespace flyx::models::dv
