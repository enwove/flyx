// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// DV's description (models/dv.h) from the tiny Llama's DV file (make_tiny.py: tiny_llama_dv.flyx, the tiny Llama with
// 8 slots, adapters of rank 32 and a selector of rank 16, a beam tree of 15): its blocks and state as backend_abi.h
// defines them, each weight the file's tensor of that name; its programs; the drafter as the options choose it (DV
// only when named: by default the file loads as tiny_llama.flyx); the same section with a pool tree, and with a tree
// of 31 drafts, a causal canvas, another fill and temperature (the tiny file has their defaults); and DV sections that
// cannot be drafted refused with a message.  The same target as tiny_llama.flyx, block for block.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "tests/common/check.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;
using nlohmann::json;

namespace {

std::unique_ptr<interfaces::Model> open(const ModelFile& f, const std::string& drafter, std::string* error,
                                        bool speculative = true) {
  interfaces::ModelOptions o;
  o.drafter = drafter;
  o.speculative = speculative;
  return interfaces::open_model(f, o, error);
}

const char* name_of(const flyx_model_desc& d, int w) { return w >= 0 && w < d.n_weights ? d.weights[w].t->name : ""; }

// Weight w is the file's tensor `name`, of this shape.
bool is(const flyx_model_desc& d, int w, const std::string& name, std::vector<int64_t> shape) {
  if (name_of(d, w) != name) return std::printf("  weight %d is %s, not %s\n", w, name_of(d, w), name.c_str()), false;
  const flyx_tensor_view* t = d.weights[w].t;
  bool same = t->ndim == int(shape.size());
  for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == shape[i];
  return same;
}

std::vector<int> nodes(const std::vector<flyx_node>& p) {
  std::vector<int> all;
  for (const auto& n : p) all.insert(all.end(), {n.kind, n.block, n.arg0, n.arg1});
  return all;
}

void description(const interfaces::Model& m, const interfaces::Model& plain) {
  const flyx_model_desc& d = m.desc();
  const flyx_model_desc& t = plain.desc();
  bool target = d.n_blocks == t.n_blocks + 5 && d.n_states == t.n_states + 1;
  for (int i = 0; target && i < t.n_blocks; ++i)
    target = d.blocks[i].kind == t.blocks[i].kind && std::memcmp(&d.blocks[i], &t.blocks[i], sizeof(flyx_block)) == 0;
  for (int i = 0; target && i < t.n_weights; ++i) target = std::strcmp(name_of(d, i), name_of(t, i)) == 0;
  check(target, "the target's blocks and weights are tiny_llama.flyx's, then 4 DV layers, DV and its canvas");
  if (!target) return;
  const flyx_block& b = d.blocks[d.n_blocks - 1];
  check(b.kind == FLYX_BLOCK_DV, "the last block is DV");
  if (b.kind != FLYX_BLOCK_DV) return;
  const flyx_dv_block& v = b.dv;
  const int H = 128, q = 16 * 128, kv = 2 * 128, I = 256, R = 32, S = 16;  // R the adapters' rank, S the selector's
  check(v.hidden == H && v.vocab == 768 && v.slots == 8 && v.tree == FLYX_DV_TREE_BEAM && v.tree_width == 15 &&
            v.beam == 4 && v.cov == 1 && v.pool_ranks == 0 && v.topk == 16 && v.res_k == 16 && v.mask_token == 3 &&
            v.causal == 0 && v.fill == 0.9f && v.sel_temp == 1.f && v.sel_rank == S,
        "DV: 8 slots, beam:15:4:cov, top 16, MASK <|mask|> (3), fill 0.9, selector rank 16");
  // The embedding and head blocks, and per layer the attention and MLP blocks, as the target's program runs them.
  std::vector<int> layers;
  for (const auto& n : plain.prefill_program())
    if (n.kind == FLYX_NODE_RUN) layers.push_back(n.block);
  check(v.embed == layers.front() && v.head == layers.back() && v.first_layer == t.n_blocks && v.layers == 4,
        "DV reads the target's embedding and head; its 4 layers follow the target's blocks");
  const char* lin[FLYX_LINS] = {"q", "k", "v", "o", "gate", "up", "down"};
  const int in[FLYX_LINS] = {H, H, H, q, H, H, I}, out[FLYX_LINS] = {q, kv, kv, H, I, I, H};
  bool ok = true;
  for (int l = 0; l < v.layers; ++l) {
    const flyx_block& y = d.blocks[v.first_layer + l];
    const std::string p = "dv.L" + std::to_string(l) + ".";
    ok = ok && y.kind == FLYX_BLOCK_DV_LAYER && y.dv_layer.attn == layers[size_t(1 + 2 * l)] &&
         y.dv_layer.mlp == layers[size_t(2 + 2 * l)] && d.blocks[y.dv_layer.attn].kind == FLYX_BLOCK_ATTN &&
         d.blocks[y.dv_layer.mlp].kind == FLYX_BLOCK_MLP;
    ok = ok && is(d, y.dv_layer.norm.w, p + "ln1c", {H}) && is(d, y.dv_layer.post_norm.w, p + "ln2c", {H}) &&
         y.dv_layer.norm.offset == 0 && y.dv_layer.norm.eps == 1e-6 && y.dv_layer.post_norm.eps == 1e-6;
    for (int i = 0; i < FLYX_LINS; ++i)
      ok = ok && y.dv_layer.lora[i].rank == R && is(d, y.dv_layer.lora[i].a, p + lin[i] + ".lora_a", {R, in[i]}) &&
           is(d, y.dv_layer.lora[i].b, p + lin[i] + ".lora_b", {out[i], R});
  }
  check(ok, "each DV layer: the target's attention and MLP blocks of its layer, ln1c and ln2c, a rank-32 adapter on "
            "q, k, v, o, gate, up, down");
  check(is(d, v.norm.w, "dv.fnormc", {H}) && v.norm.eps == 1e-6 && is(d, v.mask_emb, "dv.mask_emb", {H}) &&
            is(d, v.res_in, "dv.res_in", {2 * H, H}) && is(d, v.res_out, "dv.res_out", {H, H}),
        "DV's final norm, MASK embedding, residual context");
  bool sel = is(d, v.sel_a, "dv.sel.A", {S, H}) && is(d, v.sel_h, "dv.sel.H", {S, H}) &&
             is(d, v.sel_b, "dv.sel.B", {S, H}) && is(d, v.sel_alpha, "dv.sel.alpha", {1});
  for (int w : {v.sel_a, v.sel_h, v.sel_b, v.sel_alpha}) sel = sel && d.weights[w].t->dtype == FLYX_DT_F32;
  check(sel, "DV's selector: A, H, B [16, 128] and alpha [1], float32");
  const flyx_state& c = d.states[v.canvas];
  check(v.canvas == d.n_states - 1 && c.kind == FLYX_STATE_CANVAS && c.draft == 1 && c.slots == 8 && c.width == H &&
            c.topk == 16 && c.rank == S,
        "its canvas: 8 slots of 128 wide, 16 candidates, the selector's rank (16); only drafting reads it");

  // Programs: the prefill starts the canvas over; DV's round is its tree, the check, the canvas pass; plain and given
  // rounds are the target's alone.
  const int dv_block = d.n_blocks - 1;
  std::vector<flyx_node> want = plain.prefill_program();
  want.push_back({FLYX_NODE_CONTEXT, dv_block, 0, 0});
  check(nodes(m.prefill_program()) == nodes(want), "prefill: the target's, then CONTEXT(dv)");
  std::vector<flyx_node> verify;
  for (const auto& n : plain.round_program(0, -1, false))
    if (n.kind == FLYX_NODE_RUN) verify.push_back(n);
  want = {{FLYX_NODE_DRAFT, dv_block, 15, FLYX_DRAFT_TREE}};
  want.insert(want.end(), verify.begin(), verify.end());
  want.insert(want.end(),
              {{FLYX_NODE_ACCEPT, -1, 15, 0}, {FLYX_NODE_COMMIT, -1, 0, 0}, {FLYX_NODE_CONTEXT, dv_block, 0, 0}});
  check(nodes(m.round_program(15, 0, false)) == nodes(want),
        "DV's round: DRAFT(dv, 15, tree), the target's rows, ACCEPT, COMMIT, CONTEXT(dv)");
  check(nodes(m.round_program(0, -1, false)) == nodes(plain.round_program(0, -1, false)) &&
            nodes(m.round_program(15, -1, true)) == nodes(plain.round_program(15, -1, true)),
        "plain and given rounds: the target's alone (no DV node)");
  // DV's warm-up: plain decoding's round, then the canvas pass.
  want = verify;
  want.insert(want.end(),
              {{FLYX_NODE_ACCEPT, -1, 0, 0}, {FLYX_NODE_COMMIT, -1, 0, 0}, {FLYX_NODE_CONTEXT, dv_block, 0, 0}});
  check(nodes(m.round_program(0, 0, false)) == nodes(want),
        "DV's warm-up: the target's row, ACCEPT, COMMIT, CONTEXT(dv): plain decoding and the canvas pass");
}

// A copy of the file with its header's dv section changed (dir/dv_changed.flyx), opened with DV (--drafter dv).
// The file outlives the model that reads it; one copy at a time, since the next one is written over it.
struct Changed {
  ModelFile file;
  std::unique_ptr<interfaces::Model> model;
  std::string error;  // "" when it opened
};

std::unique_ptr<Changed> with_dv(const std::string& path, const std::string& dir,
                                 const std::function<void(json&)>& change) {
  auto c = std::make_unique<Changed>();
  std::ifstream in(path, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(in), {}};
  uint64_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 16, 8);
  json h = json::parse(bytes.substr(24, size_t(hlen)));
  change(h);
  const size_t data = (24 + size_t(hlen) + 4095) / 4096 * 4096;
  const std::string out = dir + "/dv_changed.flyx";
  if (!testing::write_file(out, testing::flyx_file(h.dump(), bytes.substr(data)))) {
    c->error = "cannot write " + out;
  } else if (!c->file.open(out, &c->error)) {
    c->error = "open: " + c->error;
  } else {
    c->model = open(c->file, "dv", &c->error);
    if (c->model) c->error.clear();
  }
  return c;
}

// The section's tree as a pool of 3 ranks (ST 15/3/1) instead of its beam: 15 drafts from 7 depths of 1 + 3 d tuples.
void pool_tree(json& h) { h["dv"]["tree"] = json{{"kind", "pool"}, {"width", 15}, {"ranks", 3}}; }

// The same file with a pool tree: the DV block and programs as with its beam, but for the tree's fields.
void pool_description(const std::string& path, const std::string& dir, const interfaces::Model& beam) {
  const auto c = with_dv(path, dir, pool_tree);
  check(c->model != nullptr, "the section with tree pool:15:3:1: opens (" + c->error + ")");
  if (!c->model) return;
  const interfaces::Model& m = *c->model;
  const flyx_model_desc& d = m.desc();
  const flyx_dv_block& v = d.blocks[d.n_blocks - 1].dv;
  check(d.n_blocks == beam.desc().n_blocks && d.blocks[d.n_blocks - 1].kind == FLYX_BLOCK_DV &&
            v.tree == FLYX_DV_TREE_POOL && v.tree_width == 15 && v.pool_ranks == 3 && v.beam == 0 && v.cov == 0 &&
            v.slots == 8 && v.topk == 16 && v.sel_rank == 16 && d.states[v.canvas].rank == 16,
        "pool:15:3:1: the DV block's tree is a pool of 3 ranks and 15 drafts, no beam, no coverage");
  check(m.drafters() == 1 && m.draft_widths(0) == std::vector<int>{15} && m.drafts_trees(0) && m.restarts(0) &&
            nodes(m.round_program(15, 0, false)) == nodes(beam.round_program(15, 0, false)),
        "pool:15:3:1: rounds of 15 drafts, trees only, the same round program");
}

// What the tiny file keeps at its defaults, read from the section all the same: a tree of 31 drafts (rounds of 15 and
// of 31, its width, docs/design/dv.md), a causal canvas, another fill and selector temperature; and a tree wider than
// a tree round verifies (rounds of 15 and of FLYX_MAX_WIDE_ROWS - 1, 63).
void other_settings(const std::string& path, const std::string& dir, const interfaces::Model& beam) {
  auto c = with_dv(path, dir, [](json& h) {
    h["dv"]["tree"] = json{{"kind", "beam"}, {"width", 31}, {"beam", 5}, {"cov", false}};  // 7 depths x 5 >= 31
    h["dv"]["causal"] = true;
    h["dv"]["fill"] = 0.5;
    h["dv"]["temp"] = 0.25;
  });
  check(c->model != nullptr, "the section with beam:31:5, causal, fill 0.5, temp 0.25: opens (" + c->error + ")");
  if (!c->model) return;
  const interfaces::Model& m = *c->model;
  const flyx_dv_block& v = m.desc().blocks[m.desc().n_blocks - 1].dv;
  check(v.tree == FLYX_DV_TREE_BEAM && v.tree_width == 31 && v.beam == 5 && v.cov == 0 && v.causal == 1 &&
            v.fill == 0.5f && v.sel_temp == 0.25f,
        "beam:31:5: its tree, causal, fill 0.5 and temp 0.25 in the DV block");
  std::vector<flyx_node> want = beam.round_program(31, 0, false);
  check(m.draft_widths(0) == std::vector<int>{15, 31} && nodes(m.round_program(31, 0, false)) == nodes(want) &&
            want.front().kind == FLYX_NODE_DRAFT && want.front().arg0 == 31,
        "a tree of 31 drafts: rounds of 15 and of 31, DRAFT(dv, 31, tree), the same round program");
  c.reset();  // its file closed before the next copy is written over it
  c = with_dv(path, dir, [](json& h) {
    h["dv"]["tree"] = json{{"kind", "beam"}, {"width", 100}, {"beam", 15}, {"cov", true}};  // 7 depths x 15 >= 100
  });
  check(c->model != nullptr, "the section with beam:100:15:cov: opens (" + c->error + ")");
  if (!c->model) return;
  check(c->model->draft_widths(0) == std::vector<int>{FLYX_MAX_ROWS - 1, FLYX_MAX_WIDE_ROWS - 1},
        "a tree of 100 drafts: rounds of 15 and of FLYX_MAX_WIDE_ROWS - 1 (63), the most a tree round verifies");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 2) {
    std::fprintf(stderr, "usage: dv_models_test ASSET_DIR\n");
    return 2;
  }
  const std::string dir = argv[1], path = dir + "/tiny_llama_dv.flyx";
  ModelFile f, fp;
  std::string error;
  if (!f.open(path, &error) || !fp.open(dir + "/tiny_llama.flyx", &error)) {
    std::printf("FAIL open: %s\n", error.c_str());
    return 1;
  }
  const auto m = open(f, "dv", &error);
  const auto plain = open(fp, "", &error);
  check(m && plain, "both open: " + error);
  if (!m || !plain) return 1;
  check(m->drafters() == 1 && std::string(m->drafter()) == "dv" && std::string(m->drafter_name(0)) == "dv" &&
            m->draft_widths(0) == std::vector<int>{15} && m->drafts_trees(0) && m->restarts(0),
        "--drafter dv: DV, rounds of 15 drafts, trees only, its canvas following its own rounds only");
  check(plain->drafters() == 0, "tiny_llama.flyx: no drafter");
  description(*m, *plain);
  pool_description(path, dir, *m);
  other_settings(path, dir, *m);

  for (const std::string& name : {std::string("dv"), std::string("all")}) {
    const auto o = open(f, name, &error);
    check(o && o->drafters() == 1 && std::string(o->drafter()) == "dv", "--drafter " + name + ": DV");
  }
  // DV is no default: its rounds are slower than plain decoding past short answers (models/model.cpp).
  for (const auto& [drafter, speculative] : {std::pair<const char*, bool>{"", true}, {"none", true}, {"", false}}) {
    const auto none = open(f, drafter, &error, speculative);
    bool same = none && none->drafters() == 0 && none->desc().n_blocks == plain->desc().n_blocks &&
                none->desc().n_states == plain->desc().n_states &&
                nodes(none->prefill_program()) == nodes(plain->prefill_program());
    check(same, std::string(!speculative ? "no speculation" : *drafter ? "--drafter none" : "the default drafter") +
                    ": tiny_llama.flyx's description and programs (no DV block, state or node)");
  }
  const bool no_mtp = !open(f, "mtp", &error) && error.find("no MTP layer") != std::string::npos;
  check(no_mtp, "--drafter mtp: refused (" + error + ")");
  const bool no_dv = !open(fp, "dv", &error) && error.find("no DV drafter") != std::string::npos;
  check(no_dv, "--drafter dv on tiny_llama.flyx: refused (" + error + ")");

  // Sections the converter would refuse are refused here too: a file is read whoever wrote it.
  struct Bad {
    const char* what;
    const char* because;
    std::function<void(json&)> change;
  };
  const std::vector<Bad> bad = {
      {"slots past the bound", "dimensions past any model's", [](json& h) { h["dv"]["slots"] = 1 << 20; }},
      {"adapter rank past the bound", "dimensions past any model's", [](json& h) { h["dv"]["rank"] = 1 << 20; }},
      {"tree kind", "tree kind 'fan'", [](json& h) { h["dv"]["tree"]["kind"] = "fan"; }},
      {"beam too narrow", "fewer nodes than its width", [](json& h) { h["dv"]["tree"]["beam"] = 2; }},  // 7 x 2 < 15
      {"res_k", "topk and res_k 16", [](json& h) { h["dv"]["res_k"] = 8; }},
      {"topk", "topk and res_k 16", [](json& h) { h["dv"]["topk"] = 8; h["dv"]["res_k"] = 8; }},
      {"topk of a selector of 8", "topk and res_k 16", [](json& h) { h["dv"]["topk"] = 8; }},
      {"mask token", "mask token outside", [](json& h) { h["dv"]["mask_token"] = 768; }},
      {"fill", "not a probability", [](json& h) { h["dv"]["fill"] = 2.0; }},
      {"arch (Qwen3)", "not a Llama .flyx", [](json& h) { h["arch"] = "qwen3"; }},
      {"selector rank", "dv.sel.A: shape 16x128", [](json& h) { h["dv"]["sel_rank"] = 32; }},
      {"adapter rank", "dv.L0.q.lora_a: shape 32x128", [](json& h) { h["dv"]["rank"] = 16; }},
      {"pool rank of a slot's 16 candidates", "a pool tree that cannot give its width",
       [](json& h) { pool_tree(h); h["dv"]["tree"]["ranks"] = 16; }},
      {"pool rank 0", "a pool tree that cannot give its width",
       [](json& h) { pool_tree(h); h["dv"]["tree"]["ranks"] = 0; }},
      {"pool too small", "a pool tree that cannot give its width",  // 7 depths of rank 1: 35 tuples
       [](json& h) { pool_tree(h); h["dv"]["tree"]["ranks"] = 1; h["dv"]["tree"]["width"] = 36; }},
      {"pool without ranks", "'ranks' not found", [](json& h) { pool_tree(h); h["dv"]["tree"].erase("ranks"); }},
  };
  for (const Bad& b : bad) {
    const std::string e = with_dv(path, dir, b.change)->error;
    check(e.find(b.because) != std::string::npos, std::string("a DV section with a bad ") + b.what + ": refused (" +
                                                       e + ")");
  }
  check(with_dv(path, dir, [](json&) {})->error.empty(), "the same section rewritten: opens");
  return flyx::test::finish("dv_models_test");
}
