// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// One model, two files: the description models/<arch> builds from a GGUF file
// (llama.cpp's names and layout, said in the description's flags) must mean
// the weights of the description built from the .flyx file of the same
// model.  The flags are read here as backend_abi.h defines them, so this pins
// what they mean without a backend.  And files no model here can be built
// from are refused with a message; and NVFP4 weights take bf16 inputs on a
// device without FP4 tensor cores, NVFP4 ones when asked for by name.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "ops/dtype/ref.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

struct Side {
  const flyx_model_desc* d;
  double at(int w, size_t i) const { return ops::load(d->weights[w].t, i); }
  size_t cols(int w) const { return size_t(d->weights[w].t->shape[d->weights[w].t->ndim - 1]); }
};

struct Compare {
  Side a, b;
  double tolerance;
  std::string where;
  bool ok = true;

  void same(double x, double y, const char* what) {
    if (ok && !(std::abs(x - y) <= tolerance * std::max(1.0, std::abs(x)))) {
      ok = false;
      std::printf("  %s %s: %.9g / %.9g\n", where.c_str(), what, x, y);
    }
  }
  // Rows [ra, ra + n) of a's weight against rows [rb, rb + n) of b's (a vector: its elements).
  void rows(int wa, size_t ra, int wb, size_t rb, size_t n, const char* what) {
    const bool matrix = a.d->weights[wa].t->ndim == 2;
    const size_t k = matrix ? a.cols(wa) : 1;
    if (matrix && k != b.cols(wb)) return same(double(k), double(b.cols(wb)), "columns");
    for (size_t i = 0; ok && i < n * k; ++i) same(a.at(wa, ra * k + i), b.at(wb, rb * k + i), what);
  }
  void whole(int wa, int wb, const char* what) { rows(wa, 0, wb, 0, size_t(a.d->weights[wa].t->shape[0]), what); }
  void norm(const flyx_norm& x, const flyx_norm& y, size_t n, const char* what) {
    same(x.eps, y.eps, "eps");
    for (size_t i = 0; ok && i < n; ++i) same(x.offset + a.at(x.w, i), y.offset + b.at(y.w, i), what);
  }
};

// The stored row where head h's q (part 0) or gate (part 1) rows start.
size_t q_row(const flyx_attn_block& b, int h, int part) {
  return size_t(b.gate == FLYX_GATE_PER_HEAD ? 2 * h + part : part * b.n_heads + h) * b.head_dim;
}
// Where a file keeps value head i of key head kh.
size_t value_head(const flyx_gdn_block& b, int kh, int i) {
  return size_t(b.head_order == FLYX_GDN_HEADS_TILED ? i * b.nk + kh : kh * (b.nv / b.nk) + i);
}

void attention(Compare& c, const flyx_attn_block& x, const flyx_attn_block& y) {
  c.same(x.hidden, y.hidden, "hidden");
  c.same(x.n_heads, y.n_heads, "heads");
  c.same(x.n_kv, y.n_kv, "kv heads");
  c.same(x.head_dim, y.head_dim, "head dim");
  c.same(x.rot_dim, y.rot_dim, "rotary dims");
  c.same(x.theta, y.theta, "theta");
  c.same(x.gate != FLYX_GATE_NONE, y.gate != FLYX_GATE_NONE, "gate");
  if (!c.ok) return;
  const size_t hd = size_t(x.head_dim);
  c.norm(x.norm, y.norm, size_t(x.hidden), "norm");
  c.norm(x.q_norm, y.q_norm, hd, "q norm");
  c.norm(x.k_norm, y.k_norm, hd, "k norm");
  for (int h = 0; h < x.n_heads; ++h)
    for (int part = 0; part < (x.gate ? 2 : 1); ++part) c.rows(x.q, q_row(x, h, part), y.q, q_row(y, h, part), hd, "q");
  c.whole(x.k, y.k, "k");
  c.whole(x.v, y.v, "v");
  c.whole(x.o, y.o, "o");
}

void gdn(Compare& c, const flyx_gdn_block& x, const flyx_gdn_block& y) {
  for (auto [p, q] : {std::pair{x.hidden, y.hidden}, {x.nk, y.nk}, {x.nv, y.nv}, {x.dk, y.dk}, {x.dv, y.dv}, {x.conv, y.conv}})
    c.same(p, q, "dimensions");
  c.same(c.a.d->states[x.state].head_order, x.head_order, "state head order (a)");
  c.same(c.b.d->states[y.state].head_order, y.head_order, "state head order (b)");
  if (!c.ok) return;
  const size_t kd = size_t(2) * x.nk * x.dk, dv = size_t(x.dv), H = size_t(x.hidden);
  c.norm(x.norm, y.norm, H, "norm");
  c.norm(x.out_norm, y.out_norm, dv, "output norm");
  c.rows(x.qkv, 0, y.qkv, 0, kd, "q and k");
  c.rows(x.conv_w, 0, y.conv_w, 0, kd, "conv of q and k");
  for (int kh = 0; kh < x.nk; ++kh)
    for (int i = 0; i < x.nv / x.nk; ++i) {
      const size_t hx = value_head(x, kh, i), hy = value_head(y, kh, i);
      c.rows(x.qkv, kd + hx * dv, y.qkv, kd + hy * dv, dv, "v");
      c.rows(x.conv_w, kd + hx * dv, y.conv_w, kd + hy * dv, dv, "conv of v");
      c.rows(x.z, hx * dv, y.z, hy * dv, dv, "z");
      c.rows(x.b, hx, y.b, hy, 1, "b");
      c.rows(x.a, hx, y.a, hy, 1, "a");
      c.rows(x.dt_bias, hx, y.dt_bias, hy, 1, "dt bias");
      auto neg_exp = [](const Side& s, const flyx_gdn_block& g, size_t h) {
        const double v = s.at(g.A_log, h);
        return g.decay == FLYX_GDN_DECAY_NEG_EXP ? v : -std::exp(v);
      };
      c.same(neg_exp(c.a, x, hx), neg_exp(c.b, y, hy), "decay");
      for (size_t r = 0; c.ok && r < H; ++r)  // the output projection's columns of this head
        for (size_t j = 0; c.ok && j < dv; ++j)
          c.same(c.a.at(x.out, r * c.a.cols(x.out) + hx * dv + j), c.b.at(y.out, r * c.b.cols(y.out) + hy * dv + j), "out");
    }
}

void same_model(const std::string& canonical, const std::string& gguf, double tolerance = 4e-7) {
  ModelFile fa, fb;
  std::string error;
  if (!fa.open(canonical, &error) || !fb.open(gguf, &error)) return check(false, error);
  const interfaces::ModelOptions opts{fa.header().contains("mtp") && !fa.header()["mtp"].is_null() ? "mtp" : "none", true,
                                      ""};
  const auto ma = interfaces::open_model(fa, opts, &error);
  const auto mb = interfaces::open_model(fb, opts, &error);
  if (!ma || !mb) return check(false, gguf + ": " + error);
  const auto& da = ma->desc();
  const auto& db = mb->desc();
  bool ok = da.hidden == db.hidden && da.vocab == db.vocab && da.n_blocks == db.n_blocks && da.n_states == db.n_states;
  for (int i = 0; ok && i < da.n_blocks; ++i) {
    const auto& x = da.blocks[i];
    const auto& y = db.blocks[i];
    Compare c{{&da}, {&db}, tolerance, "block " + std::to_string(i)};
    if (x.kind != y.kind) {
      ok = false;
      break;
    }
    switch (x.kind) {
      case FLYX_BLOCK_EMBED: c.whole(x.embed.table, y.embed.table, "table"); break;
      case FLYX_BLOCK_HEAD:
        c.norm(x.head.norm, y.head.norm, size_t(da.hidden), "norm");
        c.same(x.head.w == da.blocks[0].embed.table, y.head.w == db.blocks[0].embed.table, "tied head");
        c.whole(x.head.w, y.head.w, "head");
        break;
      case FLYX_BLOCK_ATTN: attention(c, x.attn, y.attn); break;
      case FLYX_BLOCK_GDN: gdn(c, x.gdn, y.gdn); break;
      case FLYX_BLOCK_MLP:
        c.norm(x.mlp.norm, y.mlp.norm, size_t(x.mlp.hidden), "norm");
        c.whole(x.mlp.gate, y.mlp.gate, "gate");
        c.whole(x.mlp.up, y.mlp.up, "up");
        c.whole(x.mlp.down, y.mlp.down, "down");
        break;
      case FLYX_BLOCK_MTP:
        c.norm(x.mtp.norm_emb, y.mtp.norm_emb, size_t(da.hidden), "embedding norm");
        c.norm(x.mtp.norm_hidden, y.mtp.norm_hidden, size_t(da.hidden), "hidden norm");
        c.norm(x.mtp.norm, y.mtp.norm, size_t(da.hidden), "norm");
        c.whole(x.mtp.fc, y.mtp.fc, "fc");
        for (auto [p, q] : {std::pair{x.mtp.embed, y.mtp.embed}, {x.mtp.head, y.mtp.head}, {x.mtp.attn, y.mtp.attn}, {x.mtp.mlp, y.mtp.mlp}})
          c.same(p, q, "blocks");
        break;
      default: c.ok = false;
    }
    ok = c.ok;
  }
  auto programs = [](const interfaces::Model& m) {
    std::vector<int> all;
    for (const auto& p : {m.prefill_program(), m.round_program(0, -1, false), m.round_program(2, -1, true)})
      for (const auto& n : p) all.insert(all.end(), {n.kind, n.block, n.arg0, n.arg1});
    for (int d = 0; d < m.drafters(); ++d) {
      const std::vector<int> widths = m.draft_widths(d);
      all.insert(all.end(), widths.begin(), widths.end());
      for (const auto& n : m.round_program(widths.back(), d, false)) all.insert(all.end(), {n.kind, n.block, n.arg0, n.arg1});
    }
    return all;
  };
  check(ok, "the GGUF description means the .flyx description's weights: " + gguf);
  check(programs(*ma) == programs(*mb) && ma->drafters() == mb->drafters() && std::string(ma->drafter()) == mb->drafter(),
        "the same programs and drafters");
}

// A GGUF the reader reads and no model can be built from: the fixture with
// the bytes after `key` changed by `change`; the error says `because`.
void refused(const std::string& gguf, const std::string& dir, const std::string& what, const std::string& key,
             const std::function<void(char*)>& change, const std::string& because = "") {
  std::ifstream in(gguf, std::ios::binary);
  std::vector<char> data{std::istreambuf_iterator<char>(in), {}};
  const auto at = std::search(data.begin(), data.end(), key.begin(), key.end());
  if (at == data.end()) return check(false, "fixture has no " + key);
  change(&*at + key.size());
  const std::string path = dir + "/refused.gguf";
  std::ofstream(path, std::ios::binary).write(data.data(), std::streamsize(data.size()));
  ModelFile f;
  std::string error;
  const bool opened = f.open(path, &error);
  const auto m = opened ? interfaces::open_model(f, {}, &error) : nullptr;
  check(opened && !m && !error.empty() && error.find(because) != std::string::npos, "refuse " + what + ": " + error);
}

// The inputs a model's weights take by device: NVFP4 weights whose file says NVFP4 inputs keep them
// where the device computes them and take bf16 ones where it does not; every other weight's stay as the file has them.
void fp4_inputs_by_device(const std::string& path) {
  ModelFile f;
  std::string error;
  if (!f.open(path, &error)) return check(false, error);
  auto inputs = [&](bool fp4) {
    interfaces::ModelOptions o{"none", true, ""};
    o.fp4_inputs = fp4;
    std::vector<int32_t> act;
    if (const auto m = interfaces::open_model(f, o, &error))
      for (int i = 0; i < m->desc().n_weights; ++i) act.push_back(m->desc().weights[i].activations);
    return act;
  };
  const std::vector<int32_t> on = inputs(true), off = inputs(false);
  check(std::count(on.begin(), on.end(), FLYX_ACT_NVFP4) > 0, path + ": NVFP4 inputs where the device computes them");
  bool same = !off.empty() && off.size() == on.size();
  for (size_t i = 0; same && i < on.size(); ++i)
    same = off[i] == (on[i] == FLYX_ACT_NVFP4 ? FLYX_ACT_FORMAT : on[i]);
  check(same, path + ": bf16 inputs for those weights alone where it does not");
}

// NVFP4 inputs asked for by name (--activations nvfp4): every NVFP4 weight takes them whatever its file says (bf16
// inputs, or GGML NVFP4 of a K no multiple of 128, which keeps bf16 ones by default) and whatever the device (its
// backend refuses what it cannot compute), every other weight what it takes by default; the word a file says is only
// the default. A model without NVFP4 weights is what it is without the word; an unknown word is refused.
void activations_asked(const std::string& dir) {
  struct Weight {
    int32_t dtype, inputs;
  };
  auto weights = [](const std::string& path, const std::string& how, bool fp4, std::string* error) {
    std::vector<Weight> w;  // empty: refused, *error says why
    ModelFile f;
    if (!f.open(path, error)) return w;
    interfaces::ModelOptions o{"none", true, how};
    o.fp4_inputs = fp4;
    if (const auto m = interfaces::open_model(f, o, error))
      for (int i = 0; i < m->desc().n_weights; ++i) {
        const flyx_weight& x = m->desc().weights[i];
        w.push_back({x.t ? x.t->dtype : -1, x.activations});
      }
    return w;
  };
  auto nvfp4_inputs = [](const std::vector<Weight>& w) {
    return int(std::count_if(w.begin(), w.end(), [](const Weight& x) { return x.inputs == FLYX_ACT_NVFP4; }));
  };
  for (const char* name : {"tiny_w4a4_a16.flyx", "tiny_nvfp4.flyx", "tiny_nvfp4.gguf", "tiny_w4a4.flyx"}) {
    const std::string path = dir + "/" + name;
    std::string error;
    const std::vector<Weight> asked = weights(path, "nvfp4", true, &error), as_said = weights(path, "", true, &error);
    bool same = !asked.empty() && asked.size() == as_said.size();
    int nvfp4 = 0;
    for (size_t i = 0; same && i < asked.size(); ++i) {
      const bool fp4 = asked[i].dtype == FLYX_DT_NVFP4 || asked[i].dtype == FLYX_DT_GGML_NVFP4;
      nvfp4 += fp4;
      same = asked[i].dtype == as_said[i].dtype && asked[i].inputs == (fp4 ? FLYX_ACT_NVFP4 : as_said[i].inputs);
    }
    check(same && nvfp4 > 0, path + ": asked for, NVFP4 inputs for its " + std::to_string(nvfp4) +
                                 " NVFP4 weights, the others' as the file has them" +
                                 (error.empty() ? "" : " (" + error + ")"));
    const bool says_nvfp4 = std::string(name) == "tiny_w4a4.flyx";
    check(!as_said.empty() && nvfp4_inputs(as_said) == (says_nvfp4 ? nvfp4 : 0),
          path + (says_nvfp4 ? ": the same unasked (the file says NVFP4)" : ": bf16 inputs for them unless asked"));
  }
  auto same_weights = [](const std::vector<Weight>& a, const std::vector<Weight>& b) {
    bool same = !a.empty() && a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) same = a[i].dtype == b[i].dtype && a[i].inputs == b[i].inputs;
    return same;
  };
  std::string error;
  const std::string a16 = dir + "/tiny_w4a4_a16.flyx";
  check(same_weights(weights(a16, "nvfp4", false, &error), weights(a16, "nvfp4", true, &error)),
        a16 + ": asked for, the same on a device without FP4 tensor cores (its backend refuses them)" +
            (error.empty() ? "" : " (" + error + ")"));
  for (const char* name : {"tiny.flyx", "tiny_q8.gguf"}) {
    const std::string path = dir + "/" + name;
    error.clear();
    const std::vector<Weight> asked = weights(path, "nvfp4", true, &error);
    check(same_weights(asked, weights(path, "", true, &error)) && nvfp4_inputs(asked) == 0,
          path + ": no NVFP4 weights, asked for or not the same inputs" + (error.empty() ? "" : " (" + error + ")"));
  }
  error.clear();
  const bool turned_down = weights(dir + "/tiny_w4a4.flyx", "fp4", true, &error).empty();
  check(turned_down && error.find("exact, nvfp4") != std::string::npos,
        "an unknown activations word: refused (" + error + ")");
}

// DFlash2 drafter config validation: mismatched hidden size (would OOB target embed/head), odd head_dim,
// oversized sel_rank, and 32-bit signed dimension-product overflows (TG2, n_taps*hidden, ctx_width) must all be
// rejected before tensor binding.
void dflash2_validation(const std::string& dir) {
  using json = nlohmann::json;
  // Build a complete zero-filled DFlash2 drafter .flyx matching `d` so any missing config check actually passes
  // tensor lookup and opens a broken model.
  auto make_drafter = [&](const std::string& path, const json& d, bool with_tensors) {
    json tensors = json::object();
    size_t off = 0;
    auto add = [&](const std::string& name, std::initializer_list<int64_t> shape) {
      int64_t n = 1;
      for (int64_t s : shape) n *= s;
      const size_t nbytes = size_t(n) * 2;
      tensors[name] = {{"dtype", "bf16"}, {"shape", json(shape)}, {"offset", off}, {"nbytes", nbytes}};
      off = (off + nbytes + 63) / 64 * 64;
    };
    if (with_tensors) {
      const int64_t H = d.at("hidden"), layers = d.at("layers"), nh = d.at("n_heads"), nkv = d.at("n_kv");
      const int64_t hd = d.at("head_dim"), I = d.at("inter"), ct = d.at("conv_taps"), cg = d.at("conv_group");
      const int64_t sr = d.at("sel_rank"), nt = int64_t(d.at("taps").size());
      const int64_t th = d.at("target").at("hidden"), vocab = d.at("target").at("vocab");
      const int64_t q = nh * hd, kv = nkv * hd, tg2 = 2 * ct * (H / cg);
      for (int64_t l = 0; l < layers; ++l) {
        const std::string p = "draft.L" + std::to_string(l) + ".";
        add(p + "in_norm", {H});
        add(p + "post_norm", {H});
        add(p + "attn.q_norm", {hd});
        add(p + "attn.k_norm", {hd});
        add(p + "attn.q", {q, H});
        add(p + "attn.k", {kv, H});
        add(p + "attn.v", {kv, H});
        add(p + "attn.o", {H, q});
        add(p + "mlp.gate", {I, H});
        add(p + "mlp.up", {I, H});
        add(p + "mlp.down", {H, I});
        add(p + "attn_conv.proj", {tg2, H});
        add(p + "attn_conv.base", {2, ct, H});
        add(p + "mlp_conv.proj", {tg2, H});
        add(p + "mlp_conv.base", {2, ct, H});
      }
      add("draft.fc", {H, nt * th});
      add("draft.hidden_norm", {H});
      add("draft.norm", {H});
      add("draft.sel.pred", {vocab, sr});
      add("draft.sel.succ", {vocab, sr});
      add("draft.sel.hid", {sr, H});
    }
    json hdr = {{"format", "flyx"}, {"version", 1}, {"kind", "drafter"}, {"draft", d}, {"tensors", tensors}};
    std::string data(off, '\0');
    testing::write_file(path, testing::flyx_file(hdr.dump(), data));
  };

  const json base = {
      {"kind", "dflash2"},  {"hidden", 64},     {"layers", 1},      {"n_heads", 4},     {"n_kv", 2},
      {"head_dim", 16},     {"inter", 128},     {"eps", 1e-6},      {"rope_theta", 1e4},{"block", 8},
      {"mask_token", 3},    {"taps", {2, 5}},   {"conv_taps", 2},   {"conv_group", 16}, {"sel_rank", 16},
      {"sel_topk", 16},     {"window", 0},      {"target", {{"hidden", 64}, {"layers", 8}, {"vocab", 768}}},
  };
  auto check_refused = [&](const std::string& what, const json& d, bool with_tensors, const char* expect_sub) {
    const std::string path = dir + "/bad_dflash2.flyx";
    make_drafter(path, d, with_tensors);
    std::string error;
    bool ok = false, opened_model = false;
    {
      ModelFile f;
      ok = f.open(dir + "/tiny_gguf.gguf", &error) && f.add(path, &error);
      const auto m = ok ? interfaces::open_model(f, {}, &error) : nullptr;
      opened_model = m != nullptr;
    }
    std::remove(path.c_str());
    check(ok && !opened_model && error.find(expect_sub) != std::string::npos,
          "dflash2 refuse " + what + " (" + (opened_model ? "OPENED!" : error) + ")");
  };

  {  // CRITICAL: draft.hidden != target.hidden with matching tensors must still be refused
    json d = base;
    d["hidden"] = 32;
    check_refused("hidden != target hidden", d, true, "hidden");
  }
  {  // HIGH: odd head_dim breaks full-head RoPE
    json d = base;
    d["head_dim"] = 15;
    check_refused("odd head_dim", d, true, "invalid DFlash2");
  }
  {  // HIGH: 64-bit tap index must not wrap to a valid 32-bit layer index
    json d = base;
    d["taps"] = {(int64_t(1) << 32) + 1};
    check_refused("int64 tap overflow", d, false, "DFlash2 tap");
  }
  {  // HIGH: sel_rank > kMaxWidth
    json d = base;
    d["sel_rank"] = (1 << 20) + 1;
    check_refused("sel_rank > kMaxWidth", d, false, "invalid DFlash2");
  }
  {  // HIGH: 2 * layers * n_kv * head_dim overflows 32-bit int ctx_width()
    json d = base;
    d["layers"] = 1024;
    d["n_heads"] = 1024;
    d["n_kv"] = 1024;
    d["head_dim"] = 2048;
    check_refused("ctx_width int32 overflow", d, false, "invalid DFlash2");
  }
  {  // HIGH: TG2 and n_taps * hidden int32 overflows on a large-hidden synthetic model
    auto check_synth = [&](const std::string& what, json d) {
      const int H = 1 << 20;  // kMaxWidth
      d["hidden"] = H;
      d["target"] = {{"hidden", H}, {"layers", 1}, {"vocab", 64}};
      json hdr = {
          {"format", "flyx"},
          {"version", 1},
          {"arch", "qwen35"},
          {"config", {{"hidden", H}, {"layers", 1}, {"vocab", 64}, {"inter", 64}, {"n_heads", 2}, {"n_kv", 2},
                      {"head_dim", 16}, {"rot_dim", 16}, {"rope_theta", 1e4}, {"eps", 1e-6},
                      {"lin_nk", 2}, {"lin_nv", 2}, {"lin_dk", 16}, {"lin_dv", 16}, {"conv_k", 4},
                      {"layer_types", {"attn"}}}},
          {"draft", d},
          {"tensors", json::object()},
      };
      const std::string path = dir + "/bad_dflash2_synth.flyx";
      testing::write_file(path, testing::flyx_file(hdr.dump(), ""));
      std::string error;
      bool ok = false, opened_model = false;
      {
        ModelFile f;
        ok = f.open(path, &error);
        const auto m = ok ? interfaces::open_model(f, {}, &error) : nullptr;
        opened_model = m != nullptr;
      }
      std::remove(path.c_str());
      check(ok && !opened_model && error.find("invalid DFlash2") != std::string::npos,
            "dflash2 refuse " + what + " (" + (opened_model ? "OPENED!" : error) + ")");
    };
    {
      json d = base;
      d["taps"] = {1};
      d["conv_taps"] = 2048;
      d["conv_group"] = 1;  // 2 * 2048 * (1 << 20) = 2^32 > INT_MAX
      check_synth("TG2 int32 overflow", d);
    }
    {
      json d = base;
      d["taps"] = std::vector<int>(2049, 1);  // 2049 * (1 << 20) > INT_MAX
      d["conv_taps"] = 2;
      d["conv_group"] = 16;
      check_synth("n_taps * hidden int32 overflow", d);
    }
  }
}

// DFlash2 GGUF (arch="dflash") matches canonical DFlash2 .flyx block-for-block and attaches to non-Qwen3.5
// architectures (Qwen3 dense) through the shared models/drafters/ registry.
void dflash2_gguf_and_universal(const std::string& dir) {
  auto open_with = [&](ModelFile* f, const std::string& model, const std::string& comp, std::string* err) {
    if (!f->open(model, err) || !f->add(comp, err)) return std::unique_ptr<interfaces::Model>{};
    interfaces::ModelOptions o;
    o.drafter = "dflash2";
    return interfaces::open_model(*f, o, err);
  };
  std::string err;
  // 1. Qwen3.5 + tiny_draft.flyx vs Qwen3.5 + tiny_draft.gguf: identical blocks, weights, states, and programs.
  ModelFile fa, fb;
  const auto ma = open_with(&fa, dir + "/tiny_gguf.gguf", dir + "/tiny_draft.flyx", &err);
  const auto mb = open_with(&fb, dir + "/tiny_gguf.gguf", dir + "/tiny_draft.gguf", &err);
  check(ma && mb, "DFlash2 GGUF opens beside Qwen3.5 GGUF: " + err);
  if (ma && mb) {
    const auto& da = ma->desc();
    const auto& db = mb->desc();
    bool ok = da.n_blocks == db.n_blocks && da.n_states == db.n_states &&
              std::string(ma->drafter()) == std::string(mb->drafter());
    for (int i = 0; ok && i < da.n_blocks; ++i) {
      const auto& x = da.blocks[i];
      const auto& y = db.blocks[i];
      if (x.kind != y.kind) {
        ok = false;
        break;
      }
      Compare c{{&da}, {&db}, 1e-6, "dflash block " + std::to_string(i)};
      if (x.kind == FLYX_BLOCK_DFLASH2_LAYER) {
        const auto &a = x.dflash2_layer, &b = y.dflash2_layer;
        c.norm(a.norm, b.norm, size_t(da.hidden), "norm");
        c.norm(a.post_norm, b.post_norm, size_t(da.hidden), "post_norm");
        c.norm(a.q_norm, b.q_norm, size_t(da.weights[a.q_norm.w].t->shape[0]), "q_norm");
        c.norm(a.k_norm, b.k_norm, size_t(da.weights[a.k_norm.w].t->shape[0]), "k_norm");
        c.whole(a.q, b.q, "q");
        c.whole(a.k, b.k, "k");
        c.whole(a.v, b.v, "v");
        c.whole(a.o, b.o, "o");
        c.whole(a.gate, b.gate, "gate");
        c.whole(a.up, b.up, "up");
        c.whole(a.down, b.down, "down");
        c.whole(a.attn_conv_proj, b.attn_conv_proj, "attn_conv_proj");
        c.whole(a.mlp_conv_proj, b.mlp_conv_proj, "mlp_conv_proj");
        auto numel = [&](int w) {
          size_t n = 1;
          const auto* t = da.weights[w].t;
          for (int d = 0; d < t->ndim; ++d) n *= size_t(t->shape[d]);
          return n;
        };
        c.rows(a.attn_conv_base, 0, b.attn_conv_base, 0, numel(a.attn_conv_base), "attn_conv_base");
        c.rows(a.mlp_conv_base, 0, b.mlp_conv_base, 0, numel(a.mlp_conv_base), "mlp_conv_base");
      } else if (x.kind == FLYX_BLOCK_DFLASH2) {
        const auto &a = x.dflash2, &b = y.dflash2;
        c.same(a.hidden, b.hidden, "hidden");
        c.same(a.n_heads, b.n_heads, "n_heads");
        c.same(a.n_kv, b.n_kv, "n_kv");
        c.same(a.head_dim, b.head_dim, "head_dim");
        c.same(a.inter, b.inter, "inter");
        c.same(a.block, b.block, "block");
        c.same(a.mask_token, b.mask_token, "mask_token");
        c.same(a.window, b.window, "window");
        c.same(a.n_taps, b.n_taps, "n_taps");
        c.same(a.conv_taps, b.conv_taps, "conv_taps");
        c.same(a.conv_group, b.conv_group, "conv_group");
        c.same(a.sel_rank, b.sel_rank, "sel_rank");
        c.same(a.sel_topk, b.sel_topk, "sel_topk");
        c.same(a.theta, b.theta, "theta");
        c.whole(a.fc, b.fc, "fc");
        c.norm(a.hidden_norm, b.hidden_norm, size_t(a.hidden), "hidden_norm");
        c.norm(a.norm, b.norm, size_t(a.hidden), "norm");
        c.whole(a.sel_pred, b.sel_pred, "sel_pred");
        c.whole(a.sel_succ, b.sel_succ, "sel_succ");
        c.whole(a.sel_hid, b.sel_hid, "sel_hid");
      }
      ok = ok && c.ok;
    }
    auto prog_eq = [](const std::vector<flyx_node>& p, const std::vector<flyx_node>& q) {
      if (p.size() != q.size()) return false;
      for (size_t i = 0; i < p.size(); ++i)
        if (p[i].kind != q[i].kind || p[i].block != q[i].block || p[i].arg0 != q[i].arg0 || p[i].arg1 != q[i].arg1)
          return false;
      return true;
    };
    check(ok && prog_eq(ma->prefill_program(), mb->prefill_program()) &&
              prog_eq(ma->round_program(7, 0, false), mb->round_program(7, 0, false)),
          "DFlash2 GGUF matches DFlash2 .flyx block-for-block, weight-for-weight, and program-for-program");
  }

  // 2. Qwen3 (both GGUF and .flyx) attaches tiny_qwen3_draft.gguf through the shared drafter registry.
  for (const std::string& q3 : {dir + "/tiny_qwen3_gguf.gguf", dir + "/tiny_qwen3.flyx"}) {
    err.clear();
    ModelFile fq;
    const auto mq = open_with(&fq, q3, dir + "/tiny_qwen3_draft.gguf", &err);
    int taps = 0, ctx = 0;
    if (mq) {
      for (const flyx_node& n : mq->prefill_program()) {
        taps += n.kind == FLYX_NODE_TAP;
        ctx += n.kind == FLYX_NODE_CONTEXT;
      }
    }
    check(mq && mq->drafters() == 1 && std::string(mq->drafter()) == "dflash2" && taps == 2 && ctx == 1,
          "Qwen3 attaches DFlash2 GGUF (" + q3 + "): " + err);
  }

  // 3. Refusal checks on DFlash2 GGUF: mismatched hidden, causal=true.
  err.clear();
  {
    ModelFile fm;
    const auto mismatch = open_with(&fm, dir + "/tiny_w4a4.flyx", dir + "/tiny_draft.gguf", &err);
    check(!mismatch && err.find("another model") != std::string::npos,
          "DFlash2 GGUF for another hidden size is refused (" + err + ")");
  }
  {
    // Flip dflash.attention.causal from false to true in a temp copy.
    std::ifstream in(dir + "/tiny_draft.gguf", std::ios::binary);
    std::vector<char> data{std::istreambuf_iterator<char>(in), {}};
    const std::string key = "dflash.attention.causal";
    const auto at = std::search(data.begin(), data.end(), key.begin(), key.end());
    if (at != data.end()) {
      at[std::ptrdiff_t(key.size()) + 4] = 1;  // skip uint32 value type (7 = bool), set bool = true
      const std::string bad = dir + "/bad_causal_dflash.gguf";
      std::ofstream(bad, std::ios::binary).write(data.data(), std::streamsize(data.size()));
      err.clear();
      bool refused_causal = false;
      {
        ModelFile fc;
        const auto mc = open_with(&fc, dir + "/tiny_gguf.gguf", bad, &err);
        refused_causal = !mc && err.find("causal") != std::string::npos;
      }
      std::remove(bad.c_str());
      check(refused_causal, "DFlash2 GGUF with causal=true is refused (" + err + ")");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string dir = argv[1];
  fp4_inputs_by_device(dir + "/tiny_w4a4.flyx");
  activations_asked(dir);
  dflash2_validation(dir);
  dflash2_gguf_and_universal(dir);
  same_model(dir + "/tiny.flyx", dir + "/tiny_gguf.gguf");
  same_model(dir + "/tiny.flyx", dir + "/tiny_bf16.gguf");
  same_model(dir + "/tiny_explicit_hd.flyx", dir + "/tiny_explicit_hd.gguf");
  same_model(dir + "/tiny_qwen3.flyx", dir + "/tiny_qwen3_gguf.gguf");
  same_model(dir + "/tiny_qwen3.flyx", dir + "/tiny_qwen3_f16.gguf", 1e-3);
  const std::string q3 = dir + "/tiny_qwen3_gguf.gguf";
  // A metadata value follows its key and a uint32 type; a string value, a uint64 length more.  The fixture's
  // architecture, qwen3, has five letters.
  auto arch = [](const char* five) { return [five](char* p) { std::memcpy(p + 12, five, 5); }; };
  refused(q3, dir, "an architecture without a model", "general.architecture", arch("bloom"),
          "unsupported architecture 'bloom'");
  // Llama is a model this build reads from .flyx only: llama.cpp's converter permutes its q and k rows.
  refused(q3, dir, "a Llama GGUF file", "general.architecture", arch("llama"), "a Llama GGUF file");
  // ... so a Hub repository of Llama GGUFs is not said to run here (the dashboard's library, before a fetch).
  const std::vector<std::string> archs = flyx::interfaces::model_architectures();
  const auto lists = [&](const char* name) { return std::find(archs.begin(), archs.end(), name) != archs.end(); };
  check(lists("qwen35") && lists("qwen3") && !lists("llama"), "the architectures whose GGUFs open: qwen35, qwen3");
  refused(q3, dir, "a zero hidden size", "qwen3.embedding_length", [](char* p) { std::memset(p + 4, 0, 4); });
  refused(q3, dir, "a missing tensor", "blk.0.attn_k", [](char* p) { p[-1] = 'x'; });
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
