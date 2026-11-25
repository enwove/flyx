// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The DFlash2 drafter the architectures share (models/drafters/), without a backend:
//  * its config is checked before any tensor is bound: a width that is not the target's (whose embedding and head
//    it reads), dimensions whose products overflow an int;
//  * the dense targets take one: the tiny Qwen3 (GGUF and .flyx) and the tiny Llama (plain and with a V-LoRA) with a
//    drafter in llama.cpp's GGUF layout (arch "dflash", make_gguf.py: tiny_qwen3_draft.gguf, tiny_llama_draft.gguf):
//    one drafter and its block, a tap after each layer it reads and its context node in the prefill, in its round
//    and in a plain round;
//  * a drafter made for another model is refused whether or not it is chosen; what a drafter's own config says is
//    read when it drafts, and a GGUF drafter saying what the block here does not run is refused then
//    (tiny_draft_refused_*.gguf), with the file named.
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
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

// The model of `model` with `drafter_file` beside it, the drafter asked for by name ("none": no drafter).
std::unique_ptr<interfaces::Model> open_beside(ModelFile* f, const std::string& model, const std::string& drafter_file,
                                               const std::string& drafter, std::string* err) {
  err->clear();
  if (!f->open(model, err) || !f->add(drafter_file, err)) return nullptr;
  interfaces::ModelOptions o;
  o.drafter = drafter;
  return interfaces::open_model(*f, o, err);
}

// The fixtures' drafters read the target's first layer and the one before its last: a tap after each, wherever the
// target's layers run, and the drafter's context taken once a pass.
void dense_targets(const std::string& dir) {
  struct Case {
    const char* model;
    const char* drafter;
  };
  for (const Case& c : {Case{"tiny_qwen3_gguf.gguf", "tiny_qwen3_draft.gguf"},
                        Case{"tiny_qwen3.flyx", "tiny_qwen3_draft.gguf"},
                        Case{"tiny_llama.flyx", "tiny_llama_draft.gguf"},
                        Case{"tiny_llama_lora.flyx", "tiny_llama_draft.gguf"}}) {
    ModelFile f;
    std::string err;
    const auto m = open_beside(&f, dir + "/" + c.model, dir + "/" + c.drafter, "dflash2", &err);
    bool ok = m && m->drafters() == 1 && std::string(m->drafter()) == "dflash2";
    if (ok) {
      const flyx_model_desc& d = m->desc();
      int block = -1, layers = 0;
      for (int i = 0; i < d.n_blocks; ++i) {
        if (d.blocks[i].kind == FLYX_BLOCK_DFLASH2) block = i;
        layers += d.blocks[i].kind == FLYX_BLOCK_ATTN;
      }
      const int want[2] = {1, layers - 1};
      // Slot s is tapped after the attention and MLP of want[s] layers (and the embedding before them); the
      // drafter's own nodes otherwise as many as `drafts` and `contexts`.
      auto follows = [&](const std::vector<flyx_node>& p, int drafts, int contexts) {
        int runs = 0, taps = 0, n_drafts = 0, n_contexts = 0;
        bool placed = true;
        for (const flyx_node& x : p) {
          runs += x.kind == FLYX_NODE_RUN;
          if (x.block != block) continue;
          n_drafts += x.kind == FLYX_NODE_DRAFT;
          n_contexts += x.kind == FLYX_NODE_CONTEXT;
          if (x.kind != FLYX_NODE_TAP) continue;
          placed = placed && x.arg0 == taps && taps < 2 && runs == 1 + 2 * want[taps];
          ++taps;
        }
        return placed && taps == 2 && n_drafts == drafts && n_contexts == contexts;
      };
      const auto round = m->round_program(7, 0, false);
      ok = block >= 0 && d.blocks[block].dflash2.n_taps == 2 && follows(m->prefill_program(), 0, 1) &&
           follows(round, 1, 1) && round.back().kind == FLYX_NODE_CONTEXT &&
           follows(m->round_program(0, -1, false), 0, 1);
    }
    check(ok, std::string(c.model) + " takes the DFlash2 drafter of " + c.drafter +
                  ": its block, its two taps and its context in the prefill, its round and a plain round" +
                  (err.empty() ? "" : " (" + err + ")"));
  }
}

void refusals(const std::string& dir) {
  const std::string model = dir + "/tiny_gguf.gguf";
  struct Case {
    const char* file;
    const char* says;
    const char* what;
  };
  for (const Case& c : {Case{"tiny_draft_refused_causal.gguf", "causal", "causal layers"},
                        Case{"tiny_draft_refused_dspark.gguf", "DSpark", "a Markov context (DSpark)"},
                        Case{"tiny_draft_refused_anchor.gguf", "sample_from_anchor", "sampling from the anchor"},
                        Case{"tiny_draft_refused_confidence.gguf", "confidence head", "a confidence head"},
                        Case{"tiny_draft_refused_softcap.gguf", "soft-capping", "soft-capped logits"},
                        Case{"tiny_draft_refused_last_layer.gguf", "final hidden state",
                             "a tap on the target's final hidden state"},
                        Case{"tiny_draft_refused_pattern.gguf", "must be uniform", "layers that do not all slide"}}) {
    ModelFile f;
    std::string err;
    const auto m = open_beside(&f, model, dir + "/" + c.file, "dflash2", &err);
    check(!m && err.find(c.says) != std::string::npos && err.find("companion file 1: ") != std::string::npos,
          std::string("a DFlash2 GGUF with ") + c.what + ": refused, the file named (" + err + ")");
  }
  {  // What a drafter's config says is read when it drafts: not chosen, the model opens without it.
    ModelFile f;
    std::string err;
    const auto m = open_beside(&f, model, dir + "/tiny_draft_refused_causal.gguf", "none", &err);
    check(m && m->drafters() == 0,
          "a drafter the block does not run, not chosen: the model opens without it (" + err + ")");
  }
  // Made for another model (tiny_w4a4.flyx is twice as wide): refused as the file is taken, chosen or not; so is the
  // tiny Qwen3.5's drafter file beside the tiny Qwen3 (as wide, half the layers), which names its target's layers.
  for (const char* drafter : {"dflash2", "none"}) {
    for (const char* file : {"tiny_draft.gguf", "tiny_draft.flyx"}) {
      ModelFile f;
      std::string err;
      const auto m = open_beside(&f, dir + "/tiny_w4a4.flyx", dir + "/" + file, drafter, &err);
      check(!m && err.find("another model") != std::string::npos,
            std::string(file) + " beside a model twice as wide, drafter " + drafter + ": refused (" + err + ")");
    }
    ModelFile f;
    std::string err;
    const auto m = open_beside(&f, dir + "/tiny_qwen3.flyx", dir + "/tiny_draft.flyx", drafter, &err);
    check(!m && err.find("another model") != std::string::npos,
          std::string("a drafter file made for a model of other layers, drafter ") + drafter + ": refused (" + err +
              ")");
  }
  {  // A GGUF drafter names no target: one whose taps lie past the model's layers is refused when it drafts.
    ModelFile f;
    std::string err;
    const auto m = open_beside(&f, dir + "/tiny_qwen3.flyx", dir + "/tiny_draft.gguf", "dflash2", &err);
    check(!m && err.find("DFlash2 tap after layer") != std::string::npos,
          "a GGUF drafter tapping layers the model does not have: refused (" + err + ")");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string dir = argv[1];
  dflash2_validation(dir);
  dense_targets(dir);
  refusals(dir);
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
