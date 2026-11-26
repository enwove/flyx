// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Llama description's own adapters (models/llama, ABI 43: a V-LoRA) from the tiny Llama's V-LoRA files
// (make_tiny.py: tiny_llama_lora.flyx, rank 16 and scale 0.5 on tiny_llama.flyx's weights; tiny_llama_dv_lora.flyx,
// tiny_llama_dv.flyx's drafter beside a V-LoRA of rank 32 and scale 1): on every attention block its q, k, v and o
// adapters and on every MLP block its gate, up and down ones, each weight the file's tensor of that name and shape,
// the header's rank and scale; the target otherwise tiny_llama.flyx's, block for block; DV's canvas layers keep their
// own adapters (scale 1) and run the target's blocks; the same with --drafter none.  Headers and tensors that do not
// fit refused with a message: a section without its tensors, of another rank, a scale of 0 or not finite, adapters
// without the section.  The files are .flyx format 2, requiring their adapters ("requires": ["lora"]), which a build
// of format 1 only refuses instead of decoding the plain model; adapters the file does not require, a part this
// build's model does not read and a Qwen3.5 file requiring adapters are refused.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "tests/common/check.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;
using nlohmann::json;

namespace {

std::unique_ptr<interfaces::Model> open(const ModelFile& f, const std::string& drafter, std::string* error) {
  interfaces::ModelOptions o;
  o.drafter = drafter;
  return interfaces::open_model(f, o, error);
}

const char* name_of(const flyx_model_desc& d, int w) { return w >= 0 && w < d.n_weights ? d.weights[w].t->name : ""; }

bool is(const flyx_model_desc& d, int w, const std::string& name, std::vector<int64_t> shape) {
  if (name_of(d, w) != name) return std::printf("  weight %d is %s, not %s\n", w, name_of(d, w), name.c_str()), false;
  const flyx_tensor_view* t = d.weights[w].t;
  bool same = t->ndim == int(shape.size()) && t->dtype == FLYX_DT_BF16;
  for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == shape[i];
  return same;
}

// The tiny Llama's dimensions (make_tiny.py make_llama).
constexpr int H = 128, Q = 16 * 128, KV = 2 * 128, I = 256, LAYERS = 4;

// Weight w of d as the weight of plain t that has its name (-1: none, as an absent norm's).
int as_plain(const flyx_model_desc& d, const flyx_model_desc& t, int w) {
  if (w < 0) return w;
  for (int i = 0; i < t.n_weights; ++i)
    if (std::strcmp(name_of(d, w), name_of(t, i)) == 0) return i;
  return -2;
}

// Every attention and MLP block of m has its adapters of rank r and scale s, each the file's tensor of its linear;
// the blocks are plain's otherwise (their adapters taken out, their weights named as plain's), and every weight of
// plain is one of m's.
void adapters(const interfaces::Model& m, const interfaces::Model& plain, int r, float s, const std::string& what) {
  const flyx_model_desc& d = m.desc();
  const flyx_model_desc& t = plain.desc();
  int attn = 0, mlp = 0;
  int lora_weights = 0;
  for (int i = 0; i < d.n_weights; ++i)
    lora_weights += std::strncmp(name_of(d, i), "target.", 7) == 0 && std::strstr(name_of(d, i), ".lora_") != nullptr;
  bool ok = d.n_blocks >= t.n_blocks && lora_weights == LAYERS * 14;  // 7 linears a layer, a and b each
  for (int i = 0; ok && i < t.n_blocks; ++i) {
    flyx_block b = d.blocks[i];
    auto plain_w = [&](int& w) { w = as_plain(d, t, w); };
    if (b.kind == FLYX_BLOCK_ATTN) {
      const std::string p = "target.L" + std::to_string(attn++) + ".attn.";
      const char* lin[4] = {"q", "k", "v", "o"};
      const int in[4] = {H, H, H, Q}, out[4] = {Q, KV, KV, H};
      for (int j = 0; ok && j < 4; ++j)
        ok = b.attn.lora[j].rank == r && b.attn.lora[j].scale == s &&
             is(d, b.attn.lora[j].a, p + lin[j] + ".lora_a", {r, in[j]}) &&
             is(d, b.attn.lora[j].b, p + lin[j] + ".lora_b", {out[j], r});
      std::memset(b.attn.lora, 0, sizeof(b.attn.lora));
      for (int* w : {&b.attn.norm.w, &b.attn.q_norm.w, &b.attn.k_norm.w, &b.attn.q, &b.attn.k, &b.attn.v, &b.attn.o})
        plain_w(*w);
    } else if (b.kind == FLYX_BLOCK_MLP) {
      const std::string p = "target.L" + std::to_string(mlp++) + ".mlp.";
      const char* lin[3] = {"gate", "up", "down"};
      const int in[3] = {H, H, I}, out[3] = {I, I, H};
      for (int j = 0; ok && j < 3; ++j)
        ok = b.mlp.lora[j].rank == r && b.mlp.lora[j].scale == s &&
             is(d, b.mlp.lora[j].a, p + lin[j] + ".lora_a", {r, in[j]}) &&
             is(d, b.mlp.lora[j].b, p + lin[j] + ".lora_b", {out[j], r});
      std::memset(b.mlp.lora, 0, sizeof(b.mlp.lora));
      for (int* w : {&b.mlp.norm.w, &b.mlp.gate, &b.mlp.up, &b.mlp.down}) plain_w(*w);
    } else if (b.kind == FLYX_BLOCK_EMBED) {
      plain_w(b.embed.table);
    } else if (b.kind == FLYX_BLOCK_HEAD) {
      plain_w(b.head.norm.w);
      plain_w(b.head.w);
    }
    ok = ok && std::memcmp(&b, &t.blocks[i], sizeof(flyx_block)) == 0;
  }
  for (int i = 0; ok && i < t.n_weights; ++i) {
    bool found = false;
    for (int j = 0; !found && j < d.n_weights; ++j) found = std::strcmp(name_of(d, j), name_of(t, i)) == 0;
    ok = found;
  }
  check(ok && attn == LAYERS && mlp == LAYERS,
        what + ": every attention and MLP block has its adapters (rank " + std::to_string(r) + ", scale " +
            std::to_string(s) + "), the blocks otherwise tiny_llama.flyx's");
}

// The format a .flyx file's first bytes give (bytes 8 to 11).
uint32_t format_of(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  char b[12] = {};
  uint32_t v = 0;
  if (in.read(b, sizeof b)) std::memcpy(&v, b + 8, 4);
  return v;
}

// A copy of the file with its header changed (dir/lora_changed.flyx; format 2 when the header has "requires"),
// opened; "" when it opened.  The model is closed before the next copy is written over the file.
std::string with_header(const std::string& path, const std::string& dir, const std::function<void(json&)>& change) {
  std::ifstream in(path, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(in), {}};
  uint64_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 16, 8);
  json h = json::parse(bytes.substr(24, size_t(hlen)));
  change(h);
  const size_t data = (24 + size_t(hlen) + 4095) / 4096 * 4096;
  const std::string out = dir + "/lora_changed.flyx";
  const std::string header = h.dump();
  if (!testing::write_file(out, testing::flyx_file(header, bytes.substr(data), header.size(),
                                                   h.contains("requires") ? 2 : 1)))
    return "cannot write " + out;
  ModelFile f;
  std::string error;
  if (!f.open(out, &error)) return "open: " + error;
  return open(f, "", &error) ? "" : error;
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 2) {
    std::fprintf(stderr, "usage: llama_models_test ASSET_DIR [nodv]\n");
    return 2;
  }
  const std::string dir = argv[1], path = dir + "/tiny_llama_lora.flyx";
  // nodv: without tiny_llama_dv_lora.flyx, one of DV's assets (make_tiny.py --group dv: FLYX_TEST_DV's tests alone)
  const bool with_dv = !(argc > 2 && std::string(argv[2]) == "nodv");
  ModelFile f, fd, fp;
  std::string error;
  if (!f.open(path, &error) || (with_dv && !fd.open(dir + "/tiny_llama_dv_lora.flyx", &error)) ||
      !fp.open(dir + "/tiny_llama.flyx", &error)) {
    std::printf("FAIL open: %s\n", error.c_str());
    return 1;
  }
  const auto plain = open(fp, "", &error);
  const auto m = open(f, "", &error);
  check(plain && m, "tiny_llama.flyx and tiny_llama_lora.flyx open: " + error);
  if (!plain || !m) return 1;
  check(m->drafters() == 0, "a V-LoRA alone: no drafter");
  for (const std::string name : {"tiny_llama_lora.flyx", "tiny_llama_dv_lora.flyx"}) {
    if (!with_dv && name != "tiny_llama_lora.flyx") continue;
    check(format_of(dir + "/" + name) == 2 && (name == "tiny_llama_lora.flyx" ? f : fd).header().at("requires") ==
                                                  json::array({"lora"}),
          name + ": .flyx format 2, requiring its adapters (a build of format 1 only refuses it)");
  }
  check(format_of(dir + "/tiny_llama.flyx") == 1 && !fp.header().contains("requires"),
        "tiny_llama.flyx: format 1, nothing required (every build reads it)");
  adapters(*m, *plain, 16, 0.5f, "tiny_llama_lora.flyx");
  {
    bool none = true;
    for (int i = 0; i < plain->desc().n_blocks; ++i) {
      const flyx_block& b = plain->desc().blocks[i];
      if (b.kind == FLYX_BLOCK_ATTN) none = none && b.attn.lora[0].rank == 0 && b.attn.lora[3].rank == 0;
      if (b.kind == FLYX_BLOCK_MLP) none = none && b.mlp.lora[0].rank == 0 && b.mlp.lora[2].rank == 0;
    }
    check(none, "tiny_llama.flyx: no adapters (rank 0, as its blocks are built zeroed)");
  }
  if (with_dv) {
    const auto dv = open(fd, "dv", &error);
    const auto dv_none = open(fd, "none", &error);
    check(dv && dv_none && dv->drafters() == 1 && std::string(dv->drafter()) == "dv" && dv_none->drafters() == 0,
          "tiny_llama_dv_lora.flyx: DV with --drafter dv, none with --drafter none (" + error + ")");
    if (dv && dv_none) {
      adapters(*dv, *plain, 32, 1.f, "tiny_llama_dv_lora.flyx, DV");
      adapters(*dv_none, *plain, 32, 1.f, "tiny_llama_dv_lora.flyx, --drafter none");
      const flyx_model_desc& d = dv->desc();
      const flyx_dv_block& v = d.blocks[d.n_blocks - 1].dv;
      bool canvas = d.blocks[d.n_blocks - 1].kind == FLYX_BLOCK_DV && v.layers == LAYERS;
      for (int j = 0; canvas && j < v.layers; ++j) {
        const flyx_dv_layer_block& l = d.blocks[v.first_layer + j].dv_layer;
        canvas = d.blocks[l.attn].kind == FLYX_BLOCK_ATTN && d.blocks[l.mlp].kind == FLYX_BLOCK_MLP &&
                 std::string(name_of(d, l.lora[FLYX_LIN_Q].a)) == "dv.L" + std::to_string(j) + ".q.lora_a" &&
                 l.lora[FLYX_LIN_DOWN].scale == 1.f && l.lora[FLYX_LIN_Q].rank == 32;
      }
      check(canvas, "DV's canvas layers: the target's blocks (whose adapters the canvas never runs) and their own "
                    "adapters, scale 1");
    }
  }

  struct Bad {
    const char* what;
    const char* because;
    std::function<void(json&)> change;
  };
  const std::vector<Bad> bad = {
      {"rank", "target.L0.attn.q.lora_a: shape 16x128, the config says 8x128",
       [](json& h) { h["lora"]["rank"] = 8; }},
      {"rank 0", "a rank of 1 or more", [](json& h) { h["lora"]["rank"] = 0; }},
      {"rank past the bound", "past any model's (at most 16384)", [](json& h) { h["lora"]["rank"] = 1 << 20; }},
      {"scale 0", "a finite scale other than 0", [](json& h) { h["lora"]["scale"] = 0.0; }},
      {"scale left out", "'scale' not found", [](json& h) { h["lora"].erase("scale"); }},
      {"missing tensor", "missing tensor target.L3.mlp.down.lora_b",
       [](json& h) { h["tensors"].erase("target.L3.mlp.down.lora_b"); }},
      {"adapters without the section", "adapters without the header's lora section", [](json& h) { h.erase("lora"); }},
      {"section the file does not require (format 1)", "adapters the file does not require",
       [](json& h) { h.erase("requires"); }},
      {"part this build does not read", "requires its part \"later\", which this build's llama model does not read",
       [](json& h) { h["requires"].push_back("later"); }},
  };
  for (const Bad& b : bad) {
    const std::string e = with_header(path, dir, b.change);
    check(e.find(b.because) != std::string::npos, std::string("a V-LoRA with a bad ") + b.what + ": refused (" + e +
                                                       ")");
  }
  // Adapters without the header's lora section on any one of a layer's seven linears, the others' gone (each linear's
  // lora_b alone, and one lora_a alone): refused, never decoded as a file without adapters.
  for (const std::string kept : {"attn.q.lora_b", "attn.k.lora_b", "attn.v.lora_b", "attn.o.lora_b", "mlp.gate.lora_b",
                                 "mlp.up.lora_b", "mlp.down.lora_b", "attn.k.lora_a"}) {
    const std::string name = "target.L1." + kept;
    const std::string e = with_header(path, dir, [&](json& h) {
      h.erase("lora");
      json& t = h["tensors"];
      for (auto it = t.begin(); it != t.end();)
        it = it.key().find(".lora_") != std::string::npos && it.key() != name ? t.erase(it) : std::next(it);
    });
    const std::string linear = name.substr(0, name.rfind('.'));
    check(e.find(linear + ": adapters without the header's lora section") != std::string::npos,
          "adapters without the lora section, " + name + " alone: refused (" + e + ")");
  }
  {
    const std::string nan = with_header(path, dir, [](json& h) { h["lora"]["scale"] = std::nan(""); });
    check(!nan.empty(), "a V-LoRA whose scale is not a number: refused (" + nan + ")");
  }
  check(with_header(path, dir, [](json&) {}).empty(), "the same header rewritten: opens");
  {
    const std::string e = with_header(dir + "/tiny.flyx", dir, [](json& h) { h["requires"] = {"lora"}; });
    check(e.find("this build's qwen35 model does not read") != std::string::npos,
          "a Qwen3.5 file requiring adapters, which its model does not read: refused (" + e + ")");
  }
  return flyx::test::finish("llama_models_test");
}
