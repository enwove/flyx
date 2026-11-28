// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A LoRA adapter read from the directory Hugging Face PEFT saves (adapter_config.json, adapter_model.safetensors) as
// a model's companion: its description (rank, scale with and without rsLoRA, layers, linears), its tensors under a
// .flyx file's adapter names with their bytes in place, Qwen3's and Qwen3.5/3.8's (language_model, linear_attn)
// names; and what the engine cannot run refused by name.  The directories are written here, one per case.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "core/modelfile/readers.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Tensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
};

size_t elem_bytes(const std::string& dtype) { return dtype == "F32" ? 4 : 2; }

// A safetensors header of `tensors` (their bytes one after the other) and those bytes: each tensor's its index
// repeated, so a view can be checked against them.
std::string header_of(const std::vector<Tensor>& tensors, std::vector<uint8_t>* data = nullptr) {
  json h = json::object();
  size_t at = 0;
  for (size_t i = 0; i < tensors.size(); ++i) {
    size_t n = elem_bytes(tensors[i].dtype);
    for (int64_t d : tensors[i].shape) n *= size_t(d);
    h[tensors[i].name] = {{"dtype", tensors[i].dtype}, {"shape", tensors[i].shape}, {"data_offsets", {at, at + n}}};
    if (data) data->insert(data->end(), n, uint8_t(i + 1));
    at += n;
  }
  h["__metadata__"] = {{"format", "pt"}};
  return h.dump();
}

// Writes an adapter directory (header_of's tensors); `pad`: that many spaces after the header's JSON (in its length).
fs::path write_adapter(const fs::path& root, const std::string& name, const json& config,
                       const std::vector<Tensor>& tensors, bool weights = true, size_t pad = 0) {
  const fs::path dir = root / name;
  fs::create_directories(dir);
  std::ofstream(dir / "adapter_config.json") << config.dump();
  if (!weights) return dir;
  std::vector<uint8_t> data;
  const std::string hs = header_of(tensors, &data) + std::string(pad, ' ');
  const uint64_t hlen = hs.size();
  std::ofstream f(dir / "adapter_model.safetensors", std::ios::binary);
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f.write(hs.data(), std::streamsize(hs.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
  return dir;
}

json config(int r, double alpha, json extra = json::object()) {
  json c = {{"peft_type", "LORA"}, {"r", r}, {"lora_alpha", alpha}, {"base_model_name_or_path", "Qwen/Qwen3.8-27B"}};
  c.update(extra);
  return c;
}

std::vector<Tensor> pair(const std::string& prefix, int r, int in, int out, const std::string& dtype = "F32") {
  return {{prefix + ".lora_A.weight", dtype, {r, in}}, {prefix + ".lora_B.weight", dtype, {out, r}}};
}

// Adds `dir` to a fresh ModelFile; the error when it is refused.
bool add(const fs::path& dir, flyx::ModelFile* f, std::string* err) { return f->add(dir.string(), err); }

void refused(const fs::path& root, const std::string& name, const json& cfg, const std::vector<Tensor>& t,
             const std::string& says, bool weights = true) {
  flyx::ModelFile f;
  std::string err;
  const bool ok = add(write_adapter(root, name, cfg, t, weights), &f, &err);
  check(!ok && err.find(says) != std::string::npos, name + " refused: " + (ok ? "(read)" : err));
}

// The same tensors as a listing of a repository's adapters reads them (their safetensors header alone,
// peft_header_ok), refused for the same reason, and as the load refuses them (refused).
void refused_both(const fs::path& root, const std::string& name, int rank, const std::vector<Tensor>& t,
                  const std::string& says) {
  std::string why;
  const bool ok = flyx::modelfile::peft_header_ok(header_of(t), rank, &why);
  check(!ok && why.find(says) != std::string::npos, name + " refused by the listing: " + (ok ? "(listed)" : why));
  refused(root, name, config(rank, 16), t, says);
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path root = fs::temp_directory_path() / ("flyx_peft_dir_test_" + std::to_string(stamp));
  fs::remove_all(root);

  for (const json& fields : {json{{"lora_alpha", "32"}}, json{{"use_rslora", 1}}, json{{"use_rslora", "false"}},
                             json{{"use_rslora", json::object()}}, json{{"lora_alpha", 0}},
                             json{{"lora_alpha", 1e-100}}, json{{"lora_alpha", 1e100}}}) {
    const auto cfg = config(2, 16, fields);
    const auto key = fields.begin().key();
    flyx::modelfile::PeftConfig listed;
    std::string why;
    check(!flyx::modelfile::peft_config_ok(cfg.dump(), &listed, &why) && why.find(key) != std::string::npos,
          "listing refuses invalid " + key);
    refused(root, key, cfg, pair("model.layers.0.self_attn.q_proj", 2, 4, 4), key);
  }

  {  // Qwen3 (Llama naming): attention and MLP, rank 16, alpha 32, F32.
    std::vector<Tensor> t;
    for (int l = 0; l < 2; ++l) {
      const std::string p = "base_model.model.model.layers." + std::to_string(l);
      for (auto [m, in, out] : std::vector<std::tuple<std::string, int, int>>{
               {"self_attn.q_proj", 64, 128}, {"self_attn.k_proj", 64, 32}, {"self_attn.v_proj", 64, 32},
               {"self_attn.o_proj", 128, 64}, {"mlp.gate_proj", 64, 96}, {"mlp.up_proj", 64, 96},
               {"mlp.down_proj", 96, 64}})
        for (const Tensor& x : pair(p + "." + m, 16, in, out)) t.push_back(x);
    }
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "qwen3", config(16, 32, {{"use_rslora", nullptr}}), t), &f, &err);
    check(ok, "qwen3 adapter read" + (ok ? std::string() : ": " + err));
    if (ok) {
      const json& h = f.header()["companions"][0];
      const json& l = h["lora"];
      check(h["kind"] == "lora" && l["rank"] == 16 && l["rslora"] == false &&
                std::abs(l["scale"].get<double>() - 2.0) < 1e-12 && l["layers"] == 2 && l["layout"] == "hf" &&
                h["source"]["base_model"] == "Qwen/Qwen3.8-27B",
            "qwen3: kind lora, rank 16, scale alpha / r = 2, 2 layers, hf layout, its base model");
      check(l["linears"] == json({"attn.k", "attn.o", "attn.q", "attn.v", "mlp.down", "mlp.gate", "mlp.up"}),
            "qwen3: the seven linears");
      const flyx_tensor_view* a = f.tensor(flyx::ModelFile::companion_name(0, "target.L1.attn.q.lora_a"));
      const flyx_tensor_view* b = f.tensor(flyx::ModelFile::companion_name(0, "target.L1.mlp.down.lora_b"));
      // L1.q's A is tensor 15 (2 per linear, 7 linears per layer), L1.down's B the last (28).
      check(a && a->dtype == FLYX_DT_F32 && a->ndim == 2 && a->shape[0] == 16 && a->shape[1] == 64 &&
                a->nbytes == 16 * 64 * 4 && reinterpret_cast<const uint8_t*>(a->data)[0] == 15 &&
                std::string(a->name) == "target.L1.attn.q.lora_a",
            "qwen3: L1.attn.q.lora_a, F32 [16, 64], its bytes in place");
      check(b && b->shape[0] == 64 && b->shape[1] == 16 && reinterpret_cast<const uint8_t*>(b->data)[0] == 28,
            "qwen3: L1.mlp.down.lora_b [64, 16]");
    }
  }

  {  // Qwen3.8 (language_model, linear_attn), only some linears, BF16, rsLoRA.
    std::vector<Tensor> t;
    const std::string p = "base_model.model.model.language_model.layers.";
    for (const Tensor& x : pair(p + "0.linear_attn.in_proj_qkv", 8, 64, 192, "BF16")) t.push_back(x);
    for (const Tensor& x : pair(p + "0.linear_attn.out_proj", 8, 64, 64, "BF16")) t.push_back(x);
    for (const Tensor& x : pair(p + "3.self_attn.q_proj", 8, 64, 256, "BF16")) t.push_back(x);  // q with its gate
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "qwen38", config(8, 16, {{"use_rslora", true}}), t), &f, &err);
    check(ok, "qwen3.8 adapter read" + (ok ? std::string() : ": " + err));
    if (ok) {
      const json& l = f.header()["companions"][0]["lora"];
      check(l["rslora"] == true && std::abs(l["scale"].get<double>() - 16.0 / std::sqrt(8.0)) < 1e-12,
            "qwen3.8: rsLoRA's scale alpha / sqrt(r)");
      check(l["layers"] == 4 && l["linears"] == json({"attn.q", "gdn.out", "gdn.qkv"}),
            "qwen3.8: 4 layers (the last adapted one), three linears");
      const flyx_tensor_view* q = f.tensor(flyx::ModelFile::companion_name(0, "target.L3.attn.q.lora_b"));
      const flyx_tensor_view* g = f.tensor(flyx::ModelFile::companion_name(0, "target.L0.gdn.qkv.lora_a"));
      check(q && q->dtype == FLYX_DT_BF16 && q->shape[0] == 256 && g && g->shape[1] == 64,
            "qwen3.8: L3.attn.q.lora_b BF16 [256, 8] (q and gate rows), L0.gdn.qkv.lora_a");
      check(!f.tensor(flyx::ModelFile::companion_name(0, "target.L1.gdn.qkv.lora_a")), "qwen3.8: no adapter on L1");
    }
  }

  {  // The names some SVD-extraction tools write: <linear>.weight.lora_A.default, no base_model.model prefix.
    std::vector<Tensor> t = {
        {"model.language_model.layers.2.linear_attn.in_proj_a.weight.lora_A.default", "BF16", {8, 64}},
        {"model.language_model.layers.2.linear_attn.in_proj_a.weight.lora_B.default", "BF16", {4, 8}}};
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "svd_names", config(8, 8), t), &f, &err);
    const flyx_tensor_view* b = ok ? f.tensor(flyx::ModelFile::companion_name(0, "target.L2.gdn.a.lora_b")) : nullptr;
    check(ok && b && b->shape[0] == 4 && f.header()["companions"][0]["lora"]["linears"] == json({"gdn.a"}),
          "svd-extraction names: L2.gdn.a's lora_a and lora_b" + (ok ? std::string() : ": " + err));
  }

  {  // PEFT's default lora_alpha (8) when the config leaves it out.
    json c = config(4, 0);
    c.erase("lora_alpha");
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "alpha", c, pair("base_model.model.model.layers.0.mlp.up_proj", 4, 8, 8)),
                        &f, &err);
    check(ok && f.header()["companions"][0]["lora"]["scale"] == 2.0, "no lora_alpha: PEFT's 8, scale 8 / 4");
  }

  const std::string L0 = "base_model.model.model.layers.0.";
  const auto q = pair(L0 + "self_attn.q_proj", 8, 16, 16);
  refused(root, "dora", config(8, 16, {{"use_dora", true}}), q, "DoRA");
  refused(root, "rank_pattern", config(8, 16, {{"rank_pattern", {{"q_proj", 4}}}}), q, "per module");
  refused(root, "modules_to_save", config(8, 16, {{"modules_to_save", {"lm_head"}}}), q, "saved whole");
  refused(root, "not_lora", config(8, 16, {{"peft_type", "IA3"}}), q, "LORA is supported");
  refused(root, "bad_rank", config(0, 16), q, "rank of 1 to 4096");
  refused(root, "lm_head", config(8, 16), pair("base_model.model.lm_head", 8, 16, 32), "the head's adapter");
  refused(root, "embedding", config(8, 16),
          {{"base_model.model.model.embed_tokens.lora_embedding_A", "F32", {8, 32}}}, "embedding's adapter");
  refused(root, "vision", config(8, 16), pair("base_model.model.model.visual.blocks.0.attn.qkv", 8, 16, 48),
          "vision tower");
  refused(root, "unknown_linear", config(8, 16), pair(L0 + "linear_attn.in_proj_q", 8, 16, 16), "does not adapt");
  refused(root, "wrong_rank", config(8, 16), pair(L0 + "self_attn.q_proj", 4, 16, 16), "is not of rank r = 8");
  // One side of a linear's pair alone, or a second of one side (two spellings of one tensor's name): the listing of a
  // repository's adapters and the load refuse them alike, layer by layer and linear by linear.
  refused_both(root, "a_alone", 8, {{L0 + "self_attn.q_proj.lora_A.weight", "F32", {8, 16}}}, "L0.attn.q has lora_A alone");
  {
    auto t = pair(L0 + "self_attn.q_proj", 8, 16, 16);
    t.push_back({"base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight", "F32", {16, 8}});
    refused_both(root, "b_alone_other_layer", 8, t, "L1.attn.q has lora_B alone");
    t = pair(L0 + "self_attn.q_proj", 8, 16, 16);
    t.push_back({L0 + "self_attn.v_proj.lora_A.weight", "F32", {8, 16}});
    refused_both(root, "a_alone_other_linear", 8, t, "L0.attn.v has lora_A alone");
    t = pair(L0 + "self_attn.q_proj", 8, 16, 16);
    t.push_back({"model.layers.0.self_attn.q_proj.weight.lora_A.default", "F32", {8, 16}});
    refused_both(root, "second_a", 8, t, "a second lora_A on L0.attn.q");
  }
  {  // What the listing says of an adapter it calls loadable is what the load reads.
    std::vector<Tensor> t = pair(L0 + "self_attn.q_proj", 8, 16, 16);
    for (const Tensor& x : pair("base_model.model.model.layers.2.mlp.down_proj", 8, 32, 16)) t.push_back(x);
    std::string why;
    std::vector<std::string> linears;
    int layers = 0;
    const bool listed = flyx::modelfile::peft_header_ok(header_of(t), 8, &why, &linears, &layers);
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "listed", config(8, 16), t), &f, &err);
    check(listed && ok && layers == 3 && f.header()["companions"][0]["lora"]["layers"] == layers &&
              f.header()["companions"][0]["lora"]["linears"] == json(linears),
          "an adapter the listing takes loads, with the linears and layers it listed" + (ok ? std::string() : ": " + err) + why);
  }
  // Past what a load reads, refused before it is read: a header of more than 16 MiB (core/hub/hub_fetch.cpp reads no
  // more of the Hub's), and a config of more.
  {
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "big_header", config(8, 16), q, true, size_t(16) << 20), &f, &err);
    check(!ok && err.find("more than a safetensors header holds (16 MiB)") != std::string::npos &&
              err.find("truncated") == std::string::npos,
          "a safetensors header of more than 16 MiB refused before it is parsed: " + err);
    std::string why;
    const bool listed = flyx::modelfile::peft_header_ok(header_of(q) + std::string(size_t(16) << 20, ' '), 8, &why);
    check(!listed && why.find("16 MiB") != std::string::npos, "and by the listing: " + why);
    const fs::path dir = write_adapter(root, "big_config", config(8, 16), q);
    std::ofstream(dir / "adapter_config.json") << config(8, 16).dump() << std::string((size_t(16) << 20) + 1, ' ');
    flyx::ModelFile g;
    const bool read = add(dir, &g, &err);
    check(!read && err.find("more than an adapter's config holds (16 MiB)") != std::string::npos,
          "an adapter_config.json of more than 16 MiB refused before it is read: " + err);
  }
  refused(root, "dtype", config(8, 16), pair(L0 + "self_attn.q_proj", 8, 16, 16, "F64"), "F32, BF16 or F16");
  refused(root, "no_weights", config(8, 16), q, "no adapter_model.safetensors", /*weights=*/false);
  // Plain LoRA tensors whose config says they mean something else: never run as plain LoRA.
  refused(root, "alora", config(8, 16, {{"alora_invocation_tokens", {151644, 77091}}}), q, "activated LoRA");
  refused(root, "trainable_tokens", config(8, 16, {{"trainable_token_indices", {5, 6}}}), q, "token embeddings");
  for (const char* init : {"pissa", "pissa_niter_16", "olora", "corda", "loftq"})
    refused(root, std::string("init_") + init, config(8, 16, {{"init_lora_weights", init}}), q, "converted");
  {
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(write_adapter(root, "init_gaussian", config(8, 16, {{"init_lora_weights", "gaussian"}}), q), &f, &err);
    check(ok, "init_lora_weights gaussian (the base as it was) read: " + err);
  }
  // A tensor name longer than any model's is refused before it is matched, shown cut short.
  {
    flyx::ModelFile f;
    std::string err;
    const std::string longname = L0 + "mlp." + std::string(100000, 'a') + "_proj";
    const bool ok = add(write_adapter(root, "long_name", config(8, 16), pair(longname, 8, 16, 16)), &f, &err);
    check(!ok && err.find("bytes, not a decoder layer's LoRA") != std::string::npos && err.size() < 600,
          "a 100 KB tensor name refused, its message short: " + err.substr(0, 300));
  }
  {
    const fs::path dir = write_adapter(root, "pickle", config(8, 16), q, false);
    std::ofstream(dir / "adapter_model.bin") << "x";
    flyx::ModelFile f;
    std::string err;
    const bool ok = add(dir, &f, &err);
    check(!ok && err.find("pickle") != std::string::npos, "pickle refused: " + err);
  }

  fs::remove_all(root);
  return flyx::test::finish("peft_dir_test");
}
