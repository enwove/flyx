// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A .flyx file packed from a GGUF (core/modelfile/flyx_pack.cpp, flyx pack) reads as its GGUF reads: the same header
// (but for the packed file's own "recommended" and "packed"), the same tensors byte for byte under the same names and
// dtypes, the same blobs, and the files packed beside it its first companions with their roles, headers, tensors and
// a drafter directory's config.  Every GGML block type, a split GGUF, the MTP layer in a file of its own, a DFlash2
// GGUF drafter and DSpine's drafter directory.  What the writer refuses (a model not read from a GGUF, a LoRA adapter,
// the reader's keys) and what the reader refuses of a packed header.
//
//   flyx_pack_test ASSETS     (after make_gguf.py; the packed files are written there)
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <source_location>
#include <string>
#include <vector>

#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "core/modelfile/readers.h"
#include "tests/common/check.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;
using namespace flyx::testing;
using json = nlohmann::json;

namespace {

// A header as a file says what it is: without the keys a .flyx reader reads of its layout, and the packed file's own.
json described(json h) {
  for (const char* k : {"tensors", "blobs", "bundled", "requires", "companions", "recommended", "packed"}) h.erase(k);
  return h;
}

// Where two readings part ("" when they do not).
std::string parting(const ModelFile::Contents& a, const ModelFile::Contents& b) {
  if (a.arch != b.arch) return "arch " + a.arch + " / " + b.arch;
  if (described(a.header) != described(b.header)) return "the header";
  if (a.directory_config != b.directory_config) return "the directory's config";
  if (a.tensors.size() != b.tensors.size()) return "the number of tensors";
  for (size_t i = 0; i < a.tensors.size(); ++i) {
    const auto it = b.index.find(a.names[i]);
    if (it == b.index.end()) return "tensor " + a.names[i] + " missing";
    const flyx_tensor_view &x = a.tensors[i], &y = b.tensors[it->second];
    if (x.dtype != y.dtype || x.ndim != y.ndim || x.nbytes != y.nbytes ||
        !std::equal(x.shape, x.shape + x.ndim, y.shape) || std::memcmp(x.data, y.data, size_t(x.nbytes)) != 0)
      return "tensor " + a.names[i];
  }
  if (a.blobs.size() != b.blobs.size()) return "the number of blobs";
  for (const auto& [name, bytes] : a.blobs) {
    const auto it = b.blobs.find(name);
    if (it == b.blobs.end() || it->second != bytes) return "blob " + name;
  }
  return "";
}

// check(), and whether it held (a case that fails one goes no further).
bool held(bool ok, std::string_view what, std::source_location where = std::source_location::current()) {
  check(ok, what, where);
  return ok;
}

struct Case {
  std::string model;
  std::vector<std::pair<std::string, ModelFile::CompanionRole>> beside;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: flyx_pack_test ASSETS\n");
    return 2;
  }
  const std::string dir = argv[1];
  const auto drafter = ModelFile::CompanionRole::Drafter;
  std::vector<Case> cases = {
      {"tiny_gguf.gguf", {}},
      {"tiny_bf16.gguf", {}},
      {"tiny_q8.gguf", {}},
      {"tiny_nvfp4.gguf", {}},
      {"tiny_q8_split-00001-of-00002.gguf", {}},
      {"tiny_qwen3_gguf.gguf", {}},
      {"tiny_q8_target.gguf", {{"tiny_q8_mtp.gguf", drafter}}},
      {"tiny_gguf.gguf", {{"tiny_draft_q8.gguf", drafter}}},
      {"tiny_q8.gguf", {{"hf_dspine", drafter}}},
  };
  for (const auto& e : std::filesystem::directory_iterator(dir)) {  // a tiny Qwen3 for every GGML block type
    const std::string name = e.path().filename().string();
    if (name.starts_with("quant_") && name.ends_with(".gguf")) cases.push_back({name, {}});
  }
  const json recommended = {{"drafter", "mtp"}, {"backend_opts", {{"draft_weights", "auto"}}}};
  std::string err;
  int n = 0;
  for (const Case& c : cases) {
    std::string what = c.model;
    for (const auto& [file, role] : c.beside) what += " + " + file;
    ModelFile gguf;
    if (!held(gguf.open(dir + "/" + c.model, &err), what + ": opens: " + err)) continue;
    bool added = true;
    for (const auto& [file, role] : c.beside) added = added && gguf.add(dir + "/" + file, &err, {}, role);
    if (!held(added, what + ": its companions: " + err)) continue;
    const std::string packed = dir + "/packed_" + std::to_string(n++) + ".flyx";
    if (!held(modelfile::write_packed(gguf, {{"recommended", recommended}}, packed, &err), what + ": packs: " + err))
      continue;
    ModelFile back;
    if (!held(back.open(packed, &err), what + ": the packed file opens: " + err)) continue;
    const std::string part = parting(gguf.contents(), back.contents());
    check(part.empty(), what + ": the packed file reads as the GGUF" + (part.empty() ? "" : " (not " + part + ")"));
    check(back.header().at("recommended") == recommended, what + ": its recommended settings in the header");
    check(!back.header().contains("requires") && !back.header().contains("bundled"),
          what + ": the reader's parts taken off the header");
    check(back.bundled() == gguf.companions() && back.companions() == gguf.companions(),
          what + ": the files beside it its companions");
    for (int i = 0; i < gguf.companions() && i < back.companions(); ++i) {
      const std::string cpart = parting(gguf.companion(i), back.companion(i));
      check(back.companion_role(i) == gguf.companion_role(i) && cpart.empty(),
            what + ": companion " + std::to_string(i) + " reads as the file it was" +
                (cpart.empty() ? "" : " (not " + cpart + ")"));
    }
    check(interfaces::open_model(back, {}, &err) != nullptr, what + ": opens as a model: " + err);
    // What the loader reads of it before opening it: the model's own MTP layer, then each bundled drafter's kind.
    const auto kinds = packed_drafters(packed);
    const size_t own_mtp = gguf_peek(dir + "/" + c.model).mtp_layers > 0 ? 1 : 0;
    bool named = kinds.size() == own_mtp + c.beside.size();
    std::string listed;
    for (const auto& k : kinds) named = named && !k.empty(), listed += " " + k;
    check(named, what + ": its drafters read from its header:" + listed);
    std::filesystem::remove(packed);
  }

  {  // what the writer refuses
    const std::string out = dir + "/packed_refused.flyx";
    ModelFile f;
    check(f.open(dir + "/tiny.flyx", &err), "tiny.flyx opens: " + err);
    check(!modelfile::write_packed(f, {}, out, &err) && err.find("GGUF") != std::string::npos,
          "a model not read from a GGUF: refused (" + err + ")");
    ModelFile g;
    check(g.open(dir + "/tiny_gguf.gguf", &err), "tiny_gguf.gguf opens: " + err);
    for (const char* key : {"tensors", "bundled", "requires"})
      check(!modelfile::write_packed(g, {{key, 1}}, out, &err), std::string("the reader's key ") + key +
                                                                    " given: refused (" + err + ")");
    check(!modelfile::write_packed(g, {{"gguf", 1}}, out, &err), "a key of the GGUF's header given: refused (" + err + ")");
    check(g.add(dir + "/peft_qwen35", &err, {}, ModelFile::CompanionRole::Lora), "a LoRA adapter beside it: " + err);
    check(!modelfile::write_packed(g, {}, out, &err) && err.find("drafter or a projector") != std::string::npos,
          "a LoRA adapter beside it: refused (" + err + ")");
    check(!std::filesystem::exists(out), "nothing written when refused");
  }

  {  // what the reader refuses of a packed header
    const std::string path = dir + "/packed_header.flyx";
    const std::string data(64, '\0');
    const std::string gguf_head = R"("arch": "qwen35", "source": {"format": "gguf"}, "gguf": {})";
    auto opens = [&](const std::string& header, uint32_t version) {
      write_file(path, flyx_file(header, data, header.size(), version));
      ModelFile f;
      return f.open(path, &err);
    };
    check(opens("{" + gguf_head + R"(, "requires": ["gguf"], "tensors": {"w": {"dtype": "ggml_q8_0", "shape": [1, 32], )"
                                  R"("offset": 0, "nbytes": 34}}})", 2),
          "a GGML Q8_0 row of one block: read (" + err + ")");
    check(!opens("{" + gguf_head + R"(, "requires": ["gguf"], "tensors": {"w": {"dtype": "ggml_q8_0", "shape": [1, 32], )"
                                   R"("offset": 0, "nbytes": 32}}})", 2),
          "a GGML block tensor of the wrong size: refused (" + err + ")");
    check(!opens("{" + gguf_head + R"(, "requires": ["gguf"], "tensors": {"w": {"dtype": "ggml_q8_0", "shape": [2, 16], )"
                                   R"("offset": 0, "nbytes": 34}}})", 2),
          "GGML block rows not whole blocks: refused (" + err + ")");
    check(!opens(R"({"arch": "qwen35", "requires": ["gguf"], "tensors": {}})", 2) &&
              err.find("GGUF's header") != std::string::npos,
          "requires \"gguf\" without a GGUF's header: refused (" + err + ")");
    check(!opens("{" + gguf_head + R"(, "tensors": {}, "bundled": []})", 1) && err.find("not packed") != std::string::npos,
          "bundled files in a file not packed from a GGUF: refused (" + err + ")");
    check(!opens("{" + gguf_head + R"(, "requires": ["gguf"], "tensors": {}, "bundled": [{"role": "lora", "header": {}, )"
                                   R"("tensors": {}}]})", 2),
          "a bundled file of a role other than a drafter's or a projector's: refused (" + err + ")");
    const std::string bundles = "{" + gguf_head + R"(, "requires": ["gguf"], "tensors": {}, "bundled": [{"role": )"
                                R"("drafter", "header": {"arch": "dflash"}, "tensors": {}}]})";
    check(opens(bundles, 2), "a packed file with a bundled drafter: read (" + err + ")");
    write_file(path, flyx_file(bundles, data, bundles.size(), 2));
    ModelFile g;
    check(g.open(dir + "/tiny_gguf.gguf", &err) && !g.add(path, &err) && err.find("as the model") != std::string::npos,
          "a packed file given beside a model: refused (" + err + ")");
    std::filesystem::remove(path);
  }
  return flyx::test::finish("flyx_pack_test");
}
