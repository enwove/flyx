// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A block drafter read from its authors' directory (config.json and safetensors) is what
// python/convert makes of it as a drafter file: the same "draft" description and the same tensors, byte for byte.
// The tiny assets hold the drafter as z-lab publishes it (hf_draft_zlab), with a sliding window (hf_draft_window)
// and in speculators' layout (hf_draft), and python/convert's files of the first two; and a DSpine drafter as
// tencent publishes it (hf_dspine) with python/convert's file of it.
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/modelfile/modelfile.h"
#include "core/os/os.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

namespace {

// The model with one companion added from `path`.
bool with(const std::string& model, const std::string& path, flyx::ModelFile* f) {
  std::string err;
  if (!f->open(model, &err) || !f->add(path, &err)) {
    std::printf("FAIL %s: %s\n", path.c_str(), err.c_str());
    ++flyx::test::failures;
    return false;
  }
  return true;
}

// speculators' layout does not say how many layers the target has: the description's target.layers is null there.
void same(const std::string& dir, const std::string& model, const std::string& published, const std::string& file,
          bool target_layers = true) {
  flyx::ModelFile a, b;
  if (!with(model, dir + "/" + published, &a) || !with(model, dir + "/" + file, &b)) return;
  const auto& da = a.header()["companions"][0], &db = b.header()["companions"][0];
  auto want = db["draft"];
  if (!target_layers) want["target"]["layers"] = nullptr;
  check(da.value("kind", "") == "drafter" && da["draft"] == want,
        published + ": the draft description python/convert wrote into " + file);
  int tensors = 0, differ = 0;
  for (auto it = db["tensors"].begin(); it != db["tensors"].end(); ++it, ++tensors) {
    const std::string name = flyx::ModelFile::companion_name(0, it.key());
    const flyx_tensor_view *x = a.tensor(name), *y = b.tensor(name);
    const bool equal = x && y && x->dtype == y->dtype && x->ndim == y->ndim && x->nbytes == y->nbytes &&
                       std::memcmp(x->shape, y->shape, sizeof x->shape) == 0 &&
                       std::memcmp(x->data, y->data, size_t(x->nbytes)) == 0;
    if (!equal) std::printf("     %s differs\n", it.key().c_str());
    differ += !equal;
  }
  check(tensors > 0 && differ == 0, published + ": its " + std::to_string(tensors) + " tensors, byte for byte");
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 2) return std::fprintf(stderr, "usage: drafter_dir_test ASSETS\n"), 2;
  const std::string dir = argv[1], model = dir + "/tiny_gguf.gguf";
  same(dir, model, "hf_draft_zlab", "tiny_draft.flyx");
  same(dir, model, "hf_draft_window", "tiny_draft_window.flyx");
  same(dir, model, "hf_draft", "tiny_draft.flyx", false);  // speculators' layout: the same drafter
  same(dir, model, "hf_dspine", "tiny_draft_dspine.flyx");   // DSpine, its stacked bridge tensors cut per layer
  same(dir, model, "hf_dspine1", "tiny_draft_dspine1.flyx");  // one layer: no early reads ([0, code, hidden])
  {
    flyx::ModelFile f;
    std::string err;
    const bool opened = f.open(dir + "/hf_draft_zlab", &err);
    check(!opened && err.find("a drafter alone") != std::string::npos, "a drafter's directory is no model (" + err + ")");
  }
  {
    // Null optional draft_vocab_size / rope_theta fall back cleanly, and deeply nested config.json is rejected.
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / ("flyx-drafter-dir-test-" + std::to_string(flyx::os::process_id()));
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::copy(fs::path(dir) / "hf_draft", tmp, fs::copy_options::recursive, ec);
    check(!ec, "copy hf_draft fixture: " + ec.message());
    {
      std::ifstream in(tmp / "config.json");
      auto cfg = nlohmann::json::parse(in);
      cfg["draft_vocab_size"] = nullptr;
      cfg["transformer_layer_config"]["rope_theta"] = nullptr;
      std::ofstream out(tmp / "config.json", std::ios::binary | std::ios::trunc);
      out << cfg.dump();
    }
    {
      flyx::ModelFile f;
      std::string err;
      check(f.open(model, &err) && f.add(tmp.string(), &err),
            "drafter config.json with explicit null draft_vocab_size and rope_theta loads: " + err);
    }
    {
      std::ofstream out(tmp / "config.json", std::ios::binary | std::ios::trunc);
      out << R"({"dflash_config":{},"architectures":["DFlash2DraftModel"],"x":)" << std::string(200, '[')
          << std::string(200, ']') << "}";
    }
    {
      flyx::ModelFile f2;
      std::string err;
      check(f2.open(model, &err) && !f2.add(tmp.string(), &err) && err.find("nest") != std::string::npos,
            "deeply nested drafter config.json is rejected cleanly: " + err);
    }
    // A config whose target_layer_ids reach num_target_layers counts the target's hidden states (Inco's drafter for
    // Inkling-2): its ids are the taps as they are, the last the final norm's output; the selector's soft cap and its
    // mode come along.
    fs::remove_all(tmp, ec);
    fs::copy(fs::path(dir) / "hf_draft_zlab", tmp, fs::copy_options::recursive, ec);
    {
      std::ifstream in(tmp / "config.json");
      auto cfg = nlohmann::json::parse(in);
      const int n_target = cfg.at("num_target_layers").get<int>();
      cfg["dflash_config"]["target_layer_ids"] = {1, 2, n_target};
      cfg["dflash_config"]["final_logit_softcapping"] = 30.0;
      cfg["dflash_config"]["selector_softcap_mode"] = "after_correction";
      std::ofstream out(tmp / "config.json", std::ios::binary | std::ios::trunc);
      out << cfg.dump();
    }
    {
      flyx::ModelFile f3;
      std::string err;
      const bool ok = f3.open(model, &err) && f3.add(tmp.string(), &err);
      const auto draft = ok ? f3.header()["companions"][0]["draft"] : nlohmann::json::object();
      const auto taps = ok ? draft["taps"].get<std::vector<int>>() : std::vector<int>{};
      const int n_target = ok ? draft["target"]["layers"].get<int>() : -1;
      check(ok && taps == std::vector<int>({1, 2, n_target}) && draft.value("softcap", 0.0) == 30.0 &&
                draft.value("softcap_mode", "") == "after_correction",
            "target_layer_ids that reach num_target_layers are the taps as they are, with the soft cap and its mode: " +
                err);
    }
    fs::remove_all(tmp, ec);
  }
  {
    // DSpine's training-only fields (the refinement's weight, the bridge's kappa) take Tencent's readers' values, and
    // nothing else is read: a negative weight, a bridge key of another kind or a kappa that is no number is refused.
    namespace fs = std::filesystem;
    const fs::path tmp = fs::temp_directory_path() / ("flyx-dspine-dir-test-" + std::to_string(flyx::os::process_id()));
    std::error_code ec;
    auto refused = [&](const char* what, auto edit) {
      fs::remove_all(tmp, ec);
      fs::copy(fs::path(dir) / "hf_dspine", tmp, fs::copy_options::recursive, ec);
      nlohmann::json cfg;
      {
        std::ifstream in(tmp / "config.json");
        cfg = nlohmann::json::parse(in);
      }
      edit(cfg["dflash_config"]);
      {
        std::ofstream out(tmp / "config.json", std::ios::binary | std::ios::trunc);
        out << cfg.dump();
      }
      flyx::ModelFile f;
      std::string err;
      check(f.open(model, &err) && !f.add(tmp.string(), &err) && err.find("DSpine") != std::string::npos,
            std::string("a DSpine config with ") + what + " refused: " + err);
    };
    refused("a negative refinement weight",
            [](nlohmann::json& dc) { dc["selected_prefix_last_write_refine"]["weight"] = -1; });
    refused("a bridge key of another kind", [](nlohmann::json& dc) { dc["selected_prefix_code_bridge"]["scale"] = 2; });
    refused("a kappa that is no number", [](nlohmann::json& dc) { dc["selected_prefix_code_bridge"]["kappa"] = "10"; });
    fs::remove_all(tmp, ec);
  }
  return flyx::test::finish("drafter_dir_test");
}
