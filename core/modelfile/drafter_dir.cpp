// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A DFlash2 drafter as its authors publish it: config.json and *.safetensors in a directory. Read
// into what python/convert/flyx_convert makes of it as a drafter file (--draft alone): the same "draft" description
// (canonical_draft_config with no target) and the same tensors under the same names, their bytes read in place.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>

#include "core/interfaces/json_depth.h"
#include "core/modelfile/readers.h"
#include "core/os/path.h"

namespace flyx::modelfile {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

using os::from_path;
using os::to_path;

int dtype_of(const std::string& s) {
  static const std::map<std::string, int> m = {{"BF16", FLYX_DT_BF16}, {"F16", FLYX_DT_F16}, {"F32", FLYX_DT_F32},
                                               {"F64", FLYX_DT_F64},   {"I32", FLYX_DT_I32}, {"I64", FLYX_DT_I64},
                                               {"U8", FLYX_DT_U8},     {"F8_E4M3", FLYX_DT_F8E4M3}};
  auto it = m.find(s);
  return it == m.end() ? -1 : it->second;
}

// python/convert's canonical_draft_config(config, target=None).
json draft_config(const json& sc) {
  json dc, tl;
  std::vector<int64_t> taps;
  int64_t vocab, target_hidden;
  json causal;  // null: causal where the layers slide (z-lab's code)
  if (sc.contains("dflash_config")) {
    const auto archs = sc.value("architectures", json::array());
    if (std::find(archs.begin(), archs.end(), "DFlash2DraftModel") == archs.end())
      throw std::runtime_error("drafter architectures " + archs.dump() + ": DFlash2DraftModel is supported");
    dc = sc.at("dflash_config"), tl = sc;
    // target_layer_ids name the layers whose outputs the drafter reads (hidden_states[id + 1] in z-lab's code), so
    // every id is below the target's layer count.  A list that reaches num_target_layers counts the hidden states
    // themselves instead (0 the embedding, the last the final norm's output): Inco's drafter for Inkling-2, whose
    // ids [5, 23, 41, 48] of 48 visits its reference hands, as they are, to the capture DSpark's ids go to.
    std::vector<int64_t> ids;
    for (const auto& i : dc.at("target_layer_ids")) ids.push_back(i.get<int64_t>());
    const json n_target = sc.value("num_target_layers", json());
    const bool hidden_states =
        n_target.is_number_integer() && std::find(ids.begin(), ids.end(), n_target.get<int64_t>()) != ids.end();
    for (int64_t id : ids) taps.push_back(hidden_states ? id : id + 1);
    vocab = tl.at("vocab_size").get<int64_t>(), target_hidden = tl.at("hidden_size").get<int64_t>();
    causal = sc.value("is_causal", json());
  } else {
    if (sc.value("speculators_model_type", "") != "dflash2")
      throw std::runtime_error("drafter type " + sc.value("speculators_model_type", json()).dump() +
                               " is not supported (need dflash2)");
    if (sc.value("sample_from_anchor", false)) throw std::runtime_error("sample_from_anchor=True is not supported");
    dc = sc, tl = sc.at("transformer_layer_config");
    for (const auto& i : sc.at("aux_hidden_state_layer_ids")) taps.push_back(i.get<int64_t>());
    const json dv = sc.value("draft_vocab_size", json());
    vocab = dv.is_number_integer() && dv.get<int64_t>() != 0 ? dv.get<int64_t>() : tl.at("vocab_size").get<int64_t>();
    const json th = sc.value("target_hidden_size", json());
    target_hidden = th.is_number() && th.get<int64_t>() != 0 ? th.get<int64_t>() : tl.at("hidden_size").get<int64_t>();
    causal = false;
  }
  json rp = tl.value("rope_parameters", json());
  if (rp.is_null() || rp.empty()) rp = tl.value("rope_scaling", json());
  if (rp.is_null()) rp = json::object();
  const int64_t layers = tl.at("num_hidden_layers").get<int64_t>();
  if (layers <= 0 || layers > 1024) throw std::runtime_error("draft num_hidden_layers out of range");
  std::vector<std::string> types;
  if (tl.contains("layer_types") && tl["layer_types"].is_array() && !tl["layer_types"].empty())
    types = tl["layer_types"].get<std::vector<std::string>>();
  else
    types.assign(size_t(layers), "full_attention");
  int64_t window = 0;
  auto all = [&](const char* t) { return std::all_of(types.begin(), types.end(), [&](const std::string& x) { return x == t; }); };
  if (all("full_attention")) {
    window = 0;
  } else if (all("sliding_attention") && tl.value("sliding_window", json()).is_number() && tl["sliding_window"].get<int64_t>()) {
    window = tl["sliding_window"].get<int64_t>();
  } else {
    throw std::runtime_error("draft layer types: all full or all sliding with one window");
  }
  const bool is_causal = causal.is_null() ? window > 0 : causal.get<bool>();
  if (is_causal) throw std::runtime_error("causal DFlash2 layers are not supported (the block attends both ways)");
  const int64_t hidden = tl.at("hidden_size").get<int64_t>();
  const int64_t n_heads = tl.at("num_attention_heads").get<int64_t>();
  if (hidden <= 0 || n_heads <= 0) throw std::runtime_error("draft hidden_size and num_attention_heads must be positive");
  const json hd_given = tl.value("head_dim", json());
  const int64_t hd = hd_given.is_number() && hd_given.get<int64_t>() != 0 ? hd_given.get<int64_t>() : hidden / n_heads;
  if (hd <= 0) throw std::runtime_error("draft head_dim must be positive");
  double theta = 10000.0;
  if (rp.contains("rope_theta") && rp["rope_theta"].is_number()) theta = rp["rope_theta"].get<double>();
  else if (tl.contains("rope_theta") && tl["rope_theta"].is_number()) theta = tl["rope_theta"].get<double>();
  json d = json::object();
  d["kind"] = "dflash2";
  d["hidden"] = tl.at("hidden_size");
  d["layers"] = tl.at("num_hidden_layers");
  d["n_heads"] = tl.at("num_attention_heads");
  d["n_kv"] = tl.at("num_key_value_heads");
  d["head_dim"] = hd;
  d["inter"] = tl.at("intermediate_size");
  d["eps"] = tl.at("rms_norm_eps");
  d["rope_theta"] = theta;
  d["block"] = dc.at("block_size");
  d["mask_token"] = dc.at("mask_token_id");
  d["taps"] = taps;
  d["conv_taps"] = dc.value("conv_kernel_size", json(2));
  d["conv_group"] = dc.value("conv_group_size", json(16));
  d["sel_rank"] = dc.value("selector_rank", json(256));
  d["sel_topk"] = dc.value("selector_top_k", json(16));
  d["window"] = window;
  if (dc.contains("final_logit_softcapping") && dc["final_logit_softcapping"].is_number()) {
    const double sc_val = dc["final_logit_softcapping"].get<double>();
    if (!std::isfinite(sc_val) || sc_val < 0.0) throw std::runtime_error("draft final_logit_softcapping out of range");
    if (sc_val > 0.0) d["softcap"] = sc_val;
  }
  if (dc.contains("selector_softcap_mode") && dc["selector_softcap_mode"].is_string()) {
    const std::string sm = dc["selector_softcap_mode"].get<std::string>();
    if (sm != "before_correction" && sm != "after_correction")
      throw std::runtime_error("unsupported draft selector_softcap_mode: " + sm);
    d["softcap_mode"] = sm;
  }
  d["target"] = {{"hidden", target_hidden}, {"vocab", vocab}, {"layers", sc.value("num_target_layers", json())}};
  return d;
}

// python/convert's _D_SKIP and _D_RULES: the drafter's tensor names as the engine knows them, or "" to leave out.
bool canonical(const std::string& name, std::string* out) {
  static const std::regex skip(R"(^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding).*)");
  static const std::vector<std::pair<std::regex, std::string>> rules = {
      {std::regex(R"(fc\.weight)"), "fc"},
      {std::regex(R"(hidden_norm\.weight)"), "hidden_norm"},
      {std::regex(R"(norm\.weight)"), "norm"},
      {std::regex(R"(layers\.(\d+)\.input_layernorm\.weight)"), "L$1.in_norm"},
      {std::regex(R"(layers\.(\d+)\.post_attention_layernorm\.weight)"), "L$1.post_norm"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight)"), "L$1.attn.$2"},
      {std::regex(R"(layers\.(\d+)\.self_attn\.(q|k)_norm\.weight)"), "L$1.attn.$2_norm"},
      {std::regex(R"(layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight)"), "L$1.mlp.$2"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.base_kernel)"), "L$1.$2_conv.base"},
      {std::regex(R"(layers\.(\d+)\.(attention|mlp)_conv\.kernel_projection\.weight)"), "L$1.$2_conv.proj"},
      {std::regex(R"(candidate_selector\.predecessor_codebook)"), "sel.pred"},
      {std::regex(R"(candidate_selector\.successor_codebook)"), "sel.succ"},
      {std::regex(R"(candidate_selector\.hidden_projection\.weight)"), "sel.hid"},
  };
  const std::string inner = name.rfind("model.", 0) == 0 ? name.substr(6) : name;
  if (std::regex_match(inner, skip)) return out->clear(), true;
  for (const auto& [re, fmt] : rules) {
    std::smatch m;
    if (!std::regex_match(inner, m, re)) continue;
    std::string s = m.format(fmt);
    const size_t at = s.find("attention_conv");
    if (at != std::string::npos) s.replace(at, 14, "attn_conv");
    *out = s;
    return true;
  }
  return false;
}

}  // namespace

namespace {

// A drafter directory's config.json (at most 16 MiB, nested within json_depth_within).
bool read_config_json(const fs::path& root, const std::string& dir, json* sc, std::string* err) {
  constexpr size_t kMaxConfigBytes = 16u << 20;
  std::ifstream f(root / "config.json", std::ios::binary | std::ios::ate);
  const std::streamoff fsz = f.tellg();
  if (!f || fsz < 0 || size_t(fsz) > kMaxConfigBytes)
    return *err = dir + ": cannot read config.json (or larger than 16 MiB)", false;
  f.seekg(0);
  std::string cfg_text(size_t(fsz), '\0');
  if (fsz > 0 && !f.read(cfg_text.data(), fsz)) return *err = dir + ": cannot read config.json", false;
  if (!interfaces::json_depth_within(cfg_text)) return *err = dir + ": config.json nested too deeply", false;
  *sc = json::parse(cfg_text);
  return true;
}

}  // namespace

std::string drafter_dir_kind(const std::string& dir) {
  try {
    json sc;
    std::string err;
    if (!read_config_json(to_path(dir), dir, &sc, &err) || !sc.is_object()) return "";
    const auto archs = sc.value("architectures", json::array());
    if (std::find(archs.begin(), archs.end(), "DFlash2DraftModel") != archs.end() ||
        sc.value("speculators_model_type", "") == "dflash2") return "dflash2";
    return "";
  } catch (const std::exception&) {  // a config.json that is no JSON
    return "";
  }
}

bool is_drafter_dir(const std::string& path) {
  std::error_code ec;
  return fs::is_directory(to_path(path), ec) && fs::is_regular_file(to_path(path) / "config.json", ec);
}

bool read_drafter_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                      ModelFile::Contents* out, std::string* err) {
  const fs::path root = to_path(dir);
  std::vector<fs::path> shards;
  std::error_code ec;
  for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
    if (it->path().extension() == ".safetensors") shards.push_back(it->path());
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) return *err = dir + ": no *.safetensors beside its config.json", false;
  try {
    json sc;
    if (!read_config_json(root, dir, &sc, err)) return false;
    const json draft = draft_config(sc);
    *out = {};
    out->arch = "drafter";
    out->header = {{"arch", "drafter"}, {"kind", "drafter"}, {"draft", draft},
                   {"source", {{"draft", from_path(root.filename().empty() ? root.parent_path().filename() : root.filename())}}}};
    std::vector<std::string> seen;
    for (const fs::path& shard : shards) {
      auto map = std::make_unique<os::MappedFile>();
      if (!map->open(from_path(shard), err)) return false;
      const uint8_t* p = map->data();
      const size_t size = map->size();
      uint64_t hlen = 0;
      if (size < 8) return *err = from_path(shard) + ": not a safetensors file", false;
      std::memcpy(&hlen, p, 8);
      if (hlen > size - 8) return *err = from_path(shard) + ": truncated header", false;
      const std::string_view hdr_sv(reinterpret_cast<const char*>(p + 8), size_t(hlen));
      if (!interfaces::json_depth_within(hdr_sv))
        return *err = from_path(shard) + ": header JSON nested too deeply", false;
      const json h = json::parse(hdr_sv);
      const size_t data_start = 8 + size_t(hlen), data_size = size - data_start;
      for (auto it = h.begin(); it != h.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        std::string canon;
        if (!canonical(it.key(), &canon)) return *err = "unknown draft tensor " + it.key(), false;
        if (canon.empty()) continue;  // the engine uses the target's embedding and head
        const json& r = it.value();
        flyx_tensor_view v{};
        v.dtype = dtype_of(r.at("dtype").get<std::string>());
        const auto shape = r.at("shape").get<std::vector<int64_t>>();
        const auto offsets = r.at("data_offsets").get<std::vector<uint64_t>>();
        if (v.dtype < 0 || shape.size() > 4 || offsets.size() != 2 || offsets[0] > offsets[1] ||
            offsets[1] > data_size)
          return *err = it.key() + ": unsupported dtype or rank, or outside the file (truncated download?)", false;
        v.ndim = int32_t(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) {
          if (shape[i] < 0) return *err = it.key() + ": a negative dimension", false;
          v.shape[i] = shape[i];
        }
        uint64_t want_bytes = 0;
        if (!plain_tensor_nbytes(v.dtype, shape, &want_bytes) || offsets[1] - offsets[0] != want_bytes)
          return *err = it.key() + ": byte size does not match shape and dtype", false;
        v.data = p + data_start + offsets[0];
        v.nbytes = offsets[1] - offsets[0];
        const std::string name = "draft." + canon;
        out->index[name] = out->tensors.size();
        out->names.push_back(name);
        out->tensors.push_back(v);
        seen.push_back(canon);
      }
      maps->push_back(std::move(map));
    }
    for (size_t i = 0; i < out->tensors.size(); ++i) out->tensors[i].name = out->names[i].c_str();
    // python/convert's _check_draft
    std::vector<std::string> need = {"fc", "hidden_norm", "norm", "sel.pred", "sel.succ", "sel.hid"};
    for (int64_t j = 0; j < draft.at("layers").get<int64_t>(); ++j)
      for (const char* n : {"in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm",
                            "attn.k_norm", "mlp.gate", "mlp.up", "mlp.down", "attn_conv.base", "attn_conv.proj",
                            "mlp_conv.base", "mlp_conv.proj"})
        need.push_back("L" + std::to_string(j) + "." + n);
    std::sort(seen.begin(), seen.end());
    for (const std::string& n : need)
      if (!std::binary_search(seen.begin(), seen.end(), n)) return *err = dir + ": missing draft tensor " + n, false;
  } catch (const std::exception& e) {
    *err = dir + ": " + e.what();
    return false;
  }
  return true;
}

}  // namespace flyx::modelfile
