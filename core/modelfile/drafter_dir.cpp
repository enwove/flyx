// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Read directory metadata and bounded safetensors views. The selected drafter interprets their names and shapes.
#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>

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

// The block drafters a config.json names by its architecture, and the kind each reads as (draft_config).
constexpr std::pair<std::string_view, std::string_view> kDrafterArchitectures[] = {
    {"DFlash2DraftModel", "dflash2"},
    {"DFlashDraftModel", "dflash"},
    {"Qwen3DSparkModel", "dspark"},
    {"DSpineDraftModel", "dspine"},
    {"LlamaForCausalLMEagle4", "eagle4"},
};

std::string architectures_kind(const std::vector<std::string>& architectures) {
  for (const std::string& a : architectures)
    for (const auto& [architecture, kind] : kDrafterArchitectures)
      if (a == architecture) return std::string(kind);
  return "";
}

// The block drafter a config.json is: by its architecture, else speculators' type ("dflash2"); "" for none.
std::string drafter_config_kind(const json& sc) {
  std::vector<std::string> archs;
  for (const auto& a : sc.value("architectures", json::array()))
    if (a.is_string()) archs.push_back(a.get<std::string>());
  const std::string kind = architectures_kind(archs);
  return !kind.empty() ? kind : sc.value("speculators_model_type", "") == "dflash2" ? "dflash2" : "";
}

// A drafter directory's config.json (at most 16 MiB, nested within json_depth_within).
bool read_config_json(const fs::path& root, const std::string& dir, json* sc, std::string* err) {
  std::ifstream f(root / "config.json", std::ios::binary | std::ios::ate);
  const std::streamoff fsz = f.tellg();
  if (!f || fsz < 0 || size_t(fsz) > interfaces::kMaxConfigBytes)
    return *err = dir + ": cannot read config.json (or larger than 16 MiB)", false;
  f.seekg(0);
  std::string cfg_text(size_t(fsz), '\0');
  if (fsz > 0 && !f.read(cfg_text.data(), fsz)) return *err = dir + ": cannot read config.json", false;
  if (!interfaces::json_depth_within(cfg_text)) return *err = dir + ": config.json nested too deeply", false;
  *sc = json::parse(cfg_text);
  return true;
}

}  // namespace

std::string drafter_dir_kind(const std::string& dir, const PathAllowFn& allow_path) {
  try {
    const fs::path root = to_path(dir);
    if (allow_path && (!allow_path(dir) || !allow_path(from_path(root / "config.json")))) return "";
    json sc;
    std::string err;
    return read_config_json(root, dir, &sc, &err) ? drafter_config_kind(sc) : "";
  } catch (const std::exception&) {  // a config.json that is no JSON
    return "";
  }
}

bool is_drafter_dir(const std::string& path, const PathAllowFn& allow_path) {
  const fs::path root = to_path(path);
  const fs::path cfg = root / "config.json";
  if (allow_path && (!allow_path(path) || !allow_path(from_path(cfg)))) return false;
  std::error_code ec;
  return fs::is_directory(root, ec) && fs::is_regular_file(cfg, ec);
}

bool read_drafter_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                      ModelFile::Contents* out, std::string* err, const PathAllowFn& allow_path) {
  const fs::path root = to_path(dir);
  if (allow_path && !allow_path(dir))
    return *err = dir + ": path is not allowed by remote path policy", false;
  if (allow_path && !allow_path(from_path(root / "config.json")))
    return *err = from_path(root / "config.json") + ": path is not allowed by remote path policy", false;
  std::vector<fs::path> shards;
  std::error_code ec;
  for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec))
    if (it->path().extension() == ".safetensors") shards.push_back(it->path());
  std::sort(shards.begin(), shards.end());
  if (shards.empty()) return *err = dir + ": no *.safetensors beside its config.json", false;
  for (const fs::path& shard : shards) {
    if (allow_path && !allow_path(from_path(shard)))
      return *err = from_path(shard) + ": path is not allowed by remote path policy", false;
  }
  try {
    json sc;
    if (!read_config_json(root, dir, &sc, err)) return false;
    const std::string kind = drafter_config_kind(sc);
    if (kind.empty()) return *err = dir + ": unsupported drafter architecture", false;
    const bool speculators = sc.contains("transformer_layer_config") && !sc.contains("dflash_config");
    const json& dims = speculators ? sc.at("transformer_layer_config") : sc;
    json target = {{"layers", sc.value("num_target_layers", json())}};
    for (const auto [key, field] : {std::pair{"hidden_size", "hidden"}, {"vocab_size", "vocab"}})
      if (dims.contains(key)) target[field] = dims[key];
    if (speculators) {
      for (const auto [key, field] : {std::pair{"target_hidden_size", "hidden"}, {"draft_vocab_size", "vocab"}})
        if (sc.contains(key) && sc[key].is_number_integer() && sc[key] != 0) target[field] = sc[key];
    }
    *out = {};
    out->arch = "drafter";
    out->header = {{"arch", "drafter"}, {"kind", "drafter"}, {"draft", {{"kind", kind}, {"target", target}}},
                   {"source", {{"draft", from_path(root.filename().empty() ? root.parent_path().filename() : root.filename())}}}};
    out->directory_config = std::move(sc);
    for (const fs::path& shard : shards) {
      auto map = std::make_unique<os::MappedFile>();
      if (!map->open(from_path(shard), err)) return false;
      const uint8_t* p = map->data();
      const size_t size = map->size();
      uint64_t hlen = 0;
      if (size < 8) return *err = from_path(shard) + ": not a safetensors file", false;
      std::memcpy(&hlen, p, 8);
      if (hlen > kSafetensorsHeaderMaxBytes || hlen > size - 8) return *err = from_path(shard) + ": truncated header", false;
      const std::string_view hdr_sv(reinterpret_cast<const char*>(p + 8), size_t(hlen));
      if (!interfaces::json_depth_within(hdr_sv))
        return *err = from_path(shard) + ": header JSON nested too deeply", false;
      const json h = json::parse(hdr_sv);
      const size_t data_start = 8 + size_t(hlen), data_size = size - data_start;
      for (auto it = h.begin(); it != h.end(); ++it) {
        if (it.key() == "__metadata__") continue;
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
        if (!out->index.emplace(it.key(), out->tensors.size()).second)
          return *err = dir + ": duplicate tensor " + it.key(), false;
        out->names.push_back(it.key());
        out->tensors.push_back(v);
      }
      maps->push_back(std::move(map));
    }
    for (size_t i = 0; i < out->tensors.size(); ++i) out->tensors[i].name = out->names[i].c_str();
  } catch (const std::exception& e) {
    *err = dir + ": " + e.what();
    return false;
  }
  return true;
}

}  // namespace flyx::modelfile

namespace flyx {

std::string drafter_kind_of(const std::vector<std::string>& architectures) {
  return modelfile::architectures_kind(architectures);
}

}  // namespace flyx
