// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <map>
#include <stdexcept>
#include "core/modelfile/modelfile.h"

namespace flyx::models::drafters {

struct DirectoryDraft {
  nlohmann::json config;
  std::map<std::string, flyx_tensor_view> tensors;
  void add(const std::string& suffix, flyx_tensor_view view) {
    if (!tensors.emplace("draft." + suffix, view).second) throw std::runtime_error("duplicate draft tensor " + suffix);
  }
  void require(const std::string& suffix) const {
    if (!tensors.count("draft." + suffix)) throw std::runtime_error("missing draft tensor " + suffix);
  }
};

DirectoryDraft directory_draft(const ModelFile::Contents& raw);
DirectoryDraft dflash_directory(const ModelFile::Contents& raw);
DirectoryDraft dspine_directory(const ModelFile::Contents& raw);
DirectoryDraft eagle4_directory(const ModelFile::Contents& raw);

// A transformers layer config's fields every block drafter's canonical config has (python/convert's
// _layer_fields): width, layers, heads, head width, MLP width, norm epsilon, rope theta.
inline nlohmann::json layer_fields(const nlohmann::json& tl) {
  using nlohmann::json;
  const int64_t layers = tl.at("num_hidden_layers").get<int64_t>();
  if (layers <= 0 || layers > 1024) throw std::runtime_error("draft num_hidden_layers out of range");
  const int64_t hidden = tl.at("hidden_size").get<int64_t>(), n_heads = tl.at("num_attention_heads").get<int64_t>();
  if (hidden <= 0 || n_heads <= 0) throw std::runtime_error("draft hidden_size and num_attention_heads must be positive");
  const json hd_given = tl.value("head_dim", json());
  const int64_t hd = hd_given.is_number() && hd_given.get<int64_t>() != 0 ? hd_given.get<int64_t>() : hidden / n_heads;
  if (hd <= 0) throw std::runtime_error("draft head_dim must be positive");
  json rp = tl.value("rope_parameters", json());
  if (rp.is_null() || rp.empty()) rp = tl.value("rope_scaling", json());
  double theta = 10000.0;
  if (rp.is_object() && rp.contains("rope_theta") && rp["rope_theta"].is_number()) theta = rp["rope_theta"].get<double>();
  else if (tl.contains("rope_theta") && tl["rope_theta"].is_number()) theta = tl["rope_theta"].get<double>();
  return {{"hidden", hidden},      {"layers", layers},     {"n_heads", n_heads},
          {"n_kv", tl.at("num_key_value_heads")},          {"head_dim", hd},
          {"inter", tl.at("intermediate_size")},           {"eps", tl.at("rms_norm_eps")},
          {"rope_theta", theta}};
}
inline bool falsy(const nlohmann::json& v) {
  return v.is_null() || (v.is_boolean() && !v.get<bool>()) || (v.is_number() && v.get<double>() == 0) ||
         (v.is_string() && v.get<std::string>().empty()) || (v.is_array() && v.empty()) ||
         (v.is_object() && v.empty());
}

inline std::string inner_name(const std::string& name) { return name.rfind("model.", 0) == 0 ? name.substr(6) : name; }

bool canonical_layer_tensor(const std::string& name, std::string* out, bool layers_only);
inline void require_layer(const DirectoryDraft& draft, const std::string& prefix) {
  for (const char* n : {"in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm",
                        "attn.k_norm", "mlp.gate", "mlp.up", "mlp.down"})
    draft.require(prefix + n);
}

}  // namespace flyx::models::drafters
