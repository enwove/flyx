// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The matrices of a model file the tuner weighs its rules by (tune.cu): plain C++, so the file reader's JSON stays
// out of the CUDA compiler.
#include <string>
#include <vector>

#include "bench/microbench/tune_files.h"
#include "core/modelfile/modelfile.h"

namespace flyx::bench {

bool file_matrices(const std::string& path, std::vector<FileMatrix>* out, std::string* err) {
  ModelFile file;
  if (!file.open(path, err)) return false;
  // The embedding's rows are read, not multiplied, unless it is the head too (no output.weight).
  const bool own_head = file.tensor("output.weight") != nullptr;
  for (const std::string& name : file.tensor_names()) {
    const flyx_tensor_view* t = file.tensor(name);
    if (!t || t->ndim != 2 || (own_head && name.rfind("token_embd", 0) == 0)) continue;
    out->push_back({t->dtype, int(t->shape[0]), int(t->shape[1])});
  }
  return true;
}

bool file_experts(const std::string& path, std::vector<FileExperts>* out, std::string* err) {
  ModelFile file;
  if (!file.open(path, err)) return false;
  const std::string suffix = "ffn_gate_up_exps.weight", key = file.arch() + ".expert_used_count";
  for (const std::string& name : file.tensor_names()) {
    const flyx_tensor_view* t = file.tensor(name);
    if (!t || t->ndim != 3 || name.size() < suffix.size() ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    const nlohmann::json& h = file.header();
    if (!h.contains("gguf") || !h.at("gguf").contains(key) || !h.at("gguf").at(key).is_number_integer()) {
      *err = "expert matrices without " + key;
      return false;
    }
    out->push_back({t->dtype, int(t->shape[0]), int(t->shape[1]), int(t->shape[2]), h.at("gguf").at(key).get<int>()});
  }
  return true;
}

}  // namespace flyx::bench
