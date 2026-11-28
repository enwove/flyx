// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The matrices of a model file the tuner weighs its rules by (tune.cu): plain C++, so the file reader's JSON stays
// out of the CUDA compiler.
#include <string>
#include <vector>

#include "bench/microbench/tune_files.h"
#include "core/modelfile/modelfile.h"
#include "flyx/backend_abi.h"

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
  const std::string fused = "ffn_gate_up_exps.weight", gate = "ffn_gate_exps.weight", up = "ffn_up_exps.weight";
  const std::string key = file.arch() + ".expert_used_count";
  auto ends = [](const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  for (const std::string& name : file.tensor_names()) {
    const flyx_tensor_view* t = file.tensor(name);
    if (!t || t->ndim != 3) continue;
    // gate and up apart (gpt-oss's): the pair as the engine stacks them, an expert's gate rows then its up rows
    int rows_per = 1;
    if (ends(name, gate)) {
      const flyx_tensor_view* u = file.tensor(name.substr(0, name.size() - gate.size()) + up);
      if (!u || u->dtype != t->dtype || u->ndim != 3 || u->shape[0] != t->shape[0] || u->shape[1] != t->shape[1] ||
          u->shape[2] != t->shape[2]) {
        *err = name + " without an up matrix of its shape";
        return false;
      }
      rows_per = 2;
    } else if (!ends(name, fused)) {
      continue;
    }
    const nlohmann::json& h = file.header();
    if (!h.contains("gguf") || !h.at("gguf").contains(key) || !h.at("gguf").at(key).is_number_integer()) {
      *err = "expert matrices without " + key;
      return false;
    }
    const int n_experts = int(t->shape[0]), N = rows_per * int(t->shape[1]), K = int(t->shape[2]);
    const int top_k = h.at("gguf").at(key).get<int>();
    const int blk = flyx_block_values(t->dtype);
    if (n_experts <= 0 || top_k <= 0 || top_k > n_experts || N <= 0 || (N & 1) != 0 || K <= 0 ||
        (blk > 0 && K % blk != 0)) {
      *err = "invalid expert matrix " + name + " (" + std::to_string(n_experts) + "x" + std::to_string(N) + "x" +
             std::to_string(K) + ", top_k=" + std::to_string(top_k) + ")";
      return false;
    }
    out->push_back({t->dtype, n_experts, N, K, top_k});
  }
  return true;
}

}  // namespace flyx::bench
