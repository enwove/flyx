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

}  // namespace flyx::bench
