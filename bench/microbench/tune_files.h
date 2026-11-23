// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The matrices of a model file, for the tuner (tune.cu; read in tune_files.cpp).
#pragma once

#include <string>
#include <vector>

namespace flyx::bench {

struct FileMatrix {
  int dtype, N, K;  // flyx_dtype, the matrix's rows (output columns) and K
};
// The 2-D tensors of a model file (GGUF or .flyx) a GEMM multiplies: every one but an embedding that is not the
// head too.  False and *err: the file could not be read.
bool file_matrices(const std::string& path, std::vector<FileMatrix>* out, std::string* err);

}  // namespace flyx::bench
