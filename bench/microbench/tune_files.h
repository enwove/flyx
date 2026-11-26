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

struct FileExperts {
  int dtype, n_experts, N, K, top_k;  // flyx_dtype, experts, an expert's rows (gate and up: 2 x intermediate), K, and
                                      // the experts a token takes
};
// The fused gate|up expert matrices of a GGUF file's MoE layers (blk.*.ffn_gate_up_exps.weight: experts x 2
// intermediate x hidden, as models/inkling2 reads them), the experts a token takes from <arch>.expert_used_count.
// False and *err: the file could not be read, or it has such matrices and not that key.
bool file_experts(const std::string& path, std::vector<FileExperts>* out, std::string* err);

}  // namespace flyx::bench
