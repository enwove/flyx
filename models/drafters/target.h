// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Target model contract and optional MTP hooks handed to models/drafters/ so any model architecture
// can attach arbitrary speculative drafters without duplicating companion parsing or block builders.
#pragma once

#include <functional>
#include <string>
#include <utility>

namespace flyx::models::drafters {

// Target model dimensions and block handles shared by all speculative drafters.
struct TargetSpec {
  std::string arch;        // target architecture name as the file reports it ("qwen35", "qwen3", "llama", ...)
  int hidden = 0;          // residual stream width
  int vocab = 0;           // effective (unpadded) vocabulary size
  int vocab_padded = 0;    // storage vocabulary size (0 means same as vocab)
  int layers = 0;          // logical layer / visit count (1..layers are valid tap indices)
  int embed = -1;          // target FLYX_BLOCK_EMBED index (populated before attach)
  int head = -1;           // target FLYX_BLOCK_HEAD index (populated before attach)
  bool gguf = false;       // true when the target model was opened from GGUF

  int padded_vocab() const { return vocab_padded > 0 ? vocab_padded : vocab; }
};

// Optional architecture callbacks for targets that can host an embedded or sidecar-GGUF MTP layer.
struct MtpHooks {
  bool embedded = false;   // target model file itself contains an MTP layer
  double norm_offset = 0.0;
  double eps = 1e-6;
  // Maps canonical MTP top-level names ("mtp.fc", "mtp.norm_emb", "mtp.norm_hidden", "mtp.norm")
  // to file or companion tensor names (`companion == -1` for embedded MTP).
  std::function<std::string(const std::string& canonical, int companion)> tensor_name;
  // Builds the MTP layer's attention and MLP blocks in Tables and returns {attn_block, mlp_block}.
  std::function<std::pair<int, int>(int companion)> build_layer;
};

}  // namespace flyx::models::drafters
