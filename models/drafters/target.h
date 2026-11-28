// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Target model contract and optional MTP hooks handed to models/drafters/ so any model architecture
// can attach arbitrary speculative drafters without duplicating companion parsing or block builders.
#pragma once

#include <functional>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

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
  float softcap = 0.0f;    // target logit softcap (default fallback for drafters without explicit key)
  bool gguf = false;       // true when the target model was opened from GGUF
  bool has_post_norm_tap = false;  // architecture calls emit_post_norm_taps() after its own final residual mix
  bool dspark_taps = false;        // architecture's programs give a DSpark drafter its taps: emit_pre_visit_taps()
                                   // before every layer and emit_post_norm_taps(); without them it drafts from
                                   // rows nothing wrote, so its file is refused

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
  // An architecture whose MTP drafter is a block of its own (Gemma 4's assistant) sets these instead of the two
  // above: whether a companion file's header is its drafter (throwing on one made for another model), and the drafter
  // block built from that companion.
  std::function<bool(const nlohmann::json& companion_header)> claims;
  std::function<int(int companion)> build;
};

}  // namespace flyx::models::drafters
