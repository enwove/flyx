// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Model-agnostic vision interfaces between core/ and models/.
//
// Core decodes raw images into RGB888 pixels and manages caching, prefix matching,
// and GPU scheduling without knowing any model's resize rule, patch layout,
// placeholder token ID, or multimodal rotary position convention.  Each
// multimodal model under models/<arch>/ exposes a VisionAdapter implementing
// this contract.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace flyx::interfaces {

// Decoded image in RGB888 row-major order ([height, width, 3]).
struct RgbImage {
  std::vector<uint8_t> rgb;
  int width = 0;
  int height = 0;
  std::array<uint8_t, 32> sha256{};
};

// Placement and content identity of one media item inside an expanded prompt.
// Used by PrefixCache, Session, and SnapshotStore so distinct images producing
// identical placeholder token sequences never collide in KV/GDN caches.
struct MediaSpan {
  int32_t token_pos = 0;               // first token index in prompt
  int32_t n_tokens = 0;                // total tokens occupied (placeholders + any separators)
  std::array<uint8_t, 32> sha256{};    // content hash of source media
  int32_t grid_thw[3] = {0, 0, 0};     // (T, H_patches, W_patches) before spatial merge
  int32_t n_embeds = 0;                // embedding rows produced by the vision encoder
  // Offsets inside [0, n_tokens) receiving vision embeddings (empty = every offset 0..n_embeds-1).
  std::vector<int32_t> embed_offsets;
};

inline bool operator==(const MediaSpan& a, const MediaSpan& b) {
  return a.token_pos == b.token_pos && a.n_tokens == b.n_tokens && a.sha256 == b.sha256 &&
         a.grid_thw[0] == b.grid_thw[0] && a.grid_thw[1] == b.grid_thw[1] && a.grid_thw[2] == b.grid_thw[2] &&
         a.n_embeds == b.n_embeds && a.embed_offsets == b.embed_offsets;
}
inline bool operator!=(const MediaSpan& a, const MediaSpan& b) { return !(a == b); }

// Preprocessed patches and token expansion for one image (or video / tiled image).
struct PreparedMedia {
  std::vector<float> patches;          // [n_patches * patch_dim], row-major
  int32_t n_patches = 0;
  int32_t patch_dim = 0;
  std::vector<int32_t> grid_thw;       // [n_grids * 3]: (T, H_p, W_p) per image/tile
  std::vector<int32_t> cu_seqlens;     // [n_segs + 1]: bidirectional attention segment bounds
  int32_t n_embeds = 0;                // embedding rows emitted by the vision encoder
  // Expanded LLM token IDs replacing the single template placeholder token.
  std::vector<int32_t> tokens;
  // Offsets within `tokens` receiving vision embeddings (empty = every token 0..n_embeds-1).
  std::vector<int32_t> embed_offsets;
};

// Per-token position layout for prefill and decode.
struct PromptPositions {
  bool is_3d = false;                  // false: standard 1D sequential positions
  std::vector<int32_t> pos_3d;         // when is_3d: [prompt_len * 3] = (t, h, w) per token
  // When is_3d: [prompt_len + 1] decode rope_delta after prefix [0, p), for any non-media boundary p.
  std::vector<int32_t> pos_rope_deltas;
  int32_t rope_delta = 0;              // scalar offset added to 1D decode positions after prompt (= pos_rope_deltas[L])
};

// Architecture-specific vision preprocessing, placeholder expansion, and position rules.
class VisionAdapter {
 public:
  virtual ~VisionAdapter() = default;
  // Short descriptive name for logs and diagnostics (e.g. "qwen35_vit").
  virtual const char* name() const = 0;
  // Token ID emitted once by the chat template per image before expansion (e.g. <|image_pad|>).
  virtual int32_t placeholder_token() const = 0;
  // Output embedding width (matches LLM hidden_size).
  virtual int32_t out_hidden() const = 0;
  // Preprocess one decoded RGB image into encoder input patches and expanded LLM tokens.
  // max_tokens_budget: maximum visual tokens allowed for this image (0 = model default).
  virtual bool preprocess(const RgbImage& img, int max_tokens_budget,
                          PreparedMedia* out, std::string* err) const = 0;
  // Compute prefill position coordinates and decode rope_delta for an expanded prompt.
  virtual PromptPositions build_positions(std::span<const int32_t> tokens,
                                          std::span<const MediaSpan> spans) const = 0;
};

}  // namespace flyx::interfaces
