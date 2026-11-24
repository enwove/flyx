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

// Longest common token prefix between (a_tok, a_med) and (b_tok, b_med) where every
// overlapping MediaSpan is identical and the returned length never splits inside a MediaSpan.
inline size_t media_prefix_match(const std::vector<int32_t>& a_tok, const std::vector<MediaSpan>& a_med,
                                 const std::vector<int32_t>& b_tok, const std::vector<MediaSpan>& b_med) {
  size_t L = 0;
  const size_t n = a_tok.size() < b_tok.size() ? a_tok.size() : b_tok.size();
  while (L < n && a_tok[L] == b_tok[L]) ++L;
  if (a_med.empty() && b_med.empty()) return L;
  size_t i = 0;
  while (i < a_med.size() && i < b_med.size()) {
    const auto& ma = a_med[i];
    const auto& mb = b_med[i];
    const size_t a_start = size_t(ma.token_pos);
    const size_t b_start = size_t(mb.token_pos);
    if (a_start >= L && b_start >= L) break;
    const size_t a_end = a_start + size_t(ma.n_tokens);
    const size_t b_end = b_start + size_t(mb.n_tokens);
    if (a_end > L || b_end > L || ma != mb) {
      const size_t cut = a_start < b_start ? a_start : b_start;
      if (cut < L) L = cut;
      return L;
    }
    ++i;
  }
  if (i < a_med.size() && size_t(a_med[i].token_pos) < L) L = size_t(a_med[i].token_pos);
  if (i < b_med.size() && size_t(b_med[i].token_pos) < L) L = size_t(b_med[i].token_pos);
  return L;
}

// True iff (prefix_tok, prefix_med) is an exact token+media prefix of (prompt_tok, prompt_med).
inline bool media_prefix_equal(const std::vector<int32_t>& prefix_tok, const std::vector<MediaSpan>& prefix_med,
                               const std::vector<int32_t>& prompt_tok, const std::vector<MediaSpan>& prompt_med) {
  return prefix_tok.size() <= prompt_tok.size() &&
         media_prefix_match(prefix_tok, prefix_med, prompt_tok, prompt_med) == prefix_tok.size();
}

// Return all MediaSpans completely contained in [0, pos).
inline std::vector<MediaSpan> slice_media_prefix(const std::vector<MediaSpan>& media, size_t pos) {
  std::vector<MediaSpan> out;
  for (const auto& m : media)
    if (size_t(m.token_pos) + size_t(m.n_tokens) <= pos) out.push_back(m);
  return out;
}

// True if token position `pos` falls strictly inside any MediaSpan (unsafe split/checkpoint point).
inline bool inside_media_span(const std::vector<MediaSpan>& media, int32_t pos) {
  for (const auto& m : media)
    if (pos > m.token_pos && pos < m.token_pos + m.n_tokens) return true;
  return false;
}

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
  int32_t rope_delta = 0;              // scalar offset added to 1D decode positions after prompt
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
