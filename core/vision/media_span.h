// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Core helpers for matching, slicing, and boundary-checking multimodal MediaSpans
// across Session, SnapshotStore, and Generation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/interfaces/vision.h"

namespace flyx::vision {

// Longest common token prefix between (a_tok, a_med) and (b_tok, b_med) where every
// overlapping MediaSpan is identical and the returned length never splits inside a MediaSpan.
inline size_t media_prefix_match(const std::vector<int32_t>& a_tok,
                                 const std::vector<interfaces::MediaSpan>& a_med,
                                 const std::vector<int32_t>& b_tok,
                                 const std::vector<interfaces::MediaSpan>& b_med) {
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
inline bool media_prefix_equal(const std::vector<int32_t>& prefix_tok,
                               const std::vector<interfaces::MediaSpan>& prefix_med,
                               const std::vector<int32_t>& prompt_tok,
                               const std::vector<interfaces::MediaSpan>& prompt_med) {
  return prefix_tok.size() <= prompt_tok.size() &&
         media_prefix_match(prefix_tok, prefix_med, prompt_tok, prompt_med) == prefix_tok.size();
}

// Return all MediaSpans completely contained in [0, pos).
inline std::vector<interfaces::MediaSpan> slice_media_prefix(const std::vector<interfaces::MediaSpan>& media,
                                                             size_t pos) {
  std::vector<interfaces::MediaSpan> out;
  for (const auto& m : media)
    if (size_t(m.token_pos) + size_t(m.n_tokens) <= pos) out.push_back(m);
  return out;
}

// True if token position `pos` falls strictly inside any MediaSpan (unsafe split/checkpoint point).
inline bool inside_media_span(const std::vector<interfaces::MediaSpan>& media, int32_t pos) {
  for (const auto& m : media)
    if (pos > m.token_pos && pos < m.token_pos + m.n_tokens) return true;
  return false;
}

}  // namespace flyx::vision
