// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/reply_text.h"

#include <algorithm>

namespace flyx {

ReplyText::ReplyText(const Tokenizer& tok, std::vector<std::string> stop, TextSink sink)
    : tok_(tok), stop_(std::move(stop)), sink_(std::move(sink)), decoder_(&tok) {
  for (const auto& s : stop_) longest_stop_ = std::max(longest_stop_, s.size());
}

void ReplyText::show(std::string_view text) {
  if (text.empty()) return;
  shown_bytes_ += text.size();
  size_t k = published_;
  while (k < tokens_.size() && token_end_[k] <= shown_bytes_) ++k;
  if (sink_ && !sink_(text, tokens_.data() + published_, static_cast<int>(k - published_))) refused_ = true;
  published_ = k;
}

void ReplyText::add(const std::string& text) {
  held_ += text;
  if (longest_stop_ == 0) {
    show(held_);
    held_.clear();
    return;
  }
  // The stop string that ends first, as if the text came a byte at a time (so however the tokens are grouped: a
  // speculative round's width changes nothing); of those ending at the same byte, the one that starts first.  A
  // stop string's first occurrence is the one that ends first.
  size_t first = std::string::npos, first_end = std::string::npos;
  for (const auto& s : stop_) {
    if (s.empty()) continue;
    const size_t at = held_.find(s);
    if (at == std::string::npos) continue;
    const size_t end = at + s.size();
    if (end < first_end || (end == first_end && at < first)) {
      first = at;
      first_end = end;
      stop_string_ = s;
    }
  }
  if (first != std::string::npos) {
    show(std::string_view(held_).substr(0, first));
    held_.clear();
    return;
  }
  size_t keep = 0;  // the longest tail that could still grow into a stop string
  for (const auto& s : stop_) {
    if (s.empty()) continue;
    for (size_t l = std::min(s.size() - 1, held_.size()); l > keep; --l)
      if (held_.compare(held_.size() - l, l, s, 0, l) == 0) {
        keep = l;
        break;
      }
  }
  show(std::string_view(held_).substr(0, held_.size() - keep));
  held_.erase(0, held_.size() - keep);
}

void ReplyText::push(const int32_t* tokens, int n) {
  if (n <= 0 || !stop_string_.empty()) return;
  for (int i = 0; i < n; ++i) {
    tokens_.push_back(tokens[i]);
    bytes_ += tok_.token_bytes(tokens[i]).size();
    token_end_.push_back(bytes_);
  }
  add(decoder_.feed(tokens, n));
}

void ReplyText::end() {
  if (!stop_string_.empty()) return;
  add(decoder_.flush());
  if (!stop_string_.empty()) return;
  show(held_);
  held_.clear();
  // Tokens without text (padding ids past the tokenizer) at the very end.
  if (published_ < tokens_.size()) {
    if (sink_) sink_({}, tokens_.data() + published_, static_cast<int>(tokens_.size() - published_));
    published_ = tokens_.size();
  }
}

}  // namespace flyx
