// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A reply's tokens as the client sees them: text shown as its characters
// complete, ending right before the first stop string, and how many of the
// tokens that text covers.
//
// The tokens the shown text covers are the published ones: what the client
// saw, so what the next request can continue from.  Text that may still grow
// into a stop string is held back until it cannot.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "core/tokenizer/tokenizer.h"

namespace flyx {

// Receives generated text (complete UTF-8) and the tokens behind it.  Return
// false to cancel.
using TextSink = std::function<bool(std::string_view text, const int32_t* tokens, int n)>;

class ReplyText {
 public:
  // stop: the reply ends before the first of these strings to complete (of those completing at the same byte, the
  // one that starts first), however the tokens are grouped into pushes.  sink may be empty.
  ReplyText(const Tokenizer& tok, std::vector<std::string> stop, TextSink sink);

  // The next generated tokens, in order (no stop token among them).
  void push(const int32_t* tokens, int n);
  // The reply is over: what was held back is shown, unless a stop string
  // ended the reply.  Not called for a cancelled reply (nothing more is shown).
  void end();

  // The stop string that ended the reply ("" while none has).
  const std::string& stop_string() const { return stop_string_; }
  // The sink asked to stop.
  bool refused() const { return refused_; }
  // Every token pushed, and how many of them the shown text covers.
  const std::vector<int32_t>& tokens() const { return tokens_; }
  size_t published() const { return published_; }

 private:
  void show(std::string_view text);
  void add(const std::string& text);

  const Tokenizer& tok_;
  std::vector<std::string> stop_;
  TextSink sink_;
  StreamDecoder decoder_;
  std::vector<int32_t> tokens_;
  std::vector<size_t> token_end_;  // byte offset of the text after each token
  size_t bytes_ = 0, shown_bytes_ = 0, published_ = 0;
  std::string held_;               // decoded, not shown yet: might be the start of a stop string
  size_t longest_stop_ = 0;
  std::string stop_string_;
  bool refused_ = false;
};

}  // namespace flyx
