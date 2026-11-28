// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// BPE tokenizers: byte-level (Qwen3.5, Qwen3 with the Qwen2 split, MiniCPM5, Inkling-2, gpt-oss) and SentencePiece's
// (Gemma 4).
//
// Pipeline (must match HuggingFace `tokenizers` for these models):
//   added tokens split out first (leftmost-longest)  ->  optional NFC on the rest  ->
//   regex pre-tokenizer (hand-written, see pretokenize())  ->  BPE per piece;
// a raw prompt then gets the post-processor's tokens before it (prefix()).
// Gemma 4's (pretokenizer "gemma4"): no NFC; each space written U+2581 (SentencePiece's metaspace), the text cut where
// newlines begin and end (no token mixes a newline with another character, so the pieces merge as the whole would),
// a piece's symbols its characters by their text, one <0xNN> byte token for each byte of a character the vocabulary
// lacks (and of a byte no character holds), then the merges by rank.
// The tables come from the model file (python/convert/tokenizer_export.py).
//
// A chat prompt mixes the template's own text with the client's strings.
// Special tokens (<|im_start|>, <|im_end|>, ...) count only where the
// template wrote them; in a client's string the same characters are plain
// text, so a message cannot open or close turns.  Added tokens that are not
// special (<think>, <tool_call>) are read everywhere, as HuggingFace does.
#pragma once

#include <cstdint>
#include <functional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/interfaces/chat.h"

namespace flyx {

class ModelFile;
using interfaces::TextSpan;

// A tokenizer's pre-tokenizer split, as python/convert/flyx_convert/tokenizer_export.py names it:
//   kQwen35    letters with their combining marks, one number at a time
//   kQwen2     Qwen2's and Qwen3's: letters without marks, one number at a time
//   kMiniCPM5  numbers cut into groups of up to three from the left first, then Qwen2's split on each piece, a run
//              of numbers as one
//   kInkling2  GPT-4o / o200k case-aware letter runs with optional contractions, numbers in groups of 1..3,
//              punctuation with trailing newlines or slashes
enum class Pretokenizer { kQwen35, kQwen2, kMiniCPM5, kInkling2 };

class Tokenizer {
 public:
  bool load(const ModelFile& file, std::string* err);

  // literal: sorted spans where special tokens are read (the template's text);
  // null: everywhere (a raw prompt the caller wrote).
  std::vector<int32_t> encode(std::string_view text, const std::vector<TextSpan>* literal = nullptr) const;
  // A raw prompt's tokens (text the caller wrote, no chat template): the prefix, then the text's, special tokens
  // read everywhere: what transformers' tokenizer gives with its special tokens.
  std::vector<int32_t> encode_prompt(std::string_view text) const;
  // The tokens the model's tokenizer puts before a text (its post-processor's: MiniCPM5's <s>; Qwen's: none), each a
  // special token.  A chat template writes its own.
  const std::vector<int32_t>& prefix() const { return prefix_; }
  std::string decode(const std::vector<int32_t>& ids) const;
  const std::string& token_bytes(int32_t id) const;
  int32_t n_tokens() const { return static_cast<int32_t>(tokens_.size()); }
  // Id of an added token by content, or -1.
  int32_t added_id(const std::string& content) const;
  // An added token marked special (<|im_start|>, <|im_end|>, ...; not <think>).
  bool special(int32_t id) const { return id >= 0 && size_t(id) < special_.size() && special_[size_t(id)]; }

  // Exposed for tests: split (normalized) text into pre-tokenizer pieces.
  static std::vector<std::string> pretokenize(std::string_view text, Pretokenizer kind = Pretokenizer::kQwen35);
  static std::string nfc(std::string_view text);

 private:
  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept { return std::hash<std::string_view>{}(s); }
    size_t operator()(const std::string& s) const noexcept { return std::hash<std::string_view>{}(s); }
  };
  void encode_pieces(const std::vector<std::string_view>& pieces, std::vector<int32_t>* out) const;
  std::vector<int32_t> encode_uncached(std::string_view piece) const;
  void metaspace_symbols(std::string_view piece, std::vector<int32_t>* sym) const;
  void merge(std::vector<int32_t>* sym) const;

  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int32_t, StringHash, std::equal_to<>> by_bytes_;
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;  // (left,right) -> (rank, merged)
  int32_t byte_id_[256] = {};
  struct Added {
    std::string content;
    int32_t id;
    bool special;
  };
  std::vector<Added> added_;  // longest first
  bool added_first_[256] = {};  // the bytes an added token begins with
  std::vector<bool> special_;
  std::vector<int32_t> prefix_;
  bool ignore_merges_ = false;
  bool metaspace_ = false;  // Gemma 4's: by_bytes_ then maps a token's text, its spaces as U+2581, to its id
  Pretokenizer pretokenizer_ = Pretokenizer::kQwen35;
  bool normalize_nfc_ = true;
  mutable std::shared_mutex cache_mu_;  // encode() may run concurrently across HTTP threads
  // Pieces met before: words come back, so most are.  Long pieces (a run of letters without a space) are rare and
  // would take the room of thousands of words: they are merged each time.
  static constexpr size_t kCachedPieces = 200000, kCachedPieceBytes = 256;
  mutable std::unordered_map<std::string, std::vector<int32_t>, StringHash, std::equal_to<>> cache_;
};

// Turns a token stream into text without splitting UTF-8 characters.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer* tok) : tok_(tok) {}
  // Returns the text that became complete with these tokens.
  std::string feed(const int32_t* ids, int n);
  std::string flush();

 private:
  const Tokenizer* tok_;
  std::string pending_;
};

// Token positions at the given byte offsets of the decoded text of `ids`
// (offsets that fall inside a token map to the token's start).
std::vector<int32_t> token_positions(const Tokenizer& tok, const std::vector<int32_t>& ids,
                                     const std::vector<size_t>& offsets);

}  // namespace flyx
