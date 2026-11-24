// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Grammar-constrained decoding: the replies a request allows
// (tool calls in the chat format's own syntax, with arguments that fit the
// tools' schemas), checked token by token on the host with XGrammar
// (third_party/xgrammar).
//
// The engine decodes as usual, speculation included, and feeds every token
// to the request's matcher.  The first token the grammar rejects is undone
// and chosen again with the grammar's mask.  Token choice with a mask keeps
// the unmasked choice whenever that one is allowed (ops/sampling/sampling.h),
// so the output is exactly what masking every token would give, at the cost
// of one more round per rejected token: a reply that follows the format runs
// at full speed.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace flyx {

class Tokenizer;

// One reply's progress through a grammar.
class GrammarMatcher {
 public:
  ~GrammarMatcher();
  // false: the grammar does not allow the token here (the state is unchanged).
  bool accept(int32_t token);
  // The tokens allowed next: bit v % 32 of words[v / 32] (Grammar::words() words).
  void allowed(uint32_t* words);
  // The reply may end here (a stop token is allowed now).
  bool can_end();

 private:
  friend class Grammar;
  struct Impl;
  explicit GrammarMatcher(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// The vocabulary as grammars see it, and the grammars compiled for it.
class Grammar {
 public:
  Grammar();
  ~Grammar();
  // Token text from the tokenizer; special tokens never match text, but for
  // `markers` (the ones the chat format writes replies with: interfaces::
  // Chat::reply_tokens), which match theirs; stop_ids end a reply; ids past
  // the tokenizer (padding up to `vocab`) never match.
  void init(const Tokenizer& tok, int vocab, std::vector<int32_t> stop_ids, const std::vector<int32_t>& markers = {});
  bool ready() const;
  int words() const;  // bitmask words of the vocabulary
  // A matcher for one reply under an XGrammar structural tag (JSON text).
  // Compiled grammars are kept by their text (the last 16), so a client that
  // sends the same tools every turn compiles them once; *cached: this one was
  // kept.  Null with *err when the tag does not compile.  Any thread: a call
  // that finds its grammar kept never waits for another's compile.
  std::unique_ptr<GrammarMatcher> matcher(const std::string& structural_tag, std::string* err,
                                          bool* cached = nullptr);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace flyx
