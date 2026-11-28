// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A limit on how many tokens a reply thinks (Anthropic's thinking.budget_tokens): its thinking sections' tokens
// counted, all of them together, and once the budget is spent the section closed by the format's own marker, the
// model writing its answer after it.  The way SGLang's, TensorRT-LLM's and vLLM's thinking budgets end thinking
// (docs/prior-art.md): past the budget only the marker's next token may come.  Here that is the reply's constraint
// the way a grammar is (Generation::rejected / choose_again): a token that may not come is chosen again under the
// mask that allows only the marker's, so a speculative round gives the tokens plain decoding gives.
#pragma once

#include <cstdint>
#include <vector>

namespace flyx {

class ThinkingBudget {
 public:
  ThinkingBudget() = default;
  // open, close: the tokens that open and close a thinking section (the format's markers, as the model's tokenizer
  // writes them); budget: the most tokens inside the sections, the closing marker's own not counted; inside: the
  // prompt leaves a section open.  An empty close, or a budget below 0: no limit (active() false).
  ThinkingBudget(std::vector<int32_t> open, std::vector<int32_t> close, int budget, bool inside);

  bool active() const { return budget_ >= 0 && !close_.empty(); }
  // Whether token t may come next.
  bool allows(int32_t t) const;
  // Token t came (allows(t) held).  The closing marker's tokens are not counted.
  void take(int32_t t);
  // The one token that must come next, or -1 where any may.
  int32_t forced() const;
  int spent() const { return spent_; }    // tokens thought so far
  bool closed_early() const { return closed_early_; }  // a section was closed by the budget

 private:
  std::vector<int32_t> open_, close_;
  int budget_ = -1;
  bool inside_ = false;
  int spent_ = 0;
  size_t open_at_ = 0, close_at_ = 0;  // how much of the marker the last tokens matched
  bool forced_now_ = false;            // the budget forced a token of the marker being written
  bool closed_early_ = false;
};

}  // namespace flyx
