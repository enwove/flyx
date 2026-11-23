// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Replies written with tags, read back as they stream: thinking in
// <think>...</think>, tool calls in <tool_call>...</tool_call> with a body
// in the format's call language (calls.h).
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"
#include "models/formats/calls.h"

namespace flyx::models::formats {

using interfaces::OutputEvent;

struct TagSyntax {
  bool thinking = true;        // <think> blocks exist
  CallReader calls = nullptr;  // null: no tool calls, <tool_call> is text
  const CallSyntax* written = nullptr;  // how calls are written (grammars); with `calls`
};

// The structural tag (XGrammar JSON) a reply in this syntax must match; see
// interfaces::Chat::reply_grammar.
std::string reply_grammar(const TagSyntax& syntax, const std::vector<ojson>& tools,
                          const interfaces::ReplyRules& rules, bool in_reasoning);

class TaggedParser : public interfaces::OutputParser {
 public:
  // in_reasoning: the prompt ends inside an open <think> block.
  // tools: the request's tool definitions (to type parameter values).
  TaggedParser(TagSyntax syntax, bool in_reasoning, const std::vector<ojson>* tools)
      : syntax_(syntax), mode_(in_reasoning ? kThink : kText), tools_(tools) {}
  std::vector<OutputEvent> feed(std::string_view text) override;
  std::vector<OutputEvent> finish() override;
  int tool_calls() const override { return n_calls_; }

 private:
  enum Mode { kThink, kText, kCall };
  void run(std::vector<OutputEvent>* ev, bool final);
  void emit(std::vector<OutputEvent>* ev, OutputEvent::Kind k, const std::string& s);

  TagSyntax syntax_;
  Mode mode_;
  const std::vector<ojson>* tools_;
  std::string buf_;            // not yet classified
  std::string nl_;             // newlines held back (dropped before </think> and valid <tool_call>)
  std::string call_nl_;        // newlines saved across an open <tool_call> for malformed-call fallback
  std::unique_ptr<CallStream> call_stream_;  // active incremental parser while mode_ == kCall
  std::string call_raw_;                     // raw bytes inside <tool_call> until call_stream_->started()
  bool text_started_ = false;  // answer text began (leading newlines are dropped)
  bool think_started_ = false;
  int n_calls_ = 0;
};

}  // namespace flyx::models::formats
