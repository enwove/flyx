// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Chat prompts from the model's own template: the
// Jinja source the file carries (or one given at load), rendered as
// transformers' apply_chat_template renders it, by the renderer written for
// that template when its chat format has one (models/formats/), else by
// llama.cpp's Jinja engine (third_party/jinja).
//
// Besides the text, a render says which bytes the template wrote itself (the
// client's strings are marked as they pass through the template), so the
// tokenizer reads special tokens only there; where a later request is likely
// to branch (a checkpoint is kept there); and whether the generation prompt
// left a thinking block open.  How a reply is read back is the chat format's.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/interfaces/chat.h"
#include "core/tokenizer/tokenizer.h"
#include "nlohmann/json.hpp"

namespace flyx {

using ojson = nlohmann::ordered_json;
using interfaces::ChatMessage;
using interfaces::ChatOptions;
using interfaces::ContentPart;
using interfaces::ImageInput;
using interfaces::ReplyRules;
using interfaces::OutputEvent;
using interfaces::OutputParser;
using interfaces::RenderedChat;
using interfaces::RenderedImage;
using interfaces::ToolCall;

// A stable prefix requested by the client, expressed in the normalized conversation.
// With messages, the sizes describe the part of its last message already included.
// With no messages, tool_count definitions belong to the prefix.
struct ChatCachePoint {
  size_t messages = 0, content_bytes = 0, reasoning_bytes = 0, calls = 0, tool_count = 0;
  size_t parts = 0;  // when > 0, number of leading ContentParts in the last message
};

class ChatTemplate {
 public:
  ChatTemplate();
  ~ChatTemplate();
  // The model's template, or `template_override` (a Jinja source).  false with
  // *err when there is none or it does not parse: chat is unavailable then
  // (raw prompts still work), and render() says why.  native: use the chat
  // format's own renderer where it has one (false: the Jinja engine always).
  bool load(const ModelFile& file, const std::string& template_override, std::string* err, bool native = true);
  bool ok() const;
  const char* format() const;  // the chat format (interfaces::Chat::format), or "none"
  bool native() const;         // renders with the format's own renderer

  // tools: function definitions as the client sent them (OpenAI format).
  // branch_points: also find the boundaries and the generation prompt (two
  // renders of a one-message probe; on a first request, up to two renders of
  // a prefix).  false with *err when the template rejects the messages.
  // Any thread.
  bool render(const std::vector<ChatMessage>& msgs, const std::vector<ojson>& tools, const ChatOptions& opt,
              RenderedChat* out, std::string* err, bool branch_points = true,
              const std::vector<ChatCachePoint>& cache_points = {}) const;
  // A parser for the reply to a rendered prompt.
  std::unique_ptr<OutputParser> parser(const RenderedChat& r, const std::vector<ojson>* tools) const;
  // The grammar the reply to a rendered prompt must match (interfaces::Chat::reply_grammar), "" for none.
  std::string reply_grammar(const std::vector<ojson>& tools, const ReplyRules& rules, const RenderedChat& r) const;
  // The special tokens replies are written with (interfaces::Chat::reply_tokens); none without a template.
  std::vector<std::string> reply_tokens() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string error_ = "chat template not loaded";
};

}  // namespace flyx
