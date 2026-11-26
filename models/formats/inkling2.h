// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Inkling-2 chat format: repeatable section replies wrapped in
// <|message_model|>[recipient]<|content_{thinking,text,invoke_tool_json}|>...<|end_message|>
// ending with <|content_model_end_sampling|>.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"

namespace flyx::models::formats {

using interfaces::ojson;
using interfaces::OutputEvent;
using interfaces::ReplyRules;

bool is_inkling2_template(std::string_view source);

std::unique_ptr<interfaces::OutputParser> make_inkling2_parser(const std::vector<ojson>* tools);

std::string inkling2_reply_grammar(const std::vector<ojson>& tools, const ReplyRules& rules);

// The control tokens a reply is written with, by their text (interfaces::Chat::reply_tokens): the ones the grammar
// above names.  <|content_model_end_sampling|> is not among them: it ends the reply, as the model's stop token.
std::vector<std::string> inkling2_reply_tokens();

}  // namespace flyx::models::formats
