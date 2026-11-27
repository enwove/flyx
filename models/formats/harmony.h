// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// OpenAI's harmony chat format (gpt-oss): a reply is one or more messages, each
// [<|start|>assistant][<|channel|>CHANNEL][ to=RECIPIENT][<|constrain|>TYPE]<|message|>BODY<|end|>
// (the last one ending <|return|>, or <|call|> for a tool call): the analysis channel is its thinking, final and
// commentary without a recipient its text, a message to functions.NAME a call of that tool with the JSON body as its
// arguments.  The template renders the prompt (its Jinja, from the model file); this reads the replies.
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"

namespace flyx::models::formats {

bool is_harmony_template(std::string_view source);

std::unique_ptr<interfaces::OutputParser> make_harmony_parser(const std::vector<interfaces::ojson>* tools);

// The control tokens a reply is written with, by their text (interfaces::Chat::reply_tokens).  <|return|> and
// <|call|> end the reply (the model file's stop tokens), and are not among them.
std::vector<std::string> harmony_reply_tokens();

}  // namespace flyx::models::formats
