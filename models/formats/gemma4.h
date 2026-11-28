// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Gemma 4's chat format: a reply is [<|channel>thought\nREASONING<channel|>] TEXT [<|tool_call>call:NAME{ARGS}<tool_call|>]...
// ended by <turn|> (or by <|tool_response>, where the reply calls a tool: the model file's stop tokens).  ARGS are
// Gemma's values: keys bare, strings between <|"|> marks (no escapes), numbers, true / false / null, lists [..] and
// objects {..} of them; read back as JSON.  The model may open a call inside an unclosed thought, write empty or
// stray channel marks: those go.  The template renders the prompt (its Jinja, from the model file); this reads the
// replies and constrains them (XGrammar's "gemma" style for the arguments).
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"
#include "models/formats/tags.h"

namespace flyx::models::formats {

bool is_gemma4_template(std::string_view source);

// in_reasoning: the prompt ends inside an open thought (a tool result's turn with thinking on).
std::unique_ptr<interfaces::OutputParser> make_gemma4_parser(bool in_reasoning,
                                                             const std::vector<interfaces::ojson>* tools);

// Whether the generation prompt leaves a thought open (its last <|channel> not closed).
bool gemma4_opens_reasoning(std::string_view gen);

// Gemma 4's tag syntax for reply grammars: thinking between <|channel> and <channel|>, calls as above.
extern const TagSyntax kGemma4Syntax;

// A Gemma value (the arguments of a call, ARGS above) as JSON; null where it is not one.
interfaces::ojson parse_gemma4_value(std::string_view s);

}  // namespace flyx::models::formats
