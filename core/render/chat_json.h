// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Chat requests in the OpenAI and Anthropic JSON formats -> ChatMessage list
// and tool definitions for the template.
#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/render/chat.h"

namespace flyx {

// OpenAI chat messages: [{"role", "content" (string | parts), "reasoning_content",
// "tool_calls": [{"id", "function": {"name", "arguments" (JSON text)}}], "tool_call_id"}].
bool messages_from_openai(const ojson& messages, std::vector<ChatMessage>* out, std::string* err);
// OpenAI tools: [{"type": "function", "function": {...}}]; kept as given.
bool tools_from_openai(const ojson& tools, std::vector<ojson>* out, std::string* err);

// Anthropic Messages request: "system" (string | text blocks), "messages" with
// text / thinking / tool_use / tool_result blocks, "tools" [{name, description, input_schema}].
// Tools come back in the OpenAI format the template expects.
//
// Claude Code starts the system prompt with a billing line
// ("x-anthropic-billing-header: cc_version=...; cch=...;") that changes on
// every request: kept, the prompt differs from the session at its first line
// and every turn is computed again from the start. It is metadata for
// Anthropic's billing, not text for the model, so it is left out unless
// `keep_billing_header`; *dropped (if given) says whether it was there.
bool chat_from_anthropic(const ojson& request, std::vector<ChatMessage>* out, std::vector<ojson>* tools,
                         std::string* err, bool keep_billing_header = false, bool* dropped = nullptr,
                         std::vector<ChatCachePoint>* cache_points = nullptr);
// Whether a line is Claude Code's billing line.
bool is_billing_line(std::string_view line);

// OpenAI tool_choice ("auto" | "none" | "required" | {"type": "function",
// "function": {"name"}}) and parallel_tool_calls; a forced name must be one
// of the tools.
bool rules_from_openai(const ojson& request, const std::vector<ojson>& tools, ReplyRules* out, std::string* err);
// Anthropic tool_choice ({"type": "auto" | "any" | "tool" | "none", "name",
// "disable_parallel_tool_use"}).
bool rules_from_anthropic(const ojson& request, const std::vector<ojson>& tools, ReplyRules* out, std::string* err);

// OpenAI Responses: the items of a conversation (input items, and the output
// items of the responses it continues) -> ChatMessages, `instructions` first
// as a system message. Items: messages (user, assistant, system, developer;
// text parts), reasoning, function_call, function_call_output. Consecutive
// assistant items (reasoning, text, calls) are one assistant turn, as the
// model wrote them.
bool messages_from_responses(const std::string& instructions, const ojson& items, std::vector<ChatMessage>* out,
                             std::string* err);
// Chat templates take flat function names. Keep the namespace of each qualified name to return
// the original namespace and member separately on the Responses wire.
using ResponsesNamespaces = std::map<std::string, std::string>;
bool tools_from_responses(const ojson& tools, std::vector<ojson>* out, std::string* err,
                          ResponsesNamespaces* namespaces = nullptr);
// Responses tool_choice ("auto" | "none" | "required" | {"type": "function",
// "name"}), parallel_tool_calls and text.format (a JSON answer).
bool rules_from_responses(const ojson& request, const std::vector<ojson>& tools, ReplyRules* out, std::string* err,
                          const ResponsesNamespaces& namespaces = {});
// The text of a reasoning item: its reasoning_text parts, else its summary,
// else an encrypted_content this server wrote (kReasoningMark, then the text).
std::string reasoning_text(const ojson& item);
inline constexpr std::string_view kReasoningMark = "flyx:";

// Text of a content value: a string, or the text parts of a list joined.
bool content_text(const ojson& content, std::string* out, std::string* err);

}  // namespace flyx
