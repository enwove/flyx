// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// What a model file says about chat: its Jinja
// template, and the chat format that template belongs to: how replies are
// read back (thinking, tool calls) and, for templates a renderer was written
// for, that renderer.  Core renders any template with the Jinja engine
// (core/render/chat.h); the formats are the model families' (models/formats/).
// Core reaches them only through this interface (interfaces::open_chat, defined in
// models/formats/formats.cpp).
#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/json_depth.h"
#include "nlohmann/json.hpp"

namespace flyx {
class ModelFile;
}

namespace flyx::interfaces {

using ojson = nlohmann::ordered_json;

// JSON text from outside the engine (a request's body, tool-call arguments written as text, what a model wrote) is
// parsed with parse_json.  Deeper than kMaxJsonDepth it comes back discarded, as text that is not JSON does.
inline ojson parse_json(std::string_view s) {
  return json_depth_within(s) ? ojson::parse(s, nullptr, false) : ojson(ojson::value_t::discarded);
}

// Bytes [begin, end) of a text.
struct TextSpan {
  size_t begin = 0, end = 0;
};

struct ToolCall {
  std::string id;
  std::string name;
  ojson arguments = ojson::object();  // keys in the order the client sent them
};

// One image attached to a chat message (raw encoded bytes such as PNG/JPEG, or a decoded
// payload; sha256 identifies its content for prefix caching and embedding reuse).
struct ImageInput {
  std::vector<uint8_t> bytes;          // raw encoded image bytes (PNG/JPEG/BMP/GIF/TGA)
  std::array<uint8_t, 32> sha256{};    // SHA-256 of `bytes`
  std::string mime;                    // optional MIME type, e.g. "image/png"
  std::string detail;                  // optional detail hint: "", "auto", "low", "high"
};

struct ContentPart {
  enum Kind { kText = 0, kImage = 1 } kind = kText;
  std::string text;                             // kText
  std::shared_ptr<const ImageInput> image;      // kImage

  static ContentPart make_text(std::string s) { return {kText, std::move(s), nullptr}; }
  static ContentPart make_image(std::shared_ptr<const ImageInput> img) { return {kImage, {}, std::move(img)}; }
};

struct ChatMessage {
  std::string role;                      // system | user | assistant | tool
  std::string content;                   // text (when `parts` is empty)
  std::optional<std::string> reasoning;  // assistant thinking given separately (reasoning_content)
  std::vector<ToolCall> tool_calls;      // assistant
  std::string tool_call_id;              // tool
  std::string name;                      // optional tool / assistant name
  // Multimodal content parts (non-empty when a message carries images alongside text).
  // Kept last so existing aggregate initializers for text-only messages remain valid.
  std::vector<ContentPart> parts;
};

struct ChatOptions {
  bool add_generation_prompt = true;
  bool enable_thinking = true;
  // More template variables (OpenAI chat_template_kwargs), e.g. reasoning_effort.
  ojson kwargs = ojson::object();
};

// An image rendered into RenderedChat::text, anchored at the byte span of its template placeholder.
struct RenderedImage {
  size_t byte_offset = std::string::npos;  // start byte in RenderedChat::text (npos = resolve by placeholder token)
  size_t byte_end = std::string::npos;     // end byte in RenderedChat::text (exclusive)
  std::shared_ptr<const ImageInput> image; // source image bytes + SHA-256 + detail
};

struct RenderedChat {
  std::string text;
  // Bytes the template wrote itself (sorted): special tokens are read only
  // here.  The client's strings, as they pass through the template, are not.
  std::vector<TextSpan> literal;
  // Multimodal images emitted into `text`, in order of appearance (empty for text-only requests).
  std::vector<RenderedImage> images;
  // Byte offsets where a later request is likely to branch, so the engine
  // keeps checkpoints there: where this conversation ends (before the
  // generation prompt) and where earlier requests that it continues ended; on
  // a first request, the nearest earlier end of a user message or of tool
  // results whose prefix renders the same.  At most three, ascending.
  std::vector<size_t> boundaries;
  // Explicit client cache breakpoints, as byte offsets in the actual rendering.
  std::vector<size_t> cache_prefixes;
  size_t system_end = 0;
  bool in_reasoning = false;  // the generation prompt leaves a thinking block open
  // Byte offset where the conversation ends and the generation prompt begins
  // (0: not known).  A later prompt that has all the text up to here is the
  // same conversation, sent again or continued.
  size_t conversation_end = 0;
  // Byte offset where the first user message ends (0: none, or not known):
  // a request that shares the text up to here is the same conversation as
  // another, not just one with the same system prompt.
  size_t first_user_end = 0;
};

// What a reply may be, beyond what the model writes on its own:
// the API's tool_choice and parallel_tool_calls, or a JSON answer.
struct ReplyRules {
  enum Tools { kAuto, kRequired, kForced, kNone } tools = kAuto;
  std::string forced;     // kForced: the one tool to call
  bool parallel = true;   // more than one call in a reply
  bool json = false;      // the answer is JSON matching `schema` (OpenAI response_format)
  ojson schema = true;    // json: true = any JSON value
  // Whether the request left thinking on (ChatOptions::enable_thinking; the engine sets it for every chat request).
  // Off, a format whose template cannot keep its model from thinking leaves the thinking out of the reply here
  // (Inkling-2: its template writes an effort, and the model opens a thinking section at effort none too).
  bool thinking = true;
};

// A piece of a reply, as the client sees it.
// A tool call streams as kToolCallStart (call.name), zero or more kToolCallDelta
// (text: incremental JSON argument fragments), and finally kToolCall (complete call).
struct OutputEvent {
  enum Kind { kReasoning, kContent, kToolCallStart, kToolCallDelta, kToolCall } kind;
  std::string text;  // reasoning / content delta, or kToolCallDelta JSON arguments fragment
  ToolCall call;     // kToolCallStart: name; kToolCall: name and arguments (typed by the tool's JSON schema)
};

// Splits one reply into thinking, answer text and tool calls as it streams.
class OutputParser {
 public:
  virtual ~OutputParser() = default;
  virtual std::vector<OutputEvent> feed(std::string_view text) = 0;
  // End of generation: emits what is still held back.
  virtual std::vector<OutputEvent> finish() = 0;
  virtual int tool_calls() const = 0;
};

class Chat {
 public:
  virtual ~Chat() = default;
  // The Jinja source to render prompts with.
  virtual const std::string& template_source() const = 0;
  // Special tokens the template may name as variables (bos_token, eos_token).
  virtual const ojson& template_tokens() const = 0;
  // The chat format: "qwen35" (a renderer written for Qwen3.5's template),
  // "qwen3_coder" / "hermes" (replies read back, the template rendered by
  // the engine), or "plain" (text only).
  virtual const char* format() const = 0;
  // Whether replies can carry tool calls this build reads.
  virtual bool reads_tool_calls() const = 0;
  // Whether a generation prompt (what the template appends to open the
  // assistant's turn) leaves a thinking block open.
  virtual bool opens_reasoning(std::string_view generation_prompt) const = 0;
  // A parser for one reply.  in_reasoning: the prompt left a thinking block
  // open.  tools: the request's definitions (to type parameter values); null
  // or empty disables tool-call parsing.
  virtual std::unique_ptr<OutputParser> parser(bool in_reasoning, const std::vector<ojson>* tools) const = 0;
  // The grammar a reply must match, as an XGrammar structural tag (JSON
  // text): an optional thinking block, then text and tool calls in this
  // format's syntax, each call's arguments fitting its tool's schema, as
  // `rules` allow.  "" when nothing constrains the reply (no tools and no
  // JSON answer, tool_choice none, or a format without tool calls); a format
  // that can only keep thinking out of a reply with a grammar gives one for
  // rules.thinking false whatever else is asked (Inkling-2's).
  virtual std::string reply_grammar(const std::vector<ojson>& tools, const ReplyRules& rules,
                                    bool in_reasoning) const {
    return "";
  }
  // The special tokens a reply in this format is written with, by their text: a format whose section and call
  // markers are control tokens (Inkling-2's <|message_model|>, <|content_text|>, <|end_message|>).  Where a reply
  // grammar's delimiter, separator or constant holds that text it takes the token and nothing else, and the token
  // nowhere else: the same characters in ordinary tokens are text, which a section may quote where its grammar
  // allows that text, and never a delimiter.  Any other special token a grammar never takes; the model's stop tokens
  // end a reply and are not named here.  None where the markers are ordinary tokens (<think>, <tool_call>).
  virtual std::vector<std::string> reply_tokens() const { return {}; }
  // The text that opens a thinking section in a reply and the text that closes it (<think> and </think>; Inkling-2's
  // <|content_thinking|> and <|end_message|>): what a thinking budget counts between and writes to end thinking.
  // false for a format whose replies have no thinking sections.
  virtual bool thinking_markers(std::string* open, std::string* close) const { return false; }
  // Template variables that say what the engine's own options say, for a template that has other names for them:
  // one with a thinking effort and no enable_thinking is given the effort of thinking switched off.  The request's
  // own variables (ChatOptions::kwargs) come before these.
  virtual ojson option_variables(const ChatOptions& opt) const { return ojson::object(); }

  // A renderer written for exactly this template (selected by its full
  // text): the bytes and literal spans the Jinja engine gives, hundreds of
  // times faster.  Tests hold every such renderer to the engine.
  virtual bool native() const { return false; }
  // Only when native(): renders the messages (text and literal spans; the
  // generation prompt if asked); false with *err where the template raises.
  virtual bool render(std::span<const ChatMessage> msgs, const std::vector<ojson>& tools, const ChatOptions& opt,
                      bool generation_prompt, RenderedChat* out, std::string* err) const {
    return *err = "no native renderer", false;
  }
};

// The chat side of a model file: its template (or `template_override`, a
// Jinja source given at load) and the chat format that template belongs to.
// Null with *err set when there is no template.
std::unique_ptr<Chat> open_chat(const ModelFile& file, const std::string& template_override, std::string* err);

}  // namespace flyx::interfaces
