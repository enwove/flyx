// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Tool calls as models write them, one body (what stands between <tool_call>
// and </tool_call>) at a time.  Shared by the chat formats that use them:
//   XML  (Qwen3.5, Qwen3-Coder):  <function=NAME><parameter=P>VALUE</parameter>...</function>
//   JSON (Qwen3, Qwen2.5):        {"name": NAME, "arguments": {...}}
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"

namespace flyx::models::formats {

using interfaces::ojson;
using interfaces::OutputEvent;
using interfaces::ToolCall;

// Reads one call body; false if it is not a well-formed call (shown as text then).
using CallReader = bool (*)(const std::string& body, const std::vector<ojson>* tools, ToolCall* out);

// Incrementally parses one tool-call body (what stands between <tool_call> and </tool_call>)
// as tokens arrive: emits kToolCallStart once the function name is known, kToolCallDelta for
// JSON argument fragments, and kToolCall when finished.  If finish() returns false, nothing
// was emitted (started() is false), so the caller can show the raw bytes as content.
class CallStream {
 public:
  virtual ~CallStream() = default;
  virtual bool started() const = 0;
  virtual void feed(std::string_view chunk, std::vector<OutputEvent>* out) = 0;
  virtual bool finish(std::vector<OutputEvent>* out) = 0;
};

// Opens a streaming call parser for `reader` (XML, JSON, or a buffered fallback for any other CallReader).
std::unique_ptr<CallStream> open_call_stream(CallReader reader, const std::vector<ojson>* tools);

// How a call is written, for grammars: a call to NAME is
//   begin_prefix NAME begin_suffix ARGUMENTS end
// with ARGUMENTS in XGrammar's JSON-schema `style`; free text turns into a
// call at `trigger`.
struct CallSyntax {
  const char* begin_prefix;
  const char* begin_suffix;
  const char* end;
  const char* trigger;
  const char* style;
};
extern const CallSyntax kXmlCallSyntax;   // <tool_call>\n<function=NAME>\n<parameter=P>\nV\n</parameter>\n</function>\n</tool_call>
extern const CallSyntax kJsonCallSyntax;  // <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>

// Returns the function descriptor object inside an OpenAI/Anthropic tool definition, or nullptr.
inline const ojson* function_of(const ojson& tool) {
  if (!tool.is_object()) return nullptr;
  if (tool.contains("function") && tool["function"].is_object()) return &tool["function"];
  return tool.contains("name") && tool["name"].is_string() ? &tool : nullptr;
}

// XML: parameter values are text; each is typed by the tool's JSON schema
// (integer, number, boolean, object, ... as JSON or Python's True/False/None),
// strings stay strings.
bool read_xml_call(const std::string& body, const std::vector<ojson>* tools, ToolCall* out);
// JSON: arguments as an object, or as an object written as JSON text.
bool read_json_call(const std::string& body, const std::vector<ojson>* tools, ToolCall* out);

}  // namespace flyx::models::formats
