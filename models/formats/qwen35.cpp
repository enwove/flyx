// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Qwen3.5's chat format: its template (chat_template.jinja of Qwen/Qwen3.5-4B,
// sha256 a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715,
// also the NVFP4 checkpoints') written out in C++, and its replies: thinking
// in <think>, tool calls as XML (calls.h).  The renderer follows the template
// statement by statement; tests/models/formats/qwen35_test.cpp holds it to
// the Jinja engine on generated conversations, and the chat test to
// transformers.  Images and videos are not rendered (the API rejects them).
#include "models/formats/qwen35.h"

#include "models/formats/pytext.h"

namespace flyx::models::formats {
namespace {

constexpr std::string_view kToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsFooter =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> "
    "XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function "
    "call in natural language BEFORE the function call, but NOT after\n- If there is no function call "
    "available, answer the question like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

std::string_view rstrip_char(std::string_view s, char c) {
  size_t e = s.size();
  while (e > 0 && s[e - 1] == c) --e;
  return s.substr(0, e);
}

std::string_view lstrip_char(std::string_view s, char c) {
  size_t b = 0;
  while (b < s.size() && s[b] == c) ++b;
  return s.substr(b);
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

bool fail(std::string* err, const char* message) {
  *err = message;
  return false;
}

bool render_qwen35(std::span<const ChatMessage> msgs, const std::vector<ojson>& tools, const ChatOptions& opt,
                   bool generation_prompt, RenderedChat* out, std::string* err) {
  Writer w(out);
  const size_t n = msgs.size();
  if (n == 0) return fail(err, "No messages provided.");

  // The system block: the tools and their instructions, then the system message.
  if (!tools.empty()) {
    w.lit("<|im_start|>system\n");
    w.lit(kToolsHeader);
    for (const auto& t : tools) {
      w.lit("\n");
      w.text(py_json(t), holds_client_text(t));  // tojson
    }
    w.lit(kToolsFooter);
    if (msgs[0].role == "system") {
      const std::string_view content = py_strip(msgs[0].content);
      if (!content.empty()) {
        w.lit("\n\n");
        w.in(content);
      }
    }
    w.lit("<|im_end|>\n");
  } else if (msgs[0].role == "system") {
    w.lit("<|im_start|>system\n");
    w.in(py_strip(msgs[0].content));
    w.lit("<|im_end|>\n");
  }

  // The last user message that is a question (not tool results wrapped as a user message).
  bool multi_step_tool = true;
  size_t last_query = n - 1;
  for (size_t i = n; i-- > 0;) {
    if (!multi_step_tool || msgs[i].role != "user") continue;
    const std::string_view content = py_strip(msgs[i].content);
    if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
      multi_step_tool = false;
      last_query = i;
    }
  }
  if (multi_step_tool) return fail(err, "No user query found in messages.");

  for (size_t i = 0; i < n; ++i) {
    const ChatMessage& m = msgs[i];
    std::string_view content = py_strip(m.content);
    if (m.role == "system") {
      if (i != 0) return fail(err, "System message must be at the beginning.");
    } else if (m.role == "user") {
      w.lit("<|im_start|>");
      w.in(m.role);
      w.lit("\n");
      w.in(content);
      w.lit("<|im_end|>\n");
    } else if (m.role == "assistant") {
      std::string_view reasoning;
      if (m.reasoning) {
        reasoning = *m.reasoning;
      } else if (content.find("</think>") != std::string::npos) {
        const std::string_view head = rstrip_char(content.substr(0, content.find("</think>")), '\n');
        const size_t open = head.rfind("<think>");
        reasoning = lstrip_char(open == std::string::npos ? head : head.substr(open + 7), '\n');
        content = lstrip_char(content.substr(content.rfind("</think>") + 8), '\n');
      }
      reasoning = py_strip(reasoning);
      w.lit("<|im_start|>");
      w.in(m.role);
      if (i > last_query) {
        w.lit("\n<think>\n");
        w.in(reasoning);
        w.lit("\n</think>\n\n");
      } else {
        w.lit("\n");
      }
      w.in(content);
      for (size_t c = 0; c < m.tool_calls.size(); ++c) {
        const auto& call = m.tool_calls[c];
        if (c > 0) w.lit("\n<tool_call>\n<function=");
        else w.lit(py_strip(content).empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=");
        w.in(call.name);
        w.lit(">\n");
        if (call.arguments.is_object())
          for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
            w.lit("<parameter=");
            w.in(it.key());
            w.lit(">\n");
            const ojson& v = it.value();
            if (v.is_object() || v.is_array()) w.text(py_json(v), holds_client_text(v));  // tojson
            else if (v.is_string()) w.in(v.get<std::string>());   // | string keeps the client's string
            else w.lit(py_str(v));
            w.lit("\n</parameter>\n");
          }
        w.lit("</function>\n</tool_call>");
      }
      w.lit("<|im_end|>\n");
    } else if (m.role == "tool") {
      if (i > 0 && msgs[i - 1].role != "tool") w.lit("<|im_start|>user");
      w.lit("\n<tool_response>\n");
      w.in(content);
      w.lit("\n</tool_response>");
      if (i + 1 == n || msgs[i + 1].role != "tool") w.lit("<|im_end|>\n");
    } else {
      return fail(err, "Unexpected message role.");
    }
  }

  if (generation_prompt) {
    w.lit("<|im_start|>assistant\n");
    // `enable_thinking is defined and enable_thinking is false`: only a boolean false turns it off.
    const ojson* given = opt.kwargs.contains("enable_thinking") ? &opt.kwargs["enable_thinking"] : nullptr;
    const bool off = given ? (given->is_boolean() && !given->get<bool>()) : !opt.enable_thinking;
    w.lit(off ? "<think>\n\n</think>\n\n" : "<think>\n");
  }
  return true;
}

}  // namespace

const Family kQwen35 = {"qwen35", 7756, 0xcdff116097e1144aull, render_qwen35,
                        {true, read_xml_call, &kXmlCallSyntax}};

}  // namespace flyx::models::formats
