// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Reading Gemma 4's replies back (models/formats/gemma4.h): the thought channel as thinking, its label and the format's
// newlines left out; tool calls in Gemma's values as JSON arguments; an unclosed thought a call opens, empty, doubled
// and stray channel marks; a malformed call as its text; the same whether the reply arrives at once or byte by byte.
// The template recognised by its marks, a generation prompt's open thought, and the reply grammar's marks.
#include <cstdio>
#include <string>
#include <vector>

#include "models/formats/gemma4.h"
#include "tests/common/check.h"

using namespace flyx;
using namespace flyx::models::formats;
using interfaces::ojson;
using interfaces::OutputEvent;

namespace {

const std::vector<ojson> kTools = {ojson::parse(R"({"type": "function", "function": {"name": "get_weather",
  "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"}}}}})")};

struct Reply {
  std::string reasoning, content;
  std::vector<interfaces::ToolCall> calls;
};

Reply read(const std::string& reply, size_t step, bool in_reasoning, const std::vector<ojson>* tools) {
  auto p = make_gemma4_parser(in_reasoning, tools);
  Reply r;
  auto take = [&](std::vector<OutputEvent> ev) {
    for (auto& e : ev) {
      if (e.kind == OutputEvent::kReasoning) r.reasoning += e.text;
      else if (e.kind == OutputEvent::kContent) r.content += e.text;
      else if (e.kind == OutputEvent::kToolCall) r.calls.push_back(e.call);
    }
  };
  for (size_t i = 0; i < reply.size(); i += step) take(p->feed(reply.substr(i, step)));
  take(p->finish());
  return r;
}

void same_at_every_step(const std::string& what, const std::string& reply, const Reply& want,
                        bool in_reasoning = false, const std::vector<ojson>* tools = &kTools) {
  for (size_t step : {reply.size(), size_t(7), size_t(1)}) {
    const Reply got = read(reply, step, in_reasoning, tools);
    bool calls = got.calls.size() == want.calls.size();
    for (size_t i = 0; calls && i < got.calls.size(); ++i)
      calls = got.calls[i].name == want.calls[i].name && got.calls[i].arguments == want.calls[i].arguments;
    check(got.reasoning == want.reasoning && got.content == want.content && calls,
          what + " (" + std::to_string(step) + " bytes at a time): thinking \"" + got.reasoning + "\", text \"" +
              got.content + "\", " + std::to_string(got.calls.size()) + " calls");
  }
}

interfaces::ToolCall call(const std::string& name, const std::string& args) {
  interfaces::ToolCall c;
  c.name = name;
  c.arguments = ojson::parse(args);
  return c;
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  same_at_every_step("a thought, then the answer", "<|channel>thought\nThe user greets me.\n<channel|>Hello!",
                     {"The user greets me.", "Hello!", {}});
  same_at_every_step("no thought", "Just text, with < and > in it.", {"", "Just text, with < and > in it.", {}});
  same_at_every_step("the prompt opened the thought", "still thinking\n<channel|>\nDone.",
                     {"still thinking", "Done.", {}}, true);
  same_at_every_step(
      "a call after a thought",
      "<|channel>thought\nneed the weather\n<channel|><|tool_call>call:get_weather{days:2,location:<|\"|>Paris, "
      "France<|\"|>}<tool_call|>",
      {"need the weather", "", {call("get_weather", R"({"days": 2, "location": "Paris, France"})")}});
  same_at_every_step("a call from an unclosed thought",
                     "<|channel>thought\nI should call it<|tool_call>call:get_weather{location:<|\"|>Oslo<|\"|>}"
                     "<tool_call|>",
                     {"I should call it", "", {call("get_weather", R"({"location": "Oslo"})")}});
  same_at_every_step("two calls, nested values",
                     "<|tool_call>call:get_weather{location:<|\"|>a{b}:c<|\"|>,opts:{units:[<|\"|>c<|\"|>,1.5e3,true,"
                     "null],\"quoted key\":'x'}}<tool_call|><|tool_call>call:get_weather{}<tool_call|>",
                     {"", "",
                      {call("get_weather",
                            R"({"location": "a{b}:c", "opts": {"units": ["c", 1500.0, true, null], "quoted key": "x"}})"),
                       call("get_weather", "{}")}});
  same_at_every_step("an empty thought and a stray close", "<|channel>thought\n<channel|><channel|>Answer.",
                     {"", "Answer.", {}});
  same_at_every_step("a doubled channel mark", "<|channel>thought\n<|channel>thinking<channel|>A.",
                     {"thinking", "A.", {}});
  same_at_every_step("a malformed call stays text", "<|tool_call>call:get_weather{location:}<tool_call|>",
                     {"", "<|tool_call>call:get_weather{location:}<tool_call|>", {}});
  same_at_every_step("no tools: a call is text", "<|tool_call>call:f{}<tool_call|>",
                     {"", "<|tool_call>call:f{}<tool_call|>", {}}, false, nullptr);
  same_at_every_step("a stop token's text dropped", "Bye.<turn|>", {"", "Bye.", {}});

  check(parse_gemma4_value("{a:1,b:[<|\"|>x<|\"|>]}") == ojson::parse(R"({"a": 1, "b": ["x"]})"),
        "Gemma values: bare keys, <|\"|> strings");
  check(parse_gemma4_value("{a:}").is_null() && parse_gemma4_value("{a:1} trailing").is_null(),
        "Gemma values: a malformed one is none");
  check(is_gemma4_template("{{ '<|turn>' }}<|channel>x<|tool_call>call:{{ name }}"), "the template by its marks");
  check(!is_gemma4_template("<|im_start|><tool_call>"), "another template is not Gemma 4's");
  check(gemma4_opens_reasoning("<|turn>model\n<|channel>thought\n") &&
            !gemma4_opens_reasoning("<|turn>model\n<|channel>thought\n<channel|>") &&
            !gemma4_opens_reasoning("<|turn>model\n"),
        "a generation prompt's open thought");
  const std::vector<ojson> tools = kTools;
  interfaces::ReplyRules rules;
  rules.tools = interfaces::ReplyRules::kRequired;
  const std::string g = reply_grammar(kGemma4Syntax, tools, rules, false);
  check(g.find("<|tool_call>call:get_weather") != std::string::npos && g.find("<tool_call|>") != std::string::npos &&
            g.find("\"gemma\"") != std::string::npos && g.find("<think>") == std::string::npos,
        "the reply grammar: Gemma's call marks and argument style, no <think>");
  return flyx::test::finish("gemma4_format_test");
}
