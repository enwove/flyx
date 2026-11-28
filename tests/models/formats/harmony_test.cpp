// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Reading gpt-oss's harmony replies back (models/formats/harmony.h): the analysis channel as thinking, final and
// commentary as text, a message to functions.NAME as that tool's call, the same whether the reply arrives at once
// or byte by byte; a reply that is no harmony at all as text; the template recognised by its control tokens.
#include <cstdio>
#include <string>
#include <vector>

#include "models/formats/harmony.h"
#include "tests/common/check.h"

using namespace flyx;
using namespace flyx::models::formats;
using interfaces::ojson;
using interfaces::OutputEvent;

namespace {

const std::vector<ojson> kTools = {ojson::parse(R"({"type": "function", "function": {"name": "get_weather",
  "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}})")};

struct Reply {
  std::string reasoning, content;
  std::vector<interfaces::ToolCall> calls;
};

// The reply's events fed `step` bytes at a time, neighbouring deltas joined.
Reply read(const std::string& reply, size_t step, const std::vector<ojson>* tools = &kTools) {
  auto p = make_harmony_parser(tools);
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
                        const std::vector<ojson>* tools = &kTools) {
  for (size_t step : {reply.size(), size_t(7), size_t(1)}) {
    const Reply got = read(reply, step, tools);
    bool calls = got.calls.size() == want.calls.size();
    for (size_t i = 0; calls && i < got.calls.size(); ++i)
      calls = got.calls[i].name == want.calls[i].name && got.calls[i].arguments == want.calls[i].arguments;
    check(got.reasoning == want.reasoning && got.content == want.content && calls,
          what + " (" + std::to_string(step) + " bytes at a time): thinking \"" + got.reasoning + "\", text \"" +
              got.content + "\", " + std::to_string(got.calls.size()) + " calls");
  }
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  same_at_every_step("analysis then final",
                     "<|channel|>analysis<|message|>The user greets. Reply.<|end|>"
                     "<|start|>assistant<|channel|>final<|message|>Hello! <b>Hi</b><|return|>",
                     {"The user greets. Reply.", "Hello! <b>Hi</b>", {}});
  interfaces::ToolCall call;
  call.name = "get_weather";
  call.arguments = ojson::parse(R"({"location": "Tokyo"})");
  same_at_every_step("analysis then a tool call",
                     "<|channel|>analysis<|message|>Need the weather.<|end|>"
                     "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>"
                     "{\"location\": \"Tokyo\"}<|call|>",
                     {"Need the weather.", "", {call}});
  same_at_every_step("a preamble on commentary, then final",
                     "<|channel|>commentary<|message|>Checking.<|end|><|start|>assistant<|channel|>final<|message|>"
                     "Done.<|return|>",
                     {"", "Checking.Done.", {}});
  same_at_every_step("a call without tools given: its body as text",
                     "<|channel|>commentary to=functions.get_weather<|message|>{\"location\": \"x\"}<|call|>",
                     {"", "{\"location\": \"x\"}", {}}, nullptr);
  same_at_every_step("a reply cut short inside final", "<|channel|>final<|message|>Partial ans",
                     {"", "Partial ans", {}});
  same_at_every_step("a reply cut short inside the next message's header: the header left out",
                     "<|channel|>final<|message|>Answer.<|end|><|start|>assistant<|channel|>fin", {"", "Answer.", {}});
  same_at_every_step("text that is no harmony", "just words <|end|>", {"", "just words ", {}});
  check(is_harmony_template("{{- '<|start|>' + role + '<|channel|>final<|message|>' + content + '<|end|>' }}"),
        "a template with harmony's control tokens is harmony's");
  check(!is_harmony_template("<|im_start|>{{ role }}<|im_end|>"), "ChatML is not harmony");
  return flyx::test::finish("harmony_format_test");
}
