// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Reading replies back (models/formats/tags.h, calls.h): thinking, answer text and tool
// calls in both call languages, the same whether the reply arrives at once or
// byte by byte.
#include <cstdio>
#include <string>
#include <vector>

#include "models/formats/calls.h"
#include "models/formats/inkling2.h"
#include "models/formats/tags.h"

using namespace flyx::models::formats;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

const std::vector<ojson> kTools = {ojson::parse(R"({"type": "function", "function": {"name": "get_weather",
  "parameters": {"type": "object", "properties": {"location": {"type": "string"}, "days": {"type": "integer"},
  "detail": {"type": "boolean"}, "when": {"type": "object"}}}}})")};

// Events of one reply, verifying streaming tool-call framing (Start -> 0+ Delta -> Call)
// and merging neighbouring reasoning/content deltas (streaming splits them anywhere).
std::vector<OutputEvent> run(TagSyntax syntax, bool in_reasoning, const std::string& reply, size_t step,
                             const std::string& label) {
  TaggedParser p(syntax, in_reasoning, &kTools);
  std::vector<OutputEvent> raw;
  auto push = [&](std::vector<OutputEvent> more) {
    for (auto& e : more) raw.push_back(std::move(e));
  };
  for (size_t i = 0; i < reply.size(); i += step) push(p.feed(reply.substr(i, step)));
  push(p.finish());

  // Verify strict streaming tool-call framing and exact JSON delta reconstruction.
  std::vector<OutputEvent> ev;
  bool in_call = false;
  std::string started_name;
  std::string streamed_args;
  int completed = 0;
  for (auto& e : raw) {
    if (e.kind == OutputEvent::kToolCallStart) {
      check(!in_call, label + ": nested kToolCallStart");
      check(!e.call.name.empty(), label + ": empty name in kToolCallStart");
      in_call = true;
      started_name = e.call.name;
      streamed_args.clear();
    } else if (e.kind == OutputEvent::kToolCallDelta) {
      check(in_call, label + ": kToolCallDelta outside active call");
      streamed_args += e.text;
    } else if (e.kind == OutputEvent::kToolCall) {
      check(in_call, label + ": kToolCall without preceding kToolCallStart");
      check(e.call.name == started_name, label + ": name mismatch between Start and Call");
      const ojson parsed = ojson::parse(streamed_args, nullptr, false);
      check(!parsed.is_discarded() && parsed == e.call.arguments,
            label + ": streamed JSON deltas differ from final arguments\n  streamed: " + streamed_args +
                "\n  final:    " + e.call.arguments.dump());
      in_call = false;
      ++completed;
      ev.push_back(std::move(e));
    } else {
      check(!in_call, label + ": text/reasoning emitted mid-tool-call");
      if (!ev.empty() && ev.back().kind == e.kind) ev.back().text += e.text;
      else ev.push_back(std::move(e));
    }
  }
  check(!in_call, label + ": unclosed kToolCallStart at finish");
  check(p.tool_calls() == completed, label + ": tool_calls() count mismatch");
  return ev;
}

std::string show(const std::vector<OutputEvent>& ev) {
  std::string s;
  for (const auto& e : ev) {
    if (e.kind == OutputEvent::kToolCall) s += "[call " + e.call.name + " " + e.call.arguments.dump() + "]";
    else s += (e.kind == OutputEvent::kReasoning ? "[think " : "[text ") + e.text + "]";
  }
  return s;
}

void expect(const char* name, TagSyntax syntax, bool in_reasoning, const std::string& reply, const std::string& want) {
  for (size_t step : {reply.size() + 1, size_t(1), size_t(2), size_t(3), size_t(7)}) {
    const std::string label = std::string(name) + " (chunks of " + std::to_string(step) + ")";
    std::string got = show(run(syntax, in_reasoning, reply, step, label));
    check(got == want, label + "\n  want: " + want + "\n  got:  " + got);
  }
}

}  // namespace

int main() {
  const TagSyntax xml{true, read_xml_call}, json{true, read_json_call}, plain_think{true, nullptr};

  expect("qwen3_coder: thinking opened by the prompt, then a typed call", xml, true,
         "Check the weather.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n"
         "</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=detail>\nTrue\n</parameter>\n<parameter=when>\n"
         "{\"from\": \"mon\"}\n</parameter>\n</function>\n</tool_call>",
         "[think Check the weather.][call get_weather "
         "{\"location\":\"Paris\",\"days\":3,\"detail\":true,\"when\":{\"from\":\"mon\"}}]");
  expect("qwen3_coder: text, two calls, no thinking", xml, false,
         "Looking both up.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nRome\n</parameter>\n"
         "</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nOslo\n</parameter>\n"
         "</function>\n</tool_call>",
         "[text Looking both up.][call get_weather {\"location\":\"Rome\"}][call get_weather {\"location\":\"Oslo\"}]");
  expect("qwen3_coder: zero-parameter call", xml, false,
         "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
         "[call ping {}]");
  expect("qwen3_coder: escapes, newlines, html, UTF-8, and empty params", xml, false,
         "<tool_call>\n<function=get_weather>\n<parameter=location>\n\"line1\"\n\\</p>\n\n日本語🚀café\n</parameter>\n"
         "<parameter=empty1>\n</parameter>\n<parameter=empty2>\n\n</parameter>\n</function>\n</tool_call>",
         "[call get_weather {\"location\":\"\\\"line1\\\"\\n\\\\</p>\\n\\n日本語🚀café\",\"empty1\":\"\",\"empty2\":\"\"}]");
  expect("qwen3_coder: malformed body stays text without starting a call", xml, false,
         "Oops: <tool_call>\nno function tag\n</tool_call> ok",
         "[text Oops: <tool_call>\nno function tag\n</tool_call> ok]");
  expect("qwen3_coder: empty function name stays text", xml, false,
         "<tool_call>\n<function=>\n</tool_call>",
         "[text <tool_call>\n<function=>\n</tool_call>]");
  expect("qwen3_coder: truncated string param closes cleanly on finish", xml, false,
         "<tool_call>\n<function=get_weather>\n<parameter=location>\nHel",
         "[call get_weather {\"location\":\"Hel\"}]");
  expect("hermes: the model opens its own thinking block", json, false,
         "<think>\nhmm\n</think>\n\nSure.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
         "{\"location\": \"Paris\", \"days\": 2}}\n</tool_call>",
         "[think hmm][text Sure.][call get_weather {\"location\":\"Paris\",\"days\":2}]");
  expect("hermes: nested braces, escaped quotes, and UTF-8 in streamed JSON args", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"a } \\\"b\\\" 日本語\", "
         "\"when\": {\"nested\": {\"x\": 1}}}}\n</tool_call>",
         "[call get_weather {\"location\":\"a } \\\"b\\\" 日本語\",\"when\":{\"nested\":{\"x\":1}}}]");
  expect("hermes: omitted arguments object", json, false,
         "<tool_call>\n{\"name\": \"ping\"}\n</tool_call>",
         "[call ping {}]");
  expect("hermes: out-of-order keys fall back cleanly", json, false,
         "<tool_call>\n{\"arguments\": {\"location\": \"Oslo\"}, \"name\": \"get_weather\"}\n</tool_call>",
         "[call get_weather {\"location\":\"Oslo\"}]");
  expect("hermes: arguments written as JSON text", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"Lima\\\"}\"}\n</tool_call>",
         "[call get_weather {\"location\":\"Lima\"}]");
  expect("hermes: a body that is not a call stays text", json, false,
         "Here: <tool_call>\nnot json\n</tool_call> done",
         "[text Here: <tool_call>\nnot json\n</tool_call> done]");
  expect("hermes: truncated string arg closes cleanly on finish", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"San Fran",
         "[call get_weather {\"location\":\"San Fran\"}]");
  expect("hermes: truncated nested object/array closes cleanly on finish", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"when\": {\"days\": [1, 2",
         "[call get_weather {\"when\":{\"days\":[1,2]}}]");
  expect("hermes: truncated key before colon closes cleanly on finish", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"a\": 1, \"b\"",
         "[call get_weather {\"a\":1,\"b\":null}]");
  expect("hermes: truncated value after colon closes cleanly on finish", json, false,
         "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"a\":",
         "[call get_weather {\"a\":null}]");
  {
    // A truncated array ending in a trailing comma must not fabricate a spurious null element.
    TaggedParser p(json, false, &kTools);
    auto ev1 = p.feed("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"files\": [\"a.py\",");
    auto ev2 = p.finish();
    std::string streamed;
    ojson final_args;
    for (const auto& evs : {ev1, ev2})
      for (const auto& e : evs) {
        if (e.kind == OutputEvent::kToolCallDelta) streamed += e.text;
        if (e.kind == OutputEvent::kToolCall) final_args = e.call.arguments;
      }
    check(streamed.find("null") == std::string::npos && final_args == ojson::object(),
          "hermes: truncated array trailing comma does not fabricate null element");
  }
  expect("hermes: malformed arguments object stays text", json, false,
         "Oops: <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {bad}}\n</tool_call> ok",
         "[text Oops: <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {bad}}\n</tool_call> ok]");
  expect("hermes: omitted arguments with trailing garbage stays text", json, false,
         "<tool_call>\n{\"name\": \"ping\"} trailing\n</tool_call>",
         "[text <tool_call>\n{\"name\": \"ping\"} trailing\n</tool_call>]");
  expect("xml: malformed call fallback preserves surrounding newlines", xml, false,
         "Checking:\n\n<tool_call>not a call</tool_call>\n\nDone",
         "[text Checking:\n\n<tool_call>not a call</tool_call>\n\nDone]");
  expect("no call language: <tool_call> is text", plain_think, true, "a\n</think>\n\n<tool_call>x</tool_call>",
         "[think a][text <tool_call>x</tool_call>]");
  expect("answer only; trailing newlines dropped", xml, false, "\n\nJust text.\n\n", "[text Just text.]");

  // Incremental latency check: both XML and JSON emit kToolCallStart and argument deltas
  // BEFORE </tool_call> arrives.
  {
    TaggedParser p(xml, false, &kTools);
    auto ev1 = p.feed("<tool_call>\n<function=get_weather>\n");
    check(ev1.size() == 1 && ev1[0].kind == OutputEvent::kToolCallStart && ev1[0].call.name == "get_weather",
          "xml streams kToolCallStart as soon as <function=...> closes");
    auto ev2 = p.feed("<parameter=location>\nSan Francisco");
    std::string d;
    for (const auto& e : ev2) {
      check(e.kind == OutputEvent::kToolCallDelta, "xml emits kToolCallDelta before </parameter>");
      d += e.text;
    }
    check(d.find("San Francisco") != std::string::npos, "xml streams string param content before </tool_call>: " + d);
    auto ev3 = p.feed("\n</parameter>\n</function>\n</tool_call>");
    bool got_call = false;
    for (const auto& e : ev3) {
      if (e.kind == OutputEvent::kToolCallDelta) d += e.text;
      if (e.kind == OutputEvent::kToolCall) got_call = true;
    }
    check(got_call && ojson::parse(d) == ojson{{"location", "San Francisco"}}, "xml finalizes cleanly after streaming");
  }
  {
    TaggedParser p(json, false, &kTools);
    auto ev1 = p.feed("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tok");
    bool got_start = false;
    std::string d;
    for (const auto& e : ev1) {
      if (e.kind == OutputEvent::kToolCallStart && e.call.name == "get_weather") got_start = true;
      if (e.kind == OutputEvent::kToolCallDelta) d += e.text;
    }
    check(got_start && d.find("Tok") != std::string::npos,
          "json streams kToolCallStart and partial args before </tool_call>: " + d);
    auto ev2 = p.feed("yo\"}}\n</tool_call>");
    bool got_call = false;
    for (const auto& e : ev2) {
      if (e.kind == OutputEvent::kToolCallDelta) d += e.text;
      if (e.kind == OutputEvent::kToolCall) got_call = true;
    }
    check(got_call && ojson::parse(d) == ojson{{"location", "Tokyo"}}, "json finalizes cleanly after streaming");
  }

  // Inkling-2 multi-section parser (thinking, text, JSON tool calls) across arbitrary chunk sizes + grammar.
  {
    const std::string reply =
        "<|message_model|><|content_thinking|>Need weather.<|end_message|>"
        "<|message_model|><|content_text|>Checking now.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        "{\"name\":\"get_weather\",\"args\":{\"location\":\"Tokyo\",\"days\":3}}<|end_message|>"
        "<|content_model_end_sampling|>";
    for (size_t step : {size_t(1), size_t(3), size_t(17), reply.size()}) {
      auto p = make_inkling2_parser(&kTools);
      std::string got_think, got_text, got_args, got_name;
      int calls = 0;
      auto consume = [&](const std::vector<OutputEvent>& evs) {
        for (const auto& e : evs) {
          if (e.kind == OutputEvent::kReasoning) got_think += e.text;
          else if (e.kind == OutputEvent::kContent) got_text += e.text;
          else if (e.kind == OutputEvent::kToolCall) {
            ++calls;
            got_name = e.call.name;
            got_args = e.call.arguments.dump();
          }
        }
      };
      for (size_t i = 0; i < reply.size(); i += step) consume(p->feed(std::string_view(reply).substr(i, step)));
      consume(p->finish());
      check(got_think == "Need weather." && got_text == "Checking now." && calls == 1 && got_name == "get_weather" &&
                got_args == "{\"location\":\"Tokyo\",\"days\":3}" && p->tool_calls() == 1,
            "inkling2 parser step=" + std::to_string(step));
    }
    flyx::interfaces::ReplyRules rules;
    rules.tools = flyx::interfaces::ReplyRules::kRequired;
    const std::string g = inkling2_reply_grammar(kTools, rules);
    const ojson gj = ojson::parse(g, nullptr, false);
    check(!gj.is_discarded() && gj.contains("type") && gj["type"] == "structural_tag" &&
              g.find("<|content_invoke_tool_json|>") != std::string::npos,
          "inkling2 reply grammar produces valid structural_tag JSON");
  }

  std::printf("tagged: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
