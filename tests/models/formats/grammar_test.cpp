// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Reply grammars (models/formats/grammar.cpp through core/grammar): which
// replies a tool_choice or response_format admits, in both call languages,
// spelled with a real byte-level tokenizer.
//
//   grammar_test MODEL.flyx
#include <cstdio>
#include <string>
#include <vector>

#include "core/grammar/grammar.h"
#include "core/modelfile/modelfile.h"
#include "core/tokenizer/tokenizer.h"
#include "models/formats/calls.h"
#include "models/formats/tags.h"

using namespace flyx;
using namespace flyx::models::formats;
using Rules = interfaces::ReplyRules;

namespace {

int failures = 0, passed = 0;

const std::vector<ojson> kTools = {
    ojson::parse(R"({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object",
      "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
      "required": ["city"]}}})"),
    ojson::parse(R"({"type": "function", "function": {"name": "get_time", "parameters": {"type": "object",
      "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}})")};

const TagSyntax kXml{true, read_xml_call, &kXmlCallSyntax};
const TagSyntax kJson{true, read_json_call, &kJsonCallSyntax};

// Whether the grammar takes the whole reply, token by token, and lets it end there.
bool admits(Grammar& g, const Tokenizer& tok, const std::string& tag, const std::string& reply) {
  std::string err;
  auto m = g.matcher(tag, &err);
  if (!m) {
    std::printf("FAIL grammar does not compile: %s\n%s\n", err.c_str(), tag.c_str());
    ++failures;
    return false;
  }
  std::vector<uint32_t> mask(size_t(g.words()));
  for (int32_t t : tok.encode(reply)) {
    m->allowed(mask.data());  // the mask and accept() agree on every token
    const bool in_mask = (mask[size_t(t) >> 5] >> (t & 31)) & 1u;
    const bool ok = m->accept(t);
    if (ok != in_mask) {
      std::printf("FAIL mask says %d, accept says %d for token %d\n", in_mask, ok, t);
      ++failures;
    }
    if (!ok) return false;
  }
  return m->can_end();
}

void expect(Grammar& g, const Tokenizer& tok, const char* what, const std::string& tag, const std::string& reply,
            bool want) {
  if (admits(g, tok, tag, reply) == want) {
    ++passed;
    return;
  }
  ++failures;
  std::printf("FAIL %s: %s\n  reply: %s\n  tag: %s\n", what, want ? "rejected" : "admitted", reply.c_str(),
              tag.c_str());
}

Rules rules(Rules::Tools t, const std::string& forced = "", bool parallel = true) {
  Rules r;
  r.tools = t;
  r.forced = forced;
  r.parallel = parallel;
  return r;
}

std::string xml_call(const std::string& name, const std::vector<std::pair<std::string, std::string>>& params) {
  std::string s = "<tool_call>\n<function=" + name + ">\n";
  for (const auto& [k, v] : params) s += "<parameter=" + k + ">\n" + v + "\n</parameter>\n";
  return s + "</function>\n</tool_call>";
}

std::string json_call(const std::string& name, const std::string& args) {
  return "<tool_call>\n{\"name\": \"" + name + "\", \"arguments\": " + args + "}\n</tool_call>";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: grammar_test MODEL.flyx\n");
    return 2;
  }
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(argv[1], &err) || !tok.load(file, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  Grammar g;
  g.init(tok, tok.n_tokens() + 64, file.eos_ids());  // + padding ids past the tokenizer, as model vocabularies have

  const std::string think = "Let me see.\n</think>\n\n";  // the prompt opened <think>
  const std::string paris = xml_call("get_weather", {{"city", "Paris"}});
  const std::string tokyo = xml_call("get_time", {{"tz", "Asia/Tokyo"}});

  // XML calls (Qwen3.5, qwen3_coder), thinking open in the prompt.
  std::string auto_tag = reply_grammar(kXml, kTools, rules(Rules::kAuto), true);
  expect(g, tok, "auto: text", auto_tag, think + "Hello!", true);
  expect(g, tok, "auto: a call", auto_tag, think + paris, true);
  expect(g, tok, "auto: text, then two calls", auto_tag, think + "Checking.\n\n" + paris + "\n" + tokyo, true);
  expect(g, tok, "auto: an enum value", auto_tag,
         think + xml_call("get_weather", {{"city", "Oslo"}, {"unit", "celsius"}}), true);
  expect(g, tok, "auto: a value outside the enum", auto_tag,
         think + xml_call("get_weather", {{"city", "Oslo"}, {"unit", "kelvin"}}), false);
  expect(g, tok, "auto: an unknown tool", auto_tag, think + xml_call("get_stock", {{"sym", "X"}}), false);
  expect(g, tok, "auto: an unknown parameter", auto_tag, think + xml_call("get_time", {{"zone", "UTC"}}), false);
  expect(g, tok, "auto: a required parameter missing", auto_tag, think + xml_call("get_weather", {}), false);
  expect(g, tok, "auto: a second <think>", auto_tag, think + "<think>again</think>", false);

  std::string required = reply_grammar(kXml, kTools, rules(Rules::kRequired), true);
  expect(g, tok, "required: text only", required, think + "Hello!", false);
  expect(g, tok, "required: a call", required, think + tokyo, true);

  std::string forced = reply_grammar(kXml, kTools, rules(Rules::kForced, "get_time"), true);
  expect(g, tok, "forced: the named tool", forced, think + tokyo, true);
  expect(g, tok, "forced: another tool", forced, think + paris, false);
  expect(g, tok, "forced: text first", forced, think + "Sure. " + tokyo, false);

  std::string single = reply_grammar(kXml, kTools, rules(Rules::kRequired, "", false), true);
  expect(g, tok, "one call: one", single, think + paris, true);
  expect(g, tok, "one call: two", single, think + paris + "\n" + tokyo, false);

  // Thinking not opened by the prompt: a whole block, or none.
  std::string closed = reply_grammar(kXml, kTools, rules(Rules::kRequired), false);
  expect(g, tok, "closed thinking: a block, then a call", closed, "<think>\nhm\n</think>\n\n" + paris, true);
  expect(g, tok, "closed thinking: a call alone", closed, paris, true);

  // JSON calls (hermes, Qwen3).
  std::string hermes = reply_grammar(kJson, kTools, rules(Rules::kRequired), true);
  expect(g, tok, "hermes: a call", hermes, think + json_call("get_weather", R"({"city": "Paris"})"), true);
  expect(g, tok, "hermes: an unknown tool", hermes, think + json_call("get_stock", R"({"sym": "X"})"), false);
  expect(g, tok, "hermes: a wrong type", hermes, think + json_call("get_time", R"({"tz": 9})"), false);

  // A JSON answer (response_format).
  Rules json;
  json.json = true;
  json.schema = ojson::parse(R"({"type": "object", "properties": {"name": {"type": "string"},
    "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": false})");
  std::string person = reply_grammar(kXml, {}, json, true);
  expect(g, tok, "json: fits", person, think + R"({"name": "Ada", "age": 36})", true);
  expect(g, tok, "json: a field missing", person, think + R"({"name": "Ada"})", false);
  expect(g, tok, "json: text", person, think + "Ada, 36", false);

  for (bool xml : {true, false}) {
    const TagSyntax& syntax = xml ? kXml : kJson;
    const std::string call = xml ? xml_call("get_time", {{"tz", "UTC"}}) : json_call("get_time", R"({"tz": "UTC"})");
    const std::string two = call + "\n" + call;
    const std::string valid = R"({"name": "Ada", "age": 36})";
    Rules both = json;
    std::string tag = reply_grammar(syntax, kTools, both, true);
    expect(g, tok, "schema and tools: a JSON answer", tag, think + valid, true);
    expect(g, tok, "schema and tools: a JSON answer missing a field", tag, think + R"({"name": "Ada"})", false);
    expect(g, tok, "schema and tools: plain text cannot bypass the schema", tag, think + "hello", false);
    expect(g, tok, "schema and tools: parallel calls", tag, think + two, true);
    both.parallel = false;
    both.tools = Rules::kRequired;
    tag = reply_grammar(syntax, kTools, both, true);
    expect(g, tok, "schema and a required tool: a call", tag, think + call, true);
    expect(g, tok, "schema and a required tool: parallel disabled", tag, think + two, false);
    expect(g, tok, "schema and a required tool: a JSON answer alone", tag, think + valid, false);
    both.tools = Rules::kForced;
    both.forced = "get_time";
    tag = reply_grammar(syntax, kTools, both, true);
    expect(g, tok, "schema and a named tool: only a call", tag, think + call, true);
    expect(g, tok, "schema and a named tool: a JSON answer alone", tag, think + valid, false);
  }

  // XML zero-parameter tool call (matches Qwen3.5 template spelling without an extra blank line),
  // <tool_call> grammar bypass prevention, and JSON-escaped tool names.
  {
    std::vector<ojson> extra_tools = kTools;
    extra_tools.push_back(ojson::parse(
        R"({"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": false}}})"));
    std::string xml_auto = reply_grammar(kXml, extra_tools, rules(Rules::kAuto), true);
    expect(g, tok, "xml zero-param call", xml_auto, think + xml_call("ping", {}), true);
    expect(g, tok, "xml bypass: <tool_call> without newline", xml_auto,
           think + "<tool_call><function=get_stock><parameter=sym>\nX\n</parameter></function></tool_call>", false);
    expect(g, tok, "xml bypass: <tool_call> with space", xml_auto,
           think + "<tool_call> <function=get_stock></function></tool_call>", false);
    std::string xml_single = reply_grammar(kXml, extra_tools, rules(Rules::kRequired, "", false), true);
    expect(g, tok, "xml one call: cannot sneak second call without newline", xml_single,
           think + paris + "<tool_call><function=ping></function></tool_call>", false);

    std::vector<ojson> tricky = {ojson::parse(
        R"({"type": "function", "function": {"name": "say\"hi\\x", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})")};
    std::string hermes_tricky = reply_grammar(kJson, tricky, rules(Rules::kRequired), true);
    expect(g, tok, "hermes: JSON-escaped tool name", hermes_tricky,
           think + json_call(R"(say\"hi\\x)", R"({"city": "Paris"})"), true);
  }

  // Nothing to hold the reply to.
  if (!reply_grammar(kXml, {}, rules(Rules::kAuto), true).empty() ||
      !reply_grammar(kXml, kTools, rules(Rules::kNone), true).empty()) {
    ++failures;
    std::printf("FAIL a reply with no tools (or tool_choice none) and no JSON answer gets a grammar\n");
  }

  std::printf("%s: %d replies as expected, %d failures\n", failures ? "FAIL" : "ok", passed, failures);
  return failures ? 1 : 0;
}
