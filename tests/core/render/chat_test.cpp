// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Chat templates (core/render/chat.h) against transformers: every case of
// chat_cases.json (tests/e2e/make_tiny.py renders tests/core/render/chat_cases.py
// with the templates in templates/) must give the same text, the same branch
// points and the same open thinking block; the output language must be the
// one the template writes; and the tokens of a rendering hold special tokens
// only where the template wrote them, never where a client's string (content,
// a tool's definition, a call's arguments) put their text.
//
//   chat_test TINY.flyx CHAT_CASES.json TEMPLATES_DIR
#include <atomic>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "core/render/chat.h"
#include "core/render/chat_json.h"
#include "core/tokenizer/tokenizer.h"

using namespace flyx;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), {});
}

// Occurrences of `s` in text: all, and those inside one literal span.
std::pair<int, int> occurrences(const RenderedChat& r, const std::string& s) {
  int all = 0, lit = 0;
  for (size_t p = r.text.find(s); p != std::string::npos; p = r.text.find(s, p + s.size())) {
    ++all;
    for (const auto& span : r.literal)
      if (span.begin <= p && p + s.size() <= span.end) ++lit;
  }
  return {all, lit};
}

int count_id(const std::vector<int32_t>& ids, int32_t id) {
  int n = 0;
  for (int32_t t : ids) n += t == id;
  return n;
}

// The client's strings and keys with their special-token text taken out ("<|" -> "<_").
std::string defused(std::string s) {
  for (size_t p = s.find("<|"); p != std::string::npos; p = s.find("<|", p + 2)) s[p + 1] = '_';
  return s;
}
ojson defused(const ojson& j) {
  if (j.is_string()) return defused(j.get<std::string>());
  if (j.is_array()) {
    ojson a = ojson::array();
    for (const auto& v : j) a.push_back(defused(v));
    return a;
  }
  if (j.is_object()) {
    ojson o = ojson::object();
    for (auto it = j.begin(); it != j.end(); ++it) o[defused(it.key())] = defused(it.value());
    return o;
  }
  return j;
}

// A reply with one call to get_weather(location="Paris"), in the format's own words.
std::string reply_with_call(const std::string& format) {
  if (format == "qwen35" || format == "qwen3_coder")
    return "Need the weather.\n</think>\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n"
           "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
  return "<think>\nNeed the weather.\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": "
         "{\"location\": \"Paris\", \"days\": 3}}\n</tool_call>";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: chat_test TINY.flyx CHAT_CASES.json TEMPLATES_DIR\n");
    return 2;
  }
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(argv[1], &err) || !tok.load(file, &err)) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
  const auto cases = ojson::parse(read_file(argv[2]));
  const std::string dir = argv[3];
  const int32_t im_start = tok.added_id("<|im_start|>"), im_end = tok.added_id("<|im_end|>");
  const int32_t think = tok.added_id("<think>");
  check(im_start >= 0 && im_end >= 0 && think >= 0, "the tiny tokenizer has <|im_start|>, <|im_end|>, <think>");

  int n = 0;
  for (const auto& c : cases)
  for (bool native : {true, false}) {  // a format's own renderer where it has one, and the Jinja engine
    const std::string name = c["template"], label = name + " / " + c["case"].get<std::string>() +
                                                    (c["thinking"].get<bool>() ? " / thinking" : " / no thinking") +
                                                    (native ? "" : " / engine");
    // A fresh template per case: each render is a first request (no earlier conversation to continue).
    auto ct = std::make_unique<ChatTemplate>();
    if (!ct->load(file, read_file(dir + "/" + name), &err, native)) {
      check(false, name + ": " + err);
      continue;
    }
    check(ct->format() == c["format"].get<std::string>(),
          name + ": output language " + ct->format() + ", expected " + c["format"].get<std::string>());
    std::vector<ChatMessage> msgs;
    std::vector<ojson> tools;
    if (!messages_from_openai(c["messages"], &msgs, &err) ||
        (c["tools"].is_array() && !tools_from_openai(c["tools"], &tools, &err))) {
      check(false, label + ": " + err);
      continue;
    }
    ChatOptions co;
    co.enable_thinking = c["thinking"];
    RenderedChat r;
    if (!tools.empty() && std::string(ct->format()) == "plain") {  // calls this build cannot read back
      check(!ct->render(msgs, tools, co, &r, &err) && err.find("tool-call format") != std::string::npos,
            label + ": tools refused (" + err + ")");
      continue;
    }
    if (!ct->render(msgs, tools, co, &r, &err)) {
      check(false, label + ": " + err);
      continue;
    }
    ++n;
    const std::string want = c["text"];
    if (r.text != want) {
      size_t k = 0;
      while (k < r.text.size() && k < want.size() && r.text[k] == want[k]) ++k;
      check(false, label + ": text differs at byte " + std::to_string(k) + "\n  want: " +
                       want.substr(k > 40 ? k - 40 : 0, 120) + "\n  got:  " + r.text.substr(k > 40 ? k - 40 : 0, 120));
    }
    check(r.boundaries == c["boundaries"].get<std::vector<size_t>>(),
          label + ": branch points " + ojson(r.boundaries).dump() + ", expected " + c["boundaries"].dump());
    check(r.in_reasoning == c["in_reasoning"].get<bool>(), label + ": open thinking block");

    // Special tokens only where the template wrote them; <think> (an added
    // token that is not special) wherever it appears, as HuggingFace reads it.
    const auto ids = tok.encode(r.text, &r.literal);
    for (auto [s, id] : {std::pair<std::string, int32_t>{"<|im_start|>", im_start}, {"<|im_end|>", im_end}}) {
      auto [all, lit] = occurrences(r, s);
      check(count_id(ids, id) == lit, label + ": " + s + " read as a token " + std::to_string(count_id(ids, id)) +
                                          " times, the template wrote it " + std::to_string(lit) + " times");
      if (c["case"] == "special tokens in content" || c["case"] == "special tokens in tools")
        check(all > lit, label + ": the client's text holds " + s + " (the case tests nothing otherwise)");
    }
    check(count_id(ids, think) == occurrences(r, "<think>").first, label + ": <think> is read everywhere");
    // The same conversation with the client's special-token text taken out: the template writes the same special
    // tokens, so none of those it wrote came from the client (a tool's definition, a call's arguments).
    {
      std::vector<ChatMessage> m2;
      std::vector<ojson> t2;
      RenderedChat r2;
      const ojson tools2 = defused(c["tools"]);
      if (!messages_from_openai(defused(c["messages"]), &m2, &err) ||
          (tools2.is_array() && !tools_from_openai(tools2, &t2, &err)) || !ct->render(m2, t2, co, &r2, &err)) {
        check(false, label + ": without the client's special-token text: " + err);
      } else {
        for (const char* s : {"<|im_start|>", "<|im_end|>"})
          check(occurrences(r, s).second == occurrences(r2, s).second,
                label + ": the template wrote " + s + " " + std::to_string(occurrences(r2, s).second) +
                    " times; the spans it wrote hold it " + std::to_string(occurrences(r, s).second) + " times");
      }
    }

    // The reply parser for this format reads a call in the format's own words.
    if (!tools.empty() && c["thinking"].get<bool>()) {
      auto parser = ct->parser(r, &tools);
      std::vector<OutputEvent> ev = parser->feed(reply_with_call(ct->format()));
      for (auto& e : parser->finish()) ev.push_back(std::move(e));
      bool reasoning = false, call = false;
      for (const auto& e : ev) {
        reasoning |= e.kind == OutputEvent::kReasoning && e.text == "Need the weather.";
        call |= e.kind == OutputEvent::kToolCall && e.call.name == "get_weather" &&
                e.call.arguments == ojson{{"location", "Paris"}, {"days", 3}};
      }
      check(reasoning && call && parser->tool_calls() == 1, label + ": reply parser reads thinking and the call");
    }
  }

  // A request that continues an earlier one branches where the earlier conversation ended:
  // an agent loop, one call and its result, then the next.
  for (const auto& c : cases) {
    if (c["case"] != "agent loop" || !c["thinking"].get<bool>()) continue;
    ChatTemplate ct;
    ct.load(file, read_file(dir + "/" + c["template"].get<std::string>()), &err);
    if (std::string(ct.format()) == "plain") continue;  // no agent loops: tools refused above
    std::vector<ChatMessage> msgs;
    std::vector<ojson> tools;
    messages_from_openai(c["messages"], &msgs, &err);
    tools_from_openai(c["tools"], &tools, &err);
    std::vector<ChatMessage> first(msgs.begin(), msgs.begin() + 3);  // user, call, result
    RenderedChat r1, r2;
    check(ct.render(first, tools, {}, &r1, &err) && ct.render(msgs, tools, {}, &r2, &err), "agent loop renders");
    check(!r1.boundaries.empty() && r2.boundaries.size() >= 2 &&
              r2.boundaries[r2.boundaries.size() - 2] == r1.boundaries.back() &&
              r2.text.compare(0, r1.boundaries.back(), r1.text, 0, r1.boundaries.back()) == 0,
          c["template"].get<std::string>() + ": the next turn branches where the previous conversation ended " +
              ojson(r1.boundaries).dump() + " -> " + ojson(r2.boundaries).dump());
    // Where the first user message ends: the same in both, inside the shared text.
    check(r1.first_user_end > 0 && r1.first_user_end == r2.first_user_end &&
              r2.text.compare(0, r1.first_user_end, r1.text, 0, r1.first_user_end) == 0 &&
              r1.text.compare(r1.first_user_end - 11, 11, "<|im_end|>\n") == 0,
          c["template"].get<std::string>() + ": the first user message ends at " + std::to_string(r1.first_user_end));
  }

  // A Responses conversation (core/render/chat_json) is the chat that chat completions sends for
  // it: instructions, thinking, parallel calls and their outputs, text, a thinking block sent
  // back only as encrypted_content.
  {
    const ojson chat = ojson::parse(R"([
      {"role": "system", "content": "Be brief."},
      {"role": "user", "content": "Weather in Paris and Lyon?"},
      {"role": "assistant", "content": "", "reasoning_content": "Need the weather.", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Paris\"}"}},
        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Lyon\"}"}}]},
      {"role": "tool", "tool_call_id": "call_1", "content": "18 C"},
      {"role": "tool", "tool_call_id": "call_2", "content": "20 C"},
      {"role": "assistant", "content": "Paris 18 C, Lyon 20 C.", "reasoning_content": "Both known."},
      {"role": "user", "content": "Thanks"}])");
    const ojson items = ojson::parse(R"([
      {"role": "user", "content": "Weather in Paris and Lyon?"},
      {"type": "reasoning", "id": "rs_1", "summary": [], "content": [{"type": "reasoning_text", "text": "Need the weather."}]},
      {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"location\": \"Paris\"}"},
      {"type": "function_call", "call_id": "call_2", "name": "get_weather", "arguments": "{\"location\": \"Lyon\"}"},
      {"type": "function_call_output", "call_id": "call_1", "output": "18 C"},
      {"type": "function_call_output", "call_id": "call_2", "output": [{"type": "input_text", "text": "20 C"}]},
      {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "flyx:Both known."},
      {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris 18 C, Lyon 20 C."}]},
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Thanks"}]}])");
    std::vector<ChatMessage> a, b;
    check(messages_from_openai(chat, &a, &err), "chat messages: " + err);
    check(messages_from_responses("Be brief.", items, &b, &err), "Responses items: " + err);
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i) {
      same = a[i].role == b[i].role && a[i].content == b[i].content && a[i].reasoning == b[i].reasoning &&
             a[i].tool_call_id == b[i].tool_call_id && a[i].tool_calls.size() == b[i].tool_calls.size();
      for (size_t k = 0; same && k < a[i].tool_calls.size(); ++k)
        same = a[i].tool_calls[k].id == b[i].tool_calls[k].id && a[i].tool_calls[k].name == b[i].tool_calls[k].name &&
               a[i].tool_calls[k].arguments == b[i].tool_calls[k].arguments;
    }
    check(same, "Responses items give the chat messages of the same conversation");
    std::vector<ojson> ta, tb;
    tools_from_openai(ojson::parse(R"([{"type": "function", "function": {"name": "get_weather", "description": "d",
      "parameters": {"type": "object"}}}])"), &ta, &err);
    check(tools_from_responses(ojson::parse(R"([{"type": "function", "name": "get_weather", "description": "d",
      "parameters": {"type": "object"}}])"), &tb, &err) && ta == tb, "Responses tools are the chat tools: " + err);
    ChatTemplate q;
    q.load(file, read_file(dir + "/qwen3.5.jinja"), &err);
    RenderedChat ra, rb;
    check(q.render(a, ta, {}, &ra, &err) && q.render(b, tb, {}, &rb, &err) && ra.text == rb.text,
          "a Responses conversation renders as its chat");
    ReplyRules rules;
    check(rules_from_responses(ojson::parse(R"({"tool_choice": {"type": "function", "name": "get_weather"},
      "parallel_tool_calls": false, "text": {"format": {"type": "json_schema", "name": "x",
      "schema": {"type": "object"}}}})"), tb, &rules, &err) && rules.tools == ReplyRules::kForced &&
              rules.forced == "get_weather" && !rules.parallel && rules.json && rules.schema == ojson{{"type", "object"}},
          "Responses tool_choice, parallel_tool_calls and text.format: " + err);
    ojson grouped = ojson::parse(R"([
      {"type":"function","name":"read"},
      {"type":"namespace","name":"files","description":"Project files.","tools":[
        {"type":"function","name":"read","description":"Read one.","parameters":{"type":"object"}}]}])");
    ResponsesNamespaces namespaces;
    check(tools_from_responses(grouped, &tb, &err, &namespaces) && tb.size() == 2 &&
              tb[0]["function"]["name"] == "read" && tb[1]["function"]["name"] == "files.read" &&
              tb[1]["function"]["description"] == "Project files.\nRead one." &&
              namespaces == ResponsesNamespaces{{"files.read", "files"}},
          "same-named top-level and namespaced functions keep distinct model names");
    ojson forced = ojson::parse(R"({"tool_choice":{"type":"function","namespace":"files","name":"read"}})");
    check(rules_from_responses(forced, tb, &rules, &err, namespaces) && rules.forced == "files.read",
          "Responses tool choice identifies a namespace member");
    forced["tool_choice"] = {{"type", "function"}, {"name", "files.read"}};
    check(!rules_from_responses(forced, tb, &rules, &err, namespaces),
          "the name of a top-level function cannot select a namespaced function");
    ojson history = ojson::parse(R"([{"type":"function_call","call_id":"c","namespace":"files","name":"read"},
                                     {"type":"function_call_output","call_id":"c","output":"hello"}])");
    check(messages_from_responses("", history, &b, &err) && b.size() == 2 &&
              b[0].tool_calls[0].name == "files.read" && b[1].tool_call_id == "c",
          "a history containing a namespaced call renders the same qualified name");
    grouped.push_back({{"type", "function"}, {"name", "files.read"}});
    check(!tools_from_responses(grouped, &tb, &err), "ambiguous qualified and literal names are refused");
    grouped[1]["tools"][0]["type"] = "custom";
    check(!tools_from_responses(grouped, &tb, &err), "unsupported members of a namespace are refused");
    check(!messages_from_responses("", ojson::parse(R"([{"type": "item_reference", "id": "x"}])"), &b, &err) &&
              !tools_from_responses(ojson::parse(R"([{"type": "web_search"}])"), &tb, &err) &&
              !messages_from_responses("Only instructions.", ojson::array(), &b, &err),
          "item references, other tool types and an empty input are refused");
  }

  // Claude Code's per-request billing line is left out of the system prompt (a block of its own,
  // or the first line of the system text), and kept when asked.
  {
    std::vector<ChatMessage> m;
    std::vector<ojson> t;
    bool dropped = false;
    const ojson block = ojson::parse(R"({"system": [
      {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.3.abc; cc_entrypoint=cli; cch=1a2b3;"},
      {"type": "text", "text": "You are Claude Code."}], "messages": [{"role": "user", "content": "hi"}]})");
    const ojson line = ojson::parse(R"({"system": "x-anthropic-billing-header: cc_version=2.1.3; cch=9;\nYou are Claude Code.",
      "messages": [{"role": "user", "content": "hi"}]})");
    const ojson mention = ojson::parse(R"({"system": "Never print x-anthropic-billing-header: lines.",
      "messages": [{"role": "user", "content": "hi"}]})");
    check(chat_from_anthropic(block, &m, &t, &err, false, &dropped) && dropped && m.at(0).content == "You are Claude Code.",
          "the billing block is left out");
    check(chat_from_anthropic(line, &m, &t, &err, false, &dropped) && dropped && m.at(0).content == "You are Claude Code.",
          "the billing line is left out");
    check(chat_from_anthropic(block, &m, &t, &err, true, &dropped) && !dropped &&
              m.at(0).content.rfind("x-anthropic-billing-header:", 0) == 0,
          "the billing line kept when asked");
    check(chat_from_anthropic(mention, &m, &t, &err, false, &dropped) && !dropped &&
              m.at(0).content == "Never print x-anthropic-billing-header: lines.",
          "a mention of the header is not a billing line");

    // Empty system blocks marked with cache_control must not emit a bogus cache point for message 1 (user).
    std::vector<ChatCachePoint> cps;
    const ojson empty_sys = ojson::parse(R"({"system": [
      {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}],
      "messages": [{"role": "user", "content": "hi"}]})");
    check(chat_from_anthropic(empty_sys, &m, &t, &err, false, &dropped, &cps) && m.size() == 1 &&
              m[0].role == "user" && cps.empty(),
          "empty cached system block emits no misplaced cache point");

    // Wrong JSON types on chat/responses/anthropic request fields are refused cleanly.
    std::vector<ChatMessage> bad_m;
    std::vector<ojson> bad_t;
    ReplyRules bad_r;
    check(!messages_from_openai(ojson::parse(R"([{"role": "user", "content": [123]}])"), &bad_m, &err),
          "non-object OpenAI content part refused");
    check(!messages_from_openai(ojson::parse(R"([{"role": "assistant", "content": "", "tool_calls": [123]}])"),
                                &bad_m, &err),
          "non-object OpenAI tool_calls entry refused");
    check(!rules_from_openai(ojson::parse(R"({"response_format": "json_object"})"), {}, &bad_r, &err),
          "string response_format refused");
    check(!rules_from_openai(ojson::parse(R"({"response_format": {"type": "json_schema", "json_schema": "bad"}})"),
                             {}, &bad_r, &err),
          "non-object response_format.json_schema refused");
    check(!rules_from_openai(ojson::parse(R"({"parallel_tool_calls": "no"})"), {}, &bad_r, &err),
          "non-boolean parallel_tool_calls refused");
    check(!rules_from_responses(ojson::parse(R"({"text": "bad"})"), {}, &bad_r, &err) &&
              !rules_from_responses(ojson::parse(R"({"text": {"format": "json_object"}})"), {}, &bad_r, &err),
          "non-object Responses text / text.format refused");
    check(!rules_from_anthropic(ojson::parse(R"({"tool_choice": "auto"})"), {}, &bad_r, &err) &&
              !rules_from_anthropic(ojson::parse(R"({"tool_choice": {"type": "auto", "disable_parallel_tool_use": "no"}})"),
                                    {}, &bad_r, &err),
          "malformed Anthropic tool_choice refused");
    check(!chat_from_anthropic(ojson::parse(R"({"system": 123, "messages": [{"role": "user", "content": "hi"}]})"),
                               &bad_m, &bad_t, &err) &&
              !chat_from_anthropic(ojson::parse(R"({"system": ["bad"], "messages": [{"role": "user", "content": "hi"}]})"),
                                   &bad_m, &bad_t, &err) &&
              !chat_from_anthropic(ojson::parse(R"({"tools": "bad", "messages": [{"role": "user", "content": "hi"}]})"),
                                   &bad_m, &bad_t, &err),
          "malformed Anthropic system / tools refused");
  }

  // The tiny model's own template writes no tool calls: tools are refused, plain chat renders.
  ChatTemplate own;
  check(own.load(file, "", &err), "tiny.flyx's own template loads: " + err);
  check(std::string(own.format()) == "plain", std::string("tiny.flyx's template: plain, got ") + own.format());
  RenderedChat r;
  std::vector<ChatMessage> hi{{"user", "hi", std::nullopt, {}, ""}};
  check(!own.render(hi, {ojson{{"type", "function"}, {"function", {{"name", "f"}}}}}, {}, &r, &err) &&
            err.find("tool-call format") != std::string::npos,
        "tools refused when the template has no tool-call format (" + err + ")");
  check(own.render(hi, {}, {}, &r, &err) && r.text.rfind("<|im_start|>", 0) == 0, "plain chat renders: " + err);

  // The template's own error reaches the caller (raise_exception), without the source excerpt.
  ChatTemplate q35;
  q35.load(file, read_file(dir + "/qwen3.5.jinja"), &err);
  std::vector<ChatMessage> only_system{{"system", "s", std::nullopt, {}, ""}};
  check(!q35.render(only_system, {}, {}, &r, &err) && err.find("No user query found") != std::string::npos &&
            err.find('\n') == std::string::npos,
        "a template's raise_exception becomes the error: " + err);

  // Direct Jinja value builder & probe caches: invalid UTF-8 replacement, key-order cache isolation,
  // and concurrent multi-threaded renders on a shared Jinja-engine ChatTemplate.
  {
    ChatTemplate jinja_q35;
    check(jinja_q35.load(file, read_file(dir + "/qwen3.5.jinja"), &err, false),
          "qwen3.5 jinja engine loads: " + err);
    // Invalid UTF-8 bytes in user content are replaced with U+FFFD rather than failing.
    std::vector<ChatMessage> bad_utf8{{"user", std::string("hi \xFF\xFE bye"), std::nullopt, {}, ""}};
    RenderedChat ru;
    check(jinja_q35.render(bad_utf8, {}, {}, &ru, &err) &&
              ru.text.find("hi \xEF\xBF\xBD\xEF\xBF\xBD bye") != std::string::npos,
          "invalid UTF-8 replaced identically in Jinja engine: " + err);

    // Tool definitions differing only in object key insertion order must not alias in probe caches.
    const std::vector<ojson> tools_ab = {ojson::parse(
        R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"alpha":{"type":"string"},"beta":{"type":"integer"}}}}})")};
    const std::vector<ojson> tools_ba = {ojson::parse(
        R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"beta":{"type":"integer"},"alpha":{"type":"string"}}}}})")};
    const std::vector<ChatMessage> conv_msgs{
        {"system", "You are helpful.", std::nullopt, {}, ""},
        {"user", "first question", std::nullopt, {}, ""},
        {"assistant", "first answer", std::nullopt, {}, ""},
        {"user", "second question", std::nullopt, {}, ""},
    };
    const std::vector<ChatCachePoint> cps{{0, 0, 0, 0, 1}, {1, 16, 0, 0, 0}};
    RenderedChat rab1, rba, rab2;
    check(jinja_q35.render(conv_msgs, tools_ab, {}, &rab1, &err, true, cps) &&
              jinja_q35.render(conv_msgs, tools_ba, {}, &rba, &err, true, cps) &&
              jinja_q35.render(conv_msgs, tools_ab, {}, &rab2, &err, true, cps),
          "key-order probe cache renders succeed: " + err);
    check(rab1.text != rba.text && rab1.text == rab2.text && rab1.system_end == rab2.system_end &&
              rab1.cache_prefixes == rab2.cache_prefixes && rab1.first_user_end == rab2.first_user_end &&
              rab1.system_end > 0 && rba.system_end > 0 && rba.system_end == rab1.system_end &&
              rba.text.find("\"beta\"") < rba.text.find("\"alpha\"") &&
              rab1.text.find("\"alpha\"") < rab1.text.find("\"beta\""),
          "probe caches preserve parameter key order and match fresh renders");

    // Concurrent multi-threaded renders sharing probe caches match a fresh uncached ChatTemplate.
    ChatTemplate ref_ct;
    ref_ct.load(file, read_file(dir + "/qwen3.5.jinja"), &err, false);
    RenderedChat want_think, want_plain;
    ChatOptions opt_think, opt_plain;
    opt_think.enable_thinking = true;
    opt_plain.enable_thinking = false;
    check(ref_ct.render(conv_msgs, tools_ab, opt_think, &want_think, &err, true, cps) &&
              ref_ct.render(conv_msgs, tools_ba, opt_plain, &want_plain, &err, true, cps),
          "reference renders succeed");
    std::atomic<int> thread_errs{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
      workers.emplace_back([&, t] {
        for (int iter = 0; iter < 16; ++iter) {
          const bool use_think = ((t + iter) & 1) == 0;
          RenderedChat got;
          std::string e;
          const auto& want = use_think ? want_think : want_plain;
          if (!jinja_q35.render(conv_msgs, use_think ? tools_ab : tools_ba, use_think ? opt_think : opt_plain, &got, &e,
                                true, cps) ||
              got.text != want.text || got.system_end != want.system_end ||
              got.cache_prefixes != want.cache_prefixes || got.conversation_end != want.conversation_end ||
              got.first_user_end != want.first_user_end || got.in_reasoning != want.in_reasoning) {
            thread_errs.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto& w : workers) w.join();
    check(thread_errs.load() == 0, "concurrent Jinja renders with shared probe caches match reference");
  }

  // probe_conversation_end() must never scan untrusted message content when the 1-message probe does not match.
  {
    const std::string sys_req_tpl =
        "{% if messages[0]['role'] != 'system' %}{{ raise_exception('need system') }}{% endif %}"
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
        "{% if m['reasoning_content'] %}<think>{{ m['reasoning_content'] }}</think>{% endif %}"
        "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking %}<think>\n{% endif %}{% endif %}";
    ChatTemplate ct;
    check(ct.load(file, sys_req_tpl, &err, false), "system-required thinking template loads: " + err);
    const std::vector<ChatMessage> msgs{
        {"system", "You are helpful.", std::nullopt, {}, ""},
        {"user", "Explain what <think> does in XML.", std::nullopt, {}, ""},
    };
    ChatOptions no_think;
    no_think.enable_thinking = false;
    RenderedChat r_no, r_yes;
    check(ct.render(msgs, {}, no_think, &r_no, &err) && !r_no.in_reasoning,
          "untrusted user '<think>' does not hijack in_reasoning when gen probe fails");
    ChatOptions yes_think;
    yes_think.enable_thinking = true;
    check(ct.render(msgs, {}, yes_think, &r_yes, &err) && r_yes.in_reasoning,
          "trailing template literal '<think>' still opens in_reasoning when gen probe fails");
  }

  // Anthropic user message whose content array contains only empty text block(s) must still emit a user message.
  {
    std::vector<ChatMessage> ms;
    std::vector<ojson> ts;
    const ojson req = ojson::parse(R"({"messages":[{"role":"user","content":[{"type":"text","text":""}]}]})");
    check(chat_from_anthropic(req, &ms, &ts, &err) && ms.size() == 1 && ms[0].role == "user" && ms[0].content.empty(),
          "Anthropic user message with single empty text block is preserved");
  }

  std::printf("chat: %d renders, %s\n", n, failures ? "FAILED" : "all match transformers");
  return failures ? 1 : 0;
}
