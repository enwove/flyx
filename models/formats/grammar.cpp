// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The structural tag (XGrammar's JSON) a reply in a tag syntax must match.
// The shape follows XGrammar's own builders for these
// families (its Python get_model_structural_tag: "qwen_3_5" for XML calls,
// "qwen_3" for JSON calls): an optional thinking block, then free text that
// turns into a call at the call's opening, each call to a declared tool with
// arguments that fit its schema.
#include <utility>

#include "models/formats/tags.h"

namespace flyx::models::formats {
namespace {

ojson any_text(const TagSyntax& s) {
  ojson t = {{"type", "any_text"}};
  t["excludes"] = s.thinking ? ojson::array({s.think_open, s.think_close}) : ojson::array();
  return t;
}

// The thinking block: the rest of one the prompt opened, else an optional whole one.
ojson thinking(const TagSyntax& s, bool in_reasoning) {
  ojson block = ojson::array({ojson{{"type", "tag"}, {"begin", in_reasoning ? "" : s.think_open},
                                    {"content", any_text(s)}, {"end", s.think_close}}});
  if (*s.think_after) block.push_back(ojson{{"type", "const_string"}, {"value", s.think_after}});
  ojson seq = {{"type", "sequence"}, {"elements", block}};
  if (in_reasoning) return seq;
  return {{"type", "optional"}, {"content", seq}};
}

// The arguments' schema: the tool's parameters; anything when it opts out of strictness.
ojson arguments_schema(const ojson& fn) {
  if (fn.contains("strict") && fn["strict"].is_boolean() && !fn["strict"].get<bool>()) return {{"type", "object"}};
  for (const char* key : {"parameters", "input_schema"})
    if (fn.contains(key) && (fn[key].is_object() || fn[key].is_boolean())) return fn[key];
  return {{"type", "object"}};
}

ojson call_tag(const CallSyntax& w, const ojson& fn) {
  std::string name = fn.contains("name") && fn["name"].is_string() ? fn["name"].get<std::string>() : "";
  if (std::string_view(w.style) == "json") {
    const std::string dumped = ojson(name).dump(-1, ' ', false, ojson::error_handler_t::replace);
    if (dumped.size() >= 2) name = dumped.substr(1, dumped.size() - 2);
  }
  return {{"type", "tag"},
          {"begin", std::string(w.begin_prefix) + name + w.begin_suffix},
          {"content", {{"type", "json_schema"}, {"json_schema", arguments_schema(fn)}, {"style", w.style}}},
          {"end", w.end}};
}

}  // namespace

std::string reply_grammar(const TagSyntax& syntax, const std::vector<ojson>& tools,
                          const interfaces::ReplyRules& rules, bool in_reasoning) {
  using Rules = interfaces::ReplyRules;
  const bool calls = syntax.written && !tools.empty() && rules.tools != Rules::kNone;
  auto json_answer = [&] { return ojson{{"type", "json_schema"}, {"json_schema", rules.schema}, {"style", "json"}}; };
  ojson answer;
  if (calls) {
    const CallSyntax& w = *syntax.written;
    ojson tags = ojson::array();
    for (const auto& t : tools) {
      const ojson* fn = function_of(t);
      const std::string fname =
          fn && fn->contains("name") && (*fn)["name"].is_string() ? (*fn)["name"].get<std::string>() : "";
      if (!fn || (rules.tools == Rules::kForced && fname != rules.forced)) continue;
      tags.push_back(call_tag(w, *fn));
    }
    if (tags.empty()) return "";
    if (rules.tools == Rules::kForced) {
      answer = tags[0];  // exactly the one call
    } else if (rules.json) {
      // A structured answer cannot contain free-form text around a tool call.
      ojson calls_only = {{"type", "tags_with_separator"}, {"tags", std::move(tags)}, {"separator", "\n"},
                          {"at_least_one", true}, {"stop_after_first", !rules.parallel}};
      if (rules.tools == Rules::kAuto)
        answer = {{"type", "or"}, {"elements", ojson::array({json_answer(), std::move(calls_only)})}};
      else
        answer = std::move(calls_only);
    } else {
      answer = {{"type", "triggered_tags"},
                {"triggers", ojson::array({w.trigger})},
                {"tags", tags},
                {"excludes", any_text(syntax)["excludes"]},
                {"at_least_one", rules.tools == Rules::kRequired},
                {"stop_after_first", !rules.parallel}};
    }
  } else if (rules.json) {
    answer = json_answer();
  } else {
    return "";
  }
  // A thinking block where the request left thinking on, or where the prompt opened one (it is closed first).  Off,
  // the prompt closed an empty one: a block allowed here, a model that thinks by default opens it again and may never
  // come to the answer (Qwen3-8B with a json_schema answer thought until max_tokens ran out).
  const bool think = syntax.thinking && (rules.thinking || in_reasoning);
  ojson format = think
                     ? ojson{{"type", "sequence"}, {"elements", ojson::array({thinking(syntax, in_reasoning), answer})}}
                     : answer;
  return ojson{{"type", "structural_tag"}, {"format", format}}.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

}  // namespace flyx::models::formats
