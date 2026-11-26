// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/inkling2.h"

#include <algorithm>
#include <array>

#include "models/formats/calls.h"

namespace flyx::models::formats {
namespace {

constexpr std::string_view kMsgModel = "<|message_model|>";
constexpr std::string_view kContentThinking = "<|content_thinking|>";
constexpr std::string_view kContentText = "<|content_text|>";
constexpr std::string_view kContentToolError = "<|content_tool_error|>";
constexpr std::string_view kContentInvokeText = "<|content_invoke_tool_text|>";
constexpr std::string_view kContentInvokeJson = "<|content_invoke_tool_json|>";
constexpr std::string_view kEndMessage = "<|end_message|>";
constexpr std::string_view kEndSampling = "<|content_model_end_sampling|>";
constexpr std::array kReplyMarkers = {kMsgModel, kContentThinking, kContentText, kContentInvokeJson, kEndMessage};

std::string dump_safe(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

size_t partial_tail(std::string_view s, std::string_view tag) {
  for (size_t l = std::min(s.size(), tag.size() - 1); l > 0; --l)
    if (s.substr(s.size() - l) == tag.substr(0, l)) return l;
  return 0;
}

size_t max_partial_tail(std::string_view s, std::initializer_list<std::string_view> tags) {
  size_t best = 0;
  for (std::string_view t : tags) best = std::max(best, partial_tail(s, t));
  return best;
}

bool parse_inkling2_tool_call(const std::string& body, const std::string& recipient, interfaces::ToolCall* out) {
  const ojson j = interfaces::parse_json(body);
  if (!j.is_object()) return false;
  std::string name;
  if (j.contains("name") && j["name"].is_string()) name = j["name"].get<std::string>();
  else name = recipient;
  if (name.empty()) return false;
  ojson args = ojson::object();
  const char* arg_key = j.contains("args") ? "args" : (j.contains("arguments") ? "arguments" : nullptr);
  if (arg_key) {
    const ojson& a = j[arg_key];
    if (a.is_object()) {
      args = a;
    } else if (a.is_string()) {
      const ojson parsed = interfaces::parse_json(a.get_ref<const std::string&>());
      if (!parsed.is_object()) return false;
      args = parsed;
    } else if (!a.is_null()) {
      return false;
    }
  }
  out->id.clear();
  out->name = std::move(name);
  out->arguments = std::move(args);
  return true;
}

void push_text(std::vector<OutputEvent>* ev, OutputEvent::Kind k, std::string_view s) {
  if (s.empty()) return;
  if (!ev->empty() && ev->back().kind == k) ev->back().text.append(s);
  else ev->push_back({k, std::string(s), {}});
}

class Inkling2Parser final : public interfaces::OutputParser {
 public:
  explicit Inkling2Parser(const std::vector<ojson>* tools) : tools_(tools) {}

  std::vector<OutputEvent> feed(std::string_view text) override {
    std::vector<OutputEvent> ev;
    buf_.append(text);
    run(&ev, false);
    return ev;
  }

  std::vector<OutputEvent> finish() override {
    std::vector<OutputEvent> ev;
    run(&ev, true);
    return ev;
  }

  int tool_calls() const override { return n_calls_; }

 private:
  enum Mode { kIdle, kHeader, kThinking, kText, kToolJson };

  bool allow_tools() const { return tools_ && !tools_->empty(); }

  void emit_tool_json(std::vector<OutputEvent>* ev) {
    interfaces::ToolCall tc;
    if (allow_tools() && parse_inkling2_tool_call(call_buf_, recipient_, &tc)) {
      interfaces::ToolCall start_tc;
      start_tc.name = tc.name;
      ev->push_back({OutputEvent::kToolCallStart, {}, std::move(start_tc)});
      const std::string args_str = dump_safe(tc.arguments);
      if (!args_str.empty()) ev->push_back({OutputEvent::kToolCallDelta, args_str, {}});
      ev->push_back({OutputEvent::kToolCall, {}, std::move(tc)});
      ++n_calls_;
    } else {
      push_text(ev, OutputEvent::kContent, call_buf_);
    }
    call_buf_.clear();
    recipient_.clear();
  }

  bool try_enter_content(size_t pos) {
    auto take = [&](std::string_view tag, Mode next) {
      if (buf_.compare(pos, tag.size(), tag) != 0) return false;
      buf_.erase(0, pos + tag.size());
      mode_ = next;
      if (next == kToolJson) call_buf_.clear();
      return true;
    };
    return take(kContentThinking, kThinking) || take(kContentText, kText) || take(kContentToolError, kText) ||
           take(kContentInvokeText, kText) || take(kContentInvokeJson, allow_tools() ? kToolJson : kText);
  }

  void run(std::vector<OutputEvent>* ev, bool final) {
    while (!buf_.empty() || (final && mode_ == kToolJson && !call_buf_.empty())) {
      if (mode_ == kIdle) {
        if (buf_.starts_with(kEndSampling)) {
          buf_.erase(0, kEndSampling.size());
          continue;
        }
        if (buf_.starts_with(kEndMessage)) {
          buf_.erase(0, kEndMessage.size());
          continue;
        }
        if (buf_.starts_with(kMsgModel)) {
          buf_.erase(0, kMsgModel.size());
          recipient_.clear();
          mode_ = kHeader;
          continue;
        }
        if (try_enter_content(0)) {
          recipient_.clear();
          continue;
        }
        const size_t lt = buf_.find('<');
        if (lt == std::string::npos) {
          push_text(ev, OutputEvent::kContent, buf_);
          buf_.clear();
          return;
        }
        if (lt > 0) {
          push_text(ev, OutputEvent::kContent, std::string_view(buf_).substr(0, lt));
          buf_.erase(0, lt);
          continue;
        }
        // buf_ starts with '<': wait if it could complete a known Inkling-2 control token.
        if (!final) {
          const auto is_prefix = [&](std::string_view tag) {
            return buf_.size() < tag.size() && tag.starts_with(buf_);
          };
          if (is_prefix(kMsgModel) || is_prefix(kContentThinking) || is_prefix(kContentText) ||
              is_prefix(kContentToolError) || is_prefix(kContentInvokeText) || is_prefix(kContentInvokeJson) ||
              is_prefix(kEndMessage) || is_prefix(kEndSampling))
            return;
        }
        push_text(ev, OutputEvent::kContent, "<");
        buf_.erase(0, 1);
        continue;
      }

      if (mode_ == kHeader) {
        const size_t lt = buf_.find('<');
        if (lt == std::string::npos) {
          if (final) {
            push_text(ev, OutputEvent::kContent, recipient_ + buf_);
            recipient_.clear();
            buf_.clear();
            mode_ = kIdle;
          }
          return;
        }
        if (lt > 0) {
          recipient_.append(buf_, 0, lt);
          buf_.erase(0, lt);
        }
        if (try_enter_content(0)) continue;
        if (!final) {
          const auto is_prefix = [&](std::string_view tag) {
            return buf_.size() < tag.size() && tag.starts_with(buf_);
          };
          if (is_prefix(kContentThinking) || is_prefix(kContentText) || is_prefix(kContentToolError) ||
              is_prefix(kContentInvokeText) || is_prefix(kContentInvokeJson) || is_prefix(kEndSampling))
            return;
        }
        if (buf_.starts_with(kEndSampling)) {
          buf_.erase(0, kEndSampling.size());
          recipient_.clear();
          mode_ = kIdle;
          continue;
        }
        // Unexpected tag inside header: fall back to content.
        push_text(ev, OutputEvent::kContent, recipient_);
        recipient_.clear();
        mode_ = kIdle;
        continue;
      }

      // Inside a section body: kThinking, kText, or kToolJson.
      const size_t end_msg = buf_.find(kEndMessage);
      const size_t end_smp = buf_.find(kEndSampling);
      size_t stop = std::min(end_msg, end_smp);
      if (stop != std::string::npos) {
        const std::string_view piece(buf_.data(), stop);
        if (mode_ == kThinking) push_text(ev, OutputEvent::kReasoning, piece);
        else if (mode_ == kText) push_text(ev, OutputEvent::kContent, piece);
        else {
          call_buf_.append(piece);
          emit_tool_json(ev);
        }
        const size_t tag_len = (stop == end_msg) ? kEndMessage.size() : kEndSampling.size();
        buf_.erase(0, stop + tag_len);
        recipient_.clear();
        mode_ = kIdle;
        continue;
      }

      const size_t keep = final ? 0 : max_partial_tail(buf_, {kEndMessage, kEndSampling});
      const size_t safe = buf_.size() - keep;
      if (safe > 0) {
        const std::string_view piece(buf_.data(), safe);
        if (mode_ == kThinking) push_text(ev, OutputEvent::kReasoning, piece);
        else if (mode_ == kText) push_text(ev, OutputEvent::kContent, piece);
        else call_buf_.append(piece);
        buf_.erase(0, safe);
      }
      if (final && mode_ == kToolJson) {
        emit_tool_json(ev);
        mode_ = kIdle;
      }
      return;
    }
  }

  const std::vector<ojson>* tools_;
  Mode mode_ = kIdle;
  std::string buf_;
  std::string recipient_;
  std::string call_buf_;
  int n_calls_ = 0;
};

ojson arguments_schema(const ojson& fn) {
  if (fn.contains("strict") && fn["strict"].is_boolean() && !fn["strict"].get<bool>()) return {{"type", "object"}};
  for (const char* key : {"parameters", "input_schema"})
    if (fn.contains(key) && (fn[key].is_object() || fn[key].is_boolean())) return fn[key];
  return {{"type", "object"}};
}

// A section's text: anything but the two markers that end a section for the parser, spelled out in ordinary tokens
// (as tokens no text holds them: inkling2_reply_tokens, and the stop token).  The markers that open one are text
// here when spelled out, as they are to the parser (a reply may quote them): a section begins only after the one
// before it has ended, where the grammar puts it.
ojson any_section_body() {
  return {{"type", "any_text"}, {"excludes", ojson::array({std::string(kEndMessage), std::string(kEndSampling)})}};
}

ojson thinking_section() {
  return {{"type", "optional"},
          {"content",
           {{"type", "tag"},
            {"begin", "<|message_model|><|content_thinking|>"},
            {"content", any_section_body()},
            {"end", "<|end_message|>"}}}};
}

ojson text_section() {
  return {{"type", "tag"},
          {"begin", "<|message_model|><|content_text|>"},
          {"content", any_section_body()},
          {"end", "<|end_message|>"}};
}

ojson json_text_section(const ojson& schema) {
  return {{"type", "tag"},
          {"begin", "<|message_model|><|content_text|>"},
          {"content", {{"type", "json_schema"}, {"json_schema", schema}, {"style", "json"}}},
          {"end", "<|end_message|>"}};
}

// A call's header holds the tool's name between two markers: a name holding a marker's text would put that marker
// there, where the grammar takes its token and the parser reads the reply's own.
bool holds_reply_marker(std::string_view name) {
  for (std::string_view marker : kReplyMarkers)
    if (name.find(marker) != std::string_view::npos) return true;
  return false;
}

ojson call_section(const ojson& fn) {
  const std::string name = fn.contains("name") && fn["name"].is_string() ? fn["name"].get<std::string>() : "";
  const std::string quoted = dump_safe(ojson(name));
  return {{"type", "tag"},
          {"begin", "<|message_model|>" + name + "<|content_invoke_tool_json|>{\"name\":" + quoted + ",\"args\":"},
          {"content", {{"type", "json_schema"}, {"json_schema", arguments_schema(fn)}, {"style", "json"}}},
          {"end", "}<|end_message|>"}};
}

}  // namespace

bool is_inkling2_template(std::string_view source) {
  return source.find(kMsgModel) != std::string_view::npos &&
         source.find(kContentThinking) != std::string_view::npos &&
         source.find(kEndMessage) != std::string_view::npos;
}

std::vector<std::string> inkling2_reply_tokens() {
  return {kReplyMarkers.begin(), kReplyMarkers.end()};
}

std::unique_ptr<interfaces::OutputParser> make_inkling2_parser(const std::vector<ojson>* tools) {
  return std::make_unique<Inkling2Parser>(tools);
}

std::string inkling2_reply_grammar(const std::vector<ojson>& tools, const ReplyRules& rules) {
  const bool calls = !tools.empty() && rules.tools != ReplyRules::kNone;
  ojson answer;
  if (calls) {
    ojson call_tags = ojson::array();
    for (const auto& t : tools) {
      const ojson* fn = function_of(t);
      const std::string fname =
          fn && fn->contains("name") && (*fn)["name"].is_string() ? (*fn)["name"].get<std::string>() : "";
      // A tool no reply could call and be read back is not among the calls, as one that is no function is not.
      if (!fn || holds_reply_marker(fname) || (rules.tools == ReplyRules::kForced && fname != rules.forced)) continue;
      call_tags.push_back(call_section(*fn));
    }
    if (call_tags.empty()) return "";
    if (rules.tools == ReplyRules::kForced) {
      answer = call_tags[0];
    } else if (rules.json) {
      ojson calls_only = {{"type", "tags_with_separator"},
                          {"tags", std::move(call_tags)},
                          {"separator", ""},
                          {"at_least_one", true},
                          {"stop_after_first", !rules.parallel}};
      if (rules.tools == ReplyRules::kAuto)
        answer = {{"type", "or"}, {"elements", ojson::array({json_text_section(rules.schema), std::move(calls_only)})}};
      else
        answer = std::move(calls_only);
    } else if (rules.tools == ReplyRules::kRequired) {
      answer = {{"type", "tags_with_separator"},
                {"tags", std::move(call_tags)},
                {"separator", ""},
                {"at_least_one", true},
                {"stop_after_first", !rules.parallel}};
    } else {
      ojson calls_seq = {{"type", "tags_with_separator"},
                         {"tags", std::move(call_tags)},
                         {"separator", ""},
                         {"at_least_one", true},
                         {"stop_after_first", !rules.parallel}};
      ojson text_then_calls = {
          {"type", "sequence"},
          {"elements", ojson::array({ojson{{"type", "optional"}, {"content", text_section()}}, std::move(calls_seq)})}};
      answer = {{"type", "or"}, {"elements", ojson::array({text_section(), std::move(text_then_calls)})}};
    }
  } else if (rules.json) {
    answer = json_text_section(rules.schema);
  } else if (!rules.thinking) {
    // Nothing else to hold the reply to: only its thinking is left out, so as many text sections as the model writes.
    answer = {{"type", "tags_with_separator"},
              {"tags", ojson::array({text_section()})},
              {"separator", ""},
              {"at_least_one", true},
              {"stop_after_first", false}};
  } else {
    return "";
  }
  // Thinking switched off: the template has no way to say so (it writes an effort, and the model opens a thinking
  // section at effort none as well), so the reply has no thinking section to begin with.
  ojson format = rules.thinking
                     ? ojson{{"type", "sequence"}, {"elements", ojson::array({thinking_section(), std::move(answer)})}}
                     : std::move(answer);
  return dump_safe({{"type", "structural_tag"}, {"format", std::move(format)}});
}

}  // namespace flyx::models::formats
