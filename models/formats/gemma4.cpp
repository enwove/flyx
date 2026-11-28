// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/gemma4.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>

namespace flyx::models::formats {
namespace {

using interfaces::ojson;
using interfaces::OutputEvent;

constexpr std::string_view kChannelOpen = "<|channel>", kChannelClose = "<channel|>", kCallOpen = "<|tool_call>",
                           kCallClose = "<tool_call|>", kQuote = "<|\"|>", kCallPrefix = "call:";
// The stop tokens, should their text reach the parser (they end the reply; nothing of them is the reply's)
constexpr std::array kStops = {std::string_view("<turn|>"), std::string_view("<|tool_response>")};

bool has(std::string_view s, std::string_view what) { return s.find(what) != std::string_view::npos; }

void push(std::vector<OutputEvent>* ev, OutputEvent::Kind k, std::string_view s) {
  if (s.empty()) return;
  if (!ev->empty() && ev->back().kind == k) ev->back().text.append(s);
  else ev->push_back({k, std::string(s), {}});
}

// How much of s's end could begin one of the marks: held back until the next text says.
size_t partial_mark(std::string_view s) {
  size_t best = 0;
  for (std::string_view t : {kChannelOpen, kChannelClose, kCallOpen, kCallClose, kStops[0], kStops[1]})
    for (size_t l = std::min(s.size(), t.size() - 1); l > best; --l)
      if (s.substr(s.size() - l) == t.substr(0, l)) {
        best = l;
        break;
      }
  return best;
}

// Gemma's values (the reader of a call's arguments).
struct ValueReader {
  std::string_view s;
  size_t i = 0;
  bool ok = true;

  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  }
  bool eat(std::string_view t) {
    if (s.substr(i, t.size()) != t) return false;
    i += t.size();
    return true;
  }
  // A string: between <|"|> marks as it is, or JSON's "..." (escapes read), or '...' as it is.
  bool string(std::string* out) {
    if (eat(kQuote)) {
      const size_t end = s.find(kQuote, i);
      if (end == std::string_view::npos) return false;
      out->assign(s.substr(i, end - i));
      i = end + kQuote.size();
      return true;
    }
    if (i < s.size() && s[i] == '"') {
      size_t j = i + 1;
      while (j < s.size() && s[j] != '"') j += s[j] == '\\' ? 2 : 1;
      if (j >= s.size()) return false;
      const ojson v = interfaces::parse_json(s.substr(i, j + 1 - i));
      if (!v.is_string()) return false;
      *out = v.get<std::string>();
      i = j + 1;
      return true;
    }
    if (i < s.size() && s[i] == '\'') {
      const size_t end = s.find('\'', i + 1);
      if (end == std::string_view::npos) return false;
      out->assign(s.substr(i + 1, end - i - 1));
      i = end + 1;
      return true;
    }
    return false;
  }
  // A key: a string, or a bare run up to the ':'.
  bool key(std::string* out) {
    ws();
    if (string(out)) return true;
    const size_t start = i;
    while (i < s.size() && s[i] != ':' && s[i] != ',' && s[i] != '}' && !std::isspace(static_cast<unsigned char>(s[i])))
      ++i;
    if (i == start) return false;
    out->assign(s.substr(start, i - start));
    return true;
  }
  ojson value() {
    ws();
    if (i >= s.size()) return ok = false, ojson();
    std::string str;
    if (s[i] == '{') {
      ++i;
      ojson obj = ojson::object();
      ws();
      if (eat("}")) return obj;
      for (;;) {
        std::string k;
        if (!key(&k)) return ok = false, ojson();
        ws();
        if (!eat(":")) return ok = false, ojson();
        ojson v = value();
        if (!ok) return ojson();
        obj[k] = std::move(v);
        ws();
        if (eat("}")) return obj;
        if (!eat(",")) return ok = false, ojson();
      }
    }
    if (s[i] == '[') {
      ++i;
      ojson arr = ojson::array();
      ws();
      if (eat("]")) return arr;
      for (;;) {
        ojson v = value();
        if (!ok) return ojson();
        arr.push_back(std::move(v));
        ws();
        if (eat("]")) return arr;
        if (!eat(",")) return ok = false, ojson();
      }
    }
    if (string(&str)) return ojson(str);
    if (eat("true")) return ojson(true);
    if (eat("false")) return ojson(false);
    if (eat("null")) return ojson(nullptr);
    // a number, as JSON writes one
    const size_t start = i;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+' ||
                            s[i] == '.' || s[i] == 'e' || s[i] == 'E'))
      ++i;
    const ojson n = i > start ? interfaces::parse_json(s.substr(start, i - start)) : ojson();
    if (!n.is_number()) return ok = false, ojson();
    return n;
  }
};

class Gemma4Parser final : public interfaces::OutputParser {
 public:
  Gemma4Parser(bool in_reasoning, const std::vector<ojson>* tools)
      : tools_(tools && !tools->empty()), mode_(in_reasoning ? kThought : kText) {}

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
  enum Mode { kText, kLabel, kThought, kCall };

  // Text and reasoning go out as they come, but newlines are held back: a thought's last ones before <channel|> and
  // the answer's first ones are the format's, not the reply's.
  void out(std::vector<OutputEvent>* ev, std::string_view s) {
    const OutputEvent::Kind k = mode_ == kThought ? OutputEvent::kReasoning : OutputEvent::kContent;
    for (char c : s) {
      if (c == '\n') {
        nl_ += c;
        continue;
      }
      if (k == OutputEvent::kContent && !text_started_) nl_.clear();  // the answer's leading newlines
      push(ev, k, nl_);
      nl_.clear();
      push(ev, k, std::string_view(&c, 1));
      if (k == OutputEvent::kContent) text_started_ = true;
    }
  }

  void end_call(std::vector<OutputEvent>* ev, bool closed) {
    std::string_view body = call_;
    const bool named = body.substr(0, kCallPrefix.size()) == kCallPrefix;
    const size_t brace = body.find('{');
    ojson args;
    std::string name;
    if (closed && named && brace != std::string_view::npos) {
      name = std::string(body.substr(kCallPrefix.size(), brace - kCallPrefix.size()));
      args = parse_gemma4_value(body.substr(brace));
    }
    if (!name.empty() && args.is_object()) {
      interfaces::ToolCall start;
      start.name = name;
      ev->push_back({OutputEvent::kToolCallStart, {}, std::move(start)});
      ev->push_back({OutputEvent::kToolCallDelta, args.dump(-1, ' ', false, ojson::error_handler_t::replace), {}});
      interfaces::ToolCall tc;
      tc.name = name;
      tc.arguments = args;
      ev->push_back({OutputEvent::kToolCall, {}, std::move(tc)});
      ++n_calls_;
    } else {  // not a call: its text as written
      mode_ = kText;
      out(ev, std::string(kCallOpen) + call_ + (closed ? std::string(kCallClose) : std::string()));
    }
    call_.clear();
  }

  void run(std::vector<OutputEvent>* ev, bool final) {
    for (;;) {
      if (mode_ == kLabel) {  // a channel's name up to its newline ("thought"), or none before a mark
        const size_t nl = buf_.find('\n'), lt = buf_.find('<');
        if (nl == std::string::npos && lt == std::string::npos && !final) return;
        const size_t end = std::min(nl, lt);
        buf_.erase(0, end == std::string::npos ? buf_.size() : end + (end == nl ? 1 : 0));
        mode_ = kThought;
        continue;
      }
      if (mode_ == kCall) {
        const size_t at = buf_.find(kCallClose);
        if (at == std::string::npos) {
          if (!final) return;
          call_ += buf_;
          buf_.clear();
          end_call(ev, false);
          return;
        }
        call_.append(buf_, 0, at);
        buf_.erase(0, at + kCallClose.size());
        mode_ = kText;
        end_call(ev, true);
        continue;
      }
      // text or a thought: the first mark
      size_t at = std::string::npos;
      std::string_view mark;
      for (std::string_view t : {kChannelOpen, kChannelClose, kCallOpen, kStops[0], kStops[1]})
        if (const size_t p = buf_.find(t); p < at && (t != kCallOpen || tools_)) at = p, mark = t;
      if (at == std::string::npos) {
        const size_t keep = final ? 0 : partial_mark(buf_);
        out(ev, std::string_view(buf_).substr(0, buf_.size() - keep));
        buf_.erase(0, buf_.size() - keep);
        if (final && mode_ == kThought && !text_started_) nl_.clear();
        return;
      }
      out(ev, std::string_view(buf_).substr(0, at));
      buf_.erase(0, at + mark.size());
      if (mark == kChannelOpen) {  // a thought (in one already: a doubled mark, dropped)
        if (mode_ == kText) {
          nl_.clear();
          mode_ = kLabel;
        }
      } else if (mark == kChannelClose) {  // the thought's end (in text: a stray mark, dropped)
        if (mode_ == kThought) nl_.clear();
        mode_ = kText;
      } else if (mark == kCallOpen) {  // a call (from an unclosed thought too)
        nl_.clear();
        mode_ = kCall;
      }  // a stop token's text: dropped
    }
  }

  bool tools_;
  Mode mode_;
  std::string buf_, nl_, call_;
  bool text_started_ = false;
  int n_calls_ = 0;
};

}  // namespace

const TagSyntax kGemma4Syntax = [] {
  static const CallSyntax calls = {"<|tool_call>call:", "", "<tool_call|>", "<|tool_call>", "gemma"};
  TagSyntax t;
  t.thinking = true;
  t.written = &calls;
  t.think_open = "<|channel>";
  t.think_close = "<channel|>";
  t.think_after = "";
  return t;
}();

bool is_gemma4_template(std::string_view source) {
  return has(source, "<|tool_call>call:") && has(source, "<|turn>") && has(source, kChannelOpen);
}

bool gemma4_opens_reasoning(std::string_view gen) {
  const size_t open = gen.rfind(kChannelOpen);
  return open != std::string_view::npos && gen.find(kChannelClose, open) == std::string_view::npos;
}

std::unique_ptr<interfaces::OutputParser> make_gemma4_parser(bool in_reasoning, const std::vector<ojson>* tools) {
  return std::make_unique<Gemma4Parser>(in_reasoning, tools);
}

ojson parse_gemma4_value(std::string_view s) {
  ValueReader r{s};
  ojson v = r.value();
  r.ws();
  return r.ok && r.i == s.size() ? v : ojson();
}

}  // namespace flyx::models::formats
