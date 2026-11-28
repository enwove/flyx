// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/harmony.h"

#include <algorithm>
#include <array>

namespace flyx::models::formats {
namespace {

using interfaces::ojson;
using interfaces::OutputEvent;

constexpr std::string_view kStart = "<|start|>", kChannel = "<|channel|>", kMessage = "<|message|>",
                           kEnd = "<|end|>", kReturn = "<|return|>", kCall = "<|call|>", kConstrain = "<|constrain|>";
constexpr std::array kEnds = {kEnd, kReturn, kCall};
constexpr std::string_view kFunctions = "functions.";

bool has(std::string_view s, std::string_view what) { return s.find(what) != std::string_view::npos; }

void push(std::vector<OutputEvent>* ev, OutputEvent::Kind k, std::string_view s) {
  if (s.empty()) return;
  if (!ev->empty() && ev->back().kind == k) ev->back().text.append(s);
  else ev->push_back({k, std::string(s), {}});
}

// The word of `header` after `key`, up to a space or a control token.
std::string word_after(std::string_view header, std::string_view key) {
  const size_t at = header.find(key);
  if (at == std::string_view::npos) return {};
  const size_t from = at + key.size();
  const size_t to = header.find_first_of(" <", from);
  return std::string(header.substr(from, to == std::string_view::npos ? std::string_view::npos : to - from));
}

// How much of s's end could be the start of one of the end tokens: held back until the next text says.
size_t partial_end(std::string_view s) {
  size_t best = 0;
  for (std::string_view t : kEnds)
    for (size_t l = std::min(s.size(), t.size() - 1); l > best; --l)
      if (s.substr(s.size() - l) == t.substr(0, l)) {
        best = l;
        break;
      }
  return best;
}

class HarmonyParser final : public interfaces::OutputParser {
 public:
  explicit HarmonyParser(const std::vector<ojson>* tools) : tools_(tools) {}

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
  enum Mode { kHeader, kReasoning, kText, kTool };

  void body(std::vector<OutputEvent>* ev, std::string_view s) {
    if (mode_ == kTool) call_.append(s);
    else push(ev, mode_ == kReasoning ? OutputEvent::kReasoning : OutputEvent::kContent, s);
  }

  // A message to functions.NAME ended: the call, its arguments the body's JSON object (else the body as text).
  void end_call(std::vector<OutputEvent>* ev) {
    const ojson args = interfaces::parse_json(call_);
    if (args.is_object()) {
      interfaces::ToolCall start;
      start.name = tool_;
      ev->push_back({OutputEvent::kToolCallStart, {}, std::move(start)});
      ev->push_back({OutputEvent::kToolCallDelta, args.dump(-1, ' ', false, ojson::error_handler_t::replace), {}});
      interfaces::ToolCall tc;
      tc.name = tool_;
      tc.arguments = args;
      ev->push_back({OutputEvent::kToolCall, {}, std::move(tc)});
      ++n_calls_;
    } else {
      push(ev, OutputEvent::kContent, call_);
    }
    call_.clear();
  }

  void run(std::vector<OutputEvent>* ev, bool final) {
    for (;;) {
      if (mode_ == kHeader) {
        const size_t m = buf_.find(kMessage);
        if (m == std::string::npos) {
          if (final) {  // never a message: a reply not in harmony, as text (its control tokens left out)
            std::string text = header_ + buf_;
            for (std::string_view t : {kStart, kChannel, kConstrain, kEnd, kReturn, kCall})
              for (size_t at; (at = text.find(t)) != std::string::npos;) text.erase(at, t.size());
            push(ev, OutputEvent::kContent, text);
            header_.clear();
            buf_.clear();
          }
          return;
        }
        header_.append(buf_, 0, m);
        buf_.erase(0, m + kMessage.size());
        const std::string channel = word_after(header_, kChannel), to = word_after(header_, "to=");
        if (to.rfind(kFunctions, 0) == 0 && tools_ && !tools_->empty()) {
          mode_ = kTool;
          tool_ = to.substr(kFunctions.size());
        } else {
          mode_ = channel == "analysis" ? kReasoning : kText;
        }
        header_.clear();
        continue;
      }
      size_t at = std::string::npos, len = 0;
      for (std::string_view t : kEnds)
        if (const size_t p = buf_.find(t); p < at) at = p, len = t.size();
      if (at != std::string::npos) {
        body(ev, std::string_view(buf_).substr(0, at));
        buf_.erase(0, at + len);
        if (mode_ == kTool) end_call(ev);
        mode_ = kHeader;
        continue;
      }
      const size_t keep = final ? 0 : partial_end(buf_);
      body(ev, std::string_view(buf_).substr(0, buf_.size() - keep));
      buf_.erase(0, buf_.size() - keep);
      if (final && mode_ == kTool) end_call(ev);
      return;
    }
  }

  const std::vector<ojson>* tools_;
  std::string buf_, header_, call_, tool_;
  Mode mode_ = kHeader;  // the generation prompt ends "<|start|>assistant": the reply's first message's header
  int n_calls_ = 0;
};

}  // namespace

bool is_harmony_template(std::string_view source) {
  return has(source, kChannel) && has(source, kMessage) && has(source, kStart) && has(source, kEnd);
}

std::unique_ptr<interfaces::OutputParser> make_harmony_parser(const std::vector<ojson>* tools) {
  return std::make_unique<HarmonyParser>(tools);
}

std::vector<std::string> harmony_reply_tokens() {
  return {std::string(kStart), std::string(kChannel), std::string(kMessage), std::string(kEnd),
          std::string(kConstrain)};
}

}  // namespace flyx::models::formats
