// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/tags.h"

#include <algorithm>

namespace flyx::models::formats {
namespace {

constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kCallStart = "<tool_call>";
constexpr std::string_view kCallEnd = "</tool_call>";

// Length of the longest tail of s that is a proper prefix of tag.
size_t partial_tail(std::string_view s, std::string_view tag) {
  for (size_t l = std::min(s.size(), tag.size() - 1); l > 0; --l)
    if (s.substr(s.size() - l) == tag.substr(0, l)) return l;
  return 0;
}

}  // namespace

void TaggedParser::emit(std::vector<OutputEvent>* ev, OutputEvent::Kind k, const std::string& s) {
  // Trailing newlines are held: templates drop them before </think> and <tool_call>.
  size_t e = s.size();
  while (e > 0 && s[e - 1] == '\n') --e;
  if (e > 0) {
    std::string out = nl_ + s.substr(0, e);
    nl_.clear();
    if (!ev->empty() && ev->back().kind == k) ev->back().text += out;
    else ev->push_back({k, out, {}});
  }
  nl_ += s.substr(e);
}

void TaggedParser::run(std::vector<OutputEvent>* ev, bool final) {
  while (!buf_.empty() || (final && mode_ == kCall)) {
    if (mode_ == kThink) {
      if (!think_started_) {  // templates strip newlines right after <think>
        size_t b = buf_.find_first_not_of('\n');
        buf_.erase(0, b == std::string::npos ? buf_.size() : b);
        if (buf_.empty()) return;
        think_started_ = true;
      }
      size_t k = buf_.find(kThinkEnd);
      if (k != std::string::npos) {
        emit(ev, OutputEvent::kReasoning, buf_.substr(0, k));
        nl_.clear();
        buf_.erase(0, k + kThinkEnd.size());
        mode_ = kText;
        text_started_ = false;
        continue;
      }
      size_t keep = final ? 0 : partial_tail(buf_, kThinkEnd);
      emit(ev, OutputEvent::kReasoning, buf_.substr(0, buf_.size() - keep));
      buf_.erase(0, buf_.size() - keep);
      return;
    }
    if (mode_ == kText) {
      if (!text_started_) {  // newlines after </think> or between tool calls
        size_t b = buf_.find_first_not_of('\n');
        buf_.erase(0, b == std::string::npos ? buf_.size() : b);
        if (buf_.empty()) return;
        if (syntax_.thinking && n_calls_ == 0 && !think_started_) {
          if (buf_.compare(0, kThinkStart.size(), kThinkStart) == 0) {  // the model opened a think block itself
            buf_.erase(0, kThinkStart.size());
            mode_ = kThink;
            continue;
          }
          if (!final && kThinkStart.substr(0, buf_.size()) == buf_) return;  // could still become <think>
        }
        text_started_ = true;
      }
      if (!syntax_.calls) {
        emit(ev, OutputEvent::kContent, buf_);
        buf_.clear();
        return;
      }
      size_t k = buf_.find(kCallStart);
      if (k != std::string::npos) {
        emit(ev, OutputEvent::kContent, buf_.substr(0, k));
        call_nl_ = std::move(nl_);
        nl_.clear();
        buf_.erase(0, k + kCallStart.size());
        call_stream_ = open_call_stream(syntax_.calls, tools_);
        call_raw_.clear();
        mode_ = kCall;
        continue;
      }
      size_t keep = final ? 0 : partial_tail(buf_, kCallStart);
      emit(ev, OutputEvent::kContent, buf_.substr(0, buf_.size() - keep));
      buf_.erase(0, buf_.size() - keep);
      return;
    }
    // kCall: stream the call body into call_stream_ while holding back any partial </tool_call> tail.
    const size_t k = buf_.find(kCallEnd);
    if (k != std::string::npos || final) {
      const std::string piece = buf_.substr(0, k == std::string::npos ? buf_.size() : k);
      buf_.erase(0, k == std::string::npos ? buf_.size() : k + kCallEnd.size());
      if (!call_stream_->started()) call_raw_ += piece;
      call_stream_->feed(piece, ev);
      if (call_stream_->finish(ev)) {
        ++n_calls_;
        call_nl_.clear();
        text_started_ = false;
      } else {  // not a well-formed call: nothing was streamed, so show it as text with its surrounding newlines
        const std::string raw =
            std::string(kCallStart) + call_raw_ + (k == std::string::npos ? "" : std::string(kCallEnd));
        ev->push_back({OutputEvent::kContent, call_nl_ + raw, {}});
        call_nl_.clear();
        text_started_ = true;
      }
      call_stream_.reset();
      call_raw_.clear();
      mode_ = kText;
      continue;
    }
    const size_t keep = partial_tail(buf_, kCallEnd);
    const size_t safe = buf_.size() - keep;
    if (safe > 0) {
      const std::string_view piece(buf_.data(), safe);
      if (!call_stream_->started()) call_raw_.append(piece);
      call_stream_->feed(piece, ev);
      if (call_stream_->started()) call_raw_.clear();
      buf_.erase(0, safe);
    }
    return;
  }
}

std::vector<OutputEvent> TaggedParser::feed(std::string_view text) {
  std::vector<OutputEvent> ev;
  buf_.append(text);
  run(&ev, false);
  return ev;
}

std::vector<OutputEvent> TaggedParser::finish() {
  std::vector<OutputEvent> ev;
  run(&ev, true);
  nl_.clear();  // trailing newlines of the answer: templates trim them
  return ev;
}

}  // namespace flyx::models::formats
