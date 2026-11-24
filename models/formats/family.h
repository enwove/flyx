// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A chat format with a renderer written for one exact template:
// both directions of the format in one place, the renderer that
// writes prompts the way the template does and how replies are read back.
// formats.cpp lists them; a template that is not byte for byte one of
// theirs is rendered by the Jinja engine.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "core/interfaces/chat.h"
#include "models/formats/tags.h"

namespace flyx::models::formats {

using interfaces::ChatMessage;
using interfaces::ChatOptions;
using interfaces::RenderedChat;
using interfaces::TextSpan;

// Writes the messages as the template would: the text, and which bytes the
// template wrote itself (a client's string as it passes through the
// template is not).  false with *err (the template's message) where the
// template raises.
using NativeRender = bool (*)(std::span<const ChatMessage> msgs, const std::vector<ojson>& tools,
                              const ChatOptions& opt, bool generation_prompt, RenderedChat* out, std::string* err);

struct Family {
  const char* name;
  // The template the renderer is written for: its length and FNV-1a 64 hash.
  size_t template_size;
  uint64_t template_hash;
  NativeRender render;
  TagSyntax replies;
};

uint64_t fnv1a64(std::string_view s);

// Appends text, keeping the spans the template wrote (literal) apart from
// the client's strings (input), the way the Jinja engine marks them.
class Writer {
 public:
  explicit Writer(RenderedChat* out) : out_(out) {
    out_->text.clear();
    out_->literal.clear();
    out_->images.clear();
  }
  void lit(std::string_view s) {
    if (s.empty()) return;
    const size_t b = out_->text.size();
    out_->text += s;
    if (!out_->literal.empty() && out_->literal.back().end == b) out_->literal.back().end = out_->text.size();
    else out_->literal.push_back({b, out_->text.size()});
  }
  void in(std::string_view s) { out_->text += s; }
  void text(std::string_view s, bool client) { client ? in(s) : lit(s); }
  // Emits a template image placeholder as a literal span and anchors `img` at its byte span.
  void image(std::shared_ptr<const interfaces::ImageInput> img, std::string_view placeholder) {
    const size_t b = out_->text.size();
    lit(placeholder);
    out_->images.push_back({b, out_->text.size(), std::move(img)});
  }

 private:
  RenderedChat* out_;
};

}  // namespace flyx::models::formats
