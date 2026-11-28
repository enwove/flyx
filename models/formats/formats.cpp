// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The chat side of a model file (core/interfaces/chat.h): its template, and the
// chat format that template belongs to. The
// template is the single source: a file converted before this existed, or a
// template given at load, is recognised the same way.
//   * A template that is byte for byte one a family's renderer is written
//     for: that family (its renderer and its replies).
//   * Otherwise the replies are recognised from the template's own text, the
//     way llama.cpp recognises chat formats (docs/prior-art.md), and the
//     Jinja engine renders.
#include <string>

#include "core/interfaces/chat.h"
#include "core/modelfile/modelfile.h"
#include "models/formats/family.h"
#include "models/formats/gemma4.h"
#include "models/formats/harmony.h"
#include "models/formats/inkling2.h"
#include "models/formats/qwen35.h"
#include "models/formats/tags.h"

namespace flyx::models::formats {
namespace {

// The formats with a renderer of their own.
const Family* const kFamilies[] = {&kQwen35};

bool has(const std::string& s, const char* what) { return s.find(what) != std::string::npos; }

// Text as generated, nothing held back or renamed.
class PlainParser : public interfaces::OutputParser {
 public:
  std::vector<OutputEvent> feed(std::string_view text) override {
    if (text.empty()) return {};
    return {{OutputEvent::kContent, std::string(text), {}}};
  }
  std::vector<OutputEvent> finish() override { return {}; }
  int tool_calls() const override { return 0; }
};

class Chat : public interfaces::Chat {
 public:
  Chat(std::string source, ojson tokens) : source_(std::move(source)), tokens_(std::move(tokens)) {
    const uint64_t hash = fnv1a64(source_);
    for (const Family* f : kFamilies)
      if (f->template_size == source_.size() && f->template_hash == hash) family_ = f;
    if (family_) {
      format_ = family_->name;
      syntax_ = family_->replies;
      return;
    }
    if (is_inkling2_template(source_)) {
      format_ = "inkling2";
      inkling2_ = true;
      inkling2_tools_ = has(source_, "<|content_invoke_tool_json|>");
      return;
    }
    if (is_harmony_template(source_)) {  // gpt-oss: tools are messages to functions.NAME
      format_ = "harmony";
      harmony_ = true;
      return;
    }
    if (is_gemma4_template(source_)) {  // Gemma 4: <|channel>thought..<channel|>, <|tool_call>call:NAME{..}<tool_call|>
      format_ = "gemma4";
      gemma4_ = true;
      return;
    }
    // Tool calls written as XML: Qwen3.5, Qwen3-Coder (llama.cpp's test, chat.cpp "Qwen3-Coder").
    if (has(source_, "<tool_call>") && has(source_, "<function=") && has(source_, "<parameter=")) {
      format_ = "qwen3_coder";
      syntax_.calls = read_xml_call;
      syntax_.written = &kXmlCallSyntax;
    } else if (has(source_, "<tool_call>") && has(source_, "</tool_call>")) {  // JSON calls: Qwen3, Qwen2.5
      format_ = "hermes";
      syntax_.calls = read_json_call;
      syntax_.written = &kJsonCallSyntax;
    }
    syntax_.thinking = has(source_, "<think>") && has(source_, "</think>");
  }

  const std::string& template_source() const override { return source_; }
  const ojson& template_tokens() const override { return tokens_; }
  const char* format() const override { return format_; }
  bool reads_tool_calls() const override {
    return harmony_ || gemma4_ || (inkling2_ ? inkling2_tools_ : (syntax_.calls != nullptr));
  }

  bool opens_reasoning(std::string_view gen) const override {
    if (gemma4_) return gemma4_opens_reasoning(gen);
    if (inkling2_ || harmony_ || !syntax_.thinking) return false;  // harmony: its reply's header opens a channel
    size_t open = gen.rfind("<think>");
    return open != std::string_view::npos && gen.find("</think>", open) == std::string_view::npos;
  }

  std::unique_ptr<interfaces::OutputParser> parser(bool in_reasoning, const std::vector<ojson>* tools) const override {
    if (inkling2_) return make_inkling2_parser(inkling2_tools_ ? tools : nullptr);
    if (harmony_) return make_harmony_parser(tools);
    if (gemma4_) return make_gemma4_parser(in_reasoning, tools);
    TagSyntax s = syntax_;
    if (!tools || tools->empty()) s.calls = nullptr;
    if (!s.thinking && !s.calls) return std::make_unique<PlainParser>();
    return std::make_unique<TaggedParser>(s, in_reasoning, tools);
  }

  std::string reply_grammar(const std::vector<ojson>& tools, const interfaces::ReplyRules& rules,
                            bool in_reasoning) const override {
    if (inkling2_) return inkling2_reply_grammar(inkling2_tools_ ? tools : std::vector<ojson>{}, rules);
    if (harmony_) return {};  // no reply grammar for harmony's channels yet: replies unconstrained
    if (gemma4_) return formats::reply_grammar(kGemma4Syntax, tools, rules, in_reasoning);
    return formats::reply_grammar(syntax_, tools, rules, in_reasoning);
  }

  std::vector<std::string> reply_tokens() const override {
    return inkling2_ ? inkling2_reply_tokens() : harmony_ ? harmony_reply_tokens() : std::vector<std::string>();
  }

  bool thinking_markers(std::string* open, std::string* close) const override {
    if (inkling2_) return *open = "<|content_thinking|>", *close = "<|end_message|>", true;
    if (!syntax_.thinking) return false;
    return *open = "<think>", *close = "</think>", true;
  }

  // Inkling-2's template has no enable_thinking: it writes a thinking effort, 0.9 unless told (reasoning_effort),
  // and "none" is its word for no thinking.
  ojson option_variables(const interfaces::ChatOptions& opt) const override {
    if (inkling2_ && !opt.enable_thinking) return {{"reasoning_effort", "none"}};
    return ojson::object();
  }

  bool native() const override { return family_ != nullptr; }
  bool render(std::span<const ChatMessage> msgs, const std::vector<ojson>& tools, const ChatOptions& opt,
              bool generation_prompt, RenderedChat* out, std::string* err) const override {
    if (!family_) return interfaces::Chat::render(msgs, tools, opt, generation_prompt, out, err);
    return family_->render(msgs, tools, opt, generation_prompt, out, err);
  }

 private:
  std::string source_;
  ojson tokens_;
  const Family* family_ = nullptr;
  const char* format_ = "plain";
  bool inkling2_ = false;
  bool inkling2_tools_ = false;
  bool harmony_ = false;
  bool gemma4_ = false;  // its marks are text tokens (llama.cpp's USER_DEFINED): no reply_tokens
  TagSyntax syntax_;
};

}  // namespace

uint64_t fnv1a64(std::string_view s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  return h;
}

}  // namespace flyx::models::formats

namespace flyx::interfaces {

std::unique_ptr<Chat> open_chat(const ModelFile& file, const std::string& template_override, std::string* err) {
  std::string source = template_override;
  if (source.empty()) source = std::string(file.blob("chat_template.jinja"));
  if (source.empty()) {
    *err = "the model file has no chat template";
    return nullptr;
  }
  // The special tokens a template may print by name ({{ bos_token }}); files
  // converted before these were recorded have none.
  ojson tokens = ojson::object();
  const auto& h = file.header();
  if (h.contains("chat") && h["chat"].is_object())
    for (const char* k : {"bos_token", "eos_token"})
      if (h["chat"].contains(k) && h["chat"][k].is_string()) tokens[k] = h["chat"][k].get<std::string>();
  return std::make_unique<models::formats::Chat>(std::move(source), std::move(tokens));
}

}  // namespace flyx::interfaces
