// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/exec/request.h"

#include "core/os/os.h"

namespace flyx {

bool render_request(const ChatTemplate& chat, const Tokenizer& tok, const Request& rq, RenderedChat* out,
                    std::string* err, bool branch_points) {
  *out = {};
  if (rq.chat) {
    if (chat.render(rq.messages, rq.tools, rq.chat_options, out, err, branch_points, rq.cache_points)) return true;
    if (!rq.join_leading_system || !rq.cache_points.empty() || rq.messages.size() < 2 ||
        rq.messages[0].role != "system" || rq.messages[1].role != "system") return false;
    // Some templates accept only one system message. Preserve the order of its instructions and leave later turns alone.
    std::vector<ChatMessage> messages{rq.messages.front()};
    auto next = rq.messages.begin() + 1;
    for (; next != rq.messages.end() && next->role == "system"; ++next) {
      messages.front().content += "\n\n";
      messages.front().content += next->content;
    }
    messages.insert(messages.end(), next, rq.messages.end());
    std::string retry_error;
    if (!chat.render(messages, rq.tools, rq.chat_options, out, &retry_error, branch_points)) return false;
    err->clear();
    return true;
  }
  if (!rq.tokens.empty()) return true;
  if (rq.text.empty()) return *err = "empty prompt", false;
  // The prefix as text, as a template writes its <s>: a session's history (decoded text) then continues it.  Each
  // token in a span of its own, so that a longer special token cannot take it together with the caller's text.
  for (int32_t id : tok.prefix()) {
    const size_t at = out->text.size();
    out->text += tok.token_bytes(id);
    out->literal.push_back({at, out->text.size()});
  }
  const size_t at = out->text.size();
  out->text += rq.text;
  out->literal.push_back({at, out->text.size()});
  out->conversation_end = out->text.size();
  return true;
}

bool prepare_request(Engine& e, const Session& s, const Request& rq, const RenderedChat& rc, PreparedRequest* out,
                     std::string* err) {
  *out = {};
  out->gen = rq.gen;
  GenerateRequest& gen = out->gen;
  if (!rq.chat && !rq.tokens.empty()) gen.prompt = rq.tokens;
  else gen.prompt = s.tokenize(rc.text, &rc.literal, &out->tokenized);
  gen.checkpoints = token_positions(e.tokenizer(), gen.prompt, rc.boundaries);
  gen.cache_prefixes = token_positions(e.tokenizer(), gen.prompt, rc.cache_prefixes);
  if (rc.system_end) gen.system_end = token_positions(e.tokenizer(), gen.prompt, {rc.system_end}).front();
  // A question: keep the checkpoint where it ends through the tool calls that follow.
  if (rq.chat && !rq.messages.empty() && rq.messages.back().role == "user" && !gen.checkpoints.empty())
    gen.keep_checkpoint = gen.checkpoints.back();
  if (!rq.chat || gen.prefill_only) return true;

  // The grammar the reply must match: tool calls in the format's syntax with
  // arguments that fit their schemas, tool_choice, a JSON answer (core/grammar).
  const std::string tag = e.chat().reply_grammar(rq.tools, rq.rules, rc);
  if (!tag.empty()) {
    const double t0 = os::now_ms();
    std::string why;
    out->grammar = e.grammar().matcher(tag, &why, &out->grammar_cached);
    out->grammar_ms = os::now_ms() - t0;
    if (!out->grammar) {
      const bool must = rq.rules.tools == ReplyRules::kRequired || rq.rules.tools == ReplyRules::kForced ||
                        rq.rules.json;
      if (must) {
        out->invalid = e.grammar().ready();  // the compiler is there: what it was given is at fault
        return *err = why, false;
      }
      out->unconstrained = why;
    }
  }
  gen.grammar = out->grammar.get();
  if (rq.split_output)
    out->parser = e.chat().parser(rc, rq.rules.tools == ReplyRules::kNone ? nullptr : &rq.tools);
  return true;
}

void warm_request(Engine& e, const Request& rq, const RenderedChat& rc) {
  if (!rq.chat) return;
  const std::string tag = e.chat().reply_grammar(rq.tools, rq.rules, rc);
  std::string why;
  if (!tag.empty()) (void)e.grammar().matcher(tag, &why);
}

}  // namespace flyx
