// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A request, from where a wire format left it to where a session can run it.
//
// Every entry point (the HTTP server through the Scheduler, the C API, the
// command line) fills in a Request and takes the same two steps, so what a
// prompt's tokens, its checkpoints, the question's checkpoint and the reply's
// grammar are is decided in one place:
//
//   render_request   the prompt text: the model's chat template, or the raw
//                    text.  Needs no session and no device; any thread.
//   prepare_request  for the session that will run it: the tokens (the
//                    session's history keeps its own), where checkpoints go,
//                    the grammar the reply must match and the parser that
//                    splits it into thinking, answer and tool calls.
//
// warm_request does ahead, on any thread, the slow part of prepare_request
// that needs no session (compiling the reply's grammar): the server calls it
// as a request arrives, so the thread that runs the device does not stop the
// other requests' rounds for it.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/generation.h"
#include "core/grammar/grammar.h"
#include "core/render/chat.h"
#include "core/session/session.h"
#include "core/tokenizer/tokenizer.h"

namespace flyx {

// What a client asks for, whatever API it came through.  One prompt: chat
// messages, raw text, or tokens.
struct Request {
  bool chat = true;                  // messages + template, else raw text or tokens
  std::vector<ChatMessage> messages;
  bool join_leading_system = false;  // if the template rejects separate leading instruction messages
  std::vector<ojson> tools;
  ChatOptions chat_options;
  std::vector<ChatCachePoint> cache_points;
  ReplyRules rules;                  // chat: tool_choice, parallel calls, a JSON answer (core/grammar)
  bool split_output = true;          // chat: thinking and tool calls as their own events
  std::string text;                  // raw text prompt (chat == false)
  std::vector<int32_t> tokens;       // or the prompt as tokens (chat == false, no text)
  GenerateRequest gen;               // limits, sampling, stops (prepare_request fills in the rest)
  std::string label;                 // for the log, e.g. "openai chat"
  std::string conversation_id;       // opaque client identity for replacement only (empty if unknown)
  std::string response_id;           // optional public response identifier to correlate telemetry with its caller
};

// The prompt of a request: the chat template's rendering, or the raw text
// (all of it the caller's own writing, special tokens included) after the
// text of the tokenizer's prefix (MiniCPM5's <s>, which transformers puts
// before a text: Tokenizer::encode_prompt's tokens).  A prompt given as
// tokens renders to nothing.  branch_points: see ChatTemplate::render.
bool render_request(const ChatTemplate& chat, const Tokenizer& tok, const Request& rq, RenderedChat* out,
                    std::string* err, bool branch_points = true);

// A request ready to generate on one session.
struct PreparedRequest {
  GenerateRequest gen;                      // the request's, with prompt, checkpoints and grammar filled in
  std::unique_ptr<GrammarMatcher> grammar;  // what gen.grammar points to
  std::unique_ptr<OutputParser> parser;     // null: the reply is delivered as the text it is
  std::string tokenized;                    // how the tokenizer reused the session's history (for the log)
  double grammar_ms = 0;                    // getting the grammar took (compiling it, unless it was kept)
  bool grammar_cached = false;              // the engine had it compiled (warm_request, an earlier request)
  // The grammar did not compile and the request can do without it (tool calls
  // the model may or may not make): why.  Its tool calls are not constrained.
  std::string unconstrained;
  // prepare_request failed and the fault is the request's: its tools or its schema make no grammar.
  bool invalid = false;
};

// The tokens of `rc` for session `s`, the prompt positions worth a checkpoint,
// the question's checkpoint (kept through the tool calls that follow it), the
// reply's grammar and its parser.  false with *err when the request needs a
// grammar that does not compile.
bool prepare_request(Engine& e, const Session& s, const Request& rq, const RenderedChat& rc, PreparedRequest* out,
                     std::string* err);

// The reply's grammar compiled into the engine's cache (the vocabulary's
// tables built on first use), for a prepare_request to come.  Any thread; a
// failure here is prepare_request's to report.  (The tokens are left to
// prepare_request: it encodes only what the session's history does not hold.)
void warm_request(Engine& e, const Request& rq, const RenderedChat& rc);

}  // namespace flyx
