// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The C API (include/flyx/flyx.h) over Engine and Session.
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/request.h"
#include "core/session/session.h"
#include "core/render/chat_json.h"

// The engine is shared by its handle and every session made from it, so
// flyx_unload() with sessions still open only drops the handle's
// reference: the model is freed when the last session goes.
struct flyx_engine {
  std::shared_ptr<flyx::Engine> e = std::make_shared<flyx::Engine>();
};
struct flyx_session {
  std::shared_ptr<flyx::Engine> engine;  // declared first: destroyed after the session
  std::unique_ptr<flyx::Session> s;
};

namespace {

void set_err(flyx_error* err, const std::string& msg) {
  if (err) std::snprintf(err->message, sizeof(err->message), "%s", msg.c_str());
}

}  // namespace

extern "C" {

uint32_t flyx_api_version(void) { return FLYX_API_VERSION; }

flyx_engine* flyx_load(const char* path, const flyx_load_options* opts, flyx_error* err) {
  if (!path || !*path) {
    set_err(err, "model path is null or empty");
    return nullptr;
  }
  try {
    auto e = std::make_unique<flyx_engine>();
    flyx::EngineOptions o;
    if (opts) {
      if (opts->max_ctx > 0) o.max_ctx = opts->max_ctx, o.ctx_given = true;
      if (opts->max_ctx == -1) o.ctx_max = o.ctx_given = true;  // --ctx max
      if (opts->backend) o.backend = opts->backend;
      if (opts->backend_dir) o.backend_dir = opts->backend_dir;
      o.speculative = opts->speculative != 0;
      o.verbose = opts->verbose;
      if (opts->drafter) o.drafter = opts->drafter;
      if (opts->width > 0) o.width = opts->width;
      if (opts->prefill_chunk > 0) o.prefill_chunk = opts->prefill_chunk;
      if (opts->kv == FLYX_KV_FP8 || opts->kv == FLYX_KV_16) o.kv = opts->kv;
      if (opts->backend_options) o.backend_options = opts->backend_options;
      if (opts->activations) o.activations = opts->activations;
      if (opts->chat_template) o.chat_template = opts->chat_template;
      for (const char* p = opts->drafter_file; p && *p;) {  // one path a line
        const char* end = std::strchr(p, '\n');
        const size_t n = end ? size_t(end - p) : std::strlen(p);
        if (n > 0) o.drafter_files.emplace_back(p, n);
        p = end ? end + 1 : p + n;
      }
      if (opts->mmproj_file) o.mmproj_file = opts->mmproj_file;
      if (opts->no_vision) o.vision = false;
      if (opts->vision_device) o.vision_device = opts->vision_device;
      if (opts->vision_quant) o.vision_quant = opts->vision_quant;
      if (opts->vision_max_tokens > 0) o.vision_max_tokens = opts->vision_max_tokens;
    }
    std::string msg;
    if (!e->e->load(path, o, &msg)) {
      set_err(err, msg);
      return nullptr;
    }
    return e.release();
  } catch (const std::exception& ex) {
    set_err(err, ex.what());
    return nullptr;
  } catch (...) {
    set_err(err, "internal error");
    return nullptr;
  }
}

void flyx_unload(flyx_engine* e) {
  try {
    delete e;
  } catch (...) {
  }
}

const char* flyx_backend_name(const flyx_engine* e) {
  if (!e || !e->e) return "";
  try {
    return e->e->backend_name().c_str();
  } catch (...) {
    return "";
  }
}

flyx_session* flyx_session_new(flyx_engine* e) {
  if (!e || !e->e) return nullptr;
  try {
    auto s = std::make_unique<flyx_session>(flyx_session{e->e, std::make_unique<flyx::Session>(e->e.get())});
    std::string msg;
    if (!s->s->init(&msg)) return nullptr;
    return s.release();
  } catch (...) {
    return nullptr;
  }
}

void flyx_session_free(flyx_session* s) {
  try {
    delete s;
  } catch (...) {
  }
}

void flyx_cancel(flyx_session* s) {
  if (!s || !s->s) return;
  try {
    s->s->cancel();
  } catch (...) {
  }
}

int flyx_generate(flyx_session* s, const flyx_request* req, flyx_event_fn cb, void* user,
                  flyx_usage* usage, flyx_error* err) {
  if (!s || !s->s || !s->engine || !req) {
    set_err(err, "null session or request");
    return -1;
  }
  try {
    using namespace flyx;
    // The request as every entry point states it (core/exec/request.h).
    Request rq;
    rq.chat = false;
    rq.label = "api";
    rq.gen.max_new_tokens = req->max_new_tokens;
    rq.gen.ignore_eos = req->ignore_eos != 0;
    rq.gen.sampling = {req->temperature, req->top_p, req->top_k, req->seed, req->presence_penalty,
                       req->frequency_penalty};
    if (req->n_stop < 0) {
      set_err(err, "n_stop must be non-negative");
      return -1;
    }
    for (int i = 0; i < req->n_stop; ++i)
      if (req->stop && req->stop[i]) rq.gen.stop.push_back(req->stop[i]);

    std::vector<ojson>& tools = rq.tools;
    ReplyRules& rules = rq.rules;
    std::string msg;
    if (req->tokens) {
      if (req->n_tokens < 0) {
        set_err(err, "n_tokens must be non-negative");
        return -1;
      }
      rq.tokens.assign(req->tokens, req->tokens + req->n_tokens);
    } else if (req->messages || req->messages_json) {
      rq.chat = true;
      rq.split_output = req->split_output != 0;
      rq.chat_options.enable_thinking = req->enable_thinking != 0;
      std::vector<ChatMessage>& msgs = rq.messages;
      if (req->messages_json) {
        ojson j = interfaces::parse_json(req->messages_json);
        if (j.is_discarded() || !messages_from_openai(j, &msgs, &msg)) {
          set_err(err, j.is_discarded() ? "messages_json is not valid JSON" : msg);
          return -1;
        }
      } else {
        if (req->n_messages < 0) {
          set_err(err, "n_messages must be non-negative");
          return -1;
        }
        for (int i = 0; i < req->n_messages; ++i) {
          const auto& m = req->messages[i];
          ChatMessage cm{m.role ? m.role : "", m.content ? m.content : "", std::nullopt, {}, ""};
          if (m.reasoning) cm.reasoning = m.reasoning;
          msgs.push_back(std::move(cm));
        }
      }
      if (req->tools_json) {
        ojson j = interfaces::parse_json(req->tools_json);
        if (j.is_discarded() || !tools_from_openai(j, &tools, &msg)) {
          set_err(err, j.is_discarded() ? "tools_json is not valid JSON" : msg);
          return -1;
        }
      }
      // The reply rules, read as the OpenAI fields they are.
      ojson fields = ojson::object();
      const std::pair<const char*, const char*> rule_fields[] = {{"tool_choice", req->tool_choice_json},
                                                                 {"response_format", req->response_format_json}};
      for (const auto& [name, text] : rule_fields) {
        if (!text) continue;
        fields[name] = interfaces::parse_json(text);
        if (fields[name].is_discarded()) {
          set_err(err, std::string(name) + "_json is not valid JSON");
          return -1;
        }
      }
      if (req->single_tool_call) fields["parallel_tool_calls"] = false;
      if (!rules_from_openai(fields, tools, &rules, &msg)) {
        set_err(err, msg);
        return -1;
      }
      if (rules.tools == ReplyRules::kNone) tools.clear();  // as if none were offered
    } else if (req->text) {
      rq.text = req->text;
    }

    RenderedChat rc;
    PreparedRequest prepared;
    if (!render_request(s->engine->chat(), s->engine->tokenizer(), rq, &rc, &msg) ||
        !prepare_request(*s->engine, *s->s, rq, rc, &prepared, &msg)) {
      set_err(err, msg);
      return -1;
    }
    OutputParser* parser = prepared.parser.get();
    bool stop_cb = false;
    auto send = [&](int kind, const std::string& text, const int32_t* toks, int n) {
      if (!cb || stop_cb) return;
      flyx_event ev{kind, text.c_str(), text.size(), toks, n, nullptr};
      if (cb(&ev, user) != 0) stop_cb = true;
    };
    auto send_parsed = [&](const std::vector<OutputEvent>& evs) {
      for (const auto& e : evs) {
        if (e.kind == OutputEvent::kReasoning) send(FLYX_EVENT_REASONING, e.text, nullptr, 0);
        else if (e.kind == OutputEvent::kContent) send(FLYX_EVENT_TEXT, e.text, nullptr, 0);
        else if (e.kind == OutputEvent::kToolCall)
          send(FLYX_EVENT_TOOL_CALL,
               ojson{{"name", e.call.name}, {"arguments", e.call.arguments}}.dump(-1, ' ', false,
                                                                                  ojson::error_handler_t::replace),
               nullptr, 0);
      }
    };
    TextSink sink = [&](std::string_view text, const int32_t* toks, int n) {
      if (parser) send_parsed(parser->feed(text));
      else send(FLYX_EVENT_TEXT, std::string(text), toks, n);
      return !stop_cb;
    };
    GenerateResult res;
    if (generate(*s->s, prepared.gen, sink, &res, &msg) != 0) {
      set_err(err, msg);
      return -1;
    }
    if (parser) send_parsed(parser->finish());
    if (cb) {
      flyx_event ev{FLYX_EVENT_DONE, nullptr, 0, nullptr, 0, &res.usage};
      cb(&ev, user);
    }
    if (usage) *usage = res.usage;
    return 0;
  } catch (const std::exception& ex) {
    set_err(err, ex.what());
    return -1;
  } catch (...) {
    set_err(err, "internal error");
    return -1;
  }
}

int32_t flyx_tokenize(flyx_engine* e, const char* text, int32_t* out, int32_t cap) {
  if (!e || !e->e || !text || cap < 0) return -1;
  try {
    auto ids = e->e->tokenizer().encode_prompt(text);
    if (out && cap > 0) {
      for (int32_t i = 0; i < static_cast<int32_t>(ids.size()) && i < cap; ++i) out[i] = ids[i];
    }
    return static_cast<int32_t>(ids.size());
  } catch (...) {
    return -1;
  }
}

}  // extern "C"
