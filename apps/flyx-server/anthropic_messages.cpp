// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Anthropic-compatible endpoints: /v1/messages, /v1/messages/count_tokens.
//
// Thinking comes back as "thinking" blocks (with an empty signature), tool
// calls as "tool_use" blocks; usage reports reused prompt tokens as
// cache_read_input_tokens.
#include <atomic>
#include <cstdio>
#include <initializer_list>

#include "apps/flyx-server/server.h"
#include "core/render/chat_json.h"

namespace flyx::server {
namespace {

void conflict(httplib::Response& res) {
  // Retrying the old request could cancel its replacement.
  res.set_header("x-should-retry", "false");
  send_anthropic_error(res, 409, kSupersededMessage, "conflict_error");
}

// A job's error and its status: a prompt longer than the context is the request's (400, worded as the Messages API
// words it: clients look for "prompt is too long" to compact the conversation), as tools or a schema no grammar can
// be made from are (400); one the server stopped serving before it ran is to be sent again (503); any other is the
// server's (500).
int job_error(const JobEvent& ev, ojson* body) {
  if (ev.superseded) {
    *body = {{"type", "error"}, {"error", {{"type", "conflict_error"}, {"message", kSupersededMessage}}}};
    return 409;
  }
  if (ev.prompt_tokens > 0) {
    *body = {{"type", "error"},
             {"error", {{"type", "invalid_request_error"},
                        {"message", "prompt is too long: " + std::to_string(ev.prompt_tokens) + " tokens > " +
                                        std::to_string(ev.prompt_limit) + " maximum"}}}};
    return 400;
  }
  if (ev.invalid_request) {
    *body = {{"type", "error"}, {"error", {{"type", "invalid_request_error"}, {"message", ev.text}}}};
    return 400;
  }
  if (ev.unavailable) {
    *body = {{"type", "error"}, {"error", {{"type", "overloaded_error"}, {"message", ev.text}}}};
    return 503;
  }
  *body = {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", ev.text}}}};
  return 500;
}

void send_job_error(httplib::Response& res, const JobEvent& ev) {
  if (ev.superseded) return conflict(res);
  ojson body;
  const int status = job_error(ev, &body);
  send_json(res, status, body);
}

std::string stop_reason(const GenerateResult& r, int tool_calls) {
  if (r.usage.finish_reason == FLYX_FINISH_LENGTH) return "max_tokens";
  if (!r.stop_string.empty()) return "stop_sequence";
  return tool_calls > 0 ? "tool_use" : "end_turn";
}

// Claude Code can label a summary and the first turn that consumes it. Keep only the documented values in
// the request log, so these optional diagnostic headers cannot put arbitrary text or identifiers in the reader.
ojson gateway_hints(const httplib::Request& req) {
  ojson hints = ojson::object();
  auto keep = [&](const char* header, const char* field, std::initializer_list<const char*> allowed) {
    const std::string value = req.get_header_value(header);
    for (const char* candidate : allowed) {
      if (value == candidate) {
        hints[field] = candidate;
        break;
      }
    }
  };
  keep("x-claude-code-request-class", "request_class", {"main", "subagent", "workflow", "compaction", "auxiliary"});
  const auto triggers = {"manual", "auto", "reactive"};
  keep("x-claude-code-compaction", "compaction", triggers);
  keep("x-claude-code-context-compacted", "context_compacted", triggers);
  return hints;
}

ojson usage_json(const flyx_usage& u, int stop_token = -1) {
  return {{"input_tokens", u.prompt_tokens - u.reused_tokens},
          {"cache_creation_input_tokens", 0},
          {"cache_read_input_tokens", u.reused_tokens},
          {"output_tokens", output_tokens(u, stop_token)}};
}

bool read_request(Server* s, const ojson& body, Request* in, std::string* err) {
  bool dropped = false;
  if (!chat_from_anthropic(body, &in->messages, &in->tools, err, s->cfg.keep_billing_header, &dropped,
                           &in->cache_points)) return false;
  static std::atomic<bool> told{false};
  if (dropped && !told.exchange(true))
    std::fprintf(stderr,
                 "flyx-server: leaving out Claude Code's x-anthropic-billing-header line: it changes on every "
                 "request and would make every turn compute the whole prompt again (--keep-billing-header keeps "
                 "it)\n");
  if (!rules_from_anthropic(body, in->tools, &in->rules, err)) return false;
  if (in->rules.tools == ReplyRules::kNone) in->tools.clear();  // as if none were offered
  in->chat_options = s->cfg.chat_defaults;
  if (const auto t_it = body.find("thinking"); t_it != body.end() && !t_it->is_null()) {
    if (!t_it->is_object()) return *err = "thinking must be an object", false;
    std::string t = "enabled";
    if (const auto ty = t_it->find("type"); ty != t_it->end() && !ty->is_null()) {
      if (!ty->is_string()) return *err = "thinking.type must be a string", false;
      t = ty->get<std::string>();
    }
    in->chat_options.enable_thinking = t != "disabled";
  }
  if (const auto oc = body.find("output_config"); oc != body.end() && oc->is_object()) {
    if (const auto ef = oc->find("effort"); ef != oc->end() && !ef->is_null()) {
      if (!ef->is_string()) return *err = "output_config.effort must be a string", false;
      const std::string effort = ef->get<std::string>();
      if (effort != "low" && effort != "medium" && effort != "high" && effort != "max")
        return *err = "output_config.effort must be low, medium, high, or max", false;
      in->chat_options.kwargs["reasoning_effort"] = effort;
    }
  }
  if (!read_chat_template_kwargs(body, &in->chat_options, err)) return false;
  if (!read_optional_token_limit(body, "max_tokens", true, &in->gen.prefill_only, &in->gen.max_new_tokens, err))
    return false;
  if (body.contains("stop_sequences") && !read_stop(body["stop_sequences"], &in->gen.stop, err)) return false;
  return read_sampling(body, in->chat_options.enable_thinking, s->cfg, &in->gen.sampling, err);
}

void handle_messages(Server* s, const httplib::Request& req, httplib::Response& res) {
  const RequestLog::Arrival arrival;
  if (!require_auth(*s, req, res, true)) return;
  ojson body;
  if (!parse_json_object(req, res, &body, true)) return;
  Request in;
  in.label = "messages";
  const std::string session = req.get_header_value("x-claude-code-session-id");
  if (!session.empty()) {
    // The length separates the two opaque values even if either contains a delimiter.
    in.conversation_id = "messages:" + std::to_string(session.size()) + ":" + session +
                         req.get_header_value("x-claude-code-agent-id");
  }
  std::string err;
  if (!read_request(s, body, &in, &err)) return send_anthropic_error(res, 400, err);
  bool stream = false;
  if (!read_optional_bool(body, "stream", &stream, &err)) return send_anthropic_error(res, 400, err);
  if (in.rules.json && !in.messages.empty() && in.messages.back().role == "assistant")
    return send_anthropic_error(res, 400, "output_config.format cannot be combined with an assistant message prefill");
  if (in.gen.prefill_only) {
    const auto thinking = body.find("thinking");
    const auto kwargs = body.find("chat_template_kwargs");
    const bool explicit_thinking = (thinking != body.end() && thinking->is_object() &&
                                    thinking->value("type", ojson()) == "enabled") ||
                                   (kwargs != body.end() && kwargs->contains("enable_thinking"));
    if (stream || (explicit_thinking && in.chat_options.enable_thinking) || in.rules.json ||
        in.rules.tools == ReplyRules::kRequired || in.rules.tools == ReplyRules::kForced)
      return send_anthropic_error(
          res, 400, "max_tokens: 0 cannot be combined with streaming, enabled thinking, an output format, or a required tool");
  }

  AdmittedJob ad;
  if (!admit_job(s, res, std::move(in), "messages", "msg_", stream, body, arrival, true, false, send_job_error, &ad))
    return;
  if (ad.rec) ad.rec->set_client_hints(gateway_hints(req));
  const std::string& id = ad.id;
  const std::string& model = ad.served.name;

  if (!stream) {
    Collected c = collect(ad.job, req, ad.rec.get());
    if (c.superseded) return conflict(res);
    if (!c.ok) return send_job_error(res, c.failure);
    if (c.result.usage.finish_reason == FLYX_FINISH_CANCEL)
      return send_anthropic_error(res, 503, "request cancelled", "overloaded_error");
    ojson content = ojson::array();
    if (!c.reasoning.empty()) content.push_back({{"type", "thinking"}, {"thinking", c.reasoning}, {"signature", ""}});
    if (!c.content.empty()) content.push_back({{"type", "text"}, {"text", c.content}});
    for (const auto& call : c.calls)
      content.push_back({{"type", "tool_use"}, {"id", new_id("toolu_")}, {"name", call.name}, {"input", call.arguments}});
    ojson out = {{"id", id},
                 {"type", "message"},
                 {"role", "assistant"},
                 {"model", model},
                 {"content", content},
                 {"stop_reason", stop_reason(c.result, int(c.calls.size()))},
                 {"stop_sequence", c.result.stop_string.empty() ? ojson(nullptr) : ojson(c.result.stop_string)},
                 {"usage", usage_json(c.result.usage, c.result.stop_token)}};
    return send_json(res, 200, out);
  }

  // A ping is an event of its own here (the Messages API's), not an SSE comment.
  const std::string ping = "event: ping\ndata: " + ojson({{"type", "ping"}}).dump() + "\n\n";
  stream_events(
      res, ad.job, ad.served.sched, ad.rec, ping, [id, model, job = ad.job](httplib::DataSink& sink, auto& next) {
        auto event = [&](const char* name, const ojson& data) { return sse_event(sink, name, data); };
        flyx_usage prompt{};
        if (!job->wait_prompt_usage(&prompt, 0)) {
          // Queued or restoring the prefix: keep the stream live before message_start has its prompt counts.
          const std::string comment = ": keep-alive\n\n";
          if (!sink.write(comment.data(), comment.size()) ||
              !wait_stream(sink, comment, [&](int ms) { return job->wait_prompt_usage(&prompt, ms); }))
            return false;
        }
        // If the job failed before prefill (prompt_tokens == 0, e.g. context overflow or queue error after
        // kStartWaitMs), emit the error event directly without a bogus 0-token message_start/ping. Superseded
        // streams still emit message_start so the client sees the replaced message id.
        JobEvent first_ev;
        bool have_first = false;
        if (prompt.prompt_tokens == 0) {
          if (!next(&first_ev)) return false;
          if (first_ev.kind == JobEvent::kError && !first_ev.superseded) {
            ojson body;
            job_error(first_ev, &body);
            event("error", body);
            return true;
          }
          have_first = true;
        }
        ojson start = {{"type", "message_start"},
                       {"message", {{"id", id},
                                    {"type", "message"},
                                    {"role", "assistant"},
                                    {"model", model},
                                    {"content", ojson::array()},
                                    {"stop_reason", nullptr},
                                    {"stop_sequence", nullptr},
                                    {"usage", usage_json(prompt)}}}};
        bool ok = event("message_start", start) && event("ping", {{"type", "ping"}});
        int index = -1, calls = 0;
        bool call_open = false;
        std::string open;  // type of the open content block, "" if none
        auto close = [&] {
          if (open.empty()) return true;
          bool r = true;
          if (open == "thinking")
            r = event("content_block_delta", {{"type", "content_block_delta"},
                                              {"index", index},
                                              {"delta", {{"type", "signature_delta"}, {"signature", ""}}}});
          open.clear();
          return r && event("content_block_stop", {{"type", "content_block_stop"}, {"index", index}});
        };
        auto begin = [&](const std::string& type, const ojson& block) {
          if (open == type && type != "tool_use") return true;
          if (!close()) return false;
          open = type;
          ++index;
          return event("content_block_start", {{"type", "content_block_start"}, {"index", index}, {"content_block", block}});
        };
        auto poll = [&](JobEvent* out) {
          if (have_first) {
            *out = std::move(first_ev);
            have_first = false;
            return true;
          }
          return next(out);
        };
        JobEvent ev;
        while (ok && poll(&ev)) {
          if (ev.kind == JobEvent::kReasoning) {
            ok = begin("thinking", {{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}) &&
                 event("content_block_delta", {{"type", "content_block_delta"},
                                               {"index", index},
                                               {"delta", {{"type", "thinking_delta"}, {"thinking", ev.text}}}});
          } else if (ev.kind == JobEvent::kContent) {
            ok = begin("text", {{"type", "text"}, {"text", ""}}) &&
                 event("content_block_delta", {{"type", "content_block_delta"},
                                               {"index", index},
                                               {"delta", {{"type", "text_delta"}, {"text", ev.text}}}});
          } else if (ev.kind == JobEvent::kToolCallStart) {
            ++calls;
            call_open = true;
            ok = begin("tool_use",
                       {{"type", "tool_use"}, {"id", new_id("toolu_")}, {"name", ev.call.name}, {"input", ojson::object()}});
          } else if (ev.kind == JobEvent::kToolCallDelta) {
            if (!ev.text.empty())
              ok = event("content_block_delta",
                         {{"type", "content_block_delta"},
                          {"index", index},
                          {"delta", {{"type", "input_json_delta"}, {"partial_json", ev.text}}}});
          } else if (ev.kind == JobEvent::kToolCall) {
            if (call_open) {
              call_open = false;
              ok = close();
            } else {
              ++calls;
              ok = begin("tool_use",
                         {{"type", "tool_use"}, {"id", new_id("toolu_")}, {"name", ev.call.name}, {"input", ojson::object()}}) &&
                   event("content_block_delta",
                         {{"type", "content_block_delta"},
                          {"index", index},
                          {"delta", {{"type", "input_json_delta"}, {"partial_json", dump_json(ev.call.arguments)}}}}) &&
                   close();
            }
          } else if (ev.kind == JobEvent::kError) {
            ojson body;
            job_error(ev, &body);
            event("error", body);
            return true;
          } else {
            const auto& r = ev.result;
            // A superseded live stream ends with the normal message_delta + message_stop terminal sequence
            // (not an SSE error event) so Anthropic SDK clients do not auto-retry the replaced request.
            if (r.usage.finish_reason == FLYX_FINISH_CANCEL && !ev.superseded) {
              event("error", {{"type", "error"},
                              {"error", {{"type", "overloaded_error"}, {"message", "request cancelled"}}}});
              return true;
            }
            ok = close() &&
                 event("message_delta",
                       {{"type", "message_delta"},
                        {"delta", {{"stop_reason", stop_reason(r, calls)},
                                   {"stop_sequence", r.stop_string.empty() ? ojson(nullptr) : ojson(r.stop_string)}}},
                        {"usage", usage_json(r.usage, r.stop_token)}}) &&
                 event("message_stop", {{"type", "message_stop"}});
            return true;
          }
        }
        return false;  // the client went away
      });
}

}  // namespace

void add_anthropic_messages_routes(httplib::Server& http, Server* s) {
  http.Post("/v1/messages", [s](const httplib::Request& q, httplib::Response& r) { handle_messages(s, q, r); });
  http.Post("/v1/messages/count_tokens", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r, true)) return;
    ojson body;
    if (!parse_json_object(q, r, &body, true)) return;
    Request in;
    std::string err;
    if (!read_request(s, body, &in, &err)) return send_anthropic_error(r, 400, err);
    ModelHost::Served m;
    if (!serving(s, r, &m, true)) return;
    int n = m.sched->count_tokens(in, &err);
    if (n < 0) return send_anthropic_error(r, 400, err);
    send_json(r, 200, {{"input_tokens", n}});
  });
}

}  // namespace flyx::server
