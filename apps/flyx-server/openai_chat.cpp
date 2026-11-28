// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// OpenAI-compatible endpoints: /v1/chat/completions, /v1/completions, /v1/models.
//
// Extensions understood (as in vLLM): top_k; chat_template_kwargs.enable_thinking
// (or enable_thinking at the top level); the reply's thinking comes back as
// message.reasoning_content (delta.reasoning_content when streaming).
#include "apps/flyx-server/server.h"
#include "core/render/chat_json.h"

namespace flyx::server {
namespace {

// A job's error and its status: a prompt longer than the context is the request's (400, context_length_exceeded,
// worded as OpenAI words it: clients look for either to shorten the conversation), as tools or a schema no grammar
// can be made from are (400); one the server stopped serving before it ran is to be sent again (503); any other is
// the server's (500).
// `what` is what the prompt came in ("messages", "prompt").
int job_error(const JobEvent& ev, const char* what, ojson* body) {
  if (ev.prompt_tokens > 0) {
    *body = {{"error",
              {{"message", "This model's maximum context length is " + std::to_string(ev.prompt_limit) +
                               " tokens. However, your " + what + " resulted in " + std::to_string(ev.prompt_tokens) +
                               " tokens. Please reduce the length of the " + what + "."},
               {"type", "invalid_request_error"},
               {"param", what},
               {"code", "context_length_exceeded"}}}};
    return 400;
  }
  if (ev.superseded) {
    *body = {{"error", {{"message", ev.text}, {"type", "conflict_error"}, {"param", nullptr}, {"code", nullptr}}}};
    return 409;
  }
  if (ev.invalid_request) {
    *body = {{"error", {{"message", ev.text}, {"type", "invalid_request_error"}, {"param", nullptr}, {"code", nullptr}}}};
    return 400;
  }
  *body = {{"error", {{"message", ev.text}, {"type", "server_error"}, {"param", nullptr}, {"code", nullptr}}}};
  return ev.unavailable ? 503 : 500;
}

void send_job_error(httplib::Response& res, const JobEvent& ev, const char* what) {
  ojson body;
  const int status = job_error(ev, what, &body);
  send_json(res, status, body);
}

void send_cancel_error(httplib::Response& res, bool superseded) {
  if (superseded) send_openai_error(res, 409, kSupersededMessage, "conflict_error");
  else send_openai_error(res, 503, "request cancelled", "server_error");
}

const char* finish_reason(const GenerateResult& r, int tool_calls) {
  if (r.usage.finish_reason == FLYX_FINISH_LENGTH) return "length";
  return tool_calls > 0 ? "tool_calls" : "stop";
}

ojson usage_json(const GenerateResult& r) {
  const int out = output_tokens(r.usage, r.stop_token);
  return {{"prompt_tokens", r.usage.prompt_tokens},
          {"completion_tokens", out},
          {"total_tokens", r.usage.prompt_tokens + out},
          {"prompt_tokens_details", {{"cached_tokens", r.usage.reused_tokens}}}};
}

// Shared terminal frame sequence for OpenAI SSE streams (chat completions & text completions).
template <class EmitFinal>
bool finish_openai_stream(httplib::DataSink& sink, const JobEvent& ev, const char* what, const char* object_type,
                          const std::string& id, long long created, const std::string& model, bool include_usage,
                          EmitFinal&& emit_final) {
  if (ev.kind == JobEvent::kDone && ev.result.usage.finish_reason != FLYX_FINISH_CANCEL) {
    bool ok = emit_final(ev.result);
    if (ok && include_usage) {
      ok = sse_data(sink, {{"id", id},
                           {"object", object_type},
                           {"created", created},
                           {"model", model},
                           {"choices", ojson::array()},
                           {"usage", usage_json(ev.result)}});
    }
    static const std::string fin = "data: [DONE]\n\n";
    if (ok) sink.write(fin.data(), fin.size());
    return true;
  }
  JobEvent ce = ev;
  if (ev.kind == JobEvent::kDone) {
    ce.kind = JobEvent::kError;
    ce.text = ev.superseded ? kSupersededMessage : "request cancelled";
  }
  ojson body;
  job_error(ce, what, &body);
  const std::string s = "data: " + dump_json(body) + "\n\ndata: [DONE]\n\n";
  sink.write(s.data(), s.size());
  return true;
}

ojson tool_call_json(const ToolCall& c, int index, bool with_index) {
  ojson j = ojson::object();
  if (with_index) j["index"] = index;
  j["id"] = new_id("call_");
  j["type"] = "function";
  j["function"] = {{"name", c.name}, {"arguments", dump_json(c.arguments)}};
  return j;
}

// Common request fields -> job input.  Returns false with an error message.
bool read_common(const ojson& body, Request* in, bool* stream, bool* include_usage, std::string* err) {
  if (!read_optional_bool(body, "stream", stream, err)) return false;
  const auto opt = body.find("stream_options");
  if (opt != body.end() && !opt->is_null()) {
    if (!opt->is_object()) return *err = "stream_options must be an object", false;
    if (!read_optional_bool(*opt, "include_usage", include_usage, err)) return false;
  }
  if (const auto n = body.find("n"); n != body.end() && !n->is_null()) {
    const bool ok_one = (n->is_number_unsigned() && n->get<uint64_t>() == 1) ||
                        (n->is_number_integer() && n->get<int64_t>() == 1);
    if (!ok_one) {
      *err = "only n = 1 is supported";
      return false;
    }
  }
  int limit = 0;
  bool has_limit = false;
  for (const char* key : {"max_tokens", "max_completion_tokens"}) {
    const auto it = body.find(key);
    if (it == body.end() || it->is_null()) continue;
    if (!read_optional_token_limit(body, key, false, nullptr, &limit, err)) return false;
    has_limit = true;
  }
  if (has_limit) in->gen.max_new_tokens = limit;
  if (body.contains("stop") && !read_stop(body["stop"], &in->gen.stop, err)) return false;
  return true;
}

void handle_chat(Server* s, const httplib::Request& req, httplib::Response& res) {
  const RequestLog::Arrival arrival;
  if (!require_auth(*s, req, res)) return;
  ojson body;
  if (!parse_json_object(req, res, &body)) return;
  Request in;
  in.label = "chat";
  std::string err;
  bool stream = false, include_usage = false;
  if (!body.contains("messages") || !messages_from_openai(body["messages"], &in.messages, &err))
    return send_openai_error(res, 400, err.empty() ? "messages missing" : err);
  if (body.contains("tools") && !tools_from_openai(body["tools"], &in.tools, &err)) return send_openai_error(res, 400, err);
  if (!rules_from_openai(body, in.tools, &in.rules, &err)) return send_openai_error(res, 400, err);
  if (in.rules.tools == ReplyRules::kNone) in.tools.clear();  // as if none were offered
  ojson request_kwargs = ojson::object();
  if (const auto it = body.find("reasoning_effort"); it != body.end() && !it->is_null()) {
    if (!read_reasoning_effort(*it, "reasoning_effort", nullptr, &request_kwargs, &err))
      return send_openai_error(res, 400, err);
  }
  if (const auto it = body.find("enable_thinking"); it != body.end() && !it->is_null()) {
    if (!it->is_boolean()) return send_openai_error(res, 400, "enable_thinking must be a boolean");
    request_kwargs["enable_thinking"] = *it;
  }
  if (!read_chat_options(body, s->cfg.chat_defaults, request_kwargs, &in.chat_options, &err))
    return send_openai_error(res, 400, err);
  if (!read_common(body, &in, &stream, &include_usage, &err)) return send_openai_error(res, 400, err);

  AdmittedJob ad;
  if (!admit_job(s, res, std::move(in), "chat", "chatcmpl-", stream, body, arrival, false, true,
                 [](httplib::Response& r, const JobEvent& ev) { send_job_error(r, ev, "messages"); }, &ad))
    return;
  const long long created = unix_now();
  const std::string& id = ad.id;
  const std::string& model = ad.served.name;

  if (!stream) {
    Collected c = collect(ad.job, req, ad.rec.get());
    if (!c.ok) return send_job_error(res, c.failure, "messages");
    if (c.result.usage.finish_reason == FLYX_FINISH_CANCEL) return send_cancel_error(res, c.superseded);
    ojson msg = {{"role", "assistant"}};
    msg["content"] = (c.content.empty() && !c.calls.empty()) ? ojson(nullptr) : ojson(c.content);
    if (!c.reasoning.empty()) msg["reasoning_content"] = c.reasoning;
    if (!c.calls.empty()) {
      msg["tool_calls"] = ojson::array();
      for (size_t i = 0; i < c.calls.size(); ++i) msg["tool_calls"].push_back(tool_call_json(c.calls[i], int(i), false));
    }
    ojson out = {{"id", id},
                 {"object", "chat.completion"},
                 {"created", created},
                 {"model", model},
                 {"choices", ojson::array({{{"index", 0},
                                            {"message", msg},
                                            {"finish_reason", finish_reason(c.result, int(c.calls.size()))}}})},
                 {"usage", usage_json(c.result)}};
    return send_json(res, 200, out);
  }

  stream_events(res, ad.job, ad.served.sched, ad.rec, ": keep-alive\n\n",
                [id, created, model, include_usage](httplib::DataSink& sink, auto& next) {
                  auto chunk = [&](const ojson& delta, const ojson& finish) {
                    return sse_data(sink, {{"id", id},
                                           {"object", "chat.completion.chunk"},
                                           {"created", created},
                                           {"model", model},
                                           {"choices", ojson::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}});
                  };
                  bool ok = chunk({{"role", "assistant"}, {"content", ""}}, nullptr);
                  int calls = 0;
                  bool call_open = false;
                  JobEvent ev;
                  while (ok && next(&ev)) {
                    if (ev.kind == JobEvent::kReasoning) {
                      ok = chunk({{"reasoning_content", ev.text}}, nullptr);
                    } else if (ev.kind == JobEvent::kContent) {
                      ok = chunk({{"content", ev.text}}, nullptr);
                    } else if (ev.kind == JobEvent::kToolCallStart) {
                      call_open = true;
                      ok = chunk({{"tool_calls", ojson::array({{{"index", calls},
                                                                {"id", new_id("call_")},
                                                                {"type", "function"},
                                                                {"function", {{"name", ev.call.name}, {"arguments", ""}}}}})}},
                                 nullptr);
                    } else if (ev.kind == JobEvent::kToolCallDelta) {
                      if (!ev.text.empty())
                        ok = chunk({{"tool_calls", ojson::array({{{"index", calls}, {"function", {{"arguments", ev.text}}}}})}},
                                   nullptr);
                    } else if (ev.kind == JobEvent::kToolCall) {
                      if (call_open) {
                        call_open = false;
                      } else {
                        ok = chunk({{"tool_calls", ojson::array({tool_call_json(ev.call, calls, true)})}}, nullptr);
                      }
                      ++calls;
                    } else {
                      return finish_openai_stream(
                          sink, ev, "messages", "chat.completion.chunk", id, created, model, include_usage,
                          [&](const GenerateResult& r) { return chunk(ojson::object(), finish_reason(r, calls)); });
                    }
                  }
                  return false;  // the client went away
                });
}

void handle_completions(Server* s, const httplib::Request& req, httplib::Response& res) {
  const RequestLog::Arrival arrival;
  if (!require_auth(*s, req, res)) return;
  ojson body;
  if (!parse_json_object(req, res, &body)) return;
  Request in;
  in.label = "completion";
  in.chat = false;
  std::string err;
  bool stream = false, include_usage = false;
  if (!body.contains("prompt") || !body["prompt"].is_string())
    return send_openai_error(res, 400, "prompt must be a string (lists of prompts are not supported)");
  in.text = body["prompt"].get<std::string>();
  in.gen.max_new_tokens = 16;  // the legacy Completions default, also used for max_tokens: null
  if (!read_common(body, &in, &stream, &include_usage, &err)) return send_openai_error(res, 400, err);

  AdmittedJob ad;
  if (!admit_job(s, res, std::move(in), "completions", "cmpl-", stream, body, arrival, false, false,
                 [](httplib::Response& r, const JobEvent& ev) { send_job_error(r, ev, "prompt"); }, &ad))
    return;
  const long long created = unix_now();
  const std::string& id = ad.id;
  const std::string& model = ad.served.name;

  if (!stream) {
    Collected c = collect(ad.job, req, ad.rec.get());
    if (!c.ok) return send_job_error(res, c.failure, "prompt");
    if (c.result.usage.finish_reason == FLYX_FINISH_CANCEL) return send_cancel_error(res, c.superseded);
    ojson out = {{"id", id},
                 {"object", "text_completion"},
                 {"created", created},
                 {"model", model},
                 {"choices", ojson::array({{{"index", 0},
                                            {"text", c.content},
                                            {"logprobs", nullptr},
                                            {"finish_reason", finish_reason(c.result, 0)}}})},
                 {"usage", usage_json(c.result)}};
    return send_json(res, 200, out);
  }
  stream_events(res, ad.job, ad.served.sched, ad.rec, ": keep-alive\n\n",
                [id, created, model, include_usage](httplib::DataSink& sink, auto& next) {
                  auto chunk = [&](const std::string& text, const ojson& finish) {
                    return sse_data(sink, {{"id", id},
                                           {"object", "text_completion"},
                                           {"created", created},
                                           {"model", model},
                                           {"choices", ojson::array({{{"index", 0},
                                                                      {"text", text},
                                                                      {"logprobs", nullptr},
                                                                      {"finish_reason", finish}}})}});
                  };
                  JobEvent ev;
                  bool ok = true;
                  while (ok && next(&ev)) {
                    if (ev.kind == JobEvent::kContent) {
                      ok = chunk(ev.text, nullptr);
                    } else if (ev.kind == JobEvent::kError || ev.kind == JobEvent::kDone) {
                      return finish_openai_stream(
                          sink, ev, "prompt", "text_completion", id, created, model, include_usage,
                          [&](const GenerateResult& r) { return chunk("", finish_reason(r, 0)); });
                    }
                  }
                  return false;  // the client went away
                });
}

}  // namespace

void add_openai_chat_routes(httplib::Server& http, Server* s) {
  auto model_object = [](const ModelHost::Served& m) -> ojson {
    return {{"id", m.name},
            {"object", "model"},
            {"created", unix_now()},
            {"owned_by", "flyx"},
            {"max_model_len", m.sched->max_ctx()}};
  };
  http.Post("/v1/chat/completions", [s](const httplib::Request& q, httplib::Response& r) { handle_chat(s, q, r); });
  http.Post("/v1/completions", [s](const httplib::Request& q, httplib::Response& r) { handle_completions(s, q, r); });
  http.Get("/v1/models", [s, model_object](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    const ModelHost::Served m = s->host->served();
    ojson data = ojson::array();  // none while no model is loaded
    if (m.sched) data.push_back(model_object(m));
    send_json(r, 200, {{"object", "list"}, {"data", data}});
  });
  http.Get(R"(/v1/models/(.+))", [s, model_object](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    const std::string id = q.matches[1].str();
    const ModelHost::Served m = s->host->served();
    if (!m.sched || m.name != id)
      return send_openai_error(r, 404, "The model '" + id + "' does not exist", "invalid_request_error", "model",
                               "model_not_found");
    send_json(r, 200, model_object(m));
  });
}

}  // namespace flyx::server
