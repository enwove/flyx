// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// OpenAI Responses API: POST /v1/responses (streamed or not), GET and DELETE
// /v1/responses/{id}, GET /v1/responses/{id}/input_items.
//
// Function tools only. Thinking comes back as reasoning items with
// reasoning_text content (as OpenAI's open-weight models return it); a
// reasoning item sent back is rendered as its turn's thinking, so the prompt
// continues what was generated and the session's state is reused. With
// include: ["reasoning.encrypted_content"] the item also carries its text in
// encrypted_content (not encrypted: this server keeps nothing secret from its
// client), for clients that send back only that field.
//
// Responses are kept in memory for previous_response_id unless the request
// says store: false: each response's own input and output items and the
// response it continued (a chain, not a copy of the conversation per
// response), the oldest dropped past a byte budget.
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "apps/common/numbers.h"
#include "apps/flyx-server/response_store.h"
#include "apps/flyx-server/server.h"
#include "core/render/chat_json.h"

namespace flyx::server {
namespace {

// A job's error (code, message) and its status: a prompt longer than the context is the request's (400,
// context_length_exceeded, as clients look for), as tools or a schema no grammar can be made from are (400); one the
// server stopped serving before it ran is to be sent again (503); any other is the server's (500).
int job_error(const JobEvent& ev, std::string* code, std::string* msg) {
  if (ev.prompt_tokens > 0) {
    *code = "context_length_exceeded";
    *msg = "Your input exceeds the context window of this model: " + std::to_string(ev.prompt_tokens) +
           " tokens, at most " + std::to_string(ev.prompt_limit) + ". Please adjust your input and try again.";
    return 400;
  }
  *code = ev.invalid_request ? "invalid_request_error" : "server_error";
  *msg = ev.text;
  return ev.invalid_request ? 400 : ev.unavailable ? 503 : 500;
}

void send_job_error(httplib::Response& res, const JobEvent& ev) {
  std::string code, msg;
  const int status = job_error(ev, &code, &msg);
  send_openai_error(res, status, msg, status == 400 ? "invalid_request_error" : "server_error",
                    ev.prompt_tokens > 0 ? ojson("input") : ojson(nullptr), code);
}

std::string str(const ojson& j, const char* key) {
  return j.contains(key) && j[key].is_string() ? j[key].get<std::string>() : "";
}

// Output items as the reply arrives, in the order the model wrote them, and
// the stream events that build them (`emit`; a no-op when not streaming).
class Output {
 public:
  using Emit = std::function<bool(const std::string& type, const ojson& event)>;

  Output(Emit emit, bool encrypted, const ResponsesNamespaces& namespaces)
      : emit_(std::move(emit)), encrypted_(encrypted), namespaces_(namespaces) {}

  bool reasoning(const std::string& t) {
    if (kind_ != kReasoning && !open(kReasoning)) return false;
    text_ += t;
    return emit_("response.reasoning_text.delta",
                 {{"item_id", id_}, {"output_index", index()}, {"content_index", 0}, {"delta", t}});
  }

  bool content(const std::string& t) {
    if (kind_ != kText) {
      // Whitespace between the thinking and the calls makes no message.
      pending_ += t;
      if (pending_.find_first_not_of(" \t\r\n") == std::string::npos) return true;
      if (!open(kText)) return false;
      std::string first;
      first.swap(pending_);
      return text_delta(first);
    }
    return text_delta(t);
  }

  bool call_start(const std::string& name) {
    pending_.clear();
    if (!close(true)) return false;
    kind_ = kCall;
    text_.clear();
    id_ = new_id("fc_");
    call_item_ = {{"id", id_}, {"type", "function_call"}, {"status", "in_progress"},
                  {"call_id", new_id("call_")}, {"name", name}, {"arguments", ""}};
    const auto scope = namespaces_.find(name);
    if (scope != namespaces_.end()) {
      call_item_["namespace"] = scope->second;
      call_item_["name"] = name.substr(scope->second.size() + 1);
    }
    return emit_("response.output_item.added", {{"output_index", index()}, {"item", call_item_}});
  }

  bool call_delta(const std::string& delta) {
    if (delta.empty() || kind_ != kCall) return true;
    text_ += delta;
    return emit_("response.function_call_arguments.delta",
                 {{"item_id", id_}, {"output_index", index()}, {"delta", delta}});
  }

  bool call(const ToolCall& c) {
    if (kind_ == kCall) return close(true);
    return call_start(c.name) && call_delta(dump_json(c.arguments)) && close(true);
  }

  // A limit or cancellation leaves the last open item incomplete.
  bool close(bool complete) {
    if (kind_ == kNone) return true;
    const Kind k = kind_;
    kind_ = kNone;
    const int i = index();
    const char* status = complete ? "completed" : "incomplete";
    if (k == kCall) {
      bool ok = emit_("response.function_call_arguments.done",
                      {{"item_id", id_}, {"output_index", i}, {"name", call_item_["name"]}, {"arguments", text_}});
      call_item_["arguments"] = text_;
      call_item_["status"] = status;
      items.push_back(std::move(call_item_));
      return ok && emit_("response.output_item.done", {{"output_index", i}, {"item", items.back()}});
    }
    ojson item, part;
    bool ok;
    if (k == kReasoning) {
      thinking += text_;
      part = {{"type", "reasoning_text"}, {"text", text_}};
      item = {{"id", id_}, {"type", "reasoning"}, {"summary", ojson::array()}, {"content", ojson::array({part})},
              {"status", status}};
      if (encrypted_) item["encrypted_content"] = std::string(kReasoningMark) + text_;
      ok = emit_("response.reasoning_text.done",
                 {{"item_id", id_}, {"output_index", i}, {"content_index", 0}, {"text", text_}});
    } else {
      part = {{"type", "output_text"}, {"text", text_}, {"annotations", ojson::array()}, {"logprobs", ojson::array()}};
      item = {{"id", id_}, {"type", "message"}, {"status", status}, {"role", "assistant"},
              {"content", ojson::array({part})}};
      ok = emit_("response.output_text.done", {{"item_id", id_}, {"output_index", i}, {"content_index", 0},
                                               {"text", text_}, {"logprobs", ojson::array()}});
    }
    ok = ok && emit_("response.content_part.done",
                     {{"item_id", id_}, {"output_index", i}, {"content_index", 0}, {"part", part}});
    items.push_back(item);
    return ok && emit_("response.output_item.done", {{"output_index", i}, {"item", item}});
  }

  ojson items = ojson::array();  // finished items
  std::string thinking;          // all reasoning text (for reasoning_tokens)

 private:
  enum Kind { kNone, kReasoning, kText, kCall };

  int index() const { return static_cast<int>(items.size()); }  // the open item's place

  bool open(Kind k) {
    if (!close(true)) return false;
    kind_ = k;
    text_.clear();
    id_ = new_id(k == kReasoning ? "rs_" : "msg_");
    ojson item = k == kReasoning ? ojson{{"id", id_}, {"type", "reasoning"}, {"summary", ojson::array()},
                                         {"content", ojson::array()}, {"status", "in_progress"}}
                                 : ojson{{"id", id_}, {"type", "message"}, {"status", "in_progress"},
                                         {"role", "assistant"}, {"content", ojson::array()}};
    ojson part = k == kReasoning ? ojson{{"type", "reasoning_text"}, {"text", ""}}
                                 : ojson{{"type", "output_text"}, {"text", ""}, {"annotations", ojson::array()}};
    return emit_("response.output_item.added", {{"output_index", index()}, {"item", item}}) &&
           emit_("response.content_part.added",
                 {{"item_id", id_}, {"output_index", index()}, {"content_index", 0}, {"part", part}});
  }

  bool text_delta(const std::string& t) {
    text_ += t;
    return emit_("response.output_text.delta", {{"item_id", id_}, {"output_index", index()}, {"content_index", 0},
                                                {"delta", t}, {"logprobs", ojson::array()}});
  }

  Emit emit_;
  bool encrypted_;
  const ResponsesNamespaces& namespaces_;
  Kind kind_ = kNone;
  std::string id_, text_, pending_;
  ojson call_item_;
};

// The finished response: status, output items and usage.
ojson finished(ojson r, const Output& out, const GenerateResult& g, const Scheduler& sched) {
  const bool complete = g.usage.finish_reason == FLYX_FINISH_STOP;
  r["status"] = complete ? "completed" : "incomplete";
  if (complete) r["completed_at"] = unix_now();
  if (g.usage.finish_reason == FLYX_FINISH_LENGTH) r["incomplete_details"] = {{"reason", "max_output_tokens"}};
  r["output"] = out.items;
  const int produced = output_tokens(g.usage, g.stop_token);
  const int thinking = out.thinking.empty() ? 0 : std::min(produced, sched.count_text_tokens(out.thinking));
  r["usage"] = {{"input_tokens", g.usage.prompt_tokens},
                {"input_tokens_details", {{"cached_tokens", g.usage.reused_tokens}, {"cache_write_tokens", 0}}},
                {"output_tokens", produced},
                {"output_tokens_details", {{"reasoning_tokens", thinking}}},
                {"total_tokens", g.usage.prompt_tokens + produced}};
  return r;
}

// The list endpoint returns typed content parts even for the compact string forms accepted on input.
void stored_text_parts(ojson& parts, bool output) {
  if (parts.is_null()) parts = ojson::array();
  if (parts.is_string()) parts = ojson::array({parts});
  for (auto& part : parts) {
    if (!part.is_object()) part = {{"text", part.is_string() ? part : ojson("")}};
    if (!part.contains("text") || !part["text"].is_string()) part["text"] = "";
    part["type"] = output ? "output_text" : "input_text";
    if (output && !part.contains("annotations")) part["annotations"] = ojson::array();
  }
}

ojson stored_input(ojson input) {
  // A list cursor names one item. Reserve every supplied id before minting any for missing or repeated ids.
  std::unordered_map<std::string, size_t> owner;
  for (size_t i = 0; i < input.size(); ++i) {
    const auto id = str(input[i], "id");
    if (!id.empty()) owner.try_emplace(id, i);
  }
  for (size_t i = 0; i < input.size(); ++i) {
    auto& item = input[i];
    if (!item.contains("type")) item["type"] = "message";  // a role and content are enough on input
    const auto type = str(item, "type");
    auto id = str(item, "id");
    if (id.empty() || owner.at(id) != i) {
      const char* prefix = type == "function_call_output" ? "fco_" : type == "function_call" ? "fc_"
                           : type == "reasoning" ? "rs_" : "msg_";
      do {
        id = new_id(prefix);
      } while (!owner.try_emplace(id, i).second);
      item["id"] = std::move(id);
    }
    if (type == "message") {
      stored_text_parts(item["content"], item["role"] == "assistant");
    } else if (type == "function_call_output" && item.contains("output") && item["output"].is_array()) {
      stored_text_parts(item["output"], false);
    } else if (type == "function_call" && item.contains("arguments") && item["arguments"].is_object()) {
      item["arguments"] = dump_json(item["arguments"]);
    } else if (type == "reasoning" && !item.contains("summary")) {
      item["summary"] = ojson::array();
    }
    if (!item.contains("status")) item["status"] = "completed";
  }
  return input;
}

void handle_responses(Server* s, ResponseStore* store, const httplib::Request& req, httplib::Response& res) {
  const RequestLog::Arrival arrival;
  if (!require_auth(*s, req, res)) return;
  ojson body;
  if (!parse_json_object(req, res, &body)) return;
  std::string err, missing;
  bool background = false, keep = true, parallel_tool_calls = true, stream = false;
  if (!read_optional_bool(body, "background", &background, &err) ||
      !read_optional_bool(body, "store", &keep, &err) ||
      !read_optional_bool(body, "parallel_tool_calls", &parallel_tool_calls, &err) ||
      !read_optional_bool(body, "stream", &stream, &err))
    return send_openai_error(res, 400, err);
  if (background) return send_openai_error(res, 400, "background responses are not supported");
  for (const char* key : {"conversation", "prompt"})
    if (body.contains(key) && !body[key].is_null())
      return send_openai_error(res, 400, std::string(key) + " is not supported");

  // The conversation: the responses this one continues, then its input.
  std::string previous, instructions;
  if (!read_optional_string(body, "previous_response_id", &previous, &err) ||
      !read_optional_string(body, "instructions", &instructions, &err))
    return send_openai_error(res, 400, err);
  ojson items = ojson::array();
  if (!previous.empty() && !store->conversation(previous, &items, &missing))
    return send_openai_error(res, 400, "previous response '" + missing + "' not found (not stored, or dropped)");
  auto supplied = body.find("input");
  if (supplied == body.end() || (!supplied->is_string() && !supplied->is_array()))
    return send_openai_error(res, 400, "input must be a string or a list of items");
  ojson wrapped;
  if (supplied->is_string()) wrapped = ojson::array({{{"type", "message"}, {"role", "user"}, {"content", *supplied}}});
  ojson& input = supplied->is_array() ? *supplied : wrapped;
  if (!previous.empty()) {
    for (const auto& i : input) items.push_back(i);
  }

  Request in;
  in.label = "responses";
  const std::string thread = req.get_header_value("thread-id");
  if (!thread.empty()) in.conversation_id = "responses:" + thread;
  in.join_leading_system = true;
  ResponsesNamespaces namespaces;
  if (!messages_from_responses(instructions, previous.empty() ? input : items, &in.messages, &err))
    return send_openai_error(res, 400, err);
  if (body.contains("tools") && !tools_from_responses(body["tools"], &in.tools, &err, &namespaces))
    return send_openai_error(res, 400, err);
  if (!rules_from_responses(body, in.tools, &in.rules, &err, namespaces)) return send_openai_error(res, 400, err);
  if (in.rules.tools == ReplyRules::kNone) in.tools.clear();  // as if none were offered

  // Thinking: reasoning.effort "none" turns it off; chat_template_kwargs as in chat completions.
  bool thinking = s->cfg.thinking;
  ojson effort = nullptr;
  if (const auto r_it = body.find("reasoning"); r_it != body.end() && !r_it->is_null()) {
    if (!r_it->is_object()) return send_openai_error(res, 400, "reasoning must be an object");
    if (const auto e_it = r_it->find("effort"); e_it != r_it->end() && !e_it->is_null()) {
      if (!read_reasoning_effort(*e_it, "reasoning.effort", &effort, &thinking, &in.chat_options.kwargs, &err))
        return send_openai_error(res, 400, err);
    }
  }
  if (!read_chat_template_kwargs(body, &in.chat_options.kwargs, &thinking, &err))
    return send_openai_error(res, 400, err);
  in.chat_options.enable_thinking = thinking;
  if (!read_optional_token_limit(body, "max_output_tokens", false, nullptr, &in.gen.max_new_tokens, &err))
    return send_openai_error(res, 400, err);
  if (body.contains("stop") && !read_stop(body["stop"], &in.gen.stop, &err)) return send_openai_error(res, 400, err);
  if (!read_sampling(body, thinking, s->cfg, &in.gen.sampling, &err)) return send_openai_error(res, 400, err);

  const float temperature = in.gen.sampling.temperature;
  const float top_p = in.gen.sampling.top_p;
  bool encrypted = false;
  if (const auto inc = body.find("include"); inc != body.end() && !inc->is_null()) {
    if (!inc->is_array()) return send_openai_error(res, 400, "include must be a list of strings");
    for (const auto& i : *inc) {
      if (!i.is_string()) return send_openai_error(res, 400, "include must be a list of strings");
      encrypted |= i == "reasoning.encrypted_content";
    }
  }
  if (const auto md = body.find("metadata"); md != body.end() && !md->is_null() && !md->is_object())
    return send_openai_error(res, 400, "metadata must be an object");

  AdmittedJob ad;
  if (!admit_job(s, res, std::move(in), "responses", "resp_", stream, body, arrival, false, false, send_job_error, &ad))
    return;

  const ojson format = body.contains("text") && body["text"].is_object() && body["text"].contains("format")
                           ? body["text"]["format"]
                           : ojson{{"type", "text"}};
  ojson r = {{"id", ad.id},
             {"object", "response"},
             {"created_at", unix_now()},
             {"status", "in_progress"},
             {"background", false},
             {"error", nullptr},
             {"incomplete_details", nullptr},
             {"instructions", instructions.empty() ? ojson(nullptr) : ojson(instructions)},
             {"max_output_tokens", body.contains("max_output_tokens") ? body["max_output_tokens"] : ojson(nullptr)},
             {"max_tool_calls", nullptr},
             {"model", ad.served.name},
             {"output", ojson::array()},
             {"parallel_tool_calls", parallel_tool_calls},
             {"previous_response_id", previous.empty() ? ojson(nullptr) : ojson(previous)},
             {"reasoning", {{"effort", effort}, {"summary", nullptr}}},
             {"store", keep},
             {"temperature", temperature},
             {"text", {{"format", format}}},
             {"tool_choice", body.contains("tool_choice") ? body["tool_choice"] : ojson("auto")},
             {"tools", body.contains("tools") && body["tools"].is_array() ? body["tools"] : ojson::array()},
             {"top_p", top_p},
             {"truncation", "disabled"},
             {"usage", nullptr},
             {"user", nullptr},
             {"metadata", body.contains("metadata") && body["metadata"].is_object() ? body["metadata"]
                                                                                       : ojson::object()}};

  std::shared_ptr<Scheduler> sched = ad.served.sched;
  // The request log has read the original body. Only a persisted response still needs its own input.
  auto keep_it = [store, previous, input = keep ? std::move(input) : ojson(nullptr)](const ojson& done) mutable {
    if (input.is_array()) store->put({done, stored_input(std::move(input)), previous});
  };

  if (!stream) {
    Output out([](const std::string&, const ojson&) { return true; }, encrypted, namespaces);
    auto end = collect_events(ad.job, req, ad.rec.get(), [&](JobEvent& ev) {
      if (ev.kind == JobEvent::kReasoning) out.reasoning(ev.text);
      else if (ev.kind == JobEvent::kContent) out.content(ev.text);
      else if (ev.kind == JobEvent::kToolCall) out.call(ev.call);
    });
    if (end.kind == JobEvent::kError) return send_job_error(res, end);
    out.close(end.result.usage.finish_reason == FLYX_FINISH_STOP);
    ojson done = finished(std::move(r), out, end.result, *sched);
    keep_it(done);
    return send_json(res, 200, done);
  }

  stream_events(res, ad.job, sched, ad.rec, ": keep-alive\n\n",
                [r = std::move(r), encrypted, sched, keep_it = std::move(keep_it),
                 namespaces = std::move(namespaces)](httplib::DataSink& sink, auto& next) mutable {
                  int seq = 0;
                  auto write = [&](const std::string& type, const ojson& payload) {
                    ojson ev = {{"type", type}, {"sequence_number", seq++}};
                    for (auto it = payload.begin(); it != payload.end(); ++it) ev[it.key()] = it.value();
                    return sse_event(sink, type, ev);
                  };
                  bool ok = write("response.created", {{"response", r}}) && write("response.in_progress", {{"response", r}});
                  Output out(write, encrypted, namespaces);
                  JobEvent ev;
                  while (ok && next(&ev)) {
                    if (ev.kind == JobEvent::kReasoning) {
                      ok = out.reasoning(ev.text);
                    } else if (ev.kind == JobEvent::kContent) {
                      ok = out.content(ev.text);
                    } else if (ev.kind == JobEvent::kToolCallStart) {
                      ok = out.call_start(ev.call.name);
                    } else if (ev.kind == JobEvent::kToolCallDelta) {
                      ok = out.call_delta(ev.text);
                    } else if (ev.kind == JobEvent::kToolCall) {
                      ok = out.call(ev.call);
                    } else if (ev.kind == JobEvent::kError) {
                      out.close(false);
                      ojson failed = r;
                      failed["status"] = "failed";
                      std::string code, msg;
                      job_error(ev, &code, &msg);
                      failed["error"] = {{"code", code}, {"message", msg}};
                      failed["output"] = out.items;
                      write("response.failed", {{"response", failed}});
                      return true;
                    } else {
                      const bool complete = ev.result.usage.finish_reason == FLYX_FINISH_STOP;
                      ok = out.close(complete);
                      ojson done = finished(r, out, ev.result, *sched);
                      keep_it(done);
                      if (ok) write(complete ? "response.completed" : "response.incomplete", {{"response", done}});
                      return true;
                    }
                  }
                  return false;  // the client went away
                });
}

}  // namespace

void add_openai_responses_routes(httplib::Server& http, Server* s) {
  auto store = std::make_shared<ResponseStore>(size_t(256) << 20);
  http.Post("/v1/responses", [s, store](const httplib::Request& q, httplib::Response& r) {
    handle_responses(s, store.get(), q, r);
  });
  http.Get(R"(/v1/responses/([^/]+))", [s, store](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    auto e = store->get(q.matches[1]);
    if (!e) return send_openai_error(r, 404, "no response '" + q.matches[1].str() + "'");
    send_json(r, 200, e->response);
  });
  http.Delete(R"(/v1/responses/([^/]+))", [s, store](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    if (!store->erase(q.matches[1])) return send_openai_error(r, 404, "no response '" + q.matches[1].str() + "'");
    send_json(r, 200, {{"id", q.matches[1].str()}, {"object", "response"}, {"deleted", true}});
  });
  http.Get(R"(/v1/responses/([^/]+)/input_items)", [s, store](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    const std::string order = q.has_param("order") ? q.get_param_value("order") : "desc";
    if (order != "asc" && order != "desc") return send_openai_error(r, 400, "order must be asc or desc");
    long long limit = 20;
    if (q.has_param("limit") && !apps::parse_whole(q.get_param_value("limit"), 1, 100, &limit))
      return send_openai_error(r, 400, "limit must be an integer from 1 to 100");
    auto e = store->get(q.matches[1]);
    if (!e) return send_openai_error(r, 404, "no response '" + q.matches[1].str() + "'");
    const auto& items = e->input;
    const bool asc = order == "asc";
    auto item = [&](size_t i) -> const ojson& { return items[asc ? i : items.size() - 1 - i]; };
    size_t next = 0;
    if (q.has_param("after")) {
      const std::string after = q.get_param_value("after");
      while (next < items.size() && item(next).at("id") != after) ++next;
      if (next == items.size()) return send_openai_error(r, 400, "no input item '" + after + "' in this response");
      ++next;  // the cursor itself was on the previous page
    }
    ojson page = ojson::array();
    while (next < items.size() && page.size() < static_cast<size_t>(limit)) page.push_back(item(next++));
    send_json(r, 200, {{"object", "list"},
                       {"data", page},
                       {"first_id", page.empty() ? ojson(nullptr) : page.front()["id"]},
                       {"last_id", page.empty() ? ojson(nullptr) : page.back()["id"]},
                       {"has_more", next < items.size()}});
  });
}

}  // namespace flyx::server
