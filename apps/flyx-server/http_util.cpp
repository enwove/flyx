// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared HTTP, JSON, SSE, auth, security, query-parameter, and job-admission helpers for flyx-server routes.
#include "apps/flyx-server/http_util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <random>

#include "apps/common/numbers.h"
#include "apps/flyx-server/server.h"

namespace flyx::server {

std::string new_id(const char* prefix) {
  static std::mt19937_64 rng(std::random_device{}());
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  static const char* digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s += digits[rng() % 62];
  return s;
}

long long unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string dump_json(const ojson& body) {
  return body.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

void send_json(httplib::Response& res, int status, const ojson& body) {
  res.status = status;
  res.set_content(dump_json(body), "application/json");
}

void send_openai_error(httplib::Response& res, int status, const std::string& msg, const char* type, const ojson& param,
                       const ojson& code) {
  send_json(res, status, {{"error", {{"message", msg}, {"type", type}, {"param", param}, {"code", code}}}});
}

void send_anthropic_error(httplib::Response& res, int status, const std::string& msg, const char* type) {
  send_json(res, status, {{"type", "error"}, {"error", {{"type", type}, {"message", msg}}}});
}

bool sse_data(httplib::DataSink& sink, const ojson& payload) {
  const std::string frame = "data: " + dump_json(payload) + "\n\n";
  return sink.write(frame.data(), frame.size());
}

bool sse_event(httplib::DataSink& sink, std::string_view event, const ojson& payload) {
  std::string frame;
  frame.reserve(event.size() + 18);
  frame.append("event: ");
  frame.append(event);
  frame.append("\ndata: ");
  frame.append(dump_json(payload));
  frame.append("\n\n");
  return sink.write(frame.data(), frame.size());
}

bool read_optional_bool(const ojson& body, const char* key, bool* value, std::string* err) {
  const auto field = body.find(key);
  if (field == body.end() || field->is_null()) return true;
  if (!field->is_boolean()) return *err = std::string(key) + " must be a boolean", false;
  *value = field->get<bool>();
  return true;
}

bool read_optional_string(const ojson& body, const char* key, std::string* value, std::string* err) {
  const auto field = body.find(key);
  if (field == body.end() || field->is_null()) return true;
  if (!field->is_string()) return *err = std::string(key) + " must be a string", false;
  *value = field->get<std::string>();
  return true;
}

bool read_reasoning_effort(const ojson& v, const char* field, ojson* effort, bool* thinking, ojson* kwargs,
                           std::string* err) {
  if (!v.is_string()) return *err = std::string(field) + " must be a string", false;
  const std::string s = v.get<std::string>();
  if (s != "none" && s != "minimal" && s != "low" && s != "medium" && s != "high" && s != "xhigh" && s != "max")
    return *err = std::string(field) + " must be none, minimal, low, medium, high, xhigh, or max", false;
  if (effort) *effort = s;
  if (thinking) *thinking = s != "none";
  if (kwargs) (*kwargs)["reasoning_effort"] = s;
  return true;
}

bool read_chat_template_kwargs(const ojson& body, ojson* kwargs, bool* thinking, std::string* err) {
  const auto it = body.find("chat_template_kwargs");
  if (it == body.end() || it->is_null()) return true;
  if (!it->is_object()) return *err = "chat_template_kwargs must be an object", false;
  for (auto kv = it->begin(); kv != it->end(); ++kv) (*kwargs)[kv.key()] = kv.value();
  if (kwargs->contains("enable_thinking") && (*kwargs)["enable_thinking"].is_boolean())
    *thinking = (*kwargs)["enable_thinking"].get<bool>();
  return true;
}

bool read_optional_token_limit(const ojson& body, const char* key, bool allow_zero, bool* is_zero, int* out,
                               std::string* err) {
  const auto it = body.find(key);
  if (it == body.end() || it->is_null()) return true;
  if (!it->is_number_integer() && !it->is_number_unsigned())
    return *err = std::string(key) + " must be an integer", false;
  if (it->is_number_unsigned()) {
    const uint64_t u = it->get<uint64_t>();
    if (u == 0) {
      if (!allow_zero) return *err = std::string(key) + " must be at least 1", false;
      if (is_zero) *is_zero = true;
      return true;
    }
    *out = u >= static_cast<uint64_t>(INT_MAX) ? INT_MAX : static_cast<int>(u);
    return true;
  }
  const int64_t n = it->get<int64_t>();
  if (n < (allow_zero ? 0 : 1))
    return *err = std::string(key) + (allow_zero ? " must be non-negative" : " must be at least 1"), false;
  if (n == 0) {
    if (is_zero) *is_zero = true;
    return true;
  }
  *out = n >= static_cast<int64_t>(INT_MAX) ? INT_MAX : static_cast<int>(n);
  return true;
}

bool read_sampling(const ojson& body, bool thinking, const Config& cfg, SamplingParams* out, std::string* err) {
  *out = thinking ? cfg.think_defaults : cfg.plain_defaults;
  auto num = [&](const char* key, double lo, double hi, double* v) {
    if (!body.contains(key) || body[key].is_null()) return true;
    if (!body[key].is_number()) {
      *err = std::string(key) + " must be a number";
      return false;
    }
    *v = body[key].get<double>();
    if (*v < lo || *v > hi) {
      *err = std::string(key) + " out of range";
      return false;
    }
    return true;
  };
  double t = out->temperature, p = out->top_p, k = out->top_k;
  double presence = out->presence_penalty, frequency = out->frequency_penalty;
  double min_p = out->accept_min_p, typical = out->accept_typical;
  if (!num("temperature", 0, 100, &t) || !num("top_p", 0, 1, &p) || !num("top_k", -1, 1e9, &k)) return false;
  if (!num("presence_penalty", -2, 2, &presence) || !num("frequency_penalty", -2, 2, &frequency)) return false;
  // flyx's own (lossy, off by default): relaxed acceptance of drafts (SamplingParams)
  if (!num("accept_min_p", 0, 1, &min_p) || !num("accept_typical", 0, 1, &typical)) return false;
  if (const auto it = body.find("seed"); it != body.end() && !it->is_null()) {
    if (it->is_number_unsigned()) {
      out->seed = it->get<uint64_t>();
    } else if (it->is_number_integer()) {
      out->seed = uint64_t(it->get<int64_t>());
    } else {
      *err = "seed must be an integer";
      return false;
    }
  }
  out->presence_penalty = float(presence);
  out->frequency_penalty = float(frequency);
  out->accept_min_p = float(min_p);
  out->accept_typical = float(typical);
  out->temperature = float(t);
  const bool zero_top_p = body.contains("top_p") && !body["top_p"].is_null() && p == 0;
  out->top_p = p > 0 ? float(p) : 1.f;
  out->top_k = zero_top_p ? 1 : (k > 0 ? int(k) : 0);
  return true;
}

bool read_stop(const ojson& v, std::vector<std::string>* out, std::string* err) {
  if (v.is_null()) return true;
  if (v.is_string()) {
    if (!v.get<std::string>().empty()) out->push_back(v.get<std::string>());
    return true;
  }
  if (!v.is_array()) {
    *err = "stop must be a string or a list of strings";
    return false;
  }
  for (const auto& s : v) {
    if (!s.is_string()) {
      *err = "stop must be a string or a list of strings";
      return false;
    }
    if (!s.get<std::string>().empty()) out->push_back(s.get<std::string>());
  }
  return true;
}

namespace {

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
  return true;
}

bool is_loopback_ip(const std::string& a) {
  return a.rfind("127.", 0) == 0 || a == "::1" || a.rfind("::ffff:127.", 0) == 0;
}

bool is_loopback_host(std::string_view authority) {
  if (authority.empty()) return false;
  auto valid_port = [](std::string_view port) {
    if (port.empty()) return false;
    for (char c : port)
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    return true;
  };
  std::string_view host = authority;
  if (host.front() == '[') {
    const size_t close = host.find(']');
    if (close == std::string_view::npos) return false;
    if (close + 1 != host.size()) {
      if (host[close + 1] != ':' || !valid_port(host.substr(close + 2))) return false;
    }
    host = host.substr(1, close - 1);
    return host == "::1";
  }
  const size_t colon = host.rfind(':');
  if (colon != std::string_view::npos && host.find(':') == colon) {
    if (!valid_port(host.substr(colon + 1))) return false;
    host = host.substr(0, colon);
  }
  if (iequals(host, "localhost") || host == "::1") return true;
  if (host.size() >= 5 && host.substr(0, 4) == "127.") {
    for (size_t i = 4; i < host.size(); ++i)
      if (!std::isdigit(static_cast<unsigned char>(host[i])) && host[i] != '.') return false;
    return true;
  }
  return false;
}

std::string_view origin_authority(std::string_view origin) {
  if (origin.size() > 7 && iequals(origin.substr(0, 7), "http://")) origin.remove_prefix(7);
  else if (origin.size() > 8 && iequals(origin.substr(0, 8), "https://")) origin.remove_prefix(8);
  else return {};
  const size_t slash = origin.find('/');
  return slash == std::string_view::npos ? origin : origin.substr(0, slash);
}

}  // namespace

namespace {

// The inference API: what a page of an allowed origin may call.  Not the model panel, the request log, the logs,
// the telemetry or the dashboard's own files: those stay this origin's.
bool inference_path(const std::string& path) {
  for (const std::string_view api : {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages", "/v1/models"})
    if (path.compare(0, api.size(), api) == 0 && (path.size() == api.size() || path[api.size()] == '/')) return true;
  return path == "/health";
}

std::string lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

// Says once, per origin and for the first few, what lets its pages in: the page itself sees only that its request
// failed, and whoever runs the server reads this.
void log_refused_origin(const std::string& origin) {
  static std::mutex mu;
  static std::vector<std::string> told;
  std::lock_guard<std::mutex> lock(mu);
  if (told.size() >= 16 || std::find(told.begin(), told.end(), origin) != told.end()) return;
  told.push_back(origin);
  std::fprintf(stderr, "flyx-server: refused a browser request from a page of %s: --cors %s lets its pages call the API\n",
               origin.c_str(), origin.c_str());
}

}  // namespace

std::string cors_origin_of(std::string_view value) {
  std::string o = lower(value);
  if (o == "null") return o;
  if (!o.empty() && o.back() == '/') o.pop_back();
  const size_t sep = o.find("://");
  if (sep == std::string::npos || sep == 0 || sep + 3 >= o.size()) return "";
  for (size_t i = 0; i < sep; ++i)
    if (!std::isalnum(static_cast<unsigned char>(o[i])) && o[i] != '+' && o[i] != '-' && o[i] != '.') return "";
  for (size_t i = sep + 3; i < o.size(); ++i)  // an origin has no path, query or blank
    if (o[i] == '/' || o[i] == '?' || o[i] == '#' || o[i] == '*' || std::isspace(static_cast<unsigned char>(o[i]))) return "";
  return o;
}

bool cors_allowed(const Server& s, const httplib::Request& req) {
  const std::string origin = req.get_header_value("Origin");
  if (origin.empty() || !inference_path(req.path)) return false;
  if (s.cfg.cors_any) return true;
  if (std::find(s.cfg.cors_origins.begin(), s.cfg.cors_origins.end(), lower(origin)) != s.cfg.cors_origins.end()) return true;
  return s.cfg.cors_loopback && is_loopback_host(origin_authority(origin));
}

bool admit_browser_request(const Server& s, const httplib::Request& req, httplib::Response& res) {
  const std::string origin = req.get_header_value("Origin");
  const bool cors = cors_allowed(s, req);
  if (cors) {
    // The origin is named back, not "*": the answer is for that origin's pages, and Vary keeps a cache from handing
    // it to another's.
    res.set_header("Access-Control-Allow-Origin", origin);
    res.set_header("Vary", "Origin");
    res.set_header("Access-Control-Expose-Headers", "Retry-After, x-should-retry");
  }
  auto refuse = [&] {
    std::string why = "cross-origin or unrecognized Host header rejected";
    if (!origin.empty() && inference_path(req.path)) {
      const std::string o = origin.substr(0, 200);
      why = "pages of " + o + " may not call this server from a browser: start flyx serve with --cors " + o +
            " (--cors '*' lets any page)";
      if (cors_origin_of(o) == lower(o)) log_refused_origin(lower(o));
    } else if (!origin.empty()) {
      why = "the model panel, the request log, the logs and the telemetry are not open to pages of another origin";
    }
    send_json(res, 403, {{"error", {{"message", why}, {"type", "permission_error"}}}});
  };
  // A preflight: before a request a page may not send unasked (a JSON body, an Authorization header) the browser
  // asks whether the page's origin may.  It carries no key, and is answered before any is looked for.
  if (req.method == "OPTIONS" && !origin.empty() && req.has_header("Access-Control-Request-Method")) {
    if (!cors) return refuse(), false;
    const std::string asked = req.get_header_value("Access-Control-Request-Headers");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.set_header("Access-Control-Allow-Headers",
                   asked.empty() || asked.size() > 1024 ? "authorization, content-type, x-api-key" : asked);
    // Chrome asks this too of a public page that calls a private address (Private Network Access).
    if (iequals(req.get_header_value("Access-Control-Request-Private-Network"), "true"))
      res.set_header("Access-Control-Allow-Private-Network", "true");
    res.set_header("Access-Control-Max-Age", "600");
    res.set_header("Vary", "Origin, Access-Control-Request-Headers, Access-Control-Request-Private-Network");
    res.status = 204;
    return false;
  }
  if (req.path != "/health" && !browser_request_safe(s, req)) return refuse(), false;
  return true;
}

bool browser_request_safe(const Server& s, const httplib::Request& req) {
  if (!s.cfg.api_key.empty()) return true;
  const bool local = is_loopback_ip(req.remote_addr);
  // A page of an origin --cors lets in, calling the inference API: its being another site's is not held against it.
  const bool cors = cors_allowed(s, req);
  // A link to the dashboard followed from another site (a README, a chat) is a navigation to a static page: the
  // site that held the link reads nothing of it.  Its Host is still checked below.
  const bool page_visit = req.method == "GET" && iequals(req.get_header_value("Sec-Fetch-Mode"), "navigate") &&
                          (req.path == "/" || req.path == "/dashboard" || req.path == "/dashboard/");
  const std::string sfs = req.get_header_value("Sec-Fetch-Site");
  if (!sfs.empty() && iequals(sfs, "cross-site") && !page_visit && !cors) return false;
  const std::string host = req.get_header_value("Host");
  if (local && !host.empty() && !is_loopback_host(host)) return false;
  const std::string origin = req.get_header_value("Origin");
  if (!origin.empty() && !cors) {
    if (iequals(origin, "null")) return false;
    const std::string_view auth = origin_authority(origin);
    if (auth.empty() || host.empty() || !iequals(auth, host)) return false;
  }
  return true;
}

namespace {

// Whether `given` is the key, in a time that does not tell how much of it matched.
bool same_key(std::string_view given, std::string_view key) {
  unsigned char diff = given.size() == key.size() ? 0 : 1;
  for (size_t i = 0; i < key.size(); ++i)
    diff |= static_cast<unsigned char>(key[i]) ^ static_cast<unsigned char>(i < given.size() ? given[i] : 0);
  return diff == 0;
}

// A request a proxy or a tunnel on this machine passed on: its sender is whoever reached the proxy, not this
// machine, though the connection comes from the loopback address.
bool forwarded(const httplib::Request& req) {
  for (const char* h : {"Forwarded", "X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto", "X-Real-IP", "Via"})
    if (req.has_header(h)) return true;
  return false;
}

}  // namespace

bool authorized(const Server& s, const httplib::Request& req) {
  if (s.cfg.api_key.empty()) return true;
  if (same_key(req.get_header_value("x-api-key"), s.cfg.api_key)) return true;
  const std::string auth = req.get_header_value("Authorization");
  return auth.size() > 7 && iequals(std::string_view(auth).substr(0, 7), "Bearer ") &&
         same_key(std::string_view(auth).substr(7), s.cfg.api_key);
}

bool may_manage(const Server& s, const httplib::Request& req) {
  if (!authorized(s, req)) return false;
  if (!s.cfg.api_key.empty()) return true;
  return from_this_machine(req) && browser_request_safe(s, req);
}

bool from_this_machine(const httplib::Request& req) { return is_loopback_ip(req.remote_addr) && !forwarded(req); }

bool require_auth(const Server& s, const httplib::Request& req, httplib::Response& res, bool anthropic) {
  if (authorized(s, req)) return true;
  if (anthropic) send_anthropic_error(res, 401, "invalid API key", "authentication_error");
  else send_openai_error(res, 401, "invalid API key", "authentication_error");
  return false;
}

bool require_manage(const Server& s, const httplib::Request& req, httplib::Response& res, const char* what) {
  if (may_manage(s, req)) return true;
  send_json(res, 403, {{"error", {{"message", std::string(what) + " is for this machine (or with --api-key, the key's holder)"},
                                  {"type", "permission_error"}}}});
  return false;
}

bool parse_json_object(const httplib::Request& req, httplib::Response& res, ojson* out, bool anthropic) {
  // The depth is checked here, where the body has been read: a pre-routing handler runs before it is.
  const bool shallow = interfaces::json_depth_within(req.body);
  *out = shallow ? ojson::parse(req.body, nullptr, false) : ojson(ojson::value_t::discarded);
  if (!out->is_discarded() && out->is_object()) return true;
  const std::string why = shallow ? "request body is not a JSON object"
                                  : "request JSON is nested more than " + std::to_string(interfaces::kMaxJsonDepth) +
                                        " levels deep";
  if (anthropic) send_anthropic_error(res, 400, why);
  else send_openai_error(res, 400, why);
  return false;
}

int query_int(const httplib::Request& req, const char* name, int dflt, int lo, int hi) {
  if (!req.has_param(name)) return dflt;
  long long v = 0;
  if (!apps::parse_whole(req.get_param_value(name), LLONG_MIN, LLONG_MAX, &v)) return dflt;
  return static_cast<int>(std::clamp<long long>(v, lo, hi));
}

uint64_t query_u64(const httplib::Request& req, const char* name, uint64_t dflt) {
  if (!req.has_param(name)) return dflt;
  uint64_t v = 0;
  return apps::parse_seed(req.get_param_value(name), &v) ? v : dflt;
}

bool serving(Server* s, httplib::Response& res, ModelHost::Served* out, bool anthropic) {
  *out = s->host->served();
  if (out->sched) return true;
  const ojson st = s->host->status();
  const std::string state = st.contains("state") && st["state"].is_string() ? st["state"].get<std::string>() : "none";
  const std::string target = st.contains("target") && st["target"].is_string() ? st["target"].get<std::string>() : "";
  std::string msg = state == "fetching" || state == "loading"
                        ? "the model is being " + std::string(state == "fetching" ? "fetched" : "loaded") +
                              (target.empty() ? "" : " (" + target + ")") + ": try again shortly"
                        : "no model is loaded: name one in the dashboard (GET / in a browser) or POST /v1/load";
  res.set_header("Retry-After", "5");
  if (anthropic) send_anthropic_error(res, 503, msg, "overloaded_error");
  else send_json(res, 503, {{"error", {{"message", msg}, {"type", "server_error"}}}});
  return false;
}

bool admit_job(Server* s, httplib::Response& res, Request in, const char* api, const char* id_prefix, bool stream,
               const ojson& body, RequestLog::Arrival arrival, bool anthropic, bool set_response_id,
               const std::function<void(httplib::Response&, const JobEvent&)>& on_refused, AdmittedJob* out) {
  if (!serving(s, res, &out->served, anthropic)) return false;
  out->id = new_id(id_prefix);
  if (set_response_id) in.response_id = out->id;
  std::string err;
  int status = 0;
  out->job = out->served.sched->submit(std::move(in), &err, &status, arrival.monotonic_ms);
  if (!out->job) {
    if (anthropic) send_anthropic_error(res, status, err, status == 503 ? "overloaded_error" : "invalid_request_error");
    else send_openai_error(res, status, err, status == 503 ? "server_error" : "invalid_request_error");
    return false;
  }
  JobEvent refused;
  if (!out->job->wait_start(&refused, kStartWaitMs)) {
    on_refused(res, refused);
    return false;
  }
  out->rec = s->requests ? s->requests->start(api, out->id, out->served.name, stream, body, arrival, out->job) : nullptr;
  return true;
}

JobEvent collect_events(const std::shared_ptr<Job>& job, const httplib::Request& req, Recording* rec,
                        const std::function<void(JobEvent&)>& consume) {
  using Clock = std::chrono::steady_clock;
  constexpr auto interval = std::chrono::milliseconds(500);
  auto check_at = Clock::now() + interval;
  JobEvent ev;
  while (true) {
    if (job->next(&ev, static_cast<int>(interval.count()))) {
      if (rec) rec->see(ev);
      if (ev.kind == JobEvent::kError || ev.kind == JobEvent::kDone) {
        if (rec) rec->finish();
        return ev;
      }
      consume(ev);
    }
    const auto now = Clock::now();
    if (now >= check_at) {
      if (req.is_connection_closed()) {
        job->cancel();
        check_at = Clock::time_point::max();  // drain the terminal event without cancelling again
      } else {
        check_at = now + interval;
      }
    }
  }
}

Collected collect(const std::shared_ptr<Job>& job, const httplib::Request& req, Recording* rec) {
  Collected c;
  auto end = collect_events(job, req, rec, [&](JobEvent& ev) {
    if (ev.kind == JobEvent::kReasoning) c.reasoning += ev.text;
    else if (ev.kind == JobEvent::kContent) c.content += ev.text;
    else if (ev.kind == JobEvent::kToolCall) c.calls.push_back(std::move(ev.call));
  });
  c.ok = end.kind == JobEvent::kDone;
  c.superseded = end.superseded;
  if (c.ok) c.result = std::move(end.result);
  else c.failure = std::move(end);
  return c;
}

}  // namespace flyx::server
