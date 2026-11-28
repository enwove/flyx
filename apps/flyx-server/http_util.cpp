// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared HTTP, JSON, SSE, auth, security, query-parameter, and job-admission helpers for flyx-server routes.
#include "apps/flyx-server/http_util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <type_traits>

#include "apps/common/chat_options.h"
#include "apps/common/model_source.h"
#include "apps/common/numbers.h"
#include "apps/flyx-server/server.h"
#include "core/os/path.h"
#if defined(_WIN32)
#include <winioctl.h>
#endif

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

bool read_reasoning_effort(const ojson& v, const char* field, ojson* effort, ojson* request_kwargs, std::string* err) {
  if (!v.is_string()) return *err = std::string(field) + " must be a string", false;
  const std::string s = v.get<std::string>();
  if (s != "none" && s != "minimal" && s != "low" && s != "medium" && s != "high" && s != "xhigh" && s != "max")
    return *err = std::string(field) + " must be none, minimal, low, medium, high, xhigh, or max", false;
  if (effort) *effort = s;
  (*request_kwargs)["enable_thinking"] = s != "none";
  (*request_kwargs)["reasoning_effort"] = s;
  return true;
}

bool read_chat_options(const ojson& body, const ojson& defaults, ojson& request_kwargs, ChatOptions* out, std::string* err) {
  const auto it = body.find("chat_template_kwargs");
  if (it != body.end() && !it->is_null()) {
    if (!apps::valid_chat_kwargs(*it, err)) return *err = "chat_template_kwargs " + *err, false;
    request_kwargs.update(*it);
  }
  *out = apps::resolve_chat_options(defaults, request_kwargs);
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

bool read_sampling(const ojson& body, bool thinking, apps::SamplingOverrides* out, apps::SamplingMode* mode,
                   std::string* err) {
  *mode = thinking ? apps::SamplingMode::Thinking : apps::SamplingMode::NonThinking;
  if (auto it = body.find("sampling_profile"); it != body.end() && !it->is_null()) {
    if (*it != "thinking_coding" || !thinking)
      return *err = "sampling_profile must be thinking_coding with thinking enabled", false;
    *mode = apps::SamplingMode::ThinkingCoding;
  }
  auto num = [&](const char* key, double lo, double hi, auto& field) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return true;
    if (!it->is_number()) return *err = std::string(key) + " must be a number", false;
    double v = it->get<double>();
    if (!std::isfinite(v) || v < lo || v > hi) return *err = std::string(key) + " out of range", false;
    using T = typename std::decay_t<decltype(field)>::value_type;
    field = static_cast<T>(v);
    return true;
  };
  if (!num("temperature", 0, 100, out->temperature) || !num("top_p", 0, 1, out->top_p) ||
      !num("top_k", -1, 1e9, out->top_k) || !num("presence_penalty", -2, 2, out->presence_penalty) ||
      !num("frequency_penalty", -2, 2, out->frequency_penalty) || !num("accept_min_p", 0, 1, out->accept_min_p) ||
      !num("accept_typical", 0, 1, out->accept_typical)) return false;
  if (out->top_k && *out->top_k < 0) out->top_k = 0;
  if (out->top_p && body["top_p"] == 0) {
    out->top_p = 1.f;
    out->top_k = 1;
  }
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

bool is_loopback_ip(const std::string& a) {
  return a.rfind("127.", 0) == 0 || a == "::1" || a.rfind("::ffff:127.", 0) == 0;
}

namespace {

bool iequals(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
  return true;
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

std::string ascii_lower(std::string_view s) {
  std::string out(s);
  for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return out;
}

std::string cors_origin_of(std::string_view value) {
  std::string o = ascii_lower(value);
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
  if (std::find(s.cfg.cors_origins.begin(), s.cfg.cors_origins.end(), ascii_lower(origin)) != s.cfg.cors_origins.end())
    return true;
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
      const std::string low_o = ascii_lower(o);
      if (cors_origin_of(o) == low_o) log_refused_origin(low_o);
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

namespace {

bool parse_host_authority(std::string_view authority, std::string* host_out, int* port_out) {
  if (authority.empty()) return false;
  auto parse_port = [](std::string_view s, int* out) {
    if (s.empty() || s.size() > 5) return false;
    int val = 0;
    for (char c : s) {
      if (!std::isdigit(static_cast<unsigned char>(c))) return false;
      val = val * 10 + (c - '0');
    }
    if (val <= 0 || val > 65535) return false;
    *out = val;
    return true;
  };
  int port = 80;
  std::string_view host = authority;
  if (host.front() == '[') {
    const size_t close = host.find(']');
    if (close == std::string_view::npos || close <= 1) return false;
    if (close + 1 != host.size()) {
      if (host[close + 1] != ':' || !parse_port(host.substr(close + 2), &port)) return false;
    }
    host = host.substr(1, close - 1);
  } else {
    const size_t colon = host.rfind(':');
    if (colon != std::string_view::npos) {
      if (host.find(':') != colon) return false;  // unbracketed IPv6 is invalid in Host
      if (!parse_port(host.substr(colon + 1), &port)) return false;
      host = host.substr(0, colon);
    }
  }
  if (host.empty()) return false;
  *host_out = std::string(host);
  *port_out = port;
  return true;
}

std::string_view strip_v4_mapped(std::string_view ip) {
  if (ip.size() > 7 && iequals(ip.substr(0, 7), "::ffff:")) return ip.substr(7);
  return ip;
}

bool same_host_literal(std::string_view a, std::string_view b) {
  if (iequals(a, b)) return true;
  if (a.find(':') != std::string_view::npos && b.find(':') != std::string_view::npos) {
    in6_addr ia{}, ib{};
    const std::string sa(a), sb(b);
    if (inet_pton(AF_INET6, sa.c_str(), &ia) == 1 && inet_pton(AF_INET6, sb.c_str(), &ib) == 1)
      return std::memcmp(&ia, &ib, sizeof(ia)) == 0;
  }
  return false;
}

// Validate Host header for keyless LAN management requests against this listener's local socket address, configured
// interface addresses, explicit --host hostname, and machine hostname (stopping DNS rebinding on LAN).
bool is_allowed_lan_host(const Server& s, const httplib::Request& req, std::string_view authority) {
  std::string host;
  int port = 0;
  if (!parse_host_authority(authority, &host, &port)) return false;
  const int expected_port = req.local_port > 0 ? req.local_port : s.listener.port;
  if (expected_port > 0 && port != expected_port) return false;
  if (host == "0.0.0.0" || host == "::") return false;
  if (!req.local_addr.empty() && same_host_literal(host, strip_v4_mapped(req.local_addr))) return true;
  for (const std::string& allowed : s.listener.accepted_hosts)
    if (same_host_literal(host, allowed)) return true;
  return false;
}

bool is_sep(char c) { return c == '\\' || c == '/'; }

// Lexically normalize a Windows UNC path (\\server\share\...) into lowercased components without touching the
// filesystem or network. Returns false for Win32 device namespaces (\\?\, \\.\) or traversal escaping the share.
bool normalize_unc_components(std::string_view raw, std::vector<std::string>* out) {
  out->clear();
  if (!is_unc_or_device_path(raw, /*windows_paths=*/true)) return false;
  size_t i = 0;
  while (i < raw.size() && is_sep(raw[i])) ++i;
  while (i < raw.size()) {
    size_t j = i;
    while (j < raw.size() && !is_sep(raw[j])) ++j;
    const std::string seg = ascii_lower(raw.substr(i, j - i));
    if (out->empty() && (seg == "?" || seg == ".")) return false;  // \\?\ or \\.\ device namespace
    if (seg == ".") {
      if (out->size() < 2) return false;
    } else if (seg == "..") {
      if (out->size() <= 2) return false;  // cannot climb above \\server\share
      out->pop_back();
    } else if (!seg.empty()) {
      out->push_back(seg);
    }
    i = j;
    while (i < raw.size() && is_sep(raw[i])) ++i;
  }
  return out->size() >= 2;
}

bool has_drive_letter(std::string_view raw) {
  return raw.size() >= 2 && std::isalpha(static_cast<unsigned char>(raw[0])) && raw[1] == ':';
}

// Lexically normalize an absolute Windows drive path (X:\a\b) into lowercased drive letter + path components.
// Rejects drive-relative paths (X:foo) and traversal climbing past the drive root.
bool normalize_drive_components(std::string_view raw, char* drive_out, std::vector<std::string>* out) {
  out->clear();
  if (!has_drive_letter(raw)) return false;
  if (raw.size() >= 3 && !is_sep(raw[2])) return false;  // drive-relative X:foo
  *drive_out = static_cast<char>(std::tolower(static_cast<unsigned char>(raw[0])));
  size_t i = 2;
  while (i < raw.size() && is_sep(raw[i])) ++i;
  while (i < raw.size()) {
    size_t j = i;
    while (j < raw.size() && !is_sep(raw[j])) ++j;
    const std::string seg = ascii_lower(raw.substr(i, j - i));
    if (seg == "..") {
      if (out->empty()) return false;
      out->pop_back();
    } else if (!seg.empty() && seg != ".") {
      out->push_back(seg);
    }
    i = j;
    while (i < raw.size() && is_sep(raw[i])) ++i;
  }
  return true;
}

bool under_configured_models_dir(const std::vector<std::string>& models_dirs, std::string_view raw) {
  if (is_unc_or_device_path(raw, /*windows_paths=*/true)) {
    std::vector<std::string> target;
    if (!normalize_unc_components(raw, &target)) return false;
    for (const std::string& root : models_dirs) {
      std::vector<std::string> base;
      if (!normalize_unc_components(root, &base)) continue;
      if (target.size() >= base.size() && std::equal(base.begin(), base.end(), target.begin())) return true;
    }
    return false;
  }
  char drive = 0;
  std::vector<std::string> target;
  if (!normalize_drive_components(raw, &drive, &target)) return false;
  for (const std::string& root : models_dirs) {
    char base_drive = 0;
    std::vector<std::string> base;
    if (!normalize_drive_components(root, &base_drive, &base) || base_drive != drive) continue;
    if (target.size() >= base.size() && std::equal(base.begin(), base.end(), target.begin())) return true;
  }
  return false;
}

bool probe_is_remote_drive(char letter, const PathSafetyProbe* probe) {
  const char up = static_cast<char>(std::toupper(static_cast<unsigned char>(letter)));
  if (probe && probe->is_remote_drive) return probe->is_remote_drive(up);
#if defined(_WIN32)
  const wchar_t root[4] = {static_cast<wchar_t>(up), L':', L'\\', L'\0'};
  return GetDriveTypeW(root) == DRIVE_REMOTE;
#else
  (void)up;
  return false;
#endif
}

ReparseResult probe_reparse_info(std::string_view path, const PathSafetyProbe* probe) {
  if (probe && probe->reparse_info) return probe->reparse_info(path);
#if defined(_WIN32)
  const std::wstring wpath = os::to_path(std::string(path)).wstring();
  const DWORD attrs = GetFileAttributesW(wpath.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) {
    const DWORD err = GetLastError();
    if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND || err == ERROR_INVALID_NAME)
      return {ReparseState::kNone, {}};
    return {ReparseState::kUnknownOrError, {}};
  }
  if ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) == 0) return {ReparseState::kNone, {}};
  HANDLE h = CreateFileW(wpath.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h == INVALID_HANDLE_VALUE) return {ReparseState::kUnknownOrError, {}};
  struct ReparseDataBuf {
    ULONG ReparseTag;
    USHORT ReparseDataLength;
    USHORT Reserved;
    union {
      struct {
        USHORT SubstituteNameOffset;
        USHORT SubstituteNameLength;
        USHORT PrintNameOffset;
        USHORT PrintNameLength;
        ULONG Flags;
        WCHAR PathBuffer[1];
      } SymbolicLinkReparseBuffer;
      struct {
        USHORT SubstituteNameOffset;
        USHORT SubstituteNameLength;
        USHORT PrintNameOffset;
        USHORT PrintNameLength;
        WCHAR PathBuffer[1];
      } MountPointReparseBuffer;
    };
  };
  std::vector<unsigned char> buf(MAXIMUM_REPARSE_DATA_BUFFER_SIZE);
  DWORD returned = 0;
  const BOOL ok = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, buf.data(), DWORD(buf.size()), &returned,
                                  nullptr);
  CloseHandle(h);
  if (!ok || returned < offsetof(ReparseDataBuf, SymbolicLinkReparseBuffer))
    return {ReparseState::kUnknownOrError, {}};
  const auto* rdb = reinterpret_cast<const ReparseDataBuf*>(buf.data());
  const WCHAR* base_ptr = nullptr;
  USHORT sub_off = 0, sub_len = 0;
  size_t header_bytes = 0;
  if (rdb->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
    base_ptr = rdb->SymbolicLinkReparseBuffer.PathBuffer;
    sub_off = rdb->SymbolicLinkReparseBuffer.SubstituteNameOffset;
    sub_len = rdb->SymbolicLinkReparseBuffer.SubstituteNameLength;
    header_bytes = offsetof(ReparseDataBuf, SymbolicLinkReparseBuffer.PathBuffer);
  } else if (rdb->ReparseTag == IO_REPARSE_TAG_MOUNT_POINT) {
    base_ptr = rdb->MountPointReparseBuffer.PathBuffer;
    sub_off = rdb->MountPointReparseBuffer.SubstituteNameOffset;
    sub_len = rdb->MountPointReparseBuffer.SubstituteNameLength;
    header_bytes = offsetof(ReparseDataBuf, MountPointReparseBuffer.PathBuffer);
  } else {
    return {ReparseState::kUnknownOrError, {}};
  }
  if (header_bytes + size_t(sub_off) + size_t(sub_len) > size_t(returned) || sub_len == 0)
    return {ReparseState::kUnknownOrError, {}};
  const auto* bytes = reinterpret_cast<const unsigned char*>(base_ptr) + sub_off;
  const int wchars = int(sub_len / sizeof(WCHAR));
  const int utf8_len = WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<const WCHAR*>(bytes), wchars, nullptr, 0,
                                           nullptr, nullptr);
  if (utf8_len <= 0) return {ReparseState::kUnknownOrError, {}};
  std::string out(size_t(utf8_len), '\0');
  WideCharToMultiByte(CP_UTF8, 0, reinterpret_cast<const WCHAR*>(bytes), wchars, out.data(), utf8_len, nullptr,
                      nullptr);
  return {ReparseState::kTarget, std::move(out)};
#else
  (void)path;
  return {ReparseState::kNone, {}};
#endif
}

// Convert NT reparse targets (\??\UNC\server\share, \??\Z:\dir) into Win32 syntax; returns false for other opaque
// NT device paths (\Device\Mup\..., \??\Volume{...}) that remote callers must not traverse.
bool normalize_reparse_target(std::string_view raw, std::string* out) {
  if (raw.size() >= 8 && iequals(raw.substr(0, 8), "\\??\\UNC\\")) {
    *out = "\\\\" + std::string(raw.substr(8));
    return true;
  }
  if (raw.size() >= 4 && raw.substr(0, 4) == "\\??\\") {
    const std::string_view rest = raw.substr(4);
    if (has_drive_letter(rest)) {
      *out = std::string(rest);
      return true;
    }
    return false;
  }
  if (raw.size() >= 8 && iequals(raw.substr(0, 8), "\\Device\\")) return false;
  *out = std::string(raw);
  return true;
}

// Resolve a normalized Windows reparse target against the link's parent directory when the target is relative.
bool resolve_windows_reparse_target(std::string_view link_prefix, std::string_view norm, std::string* resolved) {
  if (norm.empty()) return false;
  if (is_unc_or_device_path(norm, /*windows_paths=*/true) || has_drive_letter(norm)) {
    *resolved = std::string(norm);
    return true;
  }
  if (!has_drive_letter(link_prefix)) return false;
  const std::string drive(link_prefix.substr(0, 2));
  std::vector<std::string> comps;
  if (!is_sep(norm[0])) {
    size_t i = 2;
    while (i < link_prefix.size() && is_sep(link_prefix[i])) ++i;
    while (i < link_prefix.size()) {
      size_t j = i;
      while (j < link_prefix.size() && !is_sep(link_prefix[j])) ++j;
      const std::string_view seg = link_prefix.substr(i, j - i);
      if (seg == "..") {
        if (comps.empty()) return false;
        comps.pop_back();
      } else if (!seg.empty() && seg != ".") {
        comps.emplace_back(seg);
      }
      i = j;
      while (i < link_prefix.size() && is_sep(link_prefix[i])) ++i;
    }
    if (!comps.empty()) comps.pop_back();  // parent directory of the link itself
  }
  size_t i = 0;
  while (i < norm.size() && is_sep(norm[i])) ++i;
  while (i < norm.size()) {
    size_t j = i;
    while (j < norm.size() && !is_sep(norm[j])) ++j;
    const std::string_view seg = norm.substr(i, j - i);
    if (seg == "..") {
      if (comps.empty()) return false;
      comps.pop_back();
    } else if (!seg.empty() && seg != ".") {
      comps.emplace_back(seg);
    }
    i = j;
    while (i < norm.size() && is_sep(norm[i])) ++i;
  }
  *resolved = drive + "\\";
  for (size_t k = 0; k < comps.size(); ++k) {
    if (k > 0) resolved->push_back('\\');
    resolved->append(comps[k]);
  }
  return true;
}

bool windows_path_safe(const std::vector<std::string>& models_dirs, std::string_view raw, const PathSafetyProbe* probe,
                       int depth = 0) {
  if (depth > 4) return false;
  if (is_unc_or_device_path(raw, /*windows_paths=*/true)) return under_configured_models_dir(models_dirs, raw);
  if (has_drive_letter(raw) && probe_is_remote_drive(raw[0], probe))
    return under_configured_models_dir(models_dirs, raw);
  // Inspect lexical prefixes locally for symlinks/junctions whose target redirects to UNC/device/remote drives.
  size_t pos = has_drive_letter(raw) ? 2 : 0;
  while (pos < raw.size() && is_sep(raw[pos])) ++pos;
  int steps = 0;
  while (pos < raw.size()) {
    if (++steps > 64) return false;  // fail closed on deeply nested paths rather than skipping remaining components
    size_t next = pos;
    while (next < raw.size() && !is_sep(raw[next])) ++next;
    const std::string_view prefix = raw.substr(0, next);
    const ReparseResult rp = probe_reparse_info(prefix, probe);
    if (rp.state == ReparseState::kUnknownOrError) return false;
    if (rp.state == ReparseState::kTarget) {
      std::string norm, resolved;
      if (!normalize_reparse_target(rp.target, &norm)) return false;
      if (!resolve_windows_reparse_target(prefix, norm, &resolved)) return false;
      std::string full = resolved;
      std::string_view tail = raw.substr(next);
      while (!tail.empty() && is_sep(tail.front())) tail.remove_prefix(1);
      if (!tail.empty()) {
        if (!full.empty() && !is_sep(full.back())) full.push_back('\\');
        full.append(tail);
      }
      if (!windows_path_safe(models_dirs, full, probe, depth + 1)) return false;
    }
    pos = next;
    while (pos < raw.size() && is_sep(raw[pos])) ++pos;
  }
  return true;
}

bool path_allowed_for_windows(const std::vector<std::string>& models_dirs, std::string_view path,
                              const PathSafetyProbe* probe) {
  if (path.empty()) return true;
  std::string expanded = path[0] == '~' ? apps::expand_user_path(std::string(path)) : std::string(path);
  if (!probe && !expanded.empty() && !has_drive_letter(expanded) &&
      !is_unc_or_device_path(expanded, /*windows_paths=*/true)) {
    std::error_code ec;
    const auto abs = std::filesystem::absolute(os::to_path(expanded), ec);
    if (ec) return false;
    expanded = os::from_path(abs);
  }
  return windows_path_safe(models_dirs, expanded, probe);
}

}  // namespace

bool is_unc_or_device_path(std::string_view path, bool windows_paths) {
  return windows_paths && path.size() >= 2 && is_sep(path[0]) && is_sep(path[1]);
}

bool remote_path_allowed(const Server& s, const httplib::Request& req, std::string_view path, bool windows_paths,
                         const PathSafetyProbe* probe) {
  if (path.empty() || from_this_machine(req) || !windows_paths) return true;
  return path_allowed_for_windows(s.models_dirs, path, probe);
}

PathAllowFn remote_path_policy(const Server& s, const httplib::Request& req, bool windows_paths,
                               const PathSafetyProbe* probe) {
  if (from_this_machine(req) || !windows_paths) return {};
  std::vector<std::string> models_dirs = s.models_dirs;
  std::optional<PathSafetyProbe> probe_copy;
  if (probe) probe_copy = *probe;
  return [dirs = std::move(models_dirs), p = std::move(probe_copy)](std::string_view path) {
    return path_allowed_for_windows(dirs, path, p ? &*p : nullptr);
  };
}

bool require_local_path(const Server& s, const httplib::Request& req, httplib::Response& res, std::string_view path) {
  if (remote_path_allowed(s, req, path)) return true;
  send_json(res, 403,
            {{"error",
              {{"message", "network or device paths are not allowed from remote clients unless configured in --models-dir"},
               {"type", "permission_error"}}}});
  return false;
}

bool browser_request_safe(const Server& s, const httplib::Request& req) {
  if (!s.cfg.api_key.empty()) return true;
  const bool local = is_loopback_ip(req.remote_addr);
  // A page of an origin --cors lets in, calling the inference API: its being another site's is not held against it.
  const bool cors = cors_allowed(s, req);
  // A link to the dashboard followed from another site (a README, a chat) is a navigation to a static page: the
  // site that held the link reads nothing of it.  Its Host is still checked below for loopback peers.
  const bool page_visit = req.method == "GET" && iequals(req.get_header_value("Sec-Fetch-Mode"), "navigate") &&
                          (req.path == "/" || req.path == "/dashboard" || req.path == "/dashboard/");
  const std::string sfs = req.get_header_value("Sec-Fetch-Site");
  if (!sfs.empty() && iequals(sfs, "cross-site") && !page_visit && !cors) return false;
  const std::string host = req.get_header_value("Host");
  // Enforce loopback Host for loopback peers (stops DNS rebinding against 127.0.0.1). Non-loopback LAN inference
  // clients may arrive via custom DNS, Tailscale MagicDNS, Docker port-mapping, or HTTP/1.0; keyless LAN management
  // and private dashboard read endpoints additionally enforce is_allowed_lan_host() inside has_access().
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
  for (const auto& [k, _] : req.headers) {
    if (k.size() >= 12 && iequals(std::string_view(k).substr(0, 12), "x-forwarded-")) return true;
    if (iequals(k, "forwarded") || iequals(k, "cf-connecting-ip") || iequals(k, "true-client-ip") ||
        iequals(k, "x-real-ip") || iequals(k, "via"))
      return true;
  }
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

bool has_access(const Server& s, const httplib::Request& req, Access level) {
  if (!authorized(s, req)) return false;
  const bool keyed = !s.cfg.api_key.empty();
  if (level == Access::kLocalOnly) return from_this_machine(req) && (keyed || browser_request_safe(s, req));
  if (keyed) return true;
  if (forwarded(req) || !browser_request_safe(s, req)) return false;
  if (is_loopback_ip(req.remote_addr)) return true;
  if (level == Access::kLocalOrKeyed) return false;
  if (!is_allowed_lan_host(s, req, req.get_header_value("Host"))) return false;
  if (level == Access::kDashboardRead) return true;
  return s.lan_manage.effective(s.listener, false);
}

bool from_this_machine(const httplib::Request& req) { return is_loopback_ip(req.remote_addr) && !forwarded(req); }

bool require_auth(const Server& s, const httplib::Request& req, httplib::Response& res, bool anthropic) {
  if (authorized(s, req)) return true;
  if (anthropic) send_anthropic_error(res, 401, "invalid API key", "authentication_error");
  else send_openai_error(res, 401, "invalid API key", "authentication_error");
  return false;
}

bool require_access(const Server& s, const httplib::Request& req, httplib::Response& res, Access level,
                    const char* what) {
  if (has_access(s, req, level)) return true;
  std::string msg;
  switch (level) {
    case Access::kDashboardRead:
      msg = std::string(what) + " requires a trusted host or --api-key";
      break;
    case Access::kManage:
      msg = std::string(what) + " is for this machine (use --api-key KEY, or --no-auth on a trusted LAN)";
      break;
    case Access::kLocalOrKeyed:
      msg = std::string(what) + " is for this machine or an --api-key holder";
      break;
    case Access::kLocalOnly:
      msg = std::string(what) + " is only allowed from this machine";
      break;
  }
  send_json(res, 403, {{"error", {{"message", msg}, {"type", "permission_error"}}}});
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
  apps::SamplingOverrides request;
  apps::SamplingMode mode;
  std::string err;
  if (!read_sampling(body, in.chat && in.chat_options.enable_thinking, &request, &mode, &err)) {
    if (anthropic) send_anthropic_error(res, 400, err);
    else send_openai_error(res, 400, err);
    return false;
  }
  if (!serving(s, res, &out->served, anthropic)) return false;
  const bool images = std::any_of(in.messages.begin(), in.messages.end(), [](const ChatMessage& message) {
    return std::any_of(message.parts.begin(), message.parts.end(), [](const auto& part) {
      return part.kind == interfaces::ContentPart::kImage;
    });
  });
  auto resolved = out->served.sampling->get(mode, images);
  resolved.overlay(request, apps::SamplingOrigin::Request);
  out->sampling = resolved.params();
  request.apply_controls(&out->sampling);
  in.gen.sampling = out->sampling;
  out->id = new_id(id_prefix);
  if (set_response_id) in.response_id = out->id;
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
