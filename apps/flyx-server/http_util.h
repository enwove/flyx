// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Shared HTTP, JSON, SSE, auth, security, query-parameter, and job-admission helpers for flyx-server routes.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "apps/flyx-server/model_host.h"
#include "apps/flyx-server/request_log.h"
#include "core/exec/scheduler.h"
#include "httplib/httplib.h"

namespace flyx::server {

struct Server;
struct Config;

inline constexpr const char* kSupersededMessage = "A newer request took over this conversation.";

inline int output_tokens(const flyx_usage& u, int stop_token = -1) {
  return u.new_tokens + (stop_token >= 0 ? 1 : 0);
}

std::string new_id(const char* prefix);
long long unix_now();
std::string dump_json(const ojson& body);
void send_json(httplib::Response& res, int status, const ojson& body);
void send_openai_error(httplib::Response& res, int status, const std::string& msg,
                       const char* type = "invalid_request_error", const ojson& param = nullptr,
                       const ojson& code = nullptr);
void send_anthropic_error(httplib::Response& res, int status, const std::string& msg,
                          const char* type = "invalid_request_error");

bool sse_data(httplib::DataSink& sink, const ojson& payload);
bool sse_event(httplib::DataSink& sink, std::string_view event, const ojson& payload);

enum class Access {
  kDashboardRead,  // Private dashboard metadata (/v1/telemetry, GET /v1/load): API key, direct loopback, or keyless
                   // LAN with a trusted listener Host (never forwarded)
  kManage,         // API key, direct loopback, or keyless LAN when lan_manage is effective and Host matches listener
  kLocalOrKeyed,   // API key or direct loopback (never keyless LAN; e.g. Hub mirror mutation)
  kLocalOnly,      // Direct loopback only (desktop dialogs, shell launch, closing local apps)
};

#if defined(_WIN32)
inline constexpr bool kDefaultWindowsPaths = true;
#else
inline constexpr bool kDefaultWindowsPaths = false;
#endif

// Result of probing whether a Windows path component is a reparse point (symlink/junction).
enum class ReparseState {
  kNone,            // Not a reparse point (ordinary file/directory or does not exist).
  kTarget,          // Symlink or mount point with decoded substitute target in `target`.
  kUnknownOrError,  // Reparse point inspection failed or encountered an unsupported/malformed tag (fail closed).
};

struct ReparseResult {
  ReparseState state = ReparseState::kNone;
  std::string target;
};

// Injectable filesystem probe for testing Windows mapped-drive and reparse-point checks without touching SMB.
struct PathSafetyProbe {
  std::function<bool(char drive_letter)> is_remote_drive;
  std::function<ReparseResult(std::string_view path)> reparse_info;
};

std::string ascii_lower(std::string_view s);

// Bearer token or x-api-key, when the server has a key.
bool authorized(const Server& s, const httplib::Request& req);
// Unified management access policy check.
bool has_access(const Server& s, const httplib::Request& req, Access level);
// Whether `req` came directly from this machine itself (loopback peer without proxy forwarding headers).
bool from_this_machine(const httplib::Request& req);
bool is_loopback_ip(const std::string& address);
// True for Windows UNC shares (\\server\share, //server/share) and Win32 device namespaces (\\?\..., \\.\...).
// Always false when windows_paths is false (POSIX paths like ///mnt/models are ordinary local paths).
bool is_unc_or_device_path(std::string_view path, bool windows_paths = kDefaultWindowsPaths);
// Whether `req` may pass `path` to local filesystem inspection: direct loopback always may; on Windows, remote
// callers may not trigger UNC/device lookups, mapped network drives (DRIVE_REMOTE), or symlinks/junctions whose
// target points to a UNC/device path or remote drive, unless lexically under an operator-configured --models-dir.
bool remote_path_allowed(const Server& s, const httplib::Request& req, std::string_view path,
                         bool windows_paths = kDefaultWindowsPaths, const PathSafetyProbe* probe = nullptr);
// Returns a lifetime-independent path predicate for async/deep filesystem operations (Hub cache reads, split shards,
// LoRA/drafter dir files, companion scans). Returns an empty predicate when loopback or !windows_paths.
PathAllowFn remote_path_policy(const Server& s, const httplib::Request& req,
                               bool windows_paths = kDefaultWindowsPaths, const PathSafetyProbe* probe = nullptr);
// Guard handler helper: returns true if remote_path_allowed; otherwise answers 403 and returns false.
bool require_local_path(const Server& s, const httplib::Request& req, httplib::Response& res, std::string_view path);
// Reject browser CSRF (all keyless requests) and loopback DNS rebinding when no --api-key guards the server.
bool browser_request_safe(const Server& s, const httplib::Request& req);
// Whether `req` is a page of another origin calling the inference API with leave to (Config's cors_*): a browser
// then lets the page read the answer, which says so.
bool cors_allowed(const Server& s, const httplib::Request& req);
// A --cors value as the origin it names ("https://host[:port]", an extension's "chrome-extension://ID", "null"), in
// lower case; "" when it names none.
std::string cors_origin_of(std::string_view value);
// What a request gets before any handler: the leave a browser asks for on behalf of another origin's page (its
// preflight answered, the answer to come marked), or a refusal of what a browser must not bring.  false when the
// request was answered here.
bool admit_browser_request(const Server& s, const httplib::Request& req, httplib::Response& res);

bool require_auth(const Server& s, const httplib::Request& req, httplib::Response& res, bool anthropic = false);
bool require_access(const Server& s, const httplib::Request& req, httplib::Response& res, Access level,
                    const char* what);
bool parse_json_object(const httplib::Request& req, httplib::Response& res, ojson* out, bool anthropic = false);

// Query parameter parsers backed by apps::parse_whole / apps::parse_seed (falling back to dflt if absent/non-numeric).
int query_int(const httplib::Request& req, const char* name, int dflt, int lo, int hi);
uint64_t query_u64(const httplib::Request& req, const char* name, uint64_t dflt);

// Leave the caller's default in place when the field is missing or null.
bool read_optional_bool(const ojson& body, const char* key, bool* value, std::string* err);
bool read_optional_string(const ojson& body, const char* key, std::string* value, std::string* err);
// OpenAI reasoning effort ("none", "minimal", "low", "medium", "high", "xhigh"):
// "none" disables thinking, other values enable it, and all reach the template as kwargs["reasoning_effort"].
bool read_reasoning_effort(const ojson& v, const char* field, ojson* effort, bool* thinking, ojson* kwargs,
                           std::string* err);
// Merge body["chat_template_kwargs"] into *kwargs and apply its boolean enable_thinking if present.
bool read_chat_template_kwargs(const ojson& body, ojson* kwargs, bool* thinking, std::string* err);
// Parse an optional integer token limit (clamping values above INT_MAX without signed overflow).
// Missing or null leaves *out unchanged; when allow_zero is true, 0 sets *is_zero = true.
bool read_optional_token_limit(const ojson& body, const char* key, bool allow_zero, bool* is_zero, int* out,
                               std::string* err);
// temperature / top_p / top_k / seed / presence_penalty / frequency_penalty
// from a request body; request values win over the server defaults for the thinking mode.
bool read_sampling(const ojson& body, bool thinking, const Config& cfg, SamplingParams* out, std::string* err);
// "stop": a string or a list of strings.
bool read_stop(const ojson& v, std::vector<std::string>* out, std::string* err);

// The model a request goes to (*out), or false with 503 answered when the server has none.
bool serving(Server* s, httplib::Response& res, ModelHost::Served* out, bool anthropic = false);

struct AdmittedJob {
  ModelHost::Served served;
  std::shared_ptr<Job> job;
  std::string id;
  std::shared_ptr<Recording> rec;
};

// Shared admission pipeline: check serving -> mint id -> submit -> wait_start -> start RequestLog recording.
bool admit_job(Server* s, httplib::Response& res, Request in, const char* api, const char* id_prefix, bool stream,
               const ojson& body, RequestLog::Arrival arrival, bool anthropic, bool set_response_id,
               const std::function<void(httplib::Response&, const JobEvent&)>& on_refused, AdmittedJob* out);

}  // namespace flyx::server
