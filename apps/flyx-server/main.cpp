// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx serve: OpenAI- and Anthropic-compatible HTTP API, a command of the one flyx executable (apps/flyx-cli/main.cpp;
// called by the name flyx-server, a link to it, the same).
//
//   flyx serve [MODEL] [--host 127.0.0.1] [--port 8080] [options]
//
// MODEL: a .gguf or .flyx file, or a Hugging Face name (org/name:QUANT), fetched into the Hugging Face cache.  Without
// one the server starts with the model it served last (the settings file), else none: the dashboard's model panel
// loads one, and loads another while it runs.
//
// Endpoints: POST /v1/chat/completions, POST /v1/completions, GET /v1/models,
// POST /v1/responses, GET / DELETE /v1/responses/{id} (OpenAI); POST
// /v1/messages, POST /v1/messages/count_tokens (Anthropic); GET /v1/telemetry
// (monitoring), GET /v1/requests (the last requests), GET
// /v1/logs (the last lines on standard error), GET
// /health; GET / (a browser: the dashboard page, else the endpoints as JSON),
// GET /dashboard and its files under /dashboard/. One thread runs the model; requests queue behind it.
//
// On a terminal it draws its dashboard there (monitor.h: the model loading,
// then the live panel and its log) and opens the page in the desktop's
// browser once it serves; --no-monitor and --no-browser leave
// either out, and neither happens when its output is not a terminal.
#include <algorithm>
#include <atomic>
#include <climits>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

#include "apps/common/defaults.h"
#include "apps/common/engine_flags.h"
#include "apps/common/model_source.h"
#include "apps/common/numbers.h"
#include "apps/flyx-cli/monitor.h"
#include "apps/flyx-server/server.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace {

volatile std::sig_atomic_t g_stop_signal = 0;

void on_signal(int) { g_stop_signal = 1; }

// The address a browser on this machine reaches a server listening on `host` at.
std::string local_url(const std::string& host, int port) {
  std::string h = host == "0.0.0.0" || host == "::" || host.empty() ? "127.0.0.1" : host;
  if (h.find(':') != std::string::npos && h.front() != '[') h = "[" + h + "]";
  return "http://" + h + ":" + std::to_string(port);
}

bool flyx_running_at(const std::string& url) {
  httplib::Client cli(url);
  cli.set_connection_timeout(0, 200000);
  cli.set_read_timeout(0, 300000);
  auto res = cli.Get("/", {{"Accept", "application/json"}});
  if (!res || res->status != 200) return false;
  const flyx::ojson body = flyx::interfaces::parse_json(res->body);
  return body.is_object() && body.contains("name") && body["name"] == "flyx";
}

// Keep an Explorer-launched console window open long enough for the user to read a startup error.
void pause_owned_console() {
  if (!flyx::os::owns_console()) return;
  flyx::os::write_original_stderr("\nPress Enter to close...");
  (void)std::getchar();
}

[[noreturn]] void usage(bool asked = false) {
  std::fprintf(asked ? stdout : stderr,
               "usage: flyx serve [MODEL] [options]      (flyx-server [MODEL] [options]: the same)\n"
               "  MODEL                a .gguf or .flyx file, or a Hugging Face name (org/name:QUANT), fetched into\n"
               "                       the Hugging Face cache; without it the last model served, or none until the\n"
               "                       dashboard's model panel loads one\n"
               "  --host H             address to listen on (default 127.0.0.1)\n"
               "  --port P             port (default 8080)\n"
               "  --name NAME          model name reported by the API (default: the file's name)\n"
               "  --models-dir DIR     a directory of model files (.gguf, .flyx) the dashboard lists beside the\n"
               "                       Hugging Face cache's, searched 6 directories deep (repeatable; FLYX_MODELS_DIR:\n"
               "                       more, separated as PATH is); flyx never deletes from it\n"
               "  --sessions N         conversations kept on the GPU (default 8; no more than fit)\n"
               "  --host-cache GB      host memory for the checkpoints of conversations that gave their session\n"
               "                       to another, loaded when they come back (default 8, less when a long context's\n"
               "                       K/V take host memory first; 0: none)\n"
               "  --kv-disk DIR[:GB]   keep reusable prompt prefixes on disk as the conversation grows, and spill\n"
               "                       host checkpoints there when memory runs short (default GB: 64, at most\n"
               "                       half the free space)\n"
               "  --no-batch           requests running at once take turns, a round each, instead of sharing rounds\n"
               "  --no-think           requests that do not say otherwise get no thinking block\n"
               "  --keep-billing-header  keep Claude Code's x-anthropic-billing-header line in the system prompt\n"
               "                       (left out by default: it changes on every request and defeats prompt reuse)\n"
               "  --temperature T      default sampling for requests that do not set it\n"
               "  --top-p P / --top-k K\n"
               "  --presence-penalty P  default presence penalty (-2..2); without it Qwen3.5's own: 1.5 for\n"
               "                       requests without thinking, 0 with\n"
               "  --frequency-penalty F  default frequency penalty (-2..2, default 0)\n"
               "  --api-key KEY        require this key (Authorization: Bearer KEY or x-api-key; default: FLYX_API_KEY)\n"
               "  --cors ORIGIN        let the pages of ORIGIN call the API from a browser (repeatable): https://host[:port],\n"
               "                       an extension's origin, null (a page opened from a file); '*': any page; none: not\n"
               "                       even this machine's own pages (localhost), which may by default\n"
               "  --queue N            requests allowed to wait (default 16)\n"
               "  --queue-timeout S    longest wait in the queue (default 600)\n"
               "  --idle-unload S      free sessions after S idle seconds (default 0 = never)\n"
               "  --keep-alive S       unload the model after S idle seconds, reload on demand (default 0 = never)\n"
               "  --threads N          HTTP worker threads (default 64): one per request in flight, queued or running\n"
               "  --no-monitor         no dashboard on the terminal, the log as it comes (also when the output is\n"
               "                       not a terminal)\n"
               "  --no-browser         do not open the dashboard page (GET /) in the browser once serving\n"
               "  --no-request-log     keep no log of requests: by default the last 50 prompts and answers are kept\n"
               "                       in memory (never on disk) for the dashboard, readable from this machine or\n"
               "                       with the API key (GET /v1/requests)\n"
               "  --no-system-log      keep no log of what the server writes to standard error: by default the last\n"
               "                       2000 lines are kept in memory for the dashboard's Logs page (GET /v1/logs)\n"
               "  --dashboard-dir DIR  serve the dashboard's files from DIR (apps/flyx-server/dashboard in the\n"
               "                       repository) where it has them: a change to the page shows on a reload\n"
               "  -v                   log every request\n"
               "the engine:\n%s",
               flyx::apps::engine_flags_help(0).c_str());
  std::exit(asked ? 0 : 2);
}

// An option that cannot be followed: which one and why, not the usage it would be lost in.
[[noreturn]] void bad_usage(const std::string& why) {
  std::fprintf(stderr, "error: %s (flyx serve --help lists the options)\n", why.c_str());
  std::exit(2);
}

// --kv-disk DIR[:GB]: GB after the last colon when it is a number (a Windows drive's colon is not).
void kv_disk(const std::string& v, flyx::SchedulerOptions* so) {
  const size_t colon = v.rfind(':');
  std::string dir = v;
  double gb = -1;
  if (colon != std::string::npos && colon + 1 < v.size() &&
      v.find_first_not_of("0123456789.", colon + 1) == std::string::npos) {
    dir = v.substr(0, colon);
    gb = flyx::apps::number_flag("--kv-disk", v.substr(colon + 1), 0, 1e6);
  }
  if (dir.empty()) std::fprintf(stderr, "error: --kv-disk DIR[:GB]\n"), std::exit(2);
  uint64_t bytes = gb >= 0 ? uint64_t(gb * 1e9) : flyx::apps::kDefaultDiskCacheBytes;
  if (gb < 0) {
    std::error_code ec;
    auto p = flyx::os::to_path(dir);
    while (!p.empty() && !std::filesystem::exists(p, ec) && p.has_parent_path() && p.parent_path() != p)
      p = p.parent_path();
    const std::filesystem::space_info s = std::filesystem::space(p, ec);
    if (!ec) bytes = std::min<uint64_t>(bytes, s.available / 2);
  }
  so->disk_cache_dir = dir;
  so->disk_cache_bytes = bytes;
}

}  // namespace

namespace flyx::server {

namespace {

// What the command line asks for.
struct ServeArgs {
  std::string model, name, host = flyx::apps::kDefaultHost;
  std::vector<std::string> models_dirs;
  std::string dashboard_dir;  // --dashboard-dir: the page's files read from here first
  int port = flyx::apps::kDefaultPort, threads = flyx::apps::kDefaultHttpThreads;
  bool port_given = false;
  bool monitor_wanted = true, browser_wanted = true, request_log = true, system_log = true;
  EngineOptions eo;
  SchedulerOptions so;
  Config cfg;
};

// argv[1] is the model unless it is an option, the rest the options (argv[0]: "serve", or the name the executable was
// called by).  A bad option prints the usage and exits.
ServeArgs parse_args(int argc, char** argv) {
  ServeArgs a;
  int first = 1;
  if (argc >= 2 && argv[1][0] != '-') a.model = argv[1], first = 2;
  a.eo.ctx_max = true;  // default: model's trained context, or the longest GPU and host memory hold
  apps::SamplingOverrides sampling_overrides;
  for (int i = first; i < argc; ++i) {
    const std::string f = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) bad_usage(f + " needs a value");
      return argv[++i];
    };
    if (f == "--host") a.host = next();
    else if (f == "--port") {
      a.port = apps::whole_flag("--port", next(), 1, 65535);
      a.port_given = true;
    }
    else if (f == "--name") a.name = next();
    else if (f == "--models-dir") a.models_dirs.push_back(next());
    else if (f == "--help" || f == "-h") usage(true);
    else if (f == "--sessions") a.so.max_sessions = apps::whole_flag("--sessions", next(), 1, INT_MAX);
    else if (f == "--host-cache") a.so.host_cache_bytes = uint64_t(apps::number_flag("--host-cache", next(), 0, 1e9) * 1e9);
    else if (f == "--kv-disk") kv_disk(next(), &a.so);
    else if (f == "--no-batch") a.so.batch_rounds = false;
    else if (f == "--no-think") a.cfg.thinking = false;
    else if (f == "--keep-billing-header") a.cfg.keep_billing_header = true;
    else if (apps::sampling_flag(f, next, &sampling_overrides)) continue;
    else if (f == "--api-key") a.cfg.api_key = next();
    else if (f == "--cors") {
      const std::string v = next();
      if (v == "*") a.cfg.cors_any = true;
      else if (v == "none") a.cfg.cors_loopback = false;
      else if (const std::string o = cors_origin_of(v); !o.empty()) a.cfg.cors_origins.push_back(o);
      else bad_usage("--cors " + v + ": an origin (https://host[:port], chrome-extension://ID, null), '*' or none");
    }
    else if (f == "--queue") a.so.queue_limit = apps::whole_flag("--queue", next(), 1, INT_MAX);
    else if (f == "--queue-timeout") a.so.queue_timeout_ms = apps::whole_flag("--queue-timeout", next(), 1, INT_MAX / 1000) * 1000;
    else if (f == "--idle-unload") a.so.idle_unload_ms = apps::whole_flag("--idle-unload", next(), 0, INT_MAX / 1000) * 1000;
    else if (f == "--keep-alive") a.so.keep_alive_ms = apps::whole_flag("--keep-alive", next(), 0, INT_MAX / 1000) * 1000;
    else if (f == "--threads") a.threads = apps::whole_flag("--threads", next(), 1, 4096);
    else if (f == "--no-monitor") a.monitor_wanted = false;
    else if (f == "--no-browser") a.browser_wanted = false;
    else if (f == "--no-request-log") a.request_log = false;
    else if (f == "--no-system-log") a.system_log = false;
    else if (f == "--dashboard-dir") a.dashboard_dir = next();
    else if (f == "-v") a.cfg.verbose = a.so.verbose = a.eo.verbose = 1;
    else if (!apps::engine_flag(f, next, &a.eo))
      bad_usage(!f.empty() && f[0] == '-' ? "unknown option " + f : "unexpected argument '" + f + "' (one model at most)");
  }
  sampling_overrides.apply(&a.cfg.think_defaults);
  sampling_overrides.apply(&a.cfg.plain_defaults);
  if (a.cfg.api_key.empty()) {
    a.cfg.api_key = os::getenv_str(env::kApiKey);
    if (a.cfg.api_key.empty()) a.cfg.api_key = os::getenv_str(env::kKeyAlias);
  }
  for (const std::string& d : apps::models_dirs_from_env()) a.models_dirs.push_back(d);
  return a;
}

// The dashboard's files: built in, or from --dashboard-dir where that has the file, so a change to
// the page shows on a reload.  `path` is the file's path under /dashboard/; nothing outside the directory is read.
const char* content_type(const std::string& path) {
  const size_t dot = path.rfind('.');
  if (dot == std::string::npos) return "application/octet-stream";
  const std::string ext = path.substr(dot + 1);
  return ext == "html" ? "text/html; charset=utf-8" : ext == "js" ? "text/javascript; charset=utf-8"
       : ext == "css" ? "text/css; charset=utf-8" : ext == "svg" ? "image/svg+xml"
       : ext == "ico" ? "image/x-icon" : ext == "png" ? "image/png" : "application/octet-stream";
}
bool dashboard_file(const std::string& dir, const std::string& path, std::string* body) {
  if (path.empty() || path.find("..") != std::string::npos || path.find('\\') != std::string::npos ||
      path.find(':') != std::string::npos || path.find('\0') != std::string::npos || path[0] == '/')
    return false;
  if (!dir.empty()) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path rel = os::to_path(path);
    if (!rel.is_absolute() && !rel.has_root_name() && !rel.has_root_directory()) {
      const fs::path base = fs::weakly_canonical(os::to_path(dir), ec);
      const fs::path target = ec ? fs::path{} : fs::weakly_canonical(base / rel, ec);
      bool inside = !ec && !base.empty() && !target.empty();
      if (inside) {
        auto bi = base.begin(), ti = target.begin();
        for (; bi != base.end() && ti != target.end(); ++bi, ++ti)
          if (*bi != *ti) {
            inside = false;
            break;
          }
        if (bi != base.end() || ti == target.end()) inside = false;
      }
      constexpr uintmax_t kMaxAssetBytes = uintmax_t(16) << 20;
      if (inside && fs::is_regular_file(target, ec) && !ec) {
        const uintmax_t sz = fs::file_size(target, ec);
        if (!ec && sz <= kMaxAssetBytes) {
          std::ifstream f(target, std::ios::binary);
          if (f) {
            body->assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            return true;
          }
        }
      }
    }
  }
  for (size_t i = 0; i < kDashboardFilesCount; ++i)
    if (path == kDashboardFiles[i].path) {
      body->assign(reinterpret_cast<const char*>(kDashboardFiles[i].data), kDashboardFiles[i].size);
      return true;
    }
  return false;
}

// The routes besides the APIs': /health, the dashboard (built in: to a browser at /, at /dashboard to anyone, its
// files under /dashboard/), and errors as JSON.
void add_base_routes(httplib::Server& http, ModelHost& models, const std::string& dashboard_dir) {
  http.Get("/health", [&models](const httplib::Request&, httplib::Response& res) {
    const ModelHost::Served m = models.served();
    send_json(res, 200, {{"status", "ok"}, {"model", m.sched ? ojson(m.name) : ojson(nullptr)},
                         {"queued", m.sched ? m.sched->queued() : 0}, {"loaded", m.sched && m.sched->is_loaded()},
                         {"load", models.status().value("state", "none")}});
  });
  // no-cache: the browser asks again each time, so a new build (or a file edited under --dashboard-dir) is what loads.
  auto file = [dashboard_dir](const std::string& path, httplib::Response& res) {
    std::string body;
    if (!dashboard_file(dashboard_dir, path, &body)) {
      send_json(res, 404, {{"error", {{"message", "no such file: /dashboard/" + path}, {"type", "not_found"}}}});
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_content(std::move(body), content_type(path));
  };
  auto page = [file](httplib::Response& res) { file("index.html", res); };
  http.Get("/", [&models, page](const httplib::Request& req, httplib::Response& res) {
    if (req.get_header_value("Accept").find("text/html") != std::string::npos) return page(res);
    const ModelHost::Served m = models.served();
    send_json(res, 200, {{"name", "flyx"}, {"model", m.sched ? ojson(m.name) : ojson(nullptr)},
                         {"endpoints", {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/responses",
                                        "/v1/messages", "/v1/messages/count_tokens", "/v1/telemetry", "/v1/load",
                                        "/v1/unload", "/v1/hub/search", "/v1/hub/files", "/v1/hub/cached", "/v1/hub/dir", "/v1/requests",
                                        "/v1/logs", "/health",
                                        "/dashboard"}}});
  });
  http.Get("/favicon.ico", [file](const httplib::Request&, httplib::Response& res) { file("brand/favicon.ico", res); });
  http.Get("/dashboard", [page](const httplib::Request&, httplib::Response& res) { page(res); });
  http.Get("/dashboard/", [page](const httplib::Request&, httplib::Response& res) { page(res); });
  http.Get(R"(/dashboard/(.+))", [file](const httplib::Request& req, httplib::Response& res) { file(req.matches[1], res); });
  http.set_exception_handler([](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
    std::string msg = "internal error";
    try {
      std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      msg = e.what();
    } catch (...) {
    }
    if (req.path.rfind("/v1/messages", 0) == 0)
      send_json(res, 500, {{"type", "error"}, {"error", {{"type", "api_error"}, {"message", msg}}}});
    else
      send_json(res, 500, {{"error", {{"message", msg}, {"type", "server_error"}}}});
  });
}

}  // namespace

int run(int argc, char** argv) {
  const ServeArgs args = parse_args(argc, argv);
  const std::string &model = args.model, &host = args.host;
  int port = args.port;
  const EngineOptions& eo = args.eo;
  const Config& cfg = args.cfg;
  const bool interactive = os::stdout_is_terminal() && os::stdin_is_terminal();

  // The port first: one in use says so (or steps aside to a free port when --port was not pinned) before the
  // terminal dashboard opens or the model takes its time to load.
  // One server a port.  httplib's default lets a second process bind a port in use (SO_REUSEPORT on Linux, where the
  // kernel then deals connections between the two; SO_REUSEADDR on Windows, where which of them answers is not
  // defined), so a second `flyx serve` started by mistake would take half the requests with another model.
  auto make_http = [] {
    auto s = std::make_unique<httplib::Server>();
    s->set_socket_options([](socket_t sock) {
      const int on = 1;
#ifdef _WIN32
      setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char*>(&on), sizeof(on));
#else
      setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));  // a restart binds while old connections linger
#endif
    });
    return s;
  };
  std::unique_ptr<httplib::Server> http_ptr = make_http();
  bool bound = false;
  for (int p = port; p <= std::min(65535, port + 32); ++p) {
    if (p > port) http_ptr = make_http();
    if (http_ptr->bind_to_port(host, p)) {
      port = p;
      bound = true;
      break;
    }
    if (args.port_given) break;
    const std::string existing = local_url(host, p);
    if (flyx_running_at(existing)) {
      const std::string page =
          existing + "/" + (cfg.api_key.empty() ? "" : "#key=" + httplib::detail::encode_query_param(cfg.api_key));
      if (os::owns_console()) {
        if (args.browser_wanted) os::open_url(page);
        os::write_original_stderr("flyx-server: already running at " + existing + "/\n");
        return 0;
      }
      os::write_original_stderr("error: flyx-server is already running at " + existing +
                                "/ (--port N starts another)\n");
      return 1;
    }
  }
  if (!bound && !args.port_given) {
    auto cand = make_http();
    const int any = cand->bind_to_any_port(host);
    if (any > 0) {
      http_ptr = std::move(cand);
      port = any;
      bound = true;
    }
  }
  if (!bound) {
    os::write_original_stderr("error: cannot listen on " + host + ":" + std::to_string(port) +
                              " (in use by another flyx serve or another program? --port N takes another)\n");
    pause_owned_console();
    return 1;
  }
  httplib::Server& http = *http_ptr;

  // On a terminal: the dashboard from now on (what the process writes to standard error goes into its log), and the
  // page in the browser once the server answers.
  const std::string base = local_url(host, port);
  static std::atomic<bool> into_dashboard{false};
  static SystemLog system_log;  // single stderr buffer backing both GET /v1/logs and the terminal dashboard
  const bool captured = (args.system_log || (args.monitor_wanted && interactive)) &&
                        os::capture_stderr([](const std::string& line) {
                          system_log.add(line);
                          if (!into_dashboard) os::write_original_stderr(line + "\n");
                        });
  system_log.set_enabled(captured && args.system_log);
  monitor::ServeView view;
  view.model = model.empty() ? "no model yet: name one on the page" : model;
  view.url = base;
  view.key = cfg.api_key;
  view.page = base + "/";
  view.recent_logs = [](size_t n) { return system_log.last(n); };
  // The key, if any, in the address's fragment: the page keeps it and takes it off the address; it never reaches
  // the server in a request line.
  const std::string to_open =
      view.page + (cfg.api_key.empty() ? "" : "#key=" + httplib::detail::encode_query_param(cfg.api_key));
  view.open_page = [to_open] { return os::open_url(to_open); };
  g_stop_signal = 0;
  std::atomic<bool> startup_abort{false};
  std::thread dashboard;
  if (args.monitor_wanted && interactive && captured) {
    into_dashboard = true;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    dashboard = std::thread([&view, &startup_abort] {
      monitor::serve_dashboard(&view);
      if (view.listening || startup_abort.load()) return;
      // Quit while the model loads: the loading cannot be cut short, so the process goes now (as Ctrl-C did).
      into_dashboard = false;
      std::string tail;
      for (const auto& l : system_log.last(12)) tail += l + "\n";
      os::write_original_stderr(tail + "flyx-server: stopped while loading\n");
      std::_Exit(130);
    });
  }
  // The dashboard gone (the user quit, a signal, an error): the terminal back, with the last lines its log showed.
  auto end_dashboard = [&](bool aborting = false) {
    if (!dashboard.joinable()) return;
    if (aborting) startup_abort = true;
    view.quit = true;
    dashboard.join();
    into_dashboard = false;
    std::string tail;
    for (const auto& l : system_log.last(12)) tail += l + "\n";
    os::write_original_stderr(tail);
  };
  if (port != args.port)
    std::fprintf(stderr, "flyx-server: port %d is in use; listening on %s instead\n", args.port, base.c_str());

  std::string err;
  ModelHost models(args.so);
  EngineOptions start_from = eo;
  start_from.drafter_files.clear();  // the command line's model's own: a model named on the page names its drafters
  Server srv{&models, cfg, start_from, os::unix_seconds(), {}};
  RequestLog requests(args.request_log);
  srv.requests = &requests;
  srv.logs = args.system_log ? &system_log : nullptr;
  srv.models_dirs = args.models_dirs;
  // The model named here loads now, before the server listens; without one, the last one served loads once it
  // listens (the page shows it coming), or the server waits for the page to name one.
  ModelHost::Want later;
  const bool remembered = model.empty() && ModelHost::remembered(&later, start_from);
  if (!model.empty()) {
    if (!models.run(ModelHost::Want{model, eo.drafter_files, start_from, args.name}, &err)) {
      end_dashboard(true);
      os::write_original_stderr("error: " + err + "\n");
      pause_owned_console();
      return 1;
    }
  } else if (remembered) {
    std::fprintf(stderr, "flyx-server: no model named: loading the last one served, %s\n", later.model.c_str());
  } else {
    std::fprintf(stderr, "flyx-server: no model named: choose one in the dashboard's model panel\n");
  }
  if (model.empty() && !remembered) {  // the GPU now, so that the page can tell which files fit it
    std::string backend;
    flyx_device_info info{};
    if (probe_device(start_from, &backend, &info, &err)) {
      srv.device_probe = {{"name", info.name}, {"mem_total", info.mem_total ? ojson(info.mem_total) : ojson(nullptr)},
                          {"mem_used", info.mem_total ? ojson(info.mem_total - info.mem_free) : ojson(nullptr)},
                          {"backend", backend}};
      std::fprintf(stderr, "flyx-server: %s (%s), %.1f GB free\n", info.name, backend.c_str(), info.mem_free / 1e9);
    }
  }
  // Connections past the workers wait for one, as many again at most: more are closed at once, not left to hang.
  http.new_task_queue = [threads = args.threads] { return new httplib::ThreadPool(threads, threads); };
  http.set_payload_max_length(flyx::apps::kHttpMaxPayloadBytes);
  http.set_read_timeout(flyx::apps::kHttpReadTimeoutSec, 0);
  http.set_write_timeout(flyx::apps::kHttpTimeoutSec, 0);
  http.set_pre_routing_handler([&srv](const httplib::Request& req, httplib::Response& res) {
    return admit_browser_request(srv, req, res) ? httplib::Server::HandlerResponse::Unhandled
                                                : httplib::Server::HandlerResponse::Handled;
  });
  add_base_routes(http, models, args.dashboard_dir);
  add_openai_chat_routes(http, &srv);
  add_anthropic_messages_routes(http, &srv);
  add_openai_responses_routes(http, &srv);
  add_telemetry_routes(http, &srv);
  add_manage_routes(http, &srv);
  add_request_log_routes(http, &srv);
  add_system_log_routes(http, &srv);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::fprintf(stderr, "flyx-server: listening on http://%s:%d, the dashboard at %s\n", host.c_str(), port,
               view.page.c_str());
  if (cfg.cors_any && cfg.api_key.empty())
    std::fprintf(stderr, "flyx-server: --cors '*' without --api-key: any web page opened in a browser that reaches this "
                         "server can use the model\n");
  if (remembered && !models.start(later, &err)) std::fprintf(stderr, "flyx-server: %s\n", err.c_str());
  if (args.browser_wanted && interactive) {
    if (os::open_url(to_open)) std::fprintf(stderr, "flyx-server: opened the dashboard in the browser\n");
    else std::fprintf(stderr, "flyx-server: no browser to open the dashboard in here (an SSH session, no display)\n");
  }
  std::atomic<bool> listener_done{false};
  std::thread serving_thread([&http, &listener_done] {
    http.listen_after_bind();
    listener_done = true;
  });
  view.listening = true;
  while (!listener_done && !view.quit && !g_stop_signal) os::sleep_ms(50);
  end_dashboard();
  models.cancel();
  if (auto sched = models.served().sched) sched->close();
  if (!listener_done) http.stop();
  serving_thread.join();
  os::write_original_stderr("flyx-server: stopped\n");
  return 0;
}

}  // namespace flyx::server
