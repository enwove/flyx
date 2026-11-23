// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The dashboard's model panel: what the server serves, loading another, and the Hub's files.
//
//   GET /v1/load          the load's state (ModelHost::status; also telemetry's "load")
//   POST /v1/load         {"model": PATH or HUB NAME, "drafters": [...], "ctx": N,
//                         "drafter": "mtp|dflash2|dv|all|none", "speculative": BOOL, "kv": "fp8|16", "name": ID}
//                         -> 202 and the state; 409 while one loads
//   DELETE /v1/load       stops a fetch under way (what came stays for the next try)
//   GET /v1/hub/search?q=TEXT&sort=trending|downloads|likes|updated|created&limit=N   the Hub's repositories with
//                         GGUF files, by name; without q, those of the family this build runs (kHubFamily)
//   GET /v1/hub/files?repo=ORG/NAME   a repository's models: its GGUFs by quantization with their sizes (and which
//                         are in the cache already), its MTP file, the DFlash2 drafters known for it, what the Hub
//                         says of it (downloads, license, its GGUFs' architecture) and whether this build runs that
//   GET /v1/hub/dir?path=PATH   a folder of this machine: the folders in it, and the model files in it and two folders
//                         down, as names the panel can load; a model file's own path lists that file.  Without
//                         path: where to start from (the model directories, the home directory, on Windows the disks)
//   POST /v1/hub/dir      {"title": TEXT, "start": PATH} asks the person at the server's machine for a folder, with
//                         the desktop's own dialog, and answers as GET does for the one chosen ({"path": null} when
//                         they cancel).  For a browser on that machine only: 403 from another, 501 where no dialog
//                         can be shown, 409 while one is open
//   GET /v1/hub/cached    the models on this machine, as names the panel can load: every snapshot of the Hugging Face
//                         cache's repositories, and the model files in --models-dir / FLYX_MODELS_DIR
//   DELETE /v1/hub/cached?name=NAME   deletes a model the cache holds (never a file of those directories), unless it
//                         is being served or a load is under way: 409
//   POST /v1/unload       the model served goes; the settings file still names it
//   POST /v1/agents/launch  {"agent": "claude|codex", "dir": PATH} opens a window of PowerShell on the server's
//                         machine, in folder dir (without it, one the person picks in the desktop's folder dialog),
//                         running that coding agent against the model served (agent_launch.h): {"launched": true,
//                         "dir"}, or {"launched": false} when the dialog is cancelled.  For a browser on that machine
//                         only: 403 from another, 409 with no model served or a dialog open, 501 where no window can
//                         be opened (not Windows, SSH, a service)
//
// All but GET /v1/load change the machine's state or reach the Hub for it: may_manage().
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

#include "apps/flyx-server/agent_launch.h"
#include "apps/flyx-server/server.h"
#include "core/interfaces/model.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace flyx::server {

namespace {

// Single desktop folder dialog guard shared across POST /v1/hub/dir and POST /v1/agents/launch.
std::atomic<bool> g_dialog_open{false};

struct DialogGuard {
  bool held = false;
  bool acquire() { return (held = !g_dialog_open.exchange(true)); }
  ~DialogGuard() {
    if (held) g_dialog_open.store(false);
  }
};

namespace fs = std::filesystem;

using os::from_path;
using os::to_path;

bool on_disk(const std::string& path) {
  std::error_code ec;
  return fs::exists(to_path(path), ec);
}

void error(httplib::Response& res, int status, const std::string& message, const char* type) {
  send_json(res, status, {{"error", {{"message", message}, {"type", type}}}});
}

// The MTP layers a GGUF holds, remembered by its path, size and time: its header is read once, not on every listing.
int mtp_layers(const std::string& path, uint64_t size, fs::file_time_type modified) {
  constexpr size_t kMaxCachedMtpEntries = 1024;
  static std::mutex mu;
  static std::map<std::string, std::tuple<uint64_t, fs::file_time_type, int>> known;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto it = known.find(path);
    if (it != known.end() && std::get<0>(it->second) == size && std::get<1>(it->second) == modified)
      return std::get<2>(it->second);
  }
  const int n = apps::gguf_mtp_layers(path);
  std::lock_guard<std::mutex> lock(mu);
  if (known.size() >= kMaxCachedMtpEntries && known.find(path) == known.end()) known.clear();
  known[path] = {size, modified, n};
  return n;
}

// One model of GET /v1/hub/cached, and what deleting it takes.
struct Listed {
  ojson json;
  std::string repo, commit;          // hub: the snapshot it is in
  std::vector<std::string> files;    // hub: its files, as paths in the repository
  std::vector<std::string> on_disk;  // its files on disk
};

ojson or_null(const std::string& s) { return s.empty() ? ojson(nullptr) : ojson(s); }

// A model file of a directory, as the panel lists it: its path is the name that loads it.
Listed dir_entry(const apps::LocalModel& m) {
  Listed l;
  const bool gguf = m.kind == "gguf";
  l.json = {{"name", m.path}, {"kind", m.kind}, {"source", "dir"}, {"repo", nullptr}, {"quant", or_null(m.quant)},
            {"file", m.file}, {"path", m.path}, {"size", m.size}, {"modified", apps::modified_unix(m.modified)},
            {"revision", nullptr}, {"current", true},
            {"mtp", gguf && mtp_layers(m.path, m.size, m.modified) > 0 ? ojson("in-file") : ojson(nullptr)}};
  if (m.parts > 1) l.json["parts"] = m.parts;
  l.on_disk = {m.path};
  return l;
}

// What GET /v1/hub/cached lists: the cache's snapshots, refs/main's first, a file under one path once (refs/main's
// copy, else the newest snapshot's); then the model directories' files.
std::vector<Listed> listing(const Server& s) {
  const std::vector<hub::Cached> snapshots = hub::cached();
  std::set<std::string> repos_with_mtp;  // the repository's MTP layer is in the cache, in any snapshot
  for (const hub::Cached& c : snapshots) {
    hub::Repo r;
    r.files = c.files;
    if (hub::mtp_file(r)) repos_with_mtp.insert(c.repo);
  }
  std::vector<Listed> out;
  std::set<std::pair<std::string, std::string>> seen;  // (repository, path)
  for (const hub::Cached& c : snapshots) {
    hub::Repo r;
    r.id = c.repo, r.revision = c.current ? "main" : c.commit, r.commit = c.commit, r.files = c.files;
    r.architectures = c.architectures, r.from_cache = true;
    const std::string suffix = c.current ? "" : "@" + c.commit;
    auto disk_of = [&](const std::string& path) { return from_path((to_path(c.dir) / to_path(path)).make_preferred()); };
    auto entry = [&](const std::string& name, const char* kind, const std::string& quant, const std::string& file,
                     const std::string& disk, uint64_t size) {
      Listed l;
      l.repo = c.repo, l.commit = c.commit;
      std::error_code ec;
      const auto modified = fs::last_write_time(to_path(disk), ec);
      ojson mtp = nullptr;
      if (std::string(kind) == "gguf") {
        if (mtp_layers(disk, size, modified) > 0) mtp = "in-file";
        else if (repos_with_mtp.count(c.repo)) mtp = "in-repo";
      }
      l.json = {{"name", name + suffix}, {"kind", kind}, {"source", "hub"}, {"repo", c.repo}, {"quant", or_null(quant)},
                {"file", file}, {"path", disk}, {"size", size}, {"modified", apps::modified_unix(modified)},
                {"revision", c.commit}, {"current", c.current}, {"mtp", mtp}};
      return l;
    };
    const std::vector<hub::GgufChoice> ggufs = hub::gguf_choices(r);
    for (const hub::GgufChoice& g : ggufs) {
      if ((g.split && !g.complete) || !seen.insert({c.repo, g.file.path}).second) continue;
      const std::string disk = disk_of(g.file.path);
      Listed l = entry(c.repo + ":" + g.quant, "gguf", g.quant, g.file.path.substr(g.file.path.rfind('/') + 1), disk,
                       g.total_size);
      if (g.split) l.json["parts"] = int(g.parts.size());
      for (const hub::RepoFile& p : g.parts) l.files.push_back(p.path), l.on_disk.push_back(disk_of(p.path));
      out.push_back(std::move(l));
    }
    // A drafter's config and safetensors, or a lone .flyx file: what the repository's name alone loads.
    std::vector<hub::RepoFile> files;
    std::string ignored;
    if (!ggufs.empty() || !hub::select(r, {c.repo}, &files, &ignored) || !seen.insert({c.repo, files[0].path}).second)
      continue;
    const bool drafter = files[0].path == "config.json";
    Listed l;
    uint64_t size = 0;
    for (const hub::RepoFile& f : files) size += f.size;
    if (drafter) {
      l = entry(c.repo, "drafter", "", c.repo.substr(c.repo.find('/') + 1), from_path(to_path(c.dir).make_preferred()),
                size);
    } else {
      const std::string base = files[0].path.substr(files[0].path.rfind('/') + 1);
      l = entry(c.repo, "flyx", hub::quant_in_name(base), base, disk_of(files[0].path), size);
    }
    for (const hub::RepoFile& f : files) l.files.push_back(f.path), l.on_disk.push_back(disk_of(f.path));
    out.push_back(std::move(l));
  }
  for (const apps::LocalModel& m : apps::scan_model_dirs(s.models_dirs)) out.push_back(dir_entry(m));
  return out;
}

// What GET /v1/hub/dir answers for a folder (or a model file) of this machine; "" asks where to start from.
bool dir_json(const Server& s, const std::string& asked, ojson* out, std::string* err) {
  constexpr int kLevels = 3;          // a model is listed from its folder, or from one or two folders above it
  constexpr size_t kMostFolders = 2000;
  ojson dirs = ojson::array(), models = ojson::array();
  std::error_code ec;
  if (asked.empty()) {
    std::set<fs::path> seen;
    auto start = [&](const std::string& dir) {
      std::error_code e2;
      const fs::path p = fs::weakly_canonical(to_path(dir), e2);
      if (!e2 && fs::is_directory(p, e2) && seen.insert(p).second)
        dirs.push_back({{"name", from_path(p)}, {"path", from_path(p)}});
    };
    for (const std::string& d : s.models_dirs) start(d);
    start(os::home_dir());
#if defined(_WIN32)
    for (char letter = 'C'; letter <= 'Z'; ++letter) start(std::string(1, letter) + ":\\");  // the disks, where models often are
#endif
    *out = {{"path", ""}, {"parent", nullptr}, {"kind", "start"}, {"dirs", dirs}, {"models", models}};
    return true;
  }
  const fs::path given = fs::absolute(to_path(asked), ec);
  const fs::path p = ec ? fs::path{} : fs::weakly_canonical(given, ec);
  if (ec || p.empty() || !fs::exists(p, ec)) return *err = asked + ": no such folder or file on this machine", false;
  const bool is_dir = fs::is_directory(p, ec);
  const fs::path base = is_dir ? p : p.parent_path();
  if (!is_dir) {
    // The file alone: the scan of its folder knows whether it is a model's (and a split model's parts as one).
    for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(base)}, 1)) {
      std::error_code e2;
      if (fs::equivalent(to_path(m.path), p, e2)) models.push_back(dir_entry(m).json);
    }
    if (models.empty()) return *err = asked + ": not a model file (.gguf, .flyx)", false;
    *out = {{"path", from_path(p)}, {"parent", from_path(base)}, {"kind", "file"}, {"dirs", dirs}, {"models", models}};
    return true;
  }
  std::vector<fs::path> subs;
  for (fs::directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = from_path(it->path().filename());
    std::error_code e2;
    if (name.empty() || name[0] == '.' || name[0] == '$' || !it->is_directory(e2)) continue;
    subs.push_back(it->path());
    if (subs.size() >= kMostFolders) break;
  }
  std::sort(subs.begin(), subs.end());
  for (const fs::path& sub : subs) dirs.push_back({{"name", from_path(sub.filename())}, {"path", from_path(sub)}});
  for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(p)}, kLevels)) models.push_back(dir_entry(m).json);
  const fs::path up = p.parent_path();
  *out = {{"path", from_path(p)}, {"parent", up.empty() || up == p ? ojson(nullptr) : ojson(from_path(up))},
          {"kind", "dir"}, {"dirs", dirs}, {"models", models}};
  return true;
}

// Whether `file` is, or is under, one of the paths the served model reads (a link followed to its blob, a hard link
// to the same file).
bool in_use(const std::string& file, const std::vector<std::string>& used) {
  std::error_code ec;
  const fs::path f = fs::weakly_canonical(to_path(file), ec);
  for (const std::string& u : used) {
    std::error_code e2;
    const fs::path p = fs::weakly_canonical(to_path(u), e2);
    if (e2 || p.empty()) continue;
    if (f == p || fs::equivalent(to_path(file), to_path(u), e2)) return true;
    const fs::path under = f.lexically_relative(p), lexical = to_path(file).lexically_relative(to_path(u));
    for (const fs::path& rel : {under, lexical})
      if (fs::is_directory(p, e2) && !rel.empty() && *rel.begin() != "..") return true;
  }
  return false;
}

ojson repo_json(const hub::Repo& r) {
  const std::string snap = hub::snapshot_dir(r) + "/";
  ojson ggufs = ojson::array();
  for (const hub::GgufChoice& c : hub::gguf_choices(r)) {
    bool cached = c.complete;
    for (const hub::RepoFile& p : c.parts) cached = cached && on_disk(snap + p.path);
    ggufs.push_back({{"name", r.id + ":" + c.quant}, {"quant", c.quant}, {"size", c.total_size}, {"split", c.split},
                     {"parts", int(c.parts.size())}, {"complete", c.complete}, {"cached", cached}});
  }
  ojson mtp = nullptr;
  if (const hub::RepoFile* f = hub::mtp_file(r))
    mtp = {{"path", f->path}, {"size", f->size}, {"cached", on_disk(snap + f->path)}};
  // A DFlash2 drafter's directory: config.json naming the architecture, and safetensors; no GGUF.
  uint64_t weights = 0;
  bool config = false;
  for (const hub::RepoFile& f : r.files) {
    config = config || f.path == "config.json";
    if (f.path.find('/') == std::string::npos && f.path.size() > 12 &&
        f.path.compare(f.path.size() - 12, 12, ".safetensors") == 0)
      weights += f.size;
  }
  // What the Hub says of it; and, where it named the GGUFs' architecture, whether this build opens that one.
  const hub::RepoInfo& i = r.info;
  const std::vector<std::string> known = interfaces::model_architectures();
  const ojson runs = i.gguf_architecture.empty()
                         ? ojson(nullptr)
                         : ojson(std::find(known.begin(), known.end(), i.gguf_architecture) != known.end());
  const ojson info = {{"downloads", i.downloads}, {"likes", i.likes}, {"updated", or_null(i.updated)},
                      {"license", or_null(i.license)}, {"pipeline", or_null(i.pipeline)}, {"gated", i.gated},
                      {"architecture", or_null(i.gguf_architecture)},
                      {"context", i.gguf_context ? ojson(i.gguf_context) : ojson(nullptr)},
                      {"parameters", i.gguf_parameters ? ojson(i.gguf_parameters) : ojson(nullptr)}};
  return {{"repo", r.id},
          {"commit", r.commit},
          {"from_cache", r.from_cache},
          {"base_models", r.base_models},
          {"ggufs", ggufs},
          {"mtp", mtp},
          {"drafter", ggufs.empty() && config && weights > 0 && hub::is_drafter(r) ? ojson({{"size", weights}}) : ojson(nullptr)},
          {"architectures", r.architectures},
          {"known_drafters", apps::known_drafters(r.base_models)},
          {"info", r.from_cache ? ojson(nullptr) : info},
          {"runs", runs},
          {"build_architectures", known}};
}

}  // namespace

void add_manage_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    const ojson body = interfaces::parse_json(q.body);
    ModelHost::Want w;
    std::string err;
    if (body.is_discarded() || !want_from_json(body, s->base, &w, &err))
      return error(r, 400, err.empty() ? "request body is not a JSON object" : err, "invalid_request_error");
    if (!s->host->start(w, &err)) return error(r, 409, err, "conflict");
    send_json(r, 202, s->host->status());
  });
  http.Delete("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    s->host->cancel();
    send_json(r, 200, s->host->status());
  });
  http.Get("/v1/hub/files", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    hub::Name name;
    if (!q.has_param("repo") || !hub::parse_name(q.get_param_value("repo"), &name))
      return error(r, 400, "repo: a Hugging Face repository, org/name", "invalid_request_error");
    hub::Repo repo;
    std::string err;
    if (!hub::describe(name.repo, name.revision, &repo, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, repo_json(repo));
  });
  http.Get("/v1/hub/search", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    // Without a name asked for, the family this build runs: most of the Hub's GGUFs are of architectures it does not.
    const std::string asked = q.get_param_value("q");
    const std::string query = asked.empty() ? apps::kHubFamily : asked;
    std::vector<hub::Found> found;
    std::string err;
    if (!hub::search(query, q.get_param_value("sort"), query_int(q, "limit", 40, 1, 100), &found, &err))
      return error(r, 502, err, "hub_error");
    std::set<std::string> here;  // repositories with a snapshot in the cache
    for (const hub::Cached& c : hub::cached()) here.insert(c.repo);
    ojson models = ojson::array();
    for (const hub::Found& f : found)
      models.push_back({{"id", f.id}, {"downloads", f.downloads}, {"likes", f.likes}, {"updated", or_null(f.updated)},
                        {"created", or_null(f.created)}, {"pipeline", or_null(f.pipeline)}, {"gated", f.gated},
                        {"cached", here.count(f.id) > 0}});
    send_json(r, 200, {{"query", query}, {"family", asked.empty()}, {"endpoint", hub::endpoint()}, {"models", models}});
  });
  http.Get("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    ojson out;
    std::string err;
    if (!dir_json(*s, q.get_param_value("path"), &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Post("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    // The dialog opens on this machine's desktop: of use, and polite, only to someone sitting at it.
    if (!from_this_machine(q))
      return error(r, 403, "the folder dialog opens on the server's machine: give the folder's path instead", "permission_error");
    DialogGuard dialog;
    if (!dialog.acquire()) return error(r, 409, "a folder dialog is already open on the server's machine", "conflict");
    const ojson body = interfaces::parse_json(q.body);
    auto text = [&](const char* key) {
      return body.is_object() && body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    };
    const std::string title = text("title");
    std::string path, err;
    const os::Picked picked = os::pick_folder(title.empty() ? "Choose a folder of model files" : title, text("start"), &path, &err);
    if (picked == os::Picked::kUnavailable) return error(r, 501, err, "not_supported");
    if (picked == os::Picked::kCancelled) return send_json(r, 200, {{"path", nullptr}});
    ojson out;
    if (!dir_json(*s, path, &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Get("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    ojson models = ojson::array();
    for (Listed& l : listing(*s)) models.push_back(std::move(l.json));
    send_json(r, 200, {{"cache", hub::cache_dir()}, {"cache_from", hub::cache_source()}, {"dirs", s->models_dirs},
                       {"models", models}});
  });
  http.Delete("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    const std::string name = q.get_param_value("name");
    if (name.empty()) return error(r, 400, "name: a model as GET /v1/hub/cached lists it", "invalid_request_error");
    std::vector<Listed> all = listing(*s);
    const auto it = std::find_if(all.begin(), all.end(), [&](const Listed& l) { return l.json["name"] == name; });
    if (it == all.end()) return error(r, 400, name + " is not among the cache's models", "invalid_request_error");
    if (it->json["source"] == "dir")
      return error(r, 400, name + " is a file of a model directory of yours: flyx deletes only from the Hugging Face cache",
                   "invalid_request_error");
    bool served = false, ok = false;
    uint64_t freed = 0;
    std::string err;
    const bool idle = s->host->with_files_in_use(
        [&](const std::vector<std::string>& used) {
          for (const std::string& f : it->on_disk) served = served || in_use(f, used);
          if (!served) ok = hub::remove_cached(it->repo, it->commit, it->files, &freed, &err);
        },
        &err);
    if (!idle) return error(r, 409, err, "conflict");
    if (served) return error(r, 409, name + " is being served: load another model, or unload it, first", "conflict");
    if (!ok) return error(r, 500, err, "server_error");
    std::fprintf(stderr, "flyx-server: deleted %s from the cache (%.1f GB)\n", name.c_str(), double(freed) / 1e9);
    send_json(r, 200, {{"deleted", name}, {"freed", freed}});
  });
  http.Post("/v1/unload", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "loading models")) return;
    std::string err;
    if (!s->host->unload(&err)) return error(r, 409, err, "conflict");
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/agents/launch", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_manage(*s, q, r, "starting coding agents")) return;
    // The window opens on this machine's desktop, as the folder dialog does: for someone sitting at it.
    if (!from_this_machine(q))
      return error(r, 403, "the agent starts in a window on the server's machine: run its command where you are",
                   "permission_error");
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    auto text = [&](const char* key) {
      return body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    };
    const std::string agent = text("agent");
    const auto& agents = launch_agents();
    if (std::find(agents.begin(), agents.end(), agent) == agents.end())
      return error(r, 400, "agent must be one of: claude, codex", "invalid_request_error");
    if (!os::shell_window_supported())
      return error(r, 501, "opening a terminal window is supported on Windows only", "not_supported");
    const ModelHost::Served served = s->host->served();
    if (!served.sched) return error(r, 409, "no model is served: load one first", "conflict");
    // The server as this request reached it: a loopback address, the agent being on the same machine.
    std::string host = q.local_addr.empty() ? "127.0.0.1" : q.local_addr;
    if (host.find(':') != std::string::npos) host = "[" + host + "]";
    const AgentTarget target{"http://" + host + ":" + std::to_string(q.local_port), served.name,
                             served.sched->max_ctx(), !s->cfg.api_key.empty()};
    const std::string command = agent_powershell(agent, target);
    if (command.empty())
      return error(r, 409, "the model id has one of \" & | < > ^ % in it, which cannot be passed to Codex from "
                           "PowerShell: run the command on the page", "conflict");
    std::string dir = text("dir"), err;
    if (dir.empty()) {
      DialogGuard dialog;
      if (!dialog.acquire()) return error(r, 409, "a folder dialog is already open on the server's machine", "conflict");
      const os::Picked picked = os::pick_folder(
          agent == "claude" ? "Choose the project folder for Claude Code" : "Choose the project folder for Codex", "", &dir,
          &err);
      if (picked == os::Picked::kUnavailable) return error(r, 501, err, "not_supported");
      if (picked == os::Picked::kCancelled) return send_json(r, 200, {{"launched", false}});
    }
    std::error_code ec;
    if (!fs::is_directory(to_path(dir), ec)) return error(r, 404, "no such folder: " + dir, "not_found");
    std::vector<std::pair<std::string, std::string>> env;
    if (target.keyed) env.emplace_back("FLYX_KEY", s->cfg.api_key);
    if (!os::open_shell_window(dir, command, env, &err)) return error(r, 501, err, "not_supported");
    send_json(r, 200, {{"launched", true}, {"dir", dir}});
  });
}

}  // namespace flyx::server
