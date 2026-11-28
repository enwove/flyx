// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The dashboard's model panel: what the server serves, loading another, and the Hub's files.
//
//   GET /v1/load          the load's state (ModelHost::status; also telemetry's "load")
//   POST /v1/load         {"model": PATH or HUB NAME, "drafters": [...], "lora": DIR or HUB NAME, "ctx": N,
//                         "drafter": "mtp|dflash2|dflash|dspark|dspine|eagle4|dv|all|none", "speculative": BOOL,
//                         "kv": "fp8|16", "name": ID}
//                         -> 202 and the state; 409 while one loads
//   DELETE /v1/load       stops a fetch under way (what came stays for the next try)
//   GET /v1/hub/config    current Hugging Face Hub endpoint, default endpoint, source ("setting", "HF_ENDPOINT",
//                         "default"), and offline flag
//   POST /v1/hub/config   {"endpoint": URL or ""} sets or resets the Hub endpoint (persisted to serve.json)
//   GET /v1/hub/search?q=TEXT&sort=trending|downloads|likes|updated|created&limit=N&cursor=C&arch=A   the Hub's
//                         repositories with GGUF files; without q, filtered to the architectures this build runs
//                         (optionally narrowed to arch=A), paginated via next_cursor
//   GET /v1/hub/files?repo=ORG/NAME   a repository's models: its GGUFs by quantization with their sizes (and which
//                         are in the cache already), its MTP file, the drafters verified for it, what the Hub
//                         says of it (downloads, license, its GGUFs' architecture) and whether this build runs that
//   GET /v1/hub/drafter?name=NAME   the drafter a path or a Hub name holds, as its own fields say: {"name", "kind"};
//                         400 for neither, 404 for a Hub name the Hub does not have
//   GET /v1/hub/dir?path=PATH   a folder of this machine: the folders in it, and the model files in it and two folders
//                         down, as names the panel can load; a model file's own path lists that file.  Without
//                         path: where to start from (the model directories, the home directory, on Windows the disks)
//   POST /v1/hub/dir      {"title": TEXT, "start": PATH} asks the person at the server's machine for a folder, with
//                         the desktop's own dialog, and answers as GET does for the one chosen ({"path": null} when
//                         they cancel).  For a browser on that machine only: 403 from another, 501 where no dialog
//                         can be shown, 409 while one is open
//   GET /v1/hub/cached    the models on this machine, as names the panel can load: every snapshot of the Hugging Face
//                         cache's repositories, and the model files in --models-dir / FLYX_MODELS_DIR; a model's
//                         verified drafters (known_drafters), a drafter's kind (drafter_kind); LoRA adapters
//                         (kind "lora") among them, named as "lora" takes them
//   DELETE /v1/hub/cached?name=NAME   deletes a model the cache holds (never a file of those directories), unless it
//                         is being served or a load is under way: 409
//   GET /v1/hub/loras?base=ORG/NAME | model=NAME&cursor=C   the LoRA adapters the Hub files as made for a base model
//                         (its model cards' base_model): for base, or for the model card's base of model (a Hub name
//                         or a file of the cache; the base the page loads with), the most downloaded first
//   GET /v1/hub/lora?repo=ORG/NAME[@REV]   a repository's adapters (each folder with an adapter_config.json), their
//                         rank, scale and linears, and whether --lora takes each (null "why": yes): its config (at
//                         most 64 KiB) and its safetensors header (two ranged requests) checked as a load checks them,
//                         read into memory (the cache is left as it was)
//   POST /v1/unload       the model served goes; the settings file still names it
//   POST /v1/agents/launch  {"agent": "claude|codex", "dir": PATH} opens a window of PowerShell on the server's
//                         machine, in folder dir (without it, one the person picks in the desktop's folder dialog),
//                         running that coding agent against the model served (agent_launch.h), with an optional
//                         permission_mode for Claude Code: {"launched": true,
//                         "dir"}, or {"launched": false} when the dialog is cancelled.  For a browser on that machine
//                         only: 403 from another, 409 with no model served or a dialog open, 501 where no window can
//                         be opened (not Windows, SSH, a service)
//
// GET /v1/load is gated by Access::kDashboardRead; all other routes change machine state or reach the Hub
// and require Access::kManage (or Access::kLocalOnly for desktop dialogs / shell launch / closing apps).
#include <algorithm>
#include <atomic>
#include <filesystem>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

#include "apps/flyx-server/agent_launch.h"
#include "apps/flyx-server/drafter_speeds.h"
#include "apps/flyx-server/gpu_apps.h"
#include "apps/flyx-server/server.h"
#include "core/interfaces/model.h"
#include "core/modelfile/modelfile.h"
#include "core/modelfile/readers.h"
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

void hub_error(httplib::Response& res, const hub::HubOutcome& out, const std::string& err) {
  switch (out.error) {
    case hub::HubError::kInvalidName: return error(res, 400, err, "invalid_request_error");
    case hub::HubError::kNotFound: return error(res, 404, err, "not_found");
    case hub::HubError::kUnauthorized: return error(res, 403, err, "permission_error");
    case hub::HubError::kRateLimited:
      if (!out.retry_after.empty()) res.set_header("Retry-After", out.retry_after);
      return error(res, 429, err, "rate_limit_error");
    case hub::HubError::kUnreachable:
    case hub::HubError::kBadAnswer:
    case hub::HubError::kNone: return error(res, 502, err, "hub_error");
  }
}

// One model of GET /v1/hub/cached, and what deleting it takes.
struct Listed {
  ojson json;
  std::string repo, commit;          // hub: the snapshot it is in
  std::vector<std::string> files;    // hub: its files, as paths in the repository
  std::vector<std::string> on_disk;  // its files on disk
  bool vision = true;                // a model of an architecture this build reads a projector beside
};

ojson or_null(const std::string& s) { return s.empty() ? ojson(nullptr) : ojson(s); }

// A model file of a directory, as the panel lists it: its path is the name that loads it.
Listed dir_entry(const apps::LocalModel& m, const apps::PathAllowFn& allow = {}) {
  Listed l;
  const bool gguf = m.kind == "gguf";
  l.json = {{"name", m.path}, {"kind", m.kind}, {"source", "dir"}, {"repo", nullptr}, {"quant", or_null(m.quant)},
            {"file", m.file}, {"path", m.path}, {"root", or_null(m.root)}, {"size", m.size},
            {"modified", apps::modified_unix(m.modified)}, {"revision", nullptr}, {"current", true},
            {"mtp", gguf && m.mtp_layers > 0 ? ojson("in-file") : ojson(nullptr)}};
  if (m.kind == "gguf" || m.kind == "flyx") {  // what its file says it is and comes from
    const flyx::ModelOrigin o = gguf ? flyx::ModelOrigin{"", apps::gguf_header(m.path, allow).base_models}
                                     : flyx::model_origin(m.path, allow);
    l.json["known_drafters"] = known_drafters_json(m.path, apps::known_drafters(o.repo, o.base_models, false, allow));
  }
  if (m.kind == "drafter" && !m.drafter_kind.empty()) l.json["drafter_kind"] = m.drafter_kind;
  if (m.parts > 1) l.json["parts"] = m.parts;
  l.on_disk = {m.path};
  return l;
}

// What GET /v1/hub/cached lists: the cache's snapshots, refs/main's first, a file under one path once (refs/main's
// copy, else the newest snapshot's); then the model directories' files.
std::vector<Listed> listing(const Server& s, const httplib::Request* req = nullptr) {
  const apps::PathAllowFn allow = req ? remote_path_policy(s, *req) : apps::PathAllowFn{};
  const std::vector<hub::Cached> snapshots = hub::cached(allow);
  std::set<std::string> repos_with_mtp;  // the repository's MTP layer is in the cache, in any snapshot
  std::set<std::pair<std::string, std::string>> snaps_with_mmproj;  // (repo, commit) holding a vision projector
  std::set<std::pair<std::string, std::string>> snaps_with_dflash2;
  std::vector<Listed> out;
  std::set<std::pair<std::string, std::string>> seen;  // (repository, path)
  for (const hub::Cached& c : snapshots) {
    hub::Repo r;
    r.id = c.repo, r.revision = c.current ? "main" : c.commit, r.commit = c.commit, r.files = c.files;
    r.architectures = c.architectures, r.from_cache = true;
    const std::string suffix = c.current ? "" : "@" + c.commit;
    auto disk_of = [&](const std::string& path) { return from_path((to_path(c.dir) / to_path(path)).make_preferred()); };
    auto entry = [&](const std::string& name, const std::string& kind, const std::string& quant,
                     const std::string& file, const std::string& disk, uint64_t size, fs::file_time_type modified,
                     int mtp_count, const std::string& drafter_kind) {
      Listed l;
      l.repo = c.repo, l.commit = c.commit;
      l.json = {{"name", name + suffix}, {"kind", kind}, {"source", "hub"}, {"repo", c.repo}, {"quant", or_null(quant)},
                {"file", file}, {"path", disk}, {"size", size}, {"modified", apps::modified_unix(modified)},
                {"revision", c.commit}, {"current", c.current}, {"mtp", kind == "gguf" && mtp_count > 0 ? ojson("in-file") : ojson(nullptr)},
                {"mmproj", nullptr}, {"dflash2", nullptr}};
      if (kind == "drafter" && !drafter_kind.empty()) l.json["drafter_kind"] = drafter_kind;
      if (kind == "gguf" || kind == "flyx")  // what its file says it comes from
        l.json["known_drafters"] = known_drafters_json(
            name + suffix,
            apps::known_drafters(c.repo, kind == "gguf" ? apps::gguf_header(disk, allow).base_models
                                                        : flyx::model_origin(disk, allow).base_models,
                                 false, allow));
      return l;
    };
    // LoRA adapters (PEFT) with their weights here, by the name that fetches them (org/name, org/name:FOLDER).
    for (const hub::AdapterFolder& a : hub::adapter_folders(c.repo, c.files)) {
      if (!a.weights || !seen.insert({c.repo, a.config->path}).second) continue;
      const std::string last = a.folder.substr(a.folder.rfind('/') == std::string::npos ? 0 : a.folder.rfind('/') + 1);
      std::error_code ec;
      const auto modified = fs::last_write_time(to_path(disk_of(a.weights->path)), ec);
      Listed l = entry(a.name, "lora", "", last.empty() ? c.repo.substr(c.repo.find('/') + 1) : last,
                       from_path((to_path(c.dir) / to_path(a.folder)).make_preferred()), a.weights->size, modified, -1,
                       "");
      for (const hub::RepoFile* g : {a.config, a.weights}) l.files.push_back(g->path), l.on_disk.push_back(disk_of(g->path));
      out.push_back(std::move(l));
    }
    const auto ggufs = apps::hub_ggufs(r, allow);
    for (const auto& item : ggufs) {
      if (!item.cached || item.cls.kind.empty()) continue;
      const auto& g = item.choice;
      const auto& cls = item.cls;
      if (cls.kind == "mmproj") snaps_with_mmproj.insert({c.repo, c.commit});
      if (cls.drafter_kind == "dflash2") snaps_with_dflash2.insert({c.repo, c.commit});
      if (cls.drafter_kind == "mtp") repos_with_mtp.insert(c.repo);
      if (!seen.insert({c.repo, g.file.path}).second) continue;
      Listed l = entry(c.repo + ":" + item.selector, cls.kind, g.quant, g.file.path.substr(g.file.path.rfind('/') + 1),
                       item.disk, g.total_size, item.modified, cls.mtp_layers, cls.drafter_kind);
      l.vision = apps::reads_vision(cls.arch);
      if (g.split) l.json["parts"] = int(g.parts.size());
      for (const hub::RepoFile& p : g.parts) l.files.push_back(p.path), l.on_disk.push_back(disk_of(p.path));
      out.push_back(std::move(l));
    }
    // A drafter's config and safetensors, or a lone .flyx file: what the repository's name alone loads.
    std::vector<hub::RepoFile> files;
    std::string ignored;
    if (!hub::select(r, {c.repo}, &files, &ignored) || hub::model_gguf_name(files[0].path) ||
        !seen.insert({c.repo, files[0].path}).second)
      continue;
    const bool drafter = files[0].path == "config.json";
    Listed l;
    uint64_t size = 0;
    for (const hub::RepoFile& f : files) size += f.size;
    std::error_code ec;
    if (drafter) {
      const std::string disk = from_path(to_path(c.dir).make_preferred());
      const auto modified = fs::last_write_time(to_path(disk), ec);
      // its config's kind ("dflash2", "dflash", "dspark", "dspine", "eagle4")
      l = entry(c.repo, "drafter", "", c.repo.substr(c.repo.find('/') + 1), disk, size, modified, -1,
                flyx::drafter_source_kind(c.dir, allow));
    } else {
      const std::string base = files[0].path.substr(files[0].path.rfind('/') + 1);
      const std::string disk = disk_of(files[0].path);
      const auto modified = fs::last_write_time(to_path(disk), ec);
      l = entry(c.repo, "flyx", hub::quant_in_name(base), base, disk, size, modified, -1, "");
    }
    for (const hub::RepoFile& f : files) l.files.push_back(f.path), l.on_disk.push_back(disk_of(f.path));
    out.push_back(std::move(l));
  }
  for (Listed& l : out) {
    if (l.json["kind"] != "gguf") continue;
    if (l.json["mtp"].is_null() && repos_with_mtp.count(l.repo)) l.json["mtp"] = "in-repo";
    if (l.vision && snaps_with_mmproj.count({l.repo, l.commit})) l.json["mmproj"] = "in-repo";
    if (snaps_with_dflash2.count({l.repo, l.commit})) l.json["dflash2"] = "in-repo";
  }
  for (const apps::LocalModel& m : apps::scan_model_dirs(s.models_dirs, 6, allow)) out.push_back(dir_entry(m, allow));
  return out;
}

// What GET /v1/hub/dir answers for a folder (or a model file) of this machine; "" asks where to start from.
bool dir_json(const Server& s, const httplib::Request& req, const std::string& asked, ojson* out, std::string* err) {
  constexpr int kLevels = 3;          // a model is listed from its folder, or from one or two folders above it
  constexpr size_t kMostFolders = 2000;
  ojson dirs = ojson::array(), models = ojson::array();
  const apps::PathAllowFn allow = remote_path_policy(s, req);
  std::error_code ec;
  if (asked.empty()) {
    std::set<fs::path> seen;
    auto start = [&](const std::string& dir) {
      if (dir.empty() || (allow && !allow(dir))) return;
      std::error_code e2;
      const fs::path p = fs::weakly_canonical(to_path(apps::expand_user_path(dir)), e2);
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
  const fs::path given = fs::absolute(to_path(apps::expand_user_path(asked)), ec);
  const fs::path p = ec ? fs::path{} : fs::weakly_canonical(given, ec);
  if (ec || p.empty() || !fs::exists(p, ec)) return *err = asked + ": no such folder or file on this machine", false;
  const bool is_dir = fs::is_directory(p, ec);
  const fs::path base = is_dir ? p : p.parent_path();
  if (!is_dir) {
    // The file alone: the scan of its folder knows whether it is a model's (and a split model's parts as one).
    // Pre-authorize every sibling before inspection so remote callers cannot trigger SMB lookups via sibling links.
    for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(base)}, 1, allow)) {
      std::error_code e2;
      if (fs::equivalent(to_path(m.path), p, e2)) models.push_back(dir_entry(m, allow).json);
    }
    if (models.empty()) return *err = asked + ": not a model file (.gguf, .flyx)", false;
    *out = {{"path", from_path(p)}, {"parent", from_path(base)}, {"kind", "file"}, {"dirs", dirs}, {"models", models}};
    return true;
  }
  std::vector<fs::path> subs;
  for (fs::directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
       it.increment(ec)) {
    const std::string name = from_path(it->path().filename());
    if (name.empty() || name[0] == '.' || name[0] == '$') continue;
    const std::string child = from_path(it->path());
    if (allow && !allow(child)) continue;
    std::error_code e2;
    if (!it->is_directory(e2)) continue;
    subs.push_back(it->path());
    if (subs.size() >= kMostFolders) break;
  }
  std::sort(subs.begin(), subs.end());
  for (const fs::path& sub : subs) dirs.push_back({{"name", from_path(sub.filename())}, {"path", from_path(sub)}});
  for (const apps::LocalModel& m : apps::scan_model_dirs({from_path(p)}, kLevels, allow))
    models.push_back(dir_entry(m, allow).json);
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

// The repository a model name or a file of the Hub's cache is from ("" for another file).
std::string repo_of(const std::string& model) {
  hub::Name n;
  if (!on_disk(model) && hub::parse_name(model, &n)) return n.repo;
  for (const fs::path& part : os::to_path(model)) {  // <cache>/models--org--name/snapshots/<commit>/...
    const std::string p = os::from_path(part);
    if (p.rfind("models--", 0) != 0) continue;
    std::string repo = p.substr(8);
    const size_t dash = repo.find("--");
    if (dash == std::string::npos) return "";
    return repo.replace(dash, 2, "/");
  }
  return "";
}

// What adapters for `model` are made for: its repository's model card's first base_model (a GGUF repository's: the
// model it quantizes), or the repository itself when it names none.  *from: "model card" or "model".
std::string base_of(const std::string& model, std::string* from, const apps::PathAllowFn& allow = {}) {
  const std::string repo = repo_of(model);
  if (repo.empty()) return "";
  std::string rev = "main";
  hub::Name n;
  if (!on_disk(model) && hub::parse_name(model, &n) && !n.revision.empty()) rev = n.revision;
  hub::Repo r;
  std::string err;
  if (hub::describe(repo, rev, &r, &err, nullptr, allow) && !r.base_models.empty()) {
    hub::Name b;
    if (hub::parse_name(r.base_models.front(), &b)) return *from = "model card", b.repo;
  }
  return *from = "model", repo;
}

ojson repo_json(const hub::Repo& r, const apps::PathAllowFn& allow = {}) {
  ojson ggufs = ojson::array();
  ojson companions = ojson::array();
  const auto choices = apps::hub_ggufs(r, allow);
  const std::string rev_suffix = (!r.revision.empty() && r.revision != "main") ? "@" + r.revision : "";
  // The GGUFs' architecture: what the Hub says, else the first cached model's header.  No projector is offered for
  // one this build reads none for (Inkling-2).
  std::string repo_arch = r.info.gguf_architecture;
  for (const auto& item : choices)
    if (repo_arch.empty() && item.cls.kind == "gguf" && !item.cls.arch.empty()) repo_arch = item.cls.arch;
  const bool repo_vision = apps::reads_vision(repo_arch);
  const auto* preferred_mtp = apps::preferred_companion(choices, "mtp");
  const auto* preferred_mm = repo_vision ? apps::preferred_companion(choices, "mmproj") : nullptr;
  const auto* preferred_dspark = apps::preferred_companion(choices, "dspark");
  ojson mtp = nullptr, mmproj = nullptr;
  bool has_dflash = false, has_dflash_arch = false;  // a DFlash2 drafter; any arch=dflash drafter (DFlash2, DSpark)
  for (const auto& item : choices) {
    const auto& c = item.choice;
    if (item.cls.kind.empty() || (item.cls.kind == "mmproj" && !repo_vision)) continue;
    if (item.cls.kind == "gguf") {
      ggufs.push_back({{"name", r.id + ":" + item.selector + rev_suffix}, {"quant", c.quant}, {"size", c.total_size},
                       {"split", c.split}, {"parts", int(c.parts.size())}, {"complete", c.complete},
                       {"cached", item.cached}});
      continue;
    }
    if (!c.complete) continue;
    const bool is_dflash = item.cls.drafter_kind == "dflash2";
    const bool is_default =
        &item == preferred_mtp || &item == preferred_mm || &item == preferred_dspark || (is_dflash && !has_dflash);
    if (is_dflash) has_dflash = true;
    if (item.cls.kind == "drafter" && item.cls.drafter_kind != "mtp") has_dflash_arch = true;
    const auto& path = c.file.path;
    const auto base = path.substr(path.rfind('/') + 1);
    companions.push_back({{"name", r.id + ":" + base + rev_suffix}, {"path", path}, {"file", base},
                          {"kind", item.cls.drafter_kind.empty() ? item.cls.kind : item.cls.drafter_kind},
                          {"drafter_kind", item.cls.drafter_kind},
                          {"size", c.total_size}, {"cached", item.cached}, {"disk", item.cached ? ojson(item.disk) : ojson(nullptr)},
                          {"default", is_default}});
    if (&item == preferred_mtp || &item == preferred_mm)
      (&item == preferred_mtp ? mtp : mmproj) = {{"path", path}, {"size", c.total_size}, {"cached", item.cached}};
  }
  // A block drafter's directory: config.json naming the architecture, and safetensors; no GGUF.
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
  const bool dflash_gguf_repo = ggufs.empty() && has_dflash_arch;
  const ojson runs = dflash_gguf_repo
                         ? ojson(true)
                         : i.gguf_architecture.empty()
                               ? ojson(nullptr)
                               : ojson(std::find(known.begin(), known.end(), i.gguf_architecture) != known.end());
  const ojson info = {{"downloads", i.downloads}, {"likes", i.likes}, {"updated", or_null(i.updated)},
                      {"license", or_null(i.license)}, {"pipeline", or_null(i.pipeline)}, {"gated", i.gated},
                      {"architecture", or_null(i.gguf_architecture)},
                      {"context", i.gguf_context ? ojson(i.gguf_context) : ojson(nullptr)},
                      {"parameters", i.gguf_parameters ? ojson(i.gguf_parameters) : ojson(nullptr)}};
  return {{"repo", r.id},
          {"revision", r.revision},
          {"commit", r.commit},
          {"from_cache", r.from_cache},
          {"base_models", r.base_models},
          {"ggufs", ggufs},
          {"mtp", mtp},
          {"mmproj", mmproj},
          {"companions", companions},
          {"drafter", ggufs.empty() && config && weights > 0 && hub::is_drafter(r)
                          ? ojson({{"size", weights}, {"kind", hub::drafter_kind(r)}})
                          : ojson(nullptr)},
          {"drafter_repo", dflash_gguf_repo},
          {"architectures", r.architectures},
          {"known_drafters", known_drafters_json(r.id, apps::known_drafters(r.id, r.base_models, true, allow))},
          {"info", r.from_cache ? ojson(nullptr) : info},
          {"runs", runs},
          {"build_architectures", known},
          {"endpoint", hub::endpoint()}};
}

}  // namespace

void add_manage_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_auth(*s, q, r)) return;
    if (!require_access(*s, q, r, Access::kDashboardRead, "reading load status")) return;
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const ojson body = interfaces::parse_json(q.body);
    ModelHost::Want w;
    std::string err;
    if (body.is_discarded() || !want_from_json(body, s->base, &w, &err))
      return error(r, 400, err.empty() ? "request body is not a JSON object" : err, "invalid_request_error");
    if (!require_local_path(*s, q, r, w.model) || !require_local_path(*s, q, r, w.eo.mmproj_file) ||
        !require_local_path(*s, q, r, w.eo.lora_file))
      return;
    for (const std::string& d : w.drafters)
      if (!require_local_path(*s, q, r, d)) return;
    w.eo.allow_path = remote_path_policy(*s, q);
    if (!s->host->start(w, &err)) return error(r, 409, err, "conflict");
    send_json(r, 202, s->host->status());
  });
  http.Delete("/v1/load", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    s->host->cancel();
    send_json(r, 200, s->host->status());
  });
  auto hub_config_json = []() -> ojson {
    const auto endpoint = hub::resolve_endpoint();
    return {{"endpoint", endpoint.value},
            {"default_endpoint", hub::kDefaultEndpoint},
            {"source", endpoint.source},
            {"offline", hub::offline()}};
  };
  http.Get("/v1/hub/config", [s, hub_config_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "configuring Hugging Face")) return;
    send_json(r, 200, hub_config_json());
  });
  http.Post("/v1/hub/config", [s, hub_config_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kLocalOrKeyed, "configuring Hugging Face")) return;
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    std::string raw;
    if (body.contains("endpoint") && !body["endpoint"].is_null()) {
      if (!body["endpoint"].is_string())
        return error(r, 400, "\"endpoint\": a URL such as https://hf-mirror.com, or empty to reset", "invalid_request_error");
      raw = body["endpoint"].get<std::string>();
    }
    std::string normalized, err;
    if (!hub::normalize_endpoint(raw, &normalized, &err)) return error(r, 400, err, "invalid_request_error");
    // Clear the saved override when emptied, or when explicitly set back to huggingface.co with no HF_ENDPOINT env.
    const std::string saved =
        normalized.empty() || (normalized == hub::kDefaultEndpoint && os::getenv_str(env::kHfEndpoint).empty())
            ? std::string()
            : normalized;
    if (!hub::set_endpoint(saved, &err)) return error(r, 400, err, "invalid_request_error");
    if (!ModelHost::save_hub_endpoint(saved, &err)) return error(r, 500, err, "server_error");
    std::fprintf(stderr, "flyx-server: Hugging Face endpoint set to %s\n", hub::endpoint().c_str());
    send_json(r, 200, hub_config_json());
  });
  http.Get("/v1/hub/files", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    hub::Name name;
    if (!q.has_param("repo") || !hub::parse_name(q.get_param_value("repo"), &name))
      return error(r, 400, "repo: a Hugging Face repository, org/name", "invalid_request_error");
    const auto allow = remote_path_policy(*s, q);
    hub::Repo repo;
    hub::HubOutcome out;
    std::string err;
    if (!hub::describe(name.repo, name.revision, &repo, &err, &out, allow)) return hub_error(r, out, err);
    send_json(r, 200, repo_json(repo, allow));
  });
  http.Get("/v1/hub/loras", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const auto allow = remote_path_policy(*s, q);
    std::string base = q.get_param_value("base"), from = "asked";
    if (base.empty() && q.has_param("model")) {
      const std::string model = q.get_param_value("model");
      if (!require_local_path(*s, q, r, model)) return;
      base = base_of(model, &from, allow);
    }
    if (base.empty()) return send_json(r, 200, {{"base", nullptr}, {"from", nullptr}, {"models", ojson::array()},
                                                {"next_cursor", nullptr}, {"endpoint", hub::endpoint()}});
    hub::Name b;
    if (!hub::parse_name(base, &b)) return error(r, 400, "base: a Hugging Face repository, org/name", "invalid_request_error");
    std::vector<hub::Found> found;
    hub::HubOutcome out;
    std::string next, err;
    if (!hub::adapters_of(b.repo, query_int(q, "limit", 30, 1, 100), &found, &err, q.get_param_value("cursor"), &next,
                          &out))
      return hub_error(r, out, err);
    std::set<std::string> here;  // repositories with an adapter's weights in the cache
    for (const hub::Cached& c : hub::cached(allow))
      for (const hub::RepoFile& f : c.files)
        if (f.path.size() >= 25 && f.path.compare(f.path.size() - 25, 25, "adapter_model.safetensors") == 0) here.insert(c.repo);
    ojson models = ojson::array();
    for (const hub::Found& f : found)
      models.push_back({{"id", f.id}, {"downloads", f.downloads}, {"likes", f.likes}, {"updated", or_null(f.updated)},
                        {"gated", f.gated}, {"cached", here.count(f.id) > 0}});
    send_json(r, 200, {{"base", b.repo}, {"from", from}, {"models", models}, {"next_cursor", or_null(next)},
                       {"endpoint", hub::endpoint()}});
  });
  http.Get("/v1/hub/lora", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    hub::Name name;
    if (!q.has_param("repo") || !hub::parse_name(q.get_param_value("repo"), &name) || !name.quant.empty())
      return error(r, 400, "repo: a Hugging Face repository, org/name[@REV]", "invalid_request_error");
    const auto allow = remote_path_policy(*s, q);
    hub::Repo repo;
    hub::HubOutcome out;
    std::string err;
    if (!hub::describe(name.repo, name.revision, &repo, &err, &out, allow)) return hub_error(r, out, err);
    // Each adapter (at most 32), read where it is: its config (at most 64 KiB; a few hundred bytes as PEFT writes it)
    // and its weights' header, into memory alone, kReadsAtOnce adapters at a time (a request's threads and the Hub's
    // connections bounded).  The Hub's files are anyone's: every string of them is checked before it is shown
    // (modelfile::peft_config_ok, peft_header_ok).
    constexpr size_t kReadsAtOnce = 4;
    std::vector<hub::AdapterFolder> found = hub::adapter_folders(repo.id, repo.files);
    if (found.size() > 32) found.resize(32);
    struct Read {
      modelfile::PeftConfig config;
      std::string why;
      std::vector<std::string> linears;
      int layers = 0;
    };
    auto read = [&repo, &allow](const hub::AdapterFolder& a) {
      Read out;
      std::string text, header;
      if (!hub::read_text(repo, *a.config, 64u << 10, &text, &out.why, allow) ||
          !modelfile::peft_config_ok(text, &out.config, &out.why))
        return out;
      if (!a.weights) {
        out.why = a.pickle ? "adapter_model.bin (a pickle) is not read; only safetensors"
                           : "no adapter_model.safetensors beside its adapter_config.json";
        return out;
      }
      // its tensors as --lora reads them (names, dtypes, rank), from the file's header alone
      if (hub::safetensors_header(repo, *a.weights, &header, &out.why, allow))
        modelfile::peft_header_ok(header, out.config.rank, &out.why, &out.linears, &out.layers);
      return out;
    };
    std::vector<Read> got(found.size());
    for (size_t first = 0; first < found.size(); first += kReadsAtOnce) {
      std::vector<std::future<Read>> reads;
      for (size_t i = first; i < std::min(found.size(), first + kReadsAtOnce); ++i)
        reads.push_back(std::async(std::launch::async, read, std::cref(found[i])));
      for (size_t i = 0; i < reads.size(); ++i) got[first + i] = reads[i].get();
    }
    ojson adapters = ojson::array();
    for (size_t i = 0; i < found.size(); ++i) {
      const hub::AdapterFolder& a = found[i];
      const modelfile::PeftConfig& c = got[i].config;
      const bool ok = c.rank > 0;
      adapters.push_back({{"folder", a.folder}, {"name", a.name + (name.revision == "main" ? "" : "@" + name.revision)},
                          {"rank", ok ? ojson(c.rank) : ojson()}, {"scale", ok ? ojson(c.scale) : ojson()},
                          {"base", or_null(c.base)}, {"targets", c.targets}, {"linears", got[i].linears},
                          {"layers", got[i].layers}, {"size", a.weights ? ojson(a.weights->size) : ojson()},
                          {"why", or_null(got[i].why)}});
    }
    send_json(r, 200, {{"repo", repo.id}, {"revision", repo.revision}, {"adapters", adapters}});
  });
  http.Get("/v1/hub/drafter", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const std::string name = q.get_param_value("name");
    if (!require_local_path(*s, q, r, name)) return;
    const auto allow = remote_path_policy(*s, q);
    std::error_code ec;
    hub::Name hn;
    std::string kind;
    if (!name.empty() && fs::exists(to_path(name), ec)) {
      kind = apps::drafter_kind(name, allow);
    } else if (hub::parse_name(name, &hn)) {
      hub::Repo repo;
      hub::HubOutcome out;
      std::string err;
      if (!hub::describe(hn.repo, hn.revision, &repo, &err, &out, allow)) return hub_error(r, out, err);
      kind = apps::drafter_kind(repo);
    } else {
      return error(r, 400, "name: a drafter's path or Hugging Face name", "invalid_request_error");
    }
    send_json(r, 200, {{"name", name}, {"kind", or_null(kind)}});
  });
  http.Get("/v1/hub/search", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const std::string asked = q.get_param_value("q");
    const std::string sort = q.get_param_value("sort");
    const std::string cursor = q.get_param_value("cursor");
    const std::string arch_filter = q.get_param_value("arch");
    const int limit = query_int(q, "limit", 20, 1, 100);
    const std::vector<std::string> known = interfaces::model_architectures();
    std::vector<hub::Found> found;
    hub::HubOutcome out;
    std::string next_cursor, err;
    if (!asked.empty()) {
      if (!hub::search(asked, sort, limit, &found, &err, cursor, &next_cursor, /*gguf_only=*/true, &out))
        return hub_error(r, out, err);
      if (cursor.empty()) {
        std::set<std::string> seen_ids;
        for (const auto& f : found) seen_ids.insert(f.id);
        const std::string low = ascii_lower(asked);
        const bool drafter_query = low.find("dflash") != std::string::npos || low.find("dspark") != std::string::npos ||
                                   low.find("dspine") != std::string::npos || low.find("eagle") != std::string::npos ||
                                   low.find("drafter") != std::string::npos || low.find("mtp") != std::string::npos;
        // Include block drafters' safetensors repos (which filter=gguf omits) when searching drafters or when no GGUF
        // matches.
        if (found.empty() || drafter_query) {
          std::vector<hub::Found> extra;
          std::string e2;
          if (hub::search(asked, sort, limit, &extra, &e2, "", nullptr, /*gguf_only=*/false)) {
            for (auto& x : extra) {
              if (x.is_drafter && seen_ids.insert(x.id).second) found.push_back(std::move(x));
            }
          }
        }
        // If the query names an exact repository (org/name), look it up directly so safetensors drafters or newly
        // published repos are always found even when Hub search omits them.
        hub::Name exact;
        if (hub::parse_name(asked, &exact) && seen_ids.insert(exact.repo).second) {
          hub::Repo repo;
          std::string e2;
          if (hub::describe(exact.repo, exact.revision, &repo, &e2, nullptr, remote_path_policy(*s, q))) {
            hub::Found f;
            f.id = repo.id;
            f.downloads = repo.info.downloads;
            f.likes = repo.info.likes;
            f.updated = repo.info.updated;
            f.pipeline = repo.info.pipeline;
            f.architecture = repo.info.gguf_architecture;
            f.gated = repo.info.gated;
            f.is_drafter = hub::is_drafter(repo) || repo.info.gguf_architecture == "dflash";
            found.insert(found.begin(), std::move(f));
          }
        }
      }
    } else {
      // Without a query, browse GGUF repositories of every architecture this build runs (or one chosen via `arch`),
      // resuming cleanly across HF batches with a "<skip>:<hf_cursor>" cursor so nothing is skipped or duplicated.
      std::set<std::string> allow;
      if (!arch_filter.empty() && std::find(known.begin(), known.end(), arch_filter) != known.end())
        allow.insert(arch_filter);
      else
        allow.insert(known.begin(), known.end());
      size_t skip = 0;
      std::string hf_cursor;
      if (!cursor.empty()) {
        const size_t colon = cursor.find(':');
        if (colon != std::string::npos) {
          try {
            skip = std::stoull(cursor.substr(0, colon));
          } catch (const std::exception&) {
            skip = 0;
          }
          hf_cursor = cursor.substr(colon + 1);
        } else {
          hf_cursor = cursor;
        }
      }
      constexpr int kMaxBatches = 4;
      for (int step = 0; step < kMaxBatches && int(found.size()) < limit; ++step) {
        const std::string batch_start = hf_cursor;
        std::vector<hub::Found> batch;
        std::string batch_next;
        if (!hub::search("", sort, 100, &batch, &err, batch_start, &batch_next, /*gguf_only=*/true, &out)) {
          if (found.empty()) return hub_error(r, out, err);
          break;
        }
        std::vector<hub::Found> matched;
        for (auto& f : batch)
          if (allow.count(f.architecture) > 0) matched.push_back(std::move(f));
        size_t i = skip;
        for (; i < matched.size() && int(found.size()) < limit; ++i) found.push_back(std::move(matched[i]));
        if (int(found.size()) >= limit) {
          if (i < matched.size()) next_cursor = std::to_string(i) + ":" + batch_start;
          else if (!batch_next.empty()) next_cursor = "0:" + batch_next;
          break;
        }
        skip = 0;
        hf_cursor = batch_next;
        if (hf_cursor.empty()) break;
        if (step + 1 == kMaxBatches) next_cursor = "0:" + hf_cursor;
      }
    }
    std::set<std::string> here;  // repositories with a snapshot in the cache
    for (const hub::Cached& c : hub::cached(remote_path_policy(*s, q))) here.insert(c.repo);
    const std::set<std::string> known_set(known.begin(), known.end());
    ojson models = ojson::array();
    for (const hub::Found& f : found) {
      const bool drafter = f.is_drafter || f.architecture == "dflash";
      const ojson runs = drafter ? ojson(true)
                                 : f.architecture.empty() ? ojson(nullptr) : ojson(known_set.count(f.architecture) > 0);
      models.push_back({{"id", f.id}, {"downloads", f.downloads}, {"likes", f.likes}, {"updated", or_null(f.updated)},
                        {"created", or_null(f.created)}, {"pipeline", or_null(f.pipeline)}, {"gated", f.gated},
                        {"cached", here.count(f.id) > 0}, {"architecture", or_null(f.architecture)}, {"runs", runs},
                        {"drafter", drafter}});
    }
    send_json(r, 200,
              {{"query", asked},
               {"family", asked.empty()},
               {"architectures", known},
               {"arch", or_null(arch_filter)},
               {"endpoint", hub::endpoint()},
               {"next_cursor", or_null(next_cursor)},
               {"models", models}});
  });
  http.Get("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const std::string path = q.get_param_value("path");
    if (!require_local_path(*s, q, r, path)) return;
    ojson out;
    std::string err;
    if (!dir_json(*s, q, path, &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Post("/v1/hub/dir", [s](const httplib::Request& q, httplib::Response& r) {
    // The dialog opens on this machine's desktop: of use, and polite, only to someone sitting at it.
    if (!require_access(*s, q, r, Access::kLocalOnly, "opening a folder dialog")) return;
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
    if (!dir_json(*s, q, path, &out, &err)) return error(r, 404, err, "not_found");
    send_json(r, 200, out);
  });
  http.Get("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    ojson models = ojson::array();
    for (Listed& l : listing(*s, &q)) models.push_back(std::move(l.json));
    ojson dirs = ojson::array();
    for (const std::string& d : s->models_dirs) dirs.push_back(d);
    const auto cache = hub::resolve_cache_dir();
    send_json(r, 200,
              {{"cache", cache.value},
               {"cache_from", cache.source},
               {"dirs", dirs},
               {"win_paths", kDefaultWindowsPaths},
               {"models", models}});
  });
  http.Delete("/v1/hub/cached", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    const std::string name = q.get_param_value("name");
    if (name.empty()) return error(r, 400, "name: a model as GET /v1/hub/cached lists it", "invalid_request_error");
    std::vector<Listed> all = listing(*s, &q);
    const auto it = std::find_if(all.begin(), all.end(), [&](const Listed& l) { return l.json["name"] == name; });
    if (it == all.end()) return error(r, 400, name + " is not among the cache's models", "invalid_request_error");
    if (it->json["source"] == "dir")
      return error(r, 400, name + " is a file of a model directory of yours: flyx deletes only from the Hugging Face cache",
                   "invalid_request_error");
    // When deleting the last model in a snapshot, also reclaim any unreferenced mmproj files in that snapshot
    // (unless currently mounted by the served model).
    int models_in_snap = 0;
    for (const Listed& l : all)
      if (l.json["source"] == "hub" && l.repo == it->repo && l.commit == it->commit) ++models_in_snap;
    std::vector<std::pair<std::string, std::string>> orphan_mmproj;  // (repo-relative path, disk path)
    const auto allow = remote_path_policy(*s, q);
    if (models_in_snap == 1) {
      for (const hub::Cached& c : hub::cached(allow)) {
        if (c.repo != it->repo || c.commit != it->commit) continue;
        for (const hub::RepoFile& f : c.files) {
          if (hub::mmproj_gguf_name(f.path)) {
            const std::string disk = from_path((to_path(c.dir) / to_path(f.path)).make_preferred());
            orphan_mmproj.emplace_back(f.path, disk);
          }
        }
      }
    }
    bool served = false, ok = false;
    uint64_t freed = 0;
    std::string err;
    const bool idle = s->host->with_files_in_use(
        [&](const std::vector<std::string>& used) {
          for (const std::string& f : it->on_disk) served = served || in_use(f, used);
          if (!served) {
            std::vector<std::string> to_remove = it->files;
            for (const auto& [rel, disk] : orphan_mmproj)
              if (!in_use(disk, used)) to_remove.push_back(rel);
            ok = hub::remove_cached(it->repo, it->commit, to_remove, &freed, &err, allow);
          }
        },
        &err);
    if (!idle) return error(r, 409, err, "conflict");
    if (served) return error(r, 409, name + " is being served: load another model, or unload it, first", "conflict");
    if (!ok) return error(r, 500, err, "server_error");
    std::fprintf(stderr, "flyx-server: deleted %s from the cache (%.1f GB)\n", name.c_str(), double(freed) / 1e9);
    send_json(r, 200, {{"deleted", name}, {"freed", freed}});
  });
  http.Post("/v1/unload", [s](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "loading models")) return;
    std::string err;
    if (!s->host->unload(&err)) return error(r, 409, err, "conflict");
    send_json(r, 200, s->host->status());
  });
  http.Post("/v1/agents/launch", [s](const httplib::Request& q, httplib::Response& r) {
    // The window opens on this machine's desktop, as the folder dialog does: for someone sitting at it.
    if (!require_access(*s, q, r, Access::kLocalOnly, "starting coding agents")) return;
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    auto text = [&](const char* key) {
      return body.contains(key) && body[key].is_string() ? body[key].get<std::string>() : std::string();
    };
    const std::string agent = text("agent");
    const auto& agents = launch_agents();
    if (std::find(agents.begin(), agents.end(), agent) == agents.end())
      return error(r, 400, "agent must be one of: claude, codex", "invalid_request_error");
    const std::string mode = text("permission_mode");
    if (body.contains("permission_mode") &&
        (agent != "claude" || (mode != "default" && mode != "acceptEdits" && mode != "auto" && mode != "bypassPermissions")))
      return error(r, 400, "permission_mode is for Claude Code and must be default, acceptEdits, auto or bypassPermissions",
                   "invalid_request_error");
    if (!os::shell_window_supported())
      return error(r, 501, "opening a terminal window is supported on Windows only", "not_supported");
    const ModelHost::Served served = s->host->served();
    if (!served.sched) return error(r, 409, "no model is served: load one first", "conflict");
    // The server as this request reached it: a loopback address, the agent being on the same machine.
    std::string host = q.local_addr.empty() ? "127.0.0.1" : q.local_addr;
    if (host.find(':') != std::string::npos) host = "[" + host + "]";
    const AgentTarget target{"http://" + host + ":" + std::to_string(q.local_port), served.name,
                             served.sched->max_ctx(), !s->cfg.api_key.empty(), mode};
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
  auto device_hint = [s]() -> std::string {
    const ModelHost::Served served = s->host->served();
    if (served.sched) {
      TelemetrySnapshot t;
      flyx_device_stats d{};
      bool ok = false;
      served.sched->telemetry(&t, &d, &ok);
      if (!t.model.device.empty()) return t.model.device;
    }
    if (s->device_probe.is_object() && s->device_probe.contains("name") && s->device_probe["name"].is_string())
      return s->device_probe["name"].get<std::string>();
    return "";
  };
  auto gpu_snapshot_json = [](const GpuAppSnapshot& snap, const char* kind, bool can_close) {
    ojson apps = ojson::array();
    for (const auto& a : snap.apps) {
      apps.push_back({{"id", a.id},
                      {"name", a.name},
                      {"process", a.process_name},
                      {"bytes", a.bytes},
                      {"local_bytes", a.local_bytes},
                      {"processes", a.processes},
                      {"closable", a.closable},
                      {"reason", a.reason.empty() ? ojson(nullptr) : ojson(a.reason)}});
    }
    return ojson{{"kind", kind},
                 {"supported", snap.supported},
                 {"can_close", can_close},
                 {"device", snap.device_name.empty() ? ojson(nullptr) : ojson(snap.device_name)},
                 {"mem_total", snap.mem_total},
                 {"mem_used", snap.mem_used},
                 {"mem_free", snap.mem_free},
                 {"own_bytes", snap.own_bytes},
                 {"others_bytes", snap.others_bytes},
                 {"system_bytes", snap.system_bytes},
                 {"idle_igpu", snap.idle_igpu.empty() ? ojson(nullptr) : ojson(snap.idle_igpu)},
                 {"apps", std::move(apps)}};
  };
  http.Get("/v1/device/processes", [s, device_hint, gpu_snapshot_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "inspecting memory usage")) return;
    const std::string kind = q.has_param("kind") ? q.get_param_value("kind") : "vram";
    if (kind != "vram" && kind != "ram") return error(r, 400, "kind must be vram or ram", "invalid_request_error");
    r.set_header("Cache-Control", "no-store");
    const bool ram = kind == "ram";
    const GpuAppSnapshot snap = ram ? list_ram_apps() : list_gpu_apps(device_hint());
    if (!ram && snap.mem_total > 0) s->probe_mem_used.store(snap.mem_used, std::memory_order_relaxed);
    send_json(r, 200, gpu_snapshot_json(snap, ram ? "ram" : "vram", from_this_machine(q)));
  });
  http.Post("/v1/device/close", [s, device_hint, gpu_snapshot_json](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kLocalOnly, "closing applications")) return;
    const ojson body = interfaces::parse_json(q.body);
    if (!body.is_object()) return error(r, 400, "request body is not a JSON object", "invalid_request_error");
    if (!body.contains("id") || !body["id"].is_string() || body["id"].get<std::string>().empty())
      return error(r, 400, "id: an application id from GET /v1/device/processes", "invalid_request_error");
    bool force = false;
    if (body.contains("force") && !body["force"].is_null()) {
      if (!body["force"].is_boolean()) return error(r, 400, "force: true or false", "invalid_request_error");
      force = body["force"].get<bool>();
    }
    std::string kind = "vram";
    if (body.contains("kind") && !body["kind"].is_null()) {
      if (!body["kind"].is_string()) return error(r, 400, "kind must be vram or ram", "invalid_request_error");
      kind = body["kind"].get<std::string>();
      if (kind != "vram" && kind != "ram") return error(r, 400, "kind must be vram or ram", "invalid_request_error");
    }
    const bool ram = kind == "ram";
    const GpuCloseResult res =
        ram ? close_ram_app(body["id"].get<std::string>(), force) : close_gpu_app(device_hint(), body["id"].get<std::string>(), force);
    if (!ram && res.snapshot.mem_total > 0) s->probe_mem_used.store(res.snapshot.mem_used, std::memory_order_relaxed);
    if (res.status == CloseStatus::kUnsupported)
      return error(r, 501, "closing applications is not supported on this system", "not_supported");
    if (res.status == CloseStatus::kNotFound)
      return error(r, 404, "that application is no longer listed", "not_found");
    if (res.status == CloseStatus::kForbidden)
      return error(r, 403, "that application is protected and cannot be closed from FLYX", "permission_error");
    if (res.status == CloseStatus::kAccessDenied)
      return error(r, 403, "permission denied: close that application manually", "permission_error");
    const char* status = res.status == CloseStatus::kClosed   ? "closed"
                       : res.status == CloseStatus::kNoWindow ? "no_window"
                                                              : "still_running";
    send_json(r, 200, {{"status", status},
                       {"windows_signaled", res.windows_signaled},
                       {"name", res.app_name},
                       {"snapshot", gpu_snapshot_json(res.snapshot, ram ? "ram" : "vram", true)}});
  });
}

}  // namespace flyx::server
