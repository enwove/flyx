// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/common/model_source.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <tuple>

#include "core/interfaces/model.h"
#include "core/interfaces/options.h"
#include "core/modelfile/modelfile.h"
#include "core/modelfile/readers.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::apps {

namespace {

namespace fs = std::filesystem;

using os::from_path;
using os::to_path;

bool exists_on_disk(const std::string& path) {
  std::error_code ec;
  return fs::exists(to_path(path), ec);
}

std::string expand_tilde(const std::string& path) {
  if (path.empty() || path[0] != '~') return path;
  if (path.size() > 1 && path[1] != '/' && path[1] != '\\') return path;
  const std::string home = os::home_dir();
  if (home.empty()) return path;
  if (path.size() == 1) return home;
  const bool home_slash = home.back() == '/' || home.back() == '\\';
  return home_slash ? home + path.substr(2) : home + path.substr(1);
}

enum class LocalPathResult { kNotFound, kFound, kDenied };

// Matches a local path verbatim first, then with leading "~/" or "~\" expanded.
// Enforces allow_path BEFORE any filesystem probe on both raw and tilde-expanded candidates.
LocalPathResult local_path_if_exists(const std::string& spec, const PathAllowFn& allow_path, std::string* resolved,
                                     std::string* err) {
  if (allow_path && !allow_path(spec)) {
    if (err) *err = spec + ": path is not allowed by remote path policy";
    return LocalPathResult::kDenied;
  }
  if (exists_on_disk(spec)) {
    *resolved = spec;
    return LocalPathResult::kFound;
  }
  const std::string expanded = expand_tilde(spec);
  if (expanded != spec) {
    if (allow_path && !allow_path(expanded)) {
      if (err) *err = expanded + ": path is not allowed by remote path policy";
      return LocalPathResult::kDenied;
    }
    if (exists_on_disk(expanded)) {
      *resolved = expanded;
      return LocalPathResult::kFound;
    }
  }
  return LocalPathResult::kNotFound;
}

// Fallback for Windows cmd.exe pastes of single-quoted shell arguments ('...').
std::optional<std::string> strip_single_quotes(const std::string& spec) {
  if (spec.size() >= 2 && spec.front() == '\'' && spec.back() == '\'')
    return spec.substr(1, spec.size() - 2);
  return std::nullopt;
}

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The bytes of a file, or of a drafter's directory (its own files, not those below it).
uint64_t bytes_on_disk(const std::string& path, const PathAllowFn& allow_path = {}) {
  if (allow_path && !allow_path(path)) return 0;
  std::error_code ec;
  const fs::path p = to_path(path);
  if (!fs::is_directory(p, ec)) {
    const auto sz = fs::file_size(p, ec);
    return ec ? 0 : uint64_t(sz);
  }
  uint64_t n = 0;
  for (fs::directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec)) {
    if (allow_path && !allow_path(from_path(it->path()))) continue;
    std::error_code e2;
    if (!it->is_regular_file(e2) || e2) continue;
    const auto sz = it->file_size(e2);
    if (!e2) n += uint64_t(sz);
  }
  return n;
}

// The cache key describes the same shard gguf_peek actually reads.
flyx::GgufPeek gguf_peek_cached(const std::string& requested, uint64_t size, fs::file_time_type modified,
                                const PathAllowFn& allow_path = {}) {
  const std::string path = flyx::gguf_header_path(requested);
  if (allow_path && !allow_path(path)) return {};
  if (path != requested || modified == fs::file_time_type::min()) {
    std::error_code ec;
    const fs::path p = to_path(path);
    size = fs::file_size(p, ec);
    if (ec) return flyx::gguf_peek(path, allow_path);
    modified = fs::last_write_time(p, ec);
    if (ec) return flyx::gguf_peek(path, allow_path);
  }
  constexpr size_t kMaxCachedEntries = 2048;
  static std::mutex mu;
  static std::map<std::string, std::tuple<uint64_t, fs::file_time_type, flyx::GgufPeek>> cache;
  {
    std::lock_guard<std::mutex> lock(mu);
    const auto it = cache.find(path);
    if (it != cache.end() && std::get<0>(it->second) == size && std::get<1>(it->second) == modified)
      return std::get<2>(it->second);
  }
  const flyx::GgufPeek peek = flyx::gguf_peek(path, allow_path);
  std::lock_guard<std::mutex> lock(mu);
  if (cache.size() >= kMaxCachedEntries && cache.find(path) == cache.end()) cache.clear();
  cache[path] = {size, modified, peek};
  return peek;
}

GgufClassification gguf_identity(const flyx::GgufPeek& peek, const std::string& name,
                                 const std::string& repo_arch = {}) {
  const int mtp = peek.valid ? peek.mtp_layers : -1;
  const std::string arch = peek.valid ? peek.arch : "";
  if (peek.valid && peek.arch == "clip") return {"mmproj", "", arch, mtp};
  if (peek.valid && (peek.arch == "dflash" || peek.has_markov_context))
    return {"drafter", peek.has_markov_context ? "dspark" : "dflash2", arch, mtp};
  // An MTP sidecar shares the target architecture. Its directory or prefix distinguishes it
  // from a main GGUF that embeds exactly the same nextn_predict_layers metadata.
  if (hub::mtp_gguf_name(name)) return {"drafter", "mtp", arch, mtp};
  if (!peek.valid || peek.arch.empty()) {
    if (hub::mmproj_gguf_name(name)) return {"mmproj", "", arch, mtp};
    if (hub::dspark_gguf_name(name)) return {"drafter", "dspark", arch, mtp};
    if (repo_arch == "dflash" && hub::model_gguf_name(name)) {
      const auto base = lower(name.substr(name.rfind('/') + 1));
      return {"drafter", base.starts_with("dflash2-") ? "dflash2" : "", arch, mtp};
    }
  }
  // A model's header under a name that holds none (a file under old_support/): not listed.
  if (!hub::model_gguf_name(name) && !hub::mmproj_gguf_name(name) && !hub::dspark_gguf_name(name)) return {};
  return {"gguf", "", arch, mtp};
}

}  // namespace

std::string drafter_mode_of(const GgufClassification& c) {
  if (c.arch.empty()) return "";
  if (c.arch == "dflash") return c.drafter_kind;  // dspark or dflash2, by the header's Markov flag
  return c.mtp_layers > 0 ? "mtp" : "";
}

GgufClassification classify_gguf(const std::string& path, const std::string& rel_name, uint64_t size,
                                 fs::file_time_type modified, const PathAllowFn& allow_path) {
  std::string name = rel_name.empty() ? path : rel_name;
  std::replace(name.begin(), name.end(), '\\', '/');
  const std::string base = lower(name.substr(name.rfind('/') + 1));
  if (!base.ends_with(".gguf") || base.find("imatrix") != std::string::npos) return {};
  return gguf_identity(gguf_peek_cached(path, size, modified, allow_path), name);
}

std::vector<HubGguf> hub_ggufs(const hub::Repo& repo, const PathAllowFn& allow_path) {
  const fs::path root = to_path(hub::snapshot_dir(repo));
  auto choices = hub::gguf_choices(repo, /*include_companions=*/true);
  std::map<std::string, int> labels;
  for (const auto& g : choices)
    if (hub::model_gguf_name(g.file.path)) ++labels[lower(g.quant)];
  std::vector<HubGguf> out;
  for (auto& g : choices) {
    HubGguf item;
    const auto& path = g.file.path;
    item.disk = from_path((root / to_path(path)).make_preferred());
    flyx::GgufPeek peek;
    item.cached = g.complete;
    for (const auto& part : g.parts) {
      const fs::path part_path = (root / to_path(part.path)).make_preferred();
      if (allow_path && !allow_path(from_path(part_path))) {
        item.cached = false;
        continue;
      }
      std::error_code ec;
      const auto size = fs::file_size(part_path, ec);
      if (ec || (part.size && size != part.size)) {
        item.cached = false;
        continue;
      }
      if (g.complete && part.path == path) {
        item.modified = fs::last_write_time(to_path(item.disk), ec);
        peek = gguf_peek_cached(item.disk, size, item.modified, allow_path);
      }
    }
    item.cls = gguf_identity(peek, path, repo.info.gguf_architecture);
    item.selector = hub::model_gguf_name(path) && labels[lower(g.quant)] == 1
                        ? g.quant : from_path(to_path(path).filename());
    item.choice = std::move(g);
    out.push_back(std::move(item));
  }
  return out;
}

const HubGguf* preferred_companion(const std::vector<HubGguf>& ggufs, const std::string& kind) {
  hub::Repo candidates;
  const HubGguf* first = nullptr;
  for (const auto& item : ggufs) {
    if (!item.choice.complete || (item.cls.kind != kind && item.cls.drafter_kind != kind)) continue;
    if (!first) first = &item;
    candidates.files.push_back(item.choice.file);
  }
  // Preserve the Hub's preferred precision, MTP directory and default DSpark drafter when conventionally named
  // candidates exist.
  std::string preferred;
  if (kind == "mmproj") {
    if (const hub::RepoFile* f = hub::mmproj_file(candidates)) preferred = f->path;
  } else if (kind == "mtp") {
    if (const hub::RepoFile* f = hub::mtp_file(candidates)) preferred = f->path;
  } else if (kind == "dspark") {
    for (const hub::CompanionFile& c : hub::companion_files(candidates))
      if (c.kind == "dspark" && c.is_default) preferred = c.file.path;
  }
  if (!preferred.empty())
    for (const auto& item : ggufs)
      if (item.choice.file.path == preferred) return &item;
  return first;
}

flyx::GgufPeek gguf_header(const std::string& path, const PathAllowFn& allow_path) {
  return gguf_peek_cached(path, 0, fs::file_time_type::min(), allow_path);
}

double modified_unix(fs::file_time_type modified) {
  if (modified == fs::file_time_type::min()) return 0;
#if __cpp_lib_chrono >= 201907L
  const auto time = std::chrono::clock_cast<std::chrono::system_clock>(modified);
#else
  // libstdc++ exposes file_clock::to_sys before the full C++20 chrono API.
  const auto time = fs::file_time_type::clock::to_sys(modified);
#endif
  return std::chrono::duration<double>(time.time_since_epoch()).count();
}

std::vector<std::string> models_dirs_from_env() {
#if defined(_WIN32)
  constexpr char kSep = ';';
#else
  constexpr char kSep = ':';
#endif
  std::vector<std::string> out;
  const std::string v = os::getenv_str(env::kModelsDir);
  for (size_t start = 0; start <= v.size();) {
    const size_t end = std::min(v.find(kSep, start), v.size());
    if (end > start) out.push_back(v.substr(start, end - start));
    start = end + 1;
  }
  return out;
}

std::string expand_user_path(const std::string& path) {
  if (path.empty() || path[0] != '~') return path;
#if defined(_WIN32)
  auto is_sep = [](char c) { return c == '/' || c == '\\'; };
#else
  auto is_sep = [](char c) { return c == '/'; };
#endif
  if (path.size() > 1 && !is_sep(path[1])) return path;
  const std::string home = os::home_dir();
  if (home.empty()) return path;
  size_t pos = 1;
  while (pos < path.size() && is_sep(path[pos])) ++pos;
  if (pos >= path.size()) return home;
  const fs::path suffix = to_path(path.substr(pos));
  if (suffix.is_absolute() || suffix.has_root_name()) return path;
  return from_path(to_path(home) / suffix);
}

std::string canonical_model_dir(std::string_view raw) {
  if (raw.empty()) return {};
  const std::string expanded = expand_user_path(std::string(raw));
  std::error_code ec;
  fs::path p = fs::absolute(to_path(expanded), ec);
  if (ec) p = to_path(expanded);
  std::string s = from_path(p);
#if defined(_WIN32)
  std::replace(s.begin(), s.end(), '/', '\\');
  while (s.size() > 1 && s.back() == '\\') {
    if (s.size() == 3 && s[1] == ':') break;
    if (s.size() == 2 && s[0] == '\\') break;
    s.pop_back();
  }
#else
  // On POSIX, '\' is a legal filename character and ".." must not be lexically collapsed across symlinks.
  std::string norm;
  norm.reserve(s.size());
  for (char c : s) {
    if (c == '/' && !norm.empty() && norm.back() == '/') continue;
    norm.push_back(c);
  }
  while (norm.size() > 1 && norm.back() == '/') norm.pop_back();
  s = std::move(norm);
#endif
  return s;
}

namespace {

ScanFs resolve_scan_fs(const ScanFs* fs_ops) {
  ScanFs ops;
  ops.is_directory = [](const fs::path& p, std::error_code& ec) { return fs::is_directory(p, ec); };
  ops.is_regular_file = [](const fs::path& p, std::error_code& ec) { return fs::is_regular_file(p, ec); };
  ops.file_size = [](const fs::path& p, std::error_code& ec) { return uint64_t(fs::file_size(p, ec)); };
  ops.last_write_time = [](const fs::path& p, std::error_code& ec) { return fs::last_write_time(p, ec); };
  ops.weakly_canonical = [](const fs::path& p, std::error_code& ec) { return fs::weakly_canonical(p, ec); };
  ops.for_each_dir_entry = [](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                              std::error_code& ec) {
    for (fs::directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!visit(it->path())) break;
    }
  };
  ops.classify_gguf = [](const std::string& p, const std::string& rel, uint64_t sz, fs::file_time_type mt) {
    return classify_gguf(p, rel, sz, mt);
  };
  if (!fs_ops) return ops;
  if (fs_ops->is_directory) ops.is_directory = fs_ops->is_directory;
  if (fs_ops->is_regular_file) ops.is_regular_file = fs_ops->is_regular_file;
  if (fs_ops->file_size) ops.file_size = fs_ops->file_size;
  if (fs_ops->last_write_time) ops.last_write_time = fs_ops->last_write_time;
  if (fs_ops->weakly_canonical) ops.weakly_canonical = fs_ops->weakly_canonical;
  if (fs_ops->for_each_dir_entry) ops.for_each_dir_entry = fs_ops->for_each_dir_entry;
  if (fs_ops->classify_gguf) ops.classify_gguf = fs_ops->classify_gguf;
  return ops;
}

}  // namespace

std::vector<LocalModel> scan_model_dirs(const std::vector<std::string>& dirs, int deepest,
                                        const PathAllowFn& allow_path, const ScanFs* fs_ops, int max_entries) {
  const ScanFs ops = resolve_scan_fs(fs_ops);
  const int entry_cap = std::max(0, max_entries);

  std::vector<LocalModel> out;
  std::set<fs::path> seen;  // a file reached from two of the directories (one inside the other) once
  struct LocalShard {
    std::string path, root, file, rel;
    uint64_t size = 0;
    fs::file_time_type modified{};
  };
  struct SplitAcc {
    int count = 0;
    std::map<int, LocalShard> parts;
  };
  std::map<std::pair<std::string, int>, SplitAcc> splits;
  for (const std::string& dir : dirs) {
    const std::string root_str = fs_ops ? dir : canonical_model_dir(dir);
    if (root_str.empty()) continue;
    if (allow_path && !allow_path(root_str)) continue;
    std::error_code ec;
    const fs::path root = to_path(root_str);
    if (!ops.is_directory(root, ec)) continue;
    std::map<fs::path, int> visited_depth;
    const fs::path root_canon = ops.weakly_canonical(root, ec);
    if (!ec) visited_depth[root_canon] = -1;
    ec.clear();
    int entries = 0;
    // Breadth-first non-recursive walk so every entry is authorized via allow_path BEFORE any target-following
    // stat, canonicalization, open, or descent, shallower paths win over deeper symlink aliases, and directory
    // iteration halts immediately at entry_cap without buffering remaining entries.
    std::deque<std::pair<fs::path, int>> queue;
    queue.emplace_back(root, 0);
    while (!queue.empty() && entries < entry_cap) {
      auto [cur_dir, cur_depth] = std::move(queue.front());
      queue.pop_front();
      std::error_code list_ec;
      ops.for_each_dir_entry(
          cur_dir,
          [&](const fs::path& child) -> bool {
            if (entries >= entry_cap) return false;
            ++entries;
            const bool more = entries < entry_cap;
            const std::string child_str = from_path(child);
            if (allow_path && !allow_path(child_str)) return more;
            const std::string name = from_path(child.filename());
            std::error_code e2;
            if (ops.is_directory(child, e2)) {
              if (e2 || name.empty() || name[0] == '.' || name[0] == '$' ||
                  name == "node_modules" || name == "System Volume Information" || cur_depth + 1 >= deepest) {
                return more;
              }
              const fs::path dcanon = ops.weakly_canonical(child, e2);
              if (e2) return more;
              auto [vit, inserted] = visited_depth.emplace(dcanon, cur_depth);
              if (!inserted && vit->second <= cur_depth) return more;
              vit->second = cur_depth;
              queue.emplace_back(child, cur_depth + 1);
              return more;
            }
            const std::string low = lower(name);
            const bool gguf = low.ends_with(".gguf"), flyx = low.ends_with(".flyx");
            if (name == "adapter_config.json") {  // a LoRA adapter as PEFT saves it: listed by its folder
              const fs::path adir = child.parent_path(), weights = adir / "adapter_model.safetensors";
              if (allow_path && !allow_path(from_path(weights))) return more;
              const fs::path canon = ops.weakly_canonical(adir, e2);
              if (e2 || !ops.is_regular_file(weights, e2) || !seen.insert(canon).second) return more;
              LocalModel m;
              m.path = from_path(adir);
              m.root = root_str;
              m.file = from_path(adir.filename());
              m.kind = "lora";
              m.size = ops.file_size(weights, e2);
              if (e2) return more;
              std::error_code e3;
              const auto mt = ops.last_write_time(weights, e3);
              m.modified = e3 ? fs::file_time_type::min() : mt;
              out.push_back(std::move(m));
              return more;
            }
            if (!gguf && !flyx) return more;
            std::string rel = from_path(child.lexically_relative(root));
#if defined(_WIN32)
            std::replace(rel.begin(), rel.end(), '\\', '/');
#endif
            if (gguf && low.find("imatrix") != std::string::npos) return more;
            const fs::path canon = ops.weakly_canonical(child, e2);
            if (e2 || !ops.is_regular_file(child, e2) || e2 || !seen.insert(canon).second) return more;
            const uint64_t sz = ops.file_size(child, e2);
            if (e2) return more;
            std::error_code e3;
            const auto mt = ops.last_write_time(child, e3);
            const fs::file_time_type modified = e3 ? fs::file_time_type::min() : mt;
            hub::SplitGguf sp;
            if (gguf && hub::parse_split_gguf_name(child_str, &sp)) {
              auto& acc = splits[{sp.prefix, sp.count}];
              acc.count = sp.count;
              acc.parts[sp.part] = {child_str, root_str, name, rel, sz, modified};
              return more;
            }
            LocalModel m;
            m.path = child_str;
            m.root = root_str;
            m.file = name;
            if (gguf) {
              const GgufClassification cls = ops.classify_gguf(m.path, rel, sz, modified);
              if (cls.kind.empty()) return more;
              m.kind = cls.kind;
              m.drafter_kind = cls.drafter_kind;
              m.drafter_mode = drafter_mode_of(cls);
              m.mtp_layers = cls.mtp_layers;
            } else {
              m.kind = "flyx";
            }
            m.quant = hub::quant_in_name(name);
            m.size = sz;
            m.modified = modified;
            out.push_back(std::move(m));
            return more;
          },
          list_ec);
    }
  }
  for (const auto& [_, acc] : splits) {
    if (int(acc.parts.size()) != acc.count) continue;
    bool complete = true;
    uint64_t total = 0;
    fs::file_time_type latest = fs::file_time_type::min();
    int expect = 1;
    for (const auto& [part_no, shard] : acc.parts) {
      if (part_no != expect++) {
        complete = false;
        break;
      }
      total += shard.size;
      if (shard.modified > latest) latest = shard.modified;
    }
    if (!complete) continue;
    const LocalShard& first = acc.parts.begin()->second;
    const GgufClassification cls = ops.classify_gguf(first.path, first.rel, first.size, first.modified);
    if (cls.kind.empty()) continue;
    LocalModel m;
    m.path = first.path;
    m.root = first.root;
    m.file = first.file;
    m.kind = cls.kind;
    m.drafter_kind = cls.drafter_kind;
    m.drafter_mode = drafter_mode_of(cls);
    m.mtp_layers = cls.mtp_layers;
    m.quant = hub::quant_in_name(first.file);
    m.size = total;
    m.parts = acc.count;
    m.modified = latest;
    out.push_back(std::move(m));
  }
  std::sort(out.begin(), out.end(), [](const LocalModel& a, const LocalModel& b) { return a.path < b.path; });
  return out;
}

namespace {

std::string gigabytes(uint64_t n) {
  char s[32];
  std::snprintf(s, sizeof s, "%.1f GB", double(n) / 1e9);
  return s;
}

// One thing named: a path as it is, else the Hub's files for the name.  A drafter's config and safetensors
// return its directory; a single or split GGUF returns its (first) model file.
bool one(const std::string& spec, const hub::ProgressFn& progress, const std::atomic<bool>* stop, std::string* path,
         hub::Repo* repo, std::string* err, const PathAllowFn& allow_path = {}) {
  std::string local;
  const LocalPathResult r1 = local_path_if_exists(spec, allow_path, &local, err);
  if (r1 == LocalPathResult::kDenied) return false;
  if (r1 == LocalPathResult::kFound) return *path = local, true;
  hub::Name name;
  bool parsed = hub::parse_name(spec, &name);
  if (!parsed) {
    if (const auto unquoted = strip_single_quotes(spec); unquoted && !unquoted->empty()) {
      const LocalPathResult r2 = local_path_if_exists(*unquoted, allow_path, &local, err);
      if (r2 == LocalPathResult::kDenied) return false;
      if (r2 == LocalPathResult::kFound) return *path = local, true;
      parsed = hub::parse_name(*unquoted, &name);
    }
  }
  if (!parsed)
    return *err = spec + ": no such file, and not a Hugging Face name (org/name, org/name:QUANT)", false;
  std::vector<hub::RepoFile> files;
  std::vector<std::string> paths;
  if (!hub::describe(name.repo, name.revision, repo, err, nullptr, allow_path) || !hub::select(*repo, name, &files, err) ||
      !hub::fetch(*repo, files, &paths, progress, stop, err, allow_path))
    return false;
  // A drafter's config and safetensors: its directory; else the (first) model file.
  const std::string resolved = files[0].path == "config.json" ? hub::snapshot_dir(*repo) : paths[0];
  if (allow_path) {
    if (!allow_path(resolved)) return *err = resolved + ": path is not allowed by remote path policy", false;
    for (const std::string& p : paths)
      if (!allow_path(p)) return *err = p + ": path is not allowed by remote path policy", false;
  }
  *path = resolved;
  return true;
}

}  // namespace

std::string drafter_kind(const std::string& path, const PathAllowFn& allow_path) {
  if (allow_path && !allow_path(path)) return "";
  if (const flyx::GgufPeek p = gguf_header(path, allow_path); p.valid)
    return p.arch == "dflash" ? (p.has_markov_context ? "dspark" : "dflash2") : p.mtp_layers > 0 ? "mtp" : "";
  return flyx::drafter_source_kind(path, allow_path);
}

std::string drafter_kind(const hub::Repo& r, const hub::Name& name, const PathAllowFn& allow_path) {
  // The file the name selects first: a repository can hold a drafter's config and GGUF drafters of other kinds.
  std::vector<hub::RepoFile> files;
  std::string err;
  if (!hub::select(r, name, &files, &err) || files.empty()) return "";
  if (files.front().path == "config.json") return hub::drafter_kind(r);  // the drafter's directory: its config
  for (const HubGguf& g : hub_ggufs(r, allow_path))
    if (g.choice.file.path == files.front().path) return g.cached ? drafter_kind(g.disk, allow_path) : "";
  return "";
}

bool reads_vision(const std::string& arch) {
  if (arch.empty()) return true;
  const std::vector<std::string> vision = interfaces::vision_architectures();
  return std::find(vision.begin(), vision.end(), arch) != vision.end();
}

std::string model_name_of(const std::string& path, bool local) {
  const fs::path file = to_path(path).filename();
  hub::SplitGguf sp;
  if (hub::parse_split_gguf_name(from_path(file), &sp)) return sp.prefix;
  std::string extension = from_path(file.extension());
  if (local) extension = lower(extension);
  return from_path(extension == ".gguf" || extension == ".flyx" ? file.stem() : file);
}

namespace {

// The Hub's list of verified drafters (docs/models.md).
constexpr const char* kDraftersRepo = "Enwove/flyx-drafters";
constexpr const char* kDraftersFile = "drafters.json";

// drafters.json: the copy in the Hugging Face cache, read without the network; and, `ask_hub` (a call that talks to the
// Hub anyway), the Hub's main once a process (asked again after 10 minutes when it could not be had), no lock held
// while it is asked.  "" when neither has it.
std::string hub_drafters(bool ask_hub, const PathAllowFn& allow_path = {}) {
  static std::mutex mu;
  static std::string fetched;  // the Hub's, once had
  static std::chrono::steady_clock::time_point tried;
  static bool ever = false;
  {
    std::lock_guard<std::mutex> lock(mu);
    if (!fetched.empty()) return fetched;
    const auto now = std::chrono::steady_clock::now();
    ask_hub = ask_hub && !(ever && now - tried < std::chrono::minutes(10));
    if (ask_hub) ever = true, tried = now;
  }
  hub::Repo r;
  std::string err, path;
  if (ask_hub ? hub::describe(kDraftersRepo, "main", &r, &err, nullptr, allow_path)
              : hub::in_cache(kDraftersRepo, "main", &r, allow_path))
    for (const hub::RepoFile& f : r.files) {
      std::vector<std::string> paths;
      if (f.path != kDraftersFile || f.size > (1u << 20)) continue;
      if (ask_hub ? hub::fetch(r, {f}, &paths, {}, nullptr, &err, allow_path)
                  : (paths.push_back(hub::snapshot_dir(r) + "/" + f.path), true))
        path = paths[0];
    }
  if (path.empty() || (allow_path && !allow_path(path))) return "";
  std::ifstream in(to_path(path), std::ios::binary);
  std::string got = std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (!nlohmann::json::accept(got)) return "";
  if (ask_hub) {
    std::lock_guard<std::mutex> lock(mu);
    fetched = got;
  }
  return got;
}

}  // namespace

std::vector<KnownDrafter> known_drafters_in(const std::string& list, const std::string& repo,
                                            const std::vector<std::string>& base_models, const std::string& relation) {
  const nlohmann::json j = nlohmann::json::parse(list, nullptr, false);
  if (!j.is_object() || !j.contains("version") || j["version"] != 1 || !j.contains("drafters") ||
      !j["drafters"].is_array())
    return {};
  std::vector<KnownDrafter> exact, inferred;
  std::set<std::string> names;
  for (const auto& e : j["drafters"]) {
    const auto text = [&](const char* k) { return e.contains(k) && e[k].is_string(); };
    if (!e.is_object() || !text("base") || !text("repo") || !text("kind")) return {};
    const std::string base = e["base"], name = e["repo"], kind = e["kind"];
    interfaces::DrafterMode mode;
    const bool runs = interfaces::parse_drafter_mode(kind, &mode);  // a kind this build does not run: left out
    if ((runs && (mode == interfaces::DrafterMode::kAuto || mode == interfaces::DrafterMode::kAll ||
                  mode == interfaces::DrafterMode::kNone)) ||
        !names.insert(name).second)
      return {};
    if (!runs) continue;
    if (!repo.empty() && base == repo) exact.push_back({name, kind, base, true, ""});
    else if (std::find(base_models.begin(), base_models.end(), base) != base_models.end())
      inferred.push_back({name, kind, base, false, relation});
  }
  exact.insert(exact.end(), inferred.begin(), inferred.end());
  return exact;
}

std::vector<KnownDrafter> known_drafters(const std::string& repo, const std::vector<std::string>& base_models,
                                         bool hub, const PathAllowFn& allow_path) {
  if (repo.empty() && base_models.empty()) return {};
  const std::string list = hub_drafters(hub, allow_path);
  std::vector<KnownDrafter> known = known_drafters_in(list, repo, base_models, "");
  if (hub && std::any_of(known.begin(), known.end(), [](const KnownDrafter& k) { return !k.exact; }))
    known = known_drafters_in(list, repo, base_models, hub::base_relation(repo));
  return known;
}

bool wants_mtp(const EngineOptions& eo) {
  if (!eo.speculative) return false;
  interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
  return interfaces::parse_drafter_mode(eo.drafter, &mode) &&
         (mode == interfaces::DrafterMode::kAuto || mode == interfaces::DrafterMode::kMtp ||
          mode == interfaces::DrafterMode::kAll);
}

bool wants_dspark(const EngineOptions& eo, const std::vector<std::string>& drafters) {
  if (!eo.speculative) return false;
  interfaces::DrafterMode mode = interfaces::DrafterMode::kAuto;
  if (!interfaces::parse_drafter_mode(eo.drafter, &mode)) return false;
  return mode == interfaces::DrafterMode::kDSpark || mode == interfaces::DrafterMode::kAll ||
         (mode == interfaces::DrafterMode::kAuto && drafters.empty());
}

namespace {

bool is_dspark_file(const std::string& path, const PathAllowFn& allow_path = {}) {
  if (allow_path && !allow_path(path)) return false;
  const flyx::GgufPeek h = flyx::gguf_peek(path, allow_path);
  return h.valid && h.arch == "dflash" && h.has_markov_context;
}

// The verified drafters for a model (a Hub repository's, and a path's) that it was not given: made for the repository it
// is, or inferred, for the base models it names (its model card's, its file's).
void suggest_drafters(const hub::Repo& repo, const std::vector<std::string>& drafters, Fetched* out,
                      const PathAllowFn& allow_path = {}) {
  const flyx::ModelOrigin origin = flyx::model_origin(out->model, allow_path);
  std::vector<std::string> bases = repo.base_models;
  bases.insert(bases.end(), origin.base_models.begin(), origin.base_models.end());
  // A Hub model's load talks to the Hub anyway; a path's reads the cache's copy alone.
  const bool hub_model = !repo.id.empty();
  for (const KnownDrafter& known : known_drafters(hub_model ? repo.id : origin.repo, bases, hub_model, allow_path)) {
    const auto named = [&](const std::string& d) { return d.rfind(known.repo, 0) == 0; };
    if (std::any_of(drafters.begin(), drafters.end(), named)) continue;
    out->notes.push_back("a " + known.kind + " drafter verified for " +
                         (known.exact ? std::string("this model")
                                      : "its base model " + known.base +
                                            (known.relation.empty() ? "" : " (" + known.relation + ")")) +
                         " is published: --drafter-file " + known.repo);
    // Its size where the cache has it; the Hub is not asked for a suggestion.
    KnownDrafter k = known;
    hub::Repo cached;
    std::vector<hub::RepoFile> files;
    std::string ignored;
    if (hub::in_cache(known.repo, "main", &cached, allow_path) && hub::select(cached, {known.repo}, &files, &ignored)) {
      k.size = 0;
      for (const hub::RepoFile& f : files) *k.size += f.size;
    }
    out->suggested.push_back(std::move(k));
  }
}

}  // namespace

std::string dspark_beside(const std::string& model_path, const PathAllowFn& allow_path) {
  if (allow_path && !allow_path(model_path)) return "";
  std::error_code ec;
  const fs::path p = to_path(model_path);
  if (lower(from_path(p.extension())) != ".gguf" || !fs::is_regular_file(p, ec)) return "";
  // The model's name without its quantization: what a drafter made for it is named after.
  std::string name = model_name_of(model_path);
  const std::string quant = hub::quant_in_name(from_path(p.filename()));
  if (!quant.empty() && name.size() > quant.size() && lower(name).ends_with(lower(quant))) {
    name.resize(name.size() - quant.size());
    while (!name.empty() && (name.back() == '-' || name.back() == '_' || name.back() == '.')) name.pop_back();
  }
  if (name.empty()) return "";
  const std::string want = "dspark-" + lower(name);
  std::vector<std::string> found;
  const fs::path dir = p.has_parent_path() ? p.parent_path() : fs::path(".");
  if (allow_path && !allow_path(from_path(dir))) return "";
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
    const std::string low = lower(from_path(it->path().filename()));
    if (!low.ends_with(".gguf") || !low.starts_with(want) || low.size() <= want.size()) continue;
    if (const char next = low[want.size()]; next != '-' && next != '_' && next != '.') continue;  // another name
    const std::string cand = from_path(it->path());
    if (allow_path && !allow_path(cand)) continue;
    std::error_code e2;
    if (!it->is_regular_file(e2) || e2 || !is_dspark_file(cand, allow_path)) continue;
    found.push_back(cand);
  }
  std::sort(found.begin(), found.end());
  return found.empty() ? "" : found.front();
}

bool fetch_model(const std::string& model, const std::vector<std::string>& drafters, bool want_mtp,
                 const hub::ProgressFn& progress, const std::atomic<bool>* stop, Fetched* out, std::string* err,
                 const std::string& mmproj, bool want_vision, bool want_dspark, const std::string& lora,
                 const PathAllowFn& allow_path) {
  *out = {};
  hub::Repo repo;
  if (!one(model, progress, stop, &out->model, &repo, err, allow_path)) return false;
  out->name = model_name_of(out->model, repo.id.empty());
  const flyx::GgufPeek own = want_mtp ? gguf_header(out->model, allow_path) : flyx::GgufPeek{};
  const int own_mtp = own.valid ? own.mtp_layers : -1;  // -1: not a GGUF, or not asked
  if (own_mtp > 0) out->drafter_sources.push_back({"mtp", "model file", std::nullopt});
  const auto clean_hub = [](const std::string& s) {
    const auto u = strip_single_quotes(s);
    return u && !u->empty() ? *u : s;
  };
  for (const std::string& d : drafters) {
    hub::Repo drepo;
    std::string path;
    std::error_code dir_ec;
    if (!one(d, progress, stop, &path, &drepo, err, allow_path)) return false;
    out->drafters.push_back(path);
    out->drafter_sources.push_back(
        {drepo.id.empty() || !hub::is_drafter(drepo) || !fs::is_directory(to_path(path), dir_ec)
             ? drafter_kind(path, allow_path)  // a file: its own header, whatever else the repository holds
             : hub::drafter_kind(drepo),
         drepo.id.empty() ? path : clean_hub(d),
         bytes_on_disk(path, allow_path)});
  }
  if (want_vision && !mmproj.empty()) {
    hub::Repo mrepo;
    std::string mpath;
    if (!one(mmproj, progress, stop, &mpath, &mrepo, err, allow_path)) return false;
    out->mmproj = mpath;
    out->mmproj_source = mrepo.id.empty() ? mpath : clean_hub(mmproj);
    out->mmproj_size = bytes_on_disk(mpath, allow_path);
  }
  if (!lora.empty()) {
    if (!fetch_lora(lora, progress, stop, &out->lora, err, allow_path)) return false;
    // An empty path loads no adapter: never the base model answering for the adapter asked for.
    if (out->lora.empty()) return *err = lora + ": names no LoRA adapter's directory", false;
  }
  bool has_dspark = false;  // named: none other is taken
  for (const auto& ds : out->drafter_sources) has_dspark = has_dspark || ds.kind == "dspark";
  if (repo.id.empty()) {  // a path: only a DSpark drafter beside it to know of
    if (want_dspark && !has_dspark)
      if (const std::string d = dspark_beside(out->model, allow_path); !d.empty()) {
        out->drafters.push_back(d);
        out->drafter_sources.push_back({"dspark", d, bytes_on_disk(d, allow_path)});
        out->notes.push_back("the DSpark drafter beside the model: " + d);
      }
    suggest_drafters(repo, drafters, out, allow_path);
    return true;
  }
  const bool need_mmproj = want_vision && out->mmproj.empty();
  const bool need_mtp = want_mtp && own_mtp == 0 &&
      std::none_of(out->drafter_sources.begin(), out->drafter_sources.end(), [](const auto& ds) { return ds.kind == "mtp"; });
  const bool need_dspark = want_dspark && !has_dspark;
  const auto ggufs = need_mmproj || need_mtp || need_dspark ? hub_ggufs(repo, allow_path) : std::vector<HubGguf>{};
  auto check_paths = [&](const std::vector<std::string>& paths) -> bool {
    if (!allow_path) return true;
    for (const std::string& p : paths)
      if (!allow_path(p)) return *err = p + ": path is not allowed by remote path policy", false;
    return true;
  };
  if (need_mmproj) {
    const std::string own_arch = classify_gguf(out->model, "", 0, fs::file_time_type::min(), allow_path).arch;
    const auto* mm = reads_vision(own_arch.empty() ? repo.info.gguf_architecture : own_arch)
                         ? preferred_companion(ggufs, "mmproj") : nullptr;
    if (mm) {
      std::vector<std::string> paths;
      if (!hub::fetch(repo, mm->choice.parts, &paths, progress, stop, err, allow_path) || !check_paths(paths))
        return false;
      out->mmproj = paths[0];
      out->mmproj_source = repo.id + "/" + mm->choice.file.path;
      out->mmproj_size = mm->choice.total_size;
      out->notes.push_back("the vision projector from " + repo.id + "/" + mm->choice.file.path);
    }
  }
  // unsloth keeps the MTP layer of GGUFs made without it in MTP/.
  const auto* mtp = need_mtp ? preferred_companion(ggufs, "mtp") : nullptr;
  if (mtp) {
    std::vector<std::string> paths;
    if (!hub::fetch(repo, mtp->choice.parts, &paths, progress, stop, err, allow_path) || !check_paths(paths))
      return false;
    out->drafters.push_back(paths[0]);
    out->drafter_sources.push_back({"mtp", repo.id + "/" + mtp->choice.file.path, mtp->choice.total_size});
    out->notes.push_back("the MTP layer from " + repo.id + "/" + mtp->choice.file.path + " (the model's file has none)");
  }
  if (need_dspark) {
    if (const auto* ds = preferred_companion(ggufs, "dspark")) {
      const auto& choice = ds->choice;
      std::vector<std::string> paths;
      if (!hub::fetch(repo, choice.parts, &paths, progress, stop, err, allow_path) || !check_paths(paths)) return false;
      const std::string source = repo.id + "/" + choice.file.path;
      if (is_dspark_file(paths.front(), allow_path)) {
        out->drafters.push_back(paths.front());
        out->drafter_sources.push_back({"dspark", source, choice.total_size});
        out->notes.push_back("the DSpark drafter from " + source);
      } else {
        out->notes.push_back(source + " is not a DSpark drafter: not drafting with it");
      }
    } else {
      for (const auto& item : ggufs)
        if (item.cached && hub::dspark_gguf_name(item.choice.file.path) && !item.cls.arch.empty()) {
          out->notes.push_back(repo.id + "/" + item.choice.file.path + " is not a DSpark drafter: not drafting with it");
          break;
        }
    }
  }
  suggest_drafters(repo, drafters, out, allow_path);
  return true;
}

bool fetch_lora(const std::string& spec, const hub::ProgressFn& progress, const std::atomic<bool>* stop,
                std::string* path, std::string* err, const PathAllowFn& allow_path) {
  // The directory, or one of the two files PEFT saves in it (adapter_config.json, adapter_model.safetensors): their
  // directory, the current one for a bare file name.  Never empty: an empty path asks for no adapter, and the base
  // model would answer in the adapter's name.
  auto take_local = [&](const std::string& p) {
    std::error_code ec;
    std::string resolved = p;
    if (!fs::is_directory(to_path(p), ec)) {
      const fs::path file = to_path(p);
      const fs::path dir = file.has_parent_path() ? file.parent_path() : fs::path(".");
      // The file itself is one of the two, by the file system's identity (ADAPTER_CONFIG.JSON on a volume that
      // ignores case), not by the spelling typed.  Each of the two is a path of its own the policy decides on before
      // it is looked at: comparing identities follows its links and reparse points.
      bool denied = false;
      const auto is = [&](const char* name) {
        const fs::path sibling = dir / name;
        if (allow_path && !allow_path(from_path(sibling))) return denied = true, false;
        std::error_code e;
        return fs::equivalent(file, sibling, e);
      };
      if (!is("adapter_config.json") && !is("adapter_model.safetensors")) {
        if (denied) return *err = p + ": path is not allowed by remote path policy", false;
        return *err = p + ": not a LoRA adapter (its directory, of adapter_config.json and adapter_model.safetensors, "
                          "or one of those two files)",
               false;
      }
      resolved = file.has_parent_path() ? from_path(dir) : std::string(".");
    }
    if (allow_path && !allow_path(resolved))
      return *err = resolved + ": path is not allowed by remote path policy", false;
    *path = resolved;
    return true;
  };
  std::string local;
  const LocalPathResult r1 = local_path_if_exists(spec, allow_path, &local, err);
  if (r1 == LocalPathResult::kDenied) return false;
  if (r1 == LocalPathResult::kFound) return take_local(local);
  hub::Name name;
  std::string active = spec;
  bool parsed = hub::parse_adapter_name(spec, &name);
  if (!parsed) {
    if (const auto unquoted = strip_single_quotes(spec); unquoted && !unquoted->empty()) {
      const LocalPathResult r2 = local_path_if_exists(*unquoted, allow_path, &local, err);
      if (r2 == LocalPathResult::kDenied) return false;
      if (r2 == LocalPathResult::kFound) return take_local(local);
      if (hub::parse_adapter_name(*unquoted, &name)) {
        parsed = true;
        active = *unquoted;
      }
    }
  }
  if (!parsed)
    return *err = spec + ": no such directory, and not a Hugging Face name (org/name, org/name:FOLDER)", false;
  hub::Repo repo;
  if (!hub::describe(name.repo, name.revision, &repo, err, nullptr, allow_path)) return false;
  return fetch_lora_in(repo, active, name.quant, progress, stop, path, err, allow_path);
}

bool fetch_lora_in(const hub::Repo& repo, const std::string& spec, const std::string& folder,
                   const hub::ProgressFn& progress, const std::atomic<bool>* stop, std::string* path,
                   std::string* err, const PathAllowFn& allow_path) {
  // The adapter named: the folder of that path, else the one folder whose last part it is.
  const std::vector<hub::AdapterFolder> all = hub::adapter_folders(repo.id, repo.files);
  auto listing = [&] {
    std::string s;
    for (const hub::AdapterFolder& a : all) s += "\n  " + a.name;
    return s;
  };
  const hub::AdapterFolder* found = nullptr;
  for (const hub::AdapterFolder& a : all)
    if (a.folder == folder) found = &a;
  if (!found && !folder.empty())
    for (const hub::AdapterFolder& a : all)
      if (a.folder.ends_with("/" + folder)) {
        if (found) return *err = spec + ": more than one folder of that name holds an adapter:" + listing(), false;
        found = &a;
      }
  if (!found) {
    if (all.empty())
      return *err = repo.id + ": no LoRA adapter there (adapter_config.json and adapter_model.safetensors)", false;
    return *err = spec + ": no LoRA adapter " + (folder.empty() ? "at the repository's top" : "in that folder") +
                  "; it has:" + listing(),
           false;
  }
  if (!found->weights)
    return *err = found->name + (found->pickle ? ": adapter_model.bin (a pickle) is not read; save the adapter as "
                                                 "safetensors"
                                               : ": no adapter_model.safetensors beside its adapter_config.json"),
           false;
  // The config is read whole by the load (modelfile::read_peft_dir): one past what it reads is refused before either
  // file is fetched.  The transfers keep to the sizes listed (hub::fetch).
  if (found->config->size > modelfile::kPeftConfigMaxBytes)
    return *err = found->name + ": adapter_config.json of " + std::to_string(found->config->size) +
                  " bytes, more than an adapter's config holds (16 MiB): not fetched",
           false;
  const std::vector<hub::RepoFile> files = {*found->config, *found->weights};
  std::vector<std::string> paths;
  if (!hub::fetch(repo, files, &paths, progress, stop, err, allow_path)) return false;
  const std::string dir = from_path(to_path(paths[0]).parent_path());
  if (allow_path) {
    if (!allow_path(dir)) return *err = dir + ": path is not allowed by remote path policy", false;
    for (const std::string& p : paths)
      if (!allow_path(p)) return *err = p + ": path is not allowed by remote path policy", false;
  }
  *path = dir;
  return true;
}

hub::ProgressFn make_progress_logger(const char* prefix, double interval_ms, hub::ProgressFn hook) {
  auto last = std::make_shared<double>(0.0);
  return [prefix, interval_ms, hook = std::move(hook), last](const hub::Progress& p) {
    if (hook) hook(p);
    const double now = os::now_ms();
    if (now - *last < interval_ms && p.done < p.total) return;
    *last = now;
    std::fprintf(stderr, "%s: fetching %s: %s of %s, %.0f MB/s%s\n", prefix, p.file.c_str(), gigabytes(p.done).c_str(),
                 gigabytes(p.total).c_str(), p.bytes_per_s / 1e6, p.checking ? ", checking" : "");
  };
}

bool fetch_named(std::string* model, EngineOptions* eo, std::string* err) {
  auto progress = make_progress_logger("flyx", 2000);
  Fetched f;
  if (!fetch_model(*model, eo->drafter_files, wants_mtp(*eo), progress, nullptr, &f, err, eo->mmproj_file, eo->vision,
                   wants_dspark(*eo, eo->drafter_files), eo->lora_file))
    return false;
  if (f.model != *model) std::fprintf(stderr, "flyx: %s is %s\n", model->c_str(), f.model.c_str());
  for (size_t i = 0; i < eo->drafter_files.size() && i < f.drafters.size(); ++i)
    if (f.drafters[i] != eo->drafter_files[i])
      std::fprintf(stderr, "flyx: %s is %s\n", eo->drafter_files[i].c_str(), f.drafters[i].c_str());
  for (const std::string& n : f.notes) std::fprintf(stderr, "flyx: %s\n", n.c_str());
  *model = f.model;
  eo->drafter_files = f.drafters;
  eo->mmproj_file = f.mmproj;
  if (f.lora != eo->lora_file) std::fprintf(stderr, "flyx: %s is %s\n", eo->lora_file.c_str(), f.lora.c_str());
  eo->lora_file = f.lora;
  return true;
}

}  // namespace flyx::apps
