// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Hugging Face Hub cache layout (models--<org>--<name>/{blobs,snapshots,refs}): inspection, placement, and removal.
#include "core/hub/hub_internal.h"

#include <algorithm>
#include <cctype>
#include <deque>
#include <fstream>
#include <mutex>

#include "core/interfaces/json_depth.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::hub {

namespace fs = std::filesystem;
using os::from_path;
using os::to_path;

namespace {

ResolvedSetting resolve_hf_home() {
  const std::string home = os::getenv_str(env::kHfHome);
  if (!home.empty()) return {home, env::kHfHome};
  const std::string xdg = os::getenv_str(env::kXdgCacheHome);
  if (!xdg.empty()) return {xdg + "/huggingface", env::kXdgCacheHome};
  return {os::home_dir() + "/.cache/huggingface", "default"};
}

std::string token() {
  std::string t = os::getenv_str(env::kHfToken);
  if (t.empty()) {
    std::ifstream f(to_path(detail::hf_home() + "/token"));
    std::getline(f, t);
  }
  while (!t.empty() && std::isspace(static_cast<unsigned char>(t.back()))) t.pop_back();
  return t;
}

bool is_symlink_or_junction(const fs::path& p, std::error_code& ec) {
  const fs::file_status st = fs::symlink_status(p, ec);
  if (st.type() == fs::file_type::not_found) {
    ec.clear();
    return false;
  }
  if (ec) return false;
  if (fs::is_symlink(st)) return true;
#if defined(_MSC_VER)
  if (st.type() == fs::file_type::junction) return true;
#endif
  // Fallback for Windows STLs where NTFS directory junctions report a non-directory, non-regular file_type.
  if (st.type() != fs::file_type::directory && st.type() != fs::file_type::regular) {
    std::error_code e2;
    return fs::is_directory(p, e2) && !e2;
  }
  return false;
}

bool is_real_dir(const fs::path& p, const CacheFs& ops) {
  std::error_code ec;
  if (ops.is_symlink(p, ec) || ec) return false;
  ec.clear();
  return ops.is_directory(p, ec) && !ec;
}

CacheFs resolve_cache_fs(const CacheFs* fs_ops) {
  CacheFs ops;
  ops.is_symlink = [](const fs::path& p, std::error_code& ec) { return is_symlink_or_junction(p, ec); };
  ops.is_directory = [](const fs::path& p, std::error_code& ec) { return fs::is_directory(p, ec); };
  ops.is_regular_file = [](const fs::path& p, std::error_code& ec) { return fs::is_regular_file(p, ec); };
  ops.file_size = [](const fs::path& p, std::error_code& ec) { return uint64_t(fs::file_size(p, ec)); };
  ops.last_write_time = [](const fs::path& p, std::error_code& ec) { return fs::last_write_time(p, ec); };
  ops.for_each_dir_entry = [](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                              std::error_code& ec) {
    for (fs::directory_iterator it(p, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end;
         it.increment(ec)) {
      if (!visit(it->path())) break;
    }
  };
  ops.read_text = [](const fs::path& p) { return detail::read_text(p); };
  ops.weakly_canonical = [](const fs::path& p, std::error_code& ec) { return fs::weakly_canonical(p, ec); };
  ops.equivalent = [](const fs::path& a, const fs::path& b, std::error_code& ec) { return fs::equivalent(a, b, ec); };
  if (!fs_ops) return ops;
  if (fs_ops->is_symlink) {
    ops.is_symlink = fs_ops->is_symlink;
  } else if (fs_ops->is_directory) {
    // Mock-filesystem tests overriding is_directory on virtual paths do not exist on real disk.
    ops.is_symlink = [](const fs::path&, std::error_code&) { return false; };
  }
  if (fs_ops->is_directory) ops.is_directory = fs_ops->is_directory;
  if (fs_ops->is_regular_file) ops.is_regular_file = fs_ops->is_regular_file;
  if (fs_ops->file_size) ops.file_size = fs_ops->file_size;
  if (fs_ops->last_write_time) ops.last_write_time = fs_ops->last_write_time;
  if (fs_ops->for_each_dir_entry) ops.for_each_dir_entry = fs_ops->for_each_dir_entry;
  if (fs_ops->read_text) ops.read_text = fs_ops->read_text;
  if (fs_ops->weakly_canonical) ops.weakly_canonical = fs_ops->weakly_canonical;
  if (fs_ops->equivalent) ops.equivalent = fs_ops->equivalent;
  ops.root_override = fs_ops->root_override;
  return ops;
}

// The files of a snapshot on disk (file links followed, directory symlinks/junctions never descended into), sizes only.
// Every child is authorized via allow_path BEFORE any target-following stat, size lookup, or directory descent.
// `bounded` caps depth/entries for interactive hub::cached() listings; authoritative single-snapshot resolution passes
// bounded=false so deep valid snapshots are never silently truncated.
std::vector<RepoFile> snapshot_files(const fs::path& snap, const PathAllowFn& allow_path, const CacheFs& ops,
                                     bool bounded) {
  constexpr int kMaxDepth = 16;
  constexpr int kMaxEntries = 50000;
  std::vector<RepoFile> files;
  if (!is_real_dir(snap, ops)) return files;
  std::deque<std::pair<fs::path, int>> queue;
  queue.emplace_back(snap, 0);
  int seen = 0;
  while (!queue.empty() && (!bounded || seen < kMaxEntries)) {
    auto [cur, depth] = std::move(queue.front());
    queue.pop_front();
    std::error_code ec;
    ops.for_each_dir_entry(
        cur,
        [&](const fs::path& child) {
          if (bounded && ++seen > kMaxEntries) return false;
          const std::string child_str = from_path(child);
          if (allow_path && !allow_path(child_str)) return true;
          std::error_code e2;
          const bool link = ops.is_symlink(child, e2);
          if (e2) return true;
          if (!link && ops.is_directory(child, e2)) {
            if (!e2 && (!bounded || depth + 1 < kMaxDepth)) queue.emplace_back(child, depth + 1);
            return true;
          }
          e2.clear();
          if (!ops.is_regular_file(child, e2) || e2) return true;  // follows file links; skips directory links & broken links
          RepoFile f;
          f.path = from_path(child.lexically_relative(snap));  // fs::relative would follow the link to the blob
          std::replace(f.path.begin(), f.path.end(), '\\', '/');
          f.size = ops.file_size(child, e2);
          if (!e2) files.push_back(std::move(f));
          return true;
        },
        ec);
  }
  std::sort(files.begin(), files.end(), [](const RepoFile& a, const RepoFile& b) { return a.path < b.path; });
  return files;
}

// A snapshot's config.json's architectures, if it has one.
std::vector<std::string> architectures_of(const fs::path& snap, const PathAllowFn& allow_path, const CacheFs& ops) {
  std::vector<std::string> out;
  if (!is_real_dir(snap, ops)) return out;
  const fs::path cfg = snap / "config.json";
  if (allow_path && !allow_path(from_path(cfg))) return out;
  const std::string text = ops.read_text(cfg);
  if (text.empty() || text.size() > interfaces::kMaxConfigBytes || !interfaces::json_depth_within(text)) return out;
  const auto j = nlohmann::json::parse(text, nullptr, false);
  if (j.is_object() && j.contains("architectures") && j["architectures"].is_array())
    for (const auto& a : j["architectures"])
      if (a.is_string()) out.push_back(a.get<std::string>());
  return out;
}

// Whether `p` is `root` or lies under it, both made absolute with their links followed.
bool inside(const fs::path& p, const fs::path& root) {
  std::error_code ec, e2;
  const fs::path a = fs::weakly_canonical(p, ec), r = fs::weakly_canonical(root, e2);
  if (ec || e2 || r.empty()) return false;
  const fs::path rel = a.lexically_relative(r);
  return !rel.empty() && *rel.begin() != "..";
}

// A standalone copy has no link back to its blob. Verify the contents before reclaiming either file.
bool same_contents(const fs::path& a, const fs::path& b, uint64_t size) {
  constexpr uint64_t kChunk = uint64_t(1) << 20;
  std::ifstream fa(a, std::ios::binary), fb(b, std::ios::binary);
  if (!fa || !fb) return false;
  std::vector<char> x(size_t(std::min(size, kChunk))), y(x.size());
  while (size) {
    const size_t n = size_t(std::min(size, kChunk));
    if (!fa.read(x.data(), n) || !fb.read(y.data(), n) || !std::equal(x.begin(), x.begin() + n, y.begin())) return false;
    size -= n;
  }
  return true;
}

}  // namespace

namespace detail {

std::string hf_home() { return resolve_hf_home().value; }

std::vector<std::string> auth_headers(const ResolvedSetting& ep) {
  // Send the host's Hugging Face token only over HTTPS and only to the official Hub origin or an operator-configured
  // HF_ENDPOINT environment mirror—never over cleartext HTTP or to an arbitrary runtime-configured mirror.
  if (ep.value.rfind("https://", 0) != 0) return {};
  if (ep.value != kDefaultEndpoint && ep.source != env::kHfEndpoint) return {};
  const std::string t = token();
  return t.empty() ? std::vector<std::string>{} : std::vector<std::string>{"Authorization: Bearer " + t};
}

fs::path repo_dir(const std::string& repo) {
  const size_t slash = repo.find('/');
  return to_path(cache_dir()) / to_path("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1));
}

std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  const std::streamoff size = f.tellg();
  if (size < 0 || uint64_t(size) > interfaces::kMaxConfigBytes) return {};
  std::string s(size_t(size), '\0');
  if (!f.seekg(0).read(s.data(), size)) return {};
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
  return s;
}

bool has_size(const fs::path& path, uint64_t size) {
  std::error_code ec;
  return fs::is_regular_file(path, ec) && fs::file_size(path, ec) == size && !ec;
}

bool from_cache(const std::string& repo, const std::string& revision, Repo* out, const PathAllowFn& allow_path,
                const CacheFs* fs_ops) {
  if (!valid_repo(repo) || !valid_word(revision)) return false;
  const CacheFs ops = resolve_cache_fs(fs_ops);
  const std::string root_str = ops.root_override.empty() ? cache_dir() : ops.root_override;
  if (root_str.empty() || (allow_path && !allow_path(root_str))) return false;
  const size_t slash = repo.find('/');
  const fs::path dir = to_path(root_str) / to_path("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1));
  if (allow_path && !allow_path(from_path(dir))) return false;
  if (!is_real_dir(dir, ops)) return false;
  const fs::path snaps = dir / "snapshots";
  if (allow_path && !allow_path(from_path(snaps))) return false;
  if (!is_real_dir(snaps, ops)) return false;
  std::error_code ec;
  std::string commit;
  if (is_commit(revision)) {
    const fs::path direct = snaps / to_path(revision);
    if (!allow_path || allow_path(from_path(direct))) {
      if (ops.is_symlink(direct, ec) || ec) return false;
      if (is_real_dir(direct, ops)) commit = revision;
    }
  }
  if (commit.empty()) {
    const fs::path refs = dir / "refs";
    ec.clear();
    if ((allow_path && !allow_path(from_path(refs))) || ops.is_symlink(refs, ec) || ec) return false;
    const fs::path ref = refs / to_path(revision);
    if (allow_path && !allow_path(from_path(ref))) return false;
    commit = ops.read_text(ref);
  }
  if (!is_commit(commit)) return false;
  const fs::path snap = snaps / to_path(commit);
  if (allow_path && !allow_path(from_path(snap))) return false;
  if (!is_real_dir(snap, ops)) return false;
  out->commit = commit;
  out->files = snapshot_files(snap, allow_path, ops, /*bounded=*/false);
  out->architectures = architectures_of(snap, allow_path, ops);
  out->from_cache = true;
  return true;
}

void write_ref(const Repo& r) {
  if (r.revision.empty() || r.revision == r.commit || !valid_repo(r.id) || !valid_word(r.revision) ||
      !is_commit(r.commit))
    return;
  const fs::path refs = repo_dir(r.id) / "refs";
  std::error_code ec;
  fs::create_directories(refs, ec);
  std::ofstream(refs / to_path(r.revision), std::ios::binary | std::ios::trunc) << r.commit;
}

// Places a fetched blob at its snapshot path: a link to it as huggingface_hub makes, else (Windows without the right
// to make links) a hard link, else a copy.
bool place(const fs::path& blob, const fs::path& dst, std::string* err) {
  std::error_code ec;
  fs::create_directories(dst.parent_path(), ec);
  fs::remove(dst, ec);
  fs::create_symlink(fs::relative(blob, dst.parent_path(), ec), dst, ec);
  if (!ec) return true;
  ec.clear();
  fs::create_hard_link(blob, dst, ec);
  if (!ec) return true;
  ec.clear();
  fs::copy_file(blob, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) return *err = from_path(dst) + ": " + ec.message(), false;
  return true;
}

}  // namespace detail

ResolvedSetting resolve_cache_dir() {
  ResolvedSetting cache{os::getenv_str(env::kHfHubCache), env::kHfHubCache};
  if (cache.value.empty()) {
    cache = resolve_hf_home();
    cache.value += "/hub";
  }
#if defined(_WIN32)
  std::replace(cache.value.begin(), cache.value.end(), '/', '\\');  // as Windows shows paths (huggingface_hub's too)
#endif
  return cache;
}

std::string cache_dir() { return resolve_cache_dir().value; }
std::string cache_source() { return resolve_cache_dir().source; }

namespace {

std::mutex& endpoint_mutex() {
  static std::mutex mu;
  return mu;
}
std::string& endpoint_override() {
  static std::string ep;
  return ep;
}

}  // namespace

bool normalize_endpoint(const std::string& raw, std::string* normalized, std::string* err) {
  size_t a = 0, b = raw.size();
  while (a < b && std::isspace(static_cast<unsigned char>(raw[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(raw[b - 1]))) --b;
  std::string s = raw.substr(a, b - a);
  if (s.empty()) {
    if (normalized) normalized->clear();
    return true;
  }
  for (unsigned char c : s) {
    if (c <= 0x20 || c == 0x7f || c == '?' || c == '#' || c == '\\' || c == '@') {
      if (err) *err = "endpoint: invalid URL character";
      return false;
    }
  }
  const size_t scheme_sep = s.find("://");
  std::string scheme, rest;
  if (scheme_sep != std::string::npos) {
    scheme = detail::lower(s.substr(0, scheme_sep));
    if (scheme != "https" && scheme != "http") {
      if (err) *err = "endpoint: URL must use http:// or https://";
      return false;
    }
    rest = s.substr(scheme_sep + 3);
  } else {
    if (s.find(':') != std::string::npos) {
      // Allow host:port without scheme, reject other scheme-like prefixes (e.g. javascript:...).
      const size_t colon = s.find(':');
      const size_t slash = s.find('/');
      if (slash == std::string::npos || colon < slash) {
        const std::string port = s.substr(colon + 1, slash == std::string::npos ? std::string::npos : slash - colon - 1);
        const bool numeric = !port.empty() && std::all_of(port.begin(), port.end(),
                                                          [](unsigned char c) { return std::isdigit(c) != 0; });
        if (!numeric) {
          if (err) *err = "endpoint: URL must use http:// or https://";
          return false;
        }
      }
    }
    scheme = "https";
    rest = s;
  }
  while (!rest.empty() && rest.back() == '/') rest.pop_back();
  const size_t slash = rest.find('/');
  const std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
  if (hostport.empty() || hostport.front() == '.' || hostport.front() == ':' || hostport.find("..") != std::string::npos) {
    if (err) *err = "endpoint: missing or invalid host name";
    return false;
  }
  if (normalized) *normalized = scheme + "://" + rest;
  return true;
}

bool set_endpoint(const std::string& url, std::string* err) {
  std::string norm;
  if (!normalize_endpoint(url, &norm, err)) return false;
  std::lock_guard<std::mutex> lock(endpoint_mutex());
  endpoint_override() = std::move(norm);
  return true;
}

ResolvedSetting resolve_endpoint() {
  std::lock_guard<std::mutex> lock(endpoint_mutex());
  if (!endpoint_override().empty()) return {endpoint_override(), "setting"};
  std::string e = os::getenv_str(env::kHfEndpoint);
  while (!e.empty() && e.back() == '/') e.pop_back();
  if (e.empty()) return {kDefaultEndpoint, "default"};
  return {std::move(e), env::kHfEndpoint};
}

std::string endpoint() { return resolve_endpoint().value; }
std::string endpoint_source() { return resolve_endpoint().source; }

bool offline() { return os::env_bool(env::kHfHubOffline).value_or(false); }

std::string snapshot_dir(const Repo& r) { return from_path(detail::repo_dir(r.id) / "snapshots" / r.commit); }

bool in_cache(const std::string& repo, const std::string& revision, Repo* out, const PathAllowFn& allow_path) {
  Repo r;
  r.id = repo, r.revision = revision.empty() ? "main" : revision;
  if (!detail::from_cache(repo, r.revision, &r, allow_path)) return false;
  *out = std::move(r);
  return true;
}

std::vector<Cached> cached(const PathAllowFn& allow_path, const CacheFs* fs_ops) {
  const CacheFs ops = resolve_cache_fs(fs_ops);
  const std::string root_str = ops.root_override.empty() ? cache_dir() : ops.root_override;
  if (root_str.empty() || (allow_path && !allow_path(root_str))) return {};
  std::vector<Cached> out;
  std::error_code ec;
  ops.for_each_dir_entry(
      to_path(root_str),
      [&](const fs::path& repo_entry) {
        const std::string dir = from_path(repo_entry.filename());
        if (!dir.starts_with("models--")) return true;
        const std::string rest = dir.substr(8);
        const size_t sep = rest.find("--");
        if (sep == std::string::npos) return true;
        const std::string repo = rest.substr(0, sep) + "/" + rest.substr(sep + 2);
        if (!detail::valid_repo(repo)) return true;
        if (allow_path && !allow_path(from_path(repo_entry))) return true;
        if (!is_real_dir(repo_entry, ops)) return true;
        const fs::path snaps = repo_entry / "snapshots";
        if (allow_path && !allow_path(from_path(snaps))) return true;
        if (!is_real_dir(snaps, ops)) return true;
        const fs::path refs = repo_entry / "refs";
        const fs::path ref_main = refs / "main";
        std::error_code ref_ec;
        const bool refs_ok = (!allow_path || (allow_path(from_path(refs)) && allow_path(from_path(ref_main)))) &&
                             !ops.is_symlink(refs, ref_ec) && !ref_ec;
        const std::string main = refs_ok ? ops.read_text(ref_main) : "";
        struct Snapshot {
          std::string commit;
          fs::file_time_type made;
        };
        std::vector<Snapshot> found;
        std::error_code e2;
        ops.for_each_dir_entry(
            snaps,
            [&](const fs::path& s_path) {
              const std::string commit = from_path(s_path.filename());
              if (!detail::is_commit(commit)) return true;
              if (allow_path && !allow_path(from_path(s_path))) return true;
              if (is_real_dir(s_path, ops)) {
                std::error_code e3;
                found.push_back({commit, ops.last_write_time(s_path, e3)});
              }
              return true;
            },
            e2);
        std::sort(found.begin(), found.end(), [&](const Snapshot& a, const Snapshot& b) {
          if ((a.commit == main) != (b.commit == main)) return a.commit == main;
          return a.made > b.made;
        });
        for (const Snapshot& s : found) {
          const fs::path snap_dir = snaps / to_path(s.commit);
          Cached c;
          c.repo = repo, c.commit = s.commit, c.current = s.commit == main;
          c.dir = from_path(snap_dir);
          c.files = snapshot_files(snap_dir, allow_path, ops, /*bounded=*/true);
          c.architectures = architectures_of(snap_dir, allow_path, ops);
          if (!c.files.empty()) out.push_back(std::move(c));
        }
        return true;
      },
      ec);
  std::stable_sort(out.begin(), out.end(), [](const Cached& a, const Cached& b) { return a.repo < b.repo; });
  return out;
}

bool remove_cached(const std::string& repo, const std::string& commit, const std::vector<std::string>& paths,
                   uint64_t* freed, std::string* err, const PathAllowFn& allow_path, const CacheFs* fs_ops) {
  *freed = 0;
  if (!detail::valid_repo(repo) || !detail::is_commit(commit))
    return *err = "not a cached snapshot: " + repo + "@" + commit, false;
  const CacheFs ops = resolve_cache_fs(fs_ops);
  const std::string root_str = ops.root_override.empty() ? cache_dir() : ops.root_override;
  if (root_str.empty()) return *err = "cache directory is not configured", false;
  auto check_allow = [&](const fs::path& p) -> bool {
    if (!allow_path || allow_path(from_path(p))) return true;
    *err = from_path(p) + ": path is not allowed by remote path policy";
    return false;
  };
  const size_t slash = repo.find('/');
  const fs::path dir = to_path(root_str) / to_path("models--" + repo.substr(0, slash) + "--" + repo.substr(slash + 1));
  const fs::path snaps = dir / "snapshots", snap = snaps / commit, blobs = dir / "blobs";
  if (!check_allow(to_path(root_str)) || !check_allow(dir) || !check_allow(snaps) || !check_allow(snap) ||
      !check_allow(blobs))
    return false;
  // Independently reject linked repo/snapshots/commit/blobs roots so cache deletion can never remove external files.
  if (!is_real_dir(dir, ops) || !is_real_dir(snaps, ops) || !is_real_dir(snap, ops))
    return *err = repo + "@" + commit + ": snapshot directory is not a real directory in the cache", false;
  std::error_code blobs_ec;
  if (ops.is_symlink(blobs, blobs_ec) || blobs_ec)
    return *err = from_path(blobs) + ": cache blobs directory must be a real directory", false;
  const bool blobs_dir = is_real_dir(blobs, ops);
  std::error_code ec;
  // A file's last name going frees its bytes; a link, or one name of a file that has others, frees none.
  auto remove_one = [&](const fs::path& p) {
    if (!check_allow(p)) return false;
    std::error_code e;
    const bool link = fs::is_symlink(fs::symlink_status(p, e));
    const uint64_t bytes = link ? 0 : fs::file_size(p, e);
    const bool last = !link && !e && fs::hard_link_count(p, e) == 1;
    if (!fs::remove(p, e) || e) return *err = from_path(p) + ": " + (e ? e.message() : "not removed"), false;
    if (last) *freed += bytes;
    return true;
  };
  // Whether an entry of any of the repository's snapshots still reads `blob`: a link to it, or a hard link.
  // Never descends into directory symlinks/junctions; if allow_path denies any snapshot path, enumeration fails,
  // or canonicalization/identity comparison reports an error, fails closed and retains the blob.
  auto referenced = [&](const fs::path& blob) {
    if (allow_path && (!allow_path(from_path(snaps)) || !allow_path(from_path(blob)))) return true;
    std::error_code e;
    const fs::path target = ops.weakly_canonical(blob, e);
    if (e) return true;
    std::deque<fs::path> queue;
    queue.push_back(snaps);
    while (!queue.empty()) {
      const fs::path cur = std::move(queue.front());
      queue.pop_front();
      if (allow_path && !allow_path(from_path(cur))) return true;
      std::error_code e2;
      bool found = false;
      auto inspect = [&](const fs::path& child) -> bool {
        if (allow_path && !allow_path(from_path(child))) return found = true, false;
        std::error_code e3;
        const bool link_or_junc = ops.is_symlink(child, e3);
        if (e3) return found = true, false;
        if (!link_or_junc) {
          const bool is_dir = ops.is_directory(child, e3);
          if (e3) return found = true, false;
          if (is_dir) {
            queue.push_back(child);
            return true;
          }
          e3.clear();
          const bool is_reg = ops.is_regular_file(child, e3);
          if (e3) return found = true, false;
          if (is_reg) {
            const bool same = ops.equivalent(child, blob, e3);
            if (e3 || same) return found = true, false;
          }
          return true;
        }
        const fs::path resolved = ops.weakly_canonical(child, e3);
        if (e3 || resolved == target) return found = true, false;
        return true;
      };
      if (fs_ops && fs_ops->for_each_dir_entry) {
        ops.for_each_dir_entry(cur, inspect, e2);
      } else {
        fs::directory_iterator it(cur, e2), end;
        if (e2) return true;
        for (; !e2 && it != end; it.increment(e2)) {
          if (!inspect(it->path())) break;
        }
      }
      if (found || e2) return true;
    }
    return false;
  };
  for (const std::string& path : paths) {
    if (!detail::safe_path(path)) return *err = "not a repository's file: " + path, false;
    const fs::path entry = snap / to_path(path);
    for (fs::path p = entry; !p.empty() && p != snap; p = p.parent_path())
      if (!check_allow(p)) return false;
    for (fs::path d = entry.parent_path(); !d.empty() && d != snap; d = d.parent_path())
      if (!is_real_dir(d, ops)) return *err = from_path(entry) + " is outside the cache", false;
    const fs::file_status st = fs::symlink_status(entry, ec);
    std::error_code dir_ec;
    const bool entry_dir_link =
        ops.is_symlink(entry, dir_ec) && (dir_ec || !fs::is_symlink(st) || fs::is_directory(entry, dir_ec));
    if (ec || !fs::exists(st) || fs::is_directory(st) || entry_dir_link)
      return *err = repo + "@" + commit + ": " + path + " is not in the cache", false;
    if (!inside(entry.parent_path(), snap)) return *err = from_path(entry) + " is outside the cache", false;
    // The blob behind the entry: what the link names; else the blob that is the same file (a hard link); else the
    // blob the entry was copied from (place() copies where it may make neither).
    fs::path blob;
    const uint64_t size = fs::file_size(entry, ec);
    if (blobs_dir && fs::is_symlink(st)) {
      blob = ops.weakly_canonical(entry, ec);
      std::error_code bec;
      const fs::path blobs_canon = ops.weakly_canonical(blobs, bec);
      const std::string id = from_path(blob.filename());
      std::error_code st_ec;
      if (ec || bec || blob == blobs_canon || blob.parent_path() != blobs_canon || !inside(blob, blobs) ||
          (allow_path && !allow_path(from_path(blob))) || !fs::is_regular_file(fs::symlink_status(blob, st_ec)) ||
          st_ec || (id.size() != 40 && id.size() != 64) || !detail::hex_id(id))
        blob.clear();  // only delete an existing allowed hashed regular blob directly inside blobs/
    } else if (blobs_dir && !fs::is_symlink(st)) {
      const bool hard = fs::hard_link_count(entry, ec) > 1;
      for (fs::directory_iterator b(blobs, ec), bend; !ec && b != bend && blob.empty(); b.increment(ec)) {
        if (allow_path && !allow_path(from_path(b->path()))) continue;
        std::error_code e2;
        const std::string id = from_path(b->path().filename());
        if ((id.size() != 40 && id.size() != 64) || !detail::hex_id(id) ||
            !fs::is_regular_file(b->symlink_status(e2)) || b->file_size(e2) != size)
          continue;
        const bool same = ops.equivalent(entry, b->path(), e2);
        if (!e2 && (hard ? same : !same && same_contents(entry, b->path(), size))) blob = b->path();
      }
    }
    ec.clear();
    if (!remove_one(entry)) return false;
    if (!blob.empty() && !referenced(blob) && !remove_one(blob)) return false;
    for (fs::path d = entry.parent_path(); d != snap && inside(d, snap); d = d.parent_path()) {
      if (allow_path && !allow_path(from_path(d))) break;
      if (!fs::is_empty(d, ec) || ec || !fs::remove(d, ec)) break;  // the directories it leaves empty
    }
  }
  return true;
}

}  // namespace flyx::hub
