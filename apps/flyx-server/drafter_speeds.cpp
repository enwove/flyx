// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/drafter_speeds.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::server {

namespace {

namespace fs = std::filesystem;
using os::to_path;

struct Sum {
  double tokens = 0, decode_ms = 0;
};
using Key = std::pair<std::string, std::string>;  // model, drafter
using Sums = std::map<Key, Sum>;

// The file's sums; none when it is missing or is not version 1 of ours throughout (the next write replaces it).
Sums read_file() {
  std::ifstream f(to_path(drafter_speeds_path()), std::ios::binary);
  if (!f) return {};
  const nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
  if (!j.is_object() || !j.contains("version") || j["version"] != 1 || !j.contains("speeds") ||
      !j["speeds"].is_array())
    return {};
  Sums out;
  for (const auto& e : j["speeds"]) {
    const auto text = [&](const char* k) { return e.contains(k) && e[k].is_string(); };
    const auto positive = [&](const char* k) { return e.contains(k) && e[k].is_number() && e[k].get<double>() > 0; };
    if (!e.is_object() || !text("model") || !text("drafter") || !positive("tokens") || !positive("decode_ms") ||
        !out.emplace(Key{e["model"].get<std::string>(), e["drafter"].get<std::string>()},
                     Sum{e["tokens"].get<double>(), e["decode_ms"].get<double>()})
             .second)
      return {};
  }
  return out;
}

// Folds b into a.
void add_sums(Sums& a, const Sums& b) {
  for (const auto& [k, m] : b) a[k].tokens += m.tokens, a[k].decode_ms += m.decode_ms;
}

// `add` added to the file's sums as they are now (another server's included), under a lock every server sharing the
// file takes, written through a file of this process's own: the sums written, or none when the lock was not had
// within a few seconds or the write did not complete (the file then as it was).
std::optional<Sums> write_added(const fs::path& p, const Sums& add) {
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  os::FileLock lock(os::from_path(p) + ".lock");
  std::string err;
  for (int i = 0;; ++i) {
    const os::FileLock::Result r = lock.try_lock(&err);
    if (r == os::FileLock::Result::kLocked) break;
    if (r == os::FileLock::Result::kError || i == 80) return std::nullopt;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  Sums all = read_file();
  add_sums(all, add);
  nlohmann::json speeds = nlohmann::json::array();
  for (const auto& [k, m] : all)
    speeds.push_back({{"model", k.first}, {"drafter", k.second}, {"tokens", m.tokens}, {"decode_ms", m.decode_ms}});
  fs::path tmp = p;
  tmp += ".tmp." + std::to_string(os::process_id());
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f << nlohmann::json{{"version", 1}, {"speeds", speeds}}.dump(1);
    f.close();
    if (!f) {
      fs::remove(tmp, ec);
      return std::nullopt;
    }
  }
  fs::rename(tmp, p, ec);
  if (ec) {
    fs::remove(tmp, ec);
    return std::nullopt;
  }
  return all;
}

class Speeds {
 public:
  ~Speeds() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (writer_.joinable()) writer_.join();
    flush();
  }
  // A request's end: in memory alone (no file is read or written on the scheduler's thread).
  void add(const std::string& model, const std::vector<std::string>& drafters, int tokens, double decode_ms) {
    std::lock_guard<std::mutex> lock(mu_);
    for (const std::string& d : drafters) {
      Sum& m = pending_[{model, d}];
      m.tokens += tokens, m.decode_ms += decode_ms;
    }
    if (!writer_.joinable()) writer_ = std::thread([this] { run(); });
  }
  double speed(const std::string& model, const std::string& drafter) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!read_) file_ = read_file(), read_ = true;
    Sum s;
    for (const Sums* sums : {&file_, &writing_, &pending_})
      for (const auto& [k, m] : *sums)
        if (k.second == drafter && (k.first == model || k.first.rfind(model + ":", 0) == 0))
          s.tokens += m.tokens, s.decode_ms += m.decode_ms;
    return s.decode_ms > 0 ? s.tokens * 1000.0 / s.decode_ms : 0.0;
  }
  // Writes the runs not written yet; false when the write failed (they are kept for the next).
  bool flush() {
    std::lock_guard<std::mutex> one(flush_mu_);
    {
      std::lock_guard<std::mutex> lock(mu_);
      writing_.swap(pending_);
    }
    if (writing_.empty()) return true;
    const fs::path p = to_path(drafter_speeds_path());
    std::optional<Sums> all;
    if (!p.empty()) all = write_added(p, writing_);
    std::lock_guard<std::mutex> lock(mu_);
    if (p.empty()) {  // no configuration directory: this process's runs alone, in memory
      add_sums(file_, writing_);
    } else if (all) {
      file_ = std::move(*all), read_ = true;
    } else {
      add_sums(pending_, writing_);
    }
    writing_.clear();
    return p.empty() || all.has_value();
  }

 private:
  void run() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
      cv_.wait_for(lock, std::chrono::seconds(30), [this] { return stop_; });
      lock.unlock();
      flush();
      lock.lock();
    }
  }
  std::mutex mu_, flush_mu_;
  std::condition_variable cv_;
  std::thread writer_;
  bool read_ = false, stop_ = false;
  // The file's sums as last read or written; the runs being written; those not written yet.
  Sums file_, writing_, pending_;
};

Speeds& speeds() {
  static Speeds s;
  return s;
}

}  // namespace

std::string drafter_speeds_path() {
  const std::string base = os::config_dir();
  return base.empty() ? "" : base + "/flyx/drafter_speeds.json";
}

void add_drafter_speed(const std::string& model, const std::vector<std::string>& drafters, int tokens,
                       double decode_ms) {
  if (model.empty() || drafters.empty() || tokens <= 0 || !(decode_ms > 0)) return;
  speeds().add(model, drafters, tokens, decode_ms);
}

double drafter_speed(const std::string& model, const std::string& drafter) { return speeds().speed(model, drafter); }

nlohmann::ordered_json known_drafters_json(const std::string& model, const std::vector<apps::KnownDrafter>& known) {
  std::vector<std::pair<const apps::KnownDrafter*, double>> by;
  for (const apps::KnownDrafter& k : known) by.push_back({&k, drafter_speed(model, k.repo)});
  std::stable_sort(by.begin(), by.end(), [](const auto& a, const auto& b) {
    return a.first->exact != b.first->exact ? a.first->exact : a.second > b.second;
  });
  using ojson = nlohmann::ordered_json;
  ojson out = ojson::array();
  for (const auto& [k, tok_s] : by)
    out.push_back({{"repo", k->repo}, {"kind", k->kind}, {"base", k->base}, {"exact", k->exact},
                   {"relation", k->relation.empty() ? ojson(nullptr) : ojson(k->relation)},
                   {"size", k->size ? ojson(*k->size) : ojson(nullptr)}, {"tok_s", tok_s > 0 ? ojson(tok_s) : ojson(nullptr)}});
  return out;
}

bool flush_drafter_speeds() { return speeds().flush(); }

}  // namespace flyx::server
