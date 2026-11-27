// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The model flyx serve serves, and loading another while it runs. flyx serve without a model starts
// with none (or with the one it served last, remembered in the settings file); the dashboard's model panel names one
// (POST /v1/load), and the host fetches it from the Hub while the current model still answers, then ends the current
// one's requests, waits for its memory to come back, and loads the new one.  If the new one fails to load, no model is
// served until another is loaded.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "apps/common/model_source.h"
#include "core/exec/scheduler.h"

namespace flyx::server {

class ModelHost {
 public:
  struct Served {
    std::shared_ptr<Scheduler> sched;  // nullptr: none
    std::string name;              // the API's model id
  };
  struct Want {
    std::string model;                  // as named: a path or a Hub name
    std::vector<std::string> drafters;  // likewise
    EngineOptions eo;                   // the server's engine options, with what the panel changed
    std::string name;                   // the API's model id; empty: the file's name
    // The drafters' backend options the panel chose ("full", "nvfp4" / "q8" / "exact"; empty: the backend's defaults),
    // also folded into eo.backend_options; kept apart so GET /v1/load can say what was asked.
    std::string draft_head{}, draft_weights{};
  };

  explicit ModelHost(const SchedulerOptions& so) : so_(so) {}
  ~ModelHost();
  ModelHost(const ModelHost&) = delete;
  ModelHost& operator=(const ModelHost&) = delete;

  // What requests go to now.  Any thread; a request keeps what it got while it runs.
  Served served() const;
  // Loads `w` on a thread of its own; false (*err) while another load is under way.
  bool start(const Want& w, std::string* err);
  // Loads `w` here (flyx serve MODEL, before it listens).
  bool run(const Want& w, std::string* err);
  // Stops a fetch under way (what came stays for the next try); a model already loading still loads.
  void cancel();
  // Ends the model's requests and lets it go: none served until another is loaded.  The settings file still names
  // it.  false (*err) while a load is under way.
  bool unload(std::string* err);
  // Runs `fn` with the files the served model is read from (its file, its drafters' files or directories) while no
  // load can start; false (*err), without running it, while one is under way.
  bool with_files_in_use(const std::function<void(const std::vector<std::string>&)>& fn, std::string* err);
  // GET /v1/load: state ("none", "fetching", "loading", "ready", "failed"), what is wanted, the fetch's progress, the
  // last error, notes, the served model's drafters and those published for it, and what the settings file remembers.
  ojson status() const;

  // The settings file holds the last model, Hub endpoint and dashboard network preference.
  static std::string settings_path();
  static bool remembered(Want* w, const EngineOptions& base);
  static std::string remembered_hub_endpoint();
  static bool save_hub_endpoint(const std::string& endpoint, std::string* err = nullptr);
  static bool lan_access();
  static bool save_lan_access(bool enabled, std::string* err);
  static bool remote_manage();
  static bool save_remote_manage(bool enabled, std::string* err);
  // Atomically update lan_access and/or remote_manage in serve.json under settings_mutex(), invoking
  // on_remote_manage_committed before releasing the lock so in-memory state stays ordered with disk.
  static bool save_network_settings(std::optional<bool> lan_access, std::optional<bool> remote_manage,
                                    const std::function<void(bool)>& on_remote_manage_committed,
                                    bool* out_lan_access, std::string* err);

 private:
  struct BusyGuard {
    ModelHost* host = nullptr;
    BusyGuard() = default;
    explicit BusyGuard(ModelHost* h) : host(h) {}
    BusyGuard(BusyGuard&& o) noexcept : host(o.host) { o.host = nullptr; }
    BusyGuard& operator=(BusyGuard&&) = delete;
    BusyGuard(const BusyGuard&) = delete;
    BusyGuard& operator=(const BusyGuard&) = delete;
    ~BusyGuard() {
      if (!host) return;
      std::lock_guard<std::mutex> lock(host->mu_);
      host->busy_ = false;
    }
    void release() { host = nullptr; }
  };
  bool try_begin_locked(std::string* err);
  void reset_load_state_locked(const std::string& state, const std::string& target);
  bool work(const Want& w, std::string* err);
  // A scheduler of `f`'s files, or nullptr (*err); *load_s: the seconds it took.
  std::shared_ptr<Scheduler> open(const std::string& model, const apps::Fetched& f, const EngineOptions& options,
                                  double* load_s, std::string* err) const;

  SchedulerOptions so_;
  mutable std::mutex mu_;
  Served served_;
  apps::Fetched fetched_;  // the served model's files and drafters
  std::string state_ = "none", error_, target_;
  std::vector<std::string> notes_;
  hub::Progress progress_;
  double since_ms_ = 0;
  Want last_;
  bool busy_ = false;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

// POST /v1/load's body (and the settings file's "last"): {"model", "drafters", "ctx", "drafter", "speculative", "kv",
// "name"} over `base`, the command line's engine options.
bool want_from_json(const ojson& j, const EngineOptions& base, ModelHost::Want* w, std::string* err);

// A model's start-up report for the log: device, context, drafter, memory, chat template.
void log_loaded(const Scheduler& sched, const std::string& name, double load_s);

}  // namespace flyx::server
