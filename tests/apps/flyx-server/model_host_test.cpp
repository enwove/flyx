// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// ModelHost (apps/flyx-server/model_host.h): a model switch loads the new model only once the old one is gone, not
// merely once nobody holds it.  The last reference to the old model's scheduler is a request's (an HTTP handler's),
// let go on a thread of its own while the backend's close of the old model is held up there: the new model must not
// start to load until that close has returned.  The tiny model on a backend that hands every call to the float64 one,
// holding at most one model open at a time (a GPU with room for one).
//
//   model_host_test MODEL.flyx
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "apps/flyx-server/model_host.h"
#include "core/exec/engine.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "tests/common/check.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <errno.h>
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

using namespace flyx;
using flyx::server::ModelHost;

extern "C" void flyx_test_delegate(const flyx_backend_api*);

namespace {

void set_env(const char* name, const std::string& value) {  // UTF-8 on Windows too (config_dir reads it wide)
#if defined(_WIN32)
  _wputenv_s(os::to_path(name).c_str(), os::to_path(value).c_str());
#else
  setenv(name, value.c_str(), 1);
#endif
}

// The exit code of `exe` run with `args`, no shell between (a path's spaces, $ and % reach it as they are, and on
// Windows its other letters too, the arguments going over as UTF-16); -1 when it could not be started.
int run_process(const std::string& exe, const std::vector<std::string>& args) {
  std::fflush(stdout);
#if defined(_WIN32)
  // As CommandLineToArgvW reads them back: each argument quoted, the backslashes before a quote and before the
  // closing one doubled.
  auto quote = [](const std::wstring& a) {
    std::wstring q = L"\"";
    size_t slashes = 0;
    for (wchar_t c : a) {
      if (c == L'\\') {
        ++slashes;
        continue;
      }
      q.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
      slashes = 0;
      q += c;
    }
    q.append(slashes * 2, L'\\');
    return q + L"\"";
  };
  const std::wstring program = os::to_path(exe).wstring();
  std::wstring line = quote(program);
  for (const std::string& a : args) line += L" " + quote(os::to_path(a).wstring());
  STARTUPINFOW si{};
  si.cb = sizeof si;
  PROCESS_INFORMATION pi{};
  if (!CreateProcessW(program.c_str(), line.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return -1;
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 1;
  GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return int(code);
#else
  std::vector<std::string> all = {exe};
  all.insert(all.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (std::string& a : all) argv.push_back(a.data());
  argv.push_back(nullptr);
  pid_t pid = 0;
  if (posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ) != 0) return -1;
  int status = 0;
  while (waitpid(pid, &status, 0) < 0)
    if (errno != EINTR) return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string file_text(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// The backend's models, as the delegating backend sees them.
const flyx_backend_api* cpu = nullptr;
std::mutex mu;
std::condition_variable cv;
int open_models = 0;           // held now: one at most
int opens = 0, closes = 0;
bool refused = false;          // an open came while another model was still open
bool hold_next_close = false;  // the next close waits for release_close
bool closing = false;          // a held close has begun
bool release_close = false;
std::vector<std::string> events;

int one_model_open(const flyx_model_desc* model, const flyx_open_options* opts, flyx_bmodel** out) {
  {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back("open");
    if (open_models > 0) {
      refused = true;
      return -1;  // no room for a second model
    }
  }
  const int rc = cpu->open(model, opts, out);
  std::lock_guard<std::mutex> lock(mu);
  if (rc == 0) ++open_models, ++opens;
  return rc;
}

void held_close(flyx_bmodel* m) {
  std::unique_lock<std::mutex> lock(mu);
  events.push_back("close begins");
  if (hold_next_close) {
    hold_next_close = false;
    closing = true;
    cv.notify_all();
    cv.wait(lock, [] { return release_close; });
  }
  lock.unlock();
  cpu->close(m);
  lock.lock();
  --open_models, ++closes;
  events.push_back("close ends");
  cv.notify_all();
}

bool wait_for(const std::function<bool()>& done, int ms) {
  const double until = os::now_ms() + ms;
  while (!done()) {
    if (os::now_ms() > until) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return true;
}

}  // namespace

// One of two servers sharing serve.json (a process of its own, the settings in `dir`): n saves of remote_manage
// turned on and off ("network") or of a Hub endpoint ("hub"), each read back after it.  The other's
// read-modify-replace, were it not under the file's lock, would now and then write its older copy over this one's
// save.  Exit 0: every save made and none lost.
int settings_writer(const std::string& dir, const std::string& kind, int n) {
  set_env("APPDATA", dir);
  set_env("XDG_CONFIG_HOME", dir);
  int failed = 0, lost = 0;
  for (int i = 0; i < n; ++i) {
    std::string err;
    if (kind == "network") {
      const bool v = i % 2 == 0;
      if (!ModelHost::save_network_settings(std::nullopt, v, {}, nullptr, &err))
        ++failed, std::fprintf(stderr, "%s\n", err.c_str());
      else if (ModelHost::read_network_settings().remote_manage != v)
        ++lost;
    } else {
      const std::string e = "https://hub-" + std::to_string(i) + ".example";
      if (!ModelHost::save_hub_endpoint(e, &err)) ++failed, std::fprintf(stderr, "%s\n", err.c_str());
      else if (ModelHost::remembered_hub_endpoint() != e) ++lost;
    }
  }
  std::printf("%s writer: %d saves, %d failed, %d lost to the other\n", kind.c_str(), n, failed, lost);
  return failed || lost ? 1 : 0;
}

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  const std::vector<std::string> args = os::utf8_arguments(argc, argv);  // Windows: from UTF-16, not the code page
  if (args.size() == 5 && args[1] == "--settings-writer")
    return settings_writer(args[2], args[3], std::atoi(args[4].c_str()));
  if (argc < 2) {
    std::fprintf(stderr, "usage: model_host_test MODEL.flyx\n");
    return 2;
  }
  // The settings file a load writes: in a directory of this run's own, never the user's.
  namespace fs = std::filesystem;
  const fs::path config = fs::temp_directory_path() / ("flyx-model-host-test-" + std::to_string(os::process_id()));
  std::error_code ec;
  fs::create_directories(config, ec);
  set_env("APPDATA", os::from_path(config));
  set_env("XDG_CONFIG_HOME", os::from_path(config));
  const std::string settings = ModelHost::settings_path();
  if (settings.rfind(os::from_path(config), 0) != 0) {
    std::printf("the settings file would be %s, not in %s\n", settings.c_str(), os::from_path(config).c_str());
    return 1;
  }

  EngineOptions eo;
  eo.backend = "cpu_ref";
  eo.max_ctx = 256;
  eo.speculative = false;
  std::unique_ptr<os::DynLib> lib;
  flyx_device_info device{};
  std::string err;
  if (!open_backend(eo, &lib, &cpu, &device, &err)) {
    std::printf("backend: %s\n", err.c_str());
    return 1;
  }
  static flyx_backend_api one_at_a_time = *cpu;
  one_at_a_time.open = one_model_open;
  one_at_a_time.close = held_close;
  flyx_test_delegate(&one_at_a_time);
  eo.backend = "test-host-kv";

  SchedulerOptions so;
  so.max_sessions = 1;
  {
    ModelHost host(so);
    ModelHost::Want want;
    want.model = argv[1];
    want.eo = eo;
    check(host.run(want, &err), "the first model loads " + err);
    int opens_before = 0;
    {
      std::lock_guard<std::mutex> lock(mu);
      opens_before = opens;
      events.clear();
      hold_next_close = true;
    }
    // A request's hold on the model: once the switch has let go of the host's, the last one.
    const auto first_defaults = host.served().sampling;
    std::shared_ptr<Scheduler> request = host.served().sched;
    std::atomic<bool> switched{false};
    bool switch_ok = false;
    std::string switch_err;
    std::thread switcher([&] {
      switch_ok = host.run(want, &switch_err);
      switched.store(true);
    });
    const bool let_go = wait_for([&] { return request.use_count() == 1; }, 30000);
    check(let_go, "the switch lets go of the host's reference to the old model");
    check(first_defaults && !host.served().sampling, "old defaults live with the request while the host is empty");
    // The request ends on its own thread, and the old model's close is held up there.
    std::thread request_end([r = std::move(request)]() mutable { r.reset(); });
    bool held = false;
    {
      std::unique_lock<std::mutex> lock(mu);
      held = cv.wait_for(lock, std::chrono::seconds(30), [] { return closing; });
    }
    check(held, "the old model's close begins on the thread of the request that held it last");
    // Long enough for a switch that only waits for nobody to hold the old model (as it polled: every 50 ms) to load.
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    {
      std::lock_guard<std::mutex> lock(mu);
      check(held && opens == opens_before && !refused && !switched.load() &&
                events == std::vector<std::string>{"close begins"},
            "while the old model closes, the switch waits and the new model does not start to load");
      release_close = true;
    }
    cv.notify_all();
    request_end.join();
    switcher.join();
    {
      std::lock_guard<std::mutex> lock(mu);
      std::string order;
      for (const auto& e : events) order += (order.empty() ? "" : ", ") + e;
      check(switch_ok && !refused && opens == opens_before + 1 && open_models == 1 && events.size() >= 3 &&
                events[0] == "close begins" && events[1] == "close ends" && events[2] == "open",
            "the new model loads once the old one's close has returned (" + order + ") " + switch_err);
    }
    check(host.served().sampling && host.served().sampling != first_defaults,
          "the new scheduler publishes its own immutable defaults");
    // The new model serves.
    Request rq;
    rq.chat = false;
    rq.text = "The new model answers";
    rq.gen.max_new_tokens = 2;
    rq.gen.ignore_eos = true;
    auto served = host.served().sched;
    int status = 0;
    auto job = served ? served->submit(rq, &err, &status) : nullptr;
    JobEvent ev;
    bool answered = false;
    while (job && job->next(&ev, 60000)) {
      if (ev.kind == JobEvent::kDone) answered = ev.result.usage.new_tokens == 2;
      if (ev.kind == JobEvent::kDone || ev.kind == JobEvent::kError) break;
    }
    check(answered, "the new model answers a request " + err);
  }
  {
    std::lock_guard<std::mutex> lock(mu);
    check(open_models == 0 && closes == opens, "every model the backend opened is closed with the host");
  }
  {
    // ModelHost::run enforces Want::eo.allow_path before opening the model file.
    ModelHost host(so);
    ModelHost::Want blocked;
    blocked.model = argv[1];
    blocked.eo = eo;
    blocked.eo.allow_path = [](std::string_view) { return false; };
    std::string policy_err;
    const int opens_now = opens;
    check(!host.run(blocked, &policy_err) && policy_err.find("remote path policy") != std::string::npos &&
              opens == opens_now,
          "ModelHost::run rejects model path blocked by eo.allow_path before backend open: " + policy_err);

    // Item 1: Scheduler::Scheduler's own tokenizer ModelFile::open must forward eo.allow_path when opening
    // split GGUF sibling shards, failing with remote path policy before touching the denied shard or backend.
    const fs::path split1 = config / "policy_split-00001-of-00002.gguf";
    const fs::path split2 = config / "policy_split-00002-of-00002.gguf";
    {
      auto write_u16 = [](std::ofstream& f, uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
      auto write_u32 = [](std::ofstream& f, uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
      auto write_u64 = [](std::ofstream& f, uint64_t v) { f.write(reinterpret_cast<const char*>(&v), 8); };
      auto write_str = [&](std::ofstream& f, const std::string& s) {
        write_u64(f, s.size());
        f.write(s.data(), std::streamsize(s.size()));
      };
      std::ofstream f(split1, std::ios::binary | std::ios::trunc);
      f.write("GGUF", 4);
      write_u32(f, 3);  // version
      write_u64(f, 0);  // n_tensors
      write_u64(f, 2);  // n_kv
      write_str(f, "split.no");
      write_u32(f, 2);  // GGUF_TYPE_UINT16
      write_u16(f, 0);
      write_str(f, "split.count");
      write_u32(f, 2);  // GGUF_TYPE_UINT16
      write_u16(f, 2);
    }
    bool saw_shard2_check = false;
    ModelHost::Want split_want;
    split_want.model = os::from_path(split1);
    split_want.eo = eo;
    split_want.eo.allow_path = [&](std::string_view p) {
      if (p.find("00002-of-00002") != std::string_view::npos) {
        saw_shard2_check = true;
        return false;
      }
      return true;
    };
    policy_err.clear();
    const int opens_before_split = opens;
    check(!host.run(split_want, &policy_err) && saw_shard2_check &&
              policy_err.find("remote path policy") != std::string::npos && opens == opens_before_split &&
              !fs::exists(split2),
          "ModelHost::run -> Scheduler::Scheduler rejects denied split GGUF sibling shard via eo.allow_path: " +
              policy_err);
  }
  {
    // Single-lock NetworkSettingsSnapshot coherence: save_network_settings commits on_remote_manage_committed and
    // samples live_remote_manage under settings_mutex(), matching read_network_settings under concurrent toggles.
    std::atomic<bool> live_rm{false};
    auto commit_rm = [&](bool v) { live_rm.store(v, std::memory_order_relaxed); };
    auto read_live = [&]() { return live_rm.load(std::memory_order_relaxed); };
    ModelHost::NetworkSettingsSnapshot snap;
    std::string net_err;
    check(ModelHost::save_network_settings(true, true, commit_rm, &snap, &net_err, read_live) &&
              snap.lan_access && snap.remote_manage && snap.live_remote_manage,
          "save_network_settings samples committed live_remote_manage in the same critical section: " + net_err);
    const auto read_snap = ModelHost::read_network_settings(read_live);
    check(read_snap.lan_access && read_snap.remote_manage && read_snap.live_remote_manage,
          "read_network_settings returns coherent lan_access, remote_manage, and live_remote_manage");

    std::atomic<bool> stop{false};
    std::atomic<bool> torn{false};
    std::thread writer([&] {
      bool v = false;
      for (int i = 0; i < 40; ++i) {
        v = !v;
        ModelHost::NetworkSettingsSnapshot s;
        std::string e;
        if (!ModelHost::save_network_settings(v, v, commit_rm, &s, &e, read_live) ||
            s.lan_access != v || s.remote_manage != v || s.live_remote_manage != v) {
          torn.store(true, std::memory_order_relaxed);
          break;
        }
      }
      stop.store(true, std::memory_order_relaxed);
    });
    std::thread reader([&] {
      while (!stop.load(std::memory_order_relaxed)) {
        const auto s = ModelHost::read_network_settings(read_live);
        if (s.lan_access != s.remote_manage || s.remote_manage != s.live_remote_manage) {
          torn.store(true, std::memory_order_relaxed);
          break;
        }
      }
    });
    writer.join();
    reader.join();
    check(!torn.load(), "concurrent save_network_settings / read_network_settings never tear disk vs live_remote_manage");
  }
  {
    // Two servers sharing serve.json, processes of their own started with no shell between, their settings in a
    // directory whose name has spaces, $ and % and a letter outside ASCII: one turns remote_manage on and off, the
    // other saves Hub endpoints, at once.  Neither save is lost to the other's, and a fresh reader sees both last ones.
    const std::string self = os::from_path(fs::absolute(os::to_path(args[0]), ec));
    const fs::path shared = config / fs::path(u8"two servers $HOME %PATH% ü");
    fs::create_directories(shared, ec);
    const std::string shared_dir = os::from_path(shared);
    const int n = 150;
    auto writer = [&](const char* kind) {
      return run_process(self, {"--settings-writer", shared_dir, kind, std::to_string(n)});
    };
    int network_rc = -1, hub_rc = -1;
    std::thread network([&] { network_rc = writer("network"); });
    std::thread hub([&] { hub_rc = writer("hub"); });
    network.join();
    hub.join();
    check(network_rc == 0 && hub_rc == 0,
          "two processes' saves of serve.json at once: none lost to the other's (exit " + std::to_string(network_rc) +
              ", " + std::to_string(hub_rc) + ")");
    set_env("APPDATA", shared_dir);
    set_env("XDG_CONFIG_HOME", shared_dir);
    const bool last_remote_manage = (n - 1) % 2 == 0;
    check(fs::exists(shared / "flyx" / "serve.json") &&
              ModelHost::read_network_settings().remote_manage == last_remote_manage &&
              ModelHost::remembered_hub_endpoint() == "https://hub-" + std::to_string(n - 1) + ".example",
          "and a fresh reader sees both processes' last saves, in the directory named as it is");
    set_env("APPDATA", os::from_path(config));
    set_env("XDG_CONFIG_HOME", os::from_path(config));
  }
  {
    // serve.json there but no object of settings (cut short, not JSON, another root, empty): a save leaves its bytes
    // as they are and says why, rather than writing the one setting it changes over the last model and the network
    // settings; a reader gets the settings this process last had, and says why too.
    std::string err;
    check(ModelHost::save_network_settings(false, true, {}, nullptr, &err), "known settings saved: " + err);
    const fs::path file = os::to_path(ModelHost::settings_path());
    for (const std::string bytes : {std::string(R"({"last": {"model": "a.gguf")"), std::string("[1, 2]"),
                                    std::string("not json"), std::string()}) {
      { std::ofstream(file, std::ios::binary | std::ios::trunc) << bytes; }
      err.clear();
      const bool saved = ModelHost::save_hub_endpoint("https://mirror.example", &err);
      check(!saved && file_text(file) == bytes && err.find("left as it is") != std::string::npos,
            "a save over serve.json of " + std::to_string(bytes.size()) + " bytes that are no settings leaves them: " +
                err);
      const ModelHost::NetworkSettingsSnapshot snap = ModelHost::read_network_settings();
      check(!snap.lan_access && snap.remote_manage && !snap.error.empty(),
            "and a reader gets the settings last had, saying why: " + snap.error);
    }
    fs::remove(file, ec);
    check(ModelHost::save_hub_endpoint("https://mirror.example", &err) && ModelHost::read_network_settings().error.empty(),
          "no file at all: a save starts one");
  }
  {
    // Another server holding the lock: a save waits for it and gives up, the file untouched; a reader opens no file
    // (a file written behind this process's back is not read), giving the settings last had, saying why.
    std::string err;
    check(ModelHost::save_network_settings(false, true, {}, nullptr, &err), "known settings saved: " + err);
    const fs::path file = os::to_path(ModelHost::settings_path());
    const std::string behind = R"({"lan_access": true, "remote_manage": false})";
    { std::ofstream(file, std::ios::binary | std::ios::trunc) << behind; }
    {
      os::FileLock other(ModelHost::settings_path() + ".lock");
      std::string why;
      check(other.try_lock(&why) == os::FileLock::Result::kLocked, "another server's lock taken: " + why);
      const bool saved = ModelHost::save_hub_endpoint("https://held.example", &err);
      check(!saved && err.find("held by another flyx server") != std::string::npos && file_text(file) == behind,
            "a save under another server's lock gives up, the file untouched: " + err);
      const ModelHost::NetworkSettingsSnapshot snap = ModelHost::read_network_settings();
      check(!snap.lan_access && snap.remote_manage && snap.error.find("held by another flyx server") != std::string::npos,
            "a reader without the lock opens no file: the settings last had, and why: " + snap.error);
    }
    const ModelHost::NetworkSettingsSnapshot snap = ModelHost::read_network_settings();
    check(snap.lan_access && !snap.remote_manage && snap.error.empty(), "the lock let go: the file read again");
    // The lock file not to be had at all (a directory in its place): the same, at once.
    const fs::path lock = os::to_path(ModelHost::settings_path() + ".lock");
    fs::remove(lock, ec);
    fs::create_directory(lock, ec);
    err.clear();
    const bool saved = ModelHost::save_hub_endpoint("https://nolock.example", &err);
    const ModelHost::NetworkSettingsSnapshot none = ModelHost::read_network_settings();
    check(!saved && !err.empty() && file_text(file) == behind && !none.error.empty(),
          "a lock file that cannot be opened: no save, no read of the file, both saying why: " + err);
    fs::remove(lock, ec);
  }
  fs::remove_all(config, ec);
  return flyx::test::finish("model_host_test");
}
