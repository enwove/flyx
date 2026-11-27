// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/os/env.h"
#include "core/os/os.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

extern char** environ;

namespace flyx::os {

MappedFile::~MappedFile() {
  if (data_) munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) close(fd_);
}

bool MappedFile::open(const std::string& path, std::string* err) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    *err = path + ": " + strerror(errno);
    return false;
  }
  struct stat st {};
  if (fstat(fd_, &st) != 0 || st.st_size == 0) {
    *err = path + ": cannot stat or empty";
    return false;
  }
  size_ = static_cast<size_t>(st.st_size);
  void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    *err = path + ": mmap failed: " + strerror(errno);
    size_ = 0;
    return false;
  }
  data_ = static_cast<const uint8_t*>(p);
  return true;
}

void MappedFile::read_ahead(size_t offset, size_t len) const {
  if (!data_ || offset >= size_) return;
  const size_t page = size_t(sysconf(_SC_PAGESIZE));
  const size_t first = offset / page * page, end = std::min(size_, offset + len);
  madvise(const_cast<uint8_t*>(data_) + first, end - first, MADV_WILLNEED);
}

RandomAccessFile::~RandomAccessFile() { close(); }

void RandomAccessFile::close() {
  if (handle_ >= 0) ::close(int(handle_));
  handle_ = -1;
}

bool RandomAccessFile::open(const std::string& path, uint64_t size, std::string* err, Mode mode) {
  close();
  path_ = path;
  const int flags = mode == kReadExisting ? O_RDONLY : O_RDWR | O_CREAT;
  const int fd = ::open(path.c_str(), flags | O_CLOEXEC, 0644);
  if (fd < 0) return *err = path + ": " + strerror(errno), false;
  struct stat st {};
  const int stat_result = fstat(fd, &st);
  if (!stat_result && mode == kReadExisting && uint64_t(st.st_size) != size) {
    *err = path + ": unexpected file size";
    ::close(fd);
    return false;
  }
  if (stat_result || (mode == kReadWrite && uint64_t(st.st_size) != size && ftruncate(fd, off_t(size)) != 0)) {
    *err = path + ": " + strerror(errno);
    ::close(fd);
    return false;
  }
  handle_ = fd;
  return true;
}

bool RandomAccessFile::write_at(uint64_t offset, const void* data, size_t n, std::string* err) const {
  const auto* p = static_cast<const char*>(data);
  while (n > 0) {
    const ssize_t w = pwrite(int(handle_), p, n, off_t(offset));
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return *err = path_ + ": " + strerror(errno), false;
    p += w, offset += uint64_t(w), n -= size_t(w);
  }
  return true;
}

bool RandomAccessFile::read_at(uint64_t offset, void* data, size_t n, std::string* err) const {
  auto* p = static_cast<char*>(data);
  while (n > 0) {
    const ssize_t r = pread(int(handle_), p, n, off_t(offset));
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return *err = path_ + (r == 0 ? ": shorter than expected" : ": " + std::string(strerror(errno))), false;
    p += r, offset += uint64_t(r), n -= size_t(r);
  }
  return true;
}

FileLock::~FileLock() {
  if (handle_ >= 0) ::close(int(handle_));
}

FileLock::Result FileLock::try_lock(std::string* err) {
  if (handle_ >= 0) return Result::kLocked;
  const int fd = ::open(path_.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
  if (fd < 0) {
    *err = path_ + ": " + strerror(errno);
    return Result::kError;
  }
  int rc;
  do { rc = flock(fd, LOCK_EX | LOCK_NB); } while (rc != 0 && errno == EINTR);
  if (rc == 0) {
    handle_ = fd;
    return Result::kLocked;
  }
  const int why = errno;
  ::close(fd);
  if (why == EWOULDBLOCK || why == EAGAIN) return Result::kBusy;
  *err = path_ + ": " + strerror(why);
  return Result::kError;
}

DynLib::~DynLib() {
  if (handle_) dlclose(handle_);
}

bool DynLib::open(const std::string& path, std::string* err) {
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    const char* e = dlerror();
    *err = e ? e : "dlopen failed";
    return false;
  }
  return true;
}

void* DynLib::symbol(const char* name) const { return handle_ ? dlsym(handle_, name) : nullptr; }

std::vector<std::string> utf8_arguments(int argc, char** argv) { return std::vector<std::string>(argv, argv + argc); }

void raw_utf8_stdio() {}

std::string executable_dir() {
  char buf[PATH_MAX];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = 0;
  std::string s(buf);
  auto k = s.rfind('/');
  return k == std::string::npos ? "." : s.substr(0, k);
}

std::string getenv_str(const char* name, const std::string& fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::string(v) : fallback;
}

bool env_exists(const char* name) { return std::getenv(name) != nullptr; }

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double unix_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

namespace {

// A "Key:   123 kB" line of a /proc file, in bytes (0 when absent).
uint64_t proc_kb(const char* path, const std::string& key) {
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line))
    if (line.compare(0, key.size(), key) == 0) return std::strtoull(line.c_str() + key.size(), nullptr, 10) * 1024;
  return 0;
}

}  // namespace

ProcessUsage process_usage() {
  ProcessUsage u;
  rusage r{};
  if (getrusage(RUSAGE_SELF, &r) == 0)
    u.cpu_seconds = double(r.ru_utime.tv_sec + r.ru_stime.tv_sec) + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
  u.rss_bytes = proc_kb("/proc/self/status", "VmRSS:");
  return u;
}

SystemMemory system_memory() {
  SystemMemory m;
  m.total = proc_kb("/proc/meminfo", "MemTotal:");
  m.available = proc_kb("/proc/meminfo", "MemAvailable:");
  if (m.total == 0) {
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) m.total = uint64_t(pages) * uint64_t(page);
  }
  return m;
}

double system_cpu_seconds() {
#if defined(__APPLE__)
  return 0;
#else
  // /proc/stat's first line: "cpu  user nice system idle iowait irq softirq steal ...", in clock ticks.
  std::ifstream f("/proc/stat");
  std::string name;
  uint64_t v[8] = {};
  if (!(f >> name) || name != "cpu") return 0;
  for (uint64_t& x : v)
    if (!(f >> x)) return 0;
  const long hz = sysconf(_SC_CLK_TCK);
  return hz > 0 ? double(v[0] + v[1] + v[2] + v[5] + v[6] + v[7]) / double(hz) : 0;  // all but idle and iowait
#endif
}

void release_working_set() {}

int process_id() { return static_cast<int>(getpid()); }

bool process_running(int pid) { return pid > 0 && (kill(pid_t(pid), 0) == 0 || errno == EPERM); }

bool stdout_is_terminal() { return isatty(STDOUT_FILENO) != 0; }
bool stdin_is_terminal() { return isatty(STDIN_FILENO) != 0; }
bool stderr_is_terminal() { return isatty(STDERR_FILENO) != 0; }
bool owns_console() { return false; }

void sleep_ms(int ms) { usleep(useconds_t(ms) * 1000); }

void sleep_us(int64_t us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }

int cpu_count() {
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? int(n) : 1;
}

struct Terminal::Saved {
  termios mode{};
};

Terminal::Terminal() = default;

Terminal::~Terminal() {
  if (saved_) tcsetattr(STDIN_FILENO, TCSANOW, &saved_->mode);
}

bool Terminal::raw() {
  if (!isatty(STDIN_FILENO)) return false;
  auto s = std::make_unique<Saved>();
  if (tcgetattr(STDIN_FILENO, &s->mode) != 0) return false;
  termios t = s->mode;
  t.c_lflag &= ~tcflag_t(ICANON | ECHO);
  t.c_cc[VMIN] = 0;
  t.c_cc[VTIME] = 0;
  if (tcsetattr(STDIN_FILENO, TCSANOW, &t) != 0) return false;
  saved_ = std::move(s);
  return true;
}

void Terminal::size(int* cols, int* rows) const {
  winsize w{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 0) {
    *cols = w.ws_col;
    *rows = w.ws_row;
  } else {
    *cols = 100;
    *rows = 32;
  }
}

int Terminal::key(int timeout_ms) {
  pollfd p{STDIN_FILENO, POLLIN, 0};
  if (poll(&p, 1, timeout_ms) <= 0) return -1;
  unsigned char c = 0;
  return read(STDIN_FILENO, &c, 1) == 1 ? c : -1;
}

namespace {

bool is_wsl() {
  std::ifstream f("/proc/sys/kernel/osrelease");
  std::string release;
  std::getline(f, release);
  for (char& c : release) c = char(std::tolower(static_cast<unsigned char>(c)));
  return release.find("microsoft") != std::string::npos || release.find("wsl") != std::string::npos;
}

// Starts a program with its standard streams on /dev/null and reaps it on a thread of its own: false when it did not
// start.
bool spawn_detached(const std::vector<std::string>& args) {
  std::vector<char*> argv;
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  for (int fd = 0; fd < 3; ++fd) posix_spawn_file_actions_addopen(&fa, fd, "/dev/null", fd ? O_WRONLY : O_RDONLY, 0);
  pid_t pid = 0;
  const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  if (rc != 0) return false;
  std::thread([pid] {
    int status = 0;
    waitpid(pid, &status, 0);
  }).detach();
  return true;
}

// Runs a program to its end with what it writes to standard output in *out: its exit status, 127 when it did not
// start (as a shell says of a program it does not find), -1 when it ended some other way.
int run_for_output(const std::vector<std::string>& args, std::string* out) {
  int p[2];
  if (pipe(p) != 0) return 127;
  std::vector<char*> argv;
  for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
  posix_spawn_file_actions_adddup2(&fa, p[1], 1);
  posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
  posix_spawn_file_actions_addclose(&fa, p[0]);
  posix_spawn_file_actions_addclose(&fa, p[1]);
  pid_t pid = 0;
  const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  close(p[1]);
  if (rc != 0) {
    close(p[0]);
    return 127;
  }
  char buf[4096];
  for (ssize_t n; (n = read(p[0], buf, sizeof buf)) > 0 || (n < 0 && errno == EINTR);)
    if (n > 0) out->append(buf, size_t(n));
  close(p[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// capture_stderr: the sink, and the descriptor standard error had before (-1: no capture).
std::mutex g_capture_mu;
std::function<void(const std::string&)> g_capture_sink;
int g_original_stderr = -1;

void deliver(const std::string& line) {
  std::function<void(const std::string&)> sink;
  {
    std::lock_guard<std::mutex> lock(g_capture_mu);
    sink = g_capture_sink;
  }
  if (sink) sink(line);
}

void write_all(int fd, const char* p, size_t n) {
  while (n > 0) {
    const ssize_t w = write(fd, p, n);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return;
    p += w;
    n -= size_t(w);
  }
}

}  // namespace

bool gpu_memory_paged() { return is_wsl(); }

uint64_t gpu_memory_floor() { return 0; }

bool open_url(const std::string& url) {
  if (env_exists(env::kSshConnection) || env_exists(env::kSshTty)) return false;  // the browser would be the server's
#if defined(__APPLE__)
  return spawn_detached({"open", url});
#else
  if (is_wsl()) return spawn_detached({"explorer.exe", url});  // Windows' default browser
  if (!env_exists(env::kDisplay) && !env_exists(env::kWaylandDisplay)) return false;
  return spawn_detached({"xdg-open", url});
#endif
}

Picked pick_folder(const std::string& title, const std::string& start, std::string* path, std::string* err) {
  if (env_exists(env::kSshConnection) || env_exists(env::kSshTty))
    return *err = "no desktop to show a folder dialog on: this is a session over SSH", Picked::kUnavailable;
  // The desktop's dialog programs, in turn: each prints the folder chosen, and ends with 1 when the person cancels.
  std::vector<std::vector<std::string>> programs;
#if defined(__APPLE__)
  std::string quoted;  // a string of AppleScript: \ and " escaped
  for (char c : title) quoted += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
  programs.push_back({"osascript", "-e", "POSIX path of (choose folder with prompt \"" + quoted + "\")"});
#else
  if (is_wsl()) return *err = "no folder dialog under WSL: give the folder's path", Picked::kUnavailable;
  if (!env_exists(env::kDisplay) && !env_exists(env::kWaylandDisplay))
    return *err = "no desktop to show a folder dialog on: no display", Picked::kUnavailable;
  const std::string from = start.empty() ? "" : (start.back() == '/' ? start : start + "/");
  programs.push_back({"zenity", "--file-selection", "--directory", "--title=" + title});
  if (!from.empty()) programs.back().push_back("--filename=" + from);
  programs.push_back({"kdialog", "--title", title, "--getexistingdirectory", from.empty() ? "." : from});
#endif
  for (const auto& program : programs) {
    std::string out;
    const int status = run_for_output(program, &out);
    if (status == 127) continue;  // not on this machine: the next one
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    if (status == 0 && !out.empty()) return *path = out, Picked::kFolder;
    if (status == 1 || (status == 0 && out.empty())) return Picked::kCancelled;
    return *err = program[0] + " could not show a folder dialog", Picked::kUnavailable;
  }
  return *err = "no folder dialog program on this machine (zenity, kdialog)", Picked::kUnavailable;
}

bool capture_stderr(std::function<void(const std::string& line)> sink) {
  std::lock_guard<std::mutex> lock(g_capture_mu);
  if (g_original_stderr >= 0) {
    g_capture_sink = std::move(sink);
    return true;
  }
  int p[2];
  if (pipe(p) != 0) return false;
  std::fflush(stderr);
  const int saved = dup(STDERR_FILENO);
  if (saved < 0 || dup2(p[1], STDERR_FILENO) < 0) {
    close(p[0]);
    close(p[1]);
    if (saved >= 0) close(saved);
    return false;
  }
  close(p[1]);  // standard error is now the pipe's only writer
  fcntl(saved, F_SETFD, FD_CLOEXEC);
  fcntl(p[0], F_SETFD, FD_CLOEXEC);
  g_original_stderr = saved;
  g_capture_sink = std::move(sink);
  std::thread([fd = p[0]] {
    std::string line;
    char buf[4096];
    for (;;) {
      const ssize_t n = read(fd, buf, sizeof(buf));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) break;
      for (ssize_t i = 0; i < n; ++i) {
        if (buf[i] == '\n') {
          deliver(line);
          line.clear();
        } else if (buf[i] != '\r') {
          line += buf[i];
        }
      }
    }
    if (!line.empty()) deliver(line);
  }).detach();
  return true;
}

void write_original_stderr(const std::string& text) {
  int fd = STDERR_FILENO;
  {
    std::lock_guard<std::mutex> lock(g_capture_mu);
    if (g_original_stderr >= 0) fd = g_original_stderr;
  }
  write_all(fd, text.data(), text.size());
}

// A terminal of the desktop's own, with an environment set for it, differs on each: not yet.
bool shell_window_supported() { return false; }
bool open_shell_window(const std::string&, const std::string&, const std::vector<std::pair<std::string, std::string>>&,
                       std::string* err) {
  return *err = "opening a terminal window is supported on Windows only", false;
}

}  // namespace flyx::os
