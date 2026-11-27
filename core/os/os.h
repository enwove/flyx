// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Everything that differs between operating systems lives behind this header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flyx::os {

// Read-only memory map of a whole file.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  bool open(const std::string& path, std::string* err);
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  // Asks the system to read bytes offset .. offset + len - 1 from disk now, in large reads, rather than a page at a
  // time when they are first touched.  It may return before they are read (Linux) or after (Windows).
  void read_ahead(size_t offset, size_t len) const;
  // Reclaim clean resident pages of this read-only mapping. Its address and file contents stay valid.
  void release_pages() const;

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  int fd_ = -1;
};

// A file read and written at offsets by several threads at once (core/hub: each connection of a download writes the
// ranges it fetched).
class RandomAccessFile {
 public:
  RandomAccessFile() = default;
  ~RandomAccessFile();
  RandomAccessFile(const RandomAccessFile&) = delete;
  RandomAccessFile& operator=(const RandomAccessFile&) = delete;
  enum Mode { kReadWrite, kReadExisting };
  // kReadWrite creates or resizes the file. kReadExisting requires an existing file of exactly this size.
  bool open(const std::string& path, uint64_t size, std::string* err, Mode mode = kReadWrite);
  bool write_at(uint64_t offset, const void* data, size_t n, std::string* err) const;
  bool read_at(uint64_t offset, void* data, size_t n, std::string* err) const;
  void close();

 private:
  intptr_t handle_ = -1;
  std::string path_;
};

// An exclusive lock on a named file, released when this object is destroyed or the process exits.
// The file stays in place so every waiter opens the same file. try_lock never waits.
class FileLock {
 public:
  explicit FileLock(const std::string& path) : path_(path) {}
  ~FileLock();
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;
  enum class Result { kLocked, kBusy, kError };
  Result try_lock(std::string* err);

 private:
  std::string path_;
  intptr_t handle_ = -1;
};

// A loadable library's file name is "lib" + its name + this (the build gives every library the prefix).
#if defined(_WIN32)
inline constexpr const char* kDynLibSuffix = ".dll";
#else
inline constexpr const char* kDynLibSuffix = ".so";
#endif

class DynLib {
 public:
  DynLib() = default;
  ~DynLib();
  DynLib(const DynLib&) = delete;
  DynLib& operator=(const DynLib&) = delete;
  bool open(const std::string& path, std::string* err);
  void* symbol(const char* name) const;

 private:
  void* handle_ = nullptr;
};

// The command line as UTF-8: argv itself, except on Windows, where main() is handed the ANSI code page's bytes.
std::vector<std::string> utf8_arguments(int argc, char** argv);
// stdout and stderr write the bytes they are given (no "\n" -> "\r\n" on Windows), a console shows them as UTF-8.
void raw_utf8_stdio();

std::string executable_dir();
std::string getenv_str(const char* name, const std::string& fallback = "");
bool env_exists(const char* name);
double now_ms();
double unix_seconds();  // wall clock

// This process: CPU time used (user + system) and resident memory.
struct ProcessUsage {
  double cpu_seconds = 0;
  uint64_t rss_bytes = 0;
};
ProcessUsage process_usage();
// The machine: physical memory and what is available (bytes; 0 = unknown), logical CPUs.
struct SystemMemory {
  uint64_t total = 0, available = 0;
};
SystemMemory system_memory();
// The seconds the machine's processors have been busy since it started, all of them added up (a second of 16 busy
// processors is 16): two readings give its load between them.  0 where the system does not say.
double system_cpu_seconds();
// Whether the GPU's memory is Windows' to page out of the backend's sight (its video memory manager under WSL2): an
// allocation past what it lets a process keep is moved to host memory and read over the bus, never refused, and the
// driver there tells neither Windows' free memory nor a process's share.  Natively Windows pages it too, but the CUDA
// backend reads NVML's free memory there and refuses an allocation that would leave less than Windows keeps free for
// one of its size (backends/cuda: device_room), so the budget keeps its usual reserve.
bool gpu_memory_paged();
// Gives back the pages the process holds without needing them locked: on Windows its working set, so that the file
// pages a load read count as available again (they stay cached; page-locked memory stays); elsewhere nothing (the
// kernel counts clean file pages as available already).
void release_working_set();
int cpu_count();
int process_id();
// Whether a process of this id runs (one of another user's too).
bool process_running(int pid);
bool stdout_is_terminal();
bool stdin_is_terminal();
bool stderr_is_terminal();
// True on Windows when this process is the sole process attached to its console (e.g. double-clicked in Explorer).
bool owns_console();
void sleep_ms(int ms);
// A sleep of about `us` microseconds.  On Windows a high-resolution waitable timer (Windows 10 1803+): ~0.5 ms at
// least, where Sleep and std::this_thread::sleep_for end on the OS scheduler's tick, 1 to 16 ms.
void sleep_us(int64_t us);

// Opens url in the desktop's web browser and does not wait for it: false where there is no desktop to show it on (a
// session over SSH, a Linux console without a display) or the opener did not start.  WSL opens it in Windows.
bool open_url(const std::string& url);

// Asks the person at this machine for a folder with the desktop's own dialog, and waits for the answer.  `title` heads
// the dialog; it opens at `start` ("" for where the desktop puts it).  kFolder: *path is the one chosen.  kUnavailable:
// no dialog could be shown (a session over SSH, no display, no dialog program), *err says why.
enum class Picked { kFolder, kCancelled, kUnavailable };
Picked pick_folder(const std::string& title, const std::string& start, std::string* path, std::string* err);

// Opens a window of Windows PowerShell on this machine's desktop, in folder `dir`, running `command` and staying open
// after it.  The window's environment is the user's own, as a terminal they open gets it (not this process's, which
// whatever started flyx changed), with `env` set on top: a secret given there stays off every command line.  Does not wait.  false (*err) where there is no desktop (a session over SSH, a service) or the window
// did not start.  Windows only for now: shell_window_supported() says whether this system has one.
bool shell_window_supported();
bool open_shell_window(const std::string& dir, const std::string& command,
                       const std::vector<std::pair<std::string, std::string>>& env, std::string* err);

// From the first call on, what this process and the libraries it loads after write to standard error goes to `sink`,
// a line at a time, from a thread of its own (a full-screen view draws on the terminal meanwhile: flyx serve's
// dashboard); a later call replaces the sink.  false: the descriptors could not be redirected (nothing changed).
bool capture_stderr(std::function<void(const std::string& line)> sink);
// Writes to standard error as it was before capture_stderr (to standard error itself without a capture).
void write_original_stderr(const std::string& text);

// The terminal on stdin / stdout for a full-screen view (flyx monitor):
// keys read one at a time without echo, its size. The mode is restored when
// the object goes.
class Terminal {
 public:
  Terminal();
  ~Terminal();
  Terminal(const Terminal&) = delete;
  Terminal& operator=(const Terminal&) = delete;
  bool raw();                         // false when stdin is not a terminal
  void size(int* cols, int* rows) const;
  int key(int timeout_ms);            // a key (byte), or -1 when none came in time

 private:
  struct Saved;
  std::unique_ptr<Saved> saved_;
};

}  // namespace flyx::os
