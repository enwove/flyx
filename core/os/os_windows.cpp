// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// core/os/os.h on Windows (os_posix.cpp elsewhere).  Paths are UTF-8 strings; they become UTF-16 at the Win32 calls.
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/win_utf8.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")  // CommandLineToArgvW, ShellExecuteW, SHCreateItemFromParsingName
#include <shobjidl.h>
#pragma comment(lib, "ole32.lib")  // CoInitializeEx, CoCreateInstance, CoTaskMemFree
#pragma comment(lib, "uuid.lib")   // CLSID_FileOpenDialog
#include <psapi.h>
#include <userenv.h>
#pragma comment(lib, "userenv.lib")  // CreateEnvironmentBlock

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <functional>
#include <mutex>
#include <thread>

namespace flyx::os {

namespace win {

std::wstring to_wide(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

std::string to_utf8(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

}  // namespace win

namespace {

std::wstring wide(const std::string& s) { return win::to_wide(s); }
std::string narrow(const std::wstring& w) { return win::to_utf8(w); }

// The text of GetLastError()'s code.
std::string last_error() {
  const DWORD code = GetLastError();
  char* text = nullptr;
  const DWORD n = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr, code, 0, reinterpret_cast<char*>(&text), 0, nullptr);
  std::string s = n && text ? std::string(text, n) : "error " + std::to_string(code);
  if (text) LocalFree(text);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
  return s;
}

}  // namespace

MappedFile::~MappedFile() {
  if (data_) UnmapViewOfFile(data_);
}

bool MappedFile::open(const std::string& path, std::string* err) {
  HANDLE f = CreateFileW(wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) {
    *err = path + ": " + last_error();
    return false;
  }
  LARGE_INTEGER size{};
  if (!GetFileSizeEx(f, &size) || size.QuadPart == 0) {
    *err = path + ": cannot stat or empty";
    CloseHandle(f);
    return false;
  }
  HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  CloseHandle(f);  // the mapping holds the file
  if (!m) {
    *err = path + ": CreateFileMapping failed: " + last_error();
    return false;
  }
  void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(m);  // the view holds the mapping
  if (!p) {
    *err = path + ": MapViewOfFile failed: " + last_error();
    return false;
  }
  data_ = static_cast<const uint8_t*>(p);
  size_ = size_t(size.QuadPart);
  return true;
}

void MappedFile::read_ahead(size_t offset, size_t len) const {
  if (!data_ || offset >= size_) return;
  WIN32_MEMORY_RANGE_ENTRY range{const_cast<uint8_t*>(data_) + offset, std::min(len, size_ - offset)};
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
}

RandomAccessFile::~RandomAccessFile() { close(); }

void RandomAccessFile::close() {
  if (handle_ != -1) CloseHandle(reinterpret_cast<HANDLE>(handle_));
  handle_ = -1;
}

bool RandomAccessFile::open(const std::string& path, uint64_t size, std::string* err, Mode mode) {
  close();
  path_ = path;
  HANDLE h = CreateFileW(wide(path).c_str(), mode == kReadExisting ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         mode == kReadExisting ? OPEN_EXISTING : OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return *err = path + ": " + last_error(), false;
  LARGE_INTEGER now{}, want{};
  want.QuadPart = LONGLONG(size);
  const BOOL sized = GetFileSizeEx(h, &now);
  if (sized && mode == kReadExisting && uint64_t(now.QuadPart) != size) {
    *err = path + ": unexpected file size";
    CloseHandle(h);
    return false;
  }
  if (!sized || (mode == kReadWrite && uint64_t(now.QuadPart) != size &&
                 (!SetFilePointerEx(h, want, nullptr, FILE_BEGIN) || !SetEndOfFile(h)))) {
    *err = path + ": " + last_error();
    CloseHandle(h);
    return false;
  }
  handle_ = reinterpret_cast<intptr_t>(h);
  return true;
}

bool RandomAccessFile::write_at(uint64_t offset, const void* data, size_t n, std::string* err) const {
  const auto* p = static_cast<const char*>(data);
  while (n > 0) {
    OVERLAPPED at{};
    at.Offset = DWORD(offset), at.OffsetHigh = DWORD(offset >> 32);
    DWORD w = 0;
    const DWORD want = DWORD(std::min<size_t>(n, 1u << 30));
    if (!WriteFile(reinterpret_cast<HANDLE>(handle_), p, want, &w, &at) || w == 0)
      return *err = path_ + ": " + last_error(), false;
    p += w, offset += w, n -= w;
  }
  return true;
}

bool RandomAccessFile::read_at(uint64_t offset, void* data, size_t n, std::string* err) const {
  auto* p = static_cast<char*>(data);
  while (n > 0) {
    OVERLAPPED at{};
    at.Offset = DWORD(offset), at.OffsetHigh = DWORD(offset >> 32);
    DWORD r = 0;
    const DWORD want = DWORD(std::min<size_t>(n, 1u << 30));
    if (!ReadFile(reinterpret_cast<HANDLE>(handle_), p, want, &r, &at))
      return *err = path_ + ": " + last_error(), false;
    if (r == 0)
      return *err = path_ + ": shorter than expected", false;
    p += r, offset += r, n -= r;
  }
  return true;
}

FileLock::~FileLock() {
  if (handle_ != -1) CloseHandle(reinterpret_cast<HANDLE>(handle_));
}

FileLock::Result FileLock::try_lock(std::string* err) {
  if (handle_ != -1) return Result::kLocked;
  HANDLE h = CreateFileW(wide(path_).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h != INVALID_HANDLE_VALUE) {
    handle_ = reinterpret_cast<intptr_t>(h);
    return Result::kLocked;
  }
  if (GetLastError() == ERROR_SHARING_VIOLATION) return Result::kBusy;
  *err = path_ + ": " + last_error();
  return Result::kError;
}

DynLib::~DynLib() {
  if (handle_) FreeLibrary(static_cast<HMODULE>(handle_));
}

bool DynLib::open(const std::string& path, std::string* err) {
  // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an absolute path, even for an explicitly named relative directory.
  std::error_code ec;
  const auto full = std::filesystem::absolute(wide(path), ec);
  if (ec) return *err = ec.message(), false;
  handle_ = LoadLibraryExW(full.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (!handle_) {
    *err = last_error();
    return false;
  }
  return true;
}

void* DynLib::symbol(const char* name) const {
  return handle_ ? reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name)) : nullptr;
}

std::vector<std::string> utf8_arguments(int argc, char** argv) {
  int n = 0;
  LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
  if (!w) return std::vector<std::string>(argv, argv + argc);
  std::vector<std::string> out;
  for (int i = 0; i < n; ++i) out.push_back(narrow(w[i]));
  LocalFree(w);
  return out;
}

void raw_utf8_stdio() {
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
  std::fflush(stdout);
  std::fflush(stderr);
  _setmode(_fileno(stdout), _O_BINARY);
  _setmode(_fileno(stderr), _O_BINARY);
}

std::string executable_dir() {
  std::wstring buf(MAX_PATH, L'\0');
  for (;;) {
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), DWORD(buf.size()));
    if (n == 0) return ".";
    if (n < buf.size()) {
      buf.resize(n);
      break;
    }
    buf.resize(buf.size() * 2);
  }
  std::string s = narrow(buf);
  const auto k = s.find_last_of("\\/");
  return k == std::string::npos ? "." : s.substr(0, k);
}

std::string getenv_str(const char* name, const std::string& fallback) {
  const auto key = wide(name);
  DWORD size = GetEnvironmentVariableW(key.c_str(), nullptr, 0);
  while (size > 1) {
    std::wstring value(size, L'\0');
    const DWORD length = GetEnvironmentVariableW(key.c_str(), value.data(), size);
    if (!length) break;
    if (length < size) {
      value.resize(length);
      return narrow(value);
    }
    size = length;  // another thread grew the variable between the two calls
  }
  return fallback;
}

bool env_exists(const char* name) {
  const auto key = wide(name);
  SetLastError(0);
  const DWORD size = GetEnvironmentVariableW(key.c_str(), nullptr, 0);
  return size > 0 || GetLastError() != ERROR_ENVVAR_NOT_FOUND;
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

double unix_seconds() {
  using namespace std::chrono;
  return duration<double>(system_clock::now().time_since_epoch()).count();
}

ProcessUsage process_usage() {
  ProcessUsage u;
  FILETIME created, exited, kernel, user;
  if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
    auto ticks = [](const FILETIME& t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
    u.cpu_seconds = double(ticks(kernel) + ticks(user)) * 1e-7;  // 100 ns ticks
  }
  PROCESS_MEMORY_COUNTERS pm{};
  if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pm, sizeof(pm))) u.rss_bytes = pm.WorkingSetSize;
  return u;
}

bool gpu_memory_paged() { return false; }  // the CUDA backend keeps Windows' floors itself (os.h)

uint64_t gpu_memory_floor() { return 256ull << 20; }  // matches backends/cuda WDDM kPiecesFloor

SystemMemory system_memory() {
  SystemMemory m;
  MEMORYSTATUSEX s{};
  s.dwLength = sizeof(s);
  if (GlobalMemoryStatusEx(&s)) {
    m.total = s.ullTotalPhys;
    m.available = s.ullAvailPhys;
  }
  return m;
}

double system_cpu_seconds() {
  FILETIME idle, kernel, user;  // kernel's time counts the idle time in
  if (!GetSystemTimes(&idle, &kernel, &user)) return 0;
  auto ticks = [](const FILETIME& t) { return (uint64_t(t.dwHighDateTime) << 32) | t.dwLowDateTime; };
  return double(ticks(kernel) + ticks(user) - ticks(idle)) * 1e-7;  // 100 ns ticks
}

void release_working_set() { SetProcessWorkingSetSize(GetCurrentProcess(), SIZE_T(-1), SIZE_T(-1)); }

int process_id() { return int(GetCurrentProcessId()); }

bool process_running(int pid) {
  if (pid <= 0) return false;
  HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, DWORD(pid));
  if (!h) return GetLastError() == ERROR_ACCESS_DENIED;  // there, not ours to open
  DWORD code = 0;
  const bool running = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
  CloseHandle(h);
  return running;
}

bool stdout_is_terminal() { return _isatty(_fileno(stdout)) != 0; }
bool stdin_is_terminal() { return _isatty(_fileno(stdin)) != 0; }
bool stderr_is_terminal() { return _isatty(_fileno(stderr)) != 0; }
bool owns_console() {
  DWORD pids[2];
  return stdout_is_terminal() && GetConsoleProcessList(pids, 2) == 1;
}

bool open_url(const std::string& url) {
  if (env_exists(env::kSshConnection) || env_exists(env::kSshTty)) return false;  // the browser would be the server's
  const auto r = reinterpret_cast<intptr_t>(
      ShellExecuteW(nullptr, L"open", wide(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL));
  return r > 32;
}

Picked pick_folder(const std::string& title, const std::string& start, std::string* path, std::string* err) {
  if (env_exists(env::kSshConnection) || env_exists(env::kSshTty))
    return *err = "no desktop to show a folder dialog on: this is a session over SSH", Picked::kUnavailable;
  USEROBJECTFLAGS station{};
  DWORD needed = 0;
  if (GetUserObjectInformationW(GetProcessWindowStation(), UOI_FLAGS, &station, sizeof station, &needed) &&
      !(station.dwFlags & WSF_VISIBLE))
    return *err = "no desktop to show a folder dialog on: flyx runs as a service", Picked::kUnavailable;
  Picked result = Picked::kUnavailable;
  // On a thread of its own, in a single-threaded apartment as the dialog needs: the caller's thread may use COM
  // another way.
  std::thread([&] {
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE))) {
      *err = "the folder dialog could not start (COM)";
      return;
    }
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) {
      DWORD options = 0;
      dialog->GetOptions(&options);
      dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
      if (!title.empty()) dialog->SetTitle(wide(title).c_str());
      IShellItem* at = nullptr;
      if (!start.empty() && SUCCEEDED(SHCreateItemFromParsingName(wide(start).c_str(), nullptr, IID_PPV_ARGS(&at)))) {
        dialog->SetFolder(at);
        at->Release();
      }
      // Owned by a window kept above the others: the dialog of a process in the background (the browser asked for
      // it, not flyx's own window) would open behind them.  That window is a point in the middle of the screen the
      // pointer is on, where whoever asked is looking: the dialog centres itself on its owner.
      POINT pointer{};
      GetCursorPos(&pointer);
      MONITORINFO screen{};
      screen.cbSize = sizeof screen;
      int x = 0, y = 0;
      if (GetMonitorInfoW(MonitorFromPoint(pointer, MONITOR_DEFAULTTOPRIMARY), &screen))
        x = (screen.rcWork.left + screen.rcWork.right) / 2, y = (screen.rcWork.top + screen.rcWork.bottom) / 2;
      const HWND owner = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", L"", WS_POPUP, x, y, 0, 0, nullptr,
                                         nullptr, nullptr, nullptr);
      if (owner) SetForegroundWindow(owner);
      const HRESULT shown = dialog->Show(owner);
      IShellItem* item = nullptr;
      if (SUCCEEDED(shown) && SUCCEEDED(dialog->GetResult(&item))) {
        PWSTR chosen = nullptr;
        if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &chosen)) && chosen) {
          *path = win::to_utf8(chosen);
          CoTaskMemFree(chosen);
          result = Picked::kFolder;
        } else {
          *err = "the folder chosen has no path on this machine";
        }
        item->Release();
      } else if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
        result = Picked::kCancelled;
      } else {
        *err = "the folder dialog did not open";
      }
      if (owner) DestroyWindow(owner);
      dialog->Release();
    } else {
      *err = "this Windows has no folder dialog";
    }
    CoUninitialize();
  }).join();
  return result;
}

bool shell_window_supported() { return true; }

namespace {

std::string base64(const unsigned char* p, size_t n) {
  static const char k[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (size_t i = 0; i < n; i += 3) {
    const uint32_t v = uint32_t(p[i]) << 16 | (i + 1 < n ? uint32_t(p[i + 1]) << 8 : 0) | (i + 2 < n ? p[i + 2] : 0);
    out += k[v >> 18 & 63];
    out += k[v >> 12 & 63];
    out += i + 1 < n ? k[v >> 6 & 63] : '=';
    out += i + 2 < n ? k[v & 63] : '=';
  }
  return out;
}

// The user's own environment, as a terminal they open gets it (the system's and the account's variables), with `set`
// on top, as CreateProcessW takes it: NAME=VALUE strings, each ended by a zero, sorted by name without regard to
// case, then one more zero.  Not this process's: what started flyx (a virtual environment, CUDA's paths, a coding
// agent whose session variables would make the agent in the window take itself for its child) is not the window's.
// This process's where the user's cannot be had.
std::wstring environment_with(const std::vector<std::pair<std::string, std::string>>& set) {
  std::vector<std::wstring> vars;
  auto name_of = [](const std::wstring& v) { return v.substr(0, v.find(L'=', 1)); };  // "=C:=C:\x" names "=C:"
  auto same = [](const std::wstring& a, const std::wstring& b) { return _wcsicmp(a.c_str(), b.c_str()) == 0; };
  auto take = [&](const wchar_t* block) {
    for (const wchar_t* v = block; *v; v += wcslen(v) + 1) {
      std::wstring var(v);
      const std::wstring name = name_of(var);
      if (std::none_of(set.begin(), set.end(), [&](const auto& kv) { return same(name, wide(kv.first)); }))
        vars.push_back(std::move(var));
    }
  };
  HANDLE token = nullptr;
  void* user = nullptr;
  if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_DUPLICATE, &token) &&
      CreateEnvironmentBlock(&user, token, FALSE)) {
    take(static_cast<const wchar_t*>(user));
    DestroyEnvironmentBlock(user);
  } else if (wchar_t* block = GetEnvironmentStringsW()) {
    take(block);
    FreeEnvironmentStringsW(block);
  }
  if (token) CloseHandle(token);
  for (const auto& [name, value] : set) vars.push_back(wide(name) + L"=" + wide(value));
  std::sort(vars.begin(), vars.end(), [&](const std::wstring& a, const std::wstring& b) {
    return _wcsicmp(name_of(a).c_str(), name_of(b).c_str()) < 0;
  });
  std::wstring out;
  for (const auto& v : vars) out += v + L'\0';
  return out + L'\0';
}

}  // namespace

bool open_shell_window(const std::string& dir, const std::string& command,
                       const std::vector<std::pair<std::string, std::string>>& env, std::string* err) {
  if (env_exists(env::kSshConnection) || env_exists(env::kSshTty))
    return *err = "no desktop to open a window on: this is a session over SSH", false;
  USEROBJECTFLAGS station{};
  DWORD needed = 0;
  if (GetUserObjectInformationW(GetProcessWindowStation(), UOI_FLAGS, &station, sizeof station, &needed) &&
      !(station.dwFlags & WSF_VISIBLE))
    return *err = "no desktop to open a window on: flyx runs as a service", false;
  // Windows' own PowerShell, by its place under System32 rather than by a search of PATH.
  wchar_t system[MAX_PATH];
  const UINT n = GetSystemDirectoryW(system, MAX_PATH);
  if (n == 0 || n >= MAX_PATH) return *err = "the system folder was not found", false;
  const std::wstring exe = std::wstring(system, n) + L"\\WindowsPowerShell\\v1.0\\powershell.exe";
  // The command as -EncodedCommand takes it (base64 of UTF-16LE): no quoting of it can go wrong.
  const std::wstring text = wide(command);
  const std::string encoded = base64(reinterpret_cast<const unsigned char*>(text.data()), text.size() * sizeof(wchar_t));
  std::wstring line = L"\"" + exe + L"\" -NoExit -NoLogo -EncodedCommand " + wide(encoded);
  std::wstring environment = environment_with(env);
  STARTUPINFOW si{};
  si.cb = sizeof si;
  PROCESS_INFORMATION pi{};
  if (!CreateProcessW(exe.c_str(), line.data(), nullptr, nullptr, FALSE, CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT,
                      environment.data(), wide(dir).c_str(), &si, &pi))
    return *err = "PowerShell did not start (error " + std::to_string(GetLastError()) + ")", false;
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return true;
}

namespace {

// capture_stderr: the sink, and standard error's handle before (nullptr: no capture).
std::mutex g_capture_mu;
std::function<void(const std::string&)> g_capture_sink;
HANDLE g_original_stderr = nullptr;

void deliver(const std::string& line) {
  std::function<void(const std::string&)> sink;
  {
    std::lock_guard<std::mutex> lock(g_capture_mu);
    sink = g_capture_sink;
  }
  if (sink) sink(line);
}

}  // namespace

bool capture_stderr(std::function<void(const std::string& line)> sink) {
  std::lock_guard<std::mutex> lock(g_capture_mu);
  if (g_original_stderr) {
    g_capture_sink = std::move(sink);
    return true;
  }
  HANDLE r = nullptr, w = nullptr;
  if (!CreatePipe(&r, &w, nullptr, 1 << 16)) return false;
  std::fflush(stderr);
  // The handle standard error had, kept open: fd 2 is moved onto the pipe below, which closes its old handle.
  HANDLE original = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &original, 0, FALSE,
                       DUPLICATE_SAME_ACCESS))
    original = nullptr;
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(w), _O_BINARY);  // owns w from here
  if (fd < 0 || _dup2(fd, 2) != 0) {
    if (fd >= 0) _close(fd);
    else CloseHandle(w);
    CloseHandle(r);
    if (original) CloseHandle(original);
    return false;
  }
  _close(fd);
  // Libraries loaded after this (the backends, each with a C runtime of its own) take standard error from here.
  SetStdHandle(STD_ERROR_HANDLE, reinterpret_cast<HANDLE>(_get_osfhandle(2)));
  g_original_stderr = original ? original : INVALID_HANDLE_VALUE;
  g_capture_sink = std::move(sink);
  std::thread([r] {
    std::string line;
    char buf[4096];
    DWORD n = 0;
    while (ReadFile(r, buf, sizeof(buf), &n, nullptr) && n > 0) {
      for (DWORD i = 0; i < n; ++i) {
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
  HANDLE h = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_capture_mu);
    h = g_original_stderr;
  }
  if (!h) {
    std::fwrite(text.data(), 1, text.size(), stderr);
    std::fflush(stderr);
    return;
  }
  if (h == INVALID_HANDLE_VALUE) return;
  DWORD written = 0;
  WriteFile(h, text.data(), DWORD(text.size()), &written, nullptr);
}

void sleep_ms(int ms) { Sleep(DWORD(ms)); }

void sleep_us(int64_t us) {
  struct Timer {
    HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    ~Timer() {
      if (h) CloseHandle(h);
    }
  };
  thread_local Timer timer;  // one per sleeping thread
  LARGE_INTEGER due;
  due.QuadPart = -us * 10;  // relative, in 100 ns
  if (timer.h && SetWaitableTimer(timer.h, &due, 0, nullptr, nullptr, FALSE)) {
    WaitForSingleObject(timer.h, INFINITE);
    return;
  }
  std::this_thread::sleep_for(std::chrono::microseconds(us));  // before Windows 10 1803: the tick
}

int cpu_count() {
  const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  return n > 0 ? int(n) : 1;
}

struct Terminal::Saved {
  DWORD in_mode = 0, out_mode = 0;
};

Terminal::Terminal() = default;

Terminal::~Terminal() {
  if (!saved_) return;
  SetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), saved_->in_mode);
  SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), saved_->out_mode);
}

bool Terminal::raw() {
  HANDLE in = GetStdHandle(STD_INPUT_HANDLE), out = GetStdHandle(STD_OUTPUT_HANDLE);
  auto s = std::make_unique<Saved>();
  if (!GetConsoleMode(in, &s->in_mode) || !GetConsoleMode(out, &s->out_mode)) return false;
  // keys one at a time, no echo; the escape sequences the panel draws with understood; suppress auto-wrap scroll
  const DWORD out_vt = s->out_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
  if (!SetConsoleMode(in, s->in_mode & ~DWORD(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT)) ||
      (!SetConsoleMode(out, out_vt | DISABLE_NEWLINE_AUTO_RETURN) && !SetConsoleMode(out, out_vt)))
    return false;
  saved_ = std::move(s);
  return true;
}

void Terminal::size(int* cols, int* rows) const {
  CONSOLE_SCREEN_BUFFER_INFO b{};
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &b)) {
    *cols = b.srWindow.Right - b.srWindow.Left + 1;
    *rows = b.srWindow.Bottom - b.srWindow.Top + 1;
  } else {
    *cols = 100;
    *rows = 32;
  }
}

int Terminal::key(int timeout_ms) {
  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  const double end = now_ms() + timeout_ms;
  for (;;) {
    const double left = end - now_ms();
    if (WaitForSingleObject(in, left > 0 ? DWORD(left) : 0) != WAIT_OBJECT_0) return -1;
    INPUT_RECORD r{};
    DWORD n = 0;
    if (!ReadConsoleInputA(in, &r, 1, &n) || n == 0) return -1;
    if (r.EventType == KEY_EVENT && r.Event.KeyEvent.bKeyDown && r.Event.KeyEvent.uChar.AsciiChar)
      return static_cast<unsigned char>(r.Event.KeyEvent.uChar.AsciiChar);
    if (left <= 0) return -1;  // other events (focus, mouse, key up): none of them a key
  }
}

}  // namespace flyx::os
