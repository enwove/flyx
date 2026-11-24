// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/vision_disk.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace flyx::cuda {

namespace {

std::atomic<uint64_t> g_seq{0};

#ifdef _WIN32
std::string win_err(DWORD code) {
  char buf[256] = {};
  const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, buf,
                                 sizeof(buf) - 1, nullptr);
  std::string s(buf, n);
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ')) s.pop_back();
  return s.empty() ? "error " + std::to_string(code) : s;
}
#endif

}  // namespace

VisionDiskStore::~VisionDiskStore() { close(); }

void VisionDiskStore::close() {
  if (map_base_) {
#ifdef _WIN32
    UnmapViewOfFile(map_base_);
#else
    ::munmap(const_cast<uint8_t*>(map_base_), map_len_);
#endif
    map_base_ = nullptr;
    map_len_ = 0;
  }
  map_tried_ = false;
  if (handle_ == -1) return;
#ifdef _WIN32
  CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
  ::close(static_cast<int>(handle_));
#endif
  handle_ = -1;
  size_ = 0;
}

void VisionDiskStore::ensure_open() {
  if (handle_ != -1) return;
#ifdef _WIN32
  wchar_t dir[MAX_PATH + 2] = {};
  const DWORD dlen = GetTempPathW(MAX_PATH + 1, dir);
  if (dlen == 0 || dlen > MAX_PATH)
    throw std::runtime_error("vision disk store: GetTempPathW failed (" + win_err(GetLastError()) + ")");
  const DWORD pid = GetCurrentProcessId();
  const ULONGLONG tick = GetTickCount64();
  for (int attempt = 0; attempt < 64; ++attempt) {
    const uint64_t id = g_seq.fetch_add(1, std::memory_order_relaxed);
    const std::wstring path = std::wstring(dir) + L"flyx-vis-" + std::to_wstring(pid) + L"-" + std::to_wstring(tick) +
                              L"-" + std::to_wstring(id) + L".bin";
    // FILE_ATTRIBUTE_NORMAL (not FILE_ATTRIBUTE_TEMPORARY) + FlushFileBuffers ensures Windows moves
    // written pages to the clean Standby list so GlobalMemoryStatusEx counts them as available RAM.
    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, CREATE_NEW,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
      handle_ = reinterpret_cast<intptr_t>(h);
      size_ = 0;
      return;
    }
    const DWORD err = GetLastError();
    if (err != ERROR_FILE_EXISTS)
      throw std::runtime_error("vision disk store: cannot create temp file: " + win_err(err) +
                               " (use --vision-device host or gpu)");
  }
  throw std::runtime_error("vision disk store: could not create unique temp file (use --vision-device host or gpu)");
#else
  const char* env = std::getenv("TMPDIR");
  std::string dir = (env && *env) ? std::string(env) : std::string("/tmp");
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  std::string tmpl = dir + "/flyx-vis-XXXXXX";
  const int fd = ::mkstemp(tmpl.data());
  if (fd < 0)
    throw std::runtime_error("vision disk store: cannot create temp file in " + dir + ": " + std::strerror(errno) +
                             " (use --vision-device host or gpu)");
  ::fcntl(fd, F_SETFD, FD_CLOEXEC);
  ::unlink(tmpl.c_str());  // auto-reclaim inode immediately when fd closes or process terminates
  handle_ = fd;
  size_ = 0;
#endif
}

uint64_t VisionDiskStore::append(const void* data, size_t nbytes) {
  if (nbytes == 0) return size_;
  ensure_open();
  const uint64_t off = (size_ + 255) & ~uint64_t(255);
  const auto* p = static_cast<const char*>(data);
  uint64_t cur = off;
  size_t left = nbytes;
#ifdef _WIN32
  HANDLE h = reinterpret_cast<HANDLE>(handle_);
  while (left > 0) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(cur);
    ov.OffsetHigh = static_cast<DWORD>(cur >> 32);
    DWORD wrote = 0;
    const DWORD want = static_cast<DWORD>(std::min<size_t>(left, 1u << 30));
    if (!WriteFile(h, p, want, &wrote, &ov) || wrote == 0)
      throw std::runtime_error("vision disk store: WriteFile failed: " + win_err(GetLastError()) +
                               " (check temp disk space or use --vision-device host)");
    p += wrote;
    cur += wrote;
    left -= wrote;
  }
#else
  const int fd = static_cast<int>(handle_);
  while (left > 0) {
    const ssize_t wrote = ::pwrite(fd, p, left, static_cast<off_t>(cur));
    if (wrote < 0 && errno == EINTR) continue;
    if (wrote <= 0)
      throw std::runtime_error("vision disk store: pwrite failed: " + std::string(std::strerror(errno)) +
                               " (check temp disk space or use --vision-device host)");
    p += wrote;
    cur += static_cast<uint64_t>(wrote);
    left -= static_cast<size_t>(wrote);
  }
#endif
  size_ = off + nbytes;
  return off;
}

void VisionDiskStore::flush_clean() {
  if (handle_ == -1 || size_ == 0) return;
#ifdef _WIN32
  if (!FlushFileBuffers(reinterpret_cast<HANDLE>(handle_)))
    throw std::runtime_error("vision disk store: FlushFileBuffers failed: " + win_err(GetLastError()));
#else
  if (::fsync(static_cast<int>(handle_)) != 0)
    throw std::runtime_error("vision disk store: fsync failed: " + std::string(std::strerror(errno)));
#endif
}

void VisionDiskStore::ensure_mapped() const {
  if (map_tried_ || handle_ == -1 || size_ == 0) return;
  std::lock_guard<std::mutex> lk(map_mu_);
  if (map_tried_) return;
  map_tried_ = true;
#ifdef _WIN32
  HANDLE fm = CreateFileMappingW(reinterpret_cast<HANDLE>(handle_), nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!fm) return;
  void* p = MapViewOfFile(fm, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(fm);  // the mapped view retains the section object
  if (!p) return;
  map_base_ = static_cast<const uint8_t*>(p);
  map_len_ = static_cast<size_t>(size_);
#else
  void* p = ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_SHARED, static_cast<int>(handle_), 0);
  if (p == MAP_FAILED) return;
  map_base_ = static_cast<const uint8_t*>(p);
  map_len_ = static_cast<size_t>(size_);
  ::madvise(const_cast<uint8_t*>(map_base_), map_len_, MADV_SEQUENTIAL);
#endif
}

void VisionDiskStore::read_at(uint64_t offset, void* dst, size_t nbytes) const {
  if (nbytes == 0) return;
  if (handle_ == -1 || offset + nbytes > size_)
    throw std::runtime_error("vision disk store: read out of range");
  ensure_mapped();
  if (map_base_ && offset + nbytes <= map_len_) {
    std::memcpy(dst, map_base_ + offset, nbytes);
    return;
  }
  auto* p = static_cast<char*>(dst);
  uint64_t cur = offset;
  size_t left = nbytes;
#ifdef _WIN32
  HANDLE h = reinterpret_cast<HANDLE>(handle_);
  while (left > 0) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(cur);
    ov.OffsetHigh = static_cast<DWORD>(cur >> 32);
    DWORD got = 0;
    const DWORD want = static_cast<DWORD>(std::min<size_t>(left, 1u << 30));
    if (!ReadFile(h, p, want, &got, &ov) || got == 0)
      throw std::runtime_error("vision disk store: ReadFile failed: " + win_err(GetLastError()));
    p += got;
    cur += got;
    left -= got;
  }
#else
  const int fd = static_cast<int>(handle_);
  while (left > 0) {
    const ssize_t got = ::pread(fd, p, left, static_cast<off_t>(cur));
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0)
      throw std::runtime_error("vision disk store: pread failed: " +
                               std::string(got == 0 ? "unexpected EOF" : std::strerror(errno)));
    p += got;
    cur += static_cast<uint64_t>(got);
    left -= static_cast<size_t>(got);
  }
#endif
}

}  // namespace flyx::cuda
