// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "backends/cuda/runtime/offload/expert_profile.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace flyx::cuda {
namespace {
constexpr char kMagic[8] = {'F', 'L', 'Y', 'X', 'E', 'X', 'P', '1'};
constexpr uint32_t kVersion = 2;
constexpr size_t kEdge = 64u << 10;
// Serializes read/add/rename across engines and processes. The OS releases the lock on exit.
class ProfileLock {
 public:
  explicit ProfileLock(std::filesystem::path path) {
    path += ".lock";
#ifdef _WIN32
    handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                          OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    OVERLAPPED range{};
    if (held() && !LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &range)) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
#else
    fd_ = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd_ < 0) return;
    int rc;
    do { rc = flock(fd_, LOCK_EX); } while (rc != 0 && errno == EINTR);
    if (rc != 0) {
      ::close(fd_);
      fd_ = -1;
    }
#endif
  }
  ProfileLock(const ProfileLock&) = delete;
  ProfileLock& operator=(const ProfileLock&) = delete;
#ifdef _WIN32
  ~ProfileLock() { if (held()) CloseHandle(handle_); }
  bool held() const { return handle_ != INVALID_HANDLE_VALUE; }
 private:
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  ~ProfileLock() { if (held()) ::close(fd_); }
  bool held() const { return fd_ >= 0; }
 private:
  int fd_ = -1;
#endif
};

size_t file_size_for(const std::vector<ProfileLayer>& layers) {
  size_t size = 32;  // header and checksum
  for (const auto& layer : layers) size += 16 + sizeof(uint64_t) * layer.scores.size();
  return size;
}
}  // namespace

uint64_t add_routes(uint64_t a, uint64_t b) {
  return a + std::min(UINT64_MAX - a, b);
}

uint64_t fingerprint_bytes(uint64_t h, const void* data, size_t bytes) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < bytes; i += 8) {
    uint64_t word = 0;
    std::memcpy(&word, p + i, std::min<size_t>(8, bytes - i));
    h = (h ^ word) * 1099511628211ull;
  }
  return h;
}

uint64_t fingerprint_tensor(uint64_t h, const flyx_tensor_view* v, bool whole) {
  if (!v) return h;
  const int64_t meta[6] = {v->dtype, v->ndim, v->shape[0], v->shape[1], v->shape[2], int64_t(v->nbytes)};
  h = fingerprint_bytes(h, meta, sizeof(meta));
  if (!v->data) return h;
  const auto* bytes = static_cast<const unsigned char*>(v->data);
  if (whole || v->nbytes <= 2 * kEdge) return fingerprint_bytes(h, bytes, v->nbytes);
  h = fingerprint_bytes(h, bytes, kEdge);
  return fingerprint_bytes(h, bytes + v->nbytes - kEdge, kEdge);
}

std::filesystem::path expert_profile_folder(const std::string& option) {
  namespace fs = std::filesystem;
  if (option == "0") return {};
  if (option != "auto") return fs::path(std::u8string(option.begin(), option.end()));
  auto set = [](const auto* value) { return value && *value ? fs::path(value) : fs::path(); };
#if defined(_WIN32)
  fs::path base = set(_wgetenv(L"LOCALAPPDATA"));
  if (base.empty() && !set(_wgetenv(L"USERPROFILE")).empty())
    base = set(_wgetenv(L"USERPROFILE")) / "AppData" / "Local";
#else
  fs::path base = set(std::getenv("XDG_CACHE_HOME"));
  if (base.empty() && !set(std::getenv("HOME")).empty()) base = set(std::getenv("HOME")) / ".cache";
#endif
  return base.empty() ? fs::path() : base / "flyx" / "expert_profiles";
}

std::filesystem::path expert_profile_file(const std::filesystem::path& folder, uint64_t fingerprint) {
  char name[32];
  std::snprintf(name, sizeof(name), "v%u-%016llx.bin", kVersion, static_cast<unsigned long long>(fingerprint));
  return folder / name;
}

std::string encode_expert_profile(uint64_t fingerprint, const std::vector<ProfileLayer>& layers) {
  std::string bytes(kMagic, sizeof(kMagic));
  auto put = [&](const auto& value) { bytes.append(reinterpret_cast<const char*>(&value), sizeof(value)); };
  put(kVersion);
  put(uint32_t(layers.size()));
  put(fingerprint);
  for (const auto& layer : layers) {
    put(int32_t(layer.weights.first));
    put(int32_t(layer.weights.second));
    put(uint32_t(layer.scores.size()));
    put(uint32_t(0));
    bytes.append(reinterpret_cast<const char*>(layer.scores.data()), layer.scores.size() * sizeof(uint64_t));
  }
  put(fingerprint_bytes(kFingerprintSeed, bytes.data(), bytes.size()));
  return bytes;
}

bool decode_expert_profile(const std::string& bytes, uint64_t fingerprint, std::vector<ProfileLayer>* layers) {
  if (bytes.size() != file_size_for(*layers)) return false;
  size_t at = 0;
  auto get = [&](auto* value) {
    if (bytes.size() - at < sizeof(*value)) return false;
    std::memcpy(value, bytes.data() + at, sizeof(*value));
    at += sizeof(*value);
    return true;
  };
  if (bytes.size() < sizeof(kMagic) || std::memcmp(bytes.data(), kMagic, sizeof(kMagic))) return false;
  at = sizeof(kMagic);
  uint32_t version = 0, count = 0;
  uint64_t print = 0;
  if (!get(&version) || version != kVersion || !get(&count) || count != layers->size() || !get(&print) ||
      print != fingerprint)
    return false;
  std::vector<std::vector<uint64_t>> scores;
  for (const auto& layer : *layers) {
    int32_t gate = 0, down = 0;
    uint32_t n = 0, pad = 0;
    if (!get(&gate) || !get(&down) || !get(&n) || !get(&pad) || gate != layer.weights.first ||
        down != layer.weights.second || n != layer.scores.size() || (bytes.size() - at) / sizeof(uint64_t) < n)
      return false;
    scores.emplace_back(n);
    std::memcpy(scores.back().data(), bytes.data() + at, n * sizeof(uint64_t));
    at += n * sizeof(uint64_t);
  }
  const uint64_t expected = fingerprint_bytes(kFingerprintSeed, bytes.data(), at);
  uint64_t checksum = 0;
  if (!get(&checksum) || checksum != expected || at != bytes.size()) return false;
  for (size_t i = 0; i < scores.size(); ++i) (*layers)[i].scores = std::move(scores[i]);
  return true;
}

bool load_expert_profile(const std::filesystem::path& file, uint64_t fingerprint, std::vector<ProfileLayer>* layers) {
  // Bound the read by the model, even when a corrupt file declares an enormous layer or has a long tail.
  std::ifstream in(file, std::ios::binary | std::ios::ate);
  const size_t size = file_size_for(*layers);
  if (!in || in.tellg() != std::streamoff(size)) return false;
  std::string bytes(size, '\0');
  in.seekg(0);
  return in.read(bytes.data(), std::streamsize(size)) && decode_expert_profile(bytes, fingerprint, layers);
}

bool merge_expert_profile(const std::filesystem::path& file, uint64_t fingerprint,
                          std::vector<ProfileLayer>* increments) noexcept {
  namespace fs = std::filesystem;
  try {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    if (ec) return false;
    ProfileLock lock(file);
    if (!lock.held()) return false;
    auto totals = *increments;
    if (load_expert_profile(file, fingerprint, &totals))
      for (size_t i = 0; i < totals.size(); ++i)
        for (size_t e = 0; e < totals[i].scores.size(); ++e)
          totals[i].scores[e] = add_routes(totals[i].scores[e], (*increments)[i].scores[e]);
    const auto bytes = encode_expert_profile(fingerprint, totals);
    auto partial = file;
    partial += ".new";
    std::ofstream out(partial, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
    out.close();
    if (out) fs::rename(partial, file, ec);
    if (!out || ec) {
      fs::remove(partial, ec);
      return false;
    }
    *increments = std::move(totals);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace flyx::cuda
