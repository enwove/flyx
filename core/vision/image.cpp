// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/vision/image.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "core/hub/sha256.h"
#include "core/os/os.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_FAILURE_USERMSG
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4244 4456 4457)
#endif
#include "stb/stb_image.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace flyx::vision {
namespace {

int b64_val(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

bool iequals_prefix(std::string_view s, std::string_view pfx) {
  if (s.size() < pfx.size()) return false;
  for (size_t i = 0; i < pfx.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(s[i])) != std::tolower(static_cast<unsigned char>(pfx[i])))
      return false;
  return true;
}

std::string mime_from_path(const std::string& path) {
  const size_t dot = path.rfind('.');
  if (dot == std::string::npos) return "image/octet-stream";
  std::string ext = path.substr(dot + 1);
  for (char& c : ext) c = char(std::tolower(static_cast<unsigned char>(c)));
  if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
  if (ext == "png") return "image/png";
  if (ext == "bmp") return "image/bmp";
  if (ext == "gif") return "image/gif";
  if (ext == "tga") return "image/tga";
  return "image/" + ext;
}

}  // namespace

std::array<uint8_t, 32> sha256_bytes(const void* data, size_t n) {
  hub::Sha256 h;
  h.update(data, n);
  const std::string hex = h.hex();
  std::array<uint8_t, 32> out{};
  auto hex_nibble = [](char c) -> uint8_t {
    if (c >= '0' && c <= '9') return uint8_t(c - '0');
    if (c >= 'a' && c <= 'f') return uint8_t(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return uint8_t(c - 'A' + 10);
    return 0;
  };
  for (size_t i = 0; i < 32 && 2 * i + 1 < hex.size(); ++i)
    out[i] = uint8_t((hex_nibble(hex[2 * i]) << 4) | hex_nibble(hex[2 * i + 1]));
  return out;
}

bool decode_base64(std::string_view in, std::vector<uint8_t>* out, std::string* err) {
  out->clear();
  out->reserve((in.size() / 4) * 3 + 3);
  uint32_t buf = 0;
  int bits = 0;
  bool ended = false;
  for (unsigned char c : in) {
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
    if (c == '=') {
      ended = true;
      continue;
    }
    if (ended) {
      if (err) *err = "invalid base64 data: characters after padding";
      return false;
    }
    const int v = b64_val(c);
    if (v < 0) {
      if (err) *err = "invalid base64 character";
      return false;
    }
    buf = (buf << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out->push_back(uint8_t((buf >> bits) & 0xFFu));
      if (out->size() > kMaxRawImageBytes) {
        if (err) *err = "image exceeds maximum allowed size (32 MiB)";
        return false;
      }
    }
  }
  if (out->empty()) {
    if (err) *err = "empty base64 image data";
    return false;
  }
  return true;
}

bool decode_image_bytes(const uint8_t* data, size_t n, interfaces::RgbImage* out, std::string* err) {
  if (!data || n == 0) {
    if (err) *err = "empty image data";
    return false;
  }
  if (n > kMaxRawImageBytes || n > size_t(INT32_MAX)) {
    if (err) *err = "image exceeds maximum allowed size (32 MiB)";
    return false;
  }
  int w = 0, h = 0, comp = 0;
  if (!stbi_info_from_memory(data, int(n), &w, &h, &comp)) {
    if (err) *err = std::string("failed to decode image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "unsupported format");
    return false;
  }
  if (w <= 0 || h <= 0 || w > kMaxImageDim || h > kMaxImageDim ||
      size_t(w) * size_t(h) * 3 > kMaxDecodedBytes) {
    if (err) *err = "image dimensions too large (" + std::to_string(w) + "x" + std::to_string(h) + ")";
    return false;
  }
  unsigned char* rgb = stbi_load_from_memory(data, int(n), &w, &h, &comp, 3);
  if (!rgb) {
    if (err) *err = std::string("failed to decode image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "corrupt image");
    return false;
  }
  out->width = w;
  out->height = h;
  out->rgb.assign(rgb, rgb + size_t(w) * size_t(h) * 3);
  stbi_image_free(rgb);
  return true;
}

bool make_image_from_bytes(std::vector<uint8_t> bytes, std::string mime, std::string detail,
                           interfaces::ImageInput* out, std::string* err) {
  if (bytes.empty()) {
    if (err) *err = "empty image data";
    return false;
  }
  if (bytes.size() > kMaxRawImageBytes) {
    if (err) *err = "image exceeds maximum allowed size (32 MiB)";
    return false;
  }
  out->sha256 = sha256_bytes(bytes.data(), bytes.size());
  out->bytes = std::move(bytes);
  out->mime = std::move(mime);
  out->detail = std::move(detail);
  return true;
}

bool parse_image_data_uri(std::string_view uri, interfaces::ImageInput* out, std::string* err) {
  if (iequals_prefix(uri, "http://") || iequals_prefix(uri, "https://")) {
    if (err) *err = "remote image URLs are not fetched; send a base64 data:image/... URI";
    return false;
  }
  if (!iequals_prefix(uri, "data:")) {
    if (err) *err = "image url must be a data:image/...;base64,... URI";
    return false;
  }
  const size_t comma = uri.find(',');
  if (comma == std::string_view::npos) {
    if (err) *err = "malformed data URI: missing comma";
    return false;
  }
  const std::string_view meta = uri.substr(5, comma - 5);
  const std::string_view payload = uri.substr(comma + 1);
  const size_t semi = meta.find(';');
  if (semi == std::string_view::npos || !iequals_prefix(meta.substr(semi + 1), "base64")) {
    if (err) *err = "data URI must be base64-encoded (data:image/...;base64,...)";
    return false;
  }
  std::string mime(meta.substr(0, semi));
  if (!iequals_prefix(mime, "image/")) {
    if (err) *err = "data URI media type must be image/* (got '" + mime + "')";
    return false;
  }
  std::vector<uint8_t> bytes;
  if (!decode_base64(payload, &bytes, err)) return false;
  return make_image_from_bytes(std::move(bytes), std::move(mime), "", out, err);
}

bool load_image_file(const std::string& path, interfaces::ImageInput* out, std::string* err) {
  os::MappedFile mf;
  if (!mf.open(path, err)) return false;
  if (mf.size() == 0) {
    if (err) *err = "image file is empty: " + path;
    return false;
  }
  if (mf.size() > kMaxRawImageBytes) {
    if (err) *err = "image file exceeds maximum allowed size (32 MiB): " + path;
    return false;
  }
  std::vector<uint8_t> bytes(mf.data(), mf.data() + mf.size());
  return make_image_from_bytes(std::move(bytes), mime_from_path(path), "", out, err);
}

bool VisionCache::get(const VisionCacheKey& key, std::shared_ptr<const CachedVision>* out) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(key);
  if (it == index_.end()) return false;
  lru_.splice(lru_.begin(), lru_, it->second);
  *out = it->second->val;
  return true;
}

void VisionCache::put(const VisionCacheKey& key, std::shared_ptr<const CachedVision> val) {
  if (!val) return;
  const size_t b = val->embeds.size() * sizeof(float) + val->tokens.size() * sizeof(int32_t);
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(key);
  if (it != index_.end()) {
    cur_bytes_ -= it->second->bytes;
    it->second->val = std::move(val);
    it->second->bytes = b;
    cur_bytes_ += b;
    lru_.splice(lru_.begin(), lru_, it->second);
  } else {
    if (max_bytes_ > 0 && b > max_bytes_) return;
    lru_.push_front(Entry{key, std::move(val), b});
    index_[key] = lru_.begin();
    cur_bytes_ += b;
  }
  while (max_bytes_ > 0 && cur_bytes_ > max_bytes_ && lru_.size() > 1) {
    const Entry& back = lru_.back();
    cur_bytes_ -= back.bytes;
    index_.erase(back.key);
    lru_.pop_back();
  }
}

void VisionCache::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  lru_.clear();
  index_.clear();
  cur_bytes_ = 0;
}

size_t VisionCache::bytes() const {
  std::lock_guard<std::mutex> lock(mu_);
  return cur_bytes_;
}

}  // namespace flyx::vision
