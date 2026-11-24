// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Image decoding (PNG/JPEG/BMP/GIF/TGA via stb_image), base64 data URI parsing,
// file loading, SHA-256 hashing, and bounded LRU cache for encoded vision embeddings.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/interfaces/chat.h"
#include "core/interfaces/vision.h"

namespace flyx::vision {

// Maximum decoded image size (prevents OOM/DoS on malformed inputs).
constexpr int kMaxImageDim = 16384;
constexpr size_t kMaxDecodedBytes = size_t(64) << 20;  // 64 MiB RGB
constexpr size_t kMaxRawImageBytes = size_t(32) << 20;  // 32 MiB compressed

std::array<uint8_t, 32> sha256_bytes(const void* data, size_t n);

bool decode_base64(std::string_view in, std::vector<uint8_t>* out, std::string* err);

// Decode compressed image bytes (PNG/JPEG/BMP/GIF/TGA) into RGB888.
bool decode_image_bytes(const uint8_t* data, size_t n, interfaces::RgbImage* out, std::string* err);

// Wrap raw compressed image bytes into an ImageInput with SHA-256 content identity.
bool make_image_from_bytes(std::vector<uint8_t> bytes, std::string mime, std::string detail,
                           interfaces::ImageInput* out, std::string* err);

// Parse a base64 data URI ("data:image/<type>;base64,<data>"). Refuses remote http(s) URLs.
bool parse_image_data_uri(std::string_view uri, interfaces::ImageInput* out, std::string* err);

// Load an image file from disk (UTF-8 path) into an ImageInput.
bool load_image_file(const std::string& path, interfaces::ImageInput* out, std::string* err);

// Cached vision encoder output for one image under one token budget / detail setting.
struct CachedVision {
  std::vector<int32_t> tokens;         // expanded placeholder token IDs
  std::vector<int32_t> embed_offsets;  // offsets in `tokens` receiving embeddings (empty = all 0..n_embeds-1)
  int32_t n_embeds = 0;                // embedding rows emitted by the vision encoder
  std::vector<int32_t> grid_thw;       // [3] = {T, H_patches, W_patches}
  std::vector<float> embeds;           // [n_embeds * out_hidden]
  int out_hidden = 0;
};

struct VisionCacheKey {
  std::array<uint8_t, 32> sha256{};
  std::string detail;
  int max_tokens = 0;

  bool operator==(const VisionCacheKey& o) const {
    return max_tokens == o.max_tokens && sha256 == o.sha256 && detail == o.detail;
  }
};

struct VisionCacheKeyHash {
  size_t operator()(const VisionCacheKey& k) const noexcept {
    size_t h = 0;
    for (size_t i = 0; i < k.sha256.size(); i += 8) {
      size_t v = 0;
      for (size_t j = 0; j < 8 && i + j < k.sha256.size(); ++j) v |= size_t(k.sha256[i + j]) << (j * 8);
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }
    h ^= std::hash<std::string>{}(k.detail) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.max_tokens) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
  }
};

// Thread-safe bounded LRU cache mapping (image SHA-256, detail, max_tokens) -> encoded embeddings.
class VisionCache {
 public:
  explicit VisionCache(size_t max_bytes = size_t(128) << 20) : max_bytes_(max_bytes) {}

  bool get(const VisionCacheKey& key, std::shared_ptr<const CachedVision>* out);
  void put(const VisionCacheKey& key, std::shared_ptr<const CachedVision> val);
  void clear();
  size_t bytes() const;

 private:
  struct Entry {
    VisionCacheKey key;
    std::shared_ptr<const CachedVision> val;
    size_t bytes = 0;
  };
  mutable std::mutex mu_;
  size_t max_bytes_ = 0;
  size_t cur_bytes_ = 0;
  std::list<Entry> lru_;
  std::unordered_map<VisionCacheKey, std::list<Entry>::iterator, VisionCacheKeyHash> index_;
};

}  // namespace flyx::vision
