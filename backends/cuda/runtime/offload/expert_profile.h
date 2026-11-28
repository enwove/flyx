// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The mapped experts' decode route counts kept between runs on this machine (expert_profile): the file a model's
// experts name, its bytes, and where it is.  Nothing here touches the device.
#pragma once

#include "flyx/backend_abi.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace flyx::cuda {

// One layer's scores by logical expert, and the gate|up and down weights that name the layer.
struct ProfileLayer {
  std::pair<int, int> weights;
  std::vector<uint64_t> scores;
};

uint64_t add_routes(uint64_t a, uint64_t b);
constexpr uint64_t kFingerprintSeed = 14695981039346656037ull;
// FNV-1a over 8-byte words, the tail's bytes zero-padded.
uint64_t fingerprint_bytes(uint64_t h, const void* data, size_t bytes);
// A tensor's part of the fingerprint: its format, shape and size, and its bytes: all of them (whole), else the first
// and last 64 KiB.
uint64_t fingerprint_tensor(uint64_t h, const flyx_tensor_view* v, bool whole);

std::filesystem::path expert_profile_folder(const std::string& option);
std::filesystem::path expert_profile_file(const std::filesystem::path& folder, uint64_t fingerprint);

// A header (magic, version, layer count, fingerprint), then each layer's weights, expert count and scores.
std::string encode_expert_profile(uint64_t fingerprint, const std::vector<ProfileLayer>& layers);
// The scores into layers, whose weights and expert counts say what the model has; false, and layers as they were,
// unless the bytes are a whole profile of this fingerprint and these layers.
bool decode_expert_profile(const std::string& bytes, uint64_t fingerprint, std::vector<ProfileLayer>* layers);

bool load_expert_profile(const std::filesystem::path& file, uint64_t fingerprint, std::vector<ProfileLayer>* layers);
// Merge new counts with the current file atomically. On success return the new totals in increments;
// on contention or I/O failure leave the increments with the caller for a later retry.
bool merge_expert_profile(const std::filesystem::path& file, uint64_t fingerprint,
                          std::vector<ProfileLayer>* increments) noexcept;

}  // namespace flyx::cuda
