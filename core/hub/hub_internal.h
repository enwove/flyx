// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Internal helpers shared across core/hub/{hub_names,hub_cache,hub_fetch}.cpp.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/hub/hub.h"

namespace flyx::hub::detail {

std::string lower(std::string s);
bool valid_repo(const std::string& r);
bool valid_word(const std::string& w);
// A repository's file path that stays inside a snapshot: '/' separated, no "..", nothing absolute.
bool safe_path(const std::string& p);
std::string url_path(const std::string& p);   // percent-encoded, '/' kept
std::string url_query(const std::string& s);  // percent-encoded query parameter value
std::string url_decode(const std::string& s);
bool hex_id(const std::string& s);
bool is_commit(const std::string& s);
std::string base_name(const std::string& path);
std::string format_gb(uint64_t n);

std::string hf_home();
std::vector<std::string> auth_headers(const ResolvedSetting& ep);
std::filesystem::path repo_dir(const std::string& repo);
std::string read_text(const std::filesystem::path& p);
bool has_size(const std::filesystem::path& path, uint64_t size);
bool from_cache(const std::string& repo, const std::string& revision, Repo* out, const PathAllowFn& allow_path = {},
                const CacheFs* fs_ops = nullptr);
void write_ref(const Repo& r);
bool place(const std::filesystem::path& blob, const std::filesystem::path& dst, std::string* err);

}  // namespace flyx::hub::detail
