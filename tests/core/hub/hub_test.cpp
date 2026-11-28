// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// core/hub: SHA-256 against FIPS 180-4's examples on both of its paths; Hub names; which files a name
// picks in a repository; a cache with two snapshots of a repository (both listed, the old commit's file loaded by
// name@commit offline, files deleted with their blobs only where no other snapshot reads them); and fetching from a
// fake Hub on this machine (HF_ENDPOINT): the files in huggingface_hub's cache layout, found again without a request and offline, a fetch cut short going on with the pieces it lacks, a
// corrupted piece refused by the hash, a storage address that stops working.  Skipped (77) where libcurl is missing.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "core/hub/hub.h"
#include "core/hub/hub_internal.h"
#include "core/hub/sha256.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

namespace fs = std::filesystem;
using flyx::hub::Sha256;
using flyx::os::from_path;
using flyx::os::to_path;

namespace {

struct TestDirectory {
  fs::path path;
  TestDirectory() {
    const auto base = fs::canonical(fs::temp_directory_path());
    const auto prefix = "flyx-hub-test-" + std::to_string(flyx::os::process_id()) + "-";
    for (unsigned i = 0;; ++i) {
      path = base / (prefix + std::to_string(i));
      std::error_code ec;
      if (fs::create_directory(path, ec)) return;  // only a directory we created will be removed
      if (ec && ec != std::errc::file_exists) throw fs::filesystem_error("create test directory", path, ec);
    }
  }
  TestDirectory(const TestDirectory&) = delete;
  TestDirectory& operator=(const TestDirectory&) = delete;
  ~TestDirectory() {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec) std::fprintf(stderr, "could not remove test directory %s\n", from_path(path).c_str());
  }
};

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _wputenv_s(to_path(k).c_str(), to_path(v).c_str());
#else
  if (v.empty()) unsetenv(k);
  else setenv(k, v.c_str(), 1);
#endif
}

std::string sha(const std::string& s, bool portable) {
  Sha256 h(portable);
  h.update(s.data(), s.size());
  return h.hex();
}

void hashes() {
  const std::pair<std::string, const char*> known[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
      {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
  for (bool portable : {true, false})
    for (const auto& [in, want] : known)
      check(sha(in, portable) == want, std::string("sha256 of ") + std::to_string(in.size()) + " bytes, " +
                                           (portable ? "plain C++" : Sha256::accelerated() ? "SHA instructions" : "default"));
  std::mt19937 rng(7);
  std::string data(3 << 20, '\0');
  for (char& c : data) c = char(rng());
  Sha256 pieces;
  for (size_t at = 0, n = 1; at < data.size(); at += n, n = n * 3 + 1) pieces.update(data.data() + at, std::min(n, data.size() - at));
  check(pieces.hex() == sha(data, true) && sha(data, false) == sha(data, true),
        "3 MB in uneven pieces, both paths: one digest");
}

void names() {
  flyx::hub::Name n;
  check(flyx::hub::parse_name("unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS", &n) && n.repo == "unsloth/Qwen3.8-27B-GGUF" &&
            n.quant == "UD-IQ3_XXS" && n.revision == "main",
        "org/name:QUANT");
  check(flyx::hub::parse_name("https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2", &n) && n.repo == "z-lab/Qwen3.8-27B-DFlash2" &&
            n.quant.empty(),
        "a huggingface.co address");
  check(flyx::hub::parse_name("hf.co/a-b/c.d_e:Q4_K_M@v1.2", &n) && n.repo == "a-b/c.d_e" && n.quant == "Q4_K_M" &&
            n.revision == "v1.2",
        "hf.co/org/name:QUANT@revision");
  for (const char* path : {"/home/me/model.gguf", "C:\\models\\x.gguf", "D:/models/x.gguf", "model.gguf", "a/b/c",
                           "../x", "org/..", "./x", "", "org/name:", "org/na me"})
    check(!flyx::hub::parse_name(path, &n), std::string("not a Hub name: \"") + path + "\"");
  // An adapter's name: a folder of the repository after ':', a path of names.
  check(flyx::hub::parse_adapter_name("org/name:adapters/tools@v2", &n) && n.repo == "org/name" &&
            n.quant == "adapters/tools" && n.revision == "v2",
        "org/name:FOLDER/FOLDER@revision, an adapter's");
  check(flyx::hub::parse_adapter_name("https://hf.co/org/name", &n) && n.repo == "org/name" && n.quant.empty(),
        "an adapter at a repository's top");
  for (const char* bad : {"org/name:a/../b", "org/name:/a", "org/name:a//b", "org/name:a/", "org/name:", "org/name:.x",
                          "/home/me/lora", "org/name:a b"})
    check(!flyx::hub::parse_adapter_name(bad, &n), std::string("not an adapter's name: \"") + bad + "\"");
  // A repository's adapters: each folder with an adapter_config.json, named by its whole path (two folders of one
  // last name apart), its weights beside it or a pickle.
  const std::vector<flyx::hub::RepoFile> files = {
      {"adapter_config.json", 1}, {"adapter_model.safetensors", 2}, {"r6/fp8/adapter_config.json", 1},
      {"r6/fp8/adapter_model.safetensors", 2}, {"r7/fp8/adapter_config.json", 1}, {"r7/fp8/adapter_model.bin", 2},
      {"other_adapter_config.json", 1}, {"README.md", 1}};
  const auto folders = flyx::hub::adapter_folders("org/name", files);
  check(folders.size() == 3 && folders[0].name == "org/name" && folders[0].weights &&
            folders[1].name == "org/name:r6/fp8" && folders[1].weights && folders[2].folder == "r7/fp8" &&
            !folders[2].weights && folders[2].pickle,
        "a repository's adapter folders");
  {  // a file too large for read_text is refused by its listed size, before the cache or the Hub is asked
    flyx::hub::Repo r;
    r.id = "org/name", r.commit = std::string(40, 'a');
    std::string text, err;
    check(!flyx::hub::read_text(r, {"adapter_config.json", 3u << 30}, 64u << 10, &text, &err) &&
              err.find("more than 65536") != std::string::npos && text.empty(),
          "read_text refuses a 3 GB adapter_config.json: " + err);
  }
  // Endpoint normalization, override, source reporting, and mirror URL parsing.
  std::string norm, ep_err;
  check(flyx::hub::normalize_endpoint("hf-mirror.com/", &norm, &ep_err) && norm == "https://hf-mirror.com",
        "normalize_endpoint adds https:// and strips trailing slash");
  check(flyx::hub::normalize_endpoint("http://127.0.0.1:8080/hf/", &norm, &ep_err) &&
            norm == "http://127.0.0.1:8080/hf",
        "normalize_endpoint keeps explicit http:// and path prefix");
  check(flyx::hub::normalize_endpoint("http://[::1]:8080/", &norm, &ep_err) && norm == "http://[::1]:8080",
        "normalize_endpoint keeps IPv6 literal host");
  check(flyx::hub::normalize_endpoint("   ", &norm, &ep_err) && norm.empty(),
        "normalize_endpoint treats blank as reset");
  for (const char* bad : {"ftp://hf-mirror.com", "javascript:alert(1)", "https://hf-mirror.com?a=1",
                          "https://hf-mirror.com#x", "https://user@hf-mirror.com", "https://", "https://bad host"})
    check(!flyx::hub::normalize_endpoint(bad, &norm, &ep_err),
          std::string("normalize_endpoint rejects invalid URL: ") + bad);
  check(flyx::hub::set_endpoint("hf-mirror.com", &ep_err) && flyx::hub::endpoint() == "https://hf-mirror.com" &&
            flyx::hub::endpoint_source() == "setting",
        "set_endpoint overrides endpoint() and reports source=setting");
  check(flyx::hub::parse_name("https://hf-mirror.com/org/model-GGUF:Q4_K_M", &n) && n.repo == "org/model-GGUF" &&
            n.quant == "Q4_K_M",
        "parse_name strips configured custom mirror URL");
  check(flyx::hub::parse_name("hf-mirror.com/org/model-GGUF:Q8_0", &n) && n.repo == "org/model-GGUF" &&
            n.quant == "Q8_0",
        "parse_name strips configured custom mirror host without scheme");
  check(flyx::hub::set_endpoint("", &ep_err) && flyx::hub::endpoint_source() != "setting",
        "set_endpoint(\"\") clears custom override");
}

void hub_settings() {
  using namespace flyx::hub;
  auto check_endpoint = [](const std::string& url, const std::string& source) {
    const auto setting = resolve_endpoint();
    check(setting.value == url && setting.source == source, "endpoint from " + source + ": " + url);
  };
  set_env("HF_ENDPOINT", "");
  check_endpoint(kDefaultEndpoint, "default");
  set_env("HF_ENDPOINT", "///");
  check_endpoint(kDefaultEndpoint, "default");
  set_env("HF_ENDPOINT", "https://env.test/hub///");
  check_endpoint("https://env.test/hub", "HF_ENDPOINT");
  check(set_endpoint("mirror.test/"), "set runtime mirror over HF_ENDPOINT");
  check_endpoint("https://mirror.test", "setting");
  const auto before_reset = resolve_endpoint();
  check(set_endpoint("  "), "blank runtime mirror resets to HF_ENDPOINT");
  check_endpoint("https://env.test/hub", "HF_ENDPOINT");
  check(before_reset.value == "https://mirror.test" && before_reset.source == "setting",
        "a snapshot retains its value and source after reset");

  std::atomic<bool> started{false}, stop{false};
  std::thread writer([&] {
    do {
      set_endpoint("mirror.test");
      started = true;
      set_endpoint("");
    } while (!stop);
  });
  while (!started) std::this_thread::yield();
  bool consistent = true;
  for (int i = 0; i < 2000; ++i) {
    const auto setting = resolve_endpoint();
    consistent &= (setting.source == "setting" && setting.value == "https://mirror.test") ||
                  (setting.source == "HF_ENDPOINT" && setting.value == "https://env.test/hub");
  }
  stop = true;
  writer.join();
  check(consistent, "concurrent overrides and resets always report a matching endpoint and source");
  set_env("HF_ENDPOINT", "");

  auto check_cache = [](const std::string& dir, const std::string& source) {
    const auto cache = resolve_cache_dir();
    check(cache.value == from_path(to_path(dir).make_preferred()) && cache.source == source,
          "cache directory from " + source + ": " + dir);
  };
  set_env("HF_HUB_CACHE", "cache/direct");
  set_env("HF_HOME", "cache/hf");
  set_env("XDG_CACHE_HOME", "cache/xdg");
  check_cache("cache/direct", "HF_HUB_CACHE");
  set_env("HF_HUB_CACHE", "");
  check_cache("cache/hf/hub", "HF_HOME");
  set_env("HF_HOME", "");
  check_cache("cache/xdg/huggingface/hub", "XDG_CACHE_HOME");
  set_env("XDG_CACHE_HOME", "");
  check_cache(flyx::os::home_dir() + "/.cache/huggingface/hub", "default");

  // HF_TOKEN is only sent over HTTPS to the default Hub or an explicit HF_ENDPOINT override, never to http:// or a runtime mirror.
  set_env("HF_TOKEN", "hf_secret_unit_token");
  const auto has_bearer = [](const std::vector<std::string>& hdrs) {
    for (const auto& h : hdrs)
      if (h == "Authorization: Bearer hf_secret_unit_token") return true;
    return false;
  };
  check(has_bearer(detail::auth_headers({kDefaultEndpoint, "default"})),
        "auth_headers attaches Bearer token for default HTTPS Hub");
  check(has_bearer(detail::auth_headers({"https://env.mirror.test", "HF_ENDPOINT"})),
        "auth_headers attaches Bearer token for HTTPS HF_ENDPOINT");
  check(detail::auth_headers({"http://127.0.0.1:8080", "HF_ENDPOINT"}).empty(),
        "auth_headers omits Bearer token for plaintext http:// HF_ENDPOINT");
  check(detail::auth_headers({"https://evil.mirror.test", "setting"}).empty(),
        "auth_headers omits Bearer token for runtime setting mirror");
  check(has_bearer(detail::auth_headers({kDefaultEndpoint, "setting"})),
        "auth_headers attaches Bearer token if runtime setting equals default HTTPS Hub");
  set_env("HF_TOKEN", "");
}

flyx::hub::RepoFile file(const std::string& path, uint64_t size, const std::string& sha256 = "ab") {
  return {path, size, sha256, "cd"};
}

void choices() {
  flyx::hub::Repo r;
  r.id = "unsloth/Qwen3.8-27B-GGUF";
  r.files = {file("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", 50), file("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", 5),
             file("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 2), file("Qwen3.8-27B-Q4_0.gguf", 16),
             file("Qwen3.8-27B-UD-IQ3_XXS.gguf", 11), file("Qwen3.8-27B-IQ3_XXS.gguf", 10), file("mmproj-F16.gguf", 1),
             file("imatrix_unsloth.gguf", 1), file("README.md", 1, "")};
  std::map<std::string, flyx::hub::GgufChoice> got;
  for (const auto& c : flyx::hub::gguf_choices(r)) got[c.quant] = c;
  check(got.size() == 4 && got.count("Q4_0") && got.count("UD-IQ3_XXS") && got.count("IQ3_XXS") && got["BF16"].split &&
            got["BF16"].complete && got["BF16"].parts.size() == 2 && got["BF16"].total_size == 55,
        "GGUF choices: Q4_0, UD-IQ3_XXS, IQ3_XXS and the complete 2-part split BF16 (total size 55); no projector, matrix or MTP file");
  std::vector<flyx::hub::RepoFile> files;
  std::string err;
  flyx::hub::Name n{r.id, "IQ3_XXS"};
  check(flyx::hub::select(r, n, &files, &err) && files[0].path == "Qwen3.8-27B-IQ3_XXS.gguf", ":IQ3_XXS, the exact one");
  n.quant = "ud-iq3_xxs";
  check(flyx::hub::select(r, n, &files, &err) && files[0].path == "Qwen3.8-27B-UD-IQ3_XXS.gguf", ":ud-iq3_xxs, any case");
  n.quant = "Q8_0";
  check(!flyx::hub::select(r, n, &files, &err) && err.find(":UD-IQ3_XXS") != std::string::npos,
        "a quantization it lacks: the ones it has listed");
  n.quant = "BF16";
  check(flyx::hub::select(r, n, &files, &err) && files.size() == 2 &&
            files[0].path == "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf" &&
            files[1].path == "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf",
        "a complete split model selects all shards in order");
  // Incomplete split model (missing part 2 of 3) is refused with an informative error.
  flyx::hub::Repo inc;
  inc.id = "org/partial-GGUF";
  inc.files = {file("model-Q8_0-00001-of-00003.gguf", 10), file("model-Q8_0-00003-of-00003.gguf", 10)};
  const auto inc_choices = flyx::hub::gguf_choices(inc);
  check(inc_choices.size() == 1 && inc_choices[0].split && !inc_choices[0].complete,
        "incomplete split GGUF is marked incomplete");
  check(!flyx::hub::select(inc, {inc.id, "Q8_0"}, &files, &err) && err.find("incomplete") != std::string::npos,
        "incomplete split model is refused");
  flyx::hub::Repo companion_split;
  companion_split.id = "org/companion";
  companion_split.files = {file("MTP/layer-00002-of-00002.gguf", 4), file("MTP/layer-00001-of-00002.gguf", 7)};
  check(flyx::hub::select(companion_split, {companion_split.id, "layer-00001-of-00002.gguf"}, &files, &err) &&
            files.size() == 2 && files[0].path == companion_split.files[1].path,
        "an explicit split companion selects all shards starting at shard 1");
  companion_split.files.pop_back();
  check(!flyx::hub::select(companion_split, {companion_split.id, "layer-00002-of-00002.gguf"}, &files, &err) &&
            err.find("incomplete") != std::string::npos, "an explicit incomplete companion is refused");
  // Single complete split model in a repo can be selected without :QUANT.
  flyx::hub::Repo lone_split;
  lone_split.id = "org/split-only";
  lone_split.files = {file("m-BF16-00002-of-00002.gguf", 5), file("m-BF16-00001-of-00002.gguf", 50)};
  check(flyx::hub::select(lone_split, {lone_split.id}, &files, &err) && files.size() == 2 &&
            files[0].path == "m-BF16-00001-of-00002.gguf" && files[1].path == "m-BF16-00002-of-00002.gguf",
        "a lone split GGUF is selected in shard order without naming a quant");
  n.quant = "";
  check(!flyx::hub::select(r, n, &files, &err) && err.find("several") != std::string::npos, "no quantization named: which?");
  check(flyx::hub::mtp_file(r) && flyx::hub::mtp_file(r)->path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf", "the MTP file");
  check(flyx::hub::mmproj_file(r) && flyx::hub::mmproj_file(r)->path == "mmproj-F16.gguf", "the mmproj file");
  {
    const auto comps = flyx::hub::companion_files(r);
    check(comps.size() == 2 && comps[0].kind == "mtp" && comps[0].is_default &&
              comps[0].file.path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf" && comps[1].kind == "mmproj" &&
              comps[1].is_default && comps[1].file.path == "mmproj-F16.gguf",
          "companion_files lists MTP and mmproj with default flags");
    check(flyx::hub::select(r, {r.id, "mmproj-F16.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "mmproj-F16.gguf",
          "select resolves repo:mmproj-F16.gguf companion directly");
    check(flyx::hub::select(r, {r.id, "mtp-Qwen3.8-27B-Q4_0.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "MTP/mtp-Qwen3.8-27B-Q4_0.gguf",
          "select resolves repo:mtp-*.gguf companion by basename");
    flyx::hub::Repo df;
    df.id = "org/model-DFlash2-GGUF";
    df.info.gguf_architecture = "dflash";
    df.files = {file("Model-DFlash2-Draft-BF16.gguf", 5)};
    check(flyx::hub::select(df, {df.id, "Model-DFlash2-Draft-BF16.gguf"}, &files, &err) && files.size() == 1 &&
              files[0].path == "Model-DFlash2-Draft-BF16.gguf",
          "select resolves repo:file.gguf directly for dflash GGUF drafter repos");
  }
  // mmproj_gguf_name edge cases (subfolders, imatrix, MTP, negative qualifiers) and precision ranking.
  check(flyx::hub::mmproj_gguf_name("mmproj-BF16.gguf") && flyx::hub::mmproj_gguf_name("mmproj/vision.gguf") &&
            flyx::hub::mmproj_gguf_name("qwen-mmproj-f16.gguf") &&
            !flyx::hub::mmproj_gguf_name("imatrix_mmproj.gguf") && !flyx::hub::mmproj_gguf_name("MTP/mmproj.gguf") &&
            !flyx::hub::mmproj_gguf_name("Qwen3-no-mmproj-Q4_K_M.gguf") &&
            !flyx::hub::mmproj_gguf_name("Qwen3-without-mmproj-Q8_0.gguf"),
        "mmproj_gguf_name handles subfolders and excludes imatrix/MTP/no-mmproj");
  check(flyx::hub::model_gguf_name("Qwen3-no-mmproj-Q4_K_M.gguf") && !flyx::hub::model_gguf_name("mmproj/vision.gguf"),
        "model_gguf_name keeps no-mmproj models and excludes mmproj/ subfolders");
  {
    flyx::hub::Repo vr;
    vr.id = "org/vlm-GGUF";
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4), file("mmproj/mmproj-F16.gguf", 2),
                file("mmproj-BF16.gguf", 2)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj-BF16.gguf",
          "mmproj_file prefers BF16 over F16/F32/Q8_0");
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4), file("mmproj/mmproj-F16.gguf", 2)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj/mmproj-F16.gguf",
          "mmproj_file prefers F16 over F32/Q8_0 (including mmproj/ subfolder)");
    vr.files = {file("mmproj-Q8_0.gguf", 1), file("mmproj-F32.gguf", 4)};
    check(flyx::hub::mmproj_file(vr) && flyx::hub::mmproj_file(vr)->path == "mmproj-F32.gguf",
          "mmproj_file prefers F32 over Q8_0");
  }
  flyx::hub::Repo one;
  one.id = "x/y";
  one.files = {file("llama-2-7b.Q4_K_M.gguf", 4)};
  check(flyx::hub::gguf_choices(one)[0].quant == "Q4_K_M" && flyx::hub::select(one, {one.id}, &files, &err),
        "a lone GGUF, its quantization after a dot; named without one");
  flyx::hub::Repo d;
  d.id = "z-lab/Qwen3.8-27B-DFlash2";
  d.architectures = {"DFlash2DraftModel"};
  d.files = {file("config.json", 1, ""), file("model.safetensors", 3), file("assets/figure.png", 1), file("README.md", 1, "")};
  check(flyx::hub::select(d, {d.id}, &files, &err) && files.size() == 2 && files[0].path == "config.json" &&
            files[1].path == "model.safetensors",
        "a drafter's repository: its config and safetensors");
  d.id = "Qwen/Qwen3-0.6B", d.architectures = {"Qwen3ForCausalLM"};
  const bool picked = flyx::hub::select(d, {d.id}, &files, &err);
  check(!picked && err.find("transformers checkpoint") != std::string::npos,
        "a transformers checkpoint is no drafter: says what flyx reads (" + err + ")");
}

std::string read_all(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- a cache with two snapshots of one repository --------------------------------------------------------------
// As the cache holds them: the blob by its id, an entry for it in each snapshot that has it (a link, else a hard link,
// else a copy, as hub::fetch places them where links are not allowed).
void put(const fs::path& repo, const std::string& commit, const std::string& path, const std::string& content) {
  const fs::path blob = repo / "blobs" / sha(content, true), entry = repo / "snapshots" / commit / path;
  fs::create_directories(blob.parent_path());
  fs::create_directories(entry.parent_path());
  if (!fs::exists(blob)) std::ofstream(blob, std::ios::binary) << content;
  std::error_code ec;
  fs::create_symlink(fs::relative(blob, entry.parent_path()), entry, ec);
  if (ec) ec.clear(), fs::create_hard_link(blob, entry, ec);
  if (ec) fs::copy_file(blob, entry);
}

void cache_snapshots(const fs::path& tmp) {
  check(flyx::hub::quant_in_name("Qwen3.8-27B-UD-IQ2_S.gguf") == "UD-IQ2_S" && flyx::hub::quant_in_name("model.gguf").empty() &&
            flyx::hub::quant_in_name("x-Q4_K_M-00001-of-00002.gguf") == "Q4_K_M",
        "a quantization from a file's name, none where it names none");
  check(flyx::hub::model_gguf_name("a/x-Q4_0.gguf") && !flyx::hub::model_gguf_name("MTP/x.gguf") &&
            !flyx::hub::model_gguf_name("a/mtp/x.gguf") && !flyx::hub::model_gguf_name("mtp-x.gguf") &&
            !flyx::hub::model_gguf_name("mmproj-F16.gguf") && flyx::hub::split_gguf_name("x-00001-of-00002.gguf"),
        "model GGUFs by name: no MTP layers, projectors; split parts told");
  // DSpark files are companions of the models beside them. The default is independent of the listing order.
  check(flyx::hub::dspark_gguf_name("dspark-Inkling-2-Mini-Q8_0.gguf") && flyx::hub::dspark_gguf_name("DSpark/x.gguf") &&
            flyx::hub::dspark_gguf_name("a/dspark/x-BF16.gguf") && !flyx::hub::dspark_gguf_name("dspark-imatrix.gguf") &&
            !flyx::hub::dspark_gguf_name("Inkling-2-Mini-Q4_K_M.gguf") && !flyx::hub::dspark_gguf_name("dspark-x.bin") &&
            !flyx::hub::model_gguf_name("dspark-Inkling-2-Mini-Q8_0.gguf") && !flyx::hub::model_gguf_name("dspark/x.gguf") &&
            !flyx::hub::model_gguf_name("old_support/x-Q4_0.gguf"),
        "DSpark GGUFs by name: no models; nor what a repository keeps under old_support/");
  {
    flyx::hub::Repo ir;
    ir.id = "acme/Inkling-GGUF";
    ir.files = {file("Inkling-Q4_K_M.gguf", 9), file("dspark-Inkling-Q8_0_Old.gguf", 402278560),
                file("dspark/dspark-Inkling-BF16.gguf", 647792800), file("dspark-Inkling-Q8_0.gguf", 503889952)};
    for (int order = 0; order < 2; ++order) {
      const auto comps = flyx::hub::companion_files(ir);
      check(comps.size() == 3 && comps[0].file.path == "dspark-Inkling-Q8_0.gguf" && comps[0].is_default &&
                !comps[1].is_default && !comps[2].is_default && flyx::hub::gguf_choices(ir).size() == 1,
            "the canonical DSpark quantization precedes its smaller suffixed variant in either listing order");
      std::reverse(ir.files.begin(), ir.files.end());
    }
    std::vector<flyx::hub::RepoFile> picked;
    std::string why;
    check(flyx::hub::select(ir, {ir.id, "dspark-Inkling-Q8_0.gguf"}, &picked, &why) && picked.size() == 1 &&
              picked[0].path == "dspark-Inkling-Q8_0.gguf",
          "select resolves repo:dspark-*.gguf companion directly (--drafter-file ORG/NAME:dspark-*.gguf)");
  }
  const std::string old_c = "1111111111111111111111111111111111111111", new_c = "2222222222222222222222222222222222222222";
  const fs::path repo = tmp / "models--acme--two-GGUF";
  const std::string q4(5000, 'q'), iq2(3000, 'i'), q8(9000, '8');
  put(repo, old_c, "two-Q4_K_M.gguf", q4);
  put(repo, old_c, "two-IQ2_S.gguf", iq2);  // only in the old snapshot: the repository moved on
  put(repo, new_c, "two-Q4_K_M.gguf", q4);
  put(repo, new_c, "two-Q8_0.gguf", q8);
  fs::create_directories(repo / "refs");
  std::ofstream(repo / "refs" / "main", std::ios::binary) << new_c;
  set_env("HF_HUB_CACHE", from_path(tmp));
  const auto snaps = flyx::hub::cached();
  check(snaps.size() == 2 && snaps[0].commit == new_c && snaps[0].current && snaps[1].commit == old_c &&
            !snaps[1].current && snaps[1].files.size() == 2 && snaps[0].files.size() == 2,
        "every snapshot: refs/main's first, then the old one with the file only it has");
  flyx::hub::Repo r;
  std::string err;
  set_env("HF_HUB_OFFLINE", "1");
  flyx::hub::Name n;
  std::vector<flyx::hub::RepoFile> files;
  std::vector<std::string> paths;
  const bool old_ok = flyx::hub::parse_name("acme/two-GGUF:IQ2_S@" + old_c, &n) &&
                      flyx::hub::describe(n.repo, n.revision, &r, &err) && flyx::hub::select(r, n, &files, &err) &&
                      flyx::hub::fetch(r, files, &paths, nullptr, nullptr, &err);
  check(old_ok && r.commit == old_c && read_all(to_path(paths[0])) == iq2,
        "org/name:QUANT@commit: the old snapshot's file, offline" + (old_ok ? "" : ": " + err));
  check(flyx::hub::describe("acme/two-GGUF", "main", &r, &err) && r.commit == new_c && flyx::hub::gguf_choices(r).size() == 2,
        "  and main is the new snapshot");
  set_env("HF_HUB_OFFLINE", "");
  uint64_t freed = 0;
  bool ok = flyx::hub::remove_cached("acme/two-GGUF", old_c, {"two-Q4_K_M.gguf"}, &freed, &err);
  check(ok && !fs::exists(fs::symlink_status(repo / "snapshots" / old_c / "two-Q4_K_M.gguf")) &&
            read_all(repo / "snapshots" / new_c / "two-Q4_K_M.gguf") == q4,
        "deleting a file the newer snapshot has too: the newer one still reads it" + (ok ? "" : ": " + err));
  ok = flyx::hub::remove_cached("acme/two-GGUF", old_c, {"two-IQ2_S.gguf"}, &freed, &err);
  check(ok && freed >= iq2.size() && !fs::exists(repo / "blobs" / sha(iq2, true)),
        "deleting a file only one snapshot has: its blob goes, its bytes freed (" + std::to_string(freed) + ")");  // a copy and its blob: twice
  check(!flyx::hub::remove_cached("acme/two-GGUF", new_c, {"../../../x.gguf"}, &freed, &err) &&
            !flyx::hub::remove_cached("acme/two-GGUF", new_c, {"none.gguf"}, &freed, &err),
        "nothing outside the snapshot, nothing it lacks");
  {
    // A snapshot may contain a standalone copy. Two model files can share their size and both outer megabytes.
    const fs::path copy = repo / "snapshots" / new_c / "copied.gguf";
    const std::string data((2u << 20) + 1, 'a');
    std::string other = data;
    other[1u << 20] = 'b';
    const fs::path other_blob = repo / "blobs" / sha(other, true);
    const fs::path partial = repo / "blobs" / (sha(data, true) + ".incomplete");
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(other_blob, std::ios::binary) << other;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && !fs::exists(copy) && freed == data.size() && fs::exists(other_blob),
          "deleting a standalone copy keeps a different blob with identical ends");
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(partial, std::ios::binary) << data;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && freed == data.size() && fs::exists(partial), "a partial download with matching bytes is kept");
    const fs::path own_blob = repo / "blobs" / sha(data, true);
    std::ofstream(copy, std::ios::binary) << data;
    std::ofstream(own_blob, std::ios::binary) << data;
    ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"copied.gguf"}, &freed, &err);
    check(ok && !fs::exists(own_blob) && fs::exists(other_blob) && freed == 2 * data.size(),
          "a standalone copy and its matching unreferenced blob are both reclaimed");
  }
  {
    // Dangling symlinks and symlinks to partial/directory entries inside blobs/ must not abort removal or delete
    // partial files when removing cached snapshot entries.
    std::error_code sec;
    const fs::path snap = repo / "snapshots" / new_c;
    const fs::path missing_blob = repo / "blobs" / std::string(64, 'e');
    const fs::path part_blob = repo / "blobs" / (std::string(64, 'f') + ".flyx-part");
    std::ofstream(part_blob, std::ios::binary) << "partial";
    const fs::path dangling = snap / "dangling.gguf";
    const fs::path sibling = snap / "sibling.gguf";
    const fs::path part_link = snap / "part_link.gguf";
    std::ofstream(sibling, std::ios::binary) << "sibling";
    fs::create_symlink(fs::path("../../blobs") / missing_blob.filename(), dangling, sec);
    if (!sec) {
      fs::create_symlink(fs::path("../../blobs") / part_blob.filename(), part_link, sec);
      err.clear();
      ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"dangling.gguf", "sibling.gguf"}, &freed, &err);
      check(ok && !fs::exists(fs::symlink_status(dangling)) && !fs::exists(fs::symlink_status(sibling)),
            "dangling symlink does not abort remove_cached or leave remaining shards: " + err);
      err.clear();
      ok = flyx::hub::remove_cached("acme/two-GGUF", new_c, {"part_link.gguf"}, &freed, &err);
      check(ok && !fs::exists(fs::symlink_status(part_link)) && fs::exists(part_blob),
            "symlink to .flyx-part only removes symlink and keeps partial blob: " + err);
    }
    fs::remove(part_blob, sec);
    fs::remove(sibling, sec);
  }
  {
    // Path traversal / invalid revision validation across parse_name(), describe(), and fetch().
    flyx::hub::Name bad_n;
    check(!flyx::hub::parse_name("acme/two-GGUF@.", &bad_n) && !flyx::hub::parse_name("acme/two-GGUF@.hidden", &bad_n),
          "parse_name rejects '.' and leading-dot revisions");
    flyx::hub::Repo bad_r;
    err.clear();
    check(!flyx::hub::describe("acme/two-GGUF", "../../other", &bad_r, &err) &&
              err.find("revision") != std::string::npos,
          "describe rejects path-traversal revision: " + err);
    const fs::path escape_target = repo.parent_path() / "escaped_ref";
    std::ofstream(escape_target, std::ios::binary) << "keep";
    flyx::hub::Repo evil;
    evil.id = "acme/two-GGUF";
    evil.commit = new_c;
    evil.revision = "../../escaped_ref";
    std::vector<std::string> out_paths;
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err) && read_all(escape_target) == "keep",
          "fetch rejects path-traversal revision without overwriting files outside cache");
    evil.revision = "main";
    check(!flyx::hub::fetch(evil, {file("../escape.gguf", 4, std::string(64, '1'))}, &out_paths, nullptr, nullptr, &err),
          "fetch rejects unsafe RepoFile path");
    evil.id = "../bad/repo";
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err), "fetch rejects invalid repo id");
    evil.id = "acme/two-GGUF";
    evil.commit = "not-a-commit";
    check(!flyx::hub::fetch(evil, {}, &out_paths, nullptr, nullptr, &err), "fetch rejects invalid commit");
    std::error_code sec;
    fs::remove(escape_target, sec);
  }
}

// ---- a drafter's kind from a cached config ----------------------------------------------------------------------
// Its config's architecture: Tencent's DSpineDraftModel is DSpine, z-lab's DFlashDraftModel DFlash.
void cached_drafter_kinds(const fs::path& tmp) {
  const std::string commit = "3333333333333333333333333333333333333333";
  const std::pair<const char*, const char*> cases[] = {{"DSpine", "DSpineDraftModel"}, {"DFlash", "DFlashDraftModel"}};
  for (const auto& [name, arch] : cases) {
    const std::string dir = std::string("models--acme--") + name;
    const std::string config = std::string(R"({"architectures": [")") + arch + R"("], "dflash_config": {}})";
    put(tmp / dir, commit, "config.json", config);
    fs::create_directories(tmp / dir / "refs");
    std::ofstream(tmp / dir / "refs" / "main", std::ios::binary) << commit;
  }
  set_env("HF_HUB_CACHE", from_path(tmp));
  flyx::hub::Repo spine, flash;
  check(flyx::hub::in_cache("acme/DSpine", "main", &spine) && flyx::hub::drafter_kind(spine) == "dspine",
        "a DSpineDraftModel: DSpine");
  check(flyx::hub::in_cache("acme/DFlash", "main", &flash) && flyx::hub::drafter_kind(flash) == "dflash",
        "a DFlashDraftModel: DFlash");
  const fs::path cfg = tmp / "models--acme--DFlash" / "snapshots" / commit / "config.json";
  for (bool oversized : {false, true}) {
    {
      std::ofstream file(cfg, std::ios::binary | std::ios::trunc);
      file << R"({"architectures":["DFlashDraftModel"],"x":)" << std::string(70, '[') << "0" << std::string(70, ']') << "}";
      if (oversized) {
        file.seekp(0);
        file << R"({"architectures":["DFlashDraftModel"]})" << std::string(16u << 20, ' ');
      }
    }
    flyx::hub::Repo bounded;
    check(flyx::hub::in_cache("acme/DFlash", "main", &bounded) && bounded.architectures.empty(),
          "cache inventory skips config beyond its size or depth limit");
  }
}

void cached_allow_path_guard() {
  auto norm = [](const fs::path& p) {
    std::string s = from_path(p);
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
  };
  std::vector<std::string> touched_blocked;
  auto record_if_blocked = [&](const fs::path& p, const char* op) {
    const std::string s = norm(p);
    if (s.find("blocked") != std::string::npos || s.find("evil") != std::string::npos)
      touched_blocked.push_back(std::string(op) + ":" + s);
  };
  const std::string root = "/hf_cache";
  const std::string commit_ok = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  const std::string commit_blocked = "blocked000000000000000000000000000000000";
  const std::string repo_ok = root + "/models--acme--good";
  const std::string snaps_ok = repo_ok + "/snapshots";
  const std::string snap_ok = snaps_ok + "/" + commit_ok;

  flyx::hub::CacheFs fake_fs;
  fake_fs.root_override = root;
  fake_fs.is_directory = [&](const fs::path& p, std::error_code&) {
    record_if_blocked(p, "is_directory");
    const std::string s = norm(p);
    return s == root || s == repo_ok || s == snaps_ok || s == snap_ok || s == snap_ok + "/sub";
  };
  fake_fs.is_regular_file = [&](const fs::path& p, std::error_code&) {
    record_if_blocked(p, "is_regular_file");
    const std::string s = norm(p);
    return s == snap_ok + "/good-Q4_K_M.gguf" || s == snap_ok + "/sub/extra.gguf";
  };
  fake_fs.file_size = [&](const fs::path& p, std::error_code&) -> uint64_t {
    record_if_blocked(p, "file_size");
    return 4096;
  };
  fake_fs.last_write_time = [&](const fs::path& p, std::error_code&) {
    record_if_blocked(p, "last_write_time");
    return fs::file_time_type::clock::now();
  };
  fake_fs.for_each_dir_entry = [&](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                                   std::error_code&) {
    record_if_blocked(p, "for_each_dir_entry");
    const std::string s = norm(p);
    std::vector<fs::path> entries;
    if (s == root) {
      entries = {p / "models--acme--blocked", p / "models--acme--good"};
    } else if (s == snaps_ok) {
      entries = {p / commit_blocked, p / commit_ok};
    } else if (s == snap_ok) {
      entries = {p / "evil_link.gguf", p / "evil_dir", p / "good-Q4_K_M.gguf", p / "sub"};
    } else if (s == snap_ok + "/sub") {
      entries = {p / "extra.gguf"};
    }
    for (const auto& e : entries)
      if (!visit(e)) break;
  };
  fake_fs.read_text = [&](const fs::path& p) -> std::string {
    record_if_blocked(p, "read_text");
    const std::string s = norm(p);
    if (s == repo_ok + "/refs/main") return commit_ok;
    if (s == snap_ok + "/config.json") return R"({"architectures":["DFlash2DraftModel"]})";
    return {};
  };

  auto allow = [](std::string_view p) {
    return p.find("blocked") == std::string_view::npos && p.find("evil") == std::string_view::npos;
  };
  const auto snaps = flyx::hub::cached(allow, &fake_fs);
  check(touched_blocked.empty(),
        "hub::cached never invokes target-following fs ops on blocked repos/commits/entries (touched=" +
            std::to_string(touched_blocked.size()) + ")");
  check(snaps.size() == 1 && snaps[0].repo == "acme/good" && snaps[0].commit == commit_ok && snaps[0].current &&
            snaps[0].files.size() == 2 && snaps[0].architectures == std::vector<std::string>{"DFlash2DraftModel"},
        "hub::cached still enumerates allowed snapshot files and config.json when blocked entries are skipped");
  // Also verify blocked refs/main and config.json are never read.
  touched_blocked.clear();
  std::vector<std::string> reads;
  fake_fs.read_text = [&](const fs::path& p) -> std::string {
    reads.push_back(norm(p));
    return {};
  };
  auto deny_meta = [&](std::string_view p) {
    std::string s(p);
    std::replace(s.begin(), s.end(), '\\', '/');
    return allow(p) && s.find("refs/main") == std::string::npos && s.find("config.json") == std::string::npos;
  };
  const auto snaps2 = flyx::hub::cached(deny_meta, &fake_fs);
  check(reads.empty() && snaps2.size() == 1 && !snaps2[0].current && snaps2[0].architectures.empty(),
        "hub::cached skips reading refs/main and config.json when blocked by allow_path");

  // Item 3 (deterministic CacheFs): directory symlinks are never descended into, while file symlinks to blobs still work.
  std::vector<std::string> descended_links;
  flyx::hub::CacheFs sym_fs = fake_fs;
  sym_fs.is_symlink = [&](const fs::path& p, std::error_code&) {
    const std::string s = norm(p);
    return s == snap_ok + "/dir_link" || s == snap_ok + "/good-Q4_K_M.gguf";
  };
  sym_fs.is_directory = [&](const fs::path& p, std::error_code&) {
    const std::string s = norm(p);
    return s == root || s == repo_ok || s == snaps_ok || s == snap_ok || s == snap_ok + "/dir_link";
  };
  sym_fs.is_regular_file = [&](const fs::path& p, std::error_code&) {
    const std::string s = norm(p);
    return s == snap_ok + "/good-Q4_K_M.gguf";
  };
  sym_fs.for_each_dir_entry = [&](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                                  std::error_code&) {
    const std::string s = norm(p);
    if (s == snap_ok + "/dir_link") {
      descended_links.push_back(s);
      visit(p / "leaked.gguf");
      return;
    }
    std::vector<fs::path> entries;
    if (s == root) entries = {p / "models--acme--good"};
    else if (s == snaps_ok) entries = {p / commit_ok};
    else if (s == snap_ok) entries = {p / "dir_link", p / "good-Q4_K_M.gguf"};
    for (const auto& e : entries)
      if (!visit(e)) break;
  };
  const auto sym_snaps = flyx::hub::cached({}, &sym_fs);
  check(descended_links.empty() && sym_snaps.size() == 1 && sym_snaps[0].files.size() == 1 &&
            sym_snaps[0].files[0].path == "good-Q4_K_M.gguf",
        "hub::cached never descends into directory symlinks while preserving file symlinks to blobs");

  // Linked repo, snapshots, and commit roots are rejected in both hub::cached and detail::from_cache.
  for (const auto& [label, linked_root] : std::vector<std::pair<std::string, std::string>>{
           {"repo", repo_ok}, {"snapshots", snaps_ok}, {"commit", snap_ok}}) {
    std::vector<std::string> enumerated_under;
    flyx::hub::CacheFs root_link_fs = sym_fs;
    root_link_fs.is_symlink = [&](const fs::path& p, std::error_code&) {
      const std::string s = norm(p);
      return s == linked_root || s == snap_ok + "/good-Q4_K_M.gguf";
    };
    root_link_fs.for_each_dir_entry = [&](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                                          std::error_code&) {
      const std::string s = norm(p);
      if (s == linked_root || s == snap_ok) enumerated_under.push_back(s);
      std::vector<fs::path> entries;
      if (s == root) entries = {p / "models--acme--good"};
      else if (s == snaps_ok) entries = {p / commit_ok};
      else if (s == snap_ok) entries = {p / "good-Q4_K_M.gguf"};
      for (const auto& e : entries)
        if (!visit(e)) break;
    };
    const auto listed = flyx::hub::cached({}, &root_link_fs);
    flyx::hub::Repo resolved;
    const bool got_main = flyx::hub::detail::from_cache("acme/good", "main", &resolved, {}, &root_link_fs);
    const bool got_commit = flyx::hub::detail::from_cache("acme/good", commit_ok, &resolved, {}, &root_link_fs);
    check(listed.empty() && !got_main && !got_commit && enumerated_under.empty(),
          "linked " + label + " root directory is rejected before enumerating snapshot entries");
  }
}

void secondary_cache_policy_and_traversal(const fs::path& tmp) {
  auto norm = [](std::string_view p) {
    std::string s(p);
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
  };
  const std::string commit_a = "1111111111111111111111111111111111111111";
  const std::string commit_b = "2222222222222222222222222222222222222222";
  // Item 2a: hub::describe offline fallback and hub::in_cache enforce allow_path.
  {
    const fs::path cache = tmp / "offline_policy";
    const fs::path repo = cache / "models--acme--policy-GGUF";
    put(repo, commit_a, "ok-Q4_K_M.gguf", "aaaa");
    put(repo, commit_a, "evil-Q8_0.gguf", "bbbb");
    fs::create_directories(repo / "refs");
    std::ofstream(repo / "refs" / "main", std::ios::binary) << commit_a;
    set_env("HF_HUB_CACHE", from_path(cache));
    set_env("HF_HUB_OFFLINE", "1");
    auto deny_evil = [&](std::string_view p) { return norm(p).find("evil") == std::string::npos; };
    auto deny_all = [](std::string_view) { return false; };
    flyx::hub::Repo r;
    std::string err;
    check(flyx::hub::describe("acme/policy-GGUF", "main", &r, &err, nullptr, deny_evil) && r.from_cache &&
              r.files.size() == 1 && r.files[0].path == "ok-Q4_K_M.gguf",
          "describe offline fallback filters denied snapshot entries via allow_path: " + err);
    r = {};
    check(!flyx::hub::describe("acme/policy-GGUF", "main", &r, &err, nullptr, deny_all),
          "describe offline fallback fails when allow_path denies cached repo");
    check(!flyx::hub::in_cache("acme/policy-GGUF", "main", &r, deny_all),
          "in_cache fails when allow_path denies cached repo");
    set_env("HF_HUB_OFFLINE", "");
  }
  // Item 2b: hub::fetch rejects derived cache paths before creating dirs or placing files.
  {
    const fs::path denied_cache = tmp / "denied_fetch_cache";
    set_env("HF_HUB_CACHE", from_path(denied_cache));
    flyx::hub::Repo r;
    r.id = "acme/fetch-denied";
    r.commit = commit_a;
    r.revision = "main";
    const auto f = file("m-Q4_K_M.gguf", 16, std::string(64, 'a'));
    std::vector<std::string> paths;
    std::string err;
    auto deny_file = [&](std::string_view p) { return norm(p).find("m-Q4_K_M.gguf") == std::string::npos; };
    check(!flyx::hub::fetch(r, {f}, &paths, nullptr, nullptr, &err, deny_file) &&
              err.find("remote path policy") != std::string::npos && paths.empty() && !fs::exists(denied_cache),
          "hub::fetch rejects denied destination path before creating cache dirs: " + err);
    err.clear();
    auto deny_blob = [&](std::string_view p) { return norm(p).find("/blobs/") == std::string::npos; };
    check(!flyx::hub::fetch(r, {f}, &paths, nullptr, nullptr, &err, deny_blob) &&
              err.find("remote path policy") != std::string::npos && paths.empty() && !fs::exists(denied_cache),
          "hub::fetch rejects denied blob path before creating cache dirs: " + err);
    // Item 1 (PR #74 follow-up): deny ONLY the PartialBlob progress .map path and verify no filesystem I/O occurs.
    err.clear();
    bool saw_part_data = false, saw_part_map = false;
    auto deny_only_map = [&](std::string_view p) {
      const std::string s = norm(p);
      if (s.ends_with(".flyx-part")) saw_part_data = true;
      if (s.ends_with(".flyx-part.map")) {
        saw_part_map = true;
        return false;
      }
      return true;
    };
    check(!flyx::hub::fetch(r, {f}, &paths, nullptr, nullptr, &err, deny_only_map) && saw_part_data && saw_part_map &&
              err.find("remote path policy") != std::string::npos && paths.empty() && !fs::exists(denied_cache),
          "hub::fetch rejects denied PartialBlob .map path before any filesystem I/O: " + err);
  }
  // Item 2c: hub::remove_cached retains a shared/unverified blob when allow_path denies another snapshot entry,
  // and refuses to delete a denied target entry.
  {
    const fs::path cache = tmp / "remove_policy";
    const fs::path repo = cache / "models--acme--shared-GGUF";
    const std::string payload = "shared-model-bytes";
    const std::string blob_id = sha(payload, true);
    const fs::path blob = repo / "blobs" / blob_id;
    const fs::path snap_a = repo / "snapshots" / commit_a;
    const fs::path snap_b = repo / "snapshots" / commit_b;
    fs::create_directories(blob.parent_path());
    fs::create_directories(snap_a);
    fs::create_directories(snap_b);
    std::ofstream(blob, std::ios::binary) << payload;
    fs::create_hard_link(blob, snap_a / "model.gguf");
    fs::create_hard_link(blob, snap_b / "model.gguf");
    set_env("HF_HUB_CACHE", from_path(cache));
    uint64_t freed = 0;
    std::string err;
    auto deny_b = [&](std::string_view p) { return norm(p).find(commit_b) == std::string::npos; };
    check(!flyx::hub::remove_cached("acme/shared-GGUF", commit_b, {"model.gguf"}, &freed, &err, deny_b) &&
              err.find("remote path policy") != std::string::npos && fs::exists(snap_b / "model.gguf"),
          "remove_cached refuses to remove entry denied by allow_path: " + err);
    err.clear();
    check(flyx::hub::remove_cached("acme/shared-GGUF", commit_a, {"model.gguf"}, &freed, &err, deny_b) &&
              !fs::exists(snap_a / "model.gguf") && fs::exists(blob) && fs::exists(snap_b / "model.gguf"),
          "remove_cached safely retains blob when allow_path denies another snapshot during reference scan: " + err);
  }
  // Item 2 (PR #74 follow-up): remove_cached independently rejects linked repo/snapshots/commit/blobs roots (CacheFs).
  {
    const fs::path cache = tmp / "rm_linked_root";
    const fs::path repo = cache / "models--acme--rmroot";
    const fs::path snaps = repo / "snapshots";
    const fs::path snap = snaps / commit_a;
    const fs::path blobs = repo / "blobs";
    const std::string payload = "do-not-delete-external";
    const fs::path blob = blobs / sha(payload, true);
    fs::create_directories(snap);
    fs::create_directories(blobs);
    std::ofstream(blob, std::ios::binary) << payload;
    fs::create_hard_link(blob, snap / "victim.gguf");
    for (const auto& [label, target] : std::vector<std::pair<std::string, fs::path>>{
             {"repo", repo}, {"snapshots", snaps}, {"commit", snap}, {"blobs", blobs}}) {
      flyx::hub::CacheFs linked_fs;
      linked_fs.root_override = from_path(cache);
      const std::string linked_norm = norm(from_path(target));
      linked_fs.is_symlink = [&](const fs::path& p, std::error_code&) { return norm(from_path(p)) == linked_norm; };
      uint64_t freed = 0;
      std::string err;
      check(!flyx::hub::remove_cached("acme/rmroot", commit_a, {"victim.gguf"}, &freed, &err, {}, &linked_fs) &&
                fs::exists(snap / "victim.gguf") && fs::exists(blob),
            "remove_cached rejects linked " + label + " root without deleting files: " + err);
    }
  }
  // Item 3 (PR #74 follow-up): remove_cached fails closed (retains blob) on symlink canonicalization or equivalent errors.
  {
    const fs::path cache = tmp / "remove_id_err";
    const std::string payload = "shared-identity-error-bytes";
    const std::string blob_id = sha(payload, true);
    // 3a: symlink canonicalization error in another snapshot.
    {
      const fs::path repo = cache / "models--acme--iderr-sym";
      const fs::path blob = repo / "blobs" / blob_id;
      const fs::path snap_a = repo / "snapshots" / commit_a;
      const fs::path snap_b = repo / "snapshots" / commit_b;
      fs::create_directories(blob.parent_path());
      fs::create_directories(snap_a);
      fs::create_directories(snap_b);
      std::ofstream(blob, std::ios::binary) << payload;
      fs::create_hard_link(blob, snap_a / "model.gguf");
      std::error_code sym_ec;
      fs::create_symlink(fs::path("../../blobs") / blob_id, snap_b / "model.gguf", sym_ec);
      if (sym_ec) fs::create_hard_link(blob, snap_b / "model.gguf");
      bool canon_err_hit = false;
      const std::string other_entry = norm(from_path(snap_b / "model.gguf"));
      flyx::hub::CacheFs canon_err_fs;
      canon_err_fs.root_override = from_path(cache);
      canon_err_fs.is_symlink = [&](const fs::path& p, std::error_code& ec) {
        if (norm(from_path(p)) == other_entry) return true;
        const auto st = fs::symlink_status(p, ec);
        if (st.type() == fs::file_type::not_found) {
          ec.clear();
          return false;
        }
        return !ec && fs::is_symlink(st);
      };
      canon_err_fs.weakly_canonical = [&](const fs::path& p, std::error_code& ec) {
        if (norm(from_path(p)) == other_entry) {
          canon_err_hit = true;
          ec = std::make_error_code(std::errc::permission_denied);
          return fs::path{};
        }
        return fs::weakly_canonical(p, ec);
      };
      uint64_t freed = 0;
      std::string err;
      check(flyx::hub::remove_cached("acme/iderr-sym", commit_a, {"model.gguf"}, &freed, &err, {}, &canon_err_fs) &&
                canon_err_hit && !fs::exists(snap_a / "model.gguf") && fs::exists(blob) &&
                read_all(snap_b / "model.gguf") == payload,
            "remove_cached retains blob when canonicalizing another snapshot symlink errors: " + err);
    }
    // 3b: regular-file identity (equivalent) error in another snapshot.
    {
      const fs::path repo = cache / "models--acme--iderr-eq";
      const fs::path blob = repo / "blobs" / blob_id;
      const fs::path snap_a = repo / "snapshots" / commit_a;
      const fs::path snap_b = repo / "snapshots" / commit_b;
      fs::create_directories(blob.parent_path());
      fs::create_directories(snap_a);
      fs::create_directories(snap_b);
      std::ofstream(blob, std::ios::binary) << payload;
      fs::create_hard_link(blob, snap_a / "model.gguf");
      fs::create_hard_link(blob, snap_b / "model.gguf");
      bool eq_err_hit = false;
      const std::string other_entry = norm(from_path(snap_b / "model.gguf"));
      flyx::hub::CacheFs eq_err_fs;
      eq_err_fs.root_override = from_path(cache);
      eq_err_fs.equivalent = [&](const fs::path& a, const fs::path& b, std::error_code& ec) {
        if (norm(from_path(a)) == other_entry || norm(from_path(b)) == other_entry) {
          eq_err_hit = true;
          ec = std::make_error_code(std::errc::io_error);
          return false;
        }
        return fs::equivalent(a, b, ec);
      };
      uint64_t freed = 0;
      std::string err;
      check(flyx::hub::remove_cached("acme/iderr-eq", commit_a, {"model.gguf"}, &freed, &err, {}, &eq_err_fs) &&
                eq_err_hit && !fs::exists(snap_a / "model.gguf") && fs::exists(blob) &&
                read_all(snap_b / "model.gguf") == payload,
            "remove_cached retains blob when fs::equivalent errors on another snapshot entry: " + err);
    }
  }
  // Item 3 (real filesystem, when OS permits directory symlinks): external directory symlinks, ancestor cycles,
  // and linked repo/snapshots/commit roots are rejected in cached/in_cache/remove_cached.
  {
    const fs::path cache = tmp / "real_symlink";
    const fs::path outside = tmp / "outside_dir";
    const fs::path repo = cache / "models--acme--sym-GGUF";
    const fs::path snap = repo / "snapshots" / commit_a;
    fs::create_directories(outside);
    std::ofstream(outside / "secret.gguf", std::ios::binary) << "secret";
    put(repo, commit_a, "real-Q4_K_M.gguf", "ok");
    fs::create_directories(repo / "refs");
    std::ofstream(repo / "refs" / "main", std::ios::binary) << commit_a;
    std::error_code sec1, sec2;
    fs::create_directory_symlink(outside, snap / "ext_link", sec1);
    fs::create_directory_symlink(snap, snap / "cycle_link", sec2);
    if (!sec1 && !sec2) {
      const fs::path ext_repo = tmp / "ext_repo_target";
      put(ext_repo, commit_a, "leaked-repo.gguf", "x");
      fs::create_directories(ext_repo / "refs");
      std::ofstream(ext_repo / "refs" / "main", std::ios::binary) << commit_a;
      std::error_code lsec;
      fs::create_directory_symlink(ext_repo, cache / "models--acme--linked-repo", lsec);
      const fs::path linked_snaps_repo = cache / "models--acme--linked-snaps";
      fs::create_directories(linked_snaps_repo / "refs");
      std::ofstream(linked_snaps_repo / "refs" / "main", std::ios::binary) << commit_a;
      fs::create_directory_symlink(ext_repo / "snapshots", linked_snaps_repo / "snapshots", lsec);
      const fs::path linked_commit_repo = cache / "models--acme--linked-commit";
      fs::create_directories(linked_commit_repo / "snapshots");
      fs::create_directories(linked_commit_repo / "refs");
      std::ofstream(linked_commit_repo / "refs" / "main", std::ios::binary) << commit_a;
      fs::create_directory_symlink(outside, linked_commit_repo / "snapshots" / commit_a, lsec);

      set_env("HF_HUB_CACHE", from_path(cache));
      const auto listed = flyx::hub::cached();
      check(listed.size() == 1 && listed[0].repo == "acme/sym-GGUF" && listed[0].files.size() == 1 &&
                listed[0].files[0].path == "real-Q4_K_M.gguf",
            "hub::cached skips real directory symlinks, ancestor cycles, and linked repo/snapshots/commit roots");
      flyx::hub::Repo r;
      check(flyx::hub::in_cache("acme/sym-GGUF", "main", &r) && r.files.size() == 1 &&
                r.files[0].path == "real-Q4_K_M.gguf",
            "hub::in_cache skips real directory symlinks and ancestor cycles on disk");
      check(!flyx::hub::in_cache("acme/linked-repo", "main", &r) &&
                !flyx::hub::in_cache("acme/linked-snaps", "main", &r) &&
                !flyx::hub::in_cache("acme/linked-commit", "main", &r) &&
                !flyx::hub::in_cache("acme/linked-commit", commit_a, &r),
            "hub::in_cache rejects real linked repo, snapshots, and commit root directories");
      uint64_t freed = 0;
      std::string err;
      check(!flyx::hub::remove_cached("acme/linked-commit", commit_a, {"secret.gguf"}, &freed, &err) &&
                fs::exists(outside / "secret.gguf"),
            "hub::remove_cached refuses to delete external file through a real linked commit root: " + err);
    }
  }
  // Item 4: authoritative offline hub::in_cache and hub::describe return complete inventory > 16 dirs deep.
  {
    const fs::path cache = tmp / "d";
    const fs::path repo = cache / "models--a--b";
    std::string deep_rel;
    for (int i = 0; i < 18; ++i) deep_rel += "a/";
    deep_rel += "n-Q8_0.gguf";
    put(repo, commit_a, "root-Q4_K_M.gguf", "root");
    put(repo, commit_a, deep_rel, "deep");
    fs::create_directories(repo / "refs");
    std::ofstream(repo / "refs" / "main", std::ios::binary) << commit_a;
    set_env("HF_HUB_CACHE", from_path(cache));
    set_env("HF_HUB_OFFLINE", "1");
    flyx::hub::Repo r;
    std::string err;
    check(flyx::hub::describe("a/b", "main", &r, &err) && r.from_cache && r.files.size() == 2 &&
              r.files[0].path == deep_rel && r.files[1].path == "root-Q4_K_M.gguf",
          "offline hub::describe returns complete inventory nested > 16 directories deep: " + err);
    flyx::hub::Repo r2;
    check(flyx::hub::in_cache("a/b", "main", &r2) && r2.files.size() == 2,
          "hub::in_cache returns complete inventory nested > 16 directories deep");
    const auto bounded = flyx::hub::cached();
    check(bounded.size() == 1 && bounded[0].files.size() == 1 && bounded[0].files[0].path == "root-Q4_K_M.gguf",
          "interactive hub::cached keeps depth bound while offline describe/in_cache stays authoritative");
    set_env("HF_HUB_OFFLINE", "");
  }
}

// ---- a fake Hub ------------------------------------------------------------------------------------------------
struct FakeHub {
  const std::string commit = "0123456789abcdef0123456789abcdef01234567";
  std::map<std::string, std::string> files;  // path -> content
  httplib::Server http;
  int port = 0;
  std::thread serving;
  std::atomic<int> storage{0};        // requests for file bytes
  std::atomic<bool> fail_late{false};  // pieces past the first: 404
  std::atomic<bool> corrupt{false};    // one byte of the second piece flipped
  std::atomic<int> sig{0};             // the signature the last redirect carried
  std::atomic<bool> expire_first{false};  // signature 1 stops working after the first request
  std::atomic<bool> saw_auth{false};      // whether any request carried an Authorization header

  FakeHub() {
    std::mt19937 rng(11);
    std::string big(70u << 20, '\0');  // three pieces of 32 MB: 32, 32, 6
    for (char& c : big) c = char(rng());
    files["tiny-Q4_K_M.gguf"] = big;
    files["MTP/mtp-tiny-Q4_0.gguf"] = std::string(1000, 'm');
    files["config.json"] = "{}";
    http.Get("/api/models/acme/tiny-GGUF/revision/main", [this](const httplib::Request& q, httplib::Response& res) {
      if (q.has_header("Authorization")) saw_auth = true;
      nlohmann::json sib = nlohmann::json::array();
      for (const auto& [path, content] : files) {
        nlohmann::json f = {{"rfilename", path}, {"size", content.size()}, {"blobId", "00112233445566778899aabbccddeeff00112233"}};
        if (path != "config.json") f["lfs"] = {{"sha256", sha(content, true)}, {"size", content.size()}};
        sib.push_back(f);
      }
      res.set_content(nlohmann::json({{"sha", commit}, {"siblings", sib}, {"cardData", {{"base_model", "Qwen/Qwen3.8-27B"}}}}).dump(),
                      "application/json");
    });
    http.Get("/api/models/acme/nullable-GGUF/revision/main", [this](const httplib::Request&, httplib::Response& res) {
      res.set_content(nlohmann::json({{"sha", commit},
                                      {"config", nullptr},
                                      {"cardData", nullptr},
                                      {"siblings", nlohmann::json::array({
                                                       {{"rfilename", "m-Q4_0.gguf"},
                                                        {"size", nullptr},
                                                        {"blobId", nullptr},
                                                        {"lfs", {{"sha256", nullptr}, {"size", 1234}}}},
                                                       "non-object",
                                                   })}})
                          .dump(),
                      "application/json");
    });
    http.Get(R"(/api/models/acme/none/revision/main)", [](const httplib::Request&, httplib::Response& res) { res.status = 404; });
    http.Get(R"(/api/models/acme/gated/revision/main)", [](const httplib::Request&, httplib::Response& res) { res.status = 403; });
    http.Get(R"(/api/models/acme/throttled/revision/main)", [](const httplib::Request&, httplib::Response& res) {
      res.status = 429;
      res.set_header("Retry-After", "42");
    });
    http.Get("/api/models", [this](const httplib::Request& q, httplib::Response& res) {
      if (q.has_header("Authorization")) saw_auth = true;
      const std::string cursor = q.get_param_value("cursor");
      if (cursor.empty()) {
        res.set_header("Link", "<http://127.0.0.1:" + std::to_string(port) +
                                   "/api/models?filter=gguf&cursor=page%2F2%2Btok%3D%3D>; rel=\"next\"");
        res.set_content(nlohmann::json::array({
                            {{"id", "acme/Inkling-2-Mini-GGUF"},
                             {"downloads", 9000},
                             {"likes", 120},
                             {"lastModified", "2026-11-20T00:00:00.000Z"},
                             {"gguf", {{"architecture", "inkling2_loop"}}}},
                            {{"id", "acme/Qwen3-27B-GGUF"},
                             {"downloads", 8000},
                             {"likes", 90},
                             {"gguf", {{"architecture", "qwen35"}}}},
                        })
                            .dump(),
                        "application/json");
      } else if (cursor == "page/2+tok==") {
        res.set_content(nlohmann::json::array({
                            {{"id", "acme/Third-Model-GGUF"},
                             {"downloads", 500},
                             {"likes", 10},
                             {"gguf", {{"architecture", "qwen3"}}}},
                        })
                            .dump(),
                        "application/json");
      } else {
        res.status = 400;
      }
    });
    http.Get(R"(/acme/tiny-GGUF/resolve/([0-9a-f]+)/(.*))", [this](const httplib::Request& q, httplib::Response& res) {
      res.status = 302;  // to the storage, an address relative to this host as the Hub gives
      res.set_header("Location", "/storage/" + q.matches[2].str() + "?sig=" + std::to_string(++sig));
    });
    http.Get(R"(/storage/(.*))", [this](const httplib::Request& q, httplib::Response& res) {
      ++storage;
      const auto it = files.find(q.matches[1].str());
      if (it == files.end()) return void(res.status = 404);
      unsigned long long a = 0, b = it->second.size() - 1;
      if (std::sscanf(q.get_header_value("Range").c_str(), "bytes=%llu-%llu", &a, &b) != 2) a = 0, b = it->second.size() - 1;
      if (expire_first && q.get_param_value("sig") == "1" && !(a == 0 && b == 0)) return void(res.status = 403);
      if (fail_late && a >= (32u << 20)) {  // after the first piece is under way
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return void(res.status = 404);
      }
      // The whole file: httplib answers the request's range of it (206).
      if (!corrupt) return res.set_content(it->second, "application/octet-stream");
      std::string body = it->second;
      body[(32u << 20) + 100] ^= 1;  // in the second piece
      res.set_content(body, "application/octet-stream");
    });
    port = http.bind_to_any_port("127.0.0.1");
    serving = std::thread([this] { http.listen_after_bind(); });
    http.wait_until_ready();
  }
  ~FakeHub() {
    http.stop();
    serving.join();
  }
};

bool fetch_q4(const std::string& cache, std::string* path, std::string* err) {
  set_env("HF_HUB_CACHE", cache);
  flyx::hub::Repo r;
  std::vector<flyx::hub::RepoFile> files;
  std::vector<std::string> paths;
  if (!flyx::hub::describe("acme/tiny-GGUF", "main", &r, err) || !flyx::hub::select(r, {r.id, "Q4_K_M"}, &files, err) ||
      !flyx::hub::fetch(r, files, &paths, nullptr, nullptr, err))
    return false;
  *path = paths[0];
  return true;
}

void fetching(const fs::path& tmp) {
  FakeHub hub;
  set_env("HF_ENDPOINT", "http://127.0.0.1:" + std::to_string(hub.port));
  set_env("HF_TOKEN", "");
  set_env("HF_HUB_OFFLINE", "");
  set_env("HF_HOME", from_path(tmp / "home"));
  std::string path, err;
  const std::string cache = from_path(tmp / "a");
  bool ok = fetch_q4(cache, &path, &err);
  check(ok && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"],
        "a 70 MB file in three pieces, as the Hub holds it" + (ok ? "" : ": " + err));
  const fs::path repo = tmp / "a" / "models--acme--tiny-GGUF";
  const fs::path blob = repo / "blobs" / sha(hub.files["tiny-Q4_K_M.gguf"], true);
  std::error_code ec;
  const auto snapshot = to_path(path);
  const bool linked = fs::equivalent(snapshot, blob, ec);
  const bool copied = !linked && !ec && !fs::is_symlink(snapshot) && read_all(blob) == hub.files["tiny-Q4_K_M.gguf"];
  check(read_all(repo / "refs" / "main") == hub.commit &&
            snapshot == repo / "snapshots" / hub.commit / "tiny-Q4_K_M.gguf" && (linked || copied),
        "huggingface_hub's layout: refs/main, the blob by its sha256, the same file or a copy in the snapshot");
  check(hub.storage == 4, "one request for where it is, then three for the pieces: " + std::to_string(hub.storage.load()));
  int before = hub.storage;
  check(fetch_q4(cache, &path, &err) && hub.storage == before, "again: from the cache, no file requested");
  {
    flyx::hub::Repo next;
    next.id = "acme/tiny-GGUF";
    next.commit = std::string(40, 'a');
    const auto shared = file("tiny-Q4_K_M.gguf", hub.files["tiny-Q4_K_M.gguf"].size(), from_path(blob.filename()));
    std::vector<std::string> linked_paths;
    flyx::hub::Progress last;
    check(flyx::hub::fetch(next, {shared}, &linked_paths, [&](const auto& p) { last = p; }, nullptr, &err) &&
              last.total == shared.size && last.done == last.total && last.file == shared.path && hub.storage == before,
          "another revision reuses a cached blob and reports all of its bytes");
    int reports = 0;
    auto hit = [&](const flyx::hub::Progress& p) { ++reports; last = p; };
    check(flyx::hub::fetch(next, {shared}, &linked_paths, hit, nullptr, &err) && reports == 1 &&
              last.file == shared.path && last.total == shared.size && last.done == last.total &&
              !last.checking && !last.fetched && last.bytes_per_s == 0 && hub.storage == before,
          "an existing snapshot reports completion once without a transfer, nothing fetched");
    fs::remove(to_path(flyx::hub::snapshot_dir(next)) / shared.path);
    fs::remove(to_path(flyx::hub::snapshot_dir(next)));
  }
  set_env("HF_HUB_OFFLINE", "1");
  flyx::hub::Repo r;
  std::vector<std::string> paths;
  check(flyx::hub::describe("acme/tiny-GGUF", "main", &r, &err) && r.from_cache && r.commit == hub.commit &&
            flyx::hub::gguf_choices(r).size() == 1 && flyx::hub::gguf_choices(r)[0].quant == "Q4_K_M",
        "offline: the repository from the cache, its file by its own name");
  check(!flyx::hub::describe("acme/other", "main", &r, &err) && err.find("HF_HUB_OFFLINE") != std::string::npos,
        "offline, a repository not in the cache: says so");
  set_env("HF_HUB_OFFLINE", "");
  ok = flyx::hub::describe("acme/none", "main", &r, &err);
  check(!ok && err.find("not on the Hub") != std::string::npos, "a repository the Hub does not have: " + err);

  const std::string cut = from_path(tmp / "b");
  hub.fail_late = true;
  ok = fetch_q4(cut, &path, &err);
  check(!ok, "the pieces after the first fail: the fetch fails (" + err + ")");
  hub.fail_late = false;
  before = hub.storage;
  check(fetch_q4(cut, &path, &err) && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"] &&
            hub.storage - before == 3,
        "again: on from where it stopped, the two pieces it lacked (" + std::to_string(hub.storage - before - 1) + ")");

  {
    const fs::path resume = tmp / "cancel";
    set_env("HF_HUB_CACHE", from_path(resume));
    flyx::hub::Repo request;
    request.id = "acme/tiny-GGUF";
    request.commit = hub.commit;
    const auto model = file("tiny-Q4_K_M.gguf", hub.files["tiny-Q4_K_M.gguf"].size(), from_path(blob.filename()));
    std::atomic<bool> stop{true};
    before = hub.storage;
    check(!flyx::hub::fetch(request, {model}, &paths, nullptr, &stop, &err) && err == "stopped" &&
              hub.storage == before && !fs::exists(resume), "a cancelled fetch makes no requests or cache files");
    stop = false;
    bool said_fetched = true;  // every report of a fetch that fetches says so
    auto cancel_check = [&](const flyx::hub::Progress& p) {
      said_fetched = said_fetched && p.fetched;
      if (p.checking) stop = true;
    };
    check(!flyx::hub::fetch(request, {model}, &paths, cancel_check, &stop, &err) && err == "stopped",
          "a cancellation during verification preserves the completed pieces");
    check(said_fetched, "  its progress says a file is being fetched");
    stop = false;
    before = hub.storage;
    check(flyx::hub::fetch(request, {model}, &paths, nullptr, &stop, &err) && hub.storage == before &&
              read_all(to_path(paths[0])) == hub.files[model.path],
          "verified locally on retry, without requesting the storage URL");
    const std::string mtp = "MTP/mtp-tiny-Q4_0.gguf";
    const auto layer = file(mtp, hub.files[mtp].size(), sha(hub.files[mtp], true));
    const auto partial = resume / "models--acme--tiny-GGUF" / "blobs" / (layer.sha256 + ".flyx-part");
    std::ofstream(partial, std::ios::binary) << std::string(layer.size, 'x');
    std::ofstream(fs::path(partial).concat(".map"), std::ios::binary).put(1).put(1);  // two flags for a one-piece file
    before = hub.storage;
    check(flyx::hub::fetch(request, {layer}, &paths, nullptr, nullptr, &err) && hub.storage == before + 1 &&
              read_all(to_path(paths[0])) == hub.files[mtp],
          "a wrong-sized bitmap counts no saved pieces and is fetched again");
  }

  hub.corrupt = true;
  check(!fetch_q4(from_path(tmp / "c"), &path, &err) && err.find("sha256") != std::string::npos,
        "a piece changed on the way: refused by the hash");
  hub.corrupt = false;
  check(!fs::exists(fs::path(tmp / "c" / "models--acme--tiny-GGUF" / "blobs" /
                             (sha(hub.files["tiny-Q4_K_M.gguf"], true) + ".flyx-part"))),
        "  and what came is deleted");
  hub.sig = 0;
  hub.expire_first = true;
  ok = fetch_q4(from_path(tmp / "d"), &path, &err);
  check(ok && read_all(to_path(path)) == hub.files["tiny-Q4_K_M.gguf"],
        "the storage address stops working: the pieces go back to the Hub's address" + (ok ? "" : ": " + err));
  hub.expire_first = false;
  set_env("HF_HUB_CACHE", from_path(tmp / "a"));
  fs::create_directories(tmp / "a" / "models----bad" / "snapshots" / hub.commit);
  const auto cached = flyx::hub::cached();
  check(cached.size() == 1 && cached[0].repo == "acme/tiny-GGUF" && cached[0].files.size() == 1 &&
            cached[0].files[0].path == "tiny-Q4_K_M.gguf" && cached[0].current,
        "the cache's models: the repository and its file (malformed dirs skipped)");
  flyx::hub::Repo nullable;
  check(flyx::hub::describe("acme/nullable-GGUF", "main", &nullable, &err) && nullable.commit == hub.commit &&
            nullable.files.size() == 1 && nullable.files[0].path == "m-Q4_0.gguf" && nullable.files[0].size == 1234,
        "describe accepts null config/cardData/sibling fields: " + err);
  {
    std::vector<flyx::hub::Found> page1, page2;
    std::string next1, next2;
    const bool ok1 = flyx::hub::search("", "trending", 20, &page1, &err, "", &next1);
    check(ok1 && page1.size() == 2 && page1[0].id == "acme/Inkling-2-Mini-GGUF" &&
              page1[0].architecture == "inkling2_loop" && page1[1].architecture == "qwen35" && next1 == "page/2+tok==",
          "search parses gguf.architecture and Link rel=next cursor: " + err);
    const bool ok2 = flyx::hub::search("", "trending", 20, &page2, &err, next1, &next2);
    check(ok2 && page2.size() == 1 && page2[0].id == "acme/Third-Model-GGUF" && page2[0].architecture == "qwen3" &&
              next2.empty(),
          "search round-trips percent-encoded cursor to page 2: " + err);
  }
  {
    // Sum of missing file sizes overflowing uint64_t must saturate and be rejected by has_room() before any download.
    set_env("HF_HUB_CACHE", from_path(tmp / "overflow_room"));
    flyx::hub::Repo huge;
    huge.id = "acme/tiny-GGUF";
    huge.commit = hub.commit;
    const uint64_t half_plus = (UINT64_MAX / 2) + 1024;
    const auto f1 = file("huge1.gguf", half_plus, std::string(64, 'a'));
    const auto f2 = file("huge2.gguf", half_plus, std::string(64, 'b'));
    before = hub.storage;
    err.clear();
    check(!flyx::hub::fetch(huge, {f1, f2}, &paths, nullptr, nullptr, &err) &&
              err.find("free") != std::string::npos && hub.storage == before,
          "has_room saturates uint64_t sum instead of wrapping around: " + err);
  }
  {
    // Live fetch/search with HF_TOKEN set must never leak Authorization to plaintext http:// or runtime setting mirror.
    hub.saw_auth = false;
    set_env("HF_TOKEN", "hf_must_not_leak");
    std::vector<flyx::hub::Found> found;
    std::string next;
    check(flyx::hub::search("", "trending", 10, &found, &err, "", &next) && !hub.saw_auth,
          "search does not leak HF_TOKEN to plaintext http:// HF_ENDPOINT");
    std::string ep_err;
    check(flyx::hub::set_endpoint("http://127.0.0.1:" + std::to_string(hub.port), &ep_err),
          "set_endpoint to local FakeHub");
    flyx::hub::Repo described;
    check(flyx::hub::describe("acme/tiny-GGUF", "main", &described, &err) && !hub.saw_auth,
          "describe does not leak HF_TOKEN to runtime setting mirror");
    flyx::hub::set_endpoint("", &ep_err);
    set_env("HF_TOKEN", "");
  }
  {
    // Typed HubOutcome error classification (400/403/404/429 + Retry-After) without brittle string parsing.
    flyx::hub::Repo out_repo;
    flyx::hub::HubOutcome out;
    check(!flyx::hub::describe("acme/none", "main", &out_repo, &err, &out) && out.error == flyx::hub::HubError::kNotFound,
          "describe classifies 404 as kNotFound");
    out = {};
    check(!flyx::hub::describe("acme/gated", "main", &out_repo, &err, &out) &&
              out.error == flyx::hub::HubError::kUnauthorized,
          "describe classifies 403 as kUnauthorized");
    out = {};
    check(!flyx::hub::describe("acme/throttled", "main", &out_repo, &err, &out) &&
              out.error == flyx::hub::HubError::kRateLimited && out.retry_after == "42",
          "describe classifies 429 as kRateLimited and preserves Retry-After: " + out.retry_after);
    out = {};
    check(!flyx::hub::describe("acme/tiny-GGUF", "../../bad", &out_repo, &err, &out) &&
              out.error == flyx::hub::HubError::kInvalidName,
          "describe classifies invalid revision as kInvalidName");
    // RFC 9110 Retry-After sanitizer: delay-seconds (0, >86400) and HTTP-date preserved; injection/garbage dropped.
    check(flyx::hub::sanitize_retry_after("0") == "0", "sanitize_retry_after preserves 0");
    check(flyx::hub::sanitize_retry_after("  172800 ") == "172800",
          "sanitize_retry_after preserves delay-seconds > 86400 without truncation");
    check(flyx::hub::sanitize_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") == "Wed, 21 Oct 2015 07:28:00 GMT",
          "sanitize_retry_after preserves IMF-fixdate");
    check(flyx::hub::sanitize_retry_after("Sunday, 06-Nov-94 08:49:37 GMT") == "Sunday, 06-Nov-94 08:49:37 GMT",
          "sanitize_retry_after preserves obsolete rfc850-date");
    check(flyx::hub::sanitize_retry_after("Sun Nov  6 08:49:37 1994") == "Sun Nov  6 08:49:37 1994",
          "sanitize_retry_after preserves asctime-date");
    check(flyx::hub::sanitize_retry_after("-5").empty() &&
              flyx::hub::sanitize_retry_after("42\r\nX-Evil: 1").empty() &&
              flyx::hub::sanitize_retry_after("not-a-date").empty() &&
              flyx::hub::sanitize_retry_after("999999999999999999999").empty(),
          "sanitize_retry_after drops negative, CRLF-injected, malformed, and oversized values");
  }
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc == 7 && std::string(argv[1]) == "--fetch") {  // child of check_concurrent_fetch.py
    flyx::hub::Repo r;
    r.id = "acme/parallel";
    r.commit = argv[2];
    flyx::hub::RepoFile f;
    f.path = argv[3];
    f.size = std::stoull(argv[5]);
    const std::string id = argv[4];
    if (id.size() == 64) f.sha256 = id;
    else f.blob_id = id;
    std::atomic<bool> stop{false};
    std::thread timer;
    const int cancel_ms = std::atoi(argv[6]);
    if (cancel_ms > 0) timer = std::thread([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(cancel_ms));
      stop = true;
    });
    std::vector<std::string> paths;
    std::string err;
    auto progress = [&](const flyx::hub::Progress& p) {
      if (cancel_ms == -1 && p.checking) stop = true;
    };
    const bool ok = flyx::hub::fetch(r, {f}, &paths, progress, &stop, &err);
    if (timer.joinable()) timer.join();
    if (!ok) std::fprintf(stderr, "%s\n", err.c_str());
    return ok ? 0 : 1;
  }
  hashes();
  names();
  hub_settings();
  choices();
  const TestDirectory tmp;
  cache_snapshots(tmp.path / "snapshots");
  cached_drafter_kinds(tmp.path / "kinds");
  cached_allow_path_guard();
  secondary_cache_policy_and_traversal(tmp.path / "p");
#if !defined(_WIN32)
  void* curl = dlopen("libcurl.so.4", RTLD_NOW);
  if (!curl) curl = dlopen("libcurl.4.dylib", RTLD_NOW);
  if (!curl) {
    std::printf("skip: no libcurl here, fetching not tested\n");
    return flyx::test::failures ? flyx::test::finish("hub_test") : 77;
  }
#endif
  fetching(tmp.path / "fetch");
  return flyx::test::finish("hub_test");
}
