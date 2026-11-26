// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// apps/common/model_source: the DSpark drafter published beside a model, taken by default (wants_dspark): the one a
// GGUF on disk has beside it (dspark_beside: its name, then its header), and the default dspark-*.gguf of the model's
// repository, here a Hugging Face cache read offline.  Neither replaces a drafter that was named.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "apps/common/model_source.h"
#include "core/os/os.h"
#include "core/os/path.h"

namespace fs = std::filesystem;
using flyx::os::from_path;
using flyx::os::to_path;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  failures += !ok;
}

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _wputenv_s(to_path(k).c_str(), to_path(v).c_str());
#else
  if (v.empty()) unsetenv(k);
  else setenv(k, v.c_str(), 1);
#endif
}

// A GGUF of metadata alone (no tensors): general.architecture, and for a drafter whether it is DSpark's.
void write_gguf(const fs::path& path, const std::string& arch, int markov = -1) {
  std::string b = "GGUF";
  auto u32 = [&](uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); };
  auto u64 = [&](uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); };
  auto str = [&](const std::string& s) { u64(s.size()), b += s; };
  u32(3);
  u64(0);                  // tensors
  u64(markov < 0 ? 1 : 2);  // entries
  str("general.architecture"), u32(8), str(arch);
  if (markov >= 0) str("dflash.has_markov_context"), u32(7), b += char(markov);
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << b;
}

flyx::EngineOptions options(bool speculative, const std::string& drafter) {
  flyx::EngineOptions eo;
  eo.speculative = speculative;
  eo.drafter = drafter;
  return eo;
}

bool fetched(const std::string& model, const std::vector<std::string>& drafters, bool want_dspark,
             flyx::apps::Fetched* f, std::string* err) {
  return flyx::apps::fetch_model(model, drafters, false, {}, nullptr, f, err, "", false, want_dspark);
}

}  // namespace

int main() {
  // When a DSpark drafter is taken without being named: the default (auto) with no drafter file, dspark, all.
  const std::vector<std::string> none, one = {"dflash2-drafter"};
  using flyx::apps::wants_dspark;
  check(wants_dspark(options(true, ""), none) && wants_dspark(options(true, "dspark"), none) &&
            wants_dspark(options(true, "all"), none) && wants_dspark(options(true, "dspark"), one) &&
            wants_dspark(options(true, "all"), one),
        "a DSpark drafter is taken by default, and with --drafter dspark or all");
  check(!wants_dspark(options(true, ""), one) && !wants_dspark(options(true, "mtp"), none) &&
            !wants_dspark(options(true, "dflash2"), none) && !wants_dspark(options(true, "none"), none) &&
            !wants_dspark(options(false, ""), none) && !wants_dspark(options(true, "bogus"), none),
        "and not beside a named drafter file in auto, nor with mtp, dflash2, none, --no-spec");

  const fs::path tmp = fs::temp_directory_path() / ("flyx-model-source-test-" + std::to_string(flyx::os::process_id()));
  std::error_code ec;
  fs::remove_all(tmp, ec);

  // On disk: the model's name without its quantization, then the header, choose.
  const fs::path dir = tmp / "models";
  const fs::path model = dir / "Ink-Tiny-UD-Q4_K_M.gguf", other = dir / "Other-Q4_K_M.gguf";
  const fs::path drafter = dir / "dspark-Ink-Tiny-Q8_0.gguf";
  write_gguf(model, "inkling2_loop");
  write_gguf(other, "inkling2_loop");
  write_gguf(drafter, "dflash", 1);
  write_gguf(dir / "dspark-Ink-Tiny-A.gguf", "dflash", 0);   // a DFlash2 drafter by its header
  write_gguf(dir / "dspark-Ink-TinyX-Q8_0.gguf", "dflash", 1);  // another model's name
  write_gguf(dir / "dspark-Ink-Tiny-B.txt", "dflash", 1);       // not a GGUF by its name
  const std::string beside = flyx::apps::dspark_beside(from_path(model));
  check(beside == from_path(drafter), "the DSpark GGUF beside a model on disk: " + beside);
  check(flyx::apps::dspark_beside(from_path(other)).empty() && flyx::apps::dspark_beside(from_path(drafter)).empty(),
        "none beside a model no DSpark drafter is named after (nor beside a drafter)");

  flyx::apps::Fetched f;
  std::string err;
  bool ok = fetched(from_path(model), {}, true, &f, &err);
  check(ok && f.drafters == std::vector<std::string>{from_path(drafter)} && f.drafter_sources.size() == 1 &&
            f.drafter_sources[0].kind == "dspark" && f.drafter_sources[0].from == from_path(drafter) &&
            f.drafter_sources[0].size == fs::file_size(drafter),
        "fetch_model attaches it to the model on disk" + (ok ? "" : ": " + err));
  ok = fetched(from_path(model), {}, false, &f, &err);
  check(ok && f.drafters.empty() && f.drafter_sources.empty(), "not when it is not wanted");
  const fs::path named = tmp / "elsewhere" / "dspark-mine.gguf";
  write_gguf(named, "dflash", 1);
  ok = fetched(from_path(model), {from_path(named)}, true, &f, &err);
  check(ok && f.drafters == std::vector<std::string>{from_path(named)} && f.drafter_sources.size() == 1 &&
            f.drafter_sources[0].kind == "dspark",
        "a DSpark drafter named is the one taken, none other beside it");

  // A repository in the Hugging Face cache, read offline: the default dspark-*.gguf beside its GGUFs.
  set_env("HF_HUB_CACHE", from_path(tmp / "cache"));
  set_env("HF_HUB_OFFLINE", "1");
  auto cached_repo = [&](const std::string& name, const std::vector<std::pair<std::string, int>>& dsparks) {
    const std::string commit(40, 'c');
    const fs::path repo = tmp / "cache" / ("models--acme--" + name);
    fs::create_directories(repo / "refs");
    std::ofstream(repo / "refs" / "main", std::ios::binary) << commit;
    const fs::path snap = repo / "snapshots" / commit;
    write_gguf(snap / "Ink-Tiny-Q4_K_M.gguf", "inkling2_loop");
    for (const auto& [file, markov] : dsparks) write_gguf(snap / file, "dflash", markov);
    return "acme/" + name;
  };
  const std::string repo = cached_repo("Ink-Tiny-GGUF", {{"dspark-Ink-Tiny-Q8_0.gguf", 1}, {"dspark-Ink-Tiny-Z.gguf", 1}});
  ok = fetched(repo + ":Q4_K_M", {}, true, &f, &err);
  check(ok && fs::path(to_path(f.model)).filename() == "Ink-Tiny-Q4_K_M.gguf" && f.drafters.size() == 1 &&
            fs::path(to_path(f.drafters[0])).filename() == "dspark-Ink-Tiny-Q8_0.gguf" &&
            f.drafter_sources.size() == 1 && f.drafter_sources[0].kind == "dspark" &&
            f.drafter_sources[0].from == repo + "/dspark-Ink-Tiny-Q8_0.gguf",
        "fetch_model attaches the repository's default DSpark file" + (ok ? "" : ": " + err));
  ok = fetched(repo + ":Q4_K_M", {}, false, &f, &err);
  check(ok && f.drafters.empty(), "not when it is not wanted (" + std::string(ok ? "" : err) + ")");
  const std::string plain = cached_repo("Ink-Plain-GGUF", {{"dspark-Ink-Tiny-Q8_0.gguf", 0}});
  ok = fetched(plain + ":Q4_K_M", {}, true, &f, &err);
  bool said = false;
  for (const std::string& n : f.notes) said = said || n.find("is not a DSpark drafter") != std::string::npos;
  check(ok && f.drafters.empty() && said, "a dspark-*.gguf whose header is not a DSpark drafter's is not taken");

  set_env("HF_HUB_OFFLINE", "");
  set_env("HF_HUB_CACHE", "");
  fs::remove_all(tmp, ec);
  std::printf("model_source_test: %s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
