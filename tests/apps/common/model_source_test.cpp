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
#include "core/modelfile/readers.h"
#include "core/os/env.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "tests/common/check.h"

namespace fs = std::filesystem;
using flyx::os::from_path;
using flyx::os::to_path;

namespace {

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
  flyx::test::print_passes = true;
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

  // A remote listing and a cold snapshot. Local blobs stand in for completed transfers to keep this test offline.
  const auto split_repo = cached_repo("Ink-Split-GGUF", {});
  const std::string first = "dspark-Ink-Q8_0-00001-of-00002.gguf", second = "dspark-Ink-Q8_0-00002-of-00002.gguf";
  flyx::hub::Repo cold;
  check(flyx::hub::in_cache(split_repo, "main", &cold), "split model cache exists");
  cold.from_cache = false;
  const auto snap = to_path(flyx::hub::snapshot_dir(cold));
  const auto blobs = snap.parent_path().parent_path() / "blobs";
  const std::string id1(40, 'a'), id2(40, 'b');
  write_gguf(blobs / id1, "dflash", 1);
  write_gguf(blobs / id2, "dflash");
  cold.files.push_back({second, fs::file_size(blobs / id2), "", id2});
  cold.files.push_back({first, fs::file_size(blobs / id1), "", id1});
  const auto choices = flyx::apps::hub_ggufs(cold);
  const auto* split = flyx::apps::preferred_companion(choices, "dspark");
  std::vector<std::string> paths;
  check(split && !split->cached && split->choice.file.path == first && split->choice.parts.size() == 2 &&
            flyx::hub::fetch(cold, split->choice.parts, &paths, {}, nullptr, &err) &&
            fs::exists(snap / first) && fs::exists(snap / second), "cold snapshot receives both DSpark shards");
  uint64_t fetched_bytes = 0;
  ok = flyx::apps::fetch_model(split_repo + ":Q4_K_M", {}, false,
                              [&](const flyx::hub::Progress& p) { fetched_bytes = p.total; },
                              nullptr, &f, &err, "", false, true);
  const auto total = fs::file_size(blobs / id1) + fs::file_size(blobs / id2);
  check(ok && f.drafters.size() == 1 && to_path(f.drafters.front()) == snap / first &&
            f.drafter_sources.size() == 1 && f.drafter_sources.front().size == total && fetched_bytes == total,
        "auto-fetch requests and reports the complete split");

  const auto neutral_repo = cached_repo("Ink-Neutral-GGUF", {{"helper-Q8_0.gguf", 1}});
  ok = fetched(neutral_repo + ":Q4_K_M", {}, true, &f, &err);
  check(ok && f.drafter_sources.size() == 1 && f.drafter_sources.front().from == neutral_repo + "/helper-Q8_0.gguf",
        "a cached Markov header identifies a neutral-name companion");

  flyx::hub::Repo generic;
  generic.id = "acme/Generic";
  generic.commit = std::string(40, 'd');
  generic.info.gguf_architecture = "dflash";
  generic.files = {{"helper.gguf", 0, "", ""}};
  auto listing = flyx::apps::hub_ggufs(generic);
  check(listing.size() == 1 && !listing[0].cached && listing[0].cls.kind == "drafter" &&
            listing[0].cls.drafter_kind.empty(), "uncached dflash GGUF is a selectable drafter of unknown type");
  write_gguf(to_path(flyx::hub::snapshot_dir(generic)) / "helper.gguf", "dflash", 1);
  listing = flyx::apps::hub_ggufs(generic);
  check(listing.size() == 1 && listing[0].cached && listing[0].cls.drafter_kind == "dspark",
        "cached selection uses the same file's Markov header");
  // A LoRA adapter of a repository listed as the Hub lists it (fetch_lora_in), its files' blobs standing in for the
  // transfers as above: a config longer than a load reads (modelfile::kPeftConfigMaxBytes) is refused by its listed
  // size before either file is fetched, the weights' blob there to be taken; one within it comes with its weights.
  {
    flyx::hub::Repo lora;
    lora.id = "acme/Tiny-LoRA";
    lora.revision = "main";
    lora.commit = std::string(40, 'e');
    const fs::path lsnap = to_path(flyx::hub::snapshot_dir(lora));
    const fs::path lblobs = lsnap.parent_path().parent_path() / "blobs";
    const std::string cfg_id(40, '1'), weights_id(40, '2'), config = R"({"peft_type": "LORA", "r": 8})";
    fs::create_directories(lblobs);
    std::ofstream(lblobs / cfg_id, std::ios::binary) << config;
    std::ofstream(lblobs / weights_id, std::ios::binary) << std::string(64, 'w');
    lora.files = {{"adapters/big/adapter_config.json", flyx::modelfile::kPeftConfigMaxBytes + 1, "", cfg_id},
                  {"adapters/big/adapter_model.safetensors", 64, "", weights_id},
                  {"adapters/small/adapter_config.json", config.size(), "", cfg_id},
                  {"adapters/small/adapter_model.safetensors", 64, "", weights_id}};
    int progressed = 0;
    std::string path;
    const auto counted = [&](const flyx::hub::Progress&) { ++progressed; };
    ok = flyx::apps::fetch_lora_in(lora, lora.id + ":big", "big", counted, nullptr, &path, &err);
    check(!ok && err.find("more than an adapter's config holds (16 MiB)") != std::string::npos && progressed == 0 &&
              !fs::exists(lsnap / "adapters" / "big"),
          "an adapter's config listed past 16 MiB is refused before either file is fetched: " + err);
    ok = flyx::apps::fetch_lora_in(lora, lora.id + ":small", "small", counted, nullptr, &path, &err);
    check(ok && to_path(path) == lsnap / "adapters" / "small" &&
              fs::file_size(lsnap / "adapters" / "small" / "adapter_config.json") == config.size() &&
              fs::file_size(lsnap / "adapters" / "small" / "adapter_model.safetensors") == 64,
          "one within it is fetched with its weights" + (ok ? std::string() : ": " + err));
  }
  // expand_user_path and scan_model_dirs: nested non-HF layouts, mixed/unsupported files, tilde, symlinks + cycles.
  {
    const std::string home = flyx::os::home_dir();
    if (!home.empty()) {
      check(flyx::apps::expand_user_path("~") == home, "expand_user_path(\"~\") returns home_dir");
      const fs::path expected = to_path(home) / "models";
      check(to_path(flyx::apps::expand_user_path("~/models")) == expected,
            "expand_user_path(\"~/models\") expands tilde");
      check(flyx::apps::expand_user_path("/abs/models") == "/abs/models",
            "expand_user_path leaves non-tilde paths untouched");
    }
    const fs::path scan_root = tmp / "scan_root";
    const fs::path nested = scan_root / "Author" / "Model" / "Nested-Q4_K_M.gguf";
    const fs::path unsupported = scan_root / "Other" / "Unsupported-Q4.gguf";
    const fs::path ext_dir = tmp / "external_models";
    const fs::path linked_model = ext_dir / "Linked-Q4_K_M.gguf";
    write_gguf(nested, "qwen35");
    write_gguf(unsupported, "some_future_arch");
    write_gguf(linked_model, "qwen3");
    std::ofstream(scan_root / "README.txt", std::ios::binary) << "ignore me";

    // Create a directory symlink to external_models and a symlink loop back to scan_root (skip if OS disallows).
    std::error_code sym_ec;
    fs::create_directory_symlink(ext_dir, scan_root / "linked", sym_ec);
    const bool symlinks_ok = !sym_ec;
    if (symlinks_ok) {
      std::error_code loop_ec;
      fs::create_directory_symlink(scan_root, scan_root / "Author" / "loop", loop_ec);
    }

    const auto found = flyx::apps::scan_model_dirs({from_path(scan_root), from_path(tmp / "does_not_exist")});
    bool saw_nested = false, saw_unsupported = false, saw_linked = false;
    for (const auto& m : found) {
      const fs::path p = to_path(m.path);
      if (p.filename() == "Nested-Q4_K_M.gguf") saw_nested = true;
      if (p.filename() == "Unsupported-Q4.gguf") saw_unsupported = true;
      if (p.filename() == "Linked-Q4_K_M.gguf") saw_linked = true;
    }
    check(saw_nested && saw_unsupported,
          "scan_model_dirs finds nested GGUFs and does not break on unsupported architectures or non-GGUF files");
    if (symlinks_ok)
      check(saw_linked, "scan_model_dirs follows directory symlinks and survives symlink cycles");
  }
  set_env("HF_HUB_OFFLINE", "");
  set_env("HF_HUB_CACHE", "");
  fs::remove_all(tmp, ec);
  return flyx::test::finish("model_source_test");
}
