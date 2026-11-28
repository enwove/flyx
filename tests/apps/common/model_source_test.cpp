// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// apps/common/model_source: the DSpark drafter published beside a model, taken by default (wants_dspark): the one a
// GGUF on disk has beside it (dspark_beside: its name, then its header), and the default dspark-*.gguf of the model's
// repository, here a Hugging Face cache read offline.  Neither replaces a drafter that was named.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
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
  // What GET /v1/hub/drafter says of a name the dashboard was given: the selected file's own kind, unknown (the
  // loader's auto) until its header is in the cache, never DFlash2 for the repository's architecture.
  flyx::hub::Name generic_name;
  check(flyx::hub::parse_name(generic.id, &generic_name) && flyx::apps::drafter_kind(generic, generic_name).empty(),
        "a Hub name's uncached dflash GGUF is of unknown kind, not DFlash2 by its repository's architecture");
  write_gguf(to_path(flyx::hub::snapshot_dir(generic)) / "helper.gguf", "dflash", 1);
  listing = flyx::apps::hub_ggufs(generic);
  check(listing.size() == 1 && listing[0].cached && listing[0].cls.drafter_kind == "dspark",
        "cached selection uses the same file's Markov header");
  check(flyx::apps::drafter_kind(generic, generic_name) == "dspark", "and so does the Hub name's kind once cached");
  {
    flyx::hub::Repo plain_dflash = generic;
    plain_dflash.id = "acme/Generic-DFlash2";
    flyx::hub::Name n;
    write_gguf(to_path(flyx::hub::snapshot_dir(plain_dflash)) / "helper.gguf", "dflash", 0);
    check(flyx::hub::parse_name(plain_dflash.id, &n) && flyx::apps::drafter_kind(plain_dflash, n) == "dflash2",
          "a cached dflash GGUF without the Markov flag is DFlash2's");
    flyx::hub::Repo named = generic;
    named.id = "acme/Named-DSpark";
    named.files = {{"dspark-Ink-Q8_0.gguf", 0, "", ""}};
    check(flyx::hub::parse_name(named.id, &n) && flyx::apps::drafter_kind(named, n).empty(),
          "an uncached dspark-*.gguf is of unknown kind too: no kind from a name");
    // What a picked companion may tell a load (GET /v1/hub/files' drafter_kind): the cached header's kind alone; the
    // list's kind of an uncached file can be its name's guess.
    const auto named_listing = flyx::apps::hub_ggufs(named);
    check(named_listing.size() == 1 && named_listing[0].cls.drafter_kind == "dspark" &&
              named_listing[0].header_drafter_kind().empty(),
          "a misleadingly named uncached companion: listed as its name says, its mode left to the loader");
    check(listing.size() == 1 && listing[0].header_drafter_kind() == "dspark",
          "a cached one: the mode its header says");
    // A repository with a drafter's config (DFlash2) and a GGUF of another kind beside it, the GGUF named: the
    // selected file's own header (Markov: DSpark), not the repository's config.
    flyx::hub::Repo mixed = generic;
    mixed.id = "acme/Mixed-Drafters";
    mixed.architectures = {"DFlash2DraftModel"};
    mixed.files = {{"config.json", 0, "", ""}, {"model.safetensors", 0, "", ""}, {"dspark-Mixed-Q8_0.gguf", 0, "", ""}};
    flyx::hub::Name whole, gguf_named;
    check(flyx::hub::parse_name(mixed.id, &whole) && flyx::hub::parse_name(mixed.id + ":dspark-Mixed-Q8_0.gguf", &gguf_named),
          "the mixed repository's names parse");
    check(flyx::apps::drafter_kind(mixed, gguf_named).empty(), "its uncached GGUF named: unknown, not its config's DFlash2");
    write_gguf(to_path(flyx::hub::snapshot_dir(mixed)) / "dspark-Mixed-Q8_0.gguf", "dflash", 1);
    check(flyx::apps::drafter_kind(mixed, gguf_named) == "dspark", "cached: its own header's DSpark");
    check(flyx::apps::drafter_kind(mixed, whole) == "dflash2",
          "the repository named alone selects its config and safetensors: the config's DFlash2");
    // A cached mtp-*.gguf whose header is an ordinary model's, no MTP layers in it: listed as MTP by its name, but
    // neither GET /v1/hub/files' drafter_kind nor GET /v1/hub/drafter tells a load it is one.
    flyx::hub::Repo mtp_named = generic;
    mtp_named.id = "acme/Mtp-Named";
    mtp_named.files = {{"mtp-Tiny-Q8_0.gguf", 0, "", ""}};
    write_gguf(to_path(flyx::hub::snapshot_dir(mtp_named)) / "mtp-Tiny-Q8_0.gguf", "qwen35");
    const auto mtp_listing = flyx::apps::hub_ggufs(mtp_named);
    flyx::hub::Name mtp_name;
    check(mtp_listing.size() == 1 && mtp_listing[0].cached && mtp_listing[0].cls.drafter_kind == "mtp" &&
              mtp_listing[0].header_drafter_kind().empty(),
          "a cached mtp-*.gguf with no MTP layers: MTP by its name in the list, no mode for a load");
    check(flyx::hub::parse_name(mtp_named.id + ":mtp-Tiny-Q8_0.gguf", &mtp_name) &&
              flyx::apps::drafter_kind(mtp_named, mtp_name).empty(),
          "and the Hub name's kind the same: none");
    // The same file in a folder of this machine (GET /v1/hub/dir, the --models-dir part of /v1/hub/cached): listed as
    // MTP by its name, its drafter_mode, what a load is told, none; a header that proves its kind gives the mode.
    const fs::path folder = tmp / "plain-mtp";
    write_gguf(folder / "mtp-Plain-Q8_0.gguf", "qwen35");
    write_gguf(folder / "dspark-Plain-Q8_0.gguf", "dflash", 1);
    const auto here = flyx::apps::scan_model_dirs({from_path(folder)}, 1);
    const auto at = [&](const char* file) {
      for (const auto& m : here)
        if (m.file == file) return m;
      return flyx::apps::LocalModel{};
    };
    check(at("mtp-Plain-Q8_0.gguf").kind == "drafter" && at("mtp-Plain-Q8_0.gguf").drafter_kind == "mtp" &&
              at("mtp-Plain-Q8_0.gguf").drafter_mode.empty(),
          "a local mtp-*.gguf with no MTP layers: MTP in the list, no mode for a load");
    check(at("dspark-Plain-Q8_0.gguf").drafter_kind == "dspark" && at("dspark-Plain-Q8_0.gguf").drafter_mode == "dspark",
          "a local DSpark GGUF whose header says so: that mode");
  }
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
    // Expose the cached LoRA snapshot via refs/main so offline fetch_lora can resolve its Hub name.
    const fs::path lrefs = lsnap.parent_path().parent_path() / "refs";
    fs::create_directories(lrefs);
    std::ofstream(lrefs / "main", std::ios::binary) << lora.commit;
    ok = flyx::apps::fetch_lora("'" + lora.id + ":small'", {}, nullptr, &path, &err);
    check(ok && to_path(path) == lsnap / "adapters" / "small",
          "single-quoted Hub LoRA name resolves via fallback" + (ok ? std::string() : ": " + err));
  }

  // A local adapter named by one of the two files PEFT saves: their directory, the current one for a bare file name
  // (never an empty path, which would load no adapter and let the base model answer in its name); any other file is
  // refused, and fetch_model never turns an adapter asked for into none.
  {
    const fs::path adir = tmp / "bare-lora";
    fs::create_directories(adir);
    std::ofstream(adir / "adapter_config.json", std::ios::binary) << R"({"peft_type":"LORA","r":8})";
    std::ofstream(adir / "adapter_model.safetensors", std::ios::binary) << std::string(64, 'w');
    std::ofstream(adir / "notes.txt", std::ios::binary) << "not an adapter";
    const fs::path cwd = fs::current_path();
    fs::current_path(adir);
    std::string p;
    for (const std::string file : {"adapter_model.safetensors", "adapter_config.json"}) {
      p.clear();
      ok = flyx::apps::fetch_lora(file, {}, nullptr, &p, &err);
      check(ok && !p.empty() && fs::equivalent(to_path(p), adir, ec),
            "a bare " + file + " is its directory, the current one: " + (ok ? p : err));
      p.clear();
      ok = flyx::apps::fetch_lora(from_path(adir / file), {}, nullptr, &p, &err);
      check(ok && to_path(p) == adir, "a path to " + file + " is its directory: " + (ok ? p : err));
    }
    for (const std::string file : {std::string("notes.txt"), from_path(adir / "notes.txt")}) {
      p = "untouched";
      ok = flyx::apps::fetch_lora(file, {}, nullptr, &p, &err);
      check(!ok && p == "untouched" && err.find("not a LoRA adapter") != std::string::npos,
            "another file is refused, not its directory taken: " + err);
    }
    ok = flyx::apps::fetch_model(from_path(model), {}, false, {}, nullptr, &f, &err, "", false, false,
                                 "adapter_model.safetensors");
    check(ok && !f.lora.empty() && fs::equivalent(to_path(f.lora), adir, ec),
          "fetch_model keeps the adapter a bare file name asks for: " + (ok ? f.lora : err));
    ok = flyx::apps::fetch_model(from_path(model), {}, false, {}, nullptr, &f, &err, "", false, false, "notes.txt");
    check(!ok && err.find("not a LoRA adapter") != std::string::npos,
          "and refuses a load whose adapter is no adapter: " + err);
#if defined(_WIN32)
    // A volume that ignores case: the two files spelled otherwise are the same files, so their directory.
    for (const std::string file : {"ADAPTER_CONFIG.JSON", "Adapter_Model.SafeTensors"}) {
      p.clear();
      ok = flyx::apps::fetch_lora(file, {}, nullptr, &p, &err);
      check(ok && p == ".", "a bare " + file + " is the current directory on Windows: " + (ok ? p : err));
      p.clear();
      ok = flyx::apps::fetch_lora(from_path(adir / file), {}, nullptr, &p, &err);
      check(ok && to_path(p) == adir, "and a path to it, its directory: " + (ok ? p : err));
    }
#endif
    // The remote path policy on the sibling compared with the file named (its identity is looked at through its
    // links): the policy is asked of it first, and a sibling it denies is not compared.
    const std::string config_sibling = from_path(adir / "adapter_config.json");
    std::vector<std::string> asked;
    const auto deny_config = [&](std::string_view q) {
      asked.emplace_back(q);
      return q != config_sibling;
    };
    p.clear();
    ok = flyx::apps::fetch_lora(from_path(adir / "adapter_model.safetensors"), {}, nullptr, &p, &err, deny_config);
    check(ok && to_path(p) == adir && std::find(asked.begin(), asked.end(), config_sibling) != asked.end(),
          "the policy is asked of each sibling before it is compared (its weights file named: taken)");
#if defined(_WIN32)
    // A spelling of the config that is not the sibling's own: the sibling, denied, is not compared, so not taken.
    ok = flyx::apps::fetch_lora(from_path(adir / "ADAPTER_CONFIG.JSON"), {}, nullptr, &p, &err, deny_config);
    check(!ok && err.find("remote path policy") != std::string::npos,
          "a denied sibling is not looked at through the name typed: " + err);
#endif
    fs::current_path(cwd);
  }

  // Leading ~ expansion (HOME and USERPROFILE) and outer single-quote fallback (cmd.exe paste).
  {
    const std::string saved_home = flyx::os::getenv_str("HOME");
    const std::string saved_profile = flyx::os::getenv_str("USERPROFILE");
    const fs::path fake_home = tmp / "home";
    const fs::path home_model = fake_home / "models" / "Home-Tiny-Q4_K_M.gguf";
    const fs::path home_lora = fake_home / "adapters" / "tiny-lora";
    write_gguf(home_model, "inkling2_loop");
    fs::create_directories(home_lora);
    std::ofstream(home_lora / "adapter_config.json", std::ios::binary) << R"({"peft_type":"LORA","r":8})";

    set_env("HOME", from_path(fake_home));
    ok = fetched("~/models/Home-Tiny-Q4_K_M.gguf", {}, false, &f, &err);
    check(ok && fs::weakly_canonical(to_path(f.model), ec) == fs::weakly_canonical(home_model, ec),
          "fetch_model expands ~/ via HOME" + (ok ? std::string() : ": " + err));
#if defined(_WIN32)
    ok = fetched("~\\models\\Home-Tiny-Q4_K_M.gguf", {}, false, &f, &err);
    check(ok && fs::weakly_canonical(to_path(f.model), ec) == fs::weakly_canonical(home_model, ec),
          "fetch_model expands ~\\ via HOME" + (ok ? std::string() : ": " + err));
#endif
    std::string lora_path;
    ok = flyx::apps::fetch_lora("~/adapters/tiny-lora", {}, nullptr, &lora_path, &err);
    check(ok && fs::weakly_canonical(to_path(lora_path), ec) == fs::weakly_canonical(home_lora, ec),
          "fetch_lora expands ~/ via HOME" + (ok ? std::string() : ": " + err));

    // When HOME is empty, USERPROFILE is used.
    set_env("HOME", "");
    set_env("USERPROFILE", from_path(fake_home));
    ok = fetched("~/models/Home-Tiny-Q4_K_M.gguf", {}, false, &f, &err);
    check(ok && fs::weakly_canonical(to_path(f.model), ec) == fs::weakly_canonical(home_model, ec),
          "fetch_model expands ~/ via USERPROFILE when HOME is unset" + (ok ? std::string() : ": " + err));

    // Outer single-quote fallback for local paths (including ~/) and Hub specs.
    ok = fetched("'" + from_path(model) + "'", {}, false, &f, &err);
    check(ok && f.model == from_path(model),
          "fetch_model strips outer single quotes around a local path" + (ok ? std::string() : ": " + err));
    ok = fetched("'~/models/Home-Tiny-Q4_K_M.gguf'", {}, false, &f, &err);
    check(ok && fs::weakly_canonical(to_path(f.model), ec) == fs::weakly_canonical(home_model, ec),
          "fetch_model strips outer single quotes and expands ~/" + (ok ? std::string() : ": " + err));
    ok = fetched("'" + repo + ":Q4_K_M'", {}, false, &f, &err);
    check(ok && fs::path(to_path(f.model)).filename() == "Ink-Tiny-Q4_K_M.gguf",
          "fetch_model strips outer single quotes around a Hub name" + (ok ? std::string() : ": " + err));
    ok = flyx::apps::fetch_lora("'" + from_path(home_lora) + "'", {}, nullptr, &lora_path, &err);
    check(ok && fs::weakly_canonical(to_path(lora_path), ec) == fs::weakly_canonical(home_lora, ec),
          "fetch_lora strips outer single quotes around a local directory" + (ok ? std::string() : ": " + err));
    for (const std::string bad : {"''", "'", "'no-such-file'"}) {
      ok = fetched(bad, {}, false, &f, &err);
      check(!ok && err.find(bad + ": no such file, and not a Hugging Face name") != std::string::npos,
            "invalid quoted spec still reports crisp error: " + err);
    }

    // Remote path policy must be checked on tilde-expanded and single-quote-stripped paths before probing disk.
    auto deny_expanded_home = [&](std::string_view p) {
      return p.find(from_path(fake_home)) == std::string_view::npos;
    };
    auto deny_unc_after_unquote = [](std::string_view p) {
      return !p.starts_with("\\\\") && !p.starts_with("//");
    };
    ok = flyx::apps::fetch_model("~/models/Home-Tiny-Q4_K_M.gguf", {}, false, {}, nullptr, &f, &err, "", true, true, "",
                                 deny_expanded_home);
    check(!ok && err.find("remote path policy") != std::string::npos,
          "fetch_model checks allow_path on tilde-expanded path before disk access: " + err);
    ok = flyx::apps::fetch_model("'~/models/Home-Tiny-Q4_K_M.gguf'", {}, false, {}, nullptr, &f, &err, "", true, true,
                                 "", deny_expanded_home);
    check(!ok && err.find("remote path policy") != std::string::npos,
          "fetch_model checks allow_path on single-quoted tilde-expanded path before disk access: " + err);
    ok = flyx::apps::fetch_model("'\\\\attacker\\share\\model.gguf'", {}, false, {}, nullptr, &f, &err, "", true, true,
                                 "", deny_unc_after_unquote);
    check(!ok && err.find("remote path policy") != std::string::npos,
          "fetch_model checks allow_path on single-quote-stripped UNC path before disk access: " + err);
    ok = flyx::apps::fetch_lora("'\\\\attacker\\share\\lora'", {}, nullptr, &lora_path, &err, deny_unc_after_unquote);
    check(!ok && err.find("remote path policy") != std::string::npos,
          "fetch_lora checks allow_path on single-quote-stripped UNC path before disk access: " + err);

    set_env("HOME", saved_home);
    set_env("USERPROFILE", saved_profile);
  }
  // expand_user_path and scan_model_dirs: nested non-HF layouts, mixed/unsupported files, tilde, symlinks + cycles.
  {
    const std::string home = flyx::os::home_dir();
    if (!home.empty()) {
      check(flyx::apps::expand_user_path("~") == home, "expand_user_path(\"~\") returns home_dir");
      check(flyx::apps::expand_user_path("~/") == home && flyx::apps::expand_user_path("~///") == home,
            "expand_user_path(\"~/\") and \"~///\" return home_dir");
      const fs::path expected = to_path(home) / "models";
      check(to_path(flyx::apps::expand_user_path("~/models")) == expected,
            "expand_user_path(\"~/models\") expands tilde");
#if defined(_WIN32)
      check(to_path(flyx::apps::expand_user_path("~//models")) == expected &&
                to_path(flyx::apps::expand_user_path("~\\\\models")) == expected,
            "expand_user_path strips repeated slashes after ~ instead of escaping home");
#else
      check(to_path(flyx::apps::expand_user_path("~//models")) == expected,
            "expand_user_path strips repeated slashes after ~ instead of escaping home");
      check(flyx::apps::expand_user_path("~\\models") == "~\\models",
            "expand_user_path preserves literal backslash after ~ on POSIX");
#endif
      check(flyx::apps::expand_user_path("/abs/models") == "/abs/models",
            "expand_user_path leaves non-tilde paths untouched");
    }
#if !defined(_WIN32)
    // POSIX path preservation: backslashes are literal filename characters, and symlink/.. follows the real parent.
    {
      const fs::path plain_dir = tmp / "posix_dir";
      const fs::path slash_dir = tmp / "posix_dir\\";
      const fs::path target_sub = tmp / "real_parent" / "child";
      const fs::path link_path = tmp / "link_to_child";
      fs::create_directories(plain_dir);
      fs::create_directories(slash_dir);
      fs::create_directories(target_sub);
      std::error_code sym_ec;
      fs::create_directory_symlink(target_sub, link_path, sym_ec);
      check(flyx::apps::canonical_model_dir(from_path(plain_dir)) !=
                flyx::apps::canonical_model_dir(from_path(slash_dir)),
            "canonical_model_dir keeps /dir and /dir\\ distinct on POSIX");
      if (!sym_ec) {
        const std::string via_link_parent = flyx::apps::canonical_model_dir(from_path(link_path) + "/..");
        const std::string lexical_parent = flyx::apps::canonical_model_dir(from_path(tmp));
        const std::string real_parent = flyx::apps::canonical_model_dir(from_path(tmp / "real_parent"));
        std::error_code eq_ec;
        const bool resolves_real = fs::equivalent(to_path(via_link_parent), to_path(real_parent), eq_ec) && !eq_ec;
        check(via_link_parent != lexical_parent && resolves_real,
              "canonical_model_dir preserves symlink/.. (resolving to real parent, not lexical parent): " +
                  via_link_parent);
      }
    }
#endif
    const fs::path scan_root = tmp / "scan_root";
    const fs::path nested = scan_root / "Author" / "Model" / "Nested-Q4_K_M.gguf";
    const fs::path unsupported = scan_root / "Other" / "Unsupported-Q4.gguf";
    const fs::path ext_dir = tmp / "external_models";
    const fs::path linked_model = ext_dir / "Linked-Q4_K_M.gguf";
    const fs::path hidden_target = scan_root / "z-real";
    const fs::path hidden_target_model = hidden_target / "Real-Q4_K_M.gguf";
    const fs::path deep_target = scan_root / "z-deep-real";
    const fs::path deep_target_model = deep_target / "sub" / "Deep-Q4_K_M.gguf";
    write_gguf(nested, "qwen35");
    write_gguf(unsupported, "some_future_arch");
    write_gguf(linked_model, "qwen3");
    write_gguf(hidden_target_model, "qwen3");
    write_gguf(deep_target_model, "qwen3");
    std::ofstream(scan_root / "README.txt", std::ios::binary) << "ignore me";

    // Create directory symlinks: external target, cycle, hidden alias to z-real, and deep alias to z-deep-real.
    std::error_code sym_ec;
    fs::create_directory_symlink(ext_dir, scan_root / "linked", sym_ec);
    const bool symlinks_ok = !sym_ec;
    if (symlinks_ok) {
      std::error_code e1, e2, e3;
      fs::create_directory_symlink(scan_root, scan_root / "Author" / "loop", e1);
      fs::create_directory_symlink(hidden_target, scan_root / ".alias-hidden", e2);
      fs::create_directories(scan_root / "a" / "b" / "c", e3);
      fs::create_directory_symlink(deep_target, scan_root / "a" / "b" / "c" / "deep-link", e3);
    }

    const auto found = flyx::apps::scan_model_dirs({from_path(scan_root), from_path(tmp / "does_not_exist")}, 4);
    bool saw_nested = false, saw_unsupported = false, saw_linked = false, saw_hidden_real = false,
         saw_deep_real = false;
    for (const auto& m : found) {
      const fs::path p = to_path(m.path);
      if (p.filename() == "Nested-Q4_K_M.gguf") saw_nested = true;
      if (p.filename() == "Unsupported-Q4.gguf") saw_unsupported = true;
      if (p.filename() == "Linked-Q4_K_M.gguf") saw_linked = true;
      if (p.filename() == "Real-Q4_K_M.gguf") saw_hidden_real = true;
      if (p.filename() == "Deep-Q4_K_M.gguf") saw_deep_real = true;
    }
    check(saw_nested && saw_unsupported,
          "scan_model_dirs finds nested GGUFs and does not break on unsupported architectures or non-GGUF files");
    const std::string want_root = flyx::apps::canonical_model_dir(from_path(scan_root));
    check(!found.empty(), "scan_model_dirs found at least one entry");
    for (const auto& m : found) {
      check(m.root == want_root, "scan_model_dirs stamps LocalModel::root with canonical root: " + m.root);
    }
    if (symlinks_ok) {
      check(saw_linked, "scan_model_dirs follows directory symlinks and survives symlink cycles");
      check(saw_hidden_real, "excluded hidden symlink does not prevent scanning its real target directory");
      check(saw_deep_real, "deep symlink visit does not prevent shallower scan of its real target directory");
    }

    // Security regression: scan_model_dirs with allow_path + ScanFs must NEVER call any target-following filesystem
    // operation (is_directory, weakly_canonical, is_regular_file, file_size, last_write_time, classify_gguf) on a
    // blocked symlink/reparse root or child entry.
    {
      std::vector<std::string> touched_blocked;
      auto record_if_blocked = [&](const fs::path& p, const char* op) {
        const std::string s = from_path(p);
        if (s.find("to_unc") != std::string::npos || s.find("blocked_root") != std::string::npos)
          touched_blocked.push_back(std::string(op) + ":" + s);
      };
      flyx::apps::ScanFs fake_fs;
      fake_fs.is_directory = [&](const fs::path& p, std::error_code&) {
        record_if_blocked(p, "is_directory");
        const std::string s = from_path(p);
        return s == "/safe" || s == "/safe/sub";
      };
      fake_fs.weakly_canonical = [&](const fs::path& p, std::error_code&) {
        record_if_blocked(p, "weakly_canonical");
        return p;
      };
      fake_fs.for_each_dir_entry = [&](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                                       std::error_code&) {
        record_if_blocked(p, "for_each_dir_entry");
        const std::string s = from_path(p);
        std::vector<fs::path> entries;
        if (s == "/safe") entries = {fs::path("/safe/to_unc_dir"), fs::path("/safe/to_unc_file.gguf"), fs::path("/safe/sub")};
        else if (s == "/safe/sub") entries = {fs::path("/safe/sub/good-Q4_K_M.gguf")};
        for (const auto& e : entries)
          if (!visit(e)) break;
      };
      fake_fs.is_regular_file = [&](const fs::path& p, std::error_code&) {
        record_if_blocked(p, "is_regular_file");
        return from_path(p) == "/safe/sub/good-Q4_K_M.gguf";
      };
      fake_fs.file_size = [&](const fs::path& p, std::error_code&) -> uint64_t {
        record_if_blocked(p, "file_size");
        return 1024;
      };
      fake_fs.last_write_time = [&](const fs::path& p, std::error_code&) -> fs::file_time_type {
        record_if_blocked(p, "last_write_time");
        return fs::file_time_type::clock::now();
      };
      fake_fs.classify_gguf = [&](const std::string& p, const std::string&, uint64_t,
                                  fs::file_time_type) -> flyx::apps::GgufClassification {
        record_if_blocked(to_path(p), "classify_gguf");
        return {"gguf", "", "", -1};
      };
      auto allow = [](std::string_view p) {
        return p.find("to_unc") == std::string_view::npos && p.find("blocked_root") == std::string_view::npos;
      };
      const auto gated = flyx::apps::scan_model_dirs({"/blocked_root", "/safe"}, 4, allow, &fake_fs);
      check(touched_blocked.empty(),
            "scan_model_dirs never invokes target-following fs ops on blocked roots/children (touched=" +
                std::to_string(touched_blocked.size()) + ")");
      check(gated.size() == 1 && gated[0].file == "good-Q4_K_M.gguf",
            "scan_model_dirs still discovers allowed siblings when blocked links are skipped");
    }

    // Streamed entry cap: scan_model_dirs stops iterating huge directories immediately at max_entries.
    {
      int yielded = 0;
      flyx::apps::ScanFs cap_fs;
      cap_fs.is_directory = [](const fs::path& p, std::error_code&) { return from_path(p) == "/big"; };
      cap_fs.weakly_canonical = [](const fs::path& p, std::error_code&) { return p; };
      cap_fs.for_each_dir_entry = [&](const fs::path& p, const std::function<bool(const fs::path&)>& visit,
                                      std::error_code&) {
        if (from_path(p) != "/big") return;
        for (int i = 0; i < 200; ++i) {
          ++yielded;
          if (!visit(fs::path("/big/file_" + std::to_string(i) + ".txt"))) break;
        }
      };
      cap_fs.is_regular_file = [](const fs::path&, std::error_code&) { return true; };
      cap_fs.file_size = [](const fs::path&, std::error_code&) -> uint64_t { return 1; };
      cap_fs.last_write_time = [](const fs::path&, std::error_code&) { return fs::file_time_type::clock::now(); };
      (void)flyx::apps::scan_model_dirs({"/big"}, 4, {}, &cap_fs, /*max_entries=*/10);
      check(yielded == 10,
            "scan_model_dirs streams directory entries and halts at max_entries (yielded=" + std::to_string(yielded) + ")");
    }

    // Dependent-path allow_path enforcement across dspark_beside, split GGUF shards, LoRA PEFT dirs, and drafter dirs.
    {
      // 1. dspark_beside skips sibling DSpark files blocked by allow_path.
      auto deny_dspark = [](std::string_view p) { return p.find("dspark-") == std::string_view::npos; };
      check(flyx::apps::dspark_beside(from_path(model), deny_dspark).empty(),
            "dspark_beside refuses sibling drafter blocked by allow_path");

      // 2. Split GGUF shard 2 blocked by allow_path fails ModelFile::open before mapping shard 2.
      const fs::path split_dir = tmp / "split_guard";
      const fs::path shard1 = split_dir / "demo-00001-of-00002.gguf";
      const fs::path shard2 = split_dir / "demo-00002-of-00002.gguf";
      auto write_split_gguf = [](const fs::path& p, uint32_t split_no, uint32_t split_count) {
        std::string b = "GGUF";
        auto u32 = [&](uint32_t v) { b.append(reinterpret_cast<const char*>(&v), 4); };
        auto u64 = [&](uint64_t v) { b.append(reinterpret_cast<const char*>(&v), 8); };
        auto str = [&](const std::string& s) { u64(s.size()), b += s; };
        u32(3);
        u64(0);  // tensors
        u64(3);  // metadata keys
        str("general.architecture"), u32(8), str("qwen3");
        str("split.no"), u32(4), u32(split_no);
        str("split.count"), u32(4), u32(split_count);
        fs::create_directories(p.parent_path());
        std::ofstream(p, std::ios::binary) << b;
      };
      write_split_gguf(shard1, 0, 2);
      write_split_gguf(shard2, 1, 2);
      flyx::ModelFile mf;
      std::string open_err;
      auto deny_shard2 = [](std::string_view p) { return p.find("00002-of-00002") == std::string_view::npos; };
      check(!mf.open(from_path(shard1), &open_err, deny_shard2) &&
                open_err.find("remote path policy") != std::string::npos,
            "ModelFile::open blocks split GGUF shard 2 rejected by allow_path: " + open_err);

      // 3. LoRA PEFT directory: ModelFile::add blocks adapter_model.safetensors rejected by allow_path.
      const fs::path lora_dir = tmp / "lora_guard";
      fs::create_directories(lora_dir);
      std::ofstream(lora_dir / "adapter_config.json", std::ios::binary)
          << R"({"peft_type":"LORA","r":4,"lora_alpha":8})";
      std::ofstream(lora_dir / "adapter_model.safetensors", std::ios::binary) << "stub";
      std::string add_err;
      auto deny_adapter_weights = [](std::string_view p) {
        return p.find("adapter_model.safetensors") == std::string_view::npos;
      };
      check(!mf.add(from_path(lora_dir), &add_err, deny_adapter_weights) &&
                add_err.find("remote path policy") != std::string::npos,
            "ModelFile::add blocks LoRA adapter weights rejected by allow_path: " + add_err);

      // 4. Drafter directory: drafter_source_kind and ModelFile::add respect allow_path for config.json and shards.
      const fs::path drafter_dir = tmp / "drafter_guard";
      fs::create_directories(drafter_dir);
      std::ofstream(drafter_dir / "config.json", std::ios::binary)
          << R"({"architectures":["DFlash2DraftModel"],"dflash_config":{}})";
      std::ofstream(drafter_dir / "model.safetensors", std::ios::binary) << "stub";
      auto deny_config = [](std::string_view p) { return p.find("config.json") == std::string_view::npos; };
      auto deny_shard = [](std::string_view p) { return p.find("model.safetensors") == std::string_view::npos; };
      check(flyx::drafter_source_kind(from_path(drafter_dir)) == "dflash2",
            "drafter_source_kind reads allowed drafter directory");
      check(flyx::drafter_source_kind(from_path(drafter_dir), deny_config).empty(),
            "drafter_source_kind skips config.json rejected by allow_path");
      add_err.clear();
      check(!mf.add(from_path(drafter_dir), &add_err, deny_shard) &&
                add_err.find("remote path policy") != std::string::npos,
            "ModelFile::add blocks drafter safetensors shard rejected by allow_path: " + add_err);
    }
  }
  set_env("HF_HUB_OFFLINE", "");
  set_env("HF_HUB_CACHE", "");
  fs::remove_all(tmp, ec);
  return flyx::test::finish("model_source_test");
}
