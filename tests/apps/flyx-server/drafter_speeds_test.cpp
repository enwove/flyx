// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// This machine's own drafter speeds (apps/flyx-server/drafter_speeds.h): each finished request's tokens and decode time
// summed by model and drafter, their mean read back (a repository's: over its models), kept in
// <config>/flyx/drafter_speeds.json, written added to what the file has then (another server's runs kept).
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "apps/flyx-server/drafter_speeds.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"
#include "tests/common/check.h"

int main() {
  namespace fs = std::filesystem;
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path dir = fs::temp_directory_path() /
                       ("flyx_drafter_speeds_" + std::to_string(flyx::os::process_id()) + "_" + std::to_string(now));
  fs::create_directories(dir);
#if defined(_WIN32)
  _putenv_s("APPDATA", dir.string().c_str());
#else
  setenv("XDG_CONFIG_HOME", dir.c_str(), 1);
#endif
  using namespace flyx::server;
  const std::string q4 = "unsloth/Qwen3-8B-GGUF:Q4_K_M", q8 = "unsloth/Qwen3-8B-GGUF:Q8_0", path = "/models/q.flyx";
  const fs::path file = drafter_speeds_path();
  auto write = [&](const char* text) {
    fs::create_directories(file.parent_path());
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    f << text;
  };
  auto read = [&] {  // the file as written now (closed again before the next write: Windows renames over no open file)
    std::ifstream f(file, std::ios::binary);
    return nlohmann::json::parse(f, nullptr, false);
  };
  // A request's end before anything asked for a speed: it waits on no file (read at the first query), and the speed
  // asked for then is the file's runs and it.
  write(R"({"version": 1, "speeds": [{"model": "seed/m", "drafter": "a/seed", "tokens": 600, "decode_ms": 1000}]})");
  add_drafter_speed("seed/m", {"a/seed"}, 200, 1000);
  check(std::fabs(drafter_speed("seed/m", "a/seed") - 400) < 1e-9, "a first run before any query: the file's and it");
  check(drafter_speed(q4, "a/spark") == 0, "none measured before a run");
  add_drafter_speed(q4, {"a/spark"}, 300, 1000);  // 300 tok/s
  add_drafter_speed(q4, {"a/spark", "a/flash"}, 100, 1000);
  add_drafter_speed(q8, {"a/spark"}, 500, 1000);
  add_drafter_speed(path, {"a/spark"}, 50, 1000);
  add_drafter_speed(q4, {"a/spark"}, 0, 1000);  // nothing decoded: not counted
  add_drafter_speed(q4, {}, 100, 1000);
  check(std::fabs(drafter_speed(q4, "a/spark") - 200) < 1e-9, "by model: 400 tokens in 2 s");
  check(std::fabs(drafter_speed(q4, "a/flash") - 100) < 1e-9, "each drafter the request drafted with");
  check(std::fabs(drafter_speed("unsloth/Qwen3-8B-GGUF", "a/spark") - 300) < 1e-9, "a repository: its models'");
  check(std::fabs(drafter_speed(path, "a/spark") - 50) < 1e-9, "a path: its own");
  check(read()["speeds"].size() == 1, "nothing written before a flush");
  // another server wrote its runs meanwhile
  write(R"({"version": 1, "speeds": [{"model": "other", "drafter": "a/spark", "tokens": 10, "decode_ms": 1000},
           {"model": "unsloth/Qwen3-8B-GGUF:Q4_K_M", "drafter": "a/spark", "tokens": 100, "decode_ms": 1000}]})");
  check(flush_drafter_speeds(), "a flush");
  double other = 0, mine = 0, seed = 0;
  for (const auto& e : read().value("speeds", nlohmann::json::array())) {
    if (e["model"] == "other") other = e["tokens"];
    if (e["model"] == q4 && e["drafter"] == "a/spark") mine = e["tokens"];
    if (e["model"] == "seed/m") seed = e["tokens"];
  }
  check(other == 10 && mine == 500 && seed == 200,
        "written added to the file's runs: another server's kept, the same key summed");
  // a file not version 1 of ours throughout: read as none, replaced by the next write
  write(R"({"version": 1, "speeds": [{"model": "other", "drafter": "a/spark", "tokens": 10, "decode_ms": 1000},
           {"model": "other", "drafter": "a/spark", "tokens": "many", "decode_ms": 1000}]})");
  add_drafter_speed(q8, {"a/spark"}, 100, 1000);
  check(flush_drafter_speeds(), "a flush over a malformed file");
  const nlohmann::json k = read();
  check(k.value("version", 0) == 1 && k["speeds"].size() == 1 && k["speeds"][0]["model"] == q8 &&
            k["speeds"][0]["tokens"] == 100,
        "a malformed file replaced: " + k.dump());
  {  // another server writing (its lock held): the flush fails, the runs kept and counted, then written once
    flyx::os::FileLock other_writer(flyx::os::from_path(file) + ".lock");
    std::string err;
    check(other_writer.try_lock(&err) == flyx::os::FileLock::Result::kLocked, "the other writer's lock: " + err);
    add_drafter_speed(q8, {"a/spark"}, 300, 1000);
    check(!flush_drafter_speeds(), "a flush while another server writes fails");
    check(read()["speeds"][0]["tokens"] == 100, "the file as it was");
    check(std::fabs(drafter_speed(q8, "a/spark") - 200) < 1e-9, "the runs kept still counted");
  }
  check(flush_drafter_speeds(), "the next flush");
  const nlohmann::json after = read();
  check(after["speeds"].size() == 1 && after["speeds"][0]["tokens"] == 400, "the kept runs written once: " +
                                                                                 after.dump());
  check(flush_drafter_speeds() && read()["speeds"][0]["tokens"] == 400, "nothing written twice");
  // The API's list: made for the model first, then each tier the fastest here first, null before a run (q8's runs are
  // the file's now: a/spark's alone).
  using flyx::apps::KnownDrafter;
  const nlohmann::ordered_json js = known_drafters_json(
      q8, {KnownDrafter{"a/none", "dspark", "b", false, "quantized"},
           KnownDrafter{"a/flash", "dflash2", "b", false, ""},
           KnownDrafter{"a/spark", "dspark", "b", false, "quantized"},
           KnownDrafter{"a/own", "dspark", "m", true, ""}});
  std::string order;
  for (const auto& e : js) order += e["repo"].get<std::string>() + " ";
  check(order == "a/own a/spark a/none a/flash " && js[0]["relation"].is_null() && js[0]["tok_s"].is_null() &&
            js[1]["relation"] == "quantized" && js[1]["tok_s"].is_number() && js[0]["size"].is_null(),
        "known_drafters_json: " + js.dump());
  std::error_code ec;
  fs::remove_all(dir, ec);
  return flyx::test::finish("drafter_speeds_test");
}
