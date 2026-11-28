// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// flyx pack MODEL -o FILE: the GGUF model (a path or a Hugging Face name), the drafters the engine would take beside
// it (the ones named; the repository's MTP file and DSpark drafter unless --no-auto-drafters) and a projector when
// named, read as the engine reads them and written into one .flyx file (core/modelfile/flyx_pack.cpp): the blocks
// byte for byte, the drafters bundled so that the file needs nothing beside it.  The packed model is opened as the
// engine opens it before a byte is written; the files packed and their SHA-256 go in the header ("packed"), with the
// settings measured best for the file when given (--recommended: "recommended").
#include "apps/flyx-cli/pack.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "apps/common/model_source.h"
#include "core/hub/sha256.h"
#include "core/interfaces/model.h"
#include "core/modelfile/readers.h"
#include "core/os/path.h"
#include "nlohmann/json.hpp"

namespace flyx::pack {
namespace {

[[noreturn]] void usage(bool asked) {
  std::fprintf(asked ? stdout : stderr,
               "usage: flyx pack MODEL -o FILE.flyx [options]\n"
               "  MODEL                a .gguf file or a Hugging Face name (org/name:QUANT)\n"
               "  -o FILE              the .flyx file written\n"
               "  --drafter-file F     a drafter to bundle (repeatable; a path or a Hugging Face name)\n"
               "  --no-auto-drafters   only the drafters named: not the repository's MTP file or DSpark drafter\n"
               "  --mmproj F           a vision projector to bundle\n"
               "  --recommended FILE   a JSON object: the settings measured best for this file (its header's\n"
               "                       \"recommended\")\n");
  std::exit(asked ? 0 : 2);
}

[[noreturn]] void bad_usage(const std::string& why) {
  std::fprintf(stderr, "error: %s (flyx pack --help)\n", why.c_str());
  std::exit(2);
}

// A file's SHA-256, read a piece at a time; "" for a directory (a drafter as published) or a file not read.
std::string sha256_of(const std::string& path) {
  std::error_code ec;
  if (!std::filesystem::is_regular_file(os::to_path(path), ec)) return "";
  std::ifstream f(os::to_path(path), std::ios::binary);
  if (!f) return "";
  hub::Sha256 h;
  std::vector<char> buf(size_t(16) << 20);
  while (f) {
    f.read(buf.data(), std::streamsize(buf.size()));
    if (f.gcount() > 0) h.update(buf.data(), size_t(f.gcount()));
  }
  return h.hex();
}

}  // namespace

int run(int argc, char** argv) {
  std::string model, out, mmproj, recommended_file;
  std::vector<std::string> drafters;
  bool auto_drafters = true;
  for (int i = 2; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) bad_usage(s + " needs a value");
      return argv[++i];
    };
    if (s == "-h" || s == "--help") usage(true);
    else if (s == "-o") out = next();
    else if (s == "--drafter-file") drafters.push_back(next());
    else if (s == "--no-auto-drafters") auto_drafters = false;
    else if (s == "--mmproj") mmproj = next();
    else if (s == "--recommended") recommended_file = next();
    else if (!s.empty() && s[0] == '-') bad_usage("unknown option " + s);
    else if (model.empty()) model = s;
    else bad_usage("one model at a time: " + s);
  }
  if (model.empty() || out.empty()) usage(false);

  nlohmann::json extra = nlohmann::json::object();
  if (!recommended_file.empty()) {
    std::ifstream f(os::to_path(recommended_file), std::ios::binary);
    const nlohmann::json r = nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
    if (!f.is_open() || !r.is_object()) bad_usage(recommended_file + ": not a JSON object");
    extra["recommended"] = r;
  }

  std::string err;
  apps::Fetched fetched;
  if (!apps::fetch_model(model, drafters, /*want_mtp=*/auto_drafters, apps::make_progress_logger("pack", 2000, {}),
                         nullptr, &fetched, &err, mmproj, /*want_vision=*/!mmproj.empty(),
                         /*want_dspark=*/auto_drafters, /*lora=*/"", {})) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  for (const auto& note : fetched.notes) std::fprintf(stderr, "pack: %s\n", note.c_str());

  ModelFile file;
  if (!file.open(fetched.model, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  if (file.bundled()) {
    std::fprintf(stderr, "error: %s is packed already\n", fetched.model.c_str());
    return 1;
  }
  nlohmann::json files = nlohmann::json::array();
  auto packed_file = [&](const std::string& role, const std::string& path, const std::string& source) {
    std::fprintf(stderr, "pack: %s %s\n", role.c_str(), source.c_str());
    files.push_back({{"role", role},
                     {"name", os::from_path(os::to_path(path).filename())},
                     {"source", source},
                     {"sha256", sha256_of(path)}});
  };
  packed_file("model", fetched.model, model);
  // drafter_sources lists the model file's own MTP layer first when it has one: the files' sources are its last ones.
  const size_t first_source = fetched.drafter_sources.size() - fetched.drafters.size();
  for (size_t i = 0; i < fetched.drafters.size(); ++i) {
    const std::string& path = fetched.drafters[i];
    if (!file.add(path, &err, {}, ModelFile::CompanionRole::Drafter)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    packed_file("drafter", path, fetched.drafter_sources[first_source + i].from);
  }
  if (!fetched.mmproj.empty()) {
    if (!file.add(fetched.mmproj, &err, {}, ModelFile::CompanionRole::Projector)) {
      std::fprintf(stderr, "error: %s\n", err.c_str());
      return 1;
    }
    packed_file("projector", fetched.mmproj, fetched.mmproj_source);
  }
  // As the engine opens it: a file the engine would refuse is not written.
  if (!interfaces::open_model(file, {}, &err)) {
    std::fprintf(stderr, "error: %s does not open as a model: %s\n", model.c_str(), err.c_str());
    return 1;
  }
  extra["packed"] = {{"files", files}};
  if (!modelfile::write_packed(file, extra, out, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr, "pack: wrote %s (%d bundled)\n", out.c_str(), file.companions());
  return 0;
}

}  // namespace flyx::pack
