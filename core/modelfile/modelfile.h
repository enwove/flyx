// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A model file, whatever its format: its description (a JSON
// document: arch, tokenizer, generation, chat, and what the architecture's
// model reads, "config" in the schema python/convert writes or "gguf", a
// GGUF file's own metadata), its tensors by the file's names and its named
// blobs (tokenizer tables, the chat template), in one memory map.  Each file
// format has a reader that fills these from its own layout (readers.h):
// .flyx and GGUF.  models/<arch> turns a file into a model (models/model.h).
#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "core/os/os.h"
#include "flyx/backend_abi.h"
#include "nlohmann/json.hpp"

namespace flyx {

// Authorization predicate evaluated BEFORE opening or inspecting a derived path (true if allowed).
using PathAllowFn = std::function<bool(std::string_view)>;

class ModelFile {
 public:
  // Maps the file and reads it with the reader its first bytes name.  A file
  // that holds a drafter alone ("kind": "drafter", or a GGUF file whose
  // general.architecture is "dflash") is no model: it goes beside one (add).
  bool open(const std::string& path, std::string* err, const PathAllowFn& allow_path = {});
  // Another file of the same model holding a part this one lacks (a
  // companion: unsloth ships the Qwen3.8-27B GGUF's MTP layer as a GGUF of its
  // own, a DFlash2 drafter file, 0038, its authors'
  // directory of config.json and safetensors, 0060, or its GGUF file as
  // llama.cpp reads it, general.architecture "dflash").  The i-th one added has
  // its header at header()["companions"][i] and its tensors named
  // companion_name(i, their name); its blobs are not read.  models/<arch>
  // decides what it takes from each, by name: nothing of one stands in for
  // this file's tensors or another companion's.
  bool add(const std::string& path, std::string* err, const PathAllowFn& allow_path = {});
  static constexpr const char* kCompanion = "companion/";
  static std::string companion_name(int i, const std::string& name) {
    return kCompanion + std::to_string(i) + "/" + name;
  }
  int companions() const { return int(companions_.size()); }

  const nlohmann::json& header() const { return c_.header; }
  const std::string& arch() const { return c_.arch; }
  const flyx_tensor_view* tensor(const std::string& name) const;
  // The names of this file's own tensors (not its companions'), in the file's order.
  const std::vector<std::string>& tensor_names() const { return c_.names; }
  // Empty view if the blob does not exist.
  std::string_view blob(const std::string& name) const;
  std::vector<int32_t> eos_ids() const;
  // The bytes of this file and its companions.
  uint64_t bytes() const;
  // Reads this file and then its companions from disk, front to back a piece at a time (os::MappedFile::read_ahead),
  // until done or `stop`: on a thread of its own while a backend reads the weights.
  void read_ahead(const std::atomic<bool>& stop) const;

  // What a reader fills (views and blobs point into the mapping, or into `owned`).
  struct Contents {
    nlohmann::json header;
    std::string arch;
    std::vector<std::string> names;  // the tensors' names (their views point here)
    std::vector<flyx_tensor_view> tensors;
    std::map<std::string, size_t> index;  // name -> tensors[i]
    std::map<std::string, std::string_view> blobs;
    std::map<std::string, std::vector<uint8_t>> owned;  // blobs a reader built (GGUF: the tokenizer tables)
  };

 private:
  std::vector<std::unique_ptr<os::MappedFile>> maps_;  // one file, or a split GGUF's shards
  Contents c_;
  struct Companion {
    std::vector<std::unique_ptr<os::MappedFile>> maps;  // a file, or a drafter directory's shards
    Contents c;
  };
  std::vector<std::unique_ptr<Companion>> companions_;
};

// Lightweight header metadata peeked from a GGUF file without mapping the whole file.
struct GgufPeek {
  bool valid = false;
  std::string arch;    // general.architecture ("dflash": a drafter alone)
  int mtp_layers = 0;  // <arch>.nextn_predict_layers, 0 if absent
  bool has_markov_context = false;  // dflash.has_markov_context (DSpark vs DFlash2)
  // The Hub repositories it says it comes from ("org/name"): the model it was converted from (general.source.repo_url)
  // and its base models (general.base_model.<i>.repo_url), as llama.cpp's converter writes them from the model card.
  std::vector<std::string> base_models;
};
std::string gguf_header_path(const std::string& path);  // shard 1 for a split GGUF
GgufPeek gguf_peek(const std::string& path, const PathAllowFn& allow_path = {});
// Where a model file says it comes from, read from its header alone: the Hub repository it is (a .flyx file's
// "source": "repo", as flyx_convert --repo writes it; a GGUF never says which repository holds it) and the
// repositories it names as its bases (a GGUF's GgufPeek::base_models, general.source.repo_url included; a .flyx file's
// "source": "base_models").
struct ModelOrigin {
  std::string repo;
  std::vector<std::string> base_models;
};
ModelOrigin model_origin(const std::string& path, const PathAllowFn& allow_path = {});

// The block drafter a config.json's architectures name, by one table (core/modelfile/drafter_dir.cpp): "dflash2",
// "dflash", "dspark", "dspine"; "" for none.
std::string drafter_kind_of(const std::vector<std::string>& architectures);
// The block drafter a drafter directory (its config.json) or a .flyx file (its header's "draft") holds, read without
// mapping it: "dflash2", "dflash", "dspark", "dspine"; "" when it holds none (a GGUF drafter's kind is its
// header's: gguf_peek).
std::string drafter_source_kind(const std::string& path, const PathAllowFn& allow_path = {});

}  // namespace flyx
