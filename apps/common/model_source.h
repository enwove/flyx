// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Models as people name them: a file or a directory on disk, else a Hugging Face Hub name,
// "unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS" for a GGUF of a repository or "z-lab/Qwen3.8-27B-DFlash2" for a drafter,
// fetched into the Hugging Face cache (core/hub).  The engine itself is given paths.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/exec/engine.h"
#include "core/hub/hub.h"
#include "core/modelfile/modelfile.h"

namespace flyx::apps {

// A drafter that goes with a model.
struct DrafterSource {
  std::string kind;            // "mtp", or a drafter of the target's taps ("dflash2", "dflash", "dspark", "dspine",
                               // "eagle4"), as its header or the Hub says; "" if unknown
  std::string from;            // "model file" (its own MTP layer), a Hub name or file of a repository, or a path
  std::optional<uint64_t> size;  // bytes on disk; none where only the Hub could tell (or inside the model's file)
};

// A drafter verified with flyx (its output plain decoding's, and faster) for the models made from a base model (a model
// card's base_model): the list drafters.json in the Hub's Enwove/flyx-drafters keeps (docs/models.md).
struct KnownDrafter {
  std::string repo;      // its Hub name
  std::string kind;      // --drafter's word for it ("dflash2", "dflash", "dspark", "dspine", "eagle4")
  std::string base;      // the model it was made (and verified) for
  bool exact = false;    // the model is that one (its own repository); else it names it as a base model: inferred
  std::string relation;  // inferred: how the Hub says the model was made from its base ("quantized", ...), or ""
  // its files' bytes where the Hugging Face cache has them (a load's suggestions); unknown elsewhere
  std::optional<uint64_t> size = std::nullopt;
};

struct Fetched {
  std::string model;                  // a path
  std::vector<std::string> drafters;  // paths: the ones named, the MTP file of the model's repository when the
                                      // model's GGUF lacks the layer, and the DSpark drafter published beside the
                                      // model (want_dspark)
  std::string mmproj;                 // path to multimodal vision projector GGUF (empty: none)
  std::string mmproj_source;          // display label: repo/path (Hub) or local path
  std::optional<uint64_t> mmproj_size;  // bytes on disk
  std::string lora;                   // the LoRA adapter's directory (fetch_lora; empty: none)
  std::string name;                   // the model's name for the API: its file's, without the extension
  std::vector<std::string> notes;     // for the log: a drafter published for this model, the MTP file taken
  std::vector<DrafterSource> drafter_sources;  // what the model drafts with: its own MTP layer, the files above
  std::vector<KnownDrafter> suggested;         // the verified drafters for it (known_drafters) not named
};

// Authorization predicate evaluated BEFORE any target-following inspection, stat, open, or recursion (true if allowed).
using PathAllowFn = flyx::PathAllowFn;

// Fetches what is missing. want_mtp: take the repository's MTP file for a GGUF without the layer.
// want_vision: take the repository's mmproj GGUF when no explicit mmproj path was given.
// want_dspark: with no DSpark drafter among `drafters`, take the one published beside the model: its repository's
// first compatible DSpark GGUF in preference order, or one named for a local GGUF in the same directory.
// lora: an adapter (fetch_lora; empty: none).
bool fetch_model(const std::string& model, const std::vector<std::string>& drafters, bool want_mtp,
                 const hub::ProgressFn& progress, const std::atomic<bool>* stop, Fetched* out, std::string* err,
                 const std::string& mmproj = "", bool want_vision = true, bool want_dspark = false,
                 const std::string& lora = "", const PathAllowFn& allow_path = {});

// A LoRA adapter as Hugging Face PEFT saves it: a directory as it is (or one of its two files, for their directory:
// "." for a bare file name; any other file is refused), else a Hub repository's adapter_config.json and
// adapter_model.safetensors fetched into the cache, "org/name" at its top, "org/name:FOLDER" in a folder of it (a
// repository of several: the folder's path or its last part), "@REV" for a revision.  `path`: the directory, never
// empty on success.
bool fetch_lora(const std::string& spec, const hub::ProgressFn& progress, const std::atomic<bool>* stop,
                std::string* path, std::string* err, const PathAllowFn& allow_path = {});
// fetch_lora's part after the repository is described: the adapter `folder` names in `repo` (spec's FOLDER; spec
// for messages), its config refused past what a load reads (modelfile::kPeftConfigMaxBytes) before either file is
// fetched.
bool fetch_lora_in(const hub::Repo& repo, const std::string& spec, const std::string& folder,
                   const hub::ProgressFn& progress, const std::atomic<bool>* stop, std::string* path,
                   std::string* err, const PathAllowFn& allow_path = {});

// The drafter a file or directory holds, as its own fields say ("mtp", "dflash2", "dflash", "dspark", "dspine",
// "eagle4"; "" when they say none): a GGUF's header (general.architecture "dflash", or MTP layers), a drafter
// directory's config.json, a .flyx file's "draft".  The file is not mapped; no name is read for it.
std::string drafter_kind(const std::string& path, const PathAllowFn& allow_path = {});
// The same of what a Hub name selects in `r`: a drafter repository's, as its config says (its architecture); a GGUF's,
// as its header says once the whole file is in the cache, else "" (never from its name): a GGUF of architecture dflash
// is a DFlash2 or a DSpark drafter by its own header alone, so the repository's architecture names neither.
std::string drafter_kind(const hub::Repo& r, const hub::Name& name, const PathAllowFn& allow_path = {});

// Whether `eo` drafts with the MTP layer when the model has one (not --no-spec, not --drafter dflash2 or none).
bool wants_mtp(const EngineOptions& eo);

// Whether `eo` drafts with a DSpark drafter published beside the model, none being named: --drafter dspark or all,
// or the default (auto) with no drafter file named (`drafters`: the ones named, by the command line or the server).
bool wants_dspark(const EngineOptions& eo, const std::vector<std::string>& drafters);

// The verified drafters this build runs (a kind it does not know left out) for a model: those made for the repository
// it is (`repo`, "" when unknown), then, inferred, those made for the base models it names (`base_models`), each in the
// list's order.  `hub`: `repo` is a Hub repository and the call talks to the Hub anyway: the list is the Hub's (asked
// once a process), and an inferred drafter's relation is how the Hub says `repo` was made from its base; else the list
// is the Hugging Face cache's copy, read without the network, and no relation is known.
std::vector<KnownDrafter> known_drafters(const std::string& repo, const std::vector<std::string>& base_models,
                                         bool hub, const PathAllowFn& allow_path = {});
// The same from drafters.json's text, with `relation` for the inferred ones: none when the text is not version 1 of the
// list throughout (an entry that is no {base, repo, kind} of strings, a kind of auto, all or none, a repository twice).
std::vector<KnownDrafter> known_drafters_in(const std::string& list, const std::string& repo,
                                            const std::vector<std::string>& base_models, const std::string& relation);

// Throttled Hub fetch progress logger writing to stderr, optionally forwarding every tick to `hook`.
hub::ProgressFn make_progress_logger(const char* prefix, double interval_ms, hub::ProgressFn hook = {});

// The command line's way: the model, --drafter-file and --lora arguments made paths in place, progress on standard
// error.
bool fetch_named(std::string* model, EngineOptions* eo, std::string* err);

// A model's name for the API from its path: the file's name without .gguf / .flyx.
// Local paths recognize those extensions in any case; Hub model IDs keep their published spelling.
std::string model_name_of(const std::string& path, bool local = true);

// Whether this build reads a vision projector beside a model of this architecture (interfaces::vision_architectures);
// "" (the architecture is not known yet): not ruled out.
bool reads_vision(const std::string& arch);

// Canonical classification of any GGUF file on disk (local directories, Hub cache snapshots, and drafter attachment).
// Header identities take precedence over filename conventions; where no header can be read, the name decides
// (hub::mtp_gguf_name, hub::dspark_gguf_name, hub::mmproj_gguf_name).  A model GGUF whose name says it holds none
// (hub::model_gguf_name: a file under old_support/, for one) is ignored.
struct GgufClassification {
  std::string kind;          // "gguf" (model), "drafter", "mmproj", or "" (ignored, e.g. imatrix)
  std::string drafter_kind;  // "dflash2", "dspark" or "mtp" when kind == "drafter", else ""
  std::string arch;          // general.architecture, "" if the header is unreadable
  int mtp_layers = -1;       // <arch>.nextn_predict_layers when valid, -1 if unreadable/invalid
};
// The drafter mode a GGUF's header proves, the one a load may be told: architecture dflash's (dspark or dflash2, by
// its Markov flag), "mtp" for MTP layers, "" otherwise (unread, or a kind only the file's name gives: an mtp-*.gguf of
// an ordinary model with no MTP layers is "mtp" in drafter_kind, none here), as drafter_kind(path) reads it.
std::string drafter_mode_of(const GgufClassification& c);
GgufClassification classify_gguf(const std::string& path, const std::string& rel_name = "", uint64_t size = 0,
                                 std::filesystem::file_time_type modified = std::filesystem::file_time_type::min(),
                                 const PathAllowFn& allow_path = {});

// One Hub GGUF group with its local header identity, when available. The selector works with hub::select.
struct HubGguf {
  hub::GgufChoice choice;
  GgufClassification cls;
  std::string disk, selector;
  bool cached = false;
  std::filesystem::file_time_type modified = std::filesystem::file_time_type::min();
  // The drafter kind the file's own header says (a whole file in the cache, its header read), "" while it is not or
  // where the header names none: what a load may be told, as drafter_kind(path) reads it, where cls.drafter_kind can
  // be a guess from the file's name (mtp-*.gguf of an ordinary model with no MTP layers is "mtp" there, none here).
  std::string header_drafter_kind() const { return cached ? drafter_mode_of(cls) : ""; }
};
std::vector<HubGguf> hub_ggufs(const hub::Repo& repo, const PathAllowFn& allow_path = {});
// The companion of `kind` ("mmproj", "mtp", "dflash2" or "dspark") a repository offers first: the Hub's preferred
// projector or MTP file (hub::mmproj_file, hub::mtp_file), the default DSpark drafter (hub::companion_files), else
// the first complete one listed.
const HubGguf* preferred_companion(const std::vector<HubGguf>& ggufs, const std::string& kind);

// A GGUF's header (flyx::gguf_peek: its architecture, MTP layers, the repositories it comes from), read once while
// the file keeps its size and time; valid false when it is no GGUF.
flyx::GgufPeek gguf_header(const std::string& path, const PathAllowFn& allow_path = {});

// Model, drafter, and vision projector files in directories of the user's own (flyx serve --models-dir,
// FLYX_MODELS_DIR): *.gguf (classified via classify_gguf; complete split GGUFs grouped as one entry pointing to
// shard 1 with their summed size), *.flyx, and LoRA adapters' folders (adapter_config.json beside
// adapter_model.safetensors), at most 6 directories down, hidden directories and node_modules left out.
struct LocalModel {
  std::string path;          // absolute
  std::string root;          // canonical scan root that found this entry
  std::string file;          // its name
  std::string kind;          // "gguf", "flyx", "drafter", "mmproj", or "lora" (a PEFT adapter: path its folder)
  std::string drafter_kind;  // "dflash2", "dspark" or "mtp" when kind == "drafter", else "" (the list's: maybe a name's)
  std::string drafter_mode;  // the mode its header proves (drafter_mode_of): what a load may be told
  std::string quant;         // from the name, "" when it names none
  uint64_t size = 0;
  int parts = 0;             // > 1 for a complete split GGUF (path points to shard 1, size is the sum)
  int mtp_layers = -1;       // cached <arch>.nextn_predict_layers for kind == "gguf"
  std::filesystem::file_time_type modified = std::filesystem::file_time_type::min();
};
// Expands a leading "~" ("~", "~/...", and "~\..." on Windows) to the user's home directory; other paths stay unchanged.
std::string expand_user_path(const std::string& path);
// Canonicalizes a configured --models-dir / FLYX_MODELS_DIR root (expands "~", resolves absolute, normalizes
// native separators, strips trailing native separators except filesystem roots without collapsing ".." across symlinks).
std::string canonical_model_dir(std::string_view raw);
// Injectable filesystem operations for deterministic tests verifying scan_model_dirs checks allow_path first
// and bounds streamed directory iteration.
struct ScanFs {
  std::function<bool(const std::filesystem::path&, std::error_code&)> is_directory;
  std::function<bool(const std::filesystem::path&, std::error_code&)> is_regular_file;
  std::function<uint64_t(const std::filesystem::path&, std::error_code&)> file_size;
  std::function<std::filesystem::file_time_type(const std::filesystem::path&, std::error_code&)> last_write_time;
  std::function<std::filesystem::path(const std::filesystem::path&, std::error_code&)> weakly_canonical;
  // Streams directory entries one by one; stops immediately when `visit` returns false.
  std::function<void(const std::filesystem::path&, const std::function<bool(const std::filesystem::path&)>& visit,
                     std::error_code&)>
      for_each_dir_entry;
  std::function<GgufClassification(const std::string&, const std::string&, uint64_t,
                                   std::filesystem::file_time_type)>
      classify_gguf;
};
// `deepest`: how many directories down a file may be (1: only in the directory named).
// `max_entries`: upper bound on entries inspected per root; enumeration stops immediately at the cap.
std::vector<LocalModel> scan_model_dirs(const std::vector<std::string>& dirs, int deepest = 6,
                                        const PathAllowFn& allow_path = {}, const ScanFs* fs_ops = nullptr,
                                        int max_entries = 200000);
// FLYX_MODELS_DIR's directories: separated by ';' on Windows, ':' elsewhere (as PATH is).
std::vector<std::string> models_dirs_from_env();
// Unix seconds for the model listing; last_write_time's error value is reported as 0.
double modified_unix(std::filesystem::file_time_type modified);

}  // namespace flyx::apps
