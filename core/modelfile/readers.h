// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The file formats a ModelFile reads, one reader each.  A reader recognises
// its format by the first bytes and fills ModelFile::Contents from the
// mapped file.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/modelfile/modelfile.h"

namespace flyx::modelfile {

// Byte size of one element of an unblocked tensor dtype (0 for block-quantized or unknown dtypes).
inline uint64_t plain_dtype_bytes(int dtype) {
  switch (dtype) {
    case FLYX_DT_U8:
    case FLYX_DT_F8E4M3: return 1;
    case FLYX_DT_BF16:
    case FLYX_DT_F16: return 2;
    case FLYX_DT_F32:
    case FLYX_DT_I32:
    case FLYX_DT_U32: return 4;
    case FLYX_DT_F64:
    case FLYX_DT_I64: return 8;
    default: return 0;
  }
}

// Byte size of a plain (or packed NVFP4) tensor with `shape`, checking for negative dims and overflow.
inline bool plain_tensor_nbytes(int dtype, const std::vector<int64_t>& shape, uint64_t* out) {
  uint64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) return false;
    const uint64_t ud = static_cast<uint64_t>(d);
    if (ud != 0 && n > std::numeric_limits<uint64_t>::max() / ud) return false;
    n *= ud;
  }
  if (dtype == FLYX_DT_NVFP4) {
    if (n % 2 != 0) return false;
    *out = n / 2;
    return true;
  }
  const uint64_t elem = plain_dtype_bytes(dtype);
  if (!elem || n > std::numeric_limits<uint64_t>::max() / elem) return false;
  *out = n * elem;
  return true;
}

// Byte size of a tensor of any dtype a file may hold: plain_tensor_nbytes's, or GGML blocks whose rows (the last
// dimension) are whole blocks.
inline bool tensor_nbytes(int dtype, const std::vector<int64_t>& shape, uint64_t* out) {
  const int64_t per = flyx_block_values(dtype);
  if (!per) return plain_tensor_nbytes(dtype, shape, out);
  if (shape.empty() || shape.back() % per) return false;
  uint64_t blocks = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] < 0) return false;
    const uint64_t d = static_cast<uint64_t>(i + 1 == shape.size() ? shape[i] / per : shape[i]);
    if (d != 0 && blocks > std::numeric_limits<uint64_t>::max() / d) return false;
    blocks *= d;
  }
  const uint64_t bb = static_cast<uint64_t>(flyx_block_bytes(dtype));
  if (blocks > std::numeric_limits<uint64_t>::max() / bb) return false;
  *out = blocks * bb;
  return true;
}

// .flyx (python/convert/flyx_convert/fileformat.py): magic, version, a
// JSON header, tensors and blobs in an aligned data section.  A file packed
// from a GGUF (flyx pack: write_packed below) requires "gguf": its header is the
// GGUF reader's (header["gguf"], source.format "gguf"), its tensors GGML blocks
// under their GGUF names, and its "bundled" files (a drafter, a projector) are
// what the GGUF reader read of them: they come back in `bundled`, in order, to
// be the model's first companions.  The reader takes "gguf" off "requires" and
// "bundled" off the header, so a packed file's Contents are its GGUF's.
struct Bundled {
  ModelFile::CompanionRole role = ModelFile::CompanionRole::Any;
  ModelFile::Contents c;
};
bool is_flyx(const uint8_t* data, size_t size);
bool read_flyx(const uint8_t* data, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err, std::vector<Bundled>* bundled = nullptr);
// The dtype's name in a .flyx header ("bf16", "ggml_q4_k", ...); null for one a file never holds.
const char* flyx_dtype_name(int dtype);
// A .flyx file holding what `file` read of a GGUF model and its companions (drafters, a projector; not a LoRA
// adapter, which runs beside a model only when asked for), its blocks byte for byte; `extra` (an object, or null for
// none) adds keys to the header ("recommended", "packed"): the ones a reader reads are refused.  Written to a file beside `path`, then renamed.
bool write_packed(const ModelFile& file, const nlohmann::json& extra, const std::string& path, std::string* err);

// GGUF, little-endian v2 and v3 (gguf_format.cpp): the container only.  The
// metadata becomes header["gguf"], the vocabulary the tokenizer tables, and
// the tensors keep their GGUF names; models/<arch> knows which is which.
// The multi-file overload also opens and maps all shards of a split GGUF
// (split.count > 1), keeping them alive in `maps`.
bool is_gguf(const uint8_t* data, size_t size);
bool read_gguf(const uint8_t* data, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err, bool need_tokenizer = true);
bool read_gguf(std::vector<std::unique_ptr<os::MappedFile>>* maps, const std::string& path,
               ModelFile::Contents* out, std::string* err, bool need_tokenizer = true,
               const PathAllowFn& allow_path = {});

// A block drafter as published (drafter_dir.cpp): a directory with its config.json and *.safetensors,
// read with its original tensor names and configuration.  The shards stay mapped in `maps`.
bool is_drafter_dir(const std::string& path, const PathAllowFn& allow_path = {});
// The drafter kind named by config.json, or an empty string for an unknown architecture.
std::string drafter_dir_kind(const std::string& dir, const PathAllowFn& allow_path = {});
// The drafter kind from a .flyx header, or an empty string when it holds none.
std::string flyx_draft_kind(const std::string& path);
// A .flyx file's header alone, read from its first bytes (the file not mapped); false when it is none.
bool peek_flyx_header(const std::string& path, nlohmann::json* h);
bool read_drafter_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                      ModelFile::Contents* out, std::string* err, const PathAllowFn& allow_path = {});

// A LoRA adapter as Hugging Face PEFT saves it (peft_dir.cpp): a directory with adapter_config.json and
// adapter_model.safetensors, read as a "lora" companion whose tensors carry a .flyx file's adapter names
// (target.L{l}.{attn,mlp,gdn}.<linear>.lora_{a,b}) in the file's dtype and Hugging Face row order.
bool is_peft_dir(const std::string& path, const PathAllowFn& allow_path = {});
bool read_peft_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                   ModelFile::Contents* out, std::string* err, const PathAllowFn& allow_path = {});
// The most an adapter_config.json, and a safetensors file's header (its JSON), may hold: 16 MiB each, where PEFT
// writes a few hundred bytes and a few KiB.  read_peft_dir refuses past them before it reads or parses either, a
// fetch of an adapter from the Hub before it downloads either file, the Hub's read of a header
// (hub::safetensors_header) before it allocates it.
constexpr uint64_t kPeftConfigMaxBytes = uint64_t(16) << 20;
constexpr uint64_t kSafetensorsHeaderMaxBytes = uint64_t(16) << 20;
// What an adapter_config.json says of its adapter, for a list of them: rank, scale (alpha / r, or / sqrt(r)), the
// model it was trained on and the modules it targets ("" and none when not strings; a regex is one).
struct PeftConfig {
  int rank = 0;
  double scale = 0;
  std::string base;
  std::vector<std::string> targets;
};
// Whether an adapter_config.json's text is one read_peft_dir takes (plain LoRA, read_peft_dir's rules for the config;
// its tensors' names are checked only when it is read).  false with *why; *out filled as far as the text is JSON.
bool peft_config_ok(const std::string& text, PeftConfig* out, std::string* why);
// Whether adapter_model.safetensors's header (its JSON, without the 8 bytes of its length) holds what read_peft_dir
// reads for an adapter of rank r: decoder layers' linears, F32, BF16 or F16, [r, in] and [out, r].  false with *why;
// *linears: those it sits beside ("attn.q", "gdn.qkv", ...), *layers: one past its last layer.
bool peft_header_ok(const std::string& header, int rank, std::string* why, std::vector<std::string>* linears = nullptr,
                    int* layers = nullptr);

}  // namespace flyx::modelfile
