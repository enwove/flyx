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

// .flyx (python/convert/flyx_convert/fileformat.py): magic, version, a
// JSON header, tensors and blobs in an aligned data section.
bool is_flyx(const uint8_t* data, size_t size);
bool read_flyx(const uint8_t* data, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err);

// GGUF, little-endian v2 and v3 (gguf_format.cpp): the container only.  The
// metadata becomes header["gguf"], the vocabulary the tokenizer tables, and
// the tensors keep their GGUF names; models/<arch> knows which is which.
// The multi-file overload also opens and maps all shards of a split GGUF
// (split.count > 1), keeping them alive in `maps`.
bool is_gguf(const uint8_t* data, size_t size);
bool read_gguf(const uint8_t* data, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err, bool need_tokenizer = true);
bool read_gguf(std::vector<std::unique_ptr<os::MappedFile>>* maps, const std::string& path,
               ModelFile::Contents* out, std::string* err, bool need_tokenizer = true);

// A block drafter as published (drafter_dir.cpp): a directory with its config.json and *.safetensors,
// read as python/convert would write it into a drafter file.  The shards stay mapped in `maps`.
std::string drafter_dir_kind(const std::string& path);
std::string flyx_draft_kind(const std::string& path);
bool is_drafter_dir(const std::string& path);
// Its config.json's block drafter (its architecture, or speculators' type: "dflash2", "dspark", "dspine"); "" for none.
std::string drafter_dir_kind(const std::string& dir);
// A .flyx file's "draft" section's kind, from its header alone ("dflash2", "dspark", "dspine"); "" when it holds none.
std::string flyx_draft_kind(const std::string& path);
// A .flyx file's header alone, read from its first bytes (the file not mapped); false when it is none.
bool peek_flyx_header(const std::string& path, nlohmann::json* h);
bool read_drafter_dir(const std::string& dir, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                      ModelFile::Contents* out, std::string* err);

}  // namespace flyx::modelfile
