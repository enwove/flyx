// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include <cstring>
#include <fstream>
#include <limits>
#include <map>

#include "core/interfaces/json_depth.h"
#include "core/modelfile/readers.h"
#include "core/os/path.h"

namespace flyx::modelfile {
namespace {

// A file's first 8 bytes; files written before the project took the name flyx start with its old name (format 1
// all the same), and read the same.
constexpr char kMagic[8] = {'F', 'L', 'Y', 'X', '\0', '\0', '\0', '\0'};
constexpr char kOldMagic[8] = {'F', 'A', 'S', 'T', 'E', 'S', 'T', '\0'};
// Format 2 is format 1's layout with a header that names, in "requires", parts of the file a reader must not ignore
// (a target's own adapters, "lora": a build that dropped them would decode another model); the model of the file's
// architecture says which of them it reads (models/model.cpp).  A file without such parts stays format 1, which every
// build reads; a build that reads format 1 only refuses the others.
constexpr uint32_t kPlainFormat = 1, kRequiresFormat = 2;
constexpr size_t kAlignData = 4096;

// A dtype by its name in a header; GGML's blocks (a packed GGUF's) by GGML's names.
constexpr std::pair<const char*, int> kDtypes[] = {
    {"bf16", FLYX_DT_BF16},
    {"f16", FLYX_DT_F16},
    {"f32", FLYX_DT_F32},
    {"f64", FLYX_DT_F64},
    {"i32", FLYX_DT_I32},
    {"u32", FLYX_DT_U32},
    {"u8", FLYX_DT_U8},
    {"i64", FLYX_DT_I64},
    {"nvfp4", FLYX_DT_NVFP4},
    {"f8e4m3", FLYX_DT_F8E4M3},
    {"ggml_q4_0", FLYX_DT_GGML_Q4_0},
    {"ggml_q4_1", FLYX_DT_GGML_Q4_1},
    {"ggml_q5_0", FLYX_DT_GGML_Q5_0},
    {"ggml_q5_1", FLYX_DT_GGML_Q5_1},
    {"ggml_q8_0", FLYX_DT_GGML_Q8_0},
    {"ggml_q2_k", FLYX_DT_GGML_Q2_K},
    {"ggml_q3_k", FLYX_DT_GGML_Q3_K},
    {"ggml_q4_k", FLYX_DT_GGML_Q4_K},
    {"ggml_q5_k", FLYX_DT_GGML_Q5_K},
    {"ggml_q6_k", FLYX_DT_GGML_Q6_K},
    {"ggml_iq2_xxs", FLYX_DT_GGML_IQ2_XXS},
    {"ggml_iq2_xs", FLYX_DT_GGML_IQ2_XS},
    {"ggml_iq2_s", FLYX_DT_GGML_IQ2_S},
    {"ggml_iq3_xxs", FLYX_DT_GGML_IQ3_XXS},
    {"ggml_iq3_s", FLYX_DT_GGML_IQ3_S},
    {"ggml_iq1_s", FLYX_DT_GGML_IQ1_S},
    {"ggml_iq1_m", FLYX_DT_GGML_IQ1_M},
    {"ggml_iq4_nl", FLYX_DT_GGML_IQ4_NL},
    {"ggml_iq4_xs", FLYX_DT_GGML_IQ4_XS},
    {"ggml_nvfp4", FLYX_DT_GGML_NVFP4},
    {"ggml_mxfp4", FLYX_DT_GGML_MXFP4},
    {"ggml_q2_0", FLYX_DT_GGML_Q2_0},
};

int dtype_of(const std::string& s) {
  for (const auto& [name, dtype] : kDtypes)
    if (s == name) return dtype;
  return -1;
}

// The part of a file packed from a GGUF that a reader must not ignore: its layout is the GGUF's.
constexpr const char* kGgufPart = "gguf";

ModelFile::CompanionRole role_of(const std::string& s) {
  if (s == "drafter") return ModelFile::CompanionRole::Drafter;
  if (s == "projector") return ModelFile::CompanionRole::Projector;
  throw std::runtime_error("a bundled file's role is \"drafter\" or \"projector\", not \"" + s + "\"");
}

}  // namespace

const char* flyx_dtype_name(int dtype) {
  for (const auto& [name, d] : kDtypes)
    if (d == dtype) return name;
  return nullptr;
}

bool is_flyx(const uint8_t* data, size_t size) {
  return size >= 8 && (std::memcmp(data, kMagic, 8) == 0 || std::memcmp(data, kOldMagic, 8) == 0);
}

namespace {

// A .flyx file's fixed part and header JSON from its first `size` bytes: its format version, the header's length.
bool read_header(const uint8_t* p, size_t size, const std::string& path, nlohmann::json* header, uint32_t* ver,
                 uint64_t* hlen, std::string* err) {
  if (size < 24 || !is_flyx(p, size)) return *err = path + ": not a .flyx file", false;
  std::memcpy(ver, p + 8, 4);
  std::memcpy(hlen, p + 16, 8);
  if (*ver != kPlainFormat && *ver != kRequiresFormat)
    return *err = path + ": file format " + std::to_string(*ver) + ", this build reads " +
                  std::to_string(kPlainFormat) + " and " + std::to_string(kRequiresFormat),
           false;
  if (*hlen > size - 24) return *err = path + ": truncated header", false;
  const std::string_view header_sv(reinterpret_cast<const char*>(p + 24), size_t(*hlen));
  if (!interfaces::json_depth_within(header_sv)) return *err = path + ": header JSON nested too deeply", false;
  try {
    *header = nlohmann::json::parse(header_sv);
  } catch (const std::exception& e) {
    return *err = path + ": " + e.what(), false;
  }
  return true;
}

}  // namespace

bool read_flyx(const uint8_t* p, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err, std::vector<Bundled>* bundled) {
  uint32_t ver;
  uint64_t hlen;
  if (!read_header(p, size, path, &out->header, &ver, &hlen, err)) return false;
  // Offsets and sizes are the file's own numbers: compared by what is left of the file, never added up first
  // (a sum of two of them can wrap round and pass for a place inside it).
  const size_t data_start = (24 + size_t(hlen) + kAlignData - 1) / kAlignData * kAlignData;
  const size_t data_size = data_start < size ? size - data_start : 0;
  auto inside = [&](uint64_t off, uint64_t n) { return data_start <= size && off <= data_size && n <= data_size - off; };
  // A header's "tensors" as views into the data: each one inside the file, its size its shape's and dtype's.
  auto read_tensors = [&](const nlohmann::json& ts, ModelFile::Contents* c) -> bool {
    c->names.reserve(ts.size());
    c->tensors.reserve(ts.size());
    for (auto it = ts.begin(); it != ts.end(); ++it) {
      const auto& r = it.value();
      flyx_tensor_view v{};
      c->names.push_back(it.key());
      v.dtype = dtype_of(r.at("dtype").get<std::string>());
      auto shape = r.at("shape").get<std::vector<int64_t>>();
      if (v.dtype < 0 || shape.size() > 4) {
        *err = "tensor " + it.key() + ": unsupported dtype or rank";
        return false;
      }
      v.ndim = static_cast<int32_t>(shape.size());
      for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] < 0) {
          *err = "tensor " + it.key() + ": a negative dimension";
          return false;
        }
        v.shape[i] = shape[i];
      }
      uint64_t off = r.at("offset").get<uint64_t>();
      v.nbytes = r.at("nbytes").get<uint64_t>();
      if (!inside(off, v.nbytes)) {
        *err = "tensor " + it.key() + " lies outside the file (truncated download?)";
        return false;
      }
      uint64_t want_bytes = 0;
      if (!tensor_nbytes(v.dtype, shape, &want_bytes) || v.nbytes != want_bytes) {
        *err = "tensor " + it.key() + ": byte size does not match shape and dtype";
        return false;
      }
      v.data = p + data_start + off;
      c->index[it.key()] = c->tensors.size();
      c->tensors.push_back(v);
    }
    for (size_t i = 0; i < c->tensors.size(); ++i) c->tensors[i].name = c->names[i].c_str();
    return true;
  };
  try {
    out->arch = out->header.value("arch", "");
    const bool named = out->header.is_object() && out->header.contains("requires");
    if (named != (ver == kRequiresFormat)) {
      *err = path + ": file format " + std::to_string(ver) +
             (named ? " with \"requires\" (format 2's: the parts a reader must not ignore)"
                    : " without \"requires\" (the parts a reader must not ignore)");
      return false;
    }
    if (named) {
      const auto& r = out->header.at("requires");
      bool names = r.is_array() && !r.empty();
      for (const auto& x : r) names = names && x.is_string() && !x.get<std::string>().empty();
      if (!names) {
        *err = path + ": \"requires\" is not a list of the parts a reader must not ignore: " + r.dump();
        return false;
      }
    }

    if (!read_tensors(out->header.at("tensors"), out)) return false;

    // A file packed from a GGUF (write_packed): its header is the GGUF reader's, its bundled files companions.
    bool packed = false;
    if (named) {
      auto& r = out->header["requires"];
      for (auto it = r.begin(); it != r.end();)
        if (*it == kGgufPart) packed = true, it = r.erase(it);
        else ++it;
      if (r.empty()) out->header.erase("requires");
    }
    if (packed) {
      const auto& h = out->header;
      if (!h.contains("source") || !h["source"].is_object() || h["source"].value("format", "") != "gguf" ||
          !h.contains("gguf") || !h["gguf"].is_object()) {
        *err = path + ": requires \"gguf\" without a GGUF's header (source.format \"gguf\", \"gguf\")";
        return false;
      }
    }
    if (out->header.contains("bundled")) {
      if (!packed) {
        *err = path + ": \"bundled\" files in a file not packed from a GGUF (\"requires\": [\"gguf\"])";
        return false;
      }
      if (!bundled) {
        *err = path + " bundles files of its own: give it as the model, not beside one";
        return false;
      }
      for (const auto& b : out->header.at("bundled")) {
        Bundled one;
        one.role = role_of(b.at("role").get<std::string>());
        one.c.header = b.at("header");
        if (!one.c.header.is_object() || one.c.header.contains("bundled") || one.c.header.contains("companions")) {
          *err = path + ": a bundled file's header is not one file's";
          return false;
        }
        one.c.arch = one.c.header.value("arch", "");
        if (b.contains("directory_config")) one.c.directory_config = b.at("directory_config");  // a drafter directory's
        if (!read_tensors(b.at("tensors"), &one.c)) return false;
        bundled->push_back(std::move(one));
      }
      out->header.erase("bundled");
    }

    if (out->header.contains("blobs"))
      for (auto it = out->header["blobs"].begin(); it != out->header["blobs"].end(); ++it) {
        uint64_t off = it->at("offset").get<uint64_t>(), n = it->at("nbytes").get<uint64_t>();
        if (!inside(off, n)) {
          *err = "blob " + it.key() + " lies outside the file (truncated download?)";
          return false;
        }
        out->blobs[it.key()] = {reinterpret_cast<const char*>(p + data_start + off), static_cast<size_t>(n)};
      }
  } catch (const std::exception& e) {  // not JSON, or a description without what every file has
    *err = path + ": bad header: " + e.what();
    return false;
  }
  return true;
}

bool peek_flyx_header(const std::string& path, nlohmann::json* h) {
  std::ifstream f(os::to_path(path), std::ios::binary);
  std::string bytes(24, '\0');
  if (!f.read(bytes.data(), 24) || !is_flyx(reinterpret_cast<const uint8_t*>(bytes.data()), 24)) return false;
  uint64_t hlen = 0;
  std::memcpy(&hlen, bytes.data() + 16, 8);
  if (hlen > (uint64_t(64) << 20)) return false;
  bytes.resize(24 + size_t(hlen));
  if (!f.read(bytes.data() + 24, std::streamsize(hlen))) return false;
  uint32_t ver;
  std::string err;
  return read_header(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), path, h, &ver, &hlen, &err) &&
         h->is_object();
}

std::string flyx_draft_kind(const std::string& path) {
  nlohmann::json h;  // a drafter file's "draft" section says which it is
  if (!peek_flyx_header(path, &h) || !h.contains("draft") || !h["draft"].is_object()) return "";
  const auto kind = h["draft"].find("kind");
  return kind != h["draft"].end() && kind->is_string() ? kind->get<std::string>() : "";
}

}  // namespace flyx::modelfile
