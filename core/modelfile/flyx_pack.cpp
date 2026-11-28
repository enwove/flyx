// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// A .flyx file packed from a GGUF (flyx pack): what the GGUF reader read of a model and of the files beside it, written
// out so that reading the packed file gives the same Contents (flyx_format.cpp takes "gguf" off "requires" and the
// bundled files off the header).  The blocks are copied byte for byte; nothing is converted.
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

#include "core/modelfile/readers.h"
#include "core/os/path.h"

namespace flyx::modelfile {
namespace {

constexpr size_t kAlignData = 4096, kAlignEntry = 256;
constexpr uint32_t kRequiresFormat = 2;
// Header keys the reader reads of a packed file: never a caller's.
constexpr const char* kReaderKeys[] = {"tensors", "blobs", "bundled", "requires", "companions"};

uint64_t pad(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

// The data section as it is laid out: each piece at its offset (from the data's start).
struct Layout {
  struct Piece {
    uint64_t offset;
    std::string_view bytes;
  };
  std::vector<Piece> pieces;
  uint64_t end = 0;
  uint64_t place(std::string_view bytes) {
    const uint64_t at = end;
    pieces.push_back({at, bytes});
    end = pad(at + bytes.size(), kAlignEntry);
    return at;
  }
};

bool tensors_of(const ModelFile::Contents& c, Layout* data, nlohmann::json* out, std::string* err) {
  *out = nlohmann::json::object();
  for (size_t i = 0; i < c.tensors.size(); ++i) {
    const flyx_tensor_view& v = c.tensors[i];
    const char* dtype = flyx_dtype_name(v.dtype);
    if (!dtype) return *err = "tensor " + c.names[i] + ": a dtype a .flyx file does not hold", false;
    std::vector<int64_t> shape(v.shape, v.shape + v.ndim);
    const uint64_t at = data->place({static_cast<const char*>(v.data), size_t(v.nbytes)});
    (*out)[c.names[i]] = {{"dtype", dtype}, {"shape", shape}, {"offset", at}, {"nbytes", v.nbytes}};
  }
  return true;
}

// A file's header as its reader gave it, without what a .flyx file's reader puts there of the file's layout.
nlohmann::json own_header(const nlohmann::json& h) {
  nlohmann::json out = h;
  for (const char* k : kReaderKeys) out.erase(k);
  return out;
}

}  // namespace

bool write_packed(const ModelFile& file, const nlohmann::json& extra, const std::string& path, std::string* err) {
  const ModelFile::Contents& c = file.contents();
  const nlohmann::json& h = c.header;
  if (!h.is_object() || !h.contains("source") || !h["source"].is_object() ||
      h["source"].value("format", "") != "gguf" || !h.contains("gguf"))
    return *err = "flyx pack packs a GGUF model; this one is not", false;
  if (!extra.is_null() && !extra.is_object())
    return *err = "the keys added to a packed file's header are not an object", false;

  Layout data;
  nlohmann::json header = own_header(h);
  for (auto it = extra.begin(); extra.is_object() && it != extra.end(); ++it) {
    for (const char* k : kReaderKeys)
      if (it.key() == k) return *err = "\"" + it.key() + "\" is the reader's in a packed file's header", false;
    if (header.contains(it.key())) return *err = "\"" + it.key() + "\" is the GGUF reader's in this header", false;
    header[it.key()] = it.value();
  }
  header["requires"] = {"gguf"};
  if (!tensors_of(c, &data, &header["tensors"], err)) return false;
  header["blobs"] = nlohmann::json::object();
  for (const auto& [name, bytes] : c.blobs) {
    const uint64_t at = data.place(bytes);
    header["blobs"][name] = {{"offset", at}, {"nbytes", uint64_t(bytes.size())}};
  }
  header["bundled"] = nlohmann::json::array();
  for (int i = 0; i < file.companions(); ++i) {
    const auto role = file.companion_role(i);
    if (role != ModelFile::CompanionRole::Drafter && role != ModelFile::CompanionRole::Projector)
      return *err = "companion " + std::to_string(i) + ": only a drafter or a projector is packed beside the model",
             false;
    const ModelFile::Contents& b = file.companion(i);
    nlohmann::json one = {{"role", role == ModelFile::CompanionRole::Drafter ? "drafter" : "projector"},
                          {"header", own_header(b.header)}};
    if (!b.directory_config.is_null()) one["directory_config"] = b.directory_config;
    if (!tensors_of(b, &data, &one["tensors"], err)) return false;
    header["bundled"].push_back(std::move(one));
  }
  if (header["bundled"].empty()) header.erase("bundled");

  const std::string text = header.dump();
  const uint64_t hlen = text.size();
  const uint64_t data_start = pad(24 + hlen, kAlignData);

  const std::filesystem::path target = os::to_path(path);
  std::filesystem::path staged = target;
  staged += ".partial";
  {
    std::ofstream f(staged, std::ios::binary | std::ios::trunc);
    if (!f) return *err = path + ": cannot write " + os::from_path(staged), false;
    const uint32_t version = kRequiresFormat, reserved = 0;
    f.write("FLYX\0\0\0\0", 8);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f << text;
    uint64_t at = 24 + hlen;
    const std::vector<char> zeros(kAlignData, '\0');
    auto zero_to = [&](uint64_t to) {
      for (; at < to; at += std::min<uint64_t>(to - at, zeros.size()))
        f.write(zeros.data(), std::streamsize(std::min<uint64_t>(to - at, zeros.size())));
    };
    for (const auto& piece : data.pieces) {
      zero_to(data_start + piece.offset);
      constexpr size_t kChunk = size_t(64) << 20;  // the mapping read a piece at a time
      for (size_t o = 0; o < piece.bytes.size(); o += kChunk)
        f.write(piece.bytes.data() + o, std::streamsize(std::min(kChunk, piece.bytes.size() - o)));
      at += piece.bytes.size();
    }
    zero_to(data_start + data.end);
    if (!f.flush()) return *err = path + ": writing " + os::from_path(staged) + " failed", false;
  }
  std::error_code ec;
  std::filesystem::rename(staged, target, ec);
  if (ec) return *err = path + ": " + ec.message(), false;
  return true;
}

}  // namespace flyx::modelfile
