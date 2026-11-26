// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The GGUF reader, which reads the container and knows no model: files the
// gguf package wrote from the tiny models (make_gguf.py) give the tokenizer
// tables, stop tokens and chat template of the .flyx file of the same
// model, the file's own metadata, and its tensors under their GGUF names with
// row-major shapes; a damaged file is rejected with a message, nothing kept.
// A vision projector beside a model is the same split into shards as in one
// file, opened from any shard; a projector is never the model.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/modelfile/readers.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "core/tokenizer/tokenizer.h"

using namespace flyx;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok" : "FAIL", what.c_str());
  failures += !ok;
}

std::vector<uint8_t> read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}

template <class T>
void patch(std::vector<uint8_t>* data, size_t at, T value) {
  if (at > data->size() || sizeof(T) > data->size() - at) throw std::runtime_error("bad fixture patch");
  std::memcpy(data->data() + at, &value, sizeof(T));
}

size_t locate(const std::vector<uint8_t>& data, const std::string& s) {
  const auto at = std::search(data.begin(), data.end(), s.begin(), s.end());
  if (at == data.end()) throw std::runtime_error("missing fixture string " + s);
  return size_t(at - data.begin());
}

// The same model as .flyx (python/convert) and as GGUF.
void same_file(const std::string& canonical, const std::string& gguf, const std::string& arch) {
  ModelFile a, b;
  std::string error;
  if (!a.open(canonical, &error) || !b.open(gguf, &error)) {
    check(false, error);
    return;
  }
  const auto& h = b.header();
  check(b.arch() == arch && a.arch() == arch, "architecture name: " + gguf);
  check(h.at("source").at("format") == "gguf" && h.at("gguf").at("general.architecture") == arch &&
            h.at("gguf").contains(arch + ".embedding_length") && !h.at("gguf").contains("tokenizer.ggml.tokens"),
        "the file's metadata, without the vocabulary");

  auto ae = a.eos_ids(), be = b.eos_ids();
  std::sort(ae.begin(), ae.end());
  std::sort(be.begin(), be.end());
  check(ae == be, "stop tokens");
  for (const char* key : {"tokenizer.vocab", "tokenizer.merges", "chat_template.jinja"})
    check(!a.blob(key).empty() && a.blob(key) == b.blob(key), std::string("blob ") + key);
  Tokenizer ta, tb;
  check(ta.load(a, &error) && tb.load(b, &error), "tokenizer loads: " + error);
  for (const std::string s : {"Hello world! 123", "<|im_start|>user\nhello<|im_end|>\n<think>abc</think>",
                             "e\xcc\x81! \n\t<|endoftext|>"}) {
    check(ta.encode(s) == tb.encode(s), "raw prompt tokenization");
    const std::vector<TextSpan> literal;
    check(ta.encode(s, &literal) == tb.encode(s, &literal), "client text special-token handling");
  }

  // A matrix both files store the same way: the first layer's down projection.
  const auto* x = a.tensor("target.L0.mlp.down");
  const auto* y = b.tensor("blk.0.ffn_down.weight");
  bool same = x && y && x->ndim == 2 && y->ndim == 2 && x->shape[0] == y->shape[0] && x->shape[1] == y->shape[1] &&
              std::string(y->name) == "blk.0.ffn_down.weight";
  auto value = [](const flyx_tensor_view* t, size_t i) {  // float32 or bf16
    float v;
    if (t->dtype == FLYX_DT_F32) {
      std::memcpy(&v, static_cast<const uint8_t*>(t->data) + 4 * i, 4);
    } else {
      uint16_t h;
      std::memcpy(&h, static_cast<const uint8_t*>(t->data) + 2 * i, 2);
      const uint32_t bits = uint32_t(h) << 16;
      std::memcpy(&v, &bits, 4);
    }
    return v;
  };
  for (size_t i = 0; same && i < size_t(x->shape[0] * x->shape[1]); ++i) same = value(x, i) == value(y, i);
  check(same, "a matrix under its GGUF name: row-major shape and values");
}

void damaged(const std::string& path) {
  const auto original = read(path);
  auto reject = [&](const std::vector<uint8_t>& data, const std::string& what) {
    ModelFile::Contents c;
    std::string error;
    const bool ok = modelfile::read_gguf(data.data(), data.size(), "fixture", &c, &error);
    check(!ok && !error.empty() && c.tensors.empty() && c.owned.empty(), "reject " + what + (ok ? "" : ": " + error));
  };
  for (size_t length : {size_t(0), size_t(3), size_t(7), size_t(23), original.size() / 2, original.size() - 1})
    reject({original.begin(), original.begin() + length}, "truncation at " + std::to_string(length));
  for (uint32_t ver : {1u, 4u, 0x03000000u}) {
    auto data = original;
    patch(&data, 4, ver);
    reject(data, "version/byte order");
  }
  for (size_t at : {size_t(8), size_t(16), size_t(24)}) {
    auto data = original;
    patch(&data, at, UINT64_MAX);
    reject(data, "count/string overflow");
  }
  const std::string name = "token_embd.weight";
  const size_t info = locate(original, name) + name.size();
  uint32_t rank;
  std::memcpy(&rank, original.data() + info, 4);
  for (auto [at, value] : std::vector<std::pair<size_t, uint64_t>>{
           {info + 4, 0}, {info + 4, UINT64_MAX}, {info + 4 + rank * 8 + 4, UINT64_MAX - 31}, {info + 4 + rank * 8 + 4, 1}}) {
    auto data = original;
    patch(&data, at, value);
    reject(data, "tensor shape/offset");
  }
  {
    auto data = original;
    patch(&data, info, uint32_t(5));
    reject(data, "rank");
  }
  {
    auto data = original;
    patch(&data, info + 4 + rank * 8, uint32_t(999));
    reject(data, "unknown tensor type");
  }
  {
    auto data = original;
    patch(&data, info + 4 + rank * 8 + 4, uint64_t(0));
    reject(data, "overlapping tensor ranges");
  }
  const size_t tokens = locate(original, "tokenizer.ggml.tokens") + std::strlen("tokenizer.ggml.tokens") + 4;
  {
    auto data = original;
    patch(&data, tokens + 4, UINT64_MAX);
    reject(data, "array size overflow");
  }
  {
    auto data = original;
    patch(&data, tokens, uint32_t(99));
    reject(data, "unknown array element type");
  }
  const std::string nfc = "tokenizer.ggml.normalizer.nfc";
  {
    auto data = original;
    data[locate(original, nfc) + nfc.size() + 4] = 2;
    reject(data, "invalid boolean");
  }
  {
    auto data = original;
    data[locate(data, "blk.0.attn_k.weight") + 11] = 'q';
    reject(data, "duplicate tensor name");
  }
  ModelFile::Contents c;
  std::string error;
  auto data = original;
  patch(&data, 4, uint32_t(2));
  check(modelfile::read_gguf(data.data(), data.size(), "v2", &c, &error), "GGUF version 2");
  check(data.size() == original.size() && std::equal(data.begin() + 8, data.end(), original.begin() + 8),
        "the reader leaves the mapping unchanged");
  // An architecture this build has no model for is the reader's to read all the same.
  data = original;
  std::memcpy(data.data() + locate(data, "general.architecture") + std::strlen("general.architecture") + 12, "llama", 5);
  check(modelfile::read_gguf(data.data(), data.size(), "arch", &c, &error) && c.arch == "llama",
        "any architecture name is read");
}

void quantization_version(const std::string& path) {
  auto data = read(path);
  const std::string key = "general.quantization_version";
  patch(&data, locate(data, key) + key.size() + 4, uint32_t(3));
  ModelFile::Contents c;
  std::string error;
  check(!modelfile::read_gguf(data.data(), data.size(), "quant", &c, &error) &&
            error.find("quantization version") != std::string::npos,
        "reject another GGUF quantization version");
}

void split_gguf(const std::string& dir) {
  namespace fs = std::filesystem;
  const std::string single = dir + "/tiny_gguf.gguf";
  const std::string shard1 = dir + "/tiny_split-00001-of-00003.gguf";
  const std::string shard2 = dir + "/tiny_split-00002-of-00003.gguf";
  const std::string shard3 = dir + "/tiny_split-00003-of-00003.gguf";
  const std::string q8_single = dir + "/tiny_q8.gguf";
  const std::string q8_shard1 = dir + "/tiny_q8_split-00001-of-00002.gguf";

  ModelFile ref, s1, s2;
  std::string err;
  check(ref.open(single, &err), "open single GGUF: " + err);
  check(s1.open(shard1, &err), "open split GGUF from shard 1: " + err);
  check(s2.open(shard2, &err), "open split GGUF from shard 2: " + err);
  if (s1.arch().empty() || s2.arch().empty()) return;

  check(s1.header().at("source").at("shards") == 3 && s2.header().at("source").at("shards") == 3,
        "split GGUF records shard count");
  check(s1.tensor_names().size() == ref.tensor_names().size() && s2.tensor_names().size() == ref.tensor_names().size(),
        "split GGUF exposes all tensors across shards");
  bool exact = true;
  for (const std::string& name : ref.tensor_names()) {
    const auto* a = ref.tensor(name);
    const auto* b = s1.tensor(name);
    const auto* c = s2.tensor(name);
    if (!a || !b || !c || a->dtype != b->dtype || a->nbytes != b->nbytes || a->ndim != b->ndim ||
        std::string(b->name) != name || std::string(c->name) != name ||
        std::memcmp(a->data, b->data, a->nbytes) != 0 || std::memcmp(a->data, c->data, a->nbytes) != 0) {
      exact = false;
      break;
    }
  }
  check(exact, "split GGUF tensor bytes, dtypes, shapes, and names match single file");
  const uint64_t expected_bytes = fs::file_size(os::to_path(shard1)) + fs::file_size(os::to_path(shard2)) +
                                  fs::file_size(os::to_path(shard3));
  check(s1.bytes() == expected_bytes && s2.bytes() == expected_bytes, "split GGUF bytes() sums all shards");
  std::atomic<bool> stop{false};
  s1.read_ahead(stop);  // exercises multi-shard read_ahead without crashing
  check(gguf_peek_mtp_layers(shard1) == gguf_peek_mtp_layers(single) &&
            gguf_peek_mtp_layers(shard2) == gguf_peek_mtp_layers(single),
        "gguf_peek_mtp_layers works on shard 1 and shard 2");

  // Quantized split GGUF
  ModelFile qref, qs1;
  check(qref.open(q8_single, &err) && qs1.open(q8_shard1, &err), "open quantized split GGUF: " + err);
  bool qexact = qs1.tensor_names().size() == qref.tensor_names().size();
  for (const std::string& name : qref.tensor_names()) {
    const auto* a = qref.tensor(name);
    const auto* b = qs1.tensor(name);
    if (!a || !b || a->dtype != b->dtype || a->nbytes != b->nbytes || std::memcmp(a->data, b->data, a->nbytes) != 0)
      qexact = false;
  }
  check(qexact, "quantized split GGUF matches single quantized GGUF");

  // Failure modes in an isolated temp directory
  const fs::path tmp = fs::temp_directory_path() / ("flyx-split-gguf-test-" + std::to_string(os::process_id()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp, ec);
  auto cleanup = [&] { fs::remove_all(tmp, ec); };
  try {
    // Missing shard 3
    fs::copy_file(os::to_path(shard1), tmp / "m-00001-of-00003.gguf", fs::copy_options::overwrite_existing);
    fs::copy_file(os::to_path(shard2), tmp / "m-00002-of-00003.gguf", fs::copy_options::overwrite_existing);
    ModelFile bad;
    err.clear();
    check(!bad.open(os::from_path(tmp / "m-00001-of-00003.gguf"), &err) && err.find("missing split") != std::string::npos,
          "reject missing split shard: " + err);

    // Non-split filename for a split shard
    fs::copy_file(os::to_path(shard1), tmp / "wrong_name.gguf", fs::copy_options::overwrite_existing);
    err.clear();
    check(!bad.open(os::from_path(tmp / "wrong_name.gguf"), &err) && err.find("-XXXXX-of-YYYYY") != std::string::npos,
          "reject split GGUF with non-shard filename: " + err);

    // Mismatched split.no (put shard 2 bytes at shard 3 path)
    fs::copy_file(os::to_path(shard2), tmp / "m-00003-of-00003.gguf", fs::copy_options::overwrite_existing);
    err.clear();
    check(!bad.open(os::from_path(tmp / "m-00001-of-00003.gguf"), &err) && err.find("split.no") != std::string::npos,
          "reject mismatched split.no: " + err);

    // Mismatched filename total count
    fs::copy_file(os::to_path(shard1), tmp / "badcnt-00001-of-00002.gguf", fs::copy_options::overwrite_existing);
    err.clear();
    check(!bad.open(os::from_path(tmp / "badcnt-00001-of-00002.gguf"), &err) &&
              err.find("header says 3") != std::string::npos,
          "reject split.count mismatch with filename: " + err);
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
}

// A GGUF file of float tensors, as a projector's converter writes one: its metadata (strings, split.no and
// split.count as u16, split.tensors.count as i32) and its tensors, row-major shapes given.
class GgufBytes {
 public:
  void str(const std::string& key, const std::string& v) {
    this->key(key, 8);
    u64(v.size());
    bytes(v);
  }
  void u16(const std::string& key, uint16_t v) {
    this->key(key, 2);
    raw(&v, 2);
  }
  void i32(const std::string& key, int32_t v) {
    this->key(key, 5);
    raw(&v, 4);
  }
  void tensor(const std::string& name, const std::vector<uint64_t>& shape, const std::vector<float>& values) {
    tensors_.push_back({name, shape, values});
  }
  void write(const std::filesystem::path& path) const {
    std::vector<uint8_t> out = {'G', 'G', 'U', 'F'};
    auto put = [&](const void* p, size_t n) { out.insert(out.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); };
    const uint32_t version = 3;
    const uint64_t n_tensors = tensors_.size(), n_kv = n_kv_;
    put(&version, 4);
    put(&n_tensors, 8);
    put(&n_kv, 8);
    out.insert(out.end(), kv_.begin(), kv_.end());
    uint64_t offset = 0;
    for (const auto& t : tensors_) {
      const uint64_t len = t.name.size();
      const uint32_t rank = uint32_t(t.shape.size()), f32 = 0;
      put(&len, 8);
      put(t.name.data(), t.name.size());
      put(&rank, 4);
      for (size_t d = t.shape.size(); d-- > 0;) put(&t.shape[d], 8);  // the fastest dimension first
      put(&f32, 4);
      put(&offset, 8);
      offset += (t.values.size() * 4 + 31) / 32 * 32;
    }
    out.resize((out.size() + 31) / 32 * 32);
    for (const auto& t : tensors_) {
      put(t.values.data(), t.values.size() * 4);
      out.resize((out.size() + 31) / 32 * 32);
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size()));
  }

 private:
  struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<float> values;
  };
  void raw(const void* p, size_t n) { kv_.insert(kv_.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); }
  void u64(uint64_t v) { raw(&v, 8); }
  void bytes(const std::string& v) { kv_.insert(kv_.end(), v.begin(), v.end()); }
  void key(const std::string& k, uint32_t type) {
    u64(k.size());
    bytes(k);
    raw(&type, 4);
    ++n_kv_;
  }
  std::vector<uint8_t> kv_;
  uint64_t n_kv_ = 0;
  std::vector<Tensor> tensors_;
};

// A vision projector (general.type mmproj) beside a model: one file, and the same tensors in two shards whose second
// says nothing of its type, or says the same; opened from either shard, each gives the one file's tensors.  Refused:
// a projector as the model, single or split, and shards of two types.
void projector_split(const std::string& dir) {
  namespace fs = std::filesystem;
  const fs::path tmp = fs::temp_directory_path() / ("flyx-mmproj-split-test-" + std::to_string(os::process_id()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp, ec);
  struct T {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<float> values;
  };
  const std::vector<T> tensors = {{"v.patch_embd.weight", {2, 3}, {0.5f, -1.25f, 2, 3.75f, -0.125f, 6}},
                                  {"v.blk.0.attn_q.weight", {4}, {1, 2, 4, 8}},
                                  {"mm.0.weight", {3, 2}, {-6, 5, -4, 3, -2, 1}}};
  // first..last of `tensors` into one file: its metadata, then split keys when it is one of `count` shards.
  auto write = [&](const std::string& file, size_t first, size_t last, const char* type, int no, int count) {
    GgufBytes g;
    g.str("general.architecture", "clip");
    if (type) g.str("general.type", type);
    if (count > 1) {
      g.u16("split.no", uint16_t(no));
      g.u16("split.count", uint16_t(count));
      g.i32("split.tensors.count", int32_t(tensors.size()));
    }
    for (size_t i = first; i < last; ++i) g.tensor(tensors[i].name, tensors[i].shape, tensors[i].values);
    g.write(tmp / file);
    return os::from_path(tmp / file);
  };
  const std::string single = write("mmproj.gguf", 0, 3, "mmproj", 0, 1);
  const std::string split1 = write("mmproj-00001-of-00002.gguf", 0, 2, "mmproj", 0, 2);
  const std::string split2 = write("mmproj-00002-of-00002.gguf", 2, 3, nullptr, 1, 2);       // says no type
  const std::string typed1 = write("typed-00001-of-00002.gguf", 0, 1, "mmproj", 0, 2);
  const std::string typed2 = write("typed-00002-of-00002.gguf", 1, 3, "mmproj", 1, 2);       // says the same
  const std::string mixed1 = write("mixed-00001-of-00002.gguf", 0, 2, "mmproj", 0, 2);
  const std::string mixed2 = write("mixed-00002-of-00002.gguf", 2, 3, "model", 1, 2);        // another type
  const std::string model1 = write("modelmix-00001-of-00002.gguf", 0, 2, nullptr, 0, 2);
  const std::string model2 = write("modelmix-00002-of-00002.gguf", 2, 3, "mmproj", 1, 2);   // a model's shard 1

  // The reader, as ModelFile::add calls it (no tokenizer): the names and the values, in order.
  auto contents = [](const std::string& path, std::vector<std::unique_ptr<os::MappedFile>>* maps,
                     ModelFile::Contents* c, std::string* err) {
    maps->clear();
    maps->push_back(std::make_unique<os::MappedFile>());
    return maps->back()->open(path, err) && modelfile::read_gguf(maps, path, c, err, /*need_tokenizer=*/false);
  };
  std::vector<std::unique_ptr<os::MappedFile>> one_maps;
  ModelFile::Contents one;
  std::string err;
  check(contents(single, &one_maps, &one, &err), "a single-file projector opens beside a model: " + err);
  bool values_ok = one.names.size() == tensors.size();
  for (const T& t : tensors) {
    const auto it = one.index.find(t.name);
    values_ok = values_ok && it != one.index.end() && one.tensors[it->second].nbytes == t.values.size() * 4 &&
                std::memcmp(one.tensors[it->second].data, t.values.data(), t.values.size() * 4) == 0;
  }
  check(values_ok, "the single-file projector's tensors are the ones written");
  for (const std::string& path : {split1, split2, typed1, typed2}) {
    std::vector<std::unique_ptr<os::MappedFile>> maps;
    ModelFile::Contents c;
    err.clear();
    const bool opened = contents(path, &maps, &c, &err);
    bool same = opened && c.names == one.names && c.header.at("source").at("shards") == 2 &&
                c.arch == one.arch && c.header.at("gguf").at("general.type") == "mmproj";
    for (size_t i = 0; same && i < one.tensors.size(); ++i) {
      const auto& a = one.tensors[i];
      const auto& b = c.tensors[c.index.at(one.names[i])];
      same = a.dtype == b.dtype && a.ndim == b.ndim && a.nbytes == b.nbytes &&
             std::equal(a.shape, a.shape + a.ndim, b.shape) && std::memcmp(a.data, b.data, a.nbytes) == 0;
    }
    check(same, "a split projector opened from " + fs::path(path).filename().string() +
                    " gives the single file's tensor names and values " + err);
  }
  // Through ModelFile, beside the tiny model: one companion per projector, its tensors under companion names.
  ModelFile mf;
  err.clear();
  bool added = mf.open(dir + "/tiny_gguf.gguf", &err) && mf.add(single, &err) && mf.add(split1, &err) &&
               mf.add(split2, &err) && mf.add(typed2, &err);
  check(added && mf.companions() == 4, "ModelFile::add takes a single and a split projector from either shard " + err);
  bool companions_same = added && mf.companions() == 4;
  for (int i = 1; companions_same && i < 4; ++i)
    for (const T& t : tensors) {
      const auto* a = mf.tensor(ModelFile::companion_name(0, t.name));
      const auto* b = mf.tensor(ModelFile::companion_name(i, t.name));
      companions_same = companions_same && a && b && a->nbytes == b->nbytes && std::memcmp(a->data, b->data, a->nbytes) == 0;
    }
  check(companions_same, "the split projector's companions hold the single file's tensors");

  // Refused: a projector as the model (single, split from either shard), and shards of two types.
  for (const std::string& path : {single, split1, split2}) {
    ModelFile primary;
    err.clear();
    const bool refused = !primary.open(path, &err) && err.find("multimodal projector") != std::string::npos;
    check(refused, "a projector is not the model (" + fs::path(path).filename().string() + "): " + err);
  }
  for (const std::string& path : {mixed1, mixed2, model1, model2}) {
    ModelFile beside;
    err.clear();
    const bool refused = beside.open(dir + "/tiny_gguf.gguf", &err) && !beside.add(path, &err) &&
                         err.find("general.type") != std::string::npos && err.find("does not match") != std::string::npos;
    check(refused, "shards of two types are refused (" + fs::path(path).filename().string() + "): " + err);
  }
  fs::remove_all(tmp, ec);
}

void companion_skips_tokenizer(const std::string& path) {
  const auto original = read(path);
  ModelFile::Contents c;
  std::string err;
  const bool ok = modelfile::read_gguf(original.data(), original.size(), path, &c, &err, /*need_tokenizer=*/false);
  check(ok && c.owned.empty() && c.blobs.empty() && !c.tensors.empty() && c.header.contains("gguf") &&
            c.header.at("gguf").contains("general.architecture"),
        "companion GGUF skips unused tokenizer blobs while keeping metadata and tensors");

  ModelFile mf;
  check(mf.open(path, &err) && mf.add(path, &err) && mf.companions() == 1 &&
            mf.tensor(ModelFile::companion_name(0, "token_embd.weight")) != nullptr &&
            mf.header().at("companions").at(0).contains("gguf") &&
            !mf.header().at("companions").at(0).contains("tokenizer"),
        "ModelFile::add loads companion GGUF tensors and metadata without building tokenizer tables");
}

void array_stride_bounds() {
  auto append_u32 = [](std::vector<uint8_t>* b, uint32_t v) {
    const size_t at = b->size();
    b->resize(at + 4);
    std::memcpy(b->data() + at, &v, 4);
  };
  auto append_u64 = [](std::vector<uint8_t>* b, uint64_t v) {
    const size_t at = b->size();
    b->resize(at + 8);
    std::memcpy(b->data() + at, &v, 8);
  };
  auto make_bad_array = [&](const std::string& key, uint32_t elem_type, uint64_t count, size_t trailing) {
    std::vector<uint8_t> b = {'G', 'G', 'U', 'F'};
    append_u32(&b, 3);  // version
    append_u64(&b, 1);  // tensors (non-zero so header count check passes into KV parsing)
    append_u64(&b, 1);  // kv pairs
    append_u64(&b, key.size());
    b.insert(b.end(), key.begin(), key.end());
    append_u32(&b, 9);          // GGUF_TYPE_ARRAY
    append_u32(&b, elem_type);  // element type
    append_u64(&b, count);      // element count (> remaining / elem_size, but <= remaining)
    b.resize(b.size() + trailing, 0);
    return b;
  };
  ModelFile::Contents c;
  std::string err;
  // 100 strings need >= 800 bytes (8-byte length prefix each); 120 trailing bytes must reject before reserve().
  const auto bad_toks = make_bad_array("tokenizer.ggml.tokens", 8, 100, 120);
  bool ok = modelfile::read_gguf(bad_toks.data(), bad_toks.size(), "bad_toks", &c, &err);
  check(!ok && err.find("invalid GGUF array type or length") != std::string::npos,
        "reject tokenizer.ggml.tokens array exceeding remaining/8 before reserve: " + err);
  // 100 int32s need 400 bytes; 120 trailing bytes must reject before reserve().
  const auto bad_types = make_bad_array("tokenizer.ggml.token_type", 5, 100, 120);
  err.clear();
  ok = modelfile::read_gguf(bad_types.data(), bad_types.size(), "bad_types", &c, &err);
  check(!ok && err.find("invalid GGUF array type or length") != std::string::npos,
        "reject tokenizer.ggml.token_type array exceeding remaining/4 before reserve: " + err);
}

void dflash_gguf_peek_and_open(const std::string& dir) {
  const std::string model = dir + "/tiny_gguf.gguf", drafter = dir + "/tiny_draft.gguf";
  const GgufPeek pm = gguf_peek(model);
  check(pm.valid && pm.arch == "qwen35" && pm.mtp_layers == 1, "gguf_peek reports arch and mtp_layers on target GGUF");
  const GgufPeek pd = gguf_peek(drafter);
  check(pd.valid && pd.arch == "dflash" && pd.mtp_layers == 0, "gguf_peek identifies arch='dflash' drafter GGUF");
  ModelFile f;
  std::string err;
  const bool opened = f.open(drafter, &err);
  check(!opened && err.find("a drafter alone, not a model") != std::string::npos,
        "standalone arch='dflash' GGUF refused by ModelFile::open (" + err + ")");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string dir = argv[1];
  same_file(dir + "/tiny.flyx", dir + "/tiny_gguf.gguf", "qwen35");
  same_file(dir + "/tiny.flyx", dir + "/tiny_bf16.gguf", "qwen35");
  same_file(dir + "/tiny_qwen3.flyx", dir + "/tiny_qwen3_gguf.gguf", "qwen3");
  damaged(dir + "/tiny_qwen3_gguf.gguf");
  quantization_version(dir + "/quant_Q4_K.gguf");
  split_gguf(dir);
  companion_skips_tokenizer(dir + "/tiny_gguf.gguf");
  projector_split(dir);
  array_stride_bounds();
  dflash_gguf_peek_and_open(dir);
  std::printf("%d failures\n", failures);
  return failures ? 1 : 0;
}
