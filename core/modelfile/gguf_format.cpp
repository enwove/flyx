// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GGUF (little-endian, v2 and v3), the container only: its metadata becomes
// header["gguf"], its byte-level BPE vocabulary the tokenizer tables, its
// tensors keep their names, with shapes turned row-major and types named as
// backend_abi.h names them.  Which tensor is which weight of an architecture,
// and how that architecture's GGUF lays them out, is for models/<arch>;
// nothing here knows a model.
#include "core/modelfile/readers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>

#include "core/os/path.h"

namespace flyx::modelfile {
namespace {

using json = nlohmann::json;

[[noreturn]] void invalid(const std::string& message) { throw std::runtime_error(message); }

// The header, read front to back with every length checked against the file.
class Cursor {
 public:
  Cursor(const uint8_t* data, size_t size) : data_(data), size_(size) {}
  size_t position() const { return position_; }
  size_t remaining() const { return size_ - position_; }
  template <class T>
  T read() {
    if (remaining() < sizeof(T)) invalid("truncated GGUF header at byte " + std::to_string(position_));
    T v;
    std::memcpy(&v, data_ + position_, sizeof(T));
    position_ += sizeof(T);
    return v;
  }
  std::string_view view(size_t limit = std::numeric_limits<size_t>::max()) {
    const uint64_t n = read<uint64_t>();
    if (n > remaining() || n > limit) invalid("invalid GGUF string length at byte " + std::to_string(position_));
    std::string_view result(reinterpret_cast<const char*>(data_ + position_), size_t(n));
    position_ += size_t(n);
    return result;
  }
  std::string string(size_t limit = std::numeric_limits<size_t>::max()) { return std::string(view(limit)); }
  static bool is_int_type(uint32_t type) {
    return type <= 5 || type == 10 || type == 11;
  }
  int64_t read_int(uint32_t type) {
    switch (type) {
      case 0: return int64_t(read<uint8_t>());
      case 1: return int64_t(read<int8_t>());
      case 2: return int64_t(read<uint16_t>());
      case 3: return int64_t(read<int16_t>());
      case 4: return int64_t(read<uint32_t>());
      case 5: return int64_t(read<int32_t>());
      case 10: {
        const uint64_t u = read<uint64_t>();
        if (u > uint64_t(INT64_MAX)) invalid("GGUF token type: expected an integer in range");
        return int64_t(u);
      }
      case 11: return read<int64_t>();
      default: invalid("GGUF token type: expected an integer in range");
    }
  }
  static size_t min_elem_bytes(uint32_t element) {
    static constexpr size_t kMin[13] = {1, 1, 2, 2, 4, 4, 4, 1, 8, 12, 8, 8, 8};
    return element <= 12 ? kMin[element] : 0;
  }
  void skip_array(uint32_t element, uint64_t n, int depth) {
    if (depth >= 8) invalid("GGUF metadata arrays nested too deeply");
    const size_t min_sz = min_elem_bytes(element);
    if (!min_sz || n > remaining() / min_sz) invalid("invalid GGUF array type or length");
    static constexpr size_t kIntSize[13] = {1, 1, 2, 2, 4, 4, 0, 0, 0, 0, 8, 8, 0};
    if (const size_t sz = kIntSize[element]) {
      position_ += size_t(n) * sz;
      return;
    }
    switch (element) {
      case 6:
        for (uint64_t i = 0; i < n; ++i) (void)finite(read<float>());
        return;
      case 7:
        for (uint64_t i = 0; i < n; ++i)
          if (read<uint8_t>() > 1) invalid("invalid GGUF boolean");
        return;
      case 8:
        for (uint64_t i = 0; i < n; ++i) (void)view();
        return;
      case 9:
        for (uint64_t i = 0; i < n; ++i) {
          const uint32_t sub_el = read<uint32_t>();
          const uint64_t sub_n = read<uint64_t>();
          skip_array(sub_el, sub_n, depth + 1);
        }
        return;
      case 12:
        for (uint64_t i = 0; i < n; ++i) (void)finite(read<double>());
        return;
      default: invalid("unsupported GGUF metadata type " + std::to_string(element));
    }
  }
  void skip_value(uint32_t type, int depth = 0) {
    switch (type) {
      case 0: (void)read<uint8_t>(); return;
      case 1: (void)read<int8_t>(); return;
      case 2: (void)read<uint16_t>(); return;
      case 3: (void)read<int16_t>(); return;
      case 4: (void)read<uint32_t>(); return;
      case 5: (void)read<int32_t>(); return;
      case 6: (void)finite(read<float>()); return;
      case 7:
        if (read<uint8_t>() > 1) invalid("invalid GGUF boolean");
        return;
      case 8: (void)view(); return;
      case 9: {
        const uint32_t element = read<uint32_t>();
        const uint64_t n = read<uint64_t>();
        skip_array(element, n, depth);
        return;
      }
      case 10: (void)read<uint64_t>(); return;
      case 11: (void)read<int64_t>(); return;
      case 12: (void)finite(read<double>()); return;
      default: invalid("unsupported GGUF metadata type " + std::to_string(type));
    }
  }
  // A metadata value of GGUF type `type` (0..12; 9 is an array).
  json value(uint32_t type, int depth = 0) {
    switch (type) {
      case 0: return read<uint8_t>();
      case 1: return read<int8_t>();
      case 2: return read<uint16_t>();
      case 3: return read<int16_t>();
      case 4: return read<uint32_t>();
      case 5: return read<int32_t>();
      case 6: return finite(read<float>());
      case 7: {
        const uint8_t b = read<uint8_t>();
        if (b > 1) invalid("invalid GGUF boolean");
        return b != 0;
      }
      case 8: return string();
      case 9: {
        if (depth >= 8) invalid("GGUF metadata arrays nested too deeply");
        const uint32_t element = read<uint32_t>();
        const uint64_t n = read<uint64_t>();
        const size_t min_sz = min_elem_bytes(element);
        if (!min_sz || n > remaining() / min_sz) invalid("invalid GGUF array type or length");
        json result = json::array();
        for (uint64_t i = 0; i < n; ++i) result.push_back(value(element, depth + 1));
        return result;
      }
      case 10: return read<uint64_t>();
      case 11: return read<int64_t>();
      case 12: return finite(read<double>());
      default: invalid("unsupported GGUF metadata type " + std::to_string(type));
    }
  }

 private:
  static double finite(double v) {
    if (!std::isfinite(v)) invalid("non-finite GGUF metadata number");
    return v;
  }
  const uint8_t* data_;
  size_t size_, position_ = 0;
};

int64_t integer(const json& v, const std::string& name, int64_t minimum, int64_t maximum) {
  if (!v.is_number_integer() || (v.is_number_unsigned() && v.get<uint64_t>() > uint64_t(maximum)))
    invalid("GGUF " + name + ": expected an integer in range");
  const int64_t n = v.get<int64_t>();
  if (n < minimum || n > maximum) invalid("GGUF " + name + ": integer out of range");
  return n;
}

// GGML's tensor type numbers -> ours; -1 for the ones no backend reads.
int dtype_of(uint32_t ggml_type) {
  switch (ggml_type) {
    case 0: return FLYX_DT_F32;
    case 1: return FLYX_DT_F16;
    case 2: return FLYX_DT_GGML_Q4_0;
    case 3: return FLYX_DT_GGML_Q4_1;
    case 6: return FLYX_DT_GGML_Q5_0;
    case 7: return FLYX_DT_GGML_Q5_1;
    case 8: return FLYX_DT_GGML_Q8_0;
    case 10: return FLYX_DT_GGML_Q2_K;
    case 11: return FLYX_DT_GGML_Q3_K;
    case 12: return FLYX_DT_GGML_Q4_K;
    case 13: return FLYX_DT_GGML_Q5_K;
    case 14: return FLYX_DT_GGML_Q6_K;
    case 16: return FLYX_DT_GGML_IQ2_XXS;
    case 17: return FLYX_DT_GGML_IQ2_XS;
    case 18: return FLYX_DT_GGML_IQ3_XXS;
    case 19: return FLYX_DT_GGML_IQ1_S;
    case 20: return FLYX_DT_GGML_IQ4_NL;
    case 21: return FLYX_DT_GGML_IQ3_S;
    case 22: return FLYX_DT_GGML_IQ2_S;
    case 23: return FLYX_DT_GGML_IQ4_XS;
    case 28: return FLYX_DT_F64;
    case 29: return FLYX_DT_GGML_IQ1_M;
    case 30: return FLYX_DT_BF16;
    case 39: return FLYX_DT_GGML_MXFP4;
    case 40: return FLYX_DT_GGML_NVFP4;
    case 42: return FLYX_DT_GGML_Q2_0;
    default: return -1;
  }
}

void add_blob(ModelFile::Contents* out, const std::string& name, const std::string& bytes) {
  auto& buffer = out->owned[name];
  buffer.assign(bytes.begin(), bytes.end());
  out->blobs[name] = std::string_view(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// Zero-copy views of the large GGUF tokenizer arrays parsed directly out of the mapped file
// without building hundreds of thousands of transient nlohmann::json nodes.
struct RawTokenizer {
  bool seen_tokens = false, seen_types = false, seen_merges = false, seen_scores = false;
  bool has_tokens = false, has_types = false, has_merges = false;
  std::vector<std::string_view> tokens;
  std::vector<int32_t> types;
  std::vector<std::string_view> merges;
};

constexpr std::array<int16_t, 324> make_gpt2_byte_table() {
  std::array<int16_t, 324> table{};
  for (auto& v : table) v = -1;
  int extra = 256;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    table[size_t(printable ? b : extra++)] = int16_t(b);
  }
  return table;
}

constexpr std::array<int16_t, 324> kGpt2ByteTable = make_gpt2_byte_table();

// Appends a vocabulary entry of a byte-level BPE as its raw bytes: GGUF stores it in
// GPT-2's printable alphabet, one Unicode scalar (code points 33..323, 1 or 2 UTF-8 bytes) per byte.
void append_token_bytes(std::string_view encoded, std::vector<uint8_t>* out) {
  const auto* p = reinterpret_cast<const uint8_t*>(encoded.data());
  const size_t n = encoded.size();
  for (size_t i = 0; i < n;) {
    const uint8_t c0 = p[i];
    int cp = -1;
    if (c0 < 0x80) {
      cp = c0;
      ++i;
    } else if ((c0 & 0xE0) == 0xC0 && i + 1 < n) {
      const uint8_t c1 = p[i + 1];
      if ((c1 & 0xC0) != 0x80) invalid("GGUF token is outside the GPT-2 byte alphabet");
      cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
      if (cp < 0x80) invalid("GGUF token is outside the GPT-2 byte alphabet");
      i += 2;
    } else {
      invalid("GGUF token is outside the GPT-2 byte alphabet");
    }
    if (cp < 0 || cp >= int(kGpt2ByteTable.size()) || kGpt2ByteTable[size_t(cp)] < 0)
      invalid("GGUF token is outside the GPT-2 byte alphabet");
    out->push_back(static_cast<uint8_t>(kGpt2ByteTable[size_t(cp)]));
  }
}

void write_u32(uint8_t* dst, uint32_t v) { std::memcpy(dst, &v, 4); }

// The tokenizer tables core/tokenizer reads (the layout python/convert
// writes), the stop tokens and the chat template, from GGUF's tokenizer keys.
void tokenizer(const json& meta, const RawTokenizer& raw, ModelFile::Contents* out) {
  if (meta.value("tokenizer.ggml.model", "") != "gpt2") invalid("unsupported GGUF tokenizer model (need gpt2 BPE)");
  for (const char* key : {"tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token",
                          "tokenizer.ggml.add_prefix_space", "tokenizer.ggml.clean_spaces"})
    if (meta.value(key, false)) invalid(std::string("unsupported GGUF tokenizer option ") + key);
  if (!raw.seen_tokens || !raw.seen_types || !raw.seen_merges)
    invalid("GGUF tokenizer without tokens, token types or merges");
  if (!raw.has_tokens || raw.tokens.empty() || !raw.has_types || raw.types.size() != raw.tokens.size())
    invalid("invalid GGUF tokenizer vocabulary/types");
  if (!raw.has_merges) invalid("invalid GGUF BPE merges");

  const int n = int(raw.tokens.size());
  // Flat open-addressed string_view -> token id table (single allocation, stores hash32 + id).
  size_t id_cap = 16;
  while (id_cap < size_t(n) * 2) id_cap <<= 1;
  struct IdSlot {
    uint32_t hash = 0;
    uint32_t id = UINT32_MAX;
  };
  std::vector<IdSlot> id_table(id_cap);
  const size_t id_mask = id_cap - 1;
  auto hash_sv = [](std::string_view sv) -> uint64_t {
    uint64_t h = 14695981039346656037ULL;
    for (unsigned char c : sv) {
      h ^= c;
      h *= 1099511628211ULL;
    }
    h ^= h >> 33;
    return h;
  };
  auto insert_id = [&](std::string_view sv, uint32_t id) -> bool {
    const uint64_t h = hash_sv(sv);
    const uint32_t tag = static_cast<uint32_t>(h ^ (h >> 32));
    size_t slot = size_t(h) & id_mask;
    for (;;) {
      IdSlot& e = id_table[slot];
      if (e.id == UINT32_MAX) {
        e = {tag, id};
        return true;
      }
      if (e.hash == tag && raw.tokens[e.id] == sv) return false;
      slot = (slot + 1) & id_mask;
    }
  };
  auto find_id = [&](std::string_view sv) -> uint32_t {
    const uint64_t h = hash_sv(sv);
    const uint32_t tag = static_cast<uint32_t>(h ^ (h >> 32));
    size_t slot = size_t(h) & id_mask;
    for (;;) {
      const IdSlot& e = id_table[slot];
      if (e.id == UINT32_MAX) return UINT32_MAX;
      if (e.hash == tag && raw.tokens[e.id] == sv) return e.id;
      slot = (slot + 1) & id_mask;
    }
  };
  json added = json::array();

  auto& vocab_buf = out->owned["tokenizer.vocab"];
  const size_t offsets_bytes = (size_t(n) + 1) * 4;
  vocab_buf.reserve(offsets_bytes + size_t(n) * 8);
  vocab_buf.resize(offsets_bytes, 0);
  for (int i = 0; i < n; ++i) {
    const std::string_view s = raw.tokens[size_t(i)];
    if (!insert_id(s, uint32_t(i))) invalid("duplicate GGUF tokenizer token");
    // GGUF token types: 1 normal, 3 control (special), 4 user defined (added, not special), 6 byte.
    const int type = raw.types[size_t(i)];
    const bool is_added = type != 1 && type != 6;
    if (is_added) {
      vocab_buf.insert(vocab_buf.end(), reinterpret_cast<const uint8_t*>(s.data()),
                       reinterpret_cast<const uint8_t*>(s.data() + s.size()));
      added.push_back({{"id", i}, {"content", std::string(s)}, {"special", type != 4}});
    } else {
      append_token_bytes(s, &vocab_buf);
    }
    const size_t data_bytes = vocab_buf.size() - offsets_bytes;
    if (data_bytes > UINT32_MAX) invalid("GGUF tokenizer byte table is too large");
    write_u32(vocab_buf.data() + size_t(i + 1) * 4, uint32_t(data_bytes));
  }
  out->blobs["tokenizer.vocab"] = std::string_view(reinterpret_cast<const char*>(vocab_buf.data()), vocab_buf.size());

  const size_t n_merges = raw.merges.size();
  auto& merges_buf = out->owned["tokenizer.merges"];
  merges_buf.resize(n_merges * 12);
  size_t cap = 16;
  while (cap < n_merges * 2) cap <<= 1;
  std::vector<uint64_t> seen(cap, UINT64_MAX);
  const size_t mask = cap - 1;
  std::string joined;
  joined.reserve(64);
  for (size_t idx = 0; idx < n_merges; ++idx) {
    const std::string_view s = raw.merges[idx];
    const auto at = s.find(' ');
    if (at == std::string_view::npos || at == 0 || at + 1 == s.size()) invalid("invalid GGUF BPE merge");
    const std::string_view left = s.substr(0, at);
    const std::string_view right = s.substr(at + 1);
    joined.clear();
    joined.append(left);
    joined.append(right);
    const uint32_t a = find_id(left), b = find_id(right), c = find_id(std::string_view(joined));
    if (a == UINT32_MAX || b == UINT32_MAX || c == UINT32_MAX)
      invalid("GGUF BPE merge has missing tokens or duplicates");
    const uint64_t key = (uint64_t(a) << 32) | uint64_t(b);
    uint64_t h = key * 0x9e3779b97f4a7c15ULL;
    h ^= h >> 30;
    size_t slot = size_t(h) & mask;
    for (;;) {
      const uint64_t cur = seen[slot];
      if (cur == UINT64_MAX) {
        seen[slot] = key;
        break;
      }
      if (cur == key) invalid("GGUF BPE merge has missing tokens or duplicates");
      slot = (slot + 1) & mask;
    }
    uint8_t* dst = merges_buf.data() + idx * 12;
    write_u32(dst, a);
    write_u32(dst + 4, b);
    write_u32(dst + 8, c);
  }
  out->blobs["tokenizer.merges"] =
      std::string_view(reinterpret_cast<const char*>(merges_buf.data()), merges_buf.size());
  const std::string pre = meta.value("tokenizer.ggml.pre", "");
  const bool is_inkling2 = pre == "inkling2";
  const bool is_gpt4o = pre == "gpt-4o";
  out->header["tokenizer"] = {{"pretokenizer", pre},
                              {"normalizer", meta.value("tokenizer.ggml.normalizer.nfc", !(is_inkling2 || is_gpt4o)) ? "nfc" : "none"},
                              {"ignore_merges", meta.value("tokenizer.ggml.ignore_merges", is_inkling2)},
                              {"n_tokens", n},
                              {"n_merges", n_merges}};
  add_blob(out, "tokenizer.added", added.dump());

  // Where generation ends: the ids the file names, and ChatML's turn end and
  // GPT-2's text end when the vocabulary has them (files often name only
  // one; llama.cpp's loader adds the well-known ones by their text too).
  json eos = json::array();
  auto stop = [&](int id) {
    if (std::find(eos.begin(), eos.end(), json(id)) == eos.end()) eos.push_back(id);
  };
  for (std::string_view s : {"<|im_end|>", "<|endoftext|>"})
    if (const uint32_t id = find_id(s); id != UINT32_MAX) stop(int(id));
  for (const char* kind : {"eos", "eot", "eom", "bos"}) {
    const std::string key = std::string("tokenizer.ggml.") + kind + "_token_id";
    if (!meta.contains(key)) continue;
    const int id = int(integer(meta.at(key), key, 0, n - 1));
    const std::string k = kind;
    if (k != "bos") stop(id);
    if (k == "bos" || k == "eos") out->header["chat"][k + "_token"] = std::string(raw.tokens[size_t(id)]);
  }
  if (eos.empty()) invalid("GGUF has no end-of-sequence token");
  out->header["generation"] = {{"eos_ids", eos}};
  if (meta.contains("tokenizer.chat_template") && meta.at("tokenizer.chat_template").is_string())
    add_blob(out, "chat_template.jinja", meta.at("tokenizer.chat_template").get<std::string>());
}

struct SplitPath {
  std::string prefix;  // full path before "-XXXXX-of-YYYYY.gguf"
  std::string ext;     // ".gguf" (original case)
  int part = 0;        // 1-based shard index
  int count = 0;       // total shard count (>= 2)
  int digits = 5;      // width of zero-padded shard numbers
};

bool parse_split_suffix(const std::string& path, SplitPath* out) {
  if (path.size() < 18) return false;  // minimum: "-00001-of-00002.gguf" minus 1 char prefix
  const std::string ext = path.substr(path.size() - 5);
  std::string low_ext = ext;
  for (char& c : low_ext) c = char(std::tolower(static_cast<unsigned char>(c)));
  if (low_ext != ".gguf") return false;
  const std::string stem = path.substr(0, path.size() - 5);
  const size_t slash = stem.find_last_of("/\\");
  const size_t base_start = slash == std::string::npos ? 0 : slash + 1;
  const size_t of = stem.rfind("-of-");
  if (of == std::string::npos || of <= base_start) return false;
  const size_t count_start = of + 4;
  const size_t digits = stem.size() - count_start;
  if (digits < 5 || digits > 9 || of < base_start + digits + 1 || stem[of - digits - 1] != '-') return false;
  auto parse_digits = [&](size_t from, int* v) {
    int n = 0;
    for (size_t i = 0; i < digits; ++i) {
      const char c = stem[from + i];
      if (c < '0' || c > '9') return false;
      n = n * 10 + (c - '0');
    }
    *v = n;
    return true;
  };
  int part = 0, count = 0;
  if (!parse_digits(of - digits, &part) || !parse_digits(count_start, &count)) return false;
  if (count < 2 || part < 1 || part > count) return false;
  out->prefix = stem.substr(0, of - digits - 1);
  out->ext = ext;
  out->part = part;
  out->count = count;
  out->digits = int(digits);
  return true;
}

std::string shard_path(const SplitPath& sp, int index_0based) {
  auto pad = [&](int n) {
    std::string s = std::to_string(n);
    if (int(s.size()) < sp.digits) s.insert(0, size_t(sp.digits - int(s.size())), '0');
    return s;
  };
  return sp.prefix + "-" + pad(index_0based + 1) + "-of-" + pad(sp.count) + sp.ext;
}

struct ShardHeader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  size_t alignment = 32;
  json meta = json::object();
  RawTokenizer tok{};
  size_t tensor_pos = 0;
};

ShardHeader parse_shard_header(const uint8_t* data, size_t size, bool allow_zero_tensors, bool need_tokenizer) {
  if constexpr (std::endian::native != std::endian::little) invalid("the GGUF reader needs a little-endian host");
  Cursor r(data, size);
  if (r.read<uint32_t>() != 0x46554747) invalid("not a GGUF file");
  ShardHeader h;
  h.version = r.read<uint32_t>();
  if (h.version != 2 && h.version != 3)
    invalid("unsupported GGUF version or byte order (need little-endian v2 or v3)");
  h.n_tensors = r.read<uint64_t>();
  const uint64_t n_keys = r.read<uint64_t>();
  if ((!allow_zero_tensors && !h.n_tensors) || h.n_tensors > r.remaining() / 24 || n_keys > r.remaining() / 12)
    invalid("invalid GGUF tensor/metadata counts");
  auto read_str_array = [&](bool* seen, bool* valid, std::vector<std::string_view>* dst, uint32_t type) {
    if (*seen) invalid("empty or duplicate GGUF metadata key");
    *seen = true;
    if (!need_tokenizer || type != 9) {
      r.skip_value(type);
      return;
    }
    const uint32_t element = r.read<uint32_t>();
    const uint64_t n = r.read<uint64_t>();
    const size_t min_sz = Cursor::min_elem_bytes(element);
    if (!min_sz || n > r.remaining() / min_sz) invalid("invalid GGUF array type or length");
    if (element == 8 && n <= size_t(INT_MAX)) {
      dst->reserve(size_t(n));
      for (uint64_t j = 0; j < n; ++j) dst->push_back(r.view());
      *valid = true;
    } else {
      r.skip_array(element, n, 0);
    }
  };
  for (uint64_t i = 0; i < n_keys; ++i) {
    const std::string_view key = r.view(65535);
    if (key.empty() || key.find('\0') != std::string_view::npos) invalid("empty or duplicate GGUF metadata key");
    const uint32_t type = r.read<uint32_t>();
    if (key == "tokenizer.ggml.tokens") {
      read_str_array(&h.tok.seen_tokens, &h.tok.has_tokens, &h.tok.tokens, type);
    } else if (key == "tokenizer.ggml.merges") {
      read_str_array(&h.tok.seen_merges, &h.tok.has_merges, &h.tok.merges, type);
    } else if (key == "tokenizer.ggml.token_type") {
      if (h.tok.seen_types) invalid("empty or duplicate GGUF metadata key");
      h.tok.seen_types = true;
      if (!need_tokenizer || type != 9) {
        r.skip_value(type);
      } else {
        const uint32_t element = r.read<uint32_t>();
        const uint64_t n = r.read<uint64_t>();
        const size_t min_sz = Cursor::min_elem_bytes(element);
        if (!min_sz || n > r.remaining() / min_sz) invalid("invalid GGUF array type or length");
        if (Cursor::is_int_type(element) && n <= size_t(INT_MAX)) {
          h.tok.types.reserve(size_t(n));
          for (uint64_t j = 0; j < n; ++j) {
            const int64_t v = r.read_int(element);
            if (v < 1 || v > 6) invalid("GGUF token type: integer out of range");
            h.tok.types.push_back(int32_t(v));
          }
          h.tok.has_types = true;
        } else {
          r.skip_array(element, n, 0);
        }
      }
    } else if (key == "tokenizer.ggml.scores") {
      if (h.tok.seen_scores) invalid("empty or duplicate GGUF metadata key");
      h.tok.seen_scores = true;
      r.skip_value(type);
    } else {
      const std::string k(key);
      if (h.meta.contains(k)) invalid("empty or duplicate GGUF metadata key");
      h.meta[k] = r.value(type);
    }
  }
  h.alignment = h.meta.contains("general.alignment")
                    ? size_t(integer(h.meta.at("general.alignment"), "alignment", 8, 1 << 20))
                    : 32;
  if (h.alignment % 8) invalid("GGUF alignment must be a multiple of eight");
  h.tensor_pos = r.position();
  return h;
}

void append_shard_tensors(const uint8_t* data, size_t size, const ShardHeader& h, ModelFile::Contents* result,
                          bool* quantized) {
  Cursor r(data + h.tensor_pos, size - h.tensor_pos);
  struct Entry {
    uint64_t offset;
    size_t tensor;
  };
  std::vector<Entry> entries;
  entries.reserve(size_t(h.n_tensors));
  for (uint64_t i = 0; i < h.n_tensors; ++i) {
    std::string name = r.string(64);
    if (name.empty() || result->index.count(name)) invalid("empty or duplicate GGUF tensor name");
    flyx_tensor_view v{};
    const uint32_t rank = r.read<uint32_t>();
    if (rank < 1 || rank > 4) invalid("unsupported GGUF tensor rank: " + name);
    v.ndim = int32_t(rank);
    size_t n = 1;
    for (uint32_t d = 0; d < rank; ++d) {  // GGUF lists the fastest dimension first
      const uint64_t dim = r.read<uint64_t>();
      if (!dim || dim > INT_MAX || dim > std::numeric_limits<size_t>::max() / n)
        invalid("invalid GGUF tensor dimension: " + name);
      v.shape[rank - d - 1] = int64_t(dim);
      n *= size_t(dim);
    }
    const uint32_t type = r.read<uint32_t>();
    v.dtype = dtype_of(type);
    if (v.dtype < 0) invalid(name + ": GGUF tensor type " + std::to_string(type) + " is not one this build reads");
    const size_t per = size_t(flyx_block_values(v.dtype));
    if (per) {  // a row is a whole number of blocks
      if (size_t(v.shape[rank - 1]) % per) invalid("GGUF tensor rows are not whole blocks: " + name);
      const size_t blocks = n / per;
      const size_t bb = size_t(flyx_block_bytes(v.dtype));
      if (blocks > std::numeric_limits<size_t>::max() / bb) invalid("GGUF tensor byte size overflows: " + name);
      v.nbytes = blocks * bb;
      *quantized = true;
    } else {
      const size_t eb = size_t(plain_dtype_bytes(v.dtype));
      if (!eb || n > std::numeric_limits<size_t>::max() / eb) invalid("GGUF tensor byte size overflows: " + name);
      v.nbytes = n * eb;
    }
    const uint64_t offset = r.read<uint64_t>();
    if (offset % h.alignment) invalid("unaligned GGUF tensor offset: " + name);
    entries.push_back({offset, result->tensors.size()});
    result->index[name] = result->tensors.size();
    result->names.push_back(std::move(name));
    result->tensors.push_back(v);
  }
  const size_t header_end = h.tensor_pos + r.position();
  const size_t padding = (h.alignment - header_end % h.alignment) % h.alignment;
  if (padding > size - header_end) invalid("truncated GGUF tensor alignment padding");
  const size_t base = header_end + padding;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(entries.size());
  for (const Entry& e : entries) {
    auto& v = result->tensors[e.tensor];
    if (e.offset > size - base || v.nbytes > size - base - e.offset)
      invalid("GGUF tensor outside the file: " + result->names[e.tensor]);
    v.data = data + base + size_t(e.offset);
    ranges.push_back({e.offset, e.offset + v.nbytes});
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second) invalid("overlapping GGUF tensor data");
}

void finalize_contents(json&& meta, const RawTokenizer& raw_tok, uint32_t version, int shards, bool quantized,
                       bool need_tokenizer, ModelFile::Contents* result) {
  if (result->tensors.empty()) invalid("GGUF has no tensors");
  if (quantized && (!meta.contains("general.quantization_version") ||
                    integer(meta.at("general.quantization_version"), "quantization version", 0, INT_MAX) != 2))
    invalid("unsupported or missing GGUF quantization version (need 2)");
  for (size_t i = 0; i < result->tensors.size(); ++i)
    result->tensors[i].name = result->names[i].c_str();  // names is final now: its strings do not move
  if (!meta.contains("general.architecture") || !meta.at("general.architecture").is_string())
    invalid("GGUF without general.architecture");
  result->arch = meta.at("general.architecture").get<std::string>();
  result->header["arch"] = result->arch;
  result->header["source"] = {{"format", "gguf"}, {"version", version}};
  if (shards > 1) result->header["source"]["shards"] = shards;
  if (need_tokenizer) tokenizer(meta, raw_tok, result);
  // What a model reads: every key but the tokenizer's tables, which are blobs now.
  meta.erase("tokenizer.chat_template");
  result->header["gguf"] = std::move(meta);
}

}  // namespace

bool is_gguf(const uint8_t* data, size_t size) { return size >= 4 && std::memcmp(data, "GGUF", 4) == 0; }

bool read_gguf(const uint8_t* data, size_t size, const std::string& path, ModelFile::Contents* out,
               std::string* err, bool need_tokenizer) {
  try {
    ShardHeader h = parse_shard_header(data, size, false, need_tokenizer);
    const std::string gtype = h.meta.value("general.type", "model");
    if (gtype != "model" && (need_tokenizer || gtype != "mmproj"))
      invalid(gtype == "mmproj" ? "GGUF is a multimodal projector (pass beside a model with --mmproj)"
                                : "GGUF is not a complete model");
    if (h.meta.contains("split.count") && integer(h.meta.at("split.count"), "split.count", 0, INT_MAX) > 1)
      invalid("split GGUF requires opening its files from disk");
    ModelFile::Contents result;
    bool quantized = false;
    append_shard_tensors(data, size, h, &result, &quantized);
    finalize_contents(std::move(h.meta), h.tok, h.version, 1, quantized, need_tokenizer, &result);
    *out = std::move(result);
    return true;
  } catch (const std::exception& e) {
    *out = {};
    *err = path + ": " + e.what();
    return false;
  }
}

bool read_gguf(std::vector<std::unique_ptr<os::MappedFile>>* maps, const std::string& path, ModelFile::Contents* out,
               std::string* err, bool need_tokenizer) {
  try {
    if (!maps || maps->empty() || !(*maps)[0]) invalid("missing GGUF mapping");
    ShardHeader initial = parse_shard_header((*maps)[0]->data(), (*maps)[0]->size(), true, need_tokenizer);
    const int split_count =
        initial.meta.contains("split.count") ? int(integer(initial.meta.at("split.count"), "split.count", 0, 99999)) : 1;
    if (split_count <= 1) {
      if (initial.n_tensors == 0) invalid("invalid GGUF tensor/metadata counts");
      if (need_tokenizer && initial.meta.value("general.architecture", "") == "dflash")
        invalid("GGUF holds a drafter alone, not a model: give it beside the model (--drafter-file)");
      const std::string gtype = initial.meta.value("general.type", "model");
      if (gtype != "model" && (need_tokenizer || gtype != "mmproj"))
        invalid(gtype == "mmproj" ? "GGUF is a multimodal projector (pass beside a model with --mmproj)"
                                  : "GGUF is not a complete model");
      ModelFile::Contents result;
      bool quantized = false;
      append_shard_tensors((*maps)[0]->data(), (*maps)[0]->size(), initial, &result, &quantized);
      finalize_contents(std::move(initial.meta), initial.tok, initial.version, 1, quantized, need_tokenizer, &result);
      *out = std::move(result);
      return true;
    }

    const int initial_no =
        initial.meta.contains("split.no") ? int(integer(initial.meta.at("split.no"), "split.no", 0, split_count - 1)) : 0;
    SplitPath sp;
    if (!parse_split_suffix(path, &sp))
      invalid("split GGUF file name must end with -XXXXX-of-YYYYY.gguf");
    if (sp.count != split_count)
      invalid("split GGUF file name says " + std::to_string(sp.count) + " parts, header says " +
              std::to_string(split_count));
    if (sp.part != initial_no + 1)
      invalid("split GGUF file name says part " + std::to_string(sp.part) + ", header says " +
              std::to_string(initial_no + 1));

    std::vector<std::unique_ptr<os::MappedFile>> shards(static_cast<size_t>(split_count));
    shards[size_t(initial_no)] = std::move((*maps)[0]);
    for (int i = 0; i < split_count; ++i) {
      if (i == initial_no) continue;
      const std::string p = shard_path(sp, i);
      auto m = std::make_unique<os::MappedFile>();
      std::string open_err;
      if (!m->open(p, &open_err))
        invalid("missing split GGUF shard " + std::to_string(i + 1) + " of " + std::to_string(split_count) + " (" +
                open_err + ")");
      shards[size_t(i)] = std::move(m);
    }

    ShardHeader h0 = initial_no == 0 ? std::move(initial)
                                     : parse_shard_header(shards[0]->data(), shards[0]->size(), true, need_tokenizer);
    if (h0.meta.value("general.type", "model") != "model") invalid("GGUF is not a complete model");
    if (!h0.meta.contains("split.count") || int(integer(h0.meta.at("split.count"), "split.count", 0, 99999)) != split_count)
      invalid("shard 1 split.count does not match");
    if (h0.meta.contains("split.no") && integer(h0.meta.at("split.no"), "split.no", 0, split_count - 1) != 0)
      invalid("shard 1 has non-zero split.no");
    const int64_t expected_tensors = h0.meta.contains("split.tensors.count")
                                         ? integer(h0.meta.at("split.tensors.count"), "split.tensors.count", 1, INT_MAX)
                                         : -1;

    ModelFile::Contents result;
    bool quantized = false;
    append_shard_tensors(shards[0]->data(), shards[0]->size(), h0, &result, &quantized);
    for (int i = 1; i < split_count; ++i) {
      ShardHeader hi =
          i == initial_no ? std::move(initial)
                          : parse_shard_header(shards[size_t(i)]->data(), shards[size_t(i)]->size(), true, false);
      if (hi.meta.contains("general.type") && hi.meta.value("general.type", "model") != "model")
        invalid("shard " + std::to_string(i + 1) + " is not a model shard");
      if (!hi.meta.contains("split.count") ||
          int(integer(hi.meta.at("split.count"), "split.count", 0, 99999)) != split_count)
        invalid("shard " + std::to_string(i + 1) + " split.count does not match");
      if (!hi.meta.contains("split.no") ||
          int(integer(hi.meta.at("split.no"), "split.no", 0, split_count - 1)) != i)
        invalid("shard " + std::to_string(i + 1) + " split.no does not match its index");
      append_shard_tensors(shards[size_t(i)]->data(), shards[size_t(i)]->size(), hi, &result, &quantized);
    }
    if (expected_tensors >= 0 && int64_t(result.tensors.size()) != expected_tensors)
      invalid("split GGUF tensor count mismatch: header says " + std::to_string(expected_tensors) + ", shards hold " +
              std::to_string(result.tensors.size()));
    finalize_contents(std::move(h0.meta), h0.tok, h0.version, split_count, quantized, need_tokenizer, &result);
    *maps = std::move(shards);
    *out = std::move(result);
    return true;
  } catch (const std::exception& e) {
    *out = {};
    *err = path + ": " + e.what();
    return false;
  }
}

namespace {

// A GGUF header read sequentially through a buffer, skipping values not needed (such as tokenizer tables).
class HeaderReader {
 public:
  explicit HeaderReader(const std::string& path) : f_(os::to_path(path), std::ios::binary), buf_(size_t(1) << 20) {}
  bool read(void* out, size_t n) {
    char* o = static_cast<char*>(out);
    while (n > 0) {
      if (at_ == len_ && !fill()) return false;
      const size_t k = std::min(n, len_ - at_);
      std::memcpy(o, buf_.data() + at_, k);
      o += k, n -= k, at_ += k;
    }
    return true;
  }
  bool skip(uint64_t n) {
    const uint64_t here = std::min<uint64_t>(n, len_ - at_);
    at_ += size_t(here), n -= here;
    if (n == 0) return true;
    constexpr uint64_t kMaxStep = static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max());
    if (n > kMaxStep) return false;
    f_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    return bool(f_);
  }
  template <class T>
  bool get(T* v) {
    return read(v, sizeof(T));
  }

 private:
  bool fill() {
    if (!f_) return false;
    f_.read(buf_.data(), std::streamsize(buf_.size()));
    len_ = size_t(f_.gcount()), at_ = 0;
    if (f_.eof()) f_.clear();
    return len_ > 0;
  }
  std::ifstream f_;
  std::vector<char> buf_;
  size_t at_ = 0, len_ = 0;
};

}  // namespace

}  // namespace flyx::modelfile

namespace flyx {

std::string gguf_header_path(const std::string& path) {
  modelfile::SplitPath sp;
  return modelfile::parse_split_suffix(path, &sp) && sp.part != 1 ? modelfile::shard_path(sp, 0) : path;
}

GgufPeek gguf_peek(const std::string& path) {
  GgufPeek out;
  modelfile::HeaderReader r(gguf_header_path(path));
  char magic[4];
  uint32_t version = 0;
  if (!r.read(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0 || !r.get(&version) || version < 1 || version > 3)
    return out;
  auto count = [&](uint64_t* n) {
    if (version > 1) return r.get(n);
    uint32_t v = 0;
    return r.get(&v) ? (*n = v, true) : false;
  };
  constexpr uint64_t kLongest = uint64_t(16) << 20;
  constexpr uint64_t kLongestKept = 4096;
  auto text = [&](std::string* s) {
    uint64_t n = 0;
    if (!count(&n) || n > kLongest) return false;
    if (!s) return r.skip(n);
    if (n > kLongestKept) return false;
    s->resize(size_t(n));
    return r.read(s->data(), size_t(n));
  };
  static constexpr int kSize[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};  // GGUF value types; 0: not fixed
  uint64_t tensors = 0, entries = 0;
  if (!count(&tensors) || !count(&entries) || entries > (uint64_t(1) << 24)) return out;
  std::string arch;
  std::map<std::string, int64_t> layers;  // keyed by architecture: the key may come before general.architecture
  bool has_markov = false;
  for (uint64_t i = 0; i < entries; ++i) {
    std::string key;
    uint32_t type = 0;
    if (!text(&key) || !r.get(&type) || type > 12) return out;
    if (type == 9) {  // array: element type, count, elements
      uint32_t of = 0;
      uint64_t n = 0;
      if (!r.get(&of) || of > 12 || of == 9 || !count(&n)) return out;
      if (of == 8) {
        for (uint64_t k = 0; k < n; ++k)
          if (!text(nullptr)) return out;
      } else {
        const uint64_t es = uint64_t(kSize[of]);
        if (!es || n > std::numeric_limits<uint64_t>::max() / es || !r.skip(n * es)) return out;
      }
      continue;
    }
    if (type == 8) {
      if (!text(key == "general.architecture" ? &arch : nullptr)) return out;
      continue;
    }
    int64_t v = 0;
    const int size = kSize[type];
    unsigned char raw[8] = {};
    if (!r.read(raw, size_t(size))) return out;
    const bool is_signed = type == 1 || type == 3 || type == 5 || type == 11;
    for (int b = size - 1; b >= 0; --b) v = (v << 8) | raw[b];  // GGUF is little-endian
    if (is_signed && size < 8 && (raw[size - 1] & 0x80)) v -= int64_t(1) << (8 * size);
    if (key == "dflash.has_markov_context" && type == 7) out.has_markov_context = v != 0;
    static const std::string kKey = ".nextn_predict_layers";
    if (type != 6 && type != 12 && key.size() > kKey.size() && key.ends_with(kKey))
      layers[key.substr(0, key.size() - kKey.size())] = v;
    else if (type == 7 && key == "dflash.has_markov_context")
      has_markov = v != 0;
  }
  out.valid = true;
  out.arch = arch;
  const auto it = layers.find(arch);
  out.mtp_layers = it == layers.end() || it->second < 0 ? 0 : int(std::min<int64_t>(it->second, 1 << 20));
  out.has_markov_context = arch == "dflash" && has_markov;
  return out;
}

int gguf_peek_mtp_layers(const std::string& path) {
  const GgufPeek p = gguf_peek(path);
  return p.valid ? p.mtp_layers : -1;
}

}  // namespace flyx
