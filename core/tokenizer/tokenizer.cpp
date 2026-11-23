// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/tokenizer/tokenizer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <mutex>
#include <queue>
#include <shared_mutex>

#include "core/modelfile/modelfile.h"
#include "utf8proc.h"

namespace flyx {
namespace {

// Decode UTF-8 into code points, remembering byte offsets.  Invalid bytes
// become U+FFFD-like placeholders of length 1 so nothing is lost.
void decode_utf8(std::string_view s, std::vector<int32_t>* cps, std::vector<size_t>* offs) {
  size_t i = 0;
  while (i < s.size()) {
    utf8proc_int32_t cp;
    auto n = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(s.data() + i),
                              static_cast<utf8proc_ssize_t>(s.size() - i), &cp);
    if (n <= 0) {
      cp = -1;
      n = 1;
    }
    cps->push_back(cp);
    offs->push_back(i);
    i += static_cast<size_t>(n);
  }
  offs->push_back(s.size());
}

enum Cls : uint8_t { kOther = 0, kL = 1, kM = 2, kN = 4, kWS = 8, kCRLF = 16 };

bool is_white_space(int32_t c) {
  // Unicode White_Space property.
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

constexpr std::array<uint8_t, 128> make_ascii_cls() {
  std::array<uint8_t, 128> t{};
  for (int c = 0; c < 128; ++c) {
    uint8_t r = 0;
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) r |= kL;
    if (c >= '0' && c <= '9') r |= kN;
    if ((c >= 0x09 && c <= 0x0D) || c == 0x20) r |= kWS;
    if (c == '\r' || c == '\n') r |= kCRLF;
    t[static_cast<size_t>(c)] = r;
  }
  return t;
}

constexpr std::array<uint8_t, 128> kAsciiCls = make_ascii_cls();

bool is_ascii(std::string_view s) {
  for (unsigned char c : s)
    if (c >= 0x80) return false;
  return true;
}

uint8_t classify(int32_t c) {
  if (c < 0) return kOther;
  if (c < 128) return kAsciiCls[static_cast<size_t>(c)];
  uint8_t r = 0;
  switch (utf8proc_category(c)) {
    case UTF8PROC_CATEGORY_LU: case UTF8PROC_CATEGORY_LL: case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM: case UTF8PROC_CATEGORY_LO:
      r |= kL; break;
    case UTF8PROC_CATEGORY_MN: case UTF8PROC_CATEGORY_MC: case UTF8PROC_CATEGORY_ME:
      r |= kM; break;
    case UTF8PROC_CATEGORY_ND: case UTF8PROC_CATEGORY_NL: case UTF8PROC_CATEGORY_NO:
      r |= kN; break;
    default: break;
  }
  if (is_white_space(c)) r |= kWS;
  return r;
}

int32_t ascii_lower(int32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Hand-written equivalent of the split regexes of the Qwen family, on the code points [begin, end) of s, which the
// regex sees as the whole text:
//   Qwen3.5   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N}
//             | ' '?[^\s\p{L}\p{M}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//   Qwen2     the same with \p{L}+ for the letters (marks are punctuation)
//   MiniCPM5  Qwen2's with \p{N}+ for the numbers
// Alternatives are tried in order at each position, with the same greedy and
// backtracking outcome a regex engine produces.  One state machine over either
// direct ASCII bytes or decoded UTF-8 code points.
struct SplitRules {
  bool marks_in_words;  // [\p{L}\p{M}]+, else \p{L}+
  bool number_runs;     // \p{N}+, else \p{N}
};

template <class CpAt, class ClsAt, class OffAt, class Emit>
void pretokenize_impl(std::string_view s, size_t begin, size_t end, SplitRules rules, CpAt&& cp, ClsAt&& k,
                      OffAt&& off, Emit&& emit) {
  const size_t n = end;
  const uint8_t letters = rules.marks_in_words ? (kL | kM) : kL;
  auto LM = [&](size_t i) { return i < n && (k(i) & letters); };
  auto WS = [&](size_t i) { return i < n && (k(i) & kWS); };
  auto P = [&](size_t i) { return i < n && !(k(i) & (kWS | letters | kN)); };

  size_t i = begin;
  while (i < n) {
    size_t e = 0;
    // 1. contractions
    if (cp(i) == '\'' && i + 1 < n) {
      const int32_t a = ascii_lower(cp(i + 1));
      if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        e = i + 2;
      } else if (i + 2 < n) {
        const int32_t b = ascii_lower(cp(i + 2));
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = i + 3;
      }
    }
    // 2. optional non-letter/number/newline char, then letters/marks
    if (!e) {
      const bool lead = !(k(i) & (kL | kN | kCRLF));
      if (lead && LM(i + 1)) {
        e = i + 1;
        while (LM(e)) ++e;
      } else if (LM(i)) {
        e = i;
        while (LM(e)) ++e;
      }
    }
    // 3. one number, or a run of them
    if (!e && (k(i) & kN)) {
      e = i + 1;
      while (rules.number_runs && e < n && (k(e) & kN)) ++e;
    }
    // 4. optional space, punctuation run, trailing newlines
    if (!e) {
      const size_t j = (cp(i) == ' ') ? i + 1 : i;
      if (P(j)) {
        e = j;
        while (P(e)) ++e;
        while (e < n && (k(e) & kCRLF)) ++e;
      }
    }
    // 5-7. whitespace
    if (!e && WS(i)) {
      size_t r = i;
      while (WS(r)) ++r;
      size_t last_nl = SIZE_MAX;
      for (size_t j = i; j < r; ++j)
        if (k(j) & kCRLF) last_nl = j;
      if (last_nl != SIZE_MAX) {
        e = last_nl + 1;                 // \s*[\r\n]+
      } else if (r == n) {
        e = r;                           // \s+(?!\S) at end of text
      } else if (r - i >= 2) {
        e = r - 1;                       // \s+(?!\S): leave one space for the next word
      } else {
        e = r;                           // \s+
      }
    }
    if (!e) e = i + 1;
    emit(s.substr(off(i), off(e) - off(i)));
    i = e;
  }
}

// The pieces of the n code points of s by the pre-tokenizer `kind`.  MiniCPM5: \p{N}{1,3} isolated first (a run of
// numbers cut into threes from the left), then the word split on each piece by itself, the groups of numbers included.
template <class CpAt, class ClsAt, class OffAt, class Emit>
void pretokenize_kind(std::string_view s, size_t n, Pretokenizer kind, CpAt&& cp, ClsAt&& k, OffAt&& off,
                      Emit&& emit) {
  if (kind != Pretokenizer::kMiniCPM5) {
    pretokenize_impl(s, 0, n, {kind == Pretokenizer::kQwen35, false}, cp, k, off, emit);
    return;
  }
  size_t i = 0;
  while (i < n) {
    size_t e = i;
    if (k(i) & kN) {
      while (e < n && e - i < 3 && (k(e) & kN)) ++e;
    } else {
      while (e < n && !(k(e) & kN)) ++e;
    }
    pretokenize_impl(s, i, e, {false, true}, cp, k, off, emit);
    i = e;
  }
}

template <class Emit>
void pretokenize_spans(std::string_view s, Pretokenizer kind, bool ascii_known, Emit&& emit) {
  if (ascii_known || is_ascii(s)) {
    pretokenize_kind(
        s, s.size(), kind, [&](size_t i) { return static_cast<int32_t>(static_cast<unsigned char>(s[i])); },
        [&](size_t i) { return kAsciiCls[static_cast<unsigned char>(s[i])]; }, [](size_t i) { return i; }, emit);
    return;
  }
  std::vector<int32_t> cp;
  std::vector<size_t> off;
  cp.reserve(s.size());
  off.reserve(s.size() + 1);
  decode_utf8(s, &cp, &off);
  const size_t n = cp.size();
  std::vector<uint8_t> cls(n);
  for (size_t i = 0; i < n; ++i) cls[i] = classify(cp[i]);
  pretokenize_kind(
      s, n, kind, [&](size_t i) { return cp[i]; }, [&](size_t i) { return cls[i]; }, [&](size_t i) { return off[i]; },
      emit);
}

uint32_t rd32(const char* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

}  // namespace

std::string Tokenizer::nfc(std::string_view text) {
  if (text.empty() || is_ascii(text)) return std::string(text);
  utf8proc_uint8_t* out = nullptr;
  const utf8proc_ssize_t len = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
                                            static_cast<utf8proc_ssize_t>(text.size()), &out,
                                            static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
  if (len < 0 || !out) return std::string(text);  // invalid UTF-8: leave as is
  std::string r(reinterpret_cast<char*>(out), static_cast<size_t>(len));
  free(out);
  return r;
}

std::vector<std::string> Tokenizer::pretokenize(std::string_view s, Pretokenizer kind) {
  std::vector<std::string> out;
  pretokenize_spans(s, kind, false, [&](std::string_view piece) { out.emplace_back(piece); });
  return out;
}

bool Tokenizer::load(const ModelFile& file, std::string* err) {
  const auto& h = file.header();
  if (!h.is_object() || !h.contains("tokenizer") || !h["tokenizer"].is_object()) {
    *err = "model file has no tokenizer";
    return false;
  }
  const auto& t = h["tokenizer"];
  const std::string pre = t.contains("pretokenizer") && t["pretokenizer"].is_string() ? t["pretokenizer"].get<std::string>() : "";
  const std::string normalizer = t.contains("normalizer") && t["normalizer"].is_string() ? t["normalizer"].get<std::string>() : "";
  if ((pre != "qwen35" && pre != "qwen2" && pre != "minicpm5") || (normalizer != "nfc" && normalizer != "none")) {
    *err = "unsupported tokenizer pipeline (pre-tokenizer '" + pre + "', normalizer '" + normalizer + "')";
    return false;
  }
  ignore_merges_ = t.contains("ignore_merges") && t["ignore_merges"].is_boolean() && t["ignore_merges"].get<bool>();
  pretokenizer_ = pre == "qwen2"      ? Pretokenizer::kQwen2
                  : pre == "minicpm5" ? Pretokenizer::kMiniCPM5
                                      : Pretokenizer::kQwen35;
  normalize_nfc_ = normalizer == "nfc";
  by_bytes_.clear(); merges_.clear(); added_.clear(); cache_.clear();
  const int64_t n64 = t.contains("n_tokens") && t["n_tokens"].is_number_integer() ? t["n_tokens"].get<int64_t>() : 0;
  auto vb = file.blob("tokenizer.vocab");
  if (n64 <= 0 || n64 > (1 << 24) || vb.size() / 4 < static_cast<size_t>(n64) + 1) {
    *err = "tokenizer.vocab blob is truncated";
    return false;
  }
  const int n = static_cast<int>(n64);
  const size_t table_bytes = (static_cast<size_t>(n) + 1) * 4;
  const char* base = vb.data() + table_bytes;
  const size_t bytes = vb.size() - table_bytes;
  tokens_.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const size_t off = static_cast<size_t>(i) * 4;
    uint32_t a = rd32(vb.data() + off), b = rd32(vb.data() + off + 4);
    if (a > b || b > bytes) {  // the offsets are the file's: a token's bytes lie inside the blob
      *err = "tokenizer.vocab blob: token " + std::to_string(i) + " lies outside it";
      return false;
    }
    tokens_[static_cast<size_t>(i)].assign(base + a, b - a);
  }
  std::fill(std::begin(byte_id_), std::end(byte_id_), -1);
  auto ab = file.blob("tokenizer.added");
  auto added = nlohmann::json::parse(ab.begin(), ab.end(), nullptr, false);
  if (!added.is_array()) {
    *err = "tokenizer.added blob is not a JSON list";
    return false;
  }
  std::vector<bool> is_added(n, false);
  special_.assign(n, false);
  std::fill(std::begin(added_first_), std::end(added_first_), false);
  for (const auto& a : added) {
    if (!a.is_object() || !a.contains("id") || !a["id"].is_number_integer() || !a.contains("content") ||
        !a["content"].is_string()) {
      *err = "tokenizer.added blob: an added token without an id or its content";
      return false;
    }
    const int64_t id64 = a["id"].get<int64_t>();
    if (id64 < 0 || id64 >= n) {
      *err = "tokenizer.added blob: added token id out of range";
      return false;
    }
    const int id = static_cast<int>(id64);
    const bool special = !a.contains("special") || (a["special"].is_boolean() && a["special"].get<bool>());
    added_.push_back({a["content"].get<std::string>(), id, special});
    if (!added_.back().content.empty()) added_first_[static_cast<uint8_t>(added_.back().content[0])] = true;
    is_added[static_cast<size_t>(id)] = true;
    special_[static_cast<size_t>(id)] = special;
  }
  std::sort(added_.begin(), added_.end(),
            [](const Added& x, const Added& y) { return x.content.size() > y.content.size(); });
  by_bytes_.reserve(tokens_.size());
  for (int i = 0; i < n; ++i) {
    if (is_added[i]) continue;
    if (!tokens_[i].empty()) by_bytes_.emplace(tokens_[i], i);
    if (tokens_[i].size() == 1) byte_id_[static_cast<uint8_t>(tokens_[i][0])] = i;
  }
  for (int b = 0; b < 256; ++b)
    if (byte_id_[b] < 0) {
      *err = "tokenizer has no token for byte " + std::to_string(b);
      return false;
    }
  auto mb = file.blob("tokenizer.merges");
  const size_t nm = mb.size() / 12;
  merges_.reserve(nm * 2);
  for (size_t r = 0; r < nm; ++r) {
    uint32_t l = rd32(mb.data() + 12 * r), rr = rd32(mb.data() + 12 * r + 4), m = rd32(mb.data() + 12 * r + 8);
    if (l >= uint32_t(n) || rr >= uint32_t(n) || m >= uint32_t(n)) {  // a merge makes a token of this vocabulary
      *err = "tokenizer.merges blob: merge " + std::to_string(r) + " names a token the vocabulary does not have";
      return false;
    }
    merges_.emplace((uint64_t(l) << 32) | rr, std::make_pair(static_cast<int32_t>(r), static_cast<int32_t>(m)));
  }
  // Each a special token: render_request writes a raw prompt's prefix as its tokens' text where special tokens are
  // read, which must give these tokens back.
  prefix_.clear();
  const nlohmann::json none = nlohmann::json::array();
  const auto& prefix = t.contains("prefix_ids") ? t["prefix_ids"] : none;
  if (!prefix.is_array()) {
    *err = "tokenizer.prefix_ids is not a list";
    return false;
  }
  for (const auto& id : prefix) {
    const int64_t v = id.is_number_integer() ? id.get<int64_t>() : -1;
    if (v < 0 || v >= n || !special_[size_t(v)]) {
      *err = "tokenizer.prefix_ids: " + id.dump() + " is not a special token of the vocabulary";
      return false;
    }
    prefix_.push_back(static_cast<int32_t>(v));
  }
  return true;
}

std::vector<int32_t> Tokenizer::encode_prompt(std::string_view text) const {
  std::vector<int32_t> ids = prefix_;
  const auto rest = encode(text);
  ids.insert(ids.end(), rest.begin(), rest.end());
  return ids;
}

int32_t Tokenizer::added_id(const std::string& content) const {
  for (const auto& a : added_)
    if (a.content == content) return a.id;
  return -1;
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
  static const std::string empty;
  return (id >= 0 && id < static_cast<int32_t>(tokens_.size())) ? tokens_[id] : empty;
}

// BPE on one piece: of the pairs side by side that have a merge, the one of lowest rank first, the leftmost of
// several, one merge at a time, until none has one.
//
// A word takes a pass over its symbols per merge: nothing to set up, and a word is a few symbols.  A pass per merge
// is n * n steps on a piece of n bytes, though, and a piece is as long as its letters run without a space (a
// megabyte of them would hold the thread that tokenizes for minutes).  So a longer piece keeps its pairs in a heap by
// (rank, place) and its symbols in a list, and a merge costs the two pairs it makes: n log n.  An entry whose two
// symbols are no longer the ones it was made for is a pair an earlier merge took apart: dropped when it comes up.
// Both give the same tokens (tests/core/tokenizer/merge_test.cpp); the heap is ahead from some 16 symbols on.
void Tokenizer::merge(std::vector<int32_t>* sym) const {
  const int32_t n = static_cast<int32_t>(sym->size());
  if (n < 2) return;
  constexpr int32_t kPassPerMerge = 16;
  if (n <= kPassPerMerge) {
    while (sym->size() > 1) {
      int32_t best = INT32_MAX, at = -1, res = -1;
      for (size_t i = 0; i + 1 < sym->size(); ++i) {
        auto it = merges_.find((uint64_t(uint32_t((*sym)[i])) << 32) | uint32_t((*sym)[i + 1]));
        if (it != merges_.end() && it->second.first < best) {
          best = it->second.first;
          at = static_cast<int32_t>(i);
          res = it->second.second;
        }
      }
      if (at < 0) break;
      (*sym)[size_t(at)] = res;
      sym->erase(sym->begin() + at + 1);
    }
    return;
  }
  std::vector<int32_t>& s = *sym;
  constexpr int32_t kGone = -1;  // a symbol merged into the one on its left; no token has this id
  std::vector<int32_t> next(static_cast<size_t>(n)), prev(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) {
    prev[size_t(i)] = i - 1;
    next[size_t(i)] = i + 1 < n ? i + 1 : -1;
  }
  struct Pair {
    int32_t rank, at, left, right, merged;
    bool operator>(const Pair& o) const { return rank != o.rank ? rank > o.rank : at > o.at; }
  };
  std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> pairs;
  auto offer = [&](int32_t i) {  // the pair that starts at i, if it has a merge
    const int32_t j = next[size_t(i)];
    if (j < 0) return;
    auto it = merges_.find((uint64_t(uint32_t(s[size_t(i)])) << 32) | uint32_t(s[size_t(j)]));
    if (it != merges_.end()) pairs.push({it->second.first, i, s[size_t(i)], s[size_t(j)], it->second.second});
  };
  for (int32_t i = 0; i + 1 < n; ++i) offer(i);
  while (!pairs.empty()) {
    const Pair p = pairs.top();
    pairs.pop();
    const int32_t j = next[size_t(p.at)];
    if (s[size_t(p.at)] != p.left || j < 0 || s[size_t(j)] != p.right) continue;
    s[size_t(p.at)] = p.merged;
    s[size_t(j)] = kGone;
    next[size_t(p.at)] = next[size_t(j)];
    if (next[size_t(j)] >= 0) prev[size_t(next[size_t(j)])] = p.at;
    if (prev[size_t(p.at)] >= 0) offer(prev[size_t(p.at)]);
    offer(p.at);
  }
  size_t kept = 0;
  for (int32_t i = 0; i >= 0; i = next[size_t(i)]) s[kept++] = s[size_t(i)];  // i only grows: kept <= i
  s.resize(kept);
}

std::vector<int32_t> Tokenizer::encode_uncached(std::string_view piece) const {
  std::vector<int32_t> sym;
  if (ignore_merges_) {
    auto it = by_bytes_.find(piece);
    if (it != by_bytes_.end()) {
      sym.push_back(it->second);
      return sym;
    }
  }
  sym.reserve(piece.size());
  for (unsigned char b : piece) sym.push_back(byte_id_[b]);
  merge(&sym);
  return sym;
}

void Tokenizer::encode_pieces(const std::vector<std::string_view>& pieces, std::vector<int32_t>* out) const {
  if (pieces.empty()) return;
  // Fast path: one shared lock covers all already-cached pieces (zero exclusive contention across HTTP threads).
  size_t first_miss = pieces.size();
  std::unordered_map<std::string_view, std::vector<int32_t>> misses;
  {
    std::shared_lock<std::shared_mutex> rlock(cache_mu_);
    for (size_t i = 0; i < pieces.size(); ++i) {
      auto c = cache_.find(pieces[i]);
      if (c != cache_.end()) {
        out->insert(out->end(), c->second.begin(), c->second.end());
      } else {
        first_miss = i;
        break;
      }
    }
    if (first_miss == pieces.size()) return;
    for (size_t i = first_miss; i < pieces.size(); ++i) {
      const std::string_view p = pieces[i];
      if (misses.find(p) == misses.end() && cache_.find(p) == cache_.end()) misses.emplace(p, std::vector<int32_t>{});
    }
  }
  // Compute BPE merges for each unique missing piece without holding cache_mu_.
  for (auto& kv : misses) kv.second = encode_uncached(kv.first);
  // Insert newly merged pieces once under an exclusive lock, then emit the remaining pieces in order.
  std::unique_lock<std::shared_mutex> wlock(cache_mu_);
  for (const auto& kv : misses) {
    if (cache_.size() < kCachedPieces && kv.first.size() <= kCachedPieceBytes)
      cache_.try_emplace(std::string(kv.first), kv.second);
  }
  for (size_t i = first_miss; i < pieces.size(); ++i) {
    const std::string_view p = pieces[i];
    if (auto m = misses.find(p); m != misses.end()) {
      out->insert(out->end(), m->second.begin(), m->second.end());
    } else if (auto c = cache_.find(p); c != cache_.end()) {
      out->insert(out->end(), c->second.begin(), c->second.end());
    } else {
      const auto sym = encode_uncached(p);
      out->insert(out->end(), sym.begin(), sym.end());
    }
  }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, const std::vector<TextSpan>* literal) const {
  std::vector<int32_t> out;
  out.reserve(text.size() / 3 + 16);
  std::vector<std::string_view> pieces;
  size_t seg = 0, i = 0, li = 0;
  auto flush = [&](size_t end) {
    if (end <= seg) return;
    const std::string_view raw = text.substr(seg, end - seg);
    const bool ascii = is_ascii(raw);
    std::string norm_buf;
    std::string_view src = raw;
    if (normalize_nfc_ && !ascii) {
      norm_buf = nfc(raw);
      src = norm_buf;
    }
    pieces.clear();
    pretokenize_spans(src, pretokenizer_, ascii, [&](std::string_view piece) { pieces.push_back(piece); });
    encode_pieces(pieces, &out);
  };
  // Whether [b, e) lies inside one literal span (b only grows).
  auto written_by_template = [&](size_t b, size_t e) {
    if (!literal) return true;
    while (li < literal->size() && (*literal)[li].end <= b) ++li;
    return li < literal->size() && (*literal)[li].begin <= b && e <= (*literal)[li].end;
  };
  while (i < text.size()) {
    const Added* hit = nullptr;
    if (added_first_[static_cast<uint8_t>(text[i])]) {  // few bytes begin an added token: the rest skip the list
      const char ch = text[i];
      for (const auto& a : added_) {  // longest first
        if (!a.content.empty() && a.content[0] == ch && text.compare(i, a.content.size(), a.content) == 0 &&
            (!a.special || written_by_template(i, i + a.content.size()))) {
          hit = &a;
          break;
        }
      }
    }
    if (hit) {
      flush(i);
      out.push_back(hit->id);
      i += hit->content.size();
      seg = i;
    } else {
      ++i;
    }
  }
  flush(text.size());
  return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string s;
  for (int32_t id : ids) s += token_bytes(id);
  return s;
}

std::string StreamDecoder::feed(const int32_t* ids, int n) {
  for (int i = 0; i < n; ++i) pending_ += tok_->token_bytes(ids[i]);
  // Emit everything up to the last complete UTF-8 character.
  size_t cut = pending_.size();
  size_t back = 0;
  while (back < 4 && back < pending_.size()) {
    unsigned char c = static_cast<unsigned char>(pending_[pending_.size() - 1 - back]);
    if ((c & 0xC0) != 0x80) {  // a lead byte or ASCII
      int need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
      if (static_cast<int>(back) + 1 < need) cut = pending_.size() - 1 - back;
      break;
    }
    ++back;
  }
  std::string out = pending_.substr(0, cut);
  pending_.erase(0, cut);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return out;
}

std::vector<int32_t> token_positions(const Tokenizer& tok, const std::vector<int32_t>& ids,
                                     const std::vector<size_t>& offsets) {
  std::vector<int32_t> out;
  size_t bytes = 0, k = 0;
  for (size_t off : offsets) {
    while (k < ids.size() && bytes + tok.token_bytes(ids[k]).size() <= off) bytes += tok.token_bytes(ids[k++]).size();
    out.push_back(static_cast<int32_t>(k));
  }
  return out;
}

}  // namespace flyx
