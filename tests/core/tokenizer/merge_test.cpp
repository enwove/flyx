// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The tokenizer's BPE against the rule said the plain way (core/tokenizer/tokenizer.cpp, Tokenizer::merge): of the
// pairs side by side that have a merge, the one of lowest rank, the leftmost of several, one merge and look again.
// On vocabularies made here (random merges over a few letters, so pairs of equal rank and merges that undo each
// other's neighbours are common), random texts, added tokens between them, and a piece long enough that a pass over
// it per merge would not end in the time allowed; and tables that are wrong (offsets and ids the file gives, held to
// the tables they index).  No model and no Python: the files are written here.
//
//   merge_test DIR        (the files are written there)
#include <chrono>
#include <climits>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "core/tokenizer/tokenizer.h"
#include "tests/common/check.h"
#include "tests/core/modelfile/write_flyx.h"

using namespace flyx;
using namespace flyx::testing;

namespace {

void put32(std::string* s, uint32_t v) { s->append(reinterpret_cast<const char*>(&v), 4); }

// A vocabulary: the 256 bytes, then tokens made by merges in rank order, then the added tokens.
struct Vocabulary {
  std::vector<std::string> tokens;
  std::map<std::pair<int32_t, int32_t>, std::pair<int32_t, int32_t>> merges;  // (left, right) -> (rank, merged)
  std::vector<std::pair<int32_t, int32_t>> order;                              // by rank
  std::vector<std::pair<std::string, bool>> added;                             // content, special
  int32_t first_added = 0;

  // `count` merges of two tokens picked at random among those made of `letters`.
  Vocabulary(const std::string& letters, int count, std::mt19937* rng) {
    for (int b = 0; b < 256; ++b) tokens.emplace_back(1, static_cast<char>(b));
    std::vector<int32_t> pool;
    for (char c : letters) pool.push_back(static_cast<uint8_t>(c));
    std::map<std::string, int32_t> by_bytes;
    for (int tries = 0; int(order.size()) < count && tries < 100 * count; ++tries) {
      const int32_t l = pool[(*rng)() % pool.size()], r = pool[(*rng)() % pool.size()];
      const std::string bytes = tokens[size_t(l)] + tokens[size_t(r)];
      if (bytes.size() > 12 || merges.count({l, r}) || by_bytes.count(bytes)) continue;
      const int32_t id = static_cast<int32_t>(tokens.size());
      merges[{l, r}] = {static_cast<int32_t>(order.size()), id};
      order.push_back({l, r});
      by_bytes[bytes] = id;
      tokens.push_back(bytes);
      pool.push_back(id);
    }
    first_added = static_cast<int32_t>(tokens.size());
    added = {{"<|turn|>", true}, {"<note>", false}, {"<|turn|>more", false}};
    for (const auto& a : added) tokens.push_back(a.first);
  }

  // The three tables as python/convert/flyx_convert/tokenizer_export.py writes them.
  struct Tables {
    std::string vocab, merges, added;
    size_t n_tokens = 0;
  };
  Tables tables() const {
    Tables t;
    std::string bytes;
    put32(&t.vocab, 0);
    for (const auto& tok : tokens) {
      bytes += tok;
      put32(&t.vocab, static_cast<uint32_t>(bytes.size()));
    }
    t.vocab += bytes;
    for (const auto& lr : order) {
      put32(&t.merges, uint32_t(lr.first));
      put32(&t.merges, uint32_t(lr.second));
      put32(&t.merges, uint32_t(merges.at(lr).second));
    }
    t.added = "[";
    for (size_t i = 0; i < added.size(); ++i)
      t.added += std::string(i ? "," : "") + R"({"id": )" + std::to_string(first_added + int32_t(i)) +
                 R"(, "content": ")" + added[i].first + R"(", "special": )" + (added[i].second ? "true" : "false") + "}";
    t.added += "]";
    t.n_tokens = tokens.size();
    return t;
  }

  static bool write(const std::string& path, const Tables& t) {
    const std::string header =
        R"({"arch": "test", "tensors": {}, "tokenizer": {"pretokenizer": "qwen35", "normalizer": "none", "n_tokens": )" +
        std::to_string(t.n_tokens) + R"(}, "blobs": {"tokenizer.vocab": {"offset": 0, "nbytes": )" +
        std::to_string(t.vocab.size()) + R"(}, "tokenizer.merges": {"offset": )" + std::to_string(t.vocab.size()) +
        R"(, "nbytes": )" + std::to_string(t.merges.size()) + R"(}, "tokenizer.added": {"offset": )" +
        std::to_string(t.vocab.size() + t.merges.size()) + R"(, "nbytes": )" + std::to_string(t.added.size()) + "}}}";
    return write_file(path, flyx_file(header, t.vocab + t.merges + t.added));
  }
  bool write(const std::string& path) const { return write(path, tables()); }

  // The rule, a pass over the piece per merge.
  void piece(std::string_view text, std::vector<int32_t>* out) const {
    std::vector<int32_t> sym;
    for (unsigned char b : text) sym.push_back(b);
    while (sym.size() > 1) {
      int32_t best = INT32_MAX, at = -1, res = -1;
      for (size_t i = 0; i + 1 < sym.size(); ++i) {
        auto it = merges.find({sym[i], sym[i + 1]});
        if (it != merges.end() && it->second.first < best) {
          best = it->second.first;
          at = static_cast<int32_t>(i);
          res = it->second.second;
        }
      }
      if (at < 0) break;
      sym[size_t(at)] = res;
      sym.erase(sym.begin() + at + 1);
    }
    out->insert(out->end(), sym.begin(), sym.end());
  }

  // Text without added tokens in it.
  std::vector<int32_t> plain(const std::string& text) const {
    std::vector<int32_t> out;
    for (const auto& p : Tokenizer::pretokenize(text)) piece(p, &out);
    return out;
  }
};

std::string random_text(const std::string& letters, int length, int space_one_in, std::mt19937* rng) {
  std::string s;
  for (int i = 0; i < length; ++i)
    s += space_one_in && (*rng)() % unsigned(space_one_in) == 0 ? ' ' : letters[(*rng)() % letters.size()];
  return s;
}

std::string show(const std::vector<int32_t>& ids) {
  std::string s;
  for (size_t i = 0; i < ids.size() && i < 40; ++i) s += " " + std::to_string(ids[i]);
  return s + (ids.size() > 40 ? " ..." : "");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: merge_test DIR\n");
    return 2;
  }
  const std::string path = std::string(argv[1]) + "/merge_test.flyx";
  std::mt19937 rng(20261112);
  int texts = 0;
  std::string err;

  // Random vocabularies and texts: few letters and many merges, so most pairs have one.
  for (const auto& [letters, merges] : std::vector<std::pair<std::string, int>>{
           {"a", 6}, {"ab", 40}, {"abc", 200}, {"abcd", 600}, {"abcdefgh", 1500}}) {
    Vocabulary v(letters, merges, &rng);
    ModelFile file;
    Tokenizer tok;
    if (!v.write(path) || !file.open(path, &err) || !tok.load(file, &err)) {
      std::printf("FAIL the vocabulary over \"%s\": %s\n", letters.c_str(), err.c_str());
      return 1;
    }
    for (int i = 0; i < 400 && flyx::test::failures < 5; ++i, ++texts) {
      // Lengths 0..: one piece (no spaces) or words; long ones too.
      const int length = i < 300 ? int(rng() % 60) : int(rng() % 3000);
      const std::string text = random_text(letters, length, i % 3 == 0 ? 0 : 7, &rng);
      const auto want = v.plain(text), got = tok.encode(text);
      if (got != want) {
        ++flyx::test::failures;
        std::printf("FAIL \"%s\" over \"%s\"\n  want:%s\n  got: %s\n", text.substr(0, 80).c_str(), letters.c_str(),
                    show(want).c_str(), show(got).c_str());
      }
      if (tok.decode(got) != text) check(false, "decoding gives the text back: " + text.substr(0, 80));
    }

    // Added tokens between pieces: read where they stand, the longest of two that begin alike, special ones only
    // where the template wrote them; a '<' that begins none is text.
    const int32_t turn = v.first_added, note = v.first_added + 1, turn_more = v.first_added + 2;
    auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) {
      a.insert(a.end(), b.begin(), b.end());
      return a;
    };
    const std::string w = random_text(letters, 9, 0, &rng), x = random_text(letters, 5, 0, &rng);
    check(tok.encode(w + "<|turn|>" + x) == cat(cat(v.plain(w), {turn}), v.plain(x)), "a special token in a raw prompt");
    check(tok.encode(w + "<note>" + x + "<note>") == cat(cat(cat(v.plain(w), {note}), v.plain(x)), {note}),
          "an added token twice");
    check(tok.encode("<|turn|>more" + x) == cat({turn_more}, v.plain(x)), "the longest of two that begin alike");
    check(tok.encode(w + "<" + x + "<|tur" + x) == v.plain(w + "<" + x + "<|tur" + x), "a '<' that begins no token");
    const std::string in_message = w + "<|turn|>" + x + "<note>";
    const std::vector<TextSpan> none;  // the template wrote none of it: special tokens are text, the others read
    check(tok.encode(in_message, &none) == cat(v.plain(w + "<|turn|>" + x), {note}),
          "a special token in a client's string is text");
  }

  // A piece of 400,000 letters: a pass per merge is some 10^10 steps (minutes); the heap's are counted in millions.
  {
    Vocabulary v("ab", 60, &rng);
    ModelFile file;
    Tokenizer tok;
    if (!v.write(path) || !file.open(path, &err) || !tok.load(file, &err)) {
      std::printf("FAIL the vocabulary for the long piece: %s\n", err.c_str());
      return 1;
    }
    const std::string text = random_text("ab", 400000, 0, &rng);
    const auto t0 = std::chrono::steady_clock::now();
    const auto got = tok.encode(text);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check(tok.decode(got) == text, "the long piece decodes to itself");
    check(got.size() < text.size() / 2, "the long piece was merged");
    check(s < 20.0, "a piece of 400,000 letters in " + std::to_string(s) + " s (allowed 20)");
    std::printf("a piece of %zu letters: %zu tokens in %.3f s\n", text.size(), got.size(), s);
    // Its first 20,000 letters against the rule (a piece's tokens depend on all of it, so a piece of its own).
    const std::string head = text.substr(0, 20000);
    check(tok.encode(head) == v.plain(head), "a piece of 20,000 letters as the rule gives it");
  }

  // Concurrent multi-thread encode (both cold misses and warm hits) must produce identical ids and stay race-free.
  {
    Vocabulary v("abcdefgh", 500, &rng);
    ModelFile file;
    Tokenizer tok;
    if (!v.write(path) || !file.open(path, &err) || !tok.load(file, &err)) {
      std::printf("FAIL the vocabulary for concurrent encode: %s\n", err.c_str());
      return 1;
    }
    std::vector<std::string> inputs;
    std::vector<std::vector<int32_t>> expected;
    for (int i = 0; i < 16; ++i) {
      inputs.push_back(random_text("abcdefgh", 400 + i * 37, 6, &rng));
      expected.push_back(v.plain(inputs.back()));
    }
    std::vector<std::thread> threads;
    std::vector<int> thread_bad(4, 0);
    for (int t = 0; t < 4; ++t) {
      threads.emplace_back([&, t] {
        for (int rep = 0; rep < 4; ++rep) {
          for (size_t i = 0; i < inputs.size(); ++i) {
            const size_t idx = (i + size_t(t) * 3) % inputs.size();
            if (tok.encode(inputs[idx]) != expected[idx]) ++thread_bad[size_t(t)];
          }
        }
      });
    }
    for (auto& th : threads) th.join();
    int total_bad = 0;
    for (int b : thread_bad) total_bad += b;
    check(total_bad == 0, "concurrent multi-thread encode matches reference");
    // NFC fast path on pure ASCII and composition on decomposed UTF-8.
    check(Tokenizer::nfc("hello world 123\n") == "hello world 123\n", "nfc keeps ASCII identical");
    check(Tokenizer::nfc("cafe\xCC\x81") == "caf\xC3\xA9", "nfc composes combining acute accent");
  }

  // ignore_merges: true must not match added/special tokens through by_bytes_ on untrusted client text.
  {
    Vocabulary v("STOPab", 20, &rng);
    const int32_t stop_id = static_cast<int32_t>(v.tokens.size());
    v.added.push_back({"STOP", true});
    v.tokens.push_back("STOP");
    const Vocabulary::Tables t = v.tables();
    const std::string header =
        R"({"arch": "test", "tensors": {}, "tokenizer": {"pretokenizer": "qwen35", "normalizer": "none", "ignore_merges": true, "n_tokens": )" +
        std::to_string(t.n_tokens) + R"(}, "blobs": {"tokenizer.vocab": {"offset": 0, "nbytes": )" +
        std::to_string(t.vocab.size()) + R"(}, "tokenizer.merges": {"offset": )" + std::to_string(t.vocab.size()) +
        R"(, "nbytes": )" + std::to_string(t.merges.size()) + R"(}, "tokenizer.added": {"offset": )" +
        std::to_string(t.vocab.size() + t.merges.size()) + R"(, "nbytes": )" + std::to_string(t.added.size()) + "}}}";
    ModelFile file;
    Tokenizer tok;
    check(write_file(path, flyx_file(header, t.vocab + t.merges + t.added)) && file.open(path, &err) &&
              tok.load(file, &err),
          "ignore_merges vocabulary loads: " + err);
    const std::vector<TextSpan> none;
    const auto untrusted = tok.encode("STOP", &none);
    const auto trusted = tok.encode("STOP");
    check(trusted == std::vector<int32_t>{stop_id}, "raw prompt still emits special token STOP");
    check(untrusted != std::vector<int32_t>{stop_id} && tok.decode(untrusted) == "STOP",
          "ignore_merges does not bypass special-token isolation via by_bytes_");
  }

  // Tables that are wrong: the tokenizer says so when it loads, and reads nothing outside them.
  {
    Vocabulary v("ab", 20, &rng);
    const Vocabulary::Tables good = v.tables();
    auto loads = [&](const Vocabulary::Tables& t, std::string* why) {
      ModelFile file;
      Tokenizer tok;
      return Vocabulary::write(path, t) && file.open(path, why) && tok.load(file, why);
    };
    check(loads(good, &err), "the tables as they are: " + err);
    Vocabulary::Tables t = good;
    const uint32_t far = 1u << 30;
    t.vocab.replace(4 * 300, 4, reinterpret_cast<const char*>(&far), 4);  // token 299 ends a gigabyte on
    check(!loads(t, &err), "a token whose bytes lie outside the vocabulary table");
    t = good;
    t.vocab.resize(4 * 100);  // fewer offsets than tokens
    check(!loads(t, &err), "a vocabulary table shorter than its offsets");
    t = good;
    t.n_tokens = 0;
    check(!loads(t, &err), "no tokens");
    t = good;
    const uint32_t beyond = static_cast<uint32_t>(good.n_tokens);
    t.merges.replace(8, 4, reinterpret_cast<const char*>(&beyond), 4);  // the first merge makes a token past the last
    check(!loads(t, &err), "a merge that makes a token the vocabulary does not have");
    t = good;
    t.added = R"({"id": 3})";
    check(!loads(t, &err), "added tokens that are not a list");
    t = good;
    t.added = R"([{"content": "<x>"}])";
    check(!loads(t, &err), "an added token without an id");
    t = good;
    t.added = "[";
    check(!loads(t, &err), "added tokens that are not JSON");
  }

  // Inkling-2 (GPT-4o / o200k) pre-tokenizer splits: camelCase, contractions, 1..3 digit chunks, punctuation + /.
  {
    auto split = [](std::string_view s) { return Tokenizer::pretokenize(s, Pretokenizer::kInkling2); };
    check(split("HelloWorld's 12345") == std::vector<std::string>({"Hello", "World's", " ", "123", "45"}),
          "inkling2 pretokenizer: camelCase + contraction + 3-digit numbers");
    check(split("HTTP_Server don't\n/foo") == std::vector<std::string>({"HTTP", "_Server", " don't", "\n", "/foo"}),
          "inkling2 pretokenizer: acronym + underscore + contraction + newline + slash");
    check(split("HTTPServer") == std::vector<std::string>({"HTTPServer"}),
          "inkling2 pretokenizer: upper run followed by lower stays one piece");
    check(split("!!!\n/a") == std::vector<std::string>({"!!!\n/", "a"}),
          "inkling2 pretokenizer: punctuation consumes trailing newline and slash");
    check(split("你好World 42") == std::vector<std::string>({"你好World", " ", "42"}),
          "inkling2 pretokenizer: CJK (Lo) + TitleCase stays one U*L+ piece");
    check(split("你好ABC") == std::vector<std::string>({"你好", "ABC"}),
          "inkling2 pretokenizer: CJK (Lo) followed by uppercase backtracks U*L+ before Lu");
  }

  std::printf("merge: %d texts against the rule\n", texts);
  return flyx::test::finish("merge_test");
}
