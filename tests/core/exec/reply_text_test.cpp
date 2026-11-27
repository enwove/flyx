// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// ReplyText (core/exec/reply_text.h): a reply's tokens as text, shown as its
// characters complete, ending before the first stop string, and the tokens
// the shown text covers.  Overlapping stop strings end the reply where they
// would a byte at a time, however the tokens are grouped (a speculative
// round's width).  No engine and no device: a tokenizer only.
//
//   reply_text_test MODEL.flyx
#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "core/exec/reply_text.h"
#include "core/modelfile/modelfile.h"
#include "core/tokenizer/tokenizer.h"
#include "tests/common/check.h"

using namespace flyx;

namespace {

// Whole UTF-8 characters only.
bool whole_utf8(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}

struct Seen {
  std::string text;
  std::vector<int32_t> tokens;
  bool whole = true;
};

// `text` as a reply, pushed `step` tokens at a time: the text shown is `want`
// (all of it by the end), never more than `want` on the way, in whole
// characters, and the tokens handed out are exactly those whose text was
// shown in full.
void run(const Tokenizer& tok, const std::string& text, const std::vector<std::string>& stop, int step,
         const std::string& want, const std::string& want_stop) {
  const std::string what = "'" + text + "' stop " + (stop.empty() ? "-" : "'" + stop[0] + "'") + ", " +
                           std::to_string(step) + " at a time: ";
  const auto ids = tok.encode(text);
  Seen seen;
  ReplyText reply(tok, stop, [&](std::string_view t, const int32_t* toks, int n) {
    seen.text += t;
    seen.tokens.insert(seen.tokens.end(), toks, toks + n);
    seen.whole = seen.whole && whole_utf8(t);
    return true;
  });
  bool within = true;
  for (size_t i = 0; i < ids.size(); i += size_t(step)) {
    reply.push(ids.data() + i, static_cast<int>(std::min(size_t(step), ids.size() - i)));
    within = within && want.compare(0, seen.text.size(), seen.text) == 0;
  }
  reply.end();
  check(seen.text == want, what + "shows '" + seen.text + "', want '" + want + "'");
  check(within, what + "showed text beyond the stop on the way");
  check(seen.whole, what + "showed part of a character");
  check(reply.stop_string() == want_stop, what + "stop string '" + reply.stop_string() + "'");
  // Published: the tokens whose text lies inside what was shown.
  size_t bytes = 0, covered = 0;
  for (int32_t id : ids) {
    bytes += tok.token_bytes(id).size();
    if (bytes > want.size()) break;
    ++covered;
  }
  check(reply.published() == covered && seen.tokens == std::vector<int32_t>(ids.begin(), ids.begin() + covered),
        what + "published " + std::to_string(reply.published()) + " tokens, want " + std::to_string(covered));
  if (want_stop.empty()) check(reply.tokens() == ids, what + "keeps every token pushed");
}


// The rule, a byte at a time: the reply ends before the stop string that completes first; of those completing at the
// same byte, before the one that starts first.  What is shown, and the stop string ("" for none).
std::pair<std::string, std::string> ends_at(const std::string& text, const std::vector<std::string>& stop) {
  for (size_t k = 1; k <= text.size(); ++k) {
    size_t start = std::string::npos;
    std::string which;
    for (const auto& s : stop)
      if (!s.empty() && s.size() <= k && text.compare(k - s.size(), s.size(), s) == 0 && k - s.size() < start) {
        start = k - s.size();
        which = s;
      }
    if (start != std::string::npos) return {text.substr(0, start), which};
  }
  return {text, ""};
}

// A token whose text is exactly `bytes` (-1: none).
int32_t token_of(const Tokenizer& tok, const std::string& bytes) {
  for (int32_t id = 0; id < tok.n_tokens(); ++id)
    if (!tok.special(id) && tok.token_bytes(id) == bytes) return id;
  return -1;
}

// `ids` pushed in every grouping there is (2^(n-1) of them: a speculative round of any width at any point): each shows
// the same text, ends on the same stop string and publishes the same tokens, those of the rule a byte at a time.
void every_grouping(const Tokenizer& tok, const std::vector<int32_t>& ids, const std::vector<std::string>& stop,
                    const std::string& what) {
  std::string text;
  for (int32_t id : ids) text += tok.token_bytes(id);
  const auto [want, want_stop] = ends_at(text, stop);
  size_t bytes = 0, covered = 0;
  for (int32_t id : ids) {
    bytes += tok.token_bytes(id).size();
    if (bytes > want.size()) break;
    ++covered;
  }
  const std::vector<int32_t> want_tokens(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(covered));
  const size_t n = ids.size();
  int differ = 0;
  for (uint32_t cuts = 0; cuts < (1u << (n - 1)); ++cuts) {  // bit i: a push ends after token i
    Seen seen;
    ReplyText reply(tok, stop, [&](std::string_view t, const int32_t* toks, int k) {
      seen.text += t;
      seen.tokens.insert(seen.tokens.end(), toks, toks + k);
      seen.whole = seen.whole && whole_utf8(t);
      return true;
    });
    std::string groups;
    for (size_t i = 0, from = 0; i < n; ++i) {
      if (i + 1 < n && !(cuts >> i & 1)) continue;
      reply.push(ids.data() + from, static_cast<int>(i + 1 - from));
      groups += (groups.empty() ? "" : "|") + std::to_string(i + 1 - from);
      from = i + 1;
    }
    reply.end();
    const bool same = seen.text == want && reply.stop_string() == want_stop && seen.tokens == want_tokens &&
                      reply.published() == covered && seen.whole;
    if (!same && differ++ < 4)
      check(false, what + ", pushes of " + groups + " tokens: shows '" + seen.text + "' (want '" + want +
                       "'), stop '" + reply.stop_string() + "' (want '" + want_stop + "'), published " +
                       std::to_string(reply.published()) + " (want " + std::to_string(covered) + ")");
  }
  check(differ == 0, what + ": every grouping of " + std::to_string(n) + " tokens shows '" + want + "' and stops at '" +
                         want_stop + "' (" + std::to_string(differ) + " differ)");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: reply_text_test MODEL.flyx\n");
    return 2;
  }
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(argv[1], &err) || !tok.load(file, &err)) {
    std::printf("load failed: %s\n", err.c_str());
    return 1;
  }

  const std::string text = "alpha beta gamma delta, caf\xC3\xA9 \xE4\xBD\xA0\xE5\xA5\xBD \xF0\x9F\x99\x82 omega";
  struct Case {
    std::vector<std::string> stop;
    std::string stop_found;  // "" = the whole text is shown
  };
  const std::vector<Case> cases = {
      {{}, ""},                        // no stop strings
      {{"gamma"}, "gamma"},            // a stop string in the middle
      {{"mm"}, "mm"},                  // inside a word: the tokens around it are not published
      {{"ta, c"}, "ta, c"},            // across words and punctuation
      {{"gamut"}, ""},                 // a false start: "gam" is held back, then shown
      {{"omega!"}, ""},                // a false start at the very end, shown when the reply ends
      {{"delta", "beta"}, "beta"},     // the first one in the text wins
      {{"\xE5\xA5\xBD"}, "\xE5\xA5\xBD"},  // a stop string of one multi-byte character
      {{""}, ""},                      // an empty stop string never matches
      {{"alpha"}, "alpha"},            // at the very start: nothing is shown
  };
  for (const auto& c : cases) {
    const size_t at = c.stop_found.empty() ? std::string::npos : text.find(c.stop_found);
    const std::string want = at == std::string::npos ? text : text.substr(0, at);
    for (int step : {1, 2, 5, 1000}) run(tok, text, c.stop, step, want, c.stop_found);
  }

  // Overlapping stop strings, the same token ids grouped every way a round can group them.
  {
    std::vector<int32_t> letter(26, -1);
    bool found = true;
    for (char c = 'a'; c <= 'z'; ++c)
      found = (letter[size_t(c - 'a')] = token_of(tok, std::string(1, c))) >= 0 && found;
    check(found, "the tokenizer has a token for each letter");
    auto spell = [&](const std::string& word) {
      std::vector<int32_t> ids;
      for (char c : word) ids.push_back(letter[size_t(c - 'a')]);
      return ids;
    };
    if (found) {
      // The case that was wrong: one id at a time saw "b" end first, all three at once preferred "abc", which starts
      // first (empty text, no token published).
      every_grouping(tok, spell("abc"), {"abc", "b"}, "'abc' stops 'abc', 'b'");
      every_grouping(tok, spell("abc"), {"b", "abc"}, "'abc' stops 'b', 'abc'");
      every_grouping(tok, spell("abcd"), {"abcd", "bc"}, "'abcd' stops 'abcd', 'bc' (inside, ending first)");
      every_grouping(tok, spell("abcd"), {"bcd", "abc"}, "'abcd' stops 'bcd', 'abc' (overlapping)");
      // Stop strings that share a prefix: the shorter ends first; a false start of the longer.
      every_grouping(tok, spell("abcde"), {"abcd", "ab"}, "'abcde' stops 'abcd', 'ab' (shared prefix)");
      every_grouping(tok, spell("abce"), {"abcd", "abce"}, "'abce' stops 'abcd', 'abce' (shared prefix, false start)");
      every_grouping(tok, spell("xabyab"), {"aby", "abz"}, "'xabyab' stops 'aby', 'abz'");
      // Ending at the same byte: the one that starts first.
      every_grouping(tok, spell("abc"), {"bc", "abc"}, "'abc' stops 'bc', 'abc' (same end)");
      every_grouping(tok, spell("abcd"), {"d", "cd", "bcd"}, "'abcd' stops 'd', 'cd', 'bcd' (same end)");
      every_grouping(tok, spell("abcabc"), {"ca", "bcab", "cab"}, "'abcabc' stops 'ca', 'bcab', 'cab'");
    }
    // A character split across two tokens: nothing of it shows until it is whole, and a stop string ending with it
    // ends where it completes.
    const int32_t lead = token_of(tok, "\xC3"), cont = token_of(tok, "\xA9");
    check(lead >= 0 && cont >= 0, "the tokenizer has a token for each byte of a two-byte character");
    if (found && lead >= 0 && cont >= 0) {
      const std::vector<int32_t> cafe = {letter['c' - 'a'], letter['a' - 'a'], letter['f' - 'a'], lead, cont,
                                         letter['s' - 'a']};
      every_grouping(tok, cafe, {"\xC3\xA9s", "f\xC3\xA9"}, "'caf\xC3\xA9s' stops '\xC3\xA9s', 'f\xC3\xA9'");
      every_grouping(tok, cafe, {"\xC3\xA9", "af\xC3\xA9"}, "'caf\xC3\xA9s' stops '\xC3\xA9', 'af\xC3\xA9' (same end)");
      every_grouping(tok, cafe, {"\xC3\xA9s", "s"}, "'caf\xC3\xA9s' stops '\xC3\xA9s', 's' (same end)");
      every_grouping(tok, cafe, {"fa"}, "'caf\xC3\xA9s' no stop");
    }
    // The tokenizer's own tokens of a phrase.
    const auto ids = tok.encode("beta gamma");
    check(ids.size() >= 3 && ids.size() <= 14, "a phrase of a few tokens (" + std::to_string(ids.size()) + ")");
    if (ids.size() >= 3 && ids.size() <= 14)
      every_grouping(tok, ids, {"beta gamma", "ta g", "a gam"}, "'beta gamma' overlapping stops");
  }
  // The sink says stop: reported, for the caller to cancel.
  {
    const auto ids = tok.encode(text);
    int calls = 0;
    ReplyText reply(tok, {}, [&](std::string_view, const int32_t*, int) { return ++calls < 2; });
    for (int32_t id : ids) {
      if (reply.refused()) break;
      reply.push(&id, 1);
    }
    check(reply.refused() && calls == 2, "a sink that returns false is reported");
  }
  // Tokens without text at the very end (ids past the tokenizer) are still handed out.
  {
    auto ids = tok.encode("alpha");
    ids.push_back(tok.n_tokens() + 3);
    Seen seen;
    int empty_calls = 0;
    ReplyText reply(tok, {}, [&](std::string_view t, const int32_t* toks, int n) {
      seen.text += t;
      seen.tokens.insert(seen.tokens.end(), toks, toks + n);
      empty_calls += t.empty();
      return true;
    });
    reply.push(ids.data(), static_cast<int>(ids.size()));
    reply.end();
    check(seen.text == "alpha" && seen.tokens == ids && reply.published() == ids.size(),
          "a token without text at the end is published");
  }
  // No sink at all.
  {
    const auto ids = tok.encode(text);
    const std::vector<std::string> stop = {"delta"};
    ReplyText reply(tok, stop, TextSink());
    reply.push(ids.data(), static_cast<int>(ids.size()));
    reply.end();
    check(reply.stop_string() == "delta" && reply.published() > 0 && reply.published() < ids.size(),
          "without a sink the stop string still ends the reply");
  }

  return flyx::test::finish("reply_text_test");
}
