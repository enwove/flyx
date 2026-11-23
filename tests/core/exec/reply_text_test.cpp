// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// ReplyText (core/exec/reply_text.h): a reply's tokens as text, shown as its
// characters complete, ending before the first stop string, and the tokens
// the shown text covers.  No engine and no device: a tokenizer only.
//
//   reply_text_test MODEL.flyx
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "core/exec/reply_text.h"
#include "core/modelfile/modelfile.h"
#include "core/tokenizer/tokenizer.h"

using namespace flyx;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  if (ok) return;
  ++failures;
  std::printf("FAIL %s\n", what.c_str());
}

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

  std::printf("reply_text: %s\n", failures ? "FAILED" : "ok");
  return failures ? 1 : 0;
}
