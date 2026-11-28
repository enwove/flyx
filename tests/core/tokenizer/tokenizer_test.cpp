// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Tokenizer vs HuggingFace `tokenizers` on recorded cases.
//
//   tokenizer_test MODEL.flyx CASES.json
//
// CASES.json: [{"text": ..., "ids": [...]}, ...] produced by the HF tokenizer
// (tests/e2e/make_tiny.py for the tiny model,
// tests/e2e/check_real_model.py checks a real one); "prompt_ids", where a
// case has them: the ids with the special tokens HF's post-processor adds,
// which a raw prompt's tokens must be.
#include <cstdio>
#include <fstream>

#include "core/modelfile/modelfile.h"
#include "core/tokenizer/tokenizer.h"
#include "nlohmann/json.hpp"

using namespace flyx;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: tokenizer_test MODEL.flyx CASES.json\n");
    return 2;
  }
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(argv[1], &err) || !tok.load(file, &err)) {
    std::fprintf(stderr, "%s\n", err.c_str());
    return 1;
  }
  std::ifstream f(argv[2]);
  auto cases = nlohmann::json::parse(f);
  int bad = 0, n = 0;
  for (const auto& c : cases) {
    std::string text = c["text"];
    auto want = c["ids"].get<std::vector<int32_t>>();
    auto got = tok.encode(text);
    ++n;
    bool ok = got == want;
    std::string round = tok.decode(got);
    const bool nfc = file.header().at("tokenizer").value("normalizer", "nfc") == "nfc";
    std::string want_text = nfc ? Tokenizer::nfc(text) : text;
    if (file.header().at("tokenizer").value("pretokenizer", "") == "gemma4")  // a U+2581 in the text: a space back
      for (size_t at; (at = want_text.find("\xE2\x96\x81")) != std::string::npos;) want_text.replace(at, 3, " ");
    if (ok && round != want_text) {
      // Decoding must give back the normalized text.
      std::printf("decode mismatch for %s\n", nlohmann::json(text).dump().c_str());
      ok = false;
    }
    if (ok && c.contains("prompt_ids")) {
      // As a raw prompt: the special tokens the post-processor adds (MiniCPM5's <s>) before the text's.
      want = c["prompt_ids"].get<std::vector<int32_t>>();
      got = tok.encode_prompt(text);
      ok = got == want;
      if (!ok) std::printf("raw prompt mismatch for %s\n", nlohmann::json(text).dump().c_str());
    }
    if (!ok) {
      ++bad;
      std::printf("MISMATCH %s\n  want:", nlohmann::json(text).dump().c_str());
      for (int id : want) std::printf(" %d", id);
      std::printf("\n  got: ");
      for (int id : got) std::printf(" %d", id);
      std::printf("\n");
    }
  }
  std::printf("%d / %d cases match\n", n - bad, n);
  if (n == 0) std::printf("FAIL %s holds no cases: nothing was compared\n", argv[2]);
  return bad || n == 0 ? 1 : 0;
}
