// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// How many tokens rounds of more rows would take on a reply, drafting by lookup:
//
//   wide_rounds MODEL PROMPT.txt REPLY.txt [--widths 15,31,63]
//
// REPLY.txt: a greedy reply to PROMPT.txt (`flyx run`'s output), which is the same whatever the rounds verify; both
// as the model's tokenizer encodes them.  A lookup drafter of up to W drafts a round is replayed on it: each round
// drafts from the sequence so far (prompt and reply up to there), takes the drafts that match the reply and one
// token more.  Prints, per W, the rounds and tokens a round, and how many rounds verified how many rows (1: no
// earlier occurrence to draft from, where the engine runs its drafter or plain decoding instead).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <algorithm>
#include <string>
#include <vector>

#include "core/modelfile/modelfile.h"
#include "core/speculation/lookup.h"
#include "core/tokenizer/tokenizer.h"

using namespace flyx;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: wide_rounds MODEL PROMPT.txt REPLY.txt [--widths 15,31,63]\n");
    return 2;
  }
  std::vector<int> widths = {15, 31, 63};
  for (int i = 4; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--widths") {
      widths.clear();
      for (size_t a = 0; a < v.size();) {
        size_t b = v.find(',', a);
        if (b == std::string::npos) b = v.size();
        widths.push_back(std::atoi(v.substr(a, b - a).c_str()));
        a = b + 1;
      }
    }
  }
  ModelFile file;
  Tokenizer tok;
  std::string err;
  if (!file.open(argv[1], &err) || !tok.load(file, &err)) return std::fprintf(stderr, "%s\n", err.c_str()), 1;
  auto read = [](const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  };
  const std::vector<int32_t> prompt = tok.encode(read(argv[2])), reply = tok.encode(read(argv[3]));
  std::printf("prompt %zu tokens, reply %zu tokens\n", prompt.size(), reply.size());
  for (int W : widths) {
    LookupDrafter d;
    std::vector<int32_t> seq = prompt, drafts;
    std::map<int, long> by_rows;  // rows a round verified -> rounds
    long rounds = 0, drafted_rounds = 0, drafted_tokens = 0;
    size_t k = 0;
    while (k < reply.size()) {
      d.sync(seq);
      d.propose(seq, W, &drafts);
      size_t take = 0;
      while (take < drafts.size() && k + take < reply.size() && drafts[take] == reply[k + take]) ++take;
      const size_t n = std::min(reply.size() - k, take + 1);
      seq.insert(seq.end(), reply.begin() + long(k), reply.begin() + long(k + n));
      k += n;
      ++rounds;
      ++by_rows[int(drafts.size()) + 1];
      if (!drafts.empty()) ++drafted_rounds, drafted_tokens += long(n);
    }
    std::printf("W %2d: %5ld rounds, %.2f tokens a round; lookup rounds %ld (%.2f tokens each); rows:", W, rounds,
                double(reply.size()) / double(rounds), drafted_rounds,
                drafted_rounds ? double(drafted_tokens) / double(drafted_rounds) : 0.0);
    for (const auto& [r, c] : by_rows) std::printf(" %d:%ld", r, c);
    std::printf("\n");
  }
  return 0;
}
