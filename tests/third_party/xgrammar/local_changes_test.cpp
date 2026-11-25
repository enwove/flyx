// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The differences from upstream in third_party/xgrammar (README.flyx.md there), on vocabularies made by hand.  At
// every step of a reply and for every token of the vocabulary, the mask's bit, whether the matcher takes the token
// and what the grammar allows (worked out by hand) agree, and rolling an accepted token back restores the mask.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#include "dlpack/dlpack.h"
#include "xgrammar/xgrammar.h"

namespace {

using namespace xgrammar;
using Ids = std::vector<int32_t>;
int failures = 0;

std::vector<uint32_t> mask(GrammarMatcher& matcher, int vocab_size) {
  int64_t words = GetBitmaskSize(vocab_size);
  std::vector<uint32_t> bits(words);
  DLTensor t{};
  t.data = bits.data();
  t.device = DLDevice{kDLCPU, 0};
  t.ndim = 1;
  t.dtype = GetBitmaskDLType();
  t.shape = &words;
  matcher.FillNextTokenBitmask(&t);
  return bits;
}

struct Step {
  int32_t take;
  Ids allowed;
};

void reply(const char* name, const CompiledGrammar& grammar, const std::vector<Step>& steps, const Ids& at_end) {
  GrammarMatcher matcher(grammar);
  const auto tokenizer = grammar.GetTokenizerInfo();
  const auto& vocab = tokenizer.GetDecodedVocab();
  auto check_state = [&](size_t step, const Ids& allowed) {
    const auto before = mask(matcher, tokenizer.GetVocabSize());
    for (int id = 0; id < tokenizer.GetVocabSize(); ++id) {
      const bool bit = (before[id >> 5] >> (id & 31)) & 1u;
      const bool took = matcher.AcceptToken(id);
      const bool want = std::find(allowed.begin(), allowed.end(), id) != allowed.end();
      if (bit != took || took != want) {
        std::printf("FAIL %s step %zu, token '%s': mask %d, acceptance %d, expected %d\n",
                    name, step, vocab[id].c_str(), bit, took, want);
        ++failures;
      }
      if (took) {
        matcher.Rollback(1);
        if (mask(matcher, tokenizer.GetVocabSize()) != before) {
          std::printf("FAIL %s step %zu: rollback across '%s' changed the mask\n", name, step, vocab[id].c_str());
          ++failures;
        }
      }
    }
  };
  for (size_t i = 0; i < steps.size(); ++i) {
    check_state(i, steps[i].allowed);
    if (!matcher.AcceptToken(steps[i].take)) {
      std::printf("FAIL %s step %zu rejected '%s'\n", name, i, vocab[steps[i].take].c_str());
      ++failures;
      return;
    }
  }
  if (!matcher.IsCompleted() || matcher.IsTerminated() != at_end.empty()) {
    std::printf("FAIL %s ended in the wrong state\n", name);
    ++failures;
  }
  if (!matcher.IsTerminated()) check_state(steps.size(), at_end);
}

void subtree_ranges() {
  enum { Eos, A, B, C, Ab, Acab, Ac };
  std::vector<std::string> tokens{"", "a", "b", "c", "ab", "acab"};
  auto check = [&](const char* name, const Ids& allowed) {
    TokenizerInfo info(tokens, VocabType::RAW, std::nullopt, Ids{Eos});
    GrammarCompiler compiler(info, 1);
    reply(name, compiler.CompileGrammar(R"(root ::= "acab")"), {{Acab, allowed}}, {Eos});
  };
  check("subtree without ac", {A, Acab});
  tokens.push_back("ac");
  check("subtree with ordinary ac", {A, Ac, Acab});
}

}  // namespace

int main() {
  try {
    subtree_ranges();
  } catch (const std::exception& e) {
    std::printf("FAIL %s\n", e.what());
    return 1;
  }
  return failures ? 1 : 0;
}
