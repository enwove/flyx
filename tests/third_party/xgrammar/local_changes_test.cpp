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

enum { X, Y, Bang, Quote, Open, Close, LongOpen, Stop };
const std::vector<std::string> vocab{"x", "y", "!", "\"", "xx", "yy", "xxy", ""};
const Ids markers{Open, Close, LongOpen};

std::string structural_tag(const char* format) { return std::string("{\"format\":") + format + "}"; }

void atomic_literals() {
  TokenizerInfo info(vocab, VocabType::RAW, std::nullopt, Ids{Stop}, false, markers);
  GrammarCompiler compiler(info, 1);
  auto compile = [&](const char* format) { return compiler.CompileStructuralTag(structural_tag(format)); };

  const Ids text{X, Y, Quote, Close};
  reply("free text", compile(R"({"type":"tag","begin":"xx","end":"yy",
        "content":{"type":"any_text","excludes":["!"]}})"),
        {{Open, {Open}}, {X, text}, {X, text}, {Y, text}, {Y, text}, {Close, text}}, {Stop});
  reply("character budget", compile(R"({"type":"tag","begin":"xx","end":"yy",
        "content":{"type":"any_text","max_chars":1}})"),
        {{Open, {Open}}, {X, {X, Y, Bang, Quote, Close}}, {Close, {Close}}}, {Stop});
  reply("wildcard token", compile(R"({"type":"tag","begin":"xx","end":"yy","content":{"type":"exclude_token"}})"),
        {{Open, {Open}}, {X, {X, Y, Bang, Quote}}, {Close, {Close}}}, {Stop});

  const Ids string_body{X, Y, Bang, Quote}, character{X, Y, Bang};
  reply("JSON string", compile(R"({"type":"tag","begin":"xx","end":"yy",
        "content":{"type":"json_schema","json_schema":{"type":"string"}}})"),
        {{Open, {Open}}, {Quote, {Quote}}, {Y, string_body}, {Y, string_body}, {Quote, string_body}, {Close, {Close}}},
        {Stop});
  reply("length-bounded JSON string", compile(R"({"type":"tag","begin":"xx","end":"yy",
        "content":{"type":"json_schema","json_schema":{"type":"string","minLength":2,"maxLength":2}}})"),
        {{Open, {Open}}, {Quote, {Quote}}, {Y, character}, {Y, character}, {Quote, {Quote}}, {Close, {Close}}}, {Stop});

  reply("separator", compile(R"({"type":"tags_with_separator","at_least_one":true,"separator":"xx",
        "tags":[{"begin":"!","content":{"type":"const_string","value":"x"},"end":"!"}]})"),
        {{Bang, {Bang}}, {X, {X}}, {Bang, {Bang}}, {Open, {Open, Stop}}, {Bang, {Bang}}, {X, {X}}, {Bang, {Bang}}},
        {Open, Stop});
  reply("mixed literal", compile(R"({"type":"const_string","value":"!yy!xxyxx!","content":"annotation"})"),
        {{Bang, {Bang}}, {Close, {Close}}, {Bang, {Bang}}, {LongOpen, {LongOpen}}, {Open, {Open}}, {Bang, {Bang}}},
        {Stop});
  const auto ends = compile(R"({"type":"tag","begin":"x","end":["yy","!xx"],
      "content":{"type":"const_string","value":"y"}})");
  reply("first end", ends, {{X, {X}}, {Y, {Y}}, {Close, {Close, Bang}}}, {Stop});
  reply("second end", ends, {{X, {X}}, {Y, {Y}}, {Bang, {Close, Bang}}, {Open, {Open}}}, {Stop});
  reply("triggered tag end", compile(R"({"type":"triggered_tags","triggers":["!"],"stop_after_first":true,
        "tags":[{"begin":"!","content":{"type":"const_string","value":"x"},"end":"yy"}]})"),
        {{Bang, {X, Y, Bang, Quote, Stop}}, {X, {X}}, {Close, {Close}}}, {Stop});

  auto refused = [&](const char* field, const char* format) {
    try {
      compile(format);
      std::printf("FAIL accepted atomic token text in %s\n", field);
    } catch (const std::exception& e) {
      const std::string error = e.what();
      if (error.find(field) != std::string::npos && error.find("atomic token") != std::string::npos) return;
      std::printf("FAIL unexpected %s error: %s\n", field, e.what());
    }
    ++failures;
  };
  refused("trigger", R"({"type":"triggered_tags","triggers":["!","xyy!"],"at_least_one":true,"stop_after_first":true,
          "tags":[{"begin":"!","content":{"type":"const_string","value":"x"},"end":"!"}]})");
  refused("tag begin", R"({"type":"triggered_tags","triggers":["!"],"tags":[
          {"begin":"!xx","content":{"type":"const_string","value":"x"},"end":"!"}]})");
  refused("dispatch key", R"({"type":"dispatch","rules":[["!xx",{"type":"const_string","value":"x"}]]})");
}

void stop_marker() {
  TokenizerInfo info(vocab, VocabType::RAW, std::nullopt, Ids{Stop, Close}, false, markers);
  GrammarCompiler compiler(info, 1);
  // Both alternatives use the stop marker. Only the second can end on it.
  auto grammar = compiler.CompileStructuralTag(structural_tag(R"({"type":"or","elements":[
      {"type":"const_string","value":"yy!"},{"type":"const_string","value":"xyy"}]})"));
  reply("stop marker", grammar, {{X, {X}}, {Close, {Close}}}, {});
}

void byte_literals() {
  TokenizerInfo info(vocab, VocabType::RAW, std::nullopt, Ids{Stop});
  GrammarCompiler compiler(info, 1);
  auto tag = compiler.CompileStructuralTag(structural_tag(
      R"({"type":"tag","begin":"xx","end":"yy","content":{"type":"const_string","value":"!"}})"));
  reply("byte tag", tag, {{X, {X, Open}}, {X, {X}}, {Bang, {Bang}}, {Close, {Y, Close}}}, {Stop});
  auto triggered = compiler.CompileStructuralTag(structural_tag(R"({"type":"triggered_tags","triggers":["xx"],
      "stop_after_first":true,"tags":[{"begin":"xx","end":"yy","content":{"type":"const_string","value":"!"}}]})"));
  const Ids gap{X, Y, Bang, Quote, Open, Close, Stop};
  reply("byte trigger", triggered, {{Y, gap}, {Open, gap}, {Bang, {Bang}}, {Y, {Y, Close}}, {Y, {Y}}}, {Stop});
}

void subtree_ranges() {
  enum { Eos, A, B, C, Ab, Acab, Ac };
  std::vector<std::string> tokens{"", "a", "b", "c", "ab", "acab"};
  auto check = [&](const char* name, const Ids& atomic, const Ids& allowed) {
    TokenizerInfo info(tokens, VocabType::RAW, std::nullopt, Ids{Eos}, false, atomic);
    GrammarCompiler compiler(info, 1);
    reply(name, compiler.CompileGrammar(R"(root ::= "acab")"), {{Acab, allowed}}, {Eos});
  };
  check("subtree without ac", {}, {A, Acab});
  tokens.push_back("ac");
  check("subtree with ordinary ac", {}, {A, Ac, Acab});
  check("subtree with atomic ac", {Ac}, {A, Acab});
}

}  // namespace

int main() {
  try {
    subtree_ranges();
    atomic_literals();
    stop_marker();
    byte_literals();
  } catch (const std::exception& e) {
    std::printf("FAIL %s\n", e.what());
    return 1;
  }
  return failures ? 1 : 0;
}
