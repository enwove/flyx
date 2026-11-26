// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/grammar/grammar.h"

#include <algorithm>
#include <deque>
#include <exception>
#include <mutex>
#include <optional>
#include <utility>

#include "core/tokenizer/tokenizer.h"
#include "dlpack/dlpack.h"
#include "xgrammar/xgrammar.h"

namespace flyx {

struct GrammarMatcher::Impl {
  xgrammar::GrammarMatcher m;
  int words;
  const std::vector<bool>* never;  // tokens no grammar takes (Grammar::Impl::never)
};

GrammarMatcher::GrammarMatcher(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GrammarMatcher::~GrammarMatcher() = default;

bool GrammarMatcher::accept(int32_t token) {
  // Special tokens no reply is written with and padding ids never match: answered here, so XGrammar does not log a
  // warning for each one a model emits.
  if (token < 0 || size_t(token) >= impl_->never->size() || (*impl_->never)[size_t(token)]) return false;
  return impl_->m.AcceptToken(token);
}

void GrammarMatcher::allowed(uint32_t* words) {
  // XGrammar writes an int32 bitmask through DLPack; the bits are ours.
  int64_t shape[1] = {impl_->words};
  DLTensor t{};
  t.data = words;
  t.device = DLDevice{kDLCPU, 0};
  t.ndim = 1;
  t.dtype = DLDataType{kDLInt, 32, 1};
  t.shape = shape;
  impl_->m.FillNextTokenBitmask(&t);
}

bool GrammarMatcher::can_end() { return impl_->m.IsTerminated() || impl_->m.IsCompleted(); }

struct Grammar::Impl {
  std::unique_ptr<xgrammar::TokenizerInfo> info;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler;
  int words = 0;
  std::vector<bool> never;  // special tokens (neither stop tokens nor a reply's markers) and padding ids
  // The cache, held only to look in it or add to it, so a lookup never waits for a compile; compile_mu: one compile
  // at a time (sessions of one engine may run on several threads, and requests are made warm ahead on others).
  std::mutex mu, compile_mu;
  std::deque<std::pair<std::string, xgrammar::CompiledGrammar>> compiled;  // newest first

  // The compiled grammar of a tag, moved to the front, or none.
  std::optional<xgrammar::CompiledGrammar> cached(const std::string& tag) {
    std::lock_guard<std::mutex> lock(mu);
    auto it = std::find_if(compiled.begin(), compiled.end(), [&](const auto& c) { return c.first == tag; });
    if (it == compiled.end()) return std::nullopt;
    if (it != compiled.begin()) {
      auto entry = std::move(*it);
      compiled.erase(it);
      compiled.push_front(std::move(entry));
    }
    return compiled.front().second;
  }
};

Grammar::Grammar() : impl_(std::make_unique<Impl>()) {}
Grammar::~Grammar() = default;

void Grammar::init(const Tokenizer& tok, int vocab, std::vector<int32_t> stop_ids,
                   const std::vector<int32_t>& markers) {
  std::vector<std::string> text(static_cast<size_t>(vocab));
  impl_->never.assign(size_t(vocab), true);
  auto add_token = [&](int32_t id) {
    text[size_t(id)] = tok.token_bytes(id);  // "" = never matches text
    impl_->never[size_t(id)] = false;
  };
  for (int32_t id = 0; id < vocab && id < tok.n_tokens(); ++id)
    if (!tok.special(id)) add_token(id);
  // A reply's own markers: without them a format that writes its sections with control tokens could only spell
  // them, a character at a time, in ordinary tokens the model never saw there.  Each is one token to XGrammar
  // (atomic): taken where a grammar's delimiter spells it, and no part of any text.
  std::vector<int32_t> atomic;
  for (int32_t id : markers) {
    if (id < 0 || id >= vocab || id >= tok.n_tokens() || !tok.special(id) || tok.token_bytes(id).empty()) continue;
    add_token(id);
    atomic.push_back(id);
  }
  for (int32_t id : stop_ids)
    if (id >= 0 && id < vocab) impl_->never[size_t(id)] = false;
  impl_->info = std::make_unique<xgrammar::TokenizerInfo>(text, xgrammar::VocabType::RAW, vocab, std::move(stop_ids),
                                                         /*add_prefix_space=*/false, atomic);
  impl_->compiler = std::make_unique<xgrammar::GrammarCompiler>(*impl_->info, /*max_threads=*/4);
  impl_->compiled.clear();  // compiled for the vocabulary before
  impl_->words = xgrammar::GetBitmaskSize(vocab);
}

bool Grammar::ready() const { return impl_->info != nullptr; }

int Grammar::words() const { return impl_->words; }

namespace {

// What the grammar compiler said, without where in its source it said it: xgrammar's errors begin
// "[hh:mm:ss] /path/of/its/source.cc:123: " (the build machine's path) and end with a newline.
std::string compiler_said(std::string what) {
  while (!what.empty() && (what.back() == '\n' || what.back() == '\r')) what.pop_back();
  if (what.empty() || what[0] != '[') return what;
  const size_t close = what.find("] ");
  if (close == std::string::npos) return what;
  // the first ":<digits>: " after the time is the end of file:line
  for (size_t i = what.find(':', close); i != std::string::npos; i = what.find(':', i + 1)) {
    size_t j = i + 1;
    while (j < what.size() && what[j] >= '0' && what[j] <= '9') ++j;
    if (j > i + 1 && what.compare(j, 2, ": ") == 0) return what.substr(j + 2);
  }
  return what;
}

}  // namespace

std::unique_ptr<GrammarMatcher> Grammar::matcher(const std::string& structural_tag, std::string* err, bool* cached) {
  if (!impl_->compiler) return *err = "grammar not initialised", nullptr;
  try {
    std::optional<xgrammar::CompiledGrammar> g = impl_->cached(structural_tag);
    if (cached) *cached = g.has_value();
    if (!g) {
      std::lock_guard<std::mutex> compiling(impl_->compile_mu);
      g = impl_->cached(structural_tag);  // compiled while this call waited
      if (!g) {
        g = impl_->compiler->CompileStructuralTag(structural_tag);
        std::lock_guard<std::mutex> lock(impl_->mu);
        impl_->compiled.emplace_front(structural_tag, *g);
        if (impl_->compiled.size() > 16) impl_->compiled.pop_back();
      }
    }
    auto impl = std::make_unique<GrammarMatcher::Impl>(
        GrammarMatcher::Impl{xgrammar::GrammarMatcher(*g), impl_->words, &impl_->never});
    return std::unique_ptr<GrammarMatcher>(new GrammarMatcher(std::move(impl)));
  } catch (const std::exception& e) {
    *err = "reply grammar: " + compiler_said(e.what());
    return nullptr;
  }
}

}  // namespace flyx
