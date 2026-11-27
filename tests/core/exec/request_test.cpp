// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The request pipeline every entry point shares (core/exec/request.h): what
// a prompt's tokens, checkpoints, kept checkpoint, reply grammar and parser
// are, for chat messages, raw text and tokens.  The tiny model on the float64
// backend, with Qwen3.5's own template; a raw prompt on a model whose
// tokenizer puts <s> before a text (the tiny Llama).  A history keeps its
// tokens only for a request that reads its prompt's special tokens alike: a
// raw prompt after a chat prewarm whose user spelled "<|im_end|>" gets a fresh
// session's tokens.
//
//   request_test MODEL.flyx TEMPLATE.jinja [PREFIXED.flyx]
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/exec/engine.h"
#include "core/exec/request.h"
#include "core/os/os.h"
#include "core/render/chat_json.h"
#include "core/session/session.h"
#include "tests/common/check.h"

using namespace flyx;

namespace {

ChatMessage msg(const std::string& role, const std::string& content) {
  ChatMessage m;
  m.role = role;
  m.content = content;
  return m;
}

const char* kWeather = R"({"type": "function", "function": {"name": "get_weather", "description": "The weather.",
  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})";
const char* kSearch = R"({"type": "function", "function": {"name": "search", "description": "Search the web.",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}})";

// Both steps, as an entry point takes them.
bool ready(Engine& e, const Session& s, const Request& rq, RenderedChat* rc, PreparedRequest* p, std::string* err) {
  return render_request(e.chat(), e.tokenizer(), rq, rc, err) && prepare_request(e, s, rq, *rc, p, err);
}

// A tokenizer whose post-processor puts <s> before a text (MiniCPM5's; the tiny Llama's): a raw prompt is fed after
// it, as transformers feeds it, and text that continues what the session has shown (the client never saw the <s>)
// keeps the session's tokens.
void raw_prompt_after_prefix(const std::string& model) {
  EngineOptions o;
  o.backend = "cpu_ref";
  o.max_ctx = 512;
  o.speculative = false;
  Engine e;
  std::string err;
  if (!e.load(model, o, &err)) return check(false, "load " + model + ": " + err);
  const auto& tok = e.tokenizer();
  const auto& prefix = tok.prefix();
  check(!prefix.empty(), "the tiny Llama's tokenizer puts tokens before a text");
  Session s(&e);
  check(s.init(&err), "a session " + err);
  Request raw;
  raw.chat = false;
  raw.text = "The capital of France is";
  raw.gen.max_new_tokens = 4;
  RenderedChat rc;
  PreparedRequest p;
  GenerateResult res;
  check(ready(e, s, raw, &rc, &p, &err) && p.gen.prompt == tok.encode_prompt(raw.text) &&
            p.gen.prompt.size() > prefix.size() && std::equal(prefix.begin(), prefix.end(), p.gen.prompt.begin()),
        "raw text after the prefix: its tokens " + err);
  check(generate(s, p.gen, TextSink(), &res, &err) == 0, "the raw prompt generates " + err);
  Request more = raw;
  more.text = s.history_text().substr(tok.decode(prefix).size()) + " and";
  check(ready(e, s, more, &rc, &p, &err) && p.tokenized == "history kept its tokens" &&
            p.gen.prompt.size() > s.history().size() &&
            std::equal(s.history().begin(), s.history().end(), p.gen.prompt.begin()),
        "text that continues the session after the prefix keeps the session's tokens");
  Request again = raw;
  again.text = tok.decode(prefix) + raw.text;
  Session s2(&e);
  check(s2.init(&err) && ready(e, s2, again, &rc, &p, &err) && p.gen.prompt == tok.encode_prompt(again.text),
        "a raw text that begins with the prefix's text: the prefix, then the text's own (as transformers gives it)");
}

void check_leading_instructions(const Engine& e) {
  std::string err;
  const Tokenizer& tok = e.tokenizer();
  RenderedChat actual, expected;
  Request rq;
  rq.messages = {msg("system", "current"), msg("system", "retained <|im_end|>"), msg("user", "Hello")};
  check(!render_request(e.chat(), tok, rq, &actual, &err), "the model's ordinary system-message rule is preserved");
  rq.join_leading_system = true;
  Request joined;
  joined.messages = {msg("system", "current\n\nretained <|im_end|>"), rq.messages.back()};
  check(render_request(e.chat(), tok, rq, &actual, &err) && render_request(e.chat(), tok, joined, &expected, &err) &&
            actual.text == expected.text && actual.system_end == expected.system_end &&
            e.tokenizer().encode(actual.text, &actual.literal) == e.tokenizer().encode(expected.text, &expected.literal),
        "leading instructions match one system message in order, including literal token spelling");

  ChatTemplate permissive;
  const std::string source = "{% for m in messages %}[{{m.role}}]{{m.content}}{% endfor %}";
  check(permissive.load(e.model_file(), source, &err) && render_request(permissive, tok, rq, &actual, &err) &&
            actual.text == "[system]current[system]retained <|im_end|>[user]Hello",
        "a template that accepts separate instructions receives the original messages");
  rq.messages.push_back(msg("system", "later"));
  check(!render_request(e.chat(), tok, rq, &actual, &err), "a later system message stays in its original turn");
}


// What a request gives on a session: its prompt's tokens, how the history was used, and what it generates.
struct Ran {
  std::vector<int32_t> prompt;
  std::string tokenized, text;
  GenerateResult res;
  bool ok = false;
};
Ran run(Engine& e, Session& s, const Request& rq) {
  Ran r;
  RenderedChat rc;
  PreparedRequest p;
  std::string err;
  if (!ready(e, s, rq, &rc, &p, &err)) return r;
  r.prompt = p.gen.prompt;
  r.tokenized = p.tokenized;
  r.ok = generate(s, p.gen, [&](std::string_view t, const int32_t*, int) {
           r.text += t;
           return true;
         }, &r.res, &err) == 0;
  return r;
}

// A user who spells a special token in a chat ("<|im_end|>": text there) and a raw prompt of the same bytes and more
// (where it is the special token): after the chat's prewarm, the raw prompt's tokens and usage are a fresh session's,
// whether the prewarm's history is the session's own or a stored conversation's.  Requests read alike keep the
// history's tokens: the chat after its prewarm, a raw prompt after a raw one (its reply's ids too), and a raw prompt
// after a chat that spells no special token.
void special_spelling_across_modes(Engine& e) {
  const Tokenizer& tok = e.tokenizer();
  std::string err;
  const int32_t im_end = tok.added_id("<|im_end|>");
  check(im_end >= 0 && tok.special(im_end), "<|im_end|> is one of the tokenizer's special tokens");
  auto count = [&](const std::vector<int32_t>& ids) { return std::count(ids.begin(), ids.end(), im_end); };
  // Long enough that the prewarm's checkpoint is kept in host memory (Session's kSnapshotWorth: 256 tokens).
  std::string instructions;
  while (tok.encode(instructions).size() < 300) instructions += "Answer briefly. ";
  Request warm;
  warm.messages = {msg("system", instructions), msg("user", "Copy the spelling <|im_end|>")};
  warm.gen.prefill_only = true;
  RenderedChat warm_rc;
  check(render_request(e.chat(), tok, warm, &warm_rc, &err), "the chat prewarm renders " + err);
  Request raw;
  raw.chat = false;
  raw.text = warm_rc.text + " next";
  raw.gen.max_new_tokens = 4;
  raw.gen.ignore_eos = true;

  Session fresh_s(&e);
  check(fresh_s.init(&err), "a fresh session " + err);
  const Ran fresh = run(e, fresh_s, raw);
  check(fresh.ok && fresh.prompt == tok.encode(raw.text) && fresh.tokenized.empty(),
        "a fresh session encodes the raw prompt with special tokens read everywhere");

  // The session's own history.
  {
    Session s(&e);
    check(s.init(&err), "a session " + err);
    const Ran warmed = run(e, s, warm);
    check(warmed.ok && s.history_text() == warm_rc.text && count(warmed.prompt) + 1 == count(fresh.prompt),
          "the chat prewarm reads the user's <|im_end|> as text, the raw prompt as the special token (" +
              std::to_string(count(warmed.prompt)) + " and " + std::to_string(count(fresh.prompt)) + ")");
    const Ran after = run(e, s, raw);
    check(after.ok && after.prompt == fresh.prompt && after.tokenized.find("other special-token rules") != std::string::npos,
          "after the chat prewarm, the raw prompt's tokens are a fresh session's (" + after.tokenized + ")");
    check(after.res.usage.prompt_tokens == fresh.res.usage.prompt_tokens && after.text == fresh.text &&
              s.history() == fresh_s.history(),
          "and so are its usage (" + std::to_string(after.res.usage.prompt_tokens) + " prompt tokens), reply and history");
  }
  // A stored conversation's history.
  {
    SnapshotStore store(SnapshotStore::kPieceBytes);
    Session parked(&e), s(&e);
    check(parked.init(&err) && s.init(&err), "two sessions " + err);
    parked.set_store(&store);
    s.set_store(&store);
    const Ran warmed = run(e, parked, warm);
    std::string kept;
    const bool stored = warmed.ok && parked.save_to(store, &kept) >= 256 && store.continued_by(raw.text);
    check(stored, "the chat prewarm's checkpoint and history go to the store (" + kept + ")");
    const Ran after = run(e, s, raw);
    check(after.ok && after.prompt == fresh.prompt && after.tokenized.find("other special-token rules") != std::string::npos,
          "from the store, the raw prompt's tokens are a fresh session's (" + after.tokenized + ")");
    check(after.res.usage.prompt_tokens == fresh.res.usage.prompt_tokens && after.text == fresh.text &&
              s.history() == fresh_s.history(),
          "and so are its usage, reply and history");
    // The chat itself, after its stored prewarm: the stored history keeps its tokens.
    Session again(&e);
    check(again.init(&err), "a third session " + err);
    again.set_store(&store);
    Request chat = warm;
    chat.gen.prefill_only = false;
    chat.gen.max_new_tokens = 2;
    chat.gen.ignore_eos = true;
    const Ran asked = run(e, again, chat);
    check(asked.ok && asked.prompt == warmed.prompt &&
              asked.tokenized == "a stored conversation's history kept its tokens" &&
              asked.res.usage.reused_tokens > 0,
          "the chat after its stored prewarm keeps the stored history's tokens (" + asked.tokenized + ", " +
              std::to_string(asked.res.usage.reused_tokens) + " reused)");
  }
  // Read alike: the history keeps its tokens.
  {
    Session s(&e);
    check(s.init(&err), "a session " + err);
    const Ran warmed = run(e, s, warm);
    Request chat = warm;
    chat.gen.prefill_only = false;
    chat.gen.max_new_tokens = 2;
    chat.gen.ignore_eos = true;
    const Ran asked = run(e, s, chat);
    check(warmed.ok && asked.ok && asked.prompt == warmed.prompt && asked.tokenized == "history kept its tokens",
          "the chat after its prewarm keeps the prewarm's tokens");
  }
  {
    Session s(&e);
    check(s.init(&err), "a session " + err);
    Request first = raw;
    first.text = "Copy the spelling <|im_end|> and";
    const Ran said = run(e, s, first);
    const std::vector<int32_t> history = s.history();  // the prompt and the reply's ids
    Request more = raw;
    more.text = s.history_text() + " more";
    const Ran next = run(e, s, more);
    check(said.ok && next.ok && next.tokenized == "history kept its tokens" && history.size() > said.prompt.size() &&
              next.prompt.size() > history.size() && std::equal(history.begin(), history.end(), next.prompt.begin()) &&
              count(said.prompt) == 1,
          "a raw prompt after a raw one keeps its tokens, the reply's ids too");
  }
  {
    Session s(&e);
    check(s.init(&err), "a session " + err);
    Request plain = warm;
    plain.messages.back().content = "Copy the spelling of this word";
    RenderedChat plain_rc;
    const Ran warmed = run(e, s, plain);
    check(render_request(e.chat(), tok, plain, &plain_rc, &err), "a chat without a special token's spelling " + err);
    Request after = raw;
    after.text = plain_rc.text + " next";
    const Ran next = run(e, s, after);
    check(warmed.ok && next.ok && next.tokenized == "history kept its tokens" &&
              std::equal(warmed.prompt.begin(), warmed.prompt.end(), next.prompt.begin()),
          "a raw prompt after a chat prewarm that spells no special token keeps its tokens (" + next.tokenized + ")");
  }
}

}  // namespace

int main(int argc, char** argv) {
  flyx::test::print_passes = true;
  if (argc < 3) {
    std::fprintf(stderr, "usage: request_test MODEL.flyx TEMPLATE.jinja [PREFIXED.flyx]\n");
    return 2;
  }
  std::ifstream f(argv[2], std::ios::binary);
  EngineOptions o;
  o.backend = "cpu_ref";
  o.max_ctx = 512;
  o.prefill_chunk = 32;
  o.speculative = false;
  o.chat_template.assign(std::istreambuf_iterator<char>(f), {});
  Engine e;
  std::string err;
  if (o.chat_template.empty() || !e.load(argv[1], o, &err)) {
    std::printf("load failed: %s\n", err.c_str());
    return 1;
  }
  Session s(&e);
  if (!s.init(&err)) {
    std::printf("session: %s\n", err.c_str());
    return 1;
  }
  const auto& tok = e.tokenizer();
  check_leading_instructions(e);

  // A question: the template's rendering as tokens, a checkpoint where the
  // question ends (kept through what follows), the reply split by a parser.
  Request ask;
  ask.messages = {msg("system", "Be brief."), msg("user", "What is the weather in Paris?")};
  ask.gen.max_new_tokens = 4;
  RenderedChat rc;
  PreparedRequest p;
  check(ready(e, s, ask, &rc, &p, &err), "a chat request is made ready " + err);
  check(!p.gen.prompt.empty() && p.gen.prompt == tok.encode(rc.text, &rc.literal),
        "its prompt is the rendered text as tokens (" + std::to_string(p.gen.prompt.size()) + ")");
  check(!p.gen.checkpoints.empty() && p.gen.keep_checkpoint == p.gen.checkpoints.back() &&
            p.gen.keep_checkpoint < static_cast<int>(p.gen.prompt.size()),
        "a question's end is a checkpoint, and the one kept");
  check(p.gen.system_end > 0 && p.gen.cache_prefixes.empty(),
        "a plain chat keeps its system prefix without requesting periodic cache checkpoints");
  check(p.parser && !p.grammar && !p.gen.grammar && p.unconstrained.empty(),
        "no tools: a parser for the reply, no grammar");
  check(p.gen.max_new_tokens == 4, "the request's limits pass through");

  // The next question, once the session has answered this one: its prompt
  // still starts with the tokens up to the kept checkpoint (the template
  // writes the earlier turn differently from there on), and its own end is
  // the checkpoint kept from now on.
  GenerateResult res;
  std::string shown;
  const bool ran = generate(s, p.gen, [&](std::string_view text, const int32_t*, int) {
    shown += text;
    return true;
  }, &res, &err) == 0;
  check(ran, "the prepared request generates " + err);
  Request next = ask;
  next.messages.push_back(msg("assistant", shown));
  next.messages.push_back(msg("user", "And in Rome?"));
  RenderedChat rc2;
  PreparedRequest p2;
  check(ready(e, s, next, &rc2, &p2, &err), "the next question is made ready " + err);
  check(static_cast<int>(p2.gen.prompt.size()) > p.gen.keep_checkpoint &&
            std::equal(p.gen.prompt.begin(), p.gen.prompt.begin() + p.gen.keep_checkpoint, p2.gen.prompt.begin()),
        "its prompt starts with the first one's tokens up to the kept checkpoint");
  check(!p2.gen.checkpoints.empty() && p2.gen.keep_checkpoint == p2.gen.checkpoints.back() &&
            p2.gen.keep_checkpoint > p.gen.keep_checkpoint,
        "the new question's end is the checkpoint kept");

  // Tool results are not a question: checkpoints, none of them kept.
  Request tool_turn = ask;
  ChatMessage call = msg("assistant", "");
  call.tool_calls.push_back({"call_1", "get_weather", ojson{{"city", "Paris"}}});
  ChatMessage result = msg("tool", "18 C, sunny");
  result.tool_call_id = "call_1";
  tool_turn.messages.push_back(call);
  tool_turn.messages.push_back(result);
  tool_turn.tools = {ojson::parse(kWeather)};
  check(ready(e, s, tool_turn, &rc, &p, &err), "a tool turn is made ready " + err);
  check(!p.gen.checkpoints.empty() && p.gen.keep_checkpoint == -1, "a tool turn keeps no checkpoint of its own");
  check(p.grammar && p.gen.grammar == p.grammar.get(), "tools: the reply has a grammar");

  // A reply that must call a tool; text as it comes when the output is not split.
  Request must = ask;
  must.tools = {ojson::parse(kWeather)};
  must.rules.tools = ReplyRules::kRequired;
  must.split_output = false;
  check(ready(e, s, must, &rc, &p, &err) && p.grammar && !p.parser,
        "tool_choice required: a grammar; output not split: no parser " + err);

  // The grammar compiled ahead (warm_request, the server's HTTP thread): the session's step finds it kept.  A tool
  // set not seen before compiles there.
  Request warm = must;
  warm.tools = {ojson::parse(kSearch)};
  check(render_request(e.chat(), e.tokenizer(), warm, &rc, &err), "a request with another tool renders " + err);
  warm_request(e, warm, rc);
  check(prepare_request(e, s, warm, rc, &p, &err) && p.grammar && p.grammar_cached,
        "a request made warm ahead: its grammar was kept " + err);
  Request cold = must;
  cold.tools = {ojson::parse(kWeather), ojson::parse(kSearch)};
  Request prefill = cold;
  prefill.rules.tools = ReplyRules::kAuto;
  prefill.gen.prefill_only = true;
  check(render_request(e.chat(), e.tokenizer(), prefill, &rc, &err), "a prewarm with optional tools renders " + err);
  warm_request(e, prefill, rc);
  check(prepare_request(e, s, prefill, rc, &p, &err) && !p.grammar && !p.parser,
        "a prewarm uses no reply grammar or parser " + err);
  prefill.gen.prefill_only = false;
  check(prepare_request(e, s, prefill, rc, &p, &err) && p.grammar && p.grammar_cached,
        "the next reply with the same optional tools uses the prewarmed grammar " + err);
  check(ready(e, s, cold, &rc, &p, &err) && p.grammar && !p.grammar_cached,
        "a tool set not seen before: compiled when made ready " + err);

  // Cache hints describe the normalized messages but cannot contribute a byte to the model's input.
  Request marked;
  const ojson wire = ojson::parse(R"({"system":[{"type":"text","text":"Stable instructions.","cache_control":{"type":"ephemeral"}}],
      "tools":[{"name":"first_protocol_tool","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}},
               {"name":"second_protocol_tool","input_schema":{"type":"object"}}],
      "messages":[{"role":"user","content":[{"type":"text","text":"Stable question. ","cache_control":{"type":"ephemeral"}},
                                           {"type":"text","text":"Changing suffix."}]}]})");
  check(chat_from_anthropic(wire, &marked.messages, &marked.tools, &err, false, nullptr, &marked.cache_points) &&
            marked.cache_points.size() == 3,
        "Anthropic system, tool and partial user block breakpoints are recognized");
  RenderedChat marked_rc, plain_rc;
  check(render_request(e.chat(), e.tokenizer(), marked, &marked_rc, &err), "marked input renders " + err);
  marked.cache_points.clear();
  check(render_request(e.chat(), e.tokenizer(), marked, &plain_rc, &err) && plain_rc.text == marked_rc.text &&
            tok.encode(plain_rc.text, &plain_rc.literal) == tok.encode(marked_rc.text, &marked_rc.literal),
        "a breakpoint leaves rendering and special-token handling unchanged");
  auto anchored_before = [&](const std::string& before, const std::string& after) {
    const size_t begin = marked_rc.text.find(before), end = marked_rc.text.find(after);
    return begin < end && end != std::string::npos &&
           std::any_of(marked_rc.cache_prefixes.begin(), marked_rc.cache_prefixes.end(),
                       [&](size_t p) { return begin + before.size() <= p && p <= end; });
  };
  check(anchored_before("first_protocol_tool", "second_protocol_tool") &&
            anchored_before("Stable question.", "Changing suffix.") && marked_rc.system_end > 0,
        "tool and partial text anchors fall before the editable suffix in the rendered template");

  // Raw text and tokens: no template, no checkpoints, no grammar, no parser.
  Request raw;
  raw.chat = false;
  raw.text = "The capital of France is";
  Session s2(&e);
  s2.init(&err);
  check(ready(e, s2, raw, &rc, &p, &err) && p.gen.prompt == tok.encode(raw.text) && p.gen.checkpoints.empty() &&
            p.gen.keep_checkpoint == -1 && !p.grammar && !p.parser,
        "raw text: its tokens and nothing else " + err);
  // Text that continues what the session has shown: the history keeps the
  // tokens it was generated as, and only the rest is encoded.
  raw.gen.max_new_tokens = 4;
  check(ready(e, s2, raw, &rc, &p, &err) && generate(s2, p.gen, TextSink(), &res, &err) == 0,
        "the raw prompt generates " + err);
  Request more = raw;
  more.text = s2.history_text() + " and";
  check(ready(e, s2, more, &rc, &p, &err) && p.tokenized == "history kept its tokens" &&
            p.gen.prompt.size() > s2.history().size() &&
            std::equal(s2.history().begin(), s2.history().end(), p.gen.prompt.begin()),
        "text that continues the session keeps the session's tokens");
  Request ids;
  ids.chat = false;
  ids.tokens = {5, 6, 7, 8};
  check(ready(e, s, ids, &rc, &p, &err) && p.gen.prompt == ids.tokens && rc.text.empty(),
        "tokens: passed through " + err);
  Request empty;
  empty.chat = false;
  check(!ready(e, s, empty, &rc, &p, &err) && err == "empty prompt", "no prompt at all is refused");
  check(tok.prefix().empty(), "a tokenizer without a post-processor's tokens: no prefix (raw text as it is, above)");
  if (argc > 3) raw_prompt_after_prefix(argv[3]);
  special_spelling_across_modes(e);

  // Rewriting an early block invalidates its suffix, but we can resume from the natural prefill pass
  // before its breakpoint. The edited request must yield the same tokens as a cold run.
  GenerateRequest edited;
  edited.prompt.assign(221, 27);
  edited.max_new_tokens = 4;
  edited.ignore_eos = true;
  edited.cache_prefixes = {95};
  err.clear();
  check(generate(s, edited, {}, &res, &err) == 0, "prompt with an early breakpoint generates " + err);
  edited.prompt[97] = 31;
  check(generate(s, edited, {}, &res, &err) == 0 && res.usage.reused_tokens == 64,
        "edited block reuses the last completed 32-token pass before its breakpoint");
  check(generate(s2, edited, {}, &res, &err) == 0 && s2.history() == s.history(),
        "restored early prefix and cold execution generate identical tokens");

  // A parked tool turn has both the question and the later end of the prompt. A single RAM piece
  // holds either checkpoint; an edit between them must still be able to return to the question.
  GenerateRequest tool_loop;
  tool_loop.prompt.assign(352, 29);
  tool_loop.max_new_tokens = 1;
  tool_loop.checkpoints = {288};
  tool_loop.keep_checkpoint = 288;
  SnapshotStore small(SnapshotStore::kPieceBytes);
  check(generate(s, tool_loop, {}, &res, &err) == 0 && s.save_to(small, &err) > 288,
        "the tool turn has a checkpoint beyond its protected question");
  tool_loop.prompt[310] = 31;
  const auto* question = small.find(tool_loop.prompt, tool_loop.prompt.size(), 0);
  check(small.size() == 1 && question && question->prefix.size() == 288,
        "when only one checkpoint fits in RAM the protected question survives");

  // A broken last segment can be discovered by the backend only after restoring its parent.
  // A fresh restore from that parent, or from zero if the parent is broken, must give the cold tokens.
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / ("flyx-request-test-" + std::to_string(os::process_id()));
  for (bool break_parent : {false, true}) {
    SnapshotStore disk(0, 4096);
    const auto utf8 = dir.u8string();
    bool ok = disk.set_disk({utf8.begin(), utf8.end()}, 32u << 20, &err);
    s2.reset_after_error();
    s2.set_store(&disk);
    GenerateRequest stored;
    stored.prompt.assign(352, 33);
    stored.cache_prefixes = {272};
    stored.max_new_tokens = 2;
    stored.ignore_eos = true;
    ok = ok && generate(s2, stored, {}, &res, &err) == 0 && s2.save_to(disk, &err) >= 352;
    disk.settle();
    stored.prompt.push_back(34);
    const auto* tail = disk.find(stored.prompt, stored.prompt.size() - 1, 0);
    const auto* parent = tail && tail->parent ? disk.find(stored.prompt, tail->prefix.size() - 1, 0) : nullptr;
    ok = ok && parent && !parent->parent && parent->disk_id == tail->parent;
    const int expected = ok && !break_parent ? static_cast<int>(parent->prefix.size()) : 0;
    if (ok) {
      const auto& path = (break_parent ? parent : tail)->file;
      std::fstream file(fs::path(std::u8string(path.begin(), path.end())),
                        std::ios::binary | std::ios::in | std::ios::out);
      file.put('!');
      ok = file.good();
    }
    s2.reset_after_error();
    s.reset_after_error();
    ok = ok && generate(s, stored, {}, &res, &err) == 0 && generate(s2, stored, {}, &res, &err) == 0;
    check(ok && res.usage.reused_tokens == expected && res.reuse.find("incomplete snapshot") != std::string::npos &&
              s.history() == s2.history(), break_parent ? "a damaged root checkpoint reproduces cold output" :
              "a damaged child checkpoint resumes from its parent and reproduces cold output");
    s2.set_store(nullptr);
  }
  fs::remove(dir);

  return flyx::test::finish("request_test");
}
