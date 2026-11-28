// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/render/chat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "core/modelfile/modelfile.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

namespace flyx {
namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull, kFnvPrime = 1099511628211ull;

bool is_valid_utf8(const std::string& s) {
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      ++i;
      continue;
    }
    size_t len = 0;
    uint32_t cp = 0;
    if ((c & 0xE0) == 0xC0) {
      len = 2;
      cp = c & 0x1Fu;
    } else if ((c & 0xF0) == 0xE0) {
      len = 3;
      cp = c & 0x0Fu;
    } else if ((c & 0xF8) == 0xF0) {
      len = 4;
      cp = c & 0x07u;
    } else {
      return false;
    }
    if (i + len > n) return false;
    for (size_t j = 1; j < len; ++j) {
      const unsigned char b = static_cast<unsigned char>(s[i + j]);
      if ((b & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (b & 0x3Fu);
    }
    if ((len == 2 && cp < 0x80) || (len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
        (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)))
      return false;
    i += len;
  }
  return true;
}

// Replace invalid byte sequences identically to nlohmann::json::error_handler_t::replace.
std::string sanitize_utf8_slow(const std::string& s) {
  const std::string dumped = ojson(s).dump(-1, ' ', false, ojson::error_handler_t::replace);
  return ojson::parse(dumped).get<std::string>();
}

jinja::value make_jinja_string(const std::string& s, bool mark_input) {
  auto str = is_valid_utf8(s) ? jinja::mk_val<jinja::value_string>(s)
                              : jinja::mk_val<jinja::value_string>(sanitize_utf8_slow(s));
  if (mark_input) str->mark_input();
  return str;
}

void insert_input_lit(const jinja::value_object& obj, const char* key, jinja::value val) {
  auto k = jinja::mk_val<jinja::value_string>(std::string(key));
  k->mark_input();
  obj->insert(std::move(k), std::move(val));
}

// Direct in-memory ojson -> jinja::value conversion, preserving object key order, input provenance,
// UTF-8 replacement, and depth limits without JSON stringify/parse or intermediate common_json copies.
jinja::value to_jinja_value(const ojson& j, bool mark_input, int depth = 0) {
  if (j.is_null()) return jinja::mk_val<jinja::value_none>();
  if (j.is_boolean()) return jinja::mk_val<jinja::value_bool>(j.get<bool>());
  if (j.is_number_integer()) return jinja::mk_val<jinja::value_int>(j.get<int64_t>());
  if (j.is_number_float()) {
    const double d = j.get<double>();
    return std::isfinite(d) ? jinja::value(jinja::mk_val<jinja::value_float>(d))
                            : jinja::value(jinja::mk_val<jinja::value_none>());
  }
  if (j.is_string()) return make_jinja_string(j.get_ref<const std::string&>(), mark_input);
  if ((j.is_array() || j.is_object()) && depth >= JINJA_RECURSIVE_LIMIT)
    throw std::invalid_argument("JSON is nested more than " + std::to_string(JINJA_RECURSIVE_LIMIT) + " levels deep");
  if (j.is_array()) {
    auto arr = jinja::mk_val<jinja::value_array>();
    arr->val_arr.reserve(j.size());
    for (const auto& el : j) arr->push_back(to_jinja_value(el, mark_input, depth + 1));
    return arr;
  }
  if (j.is_object()) {
    auto obj = jinja::mk_val<jinja::value_object>();
    obj->val_obj.reserve(j.size());
    obj->unordered.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it)
      obj->insert(make_jinja_string(it.key(), mark_input), to_jinja_value(it.value(), mark_input, depth + 1));
    return obj;
  }
  return jinja::mk_val<jinja::value_none>();
}

// The message as transformers hands it to a template, built directly as a marked jinja::value_object.
jinja::value message_to_jinja(const ChatMessage& m) {
  auto obj = jinja::mk_val<jinja::value_object>();
  obj->val_obj.reserve(5);
  obj->unordered.reserve(5);
  insert_input_lit(obj, "role", make_jinja_string(m.role, true));
  if (!m.parts.empty()) {
    auto parts = jinja::mk_val<jinja::value_array>();
    parts->val_arr.reserve(m.parts.size());
    for (const auto& p : m.parts) {
      auto item = jinja::mk_val<jinja::value_object>();
      if (p.kind == ContentPart::kImage) {
        insert_input_lit(item, "type", make_jinja_string("image", true));
      } else {
        insert_input_lit(item, "type", make_jinja_string("text", true));
        insert_input_lit(item, "text", make_jinja_string(p.text, true));
      }
      parts->push_back(std::move(item));
    }
    insert_input_lit(obj, "content", std::move(parts));
  } else {
    insert_input_lit(obj, "content", make_jinja_string(m.content, true));
  }
  if (m.reasoning) insert_input_lit(obj, "reasoning_content", make_jinja_string(*m.reasoning, true));
  if (!m.tool_calls.empty()) {
    auto calls = jinja::mk_val<jinja::value_array>();
    calls->val_arr.reserve(m.tool_calls.size());
    for (const auto& c : m.tool_calls) {
      auto call = jinja::mk_val<jinja::value_object>();
      call->val_obj.reserve(3);
      call->unordered.reserve(3);
      if (!c.id.empty()) insert_input_lit(call, "id", make_jinja_string(c.id, true));
      auto fn_type = jinja::mk_val<jinja::value_string>(std::string("function"));
      fn_type->mark_input();
      insert_input_lit(call, "type", std::move(fn_type));
      auto fn = jinja::mk_val<jinja::value_object>();
      fn->val_obj.reserve(2);
      fn->unordered.reserve(2);
      insert_input_lit(fn, "name", make_jinja_string(c.name, true));
      insert_input_lit(fn, "arguments", to_jinja_value(c.arguments, true, 5));
      insert_input_lit(call, "function", std::move(fn));
      calls->push_back(std::move(call));
    }
    insert_input_lit(obj, "tool_calls", std::move(calls));
  }
  if (!m.tool_call_id.empty()) insert_input_lit(obj, "tool_call_id", make_jinja_string(m.tool_call_id, true));
  if (!m.name.empty()) insert_input_lit(obj, "name", make_jinja_string(m.name, true));
  return obj;
}

void hash_bytes(uint64_t& h, const void* data, size_t n) {
  const auto* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime;
}

void hash_u64(uint64_t& h, uint64_t v) { hash_bytes(h, &v, sizeof(v)); }

void hash_str(uint64_t& h, const std::string& s) {
  hash_u64(h, s.size());
  hash_bytes(h, s.data(), s.size());
}

void hash_ojson(uint64_t& h, const ojson& j) {
  const auto tag = static_cast<uint64_t>(j.type());
  hash_u64(h, tag);
  if (j.is_boolean()) {
    hash_u64(h, j.get<bool>() ? 1u : 0u);
  } else if (j.is_number_unsigned()) {
    hash_u64(h, j.get<uint64_t>());
  } else if (j.is_number_integer()) {
    hash_u64(h, static_cast<uint64_t>(j.get<int64_t>()));
  } else if (j.is_number_float()) {
    const double d = j.get<double>();
    uint64_t bits = 0;
    if (std::isfinite(d)) std::memcpy(&bits, &d, sizeof(bits));
    else bits = 0x7FF8000000000000ull;  // non-finite values all map to null
    hash_u64(h, bits);
  } else if (j.is_string()) {
    hash_str(h, j.get_ref<const std::string&>());
  } else if (j.is_array()) {
    hash_u64(h, j.size());
    for (const auto& el : j) hash_ojson(h, el);
  } else if (j.is_object()) {
    hash_u64(h, j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
      hash_str(h, it.key());
      hash_ojson(h, it.value());
    }
  }
}

// Strict value + key-order + numeric-type equality so templates that iterate keys or format
// 1 vs 1.0 or -0.0 vs 0.0 never hit a stale cache entry.
bool same_ojson(const ojson& a, const ojson& b) {
  if (a.type() != b.type()) return false;
  if (a.is_null()) return true;
  if (a.is_boolean()) return a.get<bool>() == b.get<bool>();
  if (a.is_number_unsigned()) return a.get<uint64_t>() == b.get<uint64_t>();
  if (a.is_number_integer()) return a.get<int64_t>() == b.get<int64_t>();
  if (a.is_number_float()) {
    const double da = a.get<double>(), db = b.get<double>();
    if (!std::isfinite(da) && !std::isfinite(db)) return true;
    return std::memcmp(&da, &db, sizeof(double)) == 0;
  }
  if (a.is_string()) return a.get_ref<const std::string&>() == b.get_ref<const std::string&>();
  if (a.is_array()) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (!same_ojson(a[i], b[i])) return false;
    return true;
  }
  if (a.is_object()) {
    if (a.size() != b.size()) return false;
    auto ia = a.begin(), ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib)
      if (ia.key() != ib.key() || !same_ojson(ia.value(), ib.value())) return false;
    return true;
  }
  return true;
}

bool same_ojson_vec(const std::vector<ojson>& a, const std::vector<ojson>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same_ojson(a[i], b[i])) return false;
  return true;
}

void hash_chat_message(uint64_t& h, const ChatMessage& m) {
  hash_str(h, m.role);
  hash_str(h, m.content);
  hash_u64(h, m.reasoning ? 1u : 0u);
  if (m.reasoning) hash_str(h, *m.reasoning);
  hash_u64(h, m.tool_calls.size());
  for (const auto& c : m.tool_calls) {
    hash_str(h, c.id);
    hash_str(h, c.name);
    hash_ojson(h, c.arguments);
  }
  hash_str(h, m.tool_call_id);
  hash_str(h, m.name);
  hash_u64(h, m.parts.size());
  for (const auto& p : m.parts) {
    hash_u64(h, static_cast<uint64_t>(p.kind));
    if (p.kind == ContentPart::kImage && p.image) {
      hash_bytes(h, p.image->sha256.data(), p.image->sha256.size());
      hash_str(h, p.image->detail);
    } else {
      hash_str(h, p.text);
    }
  }
}

bool same_chat_message(const ChatMessage& a, const ChatMessage& b) {
  if (a.role != b.role || a.content != b.content || a.reasoning != b.reasoning || a.tool_call_id != b.tool_call_id ||
      a.name != b.name || a.tool_calls.size() != b.tool_calls.size() || a.parts.size() != b.parts.size())
    return false;
  for (size_t i = 0; i < a.tool_calls.size(); ++i)
    if (a.tool_calls[i].id != b.tool_calls[i].id || a.tool_calls[i].name != b.tool_calls[i].name ||
        !same_ojson(a.tool_calls[i].arguments, b.tool_calls[i].arguments))
      return false;
  for (size_t i = 0; i < a.parts.size(); ++i) {
    const auto &pa = a.parts[i], &pb = b.parts[i];
    if (pa.kind != pb.kind) return false;
    if (pa.kind == ContentPart::kImage) {
      if (!pa.image != !pb.image) return false;
      if (pa.image && (pa.image->sha256 != pb.image->sha256 || pa.image->detail != pb.image->detail)) return false;
    } else if (pa.text != pb.text) {
      return false;
    }
  }
  return true;
}

bool same_chat_messages(const std::vector<ChatMessage>& a, const std::vector<ChatMessage>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same_chat_message(a[i], b[i])) return false;
  return true;
}

uint64_t hash_base_key(bool enable_thinking, const ojson& kwargs, const std::vector<ojson>& tools) {
  uint64_t h = kFnvOffset;
  hash_u64(h, enable_thinking ? 1u : 0u);
  hash_ojson(h, kwargs);
  hash_u64(h, tools.size());
  for (const auto& t : tools) hash_ojson(h, t);
  return h;
}

// The template's own words for an error (raise_exception's message) without the
// engine's source excerpt.
std::string short_error(const std::string& what) {
  size_t k = what.rfind("Error: ");
  std::string s = k == std::string::npos ? what : what.substr(k + 7);
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
  while (!s.empty() && (s.front() == '\n' || s.front() == ' ')) s.erase(0, 1);
  constexpr std::string_view kRaised = "Jinja Exception: ";  // raise_exception's prefix
  if (s.starts_with(kRaised)) s.erase(0, kRaised.size());
  return s;
}

}  // namespace

struct ChatTemplate::Impl {
  std::unique_ptr<interfaces::Chat> chat;
  jinja::program program;
  ojson own_tokens;       // chat->template_tokens(), read once at load
  bool native = false;    // the chat format's own renderer instead of the engine
  bool uses_time = false; // template calls strftime_now(); probe outputs are not pure functions

  struct Conversation {
    const std::vector<ChatMessage>& msgs;
    const std::vector<ojson>& tools;
    const ChatOptions& opt;
  };

  // The first `count` messages rendered (the generation prompt if asked):
  // text and literal spans.  false with *err where the template raises.
  bool render(const Conversation& c, size_t count, bool gen, RenderedChat* out, std::string* err) const {
    if (native) return chat->render(std::span(c.msgs).first(count), c.tools, c.opt, gen, out, err);
    try {
      engine_render(c.msgs, count, c.tools, c.opt, gen, out);
      return true;
    } catch (const std::exception& e) {
      *err = short_error(e.what());
      return false;
    }
  }

  void engine_render(const std::vector<ChatMessage>& msgs, size_t count, const std::vector<ojson>& tools,
                     const ChatOptions& opt, bool gen, RenderedChat* out) const {
    jinja::context ctx(chat->template_source());
    // What the client sent is input (marked, so its special-token text stays text) ...
    if (opt.kwargs.is_object()) {
      for (auto it = opt.kwargs.begin(); it != opt.kwargs.end(); ++it) {
        if (it.key() == "messages" || it.key() == "tools" || it.key() == "add_generation_prompt") continue;
        const std::string& k = it.key();
        ctx.set_val(is_valid_utf8(k) ? k : sanitize_utf8_slow(k), to_jinja_value(it.value(), true, 0));
      }
    }
    auto messages_val = jinja::mk_val<jinja::value_array>();
    messages_val->val_arr.reserve(count);
    for (size_t i = 0; i < count; ++i) messages_val->push_back(message_to_jinja(msgs[i]));
    ctx.set_val("messages", std::move(messages_val));
    if (!tools.empty()) {
      auto tools_val = jinja::mk_val<jinja::value_array>();
      tools_val->val_arr.reserve(tools.size());
      for (const auto& t : tools) tools_val->push_back(to_jinja_value(t, true, 1));
      ctx.set_val("tools", std::move(tools_val));
    }
    // ... what the engine and the model supply is not.
    if (own_tokens.is_object()) {
      for (auto it = own_tokens.begin(); it != own_tokens.end(); ++it) {
        const std::string& k = it.key();
        ctx.set_val(is_valid_utf8(k) ? k : sanitize_utf8_slow(k), to_jinja_value(it.value(), false, 0));
      }
    }
    ctx.set_val("add_generation_prompt", jinja::mk_val<jinja::value_bool>(gen));
    if (!opt.kwargs.is_object() || !opt.kwargs.contains("enable_thinking"))
      ctx.set_val("enable_thinking", jinja::mk_val<jinja::value_bool>(opt.enable_thinking));
    // The same options under the names this template reads them by, unless the request set those itself.
    const ojson named = chat->option_variables(opt);
    for (auto it = named.begin(); it != named.end(); ++it)
      if (!opt.kwargs.is_object() || !opt.kwargs.contains(it.key()))
        ctx.set_val(it.key(), to_jinja_value(it.value(), false, 0));

    jinja::runtime runtime(ctx);
    const jinja::value result = runtime.execute(program);
    auto parts = jinja::runtime::gather_string_parts(result);
    out->text.clear();
    out->literal.clear();
    out->images.clear();
    size_t total = 0;
    for (const auto& part : parts->val_str.parts) total += part.val.size();
    out->text.reserve(total);
    for (const auto& part : parts->val_str.parts) {
      const size_t b = out->text.size();
      out->text += part.val;
      if (part.is_input || part.val.empty()) continue;
      if (!out->literal.empty() && out->literal.back().end == b) out->literal.back().end = out->text.size();
      else out->literal.push_back({b, out->text.size()});
    }
    // Non-native Jinja path: preserve ordered images with sentinel offsets; request preparation
    // resolves them against the vision adapter's literal placeholder token IDs.
    for (size_t i = 0; i < count; ++i)
      for (const auto& p : msgs[i].parts)
        if (p.kind == ContentPart::kImage && p.image)
          out->images.push_back({std::string::npos, std::string::npos, p.image});
  }

  // Where recent conversations ended (their text up to the generation
  // prompt), as length and hash: a later request that continues one of them
  // branches there.  Newest last.
  struct Ending {
    size_t len;
    uint64_t hash;

    void extend(const std::string& text, size_t end) {
      for (; len < end; ++len) hash = (hash ^ static_cast<unsigned char>(text[len])) * kFnvPrime;
    }
  };
  static constexpr size_t kEndings = 64;
  static constexpr size_t kGenCache = 8;
  static constexpr size_t kPrefixCache = 16;
  static constexpr size_t kMaxCachedPrefixBytes = 256u << 10;

  // Pure-function probe caches for Jinja-rendered templates (!native && !uses_time).
  // Both avoid re-executing the interpreter on repeated generation-prompt and stable-prefix probes
  // across multi-turn agent loops with large tool lists.  mu is never held across render().
  struct GenPromptEntry {
    uint64_t hash = 0;
    bool enable_thinking = true;
    ojson kwargs;
    std::vector<ojson> tools;
    std::optional<std::string> gen;
  };
  struct PrefixEntry {
    uint64_t hash = 0;
    bool enable_thinking = true;
    ojson kwargs;
    std::vector<ojson> tools;
    std::vector<ChatMessage> msgs;
    bool ok = false;
    std::string text;
  };

  mutable std::mutex mu;
  std::vector<Ending> endings;
  mutable std::vector<GenPromptEntry> gen_cache;
  mutable std::vector<PrefixEntry> prefix_cache;

  // The ends of remembered conversations that `text` continues, before `limit`.
  std::vector<size_t> continued(const std::string& text, size_t limit, Ending& prefix) {
    std::vector<Ending> e;
    {
      std::lock_guard<std::mutex> lock(mu);
      for (const auto& x : endings)
        if (x.len < limit) e.push_back(x);
    }
    std::sort(e.begin(), e.end(), [](const Ending& a, const Ending& b) { return a.len < b.len; });
    std::vector<size_t> out;
    for (const auto& x : e) {
      prefix.extend(text, x.len);
      if (prefix.hash == x.hash && (out.empty() || out.back() != x.len)) out.push_back(x.len);
    }
    return out;
  }

  void remember(Ending ending) {
    std::lock_guard<std::mutex> lock(mu);
    for (const auto& x : endings)
      if (x.len == ending.len && x.hash == ending.hash) return;
    if (endings.size() == kEndings) endings.erase(endings.begin());
    endings.push_back(ending);
  }

  // The generation prompt, from a one-message conversation rendered with and without it: what the template appends
  // to open the reply.  The conversation ends where `text` ends with it.
  std::optional<size_t> probe_conversation_end(const std::vector<ojson>& tools, const ChatOptions& opt,
                                               const RenderedChat& rendered, bool* in_reasoning) const {
    const std::string& text = rendered.text;
    if (!opt.add_generation_prompt) return text.size();
    std::optional<std::string> gen;
    bool hit = false;
    const bool cacheable = !native && !uses_time;
    const uint64_t h = cacheable ? hash_base_key(opt.enable_thinking, opt.kwargs, tools) : 0;
    if (cacheable) {
      std::lock_guard<std::mutex> lock(mu);
      for (size_t i = gen_cache.size(); i-- > 0;) {
        const auto& e = gen_cache[i];
        if (e.hash == h && e.enable_thinking == opt.enable_thinking && same_ojson(e.kwargs, opt.kwargs) &&
            same_ojson_vec(e.tools, tools)) {
          gen = e.gen;
          if (i + 1 != gen_cache.size()) std::rotate(gen_cache.begin() + i, gen_cache.begin() + i + 1, gen_cache.end());
          hit = true;
          break;
        }
      }
    }
    if (!hit) {
      const std::vector<ChatMessage> probe_msgs{{"user", "x", std::nullopt, {}, ""}};
      const Conversation probe{probe_msgs, tools, opt};
      RenderedChat with, without;
      std::string ignored;
      if (render(probe, 1, true, &with, &ignored) && render(probe, 1, false, &without, &ignored) &&
          with.text.starts_with(without.text))
        gen = with.text.substr(without.text.size());
      if (cacheable) {
        std::lock_guard<std::mutex> lock(mu);
        bool exists = false;
        for (const auto& e : gen_cache) {
          if (e.hash == h && e.enable_thinking == opt.enable_thinking && same_ojson(e.kwargs, opt.kwargs) &&
              same_ojson_vec(e.tools, tools)) {
            exists = true;
            break;
          }
        }
        if (!exists) {
          if (gen_cache.size() >= kGenCache) gen_cache.erase(gen_cache.begin());
          gen_cache.push_back({h, opt.enable_thinking, opt.kwargs, tools, gen});
        }
      }
    }
    std::optional<size_t> end;
    if (gen && text.ends_with(*gen)) end = text.size() - gen->size();
    // Never scan untrusted message text if the 1-message probe did not match: only check a trailing template literal.
    std::string_view tail;
    if (end) {
      tail = std::string_view(*gen);
    } else if (!rendered.literal.empty() && rendered.literal.back().end == text.size()) {
      tail = std::string_view(text).substr(rendered.literal.back().begin);
    }
    *in_reasoning = chat->opens_reasoning(tail);
    return end;
  }

  // Render a prefix without the generation prompt, reusing cached Jinja outputs when pure.
  bool render_prefix_text(const std::vector<ChatMessage>& part, const std::vector<ojson>& definitions,
                          const ChatOptions& opt, std::string* out_text) const {
    const bool cacheable = !native && !uses_time;
    uint64_t h = 0;
    if (cacheable) {
      h = hash_base_key(opt.enable_thinking, opt.kwargs, definitions);
      hash_u64(h, part.size());
      for (const auto& m : part) hash_chat_message(h, m);
      std::lock_guard<std::mutex> lock(mu);
      for (size_t i = prefix_cache.size(); i-- > 0;) {
        const auto& e = prefix_cache[i];
        if (e.hash == h && e.enable_thinking == opt.enable_thinking && same_ojson(e.kwargs, opt.kwargs) &&
            same_ojson_vec(e.tools, definitions) && same_chat_messages(e.msgs, part)) {
          const bool ok = e.ok;
          if (ok) *out_text = e.text;
          if (i + 1 != prefix_cache.size())
            std::rotate(prefix_cache.begin() + i, prefix_cache.begin() + i + 1, prefix_cache.end());
          return ok;
        }
      }
    }
    RenderedChat r;
    std::string ignored;
    const Conversation prefix{part, definitions, opt};
    const bool ok = render(prefix, part.size(), false, &r, &ignored);
    if (cacheable && (!ok || r.text.size() <= kMaxCachedPrefixBytes)) {
      std::lock_guard<std::mutex> lock(mu);
      bool exists = false;
      for (const auto& e : prefix_cache) {
        if (e.hash == h && e.enable_thinking == opt.enable_thinking && same_ojson(e.kwargs, opt.kwargs) &&
            same_ojson_vec(e.tools, definitions) && same_chat_messages(e.msgs, part)) {
          exists = true;
          break;
        }
      }
      if (!exists) {
        if (prefix_cache.size() >= kPrefixCache) prefix_cache.erase(prefix_cache.begin());
        prefix_cache.push_back({h, opt.enable_thinking, opt.kwargs, definitions, part, ok, ok ? r.text : ""});
      }
    }
    if (ok) *out_text = std::move(r.text);
    return ok;
  }

  // A template can trim or rearrange fields, or reject a conversation without a user turn. Use only the
  // leading bytes shared with the real rendering; offsets of strings in the input JSON are not prompt offsets.
  size_t probe_cache_prefix(const std::string& text, std::vector<ChatMessage> part,
                            const std::vector<ojson>& definitions, const ChatOptions& opt) const {
    if (part.empty() || part.back().role == "system") part.push_back({"user", "", std::nullopt, {}, ""});
    std::string rendered;
    if (!render_prefix_text(part, definitions, opt, &rendered)) return 0;
    const auto common = std::mismatch(text.begin(), text.end(), rendered.begin(), rendered.end());
    return static_cast<size_t>(common.first - text.begin());
  }

  void resolve_cache_points(const Conversation& conv, std::optional<size_t> end,
                            const std::vector<ChatCachePoint>& cache_points, RenderedChat* out) {
    const auto& msgs = conv.msgs;
    const auto& tools = conv.tools;
    const auto& opt = conv.opt;
    const std::string& text = out->text;
    const size_t n = msgs.size();
    size_t instructions = 0;
    while (instructions < n && msgs[instructions].role == "system") ++instructions;
    if (instructions || !tools.empty())
      out->system_end = probe_cache_prefix(text, {msgs.begin(), msgs.begin() + instructions}, tools, opt);
    for (const auto& h : cache_points) {
      size_t at = 0;
      if (!h.messages && h.tool_count && h.tool_count <= tools.size()) {
        at = h.tool_count == tools.size()
                 ? out->system_end
                 : probe_cache_prefix(text, {msgs.begin(), msgs.begin() + instructions},
                                      {tools.begin(), tools.begin() + h.tool_count}, opt);
      } else if (h.messages && h.messages <= n) {
        const auto& m = msgs[h.messages - 1];
        const bool parts_whole = m.parts.empty() || h.parts >= m.parts.size();
        const bool whole = h.content_bytes == m.content.size() && h.calls == m.tool_calls.size() &&
                           h.reasoning_bytes == (m.reasoning ? m.reasoning->size() : 0) && parts_whole;
        if (whole && h.messages == n && end) {
          at = *end;
        } else {
          std::vector<ChatMessage> part(msgs.begin(), msgs.begin() + h.messages);
          auto& last = part.back();
          last.content.resize(std::min(last.content.size(), h.content_bytes));
          if (!last.parts.empty()) last.parts.resize(std::min(last.parts.size(), h.parts));
          // Some templates render tools after the system text. For a mark on that text, discover where
          // it ends before any following tool definitions, independently of the combined system/tool root.
          if (last.role == "system") last.content += '\x01';
          if (last.reasoning) last.reasoning->resize(std::min(last.reasoning->size(), h.reasoning_bytes));
          last.tool_calls.resize(std::min(last.tool_calls.size(), h.calls));
          at = probe_cache_prefix(text, std::move(part), tools, opt);
        }
      }
      if (at) out->cache_prefixes.push_back(at);
    }
    auto& prefixes = out->cache_prefixes;
    std::sort(prefixes.begin(), prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(), prefixes.end()), prefixes.end());
  }

  void resolve_branch_points(const Conversation& conv, const std::vector<ChatCachePoint>& cache_points,
                             RenderedChat* out) {
    const auto& msgs = conv.msgs;
    const std::string& text = out->text;
    const size_t n = msgs.size();
    std::optional<size_t> end = probe_conversation_end(conv.tools, conv.opt, *out, &out->in_reasoning);
    // Tool results whose generation prompt is not a user turn's (Gemma 4: the reply goes on in the model's turn, its
    // thought reopened): the conversation rendered without one, where the rest is the template's own text.
    if (!end && conv.opt.add_generation_prompt && n > 0 && msgs.back().role == "tool") {
      RenderedChat r;
      std::string ignored;
      const auto& lit = out->literal;
      if (render(conv, n, false, &r, &ignored) && text.size() > r.text.size() && text.starts_with(r.text) &&
          !lit.empty() && lit.back().begin <= r.text.size() && lit.back().end == text.size()) {
        end = r.text.size();
        out->in_reasoning = chat->opens_reasoning(std::string_view(text).substr(*end));
      }
    }
    if (end) out->conversation_end = *end;

    // Branch points, where a later request is likely to continue or diverge:
    // this conversation's end, and the ends of earlier requests it continues (a
    // checkpoint is restored only on an exact token match, so a point is a
    // placement, not a promise).  On a first request, nothing earlier is known:
    // the nearest earlier end of a user message or tool results whose prefix
    // renders the same (at most two renders) stands in.
    auto candidate = [&](size_t k) {
      const std::string& role = msgs[k - 1].role;
      return role == "user" || (role == "tool" && (k == n || msgs[k].role != "tool"));
    };
    Ending hashed{0, kFnvOffset};
    std::vector<size_t> points = continued(text, end.value_or(text.size()), hashed);
    if (points.empty()) {
      int tried = 0;
      for (size_t k = n - 1; k >= 1 && tried < 2; --k) {
        if (!candidate(k)) continue;
        ++tried;
        RenderedChat r;
        std::string ignored;
        if (!render(conv, k, false, &r, &ignored)) continue;  // the template rejects this prefix
        if (text.starts_with(r.text)) {
          points.push_back(r.text.size());
          break;
        }
      }
    }
    if (end && candidate(n)) points.push_back(*end);
    if (points.size() > 3) points.erase(points.begin(), points.end() - 3);
    out->boundaries = std::move(points);
    resolve_cache_points(conv, end, cache_points, out);
    // Where the first user message ends (one render of the conversation up to it).
    for (size_t k = 1; k <= n; ++k) {
      if (msgs[k - 1].role != "user") continue;
      if (k == n && end) {
        out->first_user_end = *end;
      } else if (!native && !uses_time) {
        std::vector<ChatMessage> first_part(msgs.begin(), msgs.begin() + k);
        std::string rendered;
        if (render_prefix_text(first_part, conv.tools, conv.opt, &rendered) && text.starts_with(rendered))
          out->first_user_end = rendered.size();
      } else {
        RenderedChat r;
        std::string ignored;
        if (render(conv, k, false, &r, &ignored) && text.starts_with(r.text))
          out->first_user_end = r.text.size();
      }
      break;
    }
    if (end) {
      hashed.extend(text, *end);
      remember(hashed);
    }
  }
};

ChatTemplate::ChatTemplate() = default;
ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::load(const ModelFile& file, const std::string& template_override, std::string* err, bool native) {
  impl_.reset();
  auto chat = interfaces::open_chat(file, template_override, &error_);
  if (!chat) return *err = error_, false;
  auto impl = std::make_unique<Impl>();
  try {
    jinja::lexer lexer;
    impl->program = jinja::parse_from_tokens(lexer.tokenize(chat->template_source()));
  } catch (const std::exception& e) {
    error_ = "the chat template does not parse: " + short_error(e.what());
    return *err = error_, false;
  }
  impl->own_tokens = chat->template_tokens();
  impl->native = native && chat->native();
  impl->uses_time = chat->template_source().find("strftime_now") != std::string::npos;
  impl->chat = std::move(chat);
  impl_ = std::move(impl);
  error_.clear();
  return true;
}

bool ChatTemplate::ok() const { return impl_ != nullptr; }

const char* ChatTemplate::format() const { return impl_ ? impl_->chat->format() : "none"; }

bool ChatTemplate::native() const { return impl_ && impl_->native; }

bool ChatTemplate::render(const std::vector<ChatMessage>& msgs, const std::vector<ojson>& tools,
                          const ChatOptions& opt, RenderedChat* out, std::string* err, bool branch_points,
                          const std::vector<ChatCachePoint>& cache_points) const {
  *out = {};
  if (!impl_) return *err = error_, false;
  if (msgs.empty()) return *err = "no messages provided", false;
  if (!tools.empty() && !impl_->chat->reads_tool_calls())
    return *err = std::string("tools: this model's chat template has no tool-call format this build reads (") +
                  impl_->chat->format() + ")",
           false;
  const Impl::Conversation conv{msgs, tools, opt};
  if (!impl_->render(conv, msgs.size(), opt.add_generation_prompt, out, err)) {
    *err = "chat template: " + *err;
    return false;
  }
  if (branch_points) impl_->resolve_branch_points(conv, cache_points, out);
  return true;
}

std::string ChatTemplate::reply_grammar(const std::vector<ojson>& tools, const ReplyRules& rules,
                                        const RenderedChat& r) const {
  return impl_ ? impl_->chat->reply_grammar(tools, rules, r.in_reasoning) : std::string();
}

std::vector<std::string> ChatTemplate::reply_tokens() const {
  return impl_ ? impl_->chat->reply_tokens() : std::vector<std::string>();
}

bool ChatTemplate::thinking_markers(std::string* open, std::string* close) const {
  return impl_ && impl_->chat->thinking_markers(open, close);
}

std::unique_ptr<OutputParser> ChatTemplate::parser(const RenderedChat& r, const std::vector<ojson>* tools) const {
  if (!impl_) return nullptr;
  return impl_->chat->parser(r.in_reasoning, tools);
}

}  // namespace flyx
