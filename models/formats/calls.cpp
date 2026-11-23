// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "models/formats/calls.h"

#include <algorithm>
#include <cstdio>

#include "models/formats/pytext.h"

namespace flyx::models::formats {
namespace {

std::string dump_safe(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

// Length of the longest tail of s that is a proper prefix of tag.
size_t partial_tail(std::string_view s, std::string_view tag) {
  for (size_t l = std::min(s.size(), tag.size() - 1); l > 0; --l)
    if (s.substr(s.size() - l) == tag.substr(0, l)) return l;
  return 0;
}

// Trailing bytes that begin an incomplete UTF-8 scalar (0..3 bytes), so streaming deltas
// never split a multibyte UTF-8 character across JSON chunks.
size_t trailing_incomplete_utf8(std::string_view s) {
  const size_t n = s.size();
  if (n == 0 || static_cast<unsigned char>(s[n - 1]) < 0x80) return 0;
  const size_t lim = std::min<size_t>(n, 3);
  for (size_t back = 1; back <= lim; ++back) {
    const unsigned char b = static_cast<unsigned char>(s[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: keep looking for lead
    size_t want = 0;
    if (b >= 0xC2 && b <= 0xDF) want = 2;
    else if (b >= 0xE0 && b <= 0xEF) want = 3;
    else if (b >= 0xF0 && b <= 0xF4) want = 4;
    return back < want ? back : 0;
  }
  return 0;
}

std::string json_escape_body(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char hex[7];
          std::snprintf(hex, sizeof(hex), "\\u%04x", unsigned(c));
          out += hex;
        } else {
          out.push_back(char(c));
        }
        break;
    }
  }
  return out;
}

void emit_start(std::vector<OutputEvent>* out, const std::string& name) {
  ToolCall tc;
  tc.name = name;
  out->push_back({OutputEvent::kToolCallStart, {}, std::move(tc)});
}

void emit_delta(std::vector<OutputEvent>* out, std::string piece) {
  if (piece.empty()) return;
  if (!out->empty() && out->back().kind == OutputEvent::kToolCallDelta) out->back().text += piece;
  else out->push_back({OutputEvent::kToolCallDelta, std::move(piece), {}});
}

const ojson* find_param_schema(const std::vector<ojson>* tools, const std::string& fn, const std::string& param) {
  if (!tools) return nullptr;
  for (const auto& t : *tools) {
    const ojson* f = function_of(t);
    if (!f || !f->contains("name") || !(*f)["name"].is_string() || (*f)["name"].get<std::string>() != fn) continue;
    for (const char* key : {"parameters", "input_schema"}) {
      if (!f->contains(key) || !(*f)[key].is_object() || !(*f)[key].contains("properties")) continue;
      const ojson& props = (*f)[key]["properties"];  // a reference into the tools, not a copy
      if (props.is_object() && props.contains(param)) return &props[param];
    }
  }
  return nullptr;
}

bool allows(const ojson& type, const char* name) {
  if (type.is_string()) return type.get<std::string>() == name;
  if (type.is_array())
    for (const auto& t : type)
      if (t.is_string() && t.get<std::string>() == name) return true;
  return false;
}

// True when typed_value() is guaranteed to keep the parameter value as a JSON string,
// so its bytes can be JSON-escaped and streamed incrementally before </parameter> arrives.
bool always_string_param(const ojson* schema) {
  if (!schema || !schema->is_object()) return true;
  const ojson type = schema->contains("type") ? (*schema)["type"] : ojson();
  if (!type.is_null()) return allows(type, "string");
  return !schema->contains("anyOf") && !schema->contains("oneOf") && !schema->contains("enum");
}

// A parameter value as the tool's schema types it; strings stay strings.
ojson typed_value(const std::string& v, const ojson* schema) {
  if (!schema || !schema->is_object()) return v;
  ojson type = schema->contains("type") ? (*schema)["type"] : ojson();
  if (type.is_null() || allows(type, "string")) {
    if (type.is_null() && (schema->contains("anyOf") || schema->contains("oneOf") || schema->contains("enum"))) {
      ojson j = interfaces::parse_json(v);
      if (!j.is_discarded() && !j.is_string()) return j;
    }
    return v;
  }
  ojson j = interfaces::parse_json(v);
  if (!j.is_discarded()) return j;
  if (v == "True") return true;
  if (v == "False") return false;
  if (v == "None") return nullptr;
  return v;
}

class BufferedCallStream final : public CallStream {
 public:
  BufferedCallStream(CallReader reader, const std::vector<ojson>* tools) : reader_(reader), tools_(tools) {}
  bool started() const override { return false; }
  void feed(std::string_view chunk, std::vector<OutputEvent>*) override { body_.append(chunk); }
  bool finish(std::vector<OutputEvent>* out) override {
    ToolCall call;
    if (!reader_(body_, tools_, &call)) return false;
    emit_start(out, call.name);
    emit_delta(out, dump_safe(call.arguments));
    out->push_back({OutputEvent::kToolCall, {}, std::move(call)});
    return true;
  }

 private:
  CallReader reader_;
  const std::vector<ojson>* tools_;
  std::string body_;
};

// Streaming XML tool-call parser: emits kToolCallStart as soon as <function=NAME> closes,
// streams JSON-escaped string parameters incrementally, and buffers non-string parameters
// until </parameter> so schema-driven typing matches read_xml_call() identically.
class XmlCallStream final : public CallStream {
 public:
  explicit XmlCallStream(const std::vector<ojson>* tools) : tools_(tools) {}
  bool started() const override { return started_; }

  void feed(std::string_view chunk, std::vector<OutputEvent>* out) override {
    if (state_ == kDone) return;
    buf_.append(chunk);
    step(false, out);
  }

  bool finish(std::vector<OutputEvent>* out) override {
    if (state_ != kDone) step(true, out);
    if (!started_) return false;
    if (state_ != kDone) {
      emit_delta(out, opened_obj_ ? "}" : "{}");
      state_ = kDone;
    }
    ToolCall tc;
    tc.name = name_;
    tc.arguments = std::move(args_);
    out->push_back({OutputEvent::kToolCall, {}, std::move(tc)});
    return true;
  }

 private:
  static constexpr std::string_view kFuncOpen = "<function=";
  static constexpr std::string_view kFuncClose = "</function>";
  static constexpr std::string_view kParamOpen = "<parameter=";
  static constexpr std::string_view kParamClose = "</parameter>";

  enum State { kWaitFunc, kSeekParam, kParamString, kParamTyped, kDone };

  void close_object(std::vector<OutputEvent>* out) {
    emit_delta(out, opened_obj_ ? "}" : "{}");
    state_ = kDone;
    buf_.clear();
  }

  void step(bool final, std::vector<OutputEvent>* out) {
    for (;;) {
      if (state_ == kWaitFunc) {
        const size_t f = buf_.find(kFuncOpen);
        if (f == std::string::npos) {
          if (final) {
            buf_.clear();
            return;
          }
          const size_t keep = partial_tail(buf_, kFuncOpen);
          buf_.erase(0, buf_.size() - keep);
          return;
        }
        const size_t e = buf_.find('>', f + kFuncOpen.size());
        if (e == std::string::npos) {
          if (f > 0) buf_.erase(0, f);
          return;
        }
        name_ = std::string(py_strip(std::string_view(buf_).substr(f + kFuncOpen.size(), e - (f + kFuncOpen.size()))));
        if (name_.empty()) {
          buf_.clear();
          state_ = kDone;
          return;
        }
        buf_.erase(0, e + 1);
        started_ = true;
        emit_start(out, name_);
        state_ = kSeekParam;
        continue;
      }

      if (state_ == kSeekParam) {
        const size_t fc = buf_.find(kFuncClose);
        const size_t p = buf_.find(kParamOpen);
        if (fc != std::string::npos && (p == std::string::npos || fc < p)) {
          close_object(out);
          return;
        }
        if (p == std::string::npos) {
          if (final) {
            close_object(out);
            return;
          }
          const size_t keep = std::max(partial_tail(buf_, kParamOpen), partial_tail(buf_, kFuncClose));
          buf_.erase(0, buf_.size() - keep);
          return;
        }
        const size_t q = buf_.find('>', p + kParamOpen.size());
        if (q == std::string::npos || (fc != std::string::npos && q >= fc)) {
          if (final || (fc != std::string::npos && q >= fc)) {
            close_object(out);
            return;
          }
          if (p > 0) buf_.erase(0, p);
          return;
        }
        key_ = std::string(py_strip(std::string_view(buf_).substr(p + kParamOpen.size(), q - (p + kParamOpen.size()))));
        buf_.erase(0, q + 1);
        param_schema_ = find_param_schema(tools_, name_, key_);
        val_started_ = false;
        val_raw_.clear();
        const std::string prefix = (opened_obj_ ? "," : "{") + dump_safe(ojson(key_)) + ":";
        opened_obj_ = true;
        if (always_string_param(param_schema_)) {
          emit_delta(out, prefix + "\"");
          state_ = kParamString;
        } else {
          emit_delta(out, prefix);
          state_ = kParamTyped;
        }
        continue;
      }

      if (state_ == kParamString) {
        if (!val_started_) {
          if (buf_.empty()) {
            if (!final) return;
            args_[key_] = "";
            emit_delta(out, "\"");
            close_object(out);
            return;
          }
          if (buf_[0] == '\n') buf_.erase(0, 1);
          val_started_ = true;
        }
        const size_t pc = buf_.find(kParamClose);
        const size_t fc = buf_.find(kFuncClose);
        const bool hit_pc = pc != std::string::npos && (fc == std::string::npos || pc <= fc);
        const bool hit_fc = !hit_pc && fc != std::string::npos;
        if (hit_pc || hit_fc || final) {
          const size_t end = hit_pc ? pc : hit_fc ? fc : buf_.size();
          std::string tail = buf_.substr(0, end);
          // Strip the single trailing newline before </parameter> or </function> (matching read_xml_call).
          if (!tail.empty() && tail.back() == '\n') tail.pop_back();
          emit_delta(out, json_escape_body(tail) + "\"");
          val_raw_ += tail;
          args_[key_] = std::move(val_raw_);
          if (hit_pc) {
            buf_.erase(0, pc + kParamClose.size());
            state_ = kSeekParam;
            continue;
          }
          close_object(out);
          return;
        }
        size_t keep = std::max(partial_tail(buf_, kParamClose), partial_tail(buf_, kFuncClose));
        // Hold back one '\n' immediately preceding a potential closing tag (or end of buffer),
        // because read_xml_call strips one trailing '\n' right before </parameter> or </function>.
        if (keep < buf_.size() && buf_[buf_.size() - keep - 1] == '\n') ++keep;
        size_t safe = buf_.size() - keep;
        safe -= trailing_incomplete_utf8(std::string_view(buf_).substr(0, safe));
        if (safe > 0) {
          const std::string_view chunk(buf_.data(), safe);
          emit_delta(out, json_escape_body(chunk));
          val_raw_.append(chunk);
          buf_.erase(0, safe);
        }
        return;
      }

      // kParamTyped: buffer until </parameter>, </function>, or EOF, then apply schema typing.
      const size_t pc = buf_.find(kParamClose);
      const size_t fc = buf_.find(kFuncClose);
      const bool hit_pc = pc != std::string::npos && (fc == std::string::npos || pc <= fc);
      const bool hit_fc = !hit_pc && fc != std::string::npos;
      if (!hit_pc && !hit_fc && !final) return;
      size_t end = hit_pc ? pc : hit_fc ? fc : buf_.size();
      size_t consume = hit_pc ? pc + kParamClose.size() : end;
      if (!hit_pc) {
        const size_t next_p = buf_.find(kParamOpen);
        if (next_p != std::string::npos && next_p < end) {
          end = next_p;
          consume = next_p;
        }
      }
      std::string v = buf_.substr(0, end);
      if (!v.empty() && v.front() == '\n') v.erase(0, 1);
      if (!v.empty() && v.back() == '\n') v.pop_back();
      ojson tv = typed_value(v, param_schema_);
      emit_delta(out, dump_safe(tv));
      args_[key_] = std::move(tv);
      if (hit_pc || consume < (hit_fc ? fc : buf_.size())) {
        buf_.erase(0, consume);
        state_ = kSeekParam;
        continue;
      }
      close_object(out);
      return;
    }
  }

  const std::vector<ojson>* tools_;
  State state_ = kWaitFunc;
  bool started_ = false;
  bool opened_obj_ = false;
  bool val_started_ = false;
  std::string buf_;
  std::string name_;
  std::string key_;
  std::string val_raw_;
  const ojson* param_schema_ = nullptr;
  ojson args_ = ojson::object();
};

enum class StrScan { kNeedMore, kOk, kInvalid };

// Parses a JSON string starting at s[*pos] == '"'. Advances *pos past the closing '"' on kOk.
StrScan scan_json_string(std::string_view s, size_t* pos, std::string* out) {
  size_t i = *pos;
  if (i >= s.size()) return StrScan::kNeedMore;
  if (s[i] != '"') return StrScan::kInvalid;
  ++i;
  bool esc = false;
  for (; i < s.size(); ++i) {
    const char c = s[i];
    if (esc) {
      esc = false;
      continue;
    }
    if (c == '\\') {
      esc = true;
      continue;
    }
    if (c == '"') {
      const ojson parsed = interfaces::parse_json(std::string_view(s).substr(*pos, i - *pos + 1));
      if (parsed.is_discarded() || !parsed.is_string()) return StrScan::kInvalid;
      *out = parsed.get<std::string>();
      *pos = i + 1;
      return StrScan::kOk;
    }
  }
  return StrScan::kNeedMore;
}

void skip_ws(std::string_view s, size_t* pos) {
  while (*pos < s.size()) {
    const char c = s[*pos];
    if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
    ++(*pos);
  }
}

// Streaming JSON tool-call parser: when the body starts with {"name": "<fn>", "arguments": {...,
// emits kToolCallStart immediately and streams the raw JSON arguments object incrementally.
// Falls back to buffered read_json_call() for stringified arguments or non-canonical key order.
class JsonCallStream final : public CallStream {
 public:
  explicit JsonCallStream(const std::vector<ojson>* tools) : tools_(tools) {}
  bool started() const override { return started_; }

  void feed(std::string_view chunk, std::vector<OutputEvent>* out) override {
    if (state_ == kDone) return;
    if (state_ == kBufferAll) {
      raw_.append(chunk);
      return;
    }
    if (!started_) raw_.append(chunk);
    buf_.append(chunk);
    step(false, out);
  }

  bool finish(std::vector<OutputEvent>* out) override {
    if (state_ == kHeader || state_ == kBufferAll) {
      ToolCall call;
      if (!read_json_call(raw_, tools_, &call)) return false;
      emit_start(out, call.name);
      emit_delta(out, dump_safe(call.arguments));
      out->push_back({OutputEvent::kToolCall, {}, std::move(call)});
      state_ = kDone;
      return true;
    }
    if (state_ == kStreamArgs) {
      step(true, out);
      if (!stack_.empty()) {
        std::string containers(stack_.rbegin(), stack_.rend());
        std::string str_close;
        if (in_str_) {
          if (esc_) str_close.push_back('\\');
          str_close.push_back('"');
        }
        std::vector<std::string> candidates;
        candidates.push_back(str_close + containers);
        if (in_str_) {
          candidates.push_back(str_close + ":null" + containers);
        } else {
          size_t end = args_raw_.size();
          while (end > 0 && (args_raw_[end - 1] == ' ' || args_raw_[end - 1] == '\t' || args_raw_[end - 1] == '\n' ||
                             args_raw_[end - 1] == '\r'))
            --end;
          const char last = end > 0 ? args_raw_[end - 1] : '\0';
          if (last == ':') candidates.push_back("null" + containers);
          else if (last == '"') candidates.push_back(":null" + containers);
        }
        for (const std::string& suffix : candidates) {
          ojson try_parsed = interfaces::parse_json(args_raw_ + suffix);
          if (!try_parsed.is_discarded() && try_parsed.is_object()) {
            args_raw_ += suffix;
            emit_delta(out, suffix);
            stack_.clear();
            in_str_ = false;
            esc_ = false;
            break;
          }
        }
      }
    }
    ojson parsed = interfaces::parse_json(args_raw_);
    ToolCall call;
    call.name = name_;
    call.arguments = (!parsed.is_discarded() && parsed.is_object()) ? std::move(parsed) : ojson::object();
    out->push_back({OutputEvent::kToolCall, {}, std::move(call)});
    state_ = kDone;
    return true;
  }

 private:
  enum State { kHeader, kStreamArgs, kDone, kBufferAll };

  void step(bool final, std::vector<OutputEvent>* out) {
    if (state_ == kHeader) {
      size_t p = 0;
      skip_ws(buf_, &p);
      if (p >= buf_.size()) return;
      if (buf_[p] != '{') {
        state_ = kBufferAll;
        return;
      }
      ++p;
      skip_ws(buf_, &p);
      std::string k1;
      const StrScan r1 = scan_json_string(buf_, &p, &k1);
      if (r1 == StrScan::kNeedMore) return;
      if (r1 == StrScan::kInvalid || k1 != "name") {
        state_ = kBufferAll;
        return;
      }
      skip_ws(buf_, &p);
      if (p >= buf_.size()) return;
      if (buf_[p] != ':') {
        state_ = kBufferAll;
        return;
      }
      ++p;
      skip_ws(buf_, &p);
      std::string fn;
      const StrScan rv = scan_json_string(buf_, &p, &fn);
      if (rv == StrScan::kNeedMore) return;
      if (rv == StrScan::kInvalid || fn.empty()) {
        state_ = kBufferAll;
        return;
      }
      skip_ws(buf_, &p);
      if (p >= buf_.size()) return;
      if (buf_[p] == '}') {
        // No arguments to stream: buffer so finish() validates trailing text via read_json_call.
        state_ = kBufferAll;
        return;
      }
      if (buf_[p] != ',') {
        state_ = kBufferAll;
        return;
      }
      ++p;
      skip_ws(buf_, &p);
      std::string k2;
      const StrScan r2 = scan_json_string(buf_, &p, &k2);
      if (r2 == StrScan::kNeedMore) return;
      if (r2 == StrScan::kInvalid || (k2 != "arguments" && k2 != "parameters")) {
        state_ = kBufferAll;
        return;
      }
      skip_ws(buf_, &p);
      if (p >= buf_.size()) return;
      if (buf_[p] != ':') {
        state_ = kBufferAll;
        return;
      }
      ++p;
      skip_ws(buf_, &p);
      if (p >= buf_.size()) return;
      if (buf_[p] != '{') {
        // Stringified JSON arguments, null, or invalid: let finish() handle via read_json_call.
        state_ = kBufferAll;
        return;
      }
      // Peek at the first non-whitespace byte inside the arguments object: an empty object (`}`) has nothing to
      // stream, and anything other than a JSON key quote (`"`) is malformed JSON that must stay text via read_json_call.
      size_t q = p + 1;
      skip_ws(buf_, &q);
      if (q >= buf_.size()) return;
      if (buf_[q] != '"') {
        state_ = kBufferAll;
        return;
      }
      name_ = std::move(fn);
      started_ = true;
      emit_start(out, name_);
      emit_delta(out, "{");
      args_raw_ = "{";
      stack_.assign(1, '}');
      in_str_ = false;
      esc_ = false;
      scan_ = 0;
      buf_.erase(0, p + 1);
      raw_.clear();
      state_ = kStreamArgs;
    }

    if (state_ == kStreamArgs) {
      size_t end_pos = buf_.size();
      bool closed = false;
      for (; scan_ < buf_.size(); ++scan_) {
        const char c = buf_[scan_];
        if (in_str_) {
          if (esc_) esc_ = false;
          else if (c == '\\') esc_ = true;
          else if (c == '"') in_str_ = false;
          continue;
        }
        if (c == '"') in_str_ = true;
        else if (c == '{') stack_.push_back('}');
        else if (c == '[') stack_.push_back(']');
        else if ((c == '}' || c == ']') && !stack_.empty() && stack_.back() == c) {
          stack_.pop_back();
          if (stack_.empty()) {
            end_pos = scan_ + 1;
            closed = true;
            break;
          }
        }
      }
      size_t safe = end_pos;
      if (!closed && !final) safe -= trailing_incomplete_utf8(std::string_view(buf_).substr(0, safe));
      if (safe > 0) {
        std::string piece = buf_.substr(0, safe);
        args_raw_ += piece;
        emit_delta(out, std::move(piece));
        buf_.erase(0, safe);
        scan_ = scan_ > safe ? scan_ - safe : 0;
      }
      if (closed) {
        buf_.clear();
        scan_ = 0;
        state_ = kDone;
      }
    }
  }

  const std::vector<ojson>* tools_;
  State state_ = kHeader;
  bool started_ = false;
  std::vector<char> stack_;
  bool in_str_ = false;
  bool esc_ = false;
  size_t scan_ = 0;
  std::string raw_;
  std::string buf_;
  std::string name_;
  std::string args_raw_;
};

}  // namespace

std::unique_ptr<CallStream> open_call_stream(CallReader reader, const std::vector<ojson>* tools) {
  if (reader == read_xml_call) return std::make_unique<XmlCallStream>(tools);
  if (reader == read_json_call) return std::make_unique<JsonCallStream>(tools);
  return std::make_unique<BufferedCallStream>(reader, tools);
}

// The templates' own spellings (Qwen3.5's chat_template.jinja; Qwen3's).
// XML uses "</function>\n</tool_call>" without a leading newline so zero-parameter functions match
// Qwen3.5's template spelling, and triggers on "<tool_call>" so non-newline variants cannot bypass the grammar.
const CallSyntax kXmlCallSyntax = {"<tool_call>\n<function=", ">\n", "</function>\n</tool_call>", "<tool_call>",
                                   "qwen_xml"};
const CallSyntax kJsonCallSyntax = {"<tool_call>\n{\"name\": \"", "\", \"arguments\": ", "}\n</tool_call>",
                                    "<tool_call>", "json"};

bool read_xml_call(const std::string& body, const std::vector<ojson>* tools, ToolCall* out) {
  size_t f = body.find("<function=");
  if (f == std::string::npos) return false;
  size_t e = body.find('>', f);
  if (e == std::string::npos) return false;
  out->name = py_strip(std::string_view(body).substr(f + 10, e - f - 10));
  if (out->name.empty()) return false;
  out->arguments = ojson::object();
  size_t limit = body.find("</function>", e + 1);
  if (limit == std::string::npos) limit = body.size();
  size_t pos = e + 1;
  while (pos < limit) {
    size_t p = body.find("<parameter=", pos);
    if (p == std::string::npos || p >= limit) break;
    size_t q = body.find('>', p);
    if (q == std::string::npos || q >= limit) return false;
    std::string key(py_strip(std::string_view(body).substr(p + 11, q - p - 11)));
    size_t end = body.find("</parameter>", q);
    size_t next_pos;
    if (end != std::string::npos && end <= limit) {
      next_pos = end + 12;
    } else {
      size_t next = body.find("<parameter=", q);
      end = (next != std::string::npos && next < limit) ? next : limit;
      next_pos = end;
    }
    std::string v = body.substr(q + 1, end - q - 1);
    if (!v.empty() && v.front() == '\n') v.erase(0, 1);
    if (!v.empty() && v.back() == '\n') v.pop_back();
    out->arguments[key] = typed_value(v, find_param_schema(tools, out->name, key));
    pos = next_pos;
  }
  return true;
}

bool read_json_call(const std::string& body, const std::vector<ojson>*, ToolCall* out) {
  ojson j = interfaces::parse_json(body);
  if (j.is_discarded() || !j.is_object() || !j.contains("name") || !j["name"].is_string()) return false;
  out->name = j["name"].get<std::string>();
  if (out->name.empty()) return false;
  // "arguments" (or "parameters"): an object, or an object written as JSON text.
  const char* key = j.contains("arguments") ? "arguments" : "parameters";
  ojson args = j.contains(key) ? j[key] : ojson::object();
  if (args.is_string()) args = interfaces::parse_json(args.get<std::string>());
  if (args.is_null()) args = ojson::object();
  if (!args.is_object()) return false;
  out->arguments = std::move(args);
  return true;
}

}  // namespace flyx::models::formats
