// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Qwen3.5's chat format: its template (chat_template.jinja of Qwen/Qwen3.5-4B,
// sha256 a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715,
// also the NVFP4 checkpoints') written out in C++, and its replies: thinking
// in <think>, tool calls as XML (calls.h).  The renderer follows the template
// statement by statement; tests/models/formats/qwen35_test.cpp holds it to
// the Jinja engine on generated conversations, and the chat test to
// transformers.  Text and image parts are rendered identically to the template.
#include "models/formats/qwen35.h"

#include <algorithm>
#include <cmath>

#include "models/formats/pytext.h"

namespace flyx::models::formats {
namespace {

constexpr std::string_view kToolsHeader = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
constexpr std::string_view kToolsFooter =
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
    "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> "
    "XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function "
    "call in natural language BEFORE the function call, but NOT after\n- If there is no function call "
    "available, answer the question like normal with your current knowledge and do not tell the user about "
    "function calls\n</IMPORTANT>";

std::string_view rstrip_char(std::string_view s, char c) {
  size_t e = s.size();
  while (e > 0 && s[e - 1] == c) --e;
  return s.substr(0, e);
}

std::string_view lstrip_char(std::string_view s, char c) {
  size_t b = 0;
  while (b < s.size() && s[b] == c) ++b;
  return s.substr(b);
}

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }
bool ends_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.substr(s.size() - p.size()) == p;
}

bool fail(std::string* err, const char* message) {
  *err = message;
  return false;
}

bool py_truthy(const ojson& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number_unsigned()) return v.get<uint64_t>() != 0;
  if (v.is_number_integer()) return v.get<int64_t>() != 0;
  if (v.is_number_float()) {
    const double d = v.get<double>();
    return std::isfinite(d) && d != 0.0;
  }
  if (v.is_string()) return !v.get_ref<const std::string&>().empty();
  if (v.is_array() || v.is_object()) return !v.empty();
  return false;
}

struct RenderedPiece {
  bool is_lit = false;
  std::string text;
  std::shared_ptr<const interfaces::ImageInput> img;
};

struct TrimmedContent {
  std::vector<RenderedPiece> pieces;
  std::string full;  // concatenation of pieces[i].text
};

// Mirrors Jinja `render_content(message.content, do_vision_count, is_system_content)|trim` byte for byte,
// preserving literal-vs-input provenance and image anchor pointers.
bool render_content_trimmed(const ChatMessage& m, bool do_vision_count, bool is_system, bool add_vision_id,
                            int* image_count, TrimmedContent* out, std::string* err) {
  out->pieces.clear();
  out->full.clear();
  std::vector<RenderedPiece> raw;
  if (m.parts.empty()) {
    const std::string_view s = py_strip(m.content);
    if (!s.empty()) {
      out->pieces.push_back({false, std::string(s), nullptr});
      out->full.assign(s);
    }
    return true;
  }
  raw.reserve(m.parts.size() * 3);
  for (const auto& p : m.parts) {
    if (p.kind == interfaces::ContentPart::kImage) {
      if (is_system) return fail(err, "System message cannot contain images.");
      if (do_vision_count) ++(*image_count);
      if (add_vision_id) raw.push_back({true, "Picture " + std::to_string(*image_count) + ": ", nullptr});
      raw.push_back({true, "<|vision_start|>", nullptr});
      raw.push_back({true, "<|image_pad|>", p.image});
      raw.push_back({true, "<|vision_end|>", nullptr});
    } else if (!p.text.empty()) {
      raw.push_back({false, p.text, nullptr});
    }
  }
  // Apply |trim across the joined piece sequence without stripping inside non-whitespace tokens or images.
  size_t lo = 0;
  while (lo < raw.size() && !raw[lo].is_lit) {
    const std::string_view sv = raw[lo].text;
    const std::string_view st = py_strip(sv);
    if (st.empty()) {
      ++lo;
      continue;
    }
    const size_t lead = size_t(st.data() - sv.data());
    if (lead > 0) raw[lo].text.erase(0, lead);
    break;
  }
  size_t hi = raw.size();
  while (hi > lo && !raw[hi - 1].is_lit) {
    const std::string_view sv = raw[hi - 1].text;
    const std::string_view st = py_strip(sv);
    if (st.empty()) {
      --hi;
      continue;
    }
    const size_t end = size_t(st.data() - sv.data()) + st.size();
    raw[hi - 1].text.resize(end);
    break;
  }
  for (size_t k = lo; k < hi; ++k) {
    out->full += raw[k].text;
    out->pieces.push_back(std::move(raw[k]));
  }
  return true;
}

// Emits the byte slice [begin, end) of `tc` through `w`, preserving literal spans and image anchors.
void emit_slice(Writer& w, const TrimmedContent& tc, size_t begin, size_t end) {
  if (begin >= end) return;
  size_t cur = 0;
  for (const auto& p : tc.pieces) {
    const size_t p_end = cur + p.text.size();
    if (p_end > begin && cur < end) {
      const size_t s0 = std::max(cur, begin) - cur;
      const size_t s1 = std::min(p_end, end) - cur;
      const std::string_view sub(p.text.data() + s0, s1 - s0);
      if (p.img && s0 == 0 && s1 == p.text.size()) w.image(p.img, sub);
      else if (p.is_lit) w.lit(sub);
      else w.in(sub);
    }
    cur = p_end;
  }
}

bool render_qwen35(std::span<const ChatMessage> msgs, const std::vector<ojson>& tools, const ChatOptions& opt,
                   bool generation_prompt, RenderedChat* out, std::string* err) {
  Writer w(out);
  const size_t n = msgs.size();
  if (n == 0) return fail(err, "No messages provided.");
  const bool add_vision_id =
      opt.kwargs.is_object() && opt.kwargs.contains("add_vision_id") && py_truthy(opt.kwargs["add_vision_id"]);
  int image_count = 0;

  // The system block: the tools and their instructions, then the system message.
  if (!tools.empty()) {
    w.lit("<|im_start|>system\n");
    w.lit(kToolsHeader);
    for (const auto& t : tools) {
      w.lit("\n");
      w.text(py_json(t), holds_client_text(t));  // tojson
    }
    w.lit(kToolsFooter);
    if (msgs[0].role == "system") {
      TrimmedContent sc;
      if (!render_content_trimmed(msgs[0], false, true, add_vision_id, &image_count, &sc, err)) return false;
      if (!sc.full.empty()) {
        w.lit("\n\n");
        emit_slice(w, sc, 0, sc.full.size());
      }
    }
    w.lit("<|im_end|>\n");
  } else if (msgs[0].role == "system") {
    TrimmedContent sc;
    if (!render_content_trimmed(msgs[0], false, true, add_vision_id, &image_count, &sc, err)) return false;
    w.lit("<|im_start|>system\n");
    emit_slice(w, sc, 0, sc.full.size());
    w.lit("<|im_end|>\n");
  }

  // The last user message that is a question (not tool results wrapped as a user message).
  bool multi_step_tool = true;
  size_t last_query = n - 1;
  for (size_t i = n; i-- > 0;) {
    if (!multi_step_tool || msgs[i].role != "user") continue;
    TrimmedContent uc;
    if (!render_content_trimmed(msgs[i], false, false, add_vision_id, &image_count, &uc, err)) return false;
    const std::string_view content = uc.full;
    if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
      multi_step_tool = false;
      last_query = i;
    }
  }
  if (multi_step_tool) return fail(err, "No user query found in messages.");

  for (size_t i = 0; i < n; ++i) {
    const ChatMessage& m = msgs[i];
    TrimmedContent tc;
    if (!render_content_trimmed(m, true, false, add_vision_id, &image_count, &tc, err)) return false;
    if (m.role == "system") {
      if (i != 0) return fail(err, "System message must be at the beginning.");
    } else if (m.role == "user") {
      w.lit("<|im_start|>");
      w.in(m.role);
      w.lit("\n");
      emit_slice(w, tc, 0, tc.full.size());
      w.lit("<|im_end|>\n");
    } else if (m.role == "assistant") {
      std::string_view full = tc.full;
      size_t content_begin = 0;
      const bool explicit_reasoning = m.reasoning.has_value();
      std::string_view reasoning_sv;
      size_t r_begin = 0, r_end = 0;
      if (explicit_reasoning) {
        reasoning_sv = py_strip(*m.reasoning);
      } else if (const size_t first_close = full.find("</think>"); first_close != std::string_view::npos) {
        const std::string_view head = rstrip_char(full.substr(0, first_close), '\n');
        const size_t open = head.rfind("<think>");
        const std::string_view raw_r = lstrip_char(open == std::string_view::npos ? head : head.substr(open + 7), '\n');
        const std::string_view stripped_r = py_strip(raw_r);
        r_begin = size_t(stripped_r.data() - full.data());
        r_end = r_begin + stripped_r.size();
        const size_t last_close = full.rfind("</think>");
        const std::string_view tail = lstrip_char(full.substr(last_close + 8), '\n');
        content_begin = size_t(tail.data() - full.data());
      }
      w.lit("<|im_start|>");
      w.in(m.role);
      if (i > last_query) {
        w.lit("\n<think>\n");
        if (explicit_reasoning) w.in(reasoning_sv);
        else emit_slice(w, tc, r_begin, r_end);
        w.lit("\n</think>\n\n");
      } else {
        w.lit("\n");
      }
      emit_slice(w, tc, content_begin, full.size());
      const bool has_body = !py_strip(full.substr(content_begin)).empty();
      for (size_t c = 0; c < m.tool_calls.size(); ++c) {
        const auto& call = m.tool_calls[c];
        if (c > 0) w.lit("\n<tool_call>\n<function=");
        else w.lit(!has_body ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=");
        w.in(call.name);
        w.lit(">\n");
        if (call.arguments.is_object())
          for (auto it = call.arguments.begin(); it != call.arguments.end(); ++it) {
            w.lit("<parameter=");
            w.in(it.key());
            w.lit(">\n");
            const ojson& v = it.value();
            if (v.is_object() || v.is_array()) w.text(py_json(v), holds_client_text(v));  // tojson
            else if (v.is_string()) w.in(v.get<std::string>());   // | string keeps the client's string
            else w.lit(py_str(v));
            w.lit("\n</parameter>\n");
          }
        w.lit("</function>\n</tool_call>");
      }
      w.lit("<|im_end|>\n");
    } else if (m.role == "tool") {
      if (i > 0 && msgs[i - 1].role != "tool") w.lit("<|im_start|>user");
      w.lit("\n<tool_response>\n");
      emit_slice(w, tc, 0, tc.full.size());
      w.lit("\n</tool_response>");
      if (i + 1 == n || msgs[i + 1].role != "tool") w.lit("<|im_end|>\n");
    } else {
      return fail(err, "Unexpected message role.");
    }
  }

  if (generation_prompt) {
    w.lit("<|im_start|>assistant\n");
    // `enable_thinking is defined and enable_thinking is false`: only a boolean false turns it off.
    const ojson* given = opt.kwargs.contains("enable_thinking") ? &opt.kwargs["enable_thinking"] : nullptr;
    const bool off = given ? (given->is_boolean() && !given->get<bool>()) : !opt.enable_thinking;
    w.lit(off ? "<think>\n\n</think>\n\n" : "<think>\n");
  }
  return true;
}

}  // namespace

const Family kQwen35 = {"qwen35", 7756, 0xcdff116097e1144aull, render_qwen35,
                        {true, read_xml_call, &kXmlCallSyntax}};

}  // namespace flyx::models::formats
