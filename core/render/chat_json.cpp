// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "core/render/chat_json.h"

#include <memory>
#include <set>

#include "core/vision/image.h"

namespace flyx {
namespace {

std::string str_or(const ojson& j, const char* key, const std::string& dflt = "") {
  if (!j.is_object() || !j.contains(key) || !j[key].is_string()) return dflt;
  return j[key].get<std::string>();
}

// Tool-call arguments: JSON text (OpenAI) or an object; anything else, text that is not an object's included, is
// read as no arguments.  Text nested deeper than a request's own JSON may be is refused (false, *err): parsed, it
// would be the nesting the request's body was checked not to have.
bool parse_arguments(const ojson& a, ojson* out, std::string* err) {
  *out = ojson::object();
  if (a.is_object()) return *out = a, true;
  if (!a.is_string()) return true;
  const std::string& text = a.get_ref<const std::string&>();
  if (!interfaces::json_depth_within(text))
    return *err = "tool call arguments are nested more than " + std::to_string(interfaces::kMaxJsonDepth) +
                  " levels deep", false;
  ojson j = ojson::parse(text, nullptr, false);
  if (!j.is_discarded() && j.is_object()) *out = std::move(j);
  return true;
}

bool parse_image_part(const ojson& part, const std::string& type, std::shared_ptr<const ImageInput>* out,
                      std::string* err) {
  ImageInput img;
  if (type == "image_url") {
    const auto it = part.find("image_url");
    if (it == part.end()) return *err = "image_url part missing 'image_url' field", false;
    std::string url, detail;
    if (it->is_string()) {
      url = it->get<std::string>();
    } else if (it->is_object()) {
      url = str_or(*it, "url");
      detail = str_or(*it, "detail");
    } else {
      return *err = "image_url must be a string or an object with 'url'", false;
    }
    if (!vision::parse_image_data_uri(url, &img, err)) return false;
    img.detail = std::move(detail);
  } else if (type == "input_image") {
    std::string url = str_or(part, "image_url");
    if (url.empty()) return *err = "input_image part missing 'image_url' data URI", false;
    if (!vision::parse_image_data_uri(url, &img, err)) return false;
    img.detail = str_or(part, "detail");
  } else if (type == "image") {
    const auto src = part.find("source");
    if (src != part.end() && src->is_object()) {
      const std::string stype = str_or(*src, "type");
      if (stype == "base64") {
        const std::string media = str_or(*src, "media_type");
        const std::string data = str_or(*src, "data");
        if (data.empty()) return *err = "image source missing base64 'data'", false;
        std::vector<uint8_t> bytes;
        if (!vision::decode_base64(data, &bytes, err)) return false;
        if (!vision::make_image_from_bytes(std::move(bytes), media.empty() ? "image/png" : media, "", &img, err))
          return false;
      } else if (stype == "url") {
        return *err = "remote image URLs are not fetched; send base64 image data", false;
      } else {
        return *err = "image source.type must be 'base64'", false;
      }
    } else if (part.contains("url") && part["url"].is_string()) {
      if (!vision::parse_image_data_uri(part["url"].get_ref<const std::string&>(), &img, err)) return false;
    } else {
      return *err = "image part missing 'source' object or 'url' data URI", false;
    }
  } else {
    return *err = "unsupported image part type '" + type + "'", false;
  }
  *out = std::make_shared<ImageInput>(std::move(img));
  return true;
}

bool parse_content_parts(const ojson& content, std::string* text_out, std::vector<ContentPart>* parts_out,
                         bool allow_images, std::string* err) {
  text_out->clear();
  if (parts_out) parts_out->clear();
  if (content.is_null()) return true;
  if (content.is_string()) {
    *text_out = content.get<std::string>();
    return true;
  }
  if (!content.is_array()) {
    *err = "message content must be a string or a list of parts";
    return false;
  }
  bool has_image = false;
  std::vector<ContentPart> parts;
  for (const auto& part : content) {
    if (part.is_string()) {
      const std::string s = part.get<std::string>();
      *text_out += s;
      parts.push_back(ContentPart::make_text(s));
      continue;
    }
    if (!part.is_object()) {
      *err = "every content part must be a string or an object";
      return false;
    }
    const std::string type = str_or(part, "type", "text");
    if (type == "text" || type == "input_text" || type == "output_text") {
      const std::string s = str_or(part, "text");
      *text_out += s;
      parts.push_back(ContentPart::make_text(s));
    } else if (type == "image" || type == "image_url" || type == "input_image") {
      if (!allow_images) {
        *err = "content of type '" + type + "' is not supported here";
        return false;
      }
      std::shared_ptr<const ImageInput> img;
      if (!parse_image_part(part, type, &img, err)) return false;
      has_image = true;
      parts.push_back(ContentPart::make_image(std::move(img)));
    } else if (type == "video" || type == "input_audio" || type == "file" || type == "document") {
      *err = "content of type '" + type + "' is not supported";
      return false;
    } else {
      *err = "unknown content part type '" + type + "'";
      return false;
    }
  }
  if (has_image && parts_out) *parts_out = std::move(parts);
  return true;
}

}  // namespace

bool content_text(const ojson& content, std::string* out, std::string* err) {
  return parse_content_parts(content, out, nullptr, false, err);
}

bool messages_from_openai(const ojson& messages, std::vector<ChatMessage>* out, std::string* err) {
  out->clear();
  if (!messages.is_array() || messages.empty()) {
    *err = "messages must be a non-empty list";
    return false;
  }
  for (const auto& m : messages) {
    if (!m.is_object()) {
      *err = "every message must be an object";
      return false;
    }
    ChatMessage cm;
    cm.role = str_or(m, "role");
    if (cm.role == "developer") cm.role = "system";
    if (cm.role == "function") cm.role = "tool";
    if (m.contains("content") && !parse_content_parts(m["content"], &cm.content, &cm.parts, true, err)) return false;
    for (const char* key : {"reasoning_content", "reasoning"})
      if (m.contains(key) && m[key].is_string()) {
        cm.reasoning = m[key].get<std::string>();
        break;
      }
    if (const auto tcs = m.find("tool_calls"); tcs != m.end() && !tcs->is_null()) {
      if (!tcs->is_array()) {
        *err = "tool_calls must be a list";
        return false;
      }
      for (const auto& tc : *tcs) {
        if (!tc.is_object()) {
          *err = "every tool call must be an object";
          return false;
        }
        const auto fn = tc.find("function");
        if (fn != tc.end() && !fn->is_null() && !fn->is_object()) {
          *err = "tool_call.function must be an object";
          return false;
        }
        const ojson& f = (fn != tc.end() && fn->is_object()) ? *fn : tc;
        ToolCall call;
        call.id = str_or(tc, "id");
        call.name = str_or(f, "name");
        if (f.contains("arguments") && !parse_arguments(f["arguments"], &call.arguments, err)) return false;
        cm.tool_calls.push_back(std::move(call));
      }
    }
    cm.tool_call_id = str_or(m, "tool_call_id");
    out->push_back(std::move(cm));
  }
  return true;
}

bool tools_from_openai(const ojson& tools, std::vector<ojson>* out, std::string* err) {
  out->clear();
  if (tools.is_null()) return true;
  if (!tools.is_array()) {
    *err = "tools must be a list";
    return false;
  }
  for (const auto& t : tools) {
    if (!t.is_object()) {
      *err = "every tool must be an object";
      return false;
    }
    out->push_back(t);
  }
  return true;
}

namespace {

bool has_tool(const std::vector<ojson>& tools, const std::string& name) {
  for (const auto& t : tools) {
    const ojson& f = t.contains("function") && t["function"].is_object() ? t["function"] : t;
    if (str_or(f, "name") == name && f.is_object() && f.contains("name") && f["name"].is_string()) return true;
  }
  return false;
}

std::string responses_function_name(const std::string& scope, const std::string& name) {
  return scope.empty() ? name : scope + "." + name;
}

}  // namespace

bool rules_from_openai(const ojson& req, const std::vector<ojson>& tools, ReplyRules* out, std::string* err) {
  *out = {};
  if (const auto ptc = req.find("parallel_tool_calls"); ptc != req.end() && !ptc->is_null()) {
    if (!ptc->is_boolean()) return *err = "parallel_tool_calls must be a boolean", false;
    out->parallel = ptc->get<bool>();
  }
  if (req.contains("tool_choice") && !req["tool_choice"].is_null()) {
    const ojson& tc = req["tool_choice"];
    std::string mode = tc.is_string() ? tc.get<std::string>() : str_or(tc, "type");
    if (mode == "auto") out->tools = ReplyRules::kAuto;
    else if (mode == "none") out->tools = ReplyRules::kNone;
    else if (mode == "required") out->tools = ReplyRules::kRequired;
    else if (mode == "function" && tc.is_object()) {
      const ojson& f = tc.contains("function") && tc["function"].is_object() ? tc["function"] : tc;
      out->tools = ReplyRules::kForced;
      out->forced = str_or(f, "name");
    } else {
      return *err = "tool_choice must be \"auto\", \"none\", \"required\" or a function", false;
    }
  }
  if (out->tools == ReplyRules::kForced && !has_tool(tools, out->forced))
    return *err = "tool_choice names the function '" + out->forced + "', which is not among the tools", false;
  if ((out->tools == ReplyRules::kRequired || out->tools == ReplyRules::kForced) && tools.empty())
    return *err = "tool_choice requires a tool call, but no tools are given", false;
  // response_format: a JSON answer (when the reply is not a tool call).
  if (const auto rf_it = req.find("response_format"); rf_it != req.end() && !rf_it->is_null()) {
    if (!rf_it->is_object()) return *err = "response_format must be an object", false;
    const ojson& rf = *rf_it;
    std::string type = "text";
    if (rf.contains("type")) {
      if (!rf["type"].is_string())
        return *err = "response_format type must be \"text\", \"json_object\" or \"json_schema\"", false;
      type = rf["type"].get<std::string>();
    }
    if (type == "json_object") {
      out->json = true;
      out->schema = {{"type", "object"}};
    } else if (type == "json_schema") {
      out->json = true;
      const auto js_it = rf.find("json_schema");
      if (js_it != rf.end() && !js_it->is_null() && !js_it->is_object())
        return *err = "response_format.json_schema must be an object", false;
      const ojson& js = (js_it != rf.end() && js_it->is_object()) ? *js_it : rf;
      out->schema = js.contains("schema") ? js["schema"] : ojson(true);
    } else if (type != "text") {
      return *err = "response_format type must be \"text\", \"json_object\" or \"json_schema\"", false;
    }
  }
  return true;
}

std::string reasoning_text(const ojson& item) {
  std::string text;
  for (const char* key : {"content", "summary"}) {
    if (!item.contains(key) || !item[key].is_array()) continue;
    for (const auto& part : item[key]) {
      if (!text.empty() && key[0] == 's') text += "\n\n";
      text += str_or(part, "text");
    }
    if (!text.empty()) break;
  }
  const std::string sealed = str_or(item, "encrypted_content");
  if (text.empty() && sealed.starts_with(kReasoningMark)) text = sealed.substr(kReasoningMark.size());
  return text;
}

bool messages_from_responses(const std::string& instructions, const ojson& items, std::vector<ChatMessage>* out,
                             std::string* err) {
  out->clear();
  if (!instructions.empty()) out->push_back({"system", instructions, std::nullopt, {}, ""});
  if (!items.is_array()) return *err = "input must be a string or a list of items", false;
  bool open = false;  // the last message is an assistant turn later assistant items add to
  auto assistant = [&](bool fresh) -> ChatMessage& {
    if (!open || fresh) out->push_back({"assistant", "", std::nullopt, {}, ""});
    open = true;
    return out->back();
  };
  for (const auto& it : items) {
    if (!it.is_object()) return *err = "every input item must be an object", false;
    const std::string type = str_or(it, "type", it.contains("role") ? "message" : "");
    if (type == "message") {
      std::string role = str_or(it, "role"), text;
      std::vector<ContentPart> parts;
      const auto content = it.find("content");
      if (content != it.end() && !parse_content_parts(*content, &text, &parts, role != "assistant", err)) return false;
      if (role == "assistant") {
        ChatMessage& m = assistant(open && !out->back().tool_calls.empty());  // text after calls starts a turn
        m.content += text;
        continue;
      }
      if (role == "developer") role = "system";
      if (role != "user" && role != "system")
        return *err = "message role must be user, assistant, system or developer (got '" + role + "')", false;
      ChatMessage cm{role, std::move(text), std::nullopt, {}, ""};
      cm.parts = std::move(parts);
      out->push_back(std::move(cm));
      open = false;
    } else if (type == "reasoning") {
      const ChatMessage* last = open ? &out->back() : nullptr;
      ChatMessage& m =
          assistant(last && (last->reasoning || !last->content.empty() || !last->tool_calls.empty()));
      m.reasoning = reasoning_text(it);
    } else if (type == "function_call") {
      ToolCall call;
      call.id = str_or(it, "call_id");
      call.name = str_or(it, "name");
      if (call.name.empty()) return *err = "function_call item without a name", false;
      call.name = responses_function_name(str_or(it, "namespace"), call.name);
      if (it.contains("arguments") && !parse_arguments(it["arguments"], &call.arguments, err)) return false;
      assistant(false).tool_calls.push_back(std::move(call));
    } else if (type == "function_call_output") {
      std::string text;
      std::vector<ContentPart> parts;
      const auto output = it.find("output");
      if (output != it.end() && !parse_content_parts(*output, &text, &parts, true, err)) return false;
      ChatMessage tm{"tool", std::move(text), std::nullopt, {}, str_or(it, "call_id")};
      tm.parts = std::move(parts);
      out->push_back(std::move(tm));
      open = false;
    } else if (type == "item_reference") {
      return *err = "item_reference is not supported: send the item, or previous_response_id", false;
    } else {
      return *err = "input item type '" + type + "' is not supported", false;
    }
  }
  if (out->empty() || (out->size() == 1 && !instructions.empty())) return *err = "input is empty", false;
  return true;
}

bool tools_from_responses(const ojson& tools, std::vector<ojson>* out, std::string* err,
                          ResponsesNamespaces* namespaces) {
  out->clear();
  if (namespaces) namespaces->clear();
  if (tools.is_null()) return true;
  if (!tools.is_array()) return *err = "tools must be a list", false;
  std::set<std::string> names;
  auto function = [&](const ojson& t, const std::string& scope, const std::string& description) {
    if (str_or(t, "type") != "function")
      return *err = "tool type '" + str_or(t, "type") + "' is not supported (function tools only)", false;
    const std::string name = str_or(t, "name");
    if (name.empty()) return *err = "a function tool needs a name", false;
    const std::string flat = responses_function_name(scope, name);
    if (!names.insert(flat).second) return *err = "ambiguous function name '" + flat + "'", false;
    ojson f = ojson::object();
    f["name"] = flat;
    for (const char* key : {"description", "parameters", "strict"})
      if (t.contains(key) && !t[key].is_null()) f[key] = t[key];
    if (!description.empty())
      f["description"] = description + (str_or(t, "description").empty() ? "" : "\n" + str_or(t, "description"));
    if (namespaces && !scope.empty()) namespaces->emplace(flat, scope);
    out->push_back({{"type", "function"}, {"function", f}});
    return true;
  };
  for (const auto& t : tools) {
    if (str_or(t, "type") != "namespace") {
      if (!function(t, "", "")) return false;
      continue;
    }
    const std::string scope = str_or(t, "name");
    if (scope.empty() || !t.contains("tools") || !t["tools"].is_array())
      return *err = "a tool namespace needs a name and a list of tools", false;
    const std::string description = str_or(t, "description");
    for (const auto& member : t["tools"])
      if (!function(member, scope, description)) return false;
  }
  return true;
}

bool rules_from_responses(const ojson& req, const std::vector<ojson>& tools, ReplyRules* out, std::string* err,
                          const ResponsesNamespaces& namespaces) {
  ojson chat = ojson::object();  // the same rules, named as in chat completions
  for (const char* key : {"tool_choice", "parallel_tool_calls"})
    if (req.contains(key)) chat[key] = req[key];
  if (req.contains("tool_choice") && str_or(req["tool_choice"], "type") == "function") {
    const auto& tc = req["tool_choice"];
    const std::string scope = str_or(tc, "namespace");
    const std::string flat = responses_function_name(scope, str_or(tc, "name"));
    const auto it = namespaces.find(flat);
    if ((it == namespaces.end() && !scope.empty()) || (it != namespaces.end() && it->second != scope))
      return *err = "tool_choice namespace does not match the declared function '" + flat + "'", false;
    if (!scope.empty()) chat["tool_choice"]["name"] = flat;
  }
  if (const auto text = req.find("text"); text != req.end() && !text->is_null()) {
    if (!text->is_object()) return *err = "text must be an object", false;
    if (const auto fmt = text->find("format"); fmt != text->end() && !fmt->is_null()) {
      if (!fmt->is_object()) return *err = "text.format must be an object", false;
      chat["response_format"] = *fmt;
    }
  }
  return rules_from_openai(chat, tools, out, err);
}

bool rules_from_anthropic(const ojson& req, const std::vector<ojson>& tools, ReplyRules* out, std::string* err) {
  *out = {};
  const auto config = req.find("output_config");
  if (config != req.end() && !config->is_null()) {
    if (!config->is_object()) return *err = "output_config must be an object", false;
    const auto format = config->find("format");
    if (format != config->end() && !format->is_null()) {
      if (!format->is_object() || str_or(*format, "type") != "json_schema" ||
          !format->contains("schema") || !(*format)["schema"].is_object())
        return *err = "output_config.format must have type json_schema and a schema object", false;
      out->json = true;
      out->schema = (*format)["schema"];
    }
  }
  const auto tc_it = req.find("tool_choice");
  if (tc_it == req.end() || tc_it->is_null()) return true;
  if (!tc_it->is_object()) return *err = "tool_choice must be an object", false;
  const ojson& tc = *tc_it;
  std::string type = "auto";
  if (tc.contains("type")) {
    if (!tc["type"].is_string()) return *err = "tool_choice type must be auto, any, tool or none", false;
    type = tc["type"].get<std::string>();
  }
  if (const auto dp = tc.find("disable_parallel_tool_use"); dp != tc.end() && !dp->is_null()) {
    if (!dp->is_boolean()) return *err = "tool_choice.disable_parallel_tool_use must be a boolean", false;
    out->parallel = !dp->get<bool>();
  }
  if (type == "auto") out->tools = ReplyRules::kAuto;
  else if (type == "none") out->tools = ReplyRules::kNone;
  else if (type == "any") out->tools = ReplyRules::kRequired;
  else if (type == "tool") {
    out->tools = ReplyRules::kForced;
    out->forced = str_or(tc, "name");
    if (!has_tool(tools, out->forced))
      return *err = "tool_choice names the tool '" + out->forced + "', which is not among the tools", false;
  } else {
    return *err = "tool_choice type must be auto, any, tool or none", false;
  }
  if ((out->tools == ReplyRules::kRequired || out->tools == ReplyRules::kForced) && tools.empty())
    return *err = "tool_choice requires a tool call, but no tools are given", false;
  return true;
}

bool is_billing_line(std::string_view line) {
  static constexpr std::string_view kPrefix = "x-anthropic-billing-header:";
  return line.starts_with(kPrefix);
}

namespace {

// The text without a leading billing line (a block of its own, or the first line of one).
std::string without_billing_line(const std::string& text, bool* dropped) {
  const size_t eol = text.find('\n');
  if (!is_billing_line(std::string_view(text).substr(0, eol))) return text;
  *dropped = true;
  return eol == std::string::npos ? std::string() : text.substr(eol + 1);
}

struct AnthropicCache {
  std::vector<ChatCachePoint>* points = nullptr;
  static constexpr size_t kLimit = 4;

  bool marked(const ojson& block) const {
    return points && points->size() < kLimit && block.contains("cache_control") &&
           block["cache_control"].is_object() && str_or(block["cache_control"], "type") == "ephemeral";
  }
  void mark_end(const ojson& block, size_t count, const ChatMessage& m) const {
    if (marked(block))
      points->push_back({count, m.content.size(), m.reasoning ? m.reasoning->size() : 0, m.tool_calls.size()});
  }
};

bool parse_anthropic_system(const ojson& req, bool keep_billing_header, bool* billing, const AnthropicCache& cache,
                            std::vector<ChatMessage>* out, std::string* err) {
  const auto sys_it = req.find("system");
  if (sys_it == req.end() || sys_it->is_null()) return true;
  const ojson& sys = *sys_it;
  std::string text;
  std::vector<size_t> marks;
  if (sys.is_string()) {
    text = keep_billing_header ? sys.get<std::string>() : without_billing_line(sys.get<std::string>(), billing);
  } else if (sys.is_array()) {
    for (const auto& b : sys) {
      if (!b.is_object()) return *err = "every system block must be an object", false;
      std::string part = str_or(b, "text");
      if (!keep_billing_header && text.empty() && !*billing) {
        part = without_billing_line(part, billing);
        if (part.empty()) continue;  // the line was the whole block
      }
      if (!text.empty()) text += "\n";
      text += part;
      if (cache.marked(b) && marks.size() < AnthropicCache::kLimit) marks.push_back(text.size());
    }
  } else {
    return *err = "system must be a string or a list of blocks", false;
  }
  if (!text.empty()) {
    out->push_back({"system", std::move(text), std::nullopt, {}, ""});
    for (size_t bytes : marks)
      if (cache.points && cache.points->size() < AnthropicCache::kLimit) cache.points->push_back({1, bytes});
  }
  return true;
}

bool parse_anthropic_assistant(const ojson& m, const ojson& content, const AnthropicCache& cache,
                               std::vector<ChatMessage>* out, std::string* err) {
  ChatMessage cm{"assistant", "", std::nullopt, {}, ""};
  for (const auto& b : content) {
    const std::string type = str_or(b, "type");
    if (type == "text") {
      cm.content += str_or(b, "text");
    } else if (type == "thinking") {
      cm.reasoning = cm.reasoning.value_or("") + str_or(b, "thinking");
    } else if (type == "redacted_thinking") {
      // nothing the template can show
    } else if (type == "tool_use") {
      ToolCall call;
      call.id = str_or(b, "id");
      call.name = str_or(b, "name");
      if (b.contains("input") && !parse_arguments(b["input"], &call.arguments, err)) return false;
      cm.tool_calls.push_back(std::move(call));
    } else {
      return *err = "assistant content block of type '" + type + "' is not supported", false;
    }
    cache.mark_end(b, out->size() + 1, cm);
  }
  out->push_back(std::move(cm));
  cache.mark_end(m, out->size(), out->back());
  return true;
}

bool parse_anthropic_user(const ojson& m, const ojson& content, const AnthropicCache& cache,
                          std::vector<ChatMessage>* out, std::string* err) {
  // user: tool results become tool messages, text and images after them a user message
  const size_t before = out->size();
  std::string text;
  std::vector<ContentPart> user_parts;
  bool has_user_image = false;
  std::vector<std::pair<size_t, size_t>> marked_text;
  for (const auto& b : content) {
    const std::string type = str_or(b, "type");
    if (type == "tool_result") {
      std::string result;
      std::vector<ContentPart> tparts;
      if (b.contains("content") && !parse_content_parts(b["content"], &result, &tparts, true, err)) return false;
      if (b.contains("is_error") && b["is_error"].is_boolean() && b["is_error"].get<bool>()) {
        result = "Error: " + result;
        if (!tparts.empty()) tparts.insert(tparts.begin(), ContentPart::make_text("Error: "));
      }
      ChatMessage tm{"tool", std::move(result), std::nullopt, {}, str_or(b, "tool_use_id")};
      tm.parts = std::move(tparts);
      out->push_back(std::move(tm));
      cache.mark_end(b, out->size(), out->back());
    } else if (type == "text") {
      const std::string s = str_or(b, "text");
      text += s;
      user_parts.push_back(ContentPart::make_text(s));
      if (cache.marked(b) && marked_text.size() < AnthropicCache::kLimit)
        marked_text.emplace_back(out->size(), text.size());
    } else if (type == "image") {
      std::shared_ptr<const ImageInput> img;
      if (!parse_image_part(b, type, &img, err)) return false;
      has_user_image = true;
      user_parts.push_back(ContentPart::make_image(std::move(img)));
    } else {
      return *err = "user content block of type '" + type + "' is not supported (text, image and tool results only)",
             false;
    }
  }
  if (!text.empty() || has_user_image || out->size() == before) {
    ChatMessage um{"user", std::move(text), std::nullopt, {}, ""};
    if (has_user_image) um.parts = std::move(user_parts);
    out->push_back(std::move(um));
    for (const auto& [prior, bytes] : marked_text) {
      if (cache.points->size() == AnthropicCache::kLimit) break;
      if (prior + 1 == out->size()) {
        cache.points->push_back({out->size(), bytes});
      } else if (prior > 0) {
        // A later tool result renders before this text. Only the messages ahead of that tool belong
        // to the stable prefix; the full rendered input remains exactly as for an unmarked request.
        const auto& last = (*out)[prior - 1];
        cache.points->push_back({prior, last.content.size(), last.reasoning ? last.reasoning->size() : 0,
                                 last.tool_calls.size()});
      }
    }
  }
  if (!out->empty()) cache.mark_end(m, out->size(), out->back());
  return true;
}

bool parse_anthropic_tools(const ojson& req, const AnthropicCache& cache, std::vector<ojson>* tools, std::string* err) {
  const auto it = req.find("tools");
  if (it == req.end() || it->is_null()) return true;
  if (!it->is_array()) return *err = "tools must be a list", false;
  for (const auto& t : *it) {
    if (!t.is_object()) return *err = "every tool must be an object", false;
    if (t.contains("type") && t["type"].is_string() && t["type"] != "custom" && !t.contains("input_schema"))
      continue;  // server tools (web search, ...) have no schema the model can call
    ojson f = ojson::object();
    f["name"] = str_or(t, "name");
    if (t.contains("description")) f["description"] = t["description"];
    f["parameters"] = t.contains("input_schema") ? t["input_schema"] : ojson::object();
    tools->push_back(ojson{{"type", "function"}, {"function", f}});
    if (cache.marked(t)) cache.points->push_back({0, 0, 0, 0, tools->size()});
  }
  return true;
}

}  // namespace

bool chat_from_anthropic(const ojson& req, std::vector<ChatMessage>* out, std::vector<ojson>* tools,
                         std::string* err, bool keep_billing_header, bool* dropped, std::vector<ChatCachePoint>* cache_points) {
  out->clear();
  tools->clear();
  if (cache_points) cache_points->clear();
  const AnthropicCache cache{cache_points};
  bool billing = false;
  if (!parse_anthropic_system(req, keep_billing_header, &billing, cache, out, err)) return false;
  if (dropped) *dropped = billing;
  if (!req.contains("messages") || !req["messages"].is_array() || req["messages"].empty())
    return *err = "messages must be a non-empty list", false;
  for (const auto& m : req["messages"]) {
    const std::string role = str_or(m, "role");
    if (role != "user" && role != "assistant") return *err = "message role must be 'user' or 'assistant'", false;
    const auto field = m.find("content");
    if (field == m.end() || (!field->is_string() && !field->is_array()))
      return *err = "message content must be a string or a list of blocks", false;
    const ojson& content = *field;
    if (content.is_string()) {
      out->push_back({role, content.get<std::string>(), std::nullopt, {}, ""});
      cache.mark_end(m, out->size(), out->back());
    } else if (role == "assistant") {
      if (!parse_anthropic_assistant(m, content, cache, out, err)) return false;
    } else {
      if (!parse_anthropic_user(m, content, cache, out, err)) return false;
    }
  }
  if (!parse_anthropic_tools(req, cache, tools, err)) return false;
  if (!out->empty()) cache.mark_end(req, out->size(), out->back());
  return true;
}

}  // namespace flyx
