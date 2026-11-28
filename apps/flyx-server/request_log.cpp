// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The request log and GET /v1/requests?limit=N (default and at most 50, newest first): gated by Access::kManage,
// as the model panel: it shows other people's prompts.
#include "apps/flyx-server/request_log.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string_view>

#include "apps/flyx-server/server.h"
#include "core/os/os.h"

namespace flyx::server {

namespace {

constexpr size_t kSmallestClip = 1024;
constexpr size_t kHardRequestBytes = 4 * 1024 * 1024;  // safety ceiling after clipping every string to kSmallestClip
constexpr size_t kMaxLoggedCalls = 64;
constexpr size_t kMaxClipDepth = 64;
constexpr std::array<size_t, 4> kClipCaps = {
    RequestLog::kStringBytes,
    RequestLog::kStringBytes / 4,
    RequestLog::kStringBytes / 16,
    kSmallestClip,
};
static_assert(kClipCaps[0] == 65536 && kClipCaps[1] == 16384 && kClipCaps[2] == 4096 && kClipCaps[3] == 1024);

constexpr std::string_view kTruncPrefix = "\xE2\x80\xA6 [truncated ";
constexpr std::string_view kTruncSuffix = " bytes]";
constexpr std::string_view kDepthMarker = "[truncated depth]";

// If-None-Match uses weak comparison. An opaque tag may itself contain a comma.
bool matches_etag(std::string_view values, std::string_view tag) {
  size_t pos = 0;
  while ((pos = values.find_first_not_of(" ,\t", pos)) != values.npos) {
    if (values.substr(pos) == "*") return true;
    if (values.substr(pos, 2) == "W/") pos += 2;
    if (pos == values.size() || values[pos] != '"') return false;
    const size_t end = values.find('"', pos + 1);
    if (end == values.npos) return false;
    if (values.substr(pos, end - pos + 1) == tag) return true;
    pos = end + 1;
  }
  return false;
}

size_t prefix_end(const std::string& s, size_t max) {
  size_t end = std::min(max, s.size());
  while (end > 0 && end < s.size() && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) --end;
  return end;
}

size_t decimal_digits(size_t n) {
  size_t d = 1;
  while (n >= 10) {
    n /= 10;
    ++d;
  }
  return d;
}

// Validate one UTF-8 character starting at s[i] within [0, limit), adding its compact JSON escaped length
// (matching nlohmann::json::dump(-1, ' ', false, replace)) to *acc and advancing *i.  Returns false on invalid UTF-8.
bool step_json_utf8(const std::string& s, size_t limit, size_t* i, size_t* acc) {
  const unsigned char c = static_cast<unsigned char>(s[*i]);
  if (c < 0x80) {
    if (c == '"' || c == '\\' || c == '\b' || c == '\f' || c == '\n' || c == '\r' || c == '\t') *acc += 2;
    else if (c < 0x20) *acc += 6;
    else *acc += 1;
    ++(*i);
    return true;
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
  if (*i + len > limit) return false;
  for (size_t j = 1; j < len; ++j) {
    const unsigned char b = static_cast<unsigned char>(s[*i + j]);
    if ((b & 0xC0) != 0x80) return false;
    cp = (cp << 6) | (b & 0x3Fu);
  }
  if ((len == 2 && cp < 0x80) || (len == 3 && (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF))) ||
      (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)))
    return false;
  *i += len;
  *acc += len;
  return true;
}

// Exact compact JSON length (including quotes) of an unclipped object key.
size_t escaped_key_bytes(const std::string& key) {
  size_t i = 0, acc = 0;
  while (i < key.size()) {
    if (!step_json_utf8(key, key.size(), &i, &acc)) return dump_json(ojson(key)).size();
  }
  return 2 + acc;
}

// Exact compact JSON lengths (including quotes and any truncation note) of `s` at all four clip caps in one pass.
void escaped_clipped_bytes(const std::string& s, std::array<size_t, 4>& out) {
  std::array<size_t, 4> ends{};
  for (size_t k = 0; k < 4; ++k) ends[k] = prefix_end(s, kClipCaps[k]);
  std::array<size_t, 4> esc{};
  int next_k = 3;
  while (next_k >= 0 && ends[size_t(next_k)] == 0) esc[size_t(next_k--)] = 0;
  size_t i = 0, acc = 0;
  bool ok = true;
  while (i < ends[0]) {
    if (!step_json_utf8(s, ends[0], &i, &acc)) {
      ok = false;
      break;
    }
    if (next_k >= 0 && i > ends[size_t(next_k)]) {
      ok = false;
      break;
    }
    while (next_k >= 0 && i == ends[size_t(next_k)]) esc[size_t(next_k--)] = acc;
  }
  if (!ok || next_k >= 0) {
    for (size_t k = 0; k < 4; ++k) out[k] = dump_json(ojson(RequestLog::clip(s, kClipCaps[k]))).size();
    return;
  }
  constexpr size_t kSuffixFixed = kTruncPrefix.size() + kTruncSuffix.size();
  for (size_t k = 0; k < 4; ++k) {
    out[k] = 2 + esc[k] + (s.size() > kClipCaps[k] ? kSuffixFixed + decimal_digits(s.size() - ends[k]) : 0);
  }
}

size_t scalar_json_bytes(const ojson& v) {
  if (v.is_null()) return 4;
  if (v.is_boolean()) return v.get<bool>() ? 4 : 5;
  char buf[32];
  if (v.is_number_unsigned()) {
    const auto r = std::to_chars(buf, buf + sizeof(buf), v.get<uint64_t>());
    return static_cast<size_t>(r.ptr - buf);
  }
  if (v.is_number_integer()) {
    const auto r = std::to_chars(buf, buf + sizeof(buf), v.get<int64_t>());
    return static_cast<size_t>(r.ptr - buf);
  }
  return dump_json(v).size();
}

bool iequals_prefix(std::string_view s, std::string_view pfx) {
  if (s.size() < pfx.size()) return false;
  for (size_t i = 0; i < pfx.size(); ++i)
    if (std::tolower(static_cast<unsigned char>(s[i])) != std::tolower(static_cast<unsigned char>(pfx[i])))
      return false;
  return true;
}

// Replace large inline base64 image payloads (data:image/...;base64,... and Anthropic source.data)
// with a compact placeholder so request logs do not waste 64 KiB on base64 noise or crush text clipping.
std::optional<std::string> redact_image_string(const std::string& s, const ojson* parent_obj, std::string_view key) {
  if (s.size() > 64 && iequals_prefix(s, "data:image/")) {
    const size_t comma = s.find(',');
    if (comma != std::string::npos && comma <= 128) {
      const std::string_view meta(s.data() + 5, comma - 5);
      const size_t last_semi = meta.rfind(';');
      if (last_semi != std::string_view::npos && meta.size() - (last_semi + 1) == 6 &&
          iequals_prefix(meta.substr(last_semi + 1), "base64")) {
        const size_t first_semi = meta.find(';');
        const std::string_view mime = meta.substr(0, first_semi);
        return "<base64 " + std::string(mime) + ", " + std::to_string(s.size() - comma - 1) + " chars>";
      }
    }
  }
  if (s.size() > 64 && parent_obj && key == "data" && parent_obj->contains("type") &&
      (*parent_obj)["type"].is_string() && (*parent_obj)["type"].get_ref<const std::string&>() == "base64") {
    std::string media = "image";
    if (parent_obj->contains("media_type") && (*parent_obj)["media_type"].is_string()) {
      const auto& mt = (*parent_obj)["media_type"].get_ref<const std::string&>();
      if (!mt.empty() && mt.size() <= 64) media = mt;
    }
    return "<base64 " + media + ", " + std::to_string(s.size()) + " chars>";
  }
  return std::nullopt;
}

// Single read-only iterative walk: computes exact compact serialized JSON byte length at every clip cap
// (64 KiB, 16 KiB, 4 KiB, 1 KiB), the longest string length, and whether nesting exceeds kMaxClipDepth.
struct BodyMeasure {
  std::array<size_t, 4> bytes{};
  size_t longest = 0;
  bool deep = false;
  bool redacted = false;
};

BodyMeasure measure_body(const ojson& root) {
  BodyMeasure m;
  auto add_all = [&](size_t n) {
    for (size_t k = 0; k < 4; ++k) m.bytes[k] += n;
  };
  struct Item {
    const ojson* node;
    const ojson* parent_obj;
    std::string_view key;
    size_t depth;
  };
  std::vector<Item> stack;
  stack.reserve(64);
  stack.push_back({&root, nullptr, {}, 0});
  while (!stack.empty()) {
    const auto [value, parent_obj, key, depth] = stack.back();
    stack.pop_back();
    if (depth > kMaxClipDepth) {
      m.deep = true;
      add_all(2 + kDepthMarker.size());
      continue;
    }
    if (value->is_string()) {
      const auto& raw = value->get_ref<const std::string&>();
      const auto red = redact_image_string(raw, parent_obj, key);
      if (red) m.redacted = true;
      const std::string& s = red ? *red : raw;
      m.longest = std::max(m.longest, s.size());
      std::array<size_t, 4> sb{};
      escaped_clipped_bytes(s, sb);
      for (size_t k = 0; k < 4; ++k) m.bytes[k] += sb[k];
    } else if (value->is_structured()) {
      add_all(2 + (value->empty() ? 0 : value->size() - 1));  // brackets and commas
      const ojson* p = value->is_object() ? value : nullptr;
      for (auto it = value->begin(); it != value->end(); ++it) {
        if (p) add_all(escaped_key_bytes(it.key()) + 1);  // quoted key and colon
        stack.push_back({&(*it), p, p ? std::string_view(it.key()) : std::string_view{}, depth + 1});
      }
    } else {
      add_all(scalar_json_bytes(*value));
    }
  }
  return m;
}

// Copy the retained part of each value. Iterative to avoid call-stack overflow on deep JSON.
ojson clip_strings(const ojson& original, size_t max) {
  struct Frame {
    const ojson* src;
    const ojson* parent_obj;
    std::string_view key;
    ojson* dst;
    size_t depth;
  };
  ojson root;
  std::vector<Frame> stack;
  stack.push_back({&original, nullptr, {}, &root, 0});
  while (!stack.empty()) {
    const auto [src, parent_obj, key, dst, depth] = stack.back();
    stack.pop_back();
    if (depth > kMaxClipDepth) {
      *dst = std::string(kDepthMarker);
      continue;
    }
    if (src->is_string()) {
      const auto& raw = src->get_ref<const std::string&>();
      const auto red = redact_image_string(raw, parent_obj, key);
      *dst = RequestLog::clip(red ? *red : raw, max);
    } else if (src->is_object()) {
      *dst = ojson::object();
      auto& kept = dst->get_ref<ojson::object_t&>();
      kept.reserve(src->size());
      for (auto it = src->begin(); it != src->end(); ++it) kept.emplace_back(it.key(), ojson());
      auto dit = kept.begin();
      for (auto it = src->begin(); it != src->end(); ++it, ++dit)
        stack.push_back({&(*it), src, it.key(), &dit->second, depth + 1});
    } else if (src->is_array()) {
      *dst = ojson::array();
      auto& kept = dst->get_ref<ojson::array_t&>();
      kept.resize(src->size());
      for (size_t i = 0; i < src->size(); ++i) stack.push_back({&((*src)[i]), nullptr, {}, &kept[i], depth + 1});
    } else {
      *dst = *src;
    }
  }
  return root;
}

// The request body as kept: each string at most kStringBytes, clipped further (down to kSmallestClip) to stay
// within kRequestBytes where possible; a long conversation of many short strings still keeps its shape up to
// kHardRequestBytes, and anything larger falls back to a compact summary.
ojson kept_request(const ojson& body) {
  const BodyMeasure m = measure_body(body);
  if (!m.deep) {
    for (size_t k = 0; k < 3; ++k) {
      if (m.bytes[k] <= RequestLog::kRequestBytes) {
        return (!m.redacted && m.longest <= kClipCaps[k]) ? body : clip_strings(body, kClipCaps[k]);
      }
    }
  }
  if (m.bytes[3] <= kHardRequestBytes) {
    return (!m.deep && !m.redacted && m.longest <= kSmallestClip) ? body : clip_strings(body, kSmallestClip);
  }
  ojson summary = {{"_truncated", true}, {"bytes", m.bytes[3]}};
  if (body.is_object()) {
    if (body.contains("model") && body["model"].is_string())
      summary["model"] = RequestLog::clip(body["model"].get_ref<const std::string&>(), kSmallestClip);
    if (body.contains("stream") && body["stream"].is_boolean()) summary["stream"] = body["stream"];
  }
  return summary;
}

}  // namespace

std::string RequestLog::clip(const std::string& s, size_t max, size_t total) {
  total = std::max(total, s.size());
  if (total <= max) return s;
  const size_t end = prefix_end(s, max);
  return s.substr(0, end) + std::string(kTruncPrefix) + std::to_string(total - end) + std::string(kTruncSuffix);
}

std::shared_ptr<Recording> RequestLog::start(const char* api, const std::string& id, const std::string& model,
                                             bool stream, const ojson& body, Arrival arrival,
                                             const std::shared_ptr<Job>& job) {
  if (!enabled_) return nullptr;
  return std::make_shared<Recording>(this, ojson{{"id", id},
                                                 {"api", api},
                                                 {"model", model},
                                                 {"started_at", arrival.unix_seconds},
                                                 {"stream", stream},
                                                 {"request", kept_request(body)}}, arrival.monotonic_ms, job);
}

void RequestLog::add(ojson entry) {
  // Serialize once outside the lock so GET /v1/requests never deep-copies or re-serializes immutable rows.
  auto row = std::make_shared<const std::string>(dump_json(entry));
  std::lock_guard<std::mutex> lock(mu_);
  entries_.push_front(std::move(row));
  ++revision_;
  if (entries_.size() > kKept) entries_.pop_back();
}

RequestLog::Snapshot RequestLog::snapshot(size_t limit) const {
  std::lock_guard<std::mutex> lock(mu_);
  return {revision_, {entries_.begin(), entries_.begin() + std::min(entries_.size(), limit)}};
}

std::string RequestLog::Snapshot::json_response(bool enabled) const {
  size_t total = 32;
  for (const auto& row : entries) total += row->size() + 1;
  std::string out;
  out.reserve(total);
  out += enabled ? "{\"enabled\":true,\"requests\":[" : "{\"enabled\":false,\"requests\":[";
  for (size_t i = 0; i < entries.size(); ++i) {
    if (i) out.push_back(',');
    out.append(*entries[i]);
  }
  out += "]}";
  return out;
}

Recording::Recording(RequestLog* log, ojson entry, double started_ms, std::shared_ptr<Job> job)
    : log_(log), job_(std::move(job)), entry_(std::move(entry)), started_ms_(started_ms) {}

Recording::~Recording() { finish(); }

void Recording::see(const JobEvent& ev) {
  if (ev.superseded) entry_["cancel_reason"] = "superseded";
  if (ttft_ms_ < 0 && (ev.kind == JobEvent::kReasoning || ev.kind == JobEvent::kContent ||
                       ev.kind == JobEvent::kToolCallStart || ev.kind == JobEvent::kToolCall))
    ttft_ms_ = os::now_ms() - started_ms_;
  switch (ev.kind) {
    // Clipped as they grow, not only at the end: a runaway reply does not hold more than its kept part.
    case JobEvent::kReasoning:
      reasoning_bytes_ += ev.text.size();
      if (reasoning_.size() <= RequestLog::kStringBytes) reasoning_ += ev.text;
      break;
    case JobEvent::kContent:
      content_bytes_ += ev.text.size();
      if (content_.size() <= RequestLog::kStringBytes) content_ += ev.text;
      break;
    case JobEvent::kToolCallStart:
    case JobEvent::kToolCallDelta:
      break;
    case JobEvent::kToolCall:
      if (calls_.size() < kMaxLoggedCalls)
        calls_.push_back({{"name", RequestLog::clip(ev.call.name, kSmallestClip)},
                          {"arguments", clip_strings(ev.call.arguments, RequestLog::kStringBytes)}});
      break;
    case JobEvent::kError:
      status_ = ev.superseded ? "cancelled" : "error";
      if (!ev.superseded) error_ = RequestLog::clip(ev.text, RequestLog::kStringBytes);
      break;
    case JobEvent::kDone: {
      const flyx_usage& u = ev.result.usage;
      // As the scheduler names it for /v1/telemetry's recent[].status.
      status_ = u.finish_reason == FLYX_FINISH_CANCEL   ? "cancelled"
                : u.finish_reason == FLYX_FINISH_LENGTH ? "length"
                : !calls_.empty()                       ? "tool_calls"
                                                        : "stop";
      prompt_tokens_ = u.prompt_tokens;
      cached_tokens_ = u.reused_tokens;
      output_tokens_ = output_tokens(u, ev.result.stop_token);
      const double rate = apps::decode_rate(output_tokens_, u.decode_ms);
      entry_["timing"] = {{"queue_ms", ev.queue_ms}, {"prefill_ms", u.prefill_ms}, {"decode_ms", u.decode_ms},
                          {"decode_tok_s", rate > 0 ? ojson(rate) : ojson(nullptr)},
                          {"tokens_per_round", u.rounds ? ojson(ev.tokens_per_round) : ojson(nullptr)}};
      break;
    }
  }
}

void Recording::finish() {
  if (logged_) return;
  logged_ = true;
  // The HTTP admission wait can expire while the job is still queued. A cancelled stream can already be
  // visible to its client, so keep its status and replacement reason without publishing its prompt.
  if (job_ && !job_->has_started()) {
    if (status_ != "cancelled" || !entry_["stream"].get<bool>()) return;
    entry_["request"] = nullptr;
  }
  entry_["finished_at"] = os::unix_seconds();
  entry_["status"] = status_;
  entry_["error"] = error_.empty() ? ojson(nullptr) : ojson(error_);
  entry_["usage"] = {{"prompt_tokens", prompt_tokens_}, {"cached_tokens", cached_tokens_},
                     {"output_tokens", output_tokens_}};
  entry_["ttft_ms"] = ttft_ms_ < 0 ? ojson(nullptr) : ojson(ttft_ms_);
  entry_["reasoning"] = RequestLog::clip(reasoning_, RequestLog::kStringBytes, reasoning_bytes_);
  entry_["content"] = RequestLog::clip(content_, RequestLog::kStringBytes, content_bytes_);
  entry_["tool_calls"] = std::move(calls_);
  log_->add(std::move(entry_));
}

void add_request_log_routes(httplib::Server& http, Server* s) {
  http.Get("/v1/requests", [s, instance = new_id("log-")](const httplib::Request& q, httplib::Response& r) {
    if (!require_access(*s, q, r, Access::kManage, "the request log")) return;
    const size_t limit = size_t(query_int(q, "limit", int(RequestLog::kKept), 0, int(RequestLog::kKept)));
    const bool on = s->requests && s->requests->enabled();
    r.set_header("Cache-Control", "no-store");
    const auto snapshot = on ? s->requests->snapshot(limit) : RequestLog::Snapshot{};
    const auto revision = snapshot.entries.empty() ? 0 : snapshot.revision;
    const std::string tag = '"' + instance + "-" + std::to_string(revision) + "-" + std::to_string(snapshot.entries.size()) + '"';
    r.set_header("ETag", tag);
    for (size_t i = 0; i < q.get_header_value_count("If-None-Match"); ++i) {
      if (matches_etag(q.get_header_value("If-None-Match", "", i), tag)) {
        r.status = 304;
        return;
      }
    }
    r.status = 200;
    r.set_content(snapshot.json_response(on), "application/json");
  });
}

}  // namespace flyx::server
