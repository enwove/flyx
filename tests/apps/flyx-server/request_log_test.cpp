// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// RequestLog: single-pass escaped-byte request budgeting, clipping thresholds,
// pre-serialized immutable snapshot responses, and concurrent writer/reader safety.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "apps/flyx-server/http_util.h"
#include "apps/flyx-server/request_log.h"
#include "apps/flyx-server/server.h"
#include "tests/common/check.h"

namespace flyx::server {

std::string dump_json(const ojson& body) {
  return body.dump(-1, ' ', false, ojson::error_handler_t::replace);
}
std::string new_id(const char* prefix) { return std::string(prefix) + "test"; }
bool require_manage(const Server&, const httplib::Request&, httplib::Response&, const char*) { return true; }
int query_int(const httplib::Request&, const char*, int dflt, int, int) { return dflt; }

}  // namespace flyx::server

using flyx::ojson;
using flyx::server::RequestLog;
using flyx::server::dump_json;

namespace {

ojson record_one(const ojson& body) {
  RequestLog log(true);
  auto rec = log.start("chat", "id-1", "test-model", false, body, {});
  rec->finish();
  const auto snap = log.snapshot(10);
  const auto parsed = ojson::parse(snap.json_response(true));
  return parsed.at("requests").at(0).at("request");
}

}  // namespace

int main() {
  // 1. Disabled log and empty snapshots produce exact compact JSON.
  {
    RequestLog off(false);
    check(off.start("chat", "x", "m", false, ojson::object(), {}) == nullptr, "disabled log returns nullptr");
    check(off.snapshot(10).json_response(false) == "{\"enabled\":false,\"requests\":[]}",
          "disabled empty snapshot json");
    RequestLog on(true);
    check(on.snapshot(10).json_response(true) == "{\"enabled\":true,\"requests\":[]}", "enabled empty snapshot json");
  }

  // 2. Normal body under 256 KiB with escapes, control chars, multibyte UTF-8, invalid UTF-8, and scalars stays intact.
  {
    std::string all_bytes;
    for (int b = 0; b < 256; ++b) all_bytes.push_back(static_cast<char>(b));
    ojson body = {
        {"z_first", 1},
        {"a_second", all_bytes},
        {"utf8", "中文🙂\n\t\"\\\x01\x1f\x7f"},
        {"nums", ojson::array({0, -1, std::numeric_limits<int64_t>::min(), std::numeric_limits<uint64_t>::max(), 3.14159})},
        {"flags", ojson::array({true, false, nullptr})},
        {"key\"with\\esc\n", "val"},
    };
    const ojson got = record_one(body);
    const ojson want = ojson::parse(dump_json(body));
    check(dump_json(got) == dump_json(want), "normal body with all byte values, escapes, scalars keeps exact order");
  }

  // 3. Escaped length crossover: 10 strings fit unescaped at 16 KiB, exceed 256 KiB escaped -> clipped to 4 KiB.
  {
    std::string text;
    text.reserve(20500);
    for (int i = 0; i < 5100; ++i) text += "\"\\\n\t";
    text += "\xE7\xB5\x82";  // 終
    ojson probe = ojson::array();
    for (int i = 0; i < 10; ++i) probe.push_back({{"z", i}, {"a", text}, {"empty", ojson::object()}, {"flag", false}});
    ojson body = {{"model", "m"}, {"log_probe", probe}};
    const ojson got = record_one(body);
    const std::string want_a = RequestLog::clip(text, 4096);
    ojson want = body;
    for (auto& item : want["log_probe"]) item["a"] = want_a;
    check(dump_json(got) == dump_json(want), "escaped length picks 4 KiB cap and preserves key order");
  }

  // 4. Many short values exceeding 256 KiB stay intact up to 4 MiB.
  {
    ojson probe = ojson::array();
    for (int i = 0; i < 320; ++i) probe.push_back({{"name", "call"}, {"arguments", std::string(900, 'x')}});
    ojson body = {{"model", "m"}, {"log_probe", probe}};
    const ojson got = record_one(body);
    check(dump_json(got) == dump_json(body), "many short values exceeding 256 KiB stay intact");
  }

  // 5. Deeply nested JSON (> 64 levels) triggers depth truncation marker without stack overflow.
  {
    ojson nested = "leaf";
    for (int d = 0; d < 70; ++d) nested = ojson{{"n", std::move(nested)}};
    const ojson got = record_one(nested);
    const std::string wire = dump_json(got);
    check(wire.find("[truncated depth]") != std::string::npos && wire.find("leaf") == std::string::npos,
          "deep JSON truncates past max depth");
  }

  // 6. Oversized body (> 4 MiB even at 1 KiB cap) falls back to compact summary with exact serialized byte count.
  {
    ojson arr = ojson::array();
    const std::string item(1000, 'a');
    for (int i = 0; i < 4500; ++i) arr.push_back(item);
    ojson body = {{"model", "big-model"}, {"stream", true}, {"items", std::move(arr)}};
    const ojson got = record_one(body);
    const size_t exact_bytes = dump_json(body).size();
    check(got.value("_truncated", false) && got.value("bytes", size_t(0)) == exact_bytes &&
              got.value("model", std::string()) == "big-model" && got.value("stream", false),
          "oversized body emits compact summary with exact byte count");
  }

  // 7. Ring eviction (kKept = 50), newest-first order, limit slicing, and concurrent writer/reader stress (TSan).
  {
    RequestLog log(true);
    for (int i = 0; i < 65; ++i) {
      auto rec = log.start("chat", "req-" + std::to_string(i), "m", false, {{"i", i}}, {});
      rec->finish();
    }
    const auto all = log.snapshot(RequestLog::kKept);
    const auto parsed = ojson::parse(all.json_response(true));
    check(parsed["requests"].size() == RequestLog::kKept && parsed["requests"][0]["id"] == "req-64" &&
              parsed["requests"][RequestLog::kKept - 1]["id"] == "req-15" && all.revision == 65,
          "ring keeps newest 50 entries in newest-first order");
    const auto top3 = log.snapshot(3);
    const auto parsed3 = ojson::parse(top3.json_response(true));
    check(parsed3["requests"].size() == 3 && parsed3["requests"][2]["id"] == "req-62", "snapshot limit slices newest");

    std::atomic<bool> done{false};
    std::atomic<int> bad{0};
    std::vector<std::thread> writers, readers;
    for (int w = 0; w < 4; ++w) {
      writers.emplace_back([&, w] {
        ojson payload = {{"worker", w}, {"text", std::string(4096, 'k')}};
        for (int i = 0; i < 40; ++i) {
          auto rec = log.start("messages", "w" + std::to_string(w) + "-" + std::to_string(i), "m", true, payload, {});
          rec->finish();
        }
      });
    }
    for (int r = 0; r < 2; ++r) {
      readers.emplace_back([&] {
        while (!done.load(std::memory_order_relaxed)) {
          const auto snap = log.snapshot(10);
          const std::string wire = snap.json_response(true);
          const auto j = ojson::parse(wire, nullptr, false);
          if (j.is_discarded() || !j.contains("requests") || !j["requests"].is_array())
            bad.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }
    for (auto& w : writers) w.join();
    done.store(true, std::memory_order_relaxed);
    for (auto& r : readers) r.join();
    check(bad.load() == 0, "concurrent RequestLog writers and readers produce valid JSON");
  }

  return flyx::test::finish("request_log_test");
}
