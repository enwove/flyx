// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Measure RequestLog::start on parsed agent requests. Build with the server objects as
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "apps/flyx-server/request_log.h"

using flyx::ojson;
using flyx::server::RequestLog;

namespace {
std::string repeated(size_t bytes, const std::string& line) {
  std::string text;
  text.reserve(bytes);
  while (text.size() + line.size() <= bytes) text += line;
  text.append(bytes - text.size(), 'x');
  return text;
}

ojson history(int turns, size_t result_bytes, const std::string& line) {
  ojson messages = ojson::array({{{"role", "user"}, {"content", "Inspect the project files."}}});
  const auto result = repeated(result_bytes, line);
  for (int i = 0; i < turns; ++i) {
    const std::string id = "toolu_" + std::to_string(i);
    ojson call = {{"type", "tool_use"}, {"id", id}, {"name", "read_file"},
                  {"input", {{"path", "src/a.cc"}, {"offset", i}}}};
    ojson response = {{"type", "tool_result"}, {"tool_use_id", id}, {"content", result}};
    messages.push_back({{"role", "assistant"}, {"content", ojson::array({std::move(call)})}});
    messages.push_back({{"role", "user"}, {"content", ojson::array({std::move(response)})}});
  }
  ojson schema = {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}};
  ojson tool = {{"name", "read_file"}, {"input_schema", std::move(schema)}};
  return {{"model", "fixture"}, {"max_tokens", 1}, {"tools", ojson::array({std::move(tool)})},
          {"messages", std::move(messages)}, {"stream", true}};
}

auto record(RequestLog& log, const ojson& body, const std::string& id) {
  return log.start("messages", id, "fixture", true, body, {});
}

void measure(const char* name, const ojson& body, int runs) {
  std::vector<double> elapsed;
  size_t kept = 0;
  for (int i = 0; i <= runs; ++i) {
    RequestLog log(true);
    const auto start = std::chrono::steady_clock::now();
    auto entry = record(log, body, "probe");
    const auto end = std::chrono::steady_clock::now();
    entry->finish();
    if (i == 0) kept = log.newest(1).at(0).at("request").dump().size();
    else elapsed.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(elapsed.begin(), elapsed.end());
  const size_t p95 = (95 * elapsed.size() + 99) / 100 - 1;
  std::printf("%-16s %10zu %10zu %10.4f %10.4f\n", name, body.dump().size(), kept, elapsed[elapsed.size() / 2],
              elapsed[p95]);
}
}  // namespace

int main() {
  constexpr int runs = 31;
  const std::string plain = " tool result from build and tests. ", utf8 = "工具行 😀 é\n\"path\\source\":\t";
  std::printf("%-16s %10s %10s %10s %10s\n", "fixture", "wire_bytes", "kept_bytes", "p50_ms", "p95_ms");
  measure("short", history(2, 192, utf8), runs);
  measure("long_025", history(4, 64 * 1024 - 24, plain), runs);
  measure("utf8_1", history(12, 72 * 1024, utf8), runs);
  measure("long_3", history(24, 128 * 1024 - 300, plain), runs);
  measure("one_giant", history(1, 3 * 1024 * 1024 - 512, plain), runs);
  measure("escaped", history(7, 80 * 1024, "a\x01\x02\n\t\\\""), runs);
  auto many_short = history(2800, 704, plain);
  measure("many_short", many_short, runs);
  many_short["system"] = repeated(90 * 1024, utf8);
  measure("short_plus_long", many_short, runs);
  ojson schemas = history(0, 0, plain);
  schemas["system"] = repeated(3000, plain);
  for (int tool = 0; tool < 64; ++tool) {
    ojson properties = ojson::object();
    for (int arg = 0; arg < 160; ++arg)
      properties["argument_" + std::to_string(arg)] = {{"type", "integer"}, {"minimum", arg}, {"default", arg}};
    schemas["tools"].push_back({{"name", "tool_" + std::to_string(tool)},
                                {"input_schema", {{"type", "object"}, {"properties", std::move(properties)}}}});
  }
  measure("wide_schemas", schemas, runs);

  RequestLog on(true), off(false);
  const auto repeated_body = history(700, 704, plain);
  if (record(off, repeated_body, "off") || !off.newest(50).empty()) return 1;
  for (int i = 0; i <= 50; ++i) record(on, repeated_body, std::to_string(i))->finish();
  const auto entries = on.newest(50);
  if (entries.size() != 50 || entries.front().at("id") != "50" || entries.back().at("id") != "1" ||
      entries.front().at("request") != repeated_body) return 1;
  std::printf("50 entries of %zu request bytes each occupy %zu serialized bytes\n",
              repeated_body.dump().size(), entries.dump().size());
}
