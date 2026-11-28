// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/hub_search.h"

#include <future>
#include <optional>
#include <set>
#include <utility>

#include "nlohmann/json.hpp"

namespace flyx::server {

namespace {

// Where each stream goes on from: the Hub's cursor ("" for its first page); none at its end.
struct Places {
  std::optional<std::string> gguf, drafters;
};

Places places_of(const std::string& cursor) {
  if (cursor.empty()) return {std::string(), std::string()};
  const nlohmann::json j = nlohmann::json::parse(cursor, nullptr, false);
  if (!j.is_object()) return {cursor, std::nullopt};  // the Hub's own cursor: the GGUF stream's alone
  Places p;
  if (const auto g = j.find("gguf"); g != j.end() && g->is_string()) p.gguf = g->get<std::string>();
  if (const auto d = j.find("drafters"); d != j.end() && d->is_string()) p.drafters = d->get<std::string>();
  return p;
}

std::string cursor_of(const Places& p) {
  if (!p.gguf && !p.drafters) return "";
  nlohmann::json j = nlohmann::json::object();
  if (p.gguf) j["gguf"] = *p.gguf;
  if (p.drafters) j["drafters"] = *p.drafters;
  return j.dump();
}

}  // namespace

bool search_page(const HubSearchFn& search, const std::string& query, const std::string& sort, int limit,
                 const std::string& cursor, std::vector<hub::Found>* out, std::string* next_cursor, std::string* err,
                 hub::HubOutcome* outcome) {
  const Places at = places_of(cursor);  // before *next_cursor is cleared: it may be the same string
  out->clear();
  next_cursor->clear();
  // The drafter stream asked beside the GGUF one: a page waits for the slower of the two, not for both in turn.
  std::vector<hub::Found> extra;
  std::string extra_next;
  std::future<bool> drafters;
  if (at.drafters)
    drafters = std::async(std::launch::async, [&] {
      std::string e;
      return search(query, sort, limit, &extra, &e, *at.drafters, &extra_next, /*gguf_only=*/false, nullptr);
    });
  std::vector<hub::Found> gguf;
  std::string gguf_next;
  const bool gguf_ok =
      !at.gguf || search(query, sort, limit, &gguf, err, *at.gguf, &gguf_next, /*gguf_only=*/true, outcome);
  const bool drafters_ok = drafters.valid() && drafters.get();
  if (!gguf_ok) return false;
  std::set<std::string> seen;
  for (hub::Found& f : gguf)
    if (seen.insert(f.id).second) out->push_back(std::move(f));
  if (drafters_ok)
    for (hub::Found& f : extra)
      if (f.is_drafter && f.architecture.empty() && seen.insert(f.id).second) out->push_back(std::move(f));
  Places next;
  if (at.gguf && !gguf_next.empty()) next.gguf = gguf_next;
  if (drafters_ok && !extra_next.empty()) next.drafters = extra_next;
  *next_cursor = cursor_of(next);
  return true;
}

}  // namespace flyx::server
