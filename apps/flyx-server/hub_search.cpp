// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/hub_search.h"

#include <future>
#include <optional>
#include <set>
#include <utility>

#include "nlohmann/json.hpp"

namespace flyx::server {

namespace {

// Where each stream goes on from: the Hub's cursor ("" for its first page); none at its end.  `excluded`: a
// repository a first page showed beside the streams' rows, left out of every page after it.
struct Places {
  std::optional<std::string> gguf, drafters;
  std::string excluded;
};

// The cursor's places (*ours: it is one of this search's, not the Hub's own); false, with *err, for a cursor of this
// search's version 1 that is not one.
bool places_of(const std::string& cursor, Places* p, std::string* err, bool* ours = nullptr) {
  *p = {};
  if (ours) *ours = false;
  if (cursor.empty()) {
    p->gguf = std::string(), p->drafters = std::string();
    return true;
  }
  const nlohmann::json j = nlohmann::json::parse(cursor, nullptr, false);
  if (!j.is_object() || !j.contains("v")) {  // the Hub's own cursor: the GGUF stream's alone
    p->gguf = cursor;
    return true;
  }
  auto text = [&](const char* key, std::optional<std::string>* to) {
    const auto it = j.find(key);
    if (it == j.end()) return true;
    if (!it->is_string() || it->get<std::string>().empty()) return false;
    *to = it->get<std::string>();
    return true;
  };
  std::optional<std::string> excluded;
  bool ok = j["v"] == 1 && text("g", &p->gguf) && text("d", &p->drafters) && text("x", &excluded) &&
            (p->gguf || p->drafters);
  for (auto it = j.begin(); ok && it != j.end(); ++it) ok = it.key() == "v" || it.key() == "g" || it.key() == "d" ||
                                                             it.key() == "x";
  if (!ok) return *err = "cursor: not one a page of this search gave", false;
  p->excluded = excluded.value_or("");
  if (ours) *ours = true;
  return true;
}

std::string cursor_of(const Places& p) {
  if (!p.gguf && !p.drafters) return "";
  nlohmann::json j = {{"v", 1}};
  if (p.gguf) j["g"] = *p.gguf;
  if (p.drafters) j["d"] = *p.drafters;
  if (!p.excluded.empty()) j["x"] = p.excluded;
  return j.dump();
}

}  // namespace

std::string repo_key(std::string_view id) {
  std::string k(id);
  for (char& c : k)
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  return k;
}

bool search_page(const HubSearchFn& search, const std::string& query, const std::string& sort, int limit,
                 const std::string& cursor, std::vector<hub::Found>* out, std::string* next_cursor, std::string* err,
                 hub::HubOutcome* outcome) {
  Places at;
  const bool parsed = places_of(cursor, &at, err);  // before *next_cursor is cleared: it may be the same string
  out->clear();
  next_cursor->clear();
  if (!parsed) {
    if (outcome) *outcome = {hub::HubError::kInvalidName, ""};
    return false;
  }
  // The drafter stream asked beside the GGUF one: a page waits for the slower of the two, not for both in turn.
  std::vector<hub::Found> extra;
  std::string extra_next, extra_err;
  hub::HubOutcome extra_outcome;
  std::future<bool> drafters;
  if (at.drafters)
    drafters = std::async(std::launch::async, [&] {
      return search(query, sort, limit, &extra, &extra_err, *at.drafters, &extra_next, /*gguf_only=*/false,
                    &extra_outcome);
    });
  std::vector<hub::Found> gguf;
  std::string gguf_next;
  const bool gguf_ok =
      !at.gguf || search(query, sort, limit, &gguf, err, *at.gguf, &gguf_next, /*gguf_only=*/true, outcome);
  const bool drafters_ok = drafters.valid() && drafters.get();
  if (!gguf_ok) return false;
  if (!at.gguf && !drafters_ok) {  // the drafter stream the only one left: its error, not an empty last page
    *err = extra_err;
    if (outcome) *outcome = extra_outcome;
    return false;
  }
  std::set<std::string> seen;  // repo_key's: one repository however the Hub or a cursor spells it
  if (!at.excluded.empty()) seen.insert(repo_key(at.excluded));
  for (hub::Found& f : gguf)
    if (seen.insert(repo_key(f.id)).second) out->push_back(std::move(f));
  if (drafters_ok)
    for (hub::Found& f : extra)
      if (f.is_drafter && f.architecture.empty() && seen.insert(repo_key(f.id)).second) out->push_back(std::move(f));
  Places next;
  if (at.gguf && !gguf_next.empty()) next.gguf = gguf_next;
  if (drafters_ok && !extra_next.empty()) next.drafters = extra_next;
  next.excluded = at.excluded;
  *next_cursor = cursor_of(next);
  return true;
}

std::string cursor_excluding(const std::string& next_cursor, const std::string& id) {
  Places p;
  std::string err;
  bool ours = false;
  if (next_cursor.empty() || id.empty() || !places_of(next_cursor, &p, &err, &ours) || !ours) return next_cursor;
  p.excluded = id;
  return cursor_of(p);
}

}  // namespace flyx::server
