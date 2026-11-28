// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// apps/flyx-server/hub_search: GET /v1/hub/search's pages for a query against a stand-in Hub of two streams (its
// repositories with GGUF files, and all of them with their configs): every page the next of each, the safetensors
// drafters among the second beside the first's repositories whatever words the query has, each repository once, the
// pages walked by their cursors to the end of both, a drafter stream the Hub does not answer leaving the GGUF one.
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "apps/flyx-server/hub_search.h"
#include "tests/common/check.h"

using flyx::hub::Found;
using flyx::server::HubSearchFn;

namespace {

Found repo(const std::string& id, const std::string& arch = "", bool drafter = false) {
  Found f;
  f.id = id;
  f.architecture = arch;
  f.is_drafter = drafter;
  return f;
}

// A stream's pages, by the cursor that asks for each ("" the first), and the cursor after it ("" at the end).
using Pages = std::map<std::string, std::pair<std::vector<Found>, std::string>>;

struct StandIn {
  Pages gguf, all;
  bool all_fails = false, gguf_fails = false;
  std::mutex mu;
  std::vector<std::string> asked;  // "gguf:CURSOR" / "all:CURSOR", with the limit and query checked
  int limit = 0;
  std::string query;

  HubSearchFn fn() {
    return [this](const std::string& q, const std::string&, int n, std::vector<Found>* out, std::string* err,
                  const std::string& cursor, std::string* next, bool gguf_only, flyx::hub::HubOutcome*) {
      {
        std::lock_guard<std::mutex> lock(mu);
        asked.push_back(std::string(gguf_only ? "gguf:" : "all:") + cursor);
        limit = n, query = q;
      }
      if (gguf_only ? gguf_fails : all_fails) return *err = "the Hub answered 429", false;
      const Pages& pages = gguf_only ? gguf : all;
      const auto it = pages.find(cursor);
      if (it == pages.end()) return *err = "no such page: " + cursor, false;
      *out = it->second.first;
      *next = it->second.second;
      return true;
    };
  }
};

std::vector<std::string> ids(const std::vector<Found>& found) {
  std::vector<std::string> out;
  for (const Found& f : found) out.push_back(f.id);
  return out;
}

std::string joined(const std::vector<std::string>& v) {
  std::string s;
  for (const std::string& x : v) s += (s.empty() ? "" : ", ") + x;
  return s;
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  StandIn hub;
  // The GGUF stream: models, and a GGUF drafter (architecture dflash: the Hub says it is one), over two pages.
  hub.gguf = {{"", {{repo("acme/Model-GGUF", "qwen35"), repo("acme/Model-DFlash2-GGUF", "dflash", true)}, "g2"}},
              {"g2", {{repo("acme/Model-Q8-GGUF", "qwen35")}, ""}}};
  // Every repository, with its config: the same GGUF ones again, a safetensors drafter (no GGUF architecture, a config
  // the Hub module reads as a drafter's), repositories that are neither, over three pages.
  hub.all = {{"", {{repo("acme/Model-GGUF", "qwen35"), repo("acme/Model-EAGLE4", "", true),
                    repo("acme/Model-DFlash2-GGUF", "dflash", true), repo("acme/Model-safetensors")},
                   "a2"}},
             {"a2", {{repo("other/Model-DSpine", "", true), repo("acme/Model-AWQ")}, "a3"}},
             {"a3", {{repo("acme/Model-DFlash", "", true)}, ""}}};
  std::vector<std::vector<std::string>> pages;
  std::string cursor, next, err;
  std::vector<Found> found;
  bool ok = true;
  for (int i = 0; ok && i < 10; ++i) {
    ok = flyx::server::search_page(hub.fn(), "acme model", "trending", 20, cursor, &found, &next, &err, nullptr);
    if (ok) pages.push_back(ids(found));
    if (next.empty()) break;
    cursor = next;
  }
  check(ok && next.empty() && pages.size() == 3, "the pages walk both streams to their end: " + err);
  check(pages.size() == 3 &&
            pages[0] == std::vector<std::string>{"acme/Model-GGUF", "acme/Model-DFlash2-GGUF", "acme/Model-EAGLE4"} &&
            pages[1] == std::vector<std::string>{"acme/Model-Q8-GGUF", "other/Model-DSpine"} &&
            pages[2] == std::vector<std::string>{"acme/Model-DFlash"},
        "a page: the GGUF stream's repositories, then the safetensors drafters, the GGUF ones not twice: " +
            (pages.empty() ? std::string() : joined(pages[0])));
  std::set<std::string> all_ids;
  size_t shown = 0;
  for (const auto& p : pages) shown += p.size(), all_ids.insert(p.begin(), p.end());
  check(shown == all_ids.size() && shown == 6, "every repository once across the pages");
  check(hub.limit == 20 && hub.query == "acme model",
        "each stream asked for the page's limit and the query as it is (no words of a drafter in it)");
  check(hub.asked.size() == 5, "five requests: three of the unfiltered stream, two of the GGUF one, none past an end (" +
                                   joined(hub.asked) + ")");

  // The drafter stream unanswered: the page keeps the GGUF repositories, and its cursor resumes the GGUF stream alone.
  hub.all_fails = true;
  hub.asked.clear();
  ok = flyx::server::search_page(hub.fn(), "acme model", "trending", 20, "", &found, &next, &err, nullptr);
  check(ok && ids(found) == std::vector<std::string>{"acme/Model-GGUF", "acme/Model-DFlash2-GGUF"} && !next.empty(),
        "an unanswered drafter stream leaves the page its GGUF repositories");
  ok = ok && flyx::server::search_page(hub.fn(), "acme model", "trending", 20, next, &found, &next, &err, nullptr);
  check(ok && ids(found) == std::vector<std::string>{"acme/Model-Q8-GGUF"} && next.empty() &&
            hub.asked.back() == "gguf:g2",
        "and ends that stream: the next page asks the GGUF stream alone");
  hub.all_fails = false;

  // The GGUF stream unanswered: the page fails with the Hub's reason.
  hub.gguf_fails = true;
  err.clear();
  ok = flyx::server::search_page(hub.fn(), "acme model", "trending", 20, "", &found, &next, &err, nullptr);
  check(!ok && err == "the Hub answered 429", "an unanswered GGUF stream fails the page: " + err);
  hub.gguf_fails = false;

  // A cursor that is not a page's (the Hub's own, as an older server gave): the GGUF stream from it, alone.
  hub.asked.clear();
  ok = flyx::server::search_page(hub.fn(), "acme model", "trending", 20, "g2", &found, &next, &err, nullptr);
  check(ok && ids(found) == std::vector<std::string>{"acme/Model-Q8-GGUF"} && next.empty() &&
            hub.asked == std::vector<std::string>{"gguf:g2"},
        "the Hub's own cursor resumes the GGUF stream alone (" + joined(hub.asked) + ")");

  // No GGUF repository matches: the drafters alone (the query naming a drafter's model, say).
  hub.gguf = {{"", {{}, ""}}};
  ok = flyx::server::search_page(hub.fn(), "acme model", "trending", 20, "", &found, &next, &err, nullptr);
  check(ok && ids(found) == std::vector<std::string>{"acme/Model-EAGLE4"} && !next.empty(),
        "no GGUF match: the safetensors drafters still found, and the drafter stream goes on");
  return flyx::test::finish("hub_search_test");
}
