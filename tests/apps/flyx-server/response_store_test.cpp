// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Responses retained for continuation: DELETE, FIFO eviction and the serialized byte budget.
#include <cstdio>

#include "apps/flyx-server/response_store.h"

using flyx::server::ResponseStore;

int main() {
  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      std::printf("FAIL %s\n", what);
      ++failures;
    }
  };
  auto entry = [](const std::string& id, const std::string& previous = "") {
    return ResponseStore::Entry{{{"id", id}, {"output", ResponseStore::Json::array({id + " reply"})}},
                               ResponseStore::Json::array({id + " input"}), previous};
  };
  const auto a = entry("a");
  const size_t unit = a.input.dump().size() + a.response.dump().size();
  ResponseStore store(2 * unit);
  for (int i = 0; i < 256; ++i) {
    const std::string id = std::to_string(i);
    store.put(entry(id));
    check(store.erase(id), "delete a stored response");
  }
  check(store.size() == 0, "deletes release every place in eviction order");
  store.put(a);
  store.put(entry("b", "a"));
  ResponseStore::Json items;
  std::string missing;
  const auto turns = ResponseStore::Json::array({"a input", "a reply", "b input", "b reply"});
  check(store.conversation("b", &items, &missing) && items == turns,
        "a chain exactly at the byte limit keeps both turns");
  auto large = entry("large");
  large.input.push_back(std::string(2 * unit, 'x'));
  store.put(std::move(large));
  check(store.size() == 2 && !store.get("large") && store.get("a"), "an oversized response leaves the rest");
  const auto reading = store.get("b");
  check(store.erase("b") && !store.erase("b"), "deleting twice reports the missing response");
  store.put(entry("c", "a"));
  store.put(entry("d", "c"));
  check(store.size() == 2 && !store.get("a") && store.get("c") && store.get("d"),
        "after DELETE the next overflow evicts the oldest live entry");
  check(!store.conversation("d", &items, &missing) && missing == "a" && items == turns,
        "an evicted ancestor is identified without returning a partial chain");
  const auto before_replace = store.get("c");
  store.put(entry("c"));  // replace in place by id, newest for eviction
  check(before_replace && before_replace->previous == "a" && store.get("c")->previous.empty(),
        "a replaced response remains readable by a reader that already held it");
  store.put(entry("e"));
  check(store.size() == 2 && store.get("c") && !store.get("d"), "a replaced id has one place in the order");
  check(reading && reading->input == entry("b").input && reading->previous == "a",
        "a reader can finish on the original entry after deletion and eviction");
  auto truncated = entry("x");
  truncated.response["output"][0] = std::string("word ") + "\xe2\x82";
  const auto wire = truncated.response.dump(-1, ' ', false, ResponseStore::Json::error_handler_t::replace);
  const auto received = ResponseStore::Json::parse(wire);
  ResponseStore exact(truncated.input.dump().size() + wire.size());
  exact.put(std::move(truncated));
  const auto out = exact.get("x");
  check(out && out->response == received && exact.conversation("x", &items, &missing) &&
            items.back() == received["output"][0],
        "a truncated UTF-8 character has the same bytes in stored history as in the HTTP response");
  ResponseStore cyclic(4 * unit);
  cyclic.put(entry("p"));
  cyclic.put(entry("q", "p"));
  cyclic.put(entry("p", "q"));  // replace creates a cycle p -> q -> p
  check(!cyclic.conversation("q", &items, &missing) && !missing.empty(),
        "a cycle in previous_response_id terminates instead of looping");
  return failures ? 1 : 0;
}
