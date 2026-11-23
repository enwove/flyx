// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The Responses API's stored input and output items, linked by previous_response_id.
#pragma once

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace flyx::server {

class ResponseStore {
 public:
  using Json = nlohmann::ordered_json;
  struct Entry {
    Json response;         // the response returned to the client, including its output items
    Json input;            // this request's own input items, with ids
    std::string previous;  // the response it continued
    size_t bytes = 0;
  };
  using Stored = std::shared_ptr<const Entry>;

  explicit ResponseStore(size_t budget) : budget_(budget) {}

  void put(Entry e) {
    e.bytes = normalize_and_size(e.response) + normalize_and_size(e.input);
    if (e.bytes > budget_) return;  // this response cannot fit, even with all the older entries evicted
    // Allocate the list node and index it before changing which responses are kept.
    std::list<Stored> fresh;
    auto item = fresh.insert(fresh.end(), std::make_shared<const Entry>(std::move(e)));
    const auto& id = (*item)->response.at("id").get_ref<const std::string&>();
    std::lock_guard<std::mutex> lock(mu_);
    auto [index, inserted] = by_id_.try_emplace(id, item);
    if (!inserted) {
      bytes_ -= (*index->second)->bytes;
      oldest_.erase(index->second);
      index->second = item;
    }
    oldest_.splice(oldest_.end(), fresh);
    bytes_ += (*item)->bytes;
    while (bytes_ > budget_) {
      const Entry& oldest = *oldest_.front();
      bytes_ -= oldest.bytes;
      by_id_.erase(oldest.response.at("id").get_ref<const std::string&>());
      oldest_.pop_front();
    }
  }

  Stored get(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_id_.find(id);
    return it == by_id_.end() ? nullptr : *it->second;
  }

  bool erase(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_id_.find(id);
    if (it == by_id_.end()) return false;
    bytes_ -= (*it->second)->bytes;
    oldest_.erase(it->second);
    by_id_.erase(it);
    return true;
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mu_);
    return oldest_.size();
  }

  // The input and output items of the response and its ancestors, oldest first. A missing link is reported.
  bool conversation(const std::string& id, Json* items, std::string* missing) {
    std::vector<Stored> chain;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (std::string r = id; !r.empty();) {
        if (chain.size() >= by_id_.size()) {
          *missing = r;
          return false;
        }
        auto it = by_id_.find(r);
        if (it == by_id_.end()) {
          *missing = r;
          return false;
        }
        chain.push_back(*it->second);
        r = chain.back()->previous;
      }
    }
    *items = Json::array();
    for (auto c = chain.rbegin(); c != chain.rend(); ++c) {
      for (const auto& i : (*c)->input) items->push_back(i);
      for (const auto& o : (*c)->response["output"]) items->push_back(o);
    }
    return true;
  }

 private:
  // A byte-level model can end on an incomplete UTF-8 sequence. A continuation must see the
  // same replacement characters as a client replaying the JSON response it received.
  static size_t normalize_and_size(Json& j) {
    try {
      return j.dump().size();
    } catch (const Json::type_error&) {
      const std::string wire = j.dump(-1, ' ', false, Json::error_handler_t::replace);
      j = Json::parse(wire);
      return wire.size();
    }
  }

  size_t budget_, bytes_ = 0;
  std::mutex mu_;
  std::list<Stored> oldest_;  // each id indexes its one place in eviction order
  std::unordered_map<std::string, std::list<Stored>::iterator> by_id_;
};

}  // namespace flyx::server
