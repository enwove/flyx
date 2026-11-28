// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GET /v1/hub/search's pages for a query: the Hub's repositories holding GGUF files whose names match it, and beside
// them the drafters it matches whose files are safetensors (filter=gguf leaves those out).  A drafter is known by what
// the Hub says of it (hub::Found::is_drafter: a GGUF of architecture dflash, or a config architecture
// hub::drafter_kind_of reads), never by the query's words, so a drafter family the Hub module learns needs nothing
// here.  The two are two streams of the Hub's results paged together: a page takes the next page of each, and its
// cursor resumes both.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/hub/hub.h"

namespace flyx::server {

// hub::search, or a test's stand-in for the Hub.
using HubSearchFn = std::function<bool(const std::string& query, const std::string& sort, int limit,
                                       std::vector<hub::Found>* out, std::string* err, const std::string& cursor,
                                       std::string* next_cursor, bool gguf_only, hub::HubOutcome* outcome)>;

// One page for `query`: the GGUF stream's next `limit` repositories, then the drafters among the unfiltered stream's
// next `limit` that hold no GGUF file (those come with the GGUF stream), each repository once.  `cursor`: "" for the
// first page, else a previous page's *next_cursor ("" there when both streams are at their end; a cursor that is not
// one of this function's resumes the GGUF stream alone).  A drafter stream the Hub does not answer ends that stream
// and leaves the page its GGUF repositories; a GGUF stream it does not answer fails the page (*err, *outcome).
bool search_page(const HubSearchFn& search, const std::string& query, const std::string& sort, int limit,
                 const std::string& cursor, std::vector<hub::Found>* out, std::string* next_cursor, std::string* err,
                 hub::HubOutcome* outcome);

}  // namespace flyx::server
