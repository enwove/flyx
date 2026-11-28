// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// GET /v1/hub/search's pages for a query: the Hub's repositories holding GGUF files whose names match it, and beside
// them the drafters it matches whose files are safetensors (filter=gguf leaves those out).  A drafter is known by what
// the Hub says of it (hub::Found::is_drafter: a GGUF of architecture dflash, or a config architecture
// hub::drafter_kind_of reads), never by the query's words, so a drafter family the Hub module learns needs nothing
// here.  The two are two streams of the Hub's results paged together: a page takes the next page of each, and its
// cursor resumes both.
//
// What a page is, and is not: each stream gives up to `limit` rows in the Hub's order for `sort`; the page is the
// GGUF stream's rows, then the drafter stream's, so it holds up to twice `limit` (and one more where a first page adds
// an exact name's repository) and is not one list sorted across the two.  Each repository is on a page once, and an
// exact name's repository shown on a first page is on no page after it.  The paging keeps no other state: where the
// Hub's listing changes between two requests, or a drafter the Hub lists without its GGUF architecture is in both
// streams, a repository can come again on a later page.
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
// first page, else a previous page's *next_cursor ("" there when both streams are at their end).  A cursor is
// {"v":1,"g":GGUF,"d":DRAFTERS,"x":EXCLUDED}, a stream at its end left out; any other JSON object or text is the
// Hub's own cursor, which resumes the GGUF stream alone, and one of version 1 that says no stream to resume, or a
// member not text or empty, fails (HubError::kInvalidName).  A drafter stream the Hub does not answer ends that
// stream beside an answered GGUF one, the page keeping its GGUF rows; the only stream left, or the GGUF stream,
// unanswered fails the page with the Hub's error (*err, *outcome: a Retry-After kept), the cursor good to ask again.
bool search_page(const HubSearchFn& search, const std::string& query, const std::string& sort, int limit,
                 const std::string& cursor, std::vector<hub::Found>* out, std::string* next_cursor, std::string* err,
                 hub::HubOutcome* outcome);

// `next_cursor` (a page's) with the repository `id` left out of every page it resumes: an exact name's repository a
// first page showed beside the streams' rows.  "" stays "".
std::string cursor_excluding(const std::string& next_cursor, const std::string& id);

}  // namespace flyx::server
