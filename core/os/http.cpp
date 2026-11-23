// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// The parts of core/os/http.h every system shares.
#include "core/os/http.h"

namespace flyx::os {

const char* http_user_agent() { return "flyx"; }

bool http_get(const HttpGet& req, const HttpBody& body, int* status, std::string* err) {
  HttpClient once;
  return once.get(req, body, status, err);
}

bool http_get_text(const HttpGet& req, std::string* out, int* status, std::string* err, size_t limit) {
  out->clear();
  bool too_long = false;
  const bool ok = http_get(
      req,
      [&](int, const char* data, size_t n) {
        if (out->size() + n > limit) return !(too_long = true);
        out->append(data, n);
        return true;
      },
      status, err);
  if (too_long) *err = req.url + ": an answer longer than " + std::to_string(limit) + " bytes";
  return ok;
}

}  // namespace flyx::os
