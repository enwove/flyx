// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// HTTP and HTTPS GET, for fetching models (core/hub). On Linux and macOS through the system's libcurl,
// loaded the first time a request is made; on Windows through WinHTTP.  flyx links neither: it starts without them,
// and only fetching a model needs one.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace flyx::os {

struct HttpGet {
  std::string url;
  // "Name: value" lines.  An Authorization header goes to the URL's own host only, never to the host a redirect
  // leads to (the Hub sends files from its storage hosts).
  std::vector<std::string> headers;
  bool ranged = false;  // bytes first .. last of the resource, both included
  uint64_t first = 0, last = 0;
  int connect_timeout_s = 30;
  int stall_timeout_s = 60;  // gives up when nothing (libcurl: under 1 KB/s) has come for this long
};

// Follows redirects and hands the final response's body to `body` as it arrives, with its status (the body of an
// error status too); `body` returning false stops the transfer (then false, *err "stopped").  true: a response came,
// its status in *status; false: none did, *err says why (no network, no TLS library, a timeout).
using HttpBody = std::function<bool(int status, const char* data, size_t n)>;
bool http_get(const HttpGet& req, const HttpBody& body, int* status, std::string* err);

// A client that keeps its connections from one request to the next: one for each thread that makes many requests
// (a download's connections: a new TLS connection a request was most of a small piece's time).  http_get uses one
// once.  *final_url, when asked for: where the redirects led.
class HttpClient {
 public:
  HttpClient() = default;
  ~HttpClient();
  HttpClient(const HttpClient&) = delete;
  HttpClient& operator=(const HttpClient&) = delete;
  bool get(const HttpGet& req, const HttpBody& body, int* status, std::string* err, std::string* final_url = nullptr);

 private:
  void* handle_ = nullptr;  // libcurl's easy handle; WinHTTP's session
};

// The whole body as text (API answers), at most `limit` bytes.
bool http_get_text(const HttpGet& req, std::string* out, int* status, std::string* err, size_t limit = 64u << 20);

// The User-Agent requests carry: the program and its version, nothing about the user or the machine.
const char* http_user_agent();

}  // namespace flyx::os
