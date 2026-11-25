// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// core/os/http.h on Linux and macOS: the system's libcurl, opened with dlopen the first time a request is made, so
// that flyx starts (and runs local models) where it is missing.  The few functions and option numbers used are
// libcurl's public interface (curl/curl.h, stable since libcurl 7).
#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string_view>
#include <type_traits>

#include "core/os/http.h"

namespace flyx::os {

namespace {

// curl_easy_setopt's option numbers: a type's base (long 0, pointer 10000, function 20000) plus the option's own.
enum : int {
  kOptWriteData = 10001,
  kOptUrl = 10002,
  kOptRange = 10007,
  kOptErrorBuffer = 10010,
  kOptWriteFunction = 20011,
  kOptUserAgent = 10018,
  kOptLowSpeedLimit = 19,
  kOptLowSpeedTime = 20,
  kOptHttpHeader = 10023,
  kOptHeaderData = 10029,
  kOptFollowLocation = 52,
  kOptMaxRedirs = 68,
  kOptConnectTimeout = 78,
  kOptHeaderFunction = 20079,
  kOptNoSignal = 99,
  kOptProtocols = 181,
  kOptRedirProtocols = 182,
};
constexpr long kProtoHttpHttps = 3L;  // CURLPROTO_HTTP | CURLPROTO_HTTPS
constexpr int kInfoResponseCode = 0x200002;  // CURLINFO_LONG + 2
constexpr int kInfoEffectiveUrl = 0x100001;  // CURLINFO_STRING + 1
constexpr long kGlobalDefault = 3;           // CURL_GLOBAL_SSL | CURL_GLOBAL_WIN32
constexpr int kErrorSize = 256;              // CURL_ERROR_SIZE
constexpr int kWriteError = 23;              // CURLE_WRITE_ERROR: the write callback stopped the transfer

struct Curl {
  void* lib = nullptr;
  std::string missing;  // why it is not there
  int (*global_init)(long) = nullptr;
  void* (*easy_init)() = nullptr;
  int (*easy_setopt)(void*, int, ...) = nullptr;
  int (*easy_perform)(void*) = nullptr;
  int (*easy_getinfo)(void*, int, ...) = nullptr;
  void (*easy_cleanup)(void*) = nullptr;
  void (*easy_reset)(void*) = nullptr;
  const char* (*easy_strerror)(int) = nullptr;
  void* (*slist_append)(void*, const char*) = nullptr;
  void (*slist_free_all)(void*) = nullptr;
};

const Curl& curl() {
  static Curl c;
  static std::once_flag once;
  std::call_once(once, [] {
#if defined(__APPLE__)
    const char* names[] = {"libcurl.4.dylib", "/usr/lib/libcurl.4.dylib", "libcurl.dylib"};
#else
    const char* names[] = {"libcurl.so.4", "libcurl-gnutls.so.4", "libcurl-nss.so.4", "libcurl.so"};
#endif
    for (const char* n : names)
      if ((c.lib = dlopen(n, RTLD_NOW | RTLD_LOCAL)) != nullptr) break;
    if (!c.lib) {
      c.missing = "fetching a model needs libcurl (libcurl.so.4), which this system lacks: install it (Debian, "
                  "Ubuntu: apt install libcurl4; Fedora: dnf install libcurl), or download the files yourself "
                  "(hf download) and give their paths";
      return;
    }
    auto sym = [&](const char* name, auto* fn) {
      *fn = reinterpret_cast<std::remove_pointer_t<decltype(fn)>>(dlsym(c.lib, name));
      if (!*fn && c.missing.empty()) c.missing = std::string("libcurl without ") + name;
    };
    sym("curl_global_init", &c.global_init);
    sym("curl_easy_init", &c.easy_init);
    sym("curl_easy_setopt", &c.easy_setopt);
    sym("curl_easy_perform", &c.easy_perform);
    sym("curl_easy_getinfo", &c.easy_getinfo);
    sym("curl_easy_cleanup", &c.easy_cleanup);
    sym("curl_easy_reset", &c.easy_reset);
    sym("curl_easy_strerror", &c.easy_strerror);
    sym("curl_slist_append", &c.slist_append);
    sym("curl_slist_free_all", &c.slist_free_all);
    if (c.missing.empty() && c.global_init(kGlobalDefault) != 0) c.missing = "libcurl did not initialise";
  });
  return c;
}

struct Sink {
  const HttpBody* body;
  const Curl* curl;
  void* handle;
  bool stopped = false;
};

size_t on_data(char* data, size_t size, size_t count, void* user) {
  auto* s = static_cast<Sink*>(user);
  const size_t n = size * count;
  long code = 0;
  s->curl->easy_getinfo(s->handle, kInfoResponseCode, &code);  // the final response's: redirects have no body here
  if (!(*s->body)(int(code), data, n)) {
    s->stopped = true;
    return 0;  // anything but n stops the transfer
  }
  return n;
}

size_t on_header(char* data, size_t size, size_t count, void* user) {
  auto* link = static_cast<std::string*>(user);
  const size_t n = size * count;
  const std::string_view line(data, n);
  if (line.starts_with("HTTP/")) {
    link->clear();  // new response after a redirect
    return n;
  }
  if (line.size() >= 5 && (line[0] == 'L' || line[0] == 'l') && (line[1] == 'I' || line[1] == 'i') &&
      (line[2] == 'N' || line[2] == 'n') && (line[3] == 'K' || line[3] == 'k') && line[4] == ':') {
    size_t s = 5;
    while (s < line.size() && (line[s] == ' ' || line[s] == '\t')) ++s;
    size_t e = line.size();
    while (e > s && (line[e - 1] == '\r' || line[e - 1] == '\n' || line[e - 1] == ' ')) --e;
    if (!link->empty()) *link += ", ";
    link->append(line.substr(s, e - s));
  }
  return n;
}

}  // namespace

HttpClient::~HttpClient() {
  if (handle_) curl().easy_cleanup(handle_);
}

bool HttpClient::get(const HttpGet& req, const HttpBody& body, int* status, std::string* err, std::string* final_url,
                     std::string* link_header) {
  const Curl& c = curl();
  if (!c.missing.empty()) return *err = c.missing, false;
  if (!handle_ && !(handle_ = c.easy_init())) return *err = "libcurl: no handle", false;
  void* h = handle_;
  c.easy_reset(h);  // the options go; the open connections stay
  void* headers = nullptr;
  for (const std::string& line : req.headers) headers = c.slist_append(headers, line.c_str());
  char error[kErrorSize] = {0};
  Sink sink{&body, &c, h};
  const std::string range = req.ranged ? std::to_string(req.first) + "-" + std::to_string(req.last) : "";
  c.easy_setopt(h, kOptUrl, req.url.c_str());
  c.easy_setopt(h, kOptUserAgent, http_user_agent());
  c.easy_setopt(h, kOptFollowLocation, 1L);  // libcurl drops an Authorization header when a redirect changes host
  c.easy_setopt(h, kOptMaxRedirs, 10L);
  c.easy_setopt(h, kOptProtocols, kProtoHttpHttps);
  c.easy_setopt(h, kOptRedirProtocols, kProtoHttpHttps);
  c.easy_setopt(h, kOptConnectTimeout, long(req.connect_timeout_s));
  c.easy_setopt(h, kOptLowSpeedLimit, 1024L);
  c.easy_setopt(h, kOptLowSpeedTime, long(req.stall_timeout_s));
  c.easy_setopt(h, kOptNoSignal, 1L);  // threads: no SIGALRM for timeouts
  c.easy_setopt(h, kOptErrorBuffer, error);
  c.easy_setopt(h, kOptWriteFunction, &on_data);
  c.easy_setopt(h, kOptWriteData, &sink);
  if (link_header) {
    link_header->clear();
    c.easy_setopt(h, kOptHeaderFunction, &on_header);
    c.easy_setopt(h, kOptHeaderData, link_header);
  }
  if (headers) c.easy_setopt(h, kOptHttpHeader, headers);
  if (req.ranged) c.easy_setopt(h, kOptRange, range.c_str());
  const int rc = c.easy_perform(h);
  long code = 0;
  c.easy_getinfo(h, kInfoResponseCode, &code);
  if (final_url) {
    const char* where = nullptr;
    c.easy_getinfo(h, kInfoEffectiveUrl, &where);
    *final_url = where ? where : req.url;
  }
  if (headers) c.slist_free_all(headers);
  if (sink.stopped && rc == kWriteError) return *err = "stopped", false;
  if (rc != 0) {
    *err = error[0] ? error : c.easy_strerror(rc);
    return false;
  }
  *status = int(code);
  return true;
}

}  // namespace flyx::os
