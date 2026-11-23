// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// core/os/http.h on Windows: WinHTTP, part of the system.  Redirects are followed here, not by WinHTTP, so that an
// Authorization header reaches the URL's own host only.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winhttp.h>

#include <cctype>
#include <string>
#include <vector>

#include "core/os/http.h"
#include "core/os/win_utf8.h"

#pragma comment(lib, "winhttp.lib")

namespace flyx::os {

namespace {

using win::to_utf8;
using win::to_wide;

std::string winhttp_error(const std::string& what) {
  const DWORD e = GetLastError();
  wchar_t* msg = nullptr;
  FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS,
                 GetModuleHandleW(L"winhttp.dll"), e, 0, reinterpret_cast<wchar_t*>(&msg), 0, nullptr);
  std::string text = msg ? to_utf8(msg) : "";
  if (msg) LocalFree(msg);
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) text.pop_back();
  return what + ": " + (text.empty() ? "error " + std::to_string(e) : text);
}

struct Handle {
  HINTERNET h = nullptr;
  explicit Handle(HINTERNET x) : h(x) {}
  ~Handle() {
    if (h) WinHttpCloseHandle(h);
  }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
};

struct Url {
  std::string scheme, host, path;  // path with its query
  INTERNET_PORT port = 0;
  bool secure = false;
};

bool crack(const std::string& url, Url* out) {
  const std::wstring w = to_wide(url);
  URL_COMPONENTS uc{};
  uc.dwStructSize = sizeof uc;
  uc.dwSchemeLength = uc.dwHostNameLength = uc.dwUrlPathLength = uc.dwExtraInfoLength = DWORD(-1);
  if (!WinHttpCrackUrl(w.c_str(), DWORD(w.size()), 0, &uc)) return false;
  out->scheme = to_utf8(std::wstring(uc.lpszScheme, uc.dwSchemeLength));
  out->host = to_utf8(std::wstring(uc.lpszHostName, uc.dwHostNameLength));
  out->path = to_utf8(std::wstring(uc.lpszUrlPath, uc.dwUrlPathLength) +
                      std::wstring(uc.lpszExtraInfo, uc.dwExtraInfoLength));
  if (out->path.empty()) out->path = "/";
  out->port = uc.nPort;
  out->secure = uc.nScheme == INTERNET_SCHEME_HTTPS;
  return true;
}

// Where a Location header leads from `from`: absolute, from the host's root, or beside the current path.
std::string follow(const Url& from, const std::string& location) {
  if (location.starts_with("http://") || location.starts_with("https://")) return location;
  const std::string origin = from.scheme + "://" + from.host + ":" + std::to_string(from.port);
  if (location.starts_with("//")) return from.scheme + ":" + location;
  if (!location.empty() && location[0] == '/') return origin + location;
  std::string dir = from.path.substr(0, from.path.find('?'));
  dir.resize(dir.rfind('/') + 1);
  return origin + dir + location;
}

bool header_is(const std::string& line, const char* name) {
  const size_t n = std::char_traits<char>::length(name);
  if (line.size() <= n || line[n] != ':') return false;
  for (size_t i = 0; i < n; ++i)
    if (std::tolower(static_cast<unsigned char>(line[i])) != std::tolower(static_cast<unsigned char>(name[i])))
      return false;
  return true;
}

}  // namespace

HttpClient::~HttpClient() {
  if (handle_) WinHttpCloseHandle(static_cast<HINTERNET>(handle_));
}

bool HttpClient::get(const HttpGet& req, const HttpBody& body, int* status, std::string* err, std::string* final_url) {
  if (!handle_)  // the session keeps its connections alive between requests
    handle_ = WinHttpOpen(to_wide(http_user_agent()).c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                          WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!handle_) return *err = winhttp_error("WinHttpOpen"), false;
  struct {
    HINTERNET h;
  } session{static_cast<HINTERNET>(handle_)};
  const int connect_ms = req.connect_timeout_s * 1000, stall_ms = req.stall_timeout_s * 1000;
  WinHttpSetTimeouts(session.h, connect_ms, connect_ms, connect_ms, stall_ms);
  Url origin;
  if (!crack(req.url, &origin)) return *err = "bad URL: " + req.url, false;
  std::string url = req.url;
  for (int hop = 0; hop <= 10; ++hop) {
    Url u;
    if (!crack(url, &u)) return *err = "bad URL: " + url, false;
    Handle connection(WinHttpConnect(session.h, to_wide(u.host).c_str(), u.port, 0));
    if (!connection.h) return *err = winhttp_error("WinHttpConnect"), false;
    Handle request(WinHttpOpenRequest(connection.h, L"GET", to_wide(u.path).c_str(), nullptr, WINHTTP_NO_REFERER,
                                      WINHTTP_DEFAULT_ACCEPT_TYPES, u.secure ? WINHTTP_FLAG_SECURE : 0));
    if (!request.h) return *err = winhttp_error("WinHttpOpenRequest"), false;
    DWORD no_redirects = WINHTTP_DISABLE_REDIRECTS;
    WinHttpSetOption(request.h, WINHTTP_OPTION_DISABLE_FEATURE, &no_redirects, sizeof no_redirects);
    std::string lines;
    for (const std::string& h : req.headers) {
      if (header_is(h, "Authorization") && (u.host != origin.host || u.port != origin.port || u.secure != origin.secure))
        continue;
      lines += h + "\r\n";
    }
    if (req.ranged) lines += "Range: bytes=" + std::to_string(req.first) + "-" + std::to_string(req.last) + "\r\n";
    const std::wstring wlines = to_wide(lines);
    if (!WinHttpSendRequest(request.h, wlines.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : wlines.c_str(),
                            wlines.empty() ? 0 : DWORD(-1), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !WinHttpReceiveResponse(request.h, nullptr))
      return *err = winhttp_error("request to " + u.host), false;
    DWORD code = 0, size = sizeof code;
    if (!WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &code, &size, WINHTTP_NO_HEADER_INDEX))
      return *err = winhttp_error("status"), false;
    if (code == 301 || code == 302 || code == 303 || code == 307 || code == 308) {
      DWORD bytes = 0;
      WinHttpQueryHeaders(request.h, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes,
                          WINHTTP_NO_HEADER_INDEX);
      std::wstring location(bytes / sizeof(wchar_t), L'\0');
      if (bytes == 0 || !WinHttpQueryHeaders(request.h, WINHTTP_QUERY_LOCATION, WINHTTP_HEADER_NAME_BY_INDEX,
                                             location.data(), &bytes, WINHTTP_NO_HEADER_INDEX))
        return *err = "a redirect without a Location", false;
      location.resize(bytes / sizeof(wchar_t));
      url = follow(u, to_utf8(location));
      continue;
    }
    std::vector<char> buf(1 << 20);
    for (;;) {
      DWORD got = 0;
      if (!WinHttpReadData(request.h, buf.data(), DWORD(buf.size()), &got))
        return *err = winhttp_error("reading from " + u.host), false;
      if (got == 0) break;
      if (!body(int(code), buf.data(), got)) return *err = "stopped", false;
    }
    *status = int(code);
    if (final_url) *final_url = url;
    return true;
  }
  *err = "more than 10 redirects";
  return false;
}

}  // namespace flyx::os
