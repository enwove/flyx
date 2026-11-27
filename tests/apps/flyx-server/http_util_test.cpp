// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Unit tests for flyx-server HTTP security guards: DNS rebinding Host validation, proxy forwarding isolation,
// keyless LAN management scoping, local-or-keyed restriction (POST /v1/hub/config), and Windows UNC/device path guards.
#include <string>
#include <vector>

#include "apps/flyx-server/http_util.h"
#include "apps/flyx-server/server.h"
#include "tests/common/check.h"

namespace {

using namespace flyx::server;

httplib::Request make_req(const std::string& remote_addr, const std::string& local_addr, int local_port,
                          const std::string& host, const std::string& origin = "",
                          const std::string& sec_fetch_site = "") {
  httplib::Request req;
  req.method = "GET";
  req.path = "/v1/logs";
  req.remote_addr = remote_addr;
  req.local_addr = local_addr;
  req.local_port = local_port;
  if (!host.empty()) req.set_header("Host", host);
  if (!origin.empty()) req.set_header("Origin", origin);
  if (!sec_fetch_site.empty()) req.set_header("Sec-Fetch-Site", sec_fetch_site);
  return req;
}

void test_dns_rebinding_and_lan_hosts() {
  Server s{};
  s.serving_lan = true;
  s.listen_host = "0.0.0.0";
  s.listen_port = 8080;
  s.allowed_hosts = {"192.168.1.50", "myhost", "myhost.local"};
  s.remote_manage.store(true);

  // Legitimate LAN client hitting the socket's local address or a known hostname: allowed.
  {
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080", "http://192.168.1.50:8080",
                        "same-origin");
    check(browser_request_safe(s, req), "same-origin LAN IP:port passes browser_request_safe");
  }
  {
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "MyHost.local:8080", "http://myhost.local:8080",
                        "same-origin");
    check(browser_request_safe(s, req), "same-origin LAN hostname.local:port passes browser_request_safe");
  }
  {
    // Dynamic VPN/Tailscale IP not present at startup still matches req.local_addr from getsockname().
    auto req = make_req("100.64.0.2", "100.64.0.1", 8080, "100.64.0.1:8080", "http://100.64.0.1:8080", "same-origin");
    check(browser_request_safe(s, req), "dynamic socket local_addr passes browser_request_safe");
  }
  {
    // IPv4-mapped IPv6 socket address (::ffff:192.168.1.50) matches IPv4 Host literal.
    auto req = make_req("::ffff:192.168.1.99", "::ffff:192.168.1.50", 8080, "192.168.1.50:8080");
    check(browser_request_safe(s, req), "IPv4-mapped IPv6 local_addr matches IPv4 Host");
  }
  {
    // DNS rebinding: attacker domain resolves to LAN IP, sends matching Host + Origin + Sec-Fetch-Site: same-origin.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "evil.example:8080", "http://evil.example:8080",
                        "same-origin");
    check(!browser_request_safe(s, req), "DNS rebinding attacker Host is refused on keyless LAN");
  }
  {
    // Wrong port in Host is refused.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:9999");
    check(!browser_request_safe(s, req), "mismatched port in Host is refused on keyless LAN");
  }
  {
    // Missing Host header on non-loopback keyless request is refused.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "");
    check(!browser_request_safe(s, req), "missing Host header is refused on keyless LAN");
  }
  {
    // With API key configured, custom reverse-proxy Host domain is permitted (key protects the endpoint).
    Server keyed{};
    keyed.cfg.api_key = "secret";
    keyed.serving_lan = true;
    keyed.listen_port = 8080;
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "llm.corp.example:8080");
    check(browser_request_safe(keyed, req), "custom domain Host is allowed when --api-key is set");
  }
}

void test_forwarded_and_management_policy() {
  // Saved remote_manage=true on a loopback-only server (serving_lan=false) must NOT unlock forwarded or remote calls.
  {
    Server s{};
    s.serving_lan = false;
    s.listen_host = "127.0.0.1";
    s.listen_port = 8080;
    s.remote_manage.store(true);

    auto direct_loop = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
    check(may_manage(s, direct_loop), "direct loopback may manage");
    check(may_manage_local_or_keyed(s, direct_loop), "direct loopback may call local-or-keyed endpoints");

    for (const auto& [hdr, val] : std::vector<std::pair<std::string, std::string>>{
             {"X-Forwarded-For", "192.0.2.2"}, {"Forwarded", "for=192.0.2.2"}, {"Via", "1.1 proxy"},
             {"X-Real-IP", "192.0.2.2"}}) {
      auto fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
      fwd.set_header(hdr, val);
      check(!may_manage(s, fwd), "loopback + " + hdr + " is denied even with saved remote_manage=true");
      check(!may_manage_local_or_keyed(s, fwd), "local-or-keyed denies " + hdr + " without API key");
    }

    auto non_loop = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    check(!may_manage(s, non_loop), "non-loopback peer is denied when serving_lan=false");
  }

  // On a LAN server with remote_manage=true (or --no-auth): direct LAN may manage, forwarded LAN is denied,
  // and local-or-keyed (POST /v1/hub/config) still denies keyless LAN!
  {
    Server s{};
    s.serving_lan = true;
    s.listen_host = "0.0.0.0";
    s.listen_port = 8080;
    s.remote_manage.store(true);

    auto lan = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    check(may_manage(s, lan), "direct LAN client may manage when serving_lan && remote_manage");
    check(!may_manage_local_or_keyed(s, lan), "keyless LAN client is denied local-or-keyed POST /v1/hub/config");

    auto lan_fwd = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    lan_fwd.set_header("X-Forwarded-For", "198.51.100.7");
    check(!may_manage(s, lan_fwd), "forwarded LAN request is denied without --api-key even when remote_manage=true");

    auto loop_fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
    loop_fwd.set_header("Forwarded", "for=198.51.100.7");
    check(!may_manage(s, loop_fwd), "forwarded loopback request is denied without --api-key even on LAN listener");
  }

  // With --api-key, valid Bearer key unlocks both may_manage and may_manage_local_or_keyed (including behind a proxy).
  {
    Server s{};
    s.cfg.api_key = "topsecret";
    s.serving_lan = true;
    s.listen_port = 8080;
    auto proxied = make_req("127.0.0.1", "127.0.0.1", 8080, "llm.example.com");
    proxied.set_header("X-Forwarded-For", "198.51.100.7");
    check(!may_manage(s, proxied), "proxied request without key is denied");
    proxied.set_header("Authorization", "Bearer topsecret");
    check(may_manage(s, proxied), "proxied request with valid API key may manage");
    check(may_manage_local_or_keyed(s, proxied), "proxied request with valid API key may call POST /v1/hub/config");
  }
}

void test_remote_unc_and_device_paths() {
  Server s{};
  s.serving_lan = true;
  s.listen_port = 8080;
  s.models_dirs = {R"(\\nas\models)", R"(/mnt/local)"};

  auto loop = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
  auto lan = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
  auto fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
  fwd.set_header("X-Forwarded-For", "192.0.2.2");

  for (const char* p : {R"(\\attacker\share\x.gguf)", "//attacker/share/x.gguf", R"(\/attacker/share)",
                        R"(/\attacker\share)", R"(\\?\C:\Windows\System32)", R"(\\.\pipe\test)",
                        R"(\\?\UNC\nas\models\a.gguf)"}) {
    check(is_unc_or_device_path(p), std::string("is_unc_or_device_path detects ") + p);
    check(remote_path_allowed(s, loop, p), std::string("direct loopback allows ") + p);
    check(!remote_path_allowed(s, lan, p), std::string("remote LAN blocks ") + p);
    check(!remote_path_allowed(s, fwd, p), std::string("forwarded request blocks ") + p);
  }

  // Operator-configured UNC root in models_dirs allows strictly contained paths, blocks traversal out of it.
  check(remote_path_allowed(s, lan, R"(\\NAS\Models\sub\tiny.gguf)"),
        "remote LAN allows UNC path lexically inside configured --models-dir UNC root");
  check(remote_path_allowed(s, lan, "//nas/models/sub/tiny.gguf"),
        "remote LAN allows forward-slash UNC path inside configured --models-dir UNC root");
  check(!remote_path_allowed(s, lan, R"(\\nas\models\..\secret\a.gguf)"),
        "remote LAN blocks UNC path traversing above configured --models-dir UNC root");
  check(!remote_path_allowed(s, lan, R"(\\nas\models2\a.gguf)"),
        "remote LAN blocks sibling UNC share sharing prefix with configured root");

  // Regular local paths and Hub names are unaffected.
  for (const char* ok_path : {"", "org/model-GGUF:Q4_K_M", "/mnt/local/a.gguf", R"(C:\models\a.gguf)"}) {
    check(!is_unc_or_device_path(ok_path), std::string("not UNC/device: ") + ok_path);
    check(remote_path_allowed(s, lan, ok_path), std::string("remote LAN allows non-UNC path: ") + ok_path);
  }
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  test_dns_rebinding_and_lan_hosts();
  test_forwarded_and_management_policy();
  test_remote_unc_and_device_paths();
  return flyx::test::finish("http_util_test");
}
