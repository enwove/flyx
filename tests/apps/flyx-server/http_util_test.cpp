// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
// Unit tests for flyx-server HTTP security guards: DNS rebinding Host validation, proxy forwarding isolation,
// keyless LAN management scoping, local-or-keyed restriction (POST /v1/hub/config), and Windows UNC/device path guards.
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "apps/common/model_source.h"
#include "apps/flyx-server/http_util.h"
#include "apps/flyx-server/server.h"
#include "core/hub/hub.h"
#include "core/os/os.h"
#include "core/os/path.h"
#include "tests/common/check.h"

namespace {

using namespace flyx::server;

void test_chat_kwargs() {
  flyx::ChatOptions options;
  std::string err;
  options.kwargs = {{"context", {{"initial", 1}}}, {"preserve_thinking", true}};
  const flyx::ojson body = {{"chat_template_kwargs", {{"enable_thinking", false}, {"context", {{"next", 2}}}}}};
  check(read_chat_template_kwargs(body, &options, &err) && !options.enable_thinking &&
            options.kwargs == flyx::ojson{{"context", {{"next", 2}}}, {"preserve_thinking", true}},
        "request kwargs replace whole values and keep a single thinking switch");
  check(read_chat_template_kwargs(flyx::ojson::object(), &options, &err) && !options.enable_thinking,
        "absent request kwargs preserve the resolved switch");
}

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
  s.listener.lan = true;
  s.listener.host = "0.0.0.0";
  s.listener.port = 8080;
  s.listener.accepted_hosts = {"192.168.1.50", "myhost", "myhost.local", "workstation.lan"};
  s.lan_manage.saved.store(true);

  // Legitimate LAN client hitting the socket's local address or a known hostname: allowed.
  {
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080", "http://192.168.1.50:8080",
                        "same-origin");
    check(browser_request_safe(s, req), "same-origin LAN IP:port passes browser_request_safe");
    check(has_access(s, req, Access::kManage), "same-origin LAN IP:port passes kManage");
  }
  {
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "MyHost.local:8080", "http://myhost.local:8080",
                        "same-origin");
    check(browser_request_safe(s, req), "same-origin LAN hostname.local:port passes browser_request_safe");
    check(has_access(s, req, Access::kManage), "same-origin LAN hostname.local:port passes kManage");
  }
  {
    // Explicit --host hostname alias (accepted_hosts) works for keyless LAN management.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "workstation.lan:8080", "http://workstation.lan:8080",
                        "same-origin");
    check(has_access(s, req, Access::kManage), "configured --host alias passes keyless LAN kManage");
  }
  {
    // Dynamic VPN/Tailscale IP not present at startup still matches req.local_addr from getsockname().
    auto req = make_req("100.64.0.2", "100.64.0.1", 8080, "100.64.0.1:8080", "http://100.64.0.1:8080", "same-origin");
    check(browser_request_safe(s, req), "dynamic socket local_addr passes browser_request_safe");
    check(has_access(s, req, Access::kManage), "dynamic socket local_addr passes kManage");
  }
  {
    // IPv4-mapped IPv6 socket address (::ffff:192.168.1.50) matches IPv4 Host literal.
    auto req = make_req("::ffff:192.168.1.99", "::ffff:192.168.1.50", 8080, "192.168.1.50:8080");
    check(browser_request_safe(s, req), "IPv4-mapped IPv6 local_addr matches IPv4 Host");
    check(has_access(s, req, Access::kManage), "IPv4-mapped IPv6 local_addr passes kManage");
  }
  {
    // Non-browser LAN inference clients (Docker port mapping, custom DNS, HTTP/1.0 missing Host) are NOT blocked by
    // browser_request_safe(), while keyless LAN management still refuses unrecognized/missing Host headers!
    auto docker_infer = make_req("192.168.1.99", "192.168.1.50", 8080, "llm.tailnet.ts.net:18080");
    docker_infer.path = "/v1/chat/completions";
    check(browser_request_safe(s, docker_infer),
          "non-browser LAN inference with Docker/Tailscale Host passes browser_request_safe");
    check(!has_access(s, docker_infer, Access::kManage),
          "unrecognized Host is refused by keyless LAN kManage");

    auto http10 = make_req("192.168.1.99", "192.168.1.50", 8080, "");
    http10.path = "/v1/models";
    check(browser_request_safe(s, http10), "non-browser HTTP/1.0 LAN inference without Host passes browser_request_safe");
    check(!has_access(s, http10, Access::kManage), "missing Host header is refused by keyless LAN kManage");
  }
  {
    // DNS rebinding on keyless LAN management: attacker domain resolves to LAN IP with matching Origin/Sec-Fetch-Site.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "evil.example:8080", "http://evil.example:8080",
                        "same-origin");
    check(!has_access(s, req, Access::kManage), "DNS rebinding attacker Host is refused on keyless LAN kManage");
  }
  {
    // Wrong port in Host is refused on keyless LAN management.
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:9999");
    check(!has_access(s, req, Access::kManage), "mismatched port in Host is refused on keyless LAN kManage");
  }
  {
    // Loopback listener DNS rebinding: browser_request_safe rejects non-loopback Host on loopback listeners.
    Server loop_s{};
    loop_s.listener.lan = false;
    loop_s.listener.host = "127.0.0.1";
    loop_s.listener.port = 8080;
    auto rebind = make_req("127.0.0.1", "127.0.0.1", 8080, "evil.example:8080", "http://evil.example:8080",
                           "same-origin");
    check(!browser_request_safe(loop_s, rebind), "loopback DNS rebinding Host is refused by browser_request_safe");
    auto good = make_req("127.0.0.1", "127.0.0.1", 8080, "localhost:8080", "http://localhost:8080", "same-origin");
    check(browser_request_safe(loop_s, good), "loopback localhost Host passes browser_request_safe");
  }
  {
    // With API key configured, custom reverse-proxy Host domain is permitted (key protects the endpoint).
    Server keyed{};
    keyed.cfg.api_key = "secret";
    keyed.listener.lan = true;
    keyed.listener.port = 8080;
    auto req = make_req("192.168.1.99", "192.168.1.50", 8080, "llm.corp.example:8080");
    check(browser_request_safe(keyed, req), "custom domain Host is allowed when --api-key is set");
  }
}

void test_forwarded_and_management_policy() {
  // Saved remote_manage=true on a loopback-only server (listener.lan=false) must NOT unlock forwarded or remote calls.
  {
    Server s{};
    s.listener.lan = false;
    s.listener.host = "127.0.0.1";
    s.listener.port = 8080;
    s.lan_manage.saved.store(true);

    auto direct_loop = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
    check(has_access(s, direct_loop, Access::kManage), "direct loopback has kManage");
    check(has_access(s, direct_loop, Access::kLocalOrKeyed), "direct loopback has kLocalOrKeyed");
    check(has_access(s, direct_loop, Access::kLocalOnly), "direct loopback has kLocalOnly");

    for (const auto& [hdr, val] : std::vector<std::pair<std::string, std::string>>{
             {"X-Forwarded-For", "192.0.2.2"}, {"X-Forwarded-Port", "443"}, {"X-Forwarded-Custom", "1"},
             {"Forwarded", "for=192.0.2.2"}, {"Via", "1.1 proxy"}, {"X-Real-IP", "192.0.2.2"},
             {"CF-Connecting-IP", "198.51.100.1"}, {"True-Client-IP", "198.51.100.2"}}) {
      auto fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
      fwd.set_header(hdr, val);
      check(!has_access(s, fwd, Access::kManage), "loopback + " + hdr + " denied kManage even with saved remote_manage");
      check(!has_access(s, fwd, Access::kLocalOrKeyed), "loopback + " + hdr + " denied kLocalOrKeyed without API key");
      check(!has_access(s, fwd, Access::kLocalOnly), "loopback + " + hdr + " denied kLocalOnly");
    }

    auto non_loop = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    check(!has_access(s, non_loop, Access::kManage), "non-loopback peer is denied when listener.lan=false");
  }

  // On a LAN server with remote_manage=true (or --no-auth): direct LAN gets kManage, forwarded LAN is denied,
  // kLocalOrKeyed (POST /v1/hub/config) and kLocalOnly deny keyless LAN.
  {
    Server s{};
    s.listener.lan = true;
    s.listener.host = "0.0.0.0";
    s.listener.port = 8080;
    s.lan_manage.saved.store(true);

    auto lan = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    check(has_access(s, lan, Access::kManage), "direct LAN client has kManage when lan && remote_manage");
    check(!has_access(s, lan, Access::kLocalOrKeyed), "keyless LAN client denied kLocalOrKeyed (POST /v1/hub/config)");
    check(!has_access(s, lan, Access::kLocalOnly), "keyless LAN client denied kLocalOnly");

    for (const auto& [hdr, val] : std::vector<std::pair<std::string, std::string>>{
             {"X-Forwarded-For", "198.51.100.7"}, {"X-Forwarded-Custom", "yes"},
             {"CF-Connecting-IP", "198.51.100.9"}}) {
      auto lan_fwd = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
      lan_fwd.set_header(hdr, val);
      check(!has_access(s, lan_fwd, Access::kManage),
            "forwarded LAN request (" + hdr + ") denied without --api-key even when remote_manage=true");
    }

    auto loop_fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
    loop_fwd.set_header("Forwarded", "for=198.51.100.7");
    check(!has_access(s, loop_fwd, Access::kManage), "forwarded loopback request denied without --api-key on LAN");
  }

  // With --api-key, valid Bearer key unlocks kManage and kLocalOrKeyed (even behind a proxy), but NOT kLocalOnly.
  {
    Server s{};
    s.cfg.api_key = "topsecret";
    s.listener.lan = true;
    s.listener.port = 8080;
    auto proxied = make_req("127.0.0.1", "127.0.0.1", 8080, "llm.example.com");
    proxied.set_header("X-Forwarded-For", "198.51.100.7");
    check(!has_access(s, proxied, Access::kManage), "proxied request without key is denied");
    proxied.set_header("Authorization", "Bearer topsecret");
    check(has_access(s, proxied, Access::kManage), "proxied request with valid API key has kManage");
    check(has_access(s, proxied, Access::kDashboardRead), "proxied request with valid API key has kDashboardRead");
    check(has_access(s, proxied, Access::kLocalOrKeyed), "proxied request with valid API key has kLocalOrKeyed");
    check(!has_access(s, proxied, Access::kLocalOnly), "proxied request even with API key is denied kLocalOnly");
  }

  // Access::kDashboardRead: allowed on direct loopback and direct LAN with trusted Host (even when remote_manage=false),
  // but blocks untrusted Host (DNS rebinding), cross-site browser requests, and forwarded requests when keyless.
  {
    Server s{};
    s.listener.lan = true;
    s.listener.port = 8080;
    s.listener.accepted_hosts = {"192.168.1.50", "workstation"};
    s.lan_manage.saved.store(false);

    auto loop_ok = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
    check(has_access(s, loop_ok, Access::kDashboardRead), "direct loopback has kDashboardRead");

    auto loop_rebind = make_req("127.0.0.1", "127.0.0.1", 8080, "evil.example:8080");
    check(!has_access(s, loop_rebind, Access::kDashboardRead), "loopback DNS rebinding denied kDashboardRead");

    auto lan_ok = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    check(has_access(s, lan_ok, Access::kDashboardRead),
          "direct LAN with trusted Host has kDashboardRead even when remote_manage=false");
    check(!has_access(s, lan_ok, Access::kManage),
          "direct LAN with trusted Host still lacks kManage when remote_manage=false");

    auto lan_rebind = make_req("192.168.1.99", "192.168.1.50", 8080, "evil.example:8080");
    check(!has_access(s, lan_rebind, Access::kDashboardRead),
          "direct LAN with untrusted Host denied kDashboardRead");

    auto lan_xsite = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    lan_xsite.set_header("Sec-Fetch-Site", "cross-site");
    check(!has_access(s, lan_xsite, Access::kDashboardRead),
          "cross-site LAN request denied kDashboardRead");

    auto lan_fwd = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    lan_fwd.set_header("X-Forwarded-For", "198.51.100.7");
    check(!has_access(s, lan_fwd, Access::kDashboardRead),
          "forwarded keyless LAN request denied kDashboardRead");
  }
}

void test_remote_unc_and_device_paths() {
  Server s{};
  s.listener.lan = true;
  s.listener.port = 8080;
  s.models_dirs = {R"(\\nas\models)", R"(Z:\approved)", R"(/mnt/local)"};

  auto loop = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
  auto lan = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
  auto fwd = make_req("127.0.0.1", "127.0.0.1", 8080, "127.0.0.1:8080");
  fwd.set_header("X-Forwarded-For", "192.0.2.2");

  // POSIX mode (windows_paths = false): repeated leading slashes are ordinary local POSIX paths, not Windows UNC.
  check(!is_unc_or_device_path("///mnt/models", /*windows_paths=*/false),
        "POSIX mode does not classify ///mnt/models as Windows UNC");
  check(!is_unc_or_device_path("//mnt/models", /*windows_paths=*/false),
        "POSIX mode does not classify //mnt/models as Windows UNC");
  check(remote_path_allowed(s, lan, "///mnt/models", /*windows_paths=*/false),
        "POSIX mode allows ///mnt/models for remote requests");

  // Windows mode (windows_paths = true): UNC and Win32 device prefixes are detected and blocked for remote callers.
  for (const char* p : {R"(\\attacker\share\x.gguf)", "//attacker/share/x.gguf", R"(\/attacker/share)",
                        R"(/\attacker\share)", R"(\\?\C:\Windows\System32)", R"(\\.\pipe\test)",
                        R"(\\?\UNC\nas\models\a.gguf)"}) {
    check(is_unc_or_device_path(p, /*windows_paths=*/true), std::string("is_unc_or_device_path detects ") + p);
    check(remote_path_allowed(s, loop, p, /*windows_paths=*/true), std::string("direct loopback allows ") + p);
    check(!remote_path_allowed(s, lan, p, /*windows_paths=*/true), std::string("remote LAN blocks ") + p);
    check(!remote_path_allowed(s, fwd, p, /*windows_paths=*/true), std::string("forwarded request blocks ") + p);
  }

  // Operator-configured UNC root in models_dirs allows strictly contained paths, blocks traversal out of it.
  check(remote_path_allowed(s, lan, R"(\\NAS\Models\sub\tiny.gguf)", /*windows_paths=*/true),
        "remote LAN allows UNC path lexically inside configured --models-dir UNC root");
  check(remote_path_allowed(s, lan, "//nas/models/sub/tiny.gguf", /*windows_paths=*/true),
        "remote LAN allows forward-slash UNC path inside configured --models-dir UNC root");
  check(!remote_path_allowed(s, lan, R"(\\nas\models\..\secret\a.gguf)", /*windows_paths=*/true),
        "remote LAN blocks UNC path traversing above configured --models-dir UNC root");
  check(!remote_path_allowed(s, lan, R"(\\nas\models2\a.gguf)", /*windows_paths=*/true),
        "remote LAN blocks sibling UNC share sharing prefix with configured root");

  // Windows mapped network drive (DRIVE_REMOTE) and reparse points pointing to UNC / remote drives (tested via probe).
  PathSafetyProbe probe;
  probe.is_remote_drive = [](char drive) { return drive == 'Z' || drive == 'z' || drive == 'Y' || drive == 'y'; };
  probe.reparse_info = [](std::string_view p) -> ReparseResult {
    if (p == R"(C:\links\to_unc)" || p == "C:/links/to_unc") return {ReparseState::kTarget, R"(\\attacker\share)"};
    if (p == R"(C:\links\to_mapped)") return {ReparseState::kTarget, R"(Y:\models)"};
    if (p == R"(C:\links\to_local)") return {ReparseState::kTarget, R"(D:\local\models)"};
    if (p == R"(D:\local\models\sub_unc)") return {ReparseState::kTarget, R"(\\evil\share)"};
    if (p == R"(C:\links\sub\rel_to_unc)") return {ReparseState::kTarget, R"(..\to_unc)"};
    if (p == R"(C:\links\sub\rel_to_local)") return {ReparseState::kTarget, R"(..\to_local)"};
    if (p == R"(C:\links\unknown_rp)") return {ReparseState::kUnknownOrError, ""};
    return {ReparseState::kNone, ""};
  };
  check(!remote_path_allowed(s, lan, R"(Y:\secret\a.gguf)", true, &probe),
        "remote LAN blocks unconfigured mapped network drive (DRIVE_REMOTE)");
  check(remote_path_allowed(s, loop, R"(Y:\secret\a.gguf)", true, &probe),
        "direct loopback allows mapped network drive");
  check(remote_path_allowed(s, lan, R"(Z:\approved\sub\m.gguf)", true, &probe),
        "remote LAN allows mapped network drive explicitly configured in --models-dir");
  check(!remote_path_allowed(s, lan, R"(Z:\other\m.gguf)", true, &probe),
        "remote LAN blocks sibling path on mapped drive outside --models-dir");
  check(!remote_path_allowed(s, lan, R"(C:\links\to_unc\a.gguf)", true, &probe),
        "remote LAN blocks local path traversing reparse point targeting UNC share");
  check(!remote_path_allowed(s, lan, R"(C:\links\to_mapped\a.gguf)", true, &probe),
        "remote LAN blocks local path traversing reparse point targeting mapped remote drive");
  check(remote_path_allowed(s, lan, R"(C:\links\to_local\a.gguf)", true, &probe),
        "remote LAN allows local reparse point targeting local drive");
  check(!remote_path_allowed(s, lan, R"(C:\links\to_local\sub_unc\m.gguf)", true, &probe),
        "remote LAN fails closed when remaining tail after local symlink traverses a nested UNC junction");
  check(!remote_path_allowed(s, lan, R"(C:\links\sub\rel_to_unc\m.gguf)", true, &probe),
        "remote LAN resolves relative reparse target against link parent and blocks UNC target");
  check(remote_path_allowed(s, lan, R"(C:\links\sub\rel_to_local\m.gguf)", true, &probe),
        "remote LAN resolves relative reparse target against link parent and allows local target");
  check(!remote_path_allowed(s, lan, R"(C:\links\unknown_rp\m.gguf)", true, &probe),
        "remote LAN fails closed on unknown or unreadable reparse point");

  // Paths deeper than 64 components fail closed instead of silently trusting uninspected tail components.
  std::string deep = R"(C:)";
  for (int i = 0; i < 66; ++i) deep += R"(\d)";
  deep += R"(\m.gguf)";
  check(!remote_path_allowed(s, lan, deep, true, &probe),
        "remote LAN fails closed on paths exceeding 64 components");

  // Regular local paths and Hub names are unaffected.
  for (const char* ok_path : {"", "org/model-GGUF:Q4_K_M", "/mnt/local/a.gguf", R"(C:\models\a.gguf)"}) {
    check(!is_unc_or_device_path(ok_path, /*windows_paths=*/true), std::string("not UNC/device: ") + ok_path);
    check(remote_path_allowed(s, lan, ok_path, /*windows_paths=*/true, &probe),
          std::string("remote LAN allows non-UNC local path: ") + ok_path);
  }

  // remote_path_policy returns empty for loopback or POSIX mode, and captures models_dirs + probe by value for remote Windows.
  check(!remote_path_policy(s, loop, /*windows_paths=*/true, &probe),
        "remote_path_policy returns empty predicate for direct loopback");
  check(!remote_path_policy(s, lan, /*windows_paths=*/false, &probe),
        "remote_path_policy returns empty predicate in POSIX mode");

  flyx::PathAllowFn detached_policy;
  {
    Server tmp_s{};
    tmp_s.listener.lan = true;
    tmp_s.listener.port = 8080;
    tmp_s.models_dirs = {R"(\\nas\models)"};
    auto tmp_req = make_req("192.168.1.99", "192.168.1.50", 8080, "192.168.1.50:8080");
    PathSafetyProbe tmp_probe;
    tmp_probe.is_remote_drive = [](char d) { return d == 'Y' || d == 'y'; };
    tmp_probe.reparse_info = [](std::string_view p) -> ReparseResult {
      std::string s(p);
      std::replace(s.begin(), s.end(), '\\', '/');
      if (s == "C:/cache/models--acme--ok/snapshots/1111111111111111111111111111111111111111/ok.gguf")
        return {ReparseState::kTarget, R"(..\..\blobs\abc)"};
      if (s == "C:/cache/models--acme--ok/snapshots/1111111111111111111111111111111111111111/evil.gguf")
        return {ReparseState::kTarget, R"(\\attacker\share\evil.gguf)"};
      return {ReparseState::kNone, ""};
    };
    detached_policy = remote_path_policy(tmp_s, tmp_req, /*windows_paths=*/true, &tmp_probe);
  }
  check(bool(detached_policy), "remote_path_policy returns active predicate for remote Windows caller");
  check(detached_policy(R"(\\nas\models\sub\ok.gguf)"),
        "detached remote_path_policy still allows configured UNC models_dirs after Server/Request destroy");
  check(!detached_policy(R"(Y:\secret\a.gguf)"),
        "detached remote_path_policy still blocks unconfigured remote drive after Server/Request destroy");

  // Combine detached remote_path_policy with hub::cached: normal local blob symlink is allowed, UNC symlink is skipped
  // with zero target-following filesystem calls.
  auto norm = [](const std::filesystem::path& p) {
    std::string s = flyx::os::from_path(p);
    std::replace(s.begin(), s.end(), '\\', '/');
    return s;
  };
  const std::string cache_root = "C:/cache";
  const std::string repo_dir = "C:/cache/models--acme--ok";
  const std::string snaps_dir = "C:/cache/models--acme--ok/snapshots";
  const std::string commit = "1111111111111111111111111111111111111111";
  const std::string snap_dir = snaps_dir + "/" + commit;
  const std::string ok_file = snap_dir + "/ok.gguf";
  std::vector<std::string> touched_evil;
  auto track_evil = [&](const std::filesystem::path& p, const char* op) {
    const std::string s_path = norm(p);
    if (s_path.find("evil.gguf") != std::string::npos) touched_evil.push_back(std::string(op) + ":" + s_path);
  };
  flyx::hub::CacheFs fake_cache;
  fake_cache.root_override = cache_root;
  fake_cache.is_directory = [&](const std::filesystem::path& p, std::error_code&) {
    track_evil(p, "is_directory");
    const std::string sp = norm(p);
    return sp == cache_root || sp == repo_dir || sp == snaps_dir || sp == snap_dir;
  };
  fake_cache.is_regular_file = [&](const std::filesystem::path& p, std::error_code&) {
    track_evil(p, "is_regular_file");
    return norm(p) == ok_file;
  };
  fake_cache.file_size = [&](const std::filesystem::path& p, std::error_code&) -> uint64_t {
    track_evil(p, "file_size");
    return 2048;
  };
  fake_cache.last_write_time = [&](const std::filesystem::path& p, std::error_code&) {
    track_evil(p, "last_write_time");
    return std::filesystem::file_time_type::clock::now();
  };
  fake_cache.for_each_dir_entry = [&](const std::filesystem::path& p,
                                      const std::function<bool(const std::filesystem::path&)>& visit,
                                      std::error_code&) {
    track_evil(p, "for_each_dir_entry");
    const std::string sp = norm(p);
    std::vector<std::filesystem::path> entries;
    if (sp == cache_root) entries = {p / "models--acme--ok"};
    else if (sp == snaps_dir) entries = {p / commit};
    else if (sp == snap_dir) entries = {p / "evil.gguf", p / "ok.gguf"};
    for (const auto& e : entries)
      if (!visit(e)) break;
  };
  fake_cache.read_text = [&](const std::filesystem::path& p) -> std::string {
    track_evil(p, "read_text");
    if (norm(p) == repo_dir + "/refs/main") return commit;
    return {};
  };
  const auto cached = flyx::hub::cached(detached_policy, &fake_cache);
  check(touched_evil.empty(),
        "hub::cached + remote_path_policy never performs target-following ops on UNC symlink entry");
  check(cached.size() == 1 && cached[0].files.size() == 1 && cached[0].files[0].path == "ok.gguf",
        "hub::cached + remote_path_policy keeps local relative blob symlink while skipping UNC symlink");

  // Production-path integration: combine real remote_path_policy with apps::known_drafters, hub::describe offline
  // fallback, hub::fetch, and hub::remove_cached on a temporary cache tree (cross-platform via PathSafetyProbe).
  auto set_env = [](const char* name, const std::string& val) {
#if defined(_WIN32)
    _putenv_s(name, val.c_str());
#else
    setenv(name, val.c_str(), 1);
#endif
  };
  namespace fs = std::filesystem;
  const fs::path tmp_root =
      fs::temp_directory_path() / ("flyx-http-util-hub-policy-" + std::to_string(flyx::os::process_id()));
  std::error_code tec;
  fs::remove_all(tmp_root, tec);
  const std::string c_id = "1111111111111111111111111111111111111111";
  const fs::path drafters_snap = tmp_root / "models--Enwove--flyx-drafters" / "snapshots" / c_id;
  fs::create_directories(drafters_snap);
  fs::create_directories(tmp_root / "models--Enwove--flyx-drafters" / "refs");
  std::ofstream(tmp_root / "models--Enwove--flyx-drafters" / "refs" / "main", std::ios::binary) << c_id;
  std::ofstream(drafters_snap / "drafters.json", std::ios::binary)
      << R"({"version":1,"drafters":[{"base":"acme/BaseModel","repo":"acme/BaseModel-DFlash2","kind":"dflash2"}]})";
  const fs::path repo_dir_fs = tmp_root / "models--acme--demo-GGUF";
  const fs::path demo_snap = repo_dir_fs / "snapshots" / c_id;
  fs::create_directories(demo_snap);
  fs::create_directories(repo_dir_fs / "refs");
  fs::create_directories(repo_dir_fs / "blobs");
  std::ofstream(repo_dir_fs / "refs" / "main", std::ios::binary) << c_id;
  std::ofstream(demo_snap / "ok-Q4_K_M.gguf", std::ios::binary) << "ok";
  std::ofstream(demo_snap / "unc-evil-Q8_0.gguf", std::ios::binary) << "evil";

  set_env("HF_HUB_CACHE", flyx::os::from_path(tmp_root));
  set_env("HF_HUB_OFFLINE", "1");

  PathSafetyProbe cache_probe;
  cache_probe.is_remote_drive = [](char) { return false; };
  cache_probe.reparse_info = [](std::string_view p) -> ReparseResult {
    std::string sp(p);
    std::replace(sp.begin(), sp.end(), '\\', '/');
    if (sp.find("drafters.json") != std::string::npos || sp.find("unc-evil") != std::string::npos ||
        sp.find("models--acme--fetch-unc") != std::string::npos)
      return {ReparseState::kTarget, R"(\\attacker\share\escape.bin)"};
    return {ReparseState::kNone, {}};
  };
  Server policy_srv{};
  const auto remote_req = make_req("192.168.1.50", "192.168.1.10", 8080, "192.168.1.10:8080");
  const auto prod_policy = remote_path_policy(policy_srv, remote_req, /*windows_paths=*/true, &cache_probe);
  check(static_cast<bool>(prod_policy), "remote_path_policy returns active policy for remote peer");

  // 1) Verified drafter lookup respects remote_path_policy (blocked when drafters.json redirects to UNC; allowed otherwise).
  const auto blocked_drafters = flyx::apps::known_drafters("acme/BaseModel", {}, false, prod_policy);
  check(blocked_drafters.empty(),
        "apps::known_drafters blocks UNC-redirected cached drafters.json under remote_path_policy");
  const auto allowed_drafters = flyx::apps::known_drafters("acme/BaseModel", {}, false, {});
  check(allowed_drafters.size() == 1 && allowed_drafters[0].repo == "acme/BaseModel-DFlash2",
        "apps::known_drafters reads valid cached drafters.json when allowed");

  // 2) hub::describe offline fallback filters out UNC-redirected entry via remote_path_policy.
  flyx::hub::Repo desc;
  std::string hub_err;
  check(flyx::hub::describe("acme/demo-GGUF", "main", &desc, &hub_err, nullptr, prod_policy) && desc.from_cache &&
            desc.files.size() == 1 && desc.files[0].path == "ok-Q4_K_M.gguf",
        "hub::describe offline fallback + remote_path_policy skips UNC-redirected file: " + hub_err);

  // 3) hub::fetch rejects UNC-redirected repo cache path before creating any directory.
  flyx::hub::Repo unc_fetch_repo;
  unc_fetch_repo.id = "acme/fetch-unc";
  unc_fetch_repo.commit = c_id;
  unc_fetch_repo.revision = "main";
  flyx::hub::RepoFile rf;
  rf.path = "m-Q4_K_M.gguf";
  rf.size = 8;
  rf.sha256 = std::string(64, 'a');
  std::vector<std::string> out_paths;
  hub_err.clear();
  check(!flyx::hub::fetch(unc_fetch_repo, {rf}, &out_paths, nullptr, nullptr, &hub_err, prod_policy) &&
            hub_err.find("remote path policy") != std::string::npos &&
            !fs::exists(tmp_root / "models--acme--fetch-unc"),
        "hub::fetch + remote_path_policy rejects UNC-redirected cache dir before creating dirs: " + hub_err);

  // 4) hub::remove_cached refuses to remove UNC-redirected entry under remote_path_policy.
  uint64_t freed = 0;
  hub_err.clear();
  check(!flyx::hub::remove_cached("acme/demo-GGUF", c_id, {"unc-evil-Q8_0.gguf"}, &freed, &hub_err, prod_policy) &&
            hub_err.find("remote path policy") != std::string::npos && fs::exists(demo_snap / "unc-evil-Q8_0.gguf"),
        "hub::remove_cached + remote_path_policy refuses UNC-redirected entry: " + hub_err);

  set_env("HF_HUB_OFFLINE", "");
  fs::remove_all(tmp_root, tec);
}

}  // namespace

int main() {
  flyx::test::print_passes = true;
  test_chat_kwargs();
  test_dns_rebinding_and_lan_hosts();
  test_forwarded_and_management_policy();
  test_remote_unc_and_device_paths();
  return flyx::test::finish("http_util_test");
}
