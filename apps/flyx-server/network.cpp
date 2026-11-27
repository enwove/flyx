// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/server.h"

#include <cctype>
#include <set>
#ifdef _WIN32
#include <iphlpapi.h>
#else
#include <ifaddrs.h>
#include <net/if.h>
#include <unistd.h>
#endif

namespace flyx::server {
namespace {

// A wildcard is a bind address. Clients need the address of an active interface in that family.
std::set<std::string> interface_addresses(int family) {
  std::set<std::string> addresses;
  auto add = [&](const sockaddr* sa) {
    if (!sa || sa->sa_family != family) return;
    char ip[INET6_ADDRSTRLEN];
    const void* addr;
    if (family == AF_INET6) {
      const auto& v6 = reinterpret_cast<const sockaddr_in6*>(sa)->sin6_addr;
      if (IN6_IS_ADDR_LINKLOCAL(&v6)) return;  // its zone would be specific to the client
      addr = &v6;
    } else {
      addr = &reinterpret_cast<const sockaddr_in*>(sa)->sin_addr;
    }
    if (inet_ntop(family, addr, ip, sizeof(ip)) && !is_loopback_ip(ip)) addresses.insert(ip);
  };
#ifdef _WIN32
  ULONG size = 15000;
  std::vector<unsigned char> buffer;
  const ULONG flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER;
  for (;;) {
    buffer.resize(size);
    auto* adapters = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(buffer.data());
    const ULONG rc = GetAdaptersAddresses(family, flags, nullptr, adapters, &size);
    if (rc == ERROR_BUFFER_OVERFLOW) continue;
    if (rc == NO_ERROR) {
      for (auto* a = adapters; a; a = a->Next) {
        if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) continue;
        for (auto* u = a->FirstUnicastAddress; u; u = u->Next) add(u->Address.lpSockaddr);
      }
    }
    break;
  }
#else
  ifaddrs* interfaces = nullptr;
  if (getifaddrs(&interfaces) == 0) {
    for (auto* i = interfaces; i; i = i->ifa_next)
      if ((i->ifa_flags & IFF_UP) && !(i->ifa_flags & IFF_LOOPBACK)) add(i->ifa_addr);
    freeifaddrs(interfaces);
  }
#endif
  return addresses;
}

std::string local_hostname() {
  char buf[256] = {};
  if (gethostname(buf, sizeof(buf) - 1) != 0) return "";
  buf[sizeof(buf) - 1] = '\0';
  return ascii_lower(buf);
}

ojson api_urls(const std::string& host, int port) {
  const bool wildcard = host == "0.0.0.0" || host == "::";
  const auto addresses = wildcard ? interface_addresses(host == "::" ? AF_INET6 : AF_INET)
                                  : std::set<std::string>{host};
  ojson urls = ojson::array();
  for (const auto& address : addresses) {
    const std::string authority = address.find(':') == std::string::npos ? address : "[" + address + "]";
    urls.push_back("http://" + authority + ":" + std::to_string(port) + "/v1");
  }
  return urls;
}

}  // namespace

ListenConfig make_listen_config(const std::string& bound_host, int port, std::string_view requested_host) {
  ListenConfig cfg;
  cfg.host = bound_host;
  cfg.port = port;
  cfg.lan = !is_loopback_ip(bound_host);
  if (!cfg.lan) return cfg;
  std::set<std::string> hosts;
  if (bound_host == "0.0.0.0") {
    for (const auto& ip : interface_addresses(AF_INET)) hosts.insert(ascii_lower(ip));
  } else if (bound_host == "::") {
    for (const auto& ip : interface_addresses(AF_INET)) hosts.insert(ascii_lower(ip));
    for (const auto& ip : interface_addresses(AF_INET6)) hosts.insert(ascii_lower(ip));
  } else if (!bound_host.empty()) {
    hosts.insert(ascii_lower(bound_host));
  }
  // If the operator explicitly passed --host <hostname> (e.g. a LAN DNS alias that resolved to a bound IP), accept it.
  if (!requested_host.empty() && requested_host != "0.0.0.0" && requested_host != "::") {
    std::string_view alias = requested_host;
    if (alias.size() > 2 && alias.front() == '[' && alias.back() == ']') alias = alias.substr(1, alias.size() - 2);
    if (!alias.empty() && !is_loopback_ip(std::string(alias))) hosts.insert(ascii_lower(alias));
  }
  if (const std::string hn = local_hostname(); !hn.empty()) {
    hosts.insert(hn);
    if (hn.find('.') == std::string::npos) hosts.insert(hn + ".local");
  }
  cfg.accepted_hosts.assign(hosts.begin(), hosts.end());
  return cfg;
}

void add_network_routes(httplib::Server& http, Server* s, bool host_from_cli) {
  auto status = [s, host_from_cli](const ModelHost::NetworkSettingsSnapshot& snap) -> ojson {
    const bool api_key_set = !s->cfg.api_key.empty();
    const bool serving_lan = s->listener.lan;
    const bool remote_manage = s->lan_manage.from_cli || (!api_key_set && snap.remote_manage);
    const bool remote_manage_effective = serving_lan && !api_key_set && remote_manage;
    return {{"lan_access", snap.lan_access},
            {"listen_host", s->listener.host},
            {"port", s->listener.port},
            {"host_from_cli", host_from_cli},
            {"serving_lan", serving_lan},
            {"restart_required", !host_from_cli && serving_lan != snap.lan_access},
            {"remote_manage", remote_manage},
            {"remote_manage_effective", remote_manage_effective},
            {"no_auth_from_cli", s->lan_manage.from_cli},
            {"api_key_set", api_key_set},
            {"api_urls", api_urls(s->listener.host, s->listener.port)}};
  };
  http.Get("/v1/settings/network", [s, status](const httplib::Request& req, httplib::Response& res) {
    if (!require_access(*s, req, res, Access::kManage, "network settings")) return;
    send_json(res, 200, status(ModelHost::read_network_settings()));
  });
  http.Post("/v1/settings/network", [s, status, host_from_cli](const httplib::Request& req, httplib::Response& res) {
    if (!require_access(*s, req, res, Access::kManage, "network settings")) return;
    ojson body;
    if (!parse_json_object(req, res, &body)) return;
    if (body.empty()) return send_openai_error(res, 400, "expected a boolean lan_access or remote_manage setting");
    for (auto it = body.begin(); it != body.end(); ++it) {
      if (it.key() != "lan_access" && it.key() != "remote_manage")
        return send_openai_error(res, 400, "unknown network setting: " + it.key());
      if (!it.value().is_boolean())
        return send_openai_error(res, 400, "expected a boolean " + it.key() + " setting");
    }
    if (body.contains("lan_access") && host_from_cli)
      return send_openai_error(res, 409, "the listen address is set by --host");
    if (body.contains("remote_manage")) {
      if (s->lan_manage.from_cli) return send_openai_error(res, 409, "LAN management auth is disabled by --no-auth");
      if (!s->cfg.api_key.empty())
        return send_openai_error(res, 409, "an API key is configured (--api-key or FLYX_API_KEY)");
    }
    std::optional<bool> lan_opt;
    std::optional<bool> remote_opt;
    if (body.contains("lan_access")) lan_opt = body["lan_access"].get<bool>();
    if (body.contains("remote_manage")) remote_opt = body["remote_manage"].get<bool>();
    std::string err;
    ModelHost::NetworkSettingsSnapshot snap;
    if (!ModelHost::save_network_settings(
            lan_opt, remote_opt,
            [s](bool enabled) { s->lan_manage.saved.store(enabled, std::memory_order_release); }, &snap, &err))
      return send_openai_error(res, 500, err, "server_error");
    send_json(res, 200, status(snap));
  });
}

}  // namespace flyx::server
