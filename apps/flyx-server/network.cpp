// Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#include "apps/flyx-server/server.h"

#include <set>
#ifdef _WIN32
#include <iphlpapi.h>
#else
#include <ifaddrs.h>
#include <net/if.h>
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

void add_network_routes(httplib::Server& http, Server* s, const std::string& host, int port, bool host_from_cli) {
  auto status = [host, port, host_from_cli](bool saved) -> ojson {
    const bool serving_lan = !is_loopback_ip(host);
    return {{"lan_access", saved}, {"listen_host", host}, {"port", port}, {"host_from_cli", host_from_cli},
            {"serving_lan", serving_lan}, {"restart_required", !host_from_cli && serving_lan != saved},
            {"api_urls", api_urls(host, port)}};
  };
  http.Get("/v1/settings/network", [s, status](const httplib::Request& req, httplib::Response& res) {
    if (!require_manage(*s, req, res, "network settings")) return;
    send_json(res, 200, status(ModelHost::lan_access()));
  });
  http.Post("/v1/settings/network", [s, status, host_from_cli](const httplib::Request& req, httplib::Response& res) {
    if (!require_manage(*s, req, res, "network settings")) return;
    if (host_from_cli) return send_openai_error(res, 409, "the listen address is set by --host");
    ojson body;
    if (!parse_json_object(req, res, &body)) return;
    if (body.size() != 1 || !body.contains("lan_access") || !body["lan_access"].is_boolean())
      return send_openai_error(res, 400, "expected a boolean lan_access setting");
    const bool enabled = body["lan_access"].get<bool>();
    std::string err;
    if (!ModelHost::save_lan_access(enabled, &err)) return send_openai_error(res, 500, err, "server_error");
    send_json(res, 200, status(enabled));
  });
}

}  // namespace flyx::server
