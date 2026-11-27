# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the saved preference, real listener, dashboard and management policy across launches."""
import ipaddress
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from urllib.parse import urlsplit

from local_server import Client, serve, server_env


NETWORK = "/v1/settings/network"


def check_listener(origin, host, *, saved, cli=False):
    cl = Client(origin, None)
    status, network = cl.get(NETWORK)
    assert status == 200, network
    assert network["listen_host"] == host and network["port"] == cl.port, network
    lan = host == "0.0.0.0"
    assert network["host_from_cli"] is cli and network["serving_lan"] is lan, network
    assert network["lan_access"] is saved and network["restart_required"] is (not cli and saved != lan), network
    if lan:
        for url in network["api_urls"]:
            parsed = urlsplit(url)
            addr = ipaddress.ip_address(parsed.hostname)
            assert addr.version == 4 and not addr.is_loopback and not addr.is_unspecified, url
            assert parsed.port == cl.port and parsed.path == "/v1", url
        assert Client(f"http://127.0.0.2:{cl.port}", None).get("/v1/models")[0] == 200
    else:
        assert network["api_urls"] == [origin + "/v1"], network
        with socket.socket() as sock:
            sock.settimeout(2)
            assert sock.connect_ex(("127.0.0.2", cl.port)) != 0
    return cl


def main(exe, asset, root):
    env = server_env(root, exe)
    settings = root / "flyx" / "serve.json"

    def start(check, *options):
        serve(exe, root, env, on_ready=check, options=options)

    def enable(origin):
        cl = check_listener(origin, "127.0.0.1", saved=False)
        for path, needle in [("/dashboard", b'id="app"'), ("/dashboard/components/sidebar.js", b"Server settings"),
                             ("/dashboard/pages/settings.js", NETWORK.encode())]:
            status, _, body = cl.get_raw(path)
            assert status == 200 and needle in body, path
        for bad in ({}, {"lan_access": 1}, {"lan_access": "true"}, {"lan_access": True, "port": 8080}):
            assert cl.post(NETWORK, bad)[0] == 400
        for headers in ({"Origin": "https://example.org"}, {"X-Forwarded-For": "192.0.2.2"}):
            assert cl.get_raw(NETWORK, headers)[0] == 403
            assert cl.post(NETWORK, {"lan_access": True}, headers)[0] == 403
        assert cl.get(NETWORK)[1]["lan_access"] is False

        # A write failure cannot present the choice as saved or change the active listener.
        original = settings.read_bytes()
        settings.unlink()
        settings.mkdir()
        try:
            assert cl.post(NETWORK, {"lan_access": True})[0] == 500
        finally:
            settings.rmdir()
            settings.write_bytes(original)
        assert cl.post(NETWORK, {"lan_access": True})[0] == 200
        check_listener(origin, "127.0.0.1", saved=True)
        assert json.loads(settings.read_bytes())["last"] == json.loads(original)["last"]

    serve(exe, root, env, [asset], enable, options=["--name", "test model", "--cors", "*"])

    def locked_loopback(origin):
        cl = check_listener(origin, "127.0.0.1", saved=True, cli=True)
        assert cl.post(NETWORK, {"lan_access": False})[0] == 409

    start(locked_loopback, "--host", "127.0.0.1")

    def disable(origin):
        cl = check_listener(origin, "0.0.0.0", saved=True)
        for url in cl.get(NETWORK)[1]["api_urls"][:1]:
            lan_client = Client(url, None)
            assert lan_client.get("/v1/models")[0] == 200
            assert lan_client.post(NETWORK, {"lan_access": False})[0] == 403
        assert cl.post(NETWORK, {"lan_access": False})[0] == 200
        check_listener(origin, "0.0.0.0", saved=False)

    start(disable)
    start(lambda origin: check_listener(origin, "127.0.0.1", saved=False))

    def locked_wildcard(origin):
        cl = check_listener(origin, "0.0.0.0", saved=False, cli=True)
        assert cl.post(NETWORK, {"lan_access": True})[0] == 409

    start(locked_wildcard, "--host", "0.0.0.0")

    def keyed(origin):
        assert Client(origin, None).get(NETWORK)[0] == 403
        assert Client(origin, None).post(NETWORK, {"lan_access": True})[0] == 403
        cl = Client(origin, "lan-test-key")
        assert cl.get(NETWORK)[1]["lan_access"] is False
        assert cl.get(NETWORK)[1]["api_key_set"] is True
        assert cl.post(NETWORK, {"remote_manage": True})[0] == 409
        assert cl.post(NETWORK, {"lan_access": True})[1]["restart_required"] is True

    start(keyed, "--api-key", "lan-test-key")

    # Toggle remote_manage dynamically via POST /v1/settings/network on loopback:
    # remote_manage_effective stays False on loopback, forwarded headers stay denied (403) even when remote_manage is saved.
    def toggle_remote_manage_loopback(origin):
        cl = Client(origin, None)
        fwd = {"X-Forwarded-For": "192.0.2.2"}
        assert cl.get_raw("/v1/logs", fwd)[0] == 403
        assert cl.get_raw("/v1/requests", fwd)[0] == 403
        for bad in ({"remote_manage": 1}, {"remote_manage": "true"}):
            assert cl.post(NETWORK, bad)[0] == 400
        status, net = cl.post(NETWORK, {"remote_manage": True})
        assert status == 200 and net["remote_manage"] is True and net["remote_manage_effective"] is False, net
        # Even with remote_manage saved, forwarded headers and loopback-only binding never unlock keyless management.
        for extra in (fwd, {"X-Forwarded-Custom": "1"}, {"CF-Connecting-IP": "198.51.100.1"}):
            assert cl.get_raw("/v1/logs", extra)[0] == 403, extra
            assert cl.get_raw("/v1/requests", extra)[0] == 403, extra
            assert cl.get_raw("/v1/hub/cached", extra)[0] == 403, extra
        assert cl.get_raw("/v1/logs", {"Origin": "https://example.org"})[0] == 403

    start(toggle_remote_manage_loopback, "--host", "127.0.0.1")

    # Persisted remote_manage survives restart; on 0.0.0.0 direct LAN requests work while forwarded/rebinding/hub-config/UNC stay blocked.
    def persisted_remote_manage_lan(origin):
        cl = Client(origin, None)
        fwd = {"X-Forwarded-For": "192.0.2.2"}
        status, net = cl.get(NETWORK)
        assert status == 200 and net["remote_manage"] is True and net["serving_lan"] is True, net
        assert net["remote_manage_effective"] is True, net
        # Forwarded headers remain denied even on 0.0.0.0 with remote_manage=true.
        for extra in (fwd, {"X-Forwarded-Custom": "1"}, {"CF-Connecting-IP": "198.51.100.1"}):
            assert cl.get_raw("/v1/logs", extra)[0] == 403, extra
            assert cl.get_raw("/v1/requests", extra)[0] == 403, extra
            assert cl.get_raw("/v1/hub/cached", extra)[0] == 403, extra
        for url in net["api_urls"][:1]:
            lan_cl = Client(url, None)
            assert lan_cl.get_raw("/v1/logs")[0] == 200
            assert lan_cl.get_raw("/v1/requests")[0] == 200
            assert lan_cl.get_raw("/v1/hub/cached")[0] == 200
            # Non-browser LAN inference clients behind Docker port mapping or custom DNS Host are NOT blocked!
            assert lan_cl.get_raw("/v1/models", {"Host": "docker.custom.host:19000"})[0] == 200
            # Keyless LAN clients may never mutate the Hub mirror endpoint.
            assert lan_cl.post("/v1/hub/config", {"endpoint": "https://hf-mirror.com"})[0] == 403
            # DNS-rebinding Host header is rejected on management even with matching same-origin headers.
            rebind_host = f"evil.example:{lan_cl.port}"
            assert lan_cl.get_raw("/v1/logs", {
                "Host": rebind_host,
                "Origin": f"http://{rebind_host}",
                "Sec-Fetch-Site": "same-origin",
            })[0] == 403
            # On Windows, remote UNC/SMB path probing is rejected with 403 before touching the filesystem/network;
            # on POSIX, backslashes are ordinary path characters so the nonexistent path returns 404.
            expected_unc_status = 403 if os.name == "nt" else 404
            assert lan_cl.get_raw("/v1/hub/dir?path=%5C%5Cattacker%5Cshare")[0] == expected_unc_status
            # Disabling remote_manage takes effect immediately on the LAN client.
            st_off, net_off = cl.post(NETWORK, {"remote_manage": False})
            assert st_off == 200 and net_off["remote_manage_effective"] is False, net_off
            assert lan_cl.get_raw("/v1/logs")[0] == 403
            st_on, net_on = cl.post(NETWORK, {"remote_manage": True})
            assert st_on == 200 and net_on["remote_manage_effective"] is True, net_on
            assert lan_cl.get_raw("/v1/logs")[0] == 200
        assert cl.post(NETWORK, {"remote_manage": False})[0] == 200
        assert cl.get(NETWORK)[1]["remote_manage"] is False

    start(persisted_remote_manage_lan, "--host", "0.0.0.0")

    # Atomic two-field POST /v1/settings/network persists both lan_access and remote_manage together.
    def atomic_network_save(origin):
        cl = Client(origin, None)
        status, net = cl.post(NETWORK, {"lan_access": True, "remote_manage": True})
        assert status == 200 and net["lan_access"] is True and net["remote_manage"] is True, net
        saved = json.loads(settings.read_bytes())
        assert saved["lan_access"] is True and saved["remote_manage"] is True, saved
        status, net = cl.post(NETWORK, {"lan_access": False, "remote_manage": False})
        assert status == 200 and net["lan_access"] is False and net["remote_manage"] is False, net
        saved = json.loads(settings.read_bytes())
        assert saved["lan_access"] is False and saved["remote_manage"] is False, saved

    start(atomic_network_save)

    # --no-auth unlocks direct LAN management (including logs, /v1/messages request log, and --models-dir passed before MODEL),
    # ignores FLYX_API_KEY in the environment, locks remote_manage (409), and still blocks forwarded headers, cross-site Origin,
    # DNS rebinding, keyless hub/config mutation, and remote UNC paths (403).
    models_dir = root / "custom-models"
    nested = models_dir / "Author" / "Repo" / "fake.gguf"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"GGUF\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    no_auth_env = dict(env, FLYX_API_KEY="ignored-when-no-auth")

    def check_no_auth(origin):
        cl = Client(origin, None)
        fwd = {"X-Forwarded-For": "192.0.2.2"}
        status, net = cl.get(NETWORK)
        assert status == 200 and net["remote_manage"] is True and net["no_auth_from_cli"] is True, net
        assert net["api_key_set"] is False, net
        assert cl.post(NETWORK, {"remote_manage": False})[0] == 409
        assert cl.get_raw(NETWORK, {"Origin": "https://example.org"})[0] == 403
        # Forwarded headers are always denied without --api-key, even under --no-auth.
        assert cl.get_raw("/v1/logs", fwd)[0] == 403
        assert cl.get_raw("/v1/requests", fwd)[0] == 403
        assert cl.get_raw("/v1/hub/cached", fwd)[0] == 403
        # Direct client (real LAN interface when available, else loopback) accesses management + --models-dir + /v1/messages log.
        direct_cl = Client(net["api_urls"][0], None) if net["api_urls"] else cl
        assert direct_cl.get_raw("/v1/logs")[0] == 200
        status, _, raw_cached = direct_cl.get_raw("/v1/hub/cached")
        cached = json.loads(raw_cached or b"null")
        assert status == 200, cached
        assert any(Path(d).resolve() == models_dir.resolve() for d in cached["dirs"]), cached["dirs"]
        assert any(Path(m["path"]).resolve() == nested.resolve() for m in cached["models"] if m.get("path")), cached["models"]
        msg_status, _ = direct_cl.post("/v1/messages", {
            "model": "test model", "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert msg_status == 200, msg_status
        req_status, _, raw_reqs = direct_cl.get_raw("/v1/requests")
        reqs = json.loads(raw_reqs or b"null")
        assert req_status == 200 and reqs["enabled"] is True, reqs
        assert any(r["api"] == "messages" for r in reqs["requests"]), reqs["requests"]
        if net["api_urls"]:
            assert direct_cl.post("/v1/hub/config", {"endpoint": "https://hf-mirror.com"})[0] == 403
            rebind_host = f"evil.example:{direct_cl.port}"
            assert direct_cl.get_raw("/v1/logs", {
                "Host": rebind_host,
                "Origin": f"http://{rebind_host}",
            })[0] == 403
            assert direct_cl.get_raw("/v1/hub/dir?path=%5C%5Cattacker%5Cshare")[0] == 403

    # Pass options BEFORE the positional model path to verify positional MODEL is accepted anywhere.
    serve(exe, root, no_auth_env, ["--host", "0.0.0.0", "--no-auth", "--models-dir", str(models_dir), asset],
          check_no_auth, options=["--name", "test model"])

    # A hand-edited settings file must opt in with an actual JSON boolean.
    root_settings = json.loads(settings.read_bytes())
    root_settings["lan_access"] = "true"
    root_settings["remote_manage"] = "true"
    settings.write_text(json.dumps(root_settings), encoding="utf-8")

    def check_strict_bools(origin):
        cl = check_listener(origin, "127.0.0.1", saved=False)
        assert cl.get(NETWORK)[1]["remote_manage"] is False

    start(check_strict_bools)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(*(Path(arg).resolve() for arg in sys.argv[1:]), Path(tmp))
