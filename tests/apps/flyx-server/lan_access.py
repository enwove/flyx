# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the saved preference, real listener, dashboard and management policy across launches."""
import ipaddress
import json
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

    # Toggle remote_manage dynamically via POST /v1/settings/network and verify persistence + immediate effect.
    def toggle_remote_manage(origin):
        cl = Client(origin, None)
        fwd = {"X-Forwarded-For": "192.0.2.2"}
        assert cl.get_raw("/v1/logs", fwd)[0] == 403
        assert cl.get_raw("/v1/requests", fwd)[0] == 403
        for bad in ({"remote_manage": 1}, {"remote_manage": "true"}):
            assert cl.post(NETWORK, bad)[0] == 400
        status, net = cl.post(NETWORK, {"remote_manage": True})
        assert status == 200 and net["remote_manage"] is True, net
        # Immediately unlocks forwarded/LAN management without restart, while cross-site Origin stays blocked.
        assert cl.get_raw("/v1/logs", fwd)[0] == 200
        assert cl.get_raw("/v1/requests", fwd)[0] == 200
        assert cl.get_raw("/v1/hub/cached", fwd)[0] == 200
        assert cl.get_raw("/v1/logs", {"Origin": "https://example.org"})[0] == 403

    start(toggle_remote_manage, "--host", "127.0.0.1")
    # Persisted remote_manage survives restart, and disabling takes effect immediately.
    def persisted_remote_manage(origin):
        cl = Client(origin, None)
        fwd = {"X-Forwarded-For": "192.0.2.2"}
        assert cl.get(NETWORK)[1]["remote_manage"] is True
        assert cl.get_raw("/v1/logs", fwd)[0] == 200
        assert cl.post(NETWORK, {"remote_manage": False})[0] == 200
        assert cl.get_raw("/v1/logs", fwd)[0] == 403

    start(persisted_remote_manage, "--host", "127.0.0.1")

    # --no-auth unlocks LAN management (including logs, /v1/messages request log, and --models-dir passed before MODEL),
    # ignores FLYX_API_KEY in the environment, locks remote_manage (409), and still blocks cross-site Origin (403).
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
        assert cl.get_raw("/v1/logs", fwd)[0] == 200
        status, _, raw_cached = cl.get_raw("/v1/hub/cached", fwd)
        cached = json.loads(raw_cached or b"null")
        assert status == 200, cached
        assert any(Path(d).resolve() == models_dir.resolve() for d in cached["dirs"]), cached["dirs"]
        assert any(Path(m["path"]).resolve() == nested.resolve() for m in cached["models"] if m.get("path")), cached["models"]
        # Send an Anthropic /v1/messages request and verify it is visible in /v1/requests over LAN.
        msg_status, _ = cl.post("/v1/messages", {
            "model": "test model", "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
        }, fwd)
        assert msg_status == 200, msg_status
        req_status, _, raw_reqs = cl.get_raw("/v1/requests", fwd)
        reqs = json.loads(raw_reqs or b"null")
        assert req_status == 200 and reqs["enabled"] is True, reqs
        assert any(r["api"] == "messages" for r in reqs["requests"]), reqs["requests"]

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
