# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""What a hostile or broken client sends `flyx serve`, and what a browser brings: none of it ends the process, fills its
memory, holds its workers or reads what it should not.  One server, the checks below against it in turn; then the
servers --cors makes, each with its own.

    hardening.py FLYX_EXE TINY_MODEL
"""
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time

from local_server import Client, serve, server_env

DEEP = "[" * 200000 + "]" * 200000
ME = [{"role": "user", "content": "x"}]
EXE = MODEL = ENV = TMP = None


def port_of(url):
    return int(url.rsplit(":", 1)[1])


def exchange(port, payload, wait=10.0):
    """Sends `payload` and returns what comes back until the peer closes or `wait` passes with nothing: (bytes, how it
    ended: "closed", "reset" or "silent")."""
    got, how = b"", "closed"
    with socket.create_connection(("127.0.0.1", port), timeout=wait) as s:
        try:
            s.sendall(payload)
            while True:
                d = s.recv(65536)
                if not d:
                    break
                got += d
        except socket.timeout:
            how = "silent"
        except OSError:
            how = "reset"
    return got, how


def status_of(reply):
    m = re.match(rb"HTTP/1\.1 (\d+)", reply)
    return int(m.group(1)) if m else None


def alive(cl, what):
    st, h = cl.get("/health")
    assert st == 200 and h["loaded"], (what, st, h)


def check_depth(url):
    """JSON nested past the limit is refused where the body is read, wherever a request carries JSON of the client's
    own; nested 200,000 deep it once overflowed the stack of whatever copied it."""
    cl, port = Client(url, None), port_of(url)
    host = b"Host: 127.0.0.1:%d\r\n" % port
    deep_bodies = {
        "chat tools": ("/v1/chat/completions", '{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"a":%s}}}]}'),
        "chat_template_kwargs": ("/v1/chat/completions", '{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"a":%s}}'),
        "responses metadata": ("/v1/responses", '{"max_output_tokens":1,"input":"x","metadata":{"a":%s}}'),
        "messages tools": ("/v1/messages", '{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"tools":[{"name":"f","input_schema":{"a":%s}}]}'),
        "load": ("/v1/load", '{"model":"x","a":%s}'),
    }
    for what, (path, body) in deep_bodies.items():
        reply, how = exchange(port, b"POST " + path.encode() + b" HTTP/1.1\r\n" + host + b"Content-Type: application/json\r\nConnection: close\r\n"
                              b"Content-Length: %d\r\n\r\n" % len(body % DEEP) + (body % DEEP).encode())
        assert status_of(reply) == 400, (what, how, reply[:200])
        if path != "/v1/load":
            assert b"nested more than 64 levels" in reply, (what, reply[-300:])
        alive(cl, what)
    # 64 levels are taken (a schema of twenty nested objects is about that deep); the body is then judged as any other.
    ok = '{"max_tokens":1,"messages":[{"role":"user","content":"x"}],"metadata":{"a":' + "[" * 61 + "]" * 61 + "}}"
    st, r = cl.post("/v1/chat/completions", json.loads(ok))
    assert st == 200, (st, r)
    # Arguments of an earlier tool call written as JSON text, as OpenAI's clients write them, are parsed again: the
    # same limit, told as the request's error.
    text = '{"a":' + DEEP + "}"
    for path, body in [("/v1/chat/completions", {"max_tokens": 1, "messages": ME + [{"role": "assistant", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": text}}]},
                                                                                     {"role": "tool", "tool_call_id": "a", "content": "1"}]}),
                       ("/v1/responses", {"max_output_tokens": 1, "input": [{"role": "user", "content": "x"}, {"type": "function_call", "call_id": "a", "name": "f", "arguments": text},
                                                                             {"type": "function_call_output", "call_id": "a", "output": "1"}]}),
                       ("/v1/messages", {"max_tokens": 1, "messages": ME + [{"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "f", "input": text}]},
                                                                           {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "1"}]}]})]:
        st, r = cl.post(path, body)
        assert st == 400 and "nested more than 64 levels" in json.dumps(r), (path, st, r)
        alive(cl, path + " arguments as text")


def check_limits(url):
    """A chunked body is held to the size limit a Content-Length is; a line without an end is cut; a target over the
    limit is answered at once."""
    cl, port = Client(url, None), port_of(url)
    host = b"Host: 127.0.0.1:%d\r\n" % port
    # A chunked body past the limit (64 MiB) is refused when it passes it, not read to its end.
    head = b"POST /v1/chat/completions HTTP/1.1\r\n" + host + b"Content-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"
    chunk = b"100000\r\n" + b" " * (1 << 20) + b"\r\n"
    reply, how = exchange(port, head + chunk * 65, wait=30)
    assert status_of(reply) == 413, (how, reply[:200])
    alive(cl, "a chunked body past the limit")
    # A request line with no end is cut at 64 KiB: the connection closes, nothing is kept.
    t0 = time.monotonic()
    reply, how = exchange(port, b"GET /" + b"a" * (100 << 10), wait=10)
    assert how != "silent" and reply == b"" and time.monotonic() - t0 < 8, (how, reply[:100])
    # A target over the limit is answered at once (upstream waited for headers a second time, until the read timeout).
    t0 = time.monotonic()
    reply, how = exchange(port, b"GET /" + b"a" * 9000 + b" HTTP/1.1\r\n" + host + b"\r\n", wait=10)
    assert status_of(reply) == 414 and time.monotonic() - t0 < 5, (how, reply[:100], time.monotonic() - t0)
    alive(cl, "a long target")


def check_workers_and_port(url):
    """Connections that send nothing leave workers for the others (16 of them once took the server from everyone);
    a second server does not take a port in use."""
    cl, port = Client(url, None), port_of(url)
    stalled = []
    try:
        for _ in range(24):
            s = socket.create_connection(("127.0.0.1", port), timeout=5)
            s.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n" % port)  # the headers never end
            stalled.append(s)
        time.sleep(0.3)
        t0 = time.monotonic()
        alive(cl, "stalled connections")
        assert time.monotonic() - t0 < 3, time.monotonic() - t0
    finally:
        for s in stalled:
            s.close()
    second = subprocess.run([EXE, "serve", MODEL, "--backend", "cpu_ref", "--ctx", "128", "--port", str(port), "--no-monitor", "--no-browser"],
                            cwd=TMP, env=ENV, capture_output=True, timeout=60)
    assert second.returncode == 1 and b"cannot listen on" in second.stderr, (second.returncode, second.stderr[-300:])
    alive(cl, "a second server on the port")


def check_browser(url):
    """A link to the dashboard followed from another site opens it, while another site's script reads nothing; a
    request a proxy passed on does not manage the server without a key."""
    cl = Client(url, None)
    nav = {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document", "Accept": "text/html"}
    for path in ("/", "/dashboard", "/dashboard/"):
        st, _, body = cl.get_raw(path, {**nav, "Sec-Fetch-Site": "cross-site"})
        assert st == 200 and body.lstrip().lower().startswith(b"<!doctype html"), (path, st, body[:80])
    for path, headers in [("/v1/telemetry", {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Origin": "https://example.com"}),
                          ("/v1/models", {**nav, "Sec-Fetch-Site": "cross-site"}),
                          ("/", {**nav, "Sec-Fetch-Site": "cross-site", "Host": "example.com"})]:
        st, _, body = cl.get_raw(path, headers)
        assert st == 403, (path, headers, st, body[:120])
    # What a proxy on this machine passed on comes from whoever reached the proxy: not this machine's to manage.
    st, _, _ = cl.get_raw("/v1/requests")
    assert st == 200, st
    for h in ("X-Forwarded-For", "Forwarded", "X-Real-IP", "Via"):
        st, _, body = cl.get_raw("/v1/requests", {h: "203.0.113.7"})
        assert st == 403, (h, st, body[:120])
        st, r = cl.post("/v1/unload", {}, {h: "203.0.113.7"})
        assert st == 403, (h, st, r)
    st, r = cl.post("/v1/chat/completions", {"max_tokens": 1, "messages": ME}, {"X-Forwarded-For": "203.0.113.7"})
    assert st == 200, (st, r)  # inference stays what it was: open without a key


def check_statuses(url):
    """Invalid launch options and schemas return a client error."""
    cl = Client(url, None)
    # Reject invalid launch options before opening a desktop dialog or window on any platform.
    for agent, mode in (("claude", "unknown"), ("claude", None), ("claude", ""), ("codex", "acceptEdits")):
        status, result = cl.post("/v1/agents/launch", {
            "agent": agent, "permission_mode": mode, "dir": str(Path(TMP) / "missing-project")})
        assert status == 400 and "permission_mode" in result["error"]["message"], (agent, mode, status, result)
    for path, body in [("/v1/chat/completions", {"max_tokens": 4, "messages": ME, "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": 5}}}}),
                       ("/v1/chat/completions", {"max_tokens": 4, "messages": ME, "stream": True, "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": {"$ref": "#/$defs/nope"}}}}),
                       ("/v1/responses", {"max_output_tokens": 4, "input": "x", "text": {"format": {"type": "json_schema", "name": "a", "schema": {"enum": []}}}}),
                       ("/v1/messages", {"max_tokens": 4, "messages": ME, "output_config": {"format": {"type": "json_schema", "schema": {"type": "array", "minItems": 5, "maxItems": 2}}}})]:
        st, r = cl.post(path, body)
        text = json.dumps(r)
        assert st == 400 and "invalid_request_error" in text and "reply grammar: " in text, (path, st, r)
        assert not re.search(r"third_party|\.cc:\d|\[\d\d:\d\d:\d\d\]", text), (path, text)
    alive(cl, "schemas no grammar is made from")


def call(cl, method, path, headers, body=None):
    """One request, its answer's status, headers and body (the pages of a browser are played by their headers)."""
    c = cl._conn()
    try:
        c.request(method, path, body=None if body is None else json.dumps(body), headers={**cl._headers(), **headers})
        r = c.getresponse()
        return r.status, r.headers, r.read()
    finally:
        c.close()


def preflight(cl, path, origin, more=None):
    """What a browser asks before a page of `origin` may send a JSON body or a key to `path`."""
    return call(cl, "OPTIONS", path, {"Origin": origin, "Access-Control-Request-Method": "POST",
                                      "Access-Control-Request-Headers": "authorization,content-type", **(more or {})})


def lets_in(cl, origin, what=""):
    """A page of `origin` may call the inference API: its preflight is answered, its request's answer readable."""
    st, h, _ = preflight(cl, "/v1/chat/completions", origin, {"Access-Control-Request-Private-Network": "true"})
    assert st == 204 and h.get("Access-Control-Allow-Origin") == origin, (what, origin, st, dict(h))
    assert "POST" in h.get("Access-Control-Allow-Methods", "") and h.get("Access-Control-Allow-Headers") == "authorization,content-type", dict(h)
    assert h.get("Access-Control-Allow-Private-Network") == "true" and h.get("Access-Control-Max-Age"), dict(h)
    for stream in (False, True):
        st, h, body = call(cl, "POST", "/v1/chat/completions", {"Origin": origin, "Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors"},
                           {"max_tokens": 2, "messages": ME, "stream": stream})
        assert st == 200 and h.get("Access-Control-Allow-Origin") == origin and "Origin" in h.get("Vary", ""), (what, origin, stream, st, dict(h))
        assert body, (what, origin, stream)
    st, h, _ = call(cl, "GET", "/v1/models", {"Origin": origin})
    assert st == 200 and h.get("Access-Control-Allow-Origin") == origin, (what, origin, st, dict(h))


def keeps_out(cl, origin, what=""):
    """A page of `origin` may not: no leave in the preflight's answer, the request itself refused, and told why."""
    st, h, body = preflight(cl, "/v1/chat/completions", origin)
    assert st == 403 and "Access-Control-Allow-Origin" not in h, (what, origin, st, dict(h))
    if origin != "null":
        assert b"--cors " + origin.lower().encode() in body.lower(), (what, origin, body)
    st, h, _ = call(cl, "POST", "/v1/chat/completions", {"Origin": origin, "Sec-Fetch-Site": "cross-site"}, {"max_tokens": 1, "messages": ME})
    assert st == 403 and "Access-Control-Allow-Origin" not in h, (what, origin, st, dict(h))


def panel_stays_closed(cl, origin):
    """The model panel, the request log, the logs and the telemetry are no other origin's, allowed for inference or not."""
    for path in ("/v1/load", "/v1/unload", "/v1/requests", "/v1/logs", "/v1/telemetry", "/v1/hub/dir"):
        st, h, _ = preflight(cl, path, origin)
        assert st == 403 and "Access-Control-Allow-Origin" not in h, (path, origin, st, dict(h))
        st, h, _ = call(cl, "GET", path, {"Origin": origin, "Sec-Fetch-Site": "same-site"})
        assert st == 403 and "Access-Control-Allow-Origin" not in h, (path, origin, st, dict(h))
    # A simple CORS-safelisted POST (text/plain, no preflight) from another localhost port must not invoke management routes.
    st, h, _ = call(cl, "POST", "/v1/unload",
                    {"Origin": origin, "Sec-Fetch-Site": "same-site", "Content-Type": "text/plain"}, {})
    assert st == 403 and "Access-Control-Allow-Origin" not in h, ("/v1/unload", origin, st, dict(h))


def check_cors(url):
    """By default the pages of this machine (another port of localhost) may call the inference API from a browser, and
    no other site's; without an Origin (an SDK, curl) nothing changes."""
    cl = Client(url, None)
    for origin in ("http://localhost:3000", "http://127.0.0.1:5173", "http://[::1]:8000", "https://localhost"):
        lets_in(cl, origin, "this machine's pages")
        panel_stays_closed(cl, origin)
    # Same-origin dashboard requests (Origin authority matches Host) still work.
    st, _, _ = call(cl, "GET", "/v1/requests", {"Origin": url, "Sec-Fetch-Site": "same-origin"})
    assert st == 200, (url, st)
    for origin in ("https://chat.example.com", "http://localhost.example.com", "http://[::1]evil.com",
                   "http://localhost:evil", "chrome-extension://abcdefghijklmnop", "null"):
        keeps_out(cl, origin, "another site")
    st, h, _ = call(cl, "POST", "/v1/chat/completions", {}, {"max_tokens": 1, "messages": ME})
    assert st == 200 and "Access-Control-Allow-Origin" not in h, (st, dict(h))
    st, h, _ = call(cl, "OPTIONS", "/v1/chat/completions", {})  # not a preflight: as before
    assert st == 404, st


def check_cors_named(url):
    """--cors names the origins let in; none takes this machine's own pages out."""
    cl = Client(url, None)
    for origin in ("https://chat.example.com", "HTTPS://Chat.Example.com", "chrome-extension://abcdefghijklmnop", "null"):
        lets_in(cl, origin, "named")
    panel_stays_closed(cl, "https://chat.example.com")
    for origin in ("https://other.example.com", "https://chat.example.com:8443", "http://localhost:3000"):
        keeps_out(cl, origin, "not named")
    # Another localhost port sends Sec-Fetch-Site: same-site in a real browser and must still be refused when not named.
    st, h, _ = call(cl, "POST", "/v1/chat/completions",
                    {"Origin": "http://localhost:3000", "Sec-Fetch-Site": "same-site", "Content-Type": "text/plain"},
                    {"max_tokens": 1, "messages": ME})
    assert st == 403 and "Access-Control-Allow-Origin" not in h, (st, dict(h))
    alive(cl, "named origins")


def check_cors_any(url):
    """--cors '*' lets any page call the inference API, and still nothing else."""
    cl = Client(url, None)
    for origin in ("https://anything.example", "http://localhost:3000", "null"):
        lets_in(cl, origin, "any")
        panel_stays_closed(cl, origin)


def check_cors_keyed(url):
    """With a key the preflight (which carries none) is still answered, and a refusal for the key is readable."""
    origin = "https://chat.example.com"
    lets_in(Client(url, "secret-key"), origin, "with the key")
    st, h, _ = preflight(Client(url, None), "/v1/chat/completions", origin)
    assert st == 204 and h.get("Access-Control-Allow-Origin") == origin, (st, dict(h))
    st, h, _ = call(Client(url, None), "POST", "/v1/chat/completions", {"Origin": origin}, {"max_tokens": 1, "messages": ME})
    assert st == 401 and h.get("Access-Control-Allow-Origin") == origin, (st, dict(h))
    st, h, _ = preflight(Client(url, "secret-key"), "/v1/chat/completions", "https://other.example.com")
    assert st == 403 and "Access-Control-Allow-Origin" not in h, (st, dict(h))


def check_cors_flag():
    """A value that is no origin is refused at the command line; '*' without a key is said aloud at the start."""
    bad = subprocess.run([EXE, "serve", MODEL, "--backend", "cpu_ref", "--cors", "https://chat.example.com/app"],
                         cwd=TMP, env=ENV, capture_output=True, timeout=60)
    assert bad.returncode == 2 and b"--cors https://chat.example.com/app: an origin" in bad.stderr, (bad.returncode, bad.stderr[-300:])
    said = serve(EXE, TMP, ENV, [MODEL], ctx=128, options=["--cors", "*"], expect_name="tiny", stderr=True)
    assert b"--cors '*' without --api-key" in said, said[-400:]


CHECKS = [check_depth, check_limits, check_workers_and_port, check_browser, check_statuses, check_cors]


def check_all(url):
    for check in CHECKS:
        check(url)


if __name__ == "__main__":
    EXE, MODEL = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as TMP:
        ENV = server_env(TMP, EXE)
        serve(EXE, TMP, ENV, [MODEL], check_all, ctx=128, options=["--no-spec"], expect_name="tiny")
        named = ["--cors", "https://chat.example.com", "--cors", "chrome-extension://abcdefghijklmnop", "--cors", "null", "--cors", "none"]
        for options, check in [(named, check_cors_named), (["--cors", "*"], check_cors_any),
                               (["--api-key", "secret-key", "--cors", "https://chat.example.com"], check_cors_keyed)]:
            serve(EXE, TMP, ENV, [MODEL], check, ctx=128, options=["--no-spec", *options], expect_name="tiny")
        check_cors_flag()
    print("hardening: ok")
