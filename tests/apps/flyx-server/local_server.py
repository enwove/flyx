# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A test's own `flyx serve` (cpu_ref, the tiny model): the environment it runs in, its start and its end; and a
client of a server's HTTP API (Client, wait_telemetry, wait_load), which check_server.py, the checks of a server on a
real model run by hand, takes from here too.

The server reads FLYX_* settings from its environment (FLYX_API_KEY, FLYX_BACKEND_OPTIONS, FLYX_BACKEND_DIR,
FLYX_DRAFTER, ...: core/os/env.h), and its configuration, home and Hugging Face cache from the user's folders.  A
test's server takes none of that from the shell that started the test: what is tested is what the test gives it.
"""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from urllib.error import URLError
from urllib.request import urlopen

# Settings a server would take from the shell: the engine's and the server's own, and a Hugging Face token.
_SHELL_ONLY = ("FLYX_",)
_SHELL_ONLY_NAMES = ("HF_TOKEN",)


def _shell_setting(name):
    upper = name.upper()
    return upper.startswith(_SHELL_ONLY) or upper in _SHELL_ONLY_NAMES


def server_env(tmp, exe, **extra):
    """The environment for a server of the test's own: this process's without its FLYX_* settings, Hugging Face token
    or cache, with its configuration (APPDATA, XDG_CONFIG_HOME), home and caches (Hugging Face's among them) in
    `tmp`, the backends beside `exe`, and then `extra` (a test's own settings: HF_HUB_CACHE, FLYX_MODELS_DIR, ...)."""
    tmp = Path(tmp)
    env = {k: v for k, v in os.environ.items()
           if not _shell_setting(k) and k.upper() not in ("HF_HOME", "HF_HUB_CACHE", "HF_ENDPOINT", "HF_HUB_OFFLINE",
                                                           "XDG_CACHE_HOME")}
    env.update(APPDATA=str(tmp), XDG_CONFIG_HOME=str(tmp), HOME=str(tmp), XDG_CACHE_HOME=str(tmp / "cache"),
               HF_HOME=str(tmp / "huggingface"), FLYX_BACKEND_DIR=str(Path(exe).parent))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _without_inherited(env):
    """`env` less the FLYX_* settings and Hugging Face token it holds only as this process's environment holds them
    (an env made as dict(os.environ, ...)): those came from the shell, not the test."""
    return {k: v for k, v in env.items() if not (_shell_setting(k) and os.environ.get(k) == v)}


# The dashboard URL is connectable even when the listen address is a wildcard.
_LISTENING = re.compile(rb"listening on http://[^\s,]+, the dashboard at (http://[^\s/]+)/")


def serve(exe, cwd, env, model=(), on_ready=None, *, ctx=128, options=(), expect_name="test model", stderr=False):
    """Starts `exe serve` on `model` (cpu_ref, on a port the OS chooses: --port 0, read back from where the server
    says it listens), waits until it has loaded the model named `expect_name`, calls on_ready(origin), and ends it.
    What the server writes to stderr goes on to this process's stderr, or is returned where `stderr` is true."""
    command = [exe, "serve", *model, "--backend", "cpu_ref", "--ctx", str(ctx), "--port", "0",
               "--no-monitor", "--no-browser", *options]
    server = subprocess.Popen(command, cwd=cwd, env=_without_inherited(env), stderr=subprocess.PIPE)
    said, origin, listening = [], [], threading.Event()

    def read():
        for line in iter(server.stderr.readline, b""):
            said.append(line)
            if not stderr:
                sys.stderr.buffer.write(line)
                sys.stderr.flush()
            if not origin and (m := _LISTENING.search(line)):
                origin.append(m[1].decode())
                listening.set()
        listening.set()  # the server ended without saying where

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        deadline = time.monotonic() + 30
        listening.wait(30)
        if not origin:
            raise RuntimeError(f"server did not say where it listens (exit {server.poll()})")
        while server.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(origin[0] + "/health", timeout=1) as reply:
                    health = json.load(reply)
                if health.get("loaded"):
                    assert health["model"] == expect_name, health
                    break
                if health.get("load") == "failed":
                    raise RuntimeError(f"model not restored: {health}")
            except (URLError, TimeoutError):
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"server did not load the model (exit {server.poll()})")
        if on_ready:
            on_ready(origin[0])
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        reader.join(timeout=10)
    return b"".join(said) if stderr else None


class Client:
    """Requests to a server at `url` (with `key` as its bearer token, when there is one)."""

    def __init__(self, url: str, key: str | None):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.key = key

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=600)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get_raw(self, path, headers=None):
        c = self._conn()
        try:
            c.request("GET", path, headers={**self._headers(), **(headers or {})})
            r = c.getresponse()
            return r.status, r.headers, r.read()
        finally:
            c.close()

    def get(self, path):
        status, _, data = self.get_raw(path)
        return status, json.loads(data or b"null")

    def post(self, path, body, headers=None):
        c = self._conn()
        c.request("POST", path, body=json.dumps(body), headers={**self._headers(), **(headers or {})})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def delete(self, path):
        c = self._conn()
        c.request("DELETE", path, headers=self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def stream(self, path, body, stop_after=None):
        """Yields (event_name, data) of an SSE response; stop_after closes the connection early."""
        c = self._conn()
        c.request("POST", path, body=json.dumps(body), headers=self._headers())
        r = c.getresponse()
        if r.status != 200:
            yield "http_error", {"status": r.status, "body": r.read().decode()}
            return
        event, n = None, 0
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                yield event, (data if data == "[DONE]" else json.loads(data))
                event = None
                n += 1
                if stop_after and n >= stop_after:
                    c.close()
                    return


def wait_telemetry(cl, predicate, seconds):
    """GET /v1/telemetry?recent=1 until predicate holds of it, at most `seconds`; the last one read."""
    end = time.monotonic() + seconds
    while True:
        _, t = cl.get("/v1/telemetry?recent=1")
        if predicate(t) or time.monotonic() >= end:
            return t
        time.sleep(0.02)


def wait_load(cl: Client, timeout_s: float) -> dict:
    """GET /v1/load until no fetch or load is under way, at most timeout_s."""
    deadline = time.time() + timeout_s
    s = {}
    while time.time() < deadline:
        st, s = cl.get("/v1/load")
        if st == 200 and s.get("state") not in ("fetching", "loading"):
            return s
        time.sleep(1)
    return s
