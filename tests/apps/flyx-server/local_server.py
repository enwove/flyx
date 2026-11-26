# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A test's own `flyx serve` (cpu_ref, the tiny model): the environment it runs in, its start and its end.

The server reads FLYX_* settings from its environment (FLYX_API_KEY, FLYX_BACKEND_OPTIONS, FLYX_BACKEND_DIR,
FLYX_DRAFTER, ...: core/os/env.h), and its configuration, home and Hugging Face cache from the user's folders.  A
test's server takes none of that from the shell that started the test: what is tested is what the test gives it.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import time
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


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(exe, cwd, env, model=(), on_ready=None, *, ctx=128, options=(), expect_name="test model", stderr=False):
    """Starts `exe serve` on `model` (cpu_ref, a port of its own), waits until it has loaded the model named
    `expect_name`, calls on_ready(origin), and ends it.  Returns what it wrote to stderr where `stderr` is true."""
    port = free_port()
    command = [exe, "serve", *model, "--backend", "cpu_ref", "--ctx", str(ctx), "--port", str(port),
               "--no-monitor", "--no-browser", *options]
    server = subprocess.Popen(command, cwd=cwd, env=_without_inherited(env),
                              stderr=subprocess.PIPE if stderr else None)
    try:
        deadline = time.monotonic() + 30
        while server.poll() is None and time.monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as reply:
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
            on_ready(f"http://127.0.0.1:{port}")
    finally:
        server.terminate()
        try:
            said = server.communicate(timeout=10)[1]
        except subprocess.TimeoutExpired:
            server.kill()
            said = server.communicate()[1]
    return said
