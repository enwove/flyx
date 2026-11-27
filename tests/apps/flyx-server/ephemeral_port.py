# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Start on an OS-assigned port and reach /health at the advertised dashboard URL."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen


exe = Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, FLYX_BACKEND_DIR=str(exe.parent))
    log = Path(tmp) / "stderr.log"
    command = [exe, "serve", "--backend", "cpu_ref", "--port", "0", "--no-monitor", "--no-browser"]
    with log.open("wb") as stderr:
        server = subprocess.Popen(command, env=env, stderr=stderr, stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while True:
            output = log.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"listening on (http://127\.0\.0\.1:(\d+)), the dashboard at \1/\s", output)
            if match:
                break
            if server.poll() is not None or time.monotonic() >= deadline:
                raise AssertionError(f"missing listening and dashboard URLs in server log:\n{output}")
            time.sleep(0.05)
        assert 0 < int(match[2]) <= 65535, output
        with urlopen(match[1] + "/health", timeout=5) as reply:
            assert reply.status == 200 and json.load(reply)["status"] == "ok"
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
