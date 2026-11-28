# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Cold Hub load through the CLI. Record which candidates were probed and downloaded."""
import hashlib
from ctypes.util import find_library
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


exe, fixture_dir = sys.argv[1:]
if os.name != "nt" and not find_library("curl"):
    sys.exit(77)
assets = Path(fixture_dir)
repo = "acme/Ink-GGUF"
commit = "a" * 40
wrong = "dspark-Ink-Q8_0.gguf"
incompatible = "dspark-Ink-Q8_0_A.gguf"
valid = "dspark-Ink-Q8_0_B.gguf"
unused = "dspark-Ink-Q8_0_C.gguf"
files = {
    "Ink-Q4_K_M.gguf": (assets / "tiny_inkling2_bf16.gguf").read_bytes(),
    wrong: (assets / "dflash2-tiny-bf16.gguf").read_bytes(),
    incompatible: (assets / "dspark_excess_heads.gguf").read_bytes(),
    valid: (assets / "dspark-tiny-bf16.gguf").read_bytes(),
    unused: (assets / "dspark-tiny-bf16.gguf").read_bytes(),
}
listing = json.dumps({"sha": commit, "siblings": [
    {"rfilename": name, "size": len(data), "lfs": {"sha256": hashlib.sha256(data).hexdigest()}}
    for name, data in files.items()]}).encode()
requests = []


class Hub(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path.split("?")[0] == f"/api/models/{repo}/revision/main":
            self.send_response(200)
            body = listing
        elif self.path.startswith(f"/{repo}/resolve/{commit}/") and self.path.rsplit("/", 1)[-1] in files:
            name = self.path.rsplit("/", 1)[-1]
            data = files[name]
            first, last = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((name, first, last))
            # A rejected metadata range is inconclusive. The actual download still gets local validation.
            if name == incompatible and 0 < last < len(data) - 1:
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
            body = data[first:last + 1]
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


with tempfile.TemporaryDirectory() as tmp, ThreadingHTTPServer(("127.0.0.1", 0), Hub) as server:
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        env = {**os.environ, "HF_HUB_CACHE": tmp, "HF_HOME": tmp, "HF_HUB_OFFLINE": "", "HF_TOKEN": "",
               "HF_ENDPOINT": f"http://127.0.0.1:{server.server_port}", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}
        p = subprocess.run([exe, "bench", repo + ":Q4_K_M", "--backend", "cpu_ref", "-n", "2", "--json"],
                           env=env, capture_output=True, text=True, timeout=60)
        assert p.returncode == 0 and json.loads(p.stdout)["drafter"] == "dspark", p.stderr
        wrong_ranges = [(first, last) for name, first, last in requests if name == wrong]
        assert len(wrong_ranges) == 1 and wrong_ranges[0][0] == 0 and 0 < wrong_ranges[0][1] < len(files[wrong]) - 1, requests
        assert (incompatible, 0, len(files[incompatible]) - 1) in requests, requests
        assert (valid, 0, len(files[valid]) - 1) in requests, requests
        assert all(name != unused for name, _, _ in requests), requests
    finally:
        server.shutdown()
        thread.join()
