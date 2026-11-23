# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Independent flyx processes fetching from one loopback Hub into a temporary cache."""
from collections import Counter
from contextlib import contextmanager
from ctypes.util import find_library
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


PIECE = 32 << 20


@contextmanager
def storage(files, blocked, block_at, etag):
    entered, release = threading.Event(), threading.Event()
    counts = Counter()
    mutex = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            name = self.path.rsplit("/", 1)[-1]
            data = files.get(name)
            if data is None:
                self.send_error(404)
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            with mutex:
                counts[name, start, end] += 1
                first = name == blocked and start == block_at and counts[name, start, end] == 1
            if first:
                entered.set()
                if not release.wait(20):
                    self.send_error(500)
                    return
            self.send_response(206)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            if etag:
                self.send_header("ETag", f'"{hashlib.sha256(data).hexdigest()}"')
            try:
                self.end_headers()
                self.wfile.write(memoryview(data)[start:end + 1])
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the crash test terminates the first client

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered, release, counts
    finally:
        release.set()
        server.shutdown()
        thread.join()
        server.server_close()


def run_case(exe, cache, lfs, owner_action):
    crash = owner_action == "crash"
    shared = b"w" * (PIECE + 15) if crash else b"fixture weights" * 128
    files = {"weights.gguf": shared, "renamed.gguf": shared, "other.gguf": b"other file"}
    rev1, rev2 = "a" * 40, "b" * 40
    children = []
    with storage(files, "weights.gguf", PIECE if crash else 0, lfs) as (endpoint, entered, release, counts):
        env = {**os.environ, "HF_ENDPOINT": endpoint, "HF_HUB_CACHE": str(cache), "HF_HOME": str(cache.parent),
               "HF_TOKEN": "", "HF_HUB_OFFLINE": "", "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}

        def fetch(commit, name, cancel_ms=0):
            data = files[name]
            digest = (hashlib.sha256(data).hexdigest() if lfs else
                      hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest())
            p = subprocess.Popen([exe, "--fetch", commit, name, digest, str(len(data)), str(cancel_ms)], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            children.append(p)
            return p

        def finish(p, stopped=False):
            out, _ = p.communicate(timeout=8)
            assert p.returncode == int(stopped), (p.returncode, out)
            if stopped:
                assert "stopped" in out, out

        try:
            owner = fetch(rev1, "weights.gguf", cancel_ms=-1 if owner_action == "cancel" else 0)
            assert entered.wait(8), "owner never requested the blob"
            waiter = fetch(rev2, "renamed.gguf")
            last = len(shared) - 1
            expected = {("weights.gguf", 0, last): 1}
            if crash:
                # Wait until the first piece is recorded on disk. The second response stays blocked.
                blob_id = hashlib.sha256(shared).hexdigest()
                bitmap = cache / "models--acme--parallel" / "blobs" / (blob_id + ".flyx-part.map")
                until = time.monotonic() + 8
                while bitmap.read_bytes() != bytes([1, 0]):
                    assert time.monotonic() < until, "first piece never finished"
                    time.sleep(0.02)
                owner.kill()
                owner.wait(timeout=5)
                finish(waiter)
                expected = {("weights.gguf", 0, 0): 1, ("weights.gguf", 0, PIECE - 1): 1,
                            ("weights.gguf", PIECE, last): 1, ("renamed.gguf", 0, 0): 1,
                            ("renamed.gguf", PIECE, last): 1}
            elif owner_action == "cancel":
                release.set()
                finish(owner, stopped=True)
                finish(waiter)  # checksum and publish the completed partial without a second GET
            else:
                same = fetch(rev1, "weights.gguf")
                cancelled = fetch(rev2, "weights.gguf", cancel_ms=250)
                finish(fetch(rev1, "other.gguf"))  # another blob progresses while the owner is blocked
                finish(cancelled, stopped=True)
                release.set()
                for p in (owner, same, waiter):
                    finish(p)
                expected["other.gguf", 0, len(files["other.gguf"]) - 1] = 1
            assert counts == expected, counts
            snap = cache / "models--acme--parallel" / "snapshots"
            assert (snap / rev2 / "renamed.gguf").read_bytes() == shared
            if owner_action == "finish":
                assert (snap / rev1 / "weights.gguf").read_bytes() == shared
                assert (snap / rev1 / "other.gguf").read_bytes() == files["other.gguf"]
                assert not (snap / rev2 / "weights.gguf").exists()
            else:
                assert not (snap / rev1 / "weights.gguf").exists()
        finally:
            release.set()
            for p in children:
                if p.poll() is None:
                    p.kill()
                    p.wait(timeout=5)


if __name__ == "__main__":
    if os.name != "nt" and not find_library("curl"):
        sys.exit(77)
    with tempfile.TemporaryDirectory(prefix="flyx-hub-concurrent-") as tmp:
        for case, lfs, action in (("lfs", True, "finish"), ("git", False, "finish"),
                                  ("crash", True, "crash"), ("cancel", True, "cancel")):
            run_case(sys.argv[1], Path(tmp) / case, lfs, action)
    print("PASS shared blobs across processes, independent transfers, cancellation and owner exit")
