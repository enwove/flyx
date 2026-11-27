# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Serve a relative model, then resume it from another working directory."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from local_server import serve, server_env  # serve: also this module's, for the tests that import it from here


def slow_settings(origin, settings):
    # A pipe lets us pause a status read until another handler has finished using the loaded model.
    saved = settings.read_bytes()
    settings.unlink()
    os.mkfifo(settings)

    def status():
        with urlopen(origin + "/v1/load", timeout=15) as reply:
            return json.load(reply)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(status)
            deadline = time.monotonic() + 5
            while True:
                try:
                    writer = os.open(settings, os.O_WRONLY | os.O_NONBLOCK)
                    break
                except OSError as exc:
                    if exc.errno != errno.ENXIO or pending.done() or time.monotonic() > deadline:
                        raise
                    time.sleep(0.01)
            with os.fdopen(writer, "wb") as output:
                try:
                    request = Request(origin + "/v1/completions", data=b'{"prompt":"Hello","max_tokens":1}',
                                      headers={"Content-Type": "application/json"})
                    with urlopen(request, timeout=5) as reply:
                        assert json.load(reply)["usage"]["completion_tokens"] == 1
                finally:
                    output.write(saved)
            assert pending.result()["remembered"] == json.loads(saved)["last"]
    finally:
        settings.unlink()
        settings.write_bytes(saved)


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / "settings"
        env = server_env(config, exe)
        model = root / "models with spaces" / "test model.FLYX"
        model.parent.mkdir()
        shutil.copyfile(asset, model)
        settings = config / "flyx" / "serve.json"
        # On Windows, C:file is relative to that drive's working directory.
        check_status = (lambda origin: slow_settings(origin, settings)) if hasattr(os, "mkfifo") else None
        serve(exe, model.parent, env, [model.drive + model.name], on_ready=check_status)
        saved = json.loads(settings.read_text(encoding="utf-8"))
        assert Path(saved["last"]["model"]) == model, saved
        serve(exe, config, env)
