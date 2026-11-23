# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Run the packaged CLI and HTTP server with a tiny test model."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def check_server(exe, model, env, work, port, backend):
    if port is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    key = "smoke%25&+"
    args = [exe, "serve", model, "--backend", backend, "--host", "127.0.0.1",
            "--port", str(port), "--ctx", "512", "--no-monitor", "--no-browser", "--api-key", key]
    log = work / "server.log"
    with log.open("wb") as output:
        server = subprocess.Popen(args, env=env, cwd=work, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"server exited with code {server.returncode}")
                try:
                    with http.open(origin + "/health", timeout=2) as response:
                        health = json.load(response)
                    telemetry = urllib.request.Request(origin + "/v1/telemetry?recent=0", headers={"x-api-key": key})
                    with http.open(telemetry, timeout=2) as response:
                        state = json.load(response)
                    if state["server"]["pid"] != server.pid:
                        raise RuntimeError(f"{origin} belongs to another process")
                    if state["server"].get("api_key_required") is not True:
                        raise RuntimeError("server did not report its configured API key requirement")
                    if health["status"] == "ok" and health["loaded"]:
                        if state["model"]["backend"] != backend:
                            raise RuntimeError(f"expected {backend}, got {state['model']['backend']}")
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for the packaged server")
                time.sleep(0.2)
            print("  ok   server /health")
            request = urllib.request.Request(origin + "/v1/completions",
                                             headers={"Content-Type": "application/json"},
                                             data=b'{"prompt":"hello","max_tokens":4,"temperature":0}')
            try:
                with http.open(request, timeout=30):
                    raise RuntimeError("server accepted a completion without its API key")
            except urllib.error.HTTPError as error:
                if error.code != 401:
                    raise
            request.add_header("Authorization", "Bearer " + key)
            with http.open(request, timeout=30) as response:
                if not json.load(response)["choices"]:
                    raise RuntimeError("server returned no choices")
            print(f"  ok   {backend} server completion and API key")
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exe", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--backend", choices=("cpu_ref", "cuda"),
                        help="require this backend for both CLI and HTTP (default: require CPU and also try CUDA)")
    parser.add_argument("--no-binding", action="store_true",
                        help="skip packaged Python binding check (for standalone single-file executables)")
    args = parser.parse_args()
    exe, model = args.exe.resolve(strict=True), args.model.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="flyx-package-") as tmp:
        work = Path(tmp)
        env = {key: value for key, value in os.environ.items() if not key.upper().startswith("FLYX_")}
        env.update(APPDATA=tmp, XDG_CONFIG_HOME=tmp)
        if not args.no_binding:
            subprocess.run([sys.executable, "-I", Path(__file__).with_name("smoke_binding.py"),
                            exe.parent.parent / "python", model], cwd=work, env=env, check=True, timeout=60)
        run = [exe, "run", model, "-p", "hello", "-n", "8", "--backend"]
        backend = args.backend or "cpu_ref"
        subprocess.run([*run, backend], cwd=work, env=env, check=True, timeout=60, stdout=subprocess.DEVNULL)
        print(f"  ok   CLI, {backend} backend")
        if args.backend is None:
            cuda = subprocess.run([*run, "cuda"], cwd=work, env=env, timeout=60, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            print("  ok   CLI, CUDA backend" if not cuda.returncode else "  --   CUDA backend not usable here")
        try:
            check_server(exe, model, env, work, args.port, backend)
        except Exception:
            log = work / "server.log"
            if log.exists():
                print(log.read_text(encoding="utf-8", errors="replace").rstrip())
            raise


if __name__ == "__main__":
    main()
