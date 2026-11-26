# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A LoRA adapter from the dashboard's way in: listed among a models directory's files (kind "lora"), loaded with
POST /v1/load's "lora" beside its model, shown in GET /v1/load, remembered for the next start, and refused with its
reason (no such folder, a DoRA adapter) without taking the server down; GET /v1/hub/loras and /v1/hub/lora (the Hub's
adapters) where the Hub cannot be asked.

    lora_load.py FLYX MODEL.gguf PEFT_DIR
"""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

FAILS = 0


def check(ok, what, detail=""):
    global FAILS
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"  [{detail}]" if detail and not ok else ""))
    FAILS += 0 if ok else 1


def call(origin, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(origin + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as reply:
            return reply.status, json.load(reply)
    except HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def settle(origin, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _, s = call(origin, "/v1/load")
        if s.get("state") in ("ready", "failed", "idle", None) and "elapsed_s" not in s:
            return s
        time.sleep(0.1)
    raise RuntimeError("the load did not end")


def reply(origin):
    _, r = call(origin, "/v1/completions", {"prompt": "The cat", "max_tokens": 12, "temperature": 0})
    return r["choices"][0]["text"]


class Server:
    def __init__(self, exe, env, args):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.proc = subprocess.Popen([exe, "serve", *args, "--backend", "cpu_ref", "--ctx", "128", "--port",
                                      str(self.port), "--no-monitor", "--no-browser"], env=env)
        self.origin = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                with urlopen(self.origin + "/health", timeout=1):
                    return
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.1)
        raise RuntimeError("the server did not start")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def main():
    exe, model, peft = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    tmp = Path(tempfile.mkdtemp(prefix="flyx-lora-load-"))
    try:
        models = tmp / "models"
        models.mkdir()
        shutil.copy(model, models / model.name)
        adapter = models / "my-adapter"
        shutil.copytree(peft, adapter)
        dora = tmp / "dora"
        shutil.copytree(peft, dora)
        cfg = json.loads((dora / "adapter_config.json").read_text())
        cfg["use_dora"] = True
        (dora / "adapter_config.json").write_text(json.dumps(cfg))
        env = {"XDG_CONFIG_HOME": str(tmp / "config"), "APPDATA": str(tmp / "config"), "HF_HOME": str(tmp / "hf"),
               "HF_HUB_OFFLINE": "1", "HOME": str(tmp), "PATH": "/usr/bin:/bin"}
        gguf = str(models / model.name)
        with Server(exe, env, ["--models-dir", str(models)]) as s:
            _, cached = call(s.origin, "/v1/hub/cached")
            loras = [m for m in cached.get("models", []) if m.get("kind") == "lora"]
            check(len(loras) == 1 and Path(loras[0]["path"]) == adapter and loras[0]["size"] > 0,
                  "the models directory's adapter is listed, kind lora, by its folder", json.dumps(loras))
            check(all(m.get("kind") != "lora" or m["name"] == m["path"] for m in cached.get("models", [])),
                  "a listed adapter's name is the folder \"lora\" takes")

            # The Hub's adapters: a file outside the cache has no base to ask for; offline, the Hub is not asked.
            code, found = call(s.origin, "/v1/hub/loras?model=" + quote(gguf))
            check(code == 200 and found.get("base") is None and found.get("models") == [],
                  "GET /v1/hub/loras: a local file's base is not known", json.dumps(found))
            code, found = call(s.origin, "/v1/hub/loras?base=Qwen/Qwen3-0.6B")
            check(code == 502 and "HF_HUB_OFFLINE" in json.dumps(found), "GET /v1/hub/loras offline: 502, not asked",
                  f"{code} {json.dumps(found)}")
            code, _ = call(s.origin, "/v1/hub/lora?repo=" + quote("not a repo"))
            check(code == 400, "GET /v1/hub/lora: a name that is not a repository is refused (400)", str(code))

            code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": 5})
            check(code == 400, "\"lora\": 5 is refused (400)", str(code))

            code, _ = call(s.origin, "/v1/load", {"model": gguf, "speculative": False})
            st = settle(s.origin)
            check(code == 202 and st["state"] == "ready" and st["loaded"]["lora"] is None,
                  "the model alone loads; GET /v1/load says no adapter", json.dumps(st)[:300])
            plain = reply(s.origin)

            code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": str(adapter), "speculative": False})
            st = settle(s.origin)
            check(code == 202 and st["state"] == "ready" and st["loaded"]["lora"] == str(adapter),
                  "the model with the adapter loads; GET /v1/load names it", json.dumps(st)[:300])
            adapted = reply(s.origin)
            check(adapted != plain, "the adapter changes what the model says", f"{plain!r} vs {adapted!r}")

            # A file of the adapter's folder names the folder.
            code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": str(adapter / "adapter_config.json"),
                                                  "speculative": False})
            st = settle(s.origin)
            check(st["state"] == "ready" and reply(s.origin) == adapted,
                  "a file of the adapter's folder loads the folder's adapter", json.dumps(st)[:300])

            # A name that cannot be fetched fails before the model served goes: it keeps serving.  An adapter that
            # is read and refused fails the load.
            for what, lora, state, why in (("no such folder", str(tmp / "missing"), "ready", "no such directory"),
                                           ("a DoRA adapter", str(dora), "failed", "DoRA")):
                code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": lora, "speculative": False})
                st = settle(s.origin)
                check(st["state"] == state and why in (st.get("error") or ""),
                      f"{what} is refused with its reason ({state})", json.dumps(st)[:400])

            code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": str(adapter), "speculative": False})
            st = settle(s.origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] == str(adapter), "after a refusal, the adapter loads")

        saved = json.loads(next((tmp / "config").rglob("serve.json")).read_text())
        check(saved["last"].get("lora") == str(adapter), "serve.json remembers the adapter", json.dumps(saved)[:300])
        with Server(exe, env, []) as s:
            st = settle(s.origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] == str(adapter) and reply(s.origin) == adapted,
                  "the next start loads the model with its adapter again", json.dumps(st)[:300])
            code, _ = call(s.origin, "/v1/load", {"model": gguf, "lora": None, "speculative": False})
            st = settle(s.origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] is None and reply(s.origin) == plain,
                  "\"lora\": null loads the model alone")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("lora_load: " + ("ok" if FAILS == 0 else f"{FAILS} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
