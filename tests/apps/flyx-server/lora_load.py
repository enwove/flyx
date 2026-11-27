# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A LoRA adapter from the dashboard's way in: listed among a models directory's files (kind "lora"), loaded with
POST /v1/load's "lora" beside its model, shown in GET /v1/load, remembered for the next start, and refused with its
reason (no such folder, a DoRA adapter) without taking the server down; GET /v1/hub/loras and /v1/hub/lora (the Hub's
adapters) where the Hub cannot be asked; and a repository of the Hugging Face cache with two adapters, one whole and
one with a linear's lora_A alone, which GET /v1/hub/lora lists as the load then finds them (the first loads by its
Hub name, the second is refused for that reason by both).  The servers are local_server.py's (--port 0, server_env:
none of the shell's FLYX_* settings).

    lora_load.py FLYX MODEL.gguf PEFT_DIR
"""
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from local_server import serve, server_env

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


def without_one_tensor(src, dst):
    """src's adapter with the first lora_B its safetensors header lists left out (its lora_A then alone); that
    tensor's linear as the loader names it (L<layer>.<attn|mlp>.<linear>)."""
    blob = (src / "adapter_model.safetensors").read_bytes()
    n = struct.unpack("<Q", blob[:8])[0]
    header, data = json.loads(blob[8:8 + n]), blob[8 + n:]
    dropped = next(k for k in sorted(header) if ".lora_B." in k)
    out, parts, at = {}, [], 0
    for key, t in header.items():
        if key == "__metadata__" or key == dropped:
            continue
        first, last = t["data_offsets"]
        out[key] = dict(t, data_offsets=[at, at + last - first])
        parts.append(data[first:last])
        at += last - first
    text = json.dumps(out).encode()
    dst.mkdir(parents=True)
    (dst / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(text)) + text + b"".join(parts))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    parts = dropped.split(".")
    layer = parts[parts.index("layers") + 1]
    blk, proj = parts[parts.index("layers") + 2], parts[parts.index("layers") + 3]
    return f"L{layer}.{'attn' if blk == 'self_attn' else 'mlp'}.{proj.removesuffix('_proj')}"


def main():
    exe, model, peft = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
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
        # A repository of the Hugging Face cache (read offline): whole/ the adapter, one_sided/ it with a lora_B left
        # out.
        commit = "c" * 40
        repo_dir = tmp / "hf" / "hub" / "models--acme--tiny-loras"
        (repo_dir / "refs").mkdir(parents=True)
        (repo_dir / "refs" / "main").write_text(commit)
        snapshot = repo_dir / "snapshots" / commit
        shutil.copytree(peft, snapshot / "whole")
        alone = without_one_tensor(peft, snapshot / "one_sided")
        env = server_env(tmp / "config", exe, HF_HUB_CACHE=tmp / "hf" / "hub", HF_HUB_OFFLINE="1")
        gguf = str(models / model.name)
        served = {}

        def first(origin):
            _, cached = call(origin, "/v1/hub/cached")
            loras = [m for m in cached.get("models", []) if m.get("kind") == "lora" and m.get("source") != "hub"]
            check(len(loras) == 1 and Path(loras[0]["path"]) == adapter and loras[0]["size"] > 0,
                  "the models directory's adapter is listed, kind lora, by its folder", json.dumps(loras))
            check(all(m.get("kind") != "lora" or m.get("source") == "hub" or m["name"] == m["path"]
                      for m in cached.get("models", [])),
                  "a listed adapter's name is the folder \"lora\" takes")
            hub_loras = sorted(m["name"] for m in cached.get("models", [])
                               if m.get("kind") == "lora" and m.get("source") == "hub")
            check(hub_loras == ["acme/tiny-loras:one_sided", "acme/tiny-loras:whole"],
                  "the cache's adapters are listed by the names that fetch them", json.dumps(hub_loras))

            # The Hub's adapters: a file outside the cache has no base to ask for; offline, the Hub is not asked.
            code, found = call(origin, "/v1/hub/loras?model=" + quote(gguf))
            check(code == 200 and found.get("base") is None and found.get("models") == [],
                  "GET /v1/hub/loras: a local file's base is not known", json.dumps(found))
            code, found = call(origin, "/v1/hub/loras?base=Qwen/Qwen3-0.6B")
            check(code == 502 and "HF_HUB_OFFLINE" in json.dumps(found), "GET /v1/hub/loras offline: 502, not asked",
                  f"{code} {json.dumps(found)}")
            code, _ = call(origin, "/v1/hub/lora?repo=" + quote("not a repo"))
            check(code == 400, "GET /v1/hub/lora: a name that is not a repository is refused (400)", str(code))
            # A repository's adapters as the listing reads them (config and safetensors header): what the load does.
            code, listed = call(origin, "/v1/hub/lora?repo=acme/tiny-loras")
            why = {a["folder"]: a["why"] for a in listed.get("adapters", [])}
            check(code == 200 and why.get("whole", "") is None, "GET /v1/hub/lora: the whole adapter loads (why null)",
                  json.dumps(listed)[:600])
            check(f"{alone} has lora_A alone" in (why.get("one_sided") or ""),
                  "GET /v1/hub/lora: a linear's lora_A alone is not loadable, for the load's reason",
                  json.dumps(listed)[:600])

            code, _ = call(origin, "/v1/load", {"model": gguf, "lora": 5})
            check(code == 400, "\"lora\": 5 is refused (400)", str(code))

            code, _ = call(origin, "/v1/load", {"model": gguf, "speculative": False})
            st = settle(origin)
            check(code == 202 and st["state"] == "ready" and st["loaded"]["lora"] is None,
                  "the model alone loads; GET /v1/load says no adapter", json.dumps(st)[:300])
            plain = reply(origin)

            code, _ = call(origin, "/v1/load", {"model": gguf, "lora": str(adapter), "speculative": False})
            st = settle(origin)
            check(code == 202 and st["state"] == "ready" and st["loaded"]["lora"] == str(adapter),
                  "the model with the adapter loads; GET /v1/load names it", json.dumps(st)[:300])
            adapted = reply(origin)
            check(adapted != plain, "the adapter changes what the model says", f"{plain!r} vs {adapted!r}")

            # A file of the adapter's folder names the folder.
            code, _ = call(origin, "/v1/load", {"model": gguf, "lora": str(adapter / "adapter_config.json"),
                                                "speculative": False})
            st = settle(origin)
            check(st["state"] == "ready" and reply(origin) == adapted,
                  "a file of the adapter's folder loads the folder's adapter", json.dumps(st)[:300])
            # The cache's whole adapter by its Hub name: the folder's own.
            code, _ = call(origin, "/v1/load", {"model": gguf, "lora": "acme/tiny-loras:whole", "speculative": False})
            st = settle(origin)
            check(st["state"] == "ready" and reply(origin) == adapted,
                  "a cached adapter loads by its Hub name (org/name:FOLDER)", json.dumps(st)[:300])

            # A name that cannot be fetched fails before the model served goes: it keeps serving.  An adapter that
            # is read and refused fails the load.
            for what, lora, state, why in (("no such folder", str(tmp / "missing"), "ready", "no such directory"),
                                           ("a DoRA adapter", str(dora), "failed", "DoRA"),
                                           ("a linear's lora_A alone", "acme/tiny-loras:one_sided", "failed",
                                            f"{alone} has lora_A alone")):
                code, _ = call(origin, "/v1/load", {"model": gguf, "lora": lora, "speculative": False})
                st = settle(origin)
                check(st["state"] == state and why in (st.get("error") or ""),
                      f"{what} is refused with its reason ({state})", json.dumps(st)[:400])

            code, _ = call(origin, "/v1/load", {"model": gguf, "lora": str(adapter), "speculative": False})
            st = settle(origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] == str(adapter), "after a refusal, the adapter loads")
            _, health = call(origin, "/health")
            served.update(name=health.get("model"), plain=plain, adapted=adapted)

        serve(exe, tmp, env, on_ready=first, options=["--models-dir", str(models)], expect_name=None)
        saved = json.loads(next((tmp / "config").rglob("serve.json")).read_text())
        check(saved["last"].get("lora") == str(adapter), "serve.json remembers the adapter", json.dumps(saved)[:300])

        def again(origin):
            st = settle(origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] == str(adapter) and reply(origin) == served["adapted"],
                  "the next start loads the model with its adapter again", json.dumps(st)[:300])
            call(origin, "/v1/load", {"model": gguf, "lora": None, "speculative": False})
            st = settle(origin)
            check(st["state"] == "ready" and st["loaded"]["lora"] is None and reply(origin) == served["plain"],
                  "\"lora\": null loads the model alone")

        serve(exe, tmp, env, on_ready=again, expect_name=served["name"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("lora_load: " + ("ok" if FAILS == 0 else f"{FAILS} FAILED"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
