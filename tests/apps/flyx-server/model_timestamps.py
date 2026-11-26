# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The model listing publishes stable Unix times and notices a changed, same-sized GGUF header; a switch to a model
that fails to load leaves none served."""
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from urllib.request import Request, urlopen
import time

from remembered_model import serve


def gguf(layers, arch="test"):
    def string(text):
        return struct.pack("<Q", len(text)) + text.encode("ascii")

    return (b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
            + string("general.architecture") + struct.pack("<I", 8) + string(arch)
            + string(arch + ".nextn_predict_layers") + struct.pack("<II", 4, layers))


def check_failed_switch(url, root):
    """A switch to a model that fails to load leaves the server with none (state "failed"): the model served before
    is not loaded again, and what the settings file remembers stays, for a reload that is asked for."""
    with urlopen(url + "/v1/load", timeout=5) as response:
        before = json.load(response)
    assert before["state"] == "ready" and before["model"] is not None and before["remembered"] is not None, before
    bogus = root / "not-a-model.gguf"
    bogus.write_bytes(b"not a valid gguf \x00\x01\x02")
    req = Request(url + "/v1/load", data=json.dumps({"model": str(bogus)}).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as response:
        assert response.status == 202
    deadline = time.monotonic() + 30
    after = {}
    while time.monotonic() < deadline:
        with urlopen(url + "/v1/load", timeout=5) as response:
            after = json.load(response)
        if after.get("state") not in ("fetching", "loading"):
            break
        time.sleep(0.05)
    assert after["state"] == "failed" and after["model"] is None and after["fallback"] is None, after
    assert after["error"] and after["remembered"] == before["remembered"], after


def check_listing(url, file):
    def entry():
        with urlopen(url + "/v1/hub/cached", timeout=5) as response:
            models = json.load(response)["models"]
        return next(m for m in models if m["file"] == file.name)

    stamp = 1_700_000_000_000_000_000
    file.write_bytes(gguf(1))
    os.utime(file, ns=(stamp, stamp))
    first = entry()
    assert first["mtp"] == "in-file" and first["modified"] == file.stat().st_mtime, first
    for _ in range(12):
        assert entry()["modified"] == first["modified"]

    # A 100 ns change here fits inside one double's unit. Use two seconds on coarser filesystems.
    before = file.stat().st_mtime_ns
    file.write_bytes(gguf(0))
    for step in (100, 2_000_000_000):
        os.utime(file, ns=(stamp + step, stamp + step))
        if file.stat().st_mtime_ns != before:
            break
    assert file.stat().st_mtime_ns != before
    second = entry()
    assert second["modified"] == file.stat().st_mtime and second["mtp"] is None, second

    # Complete split GGUF in --models-dir is listed once at shard 1 with summed size; incomplete split is omitted.
    root = file.parent
    p1 = root / "demo-Q4_K_M-00001-of-00002.gguf"
    p2 = root / "demo-Q4_K_M-00002-of-00002.gguf"
    inc = root / "broken-Q8_0-00001-of-00003.gguf"
    p1.write_bytes(gguf(1))
    p2.write_bytes(gguf(0) + b"\0" * 128)
    inc.write_bytes(gguf(0))
    with urlopen(url + "/v1/hub/cached", timeout=5) as response:
        models = json.load(response)["models"]
    split_rows = [m for m in models if "demo-Q4_K_M" in m["file"]]
    assert len(split_rows) == 1, split_rows
    assert split_rows[0]["file"] == p1.name and split_rows[0]["size"] == p1.stat().st_size + p2.stat().st_size, split_rows[0]
    assert split_rows[0]["quant"] == "Q4_K_M" and split_rows[0]["mtp"] == "in-file", split_rows[0]
    assert not any("broken-Q8_0" in m["file"] for m in models), models

    # Dangling symlink in --models-dir must be ignored cleanly (never reported with UINT64_MAX size).
    dangling = root / "dangling-Q4_K_M.gguf"
    try:
        os.symlink(root / "does-not-exist.gguf", dangling)
    except OSError:
        dangling = None
    if dangling is not None:
        with urlopen(url + "/v1/hub/cached", timeout=5) as response:
            models2 = json.load(response)["models"]
        assert not any("dangling-Q4_K_M" in m["file"] for m in models2), models2
        assert all(0 <= m["size"] < (1 << 60) for m in models2), models2

    # Unified GGUF header classification (apps::classify_gguf): arch="dflash" (even without "dflash" in the filename,
    # e.g. DSpark or Hub cached GGUF drafters) -> kind="drafter", drafter_kind="dflash2"; arch="clip" -> kind="mmproj".
    (root / "dspark-custom-Q8_0.gguf").write_bytes(gguf(0, arch="dflash"))
    (root / "renamed-vision-BF16.gguf").write_bytes(gguf(0, arch="clip"))
    commit = "a" * 40
    hub_snap = root / "hub" / "models--acme--Test-DFlash2-GGUF" / "snapshots" / commit
    hub_snap.mkdir(parents=True, exist_ok=True)
    (hub_snap / "Test-DFlash2-Q4_K_M.gguf").write_bytes(gguf(0, arch="dflash"))
    refs = root / "hub" / "models--acme--Test-DFlash2-GGUF" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(commit, encoding="ascii")
    with urlopen(url + "/v1/hub/cached", timeout=5) as response:
        models3 = json.load(response)["models"]
    dspark = next(m for m in models3 if m["file"] == "dspark-custom-Q8_0.gguf")
    assert dspark["kind"] == "drafter" and dspark["drafter_kind"] == "dflash2" and dspark["mtp"] is None, dspark
    vis = next(m for m in models3 if m["file"] == "renamed-vision-BF16.gguf")
    assert vis["kind"] == "mmproj" and vis["mtp"] is None, vis
    hub_df = next(m for m in models3 if m["name"] == "acme/Test-DFlash2-GGUF:Q4_K_M")
    assert hub_df["source"] == "hub" and hub_df["kind"] == "drafter" and hub_df["drafter_kind"] == "dflash2", hub_df
    assert hub_df["mtp"] is None and hub_df["mmproj"] is None, hub_df


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="flyx-model-times-") as tmp:
        root = Path(tmp)
        model = root / "test model.flyx"
        shutil.copyfile(asset, model)
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, HF_HOME=tmp, HF_HUB_CACHE=str(root / "hub"),
                   HF_HUB_OFFLINE="1", FLYX_MODELS_DIR=tmp, FLYX_BACKEND_DIR=str(exe.parent))

        def checks(url):
            check_listing(url, root / "weights.gguf")
            check_failed_switch(url, root)  # last: it leaves the server with no model

        serve(exe, root, env, [str(model)], checks)
