# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The model listing publishes stable Unix times and notices a changed, same-sized GGUF header."""
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


def gguf(layers, arch="test", has_markov=False):
    def string(text):
        return struct.pack("<Q", len(text)) + text.encode("ascii")

    kv = [(string("general.architecture") + struct.pack("<I", 8) + string(arch))]
    if arch == "dflash":
        kv.append(string("dflash.has_markov_context") + struct.pack("<IB", 7, 1 if has_markov else 0))
    else:
        kv.append(string(arch + ".nextn_predict_layers") + struct.pack("<II", 4, layers))
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(kv)


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

    # Cached Hub GGUFs with general.architecture=dflash (DFlash2 or DSpark) and local arch=dflash GGUFs
    # must be classified as kind="drafter" (never as standalone models).
    commit = "a" * 40
    for repo_dir, fname, markov, want_dk in (
        ("models--acme--Mini-DFlash2-GGUF", "Mini-DFlash2-Draft-BF16.gguf", False, "dflash2"),
        ("models--acme--Mini-DSpark-GGUF", "Mini-DSpark-Draft-Q8_0.gguf", True, "dspark"),
    ):
        rdir = root / "hub" / repo_dir
        (rdir / "refs").mkdir(parents=True, exist_ok=True)
        (rdir / "refs" / "main").write_text(commit, encoding="ascii")
        sdir = rdir / "snapshots" / commit
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / fname).write_bytes(gguf(0, arch="dflash", has_markov=markov))
    (root / "unnamed-draft-Q8_0.gguf").write_bytes(gguf(0, arch="dflash", has_markov=False))
    with urlopen(url + "/v1/hub/cached", timeout=5) as response:
        models3 = json.load(response)["models"]
    by_key = {(m["source"], m["file"]): m for m in models3}
    df_row = by_key[("hub", "Mini-DFlash2-Draft-BF16.gguf")]
    ds_row = by_key[("hub", "Mini-DSpark-Draft-Q8_0.gguf")]
    loc_row = by_key[("dir", "unnamed-draft-Q8_0.gguf")]
    assert df_row["kind"] == "drafter" and df_row["drafter_kind"] == "dflash2", df_row
    assert ds_row["kind"] == "drafter" and ds_row["drafter_kind"] == "dspark", ds_row
    assert loc_row["kind"] == "drafter" and loc_row["drafter_kind"] == "dflash2", loc_row

    # Cached Hub repos with an mmproj file: vision-capable arch (qwen35) advertises mmproj; non-vision arch
    # (inkling2_loop) suppresses mmproj in both /v1/hub/cached and /v1/hub/files.
    for repo_dir, fname, arch in (
        ("models--acme--Mini-Qwen35-GGUF", "Mini-Qwen35-Q4_K_M.gguf", "qwen35"),
        ("models--acme--Mini-Inkling2-GGUF", "Mini-Inkling2-Q4_K_M.gguf", "inkling2_loop"),
    ):
        rdir = root / "hub" / repo_dir
        (rdir / "refs").mkdir(parents=True, exist_ok=True)
        (rdir / "refs" / "main").write_text(commit, encoding="ascii")
        sdir = rdir / "snapshots" / commit
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / fname).write_bytes(gguf(0, arch=arch))
        (sdir / "mmproj-BF16.gguf").write_bytes(gguf(0, arch="clip"))
    with urlopen(url + "/v1/hub/cached", timeout=5) as response:
        by_key4 = {(m["source"], m["file"]): m for m in json.load(response)["models"]}
    assert by_key4[("hub", "Mini-Qwen35-Q4_K_M.gguf")]["mmproj"] == "in-repo", by_key4[("hub", "Mini-Qwen35-Q4_K_M.gguf")]
    assert by_key4[("hub", "Mini-Inkling2-Q4_K_M.gguf")]["mmproj"] is None, by_key4[("hub", "Mini-Inkling2-Q4_K_M.gguf")]
    with urlopen(url + "/v1/hub/files?repo=acme/Mini-Qwen35-GGUF", timeout=5) as response:
        q_repo = json.load(response)
    assert q_repo["mmproj"] is not None and any(c["kind"] == "mmproj" for c in q_repo["companions"]), q_repo
    with urlopen(url + "/v1/hub/files?repo=acme/Mini-Inkling2-GGUF", timeout=5) as response:
        i_repo = json.load(response)
    assert i_repo["mmproj"] is None and not any(c["kind"] == "mmproj" for c in i_repo["companions"]), i_repo

    # A failed POST /v1/load switch must leave the server in state="failed" with no model served (never auto-reload
    # the previous model), while keeping `remembered` intact for an explicit reload.
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


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="flyx-model-times-") as tmp:
        root = Path(tmp)
        model = root / "test model.flyx"
        shutil.copyfile(asset, model)
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, HF_HOME=tmp, HF_HUB_CACHE=str(root / "hub"),
                   HF_HUB_OFFLINE="1", FLYX_MODELS_DIR=tmp, FLYX_BACKEND_DIR=str(exe.parent))
        serve(exe, root, env, [str(model)], lambda url: check_listing(url, root / "weights.gguf"))
