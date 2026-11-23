# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The model listing publishes stable Unix times and notices a changed, same-sized GGUF header."""
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from urllib.request import urlopen

from remembered_model import serve


def gguf(layers):
    def string(text):
        return struct.pack("<Q", len(text)) + text.encode("ascii")

    return (b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
            + string("general.architecture") + struct.pack("<I", 8) + string("test")
            + string("test.nextn_predict_layers") + struct.pack("<II", 4, layers))


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


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="flyx-model-times-") as tmp:
        root = Path(tmp)
        model = root / "test model.flyx"
        shutil.copyfile(asset, model)
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, HF_HOME=tmp, HF_HUB_CACHE=str(root / "hub"),
                   HF_HUB_OFFLINE="1", FLYX_MODELS_DIR=tmp, FLYX_BACKEND_DIR=str(exe.parent))
        serve(exe, root, env, [str(model)], lambda url: check_listing(url, root / "weights.gguf"))
