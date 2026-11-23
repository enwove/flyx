# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The converter uses the given model directory literally and prints its result on a redirected console."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


if __name__ == "__main__":
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3] / "python" / "convert"))
    with tempfile.TemporaryDirectory(prefix="flyx console ") as tmp:
        assets = Path(sys.argv[1])
        model = shutil.copytree(assets / "hf_target", Path(tmp) / "Qwen [4B]")
        shutil.copytree(assets / "hf_qwen3", Path(tmp) / "Qwen 4")  # matches the brackets as a glob
        out = Path(tmp) / "模型😁.flyx"
        cmd = [sys.executable, "-X", "utf8=1", "-m", "flyx_convert", "--model", str(model), "--out", str(out)]
        for encoding in ("utf-8", "ascii"):
            env["PYTHONIOENCODING"] = encoding + ":strict"
            log = subprocess.check_output(cmd, env=env).decode(encoding)
            assert str(out).encode(encoding, "backslashreplace").decode(encoding) in log
            if encoding == "utf-8":
                expected = out.read_bytes()
            else:
                assert out.read_bytes() == expected
