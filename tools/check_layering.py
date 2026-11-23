# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Directory boundaries are architecture boundaries; this test enforces them.

    python tools/check_layering.py [REPO]

* core/, models/ and ops/ are plain C++: no CUDA (or other vendor) headers.
* Dependencies point inward: core/ depends on nothing outside core/ and
  include/; ops/ only on core/interfaces/ and include/; models/ on core/interfaces/ and
  the file reader (core/modelfile/: tensors by name, the config); backends/
  only on core/interfaces/, ops/ and include/ (they get the model as data); apps/
  tie everything together.
* Every C++/CUDA/Python source file starts with the copyright and SPDX lines.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

VENDOR = re.compile(r'#\s*include\s*[<"](cuda|cuda_runtime|cuda_bf16|cuda_fp16|cuda_fp8|cublas|cudnn|nccl|hip/|metal)',
                    re.I)
INCLUDE = re.compile(r'#\s*include\s*"([^"]+)"')
# directory -> (project include prefixes it must not use, exceptions to those)
RULES = {
    "core": (("apps/", "backends/", "models/", "ops/"), ()),
    "models": (("apps/", "backends/", "core/", "ops/"), ("core/interfaces/", "core/modelfile/")),
    "ops": (("apps/", "backends/", "core/", "models/"), ("core/interfaces/",)),
    "backends": (("apps/", "core/", "models/"), ("core/interfaces/",)),
}
PLAIN_CXX = ("core", "models", "ops")
SOURCES = (".h", ".hpp", ".cpp", ".cu", ".cuh", ".py")


def tracked(root: str) -> list[str]:
    """Files under version control (downloaded models also live under models/)."""
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout
        return [l for l in out.splitlines() if l]
    except (OSError, subprocess.CalledProcessError):
        files = []
        for d, _, fs in os.walk(root):
            files += [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for f in fs]
        return files


def main(root: str) -> int:
    bad = []
    tops = ("core", "models", "ops", "backends", "apps", "include", "python", "tools", "tests", "bench")
    for rel in tracked(root):
        top = rel.split("/", 1)[0]
        if top not in tops or not rel.endswith(SOURCES) or "__pycache__" in rel:
            continue
        p = os.path.join(root, rel)
        with open(p, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "SPDX-License-Identifier: Apache-2.0" not in text[:400]:
            bad.append(f"{rel}: no SPDX line at the top")
        if top in PLAIN_CXX and VENDOR.search(text):
            bad.append(f"{rel}: includes a vendor (CUDA) header; core/, models/ and ops/ stay plain C++")
        deny, allow = RULES.get(top, ((), ()))
        for inc in INCLUDE.findall(text):
            if inc.startswith(allow):
                continue
            for prefix in deny:
                if inc.startswith(prefix):
                    bad.append(f"{rel}: includes {inc} ({top}/ must not depend on {prefix})")
    for b in bad:
        print("FAIL", b)
    print("layering: " + ("ok" if not bad else f"{len(bad)} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")))
