# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The numerics contract (docs/design/numerics.md) covers every CUDA launcher; this test keeps it so.

    python tools/check_numerics.py [REPO]

* Every function declared in backends/cuda/kernels/kernels.cuh has a row in the
  "Launchers" table.
* Every test the table names exists in tests/backends/cuda/op_test*.cu.
"""
from __future__ import annotations

import glob
import os
import re
import sys


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(root: str) -> int:
    header = _read(os.path.join(root, "backends/cuda/kernels/kernels.cuh"))
    tests = "\n".join(_read(f) for f in sorted(glob.glob(os.path.join(root, "tests/backends/cuda/op_test*.cu"))))
    doc = _read(os.path.join(root, "docs/design/numerics.md"))
    launchers = set(re.findall(r"^(?:void|bool|int|size_t|const char\*|const bf16\*|Scratch)\s+(\w+)\(", header,
                               re.M))
    table = doc[doc.index("## Launchers"):]
    listed, named_tests = set(), set()
    for line in table.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] in ("launcher", "---"):
            continue
        listed.update(n.strip() for n in cells[0].split(","))
        named_tests.update(re.findall(r"\btest_\w+", cells[4]))
    defined_tests = set(re.findall(r"^void (test_\w+)\(", tests, re.M))
    bad = [f"{n}: declared in kernels.cuh, not in docs/design/numerics.md" for n in sorted(launchers - listed)]
    bad += [f"{t}: named in docs/design/numerics.md, not defined in op_test*.cu" for t in sorted(named_tests - defined_tests)]
    bad += [f"{n}: listed in docs/design/numerics.md, not declared in kernels.cuh" for n in sorted(listed - launchers)]
    for b in bad:
        print("FAIL", b)
    print("numerics: " + ("ok" if not bad else f"{len(bad)} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")))
