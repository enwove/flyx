# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""One page of a release's smoke tests (.github/workflows/release-smoke.yml), as Markdown:

    python tools/ci/release_smoke_report.py RESULTS.tsv...

Each RESULTS file is one platform's (tools/ci/release_smoke.sh, tools/ci/release_smoke_windows.ps1: a line a check,
check<TAB>status<TAB>detail), named after it.  First a table of the platforms (what failed on each), then every
platform's checks.
"""
from __future__ import annotations

import os
import sys


def cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main(paths: list[str]) -> None:
    platforms = []
    for path in sorted(paths):
        with open(path, encoding="utf-8-sig") as f:
            rows = [line.rstrip("\r\n").split("\t", 2) for line in f if line.strip()]
        rows = [r + [""] * (3 - len(r)) for r in rows]
        platforms.append((os.path.splitext(os.path.basename(path))[0], rows))

    print("| Platform | System | Checks | Failed |\n|---|---|---|---|")
    for name, rows in platforms:
        system = next((d for c, _, d in rows if c == "system"), "")
        failed = [c for c, s, _ in rows if s == "FAIL"]
        counts = ", ".join(f"{n} {s}" for s in ("ok", "expected", "skipped", "FAIL")
                           if (n := sum(1 for _, x, _ in rows if x == s)))
        print(f"| {cell(name)} | {cell(system)} | {counts} | {cell(', '.join(failed)) or 'none'} |")
    for name, rows in platforms:
        print(f"\n<details><summary>{cell(name)}</summary>\n\n| Check | Status | Detail |\n|---|---|---|")
        for check, status, detail in rows:
            print(f"| {cell(check)} | {'**FAIL**' if status == 'FAIL' else status} | {cell(detail)} |")
        print("\n</details>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
