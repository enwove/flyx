# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The break-even tables of tree verification from bench/tree/break_even.sh's output.

    python3 bench/tree/break_even_tables.py OUT

Up to 16 rows: tree rounds (random shape) over plain decoding's round in the same run.  Above: the 16-row tree round
plus what a pass of R rows costs over one of 16 at the context (append_prefill.py: the prefill kernels) plus the head's
increase from 16 to R rows; three lines: NVFP4 with bf16 inputs (Nx: plain decoding's precision), NVFP4 with NVFP4
inputs above 16 rows (N: W4A4), the GGUF file (M).  Prints Markdown: per context, rows -> round / plain round (ms).
"""
import re
import sys

out = sys.argv[1]
CTX = [1024, 8192, 24576]
SHAPES = ["chain", "star", "binary", "random"]


def read(name):
    return open(f"{out}/{name}", encoding="utf-8").read()


def tree(f, at):
    t = read(f"tree_{f}_{at}.txt")
    plain = float(re.search(r"plain round ([0-9.]+) ms", t).group(1))
    rows = {}
    for line in t.splitlines():
        p = line.split()
        if len(p) == 9 and p[0].isdigit():
            rows[int(p[0]) + 1] = {k: float(p[1 + 2 * i]) for i, k in enumerate(SHAPES)}
    return plain, rows


def append(v, at):
    out_rows = {}
    for line in read(f"append_{v}_{at}.txt").splitlines():
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            out_rows[int(p[0])] = float(p[1])
    return out_rows


head = {int(a): float(b) / 1000 for a, b in (l.split() for l in read("head.txt").splitlines() if l.strip())}
lines = [("NVFP4 exact (bf16 inputs)", "N", "Nx"), ("NVFP4 W4A4 above 16", "N", "N"), ("GGUF", "M", "M")]
for at in CTX:
    cols = []
    for label, f, v in lines:
        plain, rows = tree(f, at)
        pts = {1: (plain, 1.0)}
        pts.update({r: (rows[r]["random"], rows[r]["random"] / plain) for r in rows})
        a = append(v, at)
        t16 = rows[16]["random"]
        for r in sorted(a):
            if r > 16 and 16 in a:
                ms = t16 + (a[r] - a[16]) + (head[r] - head[16])
                pts[r] = (ms, ms / plain)
        cols.append(pts)
    print(f"\nContext {at}:\n")
    print("| rows | " + " | ".join(l[0] for l in lines) + " |")
    print("|---|" + "---|" * len(lines))
    for r in sorted({r for c in cols for r in c}):
        print(f"| {r} | " + " | ".join(f"{c[r][1]:.2f} ({c[r][0]:.1f} ms)" if r in c else "" for c in cols) + " |")
print("\nPower limit: " + ", ".join(l.strip() for l in read("power.txt").splitlines() if l.strip()))
