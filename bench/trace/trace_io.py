# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Shared TSV loading, kernel-name normalization, and GGML type maps for bench/trace/ scripts."""
from __future__ import annotations

import re

GGML_TYPES = {
    8: "Q8_0", 10: "Q2_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
}


def plain_kernel_name(name: str) -> str:
    """Drop anonymous/flyx::cuda namespace qualifiers from a demangled CUDA symbol."""
    return re.sub(r"\(anonymous namespace\)::|flyx::cuda::", "", name)


def short_kernel_name(name: str) -> str:
    """Short unqualified function name without return type, parameters, or template arguments."""
    n = re.sub(r"^void ", "", name).replace("(anonymous namespace)::", "")
    n = n.split("(", 1)[0]
    depth, out = 0, []
    for ch in n:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).rsplit("::", 1)[-1]


def iter_trace_rows(path: str):
    """Yield tab-split fields for each record in a kernel_trace TSV (skipping the header line)."""
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            yield line.rstrip("\n").split("\t")


def split_bursts(rows, gap_ns: int = 2_000_000):
    """Group start-sorted (start, end, ...) records into bursts separated by idle gaps > gap_ns."""
    if not rows:
        return []
    out, cur, last = [], [], None
    for r in rows:
        if last is not None and r[0] - last > gap_ns:
            out.append(cur)
            cur = []
        cur.append(r)
        last = max(last, r[1]) if last is not None and r[0] <= last else r[1]
    out.append(cur)
    return out
