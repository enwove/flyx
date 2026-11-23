# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Summarize a flyx drafter benchmark directory: per configuration and workload, median over repetitions.

    python bench/drafters_table.py DIR

DIR holds flyx_<config>.<workload>.<rep>.err stderr reports. Decode reports tok/s
((generated tokens - 1) / decode time), generated token count, and tokens/round;
the long workload reports prompt tokens and prefill tok/s.
"""
import os
import re
import statistics
import sys
from collections import defaultdict

FLYX = re.compile(r"\[prompt (\d+) \(reused \d+\) \| prefill ([\d.]+) ms \| (\d+) tokens in ([\d.]+) ms = ([\d.]+) "
                  r"tok/s \| (\d+) rounds, ([\d.]+) tokens/round")
WORKLOADS = ("code", "cache", "list", "reason", "long")


def flyx(text):
    m = FLYX.findall(text)
    if not m:
        return None
    prompt, prefill_ms, new, _, tok_s, _, per_round = m[-1]
    return {"prompt": int(prompt), "prefill_ms": float(prefill_ms), "new": int(new), "tok_s": float(tok_s),
            "per_round": float(per_round)}


def main(d):
    runs = defaultdict(list)
    for f in sorted(os.listdir(d)):
        if f.endswith(".err") and f.count(".") == 3:
            name, workload, _, _ = f.split(".")
            if not name.startswith("flyx_") or workload not in WORKLOADS:
                continue
            text = open(os.path.join(d, f), errors="replace").read()
            r = flyx(text)
            if not r or r["new"] is None or (workload == "long" and (not r["prompt"] or not r["prefill_ms"])):
                raise SystemExit(f"missing benchmark metrics in {os.path.join(d, f)}")
            runs[(name, workload)].append(r)
    if not runs:
        raise SystemExit(f"no benchmark captures in {d}")
    med = lambda xs: statistics.median(xs) if xs else float("nan")  # noqa: E731
    print(f"{'config':20s}" + "".join(f" | {w:>27s}" for w in WORKLOADS))
    for name in sorted({n for n, _ in runs}):
        cells = []
        for w in WORKLOADS:
            rs = runs.get((name, w), [])
            if not rs:
                cells.append(f"{'-':>27s}")
            elif w == "long":
                tps = med([r["prompt"] / r["prefill_ms"] * 1000 for r in rs if r["prefill_ms"]])
                cells.append(f"{int(med([r['prompt'] for r in rs])):5d} tokens {tps:7.0f} tok/s")
            else:
                pr = [r["per_round"] for r in rs if r["per_round"]]
                cells.append(f"{med([r['tok_s'] for r in rs]):6.1f} tok/s {int(med([r['new'] for r in rs])):4d} new"
                             + (f" {med(pr):4.2f}/r" if pr else "       "))
        print(f"{name:20s}" + "".join(f" | {c}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
