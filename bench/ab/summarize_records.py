# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The records bench/ab/paired_records.sh leaves: per run, decode tok/s, a round's ms, tokens per forward (bifill-dv's
count and flyx bench's), over the first N prompts or all; and which runs another program's GPU work ran beside.

    python -I bench/ab/summarize_records.py DIR [--first N] [--alone-below MIB]

A run is "shared" when a sample of sampler.log taken while it ran (nvidia-smi once a minute) shows the GPU's memory
in use at MIB or more: the run's own memory with the desktop's stays below that, another program's work on the GPU
does not (paired_records.sh's QUIET_MIB waits for it to be gone before a run starts, but not while one runs).  Then
per configuration and side the median of the runs that ran alone, B against A, and whether the two sides gave the
same answers (every run of both sides, prompt by prompt: a speed change must not change one).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import statistics
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
spec = importlib.util.spec_from_file_location("check_dv", os.path.join(ROOT, "tests/e2e/check_dv.py"))
check_dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_dv)

STAMP = "%Y/%m/%d %H:%M:%S"


def stamp(text: str) -> datetime.datetime:
    return datetime.datetime.strptime(text[:19], STAMP)


def runs_of(log: str) -> dict:
    """(side, name, pair) -> (start, end) from runs.log."""
    out, started = {}, {}
    for line in open(log, encoding="utf-8", errors="replace"):
        m = re.match(r"\[([\d/]+ [\d:]+)\] (start|end) (\w+) (\S+) (\d+):", line)
        if not m:
            continue
        key = (m.group(3), m.group(4), int(m.group(5)))
        if m.group(2) == "start":
            started[key] = stamp(m.group(1))
        elif key in started:
            out[key] = (started[key], stamp(m.group(1)))
    return out


def samples_of(path: str) -> list:
    """(time, MiB in use) from nvidia-smi's lines: timestamp, memory.used, memory.free, utilization.gpu."""
    out = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            cells = [c.strip() for c in line.lstrip("﻿").split(",")]
            if len(cells) >= 2 and re.match(r"\d{4}/\d\d/\d\d \d", cells[0]):
                out.append((stamp(cells[0]), int(cells[1].split()[0])))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--first", type=int, default=0, help="only the first N prompts (0: all)")
    ap.add_argument("--alone-below", type=int, default=0, help="MiB: a run whose samples reach it ran beside other "
                    "GPU work (0: no such test)")
    a = ap.parse_args(argv)
    runs = runs_of(os.path.join(a.dir, "runs.log"))
    samples = samples_of(os.path.join(a.dir, "sampler.log"))
    rows, answers = [], {}
    for path in sorted(glob.glob(os.path.join(a.dir, "*_*_*.json"))):
        side, rest = os.path.basename(path)[:-5].split("_", 1)
        name, pair = rest.rsplit("_", 1)
        try:
            rec = json.load(open(path, encoding="utf-8"))
        except ValueError:
            print(f"{os.path.basename(path)}: not a whole record (a run still writing it?), left out")
            continue
        per = check_dv.per_prompt(rec)
        sel = [per[i] for i in sorted(per) if not a.first or i < a.first]
        when = runs.get((side, name, int(pair)))
        during = [mib for t, mib in samples if when and when[0] <= t <= when[1]]
        peak = max(during) if during else None
        shared = bool(a.alone_below and peak is not None and peak >= a.alone_below)
        rounds = sum(r["rounds"] for r in sel)
        rows.append({"name": name, "side": side, "pair": int(pair), "rate": check_dv.decode_rate(sel),
                     "round_ms": sum(r["decode_ms"] for r in sel) / rounds if rounds else 0.0,
                     "tpf": check_dv.tpf_bifill(sel), "tpf_flyx": check_dv.tpf_flyx(sel), "n": len(sel),
                     "peak": peak, "shared": shared})
        answers.setdefault(name, {}).setdefault(side, []).append({i: r["ids"] for i, r in per.items()})
    print(f"{'config':10s} side pair prompts  tok/s  ms/round  TPF(bifill) TPF(flyx)  GPU MiB peak")
    for r in sorted(rows, key=lambda r: (r["name"], r["side"], r["pair"])):
        peak = "-" if r["peak"] is None else str(r["peak"])
        print(f"{r['name']:10s} {r['side']:4s} {r['pair']:4d} {r['n']:7d} {r['rate']:6.1f} {r['round_ms']:9.2f} "
              f"{r['tpf']:11.4f} {r['tpf_flyx']:9.4f}  {peak}{'  shared' if r['shared'] else ''}")
    print()
    for name in sorted({r["name"] for r in rows}):
        med = {}
        for side in ("a", "b"):
            alone = [r for r in rows if r["name"] == name and r["side"] == side and not r["shared"]]
            med[side] = (statistics.median(r["rate"] for r in alone), statistics.median(r["round_ms"] for r in alone),
                         len(alone)) if alone else None
        sides = answers[name]
        runs_all = [x for side in sides for x in sides[side]]
        same = all(run.get(i) == runs_all[0].get(i) for run in runs_all for i in set(run) & set(runs_all[0]))
        line = f"{name}: answers the same in every run of both sides: {'yes' if same else 'NO'}"
        if med["a"] and med["b"]:
            (ra, ma, na), (rb, mb, nb) = med["a"], med["b"]
            line += (f"; alone, median of {na} and {nb} runs: a {ra:.1f} tok/s ({ma:.2f} ms a round), b {rb:.1f} "
                     f"({mb:.2f} ms), b / a {rb / ra:.3f}" if ra else "")
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
