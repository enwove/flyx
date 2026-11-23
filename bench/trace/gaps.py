# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Where a round's time goes, from a kernel trace (bench/trace/kernel_trace.cpp).

    python bench/trace/gaps.py out.tsv [--bytes 3.38e9 --gbps 780] [--sequence]

Takes the graph launches (rounds) of the trace, groups them by graph (one per
round width) and reports, per graph, medians over its launches:

* wall: first kernel start to last node end;
* busy: the union of the nodes' intervals; idle = wall - busy, the time
  between kernels when nothing runs (overlapping nodes under PDL count once);
* each kernel (by name and grid) with its time per launch, and its exclusive
  time: what it adds to the wall after every node before it has ended.  Under
  PDL a kernel starts while the one before it finishes and its duration counts
  that overlap again; the exclusive times add up to busy.

--bytes / --gbps give the round's DRAM traffic and a streaming-read rate, and
add the time a single read of those bytes would take, the floor no kernel
arrangement goes under.  --sequence prints the median launch node by node
(duration, idle before it).
"""
from __future__ import annotations

import argparse
import collections
import statistics as st

from trace_io import iter_trace_rows, short_kernel_name as short


def load(path: str):
    rows = []
    for p in iter_trace_rows(path):
        kind, start, end, corr, graph = p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])
        gx, gy, gz, name = int(p[6]), int(p[7]), int(p[8]), p[10]
        label = short(name) if kind == "K" else name
        if kind == "K":
            label += f" [{gx}x{gy}x{gz}]"
        rows.append((kind, start, end, corr, graph, label))
    return rows


def launch_stats(nodes):
    nodes = sorted(nodes, key=lambda r: r[1])
    t0 = nodes[0][1]
    wall = max(r[2] for r in nodes) - t0
    busy, cur_s, cur_e = 0, nodes[0][1], nodes[0][2]
    idle_before = [0]
    excl = [nodes[0][2] - nodes[0][1]]
    for r in nodes[1:]:
        idle_before.append(max(0, r[1] - cur_e))
        excl.append(max(0, r[2] - max(r[1], cur_e)))
        if r[1] > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = r[1], r[2]
        else:
            cur_e = max(cur_e, r[2])
    busy += cur_e - cur_s
    return nodes, wall, busy, idle_before, excl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--bytes", type=float, default=0.0, help="DRAM bytes a round reads and writes")
    ap.add_argument("--gbps", type=float, default=780.0, help="streaming-read rate, GB/s")
    ap.add_argument("--min-launches", type=int, default=20)
    ap.add_argument("--skip", type=int, default=5, help="first launches of each graph to drop (warm-up)")
    ap.add_argument("--sequence", action="store_true")
    a = ap.parse_args()

    rows = load(a.trace)
    by_launch = collections.defaultdict(list)
    for r in rows:
        if r[4]:
            by_launch[r[3]].append(r)
    by_graph = collections.defaultdict(list)
    for corr in sorted(by_launch):
        nodes = by_launch[corr]
        by_graph[nodes[0][4]].append(nodes)
    print(f"{len(rows)} records, {len(by_launch)} graph launches, {len(by_graph)} graphs")

    for graph, launches in sorted(by_graph.items(), key=lambda kv: -len(kv[1])):
        launches = launches[a.skip:]
        if len(launches) < a.min_launches:
            continue
        stats = [launch_stats(n) for n in launches]
        n_nodes = st.median(len(s[0]) for s in stats)
        wall = st.median(s[1] for s in stats) / 1e3
        busy = st.median(s[2] for s in stats) / 1e3
        kern = st.median(sum(r[2] - r[1] for r in s[0]) for s in stats) / 1e3
        gaps = [sum(1 for g in s[3] if g > 0) for s in stats]
        print(f"\n== graph {graph}: {len(launches)} launches, {n_nodes:.0f} nodes")
        print(f"   wall {wall:8.1f} us   busy {busy:8.1f} us   idle {wall - busy:7.1f} us ({100 * (wall - busy) / wall:.1f}%)"
              f"   sum of node times {kern:8.1f} us   gaps {st.median(gaps):.0f}")
        # Between one launch's last node and the next launch's first: the
        # device idles unless something outside the graphs runs then.
        spans = sorted((min(r[1] for r in n), max(r[2] for r in n)) for n in launches)
        between = [b[0] - e[1] for e, b in zip(spans, spans[1:])]
        period = [b[0] - e[0] for e, b in zip(spans, spans[1:])]
        if between:
            outside = [r for r in rows if not r[4] and spans[0][0] <= r[1] <= spans[-1][1]]
            print(f"   launch to launch {st.median(period) / 1e3:8.1f} us   between launches {st.median(between) / 1e3:7.1f} us"
                  f" (p10 {sorted(between)[len(between) // 10] / 1e3:.1f}, p90 {sorted(between)[9 * len(between) // 10] / 1e3:.1f})"
                  f"   records outside graphs meanwhile: {len(outside)}")
        if a.bytes:
            floor = a.bytes / (a.gbps * 1e3)
            print(f"   one streaming read of {a.bytes / 1e9:.2f} GB at {a.gbps:.0f} GB/s: {floor:8.1f} us"
                  f" (wall is {wall / floor:.2f}x)")
        per = collections.defaultdict(list)
        for s in stats:
            acc = collections.Counter()
            exc = collections.Counter()
            cnt = collections.Counter()
            for r, x in zip(s[0], s[4]):
                acc[r[5]] += r[2] - r[1]
                exc[r[5]] += x
                cnt[r[5]] += 1
            for k in acc:
                per[k].append((acc[k], cnt[k], exc[k]))
        table = sorted(((st.median(v[2] for v in vals) / 1e3, st.median(v[0] for v in vals) / 1e3,
                         st.median(v[1] for v in vals), k) for k, vals in per.items()), reverse=True)
        print(f"   {'excl/launch':>11} {'%wall':>6} {'us/launch':>9} {'n':>4} {'excl each':>9} {'us each':>8}  kernel [grid]")
        for x, t, n, k in table:
            print(f"   {x:11.1f} {100 * x / wall:6.1f} {t:9.1f} {n:4.0f} {x / n:9.2f} {t / n:8.2f}  {k}")
        if a.sequence:
            mid = sorted(stats, key=lambda s: s[1])[len(stats) // 2]
            nodes, _, _, idle, excl = mid
            t0 = nodes[0][1]
            print(f"   median launch, node by node (start, duration, exclusive, idle before; us):")
            for r, x, g in zip(nodes, excl, idle):
                print(f"   {(r[1] - t0) / 1e3:8.1f} {(r[2] - r[1]) / 1e3:7.2f} {x / 1e3:7.2f} {g / 1e3:6.2f}  {r[5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
