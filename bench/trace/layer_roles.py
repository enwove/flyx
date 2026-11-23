# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Where a decoding round's time goes, by layer and by what each kernel does in it, from a kernel trace
(bench/trace/kernel_trace.cpp) of a Qwen3.5-family model (gated delta net and attention layers) on GGUF weights.

    python3 bench/trace/layer_roles.py TRACE.tsv [--layers] [--skip 5]

The round graph launched most often is split at its 8-bit RMS norms (k_rmsnorm_q8: two a layer, before the mixer and
before the MLP): before the first, the round's start (embedding, rows); a mixer part is a GDN layer's or an
attention layer's by the kernels in it; an MLP part ends with its down projection; what follows the last layer is
the output norm, the head, the choice of tokens and the drafter's own work.  A node counts what the round's end
moved while it was the last to end (its end minus the latest end before it, idle included): with programmatic
dependent launch the kernels overlap, and these add up to the round's wall time.  Medians over the launches.
--layers: every layer's mixer and MLP, one line each.
The classifier reads demangled device symbols. For raw Windows CUPTI output, cu++filt can filter the TSV before use.
"""
from __future__ import annotations

import argparse
import collections
import re
import statistics as st

from trace_io import iter_trace_rows, short_kernel_name as short

GEMM = re.compile(r"k_round_blocks|k_gemm|k_sign_rough|gemm")


def load(path):
    launches = collections.defaultdict(list)
    for p in iter_trace_rows(path):
        if p[0] != "K" or not int(p[4]):
            continue
        launches[int(p[3])].append((int(p[1]), int(p[2]), int(p[4]), short(p[10]), int(p[6])))
    return launches


def roles(nodes):
    """(part, role, layer) for each node, in start order."""
    # A layer's two norms (mixer, MLP): rounded to 8 bits for GGUF block weights, plain for NVFP4 and bf16 ones
    # (k_rmsnorm, or k_rmsnorm_spread for a round's few rows; then the output norm and a drafter's are plain too). The target's layers start at the first norm a gated delta net
    # follows (Qwen3.5's layer 0) and go on while a mixer and an MLP alternate.
    norms = [i for i, n in enumerate(nodes) if n[3] == "k_rmsnorm_q8"] or \
        [i for i, n in enumerate(nodes) if n[3] in ("k_rmsnorm", "k_rmsnorm_spread")]
    out = [("start", "embedding, rows" if n[3].startswith(("k_embed", "k_build_rows", "k_stage_q8")) else "drafter",
            -1) for n in nodes]

    def has(a, b, prefixes):
        return any(nodes[i][3].startswith(prefixes) for i in range(a, b))
    first = next((k for k in range(len(norms) - 1) if has(norms[k], norms[k + 1], "k_gdn")), None)
    if first is None and any(n[3].startswith("_Z") for n in nodes):
        raise ValueError("layer attribution needs demangled CUDA kernel names. "
                         "Run cu++filt < TRACE.tsv > DECODED.tsv, then use DECODED.tsv.")
    if first is None:
        return out
    norms = norms[first:]
    keep = 0
    mixer = ("k_gdn", "k_attn", "k_attention", "k_strata_")
    while keep + 1 < len(norms) and has(norms[keep], norms[keep + 1], mixer):
        keep += 2
    norms = norms[:keep]
    if not norms:
        return out
    bounds = norms + [len(nodes)]
    tail_from = len(nodes)
    for j in range(len(norms)):
        a, b = bounds[j], bounds[j + 1]
        seg = range(a, b)
        names = [nodes[i][3] for i in seg]
        layer = j // 2
        if j % 2 == 0:  # the mixer
            kind = "GDN" if any(n.startswith("k_gdn") for n in names) else "attention"
            core = [i for i in seg if nodes[i][3].startswith(mixer)]
            first_core = core[0] if core else b
            for i in seg:
                n = nodes[i][3]
                if i == a:
                    r = "norm"
                elif GEMM.search(n) and i < first_core:
                    r = "in-projections" if kind == "GDN" else "q, k, v projections"
                elif GEMM.search(n):
                    r = "out projection"
                elif kind == "GDN":
                    r = "recurrence (conv, gates, state, gated norm)" if n.startswith("k_gdn") else "other"
                elif n == "k_attn_prep":
                    r = "prep (q/k norm, rope, K/V write)"
                elif n.startswith("k_attention"):
                    r = "attention combine" if n == "k_attention_combine" else "attention"
                elif n.startswith("k_strata_scan"):
                    r = "Strata scan"
                elif n.startswith("k_strata_select"):
                    r = "Strata selection"
                elif n == "k_strata_delta":
                    r = "Strata correction"
                elif n in ("k_strata_claim", "k_strata_fill"):
                    r = "Strata hot-cache lookup and fill"
                elif n == "k_strata_rotate_q":
                    r = "Strata query rotation"
                elif n == "k_strata_finish":
                    r = "Strata output"
                elif n == "k_kv_stage":
                    r = "KV staging"
                elif n.startswith("k_stage_q8"):
                    r = "stage output for o-proj"
                else:
                    r = "other"
                out[i] = (kind, r, layer)
        else:  # the MLP: norm, gate|up (one launch, or two of two types), SwiGLU, down; after the last, the tail
            swiglu, down = False, False
            for i in seg:
                n = nodes[i][3]
                if i == a:
                    out[i] = ("MLP", "norm", layer)
                elif n.startswith("k_swiglu"):
                    swiglu = True
                    out[i] = ("MLP", "SwiGLU", layer)
                elif GEMM.search(n) and not swiglu:
                    out[i] = ("MLP", "gate, up", layer)
                elif GEMM.search(n) and not down:
                    down = True
                    out[i] = ("MLP", "down", layer)
                elif down and j == len(norms) - 1:
                    tail_from = i
                    break
                else:
                    out[i] = ("MLP", "other", layer)
    head_seen = False
    for i in range(tail_from, len(nodes)):
        n = nodes[i][3]
        if n.startswith("k_rmsnorm") and not head_seen:
            r = ("output", "norm")
        elif GEMM.search(n) and not head_seen and nodes[i][4] > 100:
            r, head_seen = ("output", "head"), True
        elif n.startswith(("k_argmax", "k_sample", "k_accept", "k_build_rows", "k_head_cert", "k_hring", "k_copy")):
            r = ("output", "choose tokens, accept")
        else:
            r = ("drafter", n)
        out[i] = (r[0], r[1], -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--layers", action="store_true")
    a = ap.parse_args()
    launches = load(a.trace)
    by_graph = collections.defaultdict(list)
    for corr in sorted(launches):
        nodes = sorted(launches[corr])
        by_graph[nodes[0][2]].append(nodes)
    graph, runs = max(by_graph.items(), key=lambda kv: len(kv[1]))
    runs = runs[a.skip:]
    per_role = collections.defaultdict(list)
    per_layer = collections.defaultdict(list)
    walls, idles = [], []
    for nodes in runs:
        t0 = nodes[0][0]
        last_end, idle = t0, 0
        acc, lay = collections.Counter(), collections.Counter()
        try:
            labels = roles(nodes)
        except ValueError as exc:
            ap.exit(2, f"{ap.prog}: {exc}\n")
        for (s, e, _, name, _), (part, role, layer) in zip(nodes, labels):
            gain = max(0, e - last_end)
            idle += max(0, s - last_end)
            last_end = max(last_end, e)
            acc[(part, role)] += gain
            acc[(part, "count " + role)] += 1
            if layer >= 0:
                lay[(layer, part)] += gain
        walls.append(last_end - t0)
        idles.append(idle)
        for k, v in acc.items():
            per_role[k].append(v)
        for k, v in lay.items():
            per_layer[k].append(v)
    wall = st.median(walls) / 1e3
    print(f"graph {graph}: {len(runs)} launches, {len(runs[0])} nodes; wall {wall:.1f} us a round, "
          f"of it idle between kernels {st.median(idles) / 1e3:.1f} us")
    parts = collections.OrderedDict()
    for (part, role), v in per_role.items():
        if role.startswith("count "):
            continue
        parts.setdefault(part, []).append((st.median(v) / 1e3, role, st.median(per_role[(part, "count " + role)])))
    order = ["start", "GDN", "attention", "MLP", "output", "drafter"]
    print(f"{'us/round':>9} {'%':>6} {'n':>5} {'us each':>8}  part / role")
    for part in sorted(parts, key=lambda p: order.index(p) if p in order else 9):
        rows = sorted(parts[part], reverse=True)
        tot = sum(r[0] for r in rows)
        layers = sum(kind == part for _, kind in per_layer)
        each = f"  ({layers} {'layer' if layers == 1 else 'layers'}, {tot / layers:.1f} us a layer)" if layers else ""
        print(f"{tot:9.1f} {100 * tot / wall:6.1f}        {'':8}  {part}{each}")
        for t, role, n in rows:
            print(f"{t:9.1f} {100 * t / wall:6.1f} {n:5.0f} {t / max(n, 1):8.2f}    {role}")
    if a.layers:
        print("\nlayer  mixer (us)  MLP (us)")
        for layer in sorted({k[0] for k in per_layer}):
            mix = [(p, st.median(v) / 1e3) for (l, p), v in per_layer.items() if l == layer and p != "MLP"]
            mlp = st.median(per_layer.get((layer, "MLP"), [0])) / 1e3
            print(f"{layer:5d}  {mix[0][0][:4]:4} {mix[0][1]:6.1f}  {mlp:8.1f}" if mix else f"{layer:5d}  -  {mlp:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
