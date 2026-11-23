# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Every weight matrix of a decoded token, engine against engine: which kernel ran it and for how long.

    python bench/trace/per_matrix.py MODEL.gguf NAME=TRACE.tsv[:flyx|:ggml|:matrix] ... [--roles] [--best]

Each trace is a decode under bench/trace/kernel_trace.cpp (flyx bench, llama-bench tgN, ik_llama.cpp's
llama-bench), or for `:matrix` a per-matrix table (bench/external/sgl_gguf_matrices.py analyze).  The model's
matrices are walked in the order a token runs them (layer by layer; a gated delta net's qkv, z, b, a, out; an
attention layer's q, k, v, out; the MLP's gate, up, down; last the head), and each engine's weight kernels of a
token are matched to them in that order (neighbours may run in another order: llama.cpp computes z after b and
a).  A kernel covers one matrix or a run of neighbours of its type, its time shared by their bytes:
  flyx, binary16 inputs (k_gemm_blocks): neighbours whose widths add up to its grid's columns (a stack);
  flyx, rounds in 8 bits (k_round_blocks_q8): the run of one type within a stack (its blocks of threads stay,
      so the grid does not say how many columns it took);
  ggml (llama.cpp mul_mat_vec_q, ik_llama.cpp's (fused_)mul_mat_vec_q): one matrix of one or two rows a block of
      threads, or gate and up in one kernel (the GLU).
Prints, per role and type: matrices a token, MB, and each engine's microseconds a token and GB/s; then the
totals over the matrices every engine ran.  --roles adds the same per role (the types together); --best the
total of the fastest baseline per row (every engine but the first is a baseline)."""
import argparse
import collections
import re
import statistics as st
import sys

from trace_io import GGML_TYPES as GGML, iter_trace_rows, plain_kernel_name

GDN = ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"]
ATT = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
# The matrices that share an input: flyx stacks them, a run of one type is one GEMM.
STACK = {"attn_qkv": "in", "attn_gate": "in", "ssm_beta": "in", "ssm_alpha": "in", "attn_q": "in", "attn_k": "in",
         "attn_v": "in", "ssm_out": "out", "attn_output": "out", "ffn_gate": "mlp", "ffn_up": "mlp",
         "ffn_down": "down", "head": "head"}
ROLE_GROUP = {"ffn_down": "down", "ffn_gate": "gate, up", "ffn_up": "gate, up", "attn_qkv": "GDN qkv",
              "attn_gate": "GDN z", "ssm_out": "outs", "attn_output": "outs", "attn_q": "attn q, k, v",
              "attn_k": "attn q, k, v", "attn_v": "attn q, k, v", "ssm_beta": "GDN b, a", "ssm_alpha": "GDN b, a",
              "head": "head"}


def model_order(path):
    import gguf  # the gguf package (pip install gguf)

    r = gguf.GGUFReader(path)
    t = {x.name: x for x in r.tensors}
    order, layer = [], 0
    while f"blk.{layer}.ffn_down.weight" in t:
        roles = GDN if f"blk.{layer}.attn_qkv.weight" in t else ATT
        for role in roles:
            x = t[f"blk.{layer}.{role}.weight"]
            order.append(dict(layer=layer, role=role, type=x.tensor_type.name, n=int(x.shape[1]), k=int(x.shape[0]),
                              bytes=int(x.n_bytes)))
        layer += 1
    x = t["output.weight"]
    order.append(dict(layer=layer, role="head", type=x.tensor_type.name, n=int(x.shape[1]), k=int(x.shape[0]),
                      bytes=int(x.n_bytes)))
    return order


def gemv_flyx(name, gx):
    """(type, how): ("stack", width, columns a block of threads) or ("run",)."""
    name = plain_kernel_name(name)
    m = re.search(r"k_round_blocks_q8<F(\w+),", name)
    if m:
        return m.group(1), ("run",)
    m = re.search(r"k_gemm_blocks<F(\w+), (\d+), (\d+), (\d+)>", name)
    if not m:
        return None
    cols = 8 * int(m.group(2)) * int(m.group(3))
    return m.group(1), ("stack", gx * cols, cols)


def gemv_ggml(name, gx):
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+), (\d+), (true|false)", name)
    if m:
        t, fused = int(m.group(1)), m.group(3) == "true"
    else:
        m = re.search(r"(fused_)?mul_mat_vec_q<\(ggml_type\)(\d+), (\d+), (\d+)>", name)
        if not m:
            return None
        t, fused = int(m.group(2)), m.group(1) is not None
    return GGML.get(t, str(t)), ("one", {gx, 2 * gx}, fused)


def tokens_of(path, parse, head_n, flyx):
    """The weight kernels of each token, in order: [(type, how, ns)].  flyx's tokens end at the greedy choice
    after the head (k_argmax_partial); llama-bench's at the head (the kernel as wide as the vocabulary)."""
    rows = []
    for p in iter_trace_rows(path):
        if p[0] != "K" or p[4] == "0":
            continue
        g = parse(p[10], int(p[6]))
        if g:
            rows.append((int(p[1]), g, int(p[2]) - int(p[1])))
        elif flyx and "k_argmax_partial" in p[10]:
            rows.append((int(p[1]), None, 0))
    rows.sort()
    toks, cur = [], []
    for _, g, d in rows:
        if g is None:
            if cur:
                toks.append(cur)
            cur = []
            continue
        ty, how = g
        cur.append((ty, how, d))
        if not flyx and max(how[1]) >= head_n:
            toks.append(cur)
            cur = []
    return toks


def run_at(order, out, i, ty, how):
    """The end of the run of matrices from i this kernel covers, or None."""
    if i >= len(order) or order[i]["type"] != ty or out[i]:
        return None
    if how[0] == "run":
        stack = (order[i]["layer"], STACK[order[i]["role"]])
        j = i
        while (j < len(order) and order[j]["type"] == ty and not out[j]
               and (order[j]["layer"], STACK[order[j]["role"]]) == stack):
            j += 1
        return j
    if how[0] == "stack":  # neighbours of one type whose widths add up to the kernel's, rounded up to its columns
        _, n, cols = how
        j, w = i, 0
        while j < len(order) and order[j]["type"] == ty and not out[j] and w < n:
            w += order[j]["n"]
            j += 1
            if (w + cols - 1) // cols * cols == n:
                return j
        return None
    _, widths, fused = how
    if order[i]["n"] not in widths:
        return None
    if (fused and order[i]["role"] == "ffn_gate" and i + 1 < len(order) and order[i + 1]["role"] == "ffn_up"
            and order[i + 1]["type"] == ty):
        return i + 2  # the GLU's gate and up
    return i + 1


def match(order, kernels):
    """Per matrix index, ns; None if the kernels do not follow the model's order."""
    out = [0.0] * len(order)
    i = 0
    for ty, how, d in kernels:
        for start in range(i, min(i + 4, len(order))):  # from the first matrix not yet run
            j = run_at(order, out, start, ty, how)
            if j is not None:
                break
        else:
            return None
        total = sum(order[x]["bytes"] for x in range(start, j))
        for x in range(start, j):
            out[x] += d * order[x]["bytes"] / total
        while i < len(order) and out[i]:
            i += 1
    return out if i == len(order) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("traces", nargs="+", help="NAME=TRACE.tsv[:flyx|:ggml|:matrix]")
    ap.add_argument("--roles", action="store_true")
    ap.add_argument("--best", action="store_true")
    a = ap.parse_args()
    order = model_order(a.model)
    head_n = order[-1]["n"]
    engines = []
    for arg in a.traces:
        name, spec = arg.split("=", 1)
        path, _, kind = spec.partition(":")
        if kind == "matrix":  # index -> us; nan where none ran
            us = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if p[0].isdigit():
                        us[int(p[0])] = float(p[5])
            engines.append((name, [us.get(x, float("nan")) for x in range(len(order))]))
            continue
        flyx = (kind or "ggml") == "flyx"
        toks = tokens_of(path, gemv_flyx if flyx else gemv_ggml, head_n, flyx)[2:]  # two to warm up
        per = [match(order, k) for k in toks]
        good = [p for p in per if p]
        print(f"{name}: {len(toks)} tokens, {len(good)} matched", file=sys.stderr)
        engines.append((name, [st.median(p[x] for p in good) / 1e3 for x in range(len(order))] if good else None))
    groups = collections.defaultdict(lambda: [0, 0, [0.0] * len(engines)])
    roles = collections.defaultdict(lambda: [0, [0.0] * len(engines)])
    common_b, common_t = 0, [0.0] * len(engines)
    for x, m in enumerate(order):
        g = groups[(m["role"], m["type"])]
        g[0] += 1
        g[1] += m["bytes"]
        ts = [t[x] if t else float("nan") for _, t in engines]
        for e, v in enumerate(ts):
            g[2][e] += v
        if all(v == v for v in ts):
            common_b += m["bytes"]
            common_t = [s + v for s, v in zip(common_t, ts)]
            r = roles[ROLE_GROUP[m["role"]]]
            r[0] += m["bytes"]
            r[1] = [s + v for s, v in zip(r[1], ts)]

    def cell(t, b):
        return f" | {t:13.1f} {b / (t * 1e3):5.0f}" if t == t and t else f" | {'-':>13s} {'':5s}"

    print(f"{'role':12s} {'type':8s} {'n':>3s} {'MB':>7s}" + "".join(f" | {n[:13]:>13s}  GB/s" for n, _ in engines))
    for (role, ty), (n, b, ts) in sorted(groups.items(), key=lambda kv: -kv[1][2][0]):
        print(f"{role:12s} {ty:8s} {n:3d} {b / 1e6:7.1f}" + "".join(cell(t, b) for t in ts))
    print(f"{'all run by every engine':25s} {common_b / 1e6:7.1f}" + "".join(cell(t, common_b) for t in common_t))
    if a.roles:
        print()
        for role, (b, ts) in sorted(roles.items(), key=lambda kv: -kv[1][1][0]):
            print(f"{role:25s} {b / 1e6:7.1f}" + "".join(cell(t, b) for t in ts))
    if a.best and len(engines) > 2:
        wins, best = collections.Counter(), 0.0
        for (role, ty), (n, b, ts) in groups.items():
            others = [(t if t == t else float("inf"), e) for e, t in enumerate(ts) if e > 0]
            t, e = min(others)
            wins[engines[e][0]] += 1
            best += t
        print(f"\nflyx baseline per row: {dict(wins)}; their total {best:.0f} us a token, "
              f"{engines[0][0]} {sum(g[2][0] for g in groups.values()):.0f} us")


if __name__ == "__main__":
    main()
