# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Where a DV round's time goes on the CUDA backend, from an nsys trace of DV's rounds with the backend's graphs off.

    nsys profile --trace=cuda --sample=none --cpuctxsw=none -o dv flyx bench MODEL-dv.flyx --prompts P --kv 16 \\
        --drafter dv --limit 3 --backend-opt graphs=0
    nsys export --type sqlite dv.nsys-rep      # (nsys stats writes the same dv.sqlite)
    python bench/trace/dv_round_kernels.py dv.sqlite [--linears]

A round runs from DRAFT's selector table (k_dv_selector_table, launched every round) to the canvas's slide
(k_dv_slide): the tree, the verify pass up to the accept and its compaction, then the canvas pass
(docs/design/dv.md).  A kernel is counted by the part of the round it runs in and what it does there, not by its
template: one kernel serves base weights and adapters alike.  The canvas pass's GEMMs (k_gemm_skinny_mt past 16
slots, k_gemm_tc in builds before it) come in dv_context's order (backends/cuda/regions/layers.cpp): per layer w_in,
then q, k and v's adapters (a stacked, three b), w_out, o's adapter (a, b), w_gu, gate and up's (a stacked, two b),
w_down, down's (a, b); the head last.  Kernels outside the rounds (prefills, the load) are counted apart.  Prints
milliseconds and launches a round per part (--linears: each of the canvas's linears too).
"""
from __future__ import annotations

import argparse
import collections
import sqlite3

CANVAS_LINEARS = ("w_in", "a_qkv", "b_q", "b_k", "b_v", "w_out", "a_o", "b_o", "w_gu", "a_gu", "b_gate", "b_up",
                  "w_down", "a_down", "b_down")
# Grouped (gemm_draft: past 16 slots on bf16 weights), a layer's 8 launches: a base weight with its adapter's a, then
# the adapters' b.
CANVAS_GROUPS = ("w_in+a_qkv", "b_q|b_k|b_v", "w_out+a_o", "b_o", "w_gu+a_gu", "b_gate|b_up", "w_down+a_down", "b_down")
BASE = {"w_in", "w_out", "w_gu", "w_down", "w_in+a_qkv", "w_out+a_o", "w_gu+a_gu", "w_down+a_down"}
ATTENTION = ("k_attn_prep", "k_attention", "k_flash")
HEAD = ("k_gemm_skinny_fp8", "k_cert_", "k_gemm_skinny_cols")
CANVAS_GEMMS = ("k_gemm_tc", "k_gemm_skinny_mt")  # the pass's own, not its input's (k_gemm_skinny, 16 rows at a time)
DRAFT = ("k_dv_tree_beam", "k_dv_tree_pool", "k_build_rows")
OUTSIDE = "outside the rounds (prefills, the load)"


def kernels(path):
    db = sqlite3.connect(path)
    return db.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on s.id = k.shortName order by k.start").fetchall()


def part_of(phase, name):
    """The part of a round a kernel of the round's phase (draft, verify, canvas) counts in."""
    if phase == "draft":
        return "draft: selector table, tree, its rows"
    if any(name.startswith(a) for a in ATTENTION):
        return f"{phase}: attention"
    if phase == "verify":
        if any(name.startswith(h) for h in HEAD):
            return "verify: the head (FP8, certified)"
        if name in ("k_gemm_skinny", "k_gemm_skinny_mt"):
            return "verify: the linears (bf16)"
        return "verify: GEMM input parts, norms, SwiGLU, embedding, choice, accept, compacting"
    if name in ("k_canvas_parts", "k_canvas_stats", "k_canvas_hid"):
        return "canvas: statistics (top 16, z, the selector's projection)"
    if name in ("k_gemm_skinny", "k_gemm_skinny_mt", "k_dv_glu", "k_dv_embed"):
        return "canvas: its input (the residual context's GEMMs, the embedding)"
    return "canvas: GEMM input conversions, norms, SwiGLU, slide"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", help="nsys's SQLite export of the run")
    ap.add_argument("--linears", action="store_true", help="each of the canvas's linears too")
    args = ap.parse_args(argv)
    time, count = collections.Counter(), collections.Counter()
    linear_time, linear_count = collections.Counter(), collections.Counter()
    rounds, phase, accepted, gemms, layers, embedded = 0, None, False, [], 0, False

    def add(part, ns):
        time[part] += ns
        count[part] += 1

    def end_canvas():
        add("canvas: the head (bf16 weight, float32 logits)", gemms.pop())
        names = {len(CANVAS_LINEARS): CANVAS_LINEARS, len(CANVAS_GROUPS): CANVAS_GROUPS}.get(
            len(gemms) // max(layers, 1) if len(gemms) % max(layers, 1) == 0 else 0)
        if not names:
            raise SystemExit(f"a canvas pass of {len(gemms)} GEMMs over {layers} layers before its head: not "
                             f"dv_context's {len(CANVAS_LINEARS)} or {len(CANVAS_GROUPS)} a layer")
        grouped = names is CANVAS_GROUPS
        for i, ns in enumerate(gemms):
            name = names[i % len(names)]
            linear_time[name] += ns
            linear_count[name] += 1
            if grouped:
                add("canvas: the base weights' GEMMs with their adapters' a" if name in BASE else
                    "canvas: the adapters' b GEMMs", ns)
            else:
                add("canvas: the base weights' GEMMs" if name in BASE else "canvas: the adapters' GEMMs", ns)

    passes = 0
    for start, end, name in kernels(args.trace):
        ns = end - start
        if name == "k_dv_selector_table":
            rounds, phase, accepted = rounds + 1, "draft", False
        elif phase == "draft" and name not in DRAFT:
            phase = "verify"
        elif phase == "verify" and accepted and name not in ("k_compact_kv", "k_compact_rows"):
            phase, gemms, layers, embedded = "canvas", [], 0, False
        if phase is None:
            add(OUTSIDE, ns)
        elif phase == "canvas" and name in CANVAS_GEMMS and embedded:  # the residual context's before k_dv_embed
            gemms.append(ns)
        elif phase == "canvas" and name == "k_attn_prep":
            layers += 1
            add(part_of(phase, name), ns)
        else:
            add(part_of(phase, name), ns)
        accepted |= name == "k_accept_tree"
        embedded |= phase == "canvas" and name == "k_dv_embed"
        if name == "k_dv_slide" and phase == "canvas":
            passes += 1
            end_canvas()
            phase = None
    if not rounds:
        raise SystemExit(f"{args.trace}: no DV round (k_dv_selector_table) in the trace")
    within = sum(ns for part, ns in time.items() if part != OUTSIDE)
    print(f"{rounds} rounds, {passes} canvas passes; kernels {within / rounds / 1e6:.3f} ms a round")
    for part, ns in sorted(time.items(), key=lambda x: -x[1]):
        if part != OUTSIDE:
            print(f"  {part:77s} {ns / rounds / 1e6:7.3f} ms {count[part] / rounds:7.1f} launches")
    print(f"  {OUTSIDE:77s} {time[OUTSIDE] / 1e6:7.1f} ms in all, {count[OUTSIDE]} launches")
    if args.linears:
        print("the canvas's linears, a round:")
        grouped = any(linear_count[n] for n in CANVAS_GROUPS if n not in CANVAS_LINEARS)
        for name in CANVAS_GROUPS if grouped else CANVAS_LINEARS:
            kind = "base weight" if name in BASE else "adapter"
            print(f"  {name:13s} {kind:12s} {linear_time[name] / rounds / 1e6:7.3f} ms "
                  f"{linear_count[name] / rounds:5.1f} launches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
