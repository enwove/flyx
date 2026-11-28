# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Tuning tables -> C++ (run by the build; the output is not committed).

    python tools/codegen/gen_gemm_tables.py --candidates backends/cuda/variants/candidates.toml \\
        --out build/generated/flyx/gemm_tables.inc backends/cuda/variants/*/tables/*.toml

Input: the compiled kernel configurations (candidates.toml) with the rules every
GPU without a table of its own takes (its [defaults]), and one table per GPU
model.  A table names the GPU (``sku``, ``devices``: CUDA device names), may
``inherit`` another sku's table (a GPU with the same chip), lists per GEMM shape
which configuration serves which row counts:

    sku = "rtx5070ti"
    devices = ["NVIDIA GeForce RTX 5070 Ti"]
    [[gemm]]
    format = "nvfp4"          # or "bf16", or "w12" (bf16 weights in W12: the bf16 configurations)
    n = 18432
    k = 2560
    rules = [{ rows = [1, 8], span = 128, chunks = 4, nt = 1, warps = 4, unroll = 4 }, ...]

and may carry, in its [rules], the GPU's own rules of any family the defaults
have (FAMILIES: written as the defaults write them, checked the same way).  A
lookup takes the rules of a family from the GPU's table, else from the nearest
table it inherits that has them, else the defaults: a family's rules come whole
from one place. A GPU's blocks_round list can end with `{ defaults = true }`
to append the default round rules after its measured type-specific exceptions.
A table's optional [[flash_gqa]] entries select heads per block
for FP8 attention by the exact head shape, row count and staged key-window length.
[[strata_placement]] entries spread the resident low-bit copies across the
interior layers of a measured host-to-device transfer layout.

A table may also serve the GPUs of an architecture that have no table of their
own: ``architectures = ["8.9"]``, compute capabilities, each in one table at most
. A GPU's own table, matched by its name, comes first.

A table may also say how kernels launch on its GPU: ``pdl = true``, programmatic
dependent launch unless the backend's options say otherwise (a
table that inherits takes its parent's unless it says).  It never changes the
bits: a kernel reads what the one before wrote only after waiting for it.

Checked here: every rule names a compiled configuration; the rules of a shape
cover rows 1..16 once; an NVFP4 shape uses one K partition (span, chunks) for
all its rules, because the partition decides the bits and plain decoding must
match verification exactly.

bf16 GEMMs of 17..64 rows run on k_gemm_skinny_mt (``bf16_mt``: tile, chunks,
unroll, input parts), with rules of their own by the rows and, per shape, as
the formats ``bf16_rows`` (a round's: three parts and 16 chunks, each row the
bits the 1..16-row kernel gives it, so every such configuration gives the same
bits) and ``bf16_draft`` (a drafting pass's: one part, bf16 inputs); each
covers rows 17..64 once.  A drafting pass's chunks change its sums, so they are
not a table's to choose: K alone fixes them (``draft_chunks``, the same in every
table and at every row count), so the weights one launch takes on the same rows
(one K) each get the sums of a launch of their own, and a bf16_draft rule names
only the tile and the unroll (``nt``, ``unroll``).

The 1..16-row kernel of GGML block weights (``blocks_skinny``: tiles a warp,
warps that share a window, runs of K at once) has its configurations here and
rules by a layer's columns; every one of them gives the same bits.  So has
their kernel for passes of more rows with inputs in 8 bits (``blocks_q8``:
columns a block of threads, values of K a tile of rows); its runs of K
(``blocks_q8_runs``) are a K partition, one per shape by the layer's columns and K.
Rounds with inputs in 8 bits on dp4a (``blocks_round``: the rows the kernel
holds, columns a warp takes at once, groups a lane loads ahead, warps a block,
and whether a block copies the round's rows to shared memory first) have rules by
block type, the round's rows and the layer's columns, which may name one of the
three tensor-core kernels instead (``tensor_cores``); every one gives the same
bits too.

Prefill GEMMs on the bf16 tensor cores (``prefill_tc``: block tile rows and
columns, warps along the columns, pipeline stages) have rules by the weights'
format, the pass's rows and the GEMM's 64 x 128 tiles per SM; every one gives
the same bits.

MoE layers' packed expert GEMMs on dp4a (``moe_dp4a``: output columns a warp,
units a lane has in flight; 0 and 0, the original tiles) have rules for their
fused gate|up GEMM (``moe_gate_up``) by block type and the pass's rows; the
original tiles are the one configuration every block type compiles, the others
are compiled for MOE_DP4A_TYPES alone, so only the rules naming those types may
pick another.  Every one gives the same bits.

Output: X-macros listing the configurations (template instances), the sku
list with parents and the families each table has, and the rules.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib

BF16_KEYS = ("nt", "warps", "unroll", "split")
# split: the blocks a column tile's chunks are shared by (k_gemm_skinny_split; 1, the one block of k_gemm_skinny), for
# bf16 weights by their shape's own rules (the scratch holds the chunks' sums of the widest, Sku::split_columns).


def bf16_key(d: dict) -> tuple:
    """A 1..16-row bf16 configuration: (tile, warps, unroll, blocks a tile's chunks are shared by)."""
    return key({"split": 1, **d}, BF16_KEYS)
MT_KEYS = ("nt", "chunks", "unroll", "parts")
FP4_KEYS = ("span", "chunks", "nt", "warps", "unroll")
BLOCKS_KEYS = ("nt", "nw", "kw")
Q8_KEYS = ("bn", "qk", "bm")
# Prefill GEMMs on the bf16 tensor cores (k_gemm_tc): block tile rows and columns, warps along the columns, stages.
TC_KEYS = ("bm", "bn", "wn", "stages")
TC_SMEM = 99 * 1024  # an SM's dynamic shared memory for one block (sm_86 and up: 99 KB)
ROUND_KEYS = ("rows", "columns", "ahead", "warps", "staged", "parts")
ROUND_DEFAULTS = {"warps": 8, "staged": 0, "parts": 1}  # a block of 8 warps; the rows read from global memory
# The types k_round_blocks_warps takes (gemm_blocks_round_impl.cuh: WarpsType).
WARPS_TYPES = frozenset({"IQ4_XS", "Q4_K", "Q5_K", "Q6_K"})
ROUND_ROWS = (1, 2, 4, 8, 16)  # the rows a dp4a round kernel holds; a round runs the fewest that take its rows


def round_key(d: dict) -> tuple:
    """A dp4a round configuration: (rows held, columns a warp, groups ahead, warps a block, rows staged in shared
    memory: 0 or 1, warps that share a column set, each its part of the rows)."""
    return key({**ROUND_DEFAULTS, **d}, ROUND_KEYS)
# GGML block types by name: FLYX_DT_GGML_<name> in include/flyx/backend_abi.h
GGML_TYPES = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS",
              "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "MXFP4", "Q2_0")
# The families of rules: bit i of a sku's `families` is FAMILIES[i]; the keys each takes in [defaults] or [rules].
FAMILIES = {"bf16": ("bf16",), "nvfp4": ("nvfp4_k128", "nvfp4_k64"), "blocks": ("blocks",),
            "blocks_q8": ("blocks_q8",), "blocks_q8_runs": ("blocks_q8_runs",), "blocks_round": ("blocks_round",),
            "bf16_rows": ("bf16_rows",), "bf16_draft": ("bf16_draft",), "moe_gate_up": ("moe_gate_up",),
            "prefill_tc": ("prefill_tc",)}
# MoE layers' packed expert dp4a GEMMs (moe_dp4a): output columns a warp, units a lane has in flight.
MOE_KEYS = ("columns", "units")
MOE_ORIGINAL = (0, 0)  # the original tiles, which every block type compiles
# The types every moe_dp4a configuration is compiled for, the others the original tiles alone (kernels.cuh:
# kMoeDp4aTypes).
MOE_DP4A_TYPES = frozenset({"IQ3_XXS", "IQ2_S", "IQ2_XXS", "IQ2_XS", "IQ4_NL", "Q4_K"})
# bf16 GEMMs of 17..64 rows on k_gemm_skinny_mt (bf16_mt): a round's in three parts, a drafting pass's in one.
MT_PARTS = {"bf16_rows": 3, "bf16_draft": 1}
MT_ROWS = (17, 64)
DRAFT_KEYS = ("nt", "unroll")  # a bf16_draft rule's: its chunks are K's, its parts 1
# A drafting pass's K partition: 8 chunks, 16 from K = 4096, so that a warp has 8 spans of 32 or more where K has them
# (at 32 rows MiniCPM5-1B's down GEMM, K = 4608, takes 27% longer in 8 chunks, its GEMMs over K = 1536
# and 2048 3 to 10% longer in 16).
DRAFT_LONG_K = 4096
DRAFT_CHUNKS = (8, 16)  # below DRAFT_LONG_K, from it
INF = 2**31 - 1


def draft_chunks(k: int) -> int:
    """The chunks of a drafting pass's GEMM over K (bf16_draft): by K alone, in every table."""
    return DRAFT_CHUNKS[k >= DRAFT_LONG_K]


def key(d: dict, keys) -> tuple:
    return tuple(int(d[k]) for k in keys)


class Compiled:
    """The configurations candidates.toml compiles, checked."""

    def __init__(self, cand: dict, where: str):
        self.bf16 = [bf16_key(c) for c in cand["bf16_skinny"]]
        self.fp4 = [key(c, FP4_KEYS) for c in cand["nvfp4_skinny"]]
        for c in self.fp4:
            span, chunks, nt, warps, _ = c
            if span not in (64, 128) or chunks % warps:
                raise SystemExit(f"{where}: bad nvfp4 configuration {c}")
        for c in self.bf16:
            if c[3] not in (1, 2, 4, 8, 16) or 16 % (c[1] * c[3]):
                raise SystemExit(f"{where}: bf16 warps times split must divide 16 (split 1, 2, 4, 8 or 16): {c}")
        if len(set(self.bf16)) != len(self.bf16):
            raise SystemExit(f"{where}: a bf16_skinny configuration listed twice")
        self.mt = [key(c, MT_KEYS) for c in cand["bf16_mt"]]
        for nt, chunks, unroll, parts in self.mt:
            if (nt not in (1, 2, 4) or parts not in (1, 3) or unroll < 1 or (parts == 3 and chunks != 16)
                    or (parts == 1 and chunks not in DRAFT_CHUNKS)):
                raise SystemExit(f"{where}: bf16_mt takes nt 1, 2 or 4, parts 3 with chunks 16 (the 1..16-row "
                                 f"kernel's) or parts 1 with chunks {' or '.join(map(str, DRAFT_CHUNKS))} (a drafting "
                                 f"pass's partitions): {(nt, chunks, unroll, parts)}")
        if len(set(self.mt)) != len(self.mt):
            raise SystemExit(f"{where}: a bf16_mt configuration listed twice")
        if len(self.mt) > 32:
            raise SystemExit(f"{where}: at most 32 bf16_mt configurations (kReachBf16Mt holds one bit each)")
        self.blocks = [key(c, BLOCKS_KEYS) for c in cand["blocks_skinny"]]
        for nt, nw, kw in self.blocks:
            if nt not in (1, 2, 4) or kw not in (1, 2, 4) or not 1 <= nw * kw <= 32:
                raise SystemExit(f"{where}: bad blocks configuration {(nt, nw, kw)}")
        self.q8 = [key({"bm": 128, **c}, Q8_KEYS) for c in cand["blocks_q8"]]
        # bm rows a tile, qk values of K: its pieces of 16 values shared evenly by the block's 4 x bn threads
        if any(bn not in (64, 128) or qk not in (64, 128, 256) or bm not in (64, 128) or bm * (qk // 16) % (4 * bn)
               for bn, qk, bm in self.q8):
            raise SystemExit(f"{where}: blocks_q8 takes bn 64 or 128, qk 64, 128 or 256 and bm 64 or 128, a "
                             f"tile of rows the threads share evenly; got {self.q8}")
        self.rounds = [round_key(c) for c in cand["blocks_round"]]
        if (len(set(self.rounds)) != len(self.rounds)
                or any(r not in ROUND_ROWS or c not in (0, 1, 2, 4) or u not in (0, 1, 2, 4) or w not in (8, 16)
                       or s not in (0, 1, 2) or p not in (1, 2, 4) or r % p or w % p
                       for r, c, u, w, s, p in self.rounds)):
            raise SystemExit(f"{where}: blocks_round takes rows 1, 2, 4, 8 or 16, columns 0, 1, 2 or 4, ahead "
                             f"0, 1, 2 or 4, warps 8 or 16, staged 0, 1 or 2 (windows) and parts 1, 2 or 4 dividing rows and "
                             f"warps, each configuration once; got {self.rounds}")
        self.moe = [key(c, MOE_KEYS) for c in cand["moe_dp4a"]]
        if (len(set(self.moe)) != len(self.moe) or MOE_ORIGINAL not in self.moe
                or any(m != MOE_ORIGINAL and (m[0] not in (1, 2, 4) or m[1] not in (1, 2)) for m in self.moe)):
            raise SystemExit(f"{where}: moe_dp4a takes the original tiles (columns 0, units 0) and columns 1, 2 or 4 "
                             f"with units 1 or 2, each configuration once; got {self.moe}")
        self.tc = [key(c, TC_KEYS) for c in cand["prefill_tc"]]
        for bm, bn, wn, st in self.tc:
            threads = 32 * (bm // 64) * wn
            if (bm not in (64, 128) or bn not in (16, 32, 64, 128) or bn % (16 * wn) or bm * 4 % threads
                    or bn * 4 < threads or bn * 4 % threads or bn > threads or not 2 <= st <= 8 or st * (bm + bn) * 64 > TC_SMEM):
                raise SystemExit(f"{where}: prefill_tc takes bm 64 or 128, bn 16..128 with warp tiles of an even "
                                 f"number of 8 columns, threads that share the tile's copies evenly and 2..8 stages "
                                 f"within {TC_SMEM} bytes; got {(bm, bn, wn, st)}")
        if len(set(self.tc)) != len(self.tc) or not self.tc:
            raise SystemExit(f"{where}: prefill_tc lists each configuration once")
        for fam, items in (("bf16", self.bf16), ("fp4", self.fp4), ("blocks", self.blocks),
                           ("blocks_q8", self.q8), ("rounds", self.rounds), ("moe_dp4a", self.moe)):
            if len(items) >= 32:
                raise SystemExit(f"{where}: {fam}: {len(items)} candidates exceeds 32-bit reachability mask")


def bf16_rules(rules, where, c: Compiled):
    """Rules by the rows alone, for bf16 (and W12) shapes no table lists: (lo, hi, cfg), rows 1..16 once."""
    out, covered = [], [0] * 17
    for r in rules:
        lo, hi = r["rows"]
        k = bf16_key(r)
        if not 1 <= lo <= hi <= 16 or k not in c.bf16 or k[3] != 1:
            raise SystemExit(f"{where}: a bf16 rule takes rows within 1..16 and a compiled configuration of split 1 "
                             f"(a split one serves its shapes' own rules): {r}")
        for i in range(lo, hi + 1):
            covered[i] += 1
        out.append((lo, hi, c.bf16.index(k)))
    if covered[1:] != [1] * 16:
        raise SystemExit(f"{where}: the bf16 rules cover rows 1..16 once")
    return out


def mt_rules(rules, where, c: Compiled, family: str, k_shape: int | None = None):
    """bf16 GEMMs of 17..64 rows, rows 17..64 once.  bf16_rows (a round's, three parts): (lo, hi, cfg) of bf16_mt
    configurations of three parts.  bf16_draft (a drafting pass's, one part): nt and unroll, the chunks K's
    (draft_chunks): (lo, hi, cfg) for a shape's K, k_shape; for shapes no table lists, (lo, hi, cfg, cfg_long), the
    configurations of both partitions."""
    lo_all, hi_all = MT_ROWS
    out, covered = [], [0] * (hi_all + 1)
    for r in rules:
        lo, hi = r["rows"]
        if not lo_all <= lo <= hi <= hi_all:
            raise SystemExit(f"{where}: a {family} rule takes rows within {lo_all}..{hi_all}: {r}")
        if family == "bf16_draft":
            if set(r) != {"rows", *DRAFT_KEYS}:
                raise SystemExit(f"{where}: a bf16_draft rule names rows, nt and unroll alone (a drafting pass's "
                                 f"chunks are K's, draft_chunks; its parts 1): {r}")
            nt, unroll = key(r, DRAFT_KEYS)
            chunks = DRAFT_CHUNKS if k_shape is None else (draft_chunks(k_shape),)
            configs = [(nt, ch, unroll, 1) for ch in chunks]
            if any(m not in c.mt for m in configs):
                raise SystemExit(f"{where}: no compiled bf16_mt configuration (nt, chunks, unroll, parts) "
                                 f"{' and '.join(map(str, configs))} for the bf16_draft rule {r}"
                                 + (" (shapes no table lists: both partitions)" if k_shape is None else ""))
            cfg = tuple(c.mt.index(m) for m in configs)
        else:
            k = key(r, MT_KEYS)
            if k not in c.mt or k[3] != MT_PARTS[family]:
                raise SystemExit(f"{where}: a {family} rule takes a compiled bf16_mt configuration of "
                                 f"{MT_PARTS[family]} parts: {r}")
            cfg = (c.mt.index(k),)
        for i in range(lo, hi + 1):
            covered[i] += 1
        out.append((lo, hi, *cfg))
    if covered[lo_all:] != [1] * (hi_all - lo_all + 1):
        raise SystemExit(f"{where}: the {family} rules cover rows {lo_all}..{hi_all} once")
    return out


def nvfp4_rules(d, where, c: Compiled):
    """The configuration of NVFP4 shapes no table lists: (K % 128 == 0, otherwise), one each, so one partition."""
    out = []
    for name, k_mult in (("nvfp4_k128", 128), ("nvfp4_k64", 64)):
        k = key(d[name], FP4_KEYS)
        if k not in c.fp4 or k_mult % k[0]:
            raise SystemExit(f"{where}: {name} is a compiled configuration whose span divides {k_mult}: {d[name]}")
        out.append(c.fp4.index(k))
    return tuple(out)


def blocks_rules(rules, where, c: Compiled):
    """The 1..16-row kernel of block weights by the layer's columns: (n_max, lo, hi, cfg)."""
    out, covered = [], {}
    for r in rules:
        n_max, (lo, hi) = int(r.get("n_max", INF)), r["rows"]
        k = key(r, BLOCKS_KEYS)
        if k not in c.blocks or not 1 <= lo <= hi <= 16:
            raise SystemExit(f"{where}: a blocks rule takes rows within 1..16 and a compiled configuration: {r}")
        out.append((n_max, lo, hi, c.blocks.index(k)))
        seen = covered.setdefault(n_max, [0] * 17)
        for i in range(lo, hi + 1):
            seen[i] += 1
    widths = [n for n, _, _, _ in out]
    if widths != sorted(widths) or widths[-1] != INF or any(v[1:] != [1] * 16 for v in covered.values()):
        raise SystemExit(f"{where}: the blocks rules go by rising n_max, end with rules for the rest, and cover rows "
                         "1..16 once per width")
    return out


def q8_rules(rules, where, c: Compiled):
    """Passes of more than 16 rows with inputs in 8 bits: the first rule that matches the block types it names (none:
    every other type), the pass's rows (17 and more) and the layer's columns.  Rules naming types come first, and
    each list of types, like the rules for every other type, ends with a rule for the rest, so a type is served by
    its own rules alone.  (types, lo, hi, n_max, cfg)."""
    out, groups, generic = [], [], False
    for r in rules:
        types = tuple(r.get("types", ()))
        if any(t not in GGML_TYPES for t in types):
            raise SystemExit(f"{where}: blocks_q8 rules name unknown block types {types}")
        lo, hi = r.get("rows", [17, INF])
        n_max = int(r.get("n_max", INF))
        k = key({"bm": 128, **r}, Q8_KEYS)
        if not 17 <= lo <= hi or k not in c.q8:
            raise SystemExit(f"{where}: a blocks_q8 rule takes passes of 17 rows and more and a compiled "
                             f"configuration: {r}")
        if types and generic:
            raise SystemExit(f"{where}: blocks_q8 rules that name types come before the others")
        generic = generic or not types
        if not groups or groups[-1][0] != types:
            groups.append((types, []))
        groups[-1][1].append((lo, hi, n_max))
        out.append((types, lo, hi, n_max, c.q8.index(k)))
    named = [t for types, _ in groups for t in types]
    if (not generic or len(named) != len(set(named)) or len(groups) != len({types for types, _ in groups})
            or any(rs[-1] != (17, INF, INF) for _, rs in groups)):
        raise SystemExit(f"{where}: each list of types in the blocks_q8 rules, and the rules for every other type, "
                         "are consecutive and end with a rule for the rest; a type is named once")
    return out


def q8_runs_rules(rules, where):
    """The runs of K a pass in 8 bits adds a layer up in: a K partition, so one per shape (its columns and K):
    (n_max, k_max, runs)."""
    out = [(int(r.get("n_max", INF)), int(r.get("k_max", INF)), int(r["runs"])) for r in rules]
    shapes = [(n, k) for n, k, _ in out]
    ends = {n: k for n, k in shapes}  # the last k_max of each n_max
    if (not out or shapes != sorted(set(shapes)) or shapes[-1] != (INF, INF) or any(k != INF for k in ends.values())
            or any(not 1 <= r <= 64 for _, _, r in out)):
        raise SystemExit(f"{where}: the blocks_q8_runs rules go by rising n_max, then k_max, end each n_max and the "
                         "list with a rule for the rest, and take 1..64 runs")
    return out


def round_rules(rules, where, c: Compiled, default_rounds=None):
    """Rounds of 1..16 rows with inputs in 8 bits: the first rule that matches the type, the round's rows and the
    layer's columns (more than n_min, at most n_max).  Rules naming types come first and are exceptions: where none
    of a type's matches, the rules for every type serve it, and those cover rows 1..16 at every width.  A rule's
    kernel: dp4a in a configuration (the rows the kernel holds, columns a warp takes at once, groups a lane loads
    ahead per column, 0: by the type's unit), whose rows are the fewest that take the rule's or `kernel` (more: the
    same bits); or the integer tensor cores (`tensor_cores`: "rows", the rows in the MMA's first operand,
    k_round_blocks_mma; "columns", the columns there, k_round_blocks_cols; "split", the rows there and the sums'
    streams split among blocks of threads, k_round_blocks_split; "split_columns", the same with the columns in the
    MMA's first operand, rounds of up to 8 rows; "warps", the columns there and the streams split among a block's
    warps, k_round_blocks_warps, for the types it takes: WARPS_TYPES).
    (types, lo, hi, n_min, n_max, cfg or -1, kernel: 0 dp4a, 1 rows, 2 columns, 3 split, 4 split columns, 5 warps)."""
    held = lambda rows: next(r for r in ROUND_ROWS if r >= rows)  # noqa: E731
    engines = {None: 0, "rows": 1, "columns": 2, "split": 3, "split_columns": 4, "warps": 5}
    out, generic = [], False
    for i, r in enumerate(rules):
        if "defaults" in r:
            if (set(r) != {"defaults"} or r["defaults"] is not True or default_rounds is None or
                    i != len(rules) - 1 or generic):
                raise SystemExit(f"{where}: {{ defaults = true }} must end a GPU's type-specific blocks_round rules")
            # The default family has already passed the same coverage and configuration checks.
            out.extend(default_rounds)
            break
        types = tuple(r.get("types", ()))
        lo, hi = r["rows"]
        n_min, n_max = int(r.get("n_min", 0)), int(r.get("n_max", INF))
        engine = r.get("tensor_cores")
        kernel = int(r.get("kernel", held(hi)))
        if (any(t not in GGML_TYPES for t in types) or not 1 <= lo <= hi <= 16 or held(lo) != held(hi)
                or kernel not in ROUND_ROWS or kernel < hi or not 0 <= n_min < n_max or engine not in engines
                or (engine == "split_columns" and hi > 8)
                or (engine == "warps" and (not types or any(t not in WARPS_TYPES for t in types)))
                or (engine is not None and any(k in r for k in ("kernel", "columns", "ahead", "warps", "staged", "parts")))):
            raise SystemExit(f"{where}: a blocks_round rule names known types, rows within one kernel's (1, 2, 3-4, "
                             f"5-8, 9-16), a kernel that holds them and n_min < n_max, and either a dp4a "
                             f"configuration or tensor_cores = \"rows\", \"columns\", \"split\", "
                             f"\"split_columns\" (at most 8 rows) or \"warps\" (types {sorted(WARPS_TYPES)}): {r}")
        cfg = -1
        if engine is None:
            k = round_key({**r, "rows": kernel})
            if k not in c.rounds:
                raise SystemExit(f"{where}: blocks_round rule {r}: configuration {k} is not compiled")
            cfg = c.rounds.index(k)
        if types and generic:
            raise SystemExit(f"{where}: blocks_round rules that name types come before the others")
        generic = generic or not types
        out.append((types, lo, hi, n_min, n_max, cfg, engines[engine]))
    # The rules for every type: each round's rows and every width reach one (the first that matches).
    rest = [(lo, hi, n_min, n_max) for types, lo, hi, n_min, n_max, _, _ in out if not types]
    for rows in range(1, 17):
        # a gap between the rules' widths starts at 1 or just past some rule's n_max
        for n in sorted({1, INF} | {v for *_, n_min, n_max in rest for v in (n_min + 1, n_max, n_max + 1) if v <= INF}):
            if not any(lo <= rows <= hi and n_min < n <= n_max for lo, hi, n_min, n_max in rest):
                raise SystemExit(f"{where}: no blocks_round rule for every type takes {rows} rows on a layer of {n} "
                                 "columns")
    return out


def moe_rules(rules, where, c: Compiled):
    """MoE layers' fused gate|up GEMMs on dp4a: the first rule that matches the block types it names (none: every
    other type) and the pass's rows (1 and more).  Rules naming types come first, and each list of types, like the
    rules for every other type, ends with a rule for the rest, so a type is served by its own rules alone.  A rule
    that serves a type outside MOE_DP4A_TYPES (each rule for every other type among them) takes the original tiles,
    the one configuration those types compile.  (types, lo, hi, cfg)."""
    out, groups, generic = [], [], False
    for r in rules:
        types = tuple(r.get("types", ()))
        if (any(t not in GGML_TYPES for t in types) or not set(MOE_KEYS) <= set(r)
                or set(r) - {"types", "rows", *MOE_KEYS}):
            raise SystemExit(f"{where}: a moe_gate_up rule names known block types, maybe rows, and columns and units: "
                             f"{r}")
        lo, hi = r.get("rows", [1, INF])
        k = key(r, MOE_KEYS)
        if not 1 <= lo <= hi or k not in c.moe:
            raise SystemExit(f"{where}: a moe_gate_up rule takes rows from 1 on and a compiled moe_dp4a "
                             f"configuration: {r}")
        if k != MOE_ORIGINAL and (not types or any(t not in MOE_DP4A_TYPES for t in types)):
            raise SystemExit(f"{where}: moe_gate_up rule {r}: only {', '.join(sorted(MOE_DP4A_TYPES))} compile "
                             "configurations beside the original tiles (columns 0, units 0)")
        if types and generic:
            raise SystemExit(f"{where}: moe_gate_up rules that name types come before the others")
        generic = generic or not types
        if not groups or groups[-1][0] != types:
            groups.append((types, []))
        groups[-1][1].append((lo, hi))
        out.append((types, lo, hi, c.moe.index(k)))
    named = [t for types, _ in groups for t in types]
    if (not generic or len(named) != len(set(named)) or len(groups) != len({types for types, _ in groups})
            or any(rs[-1] != (1, INF) for _, rs in groups)):
        raise SystemExit(f"{where}: each list of types in the moe_gate_up rules, and the rules for every other type, "
                         "are consecutive and end with a rule for the rest (no rows); a type is named once")
    return out


def tc_rules(rules, where, c: Compiled):
    """Prefill GEMMs: the first rule that matches the weights' format (bf16 or nvfp4; none: both), the pass's rows
    and the GEMM's 64 x 128 tiles per SM (below sm_tiles_below; none: any).  Each format ends at a rule for the
    rest.  (fmt: -1 both, 0 bf16, 1 nvfp4; lo, hi, tiles_below, cfg)."""
    out, ends = [], set()
    for r in rules:
        fmt = {None: -1, "bf16": 0, "nvfp4": 1}.get(r.get("format"), -2)
        lo, hi = r.get("rows", [1, INF])
        below = float(r.get("sm_tiles_below", 0))
        k = key(r, TC_KEYS) if set(TC_KEYS) <= set(r) else None
        if (fmt == -2 or not 1 <= lo <= hi or ("sm_tiles_below" in r and below <= 0) or k not in c.tc
                or set(r) - {"format", "rows", "sm_tiles_below", *TC_KEYS}):
            raise SystemExit(f"{where}: a prefill_tc rule takes a format (bf16 or nvfp4), rows from 1, "
                             f"sm_tiles_below > 0 and a compiled configuration: {r}")
        fmts = {0, 1} if fmt < 0 else {fmt}
        if fmts <= ends:
            raise SystemExit(f"{where}: a prefill_tc rule after the rules for the rest of its formats: {r}")
        if (lo, hi) == (1, INF) and "sm_tiles_below" not in r:
            ends |= fmts
        out.append((fmt, lo, hi, below, c.tc.index(k)))
    if ends != {0, 1}:
        raise SystemExit(f"{where}: the prefill_tc rules end, for bf16 and for nvfp4, with a rule for the rest")
    return out


def family_rules(d: dict, where: str, c: Compiled, every: bool, default_rounds=None) -> dict:
    """The families of rules in `d` (candidates.toml's [defaults]: every one; a table's [rules]: any), checked."""
    known = {k for keys in FAMILIES.values() for k in keys}
    if set(d) - known:
        raise SystemExit(f"{where}: unknown rules {sorted(set(d) - known)}; the families are {sorted(known)}")
    out = {}
    for family, keys in FAMILIES.items():
        present = [k in d for k in keys]
        if not any(present) and not every:
            continue
        if not all(present):
            raise SystemExit(f"{where}: the {family} rules take {', '.join(keys)}")
        if family == "bf16":
            out[family] = bf16_rules(d["bf16"], where, c)
        elif family == "nvfp4":
            out[family] = nvfp4_rules(d, where, c)
        elif family == "blocks":
            out[family] = blocks_rules(d["blocks"], where, c)
        elif family == "blocks_q8":
            out[family] = q8_rules(d["blocks_q8"], where, c)
        elif family == "blocks_q8_runs":
            out[family] = q8_runs_rules(d["blocks_q8_runs"], where)
        elif family in MT_PARTS:
            out[family] = mt_rules(d[family], where, c, family)
        elif family == "moe_gate_up":
            out[family] = moe_rules(d["moe_gate_up"], where, c)
        elif family == "prefill_tc":
            out[family] = tc_rules(d["prefill_tc"], where, c)
        else:
            out[family] = round_rules(d["blocks_round"], where, c, default_rounds)
    return out


def shape_rules(path, t, sid, c: Compiled):
    """A table's rules per GEMM shape: (sku, fmt, n, k, lo, hi, cfg); the formats bf16_rows and bf16_draft (fmt 3
    and 4: 17..64 rows, bf16_mt configurations) as their families' rules have them."""
    rules = []
    for g in t.get("gemm", []):
        fmt = {"bf16": 0, "nvfp4": 1, "w12": 2, "bf16_rows": 3, "bf16_draft": 4}[g["format"]]
        if fmt >= 3:
            where = f"{path}: {g['format']} {g['n']}x{g['k']}"
            rules += [(sid, fmt, g["n"], g["k"], lo, hi, cfg)
                      for lo, hi, cfg in mt_rules(g["rules"], where, c, g["format"], g["k"] if fmt == 4 else None)]
            continue
        covered = [0] * 17
        parts = set()
        for r in g["rules"]:
            lo, hi = r["rows"]
            if not 1 <= lo <= hi <= 16:
                raise SystemExit(f"{path}: rows {r['rows']} outside 1..16")
            for i in range(lo, hi + 1):
                covered[i] += 1
            if fmt != 1:
                k = bf16_key(r)
                cfg = c.bf16.index(k) if k in c.bf16 else -1
                if cfg >= 0 and k[3] > 1 and fmt != 0:
                    raise SystemExit(f"{path}: {g['format']} {g['n']}x{g['k']}: a split configuration serves bf16 "
                                     f"weights")
            else:
                k = key(r, FP4_KEYS)
                cfg = c.fp4.index(k) if k in c.fp4 else -1
                parts.add(k[:2])
                if g["k"] % k[0]:
                    raise SystemExit(f"{path}: span {k[0]} does not divide K={g['k']}")
            if cfg < 0:
                raise SystemExit(f"{path}: {g['format']} {g['n']}x{g['k']} rows {r['rows']}: configuration "
                                 f"{ {k: r[k] for k in (FP4_KEYS if fmt == 1 else BF16_KEYS) if k in r} } is not in candidates")
            rules.append((sid, fmt, g["n"], g["k"], lo, hi, cfg))
        if covered[1:] != [1] * 16:
            raise SystemExit(f"{path}: {g['format']} {g['n']}x{g['k']}: rules must cover rows 1..16 exactly once")
        if len(parts) > 1:
            raise SystemExit(f"{path}: nvfp4 {g['n']}x{g['k']}: one K partition per shape, got {sorted(parts)}")
    return rules


def flash_gqa_rules(path, table, sid):
    rules = []
    fields = ("nh", "nkv", "hd", "rows", "keys", "hb")
    for rule in table.get("flash_gqa", []):
        if set(rule) != {"fp8", *fields} or rule["fp8"] is not True:
            raise SystemExit(f"{path}: flash_gqa takes fp8 = true and {', '.join(fields)}")
        if any(type(rule[f]) is not int or not 0 < rule[f] <= INF for f in fields):
            raise SystemExit(f"{path}: flash_gqa dimensions and hb must be positive integers")
        nh, nkv, hd, rows, keys, hb = (rule[f] for f in fields)
        if (nh, nkv, hd) != (24, 4, 256) or hb not in (2, 3, 6):
            raise SystemExit(f"{path}: flash_gqa configuration is not compiled")
        shape = (sid, nh, nkv, hd, rows, keys)
        if any(r[:-1] == shape for r in rules):
            raise SystemExit(f"{path}: duplicate flash_gqa rule for {shape[1:]}")
        rules.append((*shape, hb))
    return rules


def strata_placement_rules(path, table, sid):
    fields = ("copies", "host", "bits", "nkv", "hd", "min_capacity", "edge_host")
    rules = []
    for rule in table.get("strata_placement", []):
        if set(rule) - {"scan_halves"} != set(fields) or any(type(rule[f]) is not int or not 0 < rule[f] <= INF for f in fields):
            raise SystemExit(f"{path}: strata_placement takes positive integers for {', '.join(fields)}")
        copies, host, bits, nkv, hd, capacity, edge = (rule[f] for f in fields)
        halves = rule.get("scan_halves", False)
        if type(halves) is not bool:
            raise SystemExit(f"{path}: scan_halves takes true or false")
        if host >= copies or 2 * edge > host or bits not in (2, 3, 4):
            raise SystemExit(f"{path}: invalid strata_placement host count, edges or bit width")
        shape = sid, copies, host, bits, nkv, hd
        if any(r[:len(shape)] == shape for r in rules):
            raise SystemExit(f"{path}: duplicate strata_placement for {shape[1:]}")
        rules.append((*shape, capacity, edge, int(halves)))
    return rules


def parse_arch(arch) -> int | None:
    """Parse a compute capability string like '8.9' into major * 10 + minor (89), or None if invalid."""
    major, dot, minor = str(arch).partition(".")
    if dot and major.isdigit() and minor.isdigit() and len(minor) == 1:
        return int(major) * 10 + int(minor)
    return None


def table_arch(path: str, table: dict) -> int:
    """Compute capability (80, 86, 89, 90, 100, 120) a table belongs to from its path or architectures, else 0."""
    archs = [cc for arch in table.get("architectures", []) if (cc := parse_arch(arch)) is not None]
    if len(set(archs)) == 1:
        return archs[0]
    m = re.search(r"(?:^|[\\/])sm(\d+)(?:[\\/]|$)", path)
    return int(m.group(1)) if m else 0


ARCH_BUCKETS = (80, 86, 89, 90, 100, 120, 0)  # 0: union of all tables + defaults


def compute_reachability(c: Compiled, skus, sku_archs, rules, families):
    """Exact first-match reachable kernel bitmasks per compute-capability bucket (for pruned release builds).

    Bucket 80 includes sm_80, sm_86 and sm_89 tables because an sm_80 cubin serves all three via CUDA minor-version
    binary compatibility. Every bucket includes the defaults (-1) so tuning=defaults never hits an uncompiled kernel.
    """
    fam_by_sku = {sku: fam for sku, fam in families}
    defaults = fam_by_sku[-1]

    def resolve(sid, family):
        s = sid
        while s >= 0:
            if s in fam_by_sku and family in fam_by_sku[s]:
                return fam_by_sku[s][family]
            s = skus[s][2]
        return defaults[family]

    def skus_for_bucket(bucket):
        if bucket == 0:
            return list(range(len(skus))) + [-1]
        accept = {80, 86, 89, 0} if bucket == 80 else {bucket, 0}
        return [sid for sid, cc in enumerate(sku_archs) if cc in accept] + [-1]

    dt_idx = {dt: i for i, dt in enumerate(GGML_TYPES)}
    n_b, n_dt = len(ARCH_BUCKETS), len(GGML_TYPES)
    round_dp4a = [[0] * n_dt for _ in range(n_b)]
    round_stack = [[0] * n_dt for _ in range(n_b)]
    round_glu = [[0] * n_dt for _ in range(n_b)]
    round_tc = [[0] * n_dt for _ in range(n_b)]
    blocks_q8 = [[0] * n_dt for _ in range(n_b)]
    blocks_skinny = [0] * n_b
    bf16_mask = [[0, 0] for _ in range(n_b)]
    w12_mask = [[0, 0] for _ in range(n_b)]
    fp4_mask = [[0, 0] for _ in range(n_b)]
    mt_mask = [0] * n_b  # bf16_mt configurations (k_gemm_skinny_mt, 17..64 rows)
    # moe_dp4a configurations for fused gate|up by rows band (0: rows == 1, 1: rows >= 2);
    # the original tiles stay compiled for every type in both bands (and for down / unfused projections).
    orig_bit = 1 << c.moe.index(MOE_ORIGINAL)
    moe = [[[orig_bit] * n_dt for _ in range(2)] for _ in range(n_b)]

    # Per-SKU shape rules; bf16_rows and bf16_draft (fmt 3, 4) name bf16_mt configurations, not a 1..16-row kernel's
    rules_by_sku = {}
    for sid, fmt, n, k, lo, hi, cfg in rules:
        rules_by_sku.setdefault(sid, []).append((fmt, lo, hi, cfg))

    for bi, bucket in enumerate(ARCH_BUCKETS):
        bskus = skus_for_bucket(bucket)
        # bf16 / w12 / fp4
        for sid in bskus:
            for lo, hi, cfg in resolve(sid, "bf16"):
                if lo <= 8:
                    bf16_mask[bi][0] |= 1 << cfg
                    w12_mask[bi][0] |= 1 << cfg
                if hi > 8:
                    bf16_mask[bi][1] |= 1 << cfg
                    w12_mask[bi][1] |= 1 << cfg
            for cfg in resolve(sid, "nvfp4"):
                fp4_mask[bi][0] |= 1 << cfg
                fp4_mask[bi][1] |= 1 << cfg
            for _, _, _, cfg in resolve(sid, "blocks"):
                blocks_skinny[bi] |= 1 << cfg
            # 17..64 rows: a round's rules, a drafting pass's in both K partitions
            for family in ("bf16_rows", "bf16_draft"):
                for _, _, *cfgs in resolve(sid, family):
                    for cfg in cfgs:
                        mt_mask[bi] |= 1 << cfg
            if sid >= 0:
                # Include inherited shape rules too
                s = sid
                while s >= 0:
                    for fmt, lo, hi, cfg in rules_by_sku.get(s, ()):
                        if fmt >= 3:
                            mt_mask[bi] |= 1 << cfg
                            continue
                        target = bf16_mask if fmt == 0 else fp4_mask if fmt == 1 else w12_mask
                        if lo <= 8:
                            target[bi][0] |= 1 << cfg
                        if hi > 8:
                            target[bi][1] |= 1 << cfg
                            if fmt == 1:
                                # gemm_fp4_rows chunks >16 rows into 16-row slices; the last slice may be 1..8 rows.
                                target[bi][0] |= 1 << cfg
                    s = skus[s][2]
        # blocks_q8 (exact first-match)
        for sid in bskus:
            q8r = resolve(sid, "blocks_q8")
            q8_rows = sorted({17, INF} | {v for _, lo, hi, _, _ in q8r for v in (lo, hi)})
            q8_ns = sorted({1, INF} | {v for _, _, _, nmax, _ in q8r for v in (nmax, nmax + 1) if v <= INF})
            for dt in GGML_TYPES:
                di = dt_idx[dt]
                for rows in q8_rows:
                    for n in q8_ns:
                        for types, lo, hi, nmax, cfg in q8r:
                            if (not types or dt in types) and lo <= rows <= hi and n <= nmax:
                                blocks_q8[bi][di] |= 1 << cfg
                                break
        # blocks_round (exact first-match, plus GLU at rows == 1 & N >= 128 and stack at 2048 < N <= 65536 with S != 2)
        for sid in bskus:
            rr = resolve(sid, "blocks_round")
            ns = sorted({1, 128, 2049, 65536, INF} | {v for _, _, _, nmin, nmax, _, _ in rr for v in (nmin + 1, nmax, nmax + 1) if v <= INF})
            for dt in GGML_TYPES:
                di = dt_idx[dt]
                for rows in range(1, 17):
                    for n in ns:
                        for types, lo, hi, nmin, nmax, cfg, eng in rr:
                            if (not types or dt in types) and lo <= rows <= hi and nmin < n <= nmax:
                                if eng == 0:
                                    round_dp4a[bi][di] |= 1 << cfg
                                    r_cfg, _, _, _, s_cfg, p_cfg = c.rounds[cfg]
                                    if 2048 < n <= 65536 and s_cfg != 2:
                                        round_stack[bi][di] |= 1 << cfg
                                    if rows == 1 and n >= 128 and s_cfg != 2 and p_cfg == 1 and r_cfg <= 4:
                                        round_glu[bi][di] |= 1 << cfg
                                elif eng == 1:
                                    round_tc[bi][di] |= 1 << (0 if rows <= 8 else 1)
                                elif eng == 2:
                                    round_tc[bi][di] |= 1 << (2 if rows <= 8 else 1)
                                elif eng == 3:
                                    # split falls back to mma if scratch/smem does not fit
                                    round_tc[bi][di] |= (1 << (3 if rows <= 8 else 4)) | (1 << (0 if rows <= 8 else 1))
                                elif eng == 4:
                                    # split_columns falls back to split8 then mma8
                                    round_tc[bi][di] |= (1 << 5) | (1 << 3) | (1 << 0)
                                elif eng == 5:
                                    round_tc[bi][di] |= 1 << (6 if rows <= 8 else 7)
                                break
        # moe_gate_up (exact first-match by type and rows; band 0: rows == 1, band 1: rows >= 2)
        for sid in bskus:
            mr = resolve(sid, "moe_gate_up")
            probes = sorted({1, 2, INF} | {v for _, lo, hi, _ in mr for v in (lo, hi, hi + 1) if v <= INF})
            for dt in GGML_TYPES:
                di = dt_idx[dt]
                for rows in probes:
                    for types, lo, hi, cfg in mr:
                        if (not types or dt in types) and lo <= rows <= hi:
                            moe[bi][0 if rows == 1 else 1][di] |= 1 << cfg
                            break
    return {
        "round_dp4a": round_dp4a, "round_stack": round_stack, "round_glu": round_glu,
        "round_tc": round_tc, "blocks_q8": blocks_q8, "blocks_skinny": blocks_skinny,
        "bf16": bf16_mask, "w12": w12_mask, "fp4": fp4_mask, "mt": mt_mask, "moe": moe,
    }


def emit(c: Compiled, skus, rules, families, archs=(), flash_gqa=(), strata_placement=(), sku_archs=None) -> str:
    """families: (sku, {family: rules}), the skus' first, the defaults (sku -1) last; archs: (cc, sku)."""
    out = ["// Generated by tools/codegen/gen_gemm_tables.py from backends/cuda/variants: do not edit.", "#pragma once", "",
           '#include "flyx/backend_abi.h"', ""]

    def emit_xmacro(macro: str, comment: str, items):
        out.append(f"// {comment}")
        out.append(f"#define {macro}(X) \\")
        out.extend(f"  X({i}, {', '.join(str(x) for x in v)}) \\" for i, v in enumerate(items))
        out.append("")
        out.append(f"#define {macro}_COUNT {len(items)}")

    emit_xmacro("FLYX_BF16_SKINNY", "X(index, nt, warps, unroll, split)", c.bf16)
    emit_xmacro("FLYX_BF16_MT", "X(index, nt, chunks, unroll, parts)", c.mt)
    emit_xmacro("FLYX_FP4_SKINNY", "X(index, span, chunks, nt, warps, unroll)", c.fp4)
    emit_xmacro("FLYX_BLOCKS_SKINNY", "X(index, nt, nw, kw)", c.blocks)
    emit_xmacro("FLYX_BLOCKS_Q8", "X(index, bn, qk, bm)", c.q8)
    emit_xmacro("FLYX_BLOCKS_ROUND", "X(index, rows, columns, ahead, warps, staged, parts)", c.rounds)
    emit_xmacro("FLYX_MOE_DP4A", "X(index, columns, units): 0 and 0, the original tiles", c.moe)
    emit_xmacro("FLYX_PREFILL_TC", "X(index, bm, bn, wn, stages)", c.tc)
    out.append("")
    out.append("namespace flyx::cuda::tables {")
    out.append("// The families of rules a table may have beside its shapes (bit i: the i-th), as the defaults have them.")
    out.append("enum : unsigned {")
    out += [f"  kFamily{''.join(w.capitalize() for w in f.split('_'))} = {1 << i}u," for i, f in enumerate(FAMILIES)]
    out.append("};")
    out.append("// pdl: kernels launch as programmatic dependents on this GPU unless the options say otherwise")
    out.append("// split_columns: the most columns of a shape this table's rules (or those it inherits) run in a split")
    out.append("// configuration, whose chunks' sums the scratch holds; 0: none.")
    out.append("struct Sku { const char* name; const char* const* devices; int n_devices; int parent; unsigned families; "
               "bool pdl; int split_columns; };")
    for i, (n, devs, _, _) in enumerate(skus):
        lst = ", ".join('"' + x.replace('"', '\\"') + '"' for x in devs) or "nullptr"
        out.append(f"inline const char* const kDevices{i}[] = {{{lst}}};")
    has = {sku: sum(1 << i for i, f in enumerate(FAMILIES) if f in fam) for sku, fam in families}
    split_own = [0] * len(skus)
    for sku, fmt, n, _, _, _, cfg in rules:
        if fmt == 0 and c.bf16[cfg][3] > 1:
            split_own[sku] = max(split_own[sku], n)

    def split_columns(i):
        cols = 0
        while i >= 0:
            cols, i = max(cols, split_own[i]), skus[i][2]
        return cols
    out.append("inline const Sku kSkus[] = {")
    out += [f'    {{"{n}", kDevices{i}, {len(devs)}, {p}, {has.get(i, 0)}u, {"true" if pdl else "false"}, '
            f'{split_columns(i)}}},' for i, (n, devs, p, pdl) in enumerate(skus)]
    if not skus:
        out.append('    {"none", nullptr, 0, -1, 0u, false, 0},')
    out.append("};")
    out.append(f"inline constexpr int kSkuCount = {len(skus)};")
    out.append("// The table of a GPU that has none of its own, by its compute capability (major * 10 + minor).")
    out.append("struct ArchTable { int cc, sku; };")
    out.append("inline const ArchTable kArchTables[] = {")
    out += [f"    {{{cc}, {sku}}}," for cc, sku in archs]
    if not archs:
        out.append("    {-1, -1},")
    out.append("};")
    out.append(f"inline constexpr int kArchTableCount = {len(archs)};")
    out.append("// The sku whose rules of `family` serve a GPU of table `tuning`: its own, or the nearest it inherits that")
    out.append("// has them; -1: the defaults.")
    out.append("inline int rules_sku(int tuning, unsigned family) {")
    out.append("  for (int s = tuning; s >= 0; s = kSkus[s].parent)")
    out.append("    if (kSkus[s].families & family) return s;")
    out.append("  return -1;")
    out.append("}")
    out.append("// fmt: 0 bf16, 1 nvfp4, 2 w12; 3 bf16_rows, 4 bf16_draft (cfg: a bf16_mt configuration)")
    out.append("struct Rule { int sku, fmt, n, k, lo, hi, cfg; };")
    out.append("inline const Rule kRules[] = {")
    out += [f"    {{{', '.join(str(v) for v in r)}}}," for r in rules]
    if not rules:
        out.append("    {-1, 0, 0, 0, 0, 0, 0},")
    out.append("};")
    out.append(f"inline constexpr int kRuleCount = {len(rules)};")
    out.append("struct FlashGqaRule { int sku, nh, nkv, hd, rows, keys, hb; };  // FP8 staged key windows")
    out.append("inline constexpr FlashGqaRule kFlashGqa[] = {")
    out += [f"    {{{', '.join(str(v) for v in r)}}}," for r in flash_gqa]
    if not flash_gqa:
        out.append("    {-1, 0, 0, 0, 0, 0, 0},")
    out.append("};")
    out.append("struct StrataPlacement { int sku, copies, host, bits, nkv, hd, min_capacity, edge_host, scan_halves; };")
    out.append("inline constexpr StrataPlacement kStrataPlacement[] = {")
    out += [f"    {{{', '.join(str(v) for v in r)}}}," for r in strata_placement]
    if not strata_placement:
        out.append("    {-1, 0, 0, 0, 0, 0, 0, 0, 0},")
    out.append("};")

    def rows_of(family):
        return [(sku, r) for sku, fam in families if family in fam for r in fam[family]]
    out.append("// bf16 (and W12) shapes no table lists: by the rows")
    out.append("struct RowsRule { int sku, lo, hi, cfg; };")
    out.append("inline const RowsRule kBf16Rows[] = {")
    out += [f"    {{{sku}, {lo}, {hi}, {cfg}}}," for sku, (lo, hi, cfg) in rows_of("bf16")]
    out.append("};")
    out.append("// bf16 shapes no table lists, 17..64 rows (bf16_mt configurations): a round's")
    out.append("inline const RowsRule kBf16RowsRules[] = {")
    out += [f"    {{{sku}, {lo}, {hi}, {cfg}}}," for sku, (lo, hi, cfg) in rows_of("bf16_rows")]
    out.append("};")
    out.append("// A drafting pass's K partition: its chunks by K alone, in every table and at every row count, so the")
    out.append("// weights one launch takes on the same rows each get the sums of a launch of their own.")
    out.append(f"inline constexpr int kDraftLongK = {DRAFT_LONG_K}, kDraftChunks = {DRAFT_CHUNKS[0]}, "
               f"kDraftChunksLong = {DRAFT_CHUNKS[1]};")
    out.append("inline constexpr int draft_chunks(int K) {")
    out.append("  return K >= kDraftLongK ? kDraftChunksLong : kDraftChunks;")
    out.append("}")
    out.append("// bf16 shapes no table lists, 17..64 rows, a drafting pass's: the configuration of each partition")
    out.append("struct DraftRule { int sku, lo, hi, cfg, cfg_long; };")
    out.append("inline const DraftRule kBf16DraftRules[] = {")
    out += [f"    {{{sku}, {lo}, {hi}, {cfg}, {cfg_long}}}," for sku, (lo, hi, cfg, cfg_long) in rows_of("bf16_draft")]
    out.append("};")
    out.append("// NVFP4 shapes no table lists: one configuration (one K partition) for K % 128 == 0, one for the rest")
    out.append("struct Nvfp4Rule { int sku, k128, k64; };")
    out.append("inline const Nvfp4Rule kNvfp4Rules[] = {")
    out += [f"    {{{sku}, {fam['nvfp4'][0]}, {fam['nvfp4'][1]}}}," for sku, fam in families if "nvfp4" in fam]
    out.append("};")
    out.append("struct BlocksRule { int sku, n_max, lo, hi, cfg; };  // layers of up to n_max columns, rows lo..hi")
    out.append("inline const BlocksRule kBlocks[] = {")
    out += [f"    {{{sku}, {n}, {lo}, {hi}, {cfg}}}," for sku, (n, lo, hi, cfg) in rows_of("blocks")]
    out.append("};")
    out.append("// block type (-1: every type no rule names), passes of lo..hi rows, layers of up to n_max columns")
    out.append("struct BlocksQ8Rule { int sku, dtype, lo, hi, n_max, cfg; };")
    out.append("inline const BlocksQ8Rule kBlocksQ8[] = {")
    for sku, (types, lo, hi, n, cfg) in rows_of("blocks_q8"):
        out += ([f"    {{{sku}, FLYX_DT_GGML_{t}, {lo}, {hi}, {n}, {cfg}}}," for t in types]
                or [f"    {{{sku}, -1, {lo}, {hi}, {n}, {cfg}}},"])
    out.append("};")
    out.append("struct BlocksQ8Runs { int sku, n_max, k_max, runs; };  // layers of up to n_max columns and k_max of K")
    out.append("inline const BlocksQ8Runs kBlocksQ8Runs[] = {")
    out += [f"    {{{sku}, {n}, {k}, {r}}}," for sku, (n, k, r) in rows_of("blocks_q8_runs")]
    out.append("};")
    out.append("// block type (-1: every type), rounds of lo..hi rows, layers of more than n_min and at most n_max columns;")
    out.append("// the first rule that matches.  kernel: 0 dp4a in configuration cfg; else tensor cores (cfg -1):")
    out.append("// 1 rows, 2 columns, 3 split, 4 split_columns, 5 warps.")
    out.append("struct BlocksRoundRule { int sku, dtype, lo, hi, n_min, n_max, cfg, kernel; };")
    out.append("// constexpr: a stack's kernels are compiled only in the configurations a rule gives (round_cfg_ruled)")
    out.append("inline constexpr BlocksRoundRule kBlocksRound[] = {")
    for sku, (types, lo, hi, n_min, n_max, cfg, e) in rows_of("blocks_round"):
        out += ([f"    {{{sku}, FLYX_DT_GGML_{t}, {lo}, {hi}, {n_min}, {n_max}, {cfg}, {e}}}," for t in types]
                or [f"    {{{sku}, -1, {lo}, {hi}, {n_min}, {n_max}, {cfg}, {e}}},"])
    out.append("};")
    out.append("// MoE layers' packed expert dp4a GEMMs: the original tiles' configuration (FLYX_MOE_DP4A), every type's;")
    out.append("// the fused gate|up GEMM's rules: block type (-1: every type no rule names), passes of lo..hi rows, the")
    out.append("// first rule that matches.")
    out.append(f"inline constexpr int kMoeDp4aOriginal = {c.moe.index(MOE_ORIGINAL)};")
    out.append("struct MoeGateUpRule { int sku, dtype, lo, hi, cfg; };")
    out.append("inline constexpr MoeGateUpRule kMoeGateUp[] = {")
    for sku, (types, lo, hi, cfg) in rows_of("moe_gate_up"):
        out += ([f"    {{{sku}, FLYX_DT_GGML_{t}, {lo}, {hi}, {cfg}}}," for t in types]
                or [f"    {{{sku}, -1, {lo}, {hi}, {cfg}}},"])
    out.append("};")
    out.append("// Prefill GEMMs on the bf16 tensor cores: weights' format (-1 both, 0 bf16, 1 nvfp4), passes of lo..hi")
    out.append("// rows, fewer 64 x 128 tiles per SM than tiles_below (0: any); the first rule that matches.")
    out.append("struct PrefillTcRule { int sku, fmt, lo, hi; double tiles_below; int cfg; };")
    out.append("inline constexpr PrefillTcRule kPrefillTc[] = {")
    out += [f"    {{{sku}, {fmt}, {lo}, {hi}, {below!r}, {cfg}}}," for sku, (fmt, lo, hi, below, cfg) in rows_of("prefill_tc")]
    out.append("};")
    if sku_archs is None:
        sku_archs = [0] * len(skus)
    reach = compute_reachability(c, skus, sku_archs, rules, families)
    n_buckets = len(ARCH_BUCKETS)
    any_bucket = ARCH_BUCKETS.index(0)
    bucket_doc = " ".join(f"{i}:{'any' if cc == 0 else ('80(8x)' if cc == 80 else cc)}"
                          for i, cc in enumerate(ARCH_BUCKETS))
    bucket_expr = " : ".join(f"cc == {cc} ? {i}" for i, cc in enumerate(ARCH_BUCKETS) if cc) + f" : {any_bucket}"
    out.append("// Per-SKU compute capability (0: any) and exact first-match reachable kernel bitmasks per arch bucket.")
    out.append("inline constexpr int kSkuArch[] = {" + (", ".join(str(a) for a in sku_archs) or "0") + "};")
    out.append(f"inline constexpr int kArchBucketCount = {n_buckets};  // {bucket_doc}")
    out.append(f"inline constexpr int kArchBucketAny = {any_bucket};")
    out.append("#ifdef __CUDACC__")
    out.append("#define FLYX_TABLES_HD __host__ __device__")
    out.append("#else")
    out.append("#define FLYX_TABLES_HD")
    out.append("#endif")
    out.append("FLYX_TABLES_HD constexpr int arch_bucket(int cc) {")
    out.append(f"  return {bucket_expr};")
    out.append("}")
    out.append("FLYX_TABLES_HD constexpr int ggml_type_index(int dtype) {")
    out.append("  switch (dtype) {")
    out += [f"    case FLYX_DT_GGML_{t}: return {i};" for i, t in enumerate(GGML_TYPES)]
    out.append("    default: return -1;")
    out.append("  }")
    out.append("}")
    out.append("#undef FLYX_TABLES_HD")
    def emit_mat(name, mat):
        out.append(f"inline constexpr unsigned {name}[{n_buckets}][{len(GGML_TYPES)}] = {{")
        for row in mat:
            out.append("    {" + ", ".join(f"{v:#x}u" for v in row) + "},")
        out.append("};")
    emit_mat("kReachRoundDp4a", reach["round_dp4a"])
    emit_mat("kReachRoundStack", reach["round_stack"])
    emit_mat("kReachRoundGlu", reach["round_glu"])
    emit_mat("kReachRoundTc", reach["round_tc"])  # bits: 0:mma8 1:mma16 2:cols8 3:split8 4:split16 5:split_cols8 6:warps8 7:warps16
    emit_mat("kReachBlocksQ8", reach["blocks_q8"])
    # bit i: moe_dp4a configuration i for fused gate|up ([arch_bucket][rows > 1][ggml_type])
    out.append(f"inline constexpr unsigned kReachMoeDp4a[{n_buckets}][2][{len(GGML_TYPES)}] = {{")
    for bmat in reach["moe"]:
        out.append("    {")
        for row in bmat:
            out.append("        {" + ", ".join(f"{v:#x}u" for v in row) + "},")
        out.append("    },")
    out.append("};")
    out.append(f"inline constexpr unsigned kReachBlocksSkinny[{n_buckets}] = {{" +
               ", ".join(f"{v:#x}u" for v in reach["blocks_skinny"]) + "};")
    out.append("inline constexpr unsigned kReachBf16Mt[7] = {" + ", ".join(f"{v:#x}u" for v in reach["mt"]) + "};")
    for name, key_name in (("kReachBf16Skinny", "bf16"), ("kReachW12Skinny", "w12"), ("kReachFp4Skinny", "fp4")):
        out.append(f"inline constexpr unsigned {name}[{n_buckets}][2] = {{" +
                   ", ".join(f"{{{lo:#x}u, {hi:#x}u}}" for lo, hi in reach[key_name]) + "};")
    out.append("}  // namespace flyx::cuda::tables")
    return "\n".join(out) + "\n"


def _load_toml(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("tables", nargs="*")
    a = ap.parse_args(argv)
    cand = _load_toml(a.candidates)
    c = Compiled(cand, a.candidates)
    defaults = family_rules(cand["defaults"], f"{a.candidates} [defaults]", c, every=True)

    skus, sku_archs, rules, families, archs, flash_gqa, strata_placement = [], [], [], [], [], [], []
    tables = [(p, _load_toml(p)) for p in sorted(a.tables)]
    names = [t["sku"] for _, t in tables]
    if len(set(names)) != len(names):
        raise SystemExit(f"two tables name one sku: {names}")
    # A GPU takes the table that names its CUDA device name (gemm_tuning_for): one table a name, or the last would win.
    owner = {}
    for path, t in tables:
        for dev in t.get("devices", []):
            if dev in owner:
                raise SystemExit(f"{path}: device {dev!r} is already named by the table of sku {owner[dev]!r}")
            owner[dev] = t["sku"]
    for path, t in tables:
        if t.get("inherit") and t["inherit"] not in names:
            raise SystemExit(f"{path}: inherits unknown sku {t['inherit']!r}")
        parent = names.index(t["inherit"]) if t.get("inherit") else -1
        if not isinstance(t.get("pdl", False), bool):
            raise SystemExit(f"{path}: pdl takes true or false, not {t['pdl']!r}")
        skus.append((t["sku"], list(t.get("devices", [])), parent, t.get("pdl")))
        sku_archs.append(table_arch(path, t))
        sid = len(skus) - 1
        for arch in t.get("architectures", []):
            cc = parse_arch(arch)
            if cc is None:
                raise SystemExit(f"{path}: architectures takes compute capabilities like \"8.9\", not {arch!r}")
            if any(cc == have for have, _ in archs):
                raise SystemExit(f"{path}: another table already serves compute capability {arch}")
            archs.append((cc, sid))
        rules += shape_rules(path, t, sid, c)
        flash_gqa += flash_gqa_rules(path, t, sid)
        strata_placement += strata_placement_rules(path, t, sid)
        own = family_rules(t.get("rules", {}), f"{path} [rules]", c, every=False, default_rounds=defaults["blocks_round"])
        if own:
            families.append((sid, own))
    for sid, (_, _, parent, _) in enumerate(skus):  # an inheritance chain ends
        seen, s = set(), sid
        while s >= 0:
            if s in seen:
                raise SystemExit(f"the tables' inherit chain from {skus[sid][0]} goes round")
            seen.add(s)
            s = skus[s][2]
    # A table's own pdl, else the nearest inherited table's that says, else off.
    resolved = []
    for name, devs, parent, pdl in skus:
        s = parent
        while pdl is None and s >= 0:
            pdl, s = skus[s][3], skus[s][2]
        resolved.append((name, devs, parent, bool(pdl)))
    skus = resolved
    families.append((-1, defaults))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    text = emit(c, skus, rules, families, archs, flash_gqa, strata_placement, sku_archs)
    existing = None
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            existing = f.read()
    if existing != text:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
