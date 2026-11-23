# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The round kernels' defaults (variants/candidates.toml, blocks_round) from `cuda_bench rounds` and GGUF files.

    python bench/microbench/round_rules.py ROUNDS.txt FILE.gguf... [--min-us 25] [--min-gain 0.03]
        [--only 1x1x2,1x2x0,mma,cols,...]

ROUNDS.txt: the output of `cuda_bench rounds` (GB/s, per block type, layer shape and round of 1, 2, 4, 6 or 8 rows,
of every compiled configuration of the dp4a kernel, rRxCxU: the kernel of R rows, C columns a warp, U groups ahead,
and of the two tensor-core kernels, mma: the rows in the MMA's first operand, cols: the columns there).  The files'
matrices (their types and shapes) weigh the shapes: a matrix's time is its bytes over the GB/s of the measured shape
it maps to.  For each round size and width class the candidate with the least time over all the files' matrices
becomes the rule for every type; a type gets its own rule for a class where that saves --min-us a round or more
and --min-gain of its time there.  A type's neighbouring classes with one candidate make one rule, reaching down
over classes where the rule for every type is the same; types with the same rule share it.  Prints the rules as
candidates.toml takes them and, per file and round size, the GEMMs' time a round with the current defaults (* in
ROUNDS.txt), the rules, and the best candidate per shape.  --only: the candidates the rules may use.  Needs the
`gguf` package.
"""
import argparse
import re
from collections import defaultdict

from gguf import GGUFReader

LINE = re.compile(r"^(\S+)\s+(.+?)\s+(\d+) x\s+(\d+) r(\d+)((?: \| \S+\s+\d+)+)\s*$")
CELL = re.compile(r"\| (r\d+x\d+x\d+|mma|cols|splitc|split)(\*?)\s+(\d+)")
WIDTHS = (64, 2048, 6144, 12288, 65536, 2**31 - 1)  # the width classes the rules tell apart (n_max)
ROUNDS = ((1, 1, 1), (2, 2, 2), (3, 4, 4), (5, 6, 6), (7, 8, 8))  # rows lo, hi, and the rows measured for them
INF = 2**31 - 1
TENSOR_CORES = {"mma": "rows", "cols": "columns", "split": "split", "splitc": "split_columns"}  # name -> tensor_cores


def candidate(text):
    """A candidate: a dp4a configuration (rows held, columns, ahead), or a tensor-core kernel's name."""
    if text in TENSOR_CORES:
        return text
    return tuple(int(v) for v in text[1:].split("x"))


def order(c):  # dp4a configurations first, then the tensor-core kernels
    return (1, c, (0, 0, 0)) if isinstance(c, str) else (0, "", c)


def load_rounds(path):
    gbps, default = defaultdict(dict), {}
    for line in open(path):
        m = LINE.match(line.rstrip("\n"))
        if not m:
            continue
        key = (m.group(1), m.group(2).strip(), int(m.group(5)))
        for name, star, v in CELL.findall(m.group(6)):
            gbps[key][candidate(name)] = float(v)
            if star:
                default[key] = candidate(name)
    return gbps, default


def shape_of(n, k):
    """The measured shape (cuda_bench's names for the 27B's layers) a matrix of n columns and K = k maps to."""
    if k == 17408:
        return "mlp down"
    if k == 6144:
        return "gdn out"
    if k == 10240:
        return "gdn qkv"  # the MTP layer's 5120 x 10240 projection: the nearest measured
    if k != 5120:
        return None
    if n > 100000:
        return "head"
    if n <= 64:
        return "gdn a"
    if n <= 2048:
        return "attention k"
    return {6144: "gdn z", 10240: "gdn qkv", 12288: "attention q"}.get(n, "mlp gate, up")


def matrices(path):
    """The GEMMs of a file (not the embedding: rows of it are read, not multiplied)."""
    out = []
    for t in GGUFReader(path).tensors:
        if len(t.shape) == 2 and not t.name.startswith("token_embd"):
            out.append({"type": t.tensor_type.name, "K": int(t.shape[0]), "N": int(t.shape[1]), "bytes": int(t.n_bytes)})
    return out


def width_class(n):
    return next(w for w in WIDTHS if n <= w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rounds")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-us", type=float, default=25.0)
    ap.add_argument("--min-gain", type=float, default=0.03)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    gbps, default = load_rounds(a.rounds)
    keep = {candidate(s if s in TENSOR_CORES else "r" + s) for s in a.only.split(",") if s} or None
    models = {f.split("/")[-1]: matrices(f) for f in a.files}
    every = [m for ms in models.values() for m in ms]
    key = lambda m, rows: (m["type"], shape_of(m["N"], m["K"]), rows)  # noqa: E731
    known = lambda m, rows: shape_of(m["N"], m["K"]) is not None and key(m, rows) in gbps  # noqa: E731
    cost = lambda c, ms, rows: sum(m["bytes"] / gbps[key(m, rows)][c] for m in ms) / 1e3  # us  # noqa: E731

    generic, exceptions = [], []
    for lo, hi, rows in ROUNDS:
        held = next(r for r in (1, 2, 4, 8) if r >= rows)
        allowed = lambda c: ((isinstance(c, str) or c[0] == held or (rows == 1 and c[0] == 2))  # noqa: E731
                             and (keep is None or c in keep))
        for w in WIDTHS:
            group = [m for m in every if width_class(m["N"]) == w and known(m, rows)]
            if not group:  # no matrix of this width: the default
                generic.append((lo, hi, w, next(default[k] for k in default if k[2] == rows)))
                continue
            cands = set.intersection(*({c for c in gbps[key(m, rows)] if allowed(c)} for m in group))
            best = min(sorted(cands, key=order), key=lambda c: cost(c, group, rows))
            generic.append((lo, hi, w, best))
            for t in sorted({m["type"] for m in group}):
                own = [m for m in group if m["type"] == t]
                tb = min(sorted(cands, key=order), key=lambda c: cost(c, own, rows))
                saved = cost(best, own, rows) - cost(tb, own, rows)
                if saved >= a.min_us and saved >= a.min_gain * cost(best, own, rows):
                    exceptions.append((t, lo, hi, w, tb))

    # The rules for every type: neighbouring widths with one candidate merge.
    rules = []
    for lo, hi, w, c in generic:
        if rules and rules[-1][1] == lo and rules[-1][4] == c:
            rules[-1] = ((), lo, hi, w, c)
        else:
            rules.append(((), lo, hi, w, c))
    lower = dict(zip(WIDTHS, (0,) + WIDTHS))
    every = {(lo, w): c for lo, hi, w, c in generic}
    # A type's exceptions: neighbouring classes with one candidate as one span of widths, reaching down over the
    # classes where the rule for every type already has that candidate.
    spans = []
    for t, lo, hi, w, c in sorted(exceptions, key=lambda e: (e[0], e[1], WIDTHS.index(e[3]))):
        if spans and spans[-1][0] == t and spans[-1][1] == lo and spans[-1][5] == c and spans[-1][4] == lower[w]:
            spans[-1] = spans[-1][:4] + (w, c)
            continue
        n_min = lower[w]
        while n_min and every[(lo, n_min)] == c:
            n_min = lower[n_min]
        spans.append((t, lo, hi, n_min, w, c))
    typed = []  # one rule for the types with the same span and candidate
    for t, lo, hi, n_min, n_max, c in spans:
        for i, (ts, *rest) in enumerate(typed):
            if tuple(rest) == (lo, hi, n_min, n_max, c):
                typed[i] = (ts + (t,), *rest)
                break
        else:
            typed.append(((t,), lo, hi, n_min, n_max, c))
    ordered = sorted(typed, key=lambda r: (r[1], r[3]))
    ordered += [((), lo, hi, 0, w, c) for _, lo, hi, w, c in rules]
    ordered.append(((), 9, 16, 0, INF, "mma"))
    print("blocks_round = [")
    for ts, lo, hi, n_min, n_max, c in ordered:
        names = f'types = [{", ".join(chr(34) + t + chr(34) for t in ts)}], ' if ts else ""
        widths = (f"n_min = {n_min}, " if n_min else "") + (f"n_max = {n_max}, " if n_max != INF else "")
        if isinstance(c, str):
            print(f'  {{ {names}rows = [{lo}, {hi}], {widths}tensor_cores = "{TENSOR_CORES[c]}" }},')
            continue
        kernel = f"kernel = {c[0]}, " if c[0] != next(r for r in (1, 2, 4, 8, 16) if r >= hi) else ""
        print(f"  {{ {names}rows = [{lo}, {hi}], {widths}{kernel}columns = {c[1]}, ahead = {c[2]} }},")
    print("]")
    used = sorted({x[5] for x in ordered}, key=order)
    print("candidates:", ", ".join(c if isinstance(c, str) else f"rows {c[0]} columns {c[1]} ahead {c[2]}" for c in used))

    def pick(t, rows, n):
        return next(c for ts, lo, hi, n_min, n_max, c in ordered if (not ts or t in ts) and lo <= rows <= hi
                    and n_min < n <= n_max)
    for name, ms in models.items():
        for lo, hi, rows in ROUNDS:
            mine = [m for m in ms if known(m, rows)]
            before = sum(m["bytes"] / gbps[key(m, rows)][default[key(m, rows)]] for m in mine) / 1e3
            after = sum(m["bytes"] / gbps[key(m, rows)][pick(m["type"], rows, m["N"])] for m in mine) / 1e3
            best = sum(m["bytes"] / max(gbps[key(m, rows)].values()) for m in mine) / 1e3
            print(f"{name} {rows} rows: defaults {before:.0f} us, rules {after:.0f} us ({100 * (after / before - 1):+.1f}%), "
                  f"best per shape {best:.0f} us ({100 * (best / before - 1):+.1f}%)")


if __name__ == "__main__":
    main()
