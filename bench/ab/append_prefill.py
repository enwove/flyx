# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Time to append R tokens to a live session, by R, through the Python binding.

    python bench/ab/append_prefill.py LIB MODEL TEXT [--base 4096] [--runs 5] [--rows 16,32,...]

What an agent's tool turn pays: the session holds BASE tokens (and the token it
generated), and the next request adds R - 1 more, so R rows go through the
model (the generated token and the new ones) before the first new token.  Each
measurement starts a fresh session on the same BASE tokens of TEXT, generates
one token, then appends R - 1 tokens taken from after BASE in TEXT and reads
usage.prefill_ms.  It also gives the cost of a forward pass at R rows, as a
verify-only speculative round of R rows would pay it through the prefill
kernels.  Prints the median per R: ms, rows per second, ms per row past the
first.  Backend options: FLYX_BACKEND_OPTIONS; --activations exact: bf16 or binary16 inputs everywhere (an NVFP4
file's passes of more than 16 rows then take bf16 inputs, as its rounds do).
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import flyx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lib")
    ap.add_argument("model")
    ap.add_argument("text")
    ap.add_argument("--base", type=int, default=4096, help="tokens in the session before the append")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512,800,1024")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--activations", default=None, help='"exact": bf16 or binary16 inputs (default: as the file has them)')
    a = ap.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    e = flyx.Engine(a.model, lib=a.lib, max_ctx=a.max_ctx, activations=a.activations)
    toks = e.tokenize(open(a.text, encoding="utf-8").read())
    need = a.base + max(rows)
    if len(toks) < need:
        print(f"{a.text}: {len(toks)} tokens, needs {need}")
        return 2
    base = toks[:a.base]
    first = None
    print(f"base {a.base} tokens; median of {a.runs} runs")
    print(f"{'rows':>5} {'ms':>8} {'rows/s':>9} {'ms/row past 1':>14}")
    for r in rows:
        times = []
        for _ in range(a.runs):
            s = e.session()
            gen = s.generate(tokens=base, max_new_tokens=1, ignore_eos=True)
            s.generate(tokens=base + gen + toks[a.base:a.base + r - 1], max_new_tokens=1, ignore_eos=True)
            if s.usage.reused_tokens != a.base:
                print(f"rows {r}: reused {s.usage.reused_tokens}, expected {a.base}")
                return 1
            times.append(s.usage.prefill_ms)
            s.close()
        ms = statistics.median(times)
        if first is None and r == 1:
            first = ms
        past = f"{(ms - first) / (r - 1):14.3f}" if first is not None and r > 1 else f"{'':14}"
        print(f"{r:5d} {ms:8.2f} {r / ms * 1e3:9.0f} {past}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
