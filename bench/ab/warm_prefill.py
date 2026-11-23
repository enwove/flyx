# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Prefill time of one long prompt in a warm process, through the Python binding.

    python bench/ab/warm_prefill.py LIB MODEL PROMPT.txt [runs=6] [max_ctx=8192] [activations] [tokens]

Loads the model once, then prefills the prompt `runs` times, each time in a
new session and rotated by 97 * i tokens so no prefix is reused, and prints
each run's prefill time (usage.prefill_ms).  A fresh `flyx run` process
also pays the first launch of every kernel and graph and the session's
allocation inside its first prefill (20-60 ms on WSL2); this measures what a
running server pays.  Backend options: FLYX_BACKEND_OPTIONS; activations:
"exact" (bf16 or binary16 inputs), "-" or none as the file has them.
tokens: the prompt's tokens repeated (or cut) to exactly that many, for each
of a comma-separated list, as `llama-bench -p N` takes N tokens, after one
untimed run of that length (max_ctx then at least the longest).
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import flyx  # noqa: E402


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    lib, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    runs = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    max_ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    activations = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None
    counts = [int(n) for n in sys.argv[7].split(",")] if len(sys.argv) > 7 else []
    e = flyx.Engine(model, lib=lib, max_ctx=max(max_ctx, *counts) if counts else max_ctx, activations=activations)
    base = e.tokenize(open(prompt, encoding="utf-8").read())
    for n in counts or [len(base)]:
        toks = [base[j % len(base)] for j in range(n)]
        times = []
        for i in range(-1 if counts else 0, runs):  # a length of its own: one untimed run first
            if i < 0:
                s = e.session()
                s.generate(tokens=toks, max_new_tokens=1, ignore_eos=True)
                s.close()
                continue
            s = e.session()
            rotated = [toks[(j + 97 * i) % len(toks)] for j in range(len(toks))]
            s.generate(tokens=rotated, max_new_tokens=1, ignore_eos=True)
            times.append(s.usage.prefill_ms)
            s.close()
        print(f"{len(toks)} tokens; prefill ms: {' '.join(f'{t:.1f}' for t in times)}; "
              f"median {statistics.median(times):.1f} ({len(toks) / statistics.median(times):.2f}K tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
