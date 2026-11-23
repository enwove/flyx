# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""How much a numeric option changes what prefill predicts.

    python bench/quality/next_token_agreement.py MODEL --lib LIB [--options K=V,...] [--activations exact]
        [--base-activations exact] [--step 64] [--prefill-chunk 16]

For texts of three kinds (English prose, C++ code, Chinese), every `step`
tokens from the 256th on: a fresh session prefills the text up to there and
picks the next token greedily, once with the first settings (the defaults, or
--base-activations) and once with the second (--options, --activations).
Reports, per text, how often the two picks agree and how often each matches the
text's own next token (teacher forcing: the picks never feed back).  A fresh
session per cut keeps prompt reuse out: the whole prefix runs through the
prefill path being measured.  The inputs a file's weights take in prefill
(GGUF block weights: 8 bits; NVFP4 weights of a file that declares NVFP4
inputs: NVFP4) against exact ones: --base-activations exact, and nothing for
the second run.  --prefill-chunk 16 runs every pass on the rounds' kernels
(what decoding computes with).
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "python"))
import flyx  # noqa: E402


def texts(engine) -> dict[str, list[int]]:
    rd = lambda p: open(os.path.join(ROOT, p), encoding="utf-8").read()  # noqa: E731
    out = {"english (docs/design/architecture.md)": rd("docs/design/architecture.md"),
           "code (core/exec/generation.cpp)": rd("core/exec/generation.cpp")}
    # Chinese: the model's own text with the default options (the same for both runs).
    s = engine.session()
    zh = s.chat([{"role": "user", "content": "请用中文写一篇约两千字的文章，介绍大语言模型推理引擎的设计：预填充、解码、投机解码、"
                                             "显存管理和量化。"}], max_new_tokens=1800, enable_thinking=False)
    s.close()
    out["chinese (generated)"] = zh
    return {k: engine.tokenize(v) for k, v in out.items()}


def picks(model: str, lib: str, options: str | None, activations: str | None, toks: dict[str, list[int]], step: int,
          limit: int, chunk: int):
    e = flyx.Engine(model, lib=lib, speculative=False, max_ctx=8192, backend_options=options, activations=activations,
                    prefill_chunk=chunk)
    res = {}
    for name, t in toks.items():
        cuts = list(range(256, min(len(t), limit), step))
        got = []
        for p in cuts:
            s = e.session()
            got.append(s.generate(tokens=t[:p], max_new_tokens=1, ignore_eos=True)[0])
            s.close()
        res[name] = (cuts, got)
    e.close()
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--options", default=None, help="backend options of the second run")
    ap.add_argument("--activations", default=None, help="the second run's activations (exact)")
    ap.add_argument("--base-activations", default=None, help="the first run's activations (exact)")
    ap.add_argument("--step", type=int, default=64)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--prefill-chunk", type=int, default=0, help="prompt rows per prefill pass, both runs (0: the default)")
    a = ap.parse_args()
    e = flyx.Engine(a.model, lib=a.lib, speculative=False, max_ctx=8192)
    toks = texts(e)
    e.close()
    base = picks(a.model, a.lib, None, a.base_activations, toks, a.step, a.limit, a.prefill_chunk)
    alt = picks(a.model, a.lib, a.options, a.activations, toks, a.step, a.limit, a.prefill_chunk)
    print(f"{'text':42s} {'cuts':>5s} {'agree':>7s} {'first hits':>11s} {'second hits':>12s}")
    total = [0, 0, 0, 0]
    for name, t in toks.items():
        cuts, b = base[name]
        _, o = alt[name]
        agree = sum(x == y for x, y in zip(b, o))
        hb = sum(x == t[p] for x, p in zip(b, cuts))
        ho = sum(x == t[p] for x, p in zip(o, cuts))
        n = len(cuts)
        total = [total[0] + n, total[1] + agree, total[2] + hb, total[3] + ho]
        print(f"{name:42s} {n:5d} {agree / n:7.1%} {hb / n:11.1%} {ho / n:12.1%}")
    n = total[0]
    print(f"{'all':42s} {n:5d} {total[1] / n:7.1%} {total[2] / n:11.1%} {total[3] / n:12.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
