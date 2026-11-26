# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The tiny test model alone, for a release's smoke test (.github/workflows/release-smoke.yml):

    python tools/ci/tiny_model.py SOURCE_TREE OUT_DIR      # writes OUT_DIR/tiny.flyx

tiny.flyx is the first model tests/e2e/make_tiny.py makes (a Qwen3.5-shaped target of 64 wide layers with its
DFlash2 drafter, random weights, a vocabulary of 768), made here by SOURCE_TREE's own make_tiny.py and converter:
checked out at the release's tag, it is a file of the format that release reads.  Its float64 run takes a fraction
of a second, so a release's CLI and binding can generate with it on a machine without a GPU.  The steps are
make_tiny.py's first ones, the same seed, without the golden outputs and the other models (seconds, not minutes).
"""
from __future__ import annotations

import os
import sys


def main(source: str, out: str) -> None:
    sys.path.insert(0, os.path.join(source, "tests", "e2e"))
    import make_tiny  # SOURCE_TREE's, which puts its python/reference and python/convert on the path
    import torch

    tdir, ddir = os.path.join(out, "hf_target"), os.path.join(out, "hf_draft")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(ddir, exist_ok=True)
    gen = torch.Generator().manual_seed(1234)
    tok, _ = make_tiny.make_tokenizer(tdir)
    with open(os.path.join(tdir, "chat_template.jinja"), "w", encoding="utf-8") as f:
        f.write("{# tiny #}<|im_start|>{{ messages }}<think>")
    vocab = 768
    make_tiny.make_target(tdir, vocab, gen)
    make_tiny.make_draft(ddir, vocab, tok.token_to_id("<|mask|>"), gen)
    make_tiny.convert(tdir, os.path.join(out, "tiny.flyx"), ddir, log=lambda *_: None)
    print(os.path.join(out, "tiny.flyx"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]))
