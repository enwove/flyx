# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Block types inside each stack of a GGUF file: the tensors that share an input.

    python bench/external/gguf_stacks.py FILE.gguf...

For Qwen3.5's stacks (gate|up, q|k|v, the gated delta net's qkv|z|b|a), per
file: the GEMM launches a decoding round takes for them with one launch per
tensor, one per run of neighbouring tensors of one block type (how the CUDA
backend uploads them), and one per stack; how many stacks mix
types; an example of each kind of stack.  Needs the `gguf` package.
"""
import collections
import re
import sys

from gguf import GGUFReader

STACKS = {
    "mlp": ["ffn_gate", "ffn_up"],
    "attn": ["attn_q", "attn_k", "attn_v"],
    "gdn": ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"],
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for path in sys.argv[1:]:
        types = {}
        for t in GGUFReader(path).tensors:
            m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", t.name)
            if m:
                types[(int(m.group(1)), m.group(2))] = t.tensor_type.name
        per_tensor = per_run = per_stack = 0
        mixed = collections.Counter()
        examples = {}
        for layer in sorted({l for l, _ in types}):
            for kind, names in STACKS.items():
                ts = [types.get((layer, n)) for n in names]
                if not all(ts):
                    continue
                per_tensor += len(ts)
                per_run += 1 + sum(1 for a, b in zip(ts, ts[1:]) if a != b)
                per_stack += 1
                if len(set(ts)) > 1:
                    mixed[kind] += 1
                    examples.setdefault(kind, (layer, ts))
                else:
                    examples.setdefault(kind + " (one type)", (layer, ts))
        print(f"{path.rsplit('/', 1)[-1]}: launches a round for the stacks: one per tensor {per_tensor}, "
              f"one per run of one type {per_run}, one per stack {per_stack}; stacks of mixed types: {dict(mixed)}")
        for k, (layer, ts) in sorted(examples.items()):
            print(f"    {k:18s} layer {layer}: {ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
