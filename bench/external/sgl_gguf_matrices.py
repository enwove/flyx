# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""SGLang's GGUF mat-vec (sgl-kernel's ggml_mul_mat_vec_a8) on a GGUF's own matrices, one token's worth at a time.

    CUDA_INJECTION64_PATH=<build>/bin/libflyx-trace.so FLYX_TRACE=sgl.tsv \\
        python bench/external/sgl_gguf_matrices.py run MODEL.gguf RAN.txt [passes]
    python bench/external/sgl_gguf_matrices.py analyze MODEL.gguf RAN.txt sgl.tsv > sgl_matrix.tsv

SGLang cannot load the 27B, so its kernel is run here the way decoding would run it: `run` uploads every weight
matrix as the file's bytes and multiplies one bf16 row by each in the order a token runs them
(bench/trace/per_matrix.py's order), `passes` times, each matrix read from DRAM once a pass; RAN.txt lists the
matrices the kernel took.  `analyze` takes the trace's mat-vec kernels in order, one a matrix, and prints each
matrix's median time (us) after two passes, the table per_matrix.py reads as NAME=sgl_matrix.tsv:matrix.  The
kernel rounds the row to 8 bits itself (a separate quantize kernel, not counted), and these are launches from
Python one at a time, not a graph: a bound on what SGLang would do in place, not a measurement of it.  Needs
torch and sgl-kernel (measured: torch 2.9.1+cu128, sgl-kernel 0.3.21) and the gguf package."""
import collections
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trace"))
import per_matrix  # noqa: E402


def run(model, ran_path, passes):
    import gguf
    import torch
    from sgl_kernel import ggml_mul_mat_vec_a8

    order = per_matrix.model_order(model)
    t = {x.name: x for x in gguf.GGUFReader(model).tensors}
    mats = []
    for m in order:
        x = t["output.weight" if m["role"] == "head" else f"blk.{m['layer']}.{m['role']}.weight"]
        mats.append((torch.from_numpy(x.data.copy()).cuda(), int(x.tensor_type), m["n"], m["k"]))
    xs = {k: torch.randn(1, k, dtype=torch.bfloat16, device="cuda") for k in {m[3] for m in mats}}
    bad = set()
    for _ in range(passes):
        for i, (w, qt, n, k) in enumerate(mats):
            if i in bad:
                continue
            try:
                ggml_mul_mat_vec_a8(w, xs[k], qt, n)
            except Exception as e:  # noqa: BLE001  (a type it does not take)
                bad.add(i)
                print(f"matrix {i} ({order[i]['role']} {order[i]['type']}): {e}", file=sys.stderr)
        torch.cuda.synchronize()
    print(f"{passes} passes over {len(mats) - len(bad)} of {len(mats)} matrices", file=sys.stderr)
    with open(ran_path, "w") as f:
        f.write(" ".join(str(i) for i in range(len(mats)) if i not in bad))


def analyze(model, ran_path, trace):
    order = per_matrix.model_order(model)
    ran = [int(i) for i in open(ran_path).read().split()]
    rows = []
    with open(trace, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "K":
                rows.append((int(p[1]), int(p[2]) - int(p[1]), p[10]))
    rows.sort()
    gemv = [d for _, d, n in rows if "mul_mat_vec" in n]
    passes = len(gemv) // len(ran)
    times = collections.defaultdict(list)
    for p in range(2, passes):  # the first two passes warm up
        for j, i in enumerate(ran):
            times[i].append(gemv[p * len(ran) + j])
    print("index\tlayer\trole\ttype\tbytes\tus")
    for i, m in enumerate(order):
        us = f"{st.median(times[i]) / 1e3:.2f}" if times[i] else "nan"
        print(f"{i}\t{m['layer']}\t{m['role']}\t{m['type']}\t{m['bytes']}\t{us}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
    else:
        analyze(sys.argv[2], sys.argv[3], sys.argv[4])
