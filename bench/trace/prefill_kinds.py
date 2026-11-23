# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A prefill pass, engine against engine: kernel time by kind (weight GEMMs by type), per token.

    python bench/trace/prefill_kinds.py FLYX.tsv FLYX_TOKENS LLAMA.tsv LLAMA_TOKENS [--sms 70]

The traces come from bench/trace/kernel_trace.cpp: flyx under bench/ab/warm_prefill.py (passes in a warm
process), llama.cpp under `llama-bench -p N -n 0`.  flyx's passes start at its embedding kernel
(k_embed_blocks); llama.cpp's (its embedding is looked up on the CPU) after a gap of more than 2 ms between
kernels.  The first pass of each is a warm-up and left out.  One-row GEMVs: the head on the last row, and in
llama.cpp also its last layer's matrices after the attention (it keeps only the rows it outputs there).  Prints microseconds a token per kind for both,
then how much of flyx's weight GEMMs' time SMs sat idle in a launch's last wave of blocks (a block of 64
columns two to an SM, of 128 one; `--sms`: the GPU's SMs), counting each wave as long as a full one."""
import argparse
import collections
import math
import re

from trace_io import GGML_TYPES as GGML, iter_trace_rows, plain_kernel_name as plain, split_bursts


def kind_flyx(n):
    n = plain(n)
    m = re.search(r"k_gemm_blocks_q8<F(\w+),", n)
    if m:
        return "GEMM " + m.group(1)
    m = re.search(r"k_(?:round_blocks_q8|round_blocks_mma|gemm_blocks)<F(\w+),", n)
    if m:  # the head on the last row
        return "GEMV " + m.group(1) + " (one row)"
    for k, rx in [("rows to 8 bits", r"k_stage"), ("norms", r"k_rmsnorm"), ("gated delta nets", r"gdn"),
                  ("attention", r"attention|attn|k_flash_prefill"), ("SwiGLU", r"swiglu"), ("embedding", r"k_embed")]:
        if re.search(rx, n):
            return k
    return "other: " + re.sub(r"<.*|\(.*", "", n).replace("void ", "")[:32]


def kind_llama(n):
    m = re.search(r"mul_mat_q<\(ggml_type\)(\d+)", n)
    if m:
        return "GEMM " + GGML.get(int(m.group(1)), m.group(1))
    m = re.search(r"mul_mat_vec_q<\(ggml_type\)(\d+)", n)
    if m:  # the head, and the last layer's matrices: llama.cpp runs that layer on the rows it outputs
        return "GEMV " + GGML.get(int(m.group(1)), m.group(1)) + " (one row)"
    for k, rx in [("GEMM IQ1_M (dequantized, cuBLAS)", r"dequantize_block_iq1_m|cutlass"),
                  ("GEMM split-K fix-up", r"mul_mat_q_stream_k_fixup"), ("rows to 8 bits", r"quantize"),
                  ("norms", r"rms_norm|norm_f32"),
                  ("gated delta nets", r"gated_delta|ssm_conv|ssm_scan|l2_norm|softplus|op_sigmoid"),
                  ("attention", r"flash_attn|rope|k_set_rows|soft_max"), ("SwiGLU", r"swiglu|op_silu|glu"),
                  ("other: get_rows (the rows output)", r"get_rows")]:
        if re.search(rx, n):
            return k
    return "other: " + re.sub(r"<.*|\(.*", "", n).replace("void ", "")[:32]


def load(path):
    """Kernels as (start, end, blocks, name), in order of start."""
    rows = [(int(p[1]), int(p[2]), int(p[6]) * int(p[7]) * int(p[8]), p[10])
            for p in iter_trace_rows(path) if p[0] == "K"]
    rows.sort()
    return rows


def passes_flyx(rows):
    # A prompt's prefill (one or several passes, back to back) between gaps of more than 2 ms, those with prefill
    # GEMMs: the warm-up rounds at load also start at the embedding kernel, and a prompt longer than the prefill chunk
    # takes several passes.
    prefill = re.compile(r"k_gemm_blocks_q8<|k_gemm_w4a4|k_gemm_mma")
    out = [p for p in split_bursts(rows) if any(prefill.search(n) for *_, n in p)]
    # The prompts' runs, not the calibration prefill at load (it has prefill GEMMs too, and is shorter); the first
    # run is a warm-up.
    longest = max((p[-1][1] - p[0][0] for p in out), default=0)
    return [p for p in out if p[-1][1] - p[0][0] >= longest / 2][1:]


def passes_llama(rows):
    return [p for p in split_bursts(rows) if len(p) > 100][1:]


def per_kind(passes, kind):
    acc = collections.Counter()
    span = 0
    for p in passes:
        for s, e, _, n in p:
            acc[kind(n)] += e - s
        span += p[-1][1] - p[0][0]
    return {n: v / len(passes) for n, v in acc.items()}, span / len(passes)


def idle_in_waves(passes, sms):
    """ns a pass: SMs idle in the last wave of flyx's GEMM launches; in launches of less than one wave."""
    last = under = 0.0
    for p in passes:
        for s, e, blocks, n in p:
            m = re.search(r"k_gemm_blocks_q8<\S*F\w+, (\d+)(?:, \d+)?>", plain(n))  # <type, bn[, qk]>
            if not m:
                continue
            waves = blocks / (sms * (2 if m.group(1) == "64" else 1))
            if waves < 1:
                under += (e - s) * (1 - waves)
            else:
                last += (e - s) * (1 - waves / math.ceil(waves))
    return last / len(passes), under / len(passes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flyx")
    ap.add_argument("flyx_tokens", type=int)
    ap.add_argument("llama")
    ap.add_argument("llama_tokens", type=int)
    ap.add_argument("--sms", type=int, default=70)
    a = ap.parse_args()
    fp, lp = passes_flyx(load(a.flyx)), passes_llama(load(a.llama))
    ft, lt = a.flyx_tokens, a.llama_tokens
    f, fspan = per_kind(fp, kind_flyx)
    l, lspan = per_kind(lp, kind_llama)
    print(f"flyx: {len(fp)} passes of {ft} tokens, {fspan / 1e6:.1f} ms a pass ({fspan / ft / 1e3:.1f} us a token); "
          f"llama.cpp: {len(lp)} passes of {lt}, {lspan / 1e6:.1f} ms ({lspan / lt / 1e3:.1f} us a token)")
    print(f"{'us a token':40s} {'flyx':>8s} {'llama.cpp':>9s} {'diff':>7s}")
    keys = sorted(set(f) | set(l), key=lambda k: (not k.startswith("GEMM"), -(f.get(k, 0) / ft - l.get(k, 0) / lt)))
    sums = collections.defaultdict(lambda: [0.0, 0.0])
    for k in keys:
        x, y = f.get(k, 0) / ft / 1e3, l.get(k, 0) / lt / 1e3
        group = "weight GEMMs" if k.startswith("GEMM") else "everything else"
        sums[group][0] += x
        sums[group][1] += y
        print(f"{k:40s} {x:8.1f} {y:9.1f} {x - y:+7.1f}")
    for g, (x, y) in sums.items():
        print(f"{'= ' + g:40s} {x:8.1f} {y:9.1f} {x - y:+7.1f}")
    last, under = idle_in_waves(fp, a.sms)
    print(f"flyx's weight GEMMs, SMs idle: {last / ft / 1e3:.1f} us a token in partial last waves, "
          f"{under / ft / 1e3:.1f} in launches of less than one wave")


if __name__ == "__main__":
    main()
