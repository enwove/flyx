# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Kernel time by layer category, for flyx's traces and llama.cpp's alike.

    python3 bench/trace/layer_times.py TRACE.tsv pp [AFTER_MS | last]   # a prefill: ms
    python3 bench/trace/layer_times.py TRACE.tsv tg TOKENS              # decoding: us per token

TRACE.tsv comes from bench/trace/kernel_trace.cpp (CUDA_INJECTION64_PATH=
libflyx-trace.so FLYX_TRACE=TRACE.tsv), which works on any CUDA program,
llama-bench included.  pp sums every kernel starting AFTER_MS or later after
the first one, or with `last` the last burst of kernels (idle > 2 ms between
bursts: llama-bench's measured run after its warm-up; a `flyx run` trace
holds one prefill, take all of it); tg sums the kernels of graph launches and
divides by TOKENS.  The categories are regular expressions over kernel names
and grids (Qwen3.5: gated-delta-net and attention layers); a kernel no rule
takes counts as "norms, residual, other".
"""
import collections
import re
import sys

from trace_io import iter_trace_rows

CATS = [  # first match wins: (category, regex on "name gx gy ")
    ("head", r"mul_mat_vec_f.* 248320 |k_gemm_skinny(_fp8)?<1, false, 16, 16>.* 31040 "),
    ("linear GEMM", r"mul_mat_q|mul_mat_vec_q|mul_mat_vec_f|cutlass|cublas|gemm|quantize_mmq|quantize_q8_1|"
                    r"convert_unary|k_quant_rows"),
    ("attention core", r"flash_attn|k_flash|k_attention"),
    ("attention prep", r"rope|set_rows|rms_norm_f32<128|k_attn_prep|op_sigmoid, float>.* (8192|2048|16) "),
    ("GDN recurrence", r"gated_delta|gdn_single|k_gdn_chunk|k_gdn_round"),
    ("GDN conv", r"ssm_conv|concat|k_gdn_conv"),
    ("GDN norms, gates", r"rms_norm_f32<32|softplus|op_sigmoid|k_gdn_prep|k_gdn_gated_norm|k_get_rows_float_vec|"
                         r"op_silu, float>.* (8192|16) "),
]


def category(name, gx, gy):
    s = f"{name} {gx} {gy} "
    for c, rx in CATS:
        if re.search(rx, s):
            return c
    return "norms, residual, other"


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[2] not in ("pp", "tg"):
        print(__doc__)
        return 2
    rows = list(iter_trace_rows(sys.argv[1]))
    k = sorted([r for r in rows if r[0] == "K"], key=lambda r: int(r[1]))
    if sys.argv[2] == "pp":
        arg = sys.argv[3] if len(sys.argv) > 3 else "0"
        if arg == "last":
            start, end = 0, int(k[0][2])
            for i in range(1, len(k)):
                if int(k[i][1]) - end > 2_000_000:
                    start = i
                end = max(end, int(k[i][2]))
            k = k[start:]
        else:
            t0 = int(k[0][1])
            k = [r for r in k if int(r[1]) - t0 >= float(arg) * 1e6]
        div, unit = 1e6, "ms"
    else:
        k = [r for r in k if r[4] != "0"]
        div, unit = float(sys.argv[3]) * 1e3, "us per token"
    acc = collections.Counter()
    names = collections.defaultdict(collections.Counter)
    for r in k:
        c = category(r[10], r[6], r[7])
        d = int(r[2]) - int(r[1])
        acc[c] += d
        names[c][re.sub(r"^void ", "", r[10]).split("(")[0][:50]] += d
    print(f"{sys.argv[1]}: {sum(acc.values()) / div:.1f} {unit}")
    for c, v in acc.most_common():
        top = "; ".join(f"{n} {x / div:.1f}" for n, x in names[c].most_common(3))
        print(f"  {c:24s} {v / div:8.1f}   [{top}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
