# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Layer attribution on exact and Strata attention, including the CLI's layer counts."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SCRIPT = Path(__file__).resolve().parents[3] / "bench" / "trace" / "layer_roles.py"


def analyze(path, names):
    with path.open("w", encoding="utf-8") as f:
        f.write("kind\tstart\tend\tcorr\tgraph\tstream\tgx\tgy\tgz\tbx\tname\n")
        for i, name in enumerate(names):
            f.write(f"K\t{i * 1000}\t{(i + 1) * 1000}\t1\t1\t1\t256\t1\t1\t128\t{name}\n")
    return subprocess.run([sys.executable, SCRIPT, path, "--skip", "0"], text=True, capture_output=True)


def main():
    norm = ["k_rmsnorm_q8"]
    gemm = ["void flyx::cuda::k_round_blocks_q8<SomeWeight>(int)"]
    mlp = norm + gemm + ["k_swiglu"] + gemm
    gdn = norm + gemm + ["k_gdn_round"] + gemm + mlp
    exact = ["k_attention_rows_mma", "k_attention_gqa", "k_attention_combine"]
    strata = ["k_kv_stage", "k_strata_rotate_q", "k_strata_scan_mma", "k_strata_select_part", "k_strata_select",
              "k_strata_claim", "k_strata_fill", "k_strata_delta", "k_strata_finish"]
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.tsv"
        for kernels in [exact, strata]:
            attn = norm + gemm + ["k_attn_prep"] + kernels + gemm + mlp
            result = analyze(trace, gdn + attn + ["k_rmsnorm"] + gemm)
            assert result.returncode == 0, result.stderr
            rows = result.stdout
            for expected in ["GDN  (1 layer", "attention  (1 layer", "MLP  (2 layers"]:
                assert expected in rows, rows
            if kernels == strata:
                for role in ["scan", "selection", "hot-cache lookup and fill", "correction", "query rotation", "output"]:
                    assert f"Strata {role}" in rows, rows
                assert "KV staging" in rows
            else:
                assert re.search(r"\s2\s+1\.00\s+attention$", rows, re.MULTILINE)
                assert "attention combine" in rows and "Strata scan" not in rows
        result = analyze(trace, ["_ZN4flyx4cuda12k_rmsnorm_q8Ev"])
        assert result.returncode == 2 and "cu++filt" in result.stderr and "drafter" not in result.stdout


if __name__ == "__main__":
    main()
