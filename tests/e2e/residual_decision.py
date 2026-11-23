# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The bf16 prefill residual's decision, as each backend reports it at load (-v): a stack whose every residual add is
a W4A4 GEMM keeps a prefill pass's residual in bf16 (tiny_llama_nvfp4.flyx with --activations nvfp4), and the same
stack with the target's own adapters (ABI 43, tiny_llama_nvfp4_lora.flyx) keeps it in float32, as its adapters add
float32 products to float32 rows.  No golden file reaches the second case on its own (none has NVFP4 inputs and
adapters), so the decision is what is held, on each backend given.

    residual_decision.py FLYX ASSET_DIR BACKEND [BACKEND ...]
"""
import subprocess
import sys
from pathlib import Path

LINE = "prefill passes of more than 16 rows keep the residual in "


def decision(flyx, model, backend):
    run = subprocess.run([flyx, "run", str(model), "--backend", backend, "--activations", "nvfp4", "--ctx", "256",
                          "-p", "one two three", "-n", "1", "-v"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    for line in run.stderr.splitlines():
        if LINE in line:
            return line.split(LINE, 1)[1].strip()
    sys.exit(f"{backend}, {model.name}: no decision reported (exit {run.returncode}):\n{run.stderr[-3000:]}")


def main():
    flyx, assets, backends = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    failed = 0
    for backend in backends:
        for name, want in (("tiny_llama_nvfp4.flyx", "bf16"), ("tiny_llama_nvfp4_lora.flyx", "float32")):
            got = decision(flyx, assets / name, backend)
            ok = got == want
            failed += not ok
            print(f"  {'ok' if ok else 'FAIL':4} {backend}, {name}: the residual in {got} (want {want})")
    print("residual_decision:", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
