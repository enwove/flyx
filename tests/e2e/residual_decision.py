# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The bf16 prefill residual's decision, as each backend reports it at load (-v): a stack whose every residual add is
a W4A4 GEMM keeps a prefill pass's residual in bf16 (tiny_llama_nvfp4.flyx with --activations nvfp4), and the same
stack with the target's own adapters (ABI 43, tiny_llama_nvfp4_lora.flyx) keeps it in float32, as its adapters add
float32 products to float32 rows.  So does the tiny Qwen3.5 (tiny_nvfp4.flyx, gated delta nets) with a PEFT adapter
on its GDN layers alone (peft_qwen35's linear_attn tensors, given with --lora), where the backend takes NVFP4 inputs
at its widths (cpu_ref; CUDA refuses them below K = 128, and its residual16_model reads the same predicate).  No
golden file reaches these cases on its own (none has NVFP4 inputs and adapters), so the decision is what is held, on
each backend given.  A device without FP4 tensor cores computes no NVFP4 inputs and refuses the file at open
(gguf_test's nvfp4_inputs holds it to that refusal): there is no residual to decide there, and the refusal is what
is held.

    residual_decision.py FLYX ASSET_DIR BACKEND [BACKEND ...]
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

LINE = "prefill passes of more than 16 rows keep the residual in "
REFUSED = "NVFP4 inputs, which this device or build cannot compute"  # backends/cuda/backend.cpp: nvfp4_inputs
NARROW = "NVFP4 inputs need K a multiple of 128"  # the CUDA backend, for the tiny Qwen3.5's widths


def decision(flyx, model, backend, lora=None):
    run = subprocess.run([flyx, "run", str(model), "--backend", backend, "--activations", "nvfp4", "--ctx", "256",
                          "-p", "one two three", "-n", "1", "-v"] + (["--lora", str(lora)] if lora else []),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in run.stderr.splitlines():
        if LINE in line:
            return line.split(LINE, 1)[1].strip()
    if run.returncode != 0 and REFUSED in run.stderr:
        return None
    if run.returncode != 0 and NARROW in run.stderr:
        return "narrow"
    sys.exit(f"{backend}, {model.name}: no decision reported (exit {run.returncode}):\n{run.stderr[-3000:]}")


def gdn_only(src, dst):
    """src's adapter with its linear_attn tensors alone (a safetensors file rewritten with them)."""
    blob = (src / "adapter_model.safetensors").read_bytes()
    n = struct.unpack("<Q", blob[:8])[0]
    header, data = json.loads(blob[8:8 + n]), blob[8 + n:]
    out, parts, at = {}, [], 0
    for key, t in header.items():
        if key == "__metadata__" or ".linear_attn." not in key:
            continue
        first, last = t["data_offsets"]
        out[key] = dict(t, data_offsets=[at, at + last - first])
        parts.append(data[first:last])
        at += last - first
    text = json.dumps(out).encode()
    dst.mkdir()
    (dst / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(text)) + text + b"".join(parts))
    cfg = json.loads((src / "adapter_config.json").read_text())
    cfg["target_modules"] = ["in_proj_qkv", "in_proj_z", "in_proj_a", "out_proj"]
    (dst / "adapter_config.json").write_text(json.dumps(cfg))


def main():
    flyx, assets, backends = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        gdn = Path(tmp) / "peft_qwen35_gdn"
        gdn_only(assets / "peft_qwen35", gdn)
        for backend in backends:
            for name, lora, want in (("tiny_llama_nvfp4.flyx", None, "bf16"),
                                     ("tiny_llama_nvfp4_lora.flyx", None, "float32"),
                                     ("tiny_nvfp4.flyx", None, "bf16"), ("tiny_nvfp4.flyx", gdn, "float32")):
                what = name + (" with an adapter on its GDN layers alone" if lora else "")
                got = decision(flyx, assets / name, backend, lora)
                if got is None:
                    print(f"  ok   {backend}, {what}: NVFP4 inputs refused where the device cannot compute them")
                    continue
                if got == "narrow":  # no NVFP4 inputs for its linears' widths there: no residual to decide
                    print(f"  ok   {backend}, {what}: NVFP4 inputs refused for its widths (K not a multiple of 128)")
                    continue
                ok = got == want
                failed += not ok
                print(f"  {'ok' if ok else 'FAIL':4} {backend}, {what}: the residual in {got} (want {want})")
    print("residual_decision:", "FAILED" if failed else "PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
