# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The bf16 prefill residual's decision, as each backend reports it at load (-v), held on runs that end well: every
file is run through a prompt of 40 tokens (prefill passes of more than 16 rows, where the decision is taken), and a
run must exit 0 with tokens.  A stack whose every residual add is a W4A4 GEMM keeps a prefill pass's residual in bf16
(tiny_llama_nvfp4.flyx, tiny_nvfp4.flyx and tiny_w4a4.flyx with --activations nvfp4); the same stack with the
target's own adapters (ABI 43, tiny_llama_nvfp4_lora.flyx) or a PEFT adapter on its gated delta nets alone (--lora)
keeps it in float32, as its adapters add float32 products to float32 rows.

The GDN-only adapter runs where every backend takes NVFP4 inputs: tiny_w4a4.flyx (128 wide, every linear's K a whole
number of the W4A4 kernels' tiles of 128) with an adapter of rank 8 on its GDN layers' qkv, z, a and out, made here.
Its tokens are held to the float64 backend's with the same adapter (the reference), on PROMPTS: that backend's greedy
tokens differ from the model's alone (the adapter is run), and another backend's are held to them teacher forced, each
place from the reference's tokens before it, both backends' next token from the same prefill pass.  On this random
tiny model a float32 activation rounds to another NVFP4 value than float64's often enough to turn near-ties (the model
alone, or an adapter of zeros, differs from the reference at as many places), so the measure is relative: the backend
may differ from the reference at no more than a quarter of the places where the adapter itself changes the
reference's token (measured on an RTX 5070 Ti: 6 of 135; an adapter left out or at half its scale: 123 or more).
peft_qwen35's GDN tensors beside tiny_nvfp4.flyx (64 wide) are held on the float64 backend, which takes NVFP4 inputs
at any K; the CUDA backend refuses NVFP4 inputs there (K a multiple of 128), and that refusal is held as its answer
for that file alone.  The CUDA backend computes NVFP4 inputs on FP4 tensor cores, a device of compute capability 12.0
(nvidia-smi's compute_cap of the GPU the tests take) in a build with their kernels (--fp4-kernels: FLYX_CUDA_ARCHS
takes 120), and every other file must run there; on any other device or build it refuses them all at open, and that
refusal is held as its answer (the GDN-only run then not held there).  Where nvidia-smi does not say which device it
is, a refusal is accepted only of them all.

    residual_decision.py FLYX ASSET_DIR [--fp4-kernels] BACKEND [BACKEND ...]
"""
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

LINE = "prefill passes of more than 16 rows keep the residual in "
REFUSED = "NVFP4 inputs, which this device or build cannot compute"  # backends/cuda/backend.cpp: nvfp4_inputs
NARROW = "NVFP4 inputs need K a multiple of 128"  # the CUDA backend, for the tiny Qwen3.5's widths
REFERENCE = "cpu_ref"  # the float64 backend
NARROW_FILES = {("cuda", "tiny_nvfp4.flyx")}  # where a refusal for K is the answer
PROMPT = [(7 * i + 20) % 700 for i in range(40)]  # prefill passes of more than 16 rows
PROMPTS = [[(a * i + b) % 700 for i in range(40)] for a, b in ((7, 20), (13, 5), (31, 101), (3, 11))]
FORCED = 24  # teacher-forced places a prompt


def bench(flyx, model, backend, lora, prompts, n, tmp):
    """`flyx bench` of `prompts` (token ids, each in a session of its own) with NVFP4 inputs: its exit, the decision
    it reported, each prompt's new tokens, and its stderr."""
    tag = f"{backend}-{model.stem}-{lora.name if lora else 'alone'}-{len(prompts)}"
    asked, record = tmp / f"{tag}.jsonl", tmp / f"{tag}.json"
    asked.write_text("".join(json.dumps({"ids": p}) + "\n" for p in prompts), encoding="utf-8")
    run = subprocess.run([flyx, "bench", str(model), "--prompts", str(asked), "-n", str(n), "--ignore-eos",
                          "--backend", backend, "--activations", "nvfp4", "--ctx", "256", "--no-spec", "--no-lookup",
                          "-v", "--json", str(record)] + (["--lora", str(lora)] if lora else []),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    said = [line.split(LINE, 1)[1].strip() for line in run.stderr.splitlines() if LINE in line]
    tokens = None
    if run.returncode == 0 and record.exists():
        tokens = [p["ids"] for p in json.loads(record.read_text(encoding="utf-8"))["prompts"]]
    return run.returncode, said[-1] if said else None, tokens, run.stderr


def bf16_held(x):
    """x rounded to the nearest bf16 value, as a float32 the engines read it as."""
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def write_adapter(dst, tensors, cfg):
    """A PEFT adapter directory: tensors {name: (rows, cols, float32 values)} in adapter_model.safetensors."""
    header, parts, at = {}, [], 0
    for name, (rows, cols, values) in tensors.items():
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": [rows, cols], "data_offsets": [at, at + len(data)]}
        parts.append(data)
        at += len(data)
    text = json.dumps(header).encode()
    dst.mkdir()
    (dst / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(text)) + text + b"".join(parts))
    (dst / "adapter_config.json").write_text(json.dumps(cfg), encoding="utf-8")


def gdn_w4a4(hf_dir, dst, rank=8, alpha=16):
    """An adapter of rank `rank` on the GDN layers' in_proj_qkv, in_proj_z, in_proj_a and out_proj of the model whose
    Hugging Face config is hf_dir's (none on in_proj_b: zeros beside them), values bf16 holds, drawn from a seeded
    generator of Python's own."""
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    cfg = cfg.get("text_config", cfg)
    H = cfg["hidden_size"]
    kd = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    vd = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    shapes = {"in_proj_qkv": (2 * kd + vd, H), "in_proj_z": (vd, H), "in_proj_a": (cfg["linear_num_value_heads"], H),
              "out_proj": (H, vd)}
    gen = random.Random(25)
    tensors = {}
    for layer, kind in enumerate(cfg["layer_types"]):
        if kind != "linear_attention":
            continue
        for proj, (out, inp) in shapes.items():
            name = f"base_model.model.model.layers.{layer}.linear_attn.{proj}"
            tensors[name + ".lora_A.weight"] = (rank, inp, [bf16_held(gen.gauss(0, 1) / math.sqrt(inp))
                                                            for _ in range(rank * inp)])
            tensors[name + ".lora_B.weight"] = (out, rank, [bf16_held(gen.gauss(0, 1) * 2 / math.sqrt(rank))
                                                            for _ in range(out * rank)])
    write_adapter(dst, tensors, {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha,
                                 "target_modules": list(shapes)})


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


def cuda_fp4_inputs(fp4_kernels):
    """Whether the CUDA backend can compute NVFP4 inputs here, from the device and the build rather than from what the
    backend says of itself: (True or False, the device), or (None, why) where nvidia-smi does not name the GPU the tests
    take (the first of CUDA_VISIBLE_DEVICES, else GPU 0)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None, "no nvidia-smi on PATH"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    pick = ["-i", visible.split(",")[0].strip()] if visible else []
    try:
        run = subprocess.run([smi, "--query-gpu=name,compute_cap", "--format=csv,noheader", *pick],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"nvidia-smi: {e}"
    fields = [f.strip() for f in run.stdout.strip().splitlines()[0].split(",")] if run.stdout.strip() else []
    if run.returncode != 0 or len(fields) != 2:
        return None, f"nvidia-smi names no GPU ({(run.stdout + run.stderr).strip() or 'no output'})"
    name, cc = fields
    device = f"{name}, compute capability {cc}, a build {'with' if fp4_kernels else 'without'} the FP4 kernels"
    return cc == "12.0" and fp4_kernels, device


class Held:
    def __init__(self):
        self.failed = 0

    def __call__(self, ok, what):
        self.failed += not ok
        print(f"  {'ok' if ok else 'FAIL':4} {what}")


def main():
    flyx, assets = sys.argv[1], Path(sys.argv[2])
    fp4_kernels = "--fp4-kernels" in sys.argv[3:]
    backends = [b for b in sys.argv[3:] if b != "--fp4-kernels"]
    held = Held()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gdn64, gdn128 = tmp / "peft_qwen35_gdn", tmp / "peft_w4a4_gdn"
        gdn_only(assets / "peft_qwen35", gdn64)
        gdn_w4a4(assets / "hf_w4a4", gdn128)
        cases = (("tiny_llama_nvfp4.flyx", None, "bf16"), ("tiny_llama_nvfp4_lora.flyx", None, "float32"),
                 ("tiny_nvfp4.flyx", None, "bf16"), ("tiny_nvfp4.flyx", gdn64, "float32"),
                 ("tiny_w4a4.flyx", None, "bf16"), ("tiny_w4a4.flyx", gdn128, "float32"))
        w4a4 = assets / "tiny_w4a4.flyx"
        # The reference's greedy tokens for tiny_w4a4.flyx with the GDN-only adapter, and without it.
        _, _, greedy, err = bench(flyx, w4a4, REFERENCE, gdn128, PROMPTS, FORCED, tmp)
        _, _, alone, _ = bench(flyx, w4a4, REFERENCE, None, PROMPTS, FORCED, tmp)
        held(greedy is not None and alone is not None and all(len(t) == FORCED for t in greedy),
             f"{REFERENCE}, tiny_w4a4.flyx with and without a GDN-only adapter: {FORCED} tokens a prompt" +
             ("" if greedy else "\n" + err[-2000:]))
        if greedy is None or alone is None:
            print("residual_decision: FAILED")
            return 1
        held(all(a != b for a, b in zip(greedy, alone)),
             f"{REFERENCE}, tiny_w4a4.flyx: the GDN-only adapter changes every prompt's tokens (it is run)")
        forced_prompts = [p + t[:i] for p, t in zip(PROMPTS, greedy) for i in range(FORCED)]
        reference = {}

        def forced(backend, lora):
            """Each teacher-forced place's next token (forced_prompts), or None for a run that failed."""
            code, _, got, err = bench(flyx, w4a4, backend, lora, forced_prompts, 1, tmp)
            return [t[0] for t in got] if code == 0 and got else None

        for backend in backends:
            refused = []  # the cases refused for want of NVFP4 inputs on this device
            for name, lora, decided in cases:
                what = f"{backend}, {name}" + (" with an adapter on its GDN layers alone" if lora else "")
                code, got, tokens, err = bench(flyx, assets / name, backend, lora, [PROMPT], 4, tmp)
                if code != 0 and REFUSED in err and backend != REFERENCE:
                    refused.append(what)
                    continue
                if (backend, name) in NARROW_FILES:
                    held(code != 0 and NARROW in err,
                         f"{what}: NVFP4 inputs refused for its widths (K not a multiple of 128)" +
                         ("" if code != 0 else f" (it ran: the residual in {got})"))
                    continue
                ok = code == 0 and got == decided and tokens is not None and len(tokens[0]) == 4
                held(ok, f"{what}: the residual in {got} (want {decided}), exit {code}, " +
                     f"{len(tokens[0]) if tokens else 0} tokens" + ("" if ok else "\n" + err[-2000:]))
                if not ok or name != "tiny_w4a4.flyx" or lora is None or backend == REFERENCE:
                    continue
                if not reference:  # the reference's own next tokens there, with the adapter and without it
                    reference.update(adapted=forced(REFERENCE, gdn128), alone=forced(REFERENCE, None))
                mine = forced(backend, lora)
                if not (reference["adapted"] and reference["alone"] and mine):
                    held(False, f"{what}: teacher forced, a run failed")
                    continue
                effect = sum(a != b for a, b in zip(reference["adapted"], reference["alone"]))
                apart = sum(a != b for a, b in zip(reference["adapted"], mine))
                places = len(forced_prompts)
                held(effect >= places // 2 and 4 * apart <= effect,
                     f"{what}: teacher forced at {places} places, {REFERENCE}'s token at all but {apart}, a quarter "
                     f"at most of the {effect} the adapter changes")
            # The cases whose widths take NVFP4 inputs (the others' refusal for K comes first, on any device).
            wide = sum((backend, name) not in NARROW_FILES for name, _, _ in cases)
            can, device = cuda_fp4_inputs(fp4_kernels) if backend == "cuda" else (None, "")
            if can is True:
                held(not refused, f"{backend}: NVFP4 inputs computed on FP4 tensor cores ({device}): refused in "
                                  f"{len(refused)} of {len(cases)} cases" + "".join(f"\n    {w}" for w in refused))
            elif can is False:
                held(len(refused) == wide, f"{backend}: NVFP4 inputs refused where they cannot be computed "
                                           f"({device}), in {len(refused)} of {wide} cases (all of them)")
            elif refused:  # the device unknown: every case refused there (no FP4 tensor cores), or none
                held(len(refused) == wide,
                     f"{backend}: NVFP4 inputs refused in {len(refused)} of {wide} cases (all of them, or none; "
                     f"the device's FP4 tensor cores not known: {device or 'not the CUDA backend'})")
    print("residual_decision:", "FAILED" if held.failed else "PASSED")
    return 1 if held.failed else 0


if __name__ == "__main__":
    sys.exit(main())
