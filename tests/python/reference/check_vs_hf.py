# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check the reference Qwen3.5 against HuggingFace transformers.

    python tests/python/reference/check_vs_hf.py HF_DIR MODEL.flyx \
        [--device cuda] [--dtype float64|float32|bfloat16]

Compares logits for a full-sequence forward and for token-by-token decode
(our reference keeps its own state; transformers uses its cache).  Also
checks the tap convention used by DFlash: tap n == output_hidden_states[n].
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
from flyx_ref.load import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir"); ap.add_argument("model")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float64")
    ap.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    dt = getattr(torch, a.dtype)
    hf = AutoModelForCausalLM.from_pretrained(a.hf_dir, dtype=dt).to(a.device).eval()
    target, _, header, _ = load(a.model, dtype=dt, device=a.device)
    V = header["config"]["vocab"]
    g = torch.Generator().manual_seed(7)
    ids = torch.randint(0, min(V, 600), (a.tokens,), generator=g)
    taps = (2, 5, 7) if header["config"]["layers"] >= 8 else ()
    with torch.no_grad():
        out = hf(ids[None].to(a.device), output_hidden_states=True)
        st = target.new_state()
        ours, tapped, _ = target.forward(ids.to(a.device), st, taps=taps)
        full = (out.logits[0].to(torch.float64) - ours.to(torch.float64)).abs().max().item()
        tap_err = max([(out.hidden_states[n][0] - tapped[n]).abs().max().item() for n in taps] or [0.0])
        # incremental: prefix then one token at a time
        st2 = target.new_state()
        target.forward(ids[:8].to(a.device), st2)
        inc = 0.0
        for t in range(8, a.tokens):
            lg, _, _ = target.forward(ids[t:t + 1].to(a.device), st2)
            inc = max(inc, (out.logits[0, t].to(torch.float64) - lg[0].to(torch.float64)).abs().max().item())
    scale = out.logits.abs().max().item()
    print(f"max|logit|={scale:.3g}  full-forward err={full:.3g}  incremental err={inc:.3g}  tap err={tap_err:.3g}")
    # transformers runs RoPE angles and the gated-delta rule in float32 even for
    # float64 models, so float32-level agreement is the best possible here.
    tol = 5e-6 * scale if dt == torch.float64 else 5e-2 * scale
    ok = full < tol and inc < tol and tap_err < tol * 10
    print("OK" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
