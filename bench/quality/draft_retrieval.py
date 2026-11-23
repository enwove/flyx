# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Candidate retrieval for draft logits: is the exact head's
argmax (and its top 16) among the K best scores of a cheap approximation?

    python bench/quality/draft_retrieval.py HF_DIR TEXT... [--fit FIT.txt]

The head's input comes from the transformers model on each text (the hidden
state after the final norm, at most 2000 positions a text).  Approximations:
SVD of the head at rank r; the same weighted by the hidden states' second
moment (--fit: states of another text); signs with one scale (mean |w|) per
2560 or per 64 weights; 2 bits with one scale per 16; the first 151,643 ids.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


def hidden_states(model, tok, path: str, n: int = 2000) -> torch.Tensor:
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[:, :n].cuda()
    with torch.no_grad():
        return model(ids, output_hidden_states=True).hidden_states[-1][0].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hf")
    ap.add_argument("texts", nargs="+")
    ap.add_argument("--fit", help="text whose hidden states weight the SVD")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.hf)
    model = AutoModelForCausalLM.from_pretrained(a.hf, dtype=torch.bfloat16).cuda()
    h = torch.cat([hidden_states(model, tok, p) for p in a.texts])
    hfit = hidden_states(model, tok, a.fit) if a.fit else None
    del model
    torch.cuda.empty_cache()
    idx = json.load(open(os.path.join(a.hf, "model.safetensors.index.json")))
    name = "model.language_model.embed_tokens.weight"  # Qwen3.5: the tied head
    with safe_open(os.path.join(a.hf, idx["weight_map"][name]), "pt", device="cuda") as f:
        W = f.get_tensor(name).float()
    V, d = W.shape
    print(f"{h.shape[0]} positions, head {V} x {d}")
    chunks = list(torch.split(torch.arange(h.shape[0], device="cuda"), 512))  # positions at a time (memory)
    top1, top16 = [], []
    for c in chunks:
        e = h[c] @ W.T
        top1.append(e.argmax(1))
        top16.append(e.topk(16, 1).indices)

    def report(what: str, score, ks=(64, 128, 256, 512)) -> None:
        """score(positions) -> rough scores [len(positions), V]."""
        hit1 = {k: 0.0 for k in ks}
        hit16 = {k: 0.0 for k in ks}
        for c, t1, t16 in zip(chunks, top1, top16):
            sc = score(c)
            for k in ks:
                cand = sc.topk(k, 1).indices
                hit1[k] += (cand == t1[:, None]).any(1).float().sum().item()
                hit16[k] += torch.stack([(cand == t16[:, j:j + 1]).any(1) for j in range(16)], 1).float().sum().item()
        for k in ks:
            print(f"  {what:40s} K {k:4d}: top-1 {hit1[k] / h.shape[0]:.4f}, top-16 recall {hit16[k] / h.shape[0] / 16:.4f}")

    ev, vec = torch.linalg.eigh((W.T @ W).double())
    vec = vec.flip(1).float()
    for r in (128, 256):
        B = vec[:, :r].T.contiguous()
        A = W @ B.T
        report(f"SVD rank {r}", lambda c: (h[c] @ B.T) @ A.T)
    if hfit is not None:
        C = (hfit.T @ hfit / hfit.shape[0]).double()
        lam, Q = torch.linalg.eigh(C)
        M = W.double() @ ((Q * lam.clamp(min=0).sqrt()) @ Q.T)
        ev2, R = torch.linalg.eigh(M.T @ M)
        R = R.flip(1)
        for r in (128, 256):
            A = ((M @ R[:, :r]) / ev2.flip(0)[:r].clamp(min=1e-30).sqrt()).float()  # top left singular vectors
            B = (A.T @ W).contiguous()
            report(f"SVD rank {r}, weighted by hidden states", lambda c: (h[c] @ B.T) @ A.T)
        del M
    for bits, group in ((1, d), (1, 64), (2, 16)):
        Wg = W.view(V, d // group, group)
        if bits == 1:
            Wq = torch.sign(Wg) * Wg.abs().mean(2, keepdim=True)
        else:
            step = Wg.abs().amax(2, keepdim=True).clamp(min=1e-30) / 1.5
            Wq = (torch.clamp(torch.floor(Wg / step), -2, 1) + 0.5) * step
        Wq = Wq.view(V, d)
        report(f"{bits}-bit, a scale per {group}", lambda c: h[c] @ Wq.T)
        del Wq

    def cut(c):
        e = h[c] @ W.T
        e[:, 151643:] = -1e30
        return e
    report("the first 151,643 ids", cut, ks=(1,))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
