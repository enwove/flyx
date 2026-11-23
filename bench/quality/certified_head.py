# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""How many vocabulary rows a certified output head would have to score exactly.

    python bench/quality/certified_head.py --hf ~/models/Qwen3.5-4B [--text FILE ...] [--limit 2048]

A certified head scores every vocabulary row cheaply from a low-precision copy
of the head, with a rigorous bound on each score's error, and scores exactly
only the rows whose upper bound reaches the best lower bound.  The greedy token
(and, for top-k sampling, the top k) is then exactly the full head's.  What it
saves depends on how many rows survive.

For teacher-forced positions of the texts (by default docs/design/architecture.md
and core/exec/generate.cpp; --text adds files), with the target's final hidden
state h (transformers, bf16), this reports for each copy and bound the number
of rows to score exactly (median, p90, p99, max over positions; also the union
over 8 consecutive positions, as one speculative round reads it) and the bytes
one step would read: the copy, what the bound needs, and the surviving rows in
bf16; against the full head in W12 (1.5 bytes a weight), which the engine
reads now.

Copies (--copies): sign (1 bit a weight and a scale per row); b-bit integers,
round to nearest, with a scale per 32 weights (int4/32, int5/32, int6/32) or
per row (int5/row .. int8/row).  Bounds on |h . e_i|
(e_i = the row's error):
  row-cs    ||h|| ||e_i||                          (one float per row)
  group-cs  sum_g ||h_g|| ||e_ig||                 (one per 32 weights)
  group-max sum_g ||h_g||_1 max|e_ig|              (one per 32 weights)
Every bound holds for every position here (checked); the true greedy token is
always among the survivors.
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
G = 32  # weights per group


def load(hf: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.bfloat16).cuda().eval()
    return tok, model


@torch.no_grad()
def hidden_states(model, ids: list[int]) -> torch.Tensor:
    """The head's input at every position (float32 [T, H])."""
    x = torch.tensor([ids], device="cuda")
    out = model(x, output_hidden_states=True, logits_to_keep=1)
    h = out.hidden_states[-1][0]
    head = model.get_output_embeddings()
    if not torch.allclose(head(h[-1:]).float(), out.logits[0, -1:].float(), atol=1e-2, rtol=0):
        h = model.model.norm(h)  # hidden_states[-1] before the final norm
        assert torch.allclose(head(h[-1:]).float(), out.logits[0, -1:].float(), atol=1e-2, rtol=0)
    return h.float()


class Copy:
    """A low-precision copy of W [V, H] and the per-group / per-row error data its bounds use."""

    def __init__(self, name: str, W: torch.Tensor, bits_per_weight: float, extra_bytes_per_row: float):
        self.name = name
        V, H = W.shape
        # float32: the values a kernel would compute from the codes and scales (q * s exactly).
        self.W_hat = torch.empty(V, H, device=W.device)
        self.row_err = torch.empty(V, device=W.device)
        self.group_l2 = torch.empty(V, H // G, device=W.device)
        self.group_max = torch.empty(V, H // G, device=W.device)
        for a in range(0, V, 8192):  # in slices, to keep the temporaries small
            w = W[a:a + 8192].float()
            wh = self.quantize(w)
            e = (w - wh).view(w.shape[0], H // G, G)
            self.W_hat[a:a + 8192] = wh
            self.row_err[a:a + 8192] = e.flatten(1).norm(dim=1)
            self.group_l2[a:a + 8192] = e.norm(dim=2)
            self.group_max[a:a + 8192] = e.abs().amax(dim=2)
        self.copy_bytes = V * H * bits_per_weight / 8 + V * extra_bytes_per_row

    def quantize(self, w: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Sign(Copy):
    def __init__(self, W):
        super().__init__("sign (1 bit)", W, 1, 2)

    def quantize(self, w):
        return w.abs().mean(dim=1, keepdim=True) * torch.sign(w)


class IntRow(Copy):
    """b-bit integers with one scale per row."""

    def __init__(self, W, bits: int):
        self.qmax = 2 ** (bits - 1) - 1
        super().__init__(f"int{bits} / row", W, bits, 2)

    def quantize(self, w):
        s = (w.abs().amax(dim=1, keepdim=True) / self.qmax).to(torch.float16).float().clamp_min(1e-30)
        return torch.round(w / s).clamp(-self.qmax, self.qmax) * s


class IntGroup(Copy):
    """b-bit integers with one scale per 32 weights."""

    def __init__(self, W, bits: int):
        self.qmax = 2 ** (bits - 1) - 1
        super().__init__(f"int{bits} / 32", W, bits, 2 * W.shape[1] / G)

    def quantize(self, w):
        g = w.view(w.shape[0], -1, G)
        s = (g.abs().amax(dim=2, keepdim=True) / self.qmax).to(torch.float16).float().clamp_min(1e-30)
        return (torch.round(g / s).clamp(-self.qmax, self.qmax) * s).view_as(w)


class Fp8Row(Copy):
    """e4m3 codes with one scale per row, max|w| / 448: the CUDA backend's head_fp8 copy (quantize_fp8_rows)."""

    def __init__(self, W):
        super().__init__("fp8 / row", W, 8, 4)

    def quantize(self, w):
        s = w.abs().amax(dim=1, keepdim=True) / 448
        s = torch.where(s > 0, s, torch.ones_like(s))
        return (w / s).to(torch.float8_e4m3fn).float() * s


COPIES = {
    "sign": Sign, "int4/32": lambda W: IntGroup(W, 4), "fp8/row": Fp8Row,
    "int5/row": lambda W: IntRow(W, 5), "int6/row": lambda W: IntRow(W, 6), "int7/row": lambda W: IntRow(W, 7),
    "int8/row": lambda W: IntRow(W, 8), "int5/32": lambda W: IntGroup(W, 5), "int6/32": lambda W: IntGroup(W, 6),
}


@torch.no_grad()
def survivors(Wf: torch.Tensor, c: Copy, h: torch.Tensor, bound: str, k: int = 1, window: int = 8):
    """Per position: rows whose upper bound reaches the k-th best lower bound; and per window of positions, the union."""
    counts, unions, worst = [], [], 0.0
    T, H = h.shape
    for a in range(0, T, window):
        hb = h[a:a + window]
        exact = hb @ Wf.T  # [n, V]
        approx = hb @ c.W_hat.T
        hg = hb.view(hb.shape[0], H // G, G)
        if bound == "row-cs":
            B = hb.norm(dim=1, keepdim=True) * c.row_err[None, :]
        elif bound == "group-cs":
            B = hg.norm(dim=2) @ c.group_l2.T
        else:
            B = hg.abs().sum(dim=2) @ c.group_max.T
        B = B * (1 + 1e-5) + 1e-4  # float32 rounding of the products above
        worst = max(worst, float(((exact - approx).abs() - B).max()))
        lb_k = torch.topk(approx - B, k, dim=1).values[:, -1:]
        keep = (approx + B) >= lb_k
        assert bool(keep.gather(1, exact.argmax(dim=1, keepdim=True)).all())
        counts += keep.sum(dim=1).tolist()
        unions.append(int(keep.any(dim=0).sum()))
    return counts, unions, worst


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--text", action="append", default=[])
    ap.add_argument("--limit", type=int, default=2048, help="positions per text")
    ap.add_argument("--k", type=int, default=1, help="exact top k (1: greedy)")
    ap.add_argument("--copies", default="sign,int4/32,int5/32,int6/32,int5/row,int6/row,int7/row,int8/row")
    ap.add_argument("--bounds", default="row-cs,group-cs,group-max")
    a = ap.parse_args()
    tok, model = load(a.hf)
    W = model.get_output_embeddings().weight.detach()  # [V, H] bf16
    V, H = W.shape
    files = [os.path.join(ROOT, "docs/design/architecture.md"), os.path.join(ROOT, "core/exec/generation.cpp")] + a.text
    hs = []
    for f in files:
        with open(f, encoding="utf-8") as fp:
            ids = tok(fp.read())["input_ids"][:a.limit]
        hs.append(hidden_states(model, ids))
        print(f"{os.path.relpath(f, ROOT) if f.startswith(ROOT) else os.path.basename(f)}: {len(ids)} positions")
    W = W.clone()
    del model
    torch.cuda.empty_cache()
    Wf = W.float()
    h = torch.cat(hs)
    full = V * H * 1.5
    print(f"vocab {V}, hidden {H}; {h.shape[0]} positions; exact top {a.k}; the full head in W12: {full / 1e6:.0f} MB")
    print(f"{'copy':<14} {'bound':<10} {'rows: p50':>10} {'p90':>7} {'p99':>7} {'max':>7} {'8-pos union p50':>16} "
          f"{'MB a step':>10} {'MB a round of 8':>16}")
    for name in a.copies.split(","):
        c = COPIES[name](W)
        for bound in a.bounds.split(","):
            counts, unions, worst = survivors(Wf, c, h, bound, a.k)
            if worst > 0:
                print(f"  bound violated by {worst:.3g}: not rigorous")
            bound_bytes = V * 4 if bound == "row-cs" else V * (H // G) * 2
            step = c.copy_bytes + bound_bytes + statistics.median(counts) * H * 2
            rnd = c.copy_bytes + bound_bytes + statistics.median(unions) * H * 2
            print(f"{c.name:<14} {bound:<10} {statistics.median(counts):>10.0f} {pct(counts, .9):>7} "
                  f"{pct(counts, .99):>7} {max(counts):>7} {statistics.median(unions):>16.0f} "
                  f"{step / 1e6:>10.0f} {rnd / 1e6:>16.0f}")
        del c
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
