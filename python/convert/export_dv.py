# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A DV drafter (bifill-dv's canvas adapter and path selector, PyTorch .pt) -> the directory flyx_convert --dv reads.

    python -I python/convert/export_dv.py --canvas dva_res2.pt --selector sel_orig_w8.pt -o minicpm5-dv/ \\
        [--slots 32] [--tree beam:31:8:cov | pool:31:7:1] [--fill 0.9] [--mask-token "<unused_token_477>"] \\
        [--vlora lora_g_b2h.pt]
    python -I python/convert/export_dv.py --vlora lora_g_b2h.pt -o minicpm5-vlora/       # a V-LoRA alone

Writes canvas.safetensors, selector.safetensors and dv.json (python/convert/flyx_convert/dv.py says what each holds);
with --vlora (bifill-dv's V-side LoRA, raft.py's checkpoint: the target's own adapters, which its evalgen.py VLORA runs
unmerged on every row of the target and never on the canvas's) also vlora.safetensors and vlora.json
(python/convert/flyx_convert/lora.py), its adapters in bf16 as evalgen.py casts them, its scale 1 (raft.py's LoRA has
none: y = W x + B (A x); --vlora-scale for another's), the checkpoint's sha256.  flyx_convert --dv takes the V-LoRA of
the directory with the DV (the canvas and selector were made for that target), --lora a directory of a V-LoRA alone.
The canvas runs in the model's dtype (bf16), as bifill-dv's engine casts it, so its tensors are stored in bf16
(rounded to nearest even); the selector runs in float32 and stays float32.  --slots: the canvas's slots (default the
checkpoint's, cfg n); --tree: the tree a round drafts (default bifill-dv's DYN=31:8:cov); --fill: the probability from
which a slot's top token is its next input, a setting of the run as in bifill-dv's evalgen.py (TAUS, else DV_FILL;
default 0.9, its TAUS), not the checkpoint's training TAU (cfg tau, kept in dv.json's source).  Checkpoints are read
with torch.load(weights_only=True), never unpickling code; this script needs torch, flyx_convert does not.  Refused: a
canvas with mixers, a spine, a see-gate, a second residual or a candidate memory, one without its residual context,
one whose cfg changes the canvas pass with no tensor of its own (soft_in: a MASK slot's input its z; see_guess: the
canvas also attends to the guesses), an expected embedding of other than 16 candidates (res_k), a two-token or MLP
selector, a tree that cannot be drafted on the slots; a V-LoRA of partial layers, ranks apart, adapters that do not fit
one layer's linears, or a cfg that names a scale of its own.  Ported from the earlier CUDA port's export_dv.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flyx_convert.dv import LINEARS, TOPK, parse_tree, tree_spec  # noqa: E402

MASK_TOKEN = "<unused_token_477>"  # bifill-dv's MASK on MiniCPM5 (id 130559)
DEFAULT_TREE = "beam:31:8:cov"
DEFAULT_FILL = 0.9  # bifill-dv's TAUS default (evalgen.py), what its record runs set
# A canvas checkpoint's cfg flags that change what the canvas computes (bifill-dv's train_dv.py and evalgen.py); none
# of them is in DV's blocks here.  Some add tensors as well (refused by name below), soft_in and see_guess none.
CANVAS_FLAGS = ("soft_in", "see_guess", "see_gate", "residual_h", "spine", "mem_layers")


def canvas_layers(cfg: dict, names) -> int:
    """The canvas's layers from its checkpoint's cfg and tensor names; refused unless it is the plain residual canvas
    (no mixers, spine, see-gate, second residual, candidate memory, soft inputs or guesses seen; z over 16 candidates)
    with every tensor a layer of it has."""
    if cfg.get("mixers"):
        raise ValueError("a canvas with mixers is not supported")
    if not cfg.get("residual"):
        raise ValueError("a canvas without its residual context is not supported")
    flags = [f for f in CANVAS_FLAGS if cfg.get(f)]
    if flags:
        raise ValueError(f"a canvas trained with {', '.join(flags)} is not supported (DV here runs the plain residual "
                         "canvas)")
    if int(cfg.get("res_k", TOPK)) != TOPK:
        raise ValueError(f"a canvas whose expected embedding sums {cfg['res_k']} candidates is not supported ({TOPK})")
    names = set(names)
    odd = sorted(n for n in names if n.startswith(("spine", "mixers", "res_h")) or "see_gate" in n)
    if odd:
        raise ValueError(f"a canvas with a spine, see-gate or second residual is not supported ({odd[:3]})")
    layer_ids = {int(n.split(".")[1]) for n in names if n.startswith("layers.")}
    if not layer_ids:
        raise ValueError("a canvas without layers")
    layers = 1 + max(layer_ids)
    want = {"norm", "mask_emb", "res_in.weight", "res_out.weight"}
    for layer in range(layers):
        want |= {f"layers.{layer}.ln1", f"layers.{layer}.ln2"}
        want |= {f"layers.{layer}.{n}.{m}" for n in LINEARS for m in ("A", "B")}
    if names != want:
        raise ValueError(f"canvas tensors: missing {sorted(want - names)[:5]}, unknown {sorted(names - want)[:5]}")
    return layers


def check_selector(ck: dict) -> int:
    """The selector checkpoint's rank; refused unless it is the one-token bilinear selector."""
    if ck.get("two") or ck.get("mlp"):
        raise ValueError("two-token and MLP selectors are not supported")
    names = set(ck["sel"])
    want = {"A.weight", "H.weight", "B.weight", "alpha"}
    if names != want:
        raise ValueError(f"selector tensors: missing {sorted(want - names)}, unknown {sorted(names - want)}")
    return int(ck["r"])


# Keys a LoRA's cfg would hold a scale under (PEFT's lora_alpha, others' alpha or scale): raft.py's has none.
LORA_SCALE_KEYS = ("alpha", "lora_alpha", "scale", "scaling")


def check_vlora(ck: dict) -> tuple[int, int]:
    """bifill-dv's V-LoRA checkpoint (raft.py add_v_lora: {'lora': per layer q, k, v, o, gate, up, down, each
    {'A' [rank, in], 'B' [out, rank]}, 'cfg': {'r': rank, ...}}) -> (layers, rank).  Refused: adapters that are not
    whole layers of 7, of ranks apart (or not its cfg's), that do not fit one layer's linears (q, k, v, gate, up read
    the hidden size, o reads q's outputs and down gate's and up's, o and down write the hidden size), or a cfg that
    names a scale of its own (raft.py's LoRA multiplies by none)."""
    lora = ck.get("lora")
    if not isinstance(lora, (list, tuple)) or not lora or len(lora) % len(LINEARS):
        raise ValueError(f"a V-LoRA of {len(lora) if isinstance(lora, (list, tuple)) else 'no'} adapters "
                         f"({len(LINEARS)} a layer: {', '.join(LINEARS)})")
    cfg = ck.get("cfg") or {}
    named = [k for k in LORA_SCALE_KEYS if k in cfg]
    if named:
        raise ValueError(f"a V-LoRA whose cfg names {named}: raft.py's LoRA has no scale (give --vlora-scale for "
                         "another's)")
    rank = int(cfg["r"]) if "r" in cfg else int(lora[0]["A"].shape[0])
    for i, ab in enumerate(lora):
        if set(ab) != {"A", "B"}:
            raise ValueError(f"V-LoRA adapter {i}: tensors {sorted(ab)}, not A and B")
        A, B = ab["A"], ab["B"]
        if len(A.shape) != 2 or len(B.shape) != 2 or A.shape[0] != rank or B.shape[1] != rank:
            raise ValueError(f"V-LoRA adapter {i} ({LINEARS[i % 7]} of layer {i // 7}): A {list(A.shape)}, B "
                             f"{list(B.shape)}: not of rank {rank}")
    for layer in range(len(lora) // 7):
        q, k, v, o, g, u, d = lora[7 * layer:7 * layer + 7]
        H = q["A"].shape[1]
        fits = (all(x["A"].shape[1] == H for x in (k, v, g, u)) and k["B"].shape[0] == v["B"].shape[0] and
                o["A"].shape[1] == q["B"].shape[0] and o["B"].shape[0] == H and
                g["B"].shape[0] == u["B"].shape[0] == d["A"].shape[1] and d["B"].shape[0] == H)
        if not fits:
            raise ValueError(f"V-LoRA layer {layer}: adapters that do not fit one layer's linears")
    return len(lora) // 7, rank


def export_vlora(ck: dict, out: str, source: dict | None = None, scale: float = 1.0) -> dict:
    """bifill-dv's V-LoRA as loaded -> out/vlora.safetensors (bf16, as evalgen.py casts it: nearest even) and
    out/vlora.json; returns vlora.json's contents."""
    import torch
    from safetensors.torch import save_file

    layers, rank = check_vlora(ck)
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError(f"V-LoRA scale {scale}: a finite number other than 0")
    t = {}
    for i, ab in enumerate(ck["lora"]):
        n = f"L{i // 7}.{LINEARS[i % 7]}"
        t[n + ".lora_a"] = ab["A"].detach().to(torch.bfloat16).contiguous()
        t[n + ".lora_b"] = ab["B"].detach().to(torch.bfloat16).contiguous()
    src = dict(source or {})
    if "step" in ck:
        src["step"] = int(ck["step"])
    if "name" in (ck.get("cfg") or {}):
        src["name"] = ck["cfg"]["name"]
    meta = {"kind": "lora", "layers": layers, "rank": rank, "scale": float(scale), "source": src}
    os.makedirs(out, exist_ok=True)
    save_file(t, os.path.join(out, "vlora.safetensors"))
    with open(os.path.join(out, "vlora.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dv_meta(cfg: dict, layers: int, sel_rank: int, slots: int | None = None, tree: str = DEFAULT_TREE,
            fill: float = DEFAULT_FILL, mask_token=MASK_TOKEN, source: dict | None = None) -> dict:
    """dv.json's contents for a canvas checkpoint's cfg (checked by canvas_layers): the run's settings (slots, tree,
    fill, the MASK token) and the checkpoint's own (ranks, causal); its training TAU in source.  Refused: a tree that
    cannot be drafted on the slots, a fill that is not a probability."""
    slots = int(cfg["n"]) if slots is None else int(slots)
    if not 0.0 <= float(fill) <= 1.0:
        raise ValueError(f"fill {fill} is not a probability")
    src = dict(source or {})
    if "tau" in cfg:
        src["tau"] = float(cfg["tau"])  # the slide's threshold in training; the run's is fill
    return {"kind": "dv", "layers": layers, "slots": slots, "rank": int(cfg["rank"]), "res_k": TOPK,
            "causal": bool(cfg.get("canvas_causal", False)), "fill": float(fill), "topk": TOPK,
            "tree": tree_spec(parse_tree(tree, slots)), "mask_token": mask_token, "selector": {"rank": sel_rank},
            "source": src}


def export(canvas: dict, selector: dict, out: str, slots: int | None = None, tree: str = DEFAULT_TREE,
           mask_token=MASK_TOKEN, source: dict | None = None, fill: float = DEFAULT_FILL) -> dict:
    """bifill-dv's checkpoints as loaded ({'adapter', 'cfg'}, {'sel', 'r'}) -> out/; returns dv.json's contents."""
    import torch
    from safetensors.torch import save_file

    cfg, sd = canvas["cfg"], canvas["adapter"]
    layers = canvas_layers(cfg, sd)
    sel_rank = check_selector(selector)
    meta = dv_meta(cfg, layers, sel_rank, slots, tree, fill, mask_token, source)  # refused before anything is written
    bf = lambda x: x.detach().to(torch.bfloat16).contiguous()  # noqa: E731
    t = {"fnormc": bf(sd["norm"]), "mask_emb": bf(sd["mask_emb"]), "res_in": bf(sd["res_in.weight"]),
         "res_out": bf(sd["res_out.weight"])}
    for layer in range(layers):
        p = f"layers.{layer}."
        t[f"L{layer}.ln1c"], t[f"L{layer}.ln2c"] = bf(sd[p + "ln1"]), bf(sd[p + "ln2"])
        for n in LINEARS:
            t[f"L{layer}.{n}.lora_a"], t[f"L{layer}.{n}.lora_b"] = bf(sd[p + n + ".A"]), bf(sd[p + n + ".B"])
    s = selector["sel"]
    f32 = lambda x: x.detach().float().contiguous()  # noqa: E731
    sel = {"A": f32(s["A.weight"]), "H": f32(s["H.weight"]), "B": f32(s["B.weight"]),
           "alpha": f32(s["alpha"]).reshape(1)}
    os.makedirs(out, exist_ok=True)
    save_file(t, os.path.join(out, "canvas.safetensors"))
    save_file(sel, os.path.join(out, "selector.safetensors"))
    with open(os.path.join(out, "dv.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--canvas", help="CanvasAdapter checkpoint ({'adapter', 'cfg'})")
    ap.add_argument("--selector", help="PathSelector checkpoint ({'sel', 'r'})")
    ap.add_argument("--vlora", help="V-side LoRA checkpoint (raft.py's {'lora', 'cfg'}): the target's own adapters")
    ap.add_argument("--vlora-scale", type=float, default=1.0,
                    help="what the V-LoRA's product is multiplied by (raft.py's: none, %(default)s)")
    ap.add_argument("--slots", type=int, help="canvas slots (default: the checkpoint's)")
    ap.add_argument("--tree", default=DEFAULT_TREE, help="beam:T:W[:cov] or pool:T:R[:1] (default %(default)s)")
    ap.add_argument("--fill", type=float, default=DEFAULT_FILL,
                    help="a slot's top token is its next input from this probability on (bifill-dv's TAUS or DV_FILL; "
                         "default %(default)s)")
    ap.add_argument("--mask-token", default=MASK_TOKEN, help="the MASK token's text (default %(default)s)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if bool(a.canvas) != bool(a.selector) or not (a.canvas or a.vlora):
        ap.error("--canvas and --selector (a DV), --vlora (a V-LoRA), or all three")
    import torch
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)  # noqa: E731
    if a.vlora:  # checked before anything is written, as the DV's settings are in export()
        check_vlora(load(a.vlora))
    if a.canvas:
        meta = export(load(a.canvas), load(a.selector), a.out, a.slots, a.tree, a.mask_token,
                      {"canvas": os.path.basename(a.canvas), "selector": os.path.basename(a.selector)}, a.fill)
        print(f"wrote {a.out}: {meta['layers']} layers, {meta['slots']} slots, adapter rank {meta['rank']}, selector "
              f"rank {meta['selector']['rank']}, tree {meta['tree']}, fill {meta['fill']}")
    if a.vlora:
        v = export_vlora(load(a.vlora), a.out, {"vlora": os.path.basename(a.vlora), "sha256": file_sha256(a.vlora)},
                         a.vlora_scale)
        print(f"wrote {a.out}: a V-LoRA of {v['layers']} layers, rank {v['rank']}, scale {v['scale']} "
              f"(sha256 {v['source']['sha256'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
