# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A target's own adapters (a V-LoRA, docs/design/dv.md) in a .flyx file: a rank-r LoRA on every linear of every
layer, ``y = W x + scale b (a x)`` in every pass the target runs, as ``target.L{l}.{attn,mlp}.<linear>.lora_{a,b}``
and the header's ``lora`` section, from the directory python/convert/export_dv.py --vlora writes:

    vlora.safetensors     L{l}.{q,k,v,o,gate,up,down}.lora_a [rank, in] and .lora_b [out, rank]      bfloat16
    vlora.json            kind "lora", layers, rank, scale, source {vlora, sha256, ...}

The engine keeps the adapters beside the weights (unmerged: a bf16 merge would lose most of an adapter's delta), on a
Llama or a Qwen3 (models/llama).  numpy only.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .dv import LINEARS, _SafeTensorsFile

BLOCK = {"q": "attn", "k": "attn", "v": "attn", "o": "attn", "gate": "mlp", "up": "mlp", "down": "mlp"}
FILES = ("vlora.safetensors", "vlora.json")


def has_lora(directory: str | None) -> bool:
    return bool(directory) and os.path.exists(os.path.join(directory, "vlora.json"))


def lora_shapes(cfg: dict, rank: int) -> dict:
    """Every adapter tensor's name (as vlora.safetensors names it) and shape for a target of canonical config cfg."""
    D, q, kv, inter = cfg["hidden"], cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"], cfg["inter"]
    dims = {"q": (q, D), "k": (kv, D), "v": (kv, D), "o": (D, q), "gate": (inter, D), "up": (inter, D),
            "down": (D, inter)}
    want = {}
    for layer in range(cfg["layers"]):
        for n in LINEARS:
            out, inp = dims[n]
            want[f"L{layer}.{n}.lora_a"], want[f"L{layer}.{n}.lora_b"] = [rank, inp], [out, rank]
    return want


def read_lora_meta(lora_dir: str) -> dict:
    with open(os.path.join(lora_dir, "vlora.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("kind") != "lora":
        raise ValueError(f"{lora_dir}: vlora.json is not a V-LoRA export (kind {meta.get('kind')!r})")
    for key in ("layers", "rank", "scale"):
        if key not in meta:
            raise ValueError(f"{lora_dir}: vlora.json has no {key!r}")
    return meta


def add_lora(w, lora_dir: str, cfg: dict, log=print) -> dict:
    """export_dv.py --vlora's directory -> the target's adapters in w (as stored, bfloat16) and the header's lora
    section.  Refuses adapters made for another model (layers, shapes), of ranks apart, a missing or unknown tensor,
    one that is not bfloat16, and a scale that is not a finite number other than 0."""
    meta = read_lora_meta(lora_dir)
    if int(meta["layers"]) != cfg["layers"]:
        raise ValueError(f"{lora_dir}: a V-LoRA for {meta['layers']} layers, the model has {cfg['layers']}")
    rank, scale = int(meta["rank"]), float(meta["scale"])
    if rank < 1:
        raise ValueError(f"{lora_dir}: rank {rank} (at least 1)")
    if not math.isfinite(scale) or scale == 0.0:
        raise ValueError(f"{lora_dir}: scale {scale} (a finite number other than 0)")
    need = lora_shapes(cfg, rank)
    st = _SafeTensorsFile(os.path.join(lora_dir, "vlora.safetensors"))
    unknown = sorted(set(st.index) - set(need))
    if unknown:
        raise ValueError(f"unknown V-LoRA tensors: {unknown[:5]}")
    missing = sorted(set(need) - set(st.index))
    if missing:
        raise ValueError(f"missing V-LoRA tensors: {missing[:10]}")
    for name, shape in need.items():  # in a fixed order, whatever the file's
        arr, dt = st.read(name)
        layer, linear, part = name.split(".")
        if arr.ndim == 2 and arr.shape[0 if part == "lora_a" else 1] != rank:
            raise ValueError(f"V-LoRA tensor {name}: rank {arr.shape[0 if part == 'lora_a' else 1]} "
                             f"({list(arr.shape)}), vlora.json says {rank}")
        if list(arr.shape) != shape:
            raise ValueError(f"V-LoRA tensor {name}: shape {list(arr.shape)}, the model needs {shape}")
        if dt != "bf16":
            raise ValueError(f"V-LoRA tensor {name} is {dt}, not bf16")
        w.add_tensor(f"target.{layer}.{BLOCK[linear]}.{linear}.{part}", dt, arr.shape, np.ascontiguousarray(arr))
    out = {"rank": rank, "scale": scale, "layers": cfg["layers"], "source": meta.get("source")}
    src = meta.get("source") or {}
    log(f"lora: the target's own adapters, rank {rank}, scale {scale}, on every linear of {cfg['layers']} layers "
        f"({src.get('vlora', '?')}, sha256 {str(src.get('sha256', '?'))[:12]})")
    return out
