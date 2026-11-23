# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""DV (bifill-dv's drafter, docs/design/dv.md) in a .flyx file: its canvas adapter as ``dv.*``, its selector as
``dv.sel.*`` and the header's ``dv`` section, from the directory python/convert/export_dv.py writes:

    canvas.safetensors    L{l}.ln1c, L{l}.ln2c [D]; L{l}.{q,k,v,o,gate,up,down}.lora_a [rank, in] and .lora_b
                          [out, rank]; fnormc, mask_emb [D]; res_in [2D, D]; res_out [D, D]          bfloat16
    selector.safetensors  A, H, B [sel_rank, D]; alpha [1]                                          float32
    dv.json               kind "dv", layers, slots, rank, res_k, causal, fill, topk, tree, mask_token (the token's
                          text, or its id), selector {rank}, source

The tree is "beam:T:W[:cov]" (bifill-dv's DYN=T:W[:cov]) or "pool:T:R[:1]" (ST T/R/1).  The earlier CUDA port's
export_dv.py wrote the same tensors with a dv.json in words of its own (rows: the slots; canvas_causal; tree, tree_r,
tree_m: its pool), read here the same.  The names are the ones python/reference/flyx_ref/dv.py reads.  numpy only.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

from .fileformat import NUMPY

LINEARS = ("q", "k", "v", "o", "gate", "up", "down")
# A slot's candidates (the selector's) and the ones its expected embedding sums: 16 and 16, what bifill-dv's canvases
# are trained with (train_dv.py res_k=16) and what the kernels hold (kDvTop, a warp's lanes).  bifill-dv's SEL_K < 16
# keeps fewer candidates with z over 16, which no kernel here runs: refused, as is anything but 16 and 16.
TOPK = 16
ST_DTYPES = {"BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64"}


def pool_size(depths: int, ranks: int) -> int:
    """Tuples of a pool tree over `depths` slots: at depth d the selector's best path and d * R deviations."""
    return sum(1 + d * ranks for d in range(1, depths + 1))


def parse_tree(spec: str, slots: int, topk: int = TOPK) -> dict:
    """beam:T:W[:cov] or pool:T:R[:1] -> the header's tree; refused when it cannot give T drafts on `slots` slots.
    A tree reaches min(slots - 1, T) depths: the last slot never holds candidates after a slide."""
    parts = str(spec).split(":")
    kind = parts[0]
    try:
        nums = [int(x) for x in parts[1:3]]
    except ValueError:
        nums = []
    if kind not in ("beam", "pool") or len(nums) != 2 or len(parts) not in (3, 4):
        raise ValueError(f"DV tree {spec!r}: beam:T:W[:cov] or pool:T:R[:1]")
    width, n = nums
    depths = min(slots - 1, width)
    if width < 1 or n < 1 or depths < 1:
        raise ValueError(f"DV tree {spec!r}: T, W and R are at least 1, and the canvas has at least 2 slots")
    if kind == "beam":
        if len(parts) == 4 and parts[3] != "cov":
            raise ValueError(f"DV tree {spec!r}: the fourth field can only be cov")
        if depths * n < width:
            raise ValueError(f"DV tree {spec!r}: a beam of {n} over {depths} depths ({slots} slots) holds fewer than "
                             f"{width} nodes")
        return {"kind": "beam", "width": width, "beam": n, "cov": len(parts) == 4}
    if len(parts) == 4 and parts[3] != "1":
        raise ValueError(f"DV tree {spec!r}: a pool leaves the selector's best path once (m = 1)")
    if n >= topk:
        raise ValueError(f"DV tree {spec!r}: rank {n} is past a slot's {topk} candidates")
    if pool_size(depths, n) < width:
        raise ValueError(f"DV tree {spec!r}: a pool of {pool_size(depths, n)} tuples cannot give {width} drafts")
    return {"kind": "pool", "width": width, "ranks": n}


def tree_spec(tree: dict) -> str:
    """The header's tree as its spec again (beam:31:8:cov, pool:31:7:1)."""
    if tree["kind"] == "beam":
        return f"beam:{tree['width']}:{tree['beam']}" + (":cov" if tree["cov"] else "")
    return f"pool:{tree['width']}:{tree['ranks']}:1"


def read_meta(dv_dir: str) -> dict:
    """dv.json in this module's words: the earlier CUDA port's (rows, canvas_causal, tree/tree_r/tree_m) translated."""
    with open(os.path.join(dv_dir, "dv.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("kind") != "dv":
        raise ValueError(f"{dv_dir}: dv.json is not a DV export (kind {meta.get('kind')!r})")
    if "rows" in meta and "slots" not in meta:  # the earlier CUDA port's export_dv.py
        meta["slots"] = meta.pop("rows")
        meta["causal"] = meta.pop("canvas_causal", False)
        tree, r, m = meta.pop("tree"), meta.pop("tree_r"), meta.pop("tree_m")
        meta["tree"] = f"pool:{tree}:{r}:{m}"
    for key in ("layers", "slots", "rank", "res_k", "causal", "fill", "topk", "tree", "mask_token"):
        if key not in meta:
            raise ValueError(f"{dv_dir}: dv.json has no {key!r}")
    return meta


class _SafeTensorsFile:
    """One safetensors file's tensors as raw bytes (no dtype conversion)."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        self.index = {}
        for name, rec in hdr.items():
            if name == "__metadata__":
                continue
            if rec["dtype"] not in ST_DTYPES:
                raise ValueError(f"{path}: {name} is {rec['dtype']}, not a float type")
            self.index[name] = (8 + n + rec["data_offsets"][0], rec["data_offsets"][1] - rec["data_offsets"][0],
                                ST_DTYPES[rec["dtype"]], rec["shape"])

    def read(self, name: str):
        off, size, dtype, shape = self.index[name]
        with open(self.path, "rb") as f:
            f.seek(off)
            buf = f.read(size)
        return np.frombuffer(buf, dtype=NUMPY[dtype]).reshape(shape), dtype


def canvas_shapes(cfg: dict, rank: int) -> dict:
    """Every canvas tensor's name and shape for a target of canonical config cfg."""
    D, q, kv, inter = cfg["hidden"], cfg["n_heads"] * cfg["head_dim"], cfg["n_kv"] * cfg["head_dim"], cfg["inter"]
    dims = {"q": (q, D), "k": (kv, D), "v": (kv, D), "o": (D, q), "gate": (inter, D), "up": (inter, D),
            "down": (D, inter)}
    want = {"fnormc": [D], "mask_emb": [D], "res_in": [2 * D, D], "res_out": [D, D]}
    for layer in range(cfg["layers"]):
        want[f"L{layer}.ln1c"] = want[f"L{layer}.ln2c"] = [D]
        for n, (out, inp) in dims.items():
            want[f"L{layer}.{n}.lora_a"], want[f"L{layer}.{n}.lora_b"] = [rank, inp], [out, rank]
    return want


def add_dv(w, dv_dir: str, cfg: dict, added: dict, log=print) -> dict:
    """export_dv.py's directory -> dv.* and dv.sel.* tensors in w (as stored), and the header's dv section.  Refuses
    a DV made for another model (layers, shapes), a mask token the tokenizer does not have, a tree that cannot be
    drafted, a top-k or expected embedding of other than 16 candidates, and anything but a bfloat16 canvas with a
    float32 selector."""
    meta = read_meta(dv_dir)
    if meta["layers"] != cfg["layers"]:
        raise ValueError(f"{dv_dir}: a DV for {meta['layers']} layers, the model has {cfg['layers']}")
    slots, topk, rank = int(meta["slots"]), int(meta["topk"]), int(meta["rank"])
    if slots < 2:
        raise ValueError(f"{dv_dir}: {slots} canvas slots (at least 2)")
    if topk != TOPK or int(meta["res_k"]) != TOPK:
        raise ValueError(f"{dv_dir}: topk {topk}, res_k {meta['res_k']}: a slot keeps {TOPK} candidates and its "
                         f"expected embedding sums those {TOPK} (bifill-dv's canvases; the kernels hold {TOPK})")
    fill = float(meta["fill"])
    if not 0.0 <= fill <= 1.0:
        raise ValueError(f"{dv_dir}: fill {fill} is not a probability")
    tree = parse_tree(meta["tree"], slots, topk)
    mask = meta["mask_token"]
    if isinstance(mask, str):
        if mask not in added:
            raise ValueError(f"DV mask token {mask!r} is not a token of this model")
        mask_id = added[mask]
    else:
        mask_id = int(mask)
        if not 0 <= mask_id < cfg["vocab"]:
            raise ValueError(f"DV mask token {mask_id} is outside the vocabulary")
    sel = meta.get("selector") or {}
    if not os.path.exists(os.path.join(dv_dir, "selector.safetensors")) or "rank" not in sel:
        raise ValueError(f"{dv_dir}: no selector (a DV without one drafts no tree here)")
    sel_rank = int(sel["rank"])
    D = cfg["hidden"]
    parts = (("canvas.safetensors", "dv.", canvas_shapes(cfg, rank), "bf16"),
             ("selector.safetensors", "dv.sel.", {"A": [sel_rank, D], "H": [sel_rank, D], "B": [sel_rank, D],
                                                  "alpha": [1]}, "f32"))
    for fname, prefix, need, dtype in parts:
        st = _SafeTensorsFile(os.path.join(dv_dir, fname))
        for name in st.index:
            if name not in need:
                raise ValueError(f"unknown DV tensor {fname}: {name!r}")
        missing = sorted(set(need) - set(st.index))
        if missing:
            raise ValueError(f"missing DV tensors in {fname}: {missing[:10]}")
        for name in need:  # in a fixed order, whatever the file's
            arr, dt = st.read(name)
            if list(arr.shape) != need[name]:
                raise ValueError(f"DV tensor {fname}: {name} has shape {list(arr.shape)}, the model needs {need[name]}")
            if dt != dtype:
                raise ValueError(f"DV tensor {fname}: {name} is {dt}, not {dtype}")
            w.add_tensor(prefix + name, dt, arr.shape, np.ascontiguousarray(arr))
    out = {"slots": slots, "tree": tree, "topk": TOPK, "res_k": TOPK, "fill": fill,
           "causal": bool(meta["causal"]), "mask_token": mask_id, "rank": rank, "sel_rank": sel_rank,
           "temp": float(meta.get("temp", 1.0)), "source": meta.get("source")}
    log(f"dv: {slots} canvas slots, adapter rank {rank}, selector rank {sel_rank}, tree {tree_spec(tree)}, fill "
        f"{fill}, mask token {mask_id}")
    return out
